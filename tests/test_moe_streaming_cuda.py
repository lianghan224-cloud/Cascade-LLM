import unittest

import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from layer_streaming import ExecutionPolicy
from layer_streaming.mixed_runtime import MixedDtypeRuntime, MixedResidentDeviceArena
from layer_streaming.multi_dtype_store import MultiDtypeWeightStore
from layer_streaming.moe import (
    ExpertCache,
    ExpertKey,
    ExpertResidencyManager,
    ExpertScheduler,
    ExpertTransferEngine,
    OlmoeModelAdapter,
    StreamingOlmoeSparseMoeBlock,
    UnifiedResidentWeightBudget,
)


def tiny_config():
    return OlmoeConfig(
        vocab_size=37,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        attention_bias=False,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ExistingRuntimeExpertStreamingTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda", torch.cuda.current_device())
        free_bytes, _total = torch.cuda.mem_get_info(self.device)
        if free_bytes < 32 * 1024**2:
            self.skipTest("less than 32 MiB free CUDA memory")
        self.policy = ExecutionPolicy(slot_count=2, prefetch_depth=2, vocab_chunk_bytes=4096)
        self.sidecar = OlmoeModelAdapter().build_moe_plan(tiny_config(), self.policy)
        self.plan = self.sidecar.execution_plan
        self.store = MultiDtypeWeightStore(self.plan, staging_slot_count=2)
        with torch.no_grad():
            for index, spec in enumerate(self.plan.weights.values()):
                if spec.alias_of is None:
                    self.store.view(spec.name).fill_((index % 13 + 1) / 32.0)
        self.resident = MixedResidentDeviceArena(
            self.plan, self.store, device=self.device
        )
        self.runtime = MixedDtypeRuntime(
            self.plan,
            self.store,
            self.resident,
            device=self.device,
            slot_count=2,
            prefetch_depth=2,
        )
        self.manager = ExpertResidencyManager()
        for expert_id in range(4):
            self.manager.register(ExpertKey(0, expert_id))
        sample = self.sidecar.expert_unit(0, 0)
        fixed = self.plan.resident_bytes
        budget = UnifiedResidentWeightBudget(
            fixed + sample.transfer_bytes * 2, fixed
        )
        self.cache = ExpertCache(self.manager, budget)
        self.engine = ExpertTransferEngine(
            self.runtime, self.cache, profile=True
        )
        self.scheduler = ExpertScheduler(
            self.engine,
            lambda key: self.sidecar.expert_unit(key.layer_id, key.expert_id),
        )

    def tearDown(self):
        if hasattr(self, "engine"):
            # Explicit profiling/teardown boundary; event synchronization is
            # permitted here and absent from the per-Expert hot path.
            for ticket in tuple(self.engine._tickets.values()):
                self.engine.activate(ticket)
            self.engine.compute_stream.synchronize()
            self.engine.close()
        if hasattr(self, "runtime"):
            self.runtime.close()
        if hasattr(self, "resident"):
            self.resident.close()
        if hasattr(self, "store"):
            self.store.close()

    def test_exact_prefetch_uses_existing_pinned_store_and_cuda_events(self):
        key = ExpertKey(0, 0)
        unit = self.sidecar.expert_unit(0, 0)
        ticket = self.engine.submit(key, unit)
        views = self.engine.activate(ticket)
        with torch.cuda.stream(self.engine.compute_stream):
            actual = views[unit.tensors[0].weight_name].clone()
        done = torch.cuda.Event()
        done.record(self.engine.compute_stream)
        done.synchronize()
        expected = self.store.view(unit.tensors[0].weight_name)
        torch.testing.assert_close(actual.cpu(), expected)
        self.assertEqual(self.store.profile_stats()["staging_copy_count"], 1)
        stats = self.engine.profile_stats(finalize=True)
        self.assertEqual(stats["expert_h2d_bytes"], unit.transfer_bytes)
        self.assertEqual(stats["inflight"], 0)
        self.assertGreaterEqual(stats["expert_h2d_time_ms"], 0.0)

    def test_scheduler_computes_hit_before_individually_ready_miss(self):
        first = ExpertKey(0, 0)
        ticket = self.engine.submit(first, self.sidecar.expert_unit(0, 0))
        self.engine.activate(ticket)
        self.manager.acquire(first)
        self.manager.release(first)
        schedule = self.scheduler.build_schedule(0, [0, 1])
        self.assertEqual(schedule.resident_keys, (first,))
        self.assertEqual(schedule.missing_keys, (ExpertKey(0, 1),))
        order = []

        def compute(key, views):
            order.append(key)
            unit = self.sidecar.expert_unit(key.layer_id, key.expert_id)
            return views[unit.tensors[0].weight_name].sum()

        results = self.scheduler.execute(schedule, compute)
        done = torch.cuda.Event()
        done.record(self.engine.compute_stream)
        done.synchronize()
        self.assertEqual(order, [ExpertKey(0, 0), ExpertKey(0, 1)])
        self.assertEqual(set(results), set(order))
        self.assertEqual(self.manager.stats()["inflight"], 0)
        self.assertEqual(self.manager.stats()["use_count"], 0)

    def test_cache_eviction_reuses_bounded_arena_without_growth(self):
        for expert_id in (0, 1):
            ticket = self.engine.submit(
                ExpertKey(0, expert_id), self.sidecar.expert_unit(0, expert_id)
            )
            self.engine.activate(ticket)
            self.manager.acquire(ExpertKey(0, expert_id))
            self.manager.release(ExpertKey(0, expert_id))
        used = self.engine.arena.used_bytes
        ticket = self.engine.submit(ExpertKey(0, 2), self.sidecar.expert_unit(0, 2))
        self.engine.activate(ticket)
        self.assertEqual(self.engine.arena.used_bytes, used)
        self.assertEqual(self.cache.stats()["entries"], 2)
        self.assertEqual(self.cache.stats()["evictions"], 1)

    def test_streaming_olmoe_block_matches_hf_logits_and_token(self):
        torch.manual_seed(812)
        reference = OlmoeForCausalLM(tiny_config()).to(
            device=self.device, dtype=torch.bfloat16
        ).eval()
        # Load the exact HF parameters into the real Cascade CPU arena.
        state = reference.state_dict()
        with torch.no_grad():
            for name, spec in self.plan.weights.items():
                if spec.alias_of is None:
                    self.store.view(name).copy_(state[name].detach().cpu())
        candidate = OlmoeForCausalLM(tiny_config()).to(
            device=self.device, dtype=torch.bfloat16
        ).eval()
        candidate.load_state_dict(reference.state_dict())
        block = candidate.model.layers[0].mlp
        candidate.model.layers[0].mlp = StreamingOlmoeSparseMoeBlock(
            block.gate,
            self.sidecar.moe_config,
            layer_id=0,
            scheduler=self.scheduler,
            max_iteration_tokens=1,
        )
        input_ids = torch.tensor([[3]], device=self.device)
        with torch.no_grad():
            expected = reference(input_ids).logits
            actual = candidate(input_ids).logits
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
        self.assertTrue(
            torch.equal(actual[:, -1].argmax(-1), expected[:, -1].argmax(-1))
        )
        profile = candidate.model.layers[0].mlp.profile_stats()
        self.assertEqual(profile["tokens"], 1)
        self.assertGreaterEqual(profile["expert_compute_time_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
