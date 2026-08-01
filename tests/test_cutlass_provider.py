from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import AutoConfig

from layer_streaming import (
    BackendSelection,
    ExecutionPolicy,
    Llama31DecodeExecutor,
    MixedDtypeRuntime,
    MixedResidentDeviceArena,
    MultiDtypeWeightStore,
    PlacementMode,
    adapter_for_config,
    build_backend_phase_plan,
    qualify_backend,
    unregister_linear_backend,
)
from layer_streaming.providers.cutlass import (
    CutlassW8A16Provider,
    load_cutlass_w8a16_provider,
)
from test_tiny_model_e2e import generate_checkpoint


LIBRARY = (
    Path(__file__).resolve().parents[1]
    / "layer_streaming/providers/cutlass/_build/"
    "libcascade_cutlass_sm86.so"
)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability("cuda:0") == (8, 6)
    and LIBRARY.is_file(),
    "compiled CUTLASS SM86 provider is required",
)
class CutlassProviderTest(unittest.TestCase):
    def test_real_provider_runs_prefill_and_explicit_decode_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            generate_checkpoint(checkpoint, "int8_per_channel")
            config = AutoConfig.from_pretrained(
                checkpoint, local_files_only=True
            )
            provider = load_cutlass_w8a16_provider(LIBRARY)
            try:
                inferred = ExecutionPolicy.from_config(config)
                policy = ExecutionPolicy.from_config(
                    config,
                    linear_backend="int8_dequant_bf16_fallback",
                    embedding_mode=PlacementMode.RESIDENT,
                    lm_head_mode=PlacementMode.RESIDENT,
                    vocab_chunk_bytes=4096,
                )
                self.assertEqual(
                    inferred.quantization.granularity, "per_channel"
                )
                plan = adapter_for_config(config).build_execution_plan(
                    config, policy
                )
                phase_plan = build_backend_phase_plan(
                    plan,
                    BackendSelection(
                        prefill=provider.name,
                        decode="int8_dequant_bf16_fallback",
                    ),
                    device="cuda:0",
                )
                self.assertEqual(plan.as_dict()["schema_version"], 1)
                self.assertEqual(provider.workspace_bytes(
                    next(
                        spec
                        for spec in plan.weights.values()
                        if spec.quantization is not None
                    ),
                    1,
                ), 0)
                reference_decode_topk = None
                for mode in ("full_pinned", "pinned_staging"):
                    with self.subTest(mode=mode), ExitStack() as resources:
                        store = resources.enter_context(
                            MultiDtypeWeightStore(plan, mode)
                        )
                        store.load_checkpoint(checkpoint)
                        resident = resources.enter_context(
                            MixedResidentDeviceArena(
                                plan, store, "cuda:0"
                            )
                        )
                        runtime = resources.enter_context(
                            MixedDtypeRuntime(
                                plan,
                                store,
                                resident,
                                "cuda:0",
                                profile=True,
                            )
                        )
                        runtime.configure_backend_phase_plan(phase_plan)
                        executor = resources.enter_context(
                            Llama31DecodeExecutor(
                                config,
                                resident,
                                max_cache_length=8,
                                return_full_logits=False,
                            )
                        )
                        with torch.inference_mode():
                            prefill = executor.finish(
                                runtime.run(
                                    executor,
                                    executor.begin(
                                        torch.tensor(
                                            [[1, 4]],
                                            dtype=torch.long,
                                            device="cuda:0",
                                        )
                                    ),
                                )
                            )
                            prefill_profile = dict(runtime.last_profile)
                            decode = executor.finish(
                                runtime.run(
                                    executor,
                                    executor.begin(
                                        prefill.topk_indices[:, -1:, 0]
                                    ),
                                )
                            )
                            decode_profile = dict(runtime.last_profile)
                        self.assertTrue(
                            torch.isfinite(decode.topk_values).all()
                        )
                        self.assertEqual(
                            prefill_profile["backend_phase"], "prefill"
                        )
                        self.assertEqual(
                            prefill_profile["backends"], ["fused_w8a16"]
                        )
                        self.assertEqual(
                            prefill_profile["fallback_backends"], []
                        )
                        self.assertEqual(
                            prefill_profile["backend_providers"],
                            ["cutlass_sm86_w8a16"],
                        )
                        self.assertEqual(
                            decode_profile["backend_phase"], "decode"
                        )
                        self.assertEqual(
                            decode_profile["backends"],
                            ["int8_dequant_bf16_fallback"],
                        )
                        self.assertTrue(
                            decode_profile["decode_fallback_explicit"]
                        )
                        reference_decode_topk = decode.topk_indices.detach().clone()

                fused_phase_plan = build_backend_phase_plan(
                    plan,
                    BackendSelection(
                        prefill=provider.name,
                        decode=provider.name,
                    ),
                    device="cuda:0",
                )
                with ExitStack() as resources:
                    store = resources.enter_context(
                        MultiDtypeWeightStore(plan, "pinned_staging")
                    )
                    store.load_checkpoint(checkpoint)
                    resident = resources.enter_context(
                        MixedResidentDeviceArena(plan, store, "cuda:0")
                    )
                    runtime = resources.enter_context(
                        MixedDtypeRuntime(
                            plan,
                            store,
                            resident,
                            "cuda:0",
                            profile=True,
                        )
                    )
                    runtime.configure_backend_phase_plan(fused_phase_plan)
                    executor = resources.enter_context(
                        Llama31DecodeExecutor(
                            config,
                            resident,
                            max_cache_length=8,
                            return_full_logits=False,
                        )
                    )
                    with torch.inference_mode():
                        prefill = executor.finish(
                            runtime.run(
                                executor,
                                executor.begin(
                                    torch.tensor(
                                        [[1, 4]],
                                        dtype=torch.long,
                                        device="cuda:0",
                                    )
                                ),
                            )
                        )
                        fused_decode = executor.finish(
                            runtime.run(
                                executor,
                                executor.begin(
                                    prefill.topk_indices[:, -1:, 0]
                                ),
                            )
                        )
                    self.assertEqual(
                        runtime.last_profile["backend_phase"], "decode"
                    )
                    self.assertEqual(
                        runtime.last_profile["backends"], ["fused_w8a16"]
                    )
                    self.assertEqual(
                        runtime.last_profile["fallback_backends"], []
                    )
                    self.assertEqual(
                        runtime.last_profile["backend_providers"],
                        ["cutlass_sm86_w8a16"],
                    )
                    self.assertFalse(
                        runtime.last_profile["decode_fallback_explicit"]
                    )
                    self.assertTrue(
                        torch.equal(
                            fused_decode.topk_indices,
                            reference_decode_topk,
                        )
                    )
            finally:
                unregister_linear_backend("fused_w8a16")

    def test_groupwise_int8_decode_qualification_and_numerics(self):
        from layer_streaming import QuantizationSpec, WeightSpec

        provider = load_cutlass_w8a16_provider(LIBRARY)
        spec = WeightSpec.quantized(
            "weight",
            (64, 128),
            QuantizationSpec(
                bits=8,
                granularity="per_group",
                group_size=32,
            ),
            "bf16",
            "attention_q",
        )
        try:
            provider.validate(spec, "bf16")
            decode = qualify_backend(
                spec, provider.name, 1, device="cuda:0"
            )
            prefill = qualify_backend(
                spec, provider.name, 2, device="cuda:0"
            )
            self.assertTrue(decode.supported, decode.unsupported_reason)
            self.assertEqual(decode.workspace_bytes, 0)
            self.assertFalse(prefill.supported)
            self.assertIn("decode-only", prefill.unsupported_reason)

            torch.manual_seed(31)
            x = torch.randn(
                (1, 128), dtype=torch.bfloat16, device="cuda:0"
            )
            weight = torch.randint(
                -8, 8, (64, 128), dtype=torch.int8, device="cuda:0"
            )
            scale = (
                torch.rand(
                    (64, 4), dtype=torch.bfloat16, device="cuda:0"
                )
                * 0.09
                + 0.01
            )
            with torch.inference_mode():
                actual = provider.execute(
                    x,
                    weight,
                    {"scale": scale, "weight_spec": spec},
                    None,
                )
                dequantized = (
                    weight.to(torch.bfloat16)
                    * scale.repeat_interleave(32, dim=1)
                )
                expected = torch.nn.functional.linear(x, dequantized)
            torch.testing.assert_close(
                actual, expected, atol=0.25, rtol=0.02
            )
        finally:
            unregister_linear_backend("fused_w8a16")

    def test_groupwise_int8_uses_explicit_prefill_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            generate_checkpoint(checkpoint, "int8_per_group")
            config = AutoConfig.from_pretrained(
                checkpoint, local_files_only=True
            )
            provider = load_cutlass_w8a16_provider(LIBRARY)
            try:
                policy = ExecutionPolicy.from_config(
                    config,
                    linear_backend="int8_dequant_bf16_fallback",
                    embedding_mode=PlacementMode.RESIDENT,
                    lm_head_mode=PlacementMode.RESIDENT,
                    vocab_chunk_bytes=4096,
                )
                plan = adapter_for_config(config).build_execution_plan(
                    config, policy
                )
                phase_plan = build_backend_phase_plan(
                    plan,
                    BackendSelection(
                        prefill="int8_dequant_bf16_fallback",
                        decode=provider.name,
                    ),
                    device="cuda:0",
                )
                with ExitStack() as resources:
                    store = resources.enter_context(
                        MultiDtypeWeightStore(plan, "pinned_staging")
                    )
                    store.load_checkpoint(checkpoint)
                    resident = resources.enter_context(
                        MixedResidentDeviceArena(plan, store, "cuda:0")
                    )
                    runtime = resources.enter_context(
                        MixedDtypeRuntime(
                            plan,
                            store,
                            resident,
                            "cuda:0",
                            profile=True,
                        )
                    )
                    runtime.configure_backend_phase_plan(phase_plan)
                    executor = resources.enter_context(
                        Llama31DecodeExecutor(
                            config,
                            resident,
                            max_cache_length=8,
                            return_full_logits=False,
                        )
                    )
                    with torch.inference_mode():
                        prefill = executor.finish(
                            runtime.run(
                                executor,
                                executor.begin(
                                    torch.tensor(
                                        [[1, 4]],
                                        dtype=torch.long,
                                        device="cuda:0",
                                    )
                                ),
                            )
                        )
                        prefill_profile = dict(runtime.last_profile)
                        decode = executor.finish(
                            runtime.run(
                                executor,
                                executor.begin(
                                    prefill.topk_indices[:, -1:, 0]
                                ),
                            )
                        )
                        decode_profile = dict(runtime.last_profile)
                self.assertTrue(torch.isfinite(decode.topk_values).all())
                self.assertEqual(
                    prefill_profile["backends"],
                    ["int8_dequant_bf16_fallback"],
                )
                self.assertEqual(
                    decode_profile["backends"], ["fused_w8a16"]
                )
                self.assertEqual(
                    decode_profile["backend_providers"],
                    ["cutlass_sm86_w8a16"],
                )
                self.assertFalse(
                    decode_profile["decode_fallback_explicit"]
                )
            finally:
                unregister_linear_backend("fused_w8a16")

    def test_repeated_decode_execute_has_no_allocator_drift(self):
        from layer_streaming import QuantizationSpec, WeightSpec

        provider = CutlassW8A16Provider(LIBRARY)
        spec = WeightSpec.quantized(
            "weight",
            (128, 256),
            QuantizationSpec(bits=8, granularity="per_channel"),
            "bf16",
            "attention_q",
        )
        x = torch.randn((1, 256), dtype=torch.bfloat16, device="cuda:0")
        weight = torch.randint(
            -127, 128, (128, 256), dtype=torch.int8, device="cuda:0"
        )
        scale = torch.rand(
            (128, 1), dtype=torch.bfloat16, device="cuda:0"
        )
        views = {"scale": scale, "weight_spec": spec}
        output = None
        with torch.inference_mode():
            for _ in range(20):
                output = provider.execute(x, weight, views, None)
            torch.cuda.synchronize("cuda:0")
            allocated = torch.cuda.memory_allocated("cuda:0")
            reserved = torch.cuda.memory_reserved("cuda:0")
            for _ in range(500):
                output = provider.execute(x, weight, views, None)
            torch.cuda.synchronize("cuda:0")
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(torch.cuda.memory_allocated("cuda:0"), allocated)
        self.assertEqual(torch.cuda.memory_reserved("cuda:0"), reserved)


if __name__ == "__main__":
    unittest.main()
