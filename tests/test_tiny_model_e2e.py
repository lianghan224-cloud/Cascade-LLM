from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors import safe_open
from transformers import AutoConfig, LlamaForCausalLM

from layer_streaming import (
    CheckpointManifest,
    ExecutionPolicy,
    Llama31DecodeExecutor,
    LlamaModelAdapter,
    MixedDtypeRuntime,
    MixedResidentDeviceArena,
    MixedVocabStreamingRuntime,
    MultiDtypeWeightStore,
    PlacementMode,
    backend_for_weight,
    build_static_transformer_placement,
)


ROOT = Path(__file__).resolve().parents[1]


def generate_checkpoint(
    output, quantization, shards=1, dtype="bf16", extra_args=()
):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "generate_tiny_checkpoint.py"),
            "--output",
            str(output),
            "--layers",
            "1",
            "--hidden-size",
            "32",
            "--intermediate-size",
            "64",
            "--attention-heads",
            "4",
            "--kv-heads",
            "2",
            "--vocab-size",
            "65",
            "--max-context",
            "64",
            "--dtype",
            dtype,
            "--quantization",
            quantization,
            "--group-size",
            "32",
            "--shards",
            str(shards),
        ] + list(extra_args),
        check=True,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
    )


def load_tensors(manifest):
    result = {}
    for filename in manifest.files:
        with safe_open(filename, framework="pt", device="cpu") as source:
            for name in source.keys():
                result[name] = source.get_tensor(name)
    return result


class SyntheticCheckpointTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_resident_lm_head_matches_streamed_for_tied_and_untied(self):
        with tempfile.TemporaryDirectory() as directory:
            for tied in (False, True):
                root = Path(directory) / ("tied" if tied else "untied")
                generate_checkpoint(
                    root,
                    "none",
                    dtype="bf16",
                    extra_args=("--tie-word-embeddings",) if tied else (),
                )
                config = AutoConfig.from_pretrained(root, local_files_only=True)
                device = torch.device("cuda:0")
                outputs = []
                h2d_bytes = []
                for lm_head_mode in (
                    PlacementMode.STREAMED,
                    PlacementMode.RESIDENT,
                ):
                    policy = ExecutionPolicy.from_config(
                        config,
                        embedding_mode=PlacementMode.RESIDENT,
                        lm_head_mode=lm_head_mode,
                        vocab_chunk_bytes=4096,
                    )
                    plan = LlamaModelAdapter().build_execution_plan(
                        config, policy
                    )
                    with ExitStack() as resources:
                        store = resources.enter_context(
                            MultiDtypeWeightStore(plan, "pinned_staging")
                        )
                        store.load_checkpoint(root)
                        resident = resources.enter_context(
                            MixedResidentDeviceArena(plan, store, device)
                        )
                        runtime = resources.enter_context(
                            MixedDtypeRuntime(plan, store, resident, device)
                        )
                        vocab = None
                        if plan.vocab.stream_lm_head:
                            vocab = resources.enter_context(
                                MixedVocabStreamingRuntime(
                                    plan,
                                    store,
                                    runtime,
                                    embedding_staging_rows=8,
                                    profile=True,
                                )
                            )
                        executor = resources.enter_context(
                            Llama31DecodeExecutor(
                                config,
                                resident,
                                vocab_runtime=vocab,
                                max_cache_length=8,
                                return_full_logits=True,
                            )
                        )
                        with torch.inference_mode():
                            state = executor.finish(
                                runtime.run(
                                    executor,
                                    executor.begin(
                                        torch.tensor(
                                            [[1, 4, 5]], device=device
                                        )
                                    ),
                                )
                            )
                        outputs.append(state.logits.cpu())
                        h2d_bytes.append(
                            0 if vocab is None else vocab.last_profile["h2d_bytes"]
                        )
                self.assertTrue(torch.equal(outputs[0], outputs[1]))
                self.assertGreater(h2d_bytes[0], 0)
                self.assertEqual(h2d_bytes[1], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_static_resident_transformer_matches_streamed_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_checkpoint(root, "none", dtype="bf16")
            config = AutoConfig.from_pretrained(root, local_files_only=True)
            policy = ExecutionPolicy.from_config(
                config,
                embedding_mode=PlacementMode.RESIDENT,
                lm_head_mode=PlacementMode.RESIDENT,
                vocab_chunk_bytes=4096,
            )
            plan = LlamaModelAdapter().build_execution_plan(config, policy)
            placements = (
                build_static_transformer_placement(plan, 0),
                build_static_transformer_placement(plan, 1 << 60),
            )
            device = torch.device("cuda:0")
            outputs = []
            profiles = []
            with MultiDtypeWeightStore(plan, "pinned_staging") as store:
                store.load_checkpoint(root)
                for placement in placements:
                    with ExitStack() as resources:
                        resident = resources.enter_context(
                            MixedResidentDeviceArena(
                                plan,
                                store,
                                device,
                                transformer_placement=placement,
                            )
                        )
                        runtime = resources.enter_context(
                            MixedDtypeRuntime(
                                plan,
                                store,
                                resident,
                                device,
                                profile=True,
                            )
                        )
                        executor = resources.enter_context(
                            Llama31DecodeExecutor(
                                config,
                                resident,
                                max_cache_length=16,
                                return_full_logits=True,
                            )
                        )
                        with torch.inference_mode():
                            result = executor.finish(
                                runtime.run(
                                    executor,
                                    executor.begin(
                                        torch.tensor(
                                            [[1, 4, 5]], device=device
                                        )
                                    ),
                                )
                            )
                        outputs.append(result.logits.cpu())
                        profiles.append(dict(runtime.last_profile))
            self.assertTrue(torch.equal(outputs[0], outputs[1]))
            self.assertGreater(profiles[0]["h2d_bytes"], 0)
            self.assertEqual(profiles[1]["h2d_bytes"], 0)
            self.assertEqual(profiles[1]["streamed_weight_bytes"], 0)
            self.assertEqual(profiles[1]["resident_hit_ratio"], 1.0)

    def test_generator_creates_valid_int8_multi_shard_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_checkpoint(
                root,
                "int8_per_group",
                shards=2,
                dtype="fp16",
                extra_args=(
                    "--embedding-dtype",
                    "bf16",
                    "--lm-head-dtype",
                    "bf16",
                    "--norm-dtype",
                    "bf16",
                ),
            )
            config = AutoConfig.from_pretrained(root, local_files_only=True)
            policy = ExecutionPolicy.from_config(config)
            adapter = LlamaModelAdapter()
            specs = adapter.enumerate_weights(config, policy=policy)
            manifest = CheckpointManifest.from_path(root)
            result = manifest.validate(specs)
            self.assertTrue(result.ok, result.format_errors())
            self.assertEqual(len(manifest.files), 2)
            self.assertEqual(policy.quantization.group_size, 32)
            self.assertEqual(policy.quantization.scale_dtype, "bfloat16")
            self.assertEqual(policy.embedding_dtype, "bfloat16")
            self.assertEqual(policy.lm_head_dtype, "bfloat16")
            self.assertEqual(policy.norm_dtype, "bfloat16")
            self.assertTrue((root / "reference" / "model.safetensors").is_file())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_int8_fp16_full_pinned_resident_layer_path_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_checkpoint(
                root, "int8_per_channel", dtype="fp16"
            )
            config = AutoConfig.from_pretrained(root, local_files_only=True)
            policy = ExecutionPolicy.from_config(
                config,
                granularity="layer",
                embedding_mode=PlacementMode.RESIDENT,
                lm_head_mode=PlacementMode.RESIDENT,
                vocab_chunk_bytes=4096,
            )
            adapter = LlamaModelAdapter()
            plan = adapter.build_execution_plan(config, policy)
            device = torch.device("cuda:0")
            with ExitStack() as resources:
                store = resources.enter_context(
                    MultiDtypeWeightStore(plan, "full_pinned")
                )
                store.load_checkpoint(root)
                resident = resources.enter_context(
                    MixedResidentDeviceArena(plan, store, device)
                )
                runtime = resources.enter_context(
                    MixedDtypeRuntime(
                        plan, store, resident, device, profile=True
                    )
                )
                executor = resources.enter_context(
                    Llama31DecodeExecutor(
                        config,
                        resident,
                        max_cache_length=16,
                        kv_dtype=torch.float16,
                        return_full_logits=True,
                    )
                )
                with torch.inference_mode():
                    result = executor.finish(
                        runtime.run(
                            executor,
                            executor.begin(
                                torch.tensor([[1, 4, 5]], device=device)
                            ),
                        )
                    )
                self.assertTrue(torch.isfinite(result.logits).all())
                self.assertEqual(
                    runtime.last_profile["fallback_backends"],
                    ["int8_dequant_fp16_fallback"],
                )
                self.assertGreater(
                    runtime.last_profile["dequant_event_sum_ms"], 0
                )
                self.assertGreater(
                    runtime.last_profile["gemm_event_sum_ms"], 0
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_int4_streamed_prefill_and_decode_match_explicit_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_checkpoint(root, "int4_per_group", shards=2)
            config = AutoConfig.from_pretrained(root, local_files_only=True)
            policy = ExecutionPolicy.from_config(
                config,
                embedding_mode=PlacementMode.STREAMED,
                lm_head_mode=PlacementMode.STREAMED,
                vocab_chunk_bytes=4096,
            )
            adapter = LlamaModelAdapter()
            plan = adapter.build_execution_plan(config, policy)
            manifest = CheckpointManifest.from_path(root)
            tensors = load_tensors(manifest)

            reference = LlamaForCausalLM(config).to(torch.bfloat16).eval()
            state = {}
            for name in reference.state_dict():
                spec = plan.weights[name]
                storage_name = spec.alias_of or name
                storage_spec = plan.weights[storage_name]
                if storage_spec.quantization is None:
                    state[name] = tensors[storage_name]
                else:
                    backend = backend_for_weight(storage_spec)
                    state[name] = backend.dequantize(
                        storage_spec,
                        tensors[storage_name],
                        {
                            "scale": tensors[storage_name + "_scale"],
                            "weight_spec": storage_spec,
                        },
                    )
            reference.load_state_dict(state, strict=True)
            device = torch.device("cuda:0")
            reference = reference.to(device)
            prefill_ids = torch.tensor([[1, 5, 9]], device=device)
            decode_ids = torch.tensor([[7]], device=device)
            with torch.inference_mode():
                first = reference(prefill_ids, use_cache=True)
                reference_prefill = first.logits.float()
                reference_decode = reference(
                    decode_ids,
                    past_key_values=first.past_key_values,
                    use_cache=True,
                ).logits.float()
            del reference, first

            with ExitStack() as resources:
                store = resources.enter_context(
                    MultiDtypeWeightStore(plan, "pinned_staging")
                )
                store.load_checkpoint(root)
                resident = resources.enter_context(
                    MixedResidentDeviceArena(plan, store, device)
                )
                runtime = resources.enter_context(
                    MixedDtypeRuntime(plan, store, resident, device)
                )
                vocab = resources.enter_context(
                    MixedVocabStreamingRuntime(
                        plan,
                        store,
                        runtime,
                        embedding_staging_rows=16,
                    )
                )
                executor = resources.enter_context(
                    Llama31DecodeExecutor(
                        config,
                        resident,
                        vocab_runtime=vocab,
                        max_cache_length=16,
                        return_full_logits=True,
                        top_k=10,
                    )
                )
                with torch.inference_mode():
                    candidate_prefill = executor.finish(
                        runtime.run(executor, executor.begin(prefill_ids))
                    ).logits
                    candidate_decode = executor.finish(
                        runtime.run(executor, executor.begin(decode_ids))
                    ).logits
                self.assertTrue(torch.equal(reference_prefill, candidate_prefill))
                self.assertTrue(torch.equal(reference_decode, candidate_decode))
                self.assertEqual(
                    runtime.last_profile["fallback_backends"],
                    ["int4_dequant_bf16_fallback"],
                )


if __name__ == "__main__":
    unittest.main()
