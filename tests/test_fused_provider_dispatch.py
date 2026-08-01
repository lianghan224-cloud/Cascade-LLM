from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import AutoConfig

from layer_streaming import (
    BackendInfo,
    ExecutionPolicy,
    Int8DequantBF16FallbackBackend,
    Llama31DecodeExecutor,
    MixedDtypeRuntime,
    MixedResidentDeviceArena,
    MultiDtypeWeightStore,
    PlacementMode,
    adapter_for_config,
    register_linear_backend,
    unregister_linear_backend,
)
from test_tiny_model_e2e import generate_checkpoint


class MockFusedW8A16Provider:
    """Test-only provider proving dispatch; not a performance implementation."""

    name = "fused_w8a16"
    is_fallback = False
    info = BackendInfo(
        name=name,
        storage_dtype="int8",
        activation_dtype="bfloat16",
        output_dtype="bfloat16",
        requires_dequant=False,
        is_fallback=False,
        supported_gpu_architectures=("test",),
        atol=0.02,
        rtol=0.02,
    )

    def __init__(self):
        self.reference = Int8DequantBF16FallbackBackend()

    def validate(self, weight, input_dtype):
        self.reference.validate(weight, input_dtype)

    def transfer_bytes(self, weight):
        return self.reference.transfer_bytes(weight)

    def workspace_bytes(self, weight, batch_tokens):
        return self.reference.workspace_bytes(weight, batch_tokens)

    def execute(self, x, weight_view, quant_views, workspace):
        return self.reference.execute(
            x, weight_view, quant_views, workspace
        )


class FusedProviderDispatchTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_registered_nonfallback_provider_reaches_execute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_checkpoint(root, "int8_per_group")
            config = AutoConfig.from_pretrained(
                root, local_files_only=True
            )
            register_linear_backend(MockFusedW8A16Provider())
            try:
                policy = ExecutionPolicy.from_config(
                    config,
                    linear_backend="fused_w8a16",
                    embedding_mode=PlacementMode.RESIDENT,
                    lm_head_mode=PlacementMode.RESIDENT,
                    vocab_chunk_bytes=4096,
                )
                plan = adapter_for_config(config).build_execution_plan(
                    config, policy
                )
                self.assertEqual(
                    {
                        tensor.backend
                        for unit in plan.units
                        for tensor in unit.tensors
                        if tensor.backend
                    },
                    {"fused_w8a16"},
                )
                with ExitStack() as resources:
                    store = resources.enter_context(
                        MultiDtypeWeightStore(plan, "pinned_staging")
                    )
                    store.load_checkpoint(root)
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
                    executor = resources.enter_context(
                        Llama31DecodeExecutor(
                            config,
                            resident,
                            max_cache_length=8,
                            return_full_logits=False,
                        )
                    )
                    with torch.inference_mode():
                        state = executor.finish(
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
                    self.assertTrue(
                        torch.isfinite(state.topk_values).all()
                    )
                    self.assertEqual(
                        runtime.last_profile["backends"],
                        ["fused_w8a16"],
                    )
                    self.assertEqual(
                        runtime.last_profile["fallback_backends"], []
                    )
                    self.assertEqual(
                        runtime.last_profile["dequant_event_sum_ms"], 0
                    )
            finally:
                unregister_linear_backend("fused_w8a16")


if __name__ == "__main__":
    unittest.main()
