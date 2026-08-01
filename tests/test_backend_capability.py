import json
import unittest

from layer_streaming import (
    BackendCapability,
    BackendInfo,
    BackendSelection,
    BackendUnavailableError,
    ExecutionPolicy,
    LlamaModelAdapter,
    QuantizationSpec,
    WeightFormat,
    WeightSpec,
    build_backend_phase_plan,
    qualify_backend,
    register_linear_backend,
    unregister_linear_backend,
)

from test_plan import tiny_config


class BackendCapabilityTest(unittest.TestCase):
    def test_capability_reports_m_alignment_and_group_failures(self):
        capability = BackendCapability(
            min_m=1,
            max_m=512,
            supported_sms=(86,),
            activation_dtypes=("bf16",),
            weight_formats=("int8_symmetric_per_group",),
            group_sizes=(64, 128),
            alignment_k=16,
            alignment_n=8,
        )
        spec = WeightSpec.quantized(
            "weight",
            (24, 96),
            QuantizationSpec(
                bits=8,
                granularity="per_group",
                group_size=32,
            ),
            "bf16",
            "attention_q",
        )
        self.assertIn(
            "group size 32", capability.unsupported_reason(spec, 1, sm=86)
        )
        self.assertIn(
            "SM89", capability.unsupported_reason(spec, 1, sm=89)
        )
        self.assertEqual(
            BackendCapability.from_dict(
                json.loads(json.dumps(capability.as_dict()))
            ),
            capability,
        )

    def test_qualification_rejects_decode_before_execute(self):
        class PrefillOnlyProvider:
            name = "fused_w8a16"
            is_fallback = False
            info = BackendInfo(
                name=name,
                storage_dtype="int8",
                activation_dtype="bfloat16",
                output_dtype="bfloat16",
                requires_dequant=False,
                is_fallback=False,
                supported_gpu_architectures=("sm86",),
                atol=0.02,
                rtol=0.02,
            )
            capability = BackendCapability(
                min_m=2,
                max_m=None,
                supported_sms=(86,),
                activation_dtypes=("bf16",),
                weight_formats=("int8_symmetric_per_channel",),
                group_sizes=(),
                alignment_k=16,
                alignment_n=8,
            )

            def validate(self, weight, input_dtype):
                del weight, input_dtype

            def transfer_bytes(self, weight):
                return weight.storage_nbytes

            def workspace_bytes(self, weight, batch_tokens):
                del weight, batch_tokens
                return 0

            def execute(self, x, weight_view, quant_views, workspace):
                raise AssertionError("qualification must not execute")

        spec = WeightSpec.quantized(
            "weight",
            (16, 32),
            QuantizationSpec(bits=8, granularity="per_channel"),
            "bf16",
            "attention_q",
        )
        register_linear_backend(PrefillOnlyProvider())
        try:
            result = qualify_backend(spec, "fused_w8a16", 1)
            self.assertFalse(result.supported)
            self.assertIn("minimum M=2", result.unsupported_reason)
        finally:
            unregister_linear_backend("fused_w8a16")

    def test_phase_plan_is_sidecar_and_checks_both_phases(self):
        config = tiny_config()
        policy = ExecutionPolicy(
            weight_format=WeightFormat.INT8_DEQUANT_BF16_FALLBACK,
            quantization=QuantizationSpec(
                bits=8,
                granularity="per_group",
                group_size=32,
            ),
            linear_backend="int8_dequant_bf16_fallback",
            vocab_chunk_bytes=4096,
        )
        plan = LlamaModelAdapter().build_execution_plan(config, policy)
        original = plan.as_dict()
        selection = BackendSelection(
            prefill="int8_dequant_bf16_fallback",
            decode="int8_dequant_bf16_fallback",
        )
        phase_plan = build_backend_phase_plan(plan, selection)
        self.assertEqual(plan.as_dict(), original)
        self.assertEqual(
            phase_plan.as_dict()["execution_plan_schema_version"], 1
        )
        self.assertTrue(phase_plan.selection.decode_fallback_explicit)

    def test_unregistered_fused_provider_fails_phase_plan(self):
        config = tiny_config()
        policy = ExecutionPolicy(
            weight_format=WeightFormat.INT8_DEQUANT_BF16_FALLBACK,
            quantization=QuantizationSpec(
                bits=8,
                granularity="per_channel",
            ),
            vocab_chunk_bytes=4096,
        )
        plan = LlamaModelAdapter().build_execution_plan(config, policy)
        with self.assertRaises(BackendUnavailableError):
            build_backend_phase_plan(
                plan,
                BackendSelection(
                    prefill="fused_w8a16",
                    decode="int8_dequant_bf16_fallback",
                ),
            )


if __name__ == "__main__":
    unittest.main()
