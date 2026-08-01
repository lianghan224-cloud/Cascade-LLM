import unittest

import torch

from layer_streaming import (
    BackendInfo,
    BackendUnavailableError,
    BF16LinearBackend,
    FusedW8A8Backend,
    Int4DequantBF16FallbackBackend,
    Int4DequantFP16FallbackBackend,
    Int8DequantBF16FallbackBackend,
    Int8DequantFP16FallbackBackend,
    QuantizationSpec,
    WeightSpec,
    pack_int4,
    backend_capabilities,
    backend_for_weight,
    register_linear_backend,
    unregister_linear_backend,
)


class LinearBackendTest(unittest.TestCase):
    def test_dense_backend(self):
        backend = BF16LinearBackend()
        spec = WeightSpec.dense(
            "weight", (3, 4), "bf16", "attention_q"
        )
        backend.validate(spec, "bf16")
        x = torch.arange(8, dtype=torch.bfloat16).view(2, 4)
        weight = torch.arange(12, dtype=torch.bfloat16).view(3, 4)
        self.assertTrue(
            torch.equal(
                backend.execute(x, weight, None, None),
                torch.nn.functional.linear(x, weight),
            )
        )

    def test_int8_per_group_fp16_matches_explicit_dequant(self):
        quant = QuantizationSpec(
            bits=8,
            granularity="per_group",
            group_size=32,
            scale_dtype="fp16",
        )
        spec = WeightSpec.quantized(
            "weight", (2, 64), quant, "fp16", "mlp_up"
        )
        backend = Int8DequantFP16FallbackBackend()
        backend.validate(spec, "fp16")
        raw = torch.arange(-64, 64, dtype=torch.int8).view(2, 64)
        scale = torch.tensor([[0.5, 0.25], [0.125, 0.75]], dtype=torch.float16)
        actual = backend.dequantize(
            spec, raw, {"scale": scale, "weight_spec": spec}
        )
        expected = raw.to(torch.float16) * scale.repeat_interleave(32, dim=1)
        self.assertTrue(torch.equal(actual, expected))

    def test_int8_per_channel_accepts_fp16_scale_with_bf16_activation(self):
        quant = QuantizationSpec(
            bits=8,
            granularity="per_channel",
            scale_dtype="fp16",
        )
        spec = WeightSpec.quantized(
            "weight", (3, 4), quant, "bf16", "attention_o"
        )
        raw = torch.arange(-6, 6, dtype=torch.int8).view(3, 4)
        scale = torch.tensor([[0.5], [0.25], [0.125]], dtype=torch.float16)
        backend = Int8DequantBF16FallbackBackend()
        actual = backend.dequantize(
            spec, raw, {"scale": scale, "weight_spec": spec}
        )
        expected = raw.to(torch.bfloat16) * scale.to(torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))

    def test_int4_fallback_matches_explicit_dequant(self):
        quant = QuantizationSpec(
            bits=4,
            granularity="per_group",
            group_size=32,
            scale_dtype="bf16",
        )
        spec = WeightSpec.quantized(
            "weight", (2, 32), quant, "bf16", "mlp_down"
        )
        values = (torch.arange(64) % 16 - 8).to(torch.int8).view(2, 32)
        packed = pack_int4(values)
        scale = torch.tensor([[0.25], [0.5]], dtype=torch.bfloat16)
        backend = Int4DequantBF16FallbackBackend()
        workspace = torch.empty(64, dtype=torch.bfloat16)
        actual = backend.dequantize(
            spec,
            packed,
            {"scale": scale, "weight_spec": spec},
            workspace=workspace,
        )
        expected = values.to(torch.bfloat16) * scale
        self.assertTrue(torch.equal(actual, expected))

    def test_int4_fp16_backend_has_separate_name(self):
        quant = QuantizationSpec(
            bits=4,
            granularity="per_group",
            group_size=32,
            scale_dtype="fp16",
        )
        spec = WeightSpec.quantized(
            "weight", (1, 32), quant, "fp16", "mlp_gate"
        )
        backend = Int4DequantFP16FallbackBackend()
        backend.validate(spec, "fp16")
        self.assertEqual(backend.name, "int4_dequant_fp16_fallback")

    def test_reserved_fused_backend_never_silently_falls_back(self):
        with self.assertRaises(BackendUnavailableError):
            FusedW8A8Backend().validate(None, "bf16")

    def test_explicit_provider_registration_and_removal(self):
        class Provider:
            name = "fused_w8a16"
            is_fallback = False
            info = BackendInfo(
                name=name,
                storage_dtype="int8",
                activation_dtype="bfloat16",
                output_dtype="bfloat16",
                requires_dequant=False,
                is_fallback=False,
                supported_gpu_architectures=("sm80+",),
                atol=0.02,
                rtol=0.02,
            )

            def validate(self, weight, input_dtype):
                if weight.quantization.bits != 8 or input_dtype != "bfloat16":
                    raise ValueError("provider format mismatch")

            def transfer_bytes(self, weight):
                return weight.storage_nbytes

            def workspace_bytes(self, weight, batch_tokens):
                del weight, batch_tokens
                return 0

            def execute(self, x, weight_view, quant_views, workspace):
                del weight_view, quant_views, workspace
                return x

        spec = WeightSpec.quantized(
            "weight",
            (2, 32),
            QuantizationSpec(
                bits=8,
                granularity="per_group",
                group_size=32,
            ),
            "bf16",
            "attention_q",
        )
        try:
            provider = register_linear_backend(Provider())
            self.assertIs(
                backend_for_weight(spec, backend_name="fused_w8a16"),
                provider,
            )
            self.assertTrue(
                backend_capabilities()["fused_w8a16"]["available"]
            )
        finally:
            unregister_linear_backend("fused_w8a16")
        with self.assertRaises(BackendUnavailableError):
            backend_for_weight(spec, backend_name="fused_w8a16")


if __name__ == "__main__":
    unittest.main()
