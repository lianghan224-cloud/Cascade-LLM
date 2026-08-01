from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

from layer_streaming import (
    CORE_API_VERSION,
    CheckpointManifest,
    ExecutionPlan,
    ExecutionPolicy,
    LlamaModelAdapter,
    ModelGeometry,
    QuantizationSpec,
    RunReport,
    TensorManifest,
    WeightSpec,
    core_api_contract,
    core_api_contract_sha256,
)
from test_plan import tiny_config


FIXTURE = Path(__file__).parent / "fixtures" / "core_api_v1.json"
EXPECTED_SHA256 = "a699bcda404c8f3eeb9a4b297373dd3e8434c305f126acb4b3dc14a12ff70325"


class CoreApiContractTest(unittest.TestCase):
    def test_public_contract_matches_v1_fixture(self):
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(CORE_API_VERSION, 1)
        self.assertEqual(core_api_contract(), expected)
        self.assertEqual(core_api_contract_sha256(), EXPECTED_SHA256)

    def test_frozen_specs_round_trip(self):
        geometry = LlamaModelAdapter().build_geometry(tiny_config())
        self.assertEqual(
            ModelGeometry.from_dict(geometry.as_dict()), geometry
        )
        quantization = QuantizationSpec(
            bits=4,
            granularity="per_group",
            group_size=32,
            scale_dtype="fp16",
        )
        weight = WeightSpec.quantized(
            "weight",
            (3, 64),
            quantization,
            "bf16",
            "attention_q",
        )
        self.assertEqual(WeightSpec.from_dict(weight.as_dict()), weight)
        with self.assertRaises(FrozenInstanceError):
            weight.alignment = 128

    def test_checkpoint_manifest_round_trip(self):
        tensor = TensorManifest(
            name="weight",
            shape=(2, 3),
            dtype="bfloat16",
            file="model.safetensors",
            data_offsets=(0, 12),
        )
        manifest = CheckpointManifest(
            root="/checkpoint",
            files=("model.safetensors",),
            tensors={"weight": tensor},
            aliases={"lm_head.weight": "weight"},
            total_bytes=128,
            errors=(),
            weight_map={"weight": "model.safetensors"},
        )
        restored = CheckpointManifest.from_dict(
            json.loads(json.dumps(manifest.as_dict()))
        )
        self.assertEqual(restored, manifest)

    def test_execution_plan_round_trip_is_deterministic(self):
        adapter = LlamaModelAdapter()
        policy = ExecutionPolicy(vocab_chunk_bytes=4096)
        first = adapter.build_execution_plan(tiny_config(), policy)
        second = adapter.build_execution_plan(tiny_config(), policy)
        serialized = json.loads(json.dumps(first.as_dict()))
        restored = ExecutionPlan.from_dict(serialized)
        self.assertEqual(restored.as_dict(), first.as_dict())
        self.assertEqual(first.as_dict(), second.as_dict())
        serialized["schema_version"] = 999
        with self.assertRaisesRegex(ValueError, "schema version"):
            ExecutionPlan.from_dict(serialized)

    def test_run_report_v2_round_trip_and_version_guard(self):
        report = RunReport(
            model={},
            runtime_config={},
            timings={},
            throughput={},
            memory={},
            pipeline={},
            checkpoint_validation={},
            memory_preflight={},
            hardware={},
            created_at_unix=1.0,
        )
        restored = RunReport.from_json(report.to_json())
        self.assertEqual(restored, report)
        payload = report.as_dict()
        payload["schema_version"] = 3
        with self.assertRaisesRegex(ValueError, "schema version"):
            RunReport.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
