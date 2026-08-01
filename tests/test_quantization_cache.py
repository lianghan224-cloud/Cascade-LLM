from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from layer_streaming.plugin_builtins import Int8PerChannelQuantizerPlugin
from layer_streaming.quantization_cache import (
    checkpoint_digest,
    quantization_cache_key,
)


def tiny_config():
    return {
        "model_type": "llama",
        "_name_or_path": "tiny-cache-test",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
    }


class QuantizationCacheTest(unittest.TestCase):
    def test_checkpoint_digest_and_key_are_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            weight = root / "model.safetensors"
            weight.write_bytes(b"first")
            first = checkpoint_digest(root, chunk_bytes=2)
            weight.write_bytes(b"second")
            second = checkpoint_digest(root, chunk_bytes=2)
            self.assertNotEqual(first, second)
            key, inputs = quantization_cache_key(
                second,
                {"hidden_size": 16},
                "int8_symmetric_per_channel",
                None,
                "bf16",
                "row_major",
                2,
                "0.1.0",
            )
            self.assertTrue(key.startswith("sha256-"))
            self.assertEqual(inputs["provider_abi"], 2)

    def test_quantizer_uses_and_reuses_default_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(
                json.dumps(tiny_config()), encoding="utf-8"
            )
            (checkpoint / "model.safetensors").write_bytes(b"weights")
            cache = root / "cache"

            def fake_run(_name, arguments):
                output = Path(
                    arguments[arguments.index("--output") + 1]
                )
                output.mkdir(parents=True)
                return 0

            plugin = Int8PerChannelQuantizerPlugin()
            arguments = ["--input", str(checkpoint), "--scale-dtype", "bf16"]
            with mock.patch.dict(
                os.environ, {"CASCADE_CACHE_ROOT": str(cache)}
            ), mock.patch(
                "layer_streaming.cli.run_tool", side_effect=fake_run
            ) as run:
                self.assertEqual(plugin.run(arguments), 0)
                self.assertEqual(run.call_count, 1)
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(plugin.run(arguments), 0)
                self.assertEqual(run.call_count, 1)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["cache_hit"])
            self.assertTrue((Path(payload["output"]) / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
