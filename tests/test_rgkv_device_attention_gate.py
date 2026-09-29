import inspect
import unittest

import torch

from layer_streaming.attention.paged import PagedAttentionOutput
from layer_streaming.kv.runtime import PagedKVRuntime
from layer_streaming.kv_policy import KVPolicy


def _cpu_factory(shape, **kwargs):
    return torch.empty(shape, dtype=kwargs["dtype"], device="cpu")


class RGKVDeviceAttentionGateTest(unittest.TestCase):
    def make_runtime(self):
        page_size = 16
        logical_page_bytes = 2 * 1 * 1 * page_size * 4 * 2
        return PagedKVRuntime(
            layer_count=1,
            num_query_heads=2,
            num_kv_heads=1,
            head_dim=4,
            page_count=8,
            page_size=page_size,
            dtype=torch.float16,
            device="cpu",
            policy=KVPolicy(
                accuracy="sparse",
                storage="gpu_cpu",
                dtype="fp16",
                selection="rgkv",
                rgkv_scorer="torch_tensorized",
                page_budget=2,
                recent_window=16,
                attention_backend="reference_paged_exact",
                page_size=page_size,
                gpu_hot_budget_bytes=4 * logical_page_bytes,
                cpu_budget_bytes=8 * logical_page_bytes,
                gpu_high_watermark_bytes=4 * logical_page_bytes,
                gpu_low_watermark_bytes=3 * logical_page_bytes,
            ),
            allow_reference=True,
            tier_tensor_factory=_cpu_factory,
        )

    def test_1000_device_hit_launches_have_zero_bridge_metrics(self):
        runtime = self.make_runtime()
        state = runtime.create_request(32)
        key = torch.randn(32, 1, 4, dtype=torch.float16)
        runtime.append((state,), 0, key, key, (32,))
        query = torch.randn(1, 2, 4, dtype=torch.float16)
        batch = runtime.prepare_batch((state,), (1,), 0)
        selected = runtime.selection.select((state,), 0, query, batch)
        request_type = __import__(
            "layer_streaming.attention.paged",
            fromlist=["DevicePagedAttentionInput"],
        ).DevicePagedAttentionInput
        key_pool, value_pool = runtime.store.layer_view(0)
        request = request_type(
            query=query,
            key_pool_view=key_pool,
            value_pool_view=value_pool,
            batch_view=batch,
            page_size=runtime.page_size,
            num_query_heads=runtime.num_query_heads,
            num_kv_heads=runtime.num_kv_heads,
            head_dim=runtime.head_dim,
            softmax_scale=runtime.softmax_scale,
            causal=True,
            kv_dtype="fp16",
            output_dtype="fp16",
            selected_pages=selected,
        )

        def execute_device(current, *, phase):
            self.assertEqual(phase, "decode")
            return PagedAttentionOutput(
                output=current.query.clone(),
                logsumexp=None,
                provider_metrics={"workspace_bytes": 0},
            )

        runtime.dispatcher.execute_device = execute_device
        for _ in range(1000):
            guard = runtime.active_tier.begin_hot_cache_execution(
                request_id=state.request_id
            )
            result = runtime.execution._attend_active_tier_device_hit(
                state, 0, request, guard
            )
            self.assertEqual(result.provider_metrics["selected_metadata_d2h"], 0)
        runtime.active_tier.quiesce(request_id=state.request_id)
        profile = runtime.profile_stats()
        self.assertEqual(profile["gpu_hit_device_attention_calls"], 1000)
        self.assertEqual(profile["selected_metadata_d2h"], 0)
        self.assertEqual(profile["host_scalar_readbacks"], 0)
        self.assertEqual(profile["explicit_sync_count"], 0)
        self.assertEqual(profile["python_attention_wave_count"], 0)
        self.assertEqual(profile["device_view_fallback_count"], 0)
        self.assertEqual(profile["active_hot_cache_execution_guards"], 0)
        self.assertEqual(profile["total_ref_count"], profile["logical_owner_count"])
        runtime.close()

    def test_device_execution_integration_has_no_selected_host_bridge(self):
        source = inspect.getsource(
            __import__(
                "layer_streaming.kv.execution", fromlist=["KVExecutionCoordinator"]
            ).KVExecutionCoordinator._attend_active_tier_device_hit
        )
        for forbidden in (
            "resolve_entries(",
            "resolve_handles(",
            ".cpu(",
            ".numpy(",
            ".tolist(",
            ".item(",
            "synchronize(",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
