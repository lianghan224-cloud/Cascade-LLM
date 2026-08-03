# KV Stack Validation

- Profile: `full`
- Status: `PASS`
- Seed: `20260803`
- Duration: `1.762s`
- Summary: PASS=69, FAIL=0, SKIPPED_WITH_REASON=8, BLOCKED=0

| ID | Validation | Status | Duration (s) | Reason / key metrics |
|---|---|---:|---:|---|
| V00 | pre-modification snapshot | PASS | 0.0000 | {"missing": [], "snapshots": ["preflight_git_status.txt", "preflight_worktree.patch", "preflight_index.patch", "preflight_untracked_files.txt", "preflight_head.txt"]} |
| V01 | existing weight-free baseline | PASS | 0.0000 | baseline records include pre-existing failures; V01 requires recording, not a clean baseline |
| V02 | allocate/release closure | PASS | 0.0223 | {"concurrent_cycles": 2000, "free_pages": 8, "minimized_repro": ["unpin"], "random_operations": 100002, "seed": 20260803} |
| V03 | non-negative ref_count | PASS | 0.0223 | {"concurrent_cycles": 2000, "free_pages": 8, "minimized_repro": ["unpin"], "random_operations": 100002, "seed": 20260803} |
| V04 | non-negative pin_count | PASS | 0.0223 | {"concurrent_cycles": 2000, "free_pages": 8, "minimized_repro": ["unpin"], "random_operations": 100002, "seed": 20260803} |
| V05 | double release/unpin rejection | PASS | 0.0223 | {"concurrent_cycles": 2000, "free_pages": 8, "minimized_repro": ["unpin"], "random_operations": 100002, "seed": 20260803} |
| V06 | generation and ABA rejection | PASS | 0.0223 | {"concurrent_cycles": 2000, "free_pages": 8, "minimized_repro": ["unpin"], "random_operations": 100002, "seed": 20260803} |
| V07 | bounded quiescence | PASS | 0.0223 | {"concurrent_cycles": 2000, "free_pages": 8, "minimized_repro": ["unpin"], "random_operations": 100002, "seed": 20260803} |
| V08 | request cancellation cleanup | PASS | 0.0223 | {"concurrent_cycles": 2000, "free_pages": 8, "minimized_repro": ["unpin"], "random_operations": 100002, "seed": 20260803} |
| V09 | simulated OOM recovery | PASS | 0.0223 | {"concurrent_cycles": 2000, "free_pages": 8, "minimized_repro": ["unpin"], "random_operations": 100002, "seed": 20260803} |
| V10 | simulated kernel failure recovery | PASS | 0.0223 | {"concurrent_cycles": 2000, "free_pages": 8, "minimized_repro": ["unpin"], "random_operations": 100002, "seed": 20260803} |
| V11 | IO submit failure rollback | PASS | 0.0223 | {"concurrent_cycles": 2000, "free_pages": 8, "minimized_repro": ["unpin"], "random_operations": 100002, "seed": 20260803} |
| V12 | IO completion failure rollback | PASS | 0.0223 | {"concurrent_cycles": 2000, "free_pages": 8, "minimized_repro": ["unpin"], "random_operations": 100002, "seed": 20260803} |
| V13 | concurrent allocate/fork/free | PASS | 0.0223 | {"concurrent_cycles": 2000, "free_pages": 8, "minimized_repro": ["unpin"], "random_operations": 100002, "seed": 20260803} |
| V14 | 100k randomized lifecycle operations | PASS | 0.0223 | {"concurrent_cycles": 2000, "free_pages": 8, "minimized_repro": ["unpin"], "random_operations": 100002, "seed": 20260803} |
| V15 | minimal reproduction reduction | PASS | 0.0223 | {"concurrent_cycles": 2000, "free_pages": 8, "minimized_repro": ["unpin"], "random_operations": 100002, "seed": 20260803} |
| V16 | Fork sharing | PASS | 0.0029 | {"cow_isolated": true, "partial_commit_length": 23, "prefix_evicted_pages": 2, "rollback_length": 3, "seed": 20260803} |
| V17 | COW isolation | PASS | 0.0029 | {"cow_isolated": true, "partial_commit_length": 23, "prefix_evicted_pages": 2, "rollback_length": 3, "seed": 20260803} |
| V18 | Prefix ownership | PASS | 0.0029 | {"cow_isolated": true, "partial_commit_length": 23, "prefix_evicted_pages": 2, "rollback_length": 3, "seed": 20260803} |
| V19 | Prefix eviction with active owner | PASS | 0.0029 | {"cow_isolated": true, "partial_commit_length": 23, "prefix_evicted_pages": 2, "rollback_length": 3, "seed": 20260803} |
| V20 | Beam fork | PASS | 0.0029 | {"cow_isolated": true, "partial_commit_length": 23, "prefix_evicted_pages": 2, "rollback_length": 3, "seed": 20260803} |
| V21 | Beam branch exit | PASS | 0.0029 | {"cow_isolated": true, "partial_commit_length": 23, "prefix_evicted_pages": 2, "rollback_length": 3, "seed": 20260803} |
| V22 | Speculative commit | PASS | 0.0029 | {"cow_isolated": true, "partial_commit_length": 23, "prefix_evicted_pages": 2, "rollback_length": 3, "seed": 20260803} |
| V23 | Speculative partial commit | PASS | 0.0029 | {"cow_isolated": true, "partial_commit_length": 23, "prefix_evicted_pages": 2, "rollback_length": 3, "seed": 20260803} |
| V24 | Speculative rollback | PASS | 0.0029 | {"cow_isolated": true, "partial_commit_length": 23, "prefix_evicted_pages": 2, "rollback_length": 3, "seed": 20260803} |
| V25 | cross-page rollback | PASS | 0.0029 | {"cow_isolated": true, "partial_commit_length": 23, "prefix_evicted_pages": 2, "rollback_length": 3, "seed": 20260803} |
| V26 | in-page append/rollback | PASS | 0.0029 | {"cow_isolated": true, "partial_commit_length": 23, "prefix_evicted_pages": 2, "rollback_length": 3, "seed": 20260803} |
| V27 | Quest index build | PASS | 0.0001 | {"candidate_count": 5, "max_score_error": 0.25, "recall": 1.0, "selected_count": 2, "serialized_bytes": 395} |
| V28 | Quest full selection | PASS | 0.0001 | {"candidate_count": 5, "max_score_error": 0.25, "recall": 1.0, "selected_count": 2, "serialized_bytes": 395} |
| V29 | Quest full reference equivalence | PASS | 0.0001 | {"candidate_count": 5, "max_score_error": 0.25, "recall": 1.0, "selected_count": 2, "serialized_bytes": 395} |
| V30 | Quest budget/top-k | PASS | 0.0001 | {"candidate_count": 5, "max_score_error": 0.25, "recall": 1.0, "selected_count": 2, "serialized_bytes": 395} |
| V31 | Quest sparse statistics | PASS | 0.0001 | {"candidate_count": 5, "max_score_error": 0.25, "recall": 1.0, "selected_count": 2, "serialized_bytes": 395} |
| V32 | Quest append update | PASS | 0.0001 | {"candidate_count": 5, "max_score_error": 0.25, "recall": 1.0, "selected_count": 2, "serialized_bytes": 395} |
| V33 | Quest Fork record sharing | PASS | 0.0001 | {"candidate_count": 5, "max_score_error": 0.25, "recall": 1.0, "selected_count": 2, "serialized_bytes": 395} |
| V34 | Quest COW record isolation | PASS | 0.0001 | {"candidate_count": 5, "max_score_error": 0.25, "recall": 1.0, "selected_count": 2, "serialized_bytes": 395} |
| V35 | Quest rollback | PASS | 0.0001 | {"candidate_count": 5, "max_score_error": 0.25, "recall": 1.0, "selected_count": 2, "serialized_bytes": 395} |
| V36 | Quest stale-version rejection | PASS | 0.0001 | {"candidate_count": 5, "max_score_error": 0.25, "recall": 1.0, "selected_count": 2, "serialized_bytes": 395} |
| V37 | Quest serialization round trip | PASS | 0.0001 | {"candidate_count": 5, "max_score_error": 0.25, "recall": 1.0, "selected_count": 2, "serialized_bytes": 395} |
| V38 | Quest boundary cases | PASS | 0.0001 | {"candidate_count": 5, "max_score_error": 0.25, "recall": 1.0, "selected_count": 2, "serialized_bytes": 395} |
| V39 | logical block to location mapping | PASS | 0.0342 | {"authoritative_unique": true, "cancellations": 1, "deduplicated": 1, "failures": 2, "migrations": 7, "roundtrip_bytes": 128} |
| V40 | Mock GPU to CPU migration | PASS | 0.0342 | {"authoritative_unique": true, "cancellations": 1, "deduplicated": 1, "failures": 2, "migrations": 7, "roundtrip_bytes": 128} |
| V41 | Mock CPU to SSD migration | PASS | 0.0342 | {"authoritative_unique": true, "cancellations": 1, "deduplicated": 1, "failures": 2, "migrations": 7, "roundtrip_bytes": 128} |
| V42 | Mock SSD/CPU/GPU round trip | PASS | 0.0342 | {"authoritative_unique": true, "cancellations": 1, "deduplicated": 1, "failures": 2, "migrations": 7, "roundtrip_bytes": 128} |
| V43 | unique authoritative copy | PASS | 0.0342 | {"authoritative_unique": true, "cancellations": 1, "deduplicated": 1, "failures": 2, "migrations": 7, "roundtrip_bytes": 128} |
| V44 | read during migration | PASS | 0.0342 | {"authoritative_unique": true, "cancellations": 1, "deduplicated": 1, "failures": 2, "migrations": 7, "roundtrip_bytes": 128} |
| V45 | migration failure rollback | PASS | 0.0342 | {"authoritative_unique": true, "cancellations": 1, "deduplicated": 1, "failures": 2, "migrations": 7, "roundtrip_bytes": 128} |
| V46 | pinned eviction exclusion | PASS | 0.0342 | {"authoritative_unique": true, "cancellations": 1, "deduplicated": 1, "failures": 2, "migrations": 7, "roundtrip_bytes": 128} |
| V47 | prefetch deduplication | PASS | 0.0342 | {"authoritative_unique": true, "cancellations": 1, "deduplicated": 1, "failures": 2, "migrations": 7, "roundtrip_bytes": 128} |
| V48 | prefetch cancellation cleanup | PASS | 0.0342 | {"authoritative_unique": true, "cancellations": 1, "deduplicated": 1, "failures": 2, "migrations": 7, "roundtrip_bytes": 128} |
| V49 | tier capacity handling | PASS | 0.0342 | {"authoritative_unique": true, "cancellations": 1, "deduplicated": 1, "failures": 2, "migrations": 7, "roundtrip_bytes": 128} |
| V50 | reload after eviction | PASS | 0.0342 | {"authoritative_unique": true, "cancellations": 1, "deduplicated": 1, "failures": 2, "migrations": 7, "roundtrip_bytes": 128} |
| V51 | atomic metadata commit | PASS | 0.0342 | {"authoritative_unique": true, "cancellations": 1, "deduplicated": 1, "failures": 2, "migrations": 7, "roundtrip_bytes": 128} |
| V52 | selection/prefetch/view end-to-end | PASS | 0.0639 | {"cancelled": 1, "compute_failures": 1, "requests": 7, "served": 7, "wait_ms": 303.0077829025686} |
| V53 | compute pin lifecycle | PASS | 0.0639 | {"cancelled": 1, "compute_failures": 1, "requests": 7, "served": 7, "wait_ms": 303.0077829025686} |
| V54 | multi-request prefetch fairness | PASS | 0.0639 | {"cancelled": 1, "compute_failures": 1, "requests": 7, "served": 7, "wait_ms": 303.0077829025686} |
| V55 | cancellation propagation | PASS | 0.0639 | {"cancelled": 1, "compute_failures": 1, "requests": 7, "served": 7, "wait_ms": 303.0077829025686} |
| V56 | Full Prefill routing | PASS | 0.0042 | {"architecture": "cpu", "attention_backend": "reference_paged_exact", "fallback_reason": "decode_suffix_reference has no full_prefill kernel; using correctness prefill fallback", "is_reference": true, "kv_kernel_backend": "torch_paged_kv", "requested": "dec... |
| V57 | Decode routing | PASS | 0.0042 | {"architecture": "cpu", "attention_backend": "decode_suffix_reference", "fallback_reason": null, "is_reference": true, "kv_kernel_backend": "torch_paged_kv", "requested": "decode_suffix_reference", "selected": "decode_suffix_reference", "supported": true, "... |
| V58 | Chunked Prefill routing | PASS | 0.0042 | {"architecture": "cpu", "attention_backend": "reference_paged_exact", "fallback_reason": "decode_suffix_reference has no chunked_prefill kernel; using correctness prefill fallback", "is_reference": true, "kv_kernel_backend": "torch_paged_kv", "requested": "... |
| V59 | correct fallback without dedicated kernel | PASS | 0.0042 | {"chunked_fallback": "decode_suffix_reference has no chunked_prefill kernel; using correctness prefill fallback", "full_fallback": "decode_suffix_reference has no full_prefill kernel; using correctness prefill fallback", "short_suffix_selected": "decode_suf... |
| V60 | synthetic Decode numerics | PASS | 0.0700 | {"architecture": "sm86", "device": "NVIDIA GeForce RTX 3080 Ti", "latency_ms": 17.25373207591474, "max_abs_error": 0.0026137828826904297, "provider": "sm86", "seed": 20260803} |
| V61 | synthetic Full Prefill numerics | PASS | 0.0700 | {"architecture": "sm86", "decision": {"architecture": "sm86", "attention_backend": "reference_paged_exact", "fallback_reason": "sm86 has no full_prefill kernel; using correctness prefill fallback", "is_reference": true, "kv_kernel_backend": "sm86_kv_kernel"... |
| V62 | synthetic Chunked Prefill numerics | PASS | 0.0700 | {"architecture": "sm86", "device": "NVIDIA GeForce RTX 3080 Ti", "max_abs_error": 0.003809213638305664, "provider": "sm86", "seed": 20260803} |
| V63 | multi-stream race | PASS | 0.0700 | {"architecture": "sm86", "device": "NVIDIA GeForce RTX 3080 Ti", "provider": "sm86", "repetitions": 8, "seed": 20260803, "streams": 2} |
| V64 | CUDA allocated-memory drift | PASS | 0.0700 | {"allocated_drift_bytes": 0, "architecture": "sm86", "device": "NVIDIA GeForce RTX 3080 Ti", "iterations": 50, "provider": "sm86", "seed": 20260803} |
| V65 | CUDA reserved-memory drift | PASS | 0.0700 | {"architecture": "sm86", "device": "NVIDIA GeForce RTX 3080 Ti", "iterations": 50, "provider": "sm86", "reserved_drift_bytes": 0, "seed": 20260803} |
| V66 | CUDA kernel failure recovery | PASS | 0.0700 | {"architecture": "sm86", "device": "NVIDIA GeForce RTX 3080 Ti", "final_pin_count": 0, "provider": "sm86", "recovered": true, "seed": 20260803} |
| V67 | current-hardware performance baseline | PASS | 0.0700 | {"architecture": "sm86", "attention_calls": 63, "decode_latency_ms": 17.25373207591474, "device": "NVIDIA GeForce RTX 3080 Ti", "drift_loop_ms": 21.617386024445295, "provider": "sm86", "seed": 20260803} |
| V68 | real-model logits/Top-1 | SKIPPED_WITH_REASON | 0.0000 | CASCADE_KV_MODEL_PATH is not configured with real model weights |
| V69 | real 8B long generation | SKIPPED_WITH_REASON | 0.0000 | CASCADE_KV_MODEL_PATH is not configured with real model weights |
| V70 | real long-context Prefill | SKIPPED_WITH_REASON | 0.0000 | CASCADE_KV_MODEL_PATH is not configured with real model weights |
| V71 | real Quest accuracy | SKIPPED_WITH_REASON | 0.0000 | CASCADE_KV_DATASET_PATH is not configured with an accuracy dataset |
| V72 | real NVMe throughput/latency | SKIPPED_WITH_REASON | 0.0000 | CASCADE_KV_NVME_PATH is not configured for dedicated NVMe/GDS validation |
| V73 | real IO/compute overlap | SKIPPED_WITH_REASON | 0.0000 | CASCADE_KV_NVME_PATH is not configured for dedicated NVMe/GDS validation |
| V74 | SM86 specialized path validation | PASS | 0.0000 | SM86 synthetic specialized path and current-hardware baseline passed; this is not a production performance claim |
| V75 | other-architecture tuning | SKIPPED_WITH_REASON | 0.0000 | only SM86 devices are present; other architecture-specific tuning requires its target GPU |
| V76 | large dynamic batch | SKIPPED_WITH_REASON | 0.0000 | real weights and CASCADE_KV_SERVING_COMMAND are not configured |

## Hardware follow-up commands

```bash
.venv/bin/python tools/validate_kv_stack.py --profile cuda-synthetic
CASCADE_KV_MODEL_PATH=/path/to/model .venv/bin/python tools/validate_kv_stack.py --profile full
CASCADE_KV_NVME_PATH=/path/on/dedicated/nvme .venv/bin/python tools/validate_kv_stack.py --profile full
```

`SKIPPED_WITH_REASON` is used only for absent weights, datasets, NVMe configuration, or target hardware. A missing implementation is never converted to a skip.
