# Result artifact policy

Git tracks only qualification summaries, compatibility matrices, and content
manifests. Full per-token traces, raw tensors, benchmark samples, and large JSON
reports are CI or release artifacts and are intentionally ignored.

`manifest.json` records the relative path, byte size, and SHA-256 of every local
payload used to produce the summaries. The current KV result is under `kv_v2/`.
Its status is `performance_qualified`, not `production`, because L4 quality has
not run and L5 still records a 2 MiB CUDA-reserved drift in the 1000-cycle
ownership test.

Regenerate the content manifest with:

```bash
.venv/bin/python tools/build_result_manifest.py
```
