# KV Framework V1 raw results

These JSON files are reproducible local qualification evidence. They do not
contain model weights or credentials.

Primary acceptance artifacts:

- `ablation_sm86_full.json`: 36-case Generic/SM86/reference matrix.
- `ablation_sm86_4k_workspace.json`: 4K workspace and latency check.
- `real8b_sm86_multipage.json`: current 8B SM86 direct-paged result.
- `real8b_legacy_gather_lm_head_ablation.json`: streamed LM Head isolation.
- `real8b_legacy_gather_resident_lm_head_ablation.json`: resident LM Head isolation.
- `tinyllama_real_generic_cuda.json`: second real GQA checkpoint comparison.
- `tinyllama_run_report.json`: user-facing real generation report.
- `soak_1000_tokens.json` and `soak_100_load_cycles.json`: synthetic soak evidence.

Supplementary diagnostic artifacts:

- `ablation_sm86_smoke.json`
- `real8b_generic_cuda_multipage.json`
- `real8b_generic_cuda_twopass.json`
- `real8b_reference_paged_single_page.json`

The supplementary files record intermediate attribution experiments and must
not be treated as the final acceptance result. See
`docs/KV_FRAMEWORK_V1_VALIDATION.md` for the interpreted results and explicit
qualification gaps.
