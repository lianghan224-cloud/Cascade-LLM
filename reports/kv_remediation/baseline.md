# KV Remediation Baseline

Captured before implementation changes on 2026-08-03.

- `.venv/bin/python -m unittest -v tests.test_kv_cache tests.test_kv_framework_v1 tests.contract.test_kv_framework_contract tests.test_fused_provider_dispatch`
  - 30 tests passed, including the applicable SM86 CUDA cases.
  - 1 module import failed because `tests/test_fused_provider_dispatch.py` imports `test_tiny_model_e2e` as a top-level module under this invocation.
  - Exit code: 1. Full output: `baseline_unittest.txt`.
- `.venv/bin/python -m pytest ...`
  - Could not start because pytest is not installed in the repository virtual environment.
  - Exit code: 1. Full output: `baseline_pytest.txt`.

Baseline failures were recorded and did not stop remediation. The unified validator uses direct executable scenarios and `unittest`, so it does not depend on pytest.
