# Contributing to Cascade-LLM

Thanks for contributing. Cascade-LLM is an experimental systems project, so a
small, reproducible change is more valuable than an unverified performance
claim.

## Before opening a pull request

1. Keep model weights, credentials, host-specific reports, compiled CUDA
   binaries, and virtual environments out of Git.
2. Preserve explicit error and fallback behavior. Do not turn an unavailable
   provider or unqualified path into a silent fallback.
3. Add or update focused tests for changed behavior.
4. Run the relevant unit tests and `git diff --check`.
5. State the environment, backend, provider, and evidence class for any CUDA
   or performance result.

## Development setup

Use Python 3.10 and the locked dependencies:

```bash
CASCADE_PYTHON_BIN=python3.10 bash scripts/bootstrap_server.sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -q
```

GPU validation is architecture-specific. Run generic CUDA correctness checks
before selecting an architecture-specific provider, and do not generalize a
result across compute capabilities.

## Pull request scope

Keep one behavioral purpose per pull request. Describe the user-visible change,
tests run, tests not run, and any known qualification gap. Do not commit raw
benchmark traces; summarize reviewed evidence in documentation instead.
