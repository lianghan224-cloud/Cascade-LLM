# Open-source Release Readiness

## Completed repository hygiene

- Runtime source, tests, tools, Docker definitions, dependency lock, and
  reviewed technical documentation are kept in the repository.
- Model checkpoints, virtual environments, compiled providers, credentials,
  local configuration, raw reports, traces, and real-result payloads are
  ignored.
- Maintainer-only agent instructions, internal handoff material, machine
  migration notes, and absolute-path documents have been removed.
- Public contribution, security, architecture, status, and documentation-index
  pages are present.

## Required before publishing

- [ ] The copyright holder selects an open-source license and adds `LICENSE`.
- [ ] The repository owner confirms the third-party notices and all included
      source provenance are complete.
- [ ] A maintainer reviews the staged diff for tokens, private prompts, model
      paths, host identifiers, and generated binaries.
- [x] Continuous integration runs the CPU unit test suite on pull requests.
- [ ] The project owner decides the public support and release-version policy.

Until the first item is complete, the repository is prepared for review but is
not licensed for public redistribution.
