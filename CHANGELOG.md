# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.0] - 2026-05-11

### Added
- [`b587bb3`](https://github.com/cumbof/team/commit/b587bb3d1fe23f504e5368f10231c21bb6d0976a) Add CHANGELOG.md and bump version to 0.4.0
- [`4845edd`](https://github.com/cumbof/team/commit/4845edd5c9d0a8b90f7c5a81aca334ddb0ff47b3) Add manual PyPI publish GitHub Actions workflow (`workflow_dispatch`)

---

## [0.3.2] - 2026-05-11

### Fixed
- [`70d6bce`](https://github.com/cumbof/team/commit/70d6bce8c451743428c39d02573720ddcb681744) Fix GPU/macOS compatibility issues; add `--no-gpu` and `--host-ollama` CLI flags, `defaults.ollama_url`

### Documentation
- [`35a2821`](https://github.com/cumbof/team/commit/35a28214ad157ec38cefedf79acf8d21a452d387) Document GPU/macOS support, `--no-gpu`, and `--host-ollama` flags in README

---

## [0.3.1] - 2026-05-10

### Fixed
- [`6110f1c`](https://github.com/cumbof/team/commit/6110f1ce660ef0d6ff48146eaf6d800690375953) Fix minor formatting issues in README

---

## [0.3.0] - 2026-05-10

### Added
- [`dbad03b`](https://github.com/cumbof/team/commit/dbad03be95293d3d72231419c500bfeddf558208) Add 8 new features (F1–F3, F6–F10)
- [`f733f2d`](https://github.com/cumbof/team/commit/f733f2d1988420b8b397947985872daebca6b75a) Add agent tool-use / function-calling support (F4)
- [`70b503e`](https://github.com/cumbof/team/commit/70b503ec329a249c17c4902c67975ced6f57411a) Add custom skill plugins loadable from local files or remote URLs
- [`ea7e082`](https://github.com/cumbof/team/commit/ea7e08201daf9406462db9c92d1e15f82908c8bf) Add automatic workspace checkpoints before each member turn
- [`7cca4d2`](https://github.com/cumbof/team/commit/7cca4d220e86b921db0e96671a2ca7ffa9cde14e) Add `write_file`, `append_file`, `list_files` built-in tools and `Transcript.stats()`
- [`7f05178`](https://github.com/cumbof/team/commit/7f05178f84f6548dcfcc51eda6da02046109666b) Add cross-team collaboration via bridge protocol
- [`789db2f`](https://github.com/cumbof/team/commit/789db2f2dd1dded9c91dd63d93bdc20a835d5d9b) Add per-agent persistent memory, shared belief board, and team rollback
- [`f5a91b0`](https://github.com/cumbof/team/commit/f5a91b0a9d1d12ea215cb627d5d6bd5faf6d0bb6) Add predefined persona library with feature summary table in README
- [`c6cd319`](https://github.com/cumbof/team/commit/c6cd3192c800bd7ed35d8ce88e7682d8f4ed0670) Add full system access for agents and richer detailed personas
- [`6cc390c`](https://github.com/cumbof/team/commit/6cc390cf2d2c66bb36abac5a6d6d7f9a17ba1084) Add shared context file, decision log, and parallel review workflow

### Changed
- [`07346e4`](https://github.com/cumbof/team/commit/07346e45286831a8a89d77eef85b9955d51b2f19) Refactor persona storage to file-based library (`personas/*.yaml`)

### Documentation
- [`55a6c7d`](https://github.com/cumbof/team/commit/55a6c7df3615cc498587cceaa7f9a10e55316ed7) Add inline comments throughout the codebase for contributors
- [`027cdab`](https://github.com/cumbof/team/commit/027cdab01d4cd1b8cfb5ad95ae67ea80028d606e) Document workspace checkpoints and audit all features in README
- [`af61c96`](https://github.com/cumbof/team/commit/af61c966eb87a10e78e222f044cee9948a8c8cf1) Add AI-assisted development notice and LLM PR welcome note

---

## [0.2.0] - 2026-05-09

### Added
- [`e9cafa7`](https://github.com/cumbof/team/commit/e9cafa75e2e62ea2e5b19670193769fa80f98dbc) Add run-resume support via `--resume` flag
- [`884e03f`](https://github.com/cumbof/team/commit/884e03f59416e3be24cc07baf4dc82bebb8c4d06) Add token-by-token streaming output during live agent turns
- [`e26c4e5`](https://github.com/cumbof/team/commit/e26c4e5b40cce5ec009a39d4bfff4b92f6c107ec) Add `team export` command generating Markdown and HTML reports
- [`9d9278b`](https://github.com/cumbof/team/commit/9d9278b4de10e2d87b7bded1ad368a09ee59da88) Add `sequential_chain` workflow
- [`bb323ac`](https://github.com/cumbof/team/commit/bb323ac761bb044e3abf4620824d5a645f8fbab3) Surface private workspace files in member turn prompts
- [`dcf0ecc`](https://github.com/cumbof/team/commit/dcf0ecc5bb5093be72a12f0759697e5c72e36a6c) Add `team check` preflight command
- [`dcbd8b7`](https://github.com/cumbof/team/commit/dcbd8b79d696d848a7452ed9967fbd709e1c1e33) Add retry with exponential back-off on transient Ollama errors
- [`9e34354`](https://github.com/cumbof/team/commit/9e3435484be6573d849e22224851fee7454ec6ff) Add human-in-the-loop intervention during runs

---

## [0.1.0] - 2026-05-09

### Added
- [`06ba6f7`](https://github.com/cumbof/team/commit/06ba6f78d0cd1ff75705874e187f3a55e240e399) Initial project scaffold
- [`3e934d7`](https://github.com/cumbof/team/commit/3e934d7e744429df3f9813193acb807cb57627cf) Initial commit — core orchestration engine
