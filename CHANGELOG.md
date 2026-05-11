# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.0] - 2026-05-11

### Added
- Manual PyPI publish GitHub Actions workflow (`workflow_dispatch`)

### Fixed
- GPU/macOS compatibility issues; added `--no-gpu` and `--host-ollama` CLI flags, `defaults.ollama_url`

### Documentation
- Documented GPU/macOS support, `--no-gpu`, and `--host-ollama` flags in README

---

## [0.3.0] - 2026-05-10

### Added
- 8 new features (F1–F3, F6–F10)
- Agent tool-use / function-calling support (F4)
- Custom skill plugins loadable from local files or remote URLs
- Automatic workspace checkpoints before each member turn
- `write_file`, `append_file`, `list_files` built-in tools and `Transcript.stats()`
- Cross-team collaboration via bridge protocol
- Per-agent persistent memory, shared belief board, and team rollback
- Predefined persona library with feature summary table in README
- Full system access for agents and richer detailed personas
- Shared context file, decision log, and parallel review workflow

### Changed
- Refactored persona storage to file-based library (`personas/*.yaml`)

### Documentation
- Added inline comments throughout the codebase for contributors
- Documented workspace checkpoints and audited all features in README

### Fixed
- Minor formatting issues in README

---

## [0.2.0] - 2026-05-09

### Added
- Token-by-token streaming output during live agent turns
- Run-resume support via `--resume` flag
- `team export` command generating Markdown and HTML reports
- `sequential_chain` workflow
- Surface private workspace files in member turn prompts
- `team check` preflight command
- Retry with exponential back-off on transient Ollama errors
- Human-in-the-loop intervention during runs

---

## [0.1.0] - 2026-05-09

### Added
- Initial project setup and core orchestration engine
