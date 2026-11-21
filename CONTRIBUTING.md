# Contributing to DataFlow-CV

Thanks for your interest in contributing! This project follows **Spec-Driven Development (SDD)** — the documents in [`specs/`](specs/) are the single source of truth, and any change that affects a documented contract updates the spec **before** the code.

## Development Setup

```bash
pip install -e .              # Editable install
pip install -e .[dev]         # With test/lint/type-check dependencies
pip install -e .[coco]        # With pycocotools for RLE + evaluation
```

## Workflow

1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. ✏️ Make your changes — **spec-first** (see below)
4. 🧪 Add or update tests as needed
5. ✅ Pass the quality gates (see below)
6. 📬 Submit a pull request

## Spec-First (SDD)

Any feature or fix that affects a contract documented in [`specs/`](specs/) must:

1. **Update the affected spec to the target state before implementing** (and bump the spec version per its own versioning rules — see `/maestro:spec`).
2. **Cross-file scan**: grep `CLAUDE.md`, `README.md`, and other specs for references to the changed contract — update those too.
3. **Verify the implementation against the spec after coding** (conformance check).
4. **List the affected spec files in the commit body.**

When a spec and the code disagree, **the spec wins** — fix the code (or go back and change the spec first, never diverge silently).

## Quality Gates

```bash
pytest                                       # All tests
ruff check dataflow tests samples            # Lint
ruff format --check dataflow tests samples   # Format check
mypy dataflow                                # Type check
```

## Conventions

- **Format ordering**: when YOLO, LabelMe, and COCO appear together in any listing, use `YOLO → LabelMe → COCO`.
- **Module ordering**: when Analyse, Convert, Visualize, and Evaluate appear together, use `Analyse → Convert → Visualize → Evaluate`.
- **Zero cross-dependencies**: Analyse/Convert/Visualize/Evaluate modules must not import from each other.
- **Commits**: follow conventional commit style (`feat:`, `fix:`, `docs:`, `chore:`).

## Documentation Conventions

- **Bilingual mirroring**: `README.md` ↔ `README.zh-CN.md` and `docs/en/` ↔ `docs/zh/` are mirrors — every change updates both sides in the same commit, and pages link only to pages of the same language.
- **Manual page types** (`docs/`): `index` (manual home + TOC), `cli` (complete command inventory — every subcommand and option), `api` (Python API reference), `faq` (common errors and fixes). When one type grows past ~3 pages, give it a subdirectory with its own `README.md` (e.g. `docs/en/pipeline/`).
- **README Quick Start stays slim**: concepts + the most common commands + pointers into the CLI reference; the end-to-end workflow is canonical and should not be trimmed.
- **Spec files**: every spec starts with a `> **Version:** vX.Y | **Last Updated:** YYYY-MM-DD` header. Minor bump = new definitions; major bump = behavioral change; date-only = wording fix.
- **CHANGELOG**: Keep a Changelog format — `## [Unreleased]` / `## [X.Y.Z] - date` sections with `### Added/Changed/Fixed` subsections and bold-lead bullets.
- **No private paths in public docs**: never reference absolute machine paths or private repositories from `README`/`docs/`/`specs/`.

## Architecture

For architecture constraints, coordinate-system semantics, and known gotchas, see [CLAUDE.md](CLAUDE.md).
