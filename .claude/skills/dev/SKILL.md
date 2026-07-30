---
name: dev
description: Run development commands — test, lint, typecheck (Python projects only). Use when the user asks to run tests, lint, type check, or check test coverage.
allowed-tools: Bash
---

# Development Commands

Project-specific paths, test commands, and tool configurations are defined in CLAUDE.md via the `{{PACKAGE_NAME}}` and `{{SRC_DIRS}}` template variables. This skill uses those variables — adapt the commands below by substituting the actual values from CLAUDE.md.

## Scope: Python Projects Only

This skill covers the Python toolchain (pytest / black / isort / flake8 / mypy). Before running any command, confirm the project is Python — `pyproject.toml`, `setup.py`, or `setup.cfg` exists at the project root. If none exists, **stop and tell the user** this skill only supports Python projects; do not guess equivalent commands for other ecosystems (npm test, go test, etc.).

## Execution Protocol

1. **Run only what was asked** — tests, lint, or typecheck. Do not fire the full battery on every invocation.
2. **Detect toolchain** — before lint/format: `grep -c "\[tool.ruff\]" pyproject.toml 2>/dev/null || echo 0`. Ruff detected → use ruff path; otherwise → legacy stack.
3. **Check availability first** for each tool about to run: `command -v <tool>`.
4. **Missing tool → report, then degrade** — tell the user explicitly (e.g., "mypy is not installed — skipping typecheck; install with `pip install -e .[dev]`"), then continue with the available tools. Never skip silently, never abort the whole run because one tool is missing.
5. **Check config expectations** — a tool that runs is not necessarily running with the project's standards (see Tool Requirements table). If a tool falls back to defaults that conflict with project config, mention it when reporting results.

## Testing

```bash
pytest                                  # All tests (from project root)
pytest -x -q                            # Stop on first failure, quiet
pytest path/to/test_file.py             # Single test file
pytest path/to/test_file.py::test_name  # Single test
pytest --cov={{PACKAGE_NAME}} --cov-report=html  # Coverage report
pytest -n auto                          # Parallel (requires pytest-xdist)
```

Additional project-specific test commands (Docker environments, dataset download scripts, etc.) are documented in CLAUDE.md.

## Toolchain Detection

Before running lint/format commands, detect which stack the project uses:

```bash
grep -c "\[tool.ruff\]" pyproject.toml 2>/dev/null || echo 0
```

- **Output > 0** → Ruff stack (unified lint + format)
- **Output = 0** → Legacy stack (black + isort + flake8)

## Linting & Formatting

### Ruff (preferred if `[tool.ruff]` detected)

```bash
ruff check {{SRC_DIRS}}            # Lint (replaces flake8 + isort)
ruff format --check {{SRC_DIRS}}   # Format check (replaces black)
ruff format {{SRC_DIRS}}           # Format (apply)
```

### Legacy (black + isort + flake8)

```bash
# Format
black {{SRC_DIRS}}
isort {{SRC_DIRS}}

# Lint
flake8 {{SRC_DIRS}}
```

### Type Check (both stacks)

```bash
mypy {{PACKAGE_NAME}}
```

## Tool Requirements

Tools are typically declared in the project's `[dev]` extras — install with `pip install -e .[dev]`. Install alone is not always enough:

| Tool | Stack | Install alone enough? | Config it reads | Behavior without config |
|------|-------|----------------------|-----------------|------------------------|
| pytest | Both | ✅ Yes | `pyproject.toml [tool.pytest.ini_options]` | Auto-discovers `test_*.py` |
| ruff | Ruff | ✅ Yes | `pyproject.toml [tool.ruff]` | Uses built-in rules, no formatting config needed |
| black | Legacy | ✅ Yes | `pyproject.toml [tool.black]` | Defaults to 88-col line length |
| isort | Legacy | ⚠️ Needs config | `pyproject.toml [tool.isort]` | May fight black's formatting — set `profile = "black"` |
| flake8 | Legacy | ⚠️ Needs config | `.flake8` / `setup.cfg` / `tox.ini` — does **NOT** read `pyproject.toml` | Defaults to 79-col limit, which conflicts with black's 88/100 — expect E501 noise |
| mypy | Both | ✅ Yes | `pyproject.toml [tool.mypy]` | Lenient defaults |
| pytest-xdist | Both | Optional extra | — | `pytest -n auto` unavailable |

## Required Configuration

Define these variables in the project's CLAUDE.md (Development Configuration section):

| Variable | Purpose | Example |
|----------|---------|---------|
| `{{PACKAGE_NAME}}` | Package for coverage / type check | `dataflow` |
| `{{SRC_DIRS}}` | Space-separated dirs for format / lint | `dataflow tests samples` |
