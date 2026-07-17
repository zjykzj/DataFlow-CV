---
name: dev
description: Run development commands — test, lint, typecheck. Use when the user asks to run tests, lint, type check, or check test coverage.
allowed-tools: Bash
---

# Development Commands

Project-specific paths, test commands, and tool configurations are defined in CLAUDE.md via the `{{PACKAGE_NAME}}` and `{{SRC_DIRS}}` template variables. This skill uses those variables — adapt the commands below by substituting the actual values from CLAUDE.md.

## Testing

```bash
pytest                                  # All tests (from project root)
pytest -x -q                            # Stop on first failure, quiet
pytest {{SRC_DIRS}}/test_file.py        # Single test file
pytest {{SRC_DIRS}}/test_file.py::name  # Single test
pytest --cov={{PACKAGE_NAME}} --cov-report=html  # Coverage report
pytest -n auto                          # Parallel (requires pytest-xdist)
```

Additional project-specific test commands (Docker environments, dataset download scripts, etc.) are documented in CLAUDE.md.

## Linting & Formatting

```bash
# Format
black {{SRC_DIRS}}
isort {{SRC_DIRS}}

# Lint
flake8 {{SRC_DIRS}}

# Type check
mypy {{PACKAGE_NAME}}
```
