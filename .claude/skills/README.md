# Skills Usage Guide

This directory contains reusable Claude Code skills. Each skill is designed to be **project-independent** and can be copied to other projects.

## Available Skills

| Skill | Purpose | Needs Config? |
|-------|---------|---------------|
| `/commit` | Git commit message format | ✅ Yes — AI model name/email |
| `/dev` | Development commands (test, lint, typecheck) — Python projects only. Auto-detects ruff vs black+isort+flake8 | ✅ Yes — Package name, source dirs |
| `/release` | Version bump and GitHub release | ✅ Yes — Version locations, repo URL |
| `/spec` | Spec authoring methodology + SDD workflow (spec-first → implement → conformance check) | ⚠️ Optional — SDD hard rules in CLAUDE.md + enforcement hook (see `CONFIGURATION.md` §3–4) |
| `/claude` | CLAUDE.md authoring guidelines | ❌ No — Pure methodology |

Each skill documents its own needs in the `## Required Configuration` section at the end of its `SKILL.md`. All skills are language-agnostic except `/dev`, which targets the Python toolchain (pytest / black / isort / flake8 / mypy) and declines gracefully in non-Python projects.

## Quick Start for New Projects

### Step 1: Copy skills

```bash
cp -r source-project/.claude/skills target-project/.claude/
```

### Step 2: Configure CLAUDE.md

**Option A: Auto-configure (recommended)** — in the new project, ask Claude:

> "Configure skills for this project"

Claude reads each skill's `## Required Configuration` section, detects the project structure, generates the CLAUDE.md sections, and asks you to confirm. The detection workflow is documented in `CONFIGURATION.md` § Auto-Configuration Helper.

**Option B: Manual configure** — copy the template sections from `CONFIGURATION.md` into your CLAUDE.md and replace the `[PLACEHOLDERS]` with your values.

### Step 3: Verify

```bash
grep "{{.*}}" CLAUDE.md
```

Every required variable should appear as a `{{VARIABLE}} = actual-value` definition line. If a value is still a `[PLACEHOLDER]`, or a variable from the table below is missing entirely, configuration is incomplete.

## Configuration Variables

Skills use `{{VARIABLE}}` syntax. Definitions live in CLAUDE.md as `{{VARIABLE}} = value` lines:

| Variable | Used By | Example Value |
|----------|---------|---------------|
| `{{AI_MODEL_NAME}}` | `/commit`, `/release` | `DeepSeek-V4.0` |
| `{{AI_MODEL_EMAIL}}` | `/commit`, `/release` | `noreply@deepseek.com` |
| `{{PACKAGE_NAME}}` | `/dev`, `/release` | `dataflow` |
| `{{SRC_DIRS}}` | `/dev` | `dataflow tests samples` |
| `{{REPO_URL}}` | `/release` | `https://github.com/owner/repo` |

## How Skills Work

1. **Self-documenting**: Each skill ends with a `## Required Configuration` section listing what it needs
2. **Self-contained**: Skills that need templates bundle them in their directory
3. **Configurable via CLAUDE.md**: Project-specific values go in CLAUDE.md, not in skills
4. **Claude-readable**: The `{{VARIABLE}} = value` format lets Claude auto-configure new projects

Configuration flow:

```
copy skills → Claude reads Required Configuration sections
            → detects project structure (package name, dirs, repo URL)
            → generates CLAUDE.md configuration sections
            → user confirms or adjusts
            → skills work in the new project
```

## Templates

The `/spec` skill includes reusable templates and an SDD enforcement hook:

```
.claude/skills/spec/
├── SKILL.md                          # Methodology
├── scripts/
│   └── sdd-reminder.sh               # SDD enforcement hook (PreToolUse)
├── templates/
│   ├── index_template.md             # For layer index files
│   ├── spec_template.md              # For spec documents
│   └── examples/
│       └── example_format_spec.md    # Reference example
```

Copy these templates when creating specs in a new project.

## Contributing Improvements

When improving skills:
- Keep them project-independent
- Document new configuration requirements in the skill's `## Required Configuration` section
- Update this README's variables table if adding new variables
- Test in at least one other project before committing
