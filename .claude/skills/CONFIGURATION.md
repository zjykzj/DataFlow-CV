# CLAUDE.md Configuration Template

This template helps you configure CLAUDE.md for skills copied from another project.

## Skills Configuration Checklist

When you copy `.claude/skills/` to a new project, check which skills need configuration:

| Skill | Needs Config? | Where to Configure |
|-------|---------------|-------------------|
| `/commit` | ✅ Yes | Git Operations section |
| `/dev` | ✅ Yes | Development Commands section |
| `/release` | ✅ Yes | Git Operations section |
| `/spec` | ⚠️ Optional | Specifications section + SDD enforcement hook (§3–4) |
| `/claude` | ❌ No | Self-contained |

## Step-by-Step Configuration

### 1. Git Operations Section

Required by: `/commit`, `/release`

```markdown
## Git Operations

Git workflows are defined as project skills. Use the corresponding skill for each task:

- **`/commit`** — commit message format, `Co-Authored-By` line, and conventional commit types. Invoke for every `git commit`.
- **`/release`** — version bump checklist, version bump commit, annotated tag, push, and GitHub Release body template. Invoke when publishing a new release.

### AI Model Configuration

The AI model used in this project is **[YOUR_MODEL_NAME]**. Configured in skills as:

\```
{{AI_MODEL_NAME}} = [YOUR_MODEL_NAME]
{{AI_MODEL_EMAIL}} = [YOUR_MODEL_EMAIL]
\```

**Common options:**
- Claude: `Claude-Opus-4.7` / `noreply@anthropic.com`
- DeepSeek: `DeepSeek-V4.0` / `noreply@deepseek.com`
- Custom: Your preferred model name / email

### Release Configuration

Version bump locations for this project:

| # | File | Field |
|---|------|-------|
| 1 | `[PACKAGE_CONFIG_FILE]` | `version = "X.Y.Z"` |
| 2 | `[PACKAGE_INIT_FILE]` | `__version__ = "X.Y.Z"` |
| 3 | `CHANGELOG.md` | `## [X.Y.Z] - YYYY-MM-DD` section header |

Verify with: `grep -rn '"X\.Y\.Z"' [SEARCH_PATHS]` (exclude `CHANGELOG.md`).

Repository URL for the `/release` skill:

\```
{{REPO_URL}} = [YOUR_REPO_URL]
\```

**Examples:**
- Python/Poetry: `pyproject.toml` + `src/package/__init__.py`
- Python/setuptools: `setup.py` + `src/package/__init__.py`
- Node.js: `package.json` only
- Go: `version.go` or Git tags only
```

### 2. Development Commands Section

Required by: `/dev`

```markdown
## Development Commands

General development workflows are defined as a project skill — use `/dev` for test, lint, and typecheck commands.

### Development Configuration

Template variables for `/dev` skill:

\```
{{PACKAGE_NAME}} = [YOUR_PACKAGE_NAME]
{{SRC_DIRS}} = [SPACE_SEPARATED_DIRECTORIES]
\```

**Examples:**
- Single package: `myapp` + `myapp tests`
- Monorepo: `src/core` + `src tests tools`
- Multi-module: `backend` + `backend tests integration`

### Installation

\```bash
pip install -e .                    # Editable install (recommended for dev)
pip install -e .[dev]               # With test/lint deps
\```

[Add any project-specific installation notes here]
```

### 3. Specifications Section (Optional)

Required by: Projects using `/spec` for SDD (Spec-Driven Development)

```markdown
## Specifications

The `specs/` directory contains the **canonical specifications** — the single source of truth for development.

### Spec Maintenance

Spec maintenance methodology is defined as a project skill. Use `/spec` when creating, modifying, or reviewing spec files. The skill covers the SDD workflow, the two-reader model, classification principles, what belongs where, and deletion rules.

**SDD hard rules:**

1. **Invoke `/spec` before any edit to `specs/` files** — the methodology must be loaded before touching spec content.
2. **Spec-first ordering**: any feat/fix that affects a contract documented in `specs/` must (a) update the affected spec to the target state **before** implementing, (b) verify the implementation against the spec **after** coding (conformance check), and (c) list the affected spec files in the commit body.
```

### 4. SDD Enforcement Hook (Optional, Recommended)

The CLAUDE.md rules above rely on Claude honoring them; a PreToolUse hook makes the reminder **mechanical** — the harness injects the SDD methodology reminder into Claude's context on every `specs/` edit.

#### Quick Setup

The `/spec` skill ships with a ready-to-use hook script. Add to `.claude/settings.json` (merge with existing hooks if present):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "sh .claude/skills/spec/scripts/sdd-reminder.sh"
          }
        ]
      }
    ]
  }
}
```

#### How It Works

The script (`.claude/skills/spec/scripts/sdd-reminder.sh`) reads the tool input from stdin, checks if the target path contains `specs/`, and outputs a hook payload that injects the SDD reminder. The script is self-contained, commented, and can be inspected before installation.

Requires `python3` on PATH (for JSON parsing). Verify:

```bash
echo '{"tool_name":"Edit","tool_input":{"file_path":"specs/x.md"}}' \
  | sh .claude/skills/spec/scripts/sdd-reminder.sh
# Should print the reminder JSON; non-specs paths should print nothing
```

## Verification

After configuration, verify:

```bash
# 1. Check all variables are defined
grep "{{.*}}" CLAUDE.md

# Each required variable should appear as a "{{VARIABLE}} = actual-value"
# definition line. No [PLACEHOLDER] values should remain.

# 2. Test skills
/dev        # Should use correct package name
/commit     # Should show correct AI model in Co-Authored-By
/release    # Should know which files to bump
```

## Auto-Configuration Helper

If you're Claude reading this in a new project, help the user by:

1. **Reading requirements** — each skill's `## Required Configuration` section lists its variables
2. **Detecting project structure**:

```bash
# Detect project type
ls pyproject.toml package.json go.mod 2>/dev/null

# Detect package name (Python example)
grep "^name = " pyproject.toml

# Detect source directories
ls -d */

# Detect repo URL
git remote get-url origin
```

3. **Asking for what cannot be detected** (e.g., AI model name/email) via `AskUserQuestion`
4. **Generating the CLAUDE.md sections** with detected values, using the templates above
5. **Asking the user to confirm** before writing

**Example workflow:**
```
User: "Configure skills for this project"

Claude:
1. Reads pyproject.toml → detects package name "myapp"
2. Lists directories → finds "myapp tests"
3. Asks which AI model to credit in Co-Authored-By
4. Generates CLAUDE.md configuration with detected values
5. Asks user to confirm or adjust
```
