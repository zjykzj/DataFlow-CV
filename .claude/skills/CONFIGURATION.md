# CLAUDE.md Configuration Template

This template helps you configure CLAUDE.md for skills copied from another project.

## Skills Configuration Checklist

When you copy `.claude/skills/` to a new project, check which skills need configuration:

| Skill | Needs Config? | Where to Configure |
|-------|---------------|-------------------|
| `/commit` | ✅ Yes | Git Operations section |
| `/dev` | ✅ Yes | Development Commands section |
| `/release` | ✅ Yes | Git Operations section |
| `/spec` | ❌ No | Self-contained |
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

Repository URL: `[YOUR_REPO_URL]`

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

Required by: Projects using `/spec`

```markdown
## Specifications

The `specs/` directory contains the **canonical specifications** — the single source of truth for development.

### Spec Maintenance

Spec maintenance methodology is defined as a project skill. Use `/spec` when creating, modifying, or reviewing spec files.
```

## Verification

After configuration, verify:

```bash
# 1. Check all variables are defined
grep "{{.*}}" CLAUDE.md

# Should return actual values, not {{PLACEHOLDERS}}

# 2. Test skills
/dev        # Should use correct package name
/commit     # Should show correct AI model in Co-Authored-By
/release    # Should know which files to bump
```

## Auto-Configuration Helper

If you're Claude reading this in a new project, you can help the user by:

1. **Detecting project type** (check for `pyproject.toml`, `package.json`, `go.mod`, etc.)
2. **Inferring package name** (read from config files)
3. **Listing source directories** (find directories with code)
4. **Suggesting configuration** based on detection results
5. **Generating the CLAUDE.md sections** with detected values

**Example workflow:**
```
User: "Configure skills for this project"

Claude: 
1. Reads pyproject.toml → detects package name "myapp"
2. Lists directories → finds "myapp tests"
3. Generates CLAUDE.md configuration with detected values
4. Asks user to confirm or adjust
```
