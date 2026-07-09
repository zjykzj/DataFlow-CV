# Skills Usage Guide

This directory contains reusable Claude Code skills. Each skill is designed to be **project-independent** and can be copied to other projects.

## 📦 Available Skills

| Skill | Purpose | Needs Config? |
|-------|---------|---------------|
| `/commit` | Git commit message format | ✅ Yes — AI model name/email |
| `/dev` | Development commands (test, lint, typecheck) | ✅ Yes — Package name, source dirs |
| `/release` | Version bump and GitHub release | ✅ Yes — Version locations, repo URL |
| `/spec` | Spec file authoring methodology | ❌ No — Self-contained with templates |
| `/claude` | CLAUDE.md authoring guidelines | ❌ No — Pure methodology |

## 🚀 Quick Start for New Projects

### Step 1: Copy skills

```bash
cp -r source-project/.claude/skills target-project/.claude/
```

### Step 2: Configure CLAUDE.md

**Option A: Auto-configure (recommended)**
```bash
# In the new project, ask Claude:
"Configure skills for this project"

# Claude will:
# 1. Read each skill's "Required Configuration" section
# 2. Detect project type and structure
# 3. Generate CLAUDE.md configuration
# 4. Ask you to confirm
```

**Option B: Manual configure**

See `CONFIGURATION.md` for the complete template. Each skill documents its required configuration in its own file.

```markdown
## Git Operations

### AI Model Configuration

{{AI_MODEL_NAME}} = Your-AI-Model
{{AI_MODEL_EMAIL}} = noreply@example.com

### Development Configuration

{{PACKAGE_NAME}} = your_package
{{SRC_DIRS}} = src tests

### Release Configuration

Version bump locations for this project:

| # | File | Field |
|---|------|-------|
| 1 | `pyproject.toml` | `version = "X.Y.Z"` |
| 2 | `src/package/__init__.py` | `__version__ = "X.Y.Z"` |
| 3 | `CHANGELOG.md` | `## [X.Y.Z] - YYYY-MM-DD` |

Repository URL: `https://github.com/owner/repo`
```

### Step 3: Verify configuration

```bash
# Check that all variables are set
grep "{{.*}}" CLAUDE.md

# Should show actual values, not {{PLACEHOLDERS}}

# Test the skills
/dev        # Should run with correct package name
/commit     # Should show correct AI model
```

## 📋 Configuration Details

Each skill's **"Required Configuration"** section documents exactly what needs to be in CLAUDE.md.

**Key insight:** Skills read their own configuration requirements, so Claude can:
1. Read the skill file to understand what's needed
2. Auto-generate the CLAUDE.md configuration
3. Verify that configuration is complete

See `CONFIGURATION.md` for the complete setup template.

## 📝 How Skills Work

1. **Self-contained**: Skills should work without modifying their content
2. **Configurable**: Project-specific values go in CLAUDE.md, not in skills
3. **Documented**: Each skill clearly states its configuration requirements
4. **Templates included**: Skills that generate content include templates

## 📝 How Skills Work

### Design Principles

1. **Self-documenting**: Each skill contains its own configuration requirements
2. **Self-contained**: Skills that need templates bundle them in their directory
3. **Configurable via CLAUDE.md**: Project-specific values go in CLAUDE.md, not in skills
4. **Claude-readable**: Configuration format allows Claude to auto-configure new projects

### Configuration Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. User copies skills to new project                    │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Claude reads skill's "Required Configuration"        │
│    section to understand what's needed                  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Claude detects project structure (package name, etc) │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Claude generates CLAUDE.md configuration sections    │
│    with detected values                                 │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 5. User confirms or adjusts values                      │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Skills work in the new project!                      │
└─────────────────────────────────────────────────────────┘
```

## 🔧 Configuration Variables

Skills use `{{VARIABLE}}` syntax for project-specific values:

| Variable | Used By | Example Value |
|----------|---------|---------------|
| `{{AI_MODEL_NAME}}` | `/commit`, `/release` | `DeepSeek-V4.0` |
| `{{AI_MODEL_EMAIL}}` | `/commit`, `/release` | `noreply@deepseek.com` |
| `{{PACKAGE_NAME}}` | `/dev` | `dataflow` |
| `{{SRC_DIRS}}` | `/dev` | `dataflow tests samples` |
| `{{REPO_URL}}` | `/release` | `https://github.com/owner/repo` |

## 📚 Templates

The `/spec` skill includes reusable templates:

```
.claude/skills/spec/
├── SKILL.md                          # Methodology
└── templates/
    ├── index_template.md             # For layer index files
    ├── spec_template.md              # For spec documents
    └── examples/
        └── example_format_spec.md    # Reference example
```

Copy these templates when creating specs in a new project.

## ✅ Validation

After copying skills to a new project, verify:

1. ✓ All `{{VARIABLES}}` are defined in CLAUDE.md
2. ✓ File paths match your project structure
3. ✓ Templates are accessible if needed
4. ✓ Skills are registered (check `/help` output)

## 🤝 Contributing Improvements

When improving skills:
- Keep them project-independent
- Document new configuration requirements
- Update this README if adding new variables
- Test in at least one other project before committing
