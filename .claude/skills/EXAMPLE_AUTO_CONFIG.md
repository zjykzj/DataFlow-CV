# Skills Auto-Configuration Example

This document demonstrates how Claude can automatically configure skills in a new project.

## Scenario: User Copies Skills to New Project

```bash
# User has a Python project with this structure:
myapp/
├── pyproject.toml
├── src/
│   └── myapp/
│       └── __init__.py
├── tests/
└── CHANGELOG.md

# User copies skills
cp -r DataFlow-CV/.claude/skills myapp/.claude/
```

## Claude's Auto-Configuration Workflow

### Step 1: User Request

```
User: "Configure skills for this project"
```

### Step 2: Claude Reads Skill Configuration Requirements

Claude reads each `SKILL.md` file and extracts "Required Configuration" sections:

```
/commit  → needs: {{AI_MODEL_NAME}}, {{AI_MODEL_EMAIL}}
/dev     → needs: {{PACKAGE_NAME}}, {{SRC_DIRS}}
/release → needs: version locations, {{REPO_URL}}
/spec    → needs: nothing (self-contained)
/claude  → needs: nothing (self-contained)
```

### Step 3: Claude Detects Project Structure

Claude runs detection commands:

```bash
# Detect project type
ls pyproject.toml package.json go.mod 2>/dev/null

# Detect package name (Python example)
grep "^name = " pyproject.toml | cut -d'"' -f2
# Output: myapp

# Detect source directories
ls -d */ | grep -E "^(src|tests|lib)/"
# Output: src/ tests/

# Detect repo URL
git remote get-url origin
# Output: https://github.com/user/myapp
```

### Step 4: Claude Asks User for Missing Info

Claude uses `AskUserQuestion` for values it cannot detect:

```
Question 1: Which AI model are you using?
- Claude-Opus-4.7 (noreply@anthropic.com)
- DeepSeek-V4.0 (noreply@deepseek.com)
- Other

Question 2: Confirm detected values?
- Package name: myapp ✓
- Source dirs: src tests ✓
- Repo URL: https://github.com/user/myapp ✓
```

### Step 5: Claude Generates CLAUDE.md

Based on detection + user answers, Claude generates:

```markdown
## Git Operations

Git workflows are defined as project skills. Use the corresponding skill for each task:

- **`/commit`** — commit message format
- **`/release`** — version bump and release

### AI Model Configuration

{{AI_MODEL_NAME}} = Claude-Opus-4.7
{{AI_MODEL_EMAIL}} = noreply@anthropic.com

### Release Configuration

Version bump locations for this project:

| # | File | Field |
|---|------|-------|
| 1 | `pyproject.toml` | `version = "X.Y.Z"` |
| 2 | `src/myapp/__init__.py` | `__version__ = "X.Y.Z"` |
| 3 | `CHANGELOG.md` | `## [X.Y.Z] - YYYY-MM-DD` |

Repository URL: `https://github.com/user/myapp`

## Development Commands

### Development Configuration

{{PACKAGE_NAME}} = myapp
{{SRC_DIRS}} = src tests
```

### Step 6: Claude Confirms Configuration

```
✅ Configuration complete!

Skills configured:
- /commit: Ready
- /dev: Ready (package: myapp, dirs: src tests)
- /release: Ready (repo: github.com/user/myapp)
- /spec: Ready (self-contained)
- /claude: Ready (self-contained)

Try: /dev to run tests, /commit to make a commit
```

## Key Advantages

1. **No manual configuration** — Claude does the work
2. **Consistent across projects** — Same skills, adapted values
3. **Discoverable** — Skills document their own requirements
4. **Verifiable** — Easy to check if configuration is complete
5. **Extensible** — Add new skills, same pattern

## Manual Fallback

If auto-configuration fails or you prefer manual setup:

1. Read `CONFIGURATION.md` for the complete template
2. Copy the template sections into your `CLAUDE.md`
3. Replace `[PLACEHOLDERS]` with your values
4. Verify with `grep "{{.*}}" CLAUDE.md` (should return no results)
