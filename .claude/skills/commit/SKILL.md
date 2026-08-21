---
name: commit
description: Create git commits following project conventions
allowed-tools: Bash, Read, Edit
---

# Git Commit Skill

Use this skill whenever committing code to the repository.

## Commit Message Format

All commits must use the following heredoc format:

```bash
git commit -m "$(cat <<'EOF'
<type>(<scope>): <subject>

<body if needed>

Co-Authored-By: {{AI_MODEL_NAME}} <{{AI_MODEL_EMAIL}}>
EOF
)"
```

The `Co-Authored-By` line is **mandatory** for all commits. `{{AI_MODEL_NAME}}` and `{{AI_MODEL_EMAIL}}` are configured per-project in CLAUDE.md.

## Conventional Commit Types

| Type | Description |
|------|------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `build` | Build system or external dependencies |
| `test` | Adding missing tests or correcting existing tests |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `style` | Changes that do not affect the meaning of the code (white-space, formatting, etc.) |
| `perf` | Code change that improves performance |
| `ci` | Changes to CI configuration files and scripts |
| `chore` | Other changes that don't modify src or test files |

## CHANGELOG Maintenance

User-facing commits **must** also update `CHANGELOG.md`. If `CHANGELOG.md` does not exist, create it first with the standard header and an empty `[Unreleased]` section:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
```

If the file exists but has no `[Unreleased]` section, create it above the most recent `## [X.Y.Z]` section. Then add the entry under it, in the subsection matching the commit type:

| Commit Type | CHANGELOG Subsection |
|-------------|----------------------|
| `feat` (removing features/APIs) | `### Removed` |
| `feat` | `### Added` |
| `fix` (security) | `### Security` |
| `fix` | `### Fixed` |
| `docs` | `### Docs` |
| `chore` | `### Changed` |

Entry style follows the existing changelog: `- **Bold summary**: description.` One line per logical change; closely related changes may share one entry. Order subsections as Added → Changed → Fixed → Removed → Security → Docs.

**Skip CHANGELOG for:**
- `test` / `ci` / `build` / `style` commits — internal housekeeping with no user-visible effect
- Pure-internal `refactor` / `perf` commits — if a refactor/perf change IS user-visible (significant speedup, behavior adjustment), record it under `### Changed`
- Version bump commits (`chore: bump version to X.Y.Z`) — the release skill renames the Unreleased section at release time
- Commits that only modify `CHANGELOG.md` itself

The CHANGELOG edit is part of the **same commit** as the code change — never a separate commit.

## Procedure

1. Review the diff to determine the appropriate type and scope
2. Update `CHANGELOG.md` (see CHANGELOG Maintenance above)
3. Write a concise subject line (imperative mood, ≤50 chars)
4. Add body if the change needs explanation
5. Append the `Co-Authored-By` line
6. Stage the CHANGELOG edit together with the code changes and execute the commit command

## Required Configuration

Define these variables in the project's CLAUDE.md (Git Operations section):

| Variable | Purpose | Example |
|----------|---------|---------|
| `{{AI_MODEL_NAME}}` | `Co-Authored-By` name | `DeepSeek-V4.0` |
| `{{AI_MODEL_EMAIL}}` | `Co-Authored-By` email | `noreply@deepseek.com` |
