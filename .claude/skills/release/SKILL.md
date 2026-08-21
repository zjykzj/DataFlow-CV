---
name: release
description: Bump version and publish a GitHub release
allowed-tools: Bash, Read, Edit, Grep
---

# Release Skill

Use this skill when bumping the project version and publishing a GitHub release.

## Step 0: Determine Version Number

Review `git log` since the last tag. Classify each commit's conventional commit type into semver:

| Commit Type(s) | Semver | Example |
|---------------|--------|---------|
| `fix:`, `docs:`, `refactor:`, `style:`, `chore:`, `test:`, `ci:`, `build:` | **PATCH** (1.6.1 → 1.6.2) | Bug fix, doc update, internal restructure |
| `feat:` | **MINOR** (1.6.2 → 1.7.0) | New public API, new CLI flag, new feature |
| `feat!:` or `BREAKING CHANGE` in commit body | **MAJOR** (1.7.0 → 2.0.0) | Remove public API, change default behavior |

**Rule:** the highest-severity commit wins. If all commits are `docs`/`refactor`/`chore` → PATCH. If any `feat` → MINOR. If any breaking change → MAJOR.

**Concrete check:** before picking a version, look at the `### Added` subsection under `## [Unreleased]` (the section you're about to rename). If it's empty (no new features) → PATCH.

## Step 1: Bump Version

Update **all** version locations configured for this project. See CLAUDE.md for the exact file paths — typical patterns include:

| Pattern | Example File | Field |
|---------|-------------|-------|
| Package init | `{{PACKAGE_NAME}}/__init__.py` | `__version__ = "X.Y.Z"` |
| Project config | `pyproject.toml` | `version = "X.Y.Z"` |
| Changelog | `CHANGELOG.md` | Rename `## [Unreleased]` → `## [X.Y.Z] - YYYY-MM-DD`, then add a fresh empty `## [Unreleased]` at the top |

The `{{PACKAGE_NAME}}` variable is defined in CLAUDE.md.

**CHANGELOG rename, not rewrite**: the commit skill maintains `## [Unreleased]` on every commit, so the release does not write new entries. Rename the section header and reset it — the section content flows unchanged into the version bump commit (Step 2) and the GitHub Release body (Step 5).

## Step 2: Commit

Use the version bump commit format — body **must** include the relevant section from `CHANGELOG.md`:

```bash
git commit -m "$(cat <<'EOF'
chore: bump version to X.Y.Z

<CHANGELOG.md [X.Y.Z] section content, omitting the ### headers>

Co-Authored-By: {{AI_MODEL_NAME}} <{{AI_MODEL_EMAIL}}>
EOF
)"
```

This ensures `git log --oneline --no-decorator` provides enough context to understand each release's contents without opening `CHANGELOG.md`.

## Step 3: Tag

Create an annotated tag with a minimal message:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
```

## Step 4: Push

```bash
git push && git push --tags
```

## Step 5: Create GitHub Release

Go to the repository's Releases page, select tag `vX.Y.Z`, and fill in:

| Field | Content |
|-------|---------|
| **Title** | `vX.Y.Z: <one-line summary>` — draw the key theme(s) from the CHANGELOG entries. Keep it concise (~5-10 words). Examples: `v1.5.0: Industry-Standard Label Positioning`, `v1.6.1: Specs Purge & Code-Spec Compliance` |
| **Body** | CHANGELOG.md `[X.Y.Z]` section content, with citation link appended at the end |

Body template:

```markdown
<CHANGELOG.md [X.Y.Z] section content, including ### Added/Changed/Fixed/Docs headings>

---

*See [CHANGELOG.md]({{REPO_URL}}/blob/main/CHANGELOG.md) for the full change history.*
```

**Rationale**: Full changelog content on the Release page lets readers see all changes without navigating away. The citation link at the bottom serves as an attribution reference.

## Required Configuration

Define these in the project's CLAUDE.md:

| Item | Purpose | Example |
|------|---------|---------|
| `{{AI_MODEL_NAME}}` / `{{AI_MODEL_EMAIL}}` | `Co-Authored-By` line in bump commit | `DeepSeek-V4.0` / `noreply@deepseek.com` |
| `{{PACKAGE_NAME}}` | Locate `__init__.py` version field | `dataflow` |
| `{{REPO_URL}}` | CHANGELOG citation link in Release body | `https://github.com/owner/repo` |
| Version bump locations table | Which files carry the version string | See CLAUDE.md "Release Configuration" |
