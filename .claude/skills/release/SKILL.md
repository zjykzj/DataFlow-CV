---
name: release
description: Bump version and publish a GitHub release
allowed-tools: Bash
---

# Release Skill

Use this skill when bumping the project version and publishing a GitHub release.

## Step 1: Bump Version

Update **all** version locations configured for this project:

| # | File | Field |
|---|------|-------|
| 1 | `<package-config>` | `version = "X.Y.Z"` |
| 2 | `<init-file>` | `__version__ = "X.Y.Z"` |
| 3 | `CHANGELOG.md` | `## [X.Y.Z] - YYYY-MM-DD` section header |

The exact file paths and verification commands vary per project. See CLAUDE.md for this project's version bump locations.

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
| **Title** | `vX.Y.Z` |
| **Body** | CHANGELOG.md `[X.Y.Z]` section content, with citation link appended at the end |

Body template:

```markdown
<CHANGELOG.md [X.Y.Z] section content, including ### Added/Changed/Fixed/Docs headings>

---

*See [CHANGELOG.md]({{REPO_URL}}/blob/main/CHANGELOG.md) for the full change history.*
```

**Rationale**: Full changelog content on the Release page lets readers see all changes without navigating away. The citation link at the bottom serves as an attribution reference.
