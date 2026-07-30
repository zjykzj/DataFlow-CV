---
name: claude
description: Write or update CLAUDE.md following project conventions. Use when adding gotchas, restructuring CLAUDE.md, or updating project documentation.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# CLAUDE.md Authoring & Maintenance

Apply these principles when writing or modifying CLAUDE.md.

## Authoring Principles

| Principle | Description |
|-----------|-------------|
| **Timeliness** | Update as code evolves — not write-once-and-forget |
| **Specificity** | Write `set connection timeout to 30s` not `configure reasonable timeout` |
| **High-frequency first** | Sort by usage frequency — most frequently referenced at top |
| **Single source** | Architecture hard constraints are authoritatively defined in `specs/modules/index.md`; CLAUDE.md references them |
| **Structure over length** | Target 300–700 lines; structure quality (Quick Reference, cross-references, categorized sections) matters more than line count. A well-structured 600-line file beats a dense 300-line file. |

## Recommended Structure

```markdown
# CLAUDE.md

## Quick Reference (Read First) — critical rules, architecture summary, common tasks
## Project Overview
## Specifications — point to specs/, declare spec authority, CLAUDE.md vs specs boundary table
## Architecture — module dependency diagram, hard constraints, data flow pipeline
## Critical Implementation Details — most error-prone areas, with cross-references to related gotchas
## Development Commands — install, test, lint, manual verification
## Known Gotchas — categorized by topic with priority markers ([!]=critical, [*]=important)
## Test Structure
```

**Key patterns:**
- **Quick Reference**: 5-second scan for the most frequent actions. Distills ~600 lines into ~20. Always at the top.
- **Cross-references**: Critical Implementation Details link to related gotcha categories (`> **Related gotchas**: [Coordinate Systems](#coordinate-systems)`), and vice versa — bidirectional navigation.
- **Content boundary table**: A table in "Specs vs CLAUDE.md" showing what belongs where — prevents scope creep in both directions.

## Known Gotchas Writing Guidelines

### Organization: Categorize, Then Prioritize

Instead of a flat numbered list, group gotchas by **topic** and mark each group with a **priority level**:

```markdown
## Known Gotchas

### [!] Coordinate Systems
N. **Item**: description + correct approach.

### [!] RLE & Encoding
N. **Item**: ...

### [*] COCO & Evaluation
N. **Item**: ...

### Visualizer Behavior  (no marker = reference)
N. **Item**: ...
```

| Marker | Meaning | When to Use |
|--------|---------|-------------|
| `[!]` | Critical | Causes silent bugs or crashes if missed |
| `[*]` | Important | Affects correctness or design decisions |
| *(none)* | Reference | Behavioral context, useful to know |

**Why categorize**: A flat list of 40+ items is unreadable. Topic groups let the reader jump to their area of concern. Priority markers let them know which items cannot be skipped.

### Entry Format

```
N. **Keyword**: problem description + correct approach.

Elements: scenario (in what operation) → wrong (common mistake) → correct (how to do it right) → reason (why)
```

Counter-example: `"Watch the encoding"` → unclear what to watch for.
Good example: `"DB connections: Use connection pool (min=5, max=20), never create per-request."` → scenario + mistake + fix + reason.

### Entry Sources

| Source | Trigger |
|--------|---------|
| Bug fixes | After each bug fix, ask: could a Gotcha have prevented this? Yes → add one |
| Onboarding | Every friction a newcomer hits is a potential Gotcha |
| Code review | Issue types repeatedly flagged in reviews |
| Architecture decisions | Constraints whose violation has severe consequences |

### Maintenance

- Sort by topic, then by frequency within each topic group
- Each Gotcha keeps bidirectional references with related specs / Critical Details (`> **Related gotchas**: ...` and `> **See also**: ...`)
- If a Gotcha's corresponding bug has been eliminated by an architecture refactor, delete it
- Categorization is the primary defense against long lists — 40+ items in ~10 topic groups is fine; a flat list of 40 is not
- Periodically review priority markers: `[*]` items that have caused recent bugs should be promoted to `[!]`

## Required Configuration

None — self-contained. Pure methodology, no project-specific variables.
