---
name: spec
description: Create or modify spec files following project methodology. Use when writing, editing, or reviewing specs/ files.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Spec Maintenance

Apply this methodology when creating or modifying spec files.

## Specs Serve Two Readers

| Reader | Needs from specs |
|--------|-----------------|
| **Agent** (Claude Code) | Behavioral contracts — "what is correct" to verify compliance |
| **Human developer** | Understanding — "why" a contract exists and "what are the boundaries" |

Both readers matter. Content that explains a contract (not just defines it) should be kept.

## Classification Principle

For each section, ask two questions:

1. *Agent question:* "When would I read this while writing code?"
2. *Human question:* "Does this content help someone understand a behavioral contract?"

| Answer | Action |
|--------|--------|
| Read on "every change" | Belongs in CLAUDE.md — move it there |
| Read for "specific task" OR helps understand a contract | Belongs in specs — keep |
| Neither | Delete |

## What Belongs Where

**What belongs in CLAUDE.md:** Architecture hard constraints, global ordering conventions, high-frequency gotchas (encoding rules, state cleanup), critical implementation details.

**What belongs in specs:**
- `formats/` (WHAT): External format definitions — fields, coordinate systems, validation rules. Includes explanations that clarify format semantics (e.g., "why (x,y) means center, not top-left").
- `evaluate/` (WHAT): Metric definitions — IoU, mAP, P/R/F1 formulas. Includes explanations of metric design (e.g., "why TN is not applicable in object detection").
- `modules/` (HOW): Module interface contracts — public API signatures, design constraints, option definitions, dependency rules.

## What Does NOT Belong in Specs — Delete on Sight

1. **Change History** — version changelogs belong in git log / CHANGELOG.md
2. **Implementation pseudocode** — "Step 1: validate dir, Step 2: scan files..." is code documentation, not a contract
3. **CLI --help output copies** — the executable is the authority
4. **Migration guides / legacy API tables** — one-time docs, delete after migration
5. **Directory tree file listings** — `ls` is the authority
6. **Standalone tutorials / how-to guides** — task selection guides, workflow recipes that don't define any contract

## Extra Check for `modules/` Specs — Interface Contract or Implementation Description?

| Interface contract → keep | Implementation description → delete |
|---------------------------|-------------------------------------|
| Public API signatures, return types | Step-by-step internal flow pseudocode |
| Design constraints and rules | Parameter-passing code snippets |
| Option/parameter definition tables | Internal helper function descriptions |
| Exception types and exit codes | Directory tree file listings |

## Before Deleting, Always Double-Check

"Does this content help explain a behavioral contract — even if it reads like education or FAQ?" If yes, keep it. Contract-defining clarifications (e.g., "coordinates must be in [0, 1]", "TN is not applicable because there is no negative class") should be preserved — they define the contract by explaining its boundaries.

## Comparison Tables — Keep if They Define Contract Differences

A table comparing two entities (detection vs segmentation, two formats, two modules) is a contract if it defines *differentiated behavioral requirements* — different inputs needed, different validation rules, different handling. It is a tutorial/guide if it just helps the user choose ("use detection when you have bbox only"). When in doubt: "does this row define a different behavior or just compare features?"

## Outdated Framing ≠ Outdated Content

Before deleting a section that seems "about the old architecture", check whether the *substance* is still correct but the *framing* is stale. Example: coordinate transformation formulas labeled "for Internal Model" are still mathematically correct — the fix is to rename the section and add a note about current architecture, not to delete the formulas.

## Spec Directory Structure

### WHAT vs HOW Separation

```
specs/
├── <contract-layer-1>/    # WHAT — external data/interface/protocol contracts
│   ├── index.md
│   └── spec_<topic>.md
│
├── <contract-layer-2>/    # WHAT — other contract layers (optional)
│   ├── index.md
│   └── spec_<topic>.md
│
└── modules/               # HOW — internal module architecture
    ├── index.md           # Architecture diagram + hard constraints (single source of truth for module dependencies)
    ├── spec_<module-1>.md
    ├── spec_<module-2>.md
    └── ...
```

WHAT layer naming depends on project domain:

| Project Type | Suggested Name | Example Content |
|-------------|---------------|-----------------|
| Data processing / format conversion | `formats/` | Data format definitions, conversion rules |
| Web API | `api/` | REST/GraphQL interface contracts |
| Library / SDK | `interfaces/` | Public API signatures, type definitions |
| Evaluation / benchmarking | `evaluate/` | Metric definitions, baselines |
| Protocol / communication | `protocols/` | Message formats, state machines |

WHAT layers may have multiple; HOW has exactly one `modules/`, mirroring code modules.

### Index Template

When creating a new layer index, use this template:

```markdown
# <Layer Name> — Specification Index

> **Status:** Canonical — these documents define the authoritative
> <contract-type> for <project-name>.

## What This Layer Covers

Briefly describe what this layer defines and what it is the ground truth for.

## Documents

| # | Document | Purpose |
|---|----------|---------|
| 1 | `spec_xxx.md` | One-line description |

## Relationship to Other Layers

- This layer (WHAT) maps to which modules in `modules/` (HOW)
- This layer is independent of which other layers

## Reading Order

Recommended reading order by task:
- Newcomer → what to read first
- Specific task → what to read
```

### What Each Spec Should Answer

| Spec Type | Questions It Answers |
|-----------|---------------------|
| Data format / protocol spec | What does this external contract look like? What required fields are defined? |
| Conversion / adapter spec | How does A become B? How are edge cases handled? |
| Module spec | What are this module's public interfaces, design constraints, and behavioral contracts? |

A spec file should not answer both "what does the data look like" and "how does the code implement it" simultaneously. If both appear, split them.

### Version Management

Each spec file starts with a version and last-updated date:

```markdown
> **Version:** vX.Y | **Last Updated:** YYYY-MM-DD
```

| Scenario | Version Change |
|----------|---------------|
| New definitions / extending existing contracts | Minor increment (v1.0 → v1.1) |
| Behavioral change (breaking change) | Major increment (v1.2 → v2.0) |
| Clarification / wording fix (no behavior change) | Update date only, keep version |
