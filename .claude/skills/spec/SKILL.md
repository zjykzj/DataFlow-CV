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
