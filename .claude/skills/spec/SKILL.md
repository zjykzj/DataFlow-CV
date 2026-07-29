---
name: spec
description: Create or modify spec files following project methodology, and drive SDD (spec-first development). Use when writing, editing, or reviewing specs/ files — and BEFORE implementing any feature or behavior change that affects a contract documented in specs/.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Spec Maintenance

Apply this methodology when creating or modifying spec files.

## TL;DR

```
Spec-first → Cross-file scan → Bump version → Implement → Full conformance check → Commit
```

| What you're doing | How to do it |
|---|---|
| Feature / bug fix | Spec first, bump version, then code |
| **Impact scan (key!)** | `grep` CLAUDE.md, README.md, other specs/ for references to the changed contract — add to change list |
| New definition | Minor bump (v1.0 → v1.1) |
| Behavioral change (breaking) | Major bump (v1.2 → v2.0) |
| Clarification / wording | Update date only, keep version |
| Commit | List **all** affected specs/docs in body |
| Unsure if content belongs | See [curation-guide.md](references/curation-guide.md) |

## SDD Workflow (Spec-Driven Development)

Any feature or behavior change that touches a contract documented in `specs/` MUST follow this loop. Specs describe the **target state** — code follows specs, not the other way around.

### Phase 1 — Spec First (before writing any code)

1. **Impact analysis**: determine which spec files/sections the change affects.
   If none, the SDD loop does not apply — proceed with normal development.
2. **Cross-file scan (mandatory)**: search the entire repo for other files that
   reference the contract being changed — CLAUDE.md, README.md, other specs in
   `specs/`, and any doc files.  These are also affected and must be updated.
   Use `grep -rn "<key-term>" CLAUDE.md README.md specs/` to find ripple effects.
3. **Update all affected docs**: the primary spec first (bump version per
   §Version Management), then every cross-referenced file found in step 2.
4. **Confirm the contract**: present the spec change to the user for approval
   before implementing.

### Phase 2 — Implement

5. Write code to satisfy the updated spec. If during implementation the contract
   turns out to be wrong, go back to Phase 1 and change the spec — never silently
   diverge in code.

### Phase 3 — Conformance Check (after code is complete)

6. **Primary spec**: Re-read every section touched in Phase 1. Verify each
   contract line against the implementation: signatures, behaviors, edge cases,
   error/exit codes.
7. **Ripple check (mandatory)**: Re-read every cross-referenced file found in
   Phase 1 step 2.  Verify they are still aligned with the implementation —
   stale keyboard shortcuts, outdated pipeline descriptions, missing new
   parameters, etc.
8. On divergence: code bug → fix the code; contract was wrong → fix the spec
   (re-bump version), then re-verify.

### Commit Ordering

- Spec and code changes belong to the same commit series; the spec commit
  **precedes or accompanies** the code commit.
- The feat/fix commit body lists **all** affected files — primary spec(s) plus
  any cross-referenced docs (CLAUDE.md, README.md) found in Phase 1 step 2.
- Any ripple-fix commits (CLAUDE.md, README.md alignment) discovered during
  Phase 3 belong in the same series, not deferred to a later session.

## Specs Serve Two Readers

| Reader | Needs from specs |
|--------|-----------------|
| **Agent** (AI coding assistant) | Behavioral contracts — "what is correct" to verify compliance |
| **Human developer** | Understanding — "why" a contract exists and "what are the boundaries" |

Both readers matter. Content that explains a contract (not just defines it) should be kept.

## Classification Principle

For each piece of content, ask: **"When would I read this while writing code?"**

| Answer | Layer | Typical Content |
|--------|-------|-----------------|
| "Every change" | **CLAUDE.md** | Architecture hard constraints, global conventions (ordering/naming), high-frequency gotchas (encoding rules, state cleanup patterns), critical implementation details that affect every edit |
| "Specific task" | **Specs — WHAT layer** | External contract definitions — what the outside world expects from this system. Data formats, API/protocol specs, metric definitions, interface contracts with external systems. Includes explanations that clarify contract semantics (e.g., "why this field uses center coordinates, not top-left"). |
| "Specific task" | **Specs — HOW layer** | Internal module contracts — public API signatures, design constraints, option/parameter definitions, dependency rules, exception types and exit codes. How modules relate to each other, not how they are implemented internally. |
| "Neither" | Delete | — |

WHAT layer naming depends on project domain (see §Spec Directory Structure for the project-type mapping). HOW has exactly one `modules/` layer.

## Content Curation

The key question: **does this define or help understand a behavioral contract?**
If unsure, see the [curation guide](references/curation-guide.md) for detailed
judgment rules (6 content types, pre-deletion checks, interface-vs-implementation
table).

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

## Required Configuration

None for the methodology itself — templates are bundled in `templates/` within this skill's directory.

For SDD enforcement (recommended), configure in the target project (templates in `skills/CONFIGURATION.md` §3–4):

- CLAUDE.md "SDD hard rules" (invoke `/spec` before specs/ edits; spec-first ordering)
- PreToolUse hook injecting the SDD reminder on `specs/` edits (`.claude/settings.json`)
