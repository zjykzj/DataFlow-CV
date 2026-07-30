# Spec Content Curation Guide

> Reference for judging what belongs in specs vs. CLAUDE.md vs. deletion.
> Linked from the main `/spec` skill.

## TL;DR

| Content Type | Judgment | Quick Rule |
|---|---|---|
| Change history / version changelogs | **Delete** | Belongs in `git log` / `CHANGELOG.md` |
| Implementation pseudocode / code examples | **Judge** | Keep if it defines a contract more precisely than prose could |
| CLI command signatures | **Keep** | They ARE the public API contract |
| Migration guides / legacy API tables | **Delete** | One-time docs — delete after migration complete |
| Directory tree file listings | **Judge by level** | Keep module-level; simplify project-level |
| Tutorials / how-to guides | **Keep, label** | Mark as `## Usage Guide` |
| Interface contracts vs implementation | **Keep contracts** | Public API, design constraints, exit codes → keep; internal plumbing → delete |

> The key question is always: **does this define or help understand a behavioral contract?**

## What Does NOT Belong in Specs

These are not "delete on sight" rules — each requires case-by-case judgment.
The key question is always: **does this define or help understand a behavioral
contract?**

### 1. Change History / Version Changelogs → Delete

Version history belongs in `git log` / `CHANGELOG.md`. Specs define current
contracts, not how they evolved. Example: "Key change from v1: Internal Model
removed" → delete; the spec should just describe current behavior.

### 2. Implementation Pseudocode / Code Examples → Judge

**Keep** if the code **defines or clarifies a behavioral contract** — it
specifies what the code must do more precisely than prose could. Examples
that ARE contracts:

- Greedy matching algorithm pseudocode — without it, the matching contract
  is ambiguous
- Coordinate transformation formulas (`cx_abs = cx * image_width`) — these
  ARE the mathematical contract
- A `try/finally` code block showing which variable must be cleaned up and
  how — prose "must clean up" is ambiguous; the code says exactly what
- Constructor calls showing parameter values like `strict_mode=False` —
  these define the behavioral contract ("visualizers never reject data")

**Delete** only if the code is **truly redundant** — the prose already
defines the same requirement with equal precision, and the code adds
nothing. Example: `data = json.load(open(path))` when the prose says
"read the file as JSON".

**Key distinction:** Is the code the *clearest*, *most precise* expression
of the requirement? If yes → keep. The fact that something is
"implementation-like" (variable names, function calls) does NOT make it a
violation — those can be the contract when they specify required behavior.

**Tiebreaker:** "If I replaced this code with prose, would the behavioral
requirement become less precise or ambiguous?" If yes → keep.

### 3. CLI Command Signatures → Keep (with nuance)

**Keep** command signatures — they ARE the module's public API contract.
Example:
```
<tool> convert <src>2<dst> [OPTIONS] <SOURCE> <TARGET>
```
This defines: subcommand name, positional argument order, argument count.
Code must implement this exact signature.

**Don't copy** full `--help` output verbatim — the executable is the
authority for option descriptions.

### 4. Migration Guides / Legacy API Tables → Delete

One-time transition docs. Ship with the release, delete after migration is
complete.

### 5. Directory Tree File Listings → Judge by Level

**Keep** module-level file listings — they ARE architecture documentation.
Example:
```
<package>/<module>/
├── base.py                # Base class + shared pipeline
├── converter.py           # Format converter
└── utils.py               # Shared utilities
```
This tells the reader: what files exist in this module, what each is
responsible for. Essential for both agent and human to navigate the codebase.

**Simplify** project-level recursive trees that just mirror `ls` output.
Example: a full `specs/` tree in an index.md → redundant; the Documents
table already serves as the index.

### 6. Tutorials / How-To Guides → Keep, Label as Usage Guide

**Keep** if it helps readers understand or apply the contract. Example:
"Metric Selection Guide" telling users which metric to pick for small-object
detection — it's not a contract, but it's context the reader needs when
using evaluate specs.

**Label** these sections clearly (e.g., "## Usage Guide") so readers know
they're guidance, not behavioral requirements.

**Delete** only truly standalone workflow recipes that don't reference any
contract (e.g., "How to set up your first ML project").

**Comparison tables** are a special case of this rule: keep if the table
defines differentiated behavioral requirements between entities. If it just
helps the user choose between options (e.g., "Metric Selection Guide"),
keep it but label as `## Usage Guide`. Tiebreaker: "does this table help
the reader apply a contract defined in this file?"

## Pre-Deletion Checks

Before deleting any content, apply two checks:

**1. Contract check:** "Does this content help explain a behavioral contract
— even if it reads like education or FAQ?" If yes, keep it. Contract-defining
clarifications (e.g., "coordinates must be in [0, 1]", "TN is not applicable
because there is no negative class") define the contract by explaining its
boundaries.

**2. Framing check:** Before deleting a section that seems "about the old
architecture", check whether the *substance* is still correct but the
*framing* is stale. Example: coordinate transformation formulas labeled
"for Internal Model" are still mathematically correct — fix by renaming
the section and adding a note about current architecture, not by deleting
the formulas.

## Extra Check for `modules/` Specs — Interface Contract or Implementation Description?

The classification table below is a **starting point**, not a hard rule.
Apply the "behavioral contract" test before deleting:

| Interface contract → keep | Implementation description → likely delete |
|---------------------------|-------------------------------------|
| Public API signatures, return types, command signatures | Function-internal variable assignments |
| Design constraints and rules (e.g., "must not import X from Y") | Step-by-step internal plumbing (e.g., "then call `ctx.obj['verbose']`") |
| Option/parameter definition tables | Internal helper function docstrings that mirror code comments |
| Exception types and exit codes | Directory tree of nested subdirectories |

**Nuance for common borderline cases:**

- **Step-by-step pipeline flow**: Keep if it defines the **contractual
  sequence** of steps (what must happen in what order). Delete if it
  describes **how** each step is implemented internally.
- **Code snippets showing constructor calls**: Judge — a constructor call
  with specific parameter values can define behavioral requirements (e.g.,
  `strict_mode=False` means "visualizers never reject data"; `logger=self.logger`
  means "visualizer passes its logger to the handler"). Delete only if the
  call merely repeats the signature without adding contract-relevant
  information.
- **Internal utility function tables**: Delete if they duplicate what's in
  the module file listing. Keep if they define behavioral differences between
  utilities that the file listing alone doesn't convey.
