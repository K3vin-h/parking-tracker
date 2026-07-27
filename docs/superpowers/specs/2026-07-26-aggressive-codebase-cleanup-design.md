# Aggressive Codebase Cleanup Design

## Goal

Reduce repository clutter, remove code and assets that are demonstrably unused,
and improve maintainability and runtime efficiency without changing observable
application behavior, weakening security controls, or invalidating existing data.

## Scope and constraints

- Review the entire tracked application, including Python, templates, JavaScript,
  CSS, deployment configuration, tests, documentation, and generated artifacts.
- Preserve the user's pre-existing uncommitted edits and reconcile changes in
  overlapping files instead of reverting or replacing them.
- Treat migrations, package marker files, vendor libraries, vendor licenses,
  fonts, README-linked images, and required placeholder files as protected.
- Preserve public functions, URL names, template contracts, database schema,
  transaction boundaries, row locks, upload limits, image path checks, rate
  limiting, kiosk nonce handling, wallet ledger invariants, and CSP behavior.
- Delete a file only when repository-wide reference checks and runtime wiring
  show that it is unused or superseded. Ambiguous files remain in place.
- Make performance changes only when a benchmark, query-count assertion, or
  equivalent measurement demonstrates the current cost and the improvement.

## Baseline

The initial audit read 147 source and text files in full, covering 24,894 lines.
The worktree already contains modified configuration/documentation, CV training
plots, dashboard application setup, parking services, wallet logic, and deleted
legacy screenshots. Those changes are user-owned and are not cleanup evidence by
themselves.

The local Docker daemon is stopped, and the system Python does not have Django or
pytest installed. Ruff currently reports 16 intentional import-order violations
in the two directly executable CV training scripts. Vulture reports unused callback
arguments in upload handlers and unused test fixture variables. These findings are
the verification baseline, not proof that a framework callback or fixture can be
deleted.

## Cleanup strategy

### 1. Artifact and file cleanup

Build an inbound-reference inventory for tracked assets and documentation. Remove
the already-superseded `screenshots/` set and dashboard proof files that have no
documentation, template, test, or tooling references. Retain the seven dashboard
proof images embedded in the README, both CV training plots, bundled JavaScript,
fonts and their licenses, migrations, Python package markers, and the backgrounds
placeholder required by the documented data-generation workflow.

After deletions, scan README links, template static paths, CSS URLs, JavaScript
entry points, Django URL imports, and Docker copy rules for broken references.

### 2. Dead-code and lint cleanup

Use Ruff, Vulture, AST import analysis, Django registration patterns, URL wiring,
template references, and test references together. Rename required-but-unused
framework callback parameters with leading underscores where the framework permits
it. Do not remove Django-discovered classes, migrations, admin registrations,
management commands, model metadata, template callbacks, or pytest fixtures merely
because a static analyzer cannot see their dynamic use.

Document the direct-script import bootstrap in detector and recognizer training
scripts with narrowly scoped Ruff suppression rather than moving imports ahead of
the repository path setup and breaking documented direct execution.

### 3. Maintainability refactoring

Refactor only high-complexity functions with stable external interfaces. The first
candidate is `correct_plate()` in `apps/parking/services.py`, currently a large
transactional workflow. Extract private helpers around validation, locked-row
lookup, session reassignment, event correction, and wallet reconciliation while
keeping the public signature and atomic transaction behavior unchanged.

Review the dashboard and kiosk scan flows for repeated presentation or lookup logic.
Extract shared private helpers only when exact duplication is confirmed and tests
can exercise the boundary. Avoid a broad split of `models.py`: model classes and
their metadata are cohesive, migrations import historical model state indirectly,
and moving them would add risk without a measured runtime benefit.

Keep detector and recognizer training loops separate where their loss functions,
metrics, device behavior, and CTC fallback differ. Move only genuinely identical
plotting, checkpoint, split, or CLI validation behavior into existing shared
training utilities.

### 4. Performance work

Measure staff dashboard, session log, error queue, revenue data, public plate list,
wallet, and kiosk scan database query counts. Add `select_related`, `prefetch_related`,
conditional aggregation, or bulk operations only where measurements identify
avoidable queries. Maintain lot scoping and authorization in every optimized query.

Measure CV pipeline construction and checkpoint loading. Ensure models are reused
within the application process when safe, while preserving checkpoint validation,
device selection, evaluation mode, normalization metadata, and test isolation.
Do not introduce cross-request mutable inference state.

Avoid micro-optimizations that add complexity without measurable improvement.
Training-only changes are evaluated for duplicate I/O and avoidable tensor/device
copies, but correctness and educational readability remain higher priority.

## Error handling and security review

Every changed error path must continue to log, raise, or return an explicit error.
No broad exception swallowing or fallback success states will be introduced.

Each changed file receives an inline code-quality review, security review, and
unused-import check before work proceeds. Critical and high findings are fixed
immediately. Special attention applies to:

- uploaded image size, type, and safe-path validation;
- model checkpoint version and preprocessing validation;
- kiosk capability, nonce, replay, and rate-limit enforcement;
- transaction atomicity, row locking, wallet idempotency, and ledger history;
- authentication, staff authorization, lot scoping, CSRF, and response headers;
- generic external errors that do not leak filesystem paths or sensitive state.

## Verification gates

Verification proceeds from narrow to broad:

1. Establish a runnable environment by starting Docker Compose. Do not alter
   dependency pins solely to make the host Python environment work.
2. Run targeted tests before and after each behavioral refactor.
3. Add query-count or benchmark tests before performance changes, confirm they fail
   or expose the baseline cost, then confirm the optimized result.
4. Run Ruff and Vulture, reviewing dynamic-framework false positives explicitly.
5. Run `python manage.py check` and migration consistency checks.
6. Run the complete pytest suite and the accounts/parking coverage gate at 80%.
7. Run JavaScript tests and browser smoke tests for login, dashboard, sessions,
   revenue, kiosk, plates, wallet, top-up, and settings flows.
8. Check all tracked links and static references after asset deletion.
9. Review the final diff for accidental behavior, security, schema, or generated
   file changes.
10. Run `graphify update .` after source changes and inspect the resulting summary.

No cleanup phase is complete if it introduces a new failure. If the baseline itself
fails once Docker is available, isolate and report the pre-existing failure before
deciding whether it belongs inside this cleanup.

## Deliverables

- Removed obsolete files with reference evidence.
- Smaller, clearer high-complexity workflows with unchanged public contracts.
- Measured database or inference improvements where the current code permits them.
- Clean static-analysis results, except documented framework or direct-script
  exceptions that cannot be removed safely.
- Full verification evidence and a final list of retained ambiguous candidates.
