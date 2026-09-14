# File status and export fixes

**Goal:** Resolve the gaps confirmed in the all-tab status and export audits without changing scientific calculations or source-role semantics.

**Architecture:** Keep existing workflow-specific selectors, semantic theme colors, source identity matching, and owned Qt workers. Only immutable snapshots cross into workers; widgets and result publication stay on the GUI thread.

**Tech Stack:** PySide6, existing Python export code, unittest, project `.venv`.

**Evidence:** `docs/superpowers/reports/2026-09-11-all-tabs-file-status-audit.md` and `2026-09-11-all-tabs-export-audit.md` in that directory. User authorized implementation; retain existing uncommitted changes and do not commit or install dependencies.

## A — Existing selector presentation

- [x] Power: expose computed New / Partly processed / Processed status in group rows with existing theme colors and text; preserve ordering, role assignment and validation.
- [x] DRR: apply existing state colors to member/chosen file rows, preserving missing/unknown/background distinctions.
- [x] Test real list items and selected-state readability; no additional scans during filtering.

## B — SHG saved history and writer

- [x] Match existing SHG export metadata to exact source paths and roles; ambiguity remains Unknown rather than New or Processed.
- [x] Add visible All / New / Processed / History unknown filtering, timestamps/newest-first and state text/colors to the source selector; preserve Single/A/B/background choices through filtering and refresh.
- [x] Scan history in an owned background task, reuse snapshots for local filtering, invalidate on folder refresh and successful export. Never auto-select another source.
- [x] Replace the scalar NumPy finite check in SHG CSV formatting with an equivalent scalar check; preserve all output bytes and special-value behavior.
- [x] Test identity collisions, asynchronous stale results, refresh/export updates, selector preservation, and CSV equivalence.

## C — Peak Shift history and independent exports

Entry-point correction from Astra review: Tools launches `run_mcd_organizer.py`, which instantiates `ui_qt/mcd_organizer_window.py::McdOrganizerWindow`. Its `_export` must be fixed and verified; changing only the older `ui_qt/mcd_extract_dialog.py` does not address the active application workflow.

History semantics: Peak Shift displays historical successful exports for the exact analysis source path, consistent with PL/MCD saved-history indicators. It does not certify current raw bytes or current parameters. Capture that path and experiment root from the analyzed result, never from a later selection; do not introduce raw-file hashing or freshness certification. Read the small history index asynchronously and reuse its cached snapshot.

- [x] Peak Shift: record and display its own successful export history, explicitly separate from shared MCD source history. Use source descriptors; legacy exports with insufficient identity remain unknown. Do not claim MCD history means Peak Shift processed.
- [x] Move Peak Shift export work and MCD Organizer export work off the GUI thread using immutable input snapshots and owned workers. File dialogs stay on the GUI thread; retain output formats and explicit export actions.
- [x] Bound duplicate submissions, handle errors truthfully, preserve source identity after selection changes, and safely close while export is active.
- [x] Test actual worker result/error/finished paths, output equivalence, and source-history updates.

## Acceptance

- [x] Astra reviews each bounded implementation; fix actionable regressions before completion.
- [x] Run targeted tests using `.venv/Scripts/python.exe`, require exit code and summary; no full Qt suite.
- [x] All generated verification files stay in temporary/artifact directories, never the user's experiment outputs.
- [x] Summarize completed behavior, validation and any real remaining limitation. Preserve Slides/Tools purpose and existing duplicate-save policy.

Final verification: Astra approved A, B, and C; detailed commands, results, and test limitations are in 2026-09-11-file-status-export-review.md under reports. Targeted lifecycle/output checks passed; no full Qt suite or large-data timing claim.
