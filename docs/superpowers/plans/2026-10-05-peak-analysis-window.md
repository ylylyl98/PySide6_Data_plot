# Peak Analysis Window Implementation Plan

> Execute inline against the approved chat design; use test-driven-development and requesting-code-review at the feature boundary.

**Goal:** Integrate the English analysis preview as an independently restartable native analysis window.

**Architecture:** A small plot-toolbar bridge exchanges immutable snapshots and result files with a detached child process. A shared window delegates detection/tracking and PL fits to separate adapters; numeric content fingerprints protect overlays.

**Tech Stack:** Existing PySide6, NumPy, SciPy, Matplotlib and unittest.

**Spec:** `docs/superpowers/specs/2026-10-05-peak-analysis-window.md`

## Global Constraints

- All production UI is English; no browser runtime.
- Preserve existing uncommitted changes and existing analysis workflows.
- No automatic log or results dock from the new workflow.
- Snapshot preparation and hashing run outside the app GUI thread.
- Source changes reload with the child process; frozen code changes require rebuilding.

## Task 1: Numeric snapshots and separate adapters

Files: `core/peak_workspace.py`, `core/peak_tracking.py`, `core/pl_peak_analysis.py`, `core/drr_peak_adapter.py`, `tests/test_peak_workspace.py`.

Interfaces: `create_dataset(cube, kind, channel, source, name, view=None) -> dict`; `save_workspace(path, datasets, active_key=None)`; `load_workspace(path) -> (datasets, active_key)`; `detect(dataset, row, settings) -> list[dict]`; `track(dataset, branches, settings, cancelled=None, progress=None) -> dict`; `fit(dataset, result, settings, cancelled=None, progress=None) -> dict`.

- [x] Add failing numeric tests using two analytic moving Lorentzians and a peak/dip pair. Assert centers within measured resolution, stable branch IDs, missing-row gaps, cancellation and no source mutation.
- [x] Run `.venv/Scripts/python.exe -m unittest tests.test_peak_workspace -v` and confirm missing feature failures.
- [x] Implement immutable normalized data snapshots, atomic NPZ persistence, seed tracking, and a PL-only fitting adapter reusing the existing bounded fitter.
- [x] Verify Gaussian/Lorentzian fits recover independently specified centers and widths; persisted results preserve failure diagnostics and branch identity.

## Task 2: Native analysis window and exports

Files: `ui_qt/peak_analysis_window.py`, `ui_qt/peak_analysis_controls.py`, `core/peak_plotting.py`, `core/peak_export.py`, `tests/test_peak_analysis_window.py`.

Interfaces: `PeakAnalysisWindow(session_path, reply_server='')`; `add_datasets(datasets)`; `request_restart()`; `restart_requested` flag; `build_figure(dataset, row, show_overlay, show_residual, figure=None)`.

- [x] Write failing UI tests for linked linecut, peak/dip controls, branch selection, async completion, cancellation and persisted close.
- [x] Build with Qt layouts, existing theme infrastructure, an English command strip, dominant heatmap, linked spectrum, compact right controls and status bar.
- [x] Implement explicit preview/track/fit jobs, result and export dialogs, and local seed editing without recalculation on color/selection changes.
- [x] Run UI tests and inspect light/dark screenshots at desktop and compact sizes.

## Task 3: Detached process and app toolbar

Files: `run_peak_analysis.py`, `ui_qt/peak_analysis_bridge.py`, `ui_qt/main_window.py`, `run_qt.py`, `tests/test_peak_analysis_bridge.py`, `tests/test_peak_analysis_process.py`.

Interfaces: `PeakAnalysisBridge(owner, toolbar)`; `main(argv=None) -> int` with `--snapshot`, `--session`, `--reply-server`.

- [x] Add failing integration tests for sending actual PL/DRR/Compare/Power intensity snapshots and rejecting changed-source overlays.
- [x] Add toolbar `Peak Analysis...` and `Show peaks` controls, background snapshots, authenticated-local socket messages, and cached overlay artists.
- [x] Implement single-instance child handoff, state restoration and explicit restart after graceful save. Add frozen dispatch to `run_qt.py`.
- [x] Exercise a real subprocess that loads a snapshot, persists branches, exits and restores state in a fresh process.

## Task 4: Review and verification

- [x] Run focused new suites plus PL/DRR dual-view, Compare and Power regression suites.
- [x] Request a separate code review with the spec, changed-file list and test evidence; fix actionable findings.
- [x] Document source-vs-EXE restart behavior and file responsibilities in README. Report verified results and any limitations.

## Verification record

- Final focused command: `.venv/Scripts/python.exe -m unittest tests.test_peak_workspace tests.test_peak_analysis_window tests.test_peak_analysis_bridge tests.test_peak_analysis_process -v` — **23 passed**.
- Selected regressions plus the new suites: **82 run, 81 passed, 1 failed** before the final additional branch-redetection test. Covers PL/DRR dual views, old DRR batch integration, Compare and Power fitting.
- The failing existing case is `tests.test_drr_batch_peak_integration.BatchPeakIntegrationTests.test_side_by_side_owns_each_product_and_view_switch_keeps_results` at line 44 (missing legacy batch markers). Reproduced separately with `PeakAnalysisBridge` replaced by a no-op; the new entry point does not account for that failure. Existing source changes were preserved.
- Native screenshots inspected at 1180×820 and 960×720, light/dark, 100% and 150% scaling. Explicit Segoe UI font loading is used only in the offscreen preview harness because the Windows offscreen platform's default font rendered as missing glyphs. Production keeps the application's system-font theme policy.
- Review regressions added for in-place data changes, data changes during an outstanding fingerprint job, child-window viewport scope, and persistent branches absent from a selected spectrum. Cancellation and closing during initial load also have targeted passing tests.
- The independent follow-up reviewer verified the pending-content-check fix, reran all six bridge tests successfully, and reported no remaining Important findings in that bounded follow-up.
- Source launch/reopen and single-instance forwarding were exercised with real subprocesses. A packaged EXE was not rebuilt or tested.
- Current limitations: repeated scan coordinates require prior separation/aggregation; VP is outside this peak-analysis workflow. Main-app bridge edits require an app restart; analysis-process source edits require only reopening/restarting the analysis window.

## Follow-up: matching colors and visible DRR product tabs

- Added display metadata capture from actual map artists and restored custom colormaps, log/zero-centered normalization and split limits in the analysis window/export. Descending-energy snapshots preserve the physical association of region colors.
- Added ΔR/R / 2nd derivative tabs above the plot, with file-level dataset choices, same-coordinate switching, separate product results and disabled missing-product states. Source-revision grouping supports separate raw/derivative handoffs; superseded product analyses remain accessible.
- Selected test command: `.venv/Scripts/python.exe -m unittest tests.test_peak_display tests.test_peak_analysis_window tests.test_peak_analysis_bridge tests.test_peak_workspace tests.test_peak_analysis_process tests.test_split_color_scale tests.test_colormaps tests.test_drr_plot_reuse tests.test_pl_dual_view -v` — **78 passed**. After the final plot-title/tick-label readability adjustment, reran display/window/bridge suites — **20 passed**.
- Inspected light/dark native previews at 1180×820 and 960×720 with 100% and 150% scaling. Independent review verified both the sequential-handoff and descending-axis fixes with four passing targeted tests and no remaining scoped findings.
- This follow-up changes the main-app display handoff as well as the child UI: restart both processes and reopen the dataset from the plot toolbar to refresh old snapshots.

## Follow-up: whole-map candidates and reversible filters

- Replaced the remaining top file dropdown with file tabs (one file displays its name). Added Find & filter / Branches tabs, explicit Local extrema / SG + extrema method controls and actual cached-method/count information.
- Find all peaks now detects every measured spectrum, caches all candidates above an explicit acquisition floor (no count cap) and immediately overlays peak/dip markers. Prominence, width, spacing, count, polarity and range are reversible post-filters; rejected points and reset are available.
- Preserved identities through temporary filter splits, preserved manual seeds when refiltering, retained ambiguous positions as isolated markers and preserved actual cached-method provenance in PL fits. All four review findings have failing-before/passing-after regressions and independent verification.
- Workspace persistence includes the candidate pool and identity history. CSV includes prominence/width metrics. Main-app overlay packets omit the full candidate cache and association history unless rejected points are requested, in which case only their plotting coordinates are included.
- Final focused command: `.venv/Scripts/python.exe -m unittest tests.test_peak_candidates tests.test_peak_analysis_window tests.test_peak_workspace tests.test_peak_display tests.test_peak_analysis_bridge tests.test_peak_analysis_process -v` — **42 passed**, including the default-noise-rejection regression added after the user's annotation. `git diff --check` passed (line-ending warnings only).
- Inspected native light/dark previews at 1180×820 and 960×720, 100% and 150% scaling; reused the existing theme and compact control tokens. Compact windows can scroll the right control panel vertically; the horizontal scrollbar found during inspection was removed by correcting label sizing.
- Initial zero-floor local-extrema benchmark (300 spectra × 1201 energy samples, two features plus noise): 229,887 cached candidates, 3,976 kept; detection 7.897 s and filtering 0.229 s. The user correctly identified this as overly permissive for defaults. Comparing the same data with SG window 11 and floor .01 retained 26,317 candidates and 601 filtered positions, of which 600 were within 1 meV of known centers; detection 2.564 s, filtering .033 s. New datasets now default to SG/11, floor .01, prominence .05. Existing explicit settings remain unchanged. These are single-run synthetic measurements, not a universal accuracy or performance guarantee.
- Source users can restart only the analysis window to load its UI/algorithms. A running main app needs restart to load the shared overlay renderer's new candidate/ambiguity markers. No packaged build was produced.

## Follow-up: product-specific recommended method

- New PL/Compare/Power intensity and raw ΔR/R data use SG; DRR second/mixed/labelled first derivative data use Local, with a contextual English explanation next to the method tabs. Explicit choices and old workspace settings are preserved.
- Fresh handoffs apply recommendations in the child, including snapshots from an already-running main app with old SG defaults. Saved analysis sessions use the ordinary restore path. The actual subprocess test now exercises old-default handoff, a manual SG override, and a new-process restore retaining that override.
- Final verification: `.venv/Scripts/python.exe -m unittest tests.test_peak_workspace tests.test_peak_analysis_window tests.test_peak_candidates tests.test_peak_analysis_process tests.test_peak_analysis_bridge -v` — **43 passed**. `git diff --check` passed. Independent review reported no substantive findings; all three new targeted tests passed.
- Inspected the derivative recommendation and Local overlay in native light/dark previews at 100%/150% scaling, including 960×720 layout. This change requires only restarting the analysis window for source deployments; packaged code still requires rebuilding.
