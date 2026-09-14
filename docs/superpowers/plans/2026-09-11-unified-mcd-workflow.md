# Unified MCD Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement task-by-task. Review must be performed by Astra after Luna implementation.

**Goal:** Deliver the agreed single-page MCD and feature-analysis workflow in the real desktop application, with correct association, responsive controls, and unified reviewable exports.

**Architecture:** Retain the existing source/processing workers and Matplotlib/Qt shell. Add isolated pure analysis/export modules and a dedicated unified page/controller instead of further expanding the monolithic feature page. Legacy analysis APIs stay compatible; new page adapts their results explicitly.

**Tech Stack:** PySide6, NumPy/SciPy, Matplotlib, existing XLSX library in .venv. Use `.venv/Scripts/python.exe`.

**Spec:** This document records the approved conversation design and review refinements below.

## Constraints and approved design

- Preserve the dirty working tree. Baseline: `artifacts/unified-mcd-baseline/manifest.json` and source copies. No resets, commits, dependency installation, live experiment writes, or restarting the user's app.
- One top-level MCD page replaces the separate MCD Peak Shift navigation entry. Preserve legacy entry points/tests where useful, hide obsolete navigation rather than unsafe global index renumbering. No internal plot tabs.
- Four fixed plots: corrected MCD map TL; K/Kp spectra TR; corrected MCD-vs-B BL; feature energy/shift/splitting BR. View options overlay existing plots, never add plot tiles. Energy is eV on a labeled secondary axis; shift/splitting meV.
- Left sidebar retains source picker/status/sort; MCD center,width,metric,slope ranges; feature method,source,search bounds and advanced reference/detection controls. Processing/calibration collapsed. Plot-local B, Raw/Corrected, Inc/Dec, curve visibility and zoom. Fullscreen subplot can be a reversible display operation.
- Top candidate bar: MCD/Spectrum/All, default MCD; full numbered peak AND dip candidates, previous/next, keyboard navigation, manual seed. P/D for Raw spectral features, M+/M- for corrected MCD extrema. Highlight selected features/trajectories and search/MCD windows on plots. No display-count truncation of candidate inventory.
- Top Energy axes linked by default, bottom B axes linked. Click horizontal map cursor changes B only; dragging vertical MCD window changes window only. Raw/Corrected display does NOT change feature computation source.
- Inc and Dec independent in every computation but can be simultaneously displayed. Preserve actual field grids and show actual nearest measurement B in spectra.
- Calibrated mapping preferred; energy-based mapping explicitly optional, based on a selected reference feature at several positive-B samples and fixed over the sweep. Manual swap persists. Lack of evidence returns unknown, not invented valley. Splitting convention EK-EKp from absolute energies. Never silently change MCD measured-channel numerator when relabeling valleys.
- Raw feature method default; second derivative optional; Local fit never implicitly runs on load/save. Preserve existing owned worker lifetime, cancellation and latest-intent/generation guards. No blocking synchronous full analysis from UI callbacks.
- Auto-link is bidirectional and permits one-to-many. Independent feature detection precedes link; use support interval overlap and multi-field persistence, not nearest-center alone. Distinguish MCD-to-spectrum association from K/Kp pairing. Return scores/reasons/ambiguous/unmatched states. Manual correction and manual window edits must survive later automatic refresh within the same source.
- MCD detection considers multiple finite B slices separately by branch/sign without signed averaging cancellation. Do not label energy proximity as established physical origin.
- Defaults for +/-2T data: low [-.2,.2], high+ [1.5,2], high- [-2,-1.5]. Separate per-branch free-intercept OLS, n>=5 recommended threshold, finite unique B, no interpolation to manufacture samples. Invalid/out-of-range/insufficient samples return explicit statuses, do not extend range silently. Report slope, SE, n, actual range, residual diagnostics and jump/curvature flags. Difference SE handles overlapping sample covariance, or explicitly marks unsupported uncertainty; never false independent propagation.
- Display changes => drawing only. MCD window => trace/slopes only. Search => selected feature only. Corrected processing => corrected-dependent invalidation; Raw cache remains reusable when source unchanged.
- Export selection is explicit: current item fallback; users retain feature/window snapshots, inspect by clicking independently, include via checkboxes. Replacing a retained snapshot requires explicit Update. No forced 1:1 feature-window pairing. Stale/uncomputed items are blocked or explicitly reported; save never opportunistically analyzes them.
- One source package, readable sample/point/temp/gate/D/F/B/source-repeat prefix; actual units preserved. Decimal dots in new output names. Collision-safe export revision, same export ID in XLSX and JSON. Do not break old history/status recognition.
- XLSX at most MCD, Slopes, Energy, Shift, Splitting (omit empty): plot-ready independent B-Y columns for curves; slopes and differences with SE in one table. Identifiers, units, channel/branch/method and MCD center,width in headers; no Features/Links/Settings sheets. All detailed candidate/link/settings/fit diagnostics/source provenance in JSON. MCD map DAT and separate plot PNGs; no UI/combined screenshot by default.
- Export on worker using immutable numerical/settings snapshot. Do not access live Qt widgets/figures from worker. Agg renders independent plot products, selected visual state honored. Stage and atomically finalize coherent version; no overwrite of older revision, no redundant map multiplication for each window.

## Task 1 — Pure analysis (Luna core owner)

**Create:** `core/mcd_analysis.py`, `tests/test_mcd_analysis.py`.
**Consumes:** existing McdResult and peak result structures.
**Produces:** callable pure functions with explicit dataclass/JSON-friendly outputs. Coordinate exact APIs with UI/export owners before integration.

- [ ] Write failing numerical tests for spectral peak/dip detection, MCD opposite-sign persistence, ambiguous/unmatched association, manual link representation, inferred mapping and unknown state.
- [ ] Implement `detect_analysis_features(result)` and `associate_features(...)` with documented source/branch/energy/support metadata, deterministic IDs and conservative evidence rules.
- [ ] Implement `fit_mcd_slopes(fields, values, branches, ranges=...)` returning separate branch/region fits and differences. Tests recover known slopes; reject insufficient/constant-field inputs; verify overlapping fit covariance or unsupported flags.
- [ ] Test same-branch EK-EKp sign and no pairing by row index across unequal grids. Reuse/adapt legacy splitting without changing legacy convention silently.
- [ ] Run `.venv/Scripts/python.exe -m unittest tests.test_mcd_analysis -v`; report measured evidence and interfaces to other owners.

## Task 2 — Production unified page (Luna UI owner)

**Create:** dedicated `ui_qt/mcd_unified_page.py` (and controller if needed), `tests/test_mcd_unified_workflow.py`.
**Modify:** narrow adapters in `ui_qt/feature_pages.py`, `ui_qt/main_window.py`, `ui_qt/controllers_mcd.py`; preserve old controllers/worker ownership.

- [ ] Write failing tests for one visible MCD entry, four axes regions, candidate P/D and M filters, selection preserving independent window state, B cursor not reprocessing data, signal generation stale-source rejection.
- [ ] Implement one-page UI with existing themed Qt controls and source picker; preserve all useful existing background/feature operations through left controls or advanced expanders.
- [ ] Connect source result, pure candidates and feature worker results. Preserve separate Raw/Corrected display and computation state, fixed mapping, manual edits and source-generation scoping.
- [ ] Implement plot-local cursor/window and linked axes; use cached artists for B movement where feasible; coalesce heavy redraws. Selected feature changes can schedule owned background analysis, not all-candidate Local fit.
- [ ] Add retained-snapshot list/state and integrate exporter callable from Task 3 into explicit Save Results flow.
- [ ] Run isolated-QSettings Qt tests, show real YZ365 CSV four-plot screenshot, verify raw/corrected window interactions, dip selection and slope fits. Production source selection must work without preview injection.

## Task 3 — Unified export (Luna export owner)

**Create:** `core/mcd_unified_export.py`, `tests/test_mcd_unified_export.py`.
**Consumes:** explicit numerical snapshot schema coordinated with UI and pure analysis owner.
**Produces:** `export_mcd_analysis(snapshot, output_root, *, progress=None)` returning output paths/export ID and errors. Never reads live UI.

- [ ] Write failing tests for workbook sheets/independent field columns, peak AND dip results, chosen window snapshots, slope differences+errors, candidate/link metadata JSON only, raw source immutability.
- [ ] Implement readable prefix and revision-safe source package output compatible with existing MCD package discovery. Keep complete source identity in JSON; preserve original repeat identity.
- [ ] Generate independent PNG/map DAT from snapshot. Selected feature curves/spectrum state/slope fits honored; no composite or obsolete diagrams.
- [ ] Tests simulate partial export failure: no apparently completed revision/history after failure; test successive exports collision-free; no hidden fit call, JSON captures settings snapshot.
- [ ] Run export tests in temporary directories only; report timings for real-derived data copied into snapshot and output paths/schema.

## Task 4 — Integration and Astra review (root coordinator)

- [ ] Check all APIs, source/generation boundaries, save snapshot completeness, legacy regression checks including MCD peak, valley split, auto-refresh, source history, DRR exports, workflow review.
- [ ] Run changed-file compile and `git diff --check`. Inspect native screenshot and at least representative numerical/PNG export outputs.
- [ ] Dispatch Astra reviewer with this plan, baseline path, exact changed files and test evidence. Review scientific sign/branch conventions, UI preservation, responsiveness, retention semantics, exported units/schema.
- [ ] Have Luna fix all critical/important findings; obtain Astra re-review and rerun affected checks before final delivery. Report remaining concrete limitations honestly.

## Acceptance evidence

Implementation is not complete with preview-only widgets, hidden unconnected controls, hardcoded YZ365 data, unimplemented exporter or auto-link placeholders. A coherent end-to-end workflow from normal source selection to plotting and export must pass. Analytical assumptions and uncertain auto links remain visible, and no unverified scientific claim is implied.
