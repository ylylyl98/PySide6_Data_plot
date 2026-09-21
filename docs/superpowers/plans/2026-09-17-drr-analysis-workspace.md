# DRR Analysis workspace implementation plan

**Goal:** Implement the user-approved Organizer-style multi-dataset workspace,
with an independent entry and a DRR add-current shortcut.
**Architecture:** A pure dataset/session model owns copied cubes, settings,
provenance and results; a standalone Qt window owns views and async jobs.
Reuse existing analysis algorithms and export implementation. Preserve unrelated
workspace edits. No commits, executable packaging or running-app restart.
**Tech Stack:** PySide6, Matplotlib, NumPy, existing DRR core, openpyxl export.
**Spec:** User-selected design in this conversation, approved 2026-09-17.

- [x] Model, loading and export (`core/drr_analysis_workspace.py`, tests):
  dataset identity, immutable input snapshot, per-dataset settings/result revisions,
  batch parameter application excluding seeds, explicit background recipes,
  no-side-effect raw CSV loading and Origin workbook aggregation.
  Interface: AnalysisDataset(key,name,cube,provenance,settings,result,revision);
  create_dataset(cube,name,provenance,settings=None), load_dataset(path,background_files,
  background_mode='external',background_which='all'), apply_settings(dataset,settings),
  apply_common_settings(source,targets), analyze_dataset(dataset), export_dataset,
  export_summary. Concrete tests: copied input arrays; own settings remain after
  switching; seed values preserved in batch; changed settings invalidate results;
  summary preserves dataset identifiers and missing rows.
- [x] Native analysis window (`ui_qt/drr_analysis_window.py`, tests):
  left multi-select dataset list, central raw/derivative or two-dataset comparison
  maps plus linecuts, right reusable peak controls and SG controls, bottom result
  table. Dataset switching saves/restores parameters, seed, results, gate and view.
  Range edits trigger a debounced redraw independent of computation. Controls:
  add files, set background/reload selected, add/use current DRR, apply parameters,
  analyze selected, remove, export individual/summary. Background jobs carry dataset
  identity/revision and discard stale completions. Worker cancellation on close.
- [x] Integration (`main_window.py`, `feature_pages.py`, standalone launcher):
  independent DRR Analysis button plus add-current DRR button open same retained
  window. Duplicate current dataset focuses its existing entry. A changed current
  dataset is advertised and adopted only on explicit update. Existing DRR view
  stays unchanged. Batch controls migrate out of primary DRR visible layout while
  legacy single-spectrum inspection remains available.
- [x] Verification: failing-to-passing focused unittest coverage; native hidden
  window captures at light/dark themes; real BG18 snapshot/import and export smoke;
  independent scoped review. Document limitations and exact test results.

Parallel bounded ownership: a core agent owns model/loading/export and their tests;
the parent owns the window/integration and their tests. Review independently.
