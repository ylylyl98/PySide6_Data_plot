# DRR Analysis workspace

Implemented the approved Organizer-style workflow in the same application:
- Global source-toolbar DRR Analysis entry opens an empty/retained workspace.
- DRR Add current DRR to Analysis copies current processed data and provenance.
- Both entries reuse one window; main-window-owned close hides it, preserving the
  in-memory session. Standalone `run_drr_analysis.py` and `run_qt.py --drr-analysis`
  are also available.
- Batch raw CSV import requires an explicit external/self background recipe.
  Selected datasets can be reloaded with another recipe. Existing analysis
  settings are retained on reload and scientific results invalidated.
- Per-dataset settings, range, seed, gate inspection, view mode and manual zoom
  are restored on selection. Duplicate snapshots focus the existing entry.
- Range edits redraw the map and linecut without rerunning peak computation.
  Follow-range can be disabled to keep independent navigation and see a range
  rectangle. Use Current View and Zoom to analysis range support both directions.
- Select two datasets for DRR/derivative comparison. Batch parameter application
  preserves each target's seed coordinates. Batch errors are reported per dataset.
- Worker results require the same dataset instance and revision; remove/readd
  cannot revive a stale result. Main DRR changes notify without replacing snapshots.
- Individual analysis exports use the existing XLSX/JSON/PNG writer. Combined
  XLSX includes numeric per-dataset wide energy tables with blank missing rows,
  Peak_Info with dataset identifiers, and Parameters/provenance.

Primary DRR batch controls moved out of the visible panel; single-spectrum
inspection/fit remains. Existing controller bindings stay instantiated for
compatibility. Main DRR plot, preprocessing and original exports remain separate.

Verification:
- 84 focused workspace, numerical, export, UI integration, file label and picker
  tests passed (29.844 seconds).
- Real 157 x 1340 TG−1.087BG=18 data imported with three all-frame external
  backgrounds matches prior preprocessing arrays to 1e-12 absolute tolerance.
- Native hidden-window captures inspected in light/dark themes. Corrected clipped
  range inputs, linecut Y autoscale and scientific-canvas theme integration.
- Independent review found recipe switching and async dataset identity issues;
  both fixed. Removed/readded identity regression and batch-export test included.

Scope limitations: session retention is in memory, not a saved workspace project;
standalone exit ends the session. Seed tracking remains one branch per dataset
result and is not a physical peak identification model. No executable packaging
or restart of the running user application was performed.
