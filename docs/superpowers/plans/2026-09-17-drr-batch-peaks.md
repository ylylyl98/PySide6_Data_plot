# DRR range peak analysis implementation plan

User approved the design in this conversation: range-limited all-Y analysis of
raw DRR and full-spectrum SG second derivative, separate overlays and clean
original PNG/DAT, Origin-friendly XLSX plus JSON and comparison PNG.

Use parallel bounded implementation tasks with independent file ownership and
test-driven development. Preserve existing local changes. No commits or app restart.

## Interfaces

`core.drr_peak_analysis.PeakAnalysisSettings`: frozen dataclass with x_min,
x_max, y_min, y_max; source='both' ('raw','second','both'), polarity='both'
('peaks','dips','both'), prominence=0.05 (fraction), min_distance_mev=1.0,
max_peaks=6, max_shift_mev=3.0, sg_window=21, sg_polyorder=2.
`analyze_drr_peaks(cube, settings, cancelled=None)` returns JSON-safe dict:
schema_version, settings, y_label, y_values, products. Products keyed raw/second,
each has points list: row_index (index in original cube), y, energy, amplitude,
prominence, polarity ('peak'/'dip'), track_id (per product), status ('accepted'
or 'uncertain'). Missing detections omitted; exports retain all selected Y rows.
`export_drr_peak_analysis(folder, base, result, raw_cube)` returns dict of Paths.

## Tasks

- [x] Numerical module/tests: range validation, full-axis derivative before
  cropping, extrema detection, conservative adjacent-row matching with gaps,
  ambiguity flags, cancellation, descending/nonuniform axes and missing values.
  Run `.venv/Scripts/python.exe -m unittest tests.test_drr_peak_analysis -v`.
- [x] Independent export module/tests: XLSX DRR_Peaks/D2E_Peaks/Peak_Info/Parameters,
  numeric cells and blank missing points, stable track columns, independent PNG
  and JSON with source/processing provenance; no raw file mutation.
  Run `.venv/Scripts/python.exe -m unittest tests.test_drr_peak_export -v`.
- [x] UI controller: primary range/source controls and advanced detector settings,
  asynchronous execution with stale result protection, inspectable per-row table,
  product-specific overlays, clean original exports with separate analysis files.
  Existing single-spectrum fit retained under collapsed advanced section.
- [x] Verify synthetic numeric/export behavior, real CSV smoke test, Qt offscreen
  visual review, existing DRR regressions, independent code review.
