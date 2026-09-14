# Unified MCD export implementation evidence

## Scope

Implemented `core/mcd_unified_export.py` and its focused tests in
`tests/test_mcd_unified_export.py`. The exporter consumes an explicit plain
snapshot protocol and deep-copies numerical arrays into read-only worker data.
It does not access Qt widgets or run fitting code.

## Verification

The focused command passed:

```text
.venv/Scripts/python.exe -m unittest tests.test_mcd_unified_export -v
Ran 17 tests ... OK
```

The focused tests cover five plot-ready XLSX sheets with independent branch
field/value columns, center/width headers, peak and dip metadata, multiple
retained windows, slopes and slope differences, provenance/links in JSON,
source-array immutability, wavelength ordering, duplicate round-trip fields,
independent channel/method/pair identities, successive collision-safe
revisions, late rendering failure cleanup, and compatibility with
`discover_mcd_processing_status`. The suite also exports a real
`process_mcd` result and real `SlopeAnalysis.to_dict()`/`PeakShiftResult`
objects, and verifies unavailable spectrum products are omitted.

Changed-file compilation and whitespace checks passed:

```text
.venv/Scripts/python.exe -m py_compile core/mcd_unified_export.py tests/test_mcd_unified_export.py
git diff --check -- core/mcd_unified_export.py tests/test_mcd_unified_export.py
```

The adjacent legacy commands were also run. `tests.test_mcd_extract` passed;
the `tests.test_mcd` run had five UI timing/attribute failures/errors
(`test_loaded_mcd_renders_its_embedded_plots`,
`test_mcd_and_peak_tabs_switch_plots_without_reloading_or_reanalyzing`,
`test_mcd_heatmap_click_selects_the_nearest_pair`,
`test_mcd_sidebar_fits_without_horizontal_clipping`, and
`test_mcd_theme_change_rebuilds_blit_backgrounds_before_refresh`).
This task did not establish those failures against the baseline, so they are
reported without attributing cause. No legacy source files were changed by
this task.

## Output contract

`export_mcd_analysis(snapshot, output_root, *, progress=None)` returns the
export ID, package/revision directories, XLSX, JSON, map DAT, trace CSV and
four independent PNG paths. A successful revision is staged in a temporary
directory and renamed into `<output_root>/Processed Data/MCD/<source>_MCD/rNN/`;
an exception removes the staging directory and leaves no completed revision.
The JSON filename includes `_MCD_settings_` and its source fields and
measurement descriptor retain existing MCD history discovery behavior. Legacy
window packages are left intact; callers that explicitly migrate them can
continue to use `ensure_mcd_package_dir`. Retained windows are reduced using
their requested metric and center/width for the MCD(B) table and plot. Result
objects with wavelength ordered columns are reordered to canonical ascending
energy before capture. DAT comments preserve duplicate field branch/row
identity, while feature, split, and slope products retain their analysis
identifiers and diagnostics in XLSX/JSON.
The review follow-up also verifies a real `process_mcd` result roundtrip through
`load_dat`, duplicate B fields with branch identity, real
`SlopeAnalysis.to_dict()` scalar range columns, nested source/method peak
tracks, candidate inventory separation, selected visibility, and omission of
unavailable spectrum products. The follow-up also verifies raw/corrected
plot-state selection, visible feature filtering, and numeric plotted artist
values for selected B and multiple retained windows. Invalid or stale retained
window coordinates are rejected before staging; a current fallback uses the
captured center/width settings when present. The real `process_mcd` fixture
asserts canonical ascending energies and corresponding reordered numeric
columns, including an asymmetric wavelength acquisition order. Branch
visibility is applied consistently to workbook, DAT/CSV, map, and plots;
feature checkboxes affect overlays while the workbook preserves all retained
feature and splitting products. A captured `McdResult.maps`/`DataCube` and
selected map name drive the map DAT and map PNG, preserving the cube's own
energy/gate grid instead of rebuilding a branch mesh; the map PNG includes a
numeric colorbar and selected map title. PNG legends use fixed positions so
large maps do not pay Matplotlib's expensive automatic `loc="best"` scan.
Real map artist presence and exact branch-filtered retained direct values are
covered by regression tests; a mismatched direct trace dimension now raises
instead of falling back to a recomputed map mean.
