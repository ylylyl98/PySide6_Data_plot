# DRR batch peak analysis

Implemented in the existing DRR Peak Analysis section. Select raw DRR,
second derivative, or both; set independent energy and Y bounds or copy the
current view. Analysis runs on all selected Y spectra in a background worker.
SG differentiation uses the full energy axis before range cropping.

Each product has independent peak/dip markers and conservative adjacent-Y
tracks. Missing or ambiguous candidates break tracks. Clicking a result selects
the shared Y spectrum; selected candidates can be excluded. Existing single
spectrum fitting remains available in a collapsed section. Changed data or
analysis settings invalidate results and prevent stale analysis exports.

Save PNG/DAT preserves the clean original outputs. Valid analysis additionally
produces a separate XLSX, JSON and comparison PNG. XLSX sheets DRR_Peaks and
D2E_Peaks contain numeric Y and wide peak-energy columns; missing/uncertain
positions remain blank. Peak_Info and Parameters preserve details/provenance.
JSON preserves the result and settings; an import/restore UI is not included.

Validation: 45 focused numerical, export, UI, integration and background-workflow
tests passed. Native hidden-window screenshots were inspected and range-label
and table layout issues corrected. Independent review confirmed the range
endpoint, raw-only validation and plot legend fixes; no remaining scoped blocker.
Git diff whitespace checks passed for the integration files.

Real BG=18 data with 157 Y spectra, energy range 1.60–1.80 eV, SG 21/2:
both numerical analyses completed in 0.313 seconds (excluding load, rendering
and export). XLSX, JSON and comparison PNG were generated under
artifacts/drr-batch-peaks. Detector candidates are not validated physical modes.

A broader DRR regression run also encountered
test_raw_split_scale_does_not_override_second_color_limits, concerning split
color limits outside this feature. It remains unresolved; the full suite is
not claimed to pass. Existing workspace edits were preserved. No executable
was rebuilt and no running user application was restarted.
