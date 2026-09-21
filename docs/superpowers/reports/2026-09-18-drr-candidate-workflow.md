# DRR candidate-first analysis

The standalone workspace's all-candidate mode detects local extrema once within
the selected calculation range and polarity. It saves absolute prominence,
prominence divided by that row's selected-range peak-to-peak amplitude, and full
width at half prominence interpolated on the energy axis in meV. Missing-value
segments are handled separately. Detection applies no prominence cutoff, spacing
suppression or per-row count limit. Raw DRR remains unsmoothed; the second
derivative uses the existing full-axis SG policy before cropping.

The result filter applies Energy/Y, relative prominence, min/max width, spacing,
and optional per-row count limits, then links adjacent retained rows. Ambiguous
connections remain uncertain. The original candidate pool is unchanged; lowering
thresholds can recover candidates from that pool. Manual exclusions stay excluded.
Filters are per dataset and saved with the workspace. Table, markers and exports
share filtered results; exported JSON retains filtered-out points, and XLSX
includes metrics and filter settings. Original legacy/seed results lack the
complete candidate metrics and require redetection in all-candidate mode to use
metric filters. Seed tracking remains a separate calculation.

The result filter opens after detection. Legacy pre-detection threshold controls
are hidden in candidate mode. Background, SG, calculation range, source and
polarity changes still require detection. Timing on the local YZ365 1T BG=18
saved group, 1.82–1.88 eV: 33,723 candidates in 0.859 s; one prominence/width/
spacing filter in 0.427 s (numeric filtering only, not screen rendering).

Validation includes reversible thresholding, physical widths, relinking,
manual exclusion, exports, UI integration, persistence and seed regressions.
