# MCD interaction follow-up

## Changes and causes

- Catalog-to-UI conversion omitted support fields, intervals and field sign. Preserving complete feature serialization restores supported MCD-to-spectrum associations; thresholds were not weakened.
- Programmatic/filter selection now publishes the selected identity and queues tracking. Unmatched analysis displays its reason.
- MCD candidates default to five strongest prominence values, displayed in ascending energy order. The full catalog remains available through All.
- Center/width refresh uses existing trace/window artists and updates slope snapshots. Y limits expand for stronger signals; no heatmap reconstruction is required by this path.
- Valid low/high positive/high negative fits are rendered only over their measured fitting intervals. Hidden branches have no fit segments. Full fit/difference details remain accessible separately from the compact plot summary.
- Animated spectra/trace/fit artists are painted during full canvas draws, avoiding empty frames after resize.
- The directly owned Matplotlib Figure now explicitly repositions existing axes after GridSpec updates. Pixel-based row gutters preserve titles and labels in small windows.

## Evidence

- Luna: 36 core/helper tests including the real-data serialization regression.
- Root: 40 combined UI/retention/integration/render tests passed before the final geometry follow-up; final four render regressions and fit-line test passed (5 tests).
- Actual YZ365 data: automatic MCD selection status ok; Spectrum filter triggers two measured tracks without a manual queue call; five default MCD candidates from a 155-item full catalog.
- Center changes approximately 231–234 ms in the measured three-update sample, retaining the heatmap artist. Warm B changes approximately 19–24 ms; a full-draw recovery still costs roughly 0.6 seconds. These are small-sample timings, not percentile benchmarks.
- Native 1280×720 and 1600×1000 screenshots inspected. Four panels, spectra, feature tracks, fit segments and axis labels remain present after resize.
- Compilation and changed-file whitespace checks passed.

## Review boundary

Astra independently passed the initial selection/center/sorting checks, then its final follow-up hit the account usage limit. Final serialization/fit-line/layout and renderer refinements were locally verified, but have not received a completed final Astra review. No live application restart or experimental source write was performed.
