# DRR mixed derivative implementation plan

**Goal:** Add dX dY to the second-derivative product with separate recommended/manual windows, faithful full-resolution display and exports.

**Architecture:** Local polynomial first derivatives use actual coordinates in both dimensions. A tuple transform key includes the Y window; the existing background worker and export snapshot retain independent X/Y settings. Existing d2E remains the default.

- [x] Test analytical mixed derivatives, nonuniform/descending coordinates, invalid axes, and window recommendations; implement core/drr_mixed_derivative.py.
- [x] Add mode, Y window/Auto, and recommendation/span controls; integrate worker keys and second product without changing raw/d2E behavior.
- [x] Snapshot mixed settings for paired exports; distinguish filenames, labels and processing metadata so DAT reuse remains correct.
- [x] Exercise UI mode/window changes, asynchronous computation and mixed export; run related regressions and read-only code review.

Recommendations are conservative heuristics based on point count, noise and detectable peak width, not a guarantee of an optimal smoothing window. Do not silently differentiate a constant or folded coordinate axis.
