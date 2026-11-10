# Peak Analysis layout and performance review

The existing simple analysis workflow now keeps Find & Fit / Find in a fixed
footer. Custom and Advanced controls scroll independently. Edit peaks is a
secondary footer action. Compact windows move navigation history and marker
size adjustments into More, and constrain the optional result table's height.
The embedded Matplotlib subplot positions adapt to the available height so the
spectrum's energy label stays visible, including at 150% display scaling.

## Measured hot paths

`scripts/profile_peak_ui_hotpaths.py` measures warm, synthetic operations using
the offscreen software renderer. The dense map has 32,000 result points and 400
branches; spectrum switching uses 80 measured spectra with 501 energy samples.
Values below are medians on the development machine, not user-data guarantees.

| Operation | Before | After |
| --- | ---: | ---: |
| Prepare branch overlay series | 1682.5 ms | 42.7 ms |
| Update hidden Qt result/branch lists | 20.93 ms | 0.004 ms |
| Switch the selected spectrum | 109.08 ms | 58.64 ms |
| Partial redraws for ten spectrum changes | 20 | 10 |

Overlay preparation groups points once instead of scanning every point for each
branch. Hidden lists materialize only when opened. A scheduled overlay refresh
and row-selection refresh share the final spectrum draw. The overlay still
preserves branch order, disabled branches, uncertain points, and segment gaps.

The numerical path also batches energy-index lookups, shares read-only committed
histories, and reuses completed fits only when dataset identity, effective
settings, selected peak seeds, and fitting inputs match. Changing the model,
range, or selected peaks invalidates reuse. The independent reproduction is
`python -m tests.benchmark_peak_workflow`; it uses a fixed random seed and reports
candidate counts and timings. No candidate cap, reduced precision, or modified
measurement samples is used for these speedups.

## Verification

Targeted coverage includes numerical detection/filtering and fitting, cache
invalidation, cancellation/input immutability, saved workspaces, lazy lists,
single-refresh spectrum changes, branch overlay gaps, and compact QML controls.
The compact layout check verifies that the main action remains inside the panel
with Custom expanded and that the spectrum axis title stays inside the canvas.
Preview PNGs and raw timing JSON are generated under ignored `build/` paths.
