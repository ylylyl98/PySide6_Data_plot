# CW method comparison

Organizer → Temperature → Curie–Weiss → Method comparison compares the same selected slope points using:

1. Equal-weight inverse-slope regression (existing default).
2. Equal-weight nonlinear slope regression (existing alternative).
3. Nonlinear slope regression weighted by `1/slope_se^2`.

The primary fit and existing comparison plots retain their previous methods. This tab is a sensitivity comparison, not automatic method selection. All methods use the same zero or independently fixed background; a fitted background disables the matched comparison. Missing slope fits disable the comparison for that branch, and missing/nonpositive SE disables the weighted method without substituting weights or dropping points.

The weighted model is `s = background + A/(T-theta)` on the paramagnetic branch `theta < min(T)`. Its profile interval assumes independent Gaussian slope errors and omits temperature and optical systematic errors. Unbounded scans, boundary optima and model mismatch are identified. A rejected inverse fit may show its unconstrained linear intercept as **diagnostic only**, without claiming model validity. The old methods retain their residual-scaled local intervals; interval constructions are explained in Diagnostics/tooltips. R² across different fitting spaces should not be compared.

Save method comparison exports JSON with all three rows (including failures), input points, settings, and diagnostics, even when the primary fit failed. Normal CW exports also include the comparison rows.

Validation: synthetic CW recovery, real E29 Group 2 comparison, missing SE, constant response, background restrictions, default preservation, export and stale-table clearing.

## Plots

`Method plots` shows a slope-space overlay and an inverse diagnostic for each sweep branch. Checkboxes control the three method curves; line styles identify methods. Invalid poles are shown only in the inverse diagnostic. Background comes from the control setting, independently of visible or successful methods. Saving the method comparison also writes a PNG.

The batch Compare θCW dialog retains inverse/equal as default. Select a method or enable `Compare three methods (shared axes)`, then rerun. All three methods reuse the same refitted slopes and ranges; the comparison mode uses three panels with common axis limits. Group visibility, branch marker fill, manual/automatic Y controls and hover diagnostics remain available. Diagnostic estimates carry an ×; boundary solutions without identifiable theta are omitted and counted as unavailable. JSON and `methods.csv` preserve method-specific results, and exported plots follow the chosen display mode.
