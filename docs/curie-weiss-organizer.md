# Curie–Weiss in MCD Organizer

1. Reopen the source application's MCD Organizer after updating.
2. Set **Compare different** to **Temperature (same doping and E-field)**.
3. Select the required condition series, then the **Curie–Weiss** tab.
4. Select one energy group representing the same optical transition. Use
   **Conditions** to exclude unwanted windows or repeats. One included record
   per temperature is required; records are not averaged automatically.
5. Set the temperature range and background mode. Results update automatically;
   **Refit** also applies current controls immediately.
   To change the linear MCD–B slope interval, enable **Refit slopes from MCD–B**
   and set **B min / B max** (default -0.2 to +0.2 T). The same interval is
   applied independently to both branches at every included temperature.
   **MCD–B slopes** shows measured points, selected points and the fitted line
   for a chosen temperature, with slope SE, point count, R², residual RMS and
   actual sampled field limits. At least three distinct finite field values
   are required. Failed refits are omitted explicitly; saved slopes are never
   substituted. Uncheck the option to compare against saved slopes again.
6. Use **Save CW fit…** for a new timestamped folder containing `fit.png`,
   `points.csv`, `fit.json` and `fit.txt`. This is separate from the Organizer's
   ordinary **Export selected series** action.
   With field refitting enabled, `fit.json` also contains all linear-fit
   diagnostics and measured support. `field_points.csv` marks which samples
   were included, with predictions/residuals, and `field_fit_preview.png`
   saves the displayed temperature's branch plots. The selected record ID is
   recorded in JSON. Original experimental files are not overwritten.

The temperature source defaults to the Organizer's setpoint/catalog value.
**Measured T only** uses measured temperatures without silently substituting
setpoints. Assumed default temperatures and missing measurements are omitted.

## Model and interpretation

The input is the **near-zero signed-mean MCD slope**, separately for increasing
and decreasing field. By default this uses the saved slopes. Optional field
refitting reads the saved corrected signed-mean MCD–B curves for the exact
optical window, then fits a free-intercept straight line in the requested B
interval. Other slope checkboxes do not change this physical input. High-field
slopes are not automatically subtracted. Use matching optical windows and a
linear low-field regime; an interval away from zero may not represent
zero-field susceptibility.

### Default: inverse-slope linear fit (Tang-style)

Reference: Tang et al., *Evidence of frustrated magnetic interactions in a
Wigner–Mott insulator*, Nature Nanotechnology (2023),
https://doi.org/10.1038/s41565-022-01309-8, Extended Data Fig. 7.
The paper integrates a narrow spectral window and extracts its zero-field
slope. Our input remains the saved signed window mean: for a fixed window
and consistently sampled spectrum, mean and integral differ by a constant
scale, which leaves theta unchanged. This does not correct temperature-dependent
optical conversion or inconsistent spectral sampling.

The default regresses `1/(s-s0) = m*T+b` with equal weights and reports
`theta = -b/m`, `A = 1/m`. The figure caption does not specify regression
weights: equal weighting is an explicit app choice, not an exact reproduction
of unpublished analysis code. Zero background is the default; fixed background
is supported and treated as exact. Free background is unavailable for this
linear method. At least three distinct temperatures are required.

Both plots show available slope standard errors, propagated on the right as
`sigma_inverse = sigma_s / (s-s0)^2`. Saved errors are read only when the
window, branch and slope match; otherwise error bars are omitted with an
explicit warning. Field refits provide new errors for the selected B interval.
Error bars do not change the equal-weight fit. The right plot emphasizes the
measured temperature interval and labels theta and its 95% interval.
R-squared is evaluated in inverse-slope space for this method.

Theta uncertainty uses residual-scaled linear-regression covariance and the
delta method with a Student-t multiplier. If the linear coefficient is
consistent with zero, the theta ratio is poorly constrained: no finite local
interval or definite interaction sign is reported. Slopes consistent with zero
also invalidate reciprocal-error interpretation. Temperature, fixed-background
and optical systematic errors are excluded. Schema-3 JSON and CSV exports
include inverse values/errors, inverse residuals and the chosen method.

### Optional: previous slope-space fit

The method selector retains equal-weight nonlinear fitting of
`s(T) = s0 + A/(T-theta)`, including free background. Its reciprocal plot is
only a transformation of that fit; its R-squared is in slope space. It retains
the extrapolated intercept and measured-temperature inset. It can differ from
the default because taking the reciprocal changes the least-squares objective.
Both methods allow signed optical amplitude and require theta below the
lowest selected temperature.

Positive/negative theta suggests net FM/AFM interactions only within the model's
applicability. It does not establish a magnetic ordering transition. Verify a
paramagnetic temperature range and approximately temperature-independent
conversion between the optical slope and magnetic susceptibility. Few points,
weak agreement, differing window centers and unspecified field ranges produce
diagnostic messages.

## Previous slope-space validation (2026-09-15)

- 40 tests passed across `test_curie_weiss`, `test_curie_weiss_ui`,
  `test_mcd_energy_groups`, `test_mcd_extract`, `test_mcd_organizer_async_export`.
- Actual YZ365 D6.3/F0, approximately 1.64 eV / 5 meV saved results were loaded
  through the production parser and plotted/exported through the Organizer.
  Four setpoints: 1.67, 2.2, 2.7, 3.2 K. Zero-background fits: increasing
  theta = -4.421 K (local 95% CI -16.3 to 7.5 K); decreasing theta = -9.165 K
  (local 95% CI -39.0 to 20.7 K). Both interaction signs are unresolved.
- Light and dark screenshots inspected at 1500x940 and 1180x780. Smaller windows
  scroll the new tab vertically to reach the entire figure and explanatory text.
- Independent review confirmed the bounded solver and selection fixes. A
  previously slow nearly-linear free-background case now takes about 0.01 s.
- Two older `test_mcd_organizer_ui` failures remain: expected legacy slope-axis
  count and existing curve-exclusion visibility. Both reproduce when all three
  new Organizer hooks are removed in memory; the existing checkout is otherwise
  unchanged. No full-suite success is claimed.

### Editable field interval validation

- 65 tests passed across the CW core/UI, new `test_mcd_slope_refit`, existing
  MCD slope analysis, energy groups, extraction and Organizer export suites.
- The YZ365 four-temperature dataset reproduces every saved branch slope
  when refitted in [-0.2, +0.2] T. In [-0.3, +0.3] T, all eight refitted
  slopes agree with independent NumPy line fits of the selected samples.
- Verified schema-2 exports carry the requested B range, actual sampled
  ranges, field-point counts, standard errors, per-branch diagnostics and
  sample inclusion flags. Export flushes pending B changes before saving.
- Missing/invalid curves and insufficient distinct field points do not fall
  back to saved slopes. Switching off refitting restores the saved-slope path.
- Light/dark field-fit views inspected; scroll within the tab on short windows
  to see the complete plot. Independent code review found no important issues.

### Inverse-slope method validation

- Real YZ365 four-temperature saved slopes and all eight saved standard errors
  were loaded and exported. Both theta estimates agree with an independent
  NumPy regression of reciprocal slopes: -4.004 K increasing, -10.15 K decreasing.
- In both branches the inverse-response linear coefficient is consistent with
  zero at 95%; finite local theta intervals and interaction-sign claims are
  suppressed. More temperatures are needed to constrain the intercept.
- Verified field refits at +/-0.2 T and +/-0.3 T, propagated errors, schema-3
  exports, method switching, and the light/dark UI. The exported plot was
  visually inspected. Independent review found no important issues.
