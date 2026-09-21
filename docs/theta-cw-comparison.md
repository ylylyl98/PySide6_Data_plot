# Comparing θCW across conditions

1. In Organizer, choose **Temperature (same doping and E-field)**.
2. Check at least two condition series and the desired sweep branches.
3. Click **Compare θCW…** beside the export button.
4. Select a corresponding optical resonance in each series. Group numbers
   are local labels, not a guarantee of matching resonances. A series with
   one group selects it automatically; confirm its energy before running.
5. Set a common symmetric B half-width and temperature interval. Select
   catalog/setpoint or measured temperature. The dialog starts with the CW
   panel's half-width and temperature controls. It snapshots current included
   records; close and reopen it to pick up new Organizer selections.
6. **Compare θCW** runs in the background. Each included MCD–B curve is
   refitted using a free intercept, then each series and sweep branch gets
   an equal-weight straight-line fit of 1/s versus T, with zero background.
7. Inspect Results, Comparison plot and Diagnostics. Choose θCW vs E-field,
   θCW vs doping, or B range sensitivity. Optional sensitivity repeats the
   batch at ±0.1, ±0.15, ±0.2 and ±0.3 T plus the primary width.

Plots show 95% local confidence intervals where constrained, hollow symbols
where they are unavailable, and omit failed estimates. Sensitivity lines
have gaps for failed fits. Tables keep failures and diagnostics. Missing
curves or invalid slopes fail the affected branch/range rather than silently
dropping temperatures; missing temperatures fail the series. Temperatures
outside the requested interval are excluded. Different actual temperature
grids are reported explicitly; a common interval does not guarantee identical
temperature sampling. Curvature/jump flags require inspection, not automatic
interpretation. θCW describes interaction tendencies within the CW model and
does not establish magnetic order.

**Save comparison…** writes a unique timestamp folder under
`Temperature_dependence/Theta_CW_comparison/`:

- `summary.csv`: conditions, primary/sensitivity width, N, actual temperature
  range, theta, uncertainty, status and diagnostics.
- `points.csv`: every selected curve's source, energy window, temperature,
  actual sampled field range, slope/SE, field-fit diagnostics and inverse-fit
  predictions/residuals when available.
- `comparison.json`: complete fit parameters, assumptions and results.
- `theta_efield.png`, `theta_doping.png`, `theta_sensitivity.png`.
- `diagnostics.txt`.

Changing fit controls or energy-group choices invalidates the previous result
and disables saving until a new comparison is run.
