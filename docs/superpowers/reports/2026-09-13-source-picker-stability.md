# Source selector stability audit

User request: check DRR baseline selection and other tabs for initially clipped filenames and refresh flicker.

## Findings and changes

| Selector | Finding | Change / verification |
| --- | --- | --- |
| DRR measurement / baseline | Both use the same dialog and existing unchanged-group guard | Added explicit compatible-baseline selection regression; unchanged completion preserves the selected row |
| PL, MCD, Compare dialogs | Shared repopulate always cleared the live model | Stage rows and compare all stored model roles and flags; retain live rows on identical content |
| PL main source list | Catalog completion cleared unchanged filenames | Compare ordered filenames before rebuilding |
| MCD main source list | Refresh cleared unchanged filenames | Shared row comparison; only restore list selection after an actual replacement |
| Power measurement groups | Re-render cleared unchanged groups | Shared comparison, retaining source row identity and scrolling |
| Power sweep / KK / KKp picker | Catalog signal cleared the available and chosen lists | Shared comparison for both lists |
| SHG source list | Reapplying filters cleared unchanged rows | Compare filenames, status roles, styling and flags before replacement |

Changed content still replaces rows using the prior filtering and selection fallback rules. This includes processed labels, tooltips, flags and colors; the comparison is not filename-only for styled rows. No numerical processing was changed.

## Validation

- 118 tests passed across 13 independently executed modules, including source selectors, DRR baseline/catalog startup, PL/Compare/Power/SHG workflows, MCD shared selection and filename delegates.
- Regression tests reproduced redundant model resets before the corresponding fixes.
- Independent review found no actionable regressions; font inheritance and deferred staging-widget cleanup were checked.
- Windows native light-theme stability tests: 13 passed.
- Windows native dark-theme screenshot at 150% scale inspected: complete filename wrapping, no row overlap. Artifact: `artifacts/source-picker-stability-dark-150.png`.
- Original transient filename clipping was not independently reproduced. This work establishes removal of unchanged-content model resets, not proof that every possible first-frame clipping cause is eliminated.
- Two existing Power controller tests had incomplete fake-owner catalog state. Their fixtures now supply the real PowerSeriesSource cache precondition, preserving their one-load assertions.
- Combined-process Qt regression execution aborted at an existing DRR test's `app.processEvents()` and another combined run stalled. Modules are therefore verified in separate processes; this is a test execution limitation, not a claimed all-suite pass in one process.
