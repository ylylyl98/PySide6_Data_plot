# Unified MCD Task 1 analysis evidence

Implemented the pure analysis API in `core/mcd_analysis.py` with focused
coverage in `tests/test_mcd_analysis.py`.

## Interfaces

- `detect_analysis_features(result, ...) -> FeatureAnalysis`: detects raw
  spectral `peak` and `dip` candidates independently for each source and
  corrected/raw MCD `mcd_max`/`mcd_min` extrema row-by-row. Candidates include
  deterministic IDs, source, branch, field sign, energy, measured support
  fields/interval (using `pair_b_pos`/`pair_b_neg` when available), persistence,
  confidence, and weak/ok status. No signed MCD averaging is performed.
- `associate_features(mcd_features, spectral_features, ...) ->
  AssociationResult`: performs conservative energy plus support evidence
  matching, permits one-to-many links, and represents automatic, manual,
  ambiguous, and unmatched states. Manual links accept `(mcd_id,
  spectrum_id)` pairs or mappings with those keys.
- `fit_mcd_slopes(fields, values, branches, ranges=...) -> SlopeAnalysis`:
  performs free-intercept OLS independently for every branch and region. Each
  `SlopeFit` reports slope/intercept, standard errors, sample count, actual
  field range, residual RMS, R², jump/curvature flags, and explicit
  `constant_field`, `insufficient`, and `out_of_range` statuses. Branch
  differences include explicit high-minus-low regime differences per branch;
  same-region branch differences include standard error when paired-field
  residual covariance is supported, otherwise explicitly report `unsupported`.
- `infer_valley_mapping(reference_feature, positive_field_samples) ->
  ValleyMapping`: infers a fixed channel mapping only with at least three
  distinct measured positive-field samples and consistent ordering;
  insufficient, duplicate-field, inconsistent, or tied evidence returns
  `unknown`.
- `split_same_branch_tracks(k_track, kp_track) -> SplittingResult`: matches
  actual measured field values within one branch, supports unequal grids, and
  reports the new convention `E_K-E_Kp` explicitly. Legacy
  `core.mcd_valley_split.compute_valley_splitting` remains unchanged and still
  uses its historical `E_Kp-E_K` convention.

All result dataclasses expose `to_dict()` for JSON export. Compatibility aliases
are provided for `compute_same_branch_splitting` and
`infer_energy_based_mapping`.

## Verification

Command:

```text
.venv/Scripts/python.exe -m unittest tests.test_mcd_analysis -v
```

Result: **18 tests passed**. Coverage includes peak/dip detection, opposite-sign
MCD persistence, deterministic IDs/JSON output, automatic/manual/ambiguous/
unmatched links including one-to-many list/set manual retention, slope recovery and invalid inputs, paired covariance and
unsupported uncertainty, high-minus-low regime differences, unique-field
validation, channel-specific measured-field preservation, NaN-hole alignment
on ascending/descending axes, unknown/inferred mapping validation, and
same-branch splitting on unequal field grids. `py_compile` and
`git diff --check` also pass for the two implementation/test files.

The Astra re-review regressions add automatic cross-channel one-to-many links,
same-channel ambiguous candidate retention, reverse acquisition-order covariance
alignment, and explicit unsupported results for mismatched, NaN, and duplicate
field grids. The core focused suite is now **19 tests passed**; the selected
feature adapter adds nine focused tests.

The legacy MCD/valley run remains 45/46 because the concurrently changed UI hides
the legacy candidate bar in `test_peak_candidate_bar_and_markers_offer_nearby_features`;
the analysis files do not touch that UI path.

## Selected-feature adapter and coordinate regression

The pure `analyze_selected_feature(result, feature, candidates=..., method=..., source=...)` adapter in `core/mcd_feature_analysis.py` runs seeded peak/dip analysis only for the selected feature and conservative supported links. Its JSON payload has top-level `tracks`, `analysis_results`, `links`, and `status`; analysis keys include channel, feature ID, and method, while each analysis serializes only tracks accepted by kind, branch, and energy gates. Explicit `Raw` selection overrides corrected catalog provenance, and manual same-channel one-to-many links remain separate results.

Feature detection now keeps canonical ascending energy coordinates separate from wavelength-driven measured column order. This fixes reversed wavelength-column inputs while retaining correct source indices through NaN holes. Focused core plus selected-feature suites pass **28 tests**. A real YZ365 roundtrip CSV produced valid selected peak and dip analyses with two channel tracks each. The broader 83-test MCD/peak/export run has one unrelated UI candidate-bar failure in `test_peak_candidate_bar_and_markers_offer_nearby_features`.

The selected adapter now also accepts exact `search_window_ev` bounds,
optional seeded `prominence_fraction`, and an explicit reference-energy delta
override. `enrich_valley_analysis` adds fixed/manual/swap mapping or opt-in
energy-based inference and emits only link-backed `E_K-E_Kp` rows at exactly
shared measured fields. Unknown, ambiguous, and unequal-grid cases retain an
explicit status and reason. The combined focused suites now pass **35 tests**.

The follow-up review also separates ordinary `mcd_to_spectrum` associations
from explicit valley-pair links and rejects mixed feature kinds or methods.
Asymmetric search windows retain in-window points, mark excluded samples as
`window gap`, and recalculate the default reference from retained points.

## YZ365 UI catalog trace

On the available real YZ365 F20 roundtrip files, full detected features link
the closest ~1.640 eV MCD candidate to nearby raw candidates. Reconstructing
the UI's `_unified_mcd_candidates` whitelist drops `support_fields`,
`support_interval_t`, and `field_sign`; the same association then returns
`unmatched` with `insufficient support overlap`. A lossless `AnalysisFeature.to_dict()`
entry and catalog preserve the automatic links. This is a UI serialization
loss and does not justify widening energy/support thresholds.
