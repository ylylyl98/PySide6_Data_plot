# MCD export transfer and final checks

Work continued in task `01a09754-cffd-7613-b8f8-ada678942d99` from `查找缺失的 file` (`01a08c40-6817-71b3-a94b-190d9c47fefb`). The original task confirmed it stopped implementation and review. Existing unrelated working-tree changes were preserved; no commit, rollback, experiment-directory write, or live-app restart was performed.

## Review accounting

The first production Astra review occurred in the original task. Luna High consolidated its fixes here. The second and final Astra review rejected the intermediate implementation for incorrect integral/mixed-window units, ambiguous fit ownership/region labels, and legends obscuring data. Root corrected those explicit findings and verified the final code below. No third review was run; the final corrections are verified by root, not a subsequent Astra approval.

## Final behavior

- Default output includes corrected MCD map, one independent MCD-vs-B PNG per retained window, and available feature energy/shift PNGs. Explicit valid pairing enables an additional splitting PNG; spectra are opt-in.
- Every retained window has its own metric, axis units, owned fits, and slope table. Integral axes use eV and slope tables use eV/T; mean slopes use 1/T. Retained `id` is preserved as window identity.
- High− and High+ remain distinct colored slope-table rows. Fit coefficients and actual intervals remain unchanged; extrapolated segments extend by 50% with lighter lines. Short Inc/Dec legends replace long redundant fit legends.
- Feature plots use common method/reference subtitles when applicable and concise, distinct channel/branch legends. Full feature identities remain in numerical metadata.
- Analysis serialization preserves actual reference methods; manual E₀ overrides do not retain an unrelated measured reference field. Legacy unknown reference methods are not inferred to be interpolation.
- Fixed 1200×900 canvases, 22 pt axis labels, 20 pt ticks, 16 pt auxiliary labels; no tight cropping. Visible headers omit G01 while source metadata keeps the complete file identity.

## Verification

Command:

```powershell
.venv/Scripts/python.exe -m unittest tests.test_mcd_export_final_acceptance tests.test_mcd_unified_export tests.test_mcd_unified_export_luna tests.test_mcd_retention_export_integration tests.test_mcd_feature_analysis -q
```

Final result: **47 tests passed**, 18.949 seconds. New tests first reproduced missing reference provenance, incorrect integral units/mixed-window rendering, and oversized feature legends before the respective fixes.

```powershell
.venv/Scripts/python.exe artifacts/mcd-fixed-export-preview/verify_production_export.py
```

Final result: exit 0, production revision r10. The script uses the exact 2.5 K YZ365 CSV, `process_mcd`, the real MainWindow retained-selection snapshot path, and `export_mcd_analysis`. It checks four independent retained channel/branch tracks (61/60/61/60 points), each zero-field interpolation, and six slopes independently against NumPy fits to the same retained trace. No valley pairing is invented for these four tracks.

All four real-data PNGs are 1200×900, have an 8.4 px header/axis gap, have no detected text clipping, and have zero measured/fit points covered by legends. Root visually inspected the corrected output. The script's latest machine-readable report is `artifacts/mcd-fixed-export-preview/production-verification.json`; readable contact sheet is `artifacts/mcd-fixed-export-preview/final-products.png`. The displayed contact sheet is for review; exported PNGs remain separate.

Validation is scoped to the MCD export/selection/reference paths and the named dataset, not the entire application test suite or arbitrary dense multi-analysis legend layouts.
