# Unified MCD baseline

Source snapshot: `artifacts/unified-mcd-baseline/manifest.json` plus 185 Python files. Snapshot taken before this task's production edits. Existing dirty working tree is preserved.

## Checks before implementation

`.venv/Scripts/python.exe -m unittest tests.test_mcd_valley_split tests.test_mcd_peak_export tests.test_mcd_auto_refresh -q`

20 tests, three pre-existing failures, all in `test_mcd_auto_refresh`:

- `test_leaving_local_method_cancels_obsolete_pending_fit`: worker submission count 3 versus expected 2.
- `test_load_finishing_on_peak_tab_starts_deferred_analysis`: immediate result assertion gets None.
- `test_tab_entry_analyzes_once_and_reuses_results`: immediate result assertion gets None.

These tests require investigation of current async contracts, not blindly replacing assertions. The remaining 17 tests passed.

`.venv/Scripts/python.exe -m unittest tests.test_mcd_peak_shift tests.test_mcd_valley_split tests.test_mcd_peak_export tests.test_drr_dual_export tests.test_workflow_request -q`

64 tests passed in 28.021 s. This is baseline evidence, not a claim of final integrated verification.
