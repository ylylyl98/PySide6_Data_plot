# MCD current-center save acceptance — 2026-09-12

Implemented the approved save workflow in the application workspace.

- The existing top **Save unified MCD results** action saves the current center, width and metric independently of retained selections.
- The side **Export retained results** button exports only checked snapshots; no selection disables it. Feature-only batches no longer generate an implicit current-window product.
- A window awaiting refresh is rejected before any folder dialog. Optional current feature data requires a completed matching analysis key and source generation; unavailable or updating features are omitted with a status explanation.
- Retained-feature inspection invalidates the current-feature completion key, preventing inspected historical data from being mislabeled as the current analysis.
- Both modes share the frozen snapshot and worker pipeline. Snapshots are built before the folder dialog, the session remembers a writable destination, and **Change…** selects another destination.
- Worker completion restores controls according to the active tab and retained selection. Status identifies save scope, center or counts, revision, destination, and any feature omission.

## Verification

Command (exit 0):

```powershell
.venv/Scripts/python.exe -m unittest tests.test_mcd_save_scope tests.test_mcd_retention_export_integration tests.test_mcd_result_retention tests.test_mcd_unified_export tests.test_mcd_unified_export_luna tests.test_mcd_export_final_acceptance tests.test_mcd_feature_analysis tests.test_mcd_unified_workflow -q
```

**95 tests passed.** New scope tests first reproduced wrong-window export, blocking on feature analysis, forgotten destination, post-dialog snapshot capture, unwritable destination and stale inspected-feature acceptance before fixes.

The real-data verifier `artifacts/mcd-fixed-export-preview/verify_production_export.py` completed with exit 0. It used the existing YZ365 2.5 K roundtrip source, retained A at 1.64 eV, then saved current B at 1.66 eV twice through the toolbar action. Retained output remained A; both current outputs contained B and had different revision directories. Metadata and files were checked. All retained-export figures had no clipped text or legend-covered points.

Results: `artifacts/mcd-fixed-export-preview/production-verification.json`. Offscreen layouts at 1280×720 and 1600×1000 were rendered and inspected; the top Save and side batch-export controls are accessible. At 720 px height, folder Change and feature controls require sidebar scrolling, as does the existing long controls panel. The verifier explicitly loads Segoe UI for readable offscreen screenshots.

Two bounded review rounds: first identified stale inspected-feature completion keys and missing selection-driven batch-button state. Both were fixed with regressions. The final review reported no remaining blockers within that scope.

The running instrument application was not restarted. Changes are uncommitted in the existing workspace; the unrelated dirty tree was preserved. Reload/restart the application when convenient to use the updated code.
