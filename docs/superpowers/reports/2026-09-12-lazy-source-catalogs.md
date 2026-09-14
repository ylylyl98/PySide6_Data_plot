# Lazy source catalogs — implementation and verification

Implemented the requested cache-first, background-validation, lazy-tab workflow in the existing application workspace.

## Behavior

- Startup restores the experiment folder and requests only the active data tab's catalog.
- A background worker reads its versioned JSON cache and publishes a preview before inspecting source files. An existing, newer displayed catalog is never replaced by an older preview. PL/Compare shared rows receive the same protection.
- Validation compares relative paths, sizes and nanosecond modification times. Unchanged catalogs reuse discovery results. Additions, deletions, metadata edits and data changes invalidate the cached result.
- Cache files live in the user's local app cache, separate from experiment data. They use fixed supported record types and atomic replacement; corrupt or unwritable caches fall back to discovery.
- PL, MCD, Compare, Power and SHG perform discovery only for the requested mode. Compare reuses PL's file-list format but does not invoke PL processing-history discovery. Compare metadata reads now occur in the worker.
- DRR uses its own worker, a cached source-list preview and its existing persistent per-file inspection cache.
- Slides persists its plot catalog and avoids repeated scans on unchanged tab re-entry. Watcher changes and explicit Refresh still trigger updates.
- Manual Refresh retains force intent through queued requests. Async save completion targets the exported mode. File-list refreshes preserve current selections, including selections made after the initial request.

## Measured cache behavior

Final targeted suite: **121 tests passed** (82.088 s):

```powershell
.venv/Scripts/python.exe -m unittest tests.test_source_catalog_cache tests.test_lazy_source_catalogs tests.test_drr_catalog_reuse tests.test_drr_picker_startup tests.test_workflow_latency tests.test_power_combine tests.test_compare_source_workflow tests.test_pl_source_workflow tests.test_mcd_unified_workflow tests.test_slides_latency tests.test_shg_history tests.test_source_status_transitions -q
```

`py_compile` for the cache, main window and Slides widget passed. Scoped `git diff --check` passed (only Git's existing LF/CRLF conversion notices).

`artifacts/lazy-source-catalog-verification.json` records an isolated synthetic fixture with 25 small CSV files. Each of PL, MCD, Compare, Power and SHG had discovery-call counts **1 / 0 / 1** for cold / restarted unchanged / changed source cycles. Cached previews were published before validation.

Power measured about **177 ms cold / 17 ms warm** on that fixture. The small MCD fixture measured about **10 ms cold / 12 ms warm**: inventory and JSON overhead can exceed cheap discovery on small folders. These are catalog-worker measurements, not whole-application startup claims.

## Scope and limitations

The optimization skips expensive unchanged catalog discovery; it still traverses/stat-checks the data directory in the background. Known data modes ignore image-only changes. Inventory deliberately includes recursive data and JSON conservatively, so another mode's data/metadata change can cause extra invalidation. Except for DRR's existing inspection cache, changed catalogs can rebuild the whole requested mode; this is not a per-file parser cache for every workflow.

The first visit to an uncached folder/tab still requires discovery. The running instrument application was not restarted, source files were not changed, and the preexisting dirty tree remains uncommitted.

Two bounded review passes identified queued-force and stale-selection issues. Regression tests reproduced them before fixes. The second pass caught the shared PL/Compare preview case; that case was fixed and verified locally without requesting a third review.
