# YZ365 DRR latency diagnosis

Read-only application-code investigation. User confirmed YZ365. Measurements used existing YZ365 inputs; test exports went to an automatically cleaned temporary directory. The running acquisition application was not changed.

## File selector

The DRR worker publishes a cached source-list preview, but `_open_drr_source_dialog` subscribes only to `drr_catalog_refresh_finished`. `_on_drr_catalog_refresh_result` updates `drr_available_sources` and returns on a preview without notifying that dialog. A dialog opened before the preview can remain empty until full completion. This is a missing UI notification in the previous cache-first implementation.

The persisted DRR catalog was 5,641,185 bytes with 235 source records and 1,645 inventory entries. A read measured 0.315 s; inventory validation 0.182 s; the inventory matched on that run. The wrapper reads/deserializes the cache once for preview and again within refresh.

Forced discovery under cProfile took 3.581 s, including 2.704 s in four full CSV gate inspections and 0.619 s reading/resolving metadata. Instrumentation materially adds overhead: a subsequent non-profiled discovery took 1.552 s with five CSV inspections; a second discovery using the same warmed in-memory inspection cache took 0.668 s with zero CSV inspections. The differing inspection counts indicate the fixture was not static across these reads; do not treat these as a fixed universal startup time.

Explicit selector Refresh passes `auto=False`, so it intentionally bypasses the whole-catalog cache. Per-file inspection cache reuse remains available. Discovery still traverses all `Initial Data` subfolders, including PL and REF, before classifying sources.

## Save

Representative export used the most recent saved YZ365 raw DRR and d2E numerical matrices (raw shape 295×1340), rendering new images and DAT files in a temporary directory. It did not invoke the live window's Save or measure UI queueing, original raw-source loading, or its exact display parameters.

- Fresh two-product export: 1.560 s.
- Two PNG outputs: 0.604 + 0.607 s.
- Two DAT outputs: 0.162 + 0.158 s.
- Metadata writes: 0.0017 + 0.0015 s.
- Identical second export in the same temporary directory: 0.010 s, reusing existing outputs.

After a DRR export, `_on_export_done` requests another DRR catalog refresh. Catalog inventory includes output DAT/JSON, so new exports invalidate the whole catalog and cause discovery/metadata work again. Filesystem-watch refreshes may also queue a follow-up. This adds observable background activity after file writing; the exact additional latency was not measured end-to-end in the live app.

## Corrective priorities

1. Deliver cached previews to already-open DRR dialogs without marking validation complete.
2. Deserialize each cache once per request, and separate raw-source inspection invalidation from saved-result status updates.
3. Scope discovery to REF for the newly standardized layout, keeping the approved legacy fallback.
4. Optimize PNG rendering only after the extra discovery work is removed; retain numerical export precision.

Measured artifacts: `artifacts/drr-latency-investigation.json` and `artifacts/drr-save-latency-investigation.json`. No application fixes were made during this diagnostic turn.
