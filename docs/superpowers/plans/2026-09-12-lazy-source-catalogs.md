# Lazy source catalogs implementation plan

**Goal:** Implement the user's approved cache-first, background-validation, lazy-tab discovery workflow.

**Architecture:** Keep existing selection widgets and source discovery routines. Add a versioned JSON catalog cache per folder and mode outside experiment directories. Workers publish a cached preview before validating file size/mtime inventory. Only the requested mode performs discovery. Existing generation guards reject stale workers; preview results never complete the validation lifecycle. Preserve manual Refresh as a force-refresh operation. DRR retains its per-file inspection cache beneath the new catalog cache.

**Constraints:** No restart of the live instrument app, no changes to source data, no commits of the existing dirty tree. Existing selections survive background refresh where still present; missing sources remain subject to existing load validation. Cache is advisory and corrupt/unwritable caches fall back to discovery. Initial cache read and validation both run off the GUI thread.

1. Add and test `core/source_catalog_cache.py`: JSON roundtrip, atomic storage, version/root isolation, restart reuse, changed/add/delete invalidation and forced refresh. Reuse expensive discovery only when inventory agrees.
2. Split `_scan_folder_sources_worker` by optional mode while preserving its all-mode legacy call contract. Return mode metadata and merge untouched catalogs on application. Move Compare history reading to its worker. Add cache-preview signal with separate lifecycle handling.
3. Schedule startup, tab activation and watcher refresh for the active mode. Track which modes need validation for a directory. Do not eagerly queue DRR or Power from another mode. Tab activation requests missing/invalidated catalogs; manual Refresh forces current-mode discovery.
4. Give DRR a cached source-list preview while preserving inspection cache and independent lifecycle. Avoid repeated Slides scans on unchanged tab re-entry; explicit Refresh remains available.
5. Add regressions for lazy discovery, cached-first publication, stale folder callbacks, retained selections, forced refresh and inactive catalog preservation. Run relevant existing source/picker/latency suites, fix regressions, and request one concentrated review before final verification.

**Validation:** `.venv/Scripts/python.exe -m unittest` targeted modules; restart simulated with fresh cache objects and temporary data directories, not the user's running application. Report measurements from synthetic cold/warm discovery without asserting universal startup speedups.
