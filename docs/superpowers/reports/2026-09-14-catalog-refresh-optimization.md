# File catalog refresh optimization

Normal Refresh now validates and reuses the persisted catalog instead of forcing
discovery. The existing background workers, preview publication, folder/generation
guards and selection preservation remain in use. Right-click the shared Refresh
button and select **Rebuild File Catalog** to force discovery for the active mode.
Explicit rebuild requests survive coalescing with later ordinary refreshes,
including DRR.

Power discovery reuses persisted table records whose relative filename, size and
nanosecond modification time still match. Only new/changed tables reread their
Power column; deleted files disappear and legacy groups are rebuilt normally.
Records with empty Power values are not reused during discovery, so incomplete
inspections do not become per-file cache hits. Explicit rebuild bypasses this
reuse. The existing JSON catalog already contains the necessary per-file records
and signatures, so this change does not add a database or migrate stored data.

Catalog previews and validation now share one decode, with compatibility checked
before publication or reuse. Inventory enumeration uses `os.scandir` and retains
DirEntry metadata, avoiding separate per-file Path stat/symlink queries on Windows.
Recursive metadata validation remains: this is not a watcher-only index. A changed
catalog still gets a second inventory check before persistence to avoid certifying
a snapshot built across a concurrent acquisition. Watchers, discovery of newly
added directories and external deletions retain the existing reconciliation path.

## Measurements

Windows, local temporary files, five repetitions, median, warm filesystem caches;
not a production-data or network-drive benchmark. Inventory outputs were checked
for equality against the HEAD implementation before comparing timings.

| Case | Time |
| --- | ---: |
| Inventory before: 3,000 CSVs in 40 directories | 408.87 ms |
| Inventory after: same tree | 47.93 ms |
| Normal unchanged refresh: 80 Power CSVs, 2,000 rows each | 85.04 ms |
| One changed Power file: same dataset | 439.59 ms |
| Explicit full rebuild: same dataset | 909.95 ms |

The inventory comparison measures enumeration alone. Power measurements exercise
the catalog worker, including metadata/discovery and catalog serialization, but
exclude GUI rendering and application startup. They compare refresh paths in the
updated code; the rebuild figure is not a measurement of the old entire app.

## Verification

109 relevant unittest cases passed in the final run (17.723 s). Modified Python
files passed compilation and the scoped diff passed `git diff --check`.
Independent read-only review found no actionable issue in the refresh changes.

Relevant unittest coverage includes normal versus explicit refresh, queued rebuilds,
cached preview ordering and compatibility, added/modified/deleted Power sources,
same-size edits, explicit rebuild after preserved-timestamp edits, corrupt caches,
concurrent file changes, source selection stability and DRR history layering.

An expanded run also exposed four failures in the MCD curve tests in
`test_responsive_catalog_metadata`. Those tests exercise
`_compute_unified_mcd_slopes`, whose pre-existing uncommitted implementation expects
`view.window_traces`; their SimpleNamespace test views do not provide that method.
That implementation and those tests were not modified by this refresh change.
Substituting the HEAD MCD method in memory removed the missing-trace errors but
left two numeric expectation failures; no MCD fixes are included here.
