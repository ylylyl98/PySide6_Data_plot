# DRR cache and refresh implementation plan

Goal: implement the approved follow-up audit while preserving numerical results, complete background groups, JSON persistence and wrapped filenames.
Spec: ../reports/2026-09-20-drr-further-performance-audit.md
Execution: inline, preserving the existing workspace changes.

- [x] Catalog: regressions for cross-instance decode reuse, mutation isolation, disk replacement invalidation and shared-grid round-trip; bounded memory snapshots, schema migration and exact grid references in source_catalog_cache.py.
- [x] External: prove saved recipes skip unrelated candidates and stale sources still fall back; lazy candidate inspection and request-scoped canonical paths in drr_auto_external.py.
- [x] Picker: identical publications must not rebuild groups; guard on immutable source/filter dependencies before cache invalidation, avoid unnecessary scroll-anchor scans.
- [x] Watchers: preserve watcher registrations within a folder, restrict MCD-only invalidation, asynchronous DRR validation whenever the picker opens; tests for newly arriving files and stable watcher identities.
- [x] History: cache per-metadata parsing with source dependencies and operation filtering; edit one metadata without reparsing unchanged siblings, retain race guards.
- [x] Export: reuse an index during a pair export with directory signature validation and freshly checked matched metadata; retain atomic writes and external edit detection.
- [x] Verify: focused unittest suites, independent hotpath timings and full real-data workflow timings, offscreen long-filename screenshots at multiple widths/DPI. Review changes and record measured limitations.
