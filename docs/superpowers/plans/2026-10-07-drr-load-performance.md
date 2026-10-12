# DRR load performance implementation plan

**Goal:** Remove repeated history scans and unnecessary hidden-product work from DRR loading while retaining saved views, provenance, and export correctness.

**Approved scope:** The measured load bottlenecks in this chat and the user's instruction to implement the improvements.

**Architecture:** Cache parsed display-history catalogs and index snapshots by file signatures. Capture display-memory preferences in immutable load options. Reuse file digests only during loading, and defer hidden derivative work until display, an explicit Auto action, or export requires it.

**Tech stack:** Python, PySide6, NumPy, unittest.

## Constraints

- Preserve the existing uncommitted input, display-memory, and boundary changes.
- Keep source identities, verified aliases, file precedence, template tombstones, and paired-save completion semantics.
- External metadata edits, additions, removals, and source-path changes must invalidate history caches.
- Never use cached hashes for source cleanup; deletion still requires fresh hashes.
- Saved manual regional limits must survive deferred derivative export.
- Use isolated QSettings and QT_API=pyside6 for GUI tests.
- No commits or application restart are required.

## Task 1: Display history lookup

Files: core/drr_display_settings.py; tests/test_drr_display_history_cache.py.

- [x] Test an unchanged catalog being reused for multiple missing files, including different centers.
- [x] Test in-place metadata edits, additions, removals, pending-save completion, source moves, and mutable caller results.
- [x] Add bounded index/catalog caches guarded by locks. Build source and center lookup maps once per catalog signature.
- [x] Memoize physical path resolution within a scan and compute each record's source key once.
- [x] Add restore_files / restore_centers options to find_candidates; skip irrelevant legacy scans.
- [x] Re-run portable display-memory tests and the same 1000-sidecar benchmark used during diagnosis.

## Task 2: Preference-aware worker

Files: ui_qt/common.py; ui_qt/main_window.py; ui_qt/drr_display_memory.py; tests/test_drr_display_memory.py.

- [x] Test disabled memory making no history query on either the load worker or restoration path.
- [x] Capture both restore flags in LoadOptions and pass them into find_candidates.
- [x] Preserve manual Fix values and Auto behavior when memory is disabled.
- [x] Verify changed preferences and revisions do not apply an incompatible worker result.

## Task 3: Loading-only digest cache

Files: core/provenance.py; tests/test_provenance.py.

- [x] Test unchanged files avoiding repeated content reads during verification.
- [x] Test content changes with unchanged size/mtime, file replacements, missing files, and files changed during hashing.
- [x] Add a bounded digest cache keyed by stable file identity/mtime/ctime/size. Do not cache failed or unstable reads.
- [x] Make verify_initial_data_working_file use cached digests while can_cleanup and create_working_copy continue fresh hashing.
- [x] Verify cleanup rejects a modified working copy even after a warm load cache.

## Task 4: Deferred second derivative

Files: ui_qt/main_window.py; ui_qt/drr_display_memory.py; ui_qt/common.py; a focused pure export helper if needed; tests/test_drr_display_memory.py and derivative/export tests.

- [x] Test raw split loading without a second-derivative request, then switching to second/both triggering the request.
- [x] Keep shared boundary geometry synchronized without computing the hidden derivative.
- [x] Compute automatic hidden regional colors on the export worker, retaining fixed, restored, or explicitly edited limits.
- [x] Verify deferred callbacks cannot apply to a replaced cube and repeated requests remain coalesced.
- [x] Verify raw/second paired PNG/DAT exports and saved display metadata.

## Final verification

- [x] Run relevant history, memory, provenance, input, derivative, and paired-export suites.
- [x] Run compileall and git diff --check.
- [x] Request a bounded read-only code review, fix important findings, and repeat checks affected by fixes.
- [x] Document cache invalidation and report measured query improvement separately from full-load latency.

## Verification results

- Latest main regression runs: 59 core/history/provenance/input/derivative/export tests and 122 GUI memory/save/view/reuse tests passed. A subsequent focused run verifies deferred-cube release and its affected lazy/export paths.
- compileall for core, ui_qt, and tests and git diff --check passed.
- Read-only review found cache-stamp freshness, placeholder validation, finalized regional metadata, toolbar pending Auto, and malformed-marker issues; each has a regression and has been corrected. Final bounded review reported no remaining P1/P2 findings.
- Windows strong stamps include FILE_BASIC_INFO.ChangeTime. If unavailable, readable history remains usable but is not cached. Cleanup never uses the loading digest cache.
- GUI fixtures for dual view/regressions now use explicit temporary INI QSettings instead of relying on platform-global format settings.
- Isolated synthetic benchmark with 1000 sidecars and existing source paths: first catalog query 5673.59 ms; warm missing-file/center queries median 146.10 ms; indexed exact+center median 1.41 ms. Prior repeated uncached query median was 5154.58 ms. Catalog rebuild reusing unchanged records measured 499.89 ms. These timings cover history queries, not total file loading or first rendering.
- First-time legacy history parsing, CSV parsing/copying, raw ROI statistics, and Matplotlib first rendering still have their original costs.
