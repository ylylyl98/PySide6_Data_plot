# Astra independent review — responsive pages and pickers

Date: 2026-09-14. Verdict: **Needs correction; not ready for final delivery.** Application source was read-only during review. Comparison used `artifacts/responsive-implementation-baseline-20260914/source.zip`, normalizing CRLF/LF before diffing; unrelated pre-existing git changes were excluded. Review followed requesting-code-review and verification-before-completion.

## Reproducible findings and minimum repair requirements

### R1 [P1] Region drawing does not preserve correct visible scientific plots

Locations: `ui_qt/axes_region_blitter.py:27–46`; `ui_qt/controllers_pl.py:791–819`; `ui_qt/controllers_drr.py:2266` (`_draw_drr_regions`, locate current line after edits); `ui_qt/controllers_power.py:836–849`.

The helper captures only axes.bbox, never observes normal draw/resize, and leaves lines animated. Production callers change title/ylim and peak/fit overlays but invalidate only when bbox changes. Thus ticks/title/overlays remain stale and an ordinary canvas.draw removes animated spectra/gates. A new DRR/Power plot can have the same bbox and reuse a helper still bound to the old axes/line. Capturing multiple helpers separately also performs full draws while other helpers are animated. `restore_interactive_drawing` after `prepare_full_redraw` captures the visible line in its own background, creating a subsequent ghost. MainWindow export hooks only manage the MCD unified view, not these helpers.

Evidence: `astra-review-repro.py` reports 6975 differing RGBA channels for changed title/ylim against full render and 4530 differences following ordinary canvas.draw. This is not a timing-target issue. Existing 4 region tests pass because they explicitly call prepare_full_redraw for title/limit changes and never assert the production multi-axes pixels.

Repair: establish a coordinated canvas draw lifecycle (one owner or equivalent coordinated helper management), validate canvas/axes/artist ownership, disconnect old helpers on plot replacement, use draw/resize callbacks without recursive drawing, capture backgrounds with dynamic artists excluded, repaint all dynamic artists after full draw. Invalidate static axes including title/ticks/limits/theme/overlays and their tight bounds, or deliberately use full draw when necessary. Wire export prepare/restore. Test real PL, paired DRR and Power update→full pixel equivalence, changing limits/title/overlays, same-bbox source replacement, ordinary draw, resize/theme and exported curves/legends. Do not optimize by hiding the required static updates.

### R2 [P1] Power accepts changed/deleted files via old validation cache

Locations: `ui_qt/power_group_dialog.py:448–463, 621–636`.

Validation request identity now uses accepted catalog signatures instead of current stat, while `_accept_checked` still skips the worker whenever that key is cached. Deleting a source does not change the key, so stale prefetched results authorize acceptance. The reproduction demonstrates unchanged key after deletion and actual `_accept_checked` calling accept with the old cache. This violates the explicit final-acceptance freshness requirement even if the subsequent loader detects the race.

Repair: separate memory-only request identity from verified file identity and a fresh acceptance attempt token. Every user acceptance must asynchronously verify actual selected files (existence, size, mtime, then load/reuse only if signatures match); a returned worker acceptance token may complete that exact attempt once without recursively starting another worker. Guard folder/catalog generation/request/group/selection and dialog closure. Detached/no-pool dialogs must remain pending/error, not accept. Test mutation/deletion after prefetch, selection change, refresh while old worker runs, closed dialog, actual signatures returned, and no GUI I/O.

### R3 [P2] Scoped publications erase other modes' dates; old disk snapshots never migrate

Locations: `ui_qt/main_window.py:459–468, 4334–4345`; unchanged `core/source_catalog_cache.py:123–134`.

Publisher replaces BOTH PL and MCD mtime caches for every tagged result, ignoring metadata.modes, including scopes whose dictionaries are empty. Thus visiting/refreshing another mode loses accepted date ordering. No namespace/schema bump or missing-metadata rebuild was added. With unchanged inventory `_cached_folder_sources_worker` returns the old positional payload forever, so all dates remain unavailable until an unrelated disk change or forced rebuild. The executable repro confirms `legacy_payload_reused True` without invoking the builder.

Repair: only replace caches authorized by accepted scope and folder/generation, explicitly clear on folder replacement, and force worker rebuild when required tagged fields/version are missing (or version namespace while preserving cache mode behavior). Test PL→MCD→SHG/Power scoped publications, duplicate basenames, changed/deleted files, old persisted payload unchanged inventory, stale preview/scan. Update picker probes through real worker snapshot publication; current probe only assigns lists and clears caches (line 45), so zero stat does not prove date ordering or accepted snapshot behavior.

### R4 [P2] Debounce button disabling does not guard acceptance

Locations: `ui_qt/source_picker_dialog.py:110, 149–161`.

Double-click remains directly connected to accept; `_update_ok_state` also re-enables OK on any current-item change while the timer is active. Repro emits itemDoubleClicked while a no-match query is pending: button disabled but dialog result Accepted. Hidden/stale source selection can be accepted.

Repair: guard centralized accept/activation against pending generation, either synchronously flush the memory-only filter and revalidate the current row, or keep all acceptance paths disabled until current generation settles. `_update_ok_state` must respect timer/pending state. Test double-click, Enter/default button, keyboard selection change and asynchronous status refresh during debounce.

### R5 [P2] Compare still has GUI metadata reads and force refresh cannot refresh history

Locations: `ui_qt/controllers_compare.py:121–139, 320–356`.

Missing mtime still executes Path.stat on GUI (repro throws `GUI stat`). Controller writes are forwarded to owner, so the new owner-cache branch of `_cmp_refresh_history_cache(force=True)` simply copies its own old records and returns without requesting fresh history. Repro confirms old history retained and no refresh queued. A stale nonempty history for another folder also falls through to the old rglob/read_text path.

Repair: memory-only missing mtime=0 with owned refresh request; force means queue worker even with warm records, retaining known rows while pending. Separate cache-ready from cache-requested folder/generation; accepted worker publishes and refreshes open dialog/badges exactly once. Remove remaining GUI read fallback. Add cold and warm forced refresh tests with injected slow reads, history save then refresh, folder changes, eventual completion, and closed-dialog guards.

### R6 [P2] DRR missing-selected metadata has no completion path

Locations: `ui_qt/controllers_drr.py:1249–1262`; `ui_qt/main_window.py:4537–4555, 4666–4667`.

The previous exact-selected fallback is replaced with a catalog refresh. `old_source_files` only affects added/removed statistics; it is not passed as a selected-source input to `_scan_drr_catalog_worker`, whose include_all remains unchanged. An outside-partition/external selected file remains absent after completion, and rebuilding recommendations queues the same refresh again. Safe pending alone is not an implementation of the required async completion.

Repair: exact missing selection metadata worker or explicit selected records in catalog worker, with folder/catalog/selection/request scope. Merge only appropriate plain metadata; no unsafe recommendation while any selected source unknown. Coalesce duplicate requests and retain a terminal missing/error state instead of infinite retries. Test outside current partition, external path, selection change, missing file and dialog close with finite worker completion.

### R7 [P2, confirmed planned secondary risk] Hidden pending redraw still renders shared figure

Location: `ui_qt/main_window.py:2293–2308` (unchanged baseline code).

Repro calls the actual scheduled handler with loaded PL while active mode is Compare; it invokes `_plot_mode('PL')`. The required Task 9 diagnosis now confirms this risk; it is not an arbitrary new regression attributed to this diff. The report cannot simply defer it as lacking fixture.

Repair: retain dirty/range state while hidden and consume once on valid reentry with loaded-source/generation/figure ownership checks. Keep computation alive. Test leaving before timeout, latest controls on return, pending range then gate, and MCD Peak Shift ownership. Re-measure MCD return including next Qt event, not merely its ~7 ms slot.

## Additional must-do validation / implementation gaps

- Task 3 keys include raw query text (PL/MCD/Compare), so different queries yielding identical rows still construct all Qt items. PL/MCD keys omit theme colors; SHG was not migrated. Generate filtered immutable row descriptors once, include appearance/roles/flags; unkeyed fallback must always invalidate remembered key, including equal-row early return (`source_picker_dialog.py:234–239`). Add different-query same-row population spy, theme and keyed/unkeyed tests.
- MCD reentry key at `main_window.py:3906–3923` only records identities, selection, three controls and analysis generation. Add planned feature/display/theme/resize/in-place revision tests before claiming valid reuse. `_unified_energy_order` (`8720–8728`) has no source revision key. The change copies selected columns as intended, but new numerical regression coverage is absent: `test_mcd_compact_slopes` is an export layout test, not equivalence for branches/NaNs/windows/descending storage. Add exact planned numerical matrix against prior function behavior and invalidate cached ordering on supported source revisions. No speculative scientific bug is claimed without that evidence.
- SHG actual signal is correctly rewired to a view-only callback; processing identity excludes angle cursor by inspection. Existing test invokes callback on a bare owner and does not exercise loaded single/compare, enabled fitting, pending worker or actual spin signal. Add the specified integration spies and latest-cursor publication checks before claiming zero fit across those paths. Callback currently uses default 90 ms scheduler, not the suggested 0–16 ms coalescing; this is latency, not a functional blocker alone.
- Task 9 conditional theme/layout/Slides/shutdown items need actual diagnosis or exact bounded evidence explaining why deferred. The report currently labels fixture absence as disposition rather than carrying out required fixtures. Do not implement speculative redesigns; retain conditional thresholds.
- Final plan checkboxes remain unchecked; report contains contradictory interim completion/deferred claims. Rewrite as one authoritative final matrix after fixes. Include actual snapshot picker fixtures, next-event draw costs and export/local comparison matrix. Existing split-scale DRR failure is independently documented on baseline and is **not** a new blocker for this round.

## Verification performed during review

- `python -X faulthandler -m unittest tests.test_responsive_plot_regions -v`: 4 tests, exit 0.
- `python -X faulthandler -m unittest tests.test_responsive_catalog_metadata tests.test_responsive_picker_io tests.test_mcd_compact_slopes -v`: 6 tests, exit 0. These are focused shallow checks, not the promised full acceptance matrix.
- `$env:PYTHONPATH=(Get-Location).Path; python -X faulthandler artifacts/responsive-pages-and-pickers/astra-review-repro.py`: exit 0; exact findings saved in `astra-review-repro.txt`.
- Normalized ZIP comparison: production semantic changes confined to controllers_compare/drr/mcd/pl/power/shg, main_window, power_group_dialog and source_picker_dialog, plus new axes_region_blitter; existing mcd_unified_page unchanged. No unrelated user-change loss inferred from git diff.

Positive verified structure: worker tagged PL/MCD metadata exists; missing MCD lookup avoids resolver; unchanged identical content key skips population; spectrum Line2D reuse is implemented; DRR gate-only timer delay is zero while pending range flag remains independent; SHG angle wire no longer calls parameter-processing callback. These strengths do not offset the reproducible rendering/freshness defects above.


## Re-review after first Luna correction pass (2026-09-14)

Verdict remains **needs correction**. This section supersedes the resolved portions of the initial findings, and records the source snapshot reviewed before the coordinator reopened edits. Application source was read-only. Diagnostic: `artifacts/responsive-pages-and-pickers/astra-review-repro-round2.py`, output `astra-review-repro-round2.txt`, exit 0.

### Remaining actionable findings

- **R1 [P1], axes_region_blitter.py:34–48, 89–99:** `Figure.savefig(BytesIO(), format='png')` followed by moving a line leaves its old curve in the background: 5,883 differing RGBA channels against full render. Saving renders animated artists; draw_event captures that saved renderer into an interactive background. Likewise prepare_full_redraw→restore captures artists while still nonanimated. Capture must exclude dynamics before draw, ignore/invalidates saving renderer, and restore all interactive groups after export. A separate same-title/same-limits add-scatter test yields 322 differing channels: peak overlays are not part of static invalidation. MainWindow PL reconstruction resets helper collection to None without disconnecting old callbacks. Fix owner lifecycle, static overlay/theme invalidation, export→next-interaction pixels; ordinary draw and title/limit handling have improved and do not need reverting.
- **R3 [P2], main_window.py:4397–4408:** actual MainWindow accepted PL publication still turns seeded MCD mtime cache into `{}`. Cache compatibility/migration was fixed, but publisher scope logic was not. Existing scope test checks `_catalog_payload_compatible`, not actual publication. Replace only the caches covered by accepted metadata modes, and preserve inactive ones; test real sequential publications.
- **R6 [P2], main_window.py:379–390; controllers_drr.py:1249–1263:** external selected metadata is inspected relative to external parent, then appended without rewriting `source` to the user's exact/portable identity relative to the experiment root. Repro returns `outside.csv` for selected absolute path, so baseline exact-selection comparison still fails after completion. A present but uninspectable selected file returns no records and `missing_selected=()`, hence repeats success-without-progress. Ensure every exact requested identity yields matching metadata or a terminal unknown/missing/error entry, store that state per request/selection generation, and suppress automatic retry until explicit refresh/source change. The new deleted-file tuple test alone does not prove terminal behavior in the dialog.
- **R7 [P2], main_window.py:2349–2354:** hidden guard only applies to PL and additionally bypasses `active_mode is None`. MCD Peak Shift is represented by None, so even PL still draws over that page; DRR/Power/MCD have no guard. Repro invokes actual handler for all four loaded modes while active None: all call `_plot_mode`. Guard all relevant page ownership states and preserve dirty/range data until their valid reentry; test Peak Shift, not only PL→Compare.
- **R2 residual [P2], power_group_dialog.py:529–534, 651–669:** fresh owned worker acceptance now executes and rejects deleted/racing files. However returned decoded results are stored only under `('accept', token, catalog_key)`; `_group_changed` then `_validate` look up plain catalog_key and clear `_validation_results`. Real owned-pool dialog accepts with `_validation_results == {}`. The controller consumes this field for prevalidated handoff, so it loses all fresh results and repeats load; final pairing validation also does not use the fresh decoded pair. Preserve exact token-bound results through validation and controller handoff; actual callback tests must not mock `_group_changed`. Existing four freshness tests do not cover that completed chain. Also reject an old prefetch result/error after refreshed generation, rather than merely preventing accept.
- **R5 residual completion concern [P2], controllers_compare.py:335–344:** cold cache with folder already set, ready=False and pending=False is immediately promoted to ready without worker (repro: no queue, ready True). Because controller attributes forward to owner, this owner-folder branch is self-certification, not a separate accepted cache. Global publisher does not set ready/pending; only an open-dialog completion closure clears them. Verify a cold first request after folder reset and completion after dialog closes, then reopen; readiness must originate only from accepted worker publication. GUI mtime fallback is fixed and force=True now queues a worker.

### Resolved / supported improvements

- R4 central `accept` now blocks pending timer; independent double-click reproduction is blocked. `_update_ok_state` respects pending state.
- Legacy disk payload migration now forces worker rebuild before preview, covered by focused tests.
- PL/DRR/Power helper checks axes/artist identity as well as bbox; ordinary canvas draw bridge and title/limit signatures added.
- PL/MCD/SHG/Compare row descriptor keys now describe visible content instead of raw query; theme colors included. The unchanged unkeyed path invalidates the remembered key.
- SHG actual spin is wired view-only. New signal tests spy on scheduling/request calls, but the first replaces scheduling, and neither verifies final visible nearest-angle spectrum/title with fitting enabled and a real pending result publication. Do not overstate those as complete end-to-end fit/render tests.

### Additional plan/evidence limits

The final-r1/r2/r3 picker artifacts still come from direct-list assignment and clearing PL mtime (probe lines 33–45), not accepted worker snapshots. Their zero stat counts cannot certify correct nonzero dates or mtime sorting. A separate accepted/missing snapshot fixture and nonzero-date ordering checks remain mandatory.

MCD new metric tests are still incomplete: one test uses one pair label `p`, the four-metric test explicitly skips its Integral assertion, and neither covers branch/status equivalence or empty windows. Reentry signature and cached `_unified_energy_order` remain identity-only/no revision invalidation. Do not claim the full planned science/in-place-revision matrix until covered. No additional scientific defect is asserted solely from missing tests.

Task 9 Slides diagnostics measure `queue_list.count()` instead of filtered `available_list`, so reported zero rows are not a valid acceptance count; inspect/filter controls and assert actual matching records before using timing. The diagnostic confirms 1,000 cached thumbnails and says no source change authorized, which contradicts the user's already-authorized conditional plan. Theme/layout measurement remains unperformed; Peak Shift constructs an empty result and reports no complete fixture. Shutdown's bounded waitForDone demonstration supports deferring ownership redesign, but does not itself test application close handling. Complete valid bounded diagnosis/disposition; avoid unrelated speculative redesign.

Performance: final page artifacts contain stable-limit local updates around 7–27 ms but full-draw fallbacks around 220–324 ms. MCD warm return has render_count delta 0 but render around 232–245 ms in final-r1. Slot and first-next-event are not completion latency. Report fallback conditions/counts, median/p95 completed render and heartbeat separately. Missing target alone is not a functional blocker; Task 4 requires async split only if isolated synchronous analysis exceeds 75 ms median, not when Matplotlib full draw dominates. The threshold still needs a measured analysis-only disposition after rendering correctness fixes.

### Fresh verification

`python -X faulthandler -m unittest tests.test_responsive_plot_regions tests.test_responsive_catalog_metadata tests.test_power_accept_freshness tests.test_source_picker_dialog -q`: **32 tests, exit 0**, independently rerun. Their passing status is compatible with the reproduced defects because coverage misses the relevant completed production chains. Do not count the previously classified DRR split-scale baseline failure as this round's regression.
