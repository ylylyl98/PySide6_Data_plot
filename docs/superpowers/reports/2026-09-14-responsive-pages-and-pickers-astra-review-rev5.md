# Astra independent final review — rev5

**Verdict: MUST FIX — two remaining asynchronous completion defects.** Reviewed 2026-09-14. Production source remained read-only. All 93 files in `artifacts/responsive-pages-and-pickers/rev5-source-manifest.json` match their frozen SHA256 values. Scope is the exact `artifacts/responsive-implementation-baseline-20260914/source.zip`, with line endings normalized, not the user's unrelated git changes. Prior rev4 findings are preserved in `2026-09-14-responsive-pages-and-pickers-astra-review-rev4.md`.

## Remaining must-fix findings

### R2 [P2] Old prefetch result cancels an active acceptance

`ui_qt/power_group_dialog.py:520–540` rejects stale `accept` keys, but allows an earlier ordinary prefetch/catalog key through. It then calls `_group_changed(preserve_accept=False)`, clearing the newer acceptance token/key before its worker finishes. The error callback already has the missing active-accept/nonmatching-key guard at line 574.

Independent real PowerGroupDialog callback reproduction: seed current acceptance token 2, deliver an older prefetch result; `_accept_after_validation` becomes False and `_accept_validation_key` becomes None. The original older-accept reproduction is fixed, but does not cover prefetch workers started before Open is clicked.

Minimal repair: reject noncurrent callbacks before any cache, UI or acceptance mutation while an acceptance is active; consistently capture/check catalog generation and selection identity for prefetch callbacks as well as acceptance callbacks. Preserve the current request through old success and error delivery. Test prefetch A → user acceptance B → result A → result B, asserting B still completes once with its validated decoded result. Also retain stale acceptance and refreshed/closed-dialog cases.

### R5 [P2] Final queued Compare publication never clears pending after the dialog closes

`ui_qt/controllers_compare.py:346` requires published generation exactly equal to the stored request generation. When forced refresh is queued behind scan 5, the controller stores 5. MainWindow starts the queued worker as generation 6 (`ui_qt/main_window.py:4333–4340`) and publishes 6. With no live dialog completion callback, subsequent cache reads/reopening cannot accept 6 and return indefinitely through the pending guard. Preventing generation 5 from prematurely finishing the request was necessary, but does not finish the follow-up request.

Independent reproduction uses actual MainWindow `_on_file_lists_result`, `_run_pending_catalog_refresh`, real worker `run()` and final publication. Only thread-pool scheduling is made synchronous to produce a deterministic callback order. Final observed state: requested=5, published=6, pending requests={}, `_file_refresh_pending=False`, history ready=False, history pending=True. No GUI filesystem fallback is involved.

Minimal repair: bind the pending request to the actual queued refresh token/generation, or consume a qualifying later accepted same-folder Compare publication after the full pending chain ends. Retain the intermediate-publication guard. Test close → queued follow-up publication → reopen, including empty valid history, and assert ready=True/pending=False after generation 6 while generation 5 still leaves pending. The existing dialog callback can mask this defect while the dialog stays open; exercise the no-dialog path.

## Runnable independent evidence

Run from the repository root:

```powershell
.venv/Scripts/python.exe artifacts/responsive-pages-and-pickers/astra-rev5-followup-repro.py
```

The fresh run exited 0; observations are recorded in `artifacts/responsive-pages-and-pickers/astra-rev5-followup-repro.txt`. This is a diagnostic, not a passing assertion suite. Its final-chain interface invokes `_run_pending_catalog_refresh()` with `QThreadPool.start` replaced by an ordinary callable invoking `worker.run()`, then `_cmp_refresh_history_cache()` after actual final publication. Do not weaken the final ready/pending assertion to merely checking the intermediate request remained pending.

## Resolved findings and retained evidence

- R1: actual two-channel Power local/full rendering comparison now has **0 differing RGBA channels**, independently rerun. Grouping all dynamic spectrum artists under one axes owner fixes the previous overwrite. Existing single-helper export/restore and lifecycle evidence remains applicable.
- R6: independent worker reproductions now return no missing entry for a valid known selection and preserve the exact relative external selection identity. Selection-scoped terminal handling is retained.
- R3 scoped metadata/legacy snapshot migration, R4 debounce acceptance and row content keys, R7 hidden dirty reentry, MCD numerical selected-column/in-place energy handling, and SHG latest-angle display retain prior reviewed coverage; no additional reproducible blocker was established in this round.
- Independently viewed rev8 dark export: loaded map and spectrum are visible, unlike the former blank fixture. Independently viewed MCD export: Fits 6/6 and fitted lines/legend are visible. Features remain a placeholder and label crowding remains visible; these exports do not establish a complete feature-analysis matrix or a new baseline-attributable layout regression. True screen DPI is not established by the export image alone.
- Current picker probe records actual middle selection, nonzero scroll, row identity and population callback counts across an equivalent query. It replaces the former literal-True evidence. Accepted snapshot/date metadata and completed 100/1000 samples remain useful evidence.
- Luna's `rev5-affected-tests-clean.txt` reports 204 tests passing; that count is not an independent guarantee of the callback cases above. No claim is made that this review reran all 204 tests.

## Performance and disposition

Keep rev5 page samples, rev5 picker samples and rev8 rendering evidence when repairing only these two callbacks. Do not present slot/next-event latency as completed render latency or describe every interaction as a few milliseconds. The report correctly retains slow full-render fallbacks, picker construction/debounce cost, MCD rendering, and conditional secondary-workflow limits. Slides cached filtering, unsplit resize timing, incomplete PeakShift fixture and incomplete MCD feature view are documented limits, not newly established scientific defects. Baseline-confirmed DRR split/MCD/auto-refresh failures do not block this round.

After R2/R5 repair, run their real callback completion regressions and affected tests, record the new source manifest, and perform a bounded final re-review. No other newly reproduced must-fix issue remains in this review; do not expand implementation scope or rerun unrelated performance probes solely because these two asynchronous paths change.
