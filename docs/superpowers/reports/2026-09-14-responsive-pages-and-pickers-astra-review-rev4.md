# Astra final independent review — rev4 frozen snapshot

**Verdict: MUST FIX; not READY.** Date: 2026-09-14. Application source was read-only. All files in `artifacts/responsive-pages-and-pickers/rev4-source-manifest.json` matched their frozen SHA256 values. This round was compared to `artifacts/responsive-implementation-baseline-20260914/source.zip`, not the user's unrelated git diff. Earlier review detail is preserved in `2026-09-14-responsive-pages-and-pickers-astra-review-history.md`.

## Reproducible remaining defects

### R1 [P1] Power compare blitters overwrite other channel curves

`ui_qt/controllers_power.py:825–851` creates one helper per spectrum line although those lines share one axes. Each helper restores the entire axes background, which contains or removes the other helper's curve. Independent actual `PowerController._draw_power_regions` reproduction updates two colored lines and differs from the full renderer by **4,395 RGBA channels**. This is a scientific-display defect, not a missed timing target.

Repair: group all dynamic artists sharing a region/axes under one helper, including paired gate lines sharing an axes; disconnect helpers no longer referenced. Ensure ordinary full draw, export, source replacement and overlay invalidation repaint each region only once. Add actual two-channel Power local/full pixel comparisons after both lines move, plus legend/overlay and single↔compare transitions. Existing paired-helper test checks identities/nonwhite pixels and cannot detect this corruption.

Single-helper title/limits, ordinary draw, savefig and prepare→restore have improved. The earlier single-line export ghost reproduction no longer justifies reverting those fixes.

### R2 [P2] Late old Power result cancels the newer acceptance request

`ui_qt/power_group_dialog.py:513–535`: a stale request result is still cached and passed to `_group_changed(preserve_accept=False)` before being rejected for acceptance. That call clears the current request. Reproduction uses a real PowerGroupDialog with current acceptance token 2; delivering token 1's result changes `_accept_after_validation` to False and its key to None. A user changing selection then clicking Open while an old validation runs can lose the new operation.

Repair: capture request/catalog generation and selection identity in every worker callback; reject stale results/errors before changing caches, details, group state or acceptance state. In particular, an old callback cannot cancel an active newer token. Test full real callback chain with A→selection change→B, then out-of-order A/B results and refreshed/closed dialog. Current fresh-worker existence/signature checks and preserving decoded results through accepted handoff are useful and should remain.

### R5 [P2] A previous Compare scan certifies a queued forced refresh

`ui_qt/controllers_compare.py:341–352` combines request generation with the shared `_catalog_ready_modes` token. If Compare scan 5 is running, a forced refresh queues another Compare scan but stores generation 5 as the request. Accepting the earlier scan 5 adds Compare to ready modes. Calling `_cmp_refresh_history_cache()` at that point sets ready=True/pending=False while `_catalog_pending_requests == {'Compare': True}`. This was reproduced through actual MainWindow `_on_file_lists_result`. Later completion can fail to refresh an open dialog because its history pending flag was already consumed.

Repair: bind readiness to the accepted result satisfying the exact requested refresh, not merely any accepted Compare result. A separate published generation/request token or explicit pending-Compare chain tracking can work; ensure pending force isn't cleared by an intermediate publication. Test real coalesced cold/warm refresh, close→publication→reopen, and repeat reads before final publication. Do not restore GUI stat/read fallback. Simple noncoalesced cold/forced readiness and missing-mtime memory reads now work.

### R6 [P2] DRR exact-selection merge rejects known files and loses relative identity

`ui_qt/main_window.py:376–393`; consumer `ui_qt/controllers_drr.py:1249–1263`.

- For a valid selected source already in the catalog, `identity not in known` is false. The loop falls through to its else and reports that valid source in `missing_selected`. Reproduction returns `('known.csv',)`. Production sends all selected files when any metadata is missing, so mixed known/external selections trigger this.
- For an outside-partition source selected as `other/outside.csv`, the worker appends an absolute `source`, while `_baseline_measurements` uses exact string equality against the original relative selection. Reproduction returns a nonmatching identity, so successful inspection does not finish the selection.

Repair: treat an already-known matching identity as resolved; normalize or preserve one identity representation consistently on both worker and consumer sides while keeping the correct physical path. Each requested source must produce matching metadata or a terminal failure. Also bind terminal missing publication to the captured selection/request so an old missing selection cannot clear a newly selected loaded DRR view. Test mixed known+external, relative subfolder, absolute external, same basename, uninspectable/deleted files, and selection change during inspection; assert eventual recommendation or terminal status without repeated automatic scans.

## Required evidence corrections

- `render_acceptance_rev3.py:57–65` creates dark-150 with `demo_data=False` and never loads or plots anything. Its export is uniformly dark and `results.json` explicitly says `axes: []`. The pixel metric `<245` incorrectly counts every dark background channel as content. This is an invalid fixture, not proof of a production blank-render defect. Generate a loaded dark/150% plot and assert nonempty axes/curves plus actual pixel variation. Use a fresh process for scale setup. The current MCD fixture overwrites all branches as `B increasing`, and features are empty; it does not validate the planned complete fit/legend matrix. Add a valid two-branch fitted fixture and actual paired DRR/Power exports.
- `artifacts/file_picker_audit_probe.py:87` still assigns `same_result_scroll_preserved = True`. Selection and scrollbar are not set and compared across the same-result refresh. Therefore those reported assertions are not measured facts. Set a middle selected row and nonzero scroll before refresh and compare after; count the population callback itself, not merely refresh requests. Nonzero accepted mtimes, reversed date ordering and actual completed 100/1000 samples are valid improvements.
- The visible xlabel/title crowding in exports is not attributed to this round without a matching baseline image. Do not expand scope into unrelated layout redesign based on these synthetic figures.

## Verified/resolved portions and boundaries

R3 now preserves unrelated PL/MCD cache scopes and rebuilds legacy disk snapshots. R4 central accept blocks pending debounce paths. R7 checks all active modes including None/Peak Shift and retains dirty work for reentry. MCD selected-column calculation now keys energy order on its values, and added numerical tests cover Integral with equal-NaN semantics. SHG real display integration exercises latest nearest-angle publication. These are materially improved; they do not negate the reproduced defects above.

The final picker captures support roughly PL 720 ms / MCD 392 ms open-to-shown for 1000 files, with sub-ms synchronous keystroke work and a later debounce/layout cost. They do not establish sub-25-ms completed filtering or opening. Page render medians/p95 include expensive full-draw fallbacks; keep those disclosures. Missed timing targets alone are not new functional bugs. Measured MCD analysis-only 4.19 ms is below the conditional 75-ms worker threshold. Task 9 Slides cache/filter, resize attribution, incomplete Peak Shift fixture and limited shutdown demonstration remain explicitly deferred risks; this review does not authorize broad speculative redesign or relabel them fixed.

Known DRR split-scale and documented MCD/auto-refresh failures reproduced on the exact baseline are not treated as new regressions.

## Fresh verification and executable evidence

Run the standalone reproduction with the project interpreter:

`./.venv/Scripts/python.exe -X faulthandler artifacts/responsive-pages-and-pickers/astra-final-review-repro.py`

Exit 0, recorded in `astra-final-review-repro.txt`. Output: Power pixel delta 4395; stale Power result cancels newer attempt; queued Compare force remains while ready prematurely becomes True; valid DRR source reported missing; relative DRR source mismatch. These are defect observations, not passing assertions.

Independently rerun in separate offscreen processes: responsive plot regions 8/8; Power acceptance freshness 5/5; catalog metadata 15/15; Compare catalog async 5/5; SHG display integration 5/5. **38 tests passed**, but omit the reproduced scenarios. An initial wrong SHG module name was a command-selection error; the correct discovered module was then run successfully. Passing tests and script exit codes are not substituted for the actual behavior review.
