# Responsive pages and pickers implementation report

Date: 2026-09-14

This implementation preserves the existing worker/controller and scientific
calculation paths while removing the confirmed picker metadata reads and
unchanged row construction.

## Implemented

- Added an append-only `source_metadata` worker snapshot containing PL, MCD,
  and Compare mtimes. GUI controllers consume the accepted snapshot; missing
  metadata is represented as `0.0` and never triggers a GUI stat/resolve.
- MCD source sorting/filtering uses the worker cache and a 140 ms debounce.
  Acceptance is disabled while a debounced query is pending.
- `SourcePickerDialog` accepts an optional immutable content key and skips
  staging/population when the key is unchanged. PL, MCD, and Compare supply
  keys that include visible content, statuses, mtimes, filters, and colors.
- Power saved assignment loading is part of the worker power catalog snapshot;
  the dialog no longer reads `.power-selection.json` in `refresh()`.
- MCD tab reentry checks result identity, selection, display parameters,
  generation, axes, and Figure ownership before reusing the unified view.
- SHG angle cursor changes use a dedicated view-only callback and do not
  request processing or fitting.
- PL spectrum gate updates now reuse the spectrum and gate artists, remove
  only controller-owned transient overlays, and retain title/tick/limit
  updates. Added `AxesRegionBlitter` with explicit invalidation, full-redraw,
  export-safe restoration, and disconnect lifecycle.
- DRR single/paired spectrum and Power spectrum gate updates also reuse their
  Line2D artists, preserving labels and overlays while allowing full redraw
  fallback when lifecycle state requires it.
- PL production path now owns two `AxesRegionBlitter` instances (spectrum and
  heatmap gate). Stable gate updates use region restore/draw/blit; layout bbox
  changes and overlay changes invalidate and recapture backgrounds.
- Both diagnostic probes support `--output PATH`; existing baseline JSON is
  preserved.

## Verification

`python -m compileall -q ui_qt core artifacts/responsiveness_audit_probe.py artifacts/file_picker_audit_probe.py`
completed successfully.

`python -m unittest tests.test_responsive_plot_regions -v` completed with
4/4 tests passing, including a real `PlController` canvas.draw spy and exact
`buffer_rgba()` local-versus-full render comparison. Coverage includes artist
identity, region fallback, title/limits, resize, and animated-flag restoration.

The full page probe was independently rerun with faulthandler and exited 0;
its JSON and stdout/stderr log are in
`artifacts/responsive-pages-and-pickers/diagnostics/pages-faulthandler.json`
and `pages-faulthandler.txt`.

`QT_QPA_PLATFORM=offscreen python -X faulthandler -m unittest tests.test_source_picker_dialog -v`
completed with 8 tests passing.

The initial GUI abort was root-caused to tests patching QObject class methods
with a raw `MagicMock`; PySide6 6.10.2 can access-violate while connecting a
signal to that non-native callable. The affected fixtures now use explicit
callables (`lambda self: None`). With that lifecycle correction,
`tests.test_lazy_source_catalogs` ran 12/12 and
`tests.test_source_picker_stability` ran 8/8; DRR picker/cache stability ran
13/13. Production watchers were not disabled.

## Residual disposition

Theme/layout cost, queued redraw after leaving a page, MCD Peak Shift complete
fixture, Slides large-directory thumbnail growth, and slow-worker shutdown
remain diagnosis-only because they require separate complete fixtures. No
speculative broad refactoring was applied. Full Figure draws remain the
documented fallback when ownership, layout, limits, or renderer state
invalidate cached artists.

## Final picker probe samples

Three fresh-process outputs were written to:

- `artifacts/responsive-pages-and-pickers/pickers-final-1.json`
- `artifacts/responsive-pages-and-pickers/pickers-final-2.json`
- `artifacts/responsive-pages-and-pickers/pickers-final-3.json`

All samples reported zero direct GUI metadata stat calls for PL and MCD,
zero calls during unchanged filtering, and six-key slot times below 1 ms.
For 1,000 files, settled debounced filtering ranged from 177–259 ms, while
unchanged filtering ranged from 0.58–1.67 ms. Follow-up full page probes
completed successfully in fresh processes:
`pages-final-2.json` and `pages-final-3.json` (exit 0). Unchanged MCD returns
had render-count delta 0 and slot times about 6–9 ms after initial render.
PL gate render times were about 117–125 ms; DRR remained about 216–234 ms
because the current path still performs a canvas draw after artist updates.
This is structural reuse with the latency target still missed.

The real-controller Task 6 regression run covered 38 tests. Power interaction
and combine behavior passed after handling test-double artists that reject
`remove()`. One DRR split-scale semantic test failed because its fixture
expected `split_scale=None` while the loaded second product retained its
configured split scale; it is unrelated to gate artist identity and remains
for focused DRR semantics review.

Exact failed test: `tests.test_drr_dual_view_regressions.DrrDualViewRegressionTests.test_raw_split_scale_does_not_override_second_color_limits`.
The command output was captured in the task runner transcript; no baseline
rerun has yet been made under the same fixture, so causality is intentionally
unclassified.

## Region follow-up

Production PL now configures spectrum and heatmap `AxesRegionBlitter`
instances from the real controller. `tests.test_responsive_plot_regions`
ran 4/4, including a real-controller `canvas.draw` spy and `buffer_rgba`
comparison. Three post-region full page probe outputs completed with exit 0:
`pages-after-region-1.json`, `pages-after-region-2.json`, and
`pages-after-region-3.json`. They retain render-count and draw/blit counters;
PL/DRR/Power timings still include other synchronous analysis and full-draw
paths, so the latency target is not claimed.

Performance probe outputs must be written to new names under
`artifacts/responsive-pages-and-pickers/`; the original audit JSON remains
untouched.

## Final validation status (2026-09-14)

The focused affected suite command was:
`python -X faulthandler -m unittest tests.test_responsive_catalog_metadata tests.test_responsive_plot_regions tests.test_responsive_picker_io tests.test_source_picker_dialog tests.test_drr_gate_toolbar tests.test_mcd_compact_slopes tests.test_mcd_analysis -q`
Result: 45/45, exit 0.

Fresh final probe outputs were written to `pages-final-4.json` and
`pickers-final-4.json`; both exited 0. The page probe uses the production
two-branch MCD fixture and reports unchanged reentry render delta 0. Picker
probe reports 0 direct GUI stat calls for PL/MCD and sub-millisecond six-key
slot work.

Task status: Tasks 1–4 implemented with focused evidence; Task 5 implemented
with production PL region blit and 4/4 pixel/controller tests; Task 6 DRR/
Power artist reuse and DRR zero-delay gate scheduling implemented, with
remaining DRR split-scale baseline failure separately recorded; Task 7 SHG
view-only callback implemented and callback regression covered; Task 8
Power/Compare boundaries improved, while DRR exact outside-partition metadata
worker and full changed/deleted acceptance matrix remain deferred; Task 9
theme/layout, queued hidden-page redraw, Slides memory, and slow-worker
shutdown are diagnosis/disposition items; Task 10 focused validation and
artifacts are complete, while screenshot export matrix remains deferred.

## Task 6/4 follow-up

DRR gate-only scheduling now uses a zero-delay coalesced timer while range,
axis, and parameter changes retain the 90 ms debounce. The pending range
refresh flag remains independent, so a later gate event cannot consume a
queued full refresh. `tests.test_drr_gate_toolbar` passed 5/5 in an isolated
fresh process.

MCD unified slope calculation caches energy ordering and indexes only selected
window columns before copying values. Integral, mean, absolute-mean, and
field-absolute metrics retain existing branch and NaN behavior.
`tests.test_mcd_compact_slopes` plus `tests.test_mcd_analysis` passed 23/23.

The combined runner once encountered a QtCore DLL procedure-load error after
prior native Qt process failures; affected suites were rerun in fresh
processes and only those fresh results are counted.

## Render acceptance

`python -X faulthandler artifacts/responsive-pages-and-pickers/render_acceptance.py`
exited 0 and wrote widget/export PNGs plus
`render-acceptance/results.json`. PL, DRR, Power, and MCD export images were
inspected with `view_image`; heatmaps, gate lines, spectra, legends, and MCD
annotations were present, and the PL local/full comparison showed no stale
curve ghost. Dark 150% shell capture was inspected as well. Fresh post-change
probes were also run to `pages-final-5.json` and `pickers-final-5.json`, both
exit 0.

## Astra R1 follow-up (current revision)

R1 plotting lifecycle fixes are exercised by 5 focused tests in
`tests.test_responsive_plot_regions`, exit 0. `AxesRegionBlitter` suppresses
its draw-event callback while capturing a clean static background, then
restores the dynamic artist directly; changed title, limits, ticks, or layout
perform a guarded static redraw before capture. PL helper selection checks
both Axes and artist identity, so a same-bbox replacement source cannot reuse
the previous source's background or curve.

The exact command was `python -m unittest tests.test_responsive_plot_regions
-q`. The fresh repro is
`artifacts/responsive-pages-and-pickers/astra-review-repro-r1-final.txt`;
its title/ylim local-versus-full pixel delta is 0, and the ordinary draw
scenario retains a visible dynamic curve. Its later Power section stops on
the parallel R2 dialog fixture (`SimpleNamespace.set_details`) and is outside
this R1 change. Render acceptance was rerun with exit 0 to
`artifacts/responsive-pages-and-pickers/render-acceptance-r1.txt`; PL, DRR,
Power, and MCD exports were inspected from the resulting PNGs.

## Astra R3 metadata follow-up (current revision)

Source catalog worker metadata now carries the exact requested mode, folder,
generation marker, and SHG mtime map in addition to PL/MCD/Compare data. The
cache worker validates that metadata and the catalog scope match the request
before publishing a preview. Legacy disk snapshots without scoped metadata,
or snapshots from another folder/mode, are retained for migration but force a
worker rebuild before publication. This prevents a PL snapshot from being
used as an MCD/SHG/Power catalog and preserves date maps from the fresh
inventory.

The combined command `python -m unittest
tests.test_responsive_plot_regions tests.test_responsive_catalog_metadata -q`
passed 13/13. The metadata tests cover old payload rebuild without preview,
folder/scope separation, tagged mtimes, and the DRR paired / Power same-bbox
artist ownership regressions.

## Final scientific follow-up (current revision)

`MainWindow._compute_unified_mcd_slopes` now applies the analysis window to
the canonical energy order before selecting value columns. This preserves
integral/mean semantics for descending storage while avoiding a full matrix
copy on every center change. A descending-storage equivalence regression was
added and passed.

Focused final command:
`python -m unittest tests.test_responsive_catalog_metadata
tests.test_responsive_plot_regions tests.test_drr_picker_startup
tests.test_mcd_compact_slopes tests.test_mcd_analysis
tests.test_shg_controller_regressions -q`

Result: 49/49, exit 0. The report remains the single status authority; review
items owned by the Compare and Power dialog agents are recorded there with
their separate test counts and are not duplicated in this revision.

The final scientific extension adds four-metric MCD coverage (including NaN,
duplicate columns, and descending storage) plus real MainWindow SHG cursor
signal integration. Single and compare loaded states perform view-only redraw,
submit zero processing/fit requests, and preserve a pending worker payload.
The fresh command `python -X faulthandler -m unittest
tests.test_responsive_catalog_metadata tests.test_responsive_plot_regions
tests.test_drr_picker_startup tests.test_mcd_compact_slopes
tests.test_mcd_analysis tests.test_shg_controller_regressions -q` passed
52/52, exit 0.

## Final frozen validation matrix

The final affected fresh-process run is recorded at
`artifacts/responsive-pages-and-pickers/final-affected-tests-20260914.txt`:
69/69, exit 0. The scientific extension run above is 52/52, exit 0. The only
classified baseline failure remains
`tests.test_drr_dual_view_regressions.DrrDualViewRegressionTests.test_raw_split_scale_does_not_override_second_color_limits`,
with independent baseline evidence in
`artifacts/responsive-pages-and-pickers/drr-split-baseline.txt`; no new test
failure is classified as pre-existing.

Fresh page probes `pages-final-r1.json`, `pages-final-r2.json`, and
`pages-final-r3.json` each exited 0. Fresh picker probes
`pickers-final-r1.json`, `pickers-final-r2.json`, and `pickers-final-r3.json`
each exited 0. Render acceptance exited 0 at
`artifacts/responsive-pages-and-pickers/render-acceptance-final.txt`; the
resulting PL export was visually inspected and retained its heatmap, gate,
spectrum, axes, and annotations. Full draw remains the explicit fallback when
layout/theme/export signatures change; local gate redraw avoids that path on
stable layouts. No additional performance claim is made beyond the recorded
probe timings.

## Astra R6/R7 follow-up (current revision)

DRR catalog refreshes now carry exact selected identities into the worker. The
worker inspects selected absolute files outside the experiment partition,
merges only successfully inspected `DrrSource` records, and reports deleted
or unavailable selections as terminal missing results. Refresh requests retain
coalescing and existing generation/folder/closing guards.

Hidden PL redraw requests remain dirty while another plot page is active; the
request is consumed only on valid PL reentry, preventing a hidden PL
controller from drawing into a shared Compare or MCD Peak Shift canvas.

The focused command `python -m unittest tests.test_responsive_catalog_metadata
tests.test_drr_picker_startup -q` passed 14/14. The latest repro log is
`artifacts/responsive-pages-and-pickers/astra-review-repro-r6r7.txt`; its
legacy-cache step stops at the intentional `SHOULD REBUILD` scanner failure,
confirming that the old payload is no longer published without worker rebuild.
