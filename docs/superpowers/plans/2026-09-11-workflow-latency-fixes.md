# Workflow Latency Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. In this session use the available `executing-plans` skill and the explicitly authorized two Luna implementation units; Astra reviews their changes afterward. The user already selected this execution approach.

**Goal:** Remove six confirmed avoidable workflow delays while preserving numerical results, selection behavior, diagnostics and exported output.

**Architecture:** Batch repeated GUI work, reuse valid plot/data results, and move remaining substantial calculations or discovery I/O into existing owned-worker patterns. Worker requests capture plain immutable settings/source snapshots, retain worker ownership through completion, and reject obsolete generations or sources before updating the UI. Keep units independently testable without restructuring the large window module.

**Tech Stack:** Python in `.venv`, PySide6/Qt workers and timers, NumPy, pandas, Matplotlib, unittest.

**Spec:** `artifacts/all-workflow-latency-audit.md`, reviewed by `artifacts/workflow-latency-astra-review.md`; user authorized Astra plan → Luna implementation → Astra review.

## Final delivery status

- [x] Task 1: Slides batch insertion and one cached plan.
- [x] Task 2: SHG validated processing-result reuse for refits.
- [x] Task 3: Peak Shift owned background analysis, cache, latest-only queue and lifecycle safety.
- [x] Task 4: Compare gate-only artists, including approved blit addendum and toolbar/resize handling.
- [x] Task 5: MCD range/color redraw coalescing.
- [x] Task 6: Power worker catalog and source-selection/save-open handoffs.
- [x] Astra production review and final isolated regression verification.

The detailed steps below preserve the original implementation recipe; final delivery is recorded above rather than asserting every suggested test invocation was run verbatim. Final evidence: `docs/superpowers/reports/2026-09-11-workflow-latency-fixes-implementation.md` and companion review report. No commit was requested or created.

## Global Constraints

- Work directly on current `codex/file-selection-reliability` branch. Do not commit, reset, clean, stash or overwrite unrelated dirty work.
- Baseline copies of eight production files are in `artifacts/workflow-fix-baseline`; compare against these as well as current diffs when reviewing this change.
- Use `.venv/Scripts/python.exe`; no new dependencies. Test data/output/settings must be synthetic and temporary. Do not alter experimental files or production QSettings.
- Preserve both MCD Peak Shift methods and both channels, Compare intensity/VP behavior, SHG single/compare fitting, and existing export/history/source identity logic.
- No widget access from worker code. Treat captured arrays/results as read-only; copy any array before mutating it. Do not set write flags on shared production arrays. Retain source references for worker lifetime so identity keys cannot be reused while active.
- Cache validity includes source revision/identity and every relevant numerical processing setting; a display-only choice must not invalidate numerical results. Bound caches and clear on source invalidation/close as appropriate.
- Preserve existing cancellation, generation checks, latest-only pending work, close guards and strong worker references. Obsolete workers may finish, but must not clear a newer request's state or apply results.
- Performance acceptance uses call counts and event-loop/lifecycle behavior first. Synthetic wall times are supporting evidence, not brittle test thresholds or live-app guarantees.

## Ownership and execution order

**Luna A:** Tasks 1–2, owns `ui_qt/presentation_widget.py`, `ui_qt/controllers_shg.py`, and new `tests/test_slides_latency.py`, `tests/test_shg_fit_reuse.py`. May edit its existing presentation/SHG test modules if needed. No edits to `main_window.py`.

**Luna B:** Tasks 3–6, owns `ui_qt/feature_pages.py`, `ui_qt/main_window.py`, `ui_qt/controllers_compare.py`, `ui_qt/controllers_mcd.py`, `ui_qt/controllers_power.py`, and new `tests/test_workflow_latency.py`. Owns changes to shared responsiveness/source/lifecycle tests. A dedicated small worker helper is permitted if it reduces duplication; announce its path to the coordinator first.

Execution ownership adjustment: after finishing Tasks 1–2, Luna A also implements Tasks 4–5 and owns `tests/test_compare_mcd_latency.py`. Luna B retains Tasks 3 and 6. A changes only Compare plotting/click regions of `main_window.py`; B changes Power scan and Peak Shift lifecycle regions. Astra owns `tests/test_astra_workflow_review.py` for independent review regressions.

Units run independently. Within B implement Peak Shift, Compare, MCD, then Power; run focused checks after each. Root captures baseline and coordinates; Astra reviews once both units complete. Shared protocol changes must be sent to the coordinator before changing another unit's files.

### Task 1 — Slides batch queue updates and calculate each plan once (Luna A)

**Files:** `ui_qt/presentation_widget.py:629,904–1075`; new `tests/test_slides_latency.py`; existing `tests/test_presentation_widget.py`, `tests/test_presentation.py`.

**Interface:** Add `_append_queue_paths(paths: Iterable[Path]) -> int`; keep `_append_queue_path(path: Path) -> bool` as the single-item compatibility wrapper. `_rebuild_plan()` computes one plan list and passes that snapshot through titles, slide list and preview; selection-only preview reuses the valid snapshot.

- [ ] Add regression tests for ordered unique insertion, duplicates, remove/re-add, clear/re-add, drag/move ordering and title/group changes. Spy on `_rebuild_plan` and `plan_presentation_slides`: a 30-item bulk add must rebuild/plan once. Run new tests first and confirm existing per-insert behavior fails the count assertion.
- [ ] Add a batch guard around queue model notifications; suppress handlers during the transaction and perform one final labels/status/plan update. Prefer guarding expensive callbacks rather than blocking model notifications required by the view. Maintain normalized membership and record lookup maps; refresh those on actual queue/discovery changes, not once per item. Normalize each incoming path once, preserve current duplicate semantics and queue order.
- [ ] Implement the wrapper and use the batch entry point for Add selected, Add all shown and matching MCD pairs:

```python
def _append_queue_path(self, path: Path) -> bool:
    return bool(self._append_queue_paths([path]))
# Titles use the plan already computed by _rebuild_plan:
# planned_slide_title(plans[index], self.title_edit.text())
```

- [ ] Keep cached plan invalidation explicit for queue, grouping, plots-per-slide and title changes; caption/panel controls must still update preview correctly. Rebuild plans from the final ordered queue once after multi-row remove/move.
- [ ] Run `.venv/Scripts/python.exe -m unittest tests.test_slides_latency tests.test_presentation_widget tests.test_presentation -q`; verify no output order/title/content regressions.

### Task 2 — SHG refit reuses processed sweep results (Luna A)

**Files:** `ui_qt/controllers_shg.py:20–47,495–621`; new `tests/test_shg_fit_reuse.py`; existing `tests/test_shg_controller_regressions.py`.

**Interface:** Extend the internal worker request with optional validated `(result, result_b)` processing results. Keep controller callbacks and `LoadedState` consumers compatible. Define processing identity from source identity, data/background object revisions, comparison mode and frozen `ShgSettings`; exclude `ShgFitSettings`.

- [ ] Add tests that change only fit range/weighting/branch and assert `process_shg_sweep` is not called while fitting is called; single/compare processing results match the existing full path numerically. Change cosmic/background/source/settings and assert preprocessing runs again. Run these tests and record the expected pre-fix failure.
- [ ] Keep a bounded current-source processed-result entry separate from fits; seed it from an already loaded matching result, and update only when request generation/source is still accepted. Never reuse results when background/source/settings changed or a newer processing request is pending.
- [ ] Make worker logic distinguish processing and fitting, without reading UI objects:

```python
if processed is None:
    result = process_shg_sweep(data, settings, background=background)
    result_b = process_shg_sweep(data_b, settings, background=background_b) if compare else None
else:
    result, result_b = processed
# Existing fit_shg_angular_result / fit_shg_twist_comparison branches follow.
```

- [ ] Preserve 100 ms debounce and latest-only pending requests. Test rapid fit/processing/source changes with reversed completion order; old completion must neither apply stale fits nor publish a stale processing cache. Disable fit must reuse data and clear displayed fits as before.
- [ ] Run `.venv/Scripts/python.exe -m unittest tests.test_shg_fit_reuse tests.test_shg_controller_regressions tests.test_responsiveness -q`.

### Task 3 — Default Peak Shift owned worker and shared-method cache (Luna B)

**Files:** `ui_qt/feature_pages.py:1560–1600,1765–1833,2037–2103`; `ui_qt/main_window.py` only if source/close lifecycle needs registration; new `tests/test_workflow_latency.py`; existing MCD peak/lifecycle tests.

**Interface:** Split `_analyze_mcd_peak_shift_legacy()` into GUI request/result application and a plain numerical function consuming a captured source, numeric options and cancellation event, returning `{method: {channel: analysis}}`. Preserve public callbacks. Numerical key includes source object/revision, source selector, prominence, distance, smoothing, jump, peak limit and derivative window; excludes Raw/Second derivative selector and local-only background choice.

- [ ] Add tests for four calls with the correct methods/channels, event-loop responsiveness while the worker waits on a test Event, no recompute when switching cached Raw/Second derivative, and recompute after numerical/source edits. Confirm the existing synchronous route fails the dispatch/nonblocking assertion before implementing.
- [ ] Capture widgets/settings on GUI thread, retain source reference, enqueue owned worker. Run both methods for both channels; check cancellation between analyses. Do not alter core numerical algorithms or combined diagnostics.

```python
for channel in ("pos", "neg"):
    for method in ("Raw spectrum", "Second derivative"):
        if cancel_event.is_set():
            return None
        # Apply existing channel field/interpolation adaptation and analyze_peak_shift arguments.
```

- [ ] Extract existing result application, preferred-selection restoration, table population/status/error handling without changing semantics. Cache bounded results by numerical key. Tracker-only selection chooses its channel result from the shared cached result and redraws; result application must use the currently selected valid tracker.
- [ ] Keep only latest pending request when worker already active; invalidate generation on source/mode/local-fit transition and shutdown. Retain all active workers until finished. Test source switch, local-fit switch, error, close, and completion inversion; old worker must not re-enable controls belonging to a newer active request.
- [ ] Run `.venv/Scripts/python.exe -m unittest tests.test_workflow_latency tests.test_mcd_peak_shift tests.test_mcd_peak_display tests.test_mcd_peak_async_export tests.test_worker_lifecycle -q`.

### Task 4 — Compare gate-only update for control and heatmap click (Luna B)

**Files:** `ui_qt/controllers_compare.py:1160–1199,1259–1289`; `ui_qt/main_window.py:5647,6881–6902,7447–7575,7983–7999`; `tests/test_workflow_latency.py`.

**Interface:** Add `_update_cmp_gate_only() -> bool`; return false when currently shown source/view/axes/cache is incompatible, allowing existing scheduled full redraw. Reuse `_cmp_active_cubes`, `_cmp_heatmap_axes`, `_cmp_linecut_ax` and `_ensure_cmp_gate_lines`.

- [ ] Add intensity and VP tests changing gate through the spin callback and plot-click handler. Assert linecut/gate values equal full-render results and heatmap/colorbar axes identity is retained; spy that no figure clear, derived cube calculation or heatmap rebuild occurs.
- [ ] Retain linecut artists during full rendering, update their data/label/limits when gate changes, and call `draw_idle`. Preserve nearest-gate snapping, channel-specific sampling, legends, x limits and exports. Refresh only the linecut axes if artist structure changed; never clear heatmaps for a gate-only update.

```python
if source is self.cmp_spins["gate"] and self._update_cmp_gate_only():
    return
self._schedule_plot_redraw("Compare")
```

- [ ] Route heatmap click through the same helper after setting the snapped gate. Reject fast path while source/view/settings differ from the rendered snapshot, or a full redraw is pending for non-gate settings. Full redraw must rebuild/replace its snapshot; maintain plot bookkeeping and export invalidation.
- [ ] Add source/view/background change fallback tests and run `.venv/Scripts/python.exe -m unittest tests.test_workflow_latency tests.test_compare_source_workflow tests.test_responsiveness -q`.

### Task 5 — Coalesce MCD range/color redraw bursts (Luna B)

**Files:** `ui_qt/controllers_mcd.py:223–268`; `ui_qt/main_window.py` scheduler lifecycle only if required; `tests/test_workflow_latency.py`, `tests/test_responsiveness.py`.

**Interface:** Route existing full-render fallback through `_schedule_plot_redraw("MCD")`, preserving center timer and successful trace-only paths.

- [ ] Add test sending a burst of range/color changes before processing the timer, then assert one plot uses final values; old/source-switched/closed windows must not redraw stale data. Verify center/width still use the existing 40 ms route and processing still uses its 650 ms debounce.
- [ ] Replace direct range fallback with shared scheduling; preserve automatic-range refresh ordering and export invalidation. Keep `_refresh_mcd_trace_panel` fast paths; do not add incremental image/range complexity without need.

```python
if sender in trace_only_controls and self._refresh_mcd_trace_panel():
    return
self._schedule_plot_redraw("MCD")
```

- [ ] Run `.venv/Scripts/python.exe -m unittest tests.test_workflow_latency tests.test_responsiveness tests.test_ui_split_scale_controls -q`.

### Task 6 — Worker-built Power source catalog snapshot (Luna B)

**Files:** `ui_qt/main_window.py:_scan_folder_sources_worker,_refresh_file_lists,_on_file_lists_result`; `ui_qt/controllers_power.py:57,345–373,821`; `tests/test_workflow_latency.py`; existing Power/source tests. Avoid changing `core/data_io.py` unless an explicit reusable pure discovery helper is needed.

**Interface:** Extend the folder-scan result with a versioned/tagged Power snapshot containing folder, candidate identities/signatures and discovered `PowerSeriesSource` mapping. Existing tuple compatibility handling must not confuse it with SHG extras. `_power_current_sources()` returns the current in-memory snapshot without `stat`, CSV reads or synchronous discovery.

- [ ] Add tests patching `Path.stat` and `get_power_series_sources` to raise during GUI group refresh after a supplied snapshot; assert group metadata and selections remain correct. Worker test verifies full Power column values still discovered. Test changed/deleted files, folder switch and stale scan completion.
- [ ] Compute matching Power candidates, stats and `get_power_series_sources` inside `_scan_folder_sources_worker`; preserve candidate filtering and existing folder/generation guards. Snapshot contains all metadata needed by Power group display. Apply only current snapshot before `_power_refresh_groups`.
- [ ] On missing/invalidated snapshot, schedule catalog refresh and return current valid folder snapshot or empty mapping; do not fall back to GUI I/O. Ensure refresh and empty startup converge without recursive refresh loops; folder changes cannot briefly present old-folder sources. Selection/group edits that only reorganize cached sources must not invalidate file discovery unnecessarily.

```python
# GUI-side lookup: validity is established by accepted catalog generation.
if self._power_sources_cache is not None and self._power_catalog_folder == self.current_folder:
    return self._power_sources_cache
# Request/coalesce background refresh; preserve prior valid selection while pending.
return {}
```

- [ ] Adapt old direct-controller test fixtures to provide explicit catalog snapshots rather than restoring synchronous fallback. Retain file-change discovery through the normal manual/automatic refresh path.
- [ ] Run `.venv/Scripts/python.exe -m unittest tests.test_workflow_latency tests.test_power_header_cache tests.test_power_selection_regressions tests.test_power_regressions tests.test_power_prevalidated_load tests.test_responsiveness tests.test_worker_lifecycle -q`.

## Test isolation recipe

Use this pattern in new actual-widget test setup, alongside the existing QApplication singleton and normal worker cleanup:

```python
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
from PySide6.QtCore import QSettings

tmp = TemporaryDirectory()
settings_path = str(Path(tmp.name) / "settings.ini")
with patch("ui_qt.presentation_widget.QSettings", lambda *a: QSettings(settings_path, QSettings.IniFormat)):
    from ui_qt.presentation_widget import PresentationBuilderWidget
    widget = PresentationBuilderWidget()
    try:
        assert widget.queue_list.count() == 0
    finally:
        widget.close()
tmp.cleanup()
```

Tests using MainWindow must isolate its settings/configuration the same way and bypass unrelated hardware/PowerPoint integration. Use synthetic arrays, test Events and explicitly pumped event loops; cap waits and release Events in finally blocks. Add no fixed sleep-based latency assertions.

## Integration and Astra review

- [ ] Each Luna unit records files changed, focused failing-then-passing checks, any protocol decisions and remaining limitations to root. Do not commit.
- [ ] Root runs the combined focused modules once both units finish; investigate actual regressions rather than weakening stale-state/selection tests.
- [ ] Re-run `artifacts/benchmark_workflow_targets.py` using `.venv` with temporary outputs/settings. Keep the old JSON as baseline. Peak/SHG core algorithms intentionally retain their raw runtime; measure dispatch/call reuse separately to demonstrate those improvements. Slides bulk benchmark must call the new bulk entry point as well as checking single-item compatibility.
- [ ] Astra compares production changes against `artifacts/workflow-fix-baseline`, verifies cache keys, worker ownership/close safety, source transitions, preserved two-method diagnostics and exact numerical/output behavior. Review new tests for meaningful behavior checks rather than implementation mirroring. Return concrete findings; Luna fixes them and reruns affected checks before completion.
- [ ] Report six outcomes with evidence and bounded timing claims. Organizer cold preview and DRR SG misses remain deferred: bounded cache/artist reuse mitigate Organizer repeats, and only ~0.0145 s was measured for a 301 × 2048 SG miss; neither warrants expanding this authorized priority pass without representative evidence. No export rewrite is included because the audit reproduced no excessive export time at its tested size.


## Authorized addendum — Compare regional redraw (Luna A)

Root measured the implemented gate helper through actual offscreen drawing: 301 × 2048 data still takes roughly 582 ms p50, with an eight-event burst observed at 1073 ms (`artifacts/workflow-fixes-compare-timing.json`). Avoided reconstruction alone leaves full-canvas heatmap rendering. User-authorized implementation now includes this bounded extension to Task 4; Luna A owns it, Astra reviews.

**Files:** `ui_qt/controllers_compare.py`; Compare full-render/draw-event wiring and shared toolbar save hooks in `ui_qt/main_window.py`; focused Compare regression tests owned by A. Preserve B's Power/Peak Shift edits. Reference existing MCD regional redraw at `controllers_mcd.py:1650–1761`; do not rewrite MCD behavior.

- [ ] Animate the complete Compare linecut axes and heatmap gate-marker artists. Cache static backgrounds only after a valid full draw; redraw the linecut axes including title, ticks, legend and changed y limits, plus marker regions. Use padded regions large enough for old/new tick-label and title extents, so changes leave no trails.
- [ ] Invalidate backgrounds on source/render identity, figure/axes replacement, view/layout/range changes, canvas resize or renderer/DPI changes. Return to full `draw_idle` if backend lacks blit support or any cached region is invalid. A gate event before the first completed draw must take this fallback safely.
- [ ] Reuse the accepted render's automatic-background value when checking gate-only validity, or cache estimation by retained source objects/revision and all relevant numerical settings. Gate-only checks must not repeat whole-cube percentile estimation; changes to source/background controls must still force valid full rendering.
- [ ] Extend the existing toolbar save prepare/restore dispatch so Compare's animated linecut/markers are temporarily drawn normally during save and restored in `finally`, including canceled save/error. Existing MCD save hooks must run exactly as before; Compare must not override MCD's controller state. Standard application exports remain unchanged.
- [ ] Test real helper/render behavior with temporary settings and synthetic arrays: warm gate update retains heatmap axes and avoids full canvas draw; cold cache uses fallback; source/view/resize/DPI/range invalidation recaptures correct backgrounds; intensity y ticks/title reflect new limits; VP stays ±1.05 and preserves its zero line; no old gate/label trails remain in rendered pixels.
- [ ] Test toolbar save preparation/restore with mocked file chooser/output to a temporary PNG: Compare spectrum and gate markers appear in saved pixels, restoration also occurs on save error, and existing MCD toolbar-save regressions pass. Do not test only `animated` flags.
- [ ] Root repeats the same benchmark boundary, measuring actual blit completion instead of requiring a draw_event for successful warm updates. Report warm/cold behavior separately; rapid bursts with one completed sample remain a single observation, not a percentile. Keep numerical results identical and cite actual measured improvement without imposing flaky wall-time assertions.

Acceptance: cached valid Compare gate updates avoid heavy heatmap rendering and percentile recomputation; cold/invalid/save paths remain visually correct. Astra reviews stable implementation and lifecycle/save regressions before approval.


Addendum implementation refinement: prefer one clean full-figure background over separate axis regions. Keep heatmap axes nonanimated; animate only the full linecut axes and gate-marker artists. On draw_event capture the static full figure, then render dynamics; on warm gate update restore that frame, draw dynamics and blit the full figure bbox. This removes variable title/tick-region overlap risk at modest buffer-copy cost. The same invalidation/save/fallback tests above apply; never capture the old dynamic artists into the static background.
