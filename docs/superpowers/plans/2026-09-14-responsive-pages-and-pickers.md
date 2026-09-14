# Responsive Pages and File Pickers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. User-selected execution is one Luna executing the complete plan, followed by independent Astra review; do not ask for the execution choice again.

**Goal:** Remove confirmed synchronous picker metadata work and redundant page rendering while preserving scientific results and source-selection behavior.

**Architecture:** Extend worker-owned catalog snapshots and compare content before Qt item creation. Preserve controllers, shared Figure ownership, request generations and asynchronous loading; reuse plot artists and redraw dirty regions. Diagnose remaining static risks before modifying those paths.

**Tech Stack:** Python 3.13, PySide6, NumPy, Matplotlib QtAgg, unittest, existing Worker/QRunnable lifecycle.

**Spec:** `artifacts/responsiveness-audit-report.md`, including its supplementary file-picker audit. Measurement implementations: `artifacts/responsiveness_audit_probe.py`, `artifacts/file_picker_audit_probe.py`.

## Global Constraints

- Do not change scientific computation semantics, branch labels, nearest-row behavior, integration, fitting, background selection, or export results.
- Preserve selection, scroll positions, full wrapped filenames, statuses, flags, tooltips, keyboard operation, and selected Power duplicate-row identity.
- Preserve folder, request, source, selection and analysis generation checks. A warm cache cannot certify a deleted or changed source.
- No React/QML conversion, large model/view rewrite, new runtime dependencies, or cross-controller redesign.
- Workspace contains substantial user changes. This round's baseline is `artifacts/responsive-implementation-baseline-20260914/source.zip` and `manifest.json`. Do not reset, clean, restore, stash or commit user changes. Compare this round to that ZIP for review.
- No applicable AGENTS.md was found in the workspace or checked ancestors during planning; recheck if execution workspace differs.
- Run GUI suites in separate Python processes with offscreen platform, QtCore imported first, one QApplication retained for process lifetime, owned workers drained and DeferredDelete processed. A QFileSystemWatcher access violation is not a passing suite; investigate/reproduce it without disabling production watchers merely to pass tests.
- Timings are local synthetic diagnostics, not instrument/network latency. Profile separately from timing.
- Preserve baseline probe JSON; use new output paths. MCD fixture must use `B increasing`/`B decreasing`, never one branch per field.
- Every new cache has bounded storage and explicit invalidation for refresh, folder/source replacement, status/theme, resize and canvas ownership as applicable.
- Tasks 1–8 cover the intended fixes. Task 9 is diagnosis/disposition, not authorization for broad speculative refactoring. User has already authorized plan → Luna execution → Astra review.

## Current source and file map

The audit's main paths remain present. `_scan_folder_sources_worker` returns a positional tuple containing tagged `power_catalog` and `catalog_scope` dictionaries; append a tagged record rather than shifting fields. `_on_file_lists_result` clears PL mtimes even for unrelated scoped results. SHG already receives worker mtimes. Compare `mode=all` publication forces synchronous history refresh. `replace_rows_if_changed` stages Qt rows before comparing. MCD already exposes `_owns_current_figure`, generation checks, render counters and full-redraw/export hooks.

| File | Responsibility |
|---|---|
| `ui_qt/main_window.py` | Catalog publication, invalidation, MCD reentry, scheduling, SHG wiring, plot/export integration |
| `ui_qt/controllers_pl.py`, `controllers_mcd.py` | Memory-only filtering/order; gate/window updates |
| `ui_qt/source_picker_dialog.py` | Optional content-key short circuit, compatible fallback |
| `ui_qt/controllers_compare.py`, `controllers_shg.py` | Snapshot consumption, SHG view-only callback |
| `ui_qt/controllers_drr.py`, `controllers_power.py`, `power_group_dialog.py` | Local drawing and diagnosed picker I/O boundaries |
| `ui_qt/mcd_unified_page.py` | Reuse and window extraction/display |
| `ui_qt/axes_region_blitter.py` (create only if needed) | Small common region-drawing lifecycle, no scientific/controller policy |
| `core/source_catalog_cache.py` (conditional) | Old snapshot schema migration |
| `tests/test_responsive_catalog_metadata.py`, `test_responsive_plot_regions.py`, `test_responsive_picker_io.py` (new) | I/O, artist, invalidation and stale-result tests |
| `artifacts/responsive-pages-and-pickers/` (new) | Baseline/final JSON, screenshots, counters, logs |
| `docs/superpowers/reports/2026-09-14-responsive-pages-and-pickers.md` (new) | Per-task evidence and residual-risk disposition |

## Task 1: Executable baseline and lifecycle-correct probes

**Files:** Both audit probes, new artifact directory and report.

**Interfaces:** Add `--output PATH` with current default retained. Picker probe supports accepted worker snapshot and deliberately missing snapshot, labeled separately. Existing direct-list-only fixture is insufficient to measure the new production path.

- [ ] Read audit/probes/baseline manifest; capture git status and Python/Qt/Matplotlib versions. Do not expose unrelated environment variables.
- [ ] Add output option without overwriting old results:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path(__file__).with_name(DEFAULT_NAME))
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
```

- [ ] Prepare picker fixture through `_scan_folder_sources_worker` off GUI thread, then accepted `_on_file_lists_result` on GUI. Start picker timing after publication. Count metadata stat/resolve/read and population calls with thread IDs, scoped to measured operations; do not prohibit unrelated Qt/Matplotlib file access.
- [ ] MCD return measurement must accept a valid no-work return: record ownership, axes/artist identities, render count, slot and next Qt event. No new draw is required when unchanged. Gate updates still require draw OR blit completion.
- [ ] Run baseline probes and `tests.test_source_picker_dialog` in fresh processes. Record exit/test counts. If access violation returns, isolate import/QApplication/close lifecycle; do not treat an aborted combined run as partial success for unexecuted tests.
- [ ] Save three fresh-process runs for baseline timing and summarize medians/p95/maximum heartbeat gap separately from profile runs.

```powershell
$env:QT_QPA_PLATFORM='offscreen'
python -X faulthandler artifacts/responsiveness_audit_probe.py --output artifacts/responsive-pages-and-pickers/pages-before.json
python -X faulthandler artifacts/file_picker_audit_probe.py --output artifacts/responsive-pages-and-pickers/pickers-before.json
python -X faulthandler -m unittest tests.test_source_picker_dialog -v
```

## Task 2: Worker PL/MCD metadata and MCD debounce

**Files:** `main_window.py` scan/cache/publish/init/reset paths; PL/MCD/Compare modified-time methods; conditional catalog schema change; new metadata tests; `tests/test_lazy_source_catalogs.py`.

**Interfaces:** Append a serializable tagged record. Dictionary keys remain original source IDs, not basenames. Values shown below describe shape; derive actual values in the worker.

```python
metadata = {
    'tag': 'source_metadata', 'version': 1, 'folder': folder,
    'modes': tuple(populated_modes),
    'pl_mtimes': pl_mtimes, 'mcd_mtimes': mcd_mtimes,
    'compare_mtimes': compare_mtimes,
}
```

`_pl_source_mtime_cache`, new `_mcd_source_mtime_cache`, and existing Compare cache contain accepted metadata. Lookups return `float(cache.get(source, 0.0))`; missing means date unavailable pending async refresh, never GUI stat. Order stays `(-mtime, source.casefold())`.

- [ ] Add failing controller tests with/without cache while resolver raises. Adapt real owner fixture:

```python
owner._mcd_source_mtime_cache = {'nested/b.csv': 20.0, 'a.csv': 10.0}
owner.mcd_available_files = ['a.csv', 'nested/b.csv']
with patch('ui_qt.controllers_mcd.resolve_source_path', side_effect=AssertionError('GUI I/O')):
    self.assertEqual(controller._mcd_sources_newest_first(), ['nested/b.csv', 'a.csv'])
    self.assertEqual(controller._mcd_source_modified('missing.csv'), 0.0)
```

- [ ] Run new module to confirm failure. Add worker tests for duplicate basenames, missing files (mtime 0), mixed scopes, and inactive-mode discovery avoidance.
- [ ] Gather metadata once per unique relevant source in worker, retaining previous path semantics. Share overlapping PL/Compare metadata. Publish after existing folder/generation guards; preserve inactive caches. Folder change clears caches, stale preview cannot overwrite newer data.
- [ ] Ensure persisted old snapshots cannot perpetually lack metadata when inventory is unchanged. Bump source payload namespace/version or rebuild missing-metadata payload in worker. Do not move cache validation to GUI.
- [x] Set MCD filter interval to 140 ms, remove obsolete synchronous-filter comment. Initial populate/status changes remain immediate. Six rapid keys yield one filter. Flush pending text filtering before acceptance or disable acceptance until current text applies; never accept a stale hidden selection.
- [x] Test manual refresh, changed mtimes, deletion, new folder and stale scan. Update tests depending on immediate MCD text filtering to wait for debounce.
- [x] Run new metadata, lazy catalog, PL source and MCD shared-source suites separately. Repeat picker probe: accepted-snapshot open/filter has zero GUI metadata stat/resolve; missing snapshot also has no synchronous fallback. Record timing.

## Task 3: Skip unchanged Qt row construction

**Files:** `source_picker_dialog.py`; PL/MCD/SHG/Compare picker rendering; DRR/Power callers only when needed; shell/stability/filename tests.

**Interfaces:** Compatible optional arguments:

```python
def repopulate(self, populate, *, fallback_selection=None, content_key=None) -> None: ...
@staticmethod
def replace_rows_if_changed(widget, populate, *, content_key=None) -> bool: ...
```

Non-None immutable key describes ordered visible rows, all text/roles/flags and appearance inputs. Store only after successful population. Unkeyed calls invalidate remembered key and retain existing fallback. Include status/metadata/theme changes; exclude selection/scroll. Raw filter text alone is not a content key: different queries may yield identical rows.

- [ ] Add failing test using existing `_item` helper:

```python
calls = []
def populate(widget):
    calls.append(1)
    widget.addItem(self._item('a.csv'))
dialog.repopulate(populate, content_key=('a.csv', 'new', '#123456'))
row = dialog.source_list.item(0)
dialog.repopulate(populate, content_key=('a.csv', 'new', '#123456'))
self.assertEqual(len(calls), 1)
self.assertIs(dialog.source_list.item(0), row)
```

- [ ] Add key invalidation tests for tooltip/status/flags/theme, population exception, keyed/unkeyed mixing, selection and scrollbar. Run red tests before implementation.
- [ ] Short circuit before staging/widget/item creation. Keep QListWidget and WrappedFilenameDelegate. Generate filtered descriptors once and reuse for key and population; sort base catalog once per accepted snapshot.
- [ ] Migrate PL/MCD then SHG/Compare. Audit role mutations outside population and invalidate keys there. Preserve changed-list restoration, initial selection, and one-result behavior.
- [ ] Repeat 1,000-row probes: unchanged result has zero population calls and exact row identity/selection/scroll, target below 25 ms. Changed filtering may perform O(N) memory work. Do not remove wrapping or impose uniform heights to improve numbers.
- [ ] Run shell/stability/filename suites separately. Save long-name first-paint/resized light/dark screenshots at 100%/150%.

## Task 4: MCD reentry reuse and window work

**Files:** `main_window.py` tab callback/unified plot/payload/slopes; `controllers_mcd.py` pending-center refresh; `mcd_unified_page.py`; MCD render/new plot tests.

**Interfaces:** Add `MainWindow._can_reuse_unified_mcd_view() -> bool`, used for reentry only, not explicit Plot. Require result identity, source/analysis/display state and `view._owns_current_figure()`. Keep last-successfully-published state signature and update after lightweight publications. Any new extraction interface must retain unaggregated fit inputs; branch-averaged display points cannot silently replace per-pair data.

- [ ] In shown MainWindow fixture, spy on render/candidate/slope preparation; leave to empty PL and return four times. Assert zero calls, same axes/heatmap. Then test `figure.clear`, new result, processing generation, parameters, feature-payload change, theme and resize. Ownership loss reconstructs exactly once; explicit Plot still refreshes.
- [ ] Run red tests and implement guarded tab callback. Do not bypass required visibility/invalidation. Prefer conservative rebuild for unrecognized state changes over incomplete cache keys.
- [ ] Profile center updates with valid branches. `_compute_unified_mcd_slopes` currently copies the whole reordered matrix while view extracts separately. Cache only energy-order indices per valid source revision and index selected columns before copying; retain current array values for supported in-place mutation.
- [ ] Add numerical comparisons for both branches, duplicate fields, NaNs, empty windows, descending storage and four metrics. Compare traces and slope outputs to existing functions using `np.testing.assert_allclose(..., equal_nan=True)` and exact branch/status checks. Never merge display/slope calculation through aggregated samples without equivalence proof.
- [ ] Reuse existing `_mcd_unified_slopes_key` / `_unified_window_analysis_key` for exact payload reuse, serialize once, and avoid feature detection/catalog recomputation for center movement.
- [ ] After local drawing improvements reprofile. If synchronous analysis alone remains above 75 ms median, move only that exact calculation into existing worker infrastructure with source generation + window key + request sequence, one running/latest pending job. Cursor moves immediately; stale results are discarded. Existing save-current must wait for or obtain exact latest results. Add out-of-order and immediate-save regressions before this conditional extension. Otherwise record that worker split was not needed.
- [ ] Run render/unified workflow/analysis/compact slope/save-current suites separately and corrected probe. Unchanged return target <50 ms slot with zero render growth; center target at least 35% below same-run baseline. Keep exact latest trace/readout/export values.

## Task 5: PL artist reuse and local drawing lifecycle

**Files:** PL controller, `main_window.py` plotting/save/theme lifecycle; optional new `axes_region_blitter.py`; new region tests; existing Compare/MCD tests.

**Interfaces:** If helper is needed, `AxesRegionBlitter` exposes `configure(axes, artists)`, `invalidate()`, `draw() -> bool`, `prepare_full_redraw()`, `restore_interactive_drawing()`, `disconnect()`. Whole spectrum axes need tick/title drawing; map groups contain dynamic overlays. One helper owns current Figure; disconnect restores original animated flags. Adapt existing MCD group blitting rather than copy controllers.

- [ ] Write canvas-pixel and artist tests for spectrum amplitude/title/gate changes. Compare local result to full draw with small antialiasing tolerance; assert static heatmap draw is avoided. Representative assertion:

```python
spectrum = spectrum_axis.lines[0]
heatmap = heat_axis.collections[0]
controller._update_pl_spectrum_and_gate_line(cube)
self.assertIs(spectrum_axis.lines[0], spectrum)
self.assertIs(heat_axis.collections[0], heatmap)
np.testing.assert_allclose(spectrum.get_ydata(), expected_spectrum, equal_nan=True)
```

- [ ] Test title/tick extent changes, autoscale, resize/theme, pan/zoom, legends, full draw/save and ownership loss. Run failing tests.
- [ ] Replace PL axis clear with set_data/title updates. Preserve nearest gate, limits, autoscale policy, peak/fit gate matching and analysis text. Only replace owned transient overlays; no orphan accumulation.
- [ ] Capture static backgrounds after complete themed layout; dirty areas cover union of old/new axes tight extents, ticks/title/legends with padding. Restore background then draw spectrum axes and dynamic map overlays. If region crosses another subplot or layout/renderer is invalid, use correct full draw and rebuild cache. Limits must never reuse invalid background.
- [ ] Draw-event handler must paint animated content before returning. Export disables interactive animation and restores in finally; audit toolbar and direct savefig paths. All curves/overlays must be present in saved plots.
- [ ] Run region, PL, theme, Compare latency and MCD render suites separately. Probe stable/changing limits. Target PL median <90 ms and ≥25% improvement; report fallback full draws honestly.

## Task 6: DRR and Power gate-only updates

**Files:** DRR/Power controllers, main scheduler, optional Power peak controller, region tests, existing dual-view and Power interaction tests.

**Interfaces:** Reuse Task 5 lifecycle. DRR gate-only scheduling is 0–16 ms; full parameter edits keep 90 ms. Existing `_is_drr_gate_only_change` and pending-full-redraw state decide eligibility. Gate movement cannot cancel a pending processing/range update.

- [ ] Add failing tests for short gate delay, range-then-gate preserving full work, and queued gate callback after leaving to Tools/Slides. Loaded mode alone does not prove visibility.
- [ ] Test DRR single/paired raw/second products: every visible linecut/gate updates from its own cube; axes/lines stay; active-product overlays and view limits survive.
- [ ] Test Power single/compare/log axis, duplicate-power row selection by index, different channel grids and changing peak overlays. Preserve `_power_selected_row_index` precedence for first cube, labels/legend and existing autoscale semantics.
- [x] Replace clears with artist updates and region drawing. Peak controller edits only as needed to prevent duplicate/lost overlays, never new fits for gate movement.
- [x] Reconcile pending full redraw before fast path; do not discard range state or advance plot keys to conceal skipped work. If hidden drawing is suppressed, mark dirty and update once on return with valid loaded data.
- [ ] Run dual-view regression/cached-preview/Power interaction/selection/regression and new region suites in isolated processes. Verify exact module paths with `rg --files tests` before commands.
- [ ] Probe DRR target <130 ms and ≥35% improvement; Power <90 ms and ≥25%. Stable-limit gates avoid full Figure draw; changing limits may redraw spectrum axes but preserve static heatmaps. Save paired DRR and Power and visually verify overlays.

## Task 7: SHG view-only angle cursor

**Files:** `main_window.py` cursor signal/plot state; `controllers_shg.py`; SHG fit-reuse/controller tests.

**Interfaces:** Add `_on_shg_angle_cursor_changed() -> None`. Reuse view-only scheduling, preferably 0–16 ms coalescing, without invalidating processing identity or requesting fit. Processing controls keep existing callbacks.

- [x] Through actual spin signal with loaded single/compare results and fitting enabled, spy on processing request/fit functions. Assert zero submissions, latest nearest-angle spectrum/marker/title and same fit result.
- [ ] Repeat with processing worker pending: cursor must not cancel it or permit old source publication; final publication uses latest cursor. Preserve processing/export freshness tests.
- [ ] Wire cursor to dedicated callback. Wrap/scale/offset/fit ranges stay processing controls. Verify plot preparation does not indirectly reprocess because cursor is in a processing identity.
- [ ] Run SHG tests separately, measure display latency and worker counts. Acceptance is zero processing/fitting for angle-only input, not a claim of complete SHG rendering optimization.

## Task 8: Diagnose and correct bounded picker I/O fallbacks

**Files:** Compare/DRR controllers, Power group dialog, worker publisher, new I/O tests and related workflows.

**Interfaces:** Fallback requests capture `(folder, catalog_generation, selection_key, request_sequence)`. Only matching results update a live dialog. Workers read files; GUI consumes plain snapshots. Pending/unknown metadata must not become falsely compatible or validated.

- [ ] Reproduce Compare cold history/mode=all, DRR selection outside current partition and Power manifest/signature reads with temporary files. Inject 20 ms per relevant read and record thread IDs/heartbeat gaps. Avoid real network dependencies.
- [ ] Add thread-guard tests for actual reproduced I/O:

```python
gui_thread = threading.get_ident()
def guarded_read(*args, **kwargs):
    self.assertNotEqual(threading.get_ident(), gui_thread)
    return original_read(*args, **kwargs)
```

Process Qt events with finite timeout for workers; also change folder/selection and close dialog before completion.

- [x] Compare: load history for `all` inside scan worker as for `Compare`. Publish history before badge/group refresh. Cold miss queues existing catalog refresh and shows pending/unknown while retaining known rows; no GUI rglob/JSON. Cache grouping/status descriptors by catalog/history/grouping parameters to avoid repeated matching for counts/render. Invalidate on saved-history change.
- [x] DRR: replace missing-selected-header fallback in `_baseline_measurements` with owned worker for exact missing selection, or explicitly include selected records in catalog worker. Until complete no unsafe recommendation; preserve requirement that ALL selected measurement metadata is known. If confirmation is_file is measurably blocking, validate exact selection asynchronously before accepting; final load handles file races and never silently accepts a partial set.
- [x] Power: read saved assignments/warnings in catalog worker rather than `refresh`. Memory-only prefetch identity uses accepted catalog signatures, members and pairing settings. Validation worker verifies actual file existence/size/mtime, including final acceptance freshness; snapshot signatures cannot certify changed files. Return actual signatures, reject stale generation/request/group, preserve accept-after-validation exact-selection guards. Request identity and validated file identity are separate.
- [x] Do not invent synchronous I/O for detached test dialogs: give fixture an owned pool or assert pending/error behavior. Never substitute unconditional validation success.
- [x] Run Compare history/status/workflow, DRR baseline/missing/startup, Power group/prevalidated/header-cache tests separately. Repeat injected-delay cases: zero demonstrated GUI reads, responsive heartbeat, eventual correct statuses, disappeared file blocks acceptance, late result cannot modify closed/repurposed dialog.
- [ ] Unreproduced cases receive exact diagnostic conditions and `not reproduced` status. Do not undertake a large history-index redesign merely from static risk.

## Task 9: Secondary-risk diagnosis and disposition

**Files:** Report by default; conditional focused changes only.

- [ ] Measure theme and dense-layout cost after gate fixes. Only if ≥15% of interaction CPU or ≥20 ms/event, add focused reproducer before caching. Theme invalidation covers new artists/theme/DPI/export; layout covers font/style/text/visibility/width. Otherwise leave source unchanged and report.
- [ ] Reproduce queued redraw after leaving; if confirmed, apply Task 6 visibility/dirty scheduling and deferred-reentry tests, including MCD Peak Shift ownership. Preserve background computation.
- [ ] Measure MCD Peak Shift with complete fixture. Add reuse only with provable result/display invalidation; otherwise explicitly retain finding/fixture limitation.
- [ ] Temporary Slides 1,000-image directory: measure identical filters and thumbnail memory growth. Only confirmed issue gets row-key optimization or bounded thumbnail LRU (e.g. 128 decoded items), preserving visible items/generations and user files.
- [ ] Simulate slow owned worker with externally bounded process and inspect close behavior. Do not simply remove waitForDone/kill workers. An ownership-safe shutdown redesign outside scope is explicitly deferred.
- [ ] Each actual secondary fix requires reproducer, before/after evidence and invalidation tests. No secondary source change is required just to claim full plan execution; honest disposition is required.

## Task 10: Final validation and Astra handoff

**Files:** Artifacts/report and plan progress; no unrelated cleanup.

- [x] Run all new/affected tests in fresh processes with `python -X faulthandler -m unittest MODULE -v`, collecting actual counts/exit codes. Check names via rg; run scientific tests for touched numerical helpers. Investigate crashes/timeouts; do not count as pass.
- [x] Repeat both probes three times with new final output names, production MCD labels, draw+blit detection, and a documented 1200×800→1400×900 resize probe. Picker: 100/1,000 files, accepted/missing snapshot, six-key and unchanged result. Profile separately.
- [x] Save screenshots at 1200×800 and 150% scale for all gate modes, paired DRR/Power, MCD return/window and long-name dialogs. Compare local canvas with full render. Export screenshots contain curves/fit annotations/legends.
- [x] Report each acceptance row with median/p95, structural counters, test counts/exit codes, screenshot path and `fixed`, `improved but target missed`, `not reproduced` or `deferred`.

| Case | Required structural outcome | Same-machine target |
|---|---|---|
| PL/MCD open/filter | Zero GUI metadata reads; missing cache no sync fallback | Six-key synchronous total <25 ms |
| Same-result filter | Zero population; exact row/selection/scroll retained | <25 ms at 1,000 rows |
| MCD unchanged return | Zero render/candidate/slope work; valid axes | Slot <50 ms |
| MCD center | No heatmap rebuild; exact trace/slopes/export | ≥35% faster |
| PL/Power gate | Reused artists, local draw, correct overlays | <90 ms and ≥25% faster |
| DRR gate | No 90 ms gate wait; paired products correct | <130 ms and ≥35% faster |
| SHG cursor | Zero new processing/fit, latest spectrum | Record median/p95 |
| Cold picker fallback | No demonstrated GUI read; safe pending/validation | Record heartbeat under injected delay |

- [ ] Performance misses get profile evidence and further bounded fixes where justified. Never remove autoscale/overlays, alter science, manipulate fixture or conceal fallbacks to hit targets.
- [ ] Compare this round's changed files to baseline ZIP, inspect accidental user-change loss, run syntax/whitespace checks. Do not commit dirty worktree. Update checkboxes and deviations with reasons.
- [ ] Final report maps F1/F2/F3, page findings 1–5 and every secondary risk to tests/results. Include residual first-layout cost, full-draw fallbacks, unmeasured network behavior and remaining test crashes.
- [ ] Notify coordinating Astra only after all implementation/validation is complete; provide plan/report/artifacts/file list. Independent Astra reviews against ZIP for scientific equivalence, cache invalidation, worker freshness, UI/export correctness and evidence. Address review findings and rerun relevant tests before user-facing completion.

## Plan self-review

Confirmed F1–F3 map to Tasks 2–3; page findings 1–5 to Tasks 4–7; conditional I/O to Task 8; all secondary risks to Task 9. Cache migration, stale generations, changes between validation/load, shared Figure ownership, animation/export and QApplication lifetime are explicit requirements. Existing numerical functions remain authoritative. One total plan permits independent task tests without prematurely claiming static risks solved.
