# DRR paired PNG render optimization

## Approved scope

Implement the audited title-measurement optimization, visible-cell export
clipping and PNG stage diagnostics for Raw/second-product paired exports.
Keep complete titles, the fixed 1200 x 930 canvas, original measured cell
boundaries, color normalization, full DAT data and saved-image reuse.

## Changes

- Title fitting measures whole words first. Only an overwide identifier uses
  the original character-wrapping fallback; complete math tokens remain intact.
  Measurements are cached within each font-size attempt, up to 4096 entries,
  then discarded. The font search order, obstacle checks and final text extent
  checks are preserved; there is no persistent figure/title cache.
- `visible_mesh_data` slices original edge arrays and original scalar cells
  intersecting the requested view, retaining an adjacent guard cell on each
  side. Two/three-region meshes keep the original seams and independent norms.
  An empty view keeps an empty scalar mesh and its colorbar normalization.
- Clipping is enabled only in DRR PNG export. Interactive heatmaps retain the
  complete mesh; source arrays, DAT tables and scale calculations are unchanged.
  No sampling or reduction of cells within the view is performed.
- An axis keeps its full extent when limits are nonfinite, singular/nearly
  singular, or invalid for a log axis. Matplotlib can expand or ignore those
  bounds after mesh creation, so clipping them early could otherwise leave
  blank areas. Near-singular detection uses Matplotlib's `nonsingular` helper.
- Original cube dimensions are validated before figure creation, and scalar
  dimensions are checked against original edges before clipping. Cropping
  cannot conceal a transposed/malformed source and produce a PNG inconsistent
  with its DAT. Valid cubes retain the existing orientation.
- Each newly rendered product reports `build`, `title`, `draw`, `encode_write`
  and `total` elapsed seconds, plus live phase labels. Title time is included
  in build. Measurements cover the worker PNG operation, not queue delay or
  GUI signal handling; draw includes any Agg render-lock wait.
- Timing uses a thread-scoped `ContextVar`, restored on every exit. The export
  canvas keeps Agg's one-draw/PIL-encode sequence and forwards format, metadata,
  DPI and Pillow options. There is no global canvas monkeypatch. Existing
  atomic promotion, compression level and PNG fingerprints are retained.

## Verification

New tests first reproduced excess character-prefix measurements, full-grid
construction for a narrow view and missing per-product timing diagnostics.
Follow-up pixel regressions reproduced blanking from equal and `nextafter`
limits, invalid log limits, and successful rendering of a transposed cube that
should have been rejected. These regressions pass after the corresponding
repairs.

The final combined run passed **76 tests** covering title/reference pixels,
original edge/value slices, full interactive meshes, nonuniform coordinates,
NaN/peaks, clipping, zero-centered and log color norms, log/reversed/empty views,
singular ranges, matrix validation, timing failures/concurrent scopes, full DAT
round trips, fixed-canvas one-draw saves, paired GUI workflows, metadata/history
reuse, deferred export, three regions and measured right axes.

```powershell
$env:QT_API = 'pyside6'
$env:QT_QPA_PLATFORM = 'offscreen'
.venv\Scripts\python.exe artifacts\drr-cache-cancellation\verify.py tests.test_drr_png_optimizations tests.test_drr_png_render_latency tests.test_drr_export_title tests.test_split_grid_render tests.test_dat_png_save_optimizations tests.test_drr_dual_export tests.test_drr_dual_save_workflow tests.test_export_metadata tests.test_drr_export_history_cache tests.test_drr_deferred_export tests.test_drr_three_regions tests.test_drr_right_axis
```

The GUI harness uses temporary INI settings. Matplotlib emitted its expected
warnings when existing tests supplied identical axis limits; those ranges now
retain their original rendered pixels. `compileall` of `core`, `ui_qt` and
`tests`, and `git diff --check`, completed with exit code 0. Git emitted only
line-ending conversion warnings.

An independent read-only reviewer verified both guard repairs, 8 focused tests
and 40 related tests, with no remaining P1/P2 or minor findings. Additional
probes verified descending/singleton axes, thread isolation, nested scope
restoration, encoder-failure cleanup, atomic destination preservation, image
reuse and Agg raster-format compatibility. Two final timing-scope regressions
are also included in the 76-test run.

## Measurements

The ignored local benchmark compares exact pre-turn source snapshots to the
implementation, in one process, with three trials per case and temporary PNG
outputs. Every before/after decoded PNG comparison was exactly equal.

| Single PNG case | Before median | After median |
| --- | ---: | ---: |
| 157 x 1340, 200-character title, three regions and partial view | 5.495 s | 0.704 s |
| 1024 x 4096, three regions, half X and half Y span visible | 1.895 s | 0.704 s |
| 295 x 1340, short title and full view | 0.346 s | 0.333 s |

These are synthetic PNG-operation timings, excluding derivatives, DAT writing,
source/history checks and GUI work. The long-title and narrow-view cases
benefit most; the full-view case shows little improvement. The final guards
were verified separately after the measurement run and do not alter these
cases' valid shapes or ordinary limits.

The application must be restarted to load the updated source modules. No
application restart, executable repackaging, commit, installation or real
saved-output modification was performed in this turn.
