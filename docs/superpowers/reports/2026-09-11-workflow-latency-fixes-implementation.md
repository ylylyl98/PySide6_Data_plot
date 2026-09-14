# Workflow latency fixes — implementation evidence

Date: 2026-09-11. Astra planned and reviewed; Luna High implemented. Final approval is tracked in `2026-09-11-workflow-latency-fixes-review.md`.

## Changes

- Slides inserts selected/all images in a batch, deduplicates membership and computes one final slide plan.
- SHG fit-only requests reuse validated processed sweeps; source, background, settings and revision changes invalidate reuse. Current and queued request provenance are checked.
- Peak Shift default Raw spectrum and Second derivative calculations run in an owned worker, preserve both methods/channels, cache numerical results, cancel obsolete work and keep only the latest pending request.
- Compare gate changes reuse linecut/marker artists. Compatible Qt canvases restore a clean static background and redraw dynamic artists, including linecut labels/ticks. Full redraw/resize recaptures backgrounds; toolbar PNG saves temporarily restore normal artist rendering and then restore interactive state.
- MCD range/color edits use the existing redraw scheduler; center/width and trace-only paths retain their faster handling.
- Power discovery runs in the folder worker and publishes a tagged source snapshot; selected-group and newly combined-file handoffs wait for an appropriate snapshot and retain validated data.

## Measured performance

Synthetic data, local temporary settings/files and an offscreen Qt window. These are not universal timings for the running instrument app.

| Operation | Before | After |
| --- | ---: | ---: |
| Slides add 30 entries | 1.215 s | 0.0357 s |
| Slides add 200 entries | no valid baseline | 0.2525 s |
| Compare gate, 301 x 2048, 8 samples median | 582.36 ms after artist reuse but before blit | 32.43 ms warm blit |
| Compare first cache establishment, same size | — | 551.96 ms single observation |

Compare eight-input burst produced eight blits in 287.94 ms; it is not a coalesced single update. Initial loading/source matching is excluded from these benchmarks. Ordinary resize/full redraw still needs a full frame. Slides benchmark suppresses image preview decoding and uses an empty discovery index.

SHG independent numerical checks compare full and reused processing+fit for single and comparison inputs at 1e-12 tolerance. Peak Shift retains the same numerical algorithms; moving them off the GUI thread does not make the underlying 5-second calculation instantaneous. Cache hits avoid recalculation.

## Final verification

- Root independently ran final `tests.test_astra_workflow_review` + `tests.test_workflow_latency`: **28 passed** (3.612 s), after the last Power intent fix and temporary-settings fixture migration.
- Root independently ran `tests.test_mcd_peak_shift` + `tests.test_power_combine`: **57 passed** (34.730 s); the later narrow Power supersession changes were covered by the final reviewer regressions.
- Root compiled all seven changed production modules and final workflow test fixture successfully.
- Astra independently ran the final reviewer module: **23 passed** (2.603 s), and approved all six production fixes.
- Final workflow tests use temporary INI settings, disable folder restore/update startup and use monotonic bounded waits.

## Earlier integration verification

- Root combined review/Slides/SHG/Compare-MCD check: 32 passed before the final source-retaining background-cache regression was added.
- Root real owned-worker/workflow check: 9 passed before final additional controller race regressions.
- Astra independent actual Qt checks cover Peak Shift latest-pending and close cancellation, Compare toolbar PNG/error restoration, heatmap pixels, SHG controller/cache provenance and numerical equality.
- Final test counts and any remaining findings are recorded in the companion review report; intermediate counts above must not be summed as unique tests.
- `git diff --check` passed with line-ending warnings only. Existing unrelated working-tree changes remain intact; no commit was created.

Reproducible artifacts: `artifacts/workflow-fix-baseline`, `artifacts/workflow-fixes-slides-timing.json`, `artifacts/workflow-fixes-compare-timing.json`, `artifacts/workflow-fixes-compare-blit-timing.json` and their benchmark scripts. No real experimental output was created by the performance harnesses.
