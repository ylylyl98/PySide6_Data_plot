# Workflow latency fixes — Astra review

Status: **Approved for the six planned fixes and the authorized Compare blit extension.** No remaining blocking finding was identified in the final reviewed tree. Astra reviewed against the saved task baseline, added independent regressions, and did not edit production files. No commits or resets were performed.

## Scope and findings

| Fix | Reviewed behavior and resolved findings |
| --- | --- |
| Slides batch insertion | One final plan/label refresh for ordered bulk additions; duplicate and remove/re-add behavior preserved. Corrected Qt signal int/bool arguments being mistaken for cached plans. |
| SHG fit reuse | Valid processed single/compare results bypass preprocessing for fit-only changes. Corrected worker payload placement, invalid cache reseeding after background/data revision changes, and stale pending-pair reuse. |
| Peak Shift worker/cache | Both methods and both channels remain available. Owned workers capture settings, cancel between analyses, reject stale generation/source/tracker results, and retain only the latest pending request. Corrected Raw/Local transitions, cached-result button state and triple-request ownership. Close cancels before waiting. |
| Compare gate update/blitting | Reuses image/linecut artists and automatic-background results with source identity/revision guards. Preserves intensity padding, VP limits, X range and current export gate. Corrected clipping, stale rendered-state acceptance, blit trails/blank heatmaps, and save restoration. |
| MCD redraw coalescing | Range/color redraws use the existing scheduled route; trace fast paths, 40 ms center updates and 650 ms processing debounce remain. |
| Power catalog snapshot | Discovery, column reads and metadata checks run in the scan worker; GUI lookup uses the accepted folder/options snapshot. Corrected async refresh, legacy picker metadata, combined-open generation, missing selected keys, prevalidated handoff and superseded selection replay. Symmetric intent guards ensure the latest primary selection or Save+Open supersedes older deferred actions. |

## Verification evidence

- Independent final reviewer module: **23 tests passed in 2.603 s**, including all five Power handoff/latest-intent regressions. Includes real-worker event-loop/latest-request/close tests, controller provenance checks, and actual Qt toolbar save/error handling.
- SHG synthetic full processing versus fit reuse matches every compared dataclass array/scalar at rtol/atol 1e-12 for single and compare fits, with changed fit range, weights and phase branch; input spectra remain unchanged.
- Compare heatmap interiors match clean full renders exactly after warm blit and ordinary draw. Saved PNG pixels match full rendering exactly. On-screen pixel comparison excludes a 3-pixel spine border because dynamic overlay/spine antialiasing order differs; no interior trail tolerance is used. Toolbar save success and failure restore dynamic display.
- Independent final Compare/MCD plus reviewer Compare run: 13 tests passed in 1.844 s. Independent Slides run: 4 passed.
- Coordinator independently confirmed the migrated existing Peak Shift and Power-combine modules: 57 tests passed in 34.73 s. Peak Shift tests retain numerical/selection assertions, use temporary INI settings and bounded monotonic waits.
- The reverse Save+Open regression failed before Luna added symmetric intent clearing, then passed. The two new workflow dispatch tests now use temporary INI settings for both window and presentation widgets, suppress folder restoration/update checks, and pump events to monotonic deadlines. Coordinator final combined reviewer/workflow run: **28 passed in 3.612 s**; seven production modules and the adjusted fixture compiled successfully. Astra read the final fixture changes.

Tests use synthetic inputs and bounded subprocesses. These checks cover focused numerical, controller, rendering and lifecycle behavior; they are not a complete manual session with experimental datasets.

## Performance bounds

Coordinator measurements reviewed from `artifacts/workflow-fixes-slides-timing.json` and `artifacts/workflow-fixes-compare-blit-timing.json`:

- Slides: 30 bulk additions 0.0357 s; 200 additions 0.2525 s. Earlier 30 per-insert additions took 1.215 s. Thumbnail preview was suppressed and settings isolated.
- Compare synthetic 301 × 2048: cold observation 551.96 ms; eight warm updates p50 32.43 ms and p95 39.11 ms. Prior full-canvas update p50 was 582.36 ms. An eight-event burst took 287.94 ms and performed eight blits. The endpoint is actual offscreen Qt `canvas.blit` completion; file loading and source-consistency checks are excluded.
- Peak Shift and SHG core numerical cost is preserved. The fixes move Peak Shift analysis off the GUI thread and skip SHG preprocessing for valid fit-only requests; they do not claim faster numerical algorithms.

Cold Compare drawing remains substantial; warm burst inputs are not coalesced. Slides still has final-refresh path/status lookup work. Organizer cold preview and DRR smoothing remain explicitly deferred: existing reuse limits repeated Organizer cost, and the measured 301 × 2048 DRR SG miss was about 0.0145 s. No export rewrite or universal live-app latency guarantee is included.
