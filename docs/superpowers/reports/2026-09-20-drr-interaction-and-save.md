# Picker optimization and PNG/DAT responsiveness audit

## Implemented picker changes

Group row prototypes are reused for search/filter changes and invalidated on
catalog publication. Their keys also include palette/font. Chosen paths use a
set kept synchronized on remove/clear. Batch additions suppress intermediate
repaints and refresh once. Removed acquisition summaries which were immediately
overwritten by the chosen pane's status/filename display; catalog source lookup
is now indexed. Existing file/grid checks at acceptance remain intact.

41 picker tests passed. Independent review passed 13 stability tests plus manual
Qt remove/re-add and clear/re-add checks. Re-generated 150% narrow-window PNG
was visually inspected: complete first-row filenames remain visible through
`.csv`; 32 paint extent checks found no insufficient row heights.

## Save and image interaction findings (diagnosis, not implemented changes)

- Main DRR pair export already uses a worker. An offscreen Qt test with a
  synthetic 157x1340 matrix completed successfully in 0.858 s; `_start_export`
  returned in 0.0039 s. A 10 ms GUI timer fired 39 times, with a maximum gap
  of 0.221 s. This demonstrates residual scheduling latency, not its cause.
  The successful harness pumped events and slept 5 ms to release the Python
  GIL. An earlier QTest.qWait harness timed out without completion and is not
  used as evidence of production export duration.
- The separate real-data temporary-output benchmark took 3.092 s with cold
  copied history: history lookup 1.714 s, both PNGs 0.880 s, both DATs 0.276 s,
  derivative 0.164 s. It is a different fixture from the GUI test. Real data
  and output directories were not modified.
- `_PlotToolbar.save_figure` calls the Matplotlib toolbar implementation
  synchronously. Analysis quick-export and range-amplitude PNG paths likewise
  call their writers synchronously. These paths can block GUI events during
  rendering, unlike the main DRR pair export.
- Display redraw calls `_drr_cube_with_metadata(2)` synchronously on a cache
  miss. Parameter bursts are already coalesced by a 90 ms timer, but expensive
  work can still happen when the timer fires.

## Recommended implementation sequence

1. Freeze data, labels, limits and style on the GUI thread, then render an
   independent noninteractive figure for toolbar/analysis PNG saves. Never
   render or mutate the live Qt figure from a worker. Preserve exact visible
   annotations, dimensions and DPI with image/metadata regression checks.
2. Move display derivative cache misses to workers with generation checks;
   retain the prior image until the latest requested result is ready. Discard
   stale results after a file or SG parameter change.
3. Instrument GUI heartbeat alongside render stages to locate the observed
   0.22 s gap. If worker rendering measurably competes with UI execution, test
   a persistent rendering process. Include Windows process startup and array
   transfer costs before selecting this larger change.
4. Keep current DAT numerical precision and atomic writes. DAT was not the
   dominant stage in the measured fixture; prioritize the synchronous image
   paths and cold history rather than changing file formats or precision.
