# DRR picker reopening optimization

## Changes

- Reuse grouping, summaries, and detached row prototypes across dialog opens. Retain at most two presentation entries, keyed by source catalog, folder, picker type, baseline selection, palette, and font.
- Reuse successful directory validation for 30 seconds. Watcher notifications invalidate it immediately; scan generation tokens prevent a scan overlapping a change from marking stale data fresh.
- Delay initial automatic validation by 150 ms. While the dialog remains open, check freshness each second. Manual Refresh bypasses freshness. Catalog-only validation avoids preparing strict background history and does not change committed selection or trigger Load.
- Delete closed dialogs after disconnecting signals and saving row heights. JSON row-height persistence and wrapped filename rendering remain enabled.

## Measurements

Real catalog: 648 sources. Offscreen Qt benchmark compares rebuilding presentation data with reusing it in the same implementation; this is not a before/after executable comparison or a complete DRR workflow timing.

- Rebuild first-paint times: 542, 243, 267, 394 ms.
- Reuse first-paint times: 172, 142, 150, 188 ms.
- Excluding the initial UI warm-up, rebuild median was 267 ms versus reuse median 161 ms, approximately 40% lower.
- Recent-valid catalog requested no automatic refresh during the benchmark.
- Results: `artifacts/drr-picker-reopen-timing.json`.

## Verification

82 targeted unittest cases passed, covering freshness, watcher invalidation, refresh priority, cancellation, selection preservation, presentation reuse, dialog lifetime, cached preview, startup, metadata, history, and row sizing. Python compilation and scoped diff whitespace checks passed.

Layout checks used 720 synthetic sources across 240 groups, widths 920/1120/1300, and scale factors 1.0/1.5. Both runs reported zero clipped rows. The narrow 150% screenshot was inspected visually; long filenames wrap and row heights accommodate their text.

An initially intermittent test used a fixed 200 ms sleep for a 150 ms callback. It now waits for worker dispatch with a two-second bound, allowing unrelated queued Qt work without weakening the dispatch assertion.

Watcher-covered changes invalidate immediately; missed or uncovered changes rely on the 30-second freshness fallback or manual Refresh. First-time catalog discovery and application cold startup can still cost more than warm reopening. Test runs use a temporary LOCALAPPDATA to avoid mutating the user's persistent inspection cache.
