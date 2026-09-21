# DRR background PNG and display derivatives

Implemented the accepted follow-up to the interaction/save audit.

## PNG saving

DRR toolbar PNG, batch P2P quick-export PNG, and single range-amplitude PNG
now freeze a figure and render an independent Agg figure on a dedicated,
serial background pool. The live Qt figure is never rendered or mutated by
that worker. Locally generated pickle bytes transfer the Matplotlib artists;
no user-provided pickle is loaded. Snapshot errors are reported. Output is
written to a sibling temporary file and atomically replaced on success.
Existing destinations survive rendering failure. P2P default version naming
and chosen output size/DPI remain intact; success is announced after completion.

The DRR heatmap cursor callback is now a module function instead of a local
lambda, allowing real heatmap snapshots to serialize. Qt font preparation and
toolbar publication-theme handling are preserved. Non-DRR toolbar workflows
retain their original implementation. CSV/XLSX paths were not changed.

## Display computation

Cache misses for displayed derivatives execute in a worker. The previous image
remains visible. One transform is in flight; changing source/SG settings makes
obsolete results ineligible. dE and d2E products sharing a source and SG settings
can both enter their separately keyed cache, avoiding starvation when raw range
controls and the display need different derivatives. Cached reads do not cancel
pending computations. Stale worker failures allow newer requests to proceed.

Deferred automatic ranges, split centering, explicit split-side Auto commands,
and Auto d2E requests replay after completion and are guarded by source identity.
Current-operation failures clear deferred intents to prevent automatic retries
from looping. The scientific derivative implementation is unchanged.

## Timing and validation

The main DRR pair-save log now reports GUI preparation time and maximum sampled
GUI interval, in addition to its existing worker-stage timings. Toolbar PNG logs
snapshot, total time and maximum GUI interval.

One offscreen test of the actual DRR toolbar with a synthetic 157x1340 heatmap:

- Save method returned in 0.087 s (file dialog mocked).
- Figure snapshot took 0.019 s; job completed in 0.377 s.
- Maximum sampled GUI interval was 0.188 s.

Background rendering therefore does not establish zero stutter. Python/renderer
contention and queued GUI draws remain candidates for a future isolated-process
experiment; this change uses a thread pool, not a separate process. No reduced
resolution, downsampling of saved data, or DAT precision changes were introduced.

43 relevant tests passed in two runs: 20 display/interaction/MCD toolbar tests,
and 23 PNG/save/range tests. Real DRR heatmap toolbar PNG and a standalone figure
were pixel-identical to synchronous references even after the live figure was
cleared. Tests also cover atomic failure, changed source/SG, stale failure,
simultaneous dE/d2E demands, auto-range replay, file versioning and PNG dimensions.

An old regression asserted that raw split mode never affected the d2E split
boundary. It also failed with async preparation bypassed (the pre-change
synchronous behavior); the test now asserts the current shared-boundary,
independent-color-limits contract. Other interaction tests were updated to wait
for asynchronous completion while retaining their numerical/viewport assertions.
