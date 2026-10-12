# DRR derivative cache and cooperative Load cancellation

## Scope

Implement the two remaining agreed optimizations: bound retained derivative
history by bytes and stop obsolete loads at cooperative boundaries. Persistent
indexing of legacy display-history records remains separate work. Existing
numerical formulas, settings precedence, manual Fix and export behavior are
preserved.

## Implementation

- `DrrDerivativeCache` retains the current raw, dE and second/mixed derivative
  products, with LRU eviction of older parameter products. It has a default
  128 MiB buffer budget and 12-entry cap. NumPy views count their owning backing
  arrays; raw source storage already owned by `LoadedState` is not counted
  twice. Array accounting is conservative per product, not process RSS.
- Current derivative products remain pinned so asynchronous first/second
  calculation can converge even when one full-resolution pair exceeds the
  budget. In that case all history is evicted. The budget therefore limits
  derivative history retention, with an explicit exception for required current
  products; it is not a hard limit on application RAM.
- A `LoadCancellation` event is scoped to each load worker using a `ContextVar`.
  Checks run before/after canonical reads, between measurements, around
  interpolation/averaging/derivative stages and before display-history and
  provenance work. History scanning also checks between legacy records.
  Context is reset on all exits, including cancellation.
- Cancelled workers finish silently and never publish their obsolete result.
  GUI result/error handlers also reject late signals after cancellation. The
  finish handler clears the busy state without marking cancellation as failure
  and starts the latest captured request only.
- Normal and queued loads share one launch path. The existing request tracker
  can promote a pending token on finish; that promoted request now actually
  starts its worker. During an active load, option collection first drops any
  earlier queued request and publishes a matching mode/options snapshot only
  after successful validation.
- Source Clear and folder reset invalidate/cancel active loads and remove
  corresponding queued requests. A deferred launch remains protected by its
  lifecycle generation and folder. Export workers receive no Load cancellation
  event and retain their frozen data/settings.
- Accepting a newer captured request also clears earlier MCD reload, reapply
  and angle-load continuations and its pending apply timer. Those legacy updates
  otherwise could overtake the newer request or overwrite its result. Later
  MCD edits can still schedule fresh updates after the new request is accepted.

## Verification

New tests first reproduced missing byte-bounded retention and cancellation,
the promoted-token launch gap, late cancellation errors, continued history
scanning, DRR Clear restoring an old result and failed option collection keeping
an older queued payload. A folder reset test also reproduced a missing
cancellation event despite existing result invalidation.
An additional cross-workflow regression reproduced both old MCD continuation
failures: the earlier reload replaced the latest queued DRR load, and earlier
reapply could reload MCD after the DRR result had already appeared.

The 20 focused tests pass. They include real threaded CSV reads held while the
GUI queues a newer source, clears sources or resets folder state; current
derivative pairs larger than the cache budget; release of old buffers; cached
parameter promotion; cancellation scope isolation; and manual Y Fix preservation
after the newest request loads and draws.

```powershell
$env:QT_API = 'pyside6'
$env:QT_QPA_PLATFORM = 'offscreen'
.venv\Scripts\python.exe -m unittest tests.test_drr_derivative_cache tests.test_load_cancellation tests.test_drr_display_compute -v
```

Before the final Clear/capture/folder integration repairs, 174 selected
regressions also passed, covering settings memory, deferred export, display
computation, split views, background workflows, streaming loads, history caches
and asynchronous figure saving. GUI regression harnesses use temporary INI
settings to keep the user's registry preferences unchanged.

Independent read-only review found and verified the Clear, failed-capture and
MCD-continuation integration gaps above. After their repairs the reviewer
reported no remaining P1/P2 findings and verified 43 tests across cache,
cancellation, display workers, workflow requests, worker lifecycle and MCD
shared-source/summary suites.

After the final MCD repair, 46 GUI integration regressions passed, including
normal MCD source loading, automatic recalculation/center refresh, MCD tab
switching, input commit, workflow request handling and all cancellation tests.
The broader intermediate run had 178 passes and one harness error: the preview
builder subclasses `QSettings`, but the outer registry-isolation patch supplied
a Mock instead of a class. Changing only that harness to an INI-backed
`QSettings` subclass resolved it; all seven layout tests pass in the final run.
No application fix was needed for that harness failure.

The isolated harness is `artifacts/drr-cache-cancellation/verify.py`; it accepts
test names as arguments. `compileall` and `git diff --check` also pass. Git emits
line-ending conversion warnings for existing Windows working files.

## Synthetic retained-buffer measurement

`artifacts/drr-cache-cancellation/benchmark.py` generates 12 derivative parameter
products, each containing a 1024 × 4096 float64 matrix and its axes. Retained
buffer storage changed from 384.47 MiB across 12 products with the old item-only
policy to 96.12 MiB across three products with the new default budget. Current
dE and second products remain usable in both policies.

This measures retained NumPy buffers only; it excludes raw source, plot meshes,
workers and all other application allocations. It is not a timing claim or a
measurement of total process RAM.

```powershell
.venv\Scripts\python.exe artifacts\drr-cache-cancellation\benchmark.py
```

Per-run data are in `artifacts/drr-cache-cancellation/result.json`.

## Boundaries

Cancellation is cooperative. A single file read, a NumPy call or one legacy
record already in progress finishes before the next check. No thread is forcibly
terminated. Granularity of checkpoints outside the DRR numerical path is at
existing load phases. No executable has been repackaged or application restarted.
