# DRR display cancellation, derivative buffers and incremental logs

## Scope

Implement the approved remaining display-derivative and log optimizations, and
add Load worker stage diagnostics. Persistent indexing of legacy display-history
records remains separate work. Earlier settings memory, manual Fix precedence,
cache budgets and Load cancellation changes are preserved.

## Implementation

- Each asynchronous display transform has a cooperative cancellation event.
  A newer source, X SG window/polyorder or second-product kind/Y window cancels
  obsolete work. Clear, folder reset, accepting a new loaded cube and closing
  the window also cancel it. Late results and cancellation errors are ignored.
- Uneven-grid energy derivatives check between rows; mixed derivatives check
  between polynomial windows, incomplete-row fits and derivative passes. A
  uniform-grid SciPy SG call must finish its current call before cancellation
  is observed. Cancellation therefore reduces obsolete work at available
  boundaries; it does not interrupt an arbitrary numerical kernel.
- Same-source, same-X-SG dE and second/mixed products remain useful together.
  Requesting either one does not continually cancel the other. Changing only
  second-product settings preserves an active dE computation.
- Parameter invalidation runs before validation and after blocked-signal Auto
  recommendations/clamps. An active hidden mixed product also resolves its
  effective Auto X window before invalidation. This avoids retaining obsolete
  work when a hidden panel does not request a replacement or new settings fail
  validation; it does not compute a hidden replacement derivative.
- Energy-derivative output adopts the independently allocated filter buffer,
  removing a redundant full-matrix copy. Energy/gate coordinates retain their
  copies. Numerical formulas, axis ordering and NaN behavior are unchanged.
- `BufferedLog` queues messages and appends plain text in 30 ms batches. It
  retains the latest 300 messages, including multiline messages and literal
  markup, counts Qt UTF-16 positions correctly, and disables undo retention.
  Readers at the bottom follow new messages; readers browsing older messages
  keep their text position even when old entries expire. Clear also removes
  queued messages, so an old timer cannot restore cleared text.
- `LoadedState.load_timings` stores thread-scoped inclusive elapsed times for
  `worker_total`, `data_load`, `provenance`, `csv_read`, `display_memory` and
  `coordinates` where those stages occur. One summary is written to the log.
  Nested stages overlap and must not be added together. CSV timing includes
  cache lookup/read/parse work at the canonical-read call, and worker timing
  excludes thread-pool waiting, signal delivery, GUI restoration and plotting.
  Outside a timing collector, the timing helper performs no clock calls.

## Verification

New regressions first reproduced continued obsolete row/window calculations,
the extra derivative copy, whole-document log replacement, missed cancellation
on folder reset, hidden/invalid parameter changes and hidden mixed Auto X, and
a viewport jump when the oldest log entry expired. Each repair passed the
corresponding regression using real numerical filters or real Qt widgets.

All 22 focused tests passed after the final hidden mixed Auto X repair. An
independent read-only reviewer reran those 22 tests and reported no remaining
P1/P2 findings in scope. The reviewer also checked that mixed Y edits cancel
an active mixed product while preserving a useful same-SG dE computation.

Earlier in this turn, 32 processing/cache/background/cancellation/timing tests
and 147 selected GUI regressions passed. Following the main integration
repairs, 135 combined tests passed across derivative cancellation, logging,
timing, display computation, mixed math, settings memory, deferred export,
input commits and dual-view regressions. That 135-test run preceded the final
small hidden mixed Auto X repair; the fresh focused run includes that repair.
GUI verification uses temporary INI settings and does not overwrite the user's
registry preferences.

The final affected regression run passed **49 tests**, including the final
repair, mixed derivatives and deferred exports:

```powershell
$env:QT_API = 'pyside6'
$env:QT_QPA_PLATFORM = 'offscreen'
.venv\Scripts\python.exe artifacts\drr-cache-cancellation\verify.py tests.test_drr_display_compute tests.test_drr_mixed_derivative tests.test_drr_deferred_export tests.test_derivative_cancellation tests.test_buffered_log tests.test_load_timing
```

Final `compileall` of `core`, `ui_qt` and `tests`, and `git diff --check`, both
completed with exit code 0. Git emitted only line-ending conversion warnings.

## Measurements

The ignored benchmark script compares exact pre-turn source snapshots with
the updated implementation. It checks identical numerical outputs and log
contents. These are synthetic measurements, not full-application load times.

- A 1024 x 4096 float64 derivative, measured over three runs with the source
  already allocated, reduced median `tracemalloc` peak allocation from
  **64.108 MiB to 32.406 MiB**. This measures the tested operation's allocations,
  not process RSS. Timing samples varied, so no SG speedup is claimed.
- For 2,000 log messages in bursts of 100, with a hidden Qt text document and
  a 300-message limit, document changes fell from **4,000 to 20**. Measured
  document-update time fell from **1,522.1 ms to 17.5 ms**. Both documents ended
  with identical latest-300-message contents. Visible application rendering
  and worker signal delivery are outside this benchmark.

The source application must be restarted to use the updated modules. No
executable was repackaged, no application was restarted, and no commit was
created in this turn.
