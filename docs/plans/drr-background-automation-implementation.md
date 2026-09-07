# DRR automatic background implementation

**Authorization:** User requested automatic background processing and verification on 2026-09-07. The later instruction defers pushing/merging to main while MCD Peak Shift work is considered. Keep this implementation local.

**Goal:** Restore or reliably match each measurement's background without asking users to compare backgrounds or choose a numerical algorithm. Preserve existing common-background results.

**Architecture:** Resolve concrete per-measurement assignments from saved records or conservative candidates, carry those assignments through load/reload/export, and select a common or heterogeneous numerical path from the actual effective baselines. Existing explicit and pinned baseline choices remain overrides. No unrelated refactors or dependencies.

**Spec:** User requirements and `drr-background-automation-follow-up.md`; concrete decisions below resolve its open implementation choices.

## Resolution and persistence contract

- Add a validated per-measurement assignment representation in `core/drr_sources.py`. Each entry records measurement, Self first/last or External, ordered background files, frame method, source recipe and selection reason.
- Resolve membership in saved records, including new explicit per-measurement metadata. Use the newest unique historical record; equal-time conflicting assignments are unresolved. Missing recorded files or malformed per-measurement metadata must not silently fall back to a flat background union.
- Without a saved association, only accept an unambiguous high-confidence candidate compatible with known acquisition conditions and actual spectral data. Existing low/medium-confidence guesses and time-only tie breaking are insufficient for automatic resolution. Do not silently assign Self when resolution fails.
- Persist portable paths plus exact per-measurement associations in processing metadata. Parse legacy common-background records compatibly. A display union remains useful for source provenance, never for numerical pairing.
- Automatic fallback must check finite spectral coverage, sufficient samples, and all competing constant-gate candidates; the existing guesser's high-confidence label alone is insufficient. Restore tests include moved experiment folders and cleaned temporary working copies.

## Numerical contract

- Same effective External baseline means exactly equal energy and I0 arrays (including NaN positions), with no tolerance that could alter results. Preserve input ordering and baseline frame/file weighting. Common Self mode also retains its existing path.
- Use existing common-background loaders unchanged when assignments share an effective background. Freeze regression examples including mismatched measurement grids, NaNs and all/first/last frame methods.
- For heterogeneous assignments, compute each measurement's DR/R on its native grid using its own baseline, then linearly align the resulting DR/R to the first measurement's energy/gate grid. No extrapolation; outside coverage is NaN. Reject invalid/duplicate ambiguous interpolation axes or incompatible gate semantics. Average valid values with equal per-file weight; apply derivatives after averaging through the current UI derivative flow. Mixed saved Self and External modes follow the same per-measurement rule.
- Assignment identity participates in reload/cache comparisons, including when the display union is unchanged. Refresh resolves added measurements and reports missing recorded backgrounds without shrinking the recorded set. Export records the actual path and interpolation order.

## Implementation and verification steps

Owner: Luna implementation worker; parent reviews actual diff and test output; Astra independently reviews scientific and state/persistence correctness. No worker commits or pushes.

- [x] Add focused tests for member resolution, missing/conflicting history, conservative fallback and metadata round trip; implement resolver/parser.
- [x] Add numerical tests: A=20/BG=10 gives 1, B=40/BG=40 gives 0, mean=0.5; include different grids, duplicate baseline content, mixed Self/External, no overlap and NaNs. Implement `core/data_io.py`/`core/loader.py` bridge, preserving `process_ref_avg` common path.
- [x] Wire `ui_qt/controllers_drr.py`, `ui_qt/common.py`, and DRR call sites in `ui_qt/main_window.py`. Verify selection changes, pinned/manual overrides, refresh/reload, export metadata and restore. Automatic associations are not overridden by a displayed common frame selector.
- [x] Run focused source, numerical, UI and export tests with the existing Python venv; inspect actual diff and independent review findings. Fix in-scope failures and document acceptance limits.
- [ ] Push/PR/merge deferred by the user's latest instruction. No remote publication in this stage.

Python: `C:/Users/yanli/Desktop/pyside6_data_plot/PySide6_Data_plot/.venv/Scripts/python.exe`; use unittest (pytest unavailable). Real YZ303/YZ365 files are on the experiment computer, so local fixtures prove behavior but do not replace that acceptance.

## Review checkpoints

- Initial implementation passed 57 source/dialog/loader/export tests in the parent run. Independent numeric coverage was strengthened to distinguish interpolation order, unequal background frame counts and true heterogeneous reversed axes.
- Parent confirmed two numeric regression failures before correction: a measurement outside the reference grid was silently ignored, and descending equal-grid interpolation filled NaN holes. Both require correction before merge.
- Review confirmed manual frame/Self controls could retain stale assignments and reloaded exports could overwrite current pairings with old metadata. UI correction must prove actual control-to-load-to-export behavior; passing legacy unit tests alone is insufficient.
- The pre-change common numerical path has 36 frozen reference arrays covering External/Self first/Self last, raw/first/second derivatives, mismatched grids and NaNs. Initial replay matched exactly; rerun after final corrections.

## Final local verification

- Parent consolidated run: 89 tests passed in 41.441 seconds, covering sources, classification, loader, numeric regressions, export metadata, nine new Qt integration tests and seven existing DRR UI regressions.
- Final replay: all 36 pre-change reference arrays matched exactly, including NaN locations. `core/processing_run.py` remains unchanged.
- Astra independently reproduced and rechecked both final UI blockers: explicit External-to-Self switching and unresolved automatic refresh. Both corrected; no remaining blocker in that review scope.
- Final whitespace check passed. No full test suite or packaging acceptance is claimed. Known unrelated Compare assertions also fail on the original baseline; actual YZ303/YZ365 acceptance remains on the experiment computer.
