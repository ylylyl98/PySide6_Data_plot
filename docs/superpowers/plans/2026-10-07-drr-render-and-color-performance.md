# DRR rendering and automatic color limits implementation plan

**Goal:** Fix stale CSV parsing, eliminate duplicated split grids, and keep expensive DRR color statistics off the UI thread without changing numerical, Fix, memory, or export behavior.

**Approved scope:** The user accepted the recommended next round: CSV cache correctness, split rendering, and automatic color limits. Existing uncommitted changes are retained.

**Architecture:** Share change-sensitive file fingerprints. Represent split heatmaps as nonoverlapping full-resolution column slices. Calculate frozen ROI color statistics with a bounded cache and coalescing worker; apply results only through current control/Fix/memory guards. Small datasets may keep their existing synchronous path.

**Constraints:** Every measured cell and original edge remains represented; NaN/log/clip/zero and saved limits preserve existing semantics. Data and SG/source/ROI changes invalidate results. Closing, replacing data, or editing before completion cannot apply stale values. Temporary INI QSettings only in tests. No commits or app restart by the agent.

## 1. CSV freshness

- [x] Add a same-size rewrite/restored-mtime regression and unavailable-stamp cases.
- [x] Use Windows ChangeTime for raw parse cache signatures. Uncertain fingerprints must not be cached.
- [x] Run canonical/provenance/history cache regressions.

## 2. Split grids

- [x] Require mesh cell counts to sum to the original data count for two/three regions.
- [x] Compare rendered pixels to the full masked-grid reference, including nonuniform axes, NaN, clipping, log and boundary lines.
- [x] Slice original grid edges and matrices for split mesh creation and artist reuse.
- [x] Check cursor formatting/hit testing, independent norms/colorbars, full-resolution narrow peaks, PNG/DAT output, and mesh reuse.
- [x] Repeat the 512x2048 render measurement.

## 3. Automatic color statistics

- [x] Extract exact existing ROI/region percentile math into a pure helper.
- [x] Add a bounded cache, one active worker, coalesced pending requests, current-source publication guards, and shutdown/edit safety.
- [x] Keep only inexpensive geometry on the UI thread; reuse data/ROI/log/zero/region results and skip fully fixed/restored color pairs.
- [x] Connect completion to guarded color updates and debounced redraw. Preserve frozen exports and explicit Auto.
- [x] Test unchanged-range hits, changed ROI/source invalidation, no UI-thread percentile work for large data, manual edits/Fix while pending, memory restoration, and stale callbacks.

## Verification

- [x] Run focused and affected memory/input/view/reuse/save tests.
- [x] compileall and git diff --check.
- [x] Bounded read-only review and fixes for important findings.
- [x] Update documentation and report synthetic measurements separately from complete load latency.

## Results (2026-10-08)

- CSV freshness uses the same strong file fingerprint as provenance; uncertain fingerprints bypass caching. Same-size/restored-mtime and concurrent-change regressions pass.
- Split meshes retain original edges and all cells once. Nonuniform, NaN, clipping, zero-centered and log reference renders match exactly; independent colorbars, cursor peak lookup, artist reuse and export tests pass.
- DRR maps of 250,000 cells or more use one coalescing color worker and a 64-entry weak product cache. Fully fixed/restored pairs skip work; manual edits (including changing back to an earlier value), ROI/source/SG changes, empty/error results and clear/close are guarded. Completed jobs release their frozen arrays. Initial rendering resumes even when calculated limits are unchanged.
- Explicit whole/regional/second-product Auto uses the same mechanism. Uncached dE is prepared by the derivative worker before explicit Auto. Paired saves freeze the exact numerical product and request, so saving before/after completion uses the same colors.
- Read-only review findings (frozen dE export, ROI backtracking, explicit whole Auto, unchanged-result initial rendering) were reproduced and fixed. Final bounded reviewer rechecks found no P1/P2 issues.
- Verification batches: 72 core/cache/input/render tests passed; 94 memory/view/reuse tests passed; 61 region/UI tests passed with one existing skip; 17 targeted Auto/Fix/cache checks passed. The final post-repair color/math/export/cursor/clear/reuse selection passed all 31 tests. These batches overlap and are not an aggregate count.
- `python -m compileall -q core ui_qt tests` and `git diff --check` succeeded; Git reported only pre-existing LF/CRLF normalization warnings.
- Same-process synthetic Agg render, 512x2048, three regions, three-run medians: former full masked grids 1933.48 ms; sliced grids 619.08 ms; rendered RGBA exactly equal. Total mesh cells decreased from 3145728 to 1048576.
- Synthetic 1024x4096 raw/whole/left/right statistics: former UI calculation median 277.65 ms; new initial UI dispatch 7.84 ms; cached UI refresh median 0.54 ms. Background statistics and subsequent rendering still require their normal computation time; these measurements do not represent complete file Load latency.
- No commits, installs, or app restart performed. Earlier uncommitted work was retained.