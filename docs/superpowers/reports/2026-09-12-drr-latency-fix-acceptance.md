# DRR catalog latency repair — YZ365

## Changes

- The open DRR picker receives persisted catalog previews before background validation finishes. Preview delivery does not finish the worker or enable a premature refresh.
- One cache decode supplies both preview and validation.
- Raw source inspection and saved-result metadata have separate persistent catalogs. PNG/DAT output changes do not invalidate the raw catalog; metadata changes update history without inspecting unchanged raw inputs.
- When Initial Data/REF exists, DRR defaults to that subtree plus legacy root files. All data explicitly expands discovery. Without REF, the previous Initial Data traversal remains available.
- REF and All data have separate cache namespaces. Worker results carry their captured scope, so an older REF request cannot suppress an All data preview after a filter change.

## YZ365 measurements

Measured against `D:/instrument_control_v3_1/YZ365`, using temporary catalog storage and redirecting inspection-cache writes away from the experiment. No experiment input or existing export was changed.

| Case | Preview callback | Total catalog call | Raw discovery calls |
|---|---:|---:|---:|
| First new catalog build | none | 2.012 s | 1 |
| Warm catalog | 18.3 ms | 40.7 ms | 0 |
| Simulated metadata inventory change | 7.7 ms | 0.932 s | 0 |

The REF catalog contained 5 sources. These are worker-level measurements, not end-to-end desktop startup timings. The first new catalog build could reuse the existing per-file inspection cache. The metadata measurement simulated an inventory change; temporary-fixture regression tests separately cover actual metadata creation, modification and removal.

Raw measurement data: `artifacts/drr-latency-fixed-verification.json`.

PNG export is unchanged. The earlier temporary-folder benchmark measured approximately 1.56 s for a fresh Raw + derivative pair, mostly image rendering/encoding. This repair removes unnecessary catalog work after saving, not that image export cost.

## Verification

- 34 tests: cached preview, scope-switch race, catalog layers, generic cache, catalog reuse and picker startup.
- 56 tests: source discovery, source history/dialog rules, dual save workflow and dual export.
- 12 tests: lazy source catalogs.
- 11 tests: BG-only workflow and dual-view regressions.
- 12 background UI tests passed individually in separate processes, for 125 passing tests across the completed runs.
- Modified Python modules compile; scoped `git diff --check` reports no whitespace errors.

Some combined Qt UI runs did not complete normally; they are not counted as passes. All background UI cases passed in isolated processes. The cause of the combined-run interruption is not established by this verification.

The running application was not restarted. New code takes effect on the next launch; the first launch builds the new cache namespaces.
