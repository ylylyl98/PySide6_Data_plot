# DRR coordinate reuse and streaming averages

## Scope

This round implements the two recommended first priorities: measured coordinate
collection during numerical loading, and bounded-memory multi-file averaging.
The existing manual Fix, exact-file settings and wavelength-template precedence
are preserved. Other proposed improvements remain separate work.

## Implementation

- `CoordinateAccumulator` consumes each measurement's native gate and coordinate
  vectors while the canonical spectrum is already loaded. It retains coordinate
  vectors only. Reversal, monotonicity, reference-grid coverage, cross-file
  agreement and agreement at finer native knots retain the existing rules.
- Both numerical DRR paths return the resulting coordinate mappings on
  `DataCube.gate_coordinates`. The load worker uses those mappings directly,
  avoiding its previous second canonical spectrum pass solely for the right axis.
- `StreamingNanMean` retains one float64 sum and one int64 count grid. Inputs are
  not modified or retained; only NaN is excluded. Infinity/opposing-infinity and
  empty-pixel behavior remain consistent with the original NaN averages.
- The common-baseline path still aligns each raw measurement to the first grid
  before applying its baseline. The heterogeneous path still applies each
  assignment on its native grid before aligning corrected results. Raw averages
  and derivative processing order remain unchanged. No selection-wide intensity
  list or stacked intensity tensor is created in either average path.
- The standalone `load_coordinates` helper remains available. Normal application
  loads collect coordinates in one pass. The existing raw parse cache remains a
  separate, bounded source of retained arrays.

## Verification

The new tests first reproduced the old second parse, selection-wide stacking,
retention of prior measurements and loss of coordinate metadata. After the
implementation, 218 selected tests passed, covering loading, background numerics,
axis resolution, source workflows, settings memory, deferred export, measured
right axis and analysis windows. An additional hand-derived raw/DRR average test
covers zero denominators, per-file weights, NaN, infinities and input isolation;
31 numerical/cache tests passed in the subsequent focused run.

```powershell
$env:QT_API = 'pyside6'
$env:QT_QPA_PLATFORM = 'offscreen'
.venv\Scripts\python.exe -m unittest tests.test_streaming_average tests.test_drr_load_streaming tests.test_canonical_parse_cache tests.test_loader tests.test_drr_background_numerics tests.test_y_axis_resolution tests.test_drr_sources tests.test_pl_source_workflow tests.test_drr_display_memory tests.test_drr_deferred_export tests.test_drr_right_axis tests.test_drr_analysis_window -q
.venv\Scripts\python.exe -m unittest tests.test_streaming_average tests.test_drr_load_streaming tests.test_drr_background_numerics tests.test_canonical_parse_cache -q
```

The all-NaN fixtures retain expected empty-slice warnings. The existing peak
analysis fixture also emits a zero-width peak warning. No test failure occurred.
`compileall` and `git diff --check` passed; Git emitted line-ending conversion
warnings for existing Windows working files.

Independent read-only review reported no P1/P2 or noteworthy minor findings.
The reviewer ran 27 focused tests and 45 in-memory old/new comparisons covering
1/2/7/24 files, common and heterogeneous baselines, mismatched grids, NaN/±Inf
and derivative orders None/1/2. All comparisons passed; common-path finite
outputs matched exactly.

## Synthetic measurement

The reproducible script is `artifacts/drr-streaming-optimization/benchmark.py`.
Its baseline uses commit `6fc7a86349d8a5c4688be624db6f5a4911ae9398` for the unchanged
pre-round loader, processing and coordinate functions. Both versions include
averaging and coordinate collection, including the old separate coordinate pass.
CSV parsing, its cache, history lookup, drawing and UI are excluded.

24 synthetic 256 × 1024 float64 matrices, three trials per path:

| Numerical path | Before peak allocation | After peak allocation | Before median | After median |
| --- | ---: | ---: | ---: | ---: |
| Common baseline | 208.10 MiB | 14.12 MiB | 338.54 ms | 267.56 ms |
| Heterogeneous assignments | 206.38 MiB | 12.06 MiB | 318.88 ms | 297.66 ms |

The measured peak is traced allocation during this workload, **not total process
RSS**. Timing includes allocation tracing and varies with machine load; it does
not establish a full Load time. Each trial checks a pixel against an independently
computed scalar reference. A single-file check also showed no regression:
14.52 → 14.10 MiB peak allocation and 16.94 → 14.80 ms for the common path.

```powershell
.venv\Scripts\python.exe artifacts\drr-streaming-optimization\benchmark.py --baseline --output artifacts\drr-streaming-optimization\before.json
.venv\Scripts\python.exe artifacts\drr-streaming-optimization\benchmark.py --output artifacts\drr-streaming-optimization\after.json
```

Per-trial results are recorded in `before.json`, `after.json`,
`before_single.json` and `after_single.json` in the benchmark directory.
