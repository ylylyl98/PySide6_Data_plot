# DRR selection history latency

Implemented bounded metadata caching and request-local path memoization in
`core/drr_sources.py`. Entries check metadata inventory/stat signatures and
source-path dependencies on every reuse. Dependency stamps are captured before
resolution and checked before publication, including moves during parsing.
Returned collections are isolated; the process cache retains at most eight
root/filter entries. Existing source catalog workers prewarm the strict recipe
cache even when their catalog was loaded from disk.

The measurement selection callback passes its resolution into `_start_load`,
including an unresolved default-Self result. Pending load collection preserves
that result. Existing explicit Self, saved recipes, and external assignments
retain their behavior. No spectral numerical algorithms were changed.

## Measurements

Data: YZ365 p5n2, 1 T, 720 nm, 0.08 s x40, BG=18; 157 x1340 points.
Actual historical resolver before: 4.537 and 4.826 seconds (unprofiled).
After: repeated resolver 0.124 and 0.145 seconds in an initial benchmark.

Final offscreen Qt experiment with real CSV and Self (last frame):

| Stage | Cold history | Repeated selection |
| --- | ---: | ---: |
| Selection to worker | 1.176 s | 0.141 s |
| Load worker | 0.256 s | 0.019 s |
| Apply loaded state | 0.091 s | 0.063 s |
| Process events and explicit canvas draw | 0.407 s | 0.391 s |
| Sum | 1.930 s | 0.614 s |

These are isolated-window timings, excluding app startup, catalog opening,
user interaction, and desktop compositor latency. Cold measurement deliberately
does not prewarm history; normal background catalog completion now prewarms it.
The script invokes the load worker synchronously for timing and draws explicitly.
Artifacts: `artifacts/diagnose_drr_selection.py`, `artifacts/drr-selection-timing.json`.

## Verification

- New unchanged-cache test failed before implementation (2 parses instead of 1).
- New default-Self selection test failed before implementation (2 history queries).
- New worker prewarm test failed before implementation (history reparsed).
- 61 history/source/background numerical and UI tests passed; 4 BG-only workflow tests passed.
- After adding worker prewarm, 16 targeted DRR history/catalog/worker tests passed.
- Independent review identified the concurrent-move cache race; fixed, regression
  covered, and re-review found no further actionable findings.
- Broader `test_responsive_catalog_metadata` run had four MCD metric failures
  (`captured['trace']` missing / zero traces); those metric code paths were not
  modified. This is not a claim that the entire repository suite passes.
- Scoped `git diff --check` passed.

No running user application was restarted; already-loaded code requires a new
source-based application process. No packaged executable was rebuilt.
