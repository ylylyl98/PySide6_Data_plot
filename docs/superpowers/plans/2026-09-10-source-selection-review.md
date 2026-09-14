# Source selection reliability review

Date: 2026-09-10. Reviewer: Astra, following the user-requested Astra plan, Luna execution, Astra review sequence.

## Decision

Approved for the bounded source-selection reliability change. The actionable findings raised during review are resolved. This is a targeted review approval, not a claim that the complete repository test suite passes.

Reviewed against the source-selection reliability design and implementation plan in this directory and `../specs/2026-09-10-source-selection-reliability-design.md`. Pre-existing Rot1/Rot2 edits were distinguished using the initial working-tree snapshot. No production code was edited by the reviewer.

## Resolved findings

- Compare refresh retains missing and outside-filter manual assignments and suppresses inference for an existing usable manual mapping. Explicit reassignment remains available.
- Queued catalog scans notify the open Compare picker after the final applied scan. A real worker regression verifies newly created groups appear even when Refresh was queued behind an older scan.
- Compare missing-source loading checks use the active Intensity channels or VP pair. Hidden assignments are retained.
- Closed Compare dialogs disconnect their catalog callbacks, and acceptance checks the captured experiment folder.
- PL history-unknown sources are excluded from New; malformed MCD source descriptors no longer crash catalog discovery.
- Compare history distinguishes active view and channel subsets, neutral individual-panel history, absent roles, and empty history. Identity and full-mapping regressions cover conservative association.
- DRR retains missing selections, blocks dialog acceptance until removal, checks existing measurements and baselines before background resolution, and checks newly resolved recipe paths before worker creation. Missing automatic baselines and precomputed XLSX inputs are covered.

## Verification evidence

- Coordinator independently ran the twelve focused modules in isolated processes: 124 tests passed with exit code 0 and complete unittest summaries before the last DRR guard. Logs: `C:\Users\commo\AppData\Local\Temp\codex-selection-final-p2ulw4vx`.
- After that final guard, the coordinator reran `tests.test_drr_missing_selection`: 6 tests passed with exit code 0 and a complete summary.
- Reviewer independently reran the real queued-refresh regression: 1 test passed with exit code 0 and a complete summary.
- Reviewer independently reran `test_resolved_recipe_missing_baseline_blocks_before_worker` after inspecting the final guard: 1 test passed with exit code 0 and a complete summary.
- Final reviewer `git diff --check` exited 0; output contained only Git line-ending notices.

## Full-suite limitations

The attempted combined/full runs are not reported as passing. The coordinator reproduced the QtCore DLL import failure and the MCD owned-window cleanup failure in the pre-change snapshot; isolated cleanup tests passed. A baseline full run also terminated without a final unittest summary. These observations establish baseline limitations, not a clean full-suite result. Focused results above have explicit completion summaries; progress dots alone were not counted as successful verification.
