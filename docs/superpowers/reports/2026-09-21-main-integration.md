# Local application integration with main

## Scope and history

- Local checkpoint: `7650ec1` on `codex/file-selection-reliability`.
- Integrated upstream: `cf94348778fd5f61d49d2d4158cf2b152810add7`.
- Normal merge, no force push, remote unchanged.
- 21 files conflicted after checkpointing the full local application. Comparing upstream with the checkpoint showed that the conflicting upstream code was already present or superseded by the later local implementation. Retaining the examined local conflict blocks produced exactly the checkpoint tree before the verification fixes below. An independent review of the main window, DRR controller, shared UI and feature pages found no upstream-only fix to transplant.
- New runtime modules, tests and documentation are tracked. Generated `artifacts/` and root `native-*.png` files remain local and are ignored.

## Verification fixes

1. Confirmed that three-region **Save Both** writes both real PNG products. Separately reproduced the toolbar PNG failure through the actual toolbar method: local colorbar callbacks were not picklable. Moved the same locator/formatter calculations to module-level functions, using `partial` to bind the parent and index. The toolbar regression compares pixels with direct save after clearing the live figure.
2. Isolated Save Both tests with explicit temporary-file QSettings. On Windows the organization/application constructor still used NativeFormat despite setDefaultFormat(IniFormat), leaking region settings between tests. Removed the now-unnecessary process-wide default-format/path changes.
3. Isolated dense-layout tests from user settings and update checks. A 1180 px window legitimately compresses the DRR sidebar to 320 px because the plot toolbar needs more width. The 380 px row-layout test now uses a 1400 px window and requests splitter sizes summing to the actual available width. No production layout was changed.

## Validation boundaries

- Application import smoke check passed.
- CI representative tests plus MCD/Curie-Weiss, catalog and export-history tests: 157 passed.
- DRR Save Both, toolbar snapshot, three-region layout, title and asynchronous-save regression group: 32 passed.
- Dense-layout suite: 13 passed (Qt splitter rounding allows a 1 px tolerance). Save Both suite rerun after removing global settings mutations: all 14 passed. Across the focused suites, 202 distinct tests passed.
- Full discovery was attempted, then stopped after detecting real-user QSettings interference and a stale layout test assumption. It is **not** reported as a full-suite pass. Focused suites were rerun after the fixes.
- Independent review found no blocking issue in the integration fixes. This is not a packaged executable release test.
