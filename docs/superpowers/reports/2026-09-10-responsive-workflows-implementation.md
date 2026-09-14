# Responsive workflows implementation report

The implementation keeps the existing PySide6 worker and page controllers and
adds a bounded latest request state machine in `core/workflow_request.py`.
Each active load now carries a workflow, folder identity, and generation token.
While a load is busy, the newest complete `LoadOptions` snapshot is captured;
older results, errors, and finished callbacks cannot replace or clear that
newer request.  Compare completes its first valid group assignment with one
automatic load, and SHG starts after a complete Single or A/B assignment.

Power group refresh computes processed source names once per catalog pass;
Power confirmation performs cheap catalog checks immediately and runs full
sweep/VP validation in the owned worker, keyed by folder, file signatures,
roles, and pairing mode.  Completed results are reused by the main-window
handoff.  Compare history avoids repeated scans for an unchanged folder and is
force-refreshed for catalog application and export.  Plot redraws retain the
existing fast paths and no longer synchronously load a changed Power, Compare,
or PL source; source changes are routed through the worker request entry point.
SHG reprocessing is bounded to one active job plus one latest pending snapshot,
with source and role identity checks.  The shared source bar now reports
selected-versus-shown state, and expander state is persisted without resetting
parameters, scoped by workflow.  PL/DRR/Compare range refresh requests are
coalesced with their scheduled redraw.  SHG export waits for pending
reprocessing.  Optional workflow event recording remains disabled by default,
so no latency figures are inferred from code inspection.

Focused verification was run in isolated Python processes:

- `tests.test_workflow_request`: 10 tests passed.
- `tests.test_responsiveness`: 12 tests passed.
- `tests.test_compare_source_workflow`: 17 tests passed.
- `tests.test_pl_source_workflow`: 12 tests passed.
- `tests.test_drr_missing_selection` + `tests.test_drr_source_dialog`: 25 tests passed.
- `tests.test_power_group_dialog`: 17 tests passed (detached dialog fixtures
  use the existing synchronous harness; production dialogs use the owned
  MainWindow validation worker).
- `tests.test_source_picker_dialog`: 6 tests passed.
- `tests.test_compare_rotation_mapping`: 3 tests passed.
- `tests.test_mcd_shared_source`: 8 tests passed.
- `tests.test_power_prevalidated_load`: 2 tests passed.
- `tests.test_power_selection_regressions`: 4 tests passed.
- `tests.test_shg_controller_regressions`: 3 tests passed.
- `tests.test_astra_ui_minimal`: 4 tests passed.
- `tests.test_source_status_transitions`: 4 tests passed.
- `py_compile` and `git diff --check`: passed.

Cold and warm file timings, p50/p95 latency, event-loop delay, and queue
lengths remain unmeasured because the reproducible instrumentation is opt-in
and no representative data benchmark was supplied.  Large result caches,
legacy Peak Shift migration, and Slides/Tools restructuring remain deferred
because no measurement established them as bottlenecks; the SHG request queue
is bounded because its stale-result race was directly exercised.
