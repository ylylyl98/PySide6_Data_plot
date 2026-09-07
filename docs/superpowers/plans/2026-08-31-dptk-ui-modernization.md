# DPTK Desktop UI Modernization Implementation Plan

> **For agentic workers:** Use subagent-driven development (fresh agent per task,
> review gate between tasks) or executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize the PySide6 scientific application (DPTK Desktop) without breaking
existing behavior, by hardening module boundaries, completing workflow-controller
extraction, and polishing the Fluent 2 UI on the existing theme infrastructure.

**Architecture:** Keep the proven layered shape — UI-agnostic `core/` science/I/O
engines; a thin `ui_qt` shell; per-workflow pages/controllers; one theme manager.
Phase 1 removes the shared-symbol star-import cycle via a new `ui_qt/common.py`
boundary. Later phases extract the shell, finish controller ownership, introduce
reusable components, and apply Fluent polish — always behind the existing 385-test
regression net.

**Tech Stack:** PySide6 >= 6.11, Matplotlib, NumPy/SciPy/pandas, PyInstaller,
`ui_qt/fluent_ui` (vendored Fluent 2 runtime).

**Spec:** The approved architecture map and ten-section plan from the 2026-08-31
planning turn (current architecture, target architecture, phased migration,
verification strategy, risk areas, do-not-touch list). This file is the executable
form of that plan.

## Global Constraints

- Working tree contains pre-existing uncommitted/in-flight changes. Never stash,
  reset, checkout, revert, overwrite, or commit them. Treat all pre-existing diffs
  as protected. (`git status --short` snapshot in Baseline below.)
- Do not modify MCD Peak Shift implementation or its tests.
- Do not modify `ui_qt/feature_registry.py` unless a Phase-1 task absolutely requires
  it; Phase 1 does not.
- Do not change science/processing behavior (`core/*` logic, dataclass semantics,
  export formats, provenance rules).
- No unrelated cleanup. Each task touches only the files listed in the task.
- Preserve every runtime behavior and test-pinned contract: exact tab labels/order,
  hidden inner tab bar + workflow `QTabBar`, fixed 480 px sidebar, no horizontal
  list scrolling, spin boxes ignore mouse wheel, `WrappedFilenameDelegate` cache,
  drag-and-drop highlight, update-check dialogs, `objectName`s, `QSettings` keys.
- After Phase 1, no `from ui_qt.main_window import *` may remain in production code.
- No commits at any point during Phase 1; the user reviews before committing.

## Baseline (captured before Phase 1 edits)

`git status --short` at approval time (HEAD `c9219d5`):

```text
M  .github/workflows/build-windows-release.yml
 M app_version.py
 M run_mcd_organizer.py
 M run_qt.py
 M tests/test_mcd.py
MM tests/test_ui_split_scale_controls.py
 M ui_qt/controllers_drr.py
 M ui_qt/controllers_pl.py
 M ui_qt/feature_pages.py
 M ui_qt/feature_registry.py
 M ui_qt/features_tools.py
 M ui_qt/main_window.py
MM ui_qt/mcd_organizer_window.py
 M ui_qt/presentation_widget.py
?? core/mcd_peak_shift.py
?? tests/test_mcd_peak_shift.py
?? tests/test_theme.py
?? ui_qt/fluent_ui/
?? ui_qt/theme.py
```

Pre-existing dirty files that Phase 1 must NOT alter (other than
`ui_qt/main_window.py`, `ui_qt/feature_pages.py`, `ui_qt/controllers_*.py`,
`tests/test_mcd.py`, `tests/test_ui_split_scale_controls.py` — which already contain
migration edits that Phase 1 extends): `.github/workflows/build-windows-release.yml`,
`app_version.py`, `ui_qt/feature_registry.py`, plus untracked
`core/mcd_peak_shift.py`, `tests/test_mcd_peak_shift.py`, `tests/test_theme.py`,
`ui_qt/fluent_ui/`, `ui_qt/theme.py`, and the pre-existing edits in
`ui_qt/mcd_organizer_window.py` / `ui_qt/presentation_widget.py` /
`ui_qt/features_tools.py` / `run_qt.py` / `run_mcd_organizer.py`.

---

# Phase 0 — Baseline (COMPLETE)

- [x] Fluent theme infrastructure installed (`ui_qt/theme.py`, `ui_qt/fluent_ui/`).
- [x] Stale MCD Peak Shift tab expectation fixed; full suite green (385/385).

# Phase 1 — Dependency Hygiene (ACTIVE)

Goal: introduce `ui_qt/common.py` as the explicit shared-symbol boundary, replace
star imports with explicit imports, and remove the mid-module circular import —
with no behavioral change.

Exit criteria:
- 385/385 tests pass, or more with new tests.
- No `from ui_qt.main_window import *` in production code.
- feature pages/controllers no longer depend on `main_window` as a symbol provider.
- The mid-module circular import is removed.
- No behavioral changes, no unrelated files changed, no pre-existing changes lost,
  no commit.

### Task 1.1: Create `ui_qt/common.py`

**Files:**
- Create: `ui_qt/common.py`
- Create: `tests/test_common.py`

**Interfaces:**
- Produces: `UI_METRICS`, `QDoubleSpinBox`, `QSpinBox`, `QComboBox`, `LoadedState`,
  `LoadOptions`, `ExportOptions`, `WorkerSignals`, `Worker`, `WrappedFilenameDelegate`.

- [ ] **Step 1:** Read the current definitions in `ui_qt/main_window.py`:
  - `UI_METRICS` (~line 162)
  - `QDoubleSpinBox`, `QSpinBox`, `QComboBox` (lines 178–196)
  - `LoadedState` (~218), `LoadOptions` (~257), `ExportOptions` (~305)
  - `WorkerSignals` (~350), `Worker` (~356)
  - `WrappedFilenameDelegate` (~413–483)
- [ ] **Step 2:** Create `ui_qt/common.py` with a module docstring stating it is the
  explicit shared-symbol boundary, moving the definitions **verbatim** (same code,
  same defaults, same docstrings). Add exactly the imports those definitions need
  (PySide6 QtCore/QtGui/QtWidgets, `core.loader`, `core.plotting`, typing, numpy —
  copy from what `main_window.py` currently uses for these symbols).
- [ ] **Step 3:** Create `tests/test_common.py` (unittest, `QT_QPA_PLATFORM=offscreen`):
  - `UI_METRICS["left_width"] == 480` and key metric keys exist.
  - `LoadedState`/`LoadOptions`/`ExportOptions` construct with their current fields.
  - `QDoubleSpinBox`/`QSpinBox`/`QComboBox` subclasses exist and import cleanly.
  - `WrappedFilenameDelegate` is a `QStyledItemDelegate` subclass.
- [ ] **Step 4:** Run: `./.venv/Scripts/python.exe -m unittest tests.test_common -v`
  Expected: PASS.
- [ ] **Step 5:** Run: `./.venv/Scripts/python.exe -m unittest tests.test_ui_filename_delegate -v`
  Expected: still PASS (main_window untouched so far).
- [ ] **Step 6:** Self-review: `git diff -- ui_qt/common.py tests/test_common.py`;
  confirm only these two files were added and the moved code is byte-identical
  (module-level imports may differ).

### Task 1.2: Convert `ui_qt/feature_pages.py` to explicit imports

**Files:**
- Modify: `ui_qt/feature_pages.py`
- Modify: `tests/test_common.py` (add import-isolation check)

**Interfaces:**
- Consumes: `ui_qt.common` symbols; PySide6; numpy; stdlib.
- Produces: `FeatureTabsMixin` with identical builder methods; no import of
  `ui_qt.main_window`.

- [ ] **Step 1:** Replace `from ui_qt.main_window import *` with explicit imports:
  `UI_METRICS`, `QComboBox`, `QDoubleSpinBox`, `QSpinBox` from `ui_qt.common`;
  plus every PySide6/numpy/pathlib name the module actually uses (inventory by
  running imports/tests and resolving `NameError`s).
- [ ] **Step 2:** Add to `tests/test_common.py` an isolation test that spawns a
  subprocess: `python -c "import ui_qt.feature_pages, sys; assert 'ui_qt.main_window' not in sys.modules"`.
- [ ] **Step 3:** Run: `python -m py_compile ui_qt/feature_pages.py`, then the
  isolation test, then `python -m unittest tests.test_ui_split_scale_controls -v`.
  Expected: PASS.
- [ ] **Step 4:** Self-review the diff; no behavior edits beyond import changes.

### Task 1.3: Convert the five controllers to explicit imports

**Files:**
- Modify: `ui_qt/controllers_pl.py`, `ui_qt/controllers_drr.py`,
  `ui_qt/controllers_mcd.py`, `ui_qt/controllers_compare.py`,
  `ui_qt/controllers_power.py`, `ui_qt/controllers_shg.py`
- Modify: `tests/test_common.py` (isolation checks per controller)

- [ ] **Step 1:** For each controller, replace `from ui_qt.main_window import *`
  with explicit imports (`ui_qt.common`, PySide6, numpy, stdlib, and
  `ui_qt.theme`/`ui_qt.fluent_ui.style` helpers where used).
- [ ] **Step 2:** Add subprocess isolation assertions for each controller module
  (`import ui_qt.controllers_pl; 'ui_qt.main_window' not in sys.modules`, etc.).
- [ ] **Step 3:** Run `py_compile` on all five, the isolation tests, and
  `python -m unittest tests.test_responsiveness tests.test_pl_source_workflow tests.test_ui_split_scale_controls -v`.
  Expected: PASS.
- [ ] **Step 4:** Self-review the diff.

### Task 1.4: Convert `ui_qt/main_window.py` to consume `common` and remove the circular import

**Files:**
- Modify: `ui_qt/main_window.py`

- [ ] **Step 1:** Delete the local definitions of the moved symbols and import them
  from `ui_qt.common` instead. Because the imports live in `main_window`'s module
  namespace, the existing public/test-pinned re-exports keep working without an
  `__all__` (`UI_METRICS`, `LoadedState`, `LoadOptions`, `ExportOptions`,
  `QDoubleSpinBox`, `QSpinBox`, `QComboBox`, `WrappedFilenameDelegate`, `Worker`,
  `WorkerSignals`). Keep `_vp_short_title` in `main_window`.
- [ ] **Step 2:** Move `from ui_qt.feature_pages import FeatureTabsMixin` and the five
  `from ui_qt.controllers_* import ...` lines from mid-module (~lines 486–494) into
  the top import block, after confirming those modules no longer import `main_window`.
- [ ] **Step 3:** Run: `python -m py_compile ui_qt/main_window.py` and
  `python -m unittest tests.test_mcd tests.test_pl_source_workflow tests.test_ui_split_scale_controls tests.test_responsiveness -v`.
  Expected: PASS.
- [ ] **Step 4:** Self-review: no behavioral edits, only imports/definition moves.

### Task 1.5: Final Phase-1 verification and review

- [ ] **Step 1:** `rg -n "from ui_qt.main_window import \*" ui_qt -g "*.py"`
  Expected: no matches.
- [ ] **Step 2:** Full suite: `QT_QPA_PLATFORM=offscreen ./.venv/Scripts/python.exe -m unittest discover -s tests`
  Expected: 0 failures (385 + new tests).
- [ ] **Step 3:** Smoke: `./.venv/Scripts/python.exe scripts/smoke_check.py` -> ok.
- [ ] **Step 4:** Review the Phase-1 diff vs. the baseline snapshot; confirm no
  pre-existing changes were lost and no unrelated files changed.
- [ ] **Step 5:** Report A–J per the approval message.

---

# Phase 2 — Shell Extraction

Move navigation bar, status bar, toolbar/menus, docks, and workspace stack
construction into `ui_qt/shell/` modules; `MainWindow` stays a thin facade owning
state and orchestration. Exit criteria: shell construction reviewable in isolation;
full suite green; no behavior change.

# Phase 3 — Controller Completion

Finish moving each workflow's behavior out of `main_window`; delete its
`__getattr__` shims one workflow at a time. Order: PL -> DRR -> Compare -> Power ->
SHG -> MCD (MCD last: blitting + toolbar save hooks). Exit criteria: per-workflow
tests green; no shims remain for completed workflows.

# Phase 4 — Reusable Components

`SourcePickerDialog` (migrate PL/DRR/MCD pickers), `StatusBadge`, `Expander`,
`EmptyStateOverlay`, item delegates. Exit criteria: components tested in isolation;
duplicated picker logic removed.

# Phase 5 — Fluent Polish

SVG icon set (replace emoji badges), Fluent spin steppers, dialog geometry
persistence, Load/Plot/Save shortcuts, status-bar region structure, canvas
empty/loading states. Exit criteria: existing suite + light/dark screenshots pass.

# Phase 6 — Plot/Theme Integration

Dark Matplotlib figure theme driven by the theme manager; chart color policy.
Exit criteria: render checks in both themes.

# Phase 7 — Accessibility & Verification

Screen-reader pass, high-contrast verification, 100–200% scaling, reduced-motion
setting, screenshot gallery. Exit criteria: Fluent completion-gate checklist.

---

## Self-Review (plan)

- Spec coverage: all ten approved sections map to phases above; Phase 1 tasks carry
  the required test/verification steps.
- Placeholder scan: Phase 1 steps contain concrete files, commands, and expected
  results; later phases are workstream-level per the approved plan (their task
  decomposition is generated when each phase starts).
- Type consistency: symbol names in Tasks 1.2–1.4 match Task 1.1's produced
  interfaces exactly.
