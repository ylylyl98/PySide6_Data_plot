# Source Selection Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair source refresh, missing-source handling, and historical processing labels across existing file-selection flows without changing each tab's business model.

**Architecture:** Retain the existing controllers and picker widgets. Add a small identity-only helper for metadata association, use the current async catalog worker for Compare refresh, and expose conservative Compare history from existing metadata rather than computing freshness.

**Tech Stack:** Python, PySide6/Qt Widgets, standard-library JSON/pathlib/unittest, existing NumPy and project code.

**Spec:** `docs/superpowers/specs/2026-09-10-source-selection-reliability-design.md`

## Global Constraints

- Keep the existing Python/PySide6 stack and dependencies; use `python -m unittest` because pytest is absent in the default Python 3.13 environment.
- Work on `codex/file-selection-reliability` in the existing workspace; preserve all pre-existing uncommitted Rot1/Rot2 mapping changes and tests. Do not create a separate worktree, reset files, stage, or commit.
- Preserve per-tab business differences: PL single raw/result selection, DRR measurements plus independent backgrounds, Compare groups plus manual channel mappings, MCD shared source, Power sweep/role groups, and SHG single/reference/sample/background choices.
- Processing status means saved history, not verified freshness for the current source bytes or current UI settings. Do not hash measurements during catalog refresh or introduce a generic processing-status framework.
- Keep source identities in item data and metadata; status decorations must not become loader paths.
- Preserve explicit Clear and Cancel behavior. Refresh must not silently substitute unrelated sources when a prior selection disappears.

---

## Execution context and file map

The user already selected **Astra plan → Luna execution → Astra review**. Luna executes inline with `executing-plans`; do not ask for another execution choice or approval. Each task ends with a test/diff checkpoint, not a commit. Production files already dirty before this work include `core/processing.py`, `ui_qt/controllers_compare.py`, `ui_qt/feature_pages.py`, `ui_qt/main_window.py`; compare tests also contain prior changes. Read their current contents and add narrow changes.

Primary implementation files:

- Create `core/source_identity.py`: portable exact/legacy identity matching only.
- Modify `core/data_io.py`, `core/mcd.py`: retain historical status APIs, report ambiguous legacy associations, write future MCD relative identity.
- Modify `ui_qt/main_window.py`: carry ambiguity with async scan results, emit completion to Compare dialog, preserve DRR missing choices and guard loading, pass MCD experiment root.
- Modify `ui_qt/controllers_pl.py`, `ui_qt/controllers_mcd.py`: truthful history/unknown filtering, badge text and auto-advance exclusions.
- Modify `ui_qt/controllers_compare.py`, `ui_qt/source_picker_dialog.py`: real refresh completion, preserved mappings and missing selections.
- Create `core/compare_history.py`; modify `core/export.py`: read Compare records and enrich future records with source mapping, without rewriting output files.
- Modify `ui_qt/controllers_drr.py`: missing-choice labels, dialog validation, reusable load validation.
- Modify `ui_qt/controllers_power.py`, `ui_qt/feature_pages.py`: existing Power predicate in combo rows; simple SHG hint.

Baseline supplied by parent: Compare 45 tests, PL source workflow 11, DRR source dialog 14, Power group dialog 16, MCD shared source 8 all passed (94 total). Use `tests/ui_test_helpers.py::wait_for_file_catalog` for asynchronous UI tests; do not assume `_refresh_file_lists` is synchronous.

### Task 1: Exact PL/MCD historical identity and explicit ambiguity

**Files:** Create `core/source_identity.py`, `tests/test_source_processing_identity.py`; modify `core/data_io.py::discover_pl_processing_status`, `core/mcd.py::discover_mcd_processing_status` and `export_mcd_analysis_bundle`, `ui_qt/main_window.py::_scan_folder_sources_worker`, `_on_file_lists_result`, initialization/reset and MCD export call, `ui_qt/controllers_pl.py` history predicates/filtering/auto-advance, `ui_qt/controllers_mcd.py` history filtering/badges. Extend `tests/test_pl_source_workflow.py`, `tests/test_mcd_shared_source.py`.

**Interfaces:**

```python
def match_source_identity(
    experiment_root: str | Path, sources: Sequence[str], *,
    relative_path: str = "", path: str = "", legacy_name: str = "",
) -> tuple[str | None, tuple[str, ...]]:
    # Returns (one resolved source or None, ambiguous candidates).

def discover_pl_processing_status(experiment_root, sources, *, ambiguous_sources=None) -> dict[str, str]: ...
def discover_mcd_processing_status(experiment_root, sources, *, ambiguous_sources=None) -> dict[str, str]: ...
# Both optional outputs are mutable set[str]. Existing 2-argument callers stay valid.
# Add experiment_root: str | Path | None = None to export_mcd_analysis_bundle's keyword-only arguments.
```

- [ ] **1. Reproduce both bugs with table-driven unittest cases.** Create real temporary source paths and JSON metadata using existing output folder conventions. The following assertions are mandatory (expand into unittest methods with temporary directories):

```python
sources = ["Initial Data/a/same_PL.csv", "Initial Data/b/same_PL.csv"]
metadata = {"workflow": "PL", "created_utc": "2026-09-01T00:00:00+00:00",
            "sources": [{"name": sources[0]}]}
# Write metadata under Processed Data/PL/result.metadata.json.
self.assertEqual(discover_pl_processing_status(root, sources), {sources[0]: metadata["created_utc"]})
unknown = set()
# Replace sources descriptor with {"name": "same_PL.csv"}.
self.assertEqual(discover_pl_processing_status(root, sources, ambiguous_sources=unknown), {})
self.assertEqual(unknown, set(sources))
```

For MCD use `mcd/a/same.csv`, `mcd/b/same.csv`, `Processed Data/MCD/*_MCD_settings*.json`, `workflow=MCD`, and legacy `source_file`. Test explicit `source_relative_path` and absolute `source_path`; basename-only duplicate evidence yields unknown for both. Also cover unique legacy fallback, slash/case normalization, moved experiment with explicit relative identity, an exact missing source plus a same-name different source (no fallback), corrupt/non-object JSON, non-list sources, wrong workflow, and exact evidence overriding unknown for only its matched candidate. Root-level same-name source versus nested duplicate must use explicit relative/absolute evidence; do not treat a legacy basename as proven root-level identity.

- [x] **2. Run the new tests and record the failing assertions.**

```powershell
python -m unittest tests.test_source_processing_identity -v
```

- [x] **3. Implement identity matching and scanner changes.** Normalize `\\` to `/` and casefold. Strong relative identity resolves only its exact candidate. A current-root absolute path becomes relative identity. A legacy path with a directory component is strong. Basename-only fallback succeeds only for exactly one candidate; otherwise return its candidate tuple as unknown. Never append basename matches after a strong match, and never recover a missing strong relative path via basename. Select descriptor fields in order `source_relative_path`, `source_path`/`path`, `name`/`source_file`/`filename`; preserve a portable relative `name` when an obsolete absolute path cannot be anchored to the current root. For conflicting usable strong fields, report no confirmed association and mark their affected candidates unknown. Keep the helper free of metadata loading and UI code.

```python
matched, uncertain = match_source_identity(
    root, raw_sources,
    relative_path=str(descriptor.get("source_relative_path", "")),
    path=str(descriptor.get("source_path", descriptor.get("path", ""))),
    legacy_name=str(descriptor.get("name", descriptor.get("filename", ""))),
)
ambiguous.update(uncertain)
if matched is not None:
    status[matched] = max(created, status.get(matched, ""))
# After all metadata:
ambiguous.difference_update(status)
if ambiguous_sources is not None:
    ambiguous_sources.update(ambiguous)
return status
```

`export_mcd_analysis_bundle` retains `source_file` and writes `source_path` plus `source_relative_path` when under the provided experiment root. Add them also to its measurement descriptor. Main-window export passes `experiment_root=folder`. If standalone callers omit root, absolute source_path still provides identity; do not infer experiment root from output-directory depth.

- [x] **4. Carry uncertainty into current UI without breaking existing worker callers.** Initialize/reset `pl_processing_ambiguous` and `mcd_processing_ambiguous` sets on folder changes. `_scan_folder_sources_worker` obtains them in the same pass as each status dictionary and appends them to its current seven result items; `_on_file_lists_result` accepts the old seven-item test/compatibility form with empty ambiguity sets and the new nine-item form. Propagate MCD pending ambiguity alongside `_mcd_status_from_refresh`.

```python
folder, csv_files, map_files, pl_files, pl_status, mcd_files, mcd_status, *ambiguity = result
pl_unknown, mcd_unknown = ambiguity if len(ambiguity) == 2 else (set(), set())
# Assign only after existing folder/generation guards succeed.
```

PL/MCD All shows Unknown rows, counts exclude unknown from New/Processed, and an `unknown` filter displays them. Keep the saved-DAT special state. PL `_auto_load_next_unprocessed_pl` excludes ambiguity. Badge/hint text uses `Saved history` or `History unknown`; include `History does not verify current settings`. Peak Shift says `MCD source history` using these same states; do not add a Peak Shift status store. Add UI assertions for unknown filter/count, auto-advance exclusion, reset on empty/different folder, and explicit MCD wording on both tabs.

- [x] **5. Verify task and inspect the diff.**

```powershell
python -m unittest tests.test_source_processing_identity tests.test_pl_source_workflow tests.test_mcd_shared_source tests.test_mcd -v
git diff --check
```

Expected: all pass; original status calls still return timestamp dictionaries; no history scan hashes or reads measurement bodies.

### Task 2: Compare Refresh actually rescans and preserves intent

**Files:** Modify `ui_qt/main_window.py` catalog completion/error notification, `ui_qt/controllers_compare.py` group picker/combo refresh/auto-assignment/selection validation, `ui_qt/source_picker_dialog.py`; extend `tests/test_compare_source_workflow.py`, `tests/test_source_picker_dialog.py`.

**Interfaces:** Add owner `file_catalog_refresh_finished = Signal(str, bool)` (folder, success), emitted after valid catalog application or terminal error. Add backward-compatible keyword `auto_select_single: bool = True` to `SourcePickerDialog`; Compare passes False. Existing catalog worker remains authoritative. Add `CompareController._cmp_missing_sources(mapping: dict[str, str]) -> list[str]` using exact resolved paths when a current folder exists.

- [ ] **1. Add a real open-dialog refresh regression.** Create group A CSV names from the existing `_angle_owner` fixtures in a temporary folder. Open a MainWindow with `_restore_last_folder` patched, wait for its first catalog, manually swap KK/KKp, open the picker with `exec` patched to schedule/check the Refresh callback through the Qt event loop. Add group B to disk after dialog creation, click Refresh, wait for catalog completion, and assert B appears while group A's key, swap flag, mapping and angle references are unchanged. Repeat deleting A so only B remains: current row must be None and OK disabled, and the main mapping remains A with missing indicators. Assert Cancel does not commit the local tolerance/group preview.

```python
# In the patched dialog exec callback, after the second scan completes:
self.assertEqual(window.compare_controller._cmp_current_mapping(), swapped)
self.assertFalse(dialog.ok_button.isEnabled())  # when deleted A was the prior selection
self.assertIsNone(dialog.source_list.currentItem())
self.assertTrue(window.compare_controller._cmp_missing_sources(swapped))
```

Test the generic picker separately: `auto_select_single=False` keeps a removed prior choice empty; default True retains prior behavior. Include close-while-refresh, scan error button reset, folder change, explicit Clear then Refresh, and a surviving manual path outside the current PL filter.

- [x] **2. Run focused tests and observe failures.**

```powershell
python -m unittest tests.test_compare_source_workflow tests.test_source_picker_dialog -v
```

- [x] **3. Wire real completion and preserve missing mappings.** Dialog Refresh calls `_refresh_file_lists(auto=True)`, marks request pending, and disables Refresh until matching completion. Callback checks the captured folder and dialog lifetime, rebuilds from the new catalog, and preserves dialog-local search/tolerance/key. Connect with a QObject/dialog lifetime context or explicitly disconnect on `finished`; do not mutate widgets from stale queued callbacks. Error completion restores Refresh and shows a concise failure detail. Prevent whole-folder scan completion from inferring new angles or replacing a missing manual mapping during automatic refresh.

```python
# In _cmp_set_channel_combo_items, after ordinary candidates:
if current and current not in candidates:
    combo.addItem(current)
    index = combo.findText(current)
    # Existing present/outside-filter rule stays distinct from truly missing.
    combo.setItemData(index, missing_or_outside_filter_tooltip, Qt.ToolTipRole)
combo.setCurrentText(current)
```

Use `_cmp_missing_sources` to retain existing mapping during `preserve_existing=True` refresh even if a file disappears. Do not let auto-assignment replace missing values; explicit Auto Assign/group acceptance can replace them. `current_folder`-less lightweight owners keep existing catalog-only tests usable. `_cmp_selection_from_ui` first computes its active view channels, validates their required paths when a real folder is set, and raises `ValueError("Compare source missing: ...")` before constructing the load request. Missing hidden channels need not block a valid VP pair, but remain visibly marked.

- [x] **4. Verify current Rot mapping and source behavior.**

```powershell
python -m unittest tests.test_compare_source_workflow tests.test_compare_rotation_mapping tests.test_compare_vp tests.test_source_picker_dialog -v
git diff --check
```

Expected: initial coherent auto-assignment, angle mapping and explicit swap/clear still pass; no unrelated fallback after deletion.

### Task 3: Retain and block missing DRR choices

**Files:** Modify `ui_qt/controllers_drr.py::_open_drr_source_dialog`, `_update_drr_selection_labels`; `ui_qt/main_window.py::_on_drr_catalog_refresh_result` and DRR branch constructing LoadOptions; extend `tests/test_drr_source_dialog.py`, `tests/test_drr_background_ui.py`; create `tests/test_drr_missing_selection.py` for UI/load guards if existing modules become crowded.

**Interfaces:** Add `DrrController._drr_missing_sources(sources: Sequence[str]) -> list[str]`, preserving input order/deduplicating and using `resolve_source_path(current_folder, source).is_file()`. Reuse it for chosen rows, summary and load guard.

- [ ] **1. Add deletion regressions using real temporary files.** Select A and B, delete B, refresh catalog, assert selection still equals `[A, B]`, B row/summary says Missing, and accepting the dialog is blocked. Remove B explicitly, assert acceptance succeeds with A. Repeat for a manual external baseline and for an absolute baseline outside the experiment that still exists (valid). Test deletion after dialog acceptance but before load; patch `thread_pool.start` or the loading worker creation to assert no worker launches. Test stale automatic-assignment baseline paths and a precomputed XLSX missing source so neither bypasses the guard.

```python
window.drr_selected_files = ["Initial Data/a_REF.csv", "Initial Data/b_REF.csv"]
(root / window.drr_selected_files[1]).unlink()
# Complete a catalog refresh, then:
self.assertEqual(window.drr_selected_files, ["Initial Data/a_REF.csv", "Initial Data/b_REF.csv"])
self.assertEqual(window.drr_controller._drr_missing_sources(window.drr_selected_files),
                 ["Initial Data/b_REF.csv"])
# Invoke the existing load entry point under a worker spy; assert not started.
```

- [x] **2. Run failing tests.**

```powershell
python -m unittest tests.test_drr_missing_selection tests.test_drr_source_dialog -v
```

- [x] **3. Implement at all three boundaries.** Stop filtering `drr_selected_files` and explicitly manual `drr_baseline_files_manual` solely to catalog membership in `_on_drr_catalog_refresh_result`. Retain their exact values, annotate missing state, and skip automatic complete-group expansion when any existing selection is missing. Preserve the existing automatic-background unresolved branch and do not turn its display union into a manual baseline recipe. Dialog `_update_type_hint` also updates missing row text/tooltips/OK enabled state after add/remove/clear/refresh. Route `buttons.accepted` through a callback that rechecks disk immediately; disabling OK alone cannot handle deletion after the last repaint.

```python
def _accept_chosen():
    paths = [str(selected_list.item(i).data(Qt.UserRole)) for i in range(selected_list.count())]
    missing = self._drr_missing_sources(paths)
    if missing:
        # Retain rows; update Missing labels and a readable error detail.
        return
    dlg.accept()
```

At the main-window DRR load boundary validate selected measurements, explicitly required manual baselines, and each automatic recipe's baseline paths **before** background resolution or worker construction. On missing inputs call the existing invalidation/status path with `Missing DRR source(s): ...` and return; never alter the selected list or launch a subset. Catch `OSError` as unavailable. Do not reject existing files just because they are not in the catalog.

- [x] **4. Verify background automation remains intact.**

```powershell
python -m unittest tests.test_drr_missing_selection tests.test_drr_source_dialog tests.test_drr_background_ui tests.test_drr_sources -v
git diff --check
```

### Task 4: Truthful Compare history for the current view and source roles

**Files:** Create `core/compare_history.py`, `tests/test_compare_history.py`; modify `core/export.py::export_compare_panels`, `ui_qt/controllers_compare.py` group badge/details/assignment summary and refresh cache, `ui_qt/main_window.py` existing catalog/export completion refresh hooks; extend `tests/test_compare_vp.py` metadata assertions.

**Interfaces:**

```python
def discover_compare_history(experiment_root: str | Path) -> list[dict]:
    # Reads only Processed Data/Compare/**/*.metadata.json; records contain metadata_path.
def compare_history_for_selection(
    records: Sequence[dict], experiment_root: str | Path,
    mapping: dict[str, str], *, view: str,
) -> list[dict]:
    # view is "Intensity" or "VP". Returned record adds history_scope:
    # "combination", "vp_pair", or "individual_panel".
```

Consumes Task 1's identity helper; source matching uses all candidate raw source identities from metadata/current mapping where required to disambiguate legacy names. Do not treat unique names within a selected two-file subset as globally unique if the experiment catalog contains duplicates. Pass/retain the full raw candidate catalog as an optional `sources: Sequence[str] = ()` parameter on `compare_history_for_selection`; the controller always supplies `_cmp_assign_candidate_files()` plus its present outside-filter mapped paths. Exact identity remains preferred.

- [ ] **1. Write metadata-only tests (no rendering).** Create JSON records for PL, `Compare/KK`, `Compare/KKp`, and `Compare/VP`. Assert PL is ignored; VP requires KK and KKp in their correct roles in one record; swapping roles excludes the old VP record; Intensity excludes VP. Add differing `compare_source_mapping` combinations, duplicate basename ambiguity, corrupt/non-dict JSON and malformed descriptor lists. Old matching panel-only records return `individual_panel`; they never produce combination status. Full matching maps return `combination`, and full mismatching maps are excluded even if that panel's input matches.

```python
mapping = {"KK": "Initial Data/a.csv", "KKp": "Initial Data/b.csv"}
record = {"operation": "Compare/VP", "sources": [
    {"role": "source_KK", "name": mapping["KK"]},
    {"role": "source_KKp", "name": mapping["KKp"]}], "created_utc": "2026-09-01"}
self.assertEqual(len(compare_history_for_selection([record], root, mapping, view="VP")), 1)
self.assertEqual(compare_history_for_selection([record], root,
    {"KK": mapping["KKp"], "KKp": mapping["KK"]}, view="VP"), [])
```

Also assert current display excludes other-view records, active channel subset is respected, current source/settings changes never produce `Current`/`Up to date` wording, and empty/different folder clears history cache. Existing panel records cannot prove joint settings/mapping; tests must not fabricate certainty from similar timestamps or filenames.

- [x] **2. Run tests to confirm missing history behavior.**

```powershell
python -m unittest tests.test_compare_history -v
```

- [x] **3. Implement the two narrow functions and future metadata field.** Validate payload/processing/sources shapes defensively. Use operation prefix exactly, retain created timestamp and path for details, and match source roles with Task 1 helper. Add `compare_source_mapping: dict(source_files)` to the `processing` dictionary of both existing panel and VP `write_export_metadata` calls. Preserve all existing fields and export names; do not add a second export or alter numerical processing.

```python
processing={
    "scale": scale_tag,
    "background_constant": correction_background,
    "clip_outliers": clip_outliers,
    "compare_source_mapping": dict(source_files),
}
# VP keeps its formula/background and adds the same mapping field.
```

Cache parsed records on the Compare controller by folder, refresh after applied catalog scans and successful Compare exports, and re-filter the cached records on mapping/view changes. Append a short text to the existing badge/summary: `Saved Compare VP history: <date>` or `Saved Compare Intensity history: <date>`, with tooltip `Historical export; current settings not verified`. Legacy panels say `Individual panel history; combined selection not verified` and list matched channels/times in details. No records means `No saved Compare history found`. Pending/failed refresh must not relabel uncertain history as New. Do not scan JSON files on every plotting parameter signal.

- [x] **4. Run tests and inspect serialization compatibility.**

```powershell
python -m unittest tests.test_compare_history tests.test_compare_vp tests.test_compare_source_workflow -v
git diff --check
```

### Task 5: Finish scoped labels and verify the integrated behavior

**Files:** Modify `ui_qt/controllers_power.py::_power_refresh_groups`, `ui_qt/controllers_mcd.py::_open_mcd_source_dialog`, `ui_qt/feature_pages.py` SHG source controls. Extend `tests/test_power_group_dialog.py`, `tests/test_mcd_shared_source.py`; create `tests/test_source_selection_hints.py` for SHG/MCD hint checks if needed.

**Interfaces:** Existing `core.power_workflow.processed_source_names(folder)` and `is_combined(folder, source.file_name)` remain authoritative. No new public status API. Existing MCD discovery priority and SHG item data stay unchanged.

- [ ] **1. Verify the uncovered label cases.** With mocked Power processed names, test a table, an all-processed legacy record group, a partial group and a combined table. Assert main sweep and KK/KKp combo display matches the existing picker policy while `itemData` and selection stay unchanged. MCD tests use an actual dedicated folder with CSV and root CSV to assert hint scope, then an empty dedicated folder to assert root fallback; discovery results must remain identical. SHG tests assert measured-angle/numeric-wavelength hint is present and an unknown CSV remains available with its original loader identity. Text-only wording checks need not become fragile whole-string tests.

```python
names = [source.file_name] if source.file_name else [r.file_name for r in source.records]
processed = combined or bool(names) and all(
    name.replace("\\", "/").casefold() in processed_names for name in names)
prefix = "[Processed · Combined] " if combined else "[Processed] " if processed else "[New] "
# Prefix existing label only; combo.addItem(prefix + label, key).
```

- [x] **2. Apply the minimal UI changes.** Reuse the same Power predicate in picker and main/role combo construction, extracting a private controller helper only if needed to prevent divergence. Do not add a new Power enum. MCD hint: `Scanning mcd/** CSV files; root CSV fallback is used when no dedicated MCD CSVs exist` with the active scope explicitly named. SHG hint: `SHG expects a measured-angle column (for example measured_value or measured angle) and numeric wavelength columns. All root CSVs remain available.` Preserve existing widgets, sorting and all-file access.

- [x] **3. Run the focused suite, then the full suite once.**

```powershell
python -m unittest tests.test_source_processing_identity tests.test_compare_history tests.test_drr_missing_selection tests.test_source_selection_hints tests.test_compare_source_workflow tests.test_compare_rotation_mapping tests.test_compare_vp tests.test_pl_source_workflow tests.test_mcd_shared_source tests.test_drr_source_dialog tests.test_power_group_dialog tests.test_source_picker_dialog -v
python -m unittest discover -s tests -v
git diff --check
git status --short
```

Remove a proposed new module from the command only if its tests were deliberately placed in the listed existing module instead. Record actual counts/results; if full-suite failures are unrelated, reproduce and document them rather than claiming a clean run. Do not broaden/repeat a passing full run without a later change or new concern.

- [ ] **4. Handoff to Astra review.** Report changed files, tests/counts, any documented false-positive recommendations left unchanged, and remaining limitations. Review must specifically check: relative-path precedence and ambiguous New counts; stale refresh/dialog callbacks; one-row fallback; missing Compare/DRR paths at loading; full-vs-panel Compare metadata semantics; current Intensity/VP filtering; per-tab reset/Cancel/Clear behavior; and preservation of the uncommitted Rot1/Rot2 changes. Do not stage or commit.

## Plan self-review

Spec coverage: Task 1 covers duplicate histories, uncertainty and Peak Shift MCD wording; Task 2 covers Compare refresh and missing mapping intent; Task 3 covers DRR dialog/catalog/load boundaries; Task 4 covers Compare operation/view/mapping history; Task 5 covers Power rows, MCD scope, SHG hints and integrated verification. The plan deliberately leaves Power picker status and MCD directory priority intact. Identity helper is the only cross-workflow utility; no status framework is introduced. Existing scanner return APIs are retained, with optional outputs and old worker result compatibility. All changes remain reviewable in one Luna execution run.
