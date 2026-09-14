# File status and export review

## Task A — Approved by Astra

No actionable presentation regressions found. Power group status text/colors retain item identities and ordering. DRR member/chosen colors preserve missing/background distinctions. Selected text uses the highlighted palette color. The presentation additions introduce no scans or changes to filtering, role assignment, or validation.

Three exact tests passed together using `.venv/Scripts/python.exe`, offscreen Qt, and temporary INI QSettings: **Ran 3 tests in 5.377s; OK; exit code 0**.

Verification command (PowerShell):

```powershell
@'
import os, tempfile, unittest
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PySide6.QtCore import QSettings
with tempfile.TemporaryDirectory(prefix='astra-task-a-settings-') as tmp:
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, tmp)
    QSettings.setPath(QSettings.IniFormat, QSettings.SystemScope, tmp)
    names = [
        'tests.test_power_group_dialog.PowerGroupDialogTests.test_group_rows_show_status_with_theme_color_and_keep_group_role',
        'tests.test_ui_split_scale_controls.SplitScaleControlTests.test_drr_member_and_chosen_rows_use_state_colors_with_missing_priority',
        'tests.test_ui_filename_delegate.WrappedFilenameDelegateTests.test_selected_status_text_uses_highlighted_palette_color',
    ]
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    raise SystemExit(not result.wasSuccessful())
'@ | .venv/Scripts/python.exe -
```

An earlier attempt used the same temporary-settings wrapper with the entire `tests.test_power_group_dialog` module plus the two exact DRR/delegate tests. It terminated with exit code 1 after 18 progress dots and no unittest summary. It is **not counted as passing**, and no cause is established by that output alone. No production files were edited during this review.

## Task B — Approved by Astra

No actionable defects found in the bounded SHG history/writer changes. History recognizes actual writer settings filenames, uses exact source identity and retains roles/legacy ambiguity. The actual `SHG Processing` export completion queues catalog refresh. History and mtime dictionaries are detached worker results; local filters use those snapshots. Existing catalog folder/generation guards and close invalidation protect publication. The scalar finite check preserves formatting behavior.

Reviewer verification used `.venv/Scripts/python.exe`, offscreen Qt, and the same temporary-QSettings wrapper above, loading `tests.test_shg_history`, `tests.test_shg_controller_regressions.ShgControllerRegressionTests.test_source_filter_uses_cached_status_and_preserves_hidden_selection`, and `tests.test_shg`: **13 tests, OK, exit code 0**. This was before the final special-value test was added. The coordinator separately verified the final `tests.test_shg_history`: **4 tests, OK, exit code 0**, including real writer discovery and CSV byte equivalence for NaN, infinities, and negative zero.

A separate temporary-directory MainWindow smoke passed with exit code 0: actual catalog history scanning occurred off the GUI thread; invoking the actual SHG export-completion handler discovered newly written settings; Single/A/B/background choices survived; changing the status filter retained the hidden single selection without another history scan. Numerical loading and error dialogs were patched out to isolate selector behavior. An earlier smoke using placeholder CSVs without those patches stalled and was terminated; it is not counted as verification. No production files were edited during review.

## Task C — Approved by Astra

No remaining actionable defects found in the bounded final implementation. This review follows the final plan's semantics: Peak Shift indicates historical successful exports for the exact analysis source path; it does not certify current source bytes or current parameters. No raw hashing is introduced. The analysis source path and experiment folder are captured before later UI selections; export inputs are detached snapshots. The small history index is read through an owned worker and cached. Missing/legacy identity and malformed history remain Unknown, while a complete record for a different path does not mark a same-named source processed. Completion invalidates the exported source's cache even after the user switches elsewhere.

The actual Tools entry point (`run_mcd_organizer.py` → `McdOrganizerWindow`) now submits its export through an owned pool. The older extract dialog also uses the worker adapter and covers `done`/`reject` shutdown. Result/error publication stays on the GUI thread and is suppressed after closing. The shared ownership implementation retains signal QObjects until queued callbacks are delivered and disposes them on their owning thread. The existing scientific export writer and output formats are retained.

Independent reviewer verification, all exit code 0:

- Final Peak subset: `.venv/Scripts/python.exe`, Qt **6.11.2**, offscreen Qt and temporary INI QSettings; `tests.test_mcd_peak_async_export`, `tests.test_mcd_peak_export`, and `tests.test_mcd_peak_shift.MCDPeakShiftUITests.test_local_export_contains_both_channel_centers_and_valley_splitting`: **10 tests in 1.523s; OK**. The asynchronous tests use the actual private MainWindow pool with a small QObject/widget harness. They exercise completion, duplicate submission, errors, source switching/clearing, A→B→A history invalidation, and stale generations. The local-fit test waits for worker completion before inspecting both exported channels and valley splitting.
- Actual Organizer window: `tests.test_mcd_organizer_async_export`: **3 tests in 0.481s; OK**. These exercise background execution with a responsive GUI heartbeat, detached nested input dictionaries, duplicate suppression, error recovery, and close/deletion suppression.
- Actual extraction writer: the three `McdExtractTests` cases `test_export_writes_origin_workbook_png_and_settings_with_descriptive_base`, `test_optional_csv_export_separates_branches`, and `test_grouped_export_adds_compact_conditions_and_preserves_energy`: **3 tests in 4.350s; OK**.
- Older `McdExtractDialog`: a separate temporary-settings event-loop smoke exercised success, worker error/button recovery, and `reject()` during active export. All three paths passed; the worker executed off the GUI thread and the rejected dialog did not show a completion message.

The lifecycle tests deliberately substitute a blocking/failing writer to isolate Qt ownership and publication; the separate real writer and local-fit tests check output semantics. This was scoped verification, not a full Qt-suite run or a large-data performance benchmark. No user experiment outputs were written and no production files were edited by the reviewer.
