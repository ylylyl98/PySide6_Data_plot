"""Warm source-picker reopen and cancellation use only isolated synthetic data."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickCatalogClearReopenProcessTests(unittest.TestCase):
    def test_clear_reopen_in_isolated_gui_process(self):
        with TemporaryDirectory(prefix="catalog-clear-reopen-process-") as directory:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                MPLBACKEND="agg", MPLCONFIGDIR=directory, XDG_CONFIG_HOME=directory,
                XDG_CACHE_HOME=directory, LOCALAPPDATA=directory, QML_DISK_CACHE_PATH=directory,
                PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_CATALOG_CLEAR_REOPEN_TESTS=", result.stdout)
        print(result.stdout.strip())


def _run_isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from contextlib import contextmanager
    import json
    from threading import Event, Thread
    from time import monotonic, sleep
    from unittest.mock import patch

    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QGuiApplication

    from ui_qt.quick import source_catalog

    app = QGuiApplication([])

    class RecordingCatalog(source_catalog.SourceCatalogBridge):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.scans = []
            self.pause_next = False
            self.entered, self.release = Event(), Event()
            self._watch_settle_ms = 60

        def _scan_catalog(self, folder, workflow, **options):
            self.scans.append((str(folder), workflow, options["force"]))
            if self.pause_next:
                self.pause_next = False
                self.entered.set()
                if not self.release.wait(12):
                    raise AssertionError("Synthetic inspection was not released")
                options["cancellation"].check()
            return super()._scan_catalog(folder, workflow, **options)

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="catalog-clear-reopen-case-")
            self.addCleanup(self.temporary.cleanup)
            self.base = Path(self.temporary.name)
            self.root = self.base / "experiment"
            self.raw = self.root / "Initial Data" / "REF"
            self.raw.mkdir(parents=True)
            self.history = self.root / "Processed Data" / "DRR"
            self.history.mkdir(parents=True)
            self.measurement = self.csv("sample_730nmc_TG_rep01.csv")
            self.settings = QSettings(str(self.base / "settings.ini"), QSettings.IniFormat)
            self.settings.setValue("drr/source_unprocessed_only", False)
            self.catalog = RecordingCatalog(settings=self.settings)
            self.addCleanup(self.catalog.close)
            self.addCleanup(self.catalog.release.set)

        def csv(self, name):
            path = self.raw / name
            path.write_text("Vbg,Vtg,730,731\n0,0,1,2\n1,2,3,4\n", encoding="utf-8")
            return path.resolve()

        def wait(self, predicate, timeout=15):
            deadline = monotonic() + timeout
            while monotonic() < deadline:
                app.processEvents()
                if predicate():
                    app.processEvents()
                    return
                sleep(.003)
            self.fail(f"Catalog did not settle: {self.catalog.snapshot}; scans={self.catalog.scans}")

        def observe(self, seconds=.6):
            deadline = monotonic() + seconds
            while monotonic() < deadline:
                app.processEvents()
                sleep(.003)

        def open_catalog(self, selected=()):
            self.catalog.open("drr", str(self.root), list(selected), False)
            self.wait(lambda: not self.catalog.snapshot["busy"])
            self.assertFalse(self.catalog.snapshot["error"], self.catalog.snapshot["error"])

        @contextmanager
        def accepted_preview(self):
            """Accept a real filename preview before its first CSV inspection."""
            from core.drr_sources import inspect_csv_gate
            from core.load_cancellation import check_load_checkpoint
            entered, release, checkpoint, continued = Event(), Event(), Event(), Event()
            self.addCleanup(release.set)
            def inspection(*args, **kwargs):
                entered.set()
                if not release.wait(12):
                    raise AssertionError("Synthetic acquisition inspection was not released")
                checkpoint.set()
                check_load_checkpoint()
                continued.set()
                return inspect_csv_gate(*args, **kwargs)
            accepted = []
            self.catalog.selectionAccepted.connect(lambda *args: accepted.append(args))
            with patch("core.drr_sources.inspect_csv_gate", side_effect=inspection) as inspect:
                self.catalog.open("drr", str(self.root), [], False)
                self.wait(lambda: entered.is_set() and self.catalog.filesModel.rowCount() > 0)
                cancellation = self.catalog._cancel
                self.addCleanup(cancellation.cancel)
                self.assertFalse(self.catalog._catalog_validated)
                self.catalog.selectFile(str(self.measurement), True)
                self.catalog.invoke("accept")
                self.assertEqual(accepted, [("drr", [str(self.measurement)])])
                self.assertFalse(cancellation.cancelled,
                    "Accepting a preview must retain the unfinished catalog inspection")
                self.assertTrue(self.catalog.snapshot["busy"])
                release.set()
                self.wait(checkpoint.is_set)
                self.observe(.12)
                self.assertFalse(continued.is_set(),
                    "The accepted preview must pause further cooperative inspection work")
                yield cancellation, inspect

        def test_main_clear_then_warm_reopen_does_not_rescan_or_reset_lists(self):
            from ui_qt.quick.application_bridge import QuickApplicationBridge
            with patch.object(source_catalog, "SourceCatalogBridge", RecordingCatalog):
                owner = QuickApplicationBridge(settings=self.settings)
            self.addCleanup(owner.close)
            self.addCleanup(owner._source_catalog.release.set)
            self.catalog = owner._source_catalog
            owner.selectWorkflow("drr")
            owner.setSourcePath(str(self.root))
            owner.invoke("select-files")
            self.wait(lambda: not self.catalog.snapshot["busy"])
            self.assertTrue(self.catalog._catalog_validated)
            self.catalog.invoke("cancel")
            rows = list(self.catalog.filesModel.rows)
            completed, busy_states, resets = [], [], []
            self.catalog.operationCompleted.connect(lambda *args: completed.append(args))
            self.catalog.snapshotChanged.connect(lambda: busy_states.append(self.catalog.snapshot["busy"]))
            self.catalog.filesModel.modelReset.connect(lambda: resets.append("files"))
            self.catalog.groupsModel.modelReset.connect(lambda: resets.append("groups"))
            owner.invoke("clear-files")
            owner.invoke("select-files")
            self.observe()
            self.assertEqual(len(self.catalog.scans), 1,
                "Clearing the main measurement selection must not invalidate an unchanged catalog")
            self.assertEqual(completed, [], "Warm reopen must not report another full inspection")
            self.assertNotIn(True, busy_states)
            self.assertEqual(resets, [])
            self.assertEqual(self.catalog.filesModel.rows, rows)
            self.assertTrue(self.catalog._catalog_validated)
            self.assertEqual(self.catalog.snapshot["selected"], [])

        def test_same_folder_open_reuses_inflight_inspection(self):
            self.catalog.pause_next = True
            self.catalog.open("drr", str(self.root), [], False)
            self.wait(self.catalog.entered.is_set)
            cancellation = self.catalog._cancel
            self.catalog.open("drr", str(self.root), [str(self.measurement)], False)
            self.assertFalse(cancellation.cancelled,
                "Opening the same in-flight catalog must not discard and restart its inspection")
            self.catalog.release.set()
            self.wait(lambda: not self.catalog.snapshot["busy"])
            self.assertEqual(len(self.catalog.scans), 1)
            self.assertEqual(self.catalog.snapshot["selected"], [str(self.measurement)])
            self.assertTrue(self.catalog._catalog_validated)

        def test_accept_preview_clear_and_reopen_resumes_the_same_worker(self):
            from ui_qt.quick.application_bridge import QuickApplicationBridge
            with patch.object(source_catalog, "SourceCatalogBridge", RecordingCatalog):
                owner = QuickApplicationBridge(settings=self.settings)
            self.addCleanup(owner.close)
            self.addCleanup(owner._source_catalog.release.set)
            self.catalog = owner._source_catalog
            owner.selectWorkflow("drr")
            owner.setSourcePath(str(self.root))
            with patch.object(owner, "selectFiles") as measurement_load:
                with self.accepted_preview() as (cancellation, inspect):
                    # The normal acceptance signal reaches the measurement
                    # loader even though catalog detail work is paused.
                    self.assertEqual(measurement_load.call_count, 1)
                    owner.invoke("clear-files")
                    owner.invoke("select-files")
                    self.wait(lambda: not self.catalog.snapshot["busy"])
                    self.assertIs(self.catalog._cancel, cancellation)
                    self.assertFalse(cancellation.cancelled)
                    self.assertEqual(len(self.catalog.scans), 1)
                    self.assertEqual(inspect.call_count, 1)
                    self.assertTrue(self.catalog._catalog_validated)
                    self.assertEqual(self.catalog.snapshot["selected"], [])

        def test_cancel_and_close_release_paused_preview_workers(self):
            for action in ("cancel", "close"):
                with self.subTest(action=action):
                    self.catalog = RecordingCatalog(settings=self.settings)
                    self.addCleanup(self.catalog.close)
                    self.addCleanup(self.catalog.release.set)
                    with self.accepted_preview() as (cancellation, _):
                        future = self.catalog._future
                        if action == "close":
                            self.catalog.close()
                        else:
                            self.catalog.invoke("cancel")
                        future.result(timeout=5)
                        self.assertTrue(cancellation.cancelled)
                        self.assertFalse(self.catalog.snapshot["busy"])
                        self.assertFalse(self.catalog._catalog_validated)

        def test_manual_refresh_releases_and_replaces_a_paused_preview(self):
            with self.accepted_preview() as (cancellation, _):
                previous = self.catalog._future
                self.catalog.invoke("refresh")
                self.wait(lambda: not self.catalog.snapshot["busy"])
                self.assertTrue(cancellation.cancelled)
                self.assertTrue(previous.done())
                self.assertIsNot(self.catalog._cancel, cancellation)
                self.assertEqual(len(self.catalog.scans), 2)
                self.assertTrue(self.catalog._catalog_validated)

        def test_paused_catalog_does_not_block_checks_inside_an_open_source(self):
            from core.load_cancellation import check_load_cancelled, load_scope
            with self.accepted_preview() as (cancellation, _):
                checked, errors = Event(), []
                def read_source():
                    try:
                        with load_scope(cancellation):
                            with self.measurement.open(encoding="utf-8") as source:
                                check_load_cancelled()
                                self.assertTrue(source.readline())
                    except Exception as error:
                        errors.append(error)
                    finally:
                        checked.set()
                reader = Thread(target=read_source, daemon=True)
                reader.start()
                try:
                    self.assertTrue(checked.wait(.5),
                        "A per-row cancellation check must finish while the catalog is paused")
                finally:
                    if not checked.is_set():
                        cancellation.cancel()
                    reader.join(timeout=2)
                self.assertEqual(errors, [])
                self.catalog.open("drr", str(self.root), [], False)
                self.wait(lambda: not self.catalog.snapshot["busy"])
                self.assertTrue(self.catalog._catalog_validated)

        def test_switching_folders_cancels_and_replaces_inflight_inspection(self):
            self.catalog.pause_next = True
            self.catalog.open("drr", str(self.root), [], False)
            self.wait(self.catalog.entered.is_set)
            cancellation = self.catalog._cancel
            other = self.base / "other-experiment"
            other_raw = other / "Initial Data" / "REF"
            other_raw.mkdir(parents=True)
            other_source = other_raw / "other_730nmc_TG_rep01.csv"
            other_source.write_text("Vbg,Vtg,730,731\n0,0,1,2\n1,2,3,4\n", encoding="utf-8")
            self.catalog.open("drr", str(other), [], False)
            self.assertTrue(cancellation.cancelled,
                "A different experiment must not reuse an unrelated pending scan")
            self.catalog.release.set()
            self.wait(lambda: not self.catalog.snapshot["busy"])
            self.assertEqual(len(self.catalog.scans), 2)
            self.assertEqual(set(self.catalog._lookup), {str(other_source.resolve())})
            self.assertEqual(self.catalog.snapshot["folder"], str(other.resolve()))

        def test_reopen_revalidates_files_and_history_changed_while_hidden(self):
            self.open_catalog()
            for change in ("add", "remove", "history"):
                with self.subTest(change=change):
                    self.catalog.invoke("cancel")
                    previous_scans = len(self.catalog.scans)
                    if change == "add":
                        second = self.csv("sample_730nmc_TG_rep02.csv")
                        ready = lambda: str(second) in self.catalog._lookup
                    elif change == "remove":
                        second.unlink()
                        ready = lambda: str(second) not in self.catalog._lookup
                    else:
                        (self.history / "group.metadata.json").write_text(json.dumps(dict(operation="DR/R",
                            processing=dict(derivative_order=0), sources=[dict(role="measurement",
                            source_path=str(self.measurement))])), encoding="utf-8")
                        (self.history / "group.dat").write_text("X\t0\n1\t2\n", encoding="utf-8")
                        ready = lambda: self.catalog._lookup.get(str(self.measurement), {}).get("status") == "processed"
                    self.catalog.open("drr", str(self.root), [], False)
                    self.wait(lambda: ready() and not self.catalog.snapshot["busy"])
                    self.observe(.5)
                    self.assertEqual(len(self.catalog.scans), previous_scans + 1)
                    self.assertTrue(self.catalog._catalog_validated)

        def test_reopened_watches_detect_later_changes_without_reopen_scan(self):
            self.open_catalog()
            self.catalog.invoke("cancel")
            self.catalog.open("drr", str(self.root), [], False)
            self.observe()
            self.assertEqual(len(self.catalog.scans), 1)
            second = self.csv("sample_730nmc_TG_rep02.csv")
            self.wait(lambda: str(second) in self.catalog._lookup and not self.catalog.snapshot["busy"])
            self.observe(.5)
            self.assertEqual(len(self.catalog.scans), 2)

        def test_obsolete_watch_result_cannot_refresh_a_new_dialog_session(self):
            self.open_catalog([str(self.measurement)])
            self.catalog._watch_timer.stop()
            self.wait(lambda: not self.catalog._watch_probe_running)
            original_inventory = self.catalog._source_inventory
            inventory = original_inventory(str(self.root), "drr")
            entered, release = Event(), Event()
            self.addCleanup(release.set)
            calls = []
            def delayed_inventory(*args, **kwargs):
                calls.append(args)
                if len(calls) == 1:
                    entered.set()
                    if not release.wait(12):
                        raise AssertionError("Obsolete watch probe was not released")
                    # This was once observed by the old session, but is no
                    # longer present when its worker returns after reopening.
                    return [*inventory, ["Initial Data/REF/obsolete.csv", 1, 1]]
                return original_inventory(*args, **kwargs)
            self.catalog._watch_settle_ms = 0
            with patch.object(self.catalog, "_source_inventory", side_effect=delayed_inventory):
                self.catalog._watch_refresh()
                self.wait(entered.is_set)
                self.catalog.invoke("accept")
                self.catalog.open("drr", str(self.root), [], False)
                release.set()
                self.wait(lambda: len(calls) >= 2 and not self.catalog._watch_probe_running)
                self.observe(.5)
            self.assertEqual(len(self.catalog.scans), 1,
                "A probe from an accepted dialog must not invalidate the reopened catalog")
            self.assertIsNone(self.catalog._watch_candidate)
            self.assertFalse(self.catalog.snapshot["busy"])

        def test_cancelled_inflight_result_cannot_publish_and_reopen_scans_again(self):
            self.catalog.pause_next = True
            completed = []
            self.catalog.operationCompleted.connect(lambda *args: completed.append(args))
            self.catalog.open("drr", str(self.root), [], False)
            self.wait(self.catalog.entered.is_set)
            cancelled = self.catalog._cancel
            self.catalog.invoke("cancel")
            self.assertTrue(cancelled.cancelled)
            self.assertFalse(self.catalog.snapshot["busy"])
            self.catalog.release.set()
            self.catalog._future.result(timeout=15)
            self.observe(.1)
            self.assertEqual(completed, [])
            self.assertFalse(self.catalog._catalog_validated)
            self.open_catalog()
            self.assertEqual(len(self.catalog.scans), 2)
            self.assertTrue(self.catalog._catalog_validated)

        def test_main_clear_discards_pending_automatic_next_intent(self):
            from ui_qt.quick.application_bridge import QuickApplicationBridge
            owner = QuickApplicationBridge(settings=self.settings)
            self.addCleanup(owner.close)
            owner._pl_auto_next_pending = str(self.measurement)
            owner.invoke("clear-files")
            self.assertIsNone(owner._pl_auto_next_pending,
                "A late catalog completion must not reopen a measurement after main Clear")

        def test_manual_refresh_and_force_rebuild_still_inspect(self):
            self.open_catalog()
            self.catalog.invoke("cancel")
            self.open_catalog()
            self.observe()
            self.assertEqual(len(self.catalog.scans), 1)
            self.catalog.invoke("refresh")
            self.wait(lambda: not self.catalog.snapshot["busy"])
            self.assertEqual(len(self.catalog.scans), 2)
            self.catalog.refreshCurrent("drr", str(self.root), [], True)
            self.wait(lambda: not self.catalog.snapshot["busy"])
            self.assertEqual(len(self.catalog.scans), 3)
            self.assertTrue(self.catalog.scans[-1][-1])

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("QUICK_CATALOG_CLEAR_REOPEN_TESTS=" + json.dumps(dict(tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors))))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
