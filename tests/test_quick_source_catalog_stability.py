"""Exercise real source watches without exposing user settings or measurements."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickSourceCatalogStabilityProcessTests(unittest.TestCase):
    def test_source_catalog_stability_in_isolated_gui_process(self):
        with TemporaryDirectory(prefix="quick-catalog-stability-process-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                QT_SCALE_FACTOR="1", MPLBACKEND="agg", MPLCONFIGDIR=temporary, LOCALAPPDATA=temporary,
                XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary, QML_DISK_CACHE_PATH=temporary,
                PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CATALOG_STABILITY_TESTS=", result.stdout)
        print(result.stdout.strip())


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import json
    from threading import Event
    from time import monotonic, sleep

    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication
    from PySide6.QtQuickControls2 import QQuickStyle

    from ui_qt.quick.source_catalog import SourceCatalogBridge

    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    if os.name == "nt" and not QFontDatabase.families():
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))
            app.setFont(QFont("Segoe UI", 9))

    class RecordingCatalog(SourceCatalogBridge):
        def __init__(self, settings):
            super().__init__(settings=settings)
            self.scan_calls = []
            self.pause_next = False
            self.entered = Event()
            self.release = Event()
            self.pause_saved_preview = False
            self.preview_entered = Event()
            self.preview_release = Event()

        def _scan_catalog(self, folder, workflow, **options):
            self.scan_calls.append((folder, workflow, options["force"]))
            original_preview = options["publish_cached"]
            def preview(value):
                original_preview(value)
                if self.pause_saved_preview and value["files"] and all(row["saved"] for row in value["files"]):
                    self.pause_saved_preview = False
                    self.preview_entered.set()
                    if not self.preview_release.wait(8):
                        raise AssertionError("Saved-only catalog preview was not released")
            options["publish_cached"] = preview
            if self.pause_next:
                self.pause_next = False
                self.entered.set()
                if not self.release.wait(8):
                    raise AssertionError("Catalog scan was not released")
            return super()._scan_catalog(folder, workflow, **options)

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="quick-catalog-stability-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name) / "experiment"
            self.raw = self.root / "Initial Data" / "REF"
            self.history = self.root / "Processed Data" / "DRR"
            self.raw.mkdir(parents=True)
            self.history.mkdir(parents=True)
            self.measurement = self.csv("sample_730nmc_TG_rep01.csv")
            # Keep caches inside the watched tree to exercise their own write noise.
            settings = QSettings(str(self.root / "settings.ini"), QSettings.IniFormat)
            settings.setValue("drr/source_unprocessed_only", False)
            self.catalog = RecordingCatalog(settings)
            self.addCleanup(self.catalog.close)
            self.addCleanup(self.catalog.release.set)
            self.addCleanup(self.catalog.preview_release.set)
            self.completed, self.busy_states = [], []
            self.catalog.operationCompleted.connect(lambda operation, elapsed: self.completed.append(operation))
            self.catalog.snapshotChanged.connect(lambda: self.busy_states.append(self.catalog.snapshot["busy"]))
            self.open_catalog()

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
            self.fail(f"Catalog did not settle: {self.catalog.snapshot}; scans={len(self.catalog.scan_calls)}")

        def observe(self, milliseconds=1000):
            deadline = monotonic() + milliseconds / 1000
            while monotonic() < deadline:
                app.processEvents()
                sleep(.003)

        def open_catalog(self):
            previous = len(self.completed)
            self.catalog.open("drr", str(self.root), [], False)
            self.wait(lambda: len(self.completed) > previous and not self.catalog.snapshot["busy"])
            self.assertFalse(self.catalog.snapshot["error"], self.catalog.snapshot["error"])

        def assert_settled(self, expected_scans):
            self.observe()
            self.assertFalse(self.catalog.snapshot["busy"])
            self.assertEqual(len(self.catalog.scan_calls), expected_scans)
            self.assertEqual(len(self.completed), expected_scans)

        def noise(self, iteration=0):
            (self.root / "preview.png").write_bytes(b"synthetic preview " + str(iteration).encode())
            (self.raw / "acquisition.log").write_text(str(iteration), encoding="utf-8")
            (self.raw / "pending.csv.tmp").write_text(str(iteration), encoding="utf-8")
            (self.root / "debug.json").write_text(json.dumps({"iteration": iteration}), encoding="utf-8")

        def test_cache_and_unrelated_writes_leave_the_catalog_idle(self):
            self.assertTrue((self.root / "quick-source-catalog").is_dir())
            self.assert_settled(1)
            busy_start = len(self.busy_states)
            for iteration in range(4):
                self.noise(iteration)
                self.observe(90)
            self.assert_settled(1)
            self.assertFalse(any(self.busy_states[busy_start:]), "Unrelated writes restarted source inspection")

        def test_raw_burst_and_saved_history_each_coalesce_to_one_refresh(self):
            second = self.csv("sample_730nmc_TG_rep02.csv")
            self.observe(60)
            third = self.csv("sample_730nmc_TG_rep03.csv")
            self.wait(lambda: len(self.completed) == 2 and not self.catalog.snapshot["busy"])
            self.assert_settled(2)
            self.assertEqual({row["id"] for row in self.catalog.filesModel.rows},
                {str(self.measurement), str(second), str(third)})
            (self.history / "group.metadata.json").write_text(json.dumps(dict(operation="DR/R",
                processing=dict(derivative_order=0), sources=[dict(role="measurement", source_path=str(self.measurement))])),
                encoding="utf-8")
            (self.history / "group.dat").write_text("X\t0\n1\t2\n", encoding="utf-8")
            self.wait(lambda: len(self.completed) == 3 and not self.catalog.snapshot["busy"])
            self.assert_settled(3)
            row = next(row for row in self.catalog.filesModel.rows if row["id"] == str(self.measurement))
            self.assertEqual(row["status"], "processed")
            third.unlink()
            self.wait(lambda: len(self.completed) == 4 and not self.catalog.snapshot["busy"])
            self.assert_settled(4)
            self.assertNotIn(str(third), {row["id"] for row in self.catalog.filesModel.rows})

        def test_noise_during_a_scan_and_local_filters_do_not_queue_another_scan(self):
            self.catalog.pause_next = True
            self.catalog.invoke("refresh")
            self.wait(self.catalog.entered.is_set)
            try:
                self.noise()
                self.observe(500)
            finally:
                self.catalog.release.set()
            self.wait(lambda: len(self.completed) == 2 and not self.catalog.snapshot["busy"])
            self.assert_settled(2)
            group = self.catalog.groupsModel.rows[0]["id"]
            self.catalog.setParameter("query", "rep01")
            self.catalog.setParameter("onlyNew", True)
            self.catalog.setParameter("includeBackgrounds", True)
            self.catalog.setParameter("selectedGroup", group)
            self.catalog.selectFile(str(self.measurement), True)
            self.catalog.setParameter("focused", str(self.measurement))
            self.catalog.setParameter("query", "")
            self.assert_settled(2)
            self.assertEqual(self.catalog.snapshot["selected"], [str(self.measurement)])

        def test_warm_saved_only_preview_keeps_raw_rows_and_groups_until_validation(self):
            (self.history / "group.metadata.json").write_text(json.dumps(dict(operation="DR/R",
                processing=dict(derivative_order=0))), encoding="utf-8")
            (self.history / "group.dat").write_text("X\t0\n1\t2\n", encoding="utf-8")
            self.wait(lambda: len(self.completed) == 2 and not self.catalog.snapshot["busy"])
            self.catalog.setParameter("selectedGroup", self.catalog.groupsModel.rows[0]["id"])
            self.catalog.selectFile(str(self.measurement), True)
            files, groups = list(self.catalog.filesModel.rows), list(self.catalog.groupsModel.rows)
            resets = []
            self.catalog.filesModel.modelReset.connect(lambda: resets.append("files"))
            self.catalog.groupsModel.modelReset.connect(lambda: resets.append("groups"))
            self.catalog.pause_saved_preview = True
            self.catalog.invoke("refresh")
            try:
                self.wait(self.catalog.preview_entered.is_set)
                app.processEvents()
                self.assertTrue(self.catalog.snapshot["busy"])
                self.assertEqual(self.catalog.filesModel.rows, files)
                self.assertEqual(self.catalog.groupsModel.rows, groups)
                self.assertEqual(resets, [])
            finally:
                self.catalog.preview_release.set()
            self.wait(lambda: len(self.completed) == 3 and not self.catalog.snapshot["busy"])
            self.assert_settled(3)
            self.assertEqual(resets, [])
            self.assertEqual(self.catalog.snapshot["selected"], [str(self.measurement)])

        def test_cancel_accept_and_close_stop_automatic_refresh_and_reopen_resumes_it(self):
            self.catalog.invoke("cancel")
            added = self.csv("sample_730nmc_TG_rep02.csv")
            self.assert_settled(1)
            self.open_catalog()
            self.assert_settled(2)
            self.assertIn(str(added), {row["id"] for row in self.catalog.filesModel.rows})
            self.catalog.selectFile(str(self.measurement), True)
            self.catalog.invoke("accept")
            self.csv("sample_730nmc_TG_rep03.csv")
            self.assert_settled(2)
            # Explicit requests still work after a dialog is dismissed.
            self.catalog.invoke("refresh")
            self.wait(lambda: len(self.completed) == 3 and not self.catalog.snapshot["busy"])
            self.assert_settled(3)
            self.catalog.open("drr", str(self.root), [], False)
            self.assert_settled(3)
            self.catalog.close()
            self.csv("sample_730nmc_TG_rep04.csv")
            self.assert_settled(3)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"CATALOG_STABILITY_TESTS={result.testsRun}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
