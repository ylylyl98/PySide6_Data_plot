"""Remember successful native loads when the main selection is later cleared.

Real PL/DRR workers and catalog scans use synthetic files and explicit temporary
settings in an isolated GUI process. No installed application profile is read.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickCatalogLoadBookmarkProcessTests(unittest.TestCase):
    def test_successful_load_bookmarks_in_isolated_gui_process(self):
        with TemporaryDirectory(prefix="catalog-load-process-") as directory:
            environment = dict(
                os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                MPLBACKEND="agg", MPLCONFIGDIR=directory, LOCALAPPDATA=directory,
                XDG_CONFIG_HOME=directory, XDG_CACHE_HOME=directory,
                QML_DISK_CACHE_PATH=directory, PYTHONIOENCODING="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=150,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CATALOG_LOAD_BOOKMARK_TESTS=", result.stdout)
        print(result.stdout.strip())


def _run_isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import json
    from threading import Event
    from time import monotonic, sleep
    import traceback
    from unittest.mock import patch

    import numpy as np
    from PySide6.QtCore import QSettings, QUrl
    from PySide6.QtGui import QGuiApplication

    from ui_qt.quick.application_bridge import QuickApplicationBridge

    app = QGuiApplication([])
    callback_errors = []

    def python_error(kind, value, trace):
        callback_errors.append("".join(traceback.format_exception(kind, value, trace)))
        traceback.print_exception(kind, value, trace)

    sys.excepthook = python_error

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="catalog-load-case-")
            self.addCleanup(self.temporary.cleanup)
            self.base = Path(self.temporary.name)
            self.owners = []
            self.addCleanup(self.close_owners)
            error_start = len(callback_errors)
            self.addCleanup(lambda: self.assertEqual(callback_errors[error_start:], []))

        def close_owners(self):
            for owner in reversed(self.owners):
                owner.close()
            app.processEvents()

        def owner(self, workflow):
            self.root = self.base / workflow
            self.raw = self.root / "Initial Data"
            self.raw.mkdir(parents=True)
            settings = QSettings(str(self.base / (workflow + ".ini")), QSettings.IniFormat)
            settings.setValue("drr/source_unprocessed_only", False)
            settings.setValue("pl/auto_load_next", False)
            self.bridge = QuickApplicationBridge(settings=settings)
            self.owners.append(self.bridge)
            self.bridge.selectWorkflow(workflow)
            self.bridge.setSourcePath(str(self.root))
            self.catalog = self.bridge.sourceCatalogBridge
            self.assertTrue(Path(settings.fileName()).is_relative_to(self.base))
            return self.bridge

        def csv(self, name, offset=0.):
            wave = np.linspace(700., 750., 41)
            gate = np.linspace(-2., 2., 9)
            profile = np.linspace(-1., 1., len(wave))
            values = 130 + offset + 15 * gate[:, None] + 4 * profile[None, :] ** 2
            path = self.raw / name
            np.savetxt(path, np.column_stack((gate * .5, gate, values)), delimiter=",",
                header="Vbg,Vtg," + ",".join(map(str, wave)), comments="")
            return path.resolve()

        def dat(self, name, offset=0.):
            folder = self.root / "Processed Data" / self.bridge._workflow.upper()
            folder.mkdir(parents=True, exist_ok=True)
            energy, gate = np.linspace(1.7, 2.7, 41), np.linspace(-2., 2., 9)
            values = 1 + offset + .1 * gate[:, None] + np.sin(energy[None, :] * 7)
            path = folder / name
            rows = ["X\t" + "\t".join(map(str, gate))]
            rows.extend("\t".join(map(str, [x, *values[:, index]])) for index, x in enumerate(energy))
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            if self.bridge._workflow == "drr":
                path.with_suffix(".metadata.json").write_text(json.dumps(dict(
                    operation="DR/R", processing=dict(derivative_order=0))), encoding="utf-8")
            return path.resolve()

        def wait(self, predicate, timeout=25):
            deadline = monotonic() + timeout
            while monotonic() < deadline:
                app.processEvents()
                if predicate():
                    app.processEvents()
                    return
                sleep(.003)
            self.fail(f"Native load/catalog did not settle: {self.bridge.snapshot['status']}; "
                f"catalog={self.catalog.snapshot}")

        def wait_load(self, expected=None):
            self.wait(lambda: not self.bridge._pending and not self.bridge.snapshot["busy"])
            if expected is not None:
                self.assertEqual(self.bridge._source, expected, self.bridge.snapshot["status"])
                self.assertTrue(self.bridge.snapshot["canExport"], self.bridge.snapshot["status"])

        def open_catalog(self, *, saved=False):
            self.bridge.invoke("saved-results" if saved else "select-files")
            self.wait(lambda: not self.catalog.snapshot["busy"])
            self.assertEqual(self.catalog.snapshot["error"], "")

        def choose(self, path):
            self.catalog.invoke("clear")
            group = next((row for row in self.catalog.groupsModel.rows if str(path) in row["members"]), None)
            if group is not None:
                self.catalog.setParameter("selectedGroup", group["id"])
            self.catalog.selectFile(str(path), True)
            self.assertTrue(self.catalog.snapshot["canAccept"], self.catalog.snapshot)

        def picker_load(self, path):
            self.open_catalog(saved=path.suffix == ".dat")
            self.choose(path)
            self.catalog.invoke("accept")
            self.wait_load(path)

        def external_load(self, path):
            if self.bridge._workflow == "pl":
                self.bridge.openFile(QUrl.fromLocalFile(str(path)))
            else:
                self.bridge.selectFiles([QUrl.fromLocalFile(str(path))])
            self.wait_load(path)

        def assert_bookmark(self, path, *, saved=False):
            self.bridge.invoke("clear-files")
            self.assertEqual(self.bridge._sources, ())
            self.open_catalog(saved=saved)
            self.assertEqual(self.catalog.snapshot["focused"], str(path))
            self.assertEqual(self.catalog.snapshot["selected"], [])
            self.assertFalse(self.catalog.snapshot["canAccept"])
            self.assertGreaterEqual(self.catalog.filesModel.indexOfId(str(path)), 0)
            if self.bridge._workflow == "drr" and not saved:
                group = next(row for row in self.catalog.groupsModel.rows if str(path) in row["members"])
                self.assertEqual(self.catalog.snapshot["selectedGroup"], group["id"])

        def loaded_pair(self, workflow):
            self.owner(workflow)
            suffix = "_PL_TG.csv" if workflow == "pl" else "_730nmc_TG_rep01.csv"
            first, second, third = [self.csv(name + suffix, index * 10.)
                for index, name in enumerate(("first", "second", "third"))]
            self.picker_load(first)
            self.external_load(second)
            return first, second, third

        def check_latest_raw_load(self, workflow):
            first, second, _third = self.loaded_pair(workflow)
            self.assertNotEqual(first, second)
            self.assert_bookmark(second)
            previous = self.catalog._catalog
            self.catalog.refresh(True)
            self.wait(lambda: not self.catalog.snapshot["busy"])
            self.assertIsNot(self.catalog._catalog, previous)
            self.assertEqual(self.catalog.snapshot["focused"], str(second))
            self.assert_bookmark(second)

        def test_drr_latest_external_load_restores_measurement_and_group(self):
            self.check_latest_raw_load("drr")

        def test_pl_latest_external_load_restores_measurement(self):
            self.check_latest_raw_load("pl")

        def check_saved_dat_load(self, workflow):
            self.owner(workflow)
            suffix = "_PL_TG.csv" if workflow == "pl" else "_730nmc_TG_rep01.csv"
            first, raw = self.csv("first" + suffix), self.csv("second" + suffix, 10.)
            saved_first, saved_second = self.dat("first.dat"), self.dat("second.dat", 1.)
            self.picker_load(first)
            self.external_load(raw)
            self.picker_load(saved_first)
            self.external_load(saved_second)
            self.assert_bookmark(saved_second, saved=True)
            self.assert_bookmark(raw)
            self.assert_bookmark(saved_second, saved=True)

        def test_drr_native_dat_load_keeps_separate_raw_bookmark(self):
            self.check_saved_dat_load("drr")

        def test_pl_native_dat_load_keeps_separate_raw_bookmark(self):
            self.check_saved_dat_load("pl")

        def test_failed_picker_load_preserves_last_successful_bookmark(self):
            for workflow in ("pl", "drr"):
                with self.subTest(workflow=workflow):
                    _first, second, third = self.loaded_pair(workflow)
                    self.open_catalog()
                    self.choose(third)
                    target = "ui_qt.quick.pl_workflow.load_pl_source" if workflow == "pl" else "ui_qt.quick.drr_workflow.load_drr"
                    with patch(target, side_effect=ValueError("Synthetic measurement failed")):
                        self.catalog.invoke("accept")
                        self.wait_load()
                    self.assertIn("Synthetic measurement failed", self.bridge.snapshot["status"])
                    self.assertFalse(self.bridge.snapshot["canExport"])
                    self.assert_bookmark(second)

        def test_superseded_successful_callback_preserves_last_bookmark(self):
            for workflow in ("pl", "drr"):
                with self.subTest(workflow=workflow):
                    _first, second, third = self.loaded_pair(workflow)
                    self.open_catalog()
                    self.choose(third)
                    entered, release = Event(), Event()
                    self.addCleanup(release.set)
                    from ui_qt.quick.drr_workflow import load_drr
                    from ui_qt.quick.pl_workflow import load_pl_source
                    loader = load_pl_source if workflow == "pl" else load_drr

                    def delayed(*args, **kwargs):
                        result = loader(*args, **kwargs)
                        entered.set()
                        if not release.wait(15):
                            raise AssertionError("Superseded synthetic worker was not released")
                        return result

                    target = "ui_qt.quick.pl_workflow.load_pl_source" if workflow == "pl" else "ui_qt.quick.drr_workflow.load_drr"
                    callbacks = []
                    self.bridge.requestFinished.connect(lambda sequence, kind, result, error, elapsed:
                        callbacks.append((sequence, error)))
                    with patch(target, side_effect=delayed):
                        try:
                            self.catalog.invoke("accept")
                            self.wait(entered.is_set)
                            sequence = self.bridge._sequence
                            cancellation = self.bridge._load_cancellation
                            # Simulate a completed scientific call whose success
                            # arrives after Clear supersedes the request. The
                            # sequence guard must reject even this valid cube.
                            with patch.object(cancellation, "check", return_value=None):
                                self.bridge.invoke("clear-files")
                                release.set()
                                self.wait_load()
                        finally:
                            release.set()
                    self.assertIn((sequence, ""), callbacks)
                    self.assertIsNone(self.bridge._source)
                    self.assertIsNone(self.bridge._cube)
                    self.assert_bookmark(second)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    app.processEvents()
    print("CATALOG_LOAD_BOOKMARK_TESTS=" + json.dumps(dict(tests=result.testsRun, success=result.wasSuccessful())))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
