"""Warm the native picker while idle and yield to real measurement loading.

The child process owns temporary INI settings and synthetic PL/DRR sources.
Event gates exercise worker ordering without machine-specific speed limits.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickCatalogIdleIntegrationProcessTests(unittest.TestCase):
    def test_idle_catalog_in_isolated_quick_process(self):
        with TemporaryDirectory(prefix="catalog-idle-process-") as directory:
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
        self.assertIn("QUICK_CATALOG_IDLE_INTEGRATION=", result.stdout)
        print(result.stdout.strip())


def _run_isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from contextlib import contextmanager
    import json
    from threading import Event
    from time import monotonic, sleep
    import traceback
    from unittest.mock import patch

    import numpy as np
    from PySide6.QtCore import QSettings, QUrl
    from PySide6.QtGui import QGuiApplication

    from ui_qt.quick import source_catalog
    from ui_qt.quick.application_bridge import QuickApplicationBridge

    app = QGuiApplication([])
    callback_errors = []

    def python_error(kind, value, trace):
        callback_errors.append("".join(traceback.format_exception(kind, value, trace)))
        traceback.print_exception(kind, value, trace)

    sys.excepthook = python_error

    class Gate:
        def __init__(self):
            self.entered, self.release, self.checkpoint, self.continued = (
                Event(), Event(), Event(), Event())

        def wait(self):
            self.entered.set()
            if not self.release.wait(15):
                raise AssertionError("Synthetic worker gate was not released")

    class RecordingCatalog(source_catalog.SourceCatalogBridge):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.scans = []
            self.gate = None
            self._watch_settle_ms = 40

        def _scan_catalog(self, folder, workflow, **options):
            self.scans.append((str(folder), workflow, bool(options["force"])))
            gate, self.gate = self.gate, None
            if gate is not None:
                gate.wait()
                gate.checkpoint.set()
                options["cancellation"].checkpoint()
                gate.continued.set()
            return super()._scan_catalog(folder, workflow, **options)

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="catalog-idle-case-")
            self.addCleanup(self.temporary.cleanup)
            self.base = Path(self.temporary.name)
            self.owners, self.gates = [], []
            self.addCleanup(self.close_owners)
            error_start = len(callback_errors)
            self.addCleanup(lambda: self.assertEqual(callback_errors[error_start:], []))

        def close_owners(self):
            for gate in self.gates:
                gate.release.set()
            for owner in reversed(self.owners):
                owner.close()
            app.processEvents()

        def gate(self):
            gate = Gate()
            self.gates.append(gate)
            return gate

        def csv(self, root, workflow, name="sample"):
            raw = root / "Initial Data" / ("PL" if workflow == "pl" else "REF")
            raw.mkdir(parents=True, exist_ok=True)
            suffix = "_PL_TG.csv" if workflow == "pl" else "_REF_730nmc_TG_rep01.csv"
            path = raw / (name + suffix)
            wave, gate = np.linspace(700., 750., 41), np.linspace(-2., 2., 9)
            profile = np.linspace(-1., 1., len(wave))
            values = 130 + 15 * gate[:, None] + 4 * profile[None, :] ** 2
            np.savetxt(path, np.column_stack((gate * .5, gate, values)), delimiter=",",
                header="Vbg,Vtg," + ",".join(map(str, wave)), comments="")
            return path.resolve()

        def owner(self, workflow="pl", *, gated=False):
            root = self.base / f"experiment-{len(self.owners)}"
            path = self.csv(root, workflow)
            settings = QSettings(str(self.base / f"profile-{len(self.owners)}.ini"), QSettings.IniFormat)
            settings.setValue("drr/source_unprocessed_only", False)
            settings.setValue("pl/auto_load_next", False)
            with patch.object(source_catalog, "SourceCatalogBridge", RecordingCatalog):
                owner = QuickApplicationBridge(settings=settings)
            self.owners.append(owner)
            # Rendering is unrelated to loader/catalog scheduling. Keep the
            # real domain loaders, validation, source scans and status owner.
            plot_patch = patch.object(owner, "_schedule_plot")
            plot_patch.start()
            self.addCleanup(plot_patch.stop)
            self.owner_bridge, self.catalog = owner, owner.sourceCatalogBridge
            self.dialog_requests = []
            owner.sourceCatalogRequested.connect(lambda: self.dialog_requests.append("catalog"))
            owner.fileDialogRequested.connect(lambda *_: self.dialog_requests.append("files"))
            owner.backgroundDialogRequested.connect(lambda: self.dialog_requests.append("background"))
            owner.selectWorkflow(workflow)
            gate = self.gate() if gated else None
            self.catalog.gate = gate
            owner.acceptFolder(workflow, "source-folder", QUrl.fromLocalFile(str(root)))
            return owner, root, path, gate

        def wait(self, predicate, timeout=12):
            deadline = monotonic() + timeout
            while monotonic() < deadline:
                app.processEvents()
                if predicate():
                    app.processEvents()
                    return
                sleep(.003)
            self.fail(f"Idle integration timed out: {self.owner_bridge.snapshot['status']}; "
                f"catalog={self.catalog.snapshot}; scans={self.catalog.scans}")

        def catalog_ready(self, workflow, root):
            self.wait(lambda: self.catalog._catalog_validated and not self.catalog.snapshot["busy"]
                and self.catalog.snapshot["workflow"] == workflow
                and self.catalog.snapshot["folder"] == str(root.resolve()))
            self.assertEqual(self.catalog.snapshot["error"], "")

        def load_idle(self, owner):
            self.wait(lambda: not owner._pending and not owner.snapshot["busy"])

        def select(self, owner, path):
            owner.selectFiles([QUrl.fromLocalFile(str(path))])

        @contextmanager
        def blocked_loader(self, workflow, *, error=False, gates=None):
            from ui_qt.quick import drr_workflow, pl_workflow
            module, name = (pl_workflow, "load_pl_source") if workflow == "pl" else (drr_workflow, "load_drr")
            real_load = getattr(module, name)
            pending = list(gates if gates is not None else [self.gate()])
            calls = []

            def load(request):
                index = len(calls)
                calls.append(request)
                pending[index].wait()
                if error:
                    raise ValueError("Synthetic foreground load failure")
                return real_load(request)

            with patch.object(module, name, side_effect=load):
                yield pending

        def assert_catalog_paused(self, gate):
            self.wait(gate.checkpoint.is_set)
            self.assertTrue(self.catalog._foreground_busy,
                "The active measurement loader must hold background inspection")
            self.assertFalse(self.catalog._cancel._resumed.is_set())
            self.assertFalse(gate.continued.is_set(),
                "Background inspection crossed its cooperative checkpoint during a foreground load")

        def test_folder_acceptance_prepares_before_any_picker_opens(self):
            for workflow in ("pl", "drr"):
                with self.subTest(workflow=workflow):
                    owner, root, path, _ = self.owner(workflow)
                    selected_status = owner.snapshot["status"]
                    self.catalog_ready(workflow, root)
                    self.assertIn(str(path), [row["id"] for row in self.catalog._catalog["files"]])
                    self.assertEqual(self.dialog_requests, [])
                    self.assertFalse(self.catalog._dialog_open)
                    self.assertEqual(owner._pending, {})
                    self.assertFalse(owner.snapshot["busy"])
                    self.assertEqual(owner.snapshot["status"], selected_status,
                        "An idle inventory completion must preserve the foreground status")
                    scans = list(self.catalog.scans)
                    owner.invoke("select-files")
                    self.assertEqual(self.dialog_requests, ["catalog"])
                    self.assertTrue(self.catalog._catalog_validated)
                    self.assertFalse(self.catalog.snapshot["busy"])
                    self.assertEqual(self.catalog.scans, scans,
                        "Opening a prepared catalog must reuse its inspected inventory")

        def test_switching_pl_and_drr_prepares_the_restored_folder(self):
            owner, pl_root, _, _ = self.owner()
            self.catalog_ready("pl", pl_root)
            owner.selectWorkflow("drr")
            drr_root = self.base / "another-drr-experiment"
            drr_path = self.csv(drr_root, "drr")
            owner.setSourcePath(str(drr_root))
            self.catalog_ready("drr", drr_root)
            self.assertIn(str(drr_path), [row["id"] for row in self.catalog._catalog["files"]])
            for workflow, root in (("pl", pl_root), ("drr", drr_root)):
                owner.selectWorkflow(workflow)
                self.catalog_ready(workflow, root)
                self.assertEqual(self.catalog.scans[-1][:2], (str(root.resolve()), workflow))
            self.assertEqual(self.dialog_requests, [])
            self.assertFalse(self.catalog._dialog_open)

        def test_cancelled_picker_keeps_idle_watching_and_refreshing(self):
            owner, root, _, _ = self.owner("drr")
            self.catalog_ready("drr", root)
            owner.invoke("select-files")
            self.catalog.invoke("cancel")
            owner._message("The current measurement status remains visible.")
            new_path = self.csv(root, "drr", "acquired-later")
            self.wait(lambda: self.catalog._catalog_validated and not self.catalog.snapshot["busy"]
                and str(new_path) in [row["id"] for row in self.catalog._catalog["files"]])
            self.assertFalse(self.catalog._dialog_open)
            self.assertEqual(self.dialog_requests, ["catalog"])
            self.assertEqual(owner.snapshot["status"], "The current measurement status remains visible.")

        def check_foreground_completion(self, workflow, *, error=False):
            owner, root, path, catalog_gate = self.owner(workflow, gated=True)
            self.wait(catalog_gate.entered.is_set)
            with self.blocked_loader(workflow, error=error) as (load_gate,):
                self.select(owner, path)
                self.wait(load_gate.entered.is_set)
                catalog_gate.release.set()
                self.assert_catalog_paused(catalog_gate)
                load_gate.release.set()
                self.load_idle(owner)
            expected_status = owner.snapshot["status"]
            if error:
                self.assertIn("Synthetic foreground load failure", expected_status)
                self.assertFalse(owner.snapshot["canExport"])
            else:
                self.assertEqual(owner._source, path, expected_status)
                self.assertTrue(owner.snapshot["canExport"], expected_status)
            self.catalog_ready(workflow, root)
            self.assertTrue(catalog_gate.continued.is_set())
            self.assertFalse(self.catalog._foreground_busy)
            self.assertEqual(owner.snapshot["status"], expected_status,
                "Background completion must not replace the load outcome")
            self.assertEqual(self.dialog_requests, [])

        def test_successful_pl_and_drr_loads_resume_idle_inspection(self):
            for workflow in ("pl", "drr"):
                with self.subTest(workflow=workflow):
                    self.check_foreground_completion(workflow)

        def test_failed_pl_and_drr_loads_resume_idle_inspection(self):
            for workflow in ("pl", "drr"):
                with self.subTest(workflow=workflow):
                    self.check_foreground_completion(workflow, error=True)

        def test_clear_resumes_catalog_after_obsolete_loader_finishes(self):
            for workflow in ("pl", "drr"):
                with self.subTest(workflow=workflow):
                    owner, root, path, catalog_gate = self.owner(workflow, gated=True)
                    self.wait(catalog_gate.entered.is_set)
                    with self.blocked_loader(workflow) as (load_gate,):
                        self.select(owner, path)
                        self.wait(load_gate.entered.is_set)
                        catalog_gate.release.set()
                        self.assert_catalog_paused(catalog_gate)
                        owner.invoke("clear-files")
                        self.assertTrue(owner._pending,
                            "The synthetic obsolete measurement is deliberately still running")
                        self.assert_catalog_paused(catalog_gate)
                        cleared_status = owner.snapshot["status"]
                        load_gate.release.set()
                        self.load_idle(owner)
                        self.catalog_ready(workflow, root)
                        self.assertTrue(catalog_gate.continued.is_set())
                        self.assertFalse(self.catalog._foreground_busy)
                    self.assertIsNone(owner._cube)
                    self.assertEqual(owner._sources, ())
                    self.assertEqual(owner.snapshot["status"], cleared_status)

        def test_obsolete_completion_does_not_resume_during_newer_load(self):
            owner, root, path, catalog_gate = self.owner("pl", gated=True)
            second = self.csv(root, "pl", "newer")
            self.wait(catalog_gate.entered.is_set)
            old_gate, new_gate = self.gate(), self.gate()
            with self.blocked_loader("pl", gates=(old_gate, new_gate)):
                self.select(owner, path)
                old_sequence = owner._sequence
                self.wait(old_gate.entered.is_set)
                self.select(owner, second)
                self.wait(new_gate.entered.is_set)
                catalog_gate.release.set()
                self.assert_catalog_paused(catalog_gate)
                old_gate.release.set()
                self.wait(lambda: old_sequence not in owner._pending)
                self.assertTrue(owner.snapshot["busy"])
                self.assert_catalog_paused(catalog_gate)
                new_gate.release.set()
                self.load_idle(owner)
            self.assertEqual(owner._source, second)
            self.assertTrue(owner.snapshot["canExport"], owner.snapshot["status"])
            self.catalog_ready("pl", root)
            self.assertTrue(catalog_gate.continued.is_set())
            self.assertFalse(self.catalog._foreground_busy)

        def test_switching_away_from_inflight_load_unblocks_idle_catalog(self):
            owner, root, path, catalog_gate = self.owner("pl", gated=True)
            self.csv(root, "drr", "reflectance")
            self.wait(catalog_gate.entered.is_set)
            with self.blocked_loader("pl") as (load_gate,):
                self.select(owner, path)
                self.wait(load_gate.entered.is_set)
                catalog_gate.release.set()
                self.assert_catalog_paused(catalog_gate)
                owner.selectWorkflow("drr")
                self.assertTrue(owner._pending)
                self.assertTrue(self.catalog._foreground_busy)
                self.assertFalse(owner.snapshot["busy"])
                load_gate.release.set()
                self.load_idle(owner)
                self.catalog_ready("drr", root)
                self.assertFalse(self.catalog._foreground_busy)
            self.assertEqual(owner.snapshot["workflow"], "drr")
            self.assertIsNone(owner._cube)
            self.assertEqual(self.dialog_requests, [])

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("QUICK_CATALOG_IDLE_INTEGRATION=" + json.dumps(dict(
        run=result.testsRun, failures=len(result.failures), errors=len(result.errors))))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
