"""Exercise native PL/DRR state and Slides with isolated settings and Quick.

The child process owns a QGuiApplication so an existing unittest QApplication
cannot conceal a Widgets dependency. All measurement and output files are
synthetic and live below temporary directories.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickApplicationBridgeTests(unittest.TestCase):
    def test_native_pl_bridge_in_widgets_free_process(self):
        with TemporaryDirectory(prefix="quick-bridge-") as temporary:
            environment = dict(
                os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                MPLBACKEND="agg", MPLCONFIGDIR=temporary, XDG_CONFIG_HOME=temporary,
                XDG_CACHE_HOME=temporary, PYTHONIOENCODING="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=150,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_APPLICATION_BRIDGE_TESTS=", result.stdout)


def _run_isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import importlib.abc
    import json
    import threading
    import time
    import traceback
    from unittest.mock import patch

    class NoWidgetImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {
                "PySide6.QtWidgets", "PySide6.QtQuickWidgets", "ui_qt.common", "ui_qt.main_window",
                "matplotlib.backends.backend_qt", "matplotlib.backends.backend_qtagg",
            }:
                raise AssertionError("Native bridge imported a Widgets dependency: " + fullname)

    sys.meta_path.insert(0, NoWidgetImports())
    from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPointF, QSettings, Qt, QUrl, qInstallMessageHandler
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication, QImage, QMouseEvent
    from PySide6.QtQuickControls2 import QQuickStyle
    from ui_qt.quick.application_bridge import QuickApplicationBridge

    initial_imports = set(sys.modules)
    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    if sys.platform == "win32":
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
        app.setFont(QFont("Segoe UI", 9))
    python_errors, qml_errors = [], []
    def python_error(kind, value, trace):
        python_errors.append("".join(traceback.format_exception(kind, value, trace)))
        traceback.print_exception(kind, value, trace)
    sys.excepthook = python_error
    def qt_message(kind, context, message):
        if any(marker in message for marker in ("ReferenceError", "TypeError", "Binding loop", "Cannot assign", "Unable to assign", "Error:")):
            qml_errors.append(str(message))
            print(message, file=sys.stderr)
    previous_handler = qInstallMessageHandler(qt_message)

    def drain_until(predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                app.processEvents()
                return
            time.sleep(.005)
        raise AssertionError("Timed out while waiting for native bridge state")

    def settle():
        for _ in range(4):
            app.processEvents()
            time.sleep(.005)

    class BridgeCases(unittest.TestCase):
        def setUp(self):
            python_start, qml_start = len(python_errors), len(qml_errors)
            self.addCleanup(lambda: self.assertEqual(python_errors[python_start:], [], "Uncaught Python exception from a Qt callback"))
            self.addCleanup(lambda: self.assertEqual(qml_errors[qml_start:], [], "QML runtime or binding error"))
            self.temporary = TemporaryDirectory(prefix="quick-bridge-case-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name)
            self.settings = QSettings(str(self.root / "settings.ini"), QSettings.Format.IniFormat)
            self.bridge = QuickApplicationBridge(settings=self.settings)
            self.addCleanup(self.bridge.close)

        def fields(self):
            return {field["key"]: field for group in self.bridge.snapshot["parameterGroups"]
                    for field in group["fields"]}

        def wait_for_plot(self):
            drain_until(lambda: self.bridge._canvas is not None
                        and self.bridge._canvas.frame_generation > 0
                        and not self.bridge._plot_timer.isActive()
                        and not self.bridge._canvas.backend._timer.isActive())
            self.assertEqual(self.bridge._canvas.last_error, "")

        def make_cube(self, offset=0):
            import numpy as np
            from core.loader import DataCube
            return DataCube(
                np.array([1.7, 2., 2.3, 2.6]), np.array([-2., 0., 3.]),
                np.array([[1., 2., 4., 3.], [3., 5., 7., 6.], [10., 20., 30., 40.]]) + offset,
                "Gate (V)", "Synthetic native PL", "PL (a.u.)",
            )

        def write_dat(self, name, *, offset=0):
            cube = self.make_cube(offset)
            rows = ["X\t" + "\t".join(map(str, cube.gate))]
            rows.extend("\t".join(map(str, [energy, *cube.Z[:, index]]))
                        for index, energy in enumerate(cube.energy))
            path = self.root / name
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            return path, cube

        def load_dat(self, name="measurement.dat", *, offset=0):
            path, cube = self.write_dat(name, offset=offset)
            self.bridge.openFile(QUrl.fromLocalFile(str(path)))
            drain_until(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending)
            self.assertEqual(self.bridge._source, path.resolve())
            self.wait_for_plot()
            return path, cube

        def write_drr_dat(self, name="reflectance.dat"):
            import numpy as np
            from core.loader import DataCube
            energy, gate = np.linspace(1.7, 2.7, 49), np.linspace(-2, 2, 9)
            values = .05 * (1 + .1 * gate[:, None]) * np.sin(energy[None, :] * 7)
            values += .02 * gate[:, None] * energy[None, :] ** 2
            cube = DataCube(energy, gate, values, "Gate (V)", "Synthetic native DRR", "ΔR/R")
            rows = ["X\t" + "\t".join(map(str, gate))]
            rows.extend("\t".join(map(str, [value, *values[:, index]])) for index, value in enumerate(energy))
            path = self.root / name
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            return path, cube

        def load_drr_dat(self):
            self.bridge.selectWorkflow("drr")
            path, cube = self.write_drr_dat()
            self.bridge.openFile(QUrl.fromLocalFile(str(path)))
            drain_until(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending)
            self.assertEqual(self.bridge._source, path.resolve(), self.bridge.snapshot["status"])
            self.wait_for_plot()
            self.assertTrue(self.bridge.snapshot["derivativeAvailable"], self.bridge.snapshot["status"])
            return path, cube

        def write_drr_csv(self, name="sample_TG_rep01.csv", offset=0.):
            import numpy as np
            wave, gate = np.linspace(630., 690., 41), np.linspace(-2., 2., 9)
            shape = np.linspace(-1., 1., len(wave))
            values = 130 + offset + 15 * gate[:, None] + 4 * shape[None, :] ** 2 + 3 * gate[:, None] * shape[None, :]
            path = self.root / name
            np.savetxt(path, np.column_stack((gate*.5, gate, values)), delimiter=",",
                header="Vbg,Vtg," + ",".join(map(str, wave)), comments="")
            return path

        def open_drr_background(self):
            self.bridge.selectWorkflow("drr")
            source = self.write_drr_csv()
            self.bridge.selectFiles([QUrl.fromLocalFile(str(source))])
            drain_until(lambda: not self.bridge._pending and not self.bridge.snapshot["busy"])
            self.wait_for_plot()
            self.assertTrue(self.bridge.snapshot["canConfigureBackground"])
            window, _ = self.make_native_window()
            self.click_item(window, self.qml_item(window, "drr_background_settings"))
            background = self.bridge.drrBackgroundBridge
            self.assertIsNotNone(background)
            drain_until(lambda: not background.snapshot["busy"])
            self.assertEqual(background.snapshot["error"], "")
            dialog = window.findChild(QObject, "drr_background_window")
            self.assertTrue(dialog.isVisible())
            return source, background, dialog

        def test_drr_native_background_applies_after_trial_and_fixed_rebinds(self):
            import numpy as np
            from core.data_io import load_drr_external_cube
            source, background, dialog = self.open_drr_background()
            external = self.write_drr_csv("external.csv", offset=40.)
            background.setChoice("kind", 6)
            self.bridge.acceptDrrBackgroundFiles(self.bridge.backgroundDialogToken, "background", [QUrl.fromLocalFile(str(external))])
            settle()
            self.click_item(dialog, self.qml_item(dialog, "drr_background_flag_fixed"))
            self.assertTrue(background.snapshot["flags"]["fixed"])
            self.assertEqual(background.draft.files, (external.name,))
            before = self.bridge._cube
            self.assertIsNone(self.bridge._drr_background_recipe)
            background.invoke("apply")
            self.assertIs(self.bridge._cube, before)
            drain_until(lambda: not background.snapshot["busy"] and not self.bridge._pending)
            self.assertEqual(background.snapshot["error"], "")
            self.assertFalse(dialog.isVisible())
            recipe = self.bridge._drr_background_recipe
            self.assertTrue(recipe.fixed)
            expected = load_drr_external_cube(str(self.root), (str(source),), (str(external),), baseline_which="all", y_axis=self.bridge._drr_request().y_axis)
            np.testing.assert_array_equal(self.bridge._cube.Z, expected.Z)
            self.assertEqual(tuple(item.measurement_file for item in recipe.assignments), (source.name,))
            self.assertTrue(self.bridge.snapshot["canExport"])
            replacement = self.write_drr_csv("sample_TG_rep02.csv", offset=7.)
            self.bridge.selectFiles([QUrl.fromLocalFile(str(replacement))])
            drain_until(lambda: not self.bridge._pending and not self.bridge.snapshot["busy"])
            self.assertTrue(self.bridge.snapshot["canExport"], self.bridge.snapshot["status"])
            self.assertEqual(tuple(item.measurement_file for item in self.bridge._drr_background_recipe.assignments), (replacement.name,))
            expected = load_drr_external_cube(str(self.root), (str(replacement),), (str(external),), baseline_which="all", y_axis=self.bridge._drr_request().y_axis)
            np.testing.assert_array_equal(self.bridge._cube.Z, expected.Z)

        def test_background_controls_share_one_draft_until_apply(self):
            _source, background, dialog = self.open_drr_background()
            external = self.write_drr_csv("external-draft.csv", offset=40.)
            background.setChoice("kind", 6)
            self.bridge.acceptDrrBackgroundFiles(self.bridge.backgroundDialogToken, "background", [QUrl.fromLocalFile(str(external))])
            before = self.bridge._drr_products
            token = self.bridge.backgroundDialogToken
            fields = {field["key"] for group in self.bridge.snapshot["processingGroups"] for field in group["fields"]}
            self.assertIn("background-settings", fields)
            self.assertFalse(fields & {"select-background", "clear-background", "backgroundMode", "backgroundFrame"})
            requests = []
            self.bridge.fileDialogRequested.connect(lambda *args: requests.append(args))
            self.bridge.invoke("select-background")
            self.assertEqual(requests, [])
            self.assertEqual(self.bridge.backgroundDialogToken, token)
            self.assertEqual(background.draft.files, (external.name,))
            self.assertEqual(self.bridge._drr_backgrounds, ())
            self.assertIs(self.bridge._drr_products, before)
            dialog.close()
            settle()
            self.assertIs(self.bridge._drr_products, before)
            self.bridge.invoke("background-settings")
            drain_until(lambda: not background.snapshot["busy"])
            self.assertEqual(background.draft.files, ())

        def test_drr_native_background_failure_and_cancel_retain_applied_result(self):
            _source, background, dialog = self.open_drr_background()
            external = self.write_drr_csv("external.csv", offset=40.)
            background.setChoice("kind", 6)
            self.bridge.acceptDrrBackgroundFiles(self.bridge.backgroundDialogToken, "background", [QUrl.fromLocalFile(str(external))])
            before, products = self.bridge._cube, self.bridge._drr_products
            settings_before = {key: self.settings.value(key) for key in self.settings.allKeys()}
            with patch("ui_qt.quick.drr_workflow.load_drr", side_effect=ValueError("synthetic trial failed")):
                background.invoke("apply")
                drain_until(lambda: not background.snapshot["busy"] and not self.bridge._pending)
            self.assertIn("synthetic trial failed", background.snapshot["error"])
            self.assertIs(self.bridge._cube, before)
            self.assertIs(self.bridge._drr_products, products)
            self.assertIsNone(self.bridge._drr_background_recipe)
            self.assertTrue(self.bridge.snapshot["canExport"])
            self.assertTrue(dialog.isVisible())
            started, release = threading.Event(), threading.Event()
            from ui_qt.quick.drr_workflow import load_drr
            def delayed_trial(*args, **kwargs):
                started.set()
                if not release.wait(15):
                    raise AssertionError("Trial test did not release worker")
                return load_drr(*args, **kwargs)
            with patch("ui_qt.quick.drr_workflow.load_drr", side_effect=delayed_trial):
                try:
                    background.invoke("apply")
                    drain_until(started.is_set)
                    background.invoke("cancel-task")
                    self.assertFalse(background.snapshot["busy"])
                    self.assertFalse(self.bridge.snapshot["busy"])
                    self.assertTrue(dialog.isVisible())
                finally:
                    release.set()
                    drain_until(lambda: not self.bridge._pending)
            self.assertIs(self.bridge._cube, before)
            self.assertIsNone(self.bridge._drr_background_recipe)
            self.assertEqual({key: self.settings.value(key) for key in self.settings.allKeys()}, settings_before)
            token = self.bridge.backgroundDialogToken
            dialog.close()
            settle()
            self.assertFalse(self.bridge._background_open)
            self.bridge.openDrrBackground()
            drain_until(lambda: not background.snapshot["busy"])
            self.bridge.acceptDrrBackgroundFiles(token, "background", [QUrl.fromLocalFile(str(external))])
            self.assertEqual(background.draft.files, ())
            self.bridge.selectWorkflow("pl")
            self.assertFalse(dialog.isVisible())

        def make_native_window(self):
            from PySide6.QtQml import QQmlApplicationEngine
            from ui_qt.quick.plot_surface import PlotSurface, register_plot_surface
            register_plot_surface()
            engine = QQmlApplicationEngine()
            engine.setInitialProperties({"appBridge": self.bridge})
            engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / "ui_qt/quick/qml/Application.qml")))
            self.assertTrue(engine.rootObjects(), "\n".join(qml_errors))
            window = engine.rootObjects()[0]
            window.setWidth(1320)
            window.setHeight(900)
            def dispose_engine():
                window.close()
                engine.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                app.processEvents()
            self.addCleanup(dispose_engine)
            surface = window.findChild(QObject, "app_plot_surface")
            self.assertIsInstance(surface, PlotSurface)
            frames = []
            window.frameSwapped.connect(lambda: frames.append(True))
            drain_until(lambda: bool(frames))
            return window, surface

        def qml_item(self, window, name):
            result = window.findChild(QObject, name)
            if result is None:
                def visit(parent):
                    if parent.objectName() == name:
                        return parent
                    for child in parent.childItems():
                        found = visit(child)
                        if found is not None:
                            return found
                    return None
                result = visit(window.contentItem())
            self.assertIsNotNone(result, name)
            return result

        def click_item(self, window, item):
            point = item.mapToScene(item.boundingRect().center())
            for kind, buttons in ((QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
                                  (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton)):
                event = QMouseEvent(kind, point, point, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)
                QCoreApplication.sendEvent(window, event)
            settle()

        def test_00_empty_shell_import_and_state_are_widgets_free_and_lazy(self):
            for name in ("PySide6.QtWidgets", "PySide6.QtQuickWidgets", "ui_qt.main_window",
                         "ui_qt.common", "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_qt",
                         "matplotlib", "numpy", "core.application_models"):
                self.assertNotIn(name, initial_imports)
            self.assertIs(type(app), QGuiApplication)
            self.assertIsNone(self.bridge.plotBackend)
            self.assertFalse(self.bridge.snapshot["inspectionVisible"])
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.assertTrue(Path(self.settings.fileName()).is_relative_to(self.root))
            groups = self.bridge.snapshot["parameterGroups"]
            self.assertEqual([group["key"] for group in groups[:4]], ["data", "display", "axes", "spectrum"])
            self.assertTrue(all(group.get("advanced") for group in groups[4:]))
            self.assertEqual(self.bridge.snapshot["parameterGroups"][0]["key"], "data")
            self.bridge.selectWorkflow("drr")
            self.assertNotIn("not available", self.bridge.snapshot["workspaceMessage"])
            self.assertIn("DRR", self.bridge.snapshot["workspaceMessage"])
            self.assertEqual(self.bridge.snapshot["parameterGroups"], [])
            self.assertFalse(self.bridge.snapshot["inspectionVisible"])
            self.assertTrue(self.bridge.parameterUrl.toLocalFile().endswith("DrrParameters.qml"))
            self.assertIsNotNone(self.bridge.drrDisplayBridge)
            self.bridge.selectWorkflow("tools")
            self.assertEqual(self.bridge.snapshot["workspaceMessage"], "")
            self.assertTrue(self.bridge.workspaceUrl.toLocalFile().endswith("ToolsWorkspace.qml"))

        def capture_native(self, window, name):
            destination = os.environ.get("DPTK_QUICK_ROOT_CAPTURE_DIR")
            if destination:
                path = Path(destination)
                path.mkdir(parents=True, exist_ok=True)
                settle()
                self.assertTrue(window.grabWindow().save(str(path / name)))

        def test_file_picker_can_be_cancelled_during_opening_without_widgets(self):
            from PySide6.QtCore import QMetaObject
            window, _surface = self.make_native_window()
            dialog = window.findChild(QObject, "app_file_dialog")
            for _ in range(3):
                self.bridge.invoke("select-files")
                QMetaObject.invokeMethod(dialog, "reject")
                settle()
            self.assertEqual(self.bridge.snapshot["sourceSummary"], "No measurement selected")
            self.assertNotIn("PySide6.QtWidgets", sys.modules)

        def test_file_and_folder_pickers_follow_changed_experiment_directory(self):
            from PySide6.QtCore import QMetaObject
            window, _surface = self.make_native_window()
            file_dialog = window.findChild(QObject, "app_file_dialog")
            folder_dialog = window.findChild(QObject, "app_folder_dialog")
            for name in ("experiment A", "experiment B # %"):
                folder = self.root / name
                folder.mkdir()
                self.bridge.setSourcePath(str(folder))
                self.bridge.fileDialogRequested.emit("catalog", True)
                settle()
                self.assertEqual(Path(file_dialog.property("currentFolder").toLocalFile()), folder)
                QMetaObject.invokeMethod(file_dialog, "reject")
                self.click_item(window, self.qml_item(window, "app_browse_folder"))
                self.assertEqual(Path(folder_dialog.property("currentFolder").toLocalFile()), folder)
                self.assertTrue(folder_dialog.property("visible"))
                QMetaObject.invokeMethod(folder_dialog, "reject")
                settle()

        def test_background_file_picker_resets_saved_filename_to_current_folder(self):
            from PySide6.QtCore import QMetaObject
            _source, background, dialog = self.open_drr_background()
            picker = dialog.findChild(QObject, "drr_background_file_dialog")
            other = self.root / "export elsewhere"
            other.mkdir()
            self.bridge.backgroundSaveDialogRequested.emit("composite-save", str(other / "draft.csv"))
            settle()
            QMetaObject.invokeMethod(picker, "reject")
            self.bridge.backgroundFileDialogRequested.emit("background", True)
            settle()
            self.assertEqual(Path(picker.property("currentFolder").toLocalFile()), self.root)
            self.assertEqual(background.draft.files, ())
            QMetaObject.invokeMethod(picker, "reject")
            dialog.close()
            settle()

        def test_power_native_selection_plot_export_and_compare_references(self):
            import numpy as np
            self.bridge.selectWorkflow("cmp")
            compare = self.bridge.workflowBridge
            compare.setParameter("inK", 12.5)
            self.bridge.selectWorkflow("power")
            child = self.bridge.workflowBridge
            self.assertEqual(child._angle_refs["in_k"], 12.5)
            self.assertTrue(self.bridge.parameterUrl.toLocalFile().endswith("PowerParameters.qml"))
            energy, powers = np.linspace(1.5, 1.7, 161), np.geomspace(1., 100., 12)
            signal = 10 + powers[:, None] * 10 / (1 + (2 * (energy[None, :] - 1.6) / .008) ** 2)
            path = self.root / "sample_KK.csv"
            np.savetxt(path, np.column_stack((powers, np.arange(12), signal)), delimiter=",",
                header="Power_uW,stage_pos," + ",".join(map(str, energy)), comments="")
            window, default_surface = self.make_native_window()
            self.bridge.setSourcePath(str(self.root))
            self.bridge.acceptFiles("power", "single", [QUrl.fromLocalFile(str(path))])
            drain_until(lambda: not child._pending and not child.snapshot["catalogBusy"])
            self.assertEqual(child.snapshot["error"], "", child.snapshot["status"])
            self.assertEqual(child.snapshot["stage"], "selection")
            child.invoke("open-sweep")
            drain_until(lambda: not child._pending and not child.snapshot["busy"])
            surface = self.qml_item(window, "power_plot_surface")
            drain_until(lambda: child._canvas is not None and surface.property("presentedGeneration") > 0
                and surface.property("presentedGeneration") == child._canvas.frame_generation and not child._plot_timer.isActive())
            self.assertIsNone(default_surface.property("backend"))
            self.assertTrue(self.bridge.snapshot["canExport"], child.snapshot["status"])
            self.assertEqual(self.qml_item(window, "app_export").property("text"), "Save power package")
            self.capture_native(window, "power-root-native.png")
            self.bridge.invoke("pan")
            self.assertEqual(self.bridge.snapshot["plotTool"], "pan")
            self.bridge.invoke("pan")
            self.bridge.invoke("export")
            drain_until(lambda: not child._pending and not child.snapshot["busy"], timeout=45)
            self.assertTrue(child.snapshot["resultPaths"], child.snapshot["status"])
            self.assertTrue(all(Path(item).is_file() for item in child.snapshot["resultPaths"]))
            self.bridge.selectWorkflow("cmp")
            compare.setParameter("inK", 17.)
            self.bridge.selectWorkflow("power")
            self.assertIs(self.bridge.workflowBridge, child)
            self.assertEqual(child._angle_refs["in_k"], 17.)
            drain_until(lambda: not child._pending)
            self.assertTrue(self.bridge.snapshot["canExport"], child.snapshot["status"])

        def test_mcd_native_root_file_folder_roles_and_preserved_workspace(self):
            import csv
            import numpy as np
            from PySide6.QtCore import QMetaObject
            energy = np.linspace(1.5, 1.9, 201)
            source = self.root / "synthetic_MCD.csv"
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["B_T", "angle_deg", *list(1239.841984 / energy)])
                for field in np.r_[np.linspace(-2, 2, 21), np.linspace(1.8, -2, 20)]:
                    for angle, sign, gain in ((90, 1, 1.), (0, -1, 1.08)):
                        signal = gain * (10000 + 1800 * np.exp(-((energy - 1.7 - sign * .001 * field) / .009) ** 2))
                        writer.writerow([field, angle, *signal])
            self.bridge.selectWorkflow("mcd")
            child = self.bridge.workflowBridge
            window, default_surface = self.make_native_window()
            self.bridge.acceptFiles("mcd", "sources", [QUrl.fromLocalFile(str(source))])
            drain_until(lambda: not child._window_timer.isActive() and not child._load_timer.isActive()
                and not child.snapshot["busy"] and not child.snapshot["analysisBusy"] and not child.snapshot["catalogBusy"], timeout=45)
            self.assertEqual(child.snapshot["error"], "", child.snapshot["status"])
            surface = self.qml_item(window, "mcd_plot_surface")
            drain_until(lambda: child._canvas is not None and surface.property("presentedGeneration") > 0
                and surface.property("presentedGeneration") == child._canvas.frame_generation and not child._plot_timer.isActive())
            self.assertIsNone(default_surface.property("backend"))
            self.assertTrue(self.bridge.snapshot["canExport"], child.snapshot["status"])
            self.assertEqual(self.qml_item(window, "app_export").property("text"), "Save MCD package")
            self.capture_native(window, "mcd-root-native.png")
            self.bridge.invoke("choose-output-folder")
            self.assertEqual(window.property("folderDialogPurpose"), "output-folder")
            self.assertEqual(window.property("folderDialogWorkflow"), "mcd")
            QMetaObject.invokeMethod(window.findChild(QObject, "app_folder_dialog"), "reject")
            output = self.root / "result-package"
            output.mkdir()
            self.bridge.acceptFolder("mcd", "output-folder", QUrl.fromLocalFile(str(output)))
            self.assertEqual(child._output, str(output))
            self.assertEqual(self.bridge.snapshot["sourcePath"], str(self.root))
            self.bridge.invoke("export")
            drain_until(lambda: not child.snapshot["busy"], timeout=60)
            self.assertTrue(child.snapshot["resultPaths"], child.snapshot["status"])
            self.assertTrue(all(Path(item).is_file() for item in child.snapshot["resultPaths"]))
            self.bridge.selectWorkflow("pl")
            before = self.bridge.snapshot["sourcePath"]
            self.bridge.acceptFolder("mcd", "source-folder", QUrl.fromLocalFile(str(output)))
            self.assertEqual(self.bridge.snapshot["sourcePath"], before)
            self.bridge.selectWorkflow("mcd")
            self.assertIs(self.bridge.workflowBridge, child)
            self.assertEqual(child.snapshot["source"], str(source))
            self.assertTrue(self.bridge.snapshot["canExport"])
            products = child._products
            self.bridge.selectWorkflow("mcd_peak_shift")
            peak = self.bridge.workflowBridge
            self.assertIs(peak._products, products, "MCD pages share accepted processing without reloading")
            drain_until(lambda: not peak.snapshot["busy"] and not peak.snapshot["analysisBusy"] and not peak._plot_timer.isActive(), timeout=45)
            self.assertEqual(peak.snapshot["error"], "", peak.snapshot["status"])
            surface = self.qml_item(window, "mcd_peak_plot_surface")
            drain_until(lambda: surface.property("presentedGeneration") > 0 and surface.property("presentedGeneration") == peak._canvas.frame_generation)
            self.assertIsNone(default_surface.property("backend"))
            self.assertTrue(self.bridge.snapshot["canExport"], peak.snapshot["status"])
            self.assertEqual(self.qml_item(window, "app_export").property("text"), "Save peak CSV + metadata")
            self.capture_native(window, "mcd-peak-root-native.png")
            self.bridge.acceptFolder("mcd_peak_shift", "output-folder", QUrl.fromLocalFile(str(output)))
            self.bridge.invoke("export")
            drain_until(lambda: not peak.snapshot["busy"], timeout=45)
            self.assertTrue(peak.snapshot["resultPaths"], peak.snapshot["status"])
            self.assertTrue(all(Path(item).is_file() for item in peak.snapshot["resultPaths"]))
            self.bridge.setParameter("max_delta_b", .11)
            drain_until(lambda: not peak._load_timer.isActive() and not peak.snapshot["busy"] and not peak.snapshot["analysisBusy"], timeout=45)
            self.assertTrue(peak._products is not products, "A changed processing setting must publish a fresh snapshot")
            self.assertIs(child._products, peak._products)
            self.bridge.selectWorkflow("mcd")
            self.assertIs(self.bridge.workflowBridge, child)
            self.assertAlmostEqual(child._values["max_delta_b"], .11)

        def test_help_updater_native_dialog_and_preference_routes(self):
            from types import SimpleNamespace
            from app_version import __version__
            from core.update_checker import CheckResult, ReleaseInfo, parse_version
            window, _ = self.make_native_window()
            self.bridge.invoke("about")
            settle()
            update = self.bridge.updateBridge
            dialog = window.findChild(QObject, "quick_update_dialog")
            self.assertIsNotNone(dialog)
            self.assertTrue(dialog.property("visible"))
            self.assertIn(__version__, update.snapshot["message"])
            self.bridge.invoke("theme-high-contrast")
            settle()
            dialog_colors = dialog.property("colors")
            if hasattr(dialog_colors, "toVariant"):
                dialog_colors = dialog_colors.toVariant()
            self.assertEqual(dialog_colors["text"], self.bridge.colors["text"])
            update.invoke("close-dialog")
            settle()
            self.assertFalse(dialog.property("visible"))
            self.bridge.selectWorkflow("cmp")
            self.bridge.setParameter("autoUpdateCheck", False)
            self.assertFalse(self.bridge.snapshot["autoUpdateCheck"])
            self.assertFalse(self.settings.value("updates/check_automatically", type=bool))
            release = ReleaseInfo("v99.0.0", (99, 0, 0), False, False,
                "https://github.com/ylylyl98/PySide6_Data_plot/releases/tag/v99.0.0", "Synthetic release notes", {})
            calls = []
            update.services = SimpleNamespace(check=lambda version: calls.append(version) or
                CheckResult("update_available", parse_version(version), (99, 0, 0), release,
                    installer_url="https://github.com/ylylyl98/PySide6_Data_plot/releases/download/v99.0.0/DPTK-99.0.0.exe",
                    sums_url="https://github.com/ylylyl98/PySide6_Data_plot/releases/download/v99.0.0/SHA256SUMS.txt",
                    message="Synthetic update"))
            self.bridge.invoke("check-updates")
            drain_until(lambda: not update.snapshot["busy"])
            self.assertEqual(calls, [__version__])
            self.assertTrue(dialog.property("visible"))
            self.assertEqual(update.snapshot["latestVersion"], "99.0.0")
            self.assertEqual(update.snapshot["releaseNotes"], "Synthetic release notes")

        def test_data_maintenance_is_bound_to_current_export_and_archives_only_saved_csv(self):
            source = self.write_drr_csv("sample_PL.csv")
            self.bridge.setParameter("pl.autoNext", False)
            self.bridge.openFile(QUrl.fromLocalFile(str(source)))
            drain_until(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending)
            self.wait_for_plot()
            self.assertFalse(self.bridge.snapshot["canMoveExportedSources"])
            self.bridge.invoke("export")
            drain_until(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending)
            drain_until(lambda: not self.bridge.snapshot["maintenanceBusy"])
            self.assertTrue(self.bridge.snapshot["canMoveExportedSources"], self.bridge.snapshot["status"])
            self.assertTrue(source.is_file())
            original_bytes = source.read_bytes()
            source.write_bytes(original_bytes + b"\n")
            self.bridge.invoke("move-exported-sources")
            drain_until(lambda: not self.bridge.snapshot["maintenanceBusy"])
            self.assertTrue(source.is_file())
            # A concurrent catalog refresh can replace the transient status.
            # The refusal must remain recorded and the source must stay put.
            self.assertIn("changed after saving", self.bridge.snapshot["logText"])
            source.write_bytes(original_bytes)
            self.bridge.openFile(QUrl.fromLocalFile(str(source)))
            drain_until(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending)
            self.bridge.invoke("export")
            drain_until(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending)
            drain_until(lambda: not self.bridge.snapshot["maintenanceBusy"])
            self.assertTrue(self.bridge.snapshot["canMoveExportedSources"], self.bridge.snapshot["status"])
            self.bridge.invoke("move-exported-sources")
            drain_until(lambda: not self.bridge.snapshot["maintenanceBusy"])
            self.assertFalse(source.exists())
            archive = self.root / "Initial data after processing" / source.name
            self.assertEqual(archive.read_bytes(), original_bytes)
            self.assertFalse(self.bridge.snapshot["canMoveExportedSources"])

        def test_organizer_menu_launches_native_entry_with_current_folder_and_isolated_settings(self):
            self.bridge.setSourcePath(str(self.root))
            with patch("PySide6.QtCore.QProcess.startDetached", return_value=(True, 4321)) as start:
                self.bridge.invoke("mcd-organizer")
            executable, arguments, working = start.call_args.args
            self.assertTrue(any(argument.endswith("run_mcd_organizer_quick.py") for argument in arguments))
            self.assertIn(str(self.root), arguments)
            child_settings = Path(arguments[arguments.index("--settings") + 1])
            self.assertEqual(child_settings, self.root / "mcd-organizer" / "quick.ini")
            self.assertTrue(child_settings.is_file())
            self.assertIn("Opened MCD Organizer", self.bridge.snapshot["status"])
            self.bridge.selectWorkflow("tools")
            window, _ = self.make_native_window()
            self.assertIsNotNone(self.qml_item(window, "tools_open_mcd_organizer"))

        def test_recent_folders_drop_and_restore_preserve_existing_settings(self):
            source, cube = self.write_dat("drop-test.dat")
            self.settings.setValue("unrelated/sentinel", "kept")
            self.assertTrue(self.bridge.dropSources([QUrl.fromLocalFile(str(source))]))
            drain_until(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending)
            self.wait_for_plot()
            self.assertEqual(self.bridge._source, source)
            self.assertEqual(self.bridge.snapshot["recentFolders"], [str(self.root)])
            self.assertEqual(self.settings.value("data/last_folder"), str(self.root))
            self.assertEqual(self.settings.value("unrelated/sentinel"), "kept")
            other = self.root / "another"
            other.mkdir()
            self.assertFalse(self.bridge.dropSources([QUrl.fromLocalFile(str(source)), QUrl.fromLocalFile(str(other))]))
            self.assertEqual(self.bridge._folder, str(self.root))
            self.assertFalse(self.bridge.dropSources([QUrl("https://example.invalid/sample.dat")]))
            self.assertTrue(self.bridge.dropSources([QUrl.fromLocalFile(str(other))]))
            self.assertEqual(self.bridge.snapshot["recentFolders"], [str(other), str(self.root)])
            self.settings.setValue("ui/appearance", "dark")
            restored = QuickApplicationBridge(settings=self.settings)
            self.addCleanup(restored.close)
            self.assertEqual(restored.snapshot["appearance"], "dark")
            restored.restoreSourceFolder()
            self.assertEqual(restored._folder, str(other))
            self.assertEqual(restored.snapshot["recentFolders"], [str(other), str(self.root)])

        def test_shg_native_source_roles_plot_export_and_selection_survive_switching(self):
            import numpy as np
            from core.shg_fit import evaluate_shg_angular_model
            self.bridge.selectWorkflow("shg")
            child = self.bridge.workflowBridge
            requests = []
            self.bridge.fileDialogRequested.connect(lambda *args: requests.append(args))
            self.bridge.invoke("choose-settings")
            self.bridge.invoke("choose-backgroundA")
            self.assertEqual(requests, [("settings", False), ("backgroundA", False)])
            self.assertEqual(self.bridge.snapshot["fileNameFilters"], ["SHG measurements (*.csv)"])
            wave, angles = np.linspace(507., 523., 161), np.linspace(0., 180., 19)
            peak = np.exp(-.5 * ((wave - 515.) / .35) ** 2) / (.35 * np.sqrt(2 * np.pi))
            area = evaluate_shg_angular_model(angles, 120., 900., 17.5)
            spectra = 800. + 1.8 * (wave[None, :] - 515.) + area[:, None] * peak[None, :]
            path = self.root / "angular-sweep.csv"
            np.savetxt(path, np.column_stack((angles, angles, np.zeros_like(angles), np.ones_like(angles), np.ones_like(angles), spectra)),
                delimiter=",", header="target_value,measured_value,position_error,move_ok,acquisition_ok," +
                    ",".join(f"{value:.8f}" for value in wave), comments="")
            window, default_surface = self.make_native_window()
            self.bridge.openFile(QUrl.fromLocalFile(str(path)))
            drain_until(lambda: not child._pending and not child.snapshot["busy"] and not child._load_timer.isActive())
            self.assertEqual(child.snapshot["error"], "", child.snapshot["status"])
            surface = self.qml_item(window, "shg_plot_surface")
            drain_until(lambda: child._canvas is not None and surface.property("presentedGeneration") > 0
                        and surface.property("presentedGeneration") == child._canvas.frame_generation
                        and not child._plot_timer.isActive())
            self.assertIsNone(default_surface.property("backend"))
            self.assertIs(self.bridge.plotBackend, child.plotBackend)
            self.assertTrue(self.qml_item(window, "quick_shg_workspace").property("visible"))
            self.assertEqual(self.qml_item(window, "app_export").property("text"), "Export analysis")
            self.assertTrue(self.bridge.snapshot["canExport"])
            self.bridge.invoke("pan")
            self.assertEqual(self.bridge.snapshot["plotTool"], "pan")
            self.bridge.invoke("pan")
            self.bridge.setParameter("showTable", True)
            self.bridge.selectWorkflow("pl")
            self.bridge.selectWorkflow("shg")
            self.assertIs(self.bridge.workflowBridge, child)
            self.assertEqual(child.snapshot["sources"]["single"], str(path.resolve()))
            self.assertTrue(child.snapshot["showTable"])
            self.bridge.invoke("export")
            self.assertTrue(self.bridge.snapshot["busy"])
            drain_until(lambda: not child._pending and not child.snapshot["busy"], timeout=45)
            self.assertEqual(child.snapshot["error"], "")
            self.assertTrue(child.snapshot["resultPaths"])
            self.assertTrue(all(Path(output).is_file() and Path(output).is_relative_to(self.root)
                                for output in child.snapshot["resultPaths"]))

        def test_compare_file_dialog_is_bound_to_the_requesting_workflow_and_channel(self):
            self.bridge.selectWorkflow("cmp")
            child = self.bridge.workflowBridge
            requests = []
            self.bridge.fileDialogRequested.connect(lambda *args: requests.append(args))
            self.bridge.invoke("select-files")
            self.bridge.invoke("choose-channel:KKp")
            self.assertEqual(requests, [("sources", True), ("channel:KKp", False)])
            urls = [QUrl.fromLocalFile(str(self.root / "sample.csv"))]
            with patch.object(child, "openFiles") as receive:
                self.bridge.acceptFiles("cmp", "channel:KKp", urls)
                receive.assert_called_once_with("channel:KKp", urls)
                self.bridge.selectWorkflow("pl")
                self.bridge.acceptFiles("cmp", "channel:KKp", urls)
                self.assertEqual(receive.call_count, 1)
            self.assertIsNone(self.bridge._source)

        def test_appearance_keeps_system_preference_and_updates_native_plot_and_children(self):
            from PySide6.QtGui import QColor
            self.load_dat()
            window, _ = self.make_native_window()
            self.bridge.invoke("theme-high-contrast")
            self.wait_for_plot()
            self.assertEqual(self.bridge.snapshot["appearance"], "high-contrast")
            self.assertEqual(self.settings.value("quick/appearance"), "high-contrast")
            self.assertEqual(QColor(self.bridge.colors["text"]).name(), "#ffffff")
            self.assertEqual(QColor(self.bridge.colors["canvas"]).name(), "#000000")
            self.assertEqual(tuple(self.bridge._canvas.figure.get_facecolor()), (0., 0., 0., 1.))
            self.bridge.selectWorkflow("cmp")
            child = self.bridge.workflowBridge
            self.assertEqual(child._palette["text"], self.bridge.colors["text"])
            self.bridge.invoke("theme-dark")
            self.assertEqual(child._palette["inset"], self.bridge.colors["inset"])
            from types import SimpleNamespace
            from ui_qt.quick.appearance import quick_colors
            mode = [Qt.ColorScheme.Light]
            # The offscreen plugin cannot change the OS theme. Deliver the
            # real Qt signal while providing its corresponding system value.
            hints = SimpleNamespace(colorScheme=lambda: mode[0])
            with patch("ui_qt.quick.appearance.quick_colors", side_effect=lambda preference: quick_colors(preference, hints=hints)):
                self.bridge.invoke("theme-system")
                light = self.bridge.colors["text"]
                mode[0] = Qt.ColorScheme.Dark
                app.styleHints().colorSchemeChanged.emit(mode[0])
                settle()
                self.assertNotEqual(self.bridge.colors["text"], light)
                self.assertEqual(child._palette["text"], self.bridge.colors["text"])
                self.assertEqual(self.bridge.snapshot["appearance"], "system")
                self.assertEqual(self.settings.value("quick/appearance"), "system")
                self.bridge.selectWorkflow("pl")
                self.wait_for_plot()
                self.assertTrue(self.qml_item(window, "app_plot_surface").property("visible"))

        def test_peak_analysis_handoff_overlay_rebuild_and_source_identity(self):
            from core.peak_plotting import overlay_payload
            from core.peak_workspace import load_workspace
            source, _ = self.load_dat()
            window, _ = self.make_native_window()
            self.assertTrue(self.bridge.snapshot["canAnalyzePeaks"])
            self.assertTrue(self.qml_item(window, "app_peak_analysis").property("enabled"))
            with patch("ui_qt.quick.peak_analysis_launcher.QProcess.startDetached", return_value=(True, 123)) as start:
                self.bridge.invoke("peak-analysis")
                launcher = self.bridge._peak_launcher
                drain_until(lambda: not launcher.snapshot["busy"])
                self.assertEqual(start.call_count, 1)
                arguments = start.call_args.args[1]
                snapshot = Path(arguments[arguments.index("--snapshot") + 1])
                self.assertTrue(snapshot.is_relative_to(self.root))
                datasets, active = load_workspace(snapshot)
                self.assertEqual(len(datasets), 1)
                dataset = datasets[0]
                self.assertIn(str(source), json.loads(dataset["source"])[2])
                self.assertEqual(dataset["key"], active)
                self.assertEqual(dataset["view"]["xlim"], list(self.bridge._maps[0][0].get_xlim()))
                branch = dict(id="test-peak", name="P1", color=0, enabled=True, polarity="peak")
                dataset["branches"] = [branch]
                dataset["result"] = dict(dataset_key=dataset["key"], branches=[branch], points=[
                    dict(branch_id="test-peak", row_index=0, energy=2., y=-2., status="accepted", segment=0)])
                packet = overlay_payload(dataset)
                launcher.accept_overlay(packet)
                drain_until(lambda: not launcher.snapshot["busy"] and bool(launcher._artists))
                self.assertTrue(self.bridge.snapshot["hasPeakOverlay"])
                self.assertIs(launcher._artists[0].axes, self.bridge._maps[0][0])
                self.bridge.setParameter("sideBySide", True)
                self.wait_for_plot()
                drain_until(lambda: not launcher.snapshot["busy"] and len(launcher._artists) == 2)
                self.assertEqual({artist.axes for artist in launcher._artists}, {entry[0] for entry in self.bridge._maps})
                self.bridge.invoke("toggle-peak-overlay")
                self.assertFalse(launcher._artists)
                self.bridge.invoke("toggle-peak-overlay")
                self.assertEqual(len(launcher._artists), 2)
                self.bridge.selectWorkflow("mcd")
                settle()
                self.assertFalse(self.bridge.snapshot["canAnalyzePeaks"])
                self.assertFalse(launcher._artists)
                self.bridge.selectWorkflow("pl")
                self.wait_for_plot()
                drain_until(lambda: not launcher.snapshot["busy"] and len(launcher._artists) == 2)
                self.load_dat("other.dat", offset=100)
                drain_until(lambda: not launcher.snapshot["busy"])
                launcher.accept_overlay(packet)
                drain_until(lambda: not launcher.snapshot["busy"])
                self.assertFalse(launcher._artists)
                self.assertFalse(self.bridge.snapshot["hasPeakOverlay"])

        def test_drr_analysis_receives_original_source_and_tools_restores_controls(self):
            import numpy as np
            from core.drr_workspace_session import load_session
            self.load_drr_dat()
            self.bridge.setParameter("drrView", "second")
            self.wait_for_plot()
            products = self.bridge._drr_products
            window, _ = self.make_native_window()
            self.assertTrue(self.bridge.snapshot["canAddDrrAnalysis"])
            with patch("ui_qt.quick.drr_analysis_launcher.QProcess.startDetached", return_value=(True, 123)) as start:
                self.bridge.invoke("drr-analysis-current")
                launcher = self.bridge._drr_analysis_launcher
                drain_until(lambda: not launcher.busy)
                self.assertEqual(start.call_count, 1)
                args = start.call_args.args[1]
                self.assertTrue(args[0].endswith("run_drr_analysis_quick.py"))
                path = Path(args[args.index("--snapshot") + 1])
                self.assertTrue(path.is_relative_to(self.root))
                datasets, views, active = load_session(path)
                self.assertEqual(len(datasets), 1)
                dataset = datasets[0]
                self.assertEqual(dataset.key, active)
                np.testing.assert_array_equal(dataset.cube.Z, products.source.Z)
                self.assertFalse(np.allclose(dataset.cube.Z, products.second.Z))
                self.assertEqual(dataset.provenance["measurement_files"], list(products.request.files))
                self.assertEqual(dataset.provenance["background_mode"], "Imported map")
                from ui_qt.quick.drr_analysis_bridge import DrrAnalysisBridge
                child = DrrAnalysisBridge(settings=QSettings(str(self.root / "analysis.ini"), QSettings.IniFormat),
                    session_path=self.root / "analysis.npz", start_folder=str(self.root))
                self.addCleanup(child.close)
                child.receiveSnapshot(str(path))
                drain_until(lambda: not child.snapshot["busy"] and bool(child.datasets))
                np.testing.assert_array_equal(child.datasets[child.active_key].cube.Z, products.source.Z)
                self.bridge.selectWorkflow("tools")
                settle()
                self.assertTrue(self.qml_item(window, "quick_tools_workspace").property("visible"))
                self.assertFalse(self.bridge.snapshot["controlsVisible"])
                self.bridge.invoke("drr-analysis")
                self.assertEqual(start.call_count, 2)
                self.assertNotIn("--snapshot", start.call_args.args[1])
                self.assertIn(str(launcher.session_path), start.call_args.args[1])
                self.bridge.invoke("toggle-log")
                self.bridge.invoke("clear-log")
                self.assertTrue(self.bridge.snapshot["logVisible"])
                self.assertEqual(self.bridge.snapshot["logText"], "")
                self.capture_native(window, "tools-root-native.png")
                self.bridge.selectWorkflow("drr")
                self.wait_for_plot()
                self.assertTrue(self.bridge.snapshot["controlsVisible"])
                self.assertIs(self.bridge._drr_products, products)

        def test_compare_native_pages_keep_pl_state_and_route_plot_controls_and_export(self):
            import numpy as np
            source, original = self.load_dat()
            original_cube = self.bridge._cube
            window, default_surface = self.make_native_window()
            folder = self.root / "channels"
            folder.mkdir()
            paths = []
            energy, gate = np.linspace(1.7, 2.7, 21), np.linspace(-2, 2, 7)
            for index, name in enumerate(("KK", "KKp")):
                path = folder / f"sample_PL_{name}.csv"
                values = 100 + index * 20 + (gate[:, None] + 3) * 7 + 50 * np.exp(-((energy[None, :] - 2.15) / .1) ** 2)
                np.savetxt(path, np.column_stack((gate, gate * .3 + 1, np.zeros_like(gate), values)),
                    delimiter=",", header="Vbg,Vtg,Vbias," + ",".join(f"{1240 / value:.12g}" for value in energy), comments="")
                paths.append(QUrl.fromLocalFile(str(path)))
            self.bridge.selectWorkflow("cmp")
            child = self.bridge.workflowBridge
            self.bridge.setSourcePath(str(folder))
            self.bridge.acceptFiles("cmp", "sources", paths)
            self.assertTrue(self.bridge.snapshot["busy"])
            self.assertFalse(self.bridge.snapshot["canExport"])
            drain_until(lambda: not child.snapshot["busy"] and not child._pending)
            self.assertEqual(child.snapshot["error"], "")
            surface = self.qml_item(window, "compare_plot_surface")
            drain_until(lambda: child._canvas is not None and surface.property("presentedGeneration") > 0
                        and surface.property("presentedGeneration") == child._canvas.frame_generation
                        and not child._plot_timer.isActive())
            self.assertIsNone(default_surface.property("backend"), "Only the active surface may resize the child canvas")
            self.assertIs(surface.property("backend"), self.bridge.plotBackend)
            self.assertTrue(self.qml_item(window, "quick_compare_workspace").property("visible"))
            self.assertTrue(self.bridge.snapshot["canExport"])
            self.assertTrue(self.bridge.snapshot["controlsVisible"])
            self.assertFalse(self.bridge.snapshot["railVisible"])
            self.assertEqual(self.bridge.snapshot["sourcePath"], str(folder.resolve()))
            self.bridge.setParameter("view", "vp")
            drain_until(lambda: self.bridge.snapshot["railVisible"])
            self.bridge.invoke("toggle-spectrum")
            self.assertFalse(child.snapshot["spectrumVisible"])
            self.bridge.selectWorkflow("pl")
            self.wait_for_plot()
            self.assertIs(self.bridge._cube, original_cube)
            self.assertEqual(self.bridge._source, source)
            np.testing.assert_allclose(self.bridge._cube.Z, original.Z)
            self.assertIs(self.bridge.plotBackend, self.bridge._canvas.backend)
            self.bridge.selectWorkflow("cmp")
            self.assertIs(self.bridge.workflowBridge, child)
            self.assertEqual(child.snapshot["view"], "vp")
            self.assertFalse(child.snapshot["spectrumVisible"])
            self.assertTrue(self.bridge.snapshot["railVisible"])
            self.assertEqual(self.bridge.snapshot["sourcePath"], str(folder.resolve()))
            self.bridge.invoke("export")
            self.assertTrue(self.bridge.snapshot["busy"])
            drain_until(lambda: not child.snapshot["busy"] and not child._pending, timeout=45)
            self.assertEqual(child.snapshot["error"], "")
            self.assertTrue(child.snapshot["resultPaths"])
            self.assertFalse(self.bridge.snapshot["resultsVisible"])
            self.assertFalse(self.bridge.snapshot["logVisible"])
            self.assertEqual(self.bridge.snapshot["resultText"], "")
            self.assertFalse(self.qml_item(window, "app_bottom_panel").isVisible())
            for path in child.snapshot["resultPaths"]:
                self.assertIn(str(path), self.bridge.snapshot["logText"])
            self.assertTrue(all(Path(path).is_file() and Path(path).is_relative_to(self.root)
                                for path in child.snapshot["resultPaths"]))
            drain_until(lambda: not self.bridge.snapshot["maintenanceBusy"])
            self.assertTrue(self.bridge.snapshot["canMoveExportedSources"])
            self.bridge.selectWorkflow("pl")
            self.assertFalse(self.bridge.snapshot["canMoveExportedSources"])
            self.bridge.selectWorkflow("cmp")
            settle()
            self.assertFalse(self.bridge.snapshot["resultsVisible"])
            self.assertEqual(self.bridge.snapshot["logText"].count("cmp: Exported files:"), 1)

        def test_demo_produces_real_scientific_quick_surface_frame(self):
            import numpy as np
            from ui_qt.quick.plot_surface import QuickFigureCanvas
            window, surface = self.make_native_window()
            self.bridge.loadDemo()
            self.wait_for_plot()
            self.assertIsInstance(self.bridge._canvas, QuickFigureCanvas)
            self.assertEqual(self.bridge._cube.Z.shape, (157, 1340))
            self.assertTrue(self.bridge.snapshot["inspectionVisible"])
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.assertEqual(len(self.bridge._maps), 1)
            axis, _ = self.bridge._maps[0]
            self.assertTrue(axis.collections)
            np.testing.assert_allclose(axis.get_xlim(), [1.7, 2.7])
            np.testing.assert_allclose(axis.get_ylim(), [-1, 1])
            drain_until(lambda: surface.presentedGeneration > 0
                        and surface.presentedGeneration == self.bridge._canvas.frame_generation)
            self.assertFalse(self.bridge._canvas.image.isNull())
            self.assertEqual(self.bridge._canvas.image.width(), round(surface.width() * window.devicePixelRatio()))
            self.assertIs(surface.canvas, self.bridge._canvas)
            self.assertFalse(window.grabWindow().isNull())

        def test_drr_domain_derivative_and_pair_curves_are_independent(self):
            import numpy as np
            from core.processing import apply_sg_derivative_energy, nearest_gate_spectrum
            _, fixture = self.load_drr_dat()
            products = self.bridge._drr_products
            np.testing.assert_allclose(products.source.Z, fixture.Z)
            expected, used_window = apply_sg_derivative_energy(
                products.source, derivative=2, window_length=21, polyorder=2)
            np.testing.assert_allclose(products.second.Z, expected.Z)
            self.assertEqual(products.second_window, used_window)
            self.assertFalse(np.allclose(products.raw.Z, products.second.Z))
            self.bridge.setParameter("drrView", "pair")
            self.wait_for_plot()
            self.assertEqual(len(self.bridge._maps), 2)
            self.assertFalse(self.bridge.snapshot["railVisible"])
            self.assertTrue(self.bridge.snapshot["controlsVisible"])
            self.assertAlmostEqual(self.bridge.snapshot["heatmapFraction"], .7)
            display = self.bridge.drrDisplayBridge
            for key, value in (("raw.vmin", -.5), ("raw.vmax", .5),
                               ("second.vmin", -8), ("second.vmax", 8)):
                self.assertTrue(display.setNumber(key, str(value)), display.error)
            drain_until(lambda: not display._debounce.isActive())
            self.wait_for_plot()
            bounds = [(axis.collections[0].norm.vmin, axis.collections[0].norm.vmax) for axis, _ in self.bridge._maps]
            self.assertEqual(bounds, [(-.5, .5), (-8, 8)])
            self.bridge.setParameter("gate", .7)
            settle()
            for cube, ((axis, gate_line), (spectrum_axis, line)) in zip(
                    (products.raw, products.second), zip(self.bridge._maps, self.bridge._spectra)):
                gate, values = nearest_gate_spectrum(cube, .7)
                np.testing.assert_allclose(line.get_ydata(), values)
                np.testing.assert_allclose(line.get_xdata(), cube.energy)
                np.testing.assert_allclose(gate_line.get_ydata(), [gate, gate])
                fraction = axis.get_position().height / (axis.get_position().height + spectrum_axis.get_position().height)
                self.assertAlmostEqual(fraction, .7)
            headers = [axis for axis in self.bridge._canvas.figure.axes if getattr(axis, "_dptk_inspection_header", False)]
            self.assertEqual(len(headers), 2)
            for header, (axis, _) in zip(headers, self.bridge._maps):
                self.assertGreaterEqual(header.get_position().y0, axis.get_position().y1 - .002)
                self.assertTrue(header.child_axes)
                for bar in header.child_axes:
                    self.assertEqual(bar._colorbar.orientation, "horizontal")
                    self.assertEqual(bar.xaxis.get_ticks_position(), "top")
            self.bridge.invoke("toggle-spectrum")
            self.wait_for_plot()
            self.assertTrue(all(not axis.get_visible() for axis, _ in self.bridge._spectra))
            self.bridge.invoke("toggle-spectrum")
            self.wait_for_plot()
            self.assertAlmostEqual(self.bridge.snapshot["heatmapFraction"], .7)
            self.assertTrue(all(axis.get_visible() for axis, _ in self.bridge._spectra))
            self.bridge.setParameter("sgWindow", 9)
            drain_until(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending)
            self.wait_for_plot()
            expected, used_window = apply_sg_derivative_energy(
                products.source, derivative=2, window_length=9, polyorder=2)
            np.testing.assert_allclose(self.bridge._drr_products.second.Z, expected.Z)
            self.assertEqual(self.bridge._drr_products.second_window, used_window)

        def test_drr_native_display_reuses_approved_controls_and_file_picker(self):
            window, surface = self.make_native_window()
            requests = []
            self.bridge.fileDialogRequested.connect(lambda *args: requests.append(args))
            self.load_drr_dat()
            drain_until(lambda: window.findChild(QObject, "drr_native_parameters") is not None)
            display = self.qml_item(window, "drr_quick_display")
            self.assertIs(display.property("bridge"), self.bridge.drrDisplayBridge)
            for name in ("drr_native_display_tab", "drr_native_processing_tab", "drr_select_measurements",
                         "drr_clear_measurements", "drr_background_settings", "drr_input_axis.xmin",
                         "drr_input_axis.xmax", "drr_color_tab_raw", "drr_color_tab_second"):
                self.qml_item(window, name)
            drain_until(lambda: surface.presentedGeneration == self.bridge._canvas.frame_generation and surface.presentedGeneration > 0)
            self.click_item(window, self.qml_item(window, "drr_color_tab_second"))
            self.assertEqual(self.bridge.drrDisplayBridge.snapshot["colorProduct"], 1)
            self.assertEqual(self.bridge.snapshot["drrView"], "raw")
            dialog = self.qml_item(window, "app_file_dialog")
            self.click_item(window, self.qml_item(window, "drr_select_measurements"))
            catalog = self.qml_item(window, "quick_source_catalog")
            self.assertTrue(catalog.property("visible"))
            self.bridge.sourceCatalogBridge.invoke("browse")
            self.assertIn(("catalog", True), requests)
            self.assertTrue(dialog.property("visible"))
            dialog.close()
            catalog.close()
            settle()
            self.click_item(window, self.qml_item(window, "drr_native_processing_tab"))
            self.assertTrue(self.qml_item(window, "drr_native_processing").isVisible())
            self.click_item(window, self.qml_item(window, "drr_clear_measurements"))
            drain_until(lambda: not self.bridge._maps and not self.bridge._canvas.backend._timer.isActive())
            self.assertIsNone(self.bridge._cube)
            self.assertIsNone(self.bridge._drr_products)
            self.assertFalse(self.bridge.snapshot["inspectionVisible"])
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.assertEqual(self.bridge.snapshot["sourceSummary"], "No measurement selected")

        def test_workflow_switch_restores_own_measurements_and_display_values(self):
            pl_path, _ = self.load_dat("pl.dat")
            pl_cube = self.bridge._cube
            self.bridge.setParameter("gate", 3)
            self.bridge.setParameter("sideBySide", True)
            self.bridge.selectWorkflow("drr")
            self.assertIsNone(self.bridge._cube)
            self.assertTrue(self.bridge.snapshot["railVisible"])
            self.assertIsNone(self.bridge._source)
            self.assertFalse(self.bridge.snapshot["inspectionVisible"])
            self.assertFalse(self.bridge._maps)
            drr_path, _ = self.load_drr_dat()
            drr_cube = self.bridge._cube
            self.bridge.setParameter("gate", -1)
            self.bridge.selectWorkflow("pl")
            self.wait_for_plot()
            self.assertIs(self.bridge._cube, pl_cube)
            self.assertEqual(self.bridge._source, pl_path.resolve())
            self.assertEqual(self.bridge._values["gate"], 3)
            self.assertTrue(self.bridge.snapshot["sideBySide"])
            self.assertFalse(self.bridge.snapshot["railVisible"])
            self.assertEqual(len(self.bridge._maps), 2)
            self.bridge.selectWorkflow("drr")
            self.wait_for_plot()
            self.assertIs(self.bridge._cube, drr_cube)
            self.assertEqual(self.bridge._source, drr_path.resolve())
            self.assertEqual(self.bridge._values["gate"], -1)
            self.assertFalse(self.bridge.snapshot["sideBySide"])
            self.assertTrue(self.bridge.snapshot["railVisible"])
            self.assertEqual(len(self.bridge._maps), 1)

        def test_slides_workflow_loads_real_native_workspace_from_app_bridge(self):
            folder = self.root / "Processed Data"
            folder.mkdir()
            image = QImage(24, 20, QImage.Format.Format_RGB32)
            image.fill(Qt.GlobalColor.red)
            self.assertTrue(image.save(str(folder / "synthetic_PL_linear.png"), "PNG"))
            self.bridge.setSourcePath(str(self.root))
            window, _ = self.make_native_window()
            self.bridge.selectWorkflow("slides")
            slides = self.bridge.workflowBridge
            self.assertIsNot(slides, self.bridge)
            self.assertTrue(self.bridge.workspaceUrl.toLocalFile().endswith("Presentation.qml"))
            self.assertFalse(self.bridge.snapshot["controlsVisible"])
            drain_until(lambda: window.findChild(QObject, "quick_presentation_workspace") is not None)
            page = self.qml_item(window, "quick_presentation_workspace")
            self.assertIs(page.property("bridge"), slides)
            drain_until(lambda: not slides.snapshot["scanning"])
            self.assertEqual(slides.snapshot["availableCount"], 1)
            self.assertEqual(Path(slides.snapshot["imageRoot"]), folder)
            for name in ("slides_source", "slides_image_root", "slides_search", "slides_build_copy"):
                self.qml_item(window, name)
            self.assertFalse(self.bridge.snapshot["inspectionVisible"])
            self.bridge.selectWorkflow("pl")
            settle()
            self.assertTrue(self.bridge.snapshot["controlsVisible"])

        def test_invalid_parameters_leave_scientific_values_unchanged(self):
            self.bridge.set_cube(self.make_cube())
            self.wait_for_plot()
            values, scales = dict(self.bridge._values), {key: dict(value) for key, value in self.bridge._scales.items()}
            for key, value in (("xmin", float("nan")), ("xmax", float("inf")),
                               ("gate", "invalid"), ("xmin", values["xmax"]),
                               ("ymax", values["ymin"]), ("vmin", scales["linear"]["vmax"]),
                               ("vmax", scales["linear"]["vmin"]), ("scale", "unknown"),
                               ("cmap", "missing-colormap"), ("missing-field", 2)):
                with self.subTest(key=key, value=value):
                    self.bridge.setParameter(key, value)
                    self.assertEqual(self.bridge._values, values)
                    self.assertEqual(self.bridge._scales, scales)
            self.assertFalse(self.bridge._plot_timer.isActive())

        def test_independent_color_limits_and_comparison_rail_policy(self):
            self.bridge.set_cube(self.make_cube())
            self.wait_for_plot()
            for key, value in (("vmin", .2), ("vmax", 50), ("scale", "log"),
                               ("vmin", .1), ("vmax", 60), ("sideBySide", True)):
                self.bridge.setParameter(key, value)
            self.wait_for_plot()
            self.assertFalse(self.bridge.snapshot["railVisible"])
            self.assertTrue(self.bridge.snapshot["controlsVisible"])
            self.assertEqual(len(self.bridge._maps), 2)
            linear, logarithmic = [axis.collections[0].norm for axis, _ in self.bridge._maps]
            self.assertEqual((linear.vmin, linear.vmax), (.2, 50))
            self.assertEqual((logarithmic.vmin, logarithmic.vmax), (.1, 60))
            axes = tuple(axis for axis, _ in self.bridge._maps)
            self.bridge.setParameter("scale", "linear")
            self.assertEqual(self.fields()["vmin"]["value"], .2)
            self.assertEqual(self.fields()["vmax"]["value"], 50)
            self.assertFalse(self.bridge._plot_timer.isActive())
            self.assertEqual(tuple(axis for axis, _ in self.bridge._maps), axes)
            self.bridge.setParameter("sideBySide", False)
            self.assertTrue(self.bridge.snapshot["railVisible"])
            self.bridge.setParameter("sideBySide", True)
            self.bridge.setRailVisible(False)
            self.bridge.setParameter("sideBySide", False)
            self.assertFalse(self.bridge.snapshot["railVisible"])
            self.bridge.setParameter("sideBySide", True)
            self.bridge.setRailVisible(True)
            self.bridge.setParameter("sideBySide", False)
            self.assertTrue(self.bridge.snapshot["railVisible"])
            self.assertTrue(self.bridge.snapshot["controlsVisible"])
            self.wait_for_plot()

        def test_gate_uses_exact_nearest_spectrum_and_preserves_map_axes(self):
            import numpy as np
            from core.processing import nearest_gate_spectrum
            cube = self.make_cube()
            self.bridge.set_cube(cube)
            self.wait_for_plot()
            axes = tuple(axis for axis, _ in self.bridge._maps)
            draws = self.bridge._canvas.draw_count
            self.bridge.setParameter("gate", 2.8)
            settle()
            expected_gate, expected_spectrum = nearest_gate_spectrum(cube, 2.8)
            self.assertEqual(self.fields()["gate"]["value"], expected_gate)
            self.assertEqual(tuple(axis for axis, _ in self.bridge._maps), axes)
            for (_, gate_line), (_, line) in zip(self.bridge._maps, self.bridge._spectra):
                np.testing.assert_array_equal(gate_line.get_ydata(), [expected_gate, expected_gate])
                np.testing.assert_array_equal(line.get_xdata(), cube.energy)
                np.testing.assert_array_equal(line.get_ydata(), expected_spectrum)
            self.assertGreater(self.bridge._canvas.blit_count, 0)
            self.assertLessEqual(self.bridge._canvas.draw_count, draws + 1)
            self.bridge.invoke("previous-gate")
            self.assertEqual(self.fields()["gate"]["value"], 0)
            self.bridge.invoke("next-gate")
            self.assertEqual(self.fields()["gate"]["value"], 3)
            self.bridge.invoke("next-gate")
            self.assertEqual(self.fields()["gate"]["value"], 3)

        def test_gate_spectrum_range_keeps_new_amplitude_visible(self):
            import numpy as np
            cube = self.make_cube()
            self.bridge.set_cube(cube)
            self.wait_for_plot()
            self.bridge.setParameter("gate", 3)
            settle()
            spectrum_axis, line = self.bridge._spectra[0]
            low, high = spectrum_axis.get_ylim()
            self.assertLessEqual(low, np.min(line.get_ydata()))
            self.assertGreaterEqual(high, np.max(line.get_ydata()))

        def test_split_and_spectrum_visibility_survive_rebuilds(self):
            self.bridge.set_cube(self.make_cube())
            self.wait_for_plot()
            self.bridge.setParameter("heatmapFraction", .53)
            self.bridge.setParameter("spectrumVisible", False)
            self.bridge.setParameter("sideBySide", True)
            self.wait_for_plot()
            self.assertAlmostEqual(self.bridge.snapshot["heatmapFraction"], .53)
            self.assertFalse(self.bridge.snapshot["spectrumVisible"])
            hidden_heights = [axis.get_position().height for axis, _ in self.bridge._maps]
            for (axis, _), (spectrum, _) in zip(self.bridge._maps, self.bridge._spectra):
                self.assertFalse(spectrum.get_visible())
                self.assertAlmostEqual(axis.get_position().y0, spectrum.get_position().y0)
            self.bridge.invoke("toggle-spectrum")
            settle()
            self.assertTrue(self.bridge.snapshot["spectrumVisible"])
            self.assertAlmostEqual(self.bridge.snapshot["heatmapFraction"], .53)
            for index, ((axis, _), (spectrum, _)) in enumerate(zip(self.bridge._maps, self.bridge._spectra)):
                self.assertTrue(spectrum.get_visible())
                self.assertLess(axis.get_position().height, hidden_heights[index])
                fraction = axis.get_position().height / (axis.get_position().height + spectrum.get_position().height)
                self.assertAlmostEqual(fraction, .53)
            divider = self.bridge.snapshot["dividerY"] + .03
            self.bridge.setParameter("dividerY", divider)
            self.assertGreater(self.bridge.snapshot["heatmapFraction"], .53)
            self.assertAlmostEqual(self.bridge.snapshot["dividerY"], divider)

        def test_dat_latest_request_wins_after_an_older_worker_finishes(self):
            import numpy as np
            from ui_qt.quick.pl_workflow import load_pl_source
            first, _ = self.write_dat("older.dat")
            latest, expected = self.write_dat("latest.dat", offset=100)
            started, release = threading.Event(), threading.Event()
            self.addCleanup(release.set)
            def delayed(request):
                if Path(request.path).name == first.name:
                    started.set()
                    if not release.wait(5):
                        raise AssertionError("Synthetic older request was not released")
                return load_pl_source(request)
            completions = []
            self.bridge.operationCompleted.connect(lambda *args: completions.append(args))
            with patch("ui_qt.quick.pl_workflow.load_pl_source", side_effect=delayed):
                self.bridge.openFile(QUrl.fromLocalFile(str(first)))
                drain_until(started.is_set)
                self.bridge.openFile(QUrl.fromLocalFile(str(latest)))
                drain_until(lambda: self.bridge._source == latest.resolve() and not self.bridge.snapshot["busy"])
                release.set()
                drain_until(lambda: not self.bridge._pending)
            self.wait_for_plot()
            self.assertEqual(self.bridge._source, latest.resolve())
            np.testing.assert_array_equal(self.bridge._cube.Z, expected.Z)
            self.assertEqual([kind for kind, _ in completions], ["load"])
            self.assertIn("latest.dat", self.bridge.snapshot["status"])

        def test_demo_supersedes_an_older_measurement_request(self):
            from ui_qt.quick.pl_workflow import load_pl_source
            first, _ = self.write_dat("older.dat")
            started, release = threading.Event(), threading.Event()
            self.addCleanup(release.set)
            def delayed(request):
                started.set()
                if not release.wait(5):
                    raise AssertionError("Synthetic request was not released")
                return load_pl_source(request)
            with patch("ui_qt.quick.pl_workflow.load_pl_source", side_effect=delayed):
                self.bridge.openFile(QUrl.fromLocalFile(str(first)))
                drain_until(started.is_set)
                self.bridge.loadDemo()
                release.set()
                drain_until(lambda: not self.bridge._pending)
            self.wait_for_plot()
            self.assertIsNone(self.bridge._source)
            self.assertEqual(self.bridge._cube.title, "Synthetic PL measurement")
            self.assertEqual(self.bridge._cube.Z.shape, (157, 1340))
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.assertFalse(self.bridge.snapshot["busy"])

        def test_close_cancels_pending_load_and_stops_publication(self):
            from core.load_cancellation import check_load_cancelled
            source, _ = self.write_dat("cancel.dat")
            started = threading.Event()
            completed = []
            def cooperative_load(*_args, **_kwargs):
                started.set()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    check_load_cancelled()
                    time.sleep(.005)
                raise AssertionError("Bridge did not cancel the synthetic worker")
            self.bridge.operationCompleted.connect(lambda *args: completed.append(args))
            with patch("ui_qt.quick.pl_workflow.load_pl_source", side_effect=cooperative_load):
                self.bridge.openFile(QUrl.fromLocalFile(str(source)))
                drain_until(started.is_set)
                self.bridge.close()
                self.bridge.close()
                settle()
            self.assertEqual(completed, [])
            self.assertEqual(self.bridge._pending, {})
            self.assertTrue(self.bridge._closed)
            self.assertFalse(self.bridge._plot_timer.isActive())
            self.assertFalse(self.bridge._publish_timer.isActive())
            self.assertIsNone(self.bridge._source)

        def test_pl_export_writes_linear_log_and_dat_without_pair_png(self):
            import numpy as np
            from core.loader import load_dat
            source, expected = self.load_dat()
            self.bridge.setParameter("sideBySide", True)
            self.wait_for_plot()
            window, surface = self.make_native_window()
            self.bridge.invoke("spectrum-find")
            self.assertTrue(self.bridge.snapshot["resultsVisible"])
            analysis = self.bridge.snapshot["resultText"]
            self.assertTrue(analysis.startswith("Peaks:"), analysis)
            self.bridge.invoke("toggle-results")
            settle()
            self.assertFalse(self.qml_item(window, "app_bottom_panel").isVisible())
            plot_height = surface.height()
            self.bridge.invoke("export")
            drain_until(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending, timeout=45)
            # Receipt verification may finish in the same event loop turn as
            # export, so either successful completion message is valid here.
            self.assertTrue(self.bridge.snapshot["status"].startswith(("Export completed", "Export verified.")),
                            self.bridge.snapshot["status"])
            self.assertFalse(self.bridge.snapshot["resultsVisible"])
            self.assertFalse(self.bridge.snapshot["logVisible"])
            self.assertEqual(self.bridge.snapshot["resultText"], analysis)
            self.assertFalse(self.qml_item(window, "app_bottom_panel").isVisible())
            self.assertAlmostEqual(surface.height(), plot_height)
            paths = [path for path in self.root.rglob("*")
                     if path.suffix in {".png", ".dat"} and path != source]
            self.assertEqual(len(paths), 3)
            self.assertEqual(sorted(path.suffix for path in paths), [".dat", ".png", ".png"])
            for path in paths:
                self.assertTrue(path.is_relative_to(self.root))
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
                self.assertIn(str(path), self.bridge.snapshot["logText"])
            pngs = list(self.root.rglob("*.png"))
            self.assertEqual(len(pngs), 2)
            self.assertTrue(any(path.name.endswith("_linear.png") for path in pngs))
            self.assertTrue(any(path.name.endswith("_log.png") for path in pngs))
            self.assertFalse(any("side_by_side" in path.name for path in pngs))
            exported = load_dat(next(path for path in paths if path.suffix == ".dat"))
            np.testing.assert_allclose(exported.energy, expected.energy)
            np.testing.assert_allclose(exported.gate, expected.gate)
            np.testing.assert_allclose(exported.Z, expected.Z)
            self.assertTrue(source.is_file())

        def test_drr_export_preserves_open_analysis_and_log_panels(self):
            self.load_drr_dat()
            window, surface = self.make_native_window()
            self.bridge.invoke("spectrum-find")
            self.bridge.invoke("toggle-log")
            panel = self.qml_item(window, "app_bottom_panel")
            drain_until(lambda: panel.property("resultsAvailable") and panel.property("logAvailable"))
            panel.setProperty("selectedPanel", "log")
            settle()
            self.assertEqual(panel.property("selectedPanel"), "log")
            analysis = self.bridge.snapshot["resultText"]
            self.assertTrue(analysis.startswith("Peaks:"), analysis)
            plot_height = surface.height()
            self.bridge.invoke("export")
            drain_until(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending, timeout=45)
            self.assertTrue(self.bridge.snapshot["status"].startswith(("Export completed", "Export verified.")),
                            self.bridge.snapshot["status"])
            self.assertTrue(self.bridge.snapshot["resultsVisible"])
            self.assertTrue(self.bridge.snapshot["logVisible"])
            self.assertEqual(self.bridge.snapshot["resultText"], analysis)
            self.assertTrue(panel.isVisible())
            self.assertEqual(panel.property("selectedPanel"), "log")
            self.assertAlmostEqual(surface.height(), plot_height)
            self.assertIn("drr: Exported files:", self.bridge.snapshot["logText"])
            self.assertTrue(list(self.root.rglob("*.png")))

        def test_drr_wide_png_toolbar_saves_both_widths_with_shared_dat(self):
            import numpy as np
            from PIL import Image
            from core.loader import load_dat
            from PySide6.QtCore import QMetaObject, Q_ARG
            self.load_drr_dat()
            self.bridge.setParameter("drrView", "pair")
            self.wait_for_plot()
            products = self.bridge._drr_products
            window, surface = self.make_native_window()
            layout = self.qml_item(window, "app_png_layout")
            editor = self.qml_item(window, "app_png_width")
            self.assertTrue(layout.isVisible())
            self.assertEqual(layout.property("currentIndex"), 0)
            self.assertFalse(editor.isEnabled())
            self.assertEqual(self.bridge.snapshot["pngWidth"], 2.)
            layout.setProperty("currentIndex", 1)
            QMetaObject.invokeMethod(layout, "activated", Qt.ConnectionType.DirectConnection, Q_ARG(int, 1))
            drain_until(editor.isEnabled)
            field = self.qml_item(window, "app_parameter_export.pngWidth")
            field.setProperty("text", "1.5")
            QMetaObject.invokeMethod(field, "textEdited")
            self.assertTrue(editor.property("dirty"))
            plot_size = (surface.width(), surface.height())
            QMetaObject.invokeMethod(window, "exportCurrent")
            self.assertEqual(self.bridge.snapshot["pngWidth"], 1.5)
            drain_until(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending, timeout=45)
            self.assertTrue(self.bridge.snapshot["status"].startswith(("Export completed", "Export verified.")),
                            self.bridge.snapshot["status"])
            directory = self.root / "Processed Data" / "DRR"
            pngs, dats = list(directory.glob("*.png")), list(directory.glob("*.dat"))
            self.assertEqual(len(pngs), 4)
            self.assertEqual(len(dats), 2)
            for path in pngs:
                with Image.open(path) as png:
                    self.assertEqual(png.size, (1800, 930) if path.stem.endswith("_1.5x") else (1200, 930))
                self.assertIn(str(path), self.bridge.snapshot["logText"])
            for path in dats:
                metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
                expected = products.second if metadata["processing"].get("derivative_order") == 2 else products.raw
                restored = load_dat(path)
                np.testing.assert_allclose(restored.energy, expected.energy)
                np.testing.assert_allclose(restored.gate, expected.gate)
                np.testing.assert_allclose(restored.Z, expected.Z)
            self.assertEqual((surface.width(), surface.height()), plot_size)
            self.assertFalse(self.bridge.snapshot["resultsVisible"])
            self.capture_native(window, "drr-wide-export-controls.png")
            window.setWidth(640)
            settle()
            save = self.qml_item(window, "app_export")
            for item in (layout, editor, save):
                position = item.mapToScene(QPointF(0, 0))
                self.assertTrue(item.isVisible())
                self.assertGreaterEqual(position.x(), 0)
                self.assertLessEqual(position.x() + item.width(), window.width())
            self.capture_native(window, "drr-wide-export-compact.png")
            self.bridge.setParameter("export.pngLayout", "original")
            self.bridge.invoke("export")
            drain_until(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending, timeout=45)
            self.assertEqual(set(directory.glob("*.dat")), set(dats))
            self.assertEqual(set(directory.glob("*.png")), set(pngs))

        def test_wide_png_choice_survives_measurement_changes_clear_and_workflows(self):
            from PySide6.QtCore import QMetaObject
            from PySide6.QtQml import QQmlComponent, QQmlEngine
            self.bridge.selectWorkflow("drr")
            first = self.write_drr_csv("first_TG_rep01.csv")
            second = self.write_drr_csv("second_TG_rep01.csv", offset=10.)

            def select_measurement(path):
                self.bridge.sourceCatalogBridge.selectionAccepted.emit("drr", [str(path)])
                drain_until(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending, timeout=45)
                self.assertEqual(self.bridge._source, path.resolve(), self.bridge.snapshot["status"])
                self.assertTrue(self.bridge.snapshot["canExport"])
                self.wait_for_plot()

            select_measurement(first)
            window, _ = self.make_native_window()
            layout = self.qml_item(window, "app_png_layout")
            width = self.qml_item(window, "app_png_width")
            adapter = QQmlComponent(QQmlEngine.contextForObject(layout).engine())
            adapter.setData(b'''import QtQuick
QtObject {
    required property Item control
    property QtObject popupObject: control ? control.popup : null
    property Item listObject: control ? control.popup.contentItem : null
}''', QUrl())
            popup_adapter = adapter.createWithInitialProperties({"control": layout})
            self.assertIsNotNone(popup_adapter, "\n".join(error.toString() for error in adapter.errors()))
            popup = popup_adapter.property("popupObject")
            listing = popup_adapter.property("listObject")

            def wide_row():
                def find(item):
                    if item.property("text") == "Wide PNG" and item.property("highlighted") is not None:
                        return item
                    return next((found for child in item.childItems() if (found := find(child)) is not None), None)
                return find(listing)

            self.click_item(window, layout)
            drain_until(lambda: popup.property("visible") and wide_row() is not None)
            self.click_item(window, wide_row())
            drain_until(lambda: not popup.property("visible"))
            drain_until(width.isEnabled)
            field = self.qml_item(window, "app_parameter_export.pngWidth")
            field.setProperty("text", "1.75")
            QMetaObject.invokeMethod(field, "textEdited")

            def assert_wide_export():
                settle()
                self.assertEqual(self.bridge.snapshot["pngLayout"], "wide")
                self.assertEqual(layout.property("currentIndex"), 1)
                self.assertTrue(width.isEnabled())
                with patch("ui_qt.quick.drr_workflow.export_drr", return_value={}) as export:
                    QMetaObject.invokeMethod(window, "exportCurrent")
                    drain_until(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending)
                self.assertEqual(export.call_count, 1)
                self.assertEqual(export.call_args.kwargs["raw_params"].png_width_scale, 1.75)
                self.assertEqual(export.call_args.kwargs["second_params"].png_width_scale, 1.75)
                self.assertEqual(self.bridge.snapshot["pngWidth"], 1.75)

            assert_wide_export()
            select_measurement(second)
            assert_wide_export()
            self.click_item(window, self.qml_item(window, "drr_clear_measurements"))
            self.assertIsNone(self.bridge._cube)
            self.assertEqual(self.bridge.snapshot["pngLayout"], "wide")
            select_measurement(first)
            assert_wide_export()
            self.bridge.selectWorkflow("pl")
            self.load_dat("pl-export-preference.dat")
            with patch("ui_qt.quick.pl_workflow.export_pl", return_value={}) as export:
                QMetaObject.invokeMethod(window, "exportCurrent")
                drain_until(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending)
            self.assertEqual({params.png_width_scale for params in export.call_args.args[2].values()}, {1.75})
            self.bridge.selectWorkflow("drr")
            self.wait_for_plot()
            assert_wide_export()

            self.settings.sync()
            reopened_settings = QSettings(str(self.root / "settings.ini"), QSettings.Format.IniFormat)
            reopened = QuickApplicationBridge(settings=reopened_settings)
            self.addCleanup(reopened.close)
            self.assertEqual(reopened.snapshot["pngWidth"], 1.75)
            self.assertEqual(reopened.snapshot["pngLayout"], "wide")

        def test_invalid_stored_png_layout_uses_original_width(self):
            for stored in ("invalid", "", "WIDE"):
                with self.subTest(stored=stored):
                    self.settings.setValue("export/png_layout", stored)
                    self.settings.setValue("export/png_width_scale", 1.5)
                    reopened = QuickApplicationBridge(settings=self.settings)
                    try:
                        self.assertEqual(reopened.snapshot["pngLayout"], "original")
                        self.assertEqual(reopened.snapshot["pngWidth"], 1.5)
                        self.assertEqual(reopened._png_export_width(), 1.)
                    finally:
                        reopened.close()

        def test_png_width_options_are_export_only_validated_and_remembered(self):
            self.load_dat()
            frame = self.bridge._canvas.frame_generation
            self.bridge.setParameter("export.pngLayout", "wide")
            self.bridge.setParameter("export.pngWidth", 1.75)
            settle()
            self.assertEqual(self.bridge._canvas.frame_generation, frame)
            with patch("ui_qt.quick.pl_workflow.export_pl", return_value={}) as export:
                self.bridge._pl_export_request()()
            self.assertEqual({params.png_width_scale for params in export.call_args.args[2].values()}, {1.75})
            for value in (float("nan"), float("inf"), 0.5, 3.5, "invalid"):
                self.bridge.setParameter("export.pngWidth", value)
                self.assertEqual(self.bridge.snapshot["pngWidth"], 1.75)
            self.bridge.setParameter("export.pngLayout", "invalid")
            self.assertEqual(self.bridge.snapshot["pngLayout"], "wide")
            reopened = QuickApplicationBridge(settings=self.settings)
            self.addCleanup(reopened.close)
            self.assertEqual(reopened.snapshot["pngWidth"], 1.75)
            self.assertEqual(reopened.snapshot["pngLayout"], "wide")
            self.bridge.setParameter("export.pngLayout", "original")
            self.settings.sync()
            original = QuickApplicationBridge(settings=QSettings(
                str(self.root / "settings.ini"), QSettings.Format.IniFormat))
            self.addCleanup(original.close)
            self.assertEqual(original.snapshot["pngLayout"], "original")
            self.assertEqual(original.snapshot["pngWidth"], 1.75)
            self.assertEqual(original._png_export_width(), 1.)

        def test_child_export_logs_once_without_reopening_panels_or_replaying_cleared_log(self):
            self.bridge.selectWorkflow("cmp")
            child = self.bridge.workflowBridge
            self.bridge.selectWorkflow("pl")
            path = self.root / "saved comparison.dat"
            child._result = [path]
            child.snapshotChanged.emit()
            settle()
            self.assertIn(str(path), self.bridge.snapshot["logText"])
            self.assertEqual(self.bridge.snapshot["resultText"], "")
            self.assertFalse(self.bridge.snapshot["resultsVisible"])
            self.assertFalse(self.bridge.snapshot["logVisible"])
            child.snapshotChanged.emit()
            self.bridge.selectWorkflow("cmp")
            settle()
            self.assertEqual(self.bridge.snapshot["logText"].count("cmp: Exported files:"), 1)
            self.bridge.invoke("clear-log")
            child.snapshotChanged.emit()
            settle()
            self.assertEqual(self.bridge.snapshot["logText"], "")
            child._result = []
            child.snapshotChanged.emit()
            child._result = [path]
            child.snapshotChanged.emit()
            settle()
            self.assertEqual(self.bridge.snapshot["logText"].count("cmp: Exported files:"), 1)
            self.assertFalse(self.bridge.snapshot["resultsVisible"])

    requested_cases = os.environ.get("DPTK_QUICK_CASE", "").split(",")
    suite = (unittest.TestSuite(BridgeCases(name) for name in requested_cases if name)
             if any(requested_cases) else unittest.defaultTestLoader.loadTestsFromTestCase(BridgeCases))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    qInstallMessageHandler(previous_handler)
    print("QUICK_APPLICATION_BRIDGE_TESTS=" + json.dumps({"tests": result.testsRun, "success": result.wasSuccessful()}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
