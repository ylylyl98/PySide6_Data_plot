"""Native advanced Power analysis with synthetic data and isolated settings."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickPowerAnalysisTests(unittest.TestCase):
    def test_widgets_free_advanced_power_analysis(self):
        with TemporaryDirectory(prefix="quick-power-analysis-process-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software", MPLBACKEND="agg",
                MPLCONFIGDIR=temporary, XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_POWER_ANALYSIS_TESTS=", result.stdout)


def _run_isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import importlib.abc
    import json
    import time
    import threading
    import traceback
    from types import SimpleNamespace
    from unittest.mock import patch

    class NoWidgets(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {"PySide6.QtWidgets", "PySide6.QtQuickWidgets", "ui_qt.main_window", "ui_qt.common",
                "ui_qt.power_peak_controller", "ui_qt.power_law_dialog", "ui_qt.power_refit_dialog",
                "matplotlib.backends.backend_qt", "matplotlib.backends.backend_qtagg"}:
                raise AssertionError("Power analysis imported a Widgets dependency: " + fullname)
    sys.meta_path.insert(0, NoWidgets())
    from PySide6.QtCore import QObject, QSettings, QUrl, QCoreApplication, QEvent, Qt, qInstallMessageHandler
    from PySide6.QtGui import QGuiApplication, QFont, QFontDatabase, QKeyEvent, QMouseEvent
    from PySide6.QtQuickControls2 import QQuickStyle
    from ui_qt.quick.power_bridge import PowerBridge
    initial_imports = set(sys.modules)
    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    if sys.platform == "win32":
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
        app.setFont(QFont("Segoe UI", 9))
    python_errors, qml_errors = [], []
    def exception(kind, value, trace):
        python_errors.append("".join(traceback.format_exception(kind, value, trace)))
        traceback.print_exception(kind, value, trace)
    sys.excepthook = exception
    def qt_message(kind, context, message):
        if any(word in message for word in ("ReferenceError", "TypeError", "Binding loop", "Cannot assign", "Unable to assign", "Error:")):
            qml_errors.append(str(message))
            print(message, file=sys.stderr)
    old_handler = qInstallMessageHandler(qt_message)
    def wait(predicate, timeout=35):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                app.processEvents()
                return
            time.sleep(.005)
        raise AssertionError("Timed out waiting for native Power analysis")

    class PowerCases(unittest.TestCase):
        def setUp(self):
            start_python, start_qml = len(python_errors), len(qml_errors)
            self.addCleanup(lambda: self.assertEqual(python_errors[start_python:], [], "Uncaught native callback"))
            self.addCleanup(lambda: self.assertEqual(qml_errors[start_qml:], [], "QML runtime error"))
            self.temporary = TemporaryDirectory(prefix="quick-power-analysis-case-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name)
            self.settings = QSettings(str(self.root / "isolated.ini"), QSettings.Format.IniFormat)
            self.settings.setValue("unrelated/preserved", "keep")
            self.bridge = PowerBridge(settings=self.settings)
            self.addCleanup(self.bridge.close)

        def install(self, channels=("Power",), model="Lorentzian", duplicates=False):
            import numpy as np
            from core.loader import DataCube
            from core.data_io import PowerSeriesResult
            from core.processing import PowerSweepPoint
            from core.power_multi_peaks import multi_peak_model
            from ui_qt.quick.power_workflow import fingerprints
            energy, power = np.linspace(1.34, 1.56, 301), np.geomspace(1., 100., 13)
            if duplicates:
                power[5] = power[4]
            cubes, records, files = {}, {}, []
            for channel_index, channel in enumerate(channels):
                spectra = np.array([multi_peak_model(energy, [500., 10., 80. * p * (1 + .2 * channel_index), 1.41, .025,
                    35. * p ** 1.35 * (1 + .1 * channel_index), 1.46, .018], model=model, origin=energy.mean()) for p in power])
                if duplicates:
                    spectra[5] = 500. + (spectra[5] - 500.) * 1.15
                path = self.root / (channel + ".csv")
                np.savetxt(path, np.column_stack((power, spectra)), delimiter=",",
                    header="power_uW," + ",".join(map(str, energy)), comments="")
                files.append(path.name)
                cubes[channel] = DataCube(energy.copy(), power.copy(), spectra, "Power (uW)", channel, "PL (a.u.)")
                records[channel] = tuple(PowerSweepPoint(path.name, float(p), float(index), index + 2,
                    source_provenance=json.dumps(dict(sources=[dict(file=f"segment-{int(index >= 6)}.csv", row=index + 2)])))
                    for index, p in enumerate(power))
            bridge = self.bridge
            bridge._folder, bridge._stage = str(self.root), "plot"
            bridge._values.update(mode="single" if len(channels) == 1 else "intensity", autoBackground=False,
                background=0., xmin=float(energy[0]), xmax=float(energy[-1]), ymin=float(power[0]), ymax=float(power[-1]),
                vmin=500., vmax=float(max(cube.Z.max() for cube in cubes.values())), gate=float(power[0]), clip=False)
            bridge._products = dict(cubes=cubes, corrected=cubes, records=records, paired=None, background=0., stagePairs=())
            bridge._raw_results = {key: PowerSeriesResult(cube, key, records[key], {key: records[key]}) for key, cube in cubes.items()}
            bridge._active_selection = dict(group="synthetic", single=files[0], KK=files[0], KKp=files[-1], mode=bridge._values["mode"], pairing="power")
            bridge._signatures = fingerprints(str(self.root), files)
            bridge._analysis_data_changed()
            bridge._schedule_plot()
            return cubes

        def settled(self):
            wait(lambda: not self.bridge._pending and not (self.bridge._running & {"fit", "refit", "law"}))

        def fitted(self, **kwargs):
            cubes = self.install(**kwargs)
            self.bridge.setParameter("analysis.enabled", True)
            self.settled()
            self.plotted()
            self.assertTrue(self.bridge.snapshot["analysisExportReady"], self.bridge.snapshot["status"])
            return cubes

        def plotted(self):
            wait(lambda: self.bridge._canvas is not None and self.bridge._canvas.frame_generation > 0
                and not self.bridge._plot_timer.isActive() and not self.bridge._canvas.backend._timer.isActive())
            self.assertEqual(self.bridge._canvas.last_error, "")
            self.assertEqual(self.bridge.snapshot["error"], "", self.bridge.snapshot["status"])

        def test_00_lazy_constructor_and_session_settings(self):
            self.assertNotIn("numpy", initial_imports)
            self.assertNotIn("matplotlib", initial_imports)
            self.assertFalse(self.bridge.snapshot["analysisEnabled"])
            self.assertTrue(self.bridge.snapshot["analysisExportReady"])
            self.assertIsNone(self.bridge.plotBackend)
            self.assertEqual(self.settings.allKeys(), ["unrelated/preserved"])

        def test_multipeak_lorentzian_gaussian_exact_core_and_detection_add_remove(self):
            import numpy as np
            from core.power_multi_peaks import fit_power_multi_peaks
            for model in ("Lorentzian", "Gaussian"):
                self.bridge._analysis_values["model"] = model
                cubes = self.fitted(model=model)
                fits = self.bridge._analysis_results["Power"]
                self.assertEqual(len(self.bridge._peak_rows), 2)
                expected = fit_power_multi_peaks(cubes["Power"], self.bridge._analysis_settings)
                for row_index, (actual, reference) in enumerate(zip(fits, expected)):
                    self.assertEqual(actual.status, "ok")
                    np.testing.assert_array_equal(actual.fitted, reference.fitted)
                    np.testing.assert_allclose([part.center_ev for part in actual.components], [1.41, 1.46], atol=1e-7)
                    np.testing.assert_allclose([part.fwhm_mev for part in actual.components], [25., 18.], rtol=1e-5)
                    np.testing.assert_allclose(actual.baseline + sum(actual.component_curves), cubes["Power"].Z[row_index], atol=1e-3)
                self.assertEqual(self.bridge.snapshot["peakResultCount"], 26)
                self.assertEqual(self.bridge.peakResultsModel.rowCount(), 26)
            self.bridge.invoke("analysis.add")
            self.settled()
            self.assertEqual(len(self.bridge._peak_rows), 3)
            self.bridge.invoke("analysis.remove")
            self.settled()
            self.assertEqual(len(self.bridge._peak_rows), 2)
            self.bridge.invoke("analysis.detect")
            self.settled()
            self.assertEqual(len(self.bridge._peak_rows), 2)

        def test_axis_metric_cursor_edits_reuse_fits_and_independent_y_scales(self):
            self.fitted(channels=("KK", "KKp"))
            results = self.bridge._analysis_results
            resets = []
            self.bridge.peakResultsModel.modelReset.connect(lambda: resets.append(True))
            with patch("ui_qt.quick.power_workflow.fit_peaks", side_effect=AssertionError("Display edit refitted spectra")):
                self.bridge.setParameter("analysis.intensityLog", False)
                self.bridge.setParameter("analysis.linewidthLog", True)
                self.bridge.setParameter("analysis.metric", "Peak height")
                self.bridge.setParameter("powerLog", True)
                self.bridge.setParameter("gate", 10.)
                self.plotted()
            self.assertIs(self.bridge._analysis_results, results)
            self.assertEqual([axis.get_yscale() for axis in self.bridge._analysis_trends], ["linear", "log"])
            self.assertEqual([axis.get_xscale() for axis in self.bridge._analysis_trends], ["log", "log"])
            self.assertEqual(len(self.bridge._analysis_artists), 8)
            self.assertEqual(len(self.bridge._lines), 2)
            self.assertEqual(resets, [])
            self.assertEqual(self.settings.allKeys(), ["unrelated/preserved"])

        def test_manual_preview_provenance_edit_invalidation_and_one_row_apply(self):
            self.fitted(channels=("KK", "KKp"), duplicates=True)
            original = dict(self.bridge._analysis_results)
            row = next(index for index, value in enumerate(self.bridge._analysis_result_rows)
                if value["channel"] == "KKp" and value["index"] == 5 and value["peak"] == 2)
            self.bridge.setParameter("analysis.selectedResult", row)
            self.bridge.invoke("analysis.manual")
            self.plotted()
            self.assertEqual(self.bridge._refit_state["channel"], "KKp")
            self.assertEqual(self.bridge._refit_state["index"], 5)
            self.assertEqual(self.bridge._refit_state["sourceRow"], 7)
            self.assertFalse(self.bridge.snapshot["refit"]["canApply"])
            self.bridge.invoke("refit.run")
            self.settled()
            self.assertTrue(self.bridge.snapshot["refit"]["canApply"])
            self.bridge.setParameter("refit.peak.0.width", 26.)
            self.assertFalse(self.bridge.snapshot["refit"]["canApply"])
            self.bridge.invoke("refit.run")
            self.settled()
            trial = self.bridge._refit_trial
            self.bridge.invoke("refit.apply")
            self.settled()
            self.plotted()
            self.assertEqual(self.bridge._stage, "plot")
            self.assertIs(self.bridge._analysis_results["KK"], original["KK"])
            for index, fit in enumerate(self.bridge._analysis_results["KKp"]):
                self.assertIs(fit, trial if index == 5 else original["KKp"][index])
            self.assertTrue(self.bridge._analysis_results["KKp"][5].manual)
            payload = self.bridge._analysis_export_payload()
            self.assertTrue(payload["results"]["KKp"][5].manual)
            from core.power_peaks import export_peak_analysis
            export_peak_analysis(self.root, payload)
            metadata = json.loads((self.root / "Power_peak_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metadata["manual_refits"]), 1)
            self.assertEqual(metadata["manual_refits"][0]["channel"], "KKp")
            self.assertEqual(metadata["manual_refits"][0]["parameters"], list(trial.parameters))

        def test_inflight_fit_generation_and_duplicate_marker_source_are_exact(self):
            import numpy as np
            from matplotlib.backend_bases import MouseEvent
            from ui_qt.quick.power_workflow import fit_peaks
            self.install(channels=("KK", "KKp"), duplicates=True)
            started, release = threading.Event(), threading.Event()
            self.addCleanup(release.set)
            def delayed(cubes, rows, model, **kwargs):
                if model == "Lorentzian":
                    started.set()
                    if not release.wait(10):
                        raise AssertionError("Synthetic initial fit was not released")
                return fit_peaks(cubes, rows, model, **kwargs)
            completions = []
            self.bridge.operationCompleted.connect(lambda kind, elapsed: completions.append(kind))
            with patch("ui_qt.quick.power_workflow.fit_peaks", side_effect=delayed):
                self.bridge.setParameter("analysis.enabled", True)
                wait(started.is_set)
                self.bridge.setParameter("analysis.model", "Gaussian")
                wait(lambda: "fit" not in self.bridge._running)
                release.set()
                self.settled()
            self.assertEqual(self.bridge._analysis_settings.model, "Gaussian")
            self.assertEqual(completions.count("fit"), 1)
            self.bridge.setParameter("analysis.model", "Lorentzian")
            self.settled()
            self.plotted()
            fit = self.bridge._analysis_results["KKp"][5]
            axis = self.bridge._analysis_trends[0]
            x, y = axis.transData.transform((fit.power_uw, fit.components[0].area))
            MouseEvent("button_press_event", self.bridge._canvas, x, y, button=1)._process()
            self.assertEqual(self.bridge._analysis_selected_channel, "KKp")
            self.assertEqual(self.bridge._analysis_selected_rows["KKp"], 5)
            np.testing.assert_array_equal(self.bridge._lines["KKp"].get_ydata(), self.bridge._products["cubes"]["KKp"].Z[5])
            self.bridge.invoke("analysis.manual")
            self.assertEqual(self.bridge._refit_state["channel"], "KKp")
            self.assertEqual(self.bridge._refit_state["index"], 5)

        def test_manual_source_guard_and_latest_settings_win(self):
            self.fitted()
            self.bridge.invoke("analysis.manual")
            self.bridge.invoke("refit.run")
            self.settled()
            source = self.root / "Power.csv"
            source.write_bytes(source.read_bytes() + b"\n")
            self.bridge.invoke("refit.apply")
            self.settled()
            self.assertIn("changed after loading", self.bridge.snapshot["error"])
            self.assertFalse(any(fit.manual for fit in self.bridge._analysis_results["Power"]))
            self.bridge.invoke("refit.cancel")
            self.bridge._error = ""
            self.install()
            self.bridge._analysis_request_fit()
            self.settled()
            first_key = self.bridge._analysis_current_key
            original_center = self.bridge._peak_rows[0]["center"]
            self.bridge.setParameter("peak.0.center", 1.7)
            self.settled()
            self.assertFalse(self.bridge.snapshot["analysisExportReady"])
            self.bridge.setParameter("peak.0.center", original_center)
            self.settled()
            self.assertEqual(self.bridge._analysis_current_key, first_key)
            self.assertEqual(self.bridge.snapshot["error"], "")
            self.assertTrue(self.bridge.snapshot["analysisExportReady"])

        def test_power_law_shared_individual_drag_suggestions_local_slope_and_exports(self):
            import csv
            import numpy as np
            from core.power_peaks import export_peak_analysis
            self.fitted(channels=("KK", "KKp"))
            results = self.bridge._analysis_results
            self.bridge.invoke("analysis.law")
            self.settled()
            self.plotted()
            state = self.bridge._law_state
            self.assertEqual([row["selected"] for row in state["rows"]], [True, False, True, False])
            self.assertTrue(self.bridge.snapshot["law"]["canApply"])
            self.assertAlmostEqual(self.bridge._law_fits[0]["alpha"], 1., places=6)
            self.assertTrue(self.bridge._law_fits[0]["crosses_sources"])
            self.bridge._analysis_law_span(2., 80.)
            self.settled()
            self.assertEqual([(row["lower"], row["upper"]) for row in state["rows"] if row["selected"]], [(2., 80.), (2., 80.)])
            self.bridge.setParameter("law.shared", False)
            self.bridge.setParameter("law.1.selected", True)
            self.bridge.setParameter("law.1.lower", 3.)
            self.bridge.setParameter("law.1.upper", 90.)
            self.settled()
            self.assertEqual(self.bridge._law_state["rows"][0]["lower"], 2.)
            self.bridge.setParameter("law.local", True)
            self.plotted()
            self.assertEqual(len(self.bridge._canvas.figure.axes), 2)
            slopes = self.bridge._law_local[("KK", 1)][1]
            np.testing.assert_allclose(slopes, np.ones(len(slopes)), atol=1e-6)
            self.bridge.invoke("law.suggest")
            self.settled()
            self.assertTrue(self.bridge._law_state["rows"][0]["suggested"])
            self.assertIn("No reliable", self.bridge._law_state["message"])
            self.assertIs(self.bridge._analysis_results, results)
            self.bridge.invoke("law.apply")
            self.plotted()
            self.assertEqual(self.bridge._stage, "plot")
            payload = self.bridge._analysis_export_payload()
            self.assertEqual(len(payload["power_law_ranges"]), 3)
            export_peak_analysis(self.root, payload)
            metadata = json.loads((self.root / "Power_peak_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["power_law_ranges"], payload["power_law_ranges"])
            self.assertEqual(metadata["intensity_axis"], "log")
            self.assertIn("excludes gain calibration", metadata["power_law_uncertainty"])
            with (self.root / "Power_peak_analysis.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 52)
            self.assertTrue(all(row["source_provenance"] for row in rows))
            self.assertTrue((self.root / "Power_peak_analysis_power_law.csv").is_file())
            self.assertGreater((self.root / "Power_peak_analysis.png").stat().st_size, 1000)

        def test_special_edits_keep_static_pixels_and_cancel_restores_main_canvas(self):
            import numpy as np
            self.fitted(channels=("KK", "KKp"))
            main = self.bridge._canvas
            maps = dict(self.bridge._maps)
            original_image = main.image
            original_draws = main.draw_count
            self.bridge.invoke("analysis.manual")
            self.plotted()
            special = self.bridge._canvas
            self.assertIsNot(special, main)
            axes = tuple(special.figure.axes)
            draws, generation = special.draw_count, special.frame_generation
            self.bridge.setParameter("refit.peak.0.width", 26.)
            self.plotted()
            self.assertEqual(tuple(special.figure.axes), axes)
            self.assertEqual(special.draw_count, draws)
            self.assertEqual(special.frame_generation, generation)
            before = np.asarray(special.buffer_rgba()).copy()
            self.bridge.setParameter("refit.peak.0.center", 1.412)
            self.plotted()
            self.assertEqual(special.draw_count, draws)
            self.assertGreater(special.frame_generation, generation)
            self.assertEqual(list(self.bridge._refit_seed_lines[0].get_xdata()), [1.412, 1.412])
            outside = np.ones(before.shape[:2], bool)
            for axis in axes:
                box = axis.bbox
                x0, x1 = max(0, int(box.x0) - 4), min(before.shape[1], int(box.x1) + 5)
                y0, y1 = max(0, before.shape[0] - int(box.y1) - 4), min(before.shape[0], before.shape[0] - int(box.y0) + 5)
                outside[y0:y1, x0:x1] = False
            np.testing.assert_array_equal(np.asarray(special.buffer_rgba())[outside], before[outside])
            self.bridge.invoke("refit.cancel")
            self.plotted()
            self.assertIs(self.bridge._canvas, main)
            self.assertEqual(self.bridge._maps, maps)
            self.assertEqual(main.draw_count, original_draws)
            self.assertEqual(main.image, original_image)
            self.bridge.invoke("analysis.law")
            self.settled()
            self.plotted()
            self.assertIs(self.bridge._canvas, special)
            axes = tuple(special.figure.axes)
            self.bridge._law_axis.set_ylim(*self.bridge._law_axis.get_ylim())
            special.draw()
            draws, blits = special.draw_count, special.blit_count
            before = np.asarray(special.buffer_rgba()).copy()
            self.bridge.setParameter("law.0.lower", 2.)
            self.settled()
            self.plotted()
            self.assertEqual(tuple(special.figure.axes), axes)
            self.assertEqual(special.draw_count, draws)
            self.assertGreater(special.blit_count, blits)
            for fit in self.bridge._law_fits:
                _, line = self.bridge._law_fit_artists[(fit["channel"], fit["peak"])]
                np.testing.assert_allclose(line.get_ydata(), fit["amplitude"] * line.get_xdata() ** fit["alpha"])
            box = self.bridge._law_axis.bbox
            outside = np.ones(before.shape[:2], bool)
            outside[max(0, before.shape[0] - int(box.y1) - 4):min(before.shape[0], before.shape[0] - int(box.y0) + 5),
                    max(0, int(box.x0) - 4):min(before.shape[1], int(box.x1) + 5)] = False
            np.testing.assert_array_equal(np.asarray(special.buffer_rgba())[outside], before[outside])
            self.bridge.setParameter("law.0.lower", 200.)
            self.settled()
            self.plotted()
            self.assertFalse(self.bridge.snapshot["law"]["canApply"])
            self.assertTrue(all(not line.get_visible() and not span.get_visible()
                                for span, line in self.bridge._law_fit_artists.values()))
            self.bridge.setParameter("law.0.lower", 2.)
            self.settled()
            self.plotted()
            self.assertTrue(self.bridge.snapshot["law"]["canApply"])
            self.assertTrue(all(line.get_visible() and span.get_visible()
                               for span, line in self.bridge._law_fit_artists.values()))
            self.bridge.invoke("law.cancel")
            self.plotted()
            self.assertIs(self.bridge._canvas, main)
            self.assertEqual(main.draw_count, original_draws)
            self.assertEqual(main.image, original_image)
            self.bridge.close()
            self.assertTrue(main.closed)
            self.assertTrue(special.closed)
            self.assertEqual(self.bridge._plot_workspaces, {})

        def test_native_refit_law_results_components_and_keyboard_actions(self):
            from PySide6.QtQml import QQmlApplicationEngine
            from shiboken6 import isValid
            from ui_qt.quick.plot_surface import register_plot_surface
            register_plot_surface()
            self.fitted()
            from ui_qt.quick.appearance import quick_colors
            colors = quick_colors("light")
            self.bridge.invoke("analysis.manual")
            self.plotted()
            engine = QQmlApplicationEngine()
            engine.setInitialProperties(dict(bridge=self.bridge, colors=colors))
            qml_folder = Path(__file__).resolve().parents[1] / "ui_qt" / "quick" / "qml"
            engine.loadData(b'''import QtQuick
import QtQuick.Window
import "."
Window {
    id: window; required property var bridge; required property var colors
    width: 900; height: 760; visible: true
    Loader { anchors.fill: parent; sourceComponent: window.bridge.snapshot.stage === "refit" ? refit : window.bridge.snapshot.stage === "law" ? law : results }
    Component { id: refit; PowerAnalysisRefit { bridge: window.bridge; colors: window.colors } }
    Component { id: law; PowerAnalysisLaw { bridge: window.bridge; colors: window.colors } }
    Component { id: results; PowerAnalysisControls { bridge: window.bridge; colors: window.colors } }
}''', QUrl.fromLocalFile(str(qml_folder / "PowerAnalysisNativeTest.qml")))
            self.assertTrue(engine.rootObjects())
            window = engine.rootObjects()[0]
            def cleanup():
                if isValid(window):
                    window.close()
                if isValid(engine):
                    engine.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                app.processEvents()
            self.addCleanup(cleanup)
            def find(root, name):
                if root.objectName() == name:
                    return root
                for child in root.childItems() if hasattr(root, "childItems") else root.children():
                    found = find(child, name)
                    if found is not None:
                        return found
                return None
            def click(name):
                item = find(window.contentItem(), name)
                self.assertIsNotNone(item, name)
                point = item.mapToScene(item.boundingRect().center())
                global_point = window.mapToGlobal(point.toPoint()).toPointF()
                for kind, buttons in ((QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton), (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton)):
                    QCoreApplication.sendEvent(window, QMouseEvent(kind, point, global_point, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier))
                app.processEvents()
                return item
            wait(lambda: find(window.contentItem(), "power_refit_run") is not None)
            click("power_refit_run")
            self.settled()
            self.assertTrue(self.bridge.snapshot["refit"]["canApply"])
            review = os.environ.get("DPTK_POWER_ANALYSIS_REVIEW_DIR", "")
            if review:
                self.plotted()
                plot = find(window.contentItem(), "power_refit_plot")
                wait(lambda: plot.property("presentedGeneration") == self.bridge._canvas.frame_generation)
                Path(review).mkdir(parents=True, exist_ok=True)
                self.assertTrue(window.grabWindow().save(str(Path(review) / "power-refit.png")))
            item = find(window.contentItem(), "power_refit_apply")
            item.forceActiveFocus()
            QCoreApplication.sendEvent(window, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " "))
            QCoreApplication.sendEvent(window, QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " "))
            self.settled()
            self.assertEqual(self.bridge._stage, "plot")
            self.bridge.setParameter("analysis.showResults", True)
            app.processEvents()
            self.assertIsNotNone(find(window.contentItem(), "power_analysis_results"))
            self.assertTrue(self.bridge.snapshot["peakResults"])
            self.bridge.invoke("analysis.law")
            self.settled()
            self.plotted()
            click("power_law_local")
            self.plotted()
            self.assertTrue(self.bridge._law_state["local"])
            click("power_law_shared")
            self.assertFalse(self.bridge._law_state["shared"])
            self.settled()
            self.plotted()
            plot = find(window.contentItem(), "power_law_plot")
            wait(lambda: plot.property("presentedGeneration") == self.bridge._canvas.frame_generation)
            from PySide6.QtCore import QPointF
            axis = self.bridge._canvas.figure.axes[0]
            center_y = .5 * (axis.bbox.y0 + axis.bbox.y1)
            start_x = axis.transData.transform((3., axis.get_ylim()[0]))[0]
            end_x = axis.transData.transform((70., axis.get_ylim()[0]))[0]
            ratio = self.bridge._canvas.device_pixel_ratio
            start = plot.mapToScene(QPointF(start_x / ratio, plot.height() - center_y / ratio))
            end = plot.mapToScene(QPointF(end_x / ratio, plot.height() - center_y / ratio))
            for kind, point, button, buttons in ((QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton),
                    (QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton),
                    (QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)):
                QCoreApplication.sendEvent(window, QMouseEvent(kind, point, window.mapToGlobal(point.toPoint()).toPointF(), button, buttons, Qt.KeyboardModifier.NoModifier))
                app.processEvents()
            self.settled()
            self.assertAlmostEqual(self.bridge._law_state["rows"][0]["lower"], 3., places=5)
            self.assertAlmostEqual(self.bridge._law_state["rows"][0]["upper"], 70., places=5)
            if review:
                self.plotted()
                wait(lambda: plot.property("presentedGeneration") == self.bridge._canvas.frame_generation)
                self.assertTrue(window.grabWindow().save(str(Path(review) / "power-law.png")))
            window.setWidth(640)
            window.setHeight(580)
            app.processEvents()
            self.assertGreater(find(window.contentItem(), "power_law_plot").height(), 100)
            self.assertIsNotNone(find(window.contentItem(), "power_law_interval_editor"))
            if review:
                self.plotted()
                wait(lambda: plot.property("presentedGeneration") == self.bridge._canvas.frame_generation)
                self.assertTrue(window.grabWindow().save(str(Path(review) / "power-law-narrow.png")))
            self.assertNotIn("PySide6.QtWidgets", sys.modules)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PowerCases))
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    qInstallMessageHandler(old_handler)
    print(f"QUICK_POWER_ANALYSIS_TESTS={result.testsRun}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
