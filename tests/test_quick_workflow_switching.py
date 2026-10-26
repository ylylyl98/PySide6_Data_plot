"""Regressions for revisiting native workflows with already rendered data.

The child process owns a widgets-free Qt Quick application, synthetic sources,
and temporary settings. Assertions count expensive rebuilds and verify the
displayed canvas, rather than depending on machine-specific timing thresholds.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickWorkflowSwitchingTests(unittest.TestCase):
    def test_retained_workflows_in_isolated_quick_process(self):
        with TemporaryDirectory(prefix="quick-switch-process-") as temporary:
            environment = dict(
                os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                MPLBACKEND="agg", MPLCONFIGDIR=temporary, XDG_CONFIG_HOME=temporary,
                XDG_CACHE_HOME=temporary, LOCALAPPDATA=temporary,
                PYTHONIOENCODING="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=150,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_WORKFLOW_SWITCHING=", result.stdout)


def _run_isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import importlib.abc
    import json
    import time
    import traceback
    from unittest.mock import patch

    class NoWidgetImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {
                "PySide6.QtWidgets", "PySide6.QtQuickWidgets", "ui_qt.main_window",
                "matplotlib.backends.backend_qt", "matplotlib.backends.backend_qtagg",
            }:
                raise AssertionError("Workflow switch imported Widgets: " + fullname)

    sys.meta_path.insert(0, NoWidgetImports())
    import numpy as np
    from PySide6.QtCore import QCoreApplication, QEvent, QObject, QSettings, QUrl, qInstallMessageHandler
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from ui_qt.quick.application_bridge import QuickApplicationBridge
    from ui_qt.quick.plot_surface import PlotSurface, register_plot_surface

    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    if sys.platform == "win32":
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
        app.setFont(QFont("Segoe UI", 9))
    register_plot_surface()
    python_errors, qml_errors = [], []

    def python_error(kind, value, trace):
        python_errors.append("".join(traceback.format_exception(kind, value, trace)))
        traceback.print_exception(kind, value, trace)

    def qt_message(kind, context, message):
        if any(marker in message for marker in (
            "ReferenceError", "TypeError", "Binding loop", "Cannot assign", "Unable to assign", "Error:",
        )):
            qml_errors.append(str(message))
            print(message, file=sys.stderr)

    sys.excepthook = python_error
    previous_handler = qInstallMessageHandler(qt_message)

    def wait(predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            if predicate():
                app.processEvents()
                return
            time.sleep(.003)
        raise AssertionError("Timed out waiting for a workflow/frame state")

    class CountingBridge(QuickApplicationBridge):
        def __init__(self, **kwargs):
            self.rebuilds = []
            super().__init__(**kwargs)

        def _rebuild_plot(self):
            self.rebuilds.append(self._workflow)
            super()._rebuild_plot()

    class SwitchingCases(unittest.TestCase):
        def setUp(self):
            python_start, qml_start = len(python_errors), len(qml_errors)
            self.addCleanup(lambda: self.assertEqual(python_errors[python_start:], []))
            self.addCleanup(lambda: self.assertEqual(qml_errors[qml_start:], []))
            self.temporary = TemporaryDirectory(prefix="quick-switch-case-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name)
            self.settings = QSettings(str(self.root / "settings.ini"), QSettings.IniFormat)
            self.bridge = CountingBridge(settings=self.settings)
            self.addCleanup(self.bridge.close)

        def settle(self):
            # Give queued zero-delay draws and the debounced PL range handler
            # time to expose unintended work after the immediate switch.
            until = time.monotonic() + .20
            while time.monotonic() < until:
                app.processEvents()
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                time.sleep(.003)

        def idle_plot(self):
            wait(lambda: not self.bridge._pending and not self.bridge.snapshot["busy"]
                 and self.bridge._canvas is not None and self.bridge._maps
                 and self.bridge._canvas.frame_generation > 0
                 and not self.bridge._plot_timer.isActive()
                 and not self.bridge._pl_range_timer.isActive()
                 and not self.bridge._canvas.backend._timer.isActive())
            self.assertEqual(self.bridge._canvas.last_error, "")

        def dat(self, name="reflectance.dat", *, offset=0.):
            energy, gate = np.linspace(1.7, 2.1, 101), np.linspace(-2., 2., 9)
            peaks = 4 / (1 + ((energy - 1.80) / .014) ** 2)
            peaks += 2 / (1 + ((energy - 1.98) / .018) ** 2)
            values = (1 + gate[:, None] * .08) * peaks[None, :] + 1 + offset
            path = self.root / name
            lines = ["X\t" + "\t".join(map(str, gate))]
            lines.extend("\t".join(map(str, [x, *values[:, index]]))
                         for index, x in enumerate(energy))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return path

        def csv(self):
            wave, gate = np.linspace(610., 730., 101), np.linspace(-2., 2., 9)
            shape = 40 / (1 + ((wave - 650) / 4) ** 2)
            shape += 25 / (1 + ((wave - 700) / 6) ** 2)
            values = 10 + (1 + gate[:, None] * .08) * shape[None, :]
            path = self.root / "sample_PL.csv"
            np.savetxt(path, np.column_stack((gate * .5, gate, values)), delimiter=",",
                       header="Vbg,Vtg," + ",".join(map(str, wave)), comments="")
            return path

        def load(self, workflow, source):
            self.bridge.selectWorkflow(workflow)
            self.bridge.openFile(QUrl.fromLocalFile(str(source)))
            self.idle_plot()
            self.assertTrue(self.bridge.snapshot["canExport"], self.bridge.snapshot["status"])
            return self.bridge._cube

        def load_pair(self):
            pl = self.load("pl", self.csv())
            self.bridge.setParameter("gate", -1.)
            self.bridge.setParameter("xmin", float(pl.energy[8]))
            self.bridge.setParameter("xmax", float(pl.energy[-9]))
            self.idle_plot()
            self.bridge.invoke("spectrum-find")
            self.assertGreater(len(self.bridge._manual_states["pl"]["indices"]), 0)
            self.idle_plot()
            pl_context = self.context()
            self.load("drr", self.dat())
            self.bridge.setParameter("gate", 1.)
            self.bridge.invoke("spectrum-find")
            self.assertGreater(len(self.bridge._manual_states["drr"]["indices"]), 0)
            self.idle_plot()
            return {"pl": pl_context, "drr": self.context()}

        def context(self):
            bridge = self.bridge
            return dict(canvas=bridge._canvas, navigation=bridge._navigation,
                        axes=tuple(axis for axis, _ in bridge._maps),
                        spectra=tuple(line for _, line in bridge._spectra),
                        blitters=tuple(bridge._blitters), artists=tuple(bridge._manual_artists),
                        cube=bridge._cube, gate=bridge._values["gate"],
                        limits=tuple((axis.get_xlim(), axis.get_ylim()) for axis, _ in bridge._maps),
                        image_key=bridge._canvas.image.cacheKey(), draws=bridge._canvas.draw_count)

        def assert_context(self, context, *, same_image=True):
            bridge = self.bridge
            self.assertIs(bridge._canvas, context["canvas"])
            self.assertIs(bridge._navigation, context["navigation"])
            self.assertIs(bridge._cube, context["cube"])
            self.assertEqual(tuple(axis for axis, _ in bridge._maps), context["axes"])
            self.assertEqual(tuple(line for _, line in bridge._spectra), context["spectra"])
            self.assertEqual(tuple(bridge._blitters), context["blitters"])
            self.assertEqual(tuple(bridge._manual_artists), context["artists"])
            self.assertEqual(bridge._values["gate"], context["gate"])
            self.assertEqual(tuple((axis.get_xlim(), axis.get_ylim()) for axis, _ in bridge._maps), context["limits"])
            row = int(np.argmin(abs(bridge._cube.gate - context["gate"])))
            np.testing.assert_array_equal(bridge._spectra[0][1].get_ydata(), bridge._cube.Z[row])
            self.assertGreater(len(bridge._manual_artists[0][3].get_offsets()), 0)
            if same_image:
                self.assertEqual(bridge._canvas.image.cacheKey(), context["image_key"])
                self.assertEqual(bridge._canvas.draw_count, context["draws"])

        def native_window(self):
            engine = QQmlApplicationEngine()
            engine.setInitialProperties({"appBridge": self.bridge})
            qml = Path(__file__).resolve().parents[1] / "ui_qt/quick/qml/Application.qml"
            engine.load(QUrl.fromLocalFile(str(qml)))
            self.assertTrue(engine.rootObjects(), "\n".join(qml_errors))
            window = engine.rootObjects()[0]
            window.setWidth(1320)
            window.setHeight(900)
            surface = window.findChild(QObject, "app_plot_surface")
            self.assertIsInstance(surface, PlotSurface)

            def dispose():
                window.close()
                engine.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                app.processEvents()

            self.addCleanup(dispose)
            return window, surface

        def presented(self, surface):
            self.idle_plot()
            def current_frame_presented():
                canvas = self.bridge._canvas
                dpr = surface.window().devicePixelRatio()
                expected_size = (max(1, round(surface.width() * dpr)),
                                 max(1, round(surface.height() * dpr)))
                return (surface.canvas is canvas
                        and not getattr(surface, "_size_pending", False)
                        and not canvas.backend._timer.isActive()
                        and canvas.get_width_height(physical=True) == expected_size
                        and surface.presentedGeneration == canvas.frame_generation)
            wait(current_frame_presented)
            self.assertGreater(surface.presentedGeneration, 0)
            self.assertFalse(surface._image.isNull())
            self.assertEqual(surface._image.cacheKey(), self.bridge._canvas.image.cacheKey())

        def test_loaded_pl_drr_revisits_reuse_figures_data_zoom_and_overlays(self):
            contexts = self.load_pair()
            before = list(self.bridge.rebuilds)
            with patch("pandas.read_csv", side_effect=AssertionError("Switch reread CSV")) as csv_read, \
                 patch("ui_qt.quick.pl_workflow.load_pl_source", side_effect=AssertionError("Switch reloaded PL")) as pl_load, \
                 patch("ui_qt.quick.drr_workflow.load_drr", side_effect=AssertionError("Switch reloaded DRR")) as drr_load:
                for _ in range(4):
                    for workflow in ("pl", "drr"):
                        self.bridge.selectWorkflow(workflow)
                        self.settle()
                        self.assert_context(contexts[workflow])
                csv_read.assert_not_called()
                pl_load.assert_not_called()
                drr_load.assert_not_called()
            self.assertEqual(self.bridge.rebuilds, before)

        def test_all_other_workflow_visits_preserve_loaded_pl(self):
            contexts = self.load_pair()
            self.bridge.selectWorkflow("pl")
            self.settle()
            before = list(self.bridge.rebuilds)
            for workflow in ("cmp", "power", "mcd", "mcd_peak_shift", "shg", "slides", "tools"):
                with self.subTest(workflow=workflow):
                    self.bridge.selectWorkflow(workflow)
                    self.settle()
                    self.bridge.selectWorkflow("pl")
                    self.settle()
                    self.assert_context(contexts["pl"])
            self.assertEqual(self.bridge.rebuilds, before)

        def test_pending_color_change_paints_once_without_rebuilding_when_returning(self):
            contexts = self.load_pair()
            self.bridge.selectWorkflow("pl")
            self.settle()
            before = len(self.bridge.rebuilds)
            draws = self.bridge._canvas.draw_count
            axes = tuple(self.bridge._canvas.figure.axes)
            self.bridge.setParameter("cmap", "plasma")
            self.assertFalse(self.bridge._plot_timer.isActive())
            self.assertTrue(self.bridge._canvas.backend._timer.isActive())
            self.bridge.selectWorkflow("drr")
            self.settle()
            self.assert_context(contexts["drr"])
            self.assertEqual(len(self.bridge.rebuilds), before)
            self.bridge.selectWorkflow("pl")
            self.idle_plot()
            self.settle()
            self.assertEqual(self.bridge.rebuilds[before:], [])
            self.assertEqual(tuple(self.bridge._canvas.figure.axes), axes)
            self.assertEqual(self.bridge._canvas.draw_count, draws + 1)
            self.assertEqual(self.bridge._maps[0][0]._dptk_heatmap_params.cmap, "plasma")
            self.assertIs(self.bridge._canvas, contexts["pl"]["canvas"])
            self.bridge.selectWorkflow("drr")
            self.settle()
            self.assert_context(contexts["drr"])

        def test_pending_pl_viewport_change_does_not_rebuild_other_workflow(self):
            contexts = self.load_pair()
            self.bridge.selectWorkflow("pl")
            self.settle()
            cube = self.bridge._cube
            bounds = (float(cube.energy[12]), float(cube.energy[-13]))
            before = len(self.bridge.rebuilds)
            self.bridge._maps[0][0].set_xlim(bounds)
            self.assertTrue(self.bridge._pl_range_timer.isActive())
            self.bridge.selectWorkflow("drr")
            self.settle()
            self.assert_context(contexts["drr"])
            self.assertEqual(len(self.bridge.rebuilds), before)
            self.bridge.selectWorkflow("pl")
            self.idle_plot()
            self.settle()
            self.assertEqual(self.bridge.rebuilds[before:], ["pl"])
            np.testing.assert_allclose(self.bridge._maps[0][0].get_xlim(), bounds)

        def test_pending_drr_display_step_survives_leaving_before_debounce(self):
            contexts = self.load_pair()
            display = self.bridge.drrDisplayBridge
            before = len(self.bridge.rebuilds)
            self.assertTrue(display.stepNumber("axis.xmin", 1))
            self.assertTrue(display._pending)
            requested = display._numbers["axis.xmin"]
            self.bridge.selectWorkflow("pl")
            self.settle()
            self.assert_context(contexts["pl"])
            self.assertEqual(len(self.bridge.rebuilds), before)
            self.bridge.selectWorkflow("drr")
            self.idle_plot()
            self.settle()
            self.assertEqual(self.bridge.rebuilds[before:], ["drr"])
            self.assertAlmostEqual(self.bridge._maps[0][0].get_xlim()[0], requested)

        def test_theme_changed_away_redraws_each_loaded_workflow_once(self):
            contexts = self.load_pair()
            before = len(self.bridge.rebuilds)
            self.bridge.invoke("theme-dark")
            self.idle_plot()
            self.settle()
            self.assertEqual(self.bridge.rebuilds[before:], ["drr"])
            self.bridge.selectWorkflow("pl")
            self.idle_plot()
            self.settle()
            self.assertEqual(self.bridge.rebuilds[before:], ["drr", "pl"])
            from matplotlib.colors import to_rgba
            self.assertEqual(self.bridge._canvas.figure.get_facecolor(), to_rgba(self.bridge.colors["canvas"]))
            for workflow in ("drr", "pl", "drr", "pl"):
                self.bridge.selectWorkflow(workflow)
                self.settle()
                self.assertIs(self.bridge._canvas, contexts[workflow]["canvas"])
                self.assertEqual(self.bridge._canvas.figure.get_facecolor(), to_rgba(self.bridge.colors["canvas"]))
            self.assertEqual(self.bridge.rebuilds[before:], ["drr", "pl"])

        def test_clear_reload_does_not_restore_stale_plot_and_close_releases_canvases(self):
            contexts = self.load_pair()
            self.bridge.invoke("clear-files")
            self.settle()
            self.assertIsNone(self.bridge._cube)
            self.assertEqual(self.bridge._maps, [])
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.bridge.selectWorkflow("pl")
            self.settle()
            self.assert_context(contexts["pl"])
            self.bridge.selectWorkflow("drr")
            self.settle()
            self.assertIsNone(self.bridge._cube)
            self.assertEqual(self.bridge._maps, [])
            replacement = self.load("drr", self.dat("replacement.dat", offset=10.))
            self.assertIsNot(replacement, contexts["drr"]["cube"])
            self.assertNotEqual(float(replacement.Z[0, 0]), float(contexts["drr"]["cube"].Z[0, 0]))
            fresh_canvas = self.bridge._canvas
            self.bridge.selectWorkflow("pl")
            self.bridge.selectWorkflow("drr")
            self.idle_plot()
            self.assertIs(self.bridge._cube, replacement)
            self.assertIs(self.bridge._plot_cubes[0], replacement)
            self.bridge.close()
            for canvas in {contexts["pl"]["canvas"], contexts["drr"]["canvas"], fresh_canvas}:
                self.assertTrue(canvas.closed)
                self.assertIsNone(canvas.figure)
                self.assertTrue(canvas.image.isNull())
                self.assertFalse(canvas.backend._timer.isActive())

        def test_real_qml_revisits_present_matching_frames_and_resize_active_context(self):
            self.load_pair()
            window, surface = self.native_window()
            self.presented(surface)
            drr_canvas = self.bridge._canvas
            self.bridge.selectWorkflow("pl")
            self.presented(surface)
            pl_canvas = self.bridge._canvas
            drr_size = drr_canvas.get_width_height()
            before = len(self.bridge.rebuilds)
            pl_draws = pl_canvas.draw_count
            window.setWidth(1450)
            window.setHeight(960)
            self.presented(surface)
            self.assertEqual(pl_canvas.draw_count, pl_draws + 1)
            self.assertEqual(drr_canvas.get_width_height(), drr_size)
            self.assertNotEqual(pl_canvas.get_width_height(), drr_size)
            self.bridge.selectWorkflow("drr")
            self.presented(surface)
            self.assertIs(surface.canvas, drr_canvas)
            self.assertEqual(drr_canvas.get_width_height(), pl_canvas.get_width_height())
            self.bridge.selectWorkflow("cmp")
            self.settle()
            self.bridge.selectWorkflow("pl")
            self.presented(surface)
            self.assertIs(surface.canvas, pl_canvas)
            self.assertEqual(len(self.bridge.rebuilds), before)

        def test_real_qml_single_and_paired_revisits_avoid_transient_viewport_redraws(self):
            self.load_pair()
            self.bridge.setParameter("drrView", "pair")
            self.idle_plot()
            _window, surface = self.native_window()
            self.presented(surface)
            drr_canvas = self.bridge._canvas
            self.assertEqual(len(self.bridge._maps), 2)
            self.bridge.selectWorkflow("pl")
            self.presented(surface)
            pl_canvas = self.bridge._canvas
            self.assertNotEqual(pl_canvas.get_width_height(), drr_canvas.get_width_height())
            counts = {"pl": pl_canvas.draw_count, "drr": drr_canvas.draw_count}
            sizes = {"pl": pl_canvas.get_width_height(), "drr": drr_canvas.get_width_height()}
            rebuilds = list(self.bridge.rebuilds)
            requests = []
            canvas_type = type(pl_canvas)
            resize = canvas_type.set_logical_size
            def record_resize(canvas, width, height, dpr=1.):
                requests.append(("pl" if canvas is pl_canvas else "drr", width, height,
                                 canvas.get_width_height(), canvas.draw_count))
                return resize(canvas, width, height, dpr)
            with patch.object(canvas_type, "set_logical_size", record_resize):
                for _ in range(3):
                    for workflow, canvas in (("drr", drr_canvas), ("pl", pl_canvas)):
                        self.bridge.selectWorkflow(workflow)
                        self.presented(surface)
                        self.settle()
                        self.assertIs(surface.canvas, canvas)
                        self.assertEqual(canvas.draw_count, counts[workflow],
                                         f"{workflow}: cached sizes {sizes}; resize requests {requests}")
            self.assertEqual(self.bridge.rebuilds, rebuilds)

        def test_canvas_pixel_size_is_exact_and_repeated_same_size_does_not_redraw(self):
            from matplotlib.figure import Figure
            from ui_qt.quick.figure_canvas import QuickFigureCanvas
            canvas = QuickFigureCanvas(Figure(figsize=(2, 2), dpi=100))
            self.addCleanup(canvas.close)
            canvas.set_logical_size(995, 701)
            wait(lambda: canvas.frame_generation > 0 and not canvas.backend._timer.isActive())
            self.assertEqual(canvas.get_width_height(physical=True), (995, 701))
            self.assertEqual((canvas.image.width(), canvas.image.height()), (995, 701))
            draws = canvas.draw_count
            for _ in range(3):
                canvas.set_logical_size(995, 701)
                self.settle()
            self.assertEqual(canvas.draw_count, draws)

        def test_inactive_canvas_resize_cannot_modify_active_workflow_axes(self):
            contexts = self.load_pair()
            active = contexts["drr"]
            positions = tuple(tuple(axis.get_position().bounds) for axis in active["axes"])
            before = len(self.bridge.rebuilds)
            contexts["pl"]["canvas"].set_logical_size(713, 527, 1.25)
            self.settle()
            self.assert_context(active)
            self.assertEqual(tuple(tuple(axis.get_position().bounds) for axis in active["axes"]), positions)
            self.assertEqual(len(self.bridge.rebuilds), before)

    try:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SwitchingCases)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        print("QUICK_WORKFLOW_SWITCHING=" + json.dumps(dict(tests=result.testsRun, passed=result.wasSuccessful())))
        return 0 if result.wasSuccessful() else 1
    finally:
        qInstallMessageHandler(previous_handler)


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
