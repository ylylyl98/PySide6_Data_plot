"""Full SHG scientific/native parity in an isolated QGuiApplication process."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickShgTests(unittest.TestCase):
    def test_widgets_free_shg_workflow(self):
        with TemporaryDirectory(prefix="quick-shg-process-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                MPLBACKEND="agg", MPLCONFIGDIR=temporary, XDG_CONFIG_HOME=temporary,
                XDG_CACHE_HOME=temporary, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_SHG_TESTS=", result.stdout)


def _run_isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import importlib.abc
    import json
    import threading
    import time
    import traceback
    from unittest.mock import patch

    class NoWidgets(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {"PySide6.QtWidgets", "PySide6.QtQuickWidgets", "ui_qt.main_window", "ui_qt.common",
                "ui_qt.controllers_shg", "matplotlib.backends.backend_qt", "matplotlib.backends.backend_qtagg"}:
                raise AssertionError("SHG imported a Widgets dependency: " + fullname)
    sys.meta_path.insert(0, NoWidgets())
    from PySide6.QtCore import QObject, QSettings, QUrl, QPoint, Qt, qInstallMessageHandler
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication
    from PySide6.QtQuickControls2 import QQuickStyle
    from ui_qt.quick.shg_bridge import ShgBridge
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
        if any(marker in message for marker in ("ReferenceError", "TypeError", "Binding loop", "Cannot assign", "Unable to assign", "Error:")):
            qml_errors.append(str(message))
            print(message, file=sys.stderr)
    previous_handler = qInstallMessageHandler(qt_message)

    def wait(predicate, timeout=25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                app.processEvents()
                return
            time.sleep(.005)
        raise AssertionError("Timed out waiting for native SHG state")

    class ShgCases(unittest.TestCase):
        def setUp(self):
            python_start, qml_start = len(python_errors), len(qml_errors)
            self.addCleanup(lambda: self.assertEqual(python_errors[python_start:], [], "Uncaught Qt callback exception"))
            self.addCleanup(lambda: self.assertEqual(qml_errors[qml_start:], [], "QML runtime error"))
            self.temporary = TemporaryDirectory(prefix="quick-shg-case-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name)
            self.settings = QSettings(str(self.root / "isolated.ini"), QSettings.Format.IniFormat)
            self.settings.setValue("unrelated/preserved", 123)
            self.bridge = ShgBridge(settings=self.settings)
            self.addCleanup(self.bridge.close)

        def source(self, name="measurement.csv", phase=17.5, count=37, spike=False, failed=False, background=False):
            import numpy as np
            from core.shg_fit import evaluate_shg_angular_model
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            wave = np.linspace(507., 523., 321)
            angles = np.linspace(0., 180., count)
            baseline = 800. + 1.8 * (wave - 515.) + .04 * (wave - 515.) ** 2
            peak = np.exp(-.5 * ((wave - 515.) / .35) ** 2) / (.35 * np.sqrt(2. * np.pi))
            area = np.zeros(count) if background else evaluate_shg_angular_model(angles, 120., 900., phase)
            spectra = baseline[None, :] + area[:, None] * peak[None, :]
            if spike:
                spectra[3, int(np.argmin(abs(wave - 513.)))] += 4000.
            move = np.ones(count)
            if failed:
                move[5] = 0.
            table = np.column_stack((angles + .1, angles, np.full(count, .1), move, np.ones(count), spectra))
            header = "target_value,measured_value,position_error,move_ok,acquisition_ok," + ",".join(f"{value:.8f}" for value in wave)
            np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.14g")
            return path.resolve()

        def settled(self):
            wait(lambda: not self.bridge._load_timer.isActive() and not self.bridge._pending and not self.bridge.snapshot["busy"])

        def plotted(self):
            wait(lambda: self.bridge._canvas is not None and self.bridge._canvas.frame_generation > 0
                and not self.bridge._plot_timer.isActive() and not self.bridge._canvas.backend._timer.isActive())
            self.assertEqual(self.bridge._canvas.last_error, "")
            self.assertEqual(self.bridge.snapshot["error"], "", self.bridge.snapshot["status"])

        def loaded(self, path=None):
            path = path or self.source()
            self.bridge.openFiles("single", [QUrl.fromLocalFile(str(path))])
            self.settled()
            self.plotted()
            self.assertTrue(self.bridge.snapshot["canExport"], self.bridge.snapshot["status"])
            return path

        def compare(self, first=None, second=None):
            first = first or self.source("reference.csv", phase=8., count=37)
            second = second or self.source("sample.csv", phase=14., count=41)
            self.bridge.setParameter("mode", "compare")
            self.bridge.openFiles("sources", [QUrl.fromLocalFile(str(path)) for path in (first, second)])
            self.settled()
            self.plotted()
            return first, second

        def fields(self):
            return {field["key"]: field for group in self.bridge.snapshot["parameterGroups"] for field in group["fields"]}

        def test_00_lazy_widgets_free_defaults_and_settings_contract(self):
            self.assertNotIn("numpy", initial_imports)
            self.assertNotIn("matplotlib", initial_imports)
            self.assertIsNone(self.bridge.plotBackend)
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.assertEqual(self.fields()["halfRange"]["value"], 3.)
            self.assertEqual(self.settings.allKeys(), ["unrelated/preserved"])
            requested = []
            self.bridge.sourceDialogRequested.connect(lambda *args: requested.append(args))
            for role in ("single", "reference", "sample", "background", "backgroundA", "backgroundB", "settings"):
                self.bridge.invoke("choose-" + role)
            self.assertEqual(requested, [(role, False) for role in ("single", "reference", "sample", "background", "backgroundA", "backgroundB", "settings")])
            self.bridge.setParameter("cosmicWindow", 8)
            self.bridge.setParameter("cosmicMaxWidth", 15)
            self.assertEqual(self.bridge._values["cosmicWindow"], 9)
            self.assertEqual(self.bridge._values["cosmicMaxWidth"], 8)
            self.bridge.close()
            self.assertEqual(self.settings.allKeys(), ["unrelated/preserved"])

        def test_single_exact_domain_results_cosmic_exclusions_and_angle_transform(self):
            import numpy as np
            from dataclasses import replace
            from core.shg import load_shg_sweep_csv, process_shg_sweep
            path = self.loaded(self.source(spike=True, failed=True))
            self.assertEqual(len(self.bridge.snapshot["rows"]), 37)
            self.assertFalse(self.bridge._products.results[0].included[5])
            self.assertGreater(self.bridge._products.results[0].cosmic_pixels_removed.sum(), 0)
            for method in ("local_linear", "local_quadratic", "none"):
                self.bridge.setParameter("backgroundMethod", method)
                self.settled()
                self.plotted()
                products = self.bridge._products
                data = replace(load_shg_sweep_csv(str(path.parent), path.name), source_file=str(path))
                expected = process_shg_sweep(data, products.request.settings)
                for key in ("cleaned_spectra", "cosmic_ray_mask", "corrected", "baseline", "integrated_area", "area_uncertainty", "included"):
                    np.testing.assert_array_equal(getattr(products.results[0], key), getattr(expected, key), err_msg=method + ": " + key)
            self.bridge.setParameter("backgroundMethod", "local_quadratic")
            self.bridge.setParameter("includeFailed", True)
            self.bridge.setParameter("angleScale", 2.)
            self.bridge.setParameter("angleOffset", -12.)
            self.bridge.setParameter("angleWrap", "180")
            self.settled()
            result = self.bridge._products.results[0]
            np.testing.assert_allclose(result.measured_angle_deg, np.mod(result.data.measured_angle_deg * 2. - 12., 180.))
            self.assertTrue(result.included[5])
            self.bridge.setParameter("cosmicEnabled", False)
            self.settled()
            self.assertEqual(self.bridge._products.results[0].cosmic_pixels_removed.sum(), 0)

        def test_cursor_reuses_spectra_and_angle_marker_preserving_zoom(self):
            import numpy as np
            self.loaded(self.source(spike=True))
            canvas = self.bridge._canvas
            axes = tuple(canvas.figure.axes)
            artists = self.bridge._cursor_artists
            raw, corrected = artists["raw"], artists["corrected"]
            raw.set_xlim(512., 518.)
            corrected.set_ylim(-100., 2000.)
            self.bridge._angle_axis.set_xlim(15., 145.)
            canvas.draw()
            draws, blits = canvas.draw_count, canvas.blit_count
            products = self.bridge._products
            self.bridge.setParameter("angleCursor", 60.)
            self.plotted()
            self.assertEqual(tuple(canvas.figure.axes), axes)
            self.assertIs(self.bridge._products, products)
            self.assertEqual(canvas.draw_count, draws)
            self.assertEqual(canvas.blit_count, blits + 1)
            self.assertEqual(raw.get_xlim(), (512., 518.))
            self.assertEqual(corrected.get_ylim(), (-100., 2000.))
            self.assertEqual(self.bridge._angle_axis.get_xlim(), (15., 145.))
            result = products.results[0]
            index = int(np.argmin(abs(result.measured_angle_deg - 60.)))
            np.testing.assert_allclose(artists["raw_line"].get_ydata(), result.data.spectra[index])
            np.testing.assert_allclose(artists["cleaned_line"].get_ydata(), result.cleaned_spectra[index])
            np.testing.assert_allclose(artists["corrected_line"].get_ydata(), result.corrected[index])
            np.testing.assert_allclose(artists["marker"].get_offsets(), [[result.measured_angle_deg[index], result.integrated_area[index]]])
            self.bridge.setParameter("spectrumView", "cleaned")
            self.plotted()
            self.assertEqual(tuple(canvas.figure.axes), axes)
            self.assertFalse(artists["raw_line"].get_visible())
            self.assertTrue(artists["cleaned_line"].get_visible())
            self.assertTrue(all("Arial" in text.get_fontproperties().get_family()
                                for text in raw.get_legend().get_texts()))
            self.assertEqual(canvas.draw_count, draws)
            self.bridge.invoke("update")
            self.settled()
            self.plotted()
            self.assertIs(self.bridge._products, products)
            self.assertEqual(tuple(canvas.figure.axes), axes)
            self.assertEqual(self.bridge._angle_axis.get_xlim(), (15., 145.))

        def test_cursor_defers_paint_while_worker_export_holds_plot_font_lock(self):
            from core.typography import matplotlib_font_context
            self.loaded()
            acquired, release = threading.Event(), threading.Event()
            def export_context():
                with matplotlib_font_context():
                    acquired.set()
                    release.wait(5)
            worker = threading.Thread(target=export_context)
            worker.start()
            try:
                self.assertTrue(acquired.wait(5))
                generation = self.bridge._canvas.frame_generation
                self.bridge.setParameter("angleCursor", 50.)
                app.processEvents()
                self.assertTrue(worker.is_alive(), "SHG cursor waited for a worker's render lock")
                self.assertEqual(self.bridge._canvas.frame_generation, generation)
            finally:
                release.set()
                worker.join(5)
            self.plotted()
            self.assertGreater(self.bridge._canvas.frame_generation, generation)

        def test_fit_edits_reuse_axes_update_latest_curve_clear_and_restore_fit(self):
            import numpy as np
            from core.shg_fit import evaluate_shg_angular_model
            self.loaded(self.source(failed=True))
            results = self.bridge._products.results
            axes = tuple(self.bridge._canvas.figure.axes)
            curve = self.bridge._single_curve
            line, residual = curve["fit_line"], curve["residual"]
            curve["axis"].set_xlim(25., 145.)
            curve["axis"].set_ylim(0., 1500.)
            curve["residual_axis"].set_ylim(-20., 20.)
            self.bridge._cursor_artists["raw"].set_xlim(512., 518.)
            self.bridge._canvas.draw()
            draws, blits = self.bridge._canvas.draw_count, self.bridge._canvas.blit_count
            before = np.asarray(self.bridge._canvas.buffer_rgba()).copy()
            self.bridge.setParameter("fitMin", 20.)
            self.bridge.setParameter("fitMax", 150.)
            self.settled()
            self.plotted()
            fit = self.bridge._products.fits[0]
            self.assertIs(self.bridge._products.results, results)
            self.assertEqual(tuple(self.bridge._canvas.figure.axes), axes)
            self.assertIs(self.bridge._single_curve["fit_line"], line)
            self.assertEqual(self.bridge._canvas.draw_count, draws)
            self.assertGreater(self.bridge._canvas.blit_count, blits)
            outside = np.ones(before.shape[:2], bool)
            for axis in axes:
                box = axis.bbox
                outside[max(0, before.shape[0] - int(box.y1) - 4):min(before.shape[0], before.shape[0] - int(box.y0) + 5),
                        max(0, int(box.x0) - 4):min(before.shape[1], int(box.x1) + 5)] = False
            np.testing.assert_array_equal(np.asarray(self.bridge._canvas.buffer_rgba())[outside], before[outside])
            np.testing.assert_allclose(line.get_ydata(), evaluate_shg_angular_model(
                line.get_xdata(), fit.i0, fit.amplitude, fit.x_center_deg))
            np.testing.assert_allclose(residual.get_offsets(), np.column_stack((
                results[0].measured_angle_deg[fit.fit_mask], fit.residual[fit.fit_mask])))
            self.assertEqual(sum(row["fitIncluded"] for row in self.bridge.snapshot["rows"]), fit.point_count)
            self.bridge.setParameter("fitMax", 25.)
            self.settled()
            self.plotted()
            self.assertIsNone(self.bridge._products.fits[0])
            self.assertFalse(line.get_visible())
            self.assertEqual(len(residual.get_offsets()), 0)
            self.assertNotIn("SHG fit", [label.get_text() for label in curve["axis"].get_legend().get_texts()])
            self.bridge.setParameter("fitMax", 170.)
            self.settled()
            self.plotted()
            fit = self.bridge._products.fits[0]
            self.assertIsNotNone(fit)
            self.assertTrue(line.get_visible())
            self.assertIn("SHG fit", [label.get_text() for label in curve["axis"].get_legend().get_texts()])
            self.assertTrue(all("Arial" in text.get_fontproperties().get_family()
                                for text in curve["axis"].get_legend().get_texts()))
            self.assertEqual(tuple(self.bridge._canvas.figure.axes), axes)
            self.assertEqual(curve["axis"].get_xlim(), (25., 145.))
            self.assertEqual(curve["axis"].get_ylim(), (0., 1500.))
            self.assertEqual(curve["residual_axis"].get_ylim(), (-20., 20.))
            self.assertEqual(self.bridge._cursor_artists["raw"].get_xlim(), (512., 518.))
            self.bridge.setParameter("halfRange", 2.8)
            self.settled()
            self.plotted()
            self.assertIsNot(self.bridge._products.results, results)
            self.assertIsNot(self.bridge._angle_axis, curve["axis"])

        def test_auto_fit_edits_update_scale_static_ticks_and_invalidate_real_zoom(self):
            import numpy as np
            from core.shg_fit import evaluate_shg_angular_model
            self.loaded()
            canvas, curve = self.bridge._canvas, self.bridge._single_curve
            axis = curve["axis"]
            limits = axis.get_ylim()
            ticks = [text.get_text() for text in axis.get_yticklabels()]
            draws, blits = canvas.draw_count, canvas.blit_count
            before = np.asarray(canvas.buffer_rgba()).copy()
            label = axis.yaxis.label.get_window_extent(canvas.get_renderer()).expanded(1.3, 1.1)
            self.bridge.setParameter("fitMin", 20.)
            self.bridge.setParameter("fitMax", 150.)
            self.settled()
            self.plotted()
            self.assertTrue(axis.get_autoscaley_on())
            lower, upper = axis.dataLim.intervaly
            pad = (upper - lower) * axis.get_ymargin()
            np.testing.assert_allclose(axis.get_ylim(), (lower - pad, upper + pad))
            self.assertEqual([text.get_text() for text in axis.get_yticklabels()], ticks)
            self.assertTrue(all("Arial" in text.get_fontproperties().get_family()
                                for text in axis.get_yticklabels() + curve["residual_axis"].get_yticklabels()))
            self.assertEqual(canvas.draw_count, draws)
            self.assertGreater(canvas.blit_count, blits)
            fit = self.bridge._products.fits[0]
            line = curve["fit_line"]
            np.testing.assert_allclose(line.get_ydata(), evaluate_shg_angular_model(
                line.get_xdata(), fit.i0, fit.amplitude, fit.x_center_deg))
            result = self.bridge._products.results[0]
            np.testing.assert_allclose(curve["residual"].get_offsets(), np.column_stack((
                result.measured_angle_deg[fit.fit_mask], fit.residual[fit.fit_mask])))
            y0, y1 = max(0, before.shape[0] - int(label.y1) - 1), min(before.shape[0], before.shape[0] - int(label.y0) + 2)
            x0, x1 = max(0, int(label.x0) - 1), min(before.shape[1], int(label.x1) + 2)
            np.testing.assert_array_equal(np.asarray(canvas.buffer_rgba())[y0:y1, x0:x1], before[y0:y1, x0:x1])
            # A real viewport change must repaint the static grid and labels.
            axis.set_xlim(30., 140.)
            self.bridge.setParameter("fitMin", 25.)
            self.settled()
            self.plotted()
            self.assertGreater(canvas.draw_count, draws)
            self.assertEqual(axis.get_xlim(), (30., 140.))

        def test_comparison_twist_branch_unequal_grids_and_display_reuses_science(self):
            from core.shg_fit import fit_shg_twist_comparison
            self.compare()
            twist = self.bridge._products.twist
            self.assertIsNotNone(twist)
            expected = fit_shg_twist_comparison(*self.bridge._products.results, self.bridge._products.request.fit_settings)
            self.assertEqual(twist.to_dict(), expected.to_dict())
            self.assertAlmostEqual(twist.nearest_delta_x_center_deg, 6., delta=.02)
            self.assertAlmostEqual(twist.signed_twist_angle_deg, 4., delta=.02)
            results = self.bridge._products.results
            self.bridge.setParameter("comparisonView", "normalized")
            self.plotted()
            self.assertIs(self.bridge._products.results, results)
            self.assertEqual(self.bridge._canvas.figure.axes[0].get_ylabel(), "Normalized area")
            self.bridge.setParameter("phaseBranch", 1)
            self.settled()
            self.assertIs(self.bridge._products.results, results)
            self.assertAlmostEqual(self.bridge._products.twist.signed_twist_angle_deg, twist.signed_twist_angle_deg + 60., places=8)
            self.bridge.setParameter("fitMin", 20.)
            self.bridge.setParameter("fitMax", 150.)
            self.settled()
            self.assertLess(self.bridge._products.fits[0].point_count, 37)
            self.bridge.setParameter("fitMax", 25.)
            self.settled()
            self.assertIsNone(self.bridge._products.twist)
            self.assertIn("at least", self.bridge.snapshot["fitMessage"])
            self.assertTrue(self.bridge.snapshot["canExport"])

        def test_external_background_each_role_and_inactive_background_is_not_required(self):
            import numpy as np
            from core.shg import process_shg_sweep
            first, second = self.compare()
            background_a = self.source("background-a.csv", count=3, background=True)
            background_b = self.source("background-b.csv", count=4, background=True)
            self.bridge.setParameter("backgroundMethod", "external")
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.assertIn("each SHG", self.bridge.snapshot["error"])
            self.bridge.openFiles("backgroundA", [QUrl.fromLocalFile(str(background_a))])
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.bridge.openFiles("backgroundB", [QUrl.fromLocalFile(str(background_b))])
            self.settled()
            self.plotted()
            products = self.bridge._products
            for result, name in zip(products.results, (background_a, background_b)):
                from dataclasses import replace
                from core.shg import load_shg_sweep_csv
                background = replace(load_shg_sweep_csv(str(name.parent), name.name), source_file=str(name))
                expected = process_shg_sweep(result.data, products.request.settings, background=background)
                np.testing.assert_array_equal(result.integrated_area, expected.integrated_area)
                self.assertEqual(result.background_file, str(name))
            self.bridge.setParameter("backgroundMethod", "local_quadratic")
            background_a.unlink()
            background_b.unlink()
            self.settled()
            self.assertTrue(self.bridge.snapshot["canExport"])
            self.assertEqual(len(self.bridge._products.signatures), 2)

        def test_real_export_values_metadata_unique_packages_history_and_settings_roundtrip(self):
            import csv
            import numpy as np
            self.bridge.setParameter("backgroundMethod", "local_quadratic")
            first, second = self.compare()
            self.bridge.setParameter("fitMin", 10.)
            self.bridge.setParameter("phaseBranch", -1)
            self.settled()
            expected = self.bridge._products
            self.bridge.invoke("export")
            self.assertTrue(self.bridge.snapshot["busy"])
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.settled()
            self.assertEqual(self.bridge.snapshot["error"], "", self.bridge.snapshot["status"])
            paths = list(map(Path, self.bridge.snapshot["resultPaths"]))
            self.assertEqual(len(paths), 8)
            self.assertTrue(all(path.is_file() and path.is_relative_to(self.root / "Processed Data" / "SHG") for path in paths))
            package = paths[0].parent
            metadata_path = next(path for path in paths if path.suffix == ".json" and "twist_fit" in json.loads(path.read_text(encoding="utf-8")))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["package"], package.name)
            self.assertEqual(metadata["workflow"], "SHG")
            self.assertEqual(metadata["twist_fit"]["phase_branch"], -1)
            self.assertEqual({item["role"] for item in metadata["inputs"]}, {"reference", "sample"})
            combined = next(path for path in paths if "combined" in path.stem)
            with combined.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 78)
            values = [float(row["background_subtracted_integrated_area_counts_nm"]) for row in rows]
            np.testing.assert_allclose(sorted(values), sorted(np.concatenate([result.integrated_area for result in expected.results])), rtol=1e-10)
            self.assertTrue(all(item["status"] == "processed" for item in self.bridge.snapshot["catalog"]))
            roles = {item["path"]: item["roles"] for item in self.bridge.snapshot["catalog"]}
            self.assertIn("reference", roles[str(first)])
            self.assertIn("sample", roles[str(second)])
            self.bridge.setParameter("historyFilter", "new")
            self.assertEqual(self.fields()["source.reference"]["value"], str(first))
            self.assertTrue(self.bridge.snapshot["canExport"])
            self.bridge.setParameter("phaseBranch", 2)
            self.bridge.setParameter("halfRange", 2.)
            self.settled()
            self.bridge.openFiles("settings", [QUrl.fromLocalFile(str(metadata_path))])
            self.settled()
            self.assertEqual(self.bridge._products.request.settings, expected.request.settings)
            self.assertEqual(self.bridge._products.request.fit_settings, expected.request.fit_settings)
            self.bridge.invoke("export")
            self.settled()
            self.assertNotEqual(Path(self.bridge.snapshot["resultPaths"][0]).parent, package)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertEqual(self.settings.allKeys(), ["unrelated/preserved"])

        def test_fit_disabled_pair_exports_both_csv_and_settings_without_twist(self):
            self.bridge.setParameter("fitEnabled", False)
            self.compare()
            self.assertIsNone(self.bridge._products.twist)
            self.bridge.invoke("export")
            self.settled()
            self.assertEqual(len(self.bridge.snapshot["resultPaths"]), 4)
            self.assertEqual(sum(Path(name).suffix == ".json" for name in self.bridge.snapshot["resultPaths"]), 2)
            for name in self.bridge.snapshot["resultPaths"]:
                if name.endswith(".json"):
                    self.assertIsNone(json.loads(Path(name).read_text(encoding="utf-8"))["angular_fit"])

        def test_fit_exclusion_weights_and_disabled_fit_settings_are_reproducible(self):
            from core.shg_fit import fit_shg_angular_result
            self.bridge.setParameter("cosmicWindow", 21)
            self.bridge.setParameter("cosmicMaxWidth", 15)
            self.loaded(self.source(failed=True))
            original = self.bridge._products.results
            self.assertFalse(self.bridge._products.fits[0].fit_mask[5])
            self.bridge.setParameter("fitIncludeExcluded", True)
            self.bridge.setParameter("fitWeights", False)
            self.bridge.setParameter("fitMin", 10.)
            self.bridge.setParameter("fitMax", 165.)
            self.settled()
            self.assertIs(self.bridge._products.results, original)
            self.assertTrue(self.bridge._products.fits[0].fit_mask[5])
            expected = fit_shg_angular_result(original[0], self.bridge._products.request.fit_settings)
            self.assertEqual(self.bridge._products.fits[0].to_dict(), expected.to_dict())
            self.bridge.setParameter("fitEnabled", False)
            self.settled()
            fit_settings = self.bridge._products.request.fit_settings
            self.bridge.invoke("export")
            self.settled()
            settings_path = next(name for name in self.bridge.snapshot["resultPaths"] if name.endswith(".json"))
            self.bridge.setParameter("fitEnabled", True)
            self.bridge.setParameter("fitMin", 0.)
            self.bridge.setParameter("cosmicWindow", 3)
            self.bridge.openFiles("settings", [QUrl.fromLocalFile(settings_path)])
            self.settled()
            self.assertEqual(self.bridge._products.request.fit_settings, fit_settings)
            self.assertEqual(self.bridge._values["cosmicWindow"], 21)
            self.assertEqual(self.bridge._values["cosmicMaxWidth"], 15)

        def test_same_basename_external_sources_export_exact_identities_and_roles(self):
            with TemporaryDirectory(prefix="quick-shg-external-") as external:
                first = self.source("inside/same.csv", phase=8.)
                second = self.source(str(Path(external) / "same.csv"), phase=14.)
                self.bridge.setExperimentFolder(str(self.root))
                self.compare(first, second)
                self.bridge.invoke("export")
                self.settled()
                self.assertEqual(self.bridge.snapshot["error"], "")
                paths = list(map(Path, self.bridge.snapshot["resultPaths"]))
                self.assertEqual(len(paths), 8)
                self.assertEqual(len(set(paths)), 8)
                roles = {item["path"]: item["roles"] for item in self.bridge.snapshot["catalog"]}
                self.assertIn("reference", roles[str(first)])
                self.assertIn("sample", roles[str(second)])
                metadata = next(json.loads(path.read_text(encoding="utf-8")) for path in paths
                    if path.suffix == ".json" and "twist_summary" in path.stem)
                self.assertEqual({item["source_path"] for item in metadata["inputs"]}, {str(first), str(second)})
                self.assertEqual(next(item for item in metadata["inputs"] if item["role"] == "reference")["source_relative_path"], "inside/same.csv")

        def test_cancel_and_close_cooperatively_discard_inflight_processing(self):
            started, stopped = threading.Event(), threading.Event()
            path = self.source()
            def cooperative(request, cancellation, previous):
                started.set()
                try:
                    while True:
                        cancellation.check()
                        time.sleep(.005)
                finally:
                    stopped.set()
            with patch("ui_qt.quick.shg_workflow.load_shg", side_effect=cooperative):
                self.bridge.openFiles("single", [QUrl.fromLocalFile(str(path))])
                wait(started.is_set)
                self.bridge.invoke("cancel")
                wait(stopped.is_set)
                self.settled()
                self.assertFalse(self.bridge.snapshot["canExport"])
                started.clear()
                stopped.clear()
                self.bridge.invoke("update")
                wait(started.is_set)
                self.bridge.close()
                self.assertTrue(stopped.is_set())
                self.assertEqual(self.bridge._pending, {})
                self.bridge.close()

        def test_changed_missing_source_and_failed_writer_do_not_publish_history(self):
            from core.shg_history import scan_shg_history
            path = self.loaded()
            before = path.stat()
            content = path.read_bytes()
            changed = content.replace(b"800", b"801", 1)
            self.assertNotEqual(changed, content)
            self.assertEqual(len(changed), len(content))
            path.write_bytes(changed)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            self.bridge.invoke("export")
            self.settled()
            self.assertIn("changed after processing", self.bridge.snapshot["error"])
            self.assertEqual(list(self.root.rglob("*_settings.json")), [])
            self.bridge.invoke("reload")
            self.settled()
            self.assertEqual(self.bridge.snapshot["error"], "")
            def fail(folder, **kwargs):
                target = Path(kwargs["processed_name"])
                target.mkdir(parents=True)
                (target / "partial_settings.json").write_text('{"workflow":"SHG"}', encoding="utf-8")
                raise OSError("Synthetic SHG writer failure")
            with patch("core.export.export_shg_results", side_effect=fail):
                self.bridge.invoke("export")
                self.settled()
            self.assertIn("Synthetic SHG writer failure", self.bridge.snapshot["error"])
            self.assertEqual(list(self.root.rglob("*_settings.json")), [])
            self.assertEqual(scan_shg_history(self.root, [str(path)]).processed_at, {})
            self.assertEqual(list(self.root.glob(".shg-export-*")), [])
            self.bridge.invoke("reload")
            self.settled()
            path.unlink()
            self.bridge.invoke("export")
            self.settled()
            self.assertIn("missing", self.bridge.snapshot["error"])
            self.assertFalse(self.bridge.snapshot["canExport"])

        def test_latest_request_wins_clear_invalidates_and_export_locks_parameters(self):
            from ui_qt.quick.shg_workflow import load_shg, export_shg
            first, latest = self.source("first.csv", phase=8.), self.source("latest.csv", phase=14.)
            started, release = threading.Event(), threading.Event()
            self.addCleanup(release.set)
            def delayed(request, *args):
                if any("first.csv" in name for _, name in request.sources):
                    started.set()
                    if not release.wait(5):
                        raise AssertionError("Synthetic SHG load was not released")
                return load_shg(request, *args)
            completions = []
            self.bridge.operationCompleted.connect(lambda *args: completions.append(args))
            with patch("ui_qt.quick.shg_workflow.load_shg", side_effect=delayed):
                self.bridge.openFiles("single", [QUrl.fromLocalFile(str(first))])
                wait(started.is_set)
                self.bridge.openFiles("single", [QUrl.fromLocalFile(str(latest))])
                wait(lambda: not self.bridge.snapshot["busy"])
                release.set()
                self.settled()
            self.assertEqual(self.bridge._products.results[0].data.source_file, str(latest))
            self.assertEqual([kind for kind, _ in completions], ["load"])
            started.clear()
            release.clear()
            def delayed_export(*args):
                started.set()
                if not release.wait(5):
                    raise AssertionError("Synthetic SHG export was not released")
                return export_shg(*args)
            with patch("ui_qt.quick.shg_workflow.export_shg", side_effect=delayed_export):
                self.bridge.invoke("export")
                wait(started.is_set)
                self.bridge.setParameter("peakCenter", 516.)
                self.bridge.openFiles("single", [QUrl.fromLocalFile(str(first))])
                self.assertEqual(self.bridge._values["peakCenter"], 515.)
                self.assertEqual(self.bridge._sources["single"], str(latest))
                release.set()
                self.settled()
            self.bridge.invoke("clear-files")
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.assertFalse(self.bridge.snapshot["inspectionVisible"])

        def test_native_qml_controls_plot_table_keyboard_and_narrow_layout(self):
            from PySide6.QtQml import QQmlApplicationEngine
            from PySide6.QtCore import QCoreApplication, QEvent
            from PySide6.QtGui import QKeyEvent, QMouseEvent
            from shiboken6 import isValid
            from ui_qt.quick.plot_surface import register_plot_surface
            register_plot_surface()
            colors = dict(surface="#F3F5F8", control="#FFFFFF", text="#1D293B", muted="#46566D",
                border="#C6D0DE", accent="#2167C5", hover="#F3F6FA", pressed="#E5EBF3", disabled="#8A97A9",
                selectedText="#FFFFFF", inset="#E9EDF3", divider="#D9E1EB", focus="#155DBC", accentText="#155DBC",
                selected="#E5EFFD", selectedBorder="#B7D0F3", error="#B42318")
            self.loaded()
            engine = QQmlApplicationEngine()
            engine.setInitialProperties(dict(bridge=self.bridge, colors=colors))
            qml_folder = Path(__file__).resolve().parents[1] / "ui_qt" / "quick" / "qml"
            engine.loadData(b'''import QtQuick
import QtQuick.Window
import QtQuick.Layouts
import "."
Window {
    id: window
    required property var bridge
    required property var colors
    width: 1050; height: 740; visible: true
    RowLayout {
        anchors.fill: parent; spacing: 0
        ShgParameters { objectName: "parameters"; bridge: window.bridge; colors: window.colors; Layout.preferredWidth: window.width < 750 ? 220 : 300; Layout.fillHeight: true }
        ShgWorkspace { objectName: "workspace"; bridge: window.bridge; colors: window.colors; Layout.fillWidth: true; Layout.fillHeight: true }
    }
}''', QUrl.fromLocalFile(str(qml_folder / "ShgNativeTest.qml")))
            self.assertTrue(engine.rootObjects(), "Native SHG QML failed to load")
            window = engine.rootObjects()[0]
            self.addCleanup(lambda: window.close() if isValid(window) else None)
            wait(lambda: window.isVisible())
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
                # Newly visible Flow children need layout polish before their
                # scene coordinates identify the intended mouse target.
                item.parentItem().ensurePolished()
                self.assertTrue(item.isVisible(), name)
                self.assertTrue(item.isEnabled(), name)
                presented = []
                connection = window.frameSwapped.connect(lambda: presented.append(True))
                try:
                    window.update()
                    wait(lambda: bool(presented))
                finally:
                    QObject.disconnect(connection)
                point = item.mapToScene(item.boundingRect().center())
                global_point = window.mapToGlobal(point.toPoint()).toPointF()
                QCoreApplication.sendEvent(window, QMouseEvent(QEvent.Type.MouseButtonPress, point, global_point,
                    Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
                QCoreApplication.sendEvent(window, QMouseEvent(QEvent.Type.MouseButtonRelease, point, global_point,
                    Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))
                app.processEvents()
                return item
            surface = window.findChild(QObject, "shg_plot_surface")
            self.assertIs(surface.property("backend"), self.bridge.plotBackend)
            wait(lambda: surface.property("presentedGeneration") == self.bridge._canvas.frame_generation
                and surface.property("presentedGeneration") > 0)
            click("shg_table")
            self.assertTrue(self.bridge.snapshot["showTable"])
            self.assertTrue(window.findChild(QObject, "shg_results_table").isVisible())
            self.assertEqual(len(self.bridge.snapshot["rows"]), 37)
            review = os.environ.get("DPTK_SHG_REVIEW_DIR", "")
            if review:
                self.plotted()
                wait(lambda: surface.property("presentedGeneration") == self.bridge._canvas.frame_generation)
                Path(review).mkdir(parents=True, exist_ok=True)
                self.assertTrue(window.grabWindow().save(str(Path(review) / "shg-wide.png")))
            self.bridge.invoke("next-angle")
            self.assertEqual(self.bridge.snapshot["angleCursor"], 5.)
            self.bridge.invoke("previous-angle")
            self.assertEqual(self.bridge.snapshot["angleCursor"], 0.)
            item = window.findChild(QObject, "shg_table")
            item.forceActiveFocus()
            QCoreApplication.sendEvent(window, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " "))
            QCoreApplication.sendEvent(window, QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " "))
            app.processEvents()
            self.assertFalse(self.bridge.snapshot["showTable"])
            window.setWidth(650)
            window.setHeight(520)
            app.processEvents()
            self.assertLessEqual(window.findChild(QObject, "parameters").width(), 221)
            self.assertGreater(window.findChild(QObject, "workspace").width(), 400)
            self.assertGreater(surface.width(), 400)
            wait(lambda: self.bridge.snapshot["compactInspection"] and not self.bridge._plot_timer.isActive()
                and not self.bridge._canvas.backend._timer.isActive())
            self.assertEqual(len(self.bridge._canvas.figure.axes), 3)
            click("shg_corrected_panel")
            self.plotted()
            self.assertEqual(self.bridge._canvas.figure.axes[0].get_ylabel(), "Corrected\ncounts")
            click("shg_raw_panel")
            self.plotted()
            self.assertEqual(self.bridge._canvas.figure.axes[0].get_ylabel(), "Counts")
            click("shg_compare")
            self.assertEqual(self.bridge.snapshot["mode"], "compare")
            self.assertFalse(self.bridge.snapshot["inspectionVisible"])
            click("shg_single")
            self.settled()
            self.plotted()
            requested = []
            self.bridge.sourceDialogRequested.connect(lambda *args: requested.append(args))
            click("app_parameter_choose-single")
            self.assertEqual(requested, [("single", False)])
            if review:
                wait(lambda: surface.property("presentedGeneration") == self.bridge._canvas.frame_generation)
                self.assertTrue(window.grabWindow().save(str(Path(review) / "shg-narrow.png")))
            window.close()
            engine.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
            self.assertNotIn("PySide6.QtWidgets", sys.modules)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ShgCases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    qInstallMessageHandler(previous_handler)
    print(f"QUICK_SHG_TESTS={result.testsRun}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
