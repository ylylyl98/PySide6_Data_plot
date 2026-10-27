"""Native Compare scientific parity in a fresh QGuiApplication-only process."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickCompareTests(unittest.TestCase):
    def test_widgets_free_compare_workflow(self):
        with TemporaryDirectory(prefix="quick-compare-process-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                MPLBACKEND="agg", MPLCONFIGDIR=temporary, XDG_CONFIG_HOME=temporary,
                XDG_CACHE_HOME=temporary, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=150)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_COMPARE_TESTS=", result.stdout)
        print(result.stdout.strip())


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
            if fullname in {"PySide6.QtWidgets", "PySide6.QtQuickWidgets", "ui_qt.common", "ui_qt.main_window",
                "ui_qt.controllers_compare", "matplotlib.backends.backend_qt", "matplotlib.backends.backend_qtagg"}:
                raise AssertionError("Compare imported a Widgets dependency: " + fullname)
    sys.meta_path.insert(0, NoWidgets())
    from PySide6.QtCore import QByteArray, QCoreApplication, QEvent, QObject, QSettings, QUrl, qInstallMessageHandler
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication
    from PySide6.QtQuickControls2 import QQuickStyle
    from ui_qt.quick.compare_bridge import CompareBridge
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

    def wait(predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                app.processEvents()
                return
            time.sleep(.005)
        raise AssertionError("Timed out waiting for native Compare state")

    class CompareCases(unittest.TestCase):
        def setUp(self):
            python_start, qml_start = len(python_errors), len(qml_errors)
            self.addCleanup(lambda: self.assertEqual(python_errors[python_start:], [], "Uncaught Qt callback exception"))
            self.addCleanup(lambda: self.assertEqual(qml_errors[qml_start:], [], "QML runtime error"))
            self.temporary = TemporaryDirectory(prefix="quick-compare-case-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name).resolve()
            self.settings = QSettings(str(self.root / "isolated.ini"), QSettings.Format.IniFormat)
            self.bridge = CompareBridge(settings=self.settings)
            self.addCleanup(self.bridge.close)

        def files(self, names=None):
            import numpy as np
            names = names or [f"sample_PL_{key}.csv" for key in ("KK", "KKp", "KpK", "KpKp")]
            energy = np.linspace(1.7, 2.7, 41)
            gate = np.linspace(-2, 2, 9)
            paths = []
            for index, name in enumerate(names):
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                intensity = (100 + index * 14 + (gate[:, None] + 2) * (index + 1) * 7
                             + 80 * np.exp(-((energy[None, :] - 2.15 - index * .025) / .1) ** 2))
                table = np.column_stack((gate, gate * .3 + 1, np.zeros_like(gate), intensity))
                header = "Vbg,Vtg,Vbias," + ",".join(f"{1240 / value:.12g}" for value in energy)
                np.savetxt(path, table, delimiter=",", header=header, comments="")
                paths.append(path.resolve())
            return paths

        def loaded(self, paths=None):
            paths = paths or self.files()
            self.bridge.openFiles("sources", [QUrl.fromLocalFile(str(path)) for path in paths])
            wait(lambda: not self.bridge.snapshot["busy"] and not self.bridge._pending)
            self.assertEqual(self.bridge.snapshot["error"], "", self.bridge.snapshot["status"])
            self.assertTrue(self.bridge._cubes, self.bridge.snapshot["status"])
            self.plotted()
            return paths

        def plotted(self):
            wait(lambda: self.bridge._canvas is not None and self.bridge._canvas.frame_generation > 0
                and not self.bridge._plot_timer.isActive() and not self.bridge._canvas.backend._timer.isActive())
            self.assertEqual(self.bridge.snapshot["error"], "")
            self.assertEqual(self.bridge._canvas.last_error, "")

        def fields(self):
            return {field["key"]: field for group in self.bridge.snapshot["parameterGroups"] for field in group["fields"]}

        def test_00_lazy_constructor_and_widgets_free_contract(self):
            self.assertNotIn("numpy", initial_imports)
            self.assertNotIn("matplotlib", initial_imports)
            self.assertNotIn("core.data_io", initial_imports)
            self.assertIsNone(self.bridge.plotBackend)
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.assertEqual(self.bridge.snapshot["parameterGroups"][0]["key"], "data")
            signals = []
            self.bridge.sourceDialogRequested.connect(lambda *args: signals.append(args))
            self.bridge.invoke("select-files")
            self.bridge.invoke("choose-channel:KpK")
            self.assertEqual(signals, [("sources", True), ("channel:KpK", False)])
            self.assertEqual(self.bridge.angleReferences, dict(inK=0., inKp=45., outK=0., outKp=45., angleTolerance=15.))
            self.bridge.setParameter("outKp", 72.5)
            self.assertEqual(self.bridge.angleReferences["outKp"], 72.5)
            self.assertFalse(any(name in sys.modules for name in ("PySide6.QtWidgets", "ui_qt.main_window", "ui_qt.common")))

        def test_coherent_four_channel_load_shared_background_and_independent_curves(self):
            import numpy as np
            from core.data_io import CompareSelection, load_compare_cubes
            from core.processing import background_correct_cube, estimate_constant_background
            paths = self.loaded()
            expected = load_compare_cubes(str(self.root), CompareSelection.from_mapping(self.bridge._assignments))
            background = estimate_constant_background({key: expected[key] for key in ("KK", "KKp")}, percentile=1.)
            self.assertAlmostEqual(self.bridge.snapshot["background"], background)
            self.assertEqual(len(self.bridge.snapshot["sourceGroups"]), 1)
            self.assertEqual(len(self.bridge._cubes), 2)
            self.assertEqual(set(self.bridge._derived), {"KK", "KKp"})
            for key in ("KK", "KKp"):
                np.testing.assert_allclose(self.bridge._derived[key].Z, background_correct_cube(expected[key], background).Z)
            self.bridge.setParameter("preset", "all")
            wait(lambda: not self.bridge._pending)
            self.plotted()
            self.assertEqual(len(self.bridge._cubes), 4)
            self.assertEqual(len(self.bridge._maps), 4)
            self.assertEqual(len(self.bridge._lines), 4)
            self.assertEqual(self.bridge._assignments, dict(zip(("KK", "KKp", "KpK", "KpKp"), map(str, paths))))
            hosts = [axis for axis in self.bridge._canvas.figure.axes if getattr(axis, "_dptk_inspection_header", False)]
            self.assertEqual(len(hosts), 4)
            self.assertTrue(all(len(axis.child_axes) == 1 for axis in hosts))
            self.assertTrue(all(child.xaxis.get_ticks_position() == "top" for axis in hosts for child in axis.child_axes))

        def test_domain_vp_and_independent_limits_fixed_bounds_and_manual_background(self):
            import numpy as np
            from core.processing import valley_polarization_cube
            self.loaded()
            self.bridge.setParameter("autoBackground", False)
            self.bridge.setParameter("background", 20.)
            self.bridge.setParameter("colorMin", 1.234567890123)
            self.bridge.setParameter("colorMax", 300.)
            self.bridge.setParameter("fix.colorMin", True)
            self.bridge.setParameter("fix.colorMax", True)
            self.bridge.setParameter("view", "vp")
            self.plotted()
            expected = valley_polarization_cube(self.bridge._cubes["KK"], self.bridge._cubes["KKp"], background=20.)
            np.testing.assert_allclose(self.bridge._derived["VP"].Z, expected.Z, equal_nan=True)
            self.bridge.setParameter("colorMin", -.4)
            self.bridge.setParameter("colorMax", .6)
            self.bridge.setParameter("fix.colorMin", True)
            self.bridge.setParameter("fix.colorMax", True)
            self.bridge.invoke("auto-color")
            self.assertEqual((self.bridge._values["vpMin"], self.bridge._values["vpMax"]), (-.4, .6))
            self.bridge.setParameter("view", "intensity")
            self.plotted()
            self.assertEqual((self.bridge._values["vmin"], self.bridge._values["vmax"]), (1.234567890123, 300.))
            self.bridge.invoke("auto-color")
            self.assertEqual(self.bridge._values["vmin"], 1.234567890123)
            self.assertEqual(self.bridge._values["vmax"], 300.)

        def test_gate_reuses_map_axes_and_agg_static_frame_with_exact_scientific_values(self):
            import numpy as np
            from core.processing import nearest_gate_spectrum
            self.loaded()
            canvas = self.bridge._canvas
            maps = dict(self.bridge._maps)
            full_draws, blits, generation = canvas.draw_count, canvas.blit_count, canvas.frame_generation
            old_image = canvas.image
            self.bridge.setParameter("gate", 1.24)
            self.assertEqual(canvas.draw_count, full_draws)
            self.assertEqual(canvas.blit_count, blits + 1)
            self.assertGreater(canvas.frame_generation, generation)
            self.assertNotEqual(canvas.image, old_image)
            self.assertEqual(self.bridge._maps, maps)
            for key, cube in self.bridge._derived.items():
                gate, expected = nearest_gate_spectrum(cube, 1.24)
                np.testing.assert_allclose(self.bridge._lines[key].get_ydata(), expected)
                self.assertEqual(list(self.bridge._gate_lines[key].get_ydata()), [gate, gate])
            selected_gate = float(next(iter(self.bridge._derived.values())).gate[
                np.argmin(np.abs(next(iter(self.bridge._derived.values())).gate - 1.24))])
            self.assertEqual(self.bridge._values["gate"], selected_gate)
            self.bridge.invoke("next-gate")
            self.assertGreater(self.bridge._values["gate"], selected_gate)
            self.bridge.invoke("previous-gate")
            self.assertEqual(self.bridge._values["gate"], selected_gate)

        def test_spectrum_y_follows_gate_x_window_and_vp_data(self):
            import numpy as np
            self.loaded()
            self.bridge.setParameter("gate", 2.)
            axis = self.bridge._spectrum
            full = axis.get_ylim()
            axis.set_xlim(1.72, 1.85)
            narrowed = axis.get_ylim()
            self.assertLess(narrowed[1], full[1] - 20.)
            self.bridge.setParameter("gate", -2.)
            self.assertLess(axis.get_ylim()[1], narrowed[1] - 10.)
            self.bridge._navigation.push_current()
            axis.set_ylim(5., 6.)
            self.bridge._navigation.push_current()
            self.bridge.setParameter("gate", 2.)
            self.assertEqual(axis.get_ylim(), (5., 6.))
            self.bridge.invoke("home")
            self.assertTrue(axis.get_autoscaley_on())
            np.testing.assert_allclose(axis.get_ylim(), narrowed)
            self.bridge.setParameter("view", "vp")
            self.plotted()
            finite = np.concatenate([line.get_ydata() for line in self.bridge._lines.values()])
            low, high = self.bridge._spectrum.get_ylim()
            self.assertLess(low, np.min(finite))
            self.assertGreater(high, np.max(finite))
            self.assertLess(high - low, 1.)

        def test_color_edits_reuse_axes_and_restore_unclipped_source_values(self):
            import numpy as np
            self.loaded()
            maps = dict(self.bridge._maps)
            renders = dict(self.bridge._heatmap_renders)
            maps["KK"].set_xlim(1.9, 2.3)
            maximum = self.bridge._values["vmax"]
            self.bridge.setParameter("colorMax", maximum * .7)
            self.plotted()
            self.assertEqual(self.bridge._maps, maps)
            self.assertEqual(maps["KK"].get_xlim(), (1.9, 2.3))
            self.bridge.setParameter("colorMax", maximum * 1.1)
            self.plotted()
            self.assertEqual(self.bridge._heatmap_renders, renders)
            for key, render in renders.items():
                np.testing.assert_allclose(render.primary.get_array(), np.minimum(self.bridge._derived[key].Z, maximum * 1.1))

        def test_gate_defers_paint_while_worker_export_holds_plot_font_lock(self):
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
                self.bridge.setParameter("gate", 1.)
                self.assertTrue(worker.is_alive(), "Gate callback waited for a worker's render lock")
                self.assertEqual(self.bridge._canvas.frame_generation, generation)
                self.assertTrue(self.bridge._canvas.backend._timer.isActive())
            finally:
                release.set()
                worker.join(5)
            self.plotted()
            self.assertGreater(self.bridge._canvas.frame_generation, generation)

        def test_manual_assignment_swap_hidden_channels_and_clear_survive_refresh(self):
            paths = self.loaded()
            unknown = self.files(["unlabelled.csv"])[0]
            self.bridge.openFiles("channel:KK", [QUrl.fromLocalFile(str(unknown))])
            wait(lambda: not self.bridge._pending)
            self.plotted()
            self.bridge.invoke("swap-kk")
            wait(lambda: not self.bridge._pending)
            self.plotted()
            assigned = dict(self.bridge._assignments)
            self.bridge.invoke("refresh-catalog")
            wait(lambda: not self.bridge._pending)
            self.assertEqual(self.bridge._assignments, assigned)
            self.bridge.setParameter("sourceFilter", "all")
            self.assertEqual(self.bridge._assignments, assigned)
            self.bridge.setParameter("preset", "custom")
            self.bridge.setParameter("show.KpK", True)
            self.bridge.setParameter("show.KK", False)
            wait(lambda: not self.bridge._pending)
            self.plotted()
            self.assertEqual(set(self.bridge._maps), {"KKp", "KpK"})
            self.assertEqual(self.bridge._assignments, assigned)
            self.bridge.invoke("clear-files")
            self.bridge.invoke("refresh-catalog")
            wait(lambda: not self.bridge._pending)
            self.assertEqual(self.bridge._assignments, {})
            self.assertFalse(self.bridge.snapshot["inspectionVisible"])
            self.assertFalse(self.bridge.snapshot["canExport"])

        def test_rotation_inference_stays_in_selected_stage_and_mapping_is_persisted(self):
            paths = self.files(["sample_PL_1077uW_Rot224deg_Stage200.csv", "sample_PL_1078uW_Rot269deg_Stage200.csv",
                                "sample_PL_1077uW_Rot224deg_Stage3600.csv", "sample_PL_1078uW_Rot269deg_Stage3600.csv"])
            self.loaded(paths)
            self.assertAlmostEqual(self.bridge._values["outK"], 24.)
            self.assertAlmostEqual(self.bridge._values["outKp"], 69.)
            assigned = dict(self.bridge._assignments)
            self.assertEqual(len(assigned), 2)
            self.assertTrue(all("Stage200" in name for name in assigned.values()))
            other = next(group for group in self.bridge._groups if "stage3600" in group["key"])
            self.bridge.setParameter("selectedGroup", other["key"])
            wait(lambda: not self.bridge._pending)
            self.plotted()
            self.assertTrue(all("Stage3600" in name for name in self.bridge._assignments.values()))
            self.bridge.setParameter("rotationMapping", "rot1_output")
            wait(lambda: not self.bridge._pending)
            self.assertEqual(self.settings.value("compare/rotation_mapping"), "rot1_output")
            self.assertEqual(len(self.bridge._groups), 2)

        def test_axis_request_reload_and_numeric_validation_preserve_stored_precision(self):
            import numpy as np
            from core.data_io import CompareSelection, load_compare_cubes
            paths = self.loaded()
            self.bridge.setParameter("yAxis", "bg")
            wait(lambda: not self.bridge._pending)
            self.plotted()
            np.testing.assert_allclose(self.bridge._cubes["KK"].gate, np.linspace(-2, 2, 9))
            self.bridge.setParameter("yA", 1.234567890123)
            self.bridge.setParameter("yB", -.25)
            self.bridge.setParameter("yC", .1)
            self.bridge.setParameter("yAxis", "advanced")
            wait(lambda: not self.bridge._pending)
            self.plotted()
            expected = load_compare_cubes(str(self.root), CompareSelection.from_mapping(self.bridge._assignments),
                                         y_axis=self.bridge._y_axis())
            np.testing.assert_allclose(self.bridge._cubes["KK"].gate, expected["KK"].gate)
            self.assertEqual(self.bridge._values["yA"], 1.234567890123)
            self.assertIn("1.234567890123", self.fields()["yEquation"]["value"])
            self.assertEqual(self.fields()["yLabel"]["value"], expected["KK"].gate_label)
            old = self.bridge._values["xmin"]
            self.bridge.setParameter("xmin", float("nan"))
            self.assertEqual(self.bridge._values["xmin"], old)
            self.bridge.setParameter("xmin", self.bridge._values["xmax"] + 1)
            self.assertEqual(self.bridge._values["xmin"], old)
            self.bridge.setParameter("xmin", old + .000000001234)
            self.plotted()
            self.assertAlmostEqual(self.bridge._values["xmin"], old + .000000001234, places=13)

        def test_split_bounds_spectrum_layout_and_real_clipped_export_history(self):
            import numpy as np
            from core.loader import load_dat
            from core.processing import background_correct_cube, valley_polarization_cube
            from core.plotting import resolve_split_boundary
            self.loaded()
            self.bridge.setParameter("autoBackground", False)
            self.bridge.setParameter("background", 20.)
            self.bridge.setParameter("split", True)
            self.bridge.setParameter("leftMin", 70.)
            self.bridge.setParameter("leftMax", 160.)
            self.bridge.setParameter("rightMin", 80.)
            self.bridge.setParameter("rightMax", 170.)
            self.plotted()
            hosts = [axis for axis in self.bridge._canvas.figure.axes if getattr(axis, "_dptk_inspection_header", False)]
            self.assertTrue(all(len(axis.child_axes) == 2 for axis in hosts))
            self.assertTrue(self.bridge.snapshot["sideBySide"])
            self.bridge.setParameter("heatmapFraction", .61)
            self.bridge.invoke("toggle-spectrum")
            self.assertFalse(self.bridge._spectrum.get_visible())
            self.bridge.invoke("toggle-spectrum")
            self.assertTrue(self.bridge._spectrum.get_visible())
            self.assertEqual(self.bridge._values["heatmapFraction"], .61)
            self.assertAlmostEqual(self.bridge._grid.get_height_ratios()[0], .61)
            self.plotted()
            self.assertTrue(all(axis.get_xlabel() == "" for axis in self.bridge._maps.values()))
            renderer = self.bridge._canvas.get_renderer()
            divider = (1 - self.bridge.snapshot["dividerY"]) * self.bridge._canvas.figure.bbox.height
            map_bottom = min(axis.get_tightbbox(renderer).y0 for axis in self.bridge._maps.values())
            spectrum_top = self.bridge._spectrum.bbox.y1
            self.assertGreater(map_bottom - divider, 15.)
            self.assertGreater(divider - spectrum_top, 15.)
            self.bridge.invoke("toggle-spectrum")
            self.assertTrue(all(axis.get_xlabel() == "Photon Energy (eV)" for axis in self.bridge._maps.values()))
            self.bridge.invoke("toggle-spectrum")
            self.plotted()
            self.bridge.invoke("export")
            wait(lambda: not self.bridge._pending, timeout=60)
            self.assertEqual(self.bridge.snapshot["error"], "", self.bridge.snapshot["status"])
            outputs = [Path(name) for name in self.bridge.snapshot["resultPaths"]]
            self.assertEqual(len(outputs), 6)
            self.assertEqual(sum(path.suffix == ".png" for path in outputs), 3)
            self.assertTrue(all(path.is_file() and path.is_relative_to(self.root / "Processed Data" / "Compare") for path in outputs))
            metadata = [json.loads(path.read_text(encoding="utf-8")) for path in self.root.rglob("*.metadata.json")]
            self.assertEqual(len(metadata), 3)
            for record in metadata:
                channel = record["operation"].split("/", 1)[1]
                data_path = next(path for path in outputs if path.suffix == ".dat" and path.with_suffix(".metadata.json").name in
                                 [candidate.name for candidate in self.root.rglob("*.metadata.json")
                                  if json.loads(candidate.read_text(encoding="utf-8"))["operation"] == record["operation"]])
                actual = load_dat(data_path)
                if channel == "VP":
                    expected = valley_polarization_cube(self.bridge._cubes["KK"], self.bridge._cubes["KKp"], background=20.)
                    np.testing.assert_allclose(actual.Z, expected.Z, rtol=1e-6, atol=1e-7, equal_nan=True)
                else:
                    expected = background_correct_cube(self.bridge._cubes[channel], 20.)
                    boundary, _ = resolve_split_boundary(expected.energy, self.bridge._values["splitX"])
                    clipped = expected.Z.copy()
                    clipped[:, :boundary] = np.clip(clipped[:, :boundary], 70., 160.)
                    clipped[:, boundary:] = np.clip(clipped[:, boundary:], 80., 170.)
                    np.testing.assert_allclose(actual.Z, clipped, rtol=1e-6, atol=1e-6)
                self.assertEqual(record["processing"]["background_constant"], 20.)
            self.assertEqual(self.bridge._groups[0]["status"], "processed")
            self.assertIn("Historical export; current settings not verified", self.fields()["history-summary"]["value"])

        def test_latest_request_wins_and_close_cancels_worker(self):
            from core.data_io import load_compare_cubes
            from core.load_cancellation import check_load_cancelled
            first = self.files(["first_PL_KK.csv", "first_PL_KKp.csv"])
            latest = self.files(["latest_PL_KK.csv", "latest_PL_KKp.csv"])
            started, release = threading.Event(), threading.Event()
            self.addCleanup(release.set)
            def delayed(folder, selection, **kwargs):
                if any("first_" in name for name in selection.as_pairs().values()):
                    started.set()
                    if not release.wait(5):
                        raise AssertionError("Synthetic Compare request was not released")
                return load_compare_cubes(folder, selection, **kwargs)
            completions = []
            self.bridge.operationCompleted.connect(lambda *args: completions.append(args))
            with patch("core.data_io.load_compare_cubes", side_effect=delayed):
                self.bridge.openFiles("sources", [QUrl.fromLocalFile(str(path)) for path in first])
                wait(started.is_set)
                self.bridge.openFiles("sources", [QUrl.fromLocalFile(str(path)) for path in latest])
                wait(lambda: not self.bridge.snapshot["busy"])
                release.set()
                wait(lambda: not self.bridge._pending)
            self.plotted()
            self.assertEqual(set(self.bridge._assignments.values()), set(map(str, latest)))
            self.assertEqual([kind for kind, _ in completions], ["load"])
            started.clear()
            def cooperative(*args, **kwargs):
                started.set()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    check_load_cancelled()
                    time.sleep(.005)
                raise AssertionError("Compare close did not cancel its worker")
            with patch("core.data_io.load_compare_cubes", side_effect=cooperative):
                self.bridge.invoke("reload")
                wait(started.is_set)
                self.bridge.close()
                self.bridge.close()
            self.assertEqual(self.bridge._pending, {})
            self.assertFalse(self.bridge._plot_timer.isActive())
            self.assertFalse(self.bridge._publish_timer.isActive())

        def test_changed_source_is_rejected_before_export(self):
            paths = self.loaded()
            paths[0].write_text(paths[0].read_text(encoding="utf-8") + "\n", encoding="utf-8")
            self.bridge.invoke("export")
            wait(lambda: not self.bridge._pending)
            self.assertIn("changed after loading", self.bridge.snapshot["error"])
            self.assertEqual(list(self.root.rglob("*.png")), [])

        def test_range_edits_reuse_science_and_group_filters_remain_filesystem_free(self):
            import core.processing as processing
            self.loaded()
            cubes = dict(self.bridge._derived)
            with patch("core.processing.background_correct_cube", wraps=processing.background_correct_cube) as correct:
                self.bridge.setParameter("xmin", self.bridge._values["xmin"] + .001)
                self.bridge.setParameter("cmap", "dptk_rdbu_r_p0p45")
                self.bridge.setParameter("clip", False)
                self.plotted()
                self.assertEqual(correct.call_count, 0)
                self.assertTrue(all(self.bridge._derived[key] is cube for key, cube in cubes.items()))
            with patch.object(Path, "stat", side_effect=AssertionError("Group filter attempted GUI-thread filesystem IO")):
                self.bridge.setParameter("historyFilter", "new")
                self.bridge.setParameter("query", "sample")
                self.bridge._update_groups()
                self.assertTrue(self.bridge.snapshot["sourceGroups"])
            self.assertTrue(all(group["modified"] > 0 for group in self.bridge._groups))

        def test_reload_refreshes_unfixed_regional_bounds_and_honors_fixed_values(self):
            self.loaded()
            self.bridge.setParameter("split", True)
            self.plotted()
            old = self.bridge._values["leftMax"]
            self.bridge.setParameter("leftMax", old + 100.)
            self.bridge.setParameter("fix.leftMax", True)
            fixed = self.bridge._values["leftMax"]
            self.bridge.setParameter("rightMax", self.bridge._values["rightMax"] + 100.)
            stale = self.bridge._values["rightMax"]
            self.bridge.invoke("reload")
            wait(lambda: not self.bridge._pending)
            self.plotted()
            self.assertEqual(self.bridge._values["leftMax"], fixed)
            self.assertNotEqual(self.bridge._values["rightMax"], stale)

        def test_unavailable_hidden_sources_do_not_block_visible_pair_load_or_export(self):
            paths = self.loaded()
            paths[2].unlink()
            paths[3].unlink()
            self.bridge.invoke("reload")
            wait(lambda: not self.bridge._pending)
            self.plotted()
            self.assertEqual(len(self.bridge._assignments), 4)
            self.assertEqual(set(self.bridge._cubes), {"KK", "KKp"})
            self.assertTrue(self.bridge.snapshot["canExport"])
            self.bridge.setParameter("preset", "all")
            wait(lambda: not self.bridge._pending)
            self.assertTrue(self.bridge.snapshot["error"])
            self.assertFalse(self.bridge.snapshot["canExport"])

        def test_real_xlsx_sources_use_existing_map_loader_and_domain_vp(self):
            import numpy as np
            from openpyxl import Workbook
            from core.loader import load_xlsx_map
            from core.processing import valley_polarization_cube
            paths = []
            for index, key in enumerate(("KK", "KKp")):
                path = self.root / ("map_PL_" + key + ".xlsx")
                workbook = Workbook()
                sheet = workbook.active
                sheet.title = "dR_R"
                sheet.append(["0", -1., 0., 1.])
                for energy, values in [(1.7, [10., 20., 30.]), (1.8, [30., 40., 50.]), (1.9, [40., 50., 60.])]:
                    sheet.append([energy, *[value + index * 8 for value in values]])
                workbook.save(path)
                workbook.close()
                paths.append(path)
            self.loaded(paths)
            for key, path in zip(("KK", "KKp"), paths):
                expected = load_xlsx_map(str(self.root), path.name)
                np.testing.assert_array_equal(self.bridge._cubes[key].Z, expected.Z)
                self.assertEqual(self.bridge._cubes[key].gate_label, "Doping (V)")
            self.bridge.setParameter("view", "vp")
            self.plotted()
            expected = valley_polarization_cube(self.bridge._cubes["KK"], self.bridge._cubes["KKp"],
                                               background=self.bridge._background)
            np.testing.assert_allclose(self.bridge._derived["VP"].Z, expected.Z, equal_nan=True)

        def test_duplicate_manual_channel_assignment_keeps_legacy_advisory_and_cannot_fake_vp(self):
            paths = self.loaded()
            self.bridge.setParameter("channel.KKp", str(paths[0]))
            wait(lambda: not self.bridge._pending)
            self.assertIn("source file assignment(s) reused", self.fields()["assignment-summary"]["value"])
            self.assertEqual(self.bridge._active_keys(), ["KK"])
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.bridge.setParameter("view", "vp")
            self.assertFalse(self.bridge._derived)
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.assertIn("distinct KK and KKp", self.bridge.snapshot["workspaceMessage"])

        def test_real_qml_parameters_and_workspace_present_native_frame_and_controls(self):
            from PySide6.QtQml import QQmlApplicationEngine
            from ui_qt.quick.appearance import quick_colors
            from ui_qt.quick.plot_surface import register_plot_surface
            register_plot_surface()
            colors = quick_colors("light")
            engine = QQmlApplicationEngine()
            engine.setInitialProperties(dict(bridge=self.bridge, colors=colors))
            qml_folder = Path(__file__).resolve().parents[1] / "ui_qt" / "quick" / "qml"
            qml = b'''import QtQuick
import QtQuick.Window
import QtQuick.Layouts
import "."
Window {
    id: window
    required property var bridge
    required property var colors
    width: 1360; height: 900; visible: true
    RowLayout {
        anchors.fill: parent; spacing: 0
        CompareParameters { objectName: "parameters"; bridge: window.bridge; colors: window.colors; Layout.preferredWidth: 320; Layout.fillHeight: true }
        CompareWorkspace { bridge: window.bridge; colors: window.colors; Layout.fillWidth: true; Layout.fillHeight: true }
    }
}'''
            engine.loadData(QByteArray(qml), QUrl.fromLocalFile(str(qml_folder / "IsolatedCompare.qml")))
            self.assertTrue(engine.rootObjects())
            window = engine.rootObjects()[0]
            def cleanup():
                window.close()
                engine.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                app.processEvents()
            self.addCleanup(cleanup)
            parameters = window.findChild(QObject, "parameters")
            wait(lambda: bool(parameters.property("schemaSignature")))
            schema = parameters.property("schemaSignature")
            self.loaded()
            self.assertEqual(parameters.property("schemaSignature"), schema, "Loading must retain existing sidebar controls and drafts")
            surface = window.findChild(QObject, "compare_plot_surface")
            self.assertIsNotNone(surface)
            wait(lambda: surface.property("presentedGeneration") == self.bridge._canvas.frame_generation
                and surface.property("presentedGeneration") > 0)
            self.assertEqual(surface.property("paintedGeneration"), surface.property("presentedGeneration"))
            self.assertAlmostEqual(parameters.property("width"), 320.)
            self.assertTrue(window.findChild(QObject, "compare_vp"))
            self.assertNotIn(str(self.root), self.bridge._groups[0]["label"])
            self.assertIn(str(self.root), self.fields()["selectedGroup"]["description"])
            signals = []
            self.bridge.sourceDialogRequested.connect(lambda *args: signals.append(args))
            def item(root, name):
                if root.objectName() == name:
                    return root
                for child in root.childItems() if hasattr(root, "childItems") else root.children():
                    found = item(child, name)
                    if found is not None:
                        return found
                return None
            select = item(window.contentItem(), "app_parameter_select-files")
            self.assertIsNotNone(select)
            select.clicked.emit()
            self.assertEqual(signals, [("sources", True)])
            self.bridge.setParameter("view", "vp")
            self.plotted()
            wait(lambda: surface.property("presentedGeneration") == self.bridge._canvas.frame_generation)
            self.assertFalse(self.bridge.snapshot["sideBySide"])
            self.assertTrue(window.findChild(QObject, "compare_vp").property("primary"))
            self.assertFalse(any(name in sys.modules for name in ("PySide6.QtWidgets", "ui_qt.common", "ui_qt.main_window")))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CompareCases))
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    qInstallMessageHandler(previous_handler)
    print("QUICK_COMPARE_TESTS=" + json.dumps(dict(tests=result.testsRun, success=result.wasSuccessful())))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
