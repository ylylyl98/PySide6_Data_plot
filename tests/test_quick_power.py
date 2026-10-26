"""Full native Power requests with temporary spectra, settings and outputs."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickPowerTests(unittest.TestCase):
    def test_widgets_free_power_workflow(self):
        with TemporaryDirectory(prefix="quick-power-process-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software", MPLBACKEND="agg",
                MPLCONFIGDIR=temporary, XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_POWER_TESTS=", result.stdout)
        print(result.stdout.strip())


def _run_isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import importlib.abc
    import json
    import threading
    import time
    import traceback
    from copy import deepcopy
    from unittest.mock import patch

    class NoWidgets(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {"PySide6.QtWidgets", "PySide6.QtQuickWidgets", "ui_qt.common", "ui_qt.main_window",
                "matplotlib.backends.backend_qt", "matplotlib.backends.backend_qtagg"} or fullname.startswith("ui_qt.controllers"):
                raise AssertionError("Native Power imported a Widgets owner: " + fullname)
    sys.meta_path.insert(0, NoWidgets())
    from PySide6.QtCore import QByteArray, QCoreApplication, QEvent, QObject, QSettings, QUrl, qInstallMessageHandler
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication
    from PySide6.QtQuickControls2 import QQuickStyle
    from ui_qt.quick.power_bridge import PowerBridge
    from ui_qt.quick import power_workflow as workflow
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

    def wait(predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                app.processEvents()
                return
            time.sleep(.005)
        raise AssertionError("Timed out waiting for native Power worker/frame state")

    class PowerCases(unittest.TestCase):
        def setUp(self):
            python_start, qml_start = len(python_errors), len(qml_errors)
            self.addCleanup(lambda: self.assertEqual(python_errors[python_start:], [], "Uncaught Qt callback exception"))
            self.addCleanup(lambda: self.assertEqual(qml_errors[qml_start:], [], "QML runtime error"))
            self.temporary = TemporaryDirectory(prefix="quick-power-case-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name)
            self.settings = QSettings(str(self.root / "isolated.ini"), QSettings.Format.IniFormat)
            self.bridge = PowerBridge(settings=self.settings)
            self.addCleanup(self.bridge.close)

        def sweep(self, name="sample_KK.csv", powers=None, *, gain=1., background=10., stages=True, energy_count=161, center=1.6):
            import numpy as np
            powers = np.geomspace(1, 100, 12) if powers is None else np.asarray(powers, float)
            energy = np.linspace(1.5, 1.7, energy_count)
            peak = 1 / (1 + (2 * (energy - center) / .008) ** 2)
            spectra = background + gain * powers[:, None] * 10 * peak
            stage = np.arange(len(powers), dtype=float) if stages is True else np.asarray(stages, float)
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            table = np.column_stack((powers, stage, spectra))
            header = "Power_uW,stage_pos," + ",".join(format(float(value), ".17g") for value in energy)
            np.savetxt(path, table, delimiter=",", header=header, comments="")
            return path.resolve()

        def scanned(self):
            self.bridge.setExperimentFolder(str(self.root))
            wait(lambda: "catalog" not in self.bridge._running)
            self.assertEqual(self.bridge.snapshot["error"], "", self.bridge.snapshot["status"])
            return self.bridge._catalog

        def opened(self, mode="single", *, paths=None):
            if paths is None:
                paths = [self.sweep()] if mode == "single" else [self.sweep(), self.sweep("sample_KKp.csv", gain=.7)]
            self.scanned()
            group = next(group for group in self.bridge._catalog["groups"] if len(group.mapping) == (1 if mode == "single" else 2))
            self.bridge.setParameter("selectedGroup", group.key)
            self.bridge.invoke({"single": "open-sweep", "intensity": "open-comparison", "vp": "open-vp"}[mode])
            wait(lambda: not self.bridge._running.intersection({"load", "products"}))
            self.assertEqual(self.bridge.snapshot["error"], "", self.bridge.snapshot["status"])
            self.assertIsNotNone(self.bridge._products)
            self.plotted()
            return paths

        def plotted(self):
            wait(lambda: self.bridge._canvas is not None and self.bridge._canvas.frame_generation > 0
                and not self.bridge._plot_timer.isActive() and not self.bridge._canvas.backend._timer.isActive())
            self.assertEqual(self.bridge.snapshot["error"], "", self.bridge.snapshot["status"])
            self.assertEqual(self.bridge._canvas.last_error, "")

        def reviewed(self, state=None, *, paths=None):
            paths = paths or [self.sweep("segments/run1_KK.csv"), self.sweep("segments/run2_KK.csv", gain=2., background=20.)]
            self.scanned()
            from core.data_io import power_sweep_source_key
            self.bridge._queue = [power_sweep_source_key(path.relative_to(self.root).as_posix()) for path in paths]
            if state is not None:
                self.bridge._review_state.update(state)
            self.bridge.invoke("review-combination")
            wait(lambda: "review" not in self.bridge._running)
            self.assertEqual(self.bridge.snapshot["error"], "", self.bridge.snapshot["status"])
            self.assertIsNotNone(self.bridge._review)
            self.plotted()
            return paths

        def test_00_lazy_qgui_only_contract_and_no_implicit_source_write(self):
            self.assertNotIn("numpy", initial_imports)
            self.assertNotIn("matplotlib", initial_imports)
            self.assertIsNone(self.bridge.plotBackend)
            self.assertEqual(self.bridge.snapshot["stage"], "selection")
            self.assertFalse(self.bridge.snapshot["canExport"])
            signals = []
            self.bridge.sourceDialogRequested.connect(lambda *args: signals.append(args))
            self.bridge.invoke("select-files")
            self.bridge.invoke("choose-source:KK")
            self.assertEqual(signals, [("sources", True), ("KK", False)])
            self.assertFalse((self.root / ".power-selection.json").exists())

        def test_single_intensity_and_vp_are_actual_domain_products(self):
            import numpy as np
            from core.processing import background_correct_cube, power_stage_paired_vp_cubes
            self.opened("intensity")
            products = self.bridge._products
            group = self.bridge.snapshot["selectedMeasurement"]
            self.assertIsInstance(group["sources"], list)
            self.assertIsInstance(group["issues"], list)
            self.assertTrue(all(isinstance(keys, list) for keys in group["duplicates"].values()))
            self.assertEqual(set(products["cubes"]), {"KK", "KKp"})
            for key in ("KK", "KKp"):
                expected = background_correct_cube(self.bridge._raw_results[key].cube, products["background"])
                np.testing.assert_allclose(products["cubes"][key].Z, expected.Z)
            self.assertTrue(self.bridge.snapshot["sideBySide"])
            self.bridge.setParameter("mode", "vp")
            wait(lambda: "products" not in self.bridge._running)
            self.plotted()
            kk, kkp = self.bridge._raw_results["KK"], self.bridge._raw_results["KKp"]
            _, _, expected, pairs = power_stage_paired_vp_cubes(kk.cube, kk.records, kkp.cube, kkp.records,
                                                               background=self.bridge._products["background"])
            np.testing.assert_allclose(self.bridge._products["cubes"]["VP"].Z, expected.Z, equal_nan=True)
            self.assertEqual(self.bridge._values["pairing"], "stage")
            self.assertEqual(len(pairs), len(self.bridge._products["stagePairs"]))
            self.assertFalse(self.bridge.snapshot["sideBySide"])

        def test_switching_single_and_pair_restores_assignments_and_loads_only_required_channels(self):
            self.opened("intensity")
            roles = {key: self.bridge._active_selection[key] for key in ("KK", "KKp")}
            self.bridge.setParameter("mode", "single")
            wait(lambda: not self.bridge._running.intersection({"load", "products"}))
            self.plotted()
            self.assertEqual(self.bridge._values["mode"], "single")
            self.assertEqual(set(self.bridge._products["cubes"]), {"Power"})
            self.assertEqual({key: self.bridge._active_selection[key] for key in ("KK", "KKp")}, roles)
            self.bridge.setParameter("mode", "intensity")
            wait(lambda: not self.bridge._running.intersection({"load", "products"}))
            self.plotted()
            self.assertEqual(set(self.bridge._products["cubes"]), {"KK", "KKp"})
            self.assertEqual({key: self.bridge._active_selection[key] for key in ("KK", "KKp")}, roles)

        def test_native_source_freshness_and_single_export_retains_all_raw_powers(self):
            import numpy as np
            from core.loader import load_dat
            powers = [-1., 0., 1., 10., 100.]
            path = self.sweep(powers=powers)
            self.opened(paths=[path])
            self.bridge.setParameter("powerLog", True)
            self.plotted()
            self.assertEqual(len(self.bridge._raw_results["Power"].cube.gate), 5)
            self.assertEqual(len(self.bridge._display_coordinates["Power"][0]), 3)
            self.bridge.invoke("export")
            wait(lambda: not self.bridge._running.intersection({"export", "catalog"}), timeout=60)
            self.assertEqual(self.bridge.snapshot["error"], "", self.bridge.snapshot["status"])
            outputs = [Path(name) for name in self.bridge.snapshot["resultPaths"]]
            self.assertTrue(any(path.suffix == ".png" for path in outputs))
            exported = load_dat(next(path for path in outputs if path.suffix == ".dat"))
            np.testing.assert_allclose(exported.gate, powers)
            np.testing.assert_allclose(exported.Z, self.bridge._products["cubes"]["Power"].Z, rtol=1e-6, atol=1e-6)
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            previous = set(self.root.rglob("*.png"))
            self.bridge.invoke("export")
            wait(lambda: "export" not in self.bridge._running)
            self.assertIn("changed after loading", self.bridge.snapshot["error"])
            self.assertEqual(set(self.root.rglob("*.png")), previous)

        def test_reload_recomputes_regional_colors_and_keeps_each_fixed_bound(self):
            import numpy as np
            self.opened()
            self.bridge.setParameter("split", True)
            self.plotted()
            self.bridge.setParameter("leftMin", -3.)
            self.bridge.setParameter("fix.leftMin", True)
            self.bridge.setParameter("rightMax", 3000.)
            self.bridge.setParameter("fix.rightMax", True)
            old_left_max = self.bridge._values["leftMax"]
            self.sweep(gain=5.)
            self.bridge.invoke("reload")
            wait(lambda: not self.bridge._running.intersection({"load", "products"}))
            self.plotted()
            value, cube = self.bridge._values, self.bridge._products["cubes"]["Power"]
            self.assertEqual(value["leftMin"], -3.)
            self.assertEqual(value["rightMax"], 3000.)
            left = cube.Z[:, cube.energy < value["splitX"]]
            right = cube.Z[:, cube.energy >= value["splitX"]]
            self.assertAlmostEqual(value["leftMax"], float(np.percentile(left, 99.99)))
            self.assertAlmostEqual(value["rightMin"], float(np.percentile(right, .01)))
            self.assertGreater(value["leftMax"], old_left_max * 4.)
            bounds = value["leftMin"], value["leftMax"], value["rightMin"], value["rightMax"]
            self.bridge.setParameter("mode", "vp")
            self.assertEqual((value["leftMin"], value["leftMax"], value["rightMin"], value["rightMax"]), bounds)

        def test_color_edits_reuse_meshes_and_keep_zoom_with_original_intensities(self):
            import numpy as np
            self.opened()
            maps, renders = dict(self.bridge._maps), dict(self.bridge._heatmap_renders)
            maps["Power"].set_xlim(1.55, 1.65)
            maximum = self.bridge._values["vmax"]
            self.bridge.setParameter("colorMax", maximum * .6)
            self.plotted()
            self.bridge.setParameter("colorMax", maximum * 1.2)
            self.plotted()
            self.assertEqual(self.bridge._maps, maps)
            self.assertEqual(self.bridge._heatmap_renders, renders)
            self.assertEqual(maps["Power"].get_xlim(), (1.55, 1.65))
            np.testing.assert_allclose(renders["Power"].primary.get_array(),
                np.minimum(self.bridge._products["cubes"]["Power"].Z, maximum * 1.2))

        def test_gate_defers_paint_while_worker_export_holds_plot_font_lock(self):
            from core.typography import matplotlib_font_context
            self.opened()
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
                self.bridge.setParameter("gate", 10.)
                self.assertTrue(worker.is_alive(), "Power gate callback waited for a worker's render lock")
                self.assertEqual(self.bridge._canvas.frame_generation, generation)
            finally:
                release.set()
                worker.join(5)
            self.plotted()
            self.assertGreater(self.bridge._canvas.frame_generation, generation)

        def test_catalog_recurses_excludes_archives_and_retains_cached_draft_across_filters(self):
            active = self.sweep("power_sweep/active_KK.csv")
            self.sweep("Initial Data/archived_KK.csv")
            self.sweep("old/old_KK.csv")
            (self.root / "analysis.csv").write_text("Power_uW,area\n1,2\n", encoding="utf-8")
            catalog = self.scanned()
            self.assertEqual(len(catalog["sources"]), 1)
            draft = deepcopy(self.bridge.snapshot["draft"])
            with patch.object(Path, "stat", side_effect=AssertionError("GUI filter performed source I/O")):
                self.bridge.setParameter("query", "no-match")
                self.bridge.setParameter("historyFilter", "processed")
                self.assertEqual(self.bridge.snapshot["measurementGroups"], [])
                self.assertEqual(self.bridge.snapshot["draft"], draft)
                self.assertEqual(self.bridge.snapshot["selectedMeasurement"]["key"], draft["group"])

        def test_first_twenty_recent_groups_and_legacy_series_opt_in(self):
            import numpy as np
            for index in range(22):
                path = self.sweep(f"scan{index}_KK.csv", powers=[1, 2], energy_count=9)
                os.utime(path, (100 + index, 100 + index))
            self.scanned()
            self.assertEqual(len(self.bridge.snapshot["measurementGroups"]), 20)
            self.assertEqual(self.bridge.snapshot["olderCount"], 2)
            self.assertIn("scan21", self.bridge.snapshot["measurementGroups"][0]["label"])
            self.bridge.invoke("show-older")
            self.assertEqual(len(self.bridge.snapshot["measurementGroups"]), 22)
            energy = np.linspace(1.5, 1.7, 41)
            for power in (1, 10):
                path = self.root / f"legacy_PL_{power}uW.csv"
                np.savetxt(path, np.column_stack(([-1., 1.], [0., 0.], [0., 0.], np.ones((2, len(energy))) * power)),
                    delimiter=",", header="Vbg,Vtg,Vbias," + ",".join(map(str, energy)), comments="")
            self.bridge.invoke("refresh")
            wait(lambda: "catalog" not in self.bridge._running)
            self.assertEqual(len(self.bridge._catalog["sources"]), 22)
            self.bridge.setParameter("includeLegacy", True)
            wait(lambda: "catalog" not in self.bridge._running)
            self.assertEqual(len(self.bridge._catalog["sources"]), 23)
            self.assertTrue(any(source.source_format == "legacy" for source in self.bridge._catalog["sources"].values()))

        def test_metadata_group_and_only_explicit_confirmed_manual_pairing_is_portable(self):
            first, second = self.sweep("a.csv"), self.sweep("b.csv", gain=.6)
            metadata = dict(session_id="session", measurement_id="measurement", files=[dict(path="a.csv", channel="KK"), dict(path="b.csv", channel="KKp")])
            (self.root / "acquisition.experiment.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            self.scanned()
            self.assertEqual(len(self.bridge._catalog["groups"]), 1)
            original = deepcopy(self.bridge.snapshot["draft"])
            self.bridge.invoke("swap-roles")
            self.bridge.invoke("cancel-selection")
            self.assertFalse((self.root / ".power-selection.json").exists())
            self.bridge.invoke("open-comparison")
            wait(lambda: not self.bridge._running.intersection({"load", "products"}))
            self.plotted()
            saved_path = self.root / ".power-selection.json"
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            saved_roles = saved["groups"][original["group"]]
            self.assertEqual(saved_roles["KK"], original["KKp"])
            self.assertFalse(any(str(self.root) in value for value in saved_roles.values()))
            restored = PowerBridge(settings=self.settings)
            self.addCleanup(restored.close)
            restored.setExperimentFolder(str(self.root))
            wait(lambda: "catalog" not in restored._running)
            self.assertEqual(restored.snapshot["draft"]["KK"], original["KKp"])
            self.assertEqual(restored.snapshot["draft"]["KKp"], original["KK"])
            self.assertFalse(restored.snapshot["draft"]["manual"])

        def test_power_interpolation_with_distinct_grids_and_repeated_stage_override(self):
            import numpy as np
            from core.processing import power_valley_polarization_cube
            paths = [self.sweep(powers=[1, 2, 4, 8, 16, 32], stages=[0, 0, 0, 0, 0, 0]),
                     self.sweep("sample_KKp.csv", powers=[1.5, 3, 6, 12, 24, 48], gain=.5, stages=[0, 0, 0, 0, 0, 0])]
            self.opened("intensity", paths=paths)
            self.assertEqual(self.bridge._values["pairing"], "power")
            self.bridge.setParameter("mode", "vp")
            wait(lambda: "products" not in self.bridge._running)
            self.plotted()
            _, _, expected = power_valley_polarization_cube(self.bridge._raw_results["KK"].cube,
                self.bridge._raw_results["KKp"].cube, background=self.bridge._products["background"])
            np.testing.assert_allclose(self.bridge._products["cubes"]["VP"].Z, expected.Z, equal_nan=True)
            self.bridge.setParameter("pairing", "stage")
            wait(lambda: "products" not in self.bridge._running)
            self.assertIn("unique stage", self.bridge.snapshot["error"])
            self.assertFalse(self.bridge.snapshot["inspectionVisible"], "A failed new pairing must not mislabel the previous VP image")
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.bridge.setParameter("pairing", "power")
            wait(lambda: "products" not in self.bridge._running)
            self.plotted()

        def test_gate_blits_exact_spectrum_and_duplicate_row_is_preserved(self):
            import numpy as np
            from matplotlib.backend_bases import MouseEvent
            paths = [self.sweep(powers=[1., 2., 4.]), self.sweep("sample_KKp.csv", powers=[2., 1., 3.], gain=.6)]
            self.opened("vp", paths=paths)
            canvas, axis = self.bridge._canvas, self.bridge._maps["VP"]
            true, display, rows = self.bridge._display_coordinates["VP"]
            self.assertEqual(true[0], true[1])
            self.assertGreater(display[1], display[0])
            x, y = axis.transData.transform((1.6, float((display[1] + display[2]) / 2 - .01)))
            self.bridge._on_press(MouseEvent("button_press_event", canvas, x, y, button=1))
            self.assertEqual(self.bridge._selected_row_index, 1)
            self.assertEqual(self.bridge._gate_lines["VP"].get_ydata()[0], display[1])
            np.testing.assert_array_equal(self.bridge._lines["VP"].get_ydata(), self.bridge._products["cubes"]["VP"].Z[1])
            drawn, blits = canvas.draw_count, canvas.blit_count
            self.bridge.setParameter("gate", 9.)
            self.assertEqual(canvas.draw_count, drawn)
            self.assertEqual(canvas.blit_count, blits + 1)
            np.testing.assert_array_equal(self.bridge._lines["VP"].get_ydata(), self.bridge._products["cubes"]["VP"].Z[2])
            self.assertEqual(self.bridge._values["gate"], 3.5)
            self.bridge.setParameter("heatmapFraction", .61)
            self.bridge.invoke("toggle-spectrum")
            self.bridge.invoke("toggle-spectrum")
            self.assertEqual(self.bridge._values["heatmapFraction"], .61)

        def test_automatic_review_calibration_save_existing_another_and_reloaded_provenance(self):
            import numpy as np
            paths = self.reviewed()
            payload = self.bridge._review
            self.assertTrue(payload["accepted"])
            self.assertAlmostEqual(payload["details"][1]["factor"], .5, places=10)
            np.testing.assert_allclose(payload["inputs"][0].cube.Z, payload["inputs"][1].cube.Z, rtol=1e-10, atol=1e-10)
            self.assertFalse(self.bridge.snapshot["review"]["canSave"])
            self.bridge.setParameter("review.compatible", True)
            self.assertTrue(self.bridge.snapshot["review"]["canSave"])
            self.bridge.invoke("save-combination")
            wait(lambda: not self.bridge._running.intersection({"save", "catalog", "load", "products"}), timeout=60)
            self.plotted()
            self.assertEqual(self.bridge._stage, "plot")
            output = list((self.root / "Processed Data" / "Power Dependence" / "Combined Sweeps").glob("*.csv"))
            self.assertEqual(len(output), 1)
            self.assertTrue(output[0].with_suffix(".combine.json").is_file())
            self.assertTrue(any("sources" in record.source_provenance for record in self.bridge._raw_results["Power"].records))
            self.assertTrue(any(key in self.bridge._catalog["combined"] for key in self.bridge._catalog["sources"]))
            self.bridge.invoke("review-combination")
            wait(lambda: "review" not in self.bridge._running)
            self.assertTrue(self.bridge._review["existing"])
            self.bridge.setParameter("review.compatible", True)
            self.bridge.invoke("save-combination")
            wait(lambda: "save" not in self.bridge._running)
            self.assertTrue(self.bridge.snapshot["review"]["needsExistingChoice"])
            self.bridge.invoke("create-another-combination")
            wait(lambda: not self.bridge._running.intersection({"save", "catalog", "load", "products"}), timeout=60)
            self.assertEqual(len(list(output[0].parent.glob("*.csv"))), 2)
            self.assertTrue(all(path.is_file() for path in paths))

        def test_manual_unscaled_multi_sweep_average_has_no_pairwise_bias(self):
            import numpy as np
            from core.data_io import power_sweep_source_key
            paths = [self.sweep(f"segments/run{index}_KK.csv", powers=[1, 2], gain=index, energy_count=9) for index in (1, 2, 6)]
            rows = {power_sweep_source_key(path.relative_to(self.root).as_posix()): dict(mode="none", background=0., factor=1., excluded=[]) for path in paths}
            self.reviewed(dict(automatic=False, rows=rows, policy="average"), paths=paths)
            payload = self.bridge._review
            expected = sum(result.cube.Z for result in payload["originals"]) / 3
            np.testing.assert_allclose(payload["preview"].cube.Z, expected)
            for record in payload["preview"].records:
                self.assertEqual(len(json.loads(record.source_provenance)["sources"]), 3)
            self.assertTrue(payload["accepted"])

        def test_review_acquisition_telemetry_exclusions_and_metadata_freshness(self):
            import numpy as np
            from urllib.parse import quote
            paths = [self.sweep("segments/run1_KK.csv"), self.sweep("segments/run2_KK.csv", gain=2.)]
            path = paths[1]
            table = np.loadtxt(path, delimiter=",", skiprows=1)
            header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            actual = table[:, 1].copy()
            actual[1] += 3.
            np.savetxt(path, np.column_stack((table[:, :2], actual, table[:, 2:])), delimiter=",",
                header=",".join(header[:2] + ["stage_pos_actual"] + header[2:]), comments="")
            metadata = path.parent / "run.experiment.metadata.json"
            metadata.write_text(json.dumps({"files": [{"path": path.name}],
                "settings": {"requested": {"exp_ms": 2000, "frames": 3, "gain": "High"}}}), encoding="utf-8")
            self.reviewed(paths=paths)
            row = self.bridge.snapshot["review"]["details"][1]
            self.assertEqual(row["flaggedRows"], [2])
            self.assertIn("2000 ms", row["acquisitionText"])
            self.assertIn("3 frame(s)", row["acquisitionText"])
            self.assertIn("High", row["acquisitionText"])
            key = self.bridge._queue[1]
            self.bridge.setParameter("review.row:" + quote(key, safe="") + ":excluded", "2")
            wait(lambda: "review" not in self.bridge._running)
            row = self.bridge.snapshot["review"]["details"][1]
            self.assertEqual(row["flaggedRows"], [])
            self.assertEqual(row["excluded"], [1])
            self.assertEqual(len(self.bridge._review["inputs"][1].records), 11)
            self.assertTrue(all(record.row_index != 3 for record in self.bridge._review["inputs"][1].records))
            metadata.write_text(metadata.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            self.bridge.setParameter("review.compatible", True)
            self.bridge.invoke("save-combination")
            wait(lambda: "save" not in self.bridge._running)
            self.assertIn("changed after loading", self.bridge.snapshot["error"])
            self.assertFalse((self.root / "Processed Data").exists())

        def test_rejected_or_invalid_current_review_keeps_preview_and_blocks_save(self):
            paths = [self.sweep("segments/one_KK.csv"), self.sweep("segments/two_KK.csv", center=1.63)]
            self.reviewed(paths=paths)
            self.assertFalse(self.bridge._review["accepted"])
            self.assertIsNotNone(self.bridge._review["preview"])
            self.bridge.setParameter("review.compatible", True)
            self.assertFalse(self.bridge.snapshot["review"]["canSave"])
            self.bridge.invoke("save-combination")
            wait(lambda: "save" not in self.bridge._running)
            self.assertFalse((self.root / "Processed Data").exists())
            self.bridge._review_state.update(automatic=False, manualBackground=True)
            for row in self.bridge._review_state["rows"].values():
                row["mode"] = "none"
            self.bridge.invoke("refresh-review")
            wait(lambda: "review" not in self.bridge._running)
            self.assertTrue(self.bridge._review["accepted"])
            from urllib.parse import quote
            key = self.bridge._queue[0]
            self.bridge.setParameter("review.row:" + quote(key, safe="") + ":excluded", "999")
            wait(lambda: "review" not in self.bridge._running)
            self.bridge.setParameter("review.compatible", True)
            self.assertFalse(self.bridge.snapshot["review"]["canSave"], "A failed new review must not save a stale accepted preview")
            self.assertIsNotNone(self.bridge._review["preview"])

        def test_combination_rejects_opposite_channels_and_original_row_reuse(self):
            self.sweep()
            self.sweep("sample_KKp.csv")
            self.scanned()
            self.bridge._queue = list(self.bridge._catalog["sources"])
            self.bridge.invoke("review-combination")
            wait(lambda: "review" not in self.bridge._running)
            self.assertIn("same-channel", self.bridge.snapshot["error"])
            self.assertIsNone(self.bridge._review)
            from core.power_combine import combine_many_power_sweeps
            from core.data_io import load_power_series_cube
            catalog = self.bridge._catalog
            result = load_power_series_cube(str(self.root), catalog["files"], group_key=next(iter(catalog["sources"])), sources=catalog["sources"])
            from dataclasses import replace
            with self.assertRaisesRegex(ValueError, "same original"):
                combine_many_power_sweeps([result, replace(result, group_key="renamed-copy")])

        def test_async_latest_selection_wins_and_close_cancels(self):
            from core.data_io import load_power_series_cube
            from core.load_cancellation import check_load_cancelled
            first, latest = self.sweep("first_KK.csv"), self.sweep("latest_KK.csv", gain=2.)
            self.scanned()
            groups = self.bridge._catalog["groups"]
            old_group = next(group for group in groups if "first" in group.label)
            latest_group = next(group for group in groups if "latest" in group.label)
            started, release = threading.Event(), threading.Event()
            self.addCleanup(release.set)
            def delayed(folder, files, **kwargs):
                if "first" in str(kwargs.get("group_key", "")):
                    started.set()
                    if not release.wait(5):
                        raise AssertionError("Old synthetic request was not released")
                return load_power_series_cube(folder, files, **kwargs)
            completions = []
            self.bridge.operationCompleted.connect(lambda *args: completions.append(args))
            with patch("core.data_io.load_power_series_cube", side_effect=delayed):
                self.bridge.setParameter("selectedGroup", old_group.key)
                self.bridge.invoke("open-sweep")
                wait(started.is_set)
                self.bridge.setParameter("selectedGroup", latest_group.key)
                self.bridge.invoke("open-sweep")
                wait(lambda: not self.bridge._running.intersection({"load", "products"}))
                release.set()
                wait(lambda: not self.bridge._pending)
            self.plotted()
            self.assertIn("latest", self.bridge._active_selection["single"])
            self.assertEqual([kind for kind, _ in completions].count("load"), 1)
            started.clear()
            def cooperative(*args, **kwargs):
                started.set()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    check_load_cancelled()
                    time.sleep(.005)
                raise AssertionError("Close did not cancel source parsing")
            with patch("core.data_io.load_power_series_cube", side_effect=cooperative):
                self.bridge.invoke("reload")
                wait(started.is_set)
                self.bridge.close()
                self.bridge.close()
            self.assertEqual(self.bridge._pending, {})
            self.assertFalse(self.bridge._running)

        def test_actual_qml_selection_plot_review_and_stable_numeric_draft(self):
            from PySide6.QtCore import Qt
            from PySide6.QtQml import QQmlApplicationEngine
            from ui_qt.quick.appearance import quick_colors
            from ui_qt.quick.plot_surface import register_plot_surface
            from urllib.parse import quote
            register_plot_surface()
            paths = [self.sweep(), self.sweep("sample_KKp.csv", gain=.7), self.sweep("segments/run2_KK.csv", gain=2., background=20.)]
            colors = quick_colors("light")
            engine = QQmlApplicationEngine()
            engine.setInitialProperties(dict(bridge=self.bridge, colors=colors))
            folder = Path(__file__).resolve().parents[1] / "ui_qt" / "quick" / "qml"
            qml = b'''import QtQuick
import QtQuick.Window
import QtQuick.Layouts
import "."
Window {
    id: window
    required property var bridge
    required property var colors
    property real sidebarWidth: 320
    width: 1400; height: 980; visible: true
    RowLayout {
        anchors.fill: parent; spacing: 0
        PowerParameters { bridge: window.bridge; colors: window.colors; Layout.preferredWidth: window.sidebarWidth; Layout.fillHeight: true }
        PowerWorkspace { bridge: window.bridge; colors: window.colors; Layout.fillWidth: true; Layout.fillHeight: true }
    }
}'''
            engine.loadData(QByteArray(qml), QUrl.fromLocalFile(str(folder / "IsolatedPower.qml")))
            self.assertTrue(engine.rootObjects())
            window = engine.rootObjects()[0]
            def cleanup():
                window.close()
                engine.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                app.processEvents()
            self.addCleanup(cleanup)
            def find(name):
                def walk(item):
                    if item.objectName() == name:
                        return item
                    for child in item.childItems() if hasattr(item, "childItems") else item.children():
                        result = walk(child)
                        if result is not None:
                            return result
                    return None
                return walk(window.contentItem())
            self.scanned()
            self.assertIsNotNone(find("power_source_list"))
            group = next(group for group in self.bridge._catalog["groups"] if len(group.mapping) == 2)
            self.bridge.setParameter("selectedGroup", group.key)
            app.processEvents()
            find("power_open_comparison").clicked.emit()
            wait(lambda: self.bridge._products is not None and not self.bridge._running.intersection({"load", "products"}))
            self.plotted()
            wait(lambda: find("power_plot_surface") is not None and find("power_plot_surface").property("presentedGeneration") == self.bridge._canvas.frame_generation)
            self.assertTrue(self.bridge.snapshot["sideBySide"])
            capture = Path(__file__).resolve().parents[1] / "build" / "quick-power"
            capture.mkdir(parents=True, exist_ok=True)
            self.assertTrue(window.grabWindow().save(str(capture / "native-power-intensity.png")))
            from core.data_io import power_sweep_source_key
            keys = [power_sweep_source_key(path.relative_to(self.root).as_posix()) for path in (paths[0], paths[2])]
            self.bridge._queue = keys
            self.bridge.invoke("review-combination")
            wait(lambda: "review" not in self.bridge._running)
            self.plotted()
            wait(lambda: find("power_review_plot") is not None and find("power_review_plot").property("presentedGeneration") == self.bridge._canvas.frame_generation)
            self.bridge.setParameter("review.advanced", True)
            self.bridge.setParameter("review.automatic", False)
            wait(lambda: "review" not in self.bridge._running)
            self.bridge.setParameter("review.row:" + quote(keys[1], safe="") + ":mode", "known")
            wait(lambda: "review" not in self.bridge._running)
            app.processEvents()
            name = "app_parameter_review.row:" + quote(keys[1], safe="") + ":factor"
            field = find(name)
            self.assertIsNotNone(field)
            self.assertTrue(field.property("enabled"))
            field.forceActiveFocus(Qt.FocusReason.OtherFocusReason)
            field.setProperty("text", "1.234567890123")
            field.textEdited.emit()
            stored = self.bridge._review_state["rows"][keys[1]]["factor"]
            self.bridge.invoke("refresh-review")
            wait(lambda: "review" not in self.bridge._running)
            app.processEvents()
            self.assertIs(find(name), field, "Review values must not rebuild existing numeric editors")
            self.assertTrue(field.property("activeFocus"))
            self.assertEqual(field.property("text"), "1.234567890123")
            self.assertEqual(self.bridge._review_state["rows"][keys[1]]["factor"], stored)
            field.editingFinished.emit()
            wait(lambda: "review" not in self.bridge._running)
            self.assertEqual(self.bridge._review_state["rows"][keys[1]]["factor"], 1.234567890123)
            self.plotted()
            wait(lambda: find("power_review_plot").property("presentedGeneration") == self.bridge._canvas.frame_generation)
            self.assertTrue(window.grabWindow().save(str(capture / "native-power-review.png")))
            window.setProperty("sidebarWidth", 240.)
            app.setFont(QFont("Segoe UI", 18))
            window.setWidth(1024)
            for _ in range(4):
                app.processEvents()
                time.sleep(.01)
            self.assertTrue(window.grabWindow().save(str(capture / "native-power-review-narrow.png")))
            app.setFont(QFont("Segoe UI", 9))
            self.bridge.invoke("back-to-selection")
            app.processEvents()
            self.assertEqual(self.bridge.snapshot["combinationKeys"], keys)
            self.assertEqual(self.bridge._review_state["rows"][keys[1]]["factor"], 1.234567890123)
            self.assertFalse(any(name in sys.modules for name in ("PySide6.QtWidgets", "ui_qt.common", "ui_qt.main_window")))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PowerCases))
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    qInstallMessageHandler(previous_handler)
    print("QUICK_POWER_TESTS=" + json.dumps(dict(tests=result.testsRun, success=result.wasSuccessful())))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
