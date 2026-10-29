"""Real native PL exports and restart restore, isolated from Widgets and settings."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class NativePlDisplayExportTests(unittest.TestCase):
    def test_native_pl_display_exports(self):
        with TemporaryDirectory(prefix="pl-display-process-") as folder:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software", MPLBACKEND="agg",
                MPLCONFIGDIR=folder, XDG_CONFIG_HOME=folder, XDG_CACHE_HOME=folder, LOCALAPPDATA=folder,
                APPDATA=folder, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_PL_DISPLAY_EXPORT_TESTS=", result.stdout)
        print(result.stdout.strip())


def isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import importlib.abc
    import json
    import time
    from unittest.mock import patch

    class NoWidgets(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {"PySide6.QtWidgets", "PySide6.QtQuickWidgets", "ui_qt.common", "ui_qt.main_window",
                            "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_qt"}:
                raise AssertionError("Native PL imported Widgets: " + fullname)

    sys.meta_path.insert(0, NoWidgets())
    from PySide6.QtCore import QSettings, QUrl
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQuickControls2 import QQuickStyle
    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    import numpy as np
    from PIL import Image
    from core import pl_display_settings as history
    from core.loader import load_dat
    from ui_qt.quick.application_bridge import QuickApplicationBridge

    def wait(predicate, timeout=45):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                app.processEvents()
                return
            time.sleep(.005)
        raise AssertionError("Native PL display did not settle")

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="pl-display-case-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name).resolve()
            self.energy = np.linspace(1.7, 1.9, 121)
            self.gates = np.array([-2., -1., 0., 1., 2.])
            self.intensity = np.asarray([1. + (gate + 3.) * np.exp(-((self.energy - 1.8) / .014) ** 2)
                                         for gate in self.gates])
            self.path = self.root / "sample.dat"
            self.path.write_text("X\t" + "\t".join(map(str, self.gates)) + "\n" + "\n".join(
                "\t".join(map(str, [energy, *self.intensity[:, index]])) for index, energy in enumerate(self.energy)) + "\n")
            self.settings_path = self.root / "settings.ini"
            self.bridge = self.make_bridge()

        def make_bridge(self):
            bridge = QuickApplicationBridge(settings=QSettings(str(self.settings_path), QSettings.IniFormat))
            bridge.setParameter("pl.autoNext", False)
            self.addCleanup(bridge.close)
            return bridge

        def idle(self):
            bridge = self.bridge
            wait(lambda: not bridge._pending and not bridge._busy and not bridge._plot_timer.isActive()
                 and not bridge._manual_task.busy and not bridge._pl_range_timer.isActive())
            if bridge._canvas:
                wait(lambda: not bridge._canvas.backend._timer.isActive())

        def load(self, path=None):
            self.bridge.openFile(QUrl.fromLocalFile(str(path or self.path)))
            self.idle()
            self.assertIsNotNone(self.bridge._cube, self.bridge.snapshot["status"])

        def three_regions(self):
            display = self.bridge._pl_display
            snapshot = display.capture_state()
            snapshot["values"].update(xmin=1.71, xmax=1.89, ymin=-1.5, ymax=1.5, gate=1., cmap="plasma")
            snapshot["geometry"].update(count=3, x0=1.76, x1=1.84, showBoundary=True)
            for scale in ("linear", "log"):
                bank = snapshot["banks"][scale]
                for side, low, high in (("left", .1, 2.), ("middle", .2, 6.), ("right", .3, 4.)):
                    bank[side + "Min"], bank[side + "Max"] = low, high
                    bank["fixed"][side + "Min"] = True
                    bank["fixed"][side + "Max"] = True
            self.assertTrue(display.restore_state(snapshot))
            self.bridge._pl_sync_scales()
            self.bridge._schedule_plot()
            self.idle()
            return snapshot

        def csv_with_distinct_gate_axes(self):
            wavelength = np.linspace(700., 750., 101)
            bg = np.arange(5., dtype=float)
            tg = 100. + 2. * bg
            intensity = 1. + (1. + bg[:, None] * .1) * np.exp(-((wavelength[None, :] - 725.) / 3.) ** 2)
            self.path = self.root / "sample_PL_TG.csv"
            np.savetxt(self.path, np.column_stack((bg, tg, intensity)), delimiter=",",
                header="Vbg,Vtg," + ",".join(map(str, wavelength)), comments="")

        def test_three_regions_both_original_and_wide_pngs_one_dat(self):
            self.load()
            self.three_regions()
            self.bridge.setParameter("pl.pngLayout", "wide")
            self.bridge.setParameter("pl.pngWidth", 1.5)
            paths = self.bridge._pl_export_request()()
            self.assertEqual(set(paths), {"png_linear", "png_log", "png_linear_wide", "png_log_wide", "dat"})
            boundaries = []
            for scale in ("linear", "log"):
                with Image.open(paths["png_" + scale]) as original, Image.open(paths["png_" + scale + "_wide"]) as wide:
                    self.assertEqual(wide.height, original.height)
                    self.assertAlmostEqual(wide.width / original.width, 1.5, delta=.01)
                params = self.bridge._params(scale)
                boundaries.append((params.split_scale.split_x, params.split_scale.split_x2))
                np.testing.assert_allclose(boundaries[-1], (1.76, 1.84), atol=np.diff(self.energy).max())
                self.assertEqual(len(params.split_scale.bounds), 3)
            self.assertEqual(boundaries[0], boundaries[1])
            output = load_dat(paths["dat"])
            np.testing.assert_allclose(output.energy, self.energy)
            np.testing.assert_allclose(output.gate, self.gates)
            np.testing.assert_allclose(output.Z, self.intensity, rtol=1e-6, atol=1e-8)
            metadata = json.loads(Path(paths["dat"]).with_suffix(".metadata.json").read_text())
            self.assertNotIn("pl_display_save_pending", metadata)
            self.assertEqual(metadata["pl_display"]["geometry"]["count"], 3)
            self.assertEqual(len(metadata["outputs"]), 5)
            for scale in ("linear", "log"):
                self.assertEqual(metadata["plot"][scale]["split_scale"]["split_x2"], boundaries[0][1])
            self.assertEqual(set(metadata["figures"]), {Path(paths[key]).name for key in paths if key != "dat"})
            for key, width in (("png_linear", 1.), ("png_log", 1.), ("png_linear_wide", 1.5), ("png_log_wide", 1.5)):
                self.assertEqual(metadata["figures"][Path(paths[key]).name]["plot"]["png_width_scale"], width)
            index = self.root / "Processed Data" / "PL" / history.INDEX_NAME
            self.assertTrue(index.is_file())
            self.assertEqual(len(list(index.parent.glob("*.dat"))), 1)
            saved = history.find_snapshot(self.root, self.path)
            self.assertEqual(saved["values"]["cmap"], "plasma")
            self.assertEqual(saved["banks"]["linear"]["middleMax"], 6.)

        def test_restart_restores_saved_view_but_explicit_manual_fix_has_priority(self):
            self.load()
            self.three_regions()
            self.bridge._pl_export_request()()
            saved_whole_colors = {scale: tuple(self.bridge._pl_banks[scale][key] for key in ("vmin", "vmax"))
                                  for scale in ("linear", "log")}
            self.bridge.setParameter("xmin", 1.72)
            self.bridge.setParameter("pl.fixed.xmin", True)
            self.idle()
            self.bridge.settings.sync()
            self.bridge.close()
            self.bridge = self.make_bridge()
            self.load()
            self.assertEqual(self.bridge._values["xmin"], 1.72)
            self.assertEqual(self.bridge._values["xmax"], 1.89)
            self.assertEqual(self.bridge._values["cmap"], "plasma")
            self.assertEqual(self.bridge._pl_display.geometry["count"], 3)
            self.assertEqual(self.bridge._pl_banks["log"]["middleMax"], 6.)
            self.assertTrue(self.bridge._pl_axis_fixed["xmin"])
            for scale in ("linear", "log"):
                self.assertEqual(tuple(self.bridge._pl_banks[scale][key] for key in ("vmin", "vmax")),
                                 saved_whole_colors[scale])

        def test_failed_history_publication_keeps_metadata_pending_and_unrestorable(self):
            self.load()
            self.three_regions()
            operation = self.bridge._pl_export_request()
            with patch.object(history, "save_snapshot", side_effect=OSError("simulated history write failure")):
                with self.assertRaisesRegex(OSError, "simulated"):
                    operation()
            directory = self.root / "Processed Data" / "PL"
            self.assertFalse((directory / history.INDEX_NAME).exists())
            records = list(directory.glob("*.metadata.json"))
            self.assertEqual(len(records), 1)
            self.assertTrue(json.loads(records[0].read_text())["pl_display_save_pending"])
            self.assertIsNone(history.find_snapshot(self.root, self.path))

        def test_open_saved_dat_with_fresh_settings_restores_its_own_exported_display(self):
            self.load()
            self.bridge.setParameter("pl.datChoice", "Custom")
            self.idle()
            self.bridge.setParameter("pl.datLabel", "Density")
            self.idle()
            self.bridge.setParameter("pl.datUnit", "cm^-2")
            self.idle()
            self.three_regions()
            paths = self.bridge._pl_export_request()()
            self.bridge.close()
            self.settings_path = self.root / "fresh-settings.ini"
            self.bridge = self.make_bridge()
            self.load(paths["dat"])
            self.assertEqual(self.bridge._values["cmap"], "plasma")
            self.assertEqual(self.bridge._pl_display.geometry["count"], 3)
            self.assertEqual(self.bridge._pl_banks["linear"]["middleMax"], 6.)
            self.assertEqual(self.bridge._pl_banks["log"]["middleMax"], 6.)
            self.assertEqual(self.bridge._cube.gate_label, "Density")
            self.assertEqual(self.bridge._cube.gate_unit, "cm^-2")
            self.assertEqual((self.bridge._values["xmin"], self.bridge._values["xmax"]), (1.71, 1.89))

        def test_metadata_completion_failure_does_not_publish_indexed_display(self):
            self.load()
            self.three_regions()
            with patch.object(history, "complete_metadata_snapshot", side_effect=OSError("simulated completion failure")):
                with self.assertRaisesRegex(OSError, "completion"):
                    self.bridge._pl_export_request()()
            directory = self.root / "Processed Data" / "PL"
            self.assertFalse((directory / history.INDEX_NAME).exists())
            self.assertIsNone(history.find_snapshot(self.root, self.path))

        def test_unsaved_explicit_bg_y_locks_restore_axis_derivation_after_restart(self):
            self.csv_with_distinct_gate_axes()
            self.load()
            np.testing.assert_allclose(self.bridge._cube.gate, 100. + 2. * np.arange(5.))
            self.bridge.setParameter("pl.yAxis", "BG")
            self.idle()
            self.bridge.setParameter("ymin", .5)
            self.bridge.setParameter("ymax", 3.5)
            self.bridge.setParameter("pl.fixed.ymin", True)
            self.bridge.setParameter("pl.fixed.ymax", True)
            self.idle()
            self.bridge.settings.sync()
            self.bridge.close()
            self.bridge = self.make_bridge()
            self.load()
            np.testing.assert_allclose(self.bridge._cube.gate, np.arange(5.))
            self.assertEqual(self.bridge._pl_values["yAxis"], "BG")
            self.assertEqual((self.bridge._values["ymin"], self.bridge._values["ymax"]), (.5, 3.5))
            self.assertTrue(self.bridge._pl_axis_fixed["ymin"])
            self.assertTrue(self.bridge._pl_axis_fixed["ymax"])

        def test_saved_bg_file_restores_derivation_and_viewport_with_fresh_settings(self):
            self.csv_with_distinct_gate_axes()
            self.load()
            self.bridge.setParameter("pl.yAxis", "BG")
            self.idle()
            self.bridge.setParameter("ymin", .5)
            self.bridge.setParameter("ymax", 3.5)
            self.idle()
            self.bridge._pl_export_request()()
            self.bridge.close()
            self.settings_path = self.root / "fresh-settings.ini"
            self.bridge = self.make_bridge()
            self.load()
            np.testing.assert_allclose(self.bridge._cube.gate, np.arange(5.))
            self.assertEqual(self.bridge._pl_values["yAxis"], "BG")
            self.assertEqual((self.bridge._values["ymin"], self.bridge._values["ymax"]), (.5, 3.5))
            self.assertFalse(self.bridge._pl_axis_fixed["ymin"])
            self.assertFalse(self.bridge._pl_axis_fixed["ymax"])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    print("QUICK_PL_DISPLAY_EXPORT_TESTS=" + json.dumps(dict(tests=result.testsRun, failures=len(result.failures),
          errors=len(result.errors), widgetsImported="PySide6.QtWidgets" in sys.modules)))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(isolated()) if "--isolated" in sys.argv else unittest.main()
