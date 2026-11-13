"""Native saved-figure handoff and display-only export, isolated from user settings."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class PowerPointRoundtripTests(unittest.TestCase):
    def test_native_roundtrip(self):
        with TemporaryDirectory() as temporary:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                       MPLBACKEND="agg", MPLCONFIGDIR=temporary, XDG_CONFIG_HOME=temporary,
                       XDG_CACHE_HOME=temporary, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, __file__, "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import json
    import time
    from dataclasses import asdict, replace
    from unittest.mock import patch
    import numpy as np
    from PIL import Image
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QGuiApplication
    from core.plotting import HeatmapParams, SplitColorScale
    from core.pptx_figure_index import FigureIndex
    from core.pptx_figure_edit import load_figure, save_figure_edit
    from ui_qt.quick.application_bridge import QuickApplicationBridge

    app = QGuiApplication([])

    def wait(predicate):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(.005)
        raise AssertionError("PowerPoint handoff did not finish")

    class Service:
        def __init__(self): self.updates = []
        def set_root(self, root): pass
        def close(self): pass
        def update_figure(self, *values): self.updates.append(values)

    class Cases(unittest.TestCase):
        def setUp(self):
            self.tmp = TemporaryDirectory()
            self.addCleanup(self.tmp.cleanup)
            self.root = Path(self.tmp.name)
            self.bridge = QuickApplicationBridge(settings=QSettings(str(self.root / "settings.ini"), QSettings.IniFormat))
            self.addCleanup(self.bridge.close)
            self.service = self.bridge._ppt_service = Service()

        def seed(self, workflow="pl", split=False):
            folder = self.root / "Processed Data" / workflow.upper()
            folder.mkdir(parents=True, exist_ok=True)
            dat = folder / "sample.dat"
            dat.write_text("X\t-1\t0\t1\n1\t1\t2\t3\n2\t2\t3\t4\n3\t3\t4\t5\n4\t4\t5\t6\n")
            png = folder / "sample.png"
            Image.new("RGB", (120, 93), "red").save(png)
            params = HeatmapParams("Saved title", "Photon Energy (eV)", "Gate", "Signal", 1., 7.,
                (1., 4.), (-1., 1.), cmap="plasma", split_scale=SplitColorScale(2.5, 1., 5., 2., 7.) if split else None)
            role = "linear" if workflow == "pl" else "second"
            (folder / "sample.metadata.json").write_text(json.dumps(dict(
                operation="PL" if workflow == "pl" else "DR/R", outputs=[png.name, dat.name],
                processing={"derivative_order": 2} if workflow == "drr" else {},
                figures={png.name: dict(data_file=dat.name, plot=asdict(params), role=role, display_title="Saved rendered title")}
            )))
            return FigureIndex(self.root).scan()[0]

        def test_pl_doubleclick_handoff_save_and_reopen(self):
            record = self.seed(split=True)
            original = record.dat_path.read_bytes()
            self.bridge._ppt_open(record)
            wait(lambda: not self.bridge._busy)
            self.assertIsNotNone(self.bridge._ppt_record, self.bridge._status)
            self.assertEqual(self.bridge._pl_display.geometry["count"], 2)
            self.assertEqual(self.bridge._pl_display.values["cmap"], "plasma")
            self.bridge.setParameter("scale", "log")
            self.bridge.setParameter("cmap", "inferno")
            self.bridge.invoke("export")
            wait(lambda: not self.bridge._busy)
            self.assertEqual(len(self.service.updates), 1, self.bridge._status)
            old_id, png_path, updated = self.service.updates[0]
            self.assertEqual(old_id, record.record_id)
            self.assertTrue(updated.plot["log_scale"])
            self.assertEqual(updated.role, "log")
            self.assertEqual(updated.plot["cmap"], "inferno")
            self.assertEqual(updated.dat_path.read_bytes(), original)
            self.assertEqual(record.dat_path.read_bytes(), original)
            self.assertNotEqual(updated.png_path, record.png_path)
            index = FigureIndex(self.root)
            index.scan()
            self.assertEqual(index.match_image(updated.png_path.read_bytes())[0].record_id, updated.record_id)
            self.bridge._ppt_updated(old_id, updated)
            self.bridge._ppt_open(updated)
            wait(lambda: not self.bridge._busy)
            self.assertEqual(self.bridge._pl_display.values["scale"], "log", self.bridge._status)
            self.assertEqual(self.bridge._pl_display.values["cmap"], "inferno")
            self.bridge.selectWorkflow("drr")
            self.assertIsNone(self.bridge._ppt_record)

        def test_derivative_dat_is_not_differentiated_again(self):
            record = self.seed("drr")
            self.bridge._ppt_open(record)
            wait(lambda: not self.bridge._busy)
            self.assertIsNotNone(self.bridge._ppt_record, self.bridge._status)
            np.testing.assert_array_equal(self.bridge._cube.Z, np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]))
            self.assertIsNone(self.bridge._drr_products.second)
            self.assertEqual(self.bridge._drr_display.heatmap_params(self.bridge._cube).vmax, 7.)
            self.assertFalse(self.bridge._drr_display.setChoice("yaxis.left", 1))
            sequence = self.bridge._sequence
            self.bridge.setParameter("rawDerivative", "second")
            self.assertEqual(self.bridge._sequence, sequence)
            self.bridge._drr_display.setNumber("raw.vmax", "8")
            self.bridge._drr_display._flush_changes()
            self.bridge.invoke("export")
            wait(lambda: not self.bridge._busy)
            self.assertEqual(len(self.service.updates), 1, self.bridge._status)
            updated = self.service.updates[0][2]
            self.assertEqual(updated.plot["vmax"], 8.)
            self.assertEqual(updated.dat_path.read_bytes(), record.dat_path.read_bytes())

        def test_failed_export_and_changed_data_leave_originals(self):
            record = self.seed()
            _, cube, params = load_figure(record)
            before = set(record.png_path.parent.iterdir())
            with patch("core.export._save_heatmap_png", side_effect=OSError("test render failure")):
                with self.assertRaisesRegex(OSError, "test render failure"):
                    save_figure_edit(record, cube, params, {})
            self.assertEqual(set(record.png_path.parent.iterdir()), before)
            record.dat_path.write_text("changed")
            with self.assertRaises(ValueError):
                save_figure_edit(record, cube, params, {})
            self.assertEqual(set(record.png_path.parent.iterdir()), before)

        def test_custom_sidecar_and_repeated_saves_preserve_original_data(self):
            original = self.seed('drr')
            custom = original.dat_path.parent / 'export.metadata.json'
            original.metadata_path.rename(custom)
            index = FigureIndex(self.root)
            record = index.find_png(original.png_path)[0]
            self.assertEqual(record.metadata_path, custom.resolve())
            source_bytes = record.dat_path.read_bytes()
            _, cube, params = load_figure(record)
            first = save_figure_edit(record, cube, params, {})
            self.assertEqual(first.dat_path.read_bytes(), source_bytes)
            self.assertEqual(first.metadata_path, first.dat_path.with_suffix('.metadata.json'))
            second = save_figure_edit(first, cube, replace(params, xlim=(1.5, 3.5)), {})
            self.assertEqual(second.dat_path.read_bytes(), source_bytes)
            self.assertEqual(second.plot['xlim'], [1.5, 3.5])
            payload = json.loads(second.metadata_path.read_text(encoding='utf-8'))
            self.assertEqual(payload['ppt_edit']['parent_png'], first.png_path.name)
            first.metadata_path.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                save_figure_edit(first, cube, params, {})

        def test_range_edit_save_only_decodes_new_png(self):
            record = self.seed("drr")
            _, cube, params = load_figure(record)
            # Even existing valid exports must not be decoded again on Save.
            source_bytes = record.dat_path.read_bytes()
            hashed = []
            original_hash = FigureIndex._hash
            def track_hash(index, path):
                hashed.append(Path(path))
                if Path(path) == record.png_path:
                    raise AssertionError("Save decoded an unrelated existing PNG")
                return original_hash(index, path)
            with patch.object(FigureIndex, "_hash", track_hash):
                updated = save_figure_edit(record, cube, replace(params, xlim=(1.5, 3.5)), {})
            self.assertEqual(set(hashed), {updated.png_path})
            self.assertEqual(updated.plot["xlim"], [1.5, 3.5])
            self.assertEqual(updated.dat_path.read_bytes(), source_bytes)

        def test_open_does_not_adopt_global_manual_locks(self):
            record = self.seed()
            display = self.bridge._pl_display
            display._manual = {"axis.xmin": 1.5}
            self.bridge._ppt_open(record)
            wait(lambda: not self.bridge._busy)
            self.assertEqual(display.values["xmin"], 1.)
            self.bridge.setParameter("cmap", "inferno")
            stored = json.loads(str(self.bridge.settings.value("pl/display/manual_locks")))
            self.assertEqual(stored["numbers"]["axis.xmin"], 1.5)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(isolated() if "--isolated" in sys.argv else unittest.main())
