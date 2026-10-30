import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "agg")
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtGui import QGuiApplication
from core.loader import DataCube
from core.peak_workspace import create_dataset as peak_dataset
from core.drr_analysis_workspace import create_dataset as drr_dataset
from ui_qt.quick.peak_analysis_bridge import PeakAnalysisBridge
from ui_qt.quick.drr_analysis_bridge import DrrAnalysisBridge

class AnalysisRowCoalescingTests(unittest.TestCase):
    def test_immediate_exports_capture_latest_row_before_plot_timer(self):
        app = QGuiApplication.instance() or QGuiApplication([])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cube = DataCube(np.linspace(1, 2, 30), np.arange(8.), np.ones((8, 30)), "Gate", "Test", "Intensity")
            settings = QSettings(str(root / "settings.ini"), QSettings.IniFormat)
            peak = PeakAnalysisBridge(settings=settings, session_path=root / "peak.npz")
            drr = DrrAnalysisBridge(settings=settings, session_path=root / "drr.npz", start_folder=root)
            try:
                peak.addDatasets([peak_dataset(cube, "PL", "PL", "test", "Test")])
                drr.addDatasets([drr_dataset(cube, "Test", {})])
                peak.setParameter("row", 4)
                self.assertTrue(peak._row_timer.isActive())
                with patch.object(peak, "_start") as start:
                    peak.saveFigure(str(root / "peak.png"))
                with patch("core.peak_plotting.build_figure") as build, patch("ui_qt.quick.peak_analysis_bridge.save_figure"):
                    start.call_args.args[1](None, None)
                self.assertEqual(build.call_args.args[1], 4)

                drr.setParameter("tab", 1)
                drr.p2p["pending"] = False
                drr.p2p["records"] = [dict(dataset_key=drr.active_key, result=np.ones((1, 6)))]
                drr.setParameter("p2p.row", 5)
                self.assertTrue(drr._row_timer.isActive())
                with patch.object(drr, "_start") as start:
                    drr.exportTo("inspection", str(root / "drr.png"))
                from matplotlib.figure import Figure
                with patch("ui_qt.quick.drr_analysis_plot.inspection_export_figure", return_value=Figure()) as build, patch("ui_qt.quick.drr_analysis_bridge.save_figure"):
                    start.call_args.args[1](None, None)
                self.assertEqual(build.call_args.args[1]["inspection_rows"][drr.active_key], 5)
            finally:
                peak.close()
                drr.close()

    def test_latest_row_is_drawn_once_per_burst(self):
        app = QGuiApplication.instance() or QGuiApplication([])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cube = DataCube(np.linspace(1, 2, 30), np.arange(8.), np.ones((8, 30)), "Gate", "Test", "Intensity")
            settings = QSettings(str(root / "settings.ini"), QSettings.IniFormat)
            peak = PeakAnalysisBridge(settings=settings, session_path=root / "peak.npz")
            drr = DrrAnalysisBridge(settings=settings, session_path=root / "drr.npz", start_folder=root)
            try:
                peak.addDatasets([peak_dataset(cube, "PL", "PL", "test", "Test")])
                drr.addDatasets([drr_dataset(cube, "Test", {})])
                # Connect timers to mocks before issuing a burst; no canvas is needed.
                peak._row_timer.timeout.disconnect()
                drr._row_timer.timeout.disconnect()
                peak_draws, drr_draws = [], []
                peak._row_timer.timeout.connect(lambda: peak_draws.append(peak.dataset["row"]))
                drr._row_timer.timeout.connect(lambda: drr_draws.append(drr._values["gate"]))
                for row in (1, 2, 3, 4):
                    peak.setParameter("row", row)
                    drr.setParameter("gate", float(row))
                self.assertEqual(peak_draws, [])
                self.assertEqual(drr_draws, [])
                app.processEvents()
                self.assertEqual(peak_draws, [4])
                self.assertEqual(drr_draws, [4.])
            finally:
                peak.close()
                drr.close()

if __name__ == "__main__":
    unittest.main()
