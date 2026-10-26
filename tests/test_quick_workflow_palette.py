"""Native workflow activation avoids needless plotting and scientific imports."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import subprocess
import sys
from textwrap import dedent
import unittest
from unittest.mock import patch

from PySide6.QtCore import QCoreApplication, QSettings

from ui_qt.quick.compare_bridge import CompareBridge
from ui_qt.quick.mcd_bridge import McdBridge
from ui_qt.quick.mcd_peak_bridge import McdPeakBridge
from ui_qt.quick.power_bridge import PowerBridge
from ui_qt.quick.shg_bridge import ShgBridge


class WorkflowPaletteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        temporary = TemporaryDirectory(prefix="quick-workflow-palette-")
        self.addCleanup(temporary.cleanup)
        self.settings = QSettings(str(Path(temporary.name) / "isolated.ini"), QSettings.Format.IniFormat)

    def test_workspace_recreation_only_replots_for_changed_colors(self):
        for kind, scheduler in ((CompareBridge, "_schedule_plot"), (PowerBridge, "_schedule_plot"),
                                (McdBridge, "_queue_plot"), (McdPeakBridge, "_queue_plot"),
                                (ShgBridge, "_schedule_plot")):
            with self.subTest(bridge=kind.__name__):
                bridge = kind(settings=self.settings)
                self.addCleanup(bridge.close)
                previous = dict(bridge._palette)
                with patch.object(bridge, scheduler) as schedule:
                    # QML supplies colors on initial assignment and again on
                    # Component.onCompleted whenever the workspace is recreated.
                    bridge.setPalette(previous)
                    bridge.setPalette(previous)
                    bridge.setPalette({"inset": previous["inset"]})
                    schedule.assert_not_called()

                    bridge.setPalette({"inset": "#010203"})
                    schedule.assert_called_once_with()
                    self.assertEqual(bridge._palette, dict(previous, inset="#010203"))
                    bridge.setPalette({"inset": "#010203"})
                    schedule.assert_called_once_with()
                    bridge.setPalette({"text": "#040506"})
                    self.assertEqual(schedule.call_count, 2)

    def test_retained_mcd_review_recolors_once_for_an_actual_theme_change(self):
        bridge = McdBridge(settings=self.settings)
        self.addCleanup(bridge.close)
        bridge._review_result = object()
        with patch.object(bridge, "_queue_plot") as schedule, patch.object(bridge, "_draw_review") as review:
            bridge.setPalette(dict(bridge._palette))
            review.assert_not_called()
            schedule.assert_not_called()
            bridge.setPalette({"text": "#010203"})
            review.assert_called_once_with()
            schedule.assert_called_once_with()

    def test_empty_drr_controls_defer_scientific_imports_until_data_is_loaded(self):
        script = dedent("""
            import importlib.abc
            import sys
            class NoScientificStack(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'numpy', 'matplotlib', 'scipy'} or fullname in {'core.plotting', 'core.color_limits'}:
                        raise AssertionError('Empty DRR workflow imported ' + fullname)
            sys.meta_path.insert(0, NoScientificStack())
            from PySide6.QtCore import QSettings
            from PySide6.QtGui import QGuiApplication
            from ui_qt.quick.application_bridge import QuickApplicationBridge
            app = QGuiApplication([])
            bridge = QuickApplicationBridge(settings=QSettings(sys.argv[1], QSettings.Format.IniFormat))
            try:
                bridge.selectWorkflow('drr')
                assert bridge.snapshot['workflow'] == 'drr'
                assert not bridge.snapshot['inspectionVisible']
                assert bridge.drrDisplayBridge.snapshot['regionCount'] == 1
                assert bridge.drrDisplayBridge.snapshot['choices']['raw.cmap']['options']
                app.processEvents()
                bridge.selectWorkflow('pl')
                app.processEvents()
            finally:
                bridge.close()
        """)
        environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software")
        result = subprocess.run([sys.executable, "-c", script, self.settings.fileName()],
                                cwd=Path(__file__).resolve().parents[1], env=environment,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
