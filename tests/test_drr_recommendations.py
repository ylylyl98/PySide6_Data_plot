"""Integration coverage for DRR display recommendations in a separate Qt process."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class DrrRecommendationIntegrationTests(unittest.TestCase):
    def test_isolated_recommendations(self):
        with TemporaryDirectory() as temporary:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                       QT_SCALE_FACTOR="1", MPLBACKEND="agg", MPLCONFIGDIR=temporary,
                       XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                                    cwd=Path(__file__).resolve().parents[1], env=env,
                                    capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import time
    from dataclasses import replace
    import numpy as np
    from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt, QUrl
    from PySide6.QtGui import QGuiApplication, QMouseEvent
    from PySide6.QtQml import QQmlComponent, QQmlEngine
    from PySide6.QtQuick import QQuickWindow
    from core.loader import DataCube
    from ui_qt.quick.drr_display_state import DrrDisplayState

    app = QGuiApplication([])
    # Windows' offscreen plugin cannot enumerate installed fonts by itself.
    from ui_qt.typography import install_application_font
    install_application_font(app)

    def settle():
        for _ in range(40):
            app.processEvents()
            time.sleep(.001)

    class Cases(unittest.TestCase):
        def setUp(self):
            self.tmp = TemporaryDirectory()
            self.addCleanup(self.tmp.cleanup)
            self.state = DrrDisplayState(settings=QSettings(self.tmp.name + '/settings.ini', QSettings.Format.IniFormat))
            self.addCleanup(self.state.close)
            x = np.linspace(1., 3., 121)
            y = np.linspace(-1., 1., 21)
            rng = np.random.default_rng(7)
            z = rng.normal(0, .02, (len(y), len(x)))
            z[:, 30:90] += np.sin(y[:, None] * 3) * np.sin(x[None, 30:90] * 12)
            self.raw = DataCube(x, y, z, 'Gate', 'Raw DRR', 'ΔR/R')
            self.second = replace(self.raw, Z=z * .01, title='Second derivative')
            self.state.sync_cube(self.raw, self.second)

        def value(self, key):
            return self.state.snapshot['numbers'][key]['value']

        def test_single_product_strength_fix_auto_and_data_are_preserved(self):
            original = [a.copy() for a in (self.raw.energy, self.raw.gate, self.raw.Z, self.second.Z)]
            cmap = self.state.heatmap_params(self.second, 'second').cmap
            raw_before = (self.value('raw.vmin'), self.value('raw.vmax'))
            self.assertTrue(self.state.setNumber('raw.vmin', '-2'))
            self.assertTrue(self.state.setFlag('fix.raw.vmin', True))
            raw_before = (self.value('raw.vmin'), self.value('raw.vmax'))
            self.assertTrue(self.state.setFlag('zero', True))
            for strength in range(4):
                self.assertTrue(self.state.setChoice('recommendStrength', strength))
                self.assertTrue(self.state.invoke('recommend.weak.second'), self.state.error)
                self.assertFalse(self.state.snapshot['flags']['zero']['checked'])
                self.assertEqual(self.state.heatmap_params(self.second, 'second').cmap, cmap)
                self.assertEqual((self.value('raw.vmin'), self.value('raw.vmax')), raw_before)
            self.assertIn('0p60', cmap.lower())
            self.assertTrue(self.state.invoke('recommend.weak.raw'), self.state.error)
            self.assertEqual(self.value('raw.vmin'), -2)
            self.assertTrue(self.state.snapshot['flags']['fix.raw.vmin']['checked'])
            self.assertTrue(self.state.invoke('auto.raw'))
            self.assertEqual(self.value('raw.vmin'), -2)
            self.assertTrue(self.state.invoke('auto.second'))
            self.assertEqual(self.value('raw.vmin'), -2)
            for actual, expected in zip((self.raw.energy, self.raw.gate, self.raw.Z, self.second.Z), original):
                np.testing.assert_array_equal(actual, expected)

        def test_split_recommendations_are_independent_and_saved(self):
            from core.drr_display_settings import find_candidates, save_snapshot
            self.assertTrue(self.state.setChoice('regionCount', 2))
            before = {k: self.value(k) for k in ('raw.left_vmin', 'raw.left_vmax', 'second.left_vmin', 'second.left_vmax', 'second.right_vmin', 'second.right_vmax')}
            self.assertTrue(self.state.invoke('recommend.overall.second.middle'), self.state.error)
            self.assertEqual(before, {k: self.value(k) for k in before})
            middle = (self.value('second.middle_vmin'), self.value('second.middle_vmax'))
            self.assertTrue(self.state.invoke('recommend.weak.raw.left'), self.state.error)
            self.assertEqual(middle, (self.value('second.middle_vmin'), self.value('second.middle_vmax')))
            expected = (self.value('raw.left_vmin'), self.value('raw.left_vmax'))
            save_snapshot(self.tmp.name, self.state.capture_state(['a_720nmc.csv']))
            restored = DrrDisplayState(settings=QSettings(self.tmp.name + '/restored.ini', QSettings.Format.IniFormat))
            self.addCleanup(restored.close)
            restored.sync_cube(self.raw, self.second)
            self.assertTrue(restored.restore_candidates(find_candidates(self.tmp.name, ['a_720nmc.csv']), folder=self.tmp.name, files=['a_720nmc.csv']))
            self.assertEqual(expected, (restored.snapshot['numbers']['raw.left_vmin']['value'], restored.snapshot['numbers']['raw.left_vmax']['value']))

        def test_empty_region_reports_error_without_changes(self):
            self.raw.Z[:, :] = np.nan
            before = (self.value('raw.vmin'), self.value('raw.vmax'))
            self.assertFalse(self.state.invoke('recommend.weak.raw'))
            self.assertTrue(self.state.error)
            self.assertEqual(before, (self.value('raw.vmin'), self.value('raw.vmax')))

        def test_x_recommendation_keeps_boundary_fixed_axis_and_colors(self):
            self.assertTrue(self.state.setChoice('regionCount', 1))
            self.state.setFlag('fix.axis.xmin', True)
            keys = ('axis.xmin', 'axis.ymin', 'axis.ymax', 'boundary.x0', 'raw.left_vmin', 'raw.left_vmax', 'second.right_vmin', 'second.right_vmax')
            before = {key: self.value(key) for key in keys}
            self.assertTrue(self.state.invoke('recommend.x'), self.state.error)
            self.assertEqual(before, {key: self.value(key) for key in keys})
            self.assertLess(self.value('axis.xmin'), self.value('boundary.x0'))
            self.assertGreater(self.value('axis.xmax'), self.value('boundary.x0'))
            self.assertTrue(self.state.invoke('auto.x'))
            self.assertEqual(self.value('axis.xmin'), before['axis.xmin'])
            self.assertAlmostEqual(self.value('axis.xmax'), self.raw.energy.max())

        def test_x_recommendation_actually_trims_empty_edges(self):
            x = np.linspace(0, 10, 501)
            y = np.linspace(-1, 1, 40)
            rng = np.random.default_rng(3)
            z = rng.normal(0, .1, (40, len(x)))
            z += 100 * np.exp(-((x - 4) / .2) ** 2)
            z += 1.2 * np.exp(-((x - 7) / .22) ** 2)
            raw = DataCube(x, y, z, 'Gate', 'Raw', 'ΔR/R')
            self.state.sync_cube(raw, replace(raw, Z=z * .01))
            self.assertTrue(self.state.invoke('auto.x'))
            self.assertTrue(self.state.invoke('recommend.x'), self.state.error)
            self.assertGreater(self.value('axis.xmin'), 0)
            self.assertLess(self.value('axis.xmin'), 3.8)
            self.assertGreater(self.value('axis.xmax'), 7.2)
            self.assertLess(self.value('axis.xmax'), 10)
            self.assertTrue(self.state.invoke('auto.x'))
            self.assertEqual((self.value('axis.xmin'), self.value('axis.xmax')), (0, 10))

        def test_qml_recommend_button_click_updates_limits(self):
            engine = QQmlEngine()
            component = QQmlComponent(engine, QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / 'ui_qt/quick/qml/Display.qml')))
            self.assertTrue(component.isReady(), str(component.errors()))
            panel = component.createWithInitialProperties({'bridge': self.state})
            self.assertIsNotNone(panel, str(component.errors()))
            window = QQuickWindow()
            window.resize(360, 2000)
            panel.setWidth(360)
            panel.setHeight(2000)
            panel.setParentItem(window.contentItem())
            window.show()
            settle()
            def find(item, name):
                if item.objectName() == name:
                    return item
                return next((found for child in item.childItems() if (found := find(child, name)) is not None), None)
            button = find(panel, 'drr_action_recommend.weak.raw')
            self.assertIsNotNone(button)
            self.assertTrue(button.isVisible())
            before = (self.value('raw.vmin'), self.value('raw.vmax'))
            point = button.mapToScene(button.boundingRect().center())
            global_point = window.mapToGlobal(point.toPoint())
            for kind, buttons in ((QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton), (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton)):
                QCoreApplication.sendEvent(window, QMouseEvent(kind, point, global_point, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier))
            settle()
            self.assertNotEqual(before, (self.value('raw.vmin'), self.value('raw.vmax')))
            shot = window.grabWindow()
            self.assertFalse(shot.isNull())
            if os.environ.get('DRR_RECOMMEND_SCREENSHOT'):
                self.assertTrue(shot.save(os.environ['DRR_RECOMMEND_SCREENSHOT']))
            self.assertTrue(self.state.setChoice('regionCount', 2))
            self.assertTrue(self.state.setChoice('colorProduct', 1))
            settle()
            for side in ('left', 'middle', 'right'):
                for mode in ('overall', 'weak'):
                    item = find(panel, 'drr_action_recommend.' + mode + '.second.' + side)
                    self.assertIsNotNone(item)
                    self.assertTrue(item.isVisible())
                    left = item.mapToItem(panel, item.boundingRect().topLeft()).x()
                    right = item.mapToItem(panel, item.boundingRect().topRight()).x()
                    self.assertGreaterEqual(left, 0, item.objectName())
                    self.assertLessEqual(right, 360, item.objectName())
            if os.environ.get('DRR_RECOMMEND_SCREENSHOT_SPLIT'):
                self.assertTrue(window.grabWindow().save(os.environ['DRR_RECOMMEND_SCREENSHOT_SPLIT']))
            panel.setParentItem(None)
            panel.deleteLater()
            window.close()
            window.deleteLater()
            engine.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            settle()

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(_isolated()) if '--isolated' in sys.argv else unittest.main()
