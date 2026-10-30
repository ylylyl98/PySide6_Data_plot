"""Rendered QML coverage for the native P+N main workflow."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickPnSimpleQmlTests(unittest.TestCase):
    def test_rendered_simple_pn(self):
        with TemporaryDirectory(prefix='quick-pn-qml-') as tmp:
            env = dict(os.environ, QT_QPA_PLATFORM='offscreen', QT_QUICK_BACKEND='software',
                       MPLBACKEND='agg', MPLCONFIGDIR=tmp, LOCALAPPDATA=tmp,
                       XDG_CONFIG_HOME=tmp, XDG_CACHE_HOME=tmp, PYTHONIOENCODING='utf-8')
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--isolated'],
                cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tempfile
    import time
    from unittest.mock import patch
    import numpy as np
    from PySide6.QtCore import QPointF, QSettings, QUrl
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication
    from PySide6.QtQml import QQmlComponent, QQmlEngine
    from PySide6.QtQuick import QQuickItem, QQuickWindow
    from ui_qt.quick.plot_surface import register_plot_surface
    from ui_qt.quick.drr_background_bridge import DrrBackgroundBridge
    from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft, generate_pn as real_generate

    app = QGuiApplication([])
    if os.name == 'nt':
        QFontDatabase.addApplicationFont(str(Path(os.environ['WINDIR'])/'Fonts'/'segoeui.ttf'))
        app.setFont(QFont('Segoe UI', 9))
    register_plot_surface()
    light = dict(surface='#f3f5f8', control='#ffffff', text='#20242b', muted='#606978', border='#c6cbd3',
        accent='#0067c0', hover='#edf1f6', pressed='#e0e7ef', error='#b42318', disabled='#939aa5',
        selectedText='#ffffff', inset='#e9edf2', divider='#dce1e8', borderHover='#aeb7c4', borderDisabled='#e1e5eb',
        controlDisabled='#edf0f4', focus='#20242b', accentText='#0067c0', selected='#e6f0fa', selectedBorder='#b4d6fa', errorSurface='#fdf1f0')
    dark = dict(light, surface='#202020', control='#303030', text='#ffffff', muted='#b0b0b0', border='#606060',
                hover='#404040', pressed='#505050', inset='#252525', divider='#505050', selectedText='#ffffff')

    def until(predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                app.processEvents()
                return
            time.sleep(.002)
        raise AssertionError('Timed out waiting for rendered P+N state')

    class Cases(unittest.TestCase):
        prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_'

        def setUp(self):
            self.temp = TemporaryDirectory(prefix='pn-qml-case-')
            self.addCleanup(self.temp.cleanup)
            self.root = Path(self.temp.name)
            self.samples = [self.write(f'TG_rep0{i}', None) for i in (1, 2, 3)]
            self.p = self.write('Pbackground_' + 'long_name_'*7 + '001', -10)
            self.n = self.write('Nbackground_001', 10)
            self.bridge = DrrBackgroundBridge(settings=QSettings(str(self.root/'settings.ini'), QSettings.IniFormat))
            self.addCleanup(self.bridge.close)
            self.bridge.open(BackgroundContext(str(self.root), tuple(p.name for p in self.samples)))
            until(lambda: not self.bridge.snapshot['busy'])
            self.bridge.setChoice('kind', 3)
            until(lambda: not self.bridge.snapshot['busy'] and not self.bridge.snapshot['pnHistoryLoading'])
            self.engine = QQmlEngine()
            self.warnings = []
            self.engine.warnings.connect(lambda messages: self.warnings.extend(m.toString() for m in messages))
            path = Path(__file__).resolve().parents[1]/'ui_qt'/'quick'/'qml'/'DrrBackgroundSettings.qml'
            self.component = QQmlComponent(self.engine, QUrl.fromLocalFile(str(path)))
            self.item = self.component.createWithInitialProperties(dict(bridge=self.bridge, colors=light))
            self.assertIsNotNone(self.item, '\n'.join(e.toString() for e in self.component.errors()))
            self.window = QQuickWindow()
            self.item.setParentItem(self.window.contentItem())
            self.window.show()
            self.addCleanup(self.cleanup_window)
            self.resize(1060, 800)

        def cleanup_window(self):
            self.window.close()
            self.item.setParentItem(None)
            self.item.deleteLater()
            app.processEvents()

        def write(self, token, constant):
            path = self.root / (self.prefix + token + '.csv')
            wave = np.linspace(630., 690., 41)
            gate = np.linspace(-2., 2., 9) if constant is None else np.full(9, constant)
            data = 150 + np.linspace(-1, 1, len(wave))[None, :]**2 + np.zeros((len(gate), len(wave)))
            if constant is None:
                data += 5*gate[:, None]
            np.savetxt(path, np.column_stack((gate, gate, data)), delimiter=',',
                header='Vbg,Vtg,' + ','.join(map(str, wave)), comments='')
            return path

        def resize(self, width, height):
            self.window.resize(width, height)
            self.item.setWidth(width)
            self.item.setHeight(height)
            until(lambda: not self.window.grabWindow().isNull())
            app.processEvents()

        def visual_items(self):
            found, pending = [], [self.item]
            while pending:
                item = pending.pop()
                found.append(item)
                pending.extend(item.childItems())
            return found

        def get(self, name):
            item = next((item for item in self.visual_items() if item.objectName() == name), None)
            self.assertIsNotNone(item, name)
            return item

        def test_layout_modes_and_preview_apply(self):
            for colors, size in ((light, (1060, 800)), (dark, (760, 600))):
                self.item.setProperty('colors', colors)
                self.resize(*size)
                self.assertEqual(self.bridge.snapshot['pnMode'], 'create')
                for name in ('drr_pn_simple_workflow', 'drr_pn_choose_P', 'drr_pn_choose_N',
                             'drr_pn_generate', 'drr_pn_training_summary'):
                    item = self.get(name)
                    self.assertTrue(item.isVisible(), name)
                    self.assertGreater(item.width(), 0, name)
                    self.assertGreater(item.height(), 0, name)
                apply = self.get('drr_background_apply')
                self.assertEqual(apply.property('text'), 'Apply to current group')
                self.assertFalse(apply.isEnabled())
                origin = apply.mapToScene(QPointF())
                self.assertGreaterEqual(origin.x(), 0)
                self.assertGreaterEqual(origin.y(), 0)
                self.assertLessEqual(origin.x() + apply.width(), size[0])
                self.assertLessEqual(origin.y() + apply.height(), size[1])
                self.assertFalse(self.window.grabWindow().isNull())
            self.bridge.selectFiles('P', [str(self.p)])
            self.bridge.selectFiles('N', [str(self.n)])
            self.resize(1060, 800)
            self.item.setProperty('colors', light)
            # A long selected file is rendered through the wrapping source card.
            texts = [item for item in self.visual_items()
                     if item.property('text') and self.p.name in str(item.property('text'))]
            self.assertTrue(texts)
            self.assertGreater(texts[0].height(), 20)
            self.window.grabWindow().save(str(Path(tempfile.gettempdir())/'drr-pn-simple-create.png'))
            def generate_fixture(context, pair, **kwargs):
                return real_generate(context, pair, mode='pn-only')
            with patch('ui_qt.quick.drr_background_workflow.generate_pn', side_effect=generate_fixture):
                self.bridge.invoke('pn-generate')
                until(lambda: not self.bridge.snapshot['busy'])
            self.assertEqual(self.bridge.snapshot['error'], '')
            self.assertTrue(self.bridge.snapshot['pnPreviewReady'])
            until(lambda: self.bridge.plotBackend is not None and self.bridge.plotBackend.frameGeneration > 0)
            self.assertTrue(self.get('drr_pn_simple_plot').isVisible())
            self.assertFalse(self.window.grabWindow().isNull())
            self.window.grabWindow().save(str(Path(tempfile.gettempdir())/'drr-pn-simple-preview.png'))
            self.bridge.setPnMode('saved')
            until(lambda: self.get('drr_pn_saved_version').isVisible())
            self.assertTrue(self.bridge.snapshot['choices']['version']['items'])
            self.assertTrue(self.bridge.snapshot['pnPreviewReady'])
            self.window.grabWindow().save(str(Path(tempfile.gettempdir())/'drr-pn-simple-saved.png'))
            actions = []
            self.bridge.applyRequested.connect(actions.append)
            self.assertTrue(self.get('drr_background_apply').isEnabled())
            self.get('drr_background_apply').clicked.emit()
            until(lambda: not self.bridge.snapshot['busy'])
            self.assertEqual(len(actions), 1)
            self.assertFalse(self.warnings, '\n'.join(self.warnings))

        def test_single_measurement_render_and_group_change(self):
            self.bridge.selectFiles('P', [str(self.p)])
            self.bridge.selectFiles('N', [str(self.n)])
            def generate_fixture(context, pair, **kwargs):
                return real_generate(context, pair, mode='pn-only')
            with patch('ui_qt.quick.drr_background_workflow.generate_pn', side_effect=generate_fixture):
                self.bridge.invoke('pn-generate')
                until(lambda: not self.bridge.snapshot['busy'])
            self.assertTrue(self.bridge.snapshot['pnPreviewReady'])
            self.bridge.open(BackgroundContext(str(self.root), (self.samples[0].name,)), BackgroundDraft(kind='pn'))
            until(lambda: not self.bridge.snapshot['pnHistoryLoading'])
            self.assertFalse(self.bridge.snapshot['pnPreviewReady'])
            self.assertFalse(self.get('drr_pn_generate').isEnabled())
            self.bridge.selectFiles('P', [str(self.p)])
            self.bridge.selectFiles('N', [str(self.n)])
            until(lambda: self.get('drr_pn_generate').isEnabled())
            self.assertEqual(self.bridge.snapshot['pnTrainingFiles'], [self.samples[0].name])
            self.assertTrue(self.bridge.snapshot['pnSingleMeasurement'])
            for label, colors, size in (('light', light, (1060, 800)), ('dark', dark, (1060, 800)),
                                        ('narrow', light, (760, 600))):
                self.item.setProperty('colors', colors)
                self.resize(*size)
                summary = self.get('drr_pn_training_summary')
                self.assertTrue(summary.isVisible())
                self.assertIn('Single measurement draft', summary.property('text'))
                self.assertIn('repeatability unavailable', summary.property('text'))
                self.assertTrue(self.get('drr_pn_generate').isEnabled())
                text_items = [item for item in self.visual_items() if item.isVisible()
                    and item.property('text') == 'Single measurement draft']
                self.assertTrue(text_items, label)
                target = Path(tempfile.gettempdir()) / ('drr-pn-single-create.png' if label == 'light'
                    else 'drr-pn-single-create-' + label + '.png')
                self.assertTrue(self.window.grabWindow().save(str(target)))
            self.assertFalse(self.warnings, '\n'.join(self.warnings))

        def test_saved_mode_and_external_picker_return(self):
            self.bridge.setPnMode('saved')
            until(lambda: self.bridge.snapshot['pnHistoryLoaded'])
            self.assertTrue(self.get('drr_pn_saved_version').isVisible())
            self.assertTrue(self.get('drr_pn_choose_saved_files').isVisible())
            self.get('drr_pn_choose_saved_files').clicked.emit()
            app.processEvents()
            self.assertTrue(self.item.property('pnExternalFiles'))
            self.assertTrue(self.item.property('inlineReferencesVisible'))
            back = next(item for item in self.visual_items() if item.property('text') == 'Back to P+N' and item.isVisible())
            back.clicked.emit()
            app.processEvents()
            until(lambda: self.get('drr_pn_saved_version').isVisible())
            self.assertFalse(self.warnings, '\n'.join(self.warnings))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    if '--isolated' in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
