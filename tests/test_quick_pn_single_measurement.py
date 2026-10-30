"""Single-measurement Quick generation and owner application in a Qt subprocess."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickPnSingleMeasurementTests(unittest.TestCase):
    def test_single_measurement_owner(self):
        with TemporaryDirectory(prefix='quick-pn-single-') as tmp:
            env = dict(os.environ, QT_QPA_PLATFORM='offscreen', QT_QUICK_BACKEND='software',
                       MPLBACKEND='agg', MPLCONFIGDIR=tmp, LOCALAPPDATA=tmp,
                       XDG_CONFIG_HOME=tmp, XDG_CACHE_HOME=tmp, PYTHONIOENCODING='utf-8')
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--isolated'],
                cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import time
    from unittest.mock import patch
    import numpy as np
    from PySide6.QtCore import QSettings, QUrl
    from PySide6.QtGui import QGuiApplication
    from core.drr_pn_reconstruction import pipeline
    from core.drr_pn_store import DrrPnStore
    from ui_qt.quick.application_bridge import QuickApplicationBridge

    application = QGuiApplication([])
    def until(predicate, timeout=30):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            application.processEvents()
            if predicate():
                application.processEvents()
                return
            time.sleep(.002)
        raise AssertionError('Timed out waiting for single P/N workflow')

    class Cases(unittest.TestCase):
        prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_'
        def setUp(self):
            self.temp = TemporaryDirectory(prefix='pn-single-quick-case-')
            self.addCleanup(self.temp.cleanup)
            self.root = Path(self.temp.name)
            self.sample = self.write('TG_rep01', None)
            self.p = self.write('Pbackground_001', -10)
            self.n = self.write('Nbackground_001', 10)
            self.other_p = self.write('Pbackground_002', -12)
            self.bridge = QuickApplicationBridge(settings=QSettings(str(self.root/'settings.ini'), QSettings.IniFormat))
            self.addCleanup(self.bridge.close)
            patcher = patch.object(self.bridge, '_schedule_plot')
            patcher.start()
            self.addCleanup(patcher.stop)
            self.bridge.selectWorkflow('drr')
            self.bridge.selectFiles([QUrl.fromLocalFile(str(self.sample))])
            until(lambda: not self.bridge._pending and not self.bridge.snapshot['busy'])
            self.bridge.openDrrBackground()
            self.bg = self.bridge.drrBackgroundBridge
            preview = patch.object(self.bg, '_draw_preview')
            preview.start()
            self.addCleanup(preview.stop)
            self.wait()
            self.bg.setChoice('kind', 3)
            self.wait()
            self.bg.setPnMode('create')
            self.bg.selectFiles('P', [str(self.p)])
            self.bg.selectFiles('N', [str(self.n)])

        def write(self, token, constant):
            path = self.root / (self.prefix + token + '.csv')
            wave = np.linspace(630., 690., 41)
            gate = np.linspace(-2., 2., 9) if constant is None else np.full(4, constant)
            values = 150 + np.linspace(-1, 1, len(wave))[None, :]**2 + np.zeros((len(gate), len(wave)))
            if constant is None:
                values += 5 * gate[:, None]
            np.savetxt(path, np.column_stack((gate, gate, values)), delimiter=',',
                       header='Vbg,Vtg,' + ','.join(map(str, wave)), comments='')
            return path

        def wait(self):
            until(lambda: not self.bg.snapshot['busy'] and not self.bridge._pending and not self.bridge.snapshot['busy'])

        def generate(self):
            calls = []
            def stub(energy, pn_mean, frames, samples, **kwargs):
                calls.append((samples.shape, kwargs['method']))
                return {'energy': energy.copy(), 'background': pn_mean.mean(axis=0)}, {
                    'algorithm_version': pipeline.CONDITIONED_SINGLE_ALGORITHM_VERSION,
                    'settings': dict(pipeline.CONDITIONED_SINGLE_PROTOCOL), 'BN_used': False}
            with patch.object(pipeline, 'reconstruct_background', side_effect=stub):
                self.bg.invoke('pn-generate')
                self.wait()
            self.assertEqual(self.bg.snapshot['error'], '')
            self.assertEqual(calls, [((1, 9, 41), 'conditioned_single')])

        def test_one_sample_generation_and_apply(self):
            self.assertTrue(self.bg.snapshot['pnTrainingReady'])
            self.assertTrue(self.bg.snapshot['pnCreateReady'])
            self.assertEqual([x['file'] for x in self.bg.snapshot['training'] if x['selected']], [self.sample.name])
            self.generate()
            self.assertTrue(self.bg.snapshot['pnPreviewReady'])
            self.assertIsNone(DrrPnStore(self.root).find_applied((self.sample.name,)))
            self.assertFalse(self.bridge.snapshot['canExport'])
            self.bg.invoke('pn-apply')
            self.wait()
            self.assertEqual(self.bg.snapshot['error'], '')
            self.assertTrue(self.bridge.snapshot['canExport'], self.bridge.snapshot['status'])
            applied = DrrPnStore(self.root).find_applied((self.sample.name,))
            self.assertIsNotNone(applied)
            self.assertEqual(applied.algorithm_version, pipeline.CONDITIONED_SINGLE_ALGORITHM_VERSION)

        def test_saved_single_version_can_be_reloaded_and_applied(self):
            self.generate()
            self.bridge.cancelDrrBackground()
            self.bridge.openDrrBackground()
            self.wait()
            self.bg.setChoice('kind', 3)
            self.bg.setPnMode('saved')
            until(lambda: self.bg.snapshot['pnHistoryLoaded'] and not self.bg.snapshot['pnHistoryLoading'])
            self.assertTrue(self.bg.snapshot['choices']['version']['items'])
            self.bg.setChoice('version', 0)
            self.wait()
            self.assertTrue(self.bg.snapshot['pnPreviewReady'])
            self.bg.invoke('pn-apply')
            self.wait()
            self.assertTrue(self.bridge.snapshot['canExport'], self.bridge.snapshot['status'])
            applied = DrrPnStore(self.root).find_applied((self.sample.name,))
            self.assertIsNotNone(applied)
            self.assertEqual(applied.algorithm_version, pipeline.CONDITIONED_SINGLE_ALGORITHM_VERSION)

        def test_changing_source_invalidates_single_preview(self):
            self.generate()
            self.bg.selectFiles('P', [str(self.other_p)])
            self.assertEqual(self.bg.snapshot['nFiles'], [self.n.name])
            self.assertFalse(self.bg.snapshot['pnPreviewReady'])
            self.assertFalse(self.bg.snapshot['pnCanApply'])
            self.assertIsNone(DrrPnStore(self.root).find_applied((self.sample.name,)))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    if '--isolated' in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
