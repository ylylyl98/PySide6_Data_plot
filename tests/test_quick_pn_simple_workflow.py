"""Quick P+N create/apply runs through the owner in an isolated Qt process."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickPnSimpleWorkflowTests(unittest.TestCase):
    def test_native_simple_pn_workflow(self):
        with TemporaryDirectory(prefix='quick-pn-simple-') as tmp:
            env = dict(os.environ, QT_QPA_PLATFORM='offscreen', QT_QUICK_BACKEND='software',
                       MPLBACKEND='agg', MPLCONFIGDIR=tmp, LOCALAPPDATA=tmp,
                       XDG_CONFIG_HOME=tmp, XDG_CACHE_HOME=tmp, PYTHONIOENCODING='utf-8')
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--isolated'],
                cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import time
    from threading import Event
    from unittest.mock import patch
    import numpy as np
    from PySide6.QtCore import QSettings, QUrl
    from PySide6.QtGui import QGuiApplication
    from core.drr_pn_store import DrrPnStore
    from ui_qt.quick.application_bridge import QuickApplicationBridge
    from ui_qt.quick.drr_background_workflow import generate_pn as real_generate

    app = QGuiApplication([])

    def until(predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                app.processEvents()
                return
            time.sleep(.002)
        raise AssertionError('Timed out waiting for P+N Quick workflow')

    class Cases(unittest.TestCase):
        prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_'

        def setUp(self):
            self.temp = TemporaryDirectory(prefix='pn-simple-case-')
            self.addCleanup(self.temp.cleanup)
            self.root = Path(self.temp.name)
            self.bridge = QuickApplicationBridge(settings=QSettings(str(self.root/'settings.ini'), QSettings.IniFormat))
            self.addCleanup(self.bridge.close)
            plot = patch.object(self.bridge, '_schedule_plot')
            plot.start()
            self.addCleanup(plot.stop)
            self.bridge.selectWorkflow('drr')
            self.samples = [self.write(f'TG_rep0{i}', None) for i in (1, 2, 3)]
            self.p = self.write('Pbackground_001', -10)
            self.n = self.write('Nbackground_001', 10)
            self.other_p = self.write('Pbackground_002', -12)
            self.select(*self.samples)
            self.bridge.openDrrBackground()
            self.bg = self.bridge.drrBackgroundBridge
            preview = patch.object(self.bg, '_draw_preview')
            preview.start()
            self.addCleanup(preview.stop)
            self.wait()
            self.bg.setChoice('kind', 3)
            self.wait()

        def write(self, token, constant):
            path = self.root / (self.prefix + token + '.csv')
            wave = np.linspace(630., 690., 41)
            gate = np.linspace(-2., 2., 9) if constant is None else np.full(9, constant)
            data = 150 + np.linspace(-1, 1, len(wave))[None, :] ** 2 + np.zeros((len(gate), len(wave)))
            if constant is None:
                data += 5*gate[:, None]
            np.savetxt(path, np.column_stack((gate, gate, data)), delimiter=',',
                       header='Vbg,Vtg,' + ','.join(map(str, wave)), comments='')
            return path

        def select(self, *paths):
            self.bridge.selectFiles([QUrl.fromLocalFile(str(p)) for p in paths])
            until(lambda: not self.bridge._pending and not self.bridge.snapshot['busy'])

        def wait(self):
            until(lambda: not self.bg.snapshot['busy'] and not self.bridge._pending
                  and not self.bridge.snapshot['busy'])

        def sources(self):
            self.bg.selectFiles('P', [str(self.p)])
            self.bg.selectFiles('N', [str(self.n)])
            self.assertEqual(self.bg.snapshot['pFiles'], [self.p.name])
            self.assertEqual(self.bg.snapshot['nFiles'], [self.n.name])

        def generate(self):
            self.sources()
            captured = []
            def generate_fixture(context, pair, **kwargs):
                captured.append(kwargs)
                # Exercise the real immutable store and trial-load/commit path;
                # numerical reconstruction has separate engine tests.
                return real_generate(context, pair, mode='pn-only')
            with patch('ui_qt.quick.drr_background_workflow.generate_pn', side_effect=generate_fixture):
                self.bg.invoke('pn-generate')
                self.wait()
            self.assertEqual(self.bg.snapshot['error'], '')
            self.assertEqual(len(captured), 1)
            return captured[0]

        def test_create_uses_automatic_training_and_apply_once(self):
            self.bg.setPnMode('create')
            args = self.generate()
            self.assertEqual(args['mode'], 'reconstruction')
            self.assertEqual(tuple(args['training_files']), (self.samples[0].name, self.samples[2].name))
            self.assertTrue(self.bg.snapshot['pnPreviewReady'])
            self.assertIsNone(DrrPnStore(self.root).find_applied(tuple(p.name for p in self.samples)))
            self.assertFalse(self.bridge.snapshot['canExport'])
            self.bg.invoke('pn-apply')
            self.wait()
            self.assertEqual(self.bg.snapshot['error'], '')
            self.assertTrue(self.bridge.snapshot['canExport'], self.bridge.snapshot['status'])
            applied = DrrPnStore(self.root).find_applied(tuple(p.name for p in self.samples))
            self.assertIsNotNone(applied)
            self.assertEqual(self.bridge._drr_background_recipe.draft.version.version_id, applied.version_id)

        def test_edit_source_invalidates_preview_and_keeps_other_side(self):
            self.bg.setPnMode('create')
            self.generate()
            self.bg.selectFiles('P', [str(self.other_p)])
            self.assertEqual(self.bg.snapshot['nFiles'], [self.n.name])
            self.assertEqual(self.bg.snapshot['pFiles'], [self.other_p.name])
            self.assertFalse(self.bg.snapshot['pnPreviewReady'])
            self.assertFalse(self.bg.snapshot['pnCanApply'])
            self.bg.invoke('pn-apply')
            self.wait()
            self.assertIsNone(DrrPnStore(self.root).find_applied(tuple(p.name for p in self.samples)))

        def test_single_repeat_is_ready_and_two_repeats_use_both(self):
            self.select(self.samples[0])
            self.bridge.openDrrBackground()
            self.wait()
            self.bg.setChoice('kind', 3)
            self.bg.setPnMode('create')
            self.sources()
            self.assertTrue(self.bg.snapshot['pnTrainingReady'])
            self.assertTrue(self.bg.snapshot['pnCreateReady'])
            self.assertEqual(self.bg.snapshot['pnTrainingFiles'], [self.samples[0].name])
            self.select(*self.samples[:2])
            self.bridge.openDrrBackground()
            self.wait()
            self.bg.setChoice('kind', 3)
            self.bg.setPnMode('create')
            args = self.generate()
            self.assertEqual(tuple(args['training_files']), tuple(p.name for p in self.samples[:2]))

        def test_training_and_frame_changes_invalidate_preview(self):
            self.bg.setPnMode('create')
            self.generate()
            self.bg.setFlag('training:' + self.samples[0].name, False)
            self.assertFalse(self.bg.snapshot['pnPreviewReady'])
            self.assertFalse(self.bg.snapshot['pnCanApply'])
            self.bg.setFlag('manual_training', False)
            self.generate()
            self.bg.setChoice('pFrame', 0)
            self.assertFalse(self.bg.snapshot['pnPreviewReady'])
            self.assertFalse(self.bg.snapshot['pnCanApply'])

        def test_generated_version_rechecks_sources_before_apply(self):
            self.bg.setPnMode('create')
            self.generate()
            with self.p.open('a') as stream:
                stream.write('\n')
            self.bg.invoke('pn-apply')
            self.wait()
            self.assertFalse(self.bridge.snapshot['canExport'])
            self.assertIsNone(DrrPnStore(self.root).find_applied(tuple(p.name for p in self.samples)))
            self.assertTrue(self.bg.snapshot['error'] or self.bridge.snapshot['status'])

        def test_saved_history_does_not_select_unapplied_latest(self):
            self.bg.setPnMode('create')
            self.generate()
            self.bridge.cancelDrrBackground()
            self.bridge.openDrrBackground()
            self.wait()
            self.bg.setChoice('kind', 3)
            until(lambda: self.bg.snapshot['pnHistoryLoaded'] and not self.bg.snapshot['pnHistoryLoading'])
            self.assertFalse(self.bg.snapshot['pnPreviewReady'])
            self.assertEqual(self.bg.snapshot['choices']['version']['current'], -1)

        def test_late_history_result_cannot_select_for_new_group(self):
            started, release = Event(), Event()
            original = DrrPnStore.list_versions
            def delayed(store):
                started.set()
                if not release.wait(15):
                    raise AssertionError('history worker was not released')
                return original(store)
            self.bridge.cancelDrrBackground()
            with patch.object(DrrPnStore, 'list_versions', delayed):
                try:
                    self.bridge.openDrrBackground()
                    self.wait()
                    self.bg.setChoice('kind', 3)
                    until(started.is_set)
                    self.select(*self.samples[:2])
                finally:
                    release.set()
                until(lambda: not self.bg.snapshot['pnHistoryLoading'])
            self.assertFalse(self.bg.snapshot['pnPreviewReady'])
            self.assertEqual(self.bg.context.measurements, tuple(p.name for p in self.samples[:2]))

        def test_existing_saved_pn_matching_remains_selectable(self):
            self.bg.setChoice('method', 1)
            self.assertEqual(self.bg.draft.kind, 'pn')
            self.assertEqual(self.bg.draft.assignment_method, 'matched')

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    if '--isolated' in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
