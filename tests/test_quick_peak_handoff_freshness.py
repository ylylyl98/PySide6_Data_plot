"""The child must never be launched with a superseded snapshot."""
import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QSettings
from PySide6.QtGui import QGuiApplication

from ui_qt.quick.peak_analysis_launcher import QuickPeakAnalysisLauncher


class PeakHandoffFreshnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QGuiApplication.instance() or QGuiApplication([])

    def test_changed_source_discards_snapshot_and_recaptures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            launcher = QuickPeakAnalysisLauncher(settings=QSettings(str(root / 'settings.ini'), QSettings.Format.IniFormat),
                                                 session_path=root / 'workspace.npz')
            try:
                stale = root / 'stale.npz'
                stale.write_bytes(b'old')
                launcher._busy = True
                launcher._operation = 1
                launcher._revision = 2
                launcher._sources = [{'kind': 'DRR', 'channel': 'raw'}]
                with patch('ui_qt.quick.peak_analysis_launcher.QProcess.startDetached') as launch, \
                     patch.object(launcher, '_prepare') as prepare:
                    launcher._finish(1, ('prepare', 1, [], [], str(stale)), '')
                launch.assert_not_called()
                prepare.assert_called_once_with(True)
                self.assertFalse(stale.exists())
            finally:
                launcher.close()

    def test_snapshot_shows_current_and_keeps_recoverable_history(self):
        import numpy as np
        from core.loader import DataCube
        from core.peak_workspace import create_dataset, save_workspace, load_workspace
        from ui_qt.quick.peak_analysis_bridge import PeakAnalysisBridge
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bridge = PeakAnalysisBridge(settings=QSettings(str(root / 'settings.ini'), QSettings.Format.IniFormat),
                                        session_path=root / 'workspace.npz')
            try:
                def dataset(name, scale):
                    cube = DataCube(np.array([1., 2., 3.]), np.array([0., 1.]),
                                    np.ones((2, 3)) * scale, gate_label='Gate', title=name, cbar_label='DRR')
                    return create_dataset(cube, 'DRR', 'raw', name, name)
                old, current = dataset('old.csv', 1), dataset('new.csv', 2)
                bridge.addDatasets([old])
                bridge._generation = 1
                bridge._busy = True
                bridge._finish(1, 'snapshot', (None, ([current], current['key'])), '', '', 0.)
                self.assertEqual(bridge.active_key, current['key'])
                self.assertEqual([x['key'] for x in bridge.snapshot['datasetChoices']], [current['key']])
                self.assertEqual(bridge.snapshot['historyCount'], 1)
                bridge.setParameter('showHistory', True)
                choices = bridge.snapshot['datasetChoices']
                self.assertEqual(len(choices), 2)
                self.assertIn('(previous)', choices[0]['label'])
                bridge.setParameter('dataset', old['key'])
                self.assertEqual(bridge.active_key, old['key'])
                bridge.setParameter('showHistory', False)
                self.assertEqual(bridge.active_key, current['key'])
                save_workspace(root / 'saved.npz', list(bridge.datasets.values()), bridge.active_key)
                saved, _ = load_workspace(root / 'saved.npz')
                self.assertEqual({item['key'] for item in saved}, {old['key'], current['key']})
            finally:
                bridge.close()

    def test_removed_source_does_not_launch_or_recapture(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            launcher = QuickPeakAnalysisLauncher(settings=QSettings(str(root / 'settings.ini'), QSettings.Format.IniFormat),
                                                 session_path=root / 'workspace.npz')
            try:
                stale = root / 'stale.npz'
                stale.write_bytes(b'old')
                launcher._busy = True
                launcher._operation = 1
                launcher._revision = 2
                with patch('ui_qt.quick.peak_analysis_launcher.QProcess.startDetached') as launch, \
                     patch.object(launcher, '_prepare') as prepare:
                    launcher._finish(1, ('prepare', 1, [], [], str(stale)), '')
                launch.assert_not_called()
                prepare.assert_not_called()
                self.assertFalse(stale.exists())
            finally:
                launcher.close()

    def test_fresh_drr_migrates_implicit_scope_and_preserves_explicit_scope(self):
        from copy import deepcopy
        import numpy as np
        from core.loader import DataCube
        from core.peak_workspace import create_dataset
        from ui_qt.quick.peak_analysis_bridge import PeakAnalysisBridge
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bridge = PeakAnalysisBridge(settings=QSettings(str(root / 'settings.ini'), QSettings.Format.IniFormat),
                                        session_path=root / 'workspace.npz')
            try:
                cube = DataCube(np.array([1., 2., 3.]), np.array([0., 1.]), np.ones((2, 3)),
                                gate_label='Gate', title='DRR', cbar_label='DRR')
                old = create_dataset(cube, 'DRR', 'raw', 'source', 'source')
                old.update(analysis_scope='spectrum', scope=2, candidates={'stale': True}, result=None)
                bridge.addDatasets([old])
                incoming = deepcopy(old)
                incoming.update(analysis_scope='map', candidates=None)
                incoming['settings']['x_min'] = 1.5
                bridge._generation = 1
                bridge._busy = True
                bridge._finish(1, 'snapshot', (None, ([incoming], incoming['key'])), '', '', 0.)
                self.assertEqual(old['analysis_scope'], 'map')
                self.assertEqual(old['settings']['x_min'], 1.5)
                self.assertIsNone(old['candidates'])
                bridge.setParameter('analysisScope', 'spectrum')
                bridge._generation = 2
                bridge._busy = True
                bridge._finish(2, 'snapshot', (None, ([incoming], incoming['key'])), '', '', 0.)
                self.assertEqual(old['analysis_scope'], 'spectrum')
                self.assertTrue(old['analysis_scope_explicit'])
            finally:
                bridge.close()

    def test_pl_fresh_handoff_migrates_only_implicit_legacy_scope(self):
        from copy import deepcopy
        import numpy as np
        from core.loader import DataCube
        from core.peak_workspace import create_dataset
        from ui_qt.quick.peak_analysis_bridge import PeakAnalysisBridge
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bridge = PeakAnalysisBridge(settings=QSettings(str(root / 'settings.ini'), QSettings.Format.IniFormat),
                                        session_path=root / 'workspace.npz')
            try:
                cube = DataCube(np.array([1., 2., 3.]), np.array([0., 1.]), np.ones((2, 3)),
                                gate_label='Gate', title='PL', cbar_label='Intensity')
                old = create_dataset(cube, 'PL', 'PL', 'source', 'source')
                old.update(analysis_scope='spectrum', analysis_workflow_version=2,
                           analysis_scope_explicit=False, scope=2, candidates=None, result=None)
                # Opening a saved workspace preserves its single-spectrum choice.
                bridge._generation = 1; bridge._busy = True
                bridge._finish(1, 'workspace', (None, ([deepcopy(old)], old['key'])), '', '', 0.)
                saved = bridge.datasets[old['key']]
                self.assertEqual(saved['analysis_scope'], 'spectrum')
                saved['candidates'] = {'stale': True}
                incoming = deepcopy(old)
                bridge._generation = 2; bridge._busy = True
                bridge._finish(2, 'snapshot', (None, ([incoming], incoming['key'])), '', '', 0.)
                self.assertEqual(saved['analysis_scope'], 'map')
                self.assertEqual(saved['scope'], 0)
                self.assertIsNone(saved['candidates'])
                bridge.setParameter('analysisScope', 'spectrum')
                incoming = deepcopy(old)
                bridge._generation = 3; bridge._busy = True
                bridge._finish(3, 'snapshot', (None, ([incoming], incoming['key'])), '', '', 0.)
                self.assertEqual(saved['analysis_scope'], 'spectrum')
                self.assertTrue(saved['analysis_scope_explicit'])
            finally:
                bridge.close()

    def test_pl_legacy_first_handoff_defaults_to_map_but_explicit_does_not(self):
        from copy import deepcopy
        import numpy as np
        from core.loader import DataCube
        from core.peak_workspace import create_dataset
        from ui_qt.quick.peak_analysis_bridge import PeakAnalysisBridge
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cube = DataCube(np.array([1., 2., 3.]), np.array([0., 1.]), np.ones((2, 3)),
                            gate_label='Gate', title='PL', cbar_label='Intensity')
            for explicit in (False, True):
                bridge = PeakAnalysisBridge(settings=QSettings(str(root / f'{explicit}.ini'), QSettings.Format.IniFormat),
                                            session_path=root / f'{explicit}.npz')
                try:
                    old = create_dataset(cube, 'PL', 'PL', 'source', 'source')
                    old.update(analysis_scope='spectrum', analysis_workflow_version=2,
                               analysis_scope_explicit=explicit, scope=2)
                    bridge._generation = 1; bridge._busy = True
                    bridge._finish(1, 'snapshot', (None, ([deepcopy(old)], old['key'])), '', '', 0.)
                    self.assertEqual(bridge.datasets[old['key']]['analysis_scope'], 'spectrum' if explicit else 'map')
                finally:
                    bridge.close()


if __name__ == '__main__':
    unittest.main()
