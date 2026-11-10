"""File-scoped history handoff and the unified save actions."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QSettings
from PySide6.QtGui import QGuiApplication

from core.peak_workspace import create_dataset, load_workspace, save_workspace
from core.peak_file_history import history_path, save_histories
from tests.test_peak_workspace import sample
from ui_qt.quick.peak_analysis_bridge import PeakAnalysisBridge
from ui_qt.quick.peak_analysis_workflow import analyze_dataset


class PeakSaveWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QGuiApplication.instance() or QGuiApplication([])

    def wait(self, bridge):
        deadline = time.monotonic() + 30
        while bridge._busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertFalse(bridge._busy)
        self.assertFalse(bridge._error, bridge._error)

    def make(self, root, name):
        path = root / 'Processed Data' / 'DRR' / (name + '.dat')
        dataset = create_dataset(sample('DRR'), 'DRR', 'second', str(path), name)
        dataset['view'] = dict(display=dict(cmap='RdBu_r', vmin=-1., vmax=1.))
        dataset.update(processed_path=str(path), history_root=str(root / 'settings'), analysis_workflow_version=3,
                       analysis_scope='map', scope=0)
        return dataset

    def bridge(self, root):
        bridge = PeakAnalysisBridge(settings=QSettings(str(root / 'settings.ini'), QSettings.Format.IniFormat),
                                    session_path=root / 'settings' / 'workspace.npz')
        bridge._schedule_plot = lambda: None
        return bridge

    def test_switch_saves_only_each_files_history_and_restores_choices(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            first, second = self.make(root, 'first'), self.make(root, 'second')
            saved = dict(first, name='saved name')
            saved['settings'] = dict(first['settings'], min_snr=8.)
            save_histories([saved, second], root / 'settings', saved['key'])
            bridge = self.bridge(root)
            try:
                incoming = root / 'incoming.npz'
                save_workspace(incoming, [first], first['key'])
                bridge.loadInitialSnapshot(incoming)
                self.wait(bridge)
                self.assertEqual(set(bridge.datasets), {first['key']})
                self.assertEqual(bridge.dataset['settings']['min_snr'], 8.)
                bridge.dataset['settings']['min_snr'] = 12.
                save_workspace(incoming, [second], second['key'])
                bridge.loadSnapshot(incoming)
                self.wait(bridge)
                self.assertEqual(set(bridge.datasets), {second['key']})
                previous, _ = load_workspace(history_path(first, root / 'settings'))
                self.assertEqual(previous[0]['settings']['min_snr'], 12.)
                self.assertFalse(bridge.session_path.exists())
            finally:
                bridge.close()
            reopened = self.bridge(root)
            try:
                reopened.restore()
                self.wait(reopened)
                self.assertEqual(set(reopened.datasets), {second['key']})
            finally:
                reopened.close()

    def test_automatic_save_data_figures_and_figure_do_not_open_dialog(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            stem = 'YZ382_pa2_9T_6KREF_730nmc_0p1sx20_RotIn39deg_TG-1BG=0_PIXIS_avg1_DR_R_BN_d2E'
            dataset = analyze_dataset(self.make(root, stem), 'detect')
            bridge = self.bridge(root)
            dialogs = []
            bridge.dialogRequested.connect(lambda *args: dialogs.append(args))
            try:
                bridge.addDatasets([dataset])
                destination = history_path(dataset, root / 'settings').parent
                self.assertEqual(Path(bridge.snapshot['saveFolder']), destination)
                bridge.invoke('save-results')
                self.wait(bridge)
                self.assertEqual(destination, Path(dataset['processed_path']).parent)
                self.assertFalse(list(destination.glob('*.csv')))
                self.assertTrue((destination / (stem + '_analysis_settings.json')).is_file())
                self.assertTrue((destination / (stem + '_peak_results.xlsx')).is_file())
                self.assertTrue((destination / (stem + '_peak_overlay.png')).is_file())
                bridge.invoke('save-results')
                self.wait(bridge)
                self.assertTrue((destination / (stem + '_peak_results_02.xlsx')).is_file())
                self.assertTrue((destination / (stem + '_peak_overlay_02.png')).is_file())
                self.assertEqual(len(bridge.snapshot['resultPaths']), 3)
                bridge.invoke('save-figure-auto')
                self.wait(bridge)
                self.assertTrue((destination / (stem + '_figure.png')).is_file())
                bridge.invoke('save-figure-auto')
                self.wait(bridge)
                self.assertTrue((destination / (stem + '_figure_02.png')).is_file())
                self.assertEqual(dialogs, [])
                self.assertTrue(history_path(dataset, root / 'settings').is_file())
                self.assertFalse(any(path.is_dir() for path in destination.iterdir()))
            finally:
                bridge.close()

    def test_existing_window_receives_direct_handoff_without_process_spawn(self):
        from ui_qt.quick.peak_analysis_launcher import QuickPeakAnalysisLauncher
        with TemporaryDirectory() as folder:
            root = Path(folder)
            launcher = QuickPeakAnalysisLauncher(settings=QSettings(str(root / 'settings.ini'), QSettings.Format.IniFormat),
                                                 session_path=root / 'workspace.npz')
            try:
                launcher._operation = launcher._revision = 1
                dataset = self.make(root, 'current')
                with patch('ui_qt.quick.peak_analysis_launcher.send_message', return_value=True) as send, \
                     patch('ui_qt.quick.peak_analysis_launcher.QProcess.startDetached') as spawn:
                    launcher._finish(1, ('prepare', 1, [], [dataset], str(root / 'inbox.npz')), '')
                send.assert_called_once()
                spawn.assert_not_called()
            finally:
                launcher.close()

    def test_wide_figure_action_saves_both_sizes_and_preserves_previous_pair(self):
        from dataclasses import asdict
        from PIL import Image
        from core.plotting import HeatmapParams
        with TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = self.make(root, 'source_d2E')
            cube = dataset['cube']
            params = HeatmapParams(cube.title, 'Energy', cube.gate_label, cube.cbar_label,
                                   -1., 1., (float(cube.energy.min()), float(cube.energy.max())),
                                   (float(cube.gate.min()), float(cube.gate.max())), png_width_scale=2.)
            dataset['source_export'] = dict(params=asdict(params), drr=True)
            bridge = self.bridge(root)
            try:
                bridge.addDatasets([dataset])
                destination = Path(dataset['processed_path']).parent
                bridge.invoke('save-figure-auto')
                self.wait(bridge)
                first = destination / 'source_d2E_figure.png'
                wide = destination / 'source_d2E_figure_2.0x.png'
                original_bytes = {p: p.read_bytes() for p in (first, wide)}
                for path, size in ((first, (1200, 930)), (wide, (2400, 930))):
                    with Image.open(path) as picture:
                        self.assertEqual(picture.size, size)
                bridge.invoke('save-figure-auto')
                self.wait(bridge)
                self.assertEqual({Path(p).name for p in bridge.snapshot['resultPaths']},
                                 {'source_d2E_figure_02.png', 'source_d2E_figure_2.0x_02.png'})
                for path, content in original_bytes.items():
                    self.assertEqual(path.read_bytes(), content)
            finally:
                bridge.close()

    def test_figure_pair_failure_removes_only_new_files(self):
        from core.peak_figure_export import export_source_figures
        from dataclasses import asdict
        from core.peak_source_figure import _fallback_params
        from shutil import copyfileobj
        with TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = self.make(root, 'source_d2E')
            params = asdict(_fallback_params(dataset))
            params['png_width_scale'] = 2.
            dataset['source_export'] = dict(params=params, drr=True)
            original = root / 'figure.png'
            original.write_bytes(b'previous')
            calls = []
            def fail_second(source, target):
                calls.append(1)
                if len(calls) == 2:
                    raise OSError('disk full')
                copyfileobj(source, target)
            with patch('core.peak_figure_export.copyfileobj', side_effect=fail_second):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    export_source_figures(original, dataset)
            self.assertEqual(list(root.iterdir()), [original])
            self.assertEqual(original.read_bytes(), b'previous')
