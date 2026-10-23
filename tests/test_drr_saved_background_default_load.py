"""Default Self loads honor the latest export for the exact selected group."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication

from core.drr_sources import DrrMeasurementAssignment, discover_drr_sources
from ui_qt import drr_background_settings_workflow as workflow
from ui_qt.drr_background_settings import BackgroundDraft
from ui_qt.main_window import MainWindow
from ui_qt.theme import install_theme


class SavedBackgroundDefaultLoadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode='light')

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        with patch.object(MainWindow, '_restore_last_folder', lambda _self: None), patch.object(
                MainWindow, '_schedule_automatic_update_check', lambda _self: None):
            self.window = MainWindow()
        self.window.current_folder = str(self.root)
        self.measurements = ('first_REF_760nmc.csv', 'second_REF_760nmc.csv')
        self.p = 'reference_Pbackground_REF_760nmc.csv'
        self.bn = 'reference_BNbackground_REF_760nmc.csv'
        for name in self.measurements:
            self.write(name, 20, varying=True)
        self.write(self.p, 10)
        self.write(self.bn, 5)
        self.window.drr_available_sources = discover_drr_sources(self.root)
        self.window.drr_selected_files = list(self.measurements)
        workflow._silent(self.window.drr_baseline_combo, 'Self (last frame)')
        workflow._silent(self.window.drr_pin_baseline_chk, False)
        workflow._silent(self.window.drr_bn_auto_chk, False)
        self.window._drr_baseline_user_selected = False
        self.window._drr_assignments = ()
        self.window._drr_assignments_automatic = False

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def write(self, name, value, *, varying=False):
        (self.root/name).write_text('Vbg,Vtg,740,760,780\n'
            + f'0,0,{value},{value},{value}\n'
            + f'{1 if varying else 0},{1 if varying else 0},{value},{value},{value}\n')

    def save(self, name, measurements, background, *, stamp):
        output = self.root/'Processed Data/DRR'
        output.mkdir(parents=True, exist_ok=True)
        assignments = tuple(DrrMeasurementAssignment(item, 'External', (background,), 'all')
                            for item in measurements)
        metadata = output/f'{name}.metadata.json'
        metadata.write_text(json.dumps({'operation': 'DR/R', 'sources': [
            *({'role': 'measurement', 'source_path': str(self.root/item)} for item in measurements),
            {'role': 'background', 'source_path': str(self.root/background)}],
            'processing': {'drr_background_assignments': [item.to_dict() for item in assignments]}}))
        os.utime(metadata, (stamp, stamp))
        return assignments

    def capture_options(self):
        jobs = []
        with patch.object(self.window.thread_pool, 'start', jobs.append):
            self.window._start_load('DRR')
        self.assertEqual(len(jobs), 1)
        return jobs[0].args[0]

    def load(self, options):
        signal = SimpleNamespace(emit=lambda *_: None)
        return self.window._load_task(options, progress=signal, log=signal)

    def test_default_self_restores_exact_group_before_newer_member_export(self):
        expected = self.save('group', self.measurements, self.p, stamp=1000)
        self.save('newer-single', self.measurements[:1], self.bn, stamp=2000)
        options = self.capture_options()
        self.assertTrue(options.drr_resolve_background)
        loaded = self.load(options)
        self.assertEqual([(item.baseline_files, item.baseline_which) for item in loaded.drr_assignments],
                         [(item.baseline_files, item.baseline_which) for item in expected])
        np.testing.assert_allclose(loaded.cube.Z, 1.)

    def test_exact_group_restoration_replaces_previous_self_intent_on_delivery(self):
        w = self.window
        self.save('group', self.measurements, self.p, stamp=1000)
        w._drr_background_intent = BackgroundDraft(kind='self', which='last')
        w._drr_background_intent_root = workflow._root_key(w)
        w._drr_background_intent_selection = ('previous_REF_760nmc.csv',)
        options = self.capture_options()
        loaded = self.load(options)
        token = w._active_load_token
        with patch.object(w, '_plot_mode'), patch.object(w.thread_pool, 'start'):
            w._on_loaded(loaded, request_token=token)
        self.assertEqual(workflow.source_intent(w).kind, 'p')
        self.assertEqual(workflow.source_intent(w).files, (self.p,))
        self.assertEqual(w._drr_background_intent_selection, self.measurements)
        self.assertFalse(w._drr_assignments_automatic)
        self.assertTrue(w.drr_pin_baseline_chk.isEnabled())
        self.assertEqual(w.drr_baseline_combo.currentText(), 'External')
        with patch.object(w, '_start_load') as reload:
            w._ensure_loaded_matches_drr_params()
        reload.assert_not_called()
        w._on_load_finished(request_token=token)

    def test_invalid_latest_exact_group_does_not_fall_back_to_newer_member_export(self):
        self.save('group', self.measurements, self.p, stamp=1000)
        self.save('latest-group', self.measurements, 'missing.csv', stamp=2000)
        self.save('newer-single', self.measurements[:1], self.bn, stamp=3000)
        self.save('newest-single', self.measurements[1:], self.bn, stamp=4000)
        with self.assertRaisesRegex(ValueError, 'recorded baseline missing'):
            self.load(self.capture_options())

    def test_no_history_retains_default_self(self):
        loaded = self.load(self.capture_options())
        self.assertTrue(all(item.baseline_mode == 'Self (last frame)'
                            for item in loaded.drr_assignments))
        self.assertEqual(loaded.drr_background_selection['selection_method'], 'explicit_per_measurement')
        self.assertFalse(loaded.drr_background_selection.get('restored_exact_group_export', False))

    def test_explicit_self_and_manual_background_do_not_query_export_history(self):
        w = self.window
        self.save('group', self.measurements, self.p, stamp=1000)
        with patch('core.drr_saved_background.resolve_saved_group_background',
                   side_effect=AssertionError('Explicit choices must remain authoritative.')):
            w._drr_baseline_user_selected = True
            loaded = self.load(self.capture_options())
            self.assertTrue(all(item.baseline_mode == 'Self (last frame)'
                                for item in loaded.drr_assignments))
            w._on_load_finished(request_token=w._active_load_token)
            workflow._silent(w.drr_baseline_combo, 'External')
            w.drr_baseline_files_manual = [self.bn]
            loaded = self.load(self.capture_options())
        self.assertTrue(all(item.baseline_files == (self.bn,) for item in loaded.drr_assignments))


if __name__ == '__main__':
    unittest.main()
