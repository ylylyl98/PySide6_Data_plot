from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_mcd_retention_export_integration as integration

_window = integration._window


class McdSaveScopeTests(unittest.TestCase):
    setUpClass = classmethod(integration.McdRetentionExportIntegrationTests.setUpClass.__func__)
    _window_with_result = integration.McdRetentionExportIntegrationTests._window_with_result

    def setUp(self):
        self.window = self._window_with_result()
        self.window._plot_mcd_unified()
        self.addCleanup(self.window.close)
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)
        self.window.loaded.mcd_result.source_file = str(Path(self.output.name) / 'synthetic.csv')
        self.workers = []
        self.messages = []
        self.enterContext(patch.object(self.window.thread_pool, 'start', side_effect=self.workers.append))
        self.enterContext(patch.object(self.window, '_status', side_effect=self.messages.append))

    def test_current_save_ignores_checked_old_window(self):
        w = self.window
        w.mcd_retention.store.add_window(_window(w), included=True)
        w._queue_unified_mcd_export(self.output.name)
        self.assertEqual(len(self.workers), 1, self.messages)
        snapshot = self.workers[0].args[0]
        self.assertEqual(snapshot['windows'][0]['center_ev'], w.mcd_window_center_spin.value())
        self.assertNotEqual(snapshot['windows'][0]['id'], 'retained-window')
        self.assertEqual(snapshot['provenance']['save_scope'], 'current')

    def test_pending_center_does_not_export_old_slopes(self):
        self.window.mcd_window_center_spin.setValue(1.66)
        self.window._queue_unified_mcd_export(self.output.name)
        self.assertEqual(self.workers, [])
        self.assertTrue(any('updating' in m for m in self.messages), self.messages)

    def test_current_save_after_coalesced_window_refresh(self):
        w = self.window
        for control, value in ((w.mcd_window_center_spin, 1.66),
                               (w.mcd_window_width_spin, 12.0)):
            with self.subTest(control=control.objectName(), value=value):
                control.setValue(value)
                w.mcd_controller._mcd_center_refresh_timer.stop()
                w.mcd_controller._apply_pending_mcd_center_refresh()
                w._queue_unified_mcd_export(self.output.name)
                self.assertTrue(self.workers, self.messages)
                worker = self.workers.pop()
                saved = worker.args[0]['windows'][0]
                self.assertEqual(saved['center_ev'], 1.66)
                self.assertEqual(saved['width_mev'], w.mcd_window_width_spin.value())
                w._on_unified_mcd_export_finished(worker)

    def test_retained_empty_never_falls_back_to_current(self):
        self.window._queue_unified_mcd_export(self.output.name, scope='retained')
        self.assertEqual(self.workers, [])

    def test_busy_feature_does_not_block_current_window(self):
        w = self.window
        w._mcd_unified_track_worker = object()
        w._queue_unified_mcd_export(self.output.name)
        self.assertEqual(len(self.workers), 1, self.messages)
        self.assertEqual(len(self.workers[0].args[0]['analysis_results']), 0)
        w._mcd_unified_track_worker = None

    def test_retained_saves_while_current_feature_is_busy(self):
        w = self.window
        w.mcd_retention.store.add_window(_window(w), included=True)
        w._mcd_unified_track_worker = object()
        w._queue_unified_mcd_export(self.output.name, scope='retained')
        self.assertEqual(len(self.workers), 1, self.messages)
        self.assertEqual(self.workers[0].args[0]['windows'][0]['id'], 'retained-window')
        w._mcd_unified_track_worker = None

    def test_destination_remembered_and_snapshot_precedes_dialog(self):
        w = self.window
        w._mcd_export_output_root = str(Path(self.output.name) / 'missing')
        before = w.mcd_window_center_spin.value()
        def choose(*args):
            w.mcd_window_center_spin.setValue(1.66)
            return self.output.name
        with patch('ui_qt.main_window.QFileDialog.getExistingDirectory', side_effect=choose) as dialog:
            w._queue_unified_mcd_export()
            self.assertEqual(len(self.workers), 1, self.messages)
            self.assertEqual(self.workers[0].args[0]['windows'][0]['center_ev'], before)
            w._on_unified_mcd_export_finished(self.workers[0])
            w._plot_mcd_unified()
            w._queue_unified_mcd_export()
            self.assertEqual(dialog.call_count, 1)
        self.assertEqual(len(self.workers), 2)

    def test_cancel_leaves_no_worker_or_destination(self):
        with patch.object(self.window, '_unified_mcd_export_folder_writable', return_value=False), patch('ui_qt.main_window.QFileDialog.getExistingDirectory', return_value=''):
            self.window._queue_unified_mcd_export()
        self.assertEqual(self.workers, [])
        self.assertFalse(getattr(self.window, '_mcd_export_output_root', None))

    def test_default_save_follows_original_source_without_dialog(self):
        w = self.window
        w.current_folder = self.output.name
        for name in ('first', 'second'):
            with self.subTest(source=name):
                dataset_dir = Path(self.output.name) / name
                source_dir = dataset_dir / 'mcd'
                source_dir.mkdir(parents=True)
                w.loaded.mcd_result.source_file = str(source_dir / 'measurement.csv')
                w._plot_mcd_unified()
                expected = dataset_dir / 'Processed Data' / 'MCD'
                self.assertIn(str(expected), w.mcd_unified_controls.save_folder_label.text())
                with patch('ui_qt.main_window.QFileDialog.getExistingDirectory', return_value='') as dialog:
                    w._queue_unified_mcd_export()
                self.assertTrue(self.workers, self.messages)
                worker = self.workers.pop()
                self.assertEqual(Path(worker.args[1]), expected)
                self.assertTrue(expected.is_dir())
                dialog.assert_not_called()
                self.assertIn(str(expected), w.mcd_unified_controls.save_folder_label.text())
                w._on_unified_mcd_export_finished(worker)

    def test_manual_folder_overrides_source_default(self):
        w = self.window
        with patch('ui_qt.main_window.QFileDialog.getExistingDirectory', return_value=self.output.name):
            w._change_unified_mcd_export_folder()
        with patch('ui_qt.main_window.QFileDialog.getExistingDirectory', return_value='') as dialog:
            w._queue_unified_mcd_export()
        self.assertEqual(Path(self.workers[0].args[1]), Path(self.output.name))
        dialog.assert_not_called()

    def test_unwritable_remembered_folder_prompts_again(self):
        w = self.window
        w._mcd_export_output_root = self.output.name
        with patch('ui_qt.main_window.tempfile.TemporaryFile', side_effect=PermissionError('read only')), patch('ui_qt.main_window.QFileDialog.getExistingDirectory', return_value='') as dialog:
            w._queue_unified_mcd_export()
        dialog.assert_called_once()
        self.assertEqual(self.workers, [])

    def test_stale_feature_is_omitted_and_toggle_excludes_numerics(self):
        w = self.window
        w.mcd_unified_view.set_candidates([{'id': 'f', 'domain': 'mcd', 'source': 'mcd', 'kind': 'peak', 'center_ev': 1.64}])
        w._mcd_unified_track_payload = {'status': 'ok', 'tracks': [{'id': 'old'}]}
        w._mcd_unified_track_payload_key = (-1, w._unified_feature_analysis_key())
        w._queue_unified_mcd_export(self.output.name)
        self.assertEqual(len(self.workers[0].args[0]['analysis_results']), 0)
        w._on_unified_mcd_export_finished(self.workers[0])
        w._mcd_unified_track_payload_key = (w.mcd_unified_view.state.source_generation, w._unified_feature_analysis_key())
        w.mcd_unified_controls.include_feature_chk.setChecked(False)
        w._queue_unified_mcd_export(self.output.name)
        self.assertEqual(len(self.workers[1].args[0]['analysis_results']), 0)

    def test_inspected_retained_feature_cannot_reuse_current_completion_key(self):
        w = self.window
        w._mcd_unified_features = [{'id': 'f', 'domain': 'mcd', 'source': 'mcd', 'kind': 'peak', 'center_ev': 1.64}]
        w.mcd_unified_view.set_candidates(w._unified_mcd_candidates())
        retained = w.mcd_retention.add_feature({'id': 'f', 'source_generation': w.mcd_unified_view.state.source_generation, 'track_payload': {'status': 'ok', 'tracks': [{'id': 'old'}]}, 'status': 'complete'}, included=True)
        w._mcd_unified_track_payload = {'status': 'ok', 'tracks': [{'id': 'new'}]}
        w._mcd_unified_track_payload_key = (w.mcd_unified_view.state.source_generation, w._unified_feature_analysis_key())
        w._inspect_retained_snapshot('feature', retained)
        w._queue_unified_mcd_export(self.output.name)
        self.assertEqual(len(self.workers[0].args[0]['analysis_results']), 0)

    def test_duplicate_click_and_finish_respect_active_tab(self):
        w = self.window
        with patch.object(w, '_active_mode', return_value='MCD'):
            w.last_plotted_mode = 'MCD'
            w._queue_unified_mcd_export(self.output.name)
            self.assertFalse(w.save_action.isEnabled())
            w._queue_unified_mcd_export(self.output.name)
            self.assertEqual(len(self.workers), 1)
        with patch.object(w, '_active_mode', return_value='PL'):
            w.last_plotted_mode = 'PL'
            w._update_action_states()
            self.assertTrue(w.save_action.isEnabled())
            w._export_in_progress = True
            w._on_unified_mcd_export_finished(self.workers[0])
            self.assertFalse(w.save_action.isEnabled())
        self.assertFalse(w.mcd_unified_controls.save_results_btn.isEnabled())

    def test_error_restores_buttons_and_frozen_arrays_survive_changes(self):
        w = self.window
        w._queue_unified_mcd_export(self.output.name)
        worker = self.workers[0]
        trace = worker.args[0]['windows'][0]['trace_values'].copy()
        w.loaded.mcd_result.pair_mcd_corrected[:] = 123
        import numpy as np
        np.testing.assert_array_equal(worker.args[0]['windows'][0]['trace_values'], trace)
        w._on_unified_mcd_export_error(worker, 'disk full')
        w._on_unified_mcd_export_finished(worker)
        self.assertFalse(w.mcd_unified_controls.save_results_btn.isEnabled())
        self.assertTrue(w.mcd_unified_controls.change_save_folder_btn.isEnabled())
        self.assertTrue(any('disk full' in m for m in self.messages))

    def test_top_save_routes_current_and_side_button_routes_retained(self):
        self.window.mcd_retention.add_window(_window(self.window), included=True)
        with patch.object(self.window, '_active_mode', return_value='MCD'), patch.object(self.window, '_queue_unified_mcd_export') as queue:
            self.window._toolbar_save()
            queue.assert_called_once_with(scope='current')
            queue.reset_mock()
            self.window.mcd_unified_controls.save_results_btn.click()
            queue.assert_called_once_with(scope='retained')


if __name__ == '__main__':
    unittest.main()
