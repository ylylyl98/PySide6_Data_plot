import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
import tempfile
import json
from pathlib import Path
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from core.mcd_center_history import merge_center_candidates
from tests.test_mcd_unified_workflow import _result
from ui_qt.main_window import MainWindow, LoadedState
from ui_qt.mcd_unified_page import McdUnifiedView


class CenterHistoryUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_history_and_recommendations_share_map_and_linecut_ids(self):
        v = McdUnifiedView()
        self.addCleanup(v.close)
        history = [{'center_ev': 1.62, 'width_mev': 10., 'uses': 1, 'last_used': '2026-09-14'}]
        recs = [{'id': 'r1', 'label': '1', 'domain': 'mcd', 'kind': 'window',
                 'center_ev': 1.65, 'width_mev': 5., 'recommended': True}]
        v.render(_result(), candidates=merge_center_candidates(recs, history))
        for name in ('mcd_map', 'mcd_spectra'):
            labels = [a.get_text() for a in v._artists['feature_selection'] if a.axes is v.axes[name] and hasattr(a, 'get_text')]
            self.assertIn('H1', labels)
            self.assertIn('R1', labels)
        v.candidate_filter_combo.setCurrentText('History')
        self.assertEqual(len(v.visible_candidates), 1)
        self.assertTrue(v.visible_candidates[0]['history'])
        v.candidate_filter_combo.setCurrentText('Recommended')
        self.assertEqual(len(v.visible_candidates), 1)
        self.assertEqual(v.visible_candidates[0]['id'], 'r1')

    def test_load_restores_latest_saved_and_history_selection_restores_width(self):
        w = MainWindow()
        self.addCleanup(w.close)
        history = ({'center_ev': 1.62, 'width_mev': 10., 'last_used': '2026-09-10'},
                   {'center_ev': 1.65, 'width_mev': 5., 'last_used': '2026-09-14'})
        w.loaded = LoadedState(mode='MCD', folder='', mcd_result=_result(), mcd_center_history=history)
        w.mcd_controller._prepare_mcd_center_for_loaded_energy()
        self.assertEqual(w.mcd_window_center_spin.value(), 1.65)
        self.assertEqual(w.mcd_window_width_spin.value(), 5.)
        w._plot_mode('MCD')
        v = w.mcd_unified_view
        v.candidate_filter_combo.setCurrentText('History')
        index = next(i for i in range(v.candidate_combo.count()) if '1.6200' in v.candidate_combo.itemText(i))
        v.candidate_combo.setCurrentIndex(index)
        w._on_unified_candidate_selected(index)
        w.mcd_controller._apply_pending_mcd_center_refresh()
        self.assertEqual(w.mcd_window_center_spin.value(), 1.62)
        self.assertEqual(w.mcd_window_width_spin.value(), 10.)
        # Same-source Apply/reprocessing must preserve the active window.
        w.mcd_controller._prepare_mcd_center_for_loaded_energy()
        self.assertEqual(w.mcd_window_center_spin.value(), 1.62)
        self.assertEqual(w.mcd_window_width_spin.value(), 10.)

    def test_saved_history_refresh_does_not_record_or_move_unsaved_center(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'a.csv').touch()
            w = MainWindow()
            self.addCleanup(w.close)
            w.loaded = LoadedState(mode='MCD', folder=temp, primary_file='a.csv', mcd_result=_result())
            w.mcd_window_center_spin.setValue(1.64)
            w._plot_mode('MCD')
            from core.mcd_center_history import read_center_history
            self.assertEqual(read_center_history(root, 'a.csv'), [])
            out = root / 'Processed Data' / 'MCD'
            out.mkdir(parents=True)
            (out / 'result_MCD_settings_saved.json').write_text(json.dumps({
                'workflow': 'MCD', 'source_file': 'a.csv', 'created_utc': '2026-09-14',
                'mcd_b': {'center_ev': 1.65, 'width_mev': 5}}))
            w._refresh_saved_mcd_centers(())
            for _ in range(100):
                QTest.qWait(20)
                if w.loaded.mcd_center_history:
                    break
            self.assertEqual(len(w.loaded.mcd_center_history), 1)
            self.assertEqual(w.loaded.mcd_center_history[0]['center_ev'], 1.65)
            self.assertEqual(w.mcd_window_center_spin.value(), 1.64)
            self.assertTrue(any(item.get('history') for item in w.mcd_unified_view._candidates))
            rec = {'id': 'r7', 'label': '7', 'domain': 'mcd', 'kind': 'window',
                   'center_ev': 1.65, 'width_mev': 5., 'recommended': False,
                   'metadata': {'suggestion': True}}
            v = w.mcd_unified_view
            v.refresh_catalog(merge_center_candidates([rec], w.loaded.mcd_center_history), v.analysis_payload)
            w._publish_saved_mcd_centers(w._mcd_history_worker, w.loaded, w.loaded.mcd_center_history)
            self.assertTrue(any(item.get('display_id') == 'H1/R7' for item in v._candidates))
