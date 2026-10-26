"""P/N assignment review is transactional and retains every measurement."""
import os
import threading
import tempfile
from pathlib import Path
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from core.drr_sources import DrrMeasurementAssignment
from tests.ui_test_helpers import dispose_owned_window
from ui_qt.drr_background_settings import BackgroundDraft, DrrBackgroundSettings
from ui_qt.theme import install_theme


def plan(*, ready=True, warnings=(), kind='p', candidates=('Pback_one.csv', 'Pback_two.csv')):
    return SimpleNamespace(kind=kind, which='all', automatic_candidates=False,
        candidate_files=candidates, warnings=warnings, ready=ready,
        rows=tuple(SimpleNamespace(measurement_file=f'measurement_{i}.csv',
            baseline_file=candidates[min(i, len(candidates) - 1)] if ready or i == 0 else '',
            candidate_files=candidates if i == 0 else (candidates[1:] or candidates),
            evidence='Matching acquisition conditions; nearest capture', warnings=warnings)
            for i in range(2)))


class PnMatchingUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode='light')

    def setUp(self):
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)
        self.windows = []

    def dialog(self, current=None, **kwargs):
        dialog = DrrBackgroundSettings(current=current or BackgroundDraft(kind='p'),
            folder='.', measurements=('measurement_0.csv', 'measurement_1.csv'),
            pool=self.pool, **kwargs)
        self.windows.append(dialog)
        return dialog

    def tearDown(self):
        for dialog in self.windows:
            dialog.reject()
        self.pool.waitForDone(10000)
        self.app.processEvents()
        for dialog in self.windows:
            dispose_owned_window(dialog)

    def wait(self, predicate):
        for _ in range(200):
            if predicate():
                return
            QTest.qWait(10)
        self.fail('P/N matching worker did not finish')

    def test_mapping_review_and_apply_keep_common_draft_unchanged(self):
        original = BackgroundDraft(kind='p', files=('old.csv',))
        dialog = self.dialog(original)
        matching = plan(warnings=('Acquisition time is unavailable; modification time proxy.',))
        dialog.set_pn_plan(matching)
        self.assertEqual(dialog.current, original)
        self.assertEqual(dialog.draft.files, ())
        self.assertEqual(dialog.pn_matching.model.rowCount(), 2)
        self.assertFalse(dialog.apply_btn.isEnabled())
        self.assertIn('review', dialog.status.text().lower())
        dialog.pn_matching.review.setChecked(True)
        self.assertTrue(dialog.apply_btn.isEnabled())
        dialog.apply_btn.click()
        self.assertEqual(dialog.result(), QDialog.Accepted)
        self.assertIs(dialog.selection.pn_plan, matching)

    def test_unresolved_rows_block_apply_and_bulk_choices_use_intersection(self):
        dialog = self.dialog()
        dialog.set_pn_plan(plan(ready=False))
        self.assertFalse(dialog.apply_btn.isEnabled())
        self.assertIn('Unresolved', dialog.pn_matching.model.item(1, 1).text())
        dialog.pn_matching.table.selectAll()
        self.assertEqual(dialog.pn_matching.choice.count(), 2)
        self.assertIsNone(dialog.pn_matching.choice.currentData())
        self.assertEqual(dialog.pn_matching.choice.itemData(1), 'Pback_two.csv')
        self.assertFalse(dialog.pn_matching.assign_btn.isEnabled())

    def test_frame_change_invalidates_matching_and_method_switch_restores_common(self):
        original = BackgroundDraft(kind='p', files=('common.csv',))
        dialog = self.dialog(original)
        dialog.pn_assignment_method.setCurrentIndex(dialog.pn_assignment_method.findData('matched'))
        dialog.set_pn_plan(plan())
        dialog.pn_frames.setCurrentIndex(dialog.pn_frames.findData('first'))
        self.assertIsNone(dialog.draft.pn_plan)
        self.assertFalse(dialog.apply_btn.isEnabled())
        dialog.pn_assignment_method.setCurrentIndex(dialog.pn_assignment_method.findData('common'))
        self.assertEqual(dialog.draft, original)

    def test_selecting_candidate_files_runs_match_on_worker_without_applying(self):
        calls = []
        def build(folder, selected, kind, candidates, which, stopped, **kwargs):
            calls.append((threading.current_thread(), kind, candidates))
            return plan()
        selected = []
        def choose(_dialog, kind, files, title):
            selected.append(kind)
            return ('Pback_one.csv', 'Pback_two.csv')
        dialog = self.dialog(select_files=choose)
        dialog.pn_assignment_method.setCurrentIndex(dialog.pn_assignment_method.findData('matched'))
        with patch('ui_qt.drr_pn_matching_panel._build_plan', side_effect=build):
            dialog.pn_matching.choose_btn.click()
            self.assertFalse(dialog.apply_btn.isEnabled())
            self.wait(lambda: not dialog._busy)
        self.assertEqual(selected, ['p_candidates'])
        self.assertNotEqual(calls[0][0], threading.current_thread())
        self.assertEqual(calls[0][1:], ('p', ('Pback_one.csv', 'Pback_two.csv')))
        self.assertTrue(dialog.apply_btn.isEnabled())
        self.assertIsNone(dialog.selection)

    def test_cancelled_picker_preserves_mapping(self):
        dialog = self.dialog(select_files=lambda *_: None)
        dialog.set_pn_plan(plan())
        before = dialog.draft
        dialog.pn_matching.choose_btn.click()
        self.assertEqual(dialog.draft, before)

    def test_assignment_choices_recognize_absolute_and_relative_pool_paths(self):
        selected = tuple(str(Path(name).resolve()) for name in plan().candidate_files)
        dialog = self.dialog(BackgroundDraft(kind='p', pn_candidates=selected))
        dialog.set_pn_plan(plan())
        dialog.pn_matching.table.selectRow(1)
        self.assertEqual(dialog.draft.pn_candidates, selected)
        self.assertEqual(dialog.pn_matching.choice.currentData(), 'Pback_two.csv')
        self.assertTrue(dialog.pn_matching.assign_btn.isEnabled())

    def test_saved_assignment_switch_never_treats_union_as_common_background(self):
        assignments = tuple(DrrMeasurementAssignment(f'measurement_{i}.csv', 'External',
            (f'Pback_{i}.csv',), 'all') for i in range(2))
        dialog = self.dialog(BackgroundDraft(kind='p', files=('Pback_0.csv', 'Pback_1.csv'),
                                             saved_assignments=assignments))
        self.assertEqual(dialog.pn_assignment_method.currentData(), 'saved')
        self.assertEqual(dialog.pn_files_model.rowCount(), 2)
        self.assertTrue(dialog.apply_btn.isEnabled())
        dialog.pn_assignment_method.setCurrentIndex(dialog.pn_assignment_method.findData('common'))
        self.assertEqual(dialog.draft.saved_assignments, ())
        self.assertEqual(dialog.draft.files, ())
        self.assertFalse(dialog.apply_btn.isEnabled())

    def test_failed_rematch_disables_prior_result(self):
        dialog = self.dialog()
        dialog.set_pn_plan(plan())
        with patch('ui_qt.drr_pn_matching_panel._build_plan', side_effect=ValueError('Cannot inspect candidate')):
            dialog.pn_matching.match_btn.click()
            self.wait(lambda: not dialog._busy)
        self.assertFalse(dialog.apply_btn.isEnabled())
        self.assertIn('Cannot inspect candidate', dialog.status.text())

    def test_bulk_manual_assignment_routes_all_selected_rows(self):
        from core.drr_pn_matching import build_pn_match_plan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, stamp, gate in (('measurement_0.csv', 1000, -1), ('measurement_1.csv', 2000, -1),
                                      ('Pbackground_a.csv', 900, -10), ('Pbackground_b.csv', 1900, -10)):
                path = root / name
                path.write_text(f'Vbg,Vtg,740,760,780\n{gate},{gate},10,11,12\n'
                                f'{gate},{gate},11,12,13\n')
                os.utime(path, (stamp, stamp))
            mapping = build_pn_match_plan(root, ('measurement_0.csv', 'measurement_1.csv'), kind='p',
                                        candidate_files=('Pbackground_a.csv', 'Pbackground_b.csv'))
            self.assertTrue(mapping.ready)
            dialog = self.dialog()
            dialog.set_pn_plan(mapping)
            dialog.pn_matching.table.selectRow(1)
            self.assertEqual(dialog.pn_matching.choice.currentData(), 'Pbackground_b.csv')
            dialog.pn_matching.table.selectAll()
            self.assertIsNone(dialog.pn_matching.choice.currentData())
            self.assertFalse(dialog.pn_matching.assign_btn.isEnabled())
            dialog.pn_matching.choice.setCurrentIndex(dialog.pn_matching.choice.findData('Pbackground_b.csv'))
            dialog.pn_matching.assign_btn.click()
            self.assertEqual([row.baseline_file for row in dialog.draft.pn_plan.rows],
                             ['Pbackground_b.csv', 'Pbackground_b.csv'])
            self.assertEqual(len(dialog.pn_matching.table.selectionModel().selectedRows()), 2)
            self.assertEqual(dialog.pn_matching.table.currentIndex().row(), 1)
            self.assertIsNone(dialog.selection)
            self.assertTrue(all('Manual assignment' in row.evidence for row in dialog.draft.pn_plan.rows))

    def test_close_cancels_worker_and_ignores_late_result(self):
        started, release = threading.Event(), threading.Event()
        def build(*args, **kwargs):
            started.set()
            release.wait(2)
            return plan()
        dialog = self.dialog()
        dialog.pn_assignment_method.setCurrentIndex(dialog.pn_assignment_method.findData('matched'))
        dialog._pn_candidates_changed(plan().candidate_files)
        before = dialog.draft
        try:
            with patch('ui_qt.drr_pn_matching_panel._build_plan', side_effect=build):
                dialog.pn_matching.match_btn.click()
                self.wait(started.is_set)
                dialog.reject()
                self.assertTrue(dialog.pn_matching._worker.cancellation.cancelled)
                release.set()
                self.wait(lambda: dialog.pn_matching._worker is None)
        finally:
            release.set()
        self.assertEqual(dialog.draft, before)
        self.assertIsNone(dialog.selection)

    def test_pending_candidate_draft_retains_its_mode_when_changing_source(self):
        dialog = self.dialog()
        dialog.pn_assignment_method.setCurrentIndex(dialog.pn_assignment_method.findData('matched'))
        dialog._pn_candidates_changed(('Pbackground_a.csv',))
        dialog.pn_method.setCurrentIndex(dialog.pn_method.findData('n'))
        dialog.pn_method.setCurrentIndex(dialog.pn_method.findData('p'))
        self.assertEqual(dialog.pn_assignment_method.currentData(), 'matched')
        self.assertEqual(dialog.draft.pn_candidates, ('Pbackground_a.csv',))
        self.assertFalse(dialog.apply_btn.isEnabled())

    def test_empty_pool_never_starts_matching(self):
        dialog = self.dialog()
        self.assertEqual(dialog.pn_assignment_method.currentData(), 'matched')
        self.assertFalse(dialog.pn_matching.match_btn.isEnabled())
        self.assertFalse(dialog.apply_btn.isEnabled())
        with patch('ui_qt.drr_pn_matching_panel._build_plan') as build:
            dialog.pn_matching.match()
            self.app.processEvents()
            build.assert_not_called()
        from ui_qt.drr_pn_matching_panel import _build_plan
        with patch('core.drr_pn_matching.build_pn_match_plan') as build:
            with self.assertRaisesRegex(ValueError, 'Choose background files'):
                _build_plan('.', ('measurement_0.csv',), 'p', None, 'all',
                            threading.Event(), progress=None, log=None)
            build.assert_not_called()

    def test_add_remove_and_clear_restrict_matching_and_retain_measurement_selection(self):
        dialog = self.dialog(select_files=lambda *_: ('./Pback_one.csv', 'Pback_three.csv'))
        dialog.set_pn_plan(plan())
        panel = dialog.pn_matching
        panel.table.selectRow(1)
        panel.table.selectAll()
        calls = []
        def build(folder, selected, kind, candidates, which, stopped, **kwargs):
            calls.append(candidates)
            return plan(candidates=candidates)
        with patch('ui_qt.drr_pn_matching_panel._build_plan', side_effect=build):
            panel.choose_btn.click()
            self.wait(lambda: not dialog._busy)
            self.assertEqual(calls[-1], ('Pback_one.csv', 'Pback_two.csv', 'Pback_three.csv'))
            self.assertEqual(panel.pool_model.rowCount(), 3)
            self.assertEqual(len(panel.table.selectionModel().selectedRows()), 2)
            self.assertEqual(panel.table.currentIndex().row(), 1)
            panel.pool_view.setCurrentIndex(panel.pool_model.index(0, 0))
            panel.remove_btn.click()
            self.wait(lambda: not dialog._busy)
            self.assertEqual(calls[-1], ('Pback_two.csv', 'Pback_three.csv'))
            self.assertEqual(panel.pool_model.rowCount(), 2)
            self.assertEqual(len(panel.table.selectionModel().selectedRows()), 2)
            self.assertEqual(panel.table.currentIndex().row(), 1)
            panel.clear_btn.click()
            self.assertEqual(len(calls), 2)
        self.assertEqual(dialog.draft.pn_candidates, ())
        self.assertIsNone(dialog.draft.pn_plan)
        self.assertFalse(panel.match_btn.isEnabled())
        self.assertFalse(dialog.apply_btn.isEnabled())


if __name__ == '__main__':
    unittest.main()
