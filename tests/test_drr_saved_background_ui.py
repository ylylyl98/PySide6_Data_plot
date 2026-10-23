"""Exact exported backgrounds survive the explicit source-type workflow."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication

from core.drr_background_regions import DrrBackgroundRegion
from core.drr_sources import DrrBackgroundResolution, DrrMeasurementAssignment, discover_drr_sources
from ui_qt.drr_background_settings import BackgroundDraft
from ui_qt import drr_background_settings_workflow as workflow
from ui_qt.drr_saved_background import SavedBackgroundRequest
from ui_qt.main_window import MainWindow
from ui_qt.theme import install_theme


class SavedBackgroundUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode='light')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with patch.object(MainWindow, '_restore_last_folder', lambda _self: None), patch.object(
                MainWindow, '_schedule_automatic_update_check', lambda _self: None):
            self.window = MainWindow()
        self.window.current_folder = str(self.root)
        self.old = 'old_REF_760nmc.csv'
        self.target = 'target_REF_760nmc.csv'
        self.p = 'reference_Pbackground_REF_760nmc.csv'
        self.bn = 'reference_BNbackground_REF_760nmc.csv'
        self.pn = 'Processed Data/DRR/PN Backgrounds/saved.csv'
        for name in (self.old, self.target):
            self.write(name, 20, varying=True)
        for name in (self.p, self.bn, self.pn):
            self.write(name, 10)
        self.window.drr_available_sources = discover_drr_sources(self.root)
        self.window.drr_selected_files = [self.old]
        workflow._silent(self.window.drr_baseline_combo, 'External')
        workflow._silent(self.window.drr_baseline_combine_combo, workflow.FRAME_LABELS['all'])
        workflow._silent(self.window.drr_pin_baseline_chk, False)
        workflow._silent(self.window.drr_bn_auto_chk, False)
        self.window._suspend_drr_autoplot = False
        self.jobs = []

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def write(self, name, value, *, varying=False):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Vbg,Vtg,740,760,780\n'
            + f'0,0,{value},{value},{value}\n'
            + f'{1 if varying else 0},{1 if varying else 0},{value},{value},{value}\n')

    def intent(self, kind='p', *, fixed=False, plan=None):
        window = self.window
        draft = BackgroundDraft(kind=kind, files=(self.p,), which='all', fixed=fixed, bn_plan=plan)
        window._drr_background_intent = draft
        window._drr_background_intent_root = workflow._root_key(window)
        window._drr_background_intent_selection = tuple(window.drr_selected_files)
        window.drr_baseline_files_manual = [self.p]
        window._drr_bn_confirmation = object() if plan is not None else None
        workflow._silent(window.drr_pin_baseline_chk, fixed)
        return draft

    def save(self, assignments):
        history = self.root/'Processed Data/DRR'
        history.mkdir(parents=True, exist_ok=True)
        metadata = history/'latest.metadata.json'
        backgrounds = tuple(dict.fromkeys(name for item in assignments for name in item.baseline_source_files))
        metadata.write_text(json.dumps({'operation': 'DR/R', 'sources': [
            *({'role': 'measurement', 'source_path': str(self.root/item.measurement_file)} for item in assignments),
            *({'role': 'background', 'source_path': str(self.root/name)} for name in backgrounds)],
            'processing': {'baseline_selection': 'External', 'baseline_which': 'all',
                           'drr_background_assignments': [item.to_dict() for item in assignments]}}))
        return metadata

    def change_group(self, selected=None):
        with patch.object(type(self.window.drr_controller), '_open_drr_source_dialog',
                          return_value=list(selected or [self.target])), patch.object(
                self.window.thread_pool, 'start', self.jobs.append):
            self.window.drr_controller._edit_drr_measurements()
        return self.window._drr_external_request

    def deliver(self):
        with patch.object(self.window.thread_pool, 'start', self.jobs.append):
            self.jobs[0].run()
            self.app.processEvents()

    def test_changed_typed_sources_restore_export_and_real_load_accepts_new_intent(self):
        for previous_kind, background, expected_kind in (
                ('p', self.bn, 'bn'), ('pn', self.p, 'p'), ('bn', self.pn, 'pn')):
            with self.subTest(previous_kind=previous_kind):
                window = self.window
                window.drr_selected_files = [self.old]
                workflow._silent(window.drr_baseline_combo, 'External')
                self.intent(previous_kind, plan=SimpleNamespace() if previous_kind == 'bn' else None)
                expected = DrrMeasurementAssignment(self.target, 'External', (background,), 'first')
                self.save((expected,))
                self.jobs.clear()
                request = self.change_group()
                self.assertIsInstance(request, SavedBackgroundRequest)
                self.assertEqual(len(self.jobs), 1)
                self.deliver()
                self.assertEqual(len(self.jobs), 2, 'The restored recipe must pass the real load guards.')
                options = self.jobs[1].args[0]
                self.assertEqual(options.drr_assignments[0].baseline_files, expected.baseline_files)
                self.assertEqual(options.drr_assignments[0].baseline_which, 'first')
                self.assertFalse(options.drr_resolve_background)
                self.assertEqual(workflow.source_intent(window).kind, expected_kind)
                self.assertIsNone(workflow.source_intent(window).bn_plan)
                self.assertIsNone(window._drr_bn_confirmation)
                self.assertEqual(window._drr_background_intent_selection, (self.target,))
                self.assertTrue(window.drr_baseline_combine_combo.isEnabled())
                self.assertTrue(window.drr_pin_baseline_chk.isEnabled())
                loaded = window._load_task(options, progress=SimpleNamespace(emit=lambda *_: None),
                                            log=SimpleNamespace(emit=lambda *_: None))
                np.testing.assert_allclose(loaded.cube.Z, 1.)
                window._on_load_finished(request_token=window._active_load_token)

    def test_heterogeneous_recipe_preserves_member_files_methods_and_regions(self):
        second = 'second_REF_760nmc.csv'
        self.write(second, 30, varying=True)
        self.intent('pn')
        region = DrrBackgroundRegion(1.61, 1.65, (self.bn,), 'first')
        expected = (DrrMeasurementAssignment(self.target, 'External', (self.p,), 'all', regions=(region,)),
                    DrrMeasurementAssignment(second, 'External', (self.bn,), 'last'))
        self.save(expected)
        self.change_group((self.target, second))
        self.deliver()
        self.assertEqual(len(self.jobs), 2)
        options = self.jobs[1].args[0]
        self.assertEqual([(a.measurement_file, a.baseline_files, a.baseline_which, a.regions)
                          for a in options.drr_assignments],
                         [(a.measurement_file, a.baseline_files, a.baseline_which, a.regions) for a in expected])
        self.assertEqual(workflow.source_intent(self.window).kind, 'custom')

    def test_saved_self_reference_is_loaded_after_typed_external(self):
        self.intent('bn')
        self.save((DrrMeasurementAssignment(self.target, 'Self (first frame)', baseline_which='first'),))
        self.change_group()
        self.deliver()
        self.assertEqual(len(self.jobs), 2)
        self.assertEqual(self.window.drr_baseline_combo.currentText(), 'Self (first frame)')
        self.assertEqual(workflow.source_intent(self.window).kind, 'self')
        self.assertEqual(self.jobs[1].args[0].drr_assignments[0].baseline_mode, 'Self (first frame)')

    def test_fixed_background_wins_over_saved_export(self):
        self.intent('p', fixed=True)
        self.save((DrrMeasurementAssignment(self.target, 'External', (self.bn,), 'first'),))
        with patch('ui_qt.drr_saved_background._resolve_saved', side_effect=AssertionError('Fixed wins.')):
            self.assertIsNone(self.change_group())
        self.assertEqual(len(self.jobs), 1)
        self.assertEqual(self.jobs[0].args[0].baseline_files, [self.p])
        self.assertEqual(workflow.source_intent(self.window).kind, 'p')

    def test_no_saved_recipe_keeps_p_type_and_requires_review_without_guessing(self):
        draft = self.intent('p')
        self.change_group()
        with patch('ui_qt.drr_auto_external.ExternalMatchRequest', side_effect=AssertionError('No cross-type guesses.')):
            self.deliver()
        self.assertEqual(len(self.jobs), 1)
        self.assertIs(workflow.source_intent(self.window), draft)
        self.assertFalse(self.window._drr_assignments)
        self.assertFalse(self.window.drr_baseline_files_manual)
        with patch.object(self.window.thread_pool, 'start', self.jobs.append):
            self.window.drr_controller._refresh_auto_external(selection_changed=True)
        self.assertEqual(len(self.jobs), 1, 'Refreshing an unchanged unresolved group must not query history again.')

    def test_no_saved_recipe_follows_bn_only_and_leaves_new_mapping_for_review(self):
        self.intent('bn')
        self.change_group()
        with patch('ui_qt.drr_auto_external.ExternalMatchRequest', side_effect=AssertionError('No P/N guessing.')):
            self.deliver()
        self.assertEqual(len(self.jobs), 2)
        self.assertIsInstance(self.window._drr_external_request, workflow._FollowBnRequest)
        self.assertEqual(workflow.source_intent(self.window).kind, 'bn')
        with patch.object(self.window, '_start_load') as start:
            self.jobs[1].run()
            self.app.processEvents()
        start.assert_not_called()
        self.assertIsNotNone(self.window._drr_pending_bn_plan)

    def test_explicit_bn_only_entry_does_not_restore_a_saved_p_recipe(self):
        self.intent('bn')
        self.save((DrrMeasurementAssignment(self.target, 'External', (self.p,), 'first'),))
        self.window.drr_selected_files = [self.target]
        with patch('ui_qt.drr_saved_background._resolve_saved', side_effect=AssertionError('BN only.')), \
                patch.object(self.window.thread_pool, 'start', self.jobs.append):
            self.window.drr_controller._start_auto_external(measured_bn_only=True)
        self.assertEqual(len(self.jobs), 1)
        self.assertIsInstance(self.window._drr_external_request, workflow._FollowBnRequest)
        self.assertEqual(workflow.source_intent(self.window).kind, 'bn')

    def test_missing_saved_background_blocks_without_recommendation_fallback(self):
        self.intent('bn')
        self.save((DrrMeasurementAssignment(self.target, 'External', ('missing.csv',), 'last'),))
        self.change_group()
        with patch.object(self.window, '_start_load') as start, patch.object(
                workflow, '_FollowBnRequest', side_effect=AssertionError('Invalid saved recipes must block.')):
            self.deliver()
        start.assert_not_called()
        self.assertFalse(self.window._drr_assignments)
        self.assertEqual(len(self.jobs), 1)
        with patch.object(workflow, '_FollowBnRequest', side_effect=AssertionError('Invalid history still blocks.')), \
                patch.object(self.window.thread_pool, 'start', self.jobs.append):
            self.window.drr_controller._refresh_auto_external(selection_changed=True)
        self.assertEqual(len(self.jobs), 1)

    def test_worker_errors_block_without_same_type_fallback(self):
        self.intent('p')
        request = self.change_group()
        with patch.object(self.window, '_start_load') as start:
            request.failed('Cannot read saved metadata.\n\ntraceback')
        start.assert_not_called()
        self.assertIsNone(self.window._drr_external_request)
        self.assertFalse(self.window._drr_assignments)

    def test_late_delivery_cannot_override_changed_folder_selection_type_or_options(self):
        for change in ('folder', 'selection', 'type', 'frame', 'axis', 'sg', 'fixed', 'cancelled'):
            with self.subTest(change=change):
                window = self.window
                window.current_folder = str(self.root)
                window.drr_selected_files = [self.old]
                workflow._silent(window.drr_baseline_combo, 'External')
                workflow._silent(window.drr_pin_baseline_chk, False)
                self.intent('p')
                self.jobs.clear()
                request = self.change_group()
                resolved = DrrBackgroundResolution(assignments=(DrrMeasurementAssignment(
                    self.target, 'External', (self.bn,), 'first'),), numerical_path='common')
                if change == 'folder':
                    window.current_folder = str(self.root/'other')
                elif change == 'selection':
                    window.drr_selected_files = [self.old]
                elif change == 'type':
                    window._drr_background_intent = replace(workflow.source_intent(window), kind='n')
                elif change == 'frame':
                    workflow._silent(window.drr_baseline_combine_combo, workflow.FRAME_LABELS['last'])
                elif change == 'axis':
                    with patch.object(window, '_selected_y_axis_spec', return_value='new-axis'):
                        self.assertFalse(request.current())
                    request.cancelled.set()
                elif change == 'sg':
                    window.drr_sg_window_spin.blockSignals(True)
                    window.drr_sg_window_spin.setValue(window.drr_sg_window_spin.value()+2)
                    window.drr_sg_window_spin.blockSignals(False)
                elif change == 'fixed':
                    workflow._silent(window.drr_pin_baseline_chk, True)
                else:
                    window.drr_controller._cancel_auto_external()
                with patch.object(window, '_start_load') as start:
                    request.complete(resolved)
                start.assert_not_called()
                self.assertFalse(window._drr_assignments)
                window.drr_controller._cancel_auto_external()

    def test_same_group_explicit_change_does_not_query_older_export(self):
        self.window.drr_selected_files = [self.target]
        self.intent('p')
        self.save((DrrMeasurementAssignment(self.target, 'External', (self.bn,), 'first'),))
        with patch.object(self.window.thread_pool, 'start', self.jobs.append):
            self.assertTrue(workflow.follow_source_type(self.window, selection_changed=True))
        self.assertEqual(self.jobs, [])
        self.assertEqual(self.window.drr_baseline_files_manual, [self.p])


if __name__ == '__main__':
    unittest.main()
