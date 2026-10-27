"""P/N mappings stay transactional and retain each measurement's own recipe."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QDialog

from core.drr_pn_matching import build_pn_match_plan, confirmation_is_current
from core.drr_sources import DrrBackgroundResolution, DrrMeasurementAssignment
from tests.test_drr_background_settings_workflow import _Owner
from ui_qt.drr_background_settings import BackgroundDraft
from ui_qt import drr_background_settings_workflow as workflow
from ui_qt.drr_saved_background import adopt_saved_background, _kind


class PnSettingsWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_'
        self.selected = tuple(prefix + f'TG_rep{index:02d}.csv' for index in (1, 2))
        self.backgrounds = tuple(prefix + f'Pbackground_{index:03d}.csv' for index in (1, 2))
        for name, stamp in zip(self.selected, (1000, 2000)):
            self.write(name, stamp, varying=True)
        for name, stamp in zip(self.backgrounds, (900, 1900)):
            self.write(name, stamp)
        self.owner = _Owner(self.root, self.selected[0], self.backgrounds[0])
        self.owner.drr_selected_files = list(self.selected)

    def write(self, name, stamp, varying=False):
        path = self.root/name
        path.write_text('Vbg,Vtg,740,760,780\n'
            + ('0,0,10,11,12\n1,1,11,12,13\n' if varying else '-10,-10,5,6,7\n-10,-10,6,7,8\n'))
        os.utime(path, (stamp, stamp))

    def draft(self, which='all'):
        plan = build_pn_match_plan(self.root, self.selected, kind='p',
            candidate_files=self.backgrounds, which=which)
        self.assertTrue(plan.ready, str(plan))
        return BackgroundDraft(kind='p', which=which, pn_plan=plan,
            pn_candidates=self.backgrounds, pn_warnings_acknowledged=True)

    def validate(self, draft, selected=None):
        return workflow._validate(str(self.root), selected or self.selected, draft, 'auto', (5, 2),
                                  progress=lambda *_: None, log=lambda *_: None)

    def commit(self, draft):
        with patch('ui_qt.drr_bn_workflow.forget_bn_choice'):
            workflow._commit(self.owner, *self.validate(draft))

    def test_apply_keeps_both_measurements_and_their_separate_backgrounds(self):
        draft = self.draft('first')
        self.commit(draft)
        self.assertEqual(self.owner.drr_selected_files, list(self.selected))
        self.assertEqual([item.baseline_files for item in self.owner._drr_assignments],
                         [(self.backgrounds[0],), (self.backgrounds[1],)])
        self.assertTrue(self.owner._drr_assignments_automatic)
        self.assertEqual(self.owner.drr_baseline_files_manual, list(self.backgrounds))
        self.assertTrue(confirmation_is_current(self.owner._drr_pn_confirmation))
        self.assertTrue(workflow.background_ready_for_load(self.owner))
        self.assertEqual(workflow.current_draft(self.owner).pn_plan, draft.pn_plan)
        self.owner._start_load.assert_called_once_with('DRR', drr_resolution=draft.pn_plan.resolution)

    def test_applied_mapping_corrects_each_measurement_before_averaging(self):
        import numpy as np
        from core.data_io import load_drr_resolved_cube
        for name, stamp, intensity, gates in (
                (self.selected[0], 1000, 20, (0, 1)), (self.selected[1], 2000, 60, (0, 1)),
                (self.backgrounds[0], 900, 10, (-10, -10)), (self.backgrounds[1], 1900, 20, (-10, -10))):
            (self.root/name).write_text('Vbg,Vtg,740,760,780\n'
                + ''.join(f'{gate},{gate},{intensity},{intensity},{intensity}\n' for gate in gates))
            os.utime(self.root/name, (stamp, stamp))
        self.commit(self.draft())
        cube = load_drr_resolved_cube(str(self.root), self.selected, self.owner._drr_assignments,
                                      y_axis='BG', derivative=None)
        # Equal repeat weights combine ratios 1 and 2. Averaging both sources
        # into a common denominator would instead produce 1.6667.
        np.testing.assert_allclose(cube.Z, 1.5)

    def test_real_bn_deactivation_does_not_invalidate_new_pn_mapping(self):
        from core.drr_bn_store import DrrBnStore
        store = DrrBnStore(self.root)
        store.deactivate(self.selected, (self.backgrounds[0],), 'all')
        draft, prepared = self.validate(self.draft())
        readiness = []
        self.owner._start_load.side_effect = lambda *_args, **_kwargs: readiness.append(
            workflow.background_ready_for_load(self.owner))
        workflow._commit(self.owner, draft, prepared)
        self.assertEqual(readiness, [True])
        self.assertTrue(confirmation_is_current(self.owner._drr_pn_confirmation))

    def test_incomplete_unacknowledged_wrong_selection_and_changed_method_fail(self):
        draft = self.draft()
        with self.assertRaisesRegex(ValueError, 'acknowledge|warnings'):
            self.validate(replace(draft, pn_warnings_acknowledged=False))
        with self.assertRaisesRegex(ValueError, 'changed|incomplete'):
            self.validate(draft, selected=(self.selected[0],))
        with self.assertRaisesRegex(ValueError, 'changed|incomplete'):
            self.validate(replace(draft, which='first'))
        empty = build_pn_match_plan(self.root, self.selected, kind='p', candidate_files=())
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.validate(replace(draft, pn_plan=empty))

    def test_changed_input_between_validation_and_commit_preserves_previous_recipe(self):
        draft, prepared = self.validate(self.draft())
        before = (tuple(self.owner.drr_baseline_files_manual), self.owner._drr_assignments)
        os.utime(self.root/self.backgrounds[0], (905, 905))
        with patch('ui_qt.drr_bn_workflow.forget_bn_choice') as forget:
            with self.assertRaisesRegex(ValueError, 'changed'):
                workflow._commit(self.owner, draft, prepared)
        forget.assert_not_called()
        self.assertEqual((tuple(self.owner.drr_baseline_files_manual), self.owner._drr_assignments), before)
        self.owner._start_load.assert_not_called()

    def test_load_guard_invalidates_stale_mapping_and_prepares_review(self):
        self.commit(self.draft())
        self.owner._start_load.reset_mock()
        os.utime(self.root/self.backgrounds[0], (906, 906))
        self.assertFalse(workflow.background_ready_for_load(self.owner))
        self.assertEqual(self.owner.drr_selected_files, list(self.selected))
        self.assertFalse(self.owner._drr_assignments)
        self.assertFalse(self.owner.drr_baseline_files_manual)
        self.assertIsInstance(self.owner._drr_external_request, workflow._FollowPnRequest)
        self.owner._start_load.assert_not_called()

    def test_changed_selection_rematches_same_type_and_preserves_candidate_pool(self):
        self.commit(self.draft())
        next_measurement = self.selected[0].replace('rep01', 'rep03')
        self.write(next_measurement, 2100, varying=True)
        self.owner.drr_selected_files = [next_measurement]
        with patch('ui_qt.drr_saved_background.begin_saved_group_restore', return_value=False):
            workflow.follow_source_type(self.owner, selection_changed=True)
        request = self.owner._drr_external_request
        self.assertIsInstance(request, workflow._FollowPnRequest)
        self.assertEqual(request.draft.kind, 'p')
        result = workflow._follow_pn(str(self.root), (next_measurement,), request.draft,
            request.cancelled, progress=lambda *_: None, log=lambda *_: None)
        request.complete(result)
        self.assertIs(self.owner._drr_pending_pn_plan, result)
        self.assertFalse(self.owner._drr_assignments)
        self.assertEqual(result.candidate_files, self.backgrounds)
        self.assertIsNone(workflow.current_draft(self.owner).pn_plan)

    def test_pending_review_is_separate_from_applied_mapping_and_cancel_is_noop(self):
        self.commit(self.draft())
        applied = workflow.source_intent(self.owner)
        pending = replace(applied.pn_plan, warnings=(*applied.pn_plan.warnings, 'Review new evidence.'))
        self.owner._drr_pending_pn_plan = pending
        dialog = Mock()
        dialog.exec.return_value = QDialog.Rejected
        dialog.selection = None
        with patch('ui_qt.drr_background_settings.DrrBackgroundSettings', return_value=dialog), patch.object(
                workflow, 'apply_settings') as apply:
            workflow.edit_settings(self.owner)
        dialog.set_pn_plan.assert_called_once_with(pending)
        apply.assert_not_called()
        self.assertIs(workflow.source_intent(self.owner), applied)

    def test_pending_plan_from_another_folder_does_not_replace_current_draft(self):
        self.commit(self.draft())
        self.owner._drr_pending_pn_plan = replace(workflow.source_intent(self.owner).pn_plan,
                                                 root=str(self.root/'another-experiment'))
        dialog = Mock()
        dialog.exec.return_value = QDialog.Rejected
        dialog.selection = None
        with patch('ui_qt.drr_background_settings.DrrBackgroundSettings', return_value=dialog):
            workflow.edit_settings(self.owner)
        dialog.set_pn_plan.assert_not_called()

    def test_saved_heterogeneous_mapping_survives_reopen_and_apply_without_union_average(self):
        assignments = tuple(DrrMeasurementAssignment(measurement, 'External', (background,), method)
            for measurement, background, method in zip(self.selected, self.backgrounds, ('first', 'last')))
        resolved = DrrBackgroundResolution(assignments, 'heterogeneous')
        def restore(value):
            self.owner._drr_assignments = value.assignments
            self.owner._drr_assignments_automatic = True
            self.owner.drr_baseline_files_manual = list(self.backgrounds)
        self.owner.drr_controller._restore_saved_drr_recipe = Mock(side_effect=restore)
        adopt_saved_background(self.owner, resolved)
        draft = workflow.current_draft(self.owner)
        self.assertEqual(draft.saved_assignments, assignments)
        self.assertTrue(workflow.background_ready_for_load(self.owner))
        self.commit(draft)
        self.assertEqual(self.owner._drr_assignments, assignments)
        self.assertFalse(workflow.source_intent(self.owner).fixed)
        self.assertEqual(self.owner._start_load.call_args.kwargs['drr_resolution'].numerical_path, 'heterogeneous')

    def test_saved_assignment_sidecar_change_invalidates_prepared_recipe(self):
        assignments = tuple(DrrMeasurementAssignment(measurement, 'External', (background,), 'all')
            for measurement, background in zip(self.selected, self.backgrounds))
        draft, prepared = self.validate(BackgroundDraft(kind='p', saved_assignments=assignments))
        (self.root/self.backgrounds[0]).with_suffix('.metadata.json').write_text('{}')
        with patch('ui_qt.drr_bn_workflow.forget_bn_choice') as forget:
            with self.assertRaisesRegex(ValueError, 'changed'):
                workflow._commit(self.owner, draft, prepared)
        forget.assert_not_called()

    def test_saved_assignment_content_change_with_restored_size_and_mtime_is_stale(self):
        assignments = tuple(DrrMeasurementAssignment(measurement, 'External', (background,), 'all')
            for measurement, background in zip(self.selected, self.backgrounds))
        draft, prepared = self.validate(BackgroundDraft(kind='p', saved_assignments=assignments))
        path = self.root/self.backgrounds[0]
        before = path.stat()
        path.write_bytes(path.read_bytes().replace(b'5,6,7', b'6,6,7'))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(path.stat().st_size, before.st_size)
        with self.assertRaisesRegex(ValueError, 'changed'):
            workflow._commit(self.owner, draft, prepared)

    def test_arbitrarily_named_matched_composites_restore_pn_type(self):
        item = DrrMeasurementAssignment(self.selected[0], 'External', ('imported.csv',), 'all',
                                       selection_reason='P/N match (pn): Same ordered native spectral grid')
        self.assertEqual(_kind((item,)), 'pn')

    def test_ready_mapping_cannot_apply_without_its_selected_candidate_pool(self):
        draft = self.draft()
        for candidates in (None, (), self.backgrounds[:1], (*self.backgrounds, 'unselected.csv')):
            with self.subTest(candidates=candidates):
                with self.assertRaisesRegex(ValueError, 'Choose background files|selected background files changed'):
                    self.validate(replace(draft, pn_candidates=candidates))

    def test_commit_rejects_candidate_pool_changed_after_verification(self):
        draft, prepared = self.validate(self.draft())
        with patch('ui_qt.drr_bn_workflow.forget_bn_choice') as forget:
            with self.assertRaisesRegex(ValueError, 'selected background files'):
                workflow._commit(self.owner, replace(draft, pn_candidates=self.backgrounds[:1]), prepared)
        forget.assert_not_called()
        self.owner._start_load.assert_not_called()

    def test_follow_without_selected_pool_starts_no_background_worker(self):
        for candidates, changed in ((None, False), ((), False), (None, True), ((), True)):
            with self.subTest(candidates=candidates, changed=changed):
                draft = BackgroundDraft(kind='p', pn_candidates=candidates)
                self.owner._drr_background_intent = draft
                self.owner._drr_background_intent_root = workflow._root_key(self.owner)
                self.owner._drr_background_intent_selection = ('earlier-group.csv',) if changed else self.selected
                with patch('ui_qt.drr_saved_background.begin_saved_group_restore',
                           side_effect=AssertionError('No background worker before files are selected.')):
                    self.assertTrue(workflow.follow_source_type(self.owner, selection_changed=changed))
                self.owner.thread_pool.start.assert_not_called()
                self.assertIn('choose background files', self.owner._status.call_args.args[0])
                self.assertFalse(self.owner._drr_assignments)

    def test_follow_uses_only_common_chosen_files_and_never_discovers_candidates(self):
        draft = BackgroundDraft(kind='p', files=self.backgrounds[:1])
        with patch('core.drr_pn_matching.drr_source_paths',
                   side_effect=AssertionError('A selected pool must never scan the experiment.')):
            plan = workflow._follow_pn(str(self.root), self.selected, draft, None,
                progress=lambda *_: None, log=lambda *_: None)
        self.assertEqual(plan.candidate_files, self.backgrounds[:1])
        self.assertFalse(plan.automatic_candidates)
        self.assertEqual([item.baseline_files for item in plan.resolution.assignments],
                         [self.backgrounds[:1], self.backgrounds[:1]])

    def test_explicit_empty_pool_does_not_fall_back_to_previously_used_files(self):
        draft = BackgroundDraft(kind='p', files=self.backgrounds, pn_candidates=())
        with patch('core.drr_pn_matching.build_pn_match_plan') as build:
            with self.assertRaisesRegex(ValueError, 'Choose background files'):
                workflow._follow_pn(str(self.root), self.selected, draft, None,
                    progress=lambda *_: None, log=lambda *_: None)
        build.assert_not_called()

    def test_legacy_discovered_plan_keeps_only_assigned_files_and_requires_rematch(self):
        unused = self.backgrounds[0].replace('_001.csv', '_003.csv')
        self.write(unused, 5000)
        legacy = build_pn_match_plan(self.root, self.selected, kind='p')
        self.assertTrue(legacy.ready)
        self.assertIn(unused, legacy.candidate_files)
        draft = BackgroundDraft(kind='p', pn_plan=legacy, pn_warnings_acknowledged=True)
        self.owner._drr_background_intent = draft
        self.owner._drr_background_intent_root = workflow._root_key(self.owner)
        self.owner._drr_background_intent_selection = self.selected
        current = workflow.current_draft(self.owner)
        self.assertEqual(current.pn_candidates, self.backgrounds)
        with self.assertRaisesRegex(ValueError, 'Rematch the selected'):
            self.validate(current)
        newly_discovered = self.backgrounds[0].replace('_001.csv', '_004.csv')
        self.write(newly_discovered, 1000)
        with patch('core.drr_pn_matching.drr_source_paths',
                   side_effect=AssertionError('Old candidates must be frozen.')):
            plan = workflow._follow_pn(str(self.root), self.selected, draft, None,
                progress=lambda *_: None, log=lambda *_: None)
        self.assertEqual(plan.candidate_files, self.backgrounds)
        self.assertNotIn(unused, plan.candidate_files)
        self.assertNotIn(newly_discovered, plan.candidate_files)
        self.assertFalse(plan.automatic_candidates)

    def test_chosen_common_reference_replaces_previous_candidate_pool(self):
        draft = BackgroundDraft(kind='p', files=self.backgrounds[:1], pn_candidates=self.backgrounds)
        self.commit(draft)
        self.assertEqual(workflow.source_intent(self.owner).pn_candidates, self.backgrounds[:1])

    def test_saved_export_cannot_replace_or_shrink_an_explicit_pool_on_group_change(self):
        from core.drr_saved_background import resolve_saved_group_background
        target = self.selected[0].replace('rep01', 'rep03')
        outside = self.backgrounds[0].replace('_001.csv', '_003.csv')
        self.write(target, 2100, varying=True)
        self.write(outside, 2099)
        output = self.root/'Processed Data'/'DRR'
        output.mkdir(parents=True)
        for saved_background in (outside, self.backgrounds[0]):
            with self.subTest(saved_background=saved_background):
                self.owner.drr_selected_files = list(self.selected)
                self.commit(self.draft())
                applied = workflow.source_intent(self.owner)
                assignment = DrrMeasurementAssignment(target, 'External', (saved_background,), 'first')
                (output/'latest.metadata.json').write_text(json.dumps({
                    'operation': 'DR/R',
                    'sources': [
                        {'role': 'measurement', 'source_path': str(self.root/target)},
                        {'role': 'background', 'source_path': str(self.root/saved_background)}],
                    'processing': {'baseline_selection': 'External', 'baseline_which': 'first',
                                   'drr_background_assignments': [assignment.to_dict()]}}))
                saved = resolve_saved_group_background(self.root, (), (target,))
                self.assertTrue(saved.resolved)
                self.assertEqual(saved.assignments[0].baseline_files, (saved_background,))
                self.owner.drr_selected_files = [target]
                self.owner._start_load.reset_mock()
                with patch('ui_qt.drr_saved_background.begin_saved_group_restore') as restore:
                    self.assertTrue(workflow.follow_source_type(self.owner, selection_changed=True))
                restore.assert_not_called()
                request = self.owner._drr_external_request
                self.assertIsInstance(request, workflow._FollowPnRequest)
                self.assertIs(workflow.source_intent(self.owner), applied)
                self.assertEqual(workflow.source_intent(self.owner).pn_candidates, self.backgrounds)
                result = workflow._follow_pn(str(self.root), (target,), request.draft,
                    request.cancelled, progress=lambda *_: None, log=lambda *_: None)
                request.complete(result)
                self.assertEqual(result.candidate_files, self.backgrounds)
                self.assertEqual(result.rows[0].baseline_file, self.backgrounds[1])
                self.assertEqual(workflow.source_intent(self.owner).pn_candidates, self.backgrounds)
                self.assertFalse(self.owner._drr_assignments)
                self.owner._start_load.assert_not_called()

    def test_returning_to_applied_group_after_unapplied_group_prepares_missing_assignments(self):
        target = self.selected[0].replace('rep01', 'rep03')
        self.write(target, 2100, varying=True)
        self.commit(self.draft())
        applied = workflow.source_intent(self.owner)
        prepared = self.owner._drr_pn_confirmation
        self.owner._start_load.reset_mock()
        self.owner.drr_selected_files = [target]
        workflow.follow_source_type(self.owner, selection_changed=True)
        other_request = self.owner._drr_external_request
        other_plan = workflow._follow_pn(str(self.root), (target,), other_request.draft,
            other_request.cancelled, progress=lambda *_: None, log=lambda *_: None)
        other_request.complete(other_plan)
        self.assertFalse(self.owner._drr_assignments)
        self.assertIs(self.owner._drr_pending_pn_plan, other_plan)

        # Return without applying the other group's suggestion. The old
        # confirmation is fresh, but its numerical assignments were cleared.
        self.owner.drr_selected_files = list(self.selected)
        self.assertTrue(confirmation_is_current(prepared))
        self.assertFalse(workflow._pn_confirmation_ready(self.owner, applied))
        workflow.follow_source_type(self.owner, selection_changed=True)
        request = self.owner._drr_external_request
        self.assertIsInstance(request, workflow._FollowPnRequest)
        self.assertEqual(request.selected, self.selected)
        self.assertIsNone(self.owner._drr_pending_pn_plan)
        self.owner._start_load.assert_not_called()
        plan = workflow._follow_pn(str(self.root), self.selected, request.draft,
            request.cancelled, progress=lambda *_: None, log=lambda *_: None)
        request.complete(plan)
        self.assertEqual(plan.candidate_files, self.backgrounds)
        self.assertIs(self.owner._drr_pending_pn_plan, plan)
        self.commit(replace(applied, pn_plan=plan, pn_warnings_acknowledged=True))
        self.assertTrue(workflow.background_ready_for_load(self.owner))
        self.assertEqual(self.owner._drr_assignments, plan.resolution.assignments)

    def test_returning_to_saved_assignment_group_requires_actual_recipe_to_be_present(self):
        target = self.selected[0].replace('rep01', 'rep03')
        self.write(target, 2100, varying=True)
        assignments = tuple(DrrMeasurementAssignment(measurement, 'External', (background,), method)
            for measurement, background, method in zip(self.selected, self.backgrounds, ('first', 'last')))
        self.commit(BackgroundDraft(kind='p', saved_assignments=assignments))
        applied = workflow.source_intent(self.owner)
        prepared = self.owner._drr_saved_assignments_confirmation
        self.assertTrue(workflow._saved_confirmation_ready(self.owner, applied))
        self.owner._drr_assignments_automatic = False
        self.assertFalse(workflow._saved_confirmation_ready(self.owner, applied))
        self.owner._drr_assignments_automatic = True

        self.owner.drr_selected_files = [target]
        workflow.follow_source_type(self.owner, selection_changed=True)
        self.owner.drr_selected_files = list(self.selected)
        self.assertTrue(workflow._saved_assignments_current(prepared))
        self.assertFalse(self.owner._drr_assignments)
        self.assertFalse(workflow._saved_confirmation_ready(self.owner, applied))
        self.assertFalse(workflow.background_ready_for_load(self.owner))
        request = self.owner._drr_external_request
        self.assertIsInstance(request, workflow._FollowPnRequest)
        self.assertEqual(request.selected, self.selected)
        self.assertEqual(request.draft.pn_candidates, self.backgrounds)


if __name__ == '__main__':
    unittest.main()
