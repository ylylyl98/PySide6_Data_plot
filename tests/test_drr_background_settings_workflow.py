"""Transactional integration of source intent, verified drafts and DRR controls."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QDialog

from core.drr_bn_matching import BnMatchCache, build_bn_match_plan, confirmation_is_current
from core.drr_sources import DrrMeasurementAssignment
from ui_qt.drr_background_settings import BackgroundDraft
from ui_qt import drr_background_settings_workflow as workflow


class _Control:
    def __init__(self, text='', *, checked=False, value=3):
        self.text, self.checked, self.number = text, checked, value
        self.blocked = False

    def blockSignals(self, blocked):
        old, self.blocked = self.blocked, blocked
        return old

    def currentText(self):
        return self.text

    def setCurrentText(self, text):
        self.text = text

    def isChecked(self):
        return self.checked

    def setChecked(self, checked):
        self.checked = checked

    def value(self):
        return self.number


class _Settings:
    def __init__(self):
        self.values = {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value


class _Owner(QObject):
    def __init__(self, root, measurement, background):
        super().__init__()
        self.current_folder = str(root)
        self.drr_selected_files = [measurement]
        self.drr_baseline_files_manual = [background]
        self.drr_baseline_files_found = []
        self.drr_baseline_combo = _Control('External')
        self.drr_baseline_combine_combo = _Control(workflow.FRAME_LABELS['all'])
        self.drr_pin_baseline_chk = _Control()
        self.drr_bn_auto_chk = _Control()
        self.drr_sg_window_spin, self.drr_sg_poly_spin = _Control(value=5), _Control(value=2)
        self._drr_background_regions, self._drr_assignments = (), ()
        self._drr_assignments_automatic = False
        self._drr_fitted_settings = {}
        self._drr_baseline_user_selected = True
        self._drr_external_auto_selected = False
        self._drr_pending_bn_plan = None
        self._drr_background_guess = None
        self._drr_external_request = None
        self._drr_settings_apply_request = None
        self._is_closing = False
        self.settings = _Settings()
        self.drr_controller = SimpleNamespace(
            _cancel_auto_external=Mock(), _update_drr_selection_labels=Mock(),
            _update_drr_baseline_controls=Mock(), _open_drr_source_dialog=Mock(),
            _start_auto_external=Mock(side_effect=AssertionError('Cross-type matching must not run.')),
        )
        self.thread_pool = SimpleNamespace(start=Mock())
        self.drr_edit_baselines_btn = SimpleNamespace(setFocus=Mock())
        self._selected_y_axis_spec = Mock(return_value='auto')
        self._status, self._show_error, self._start_load = Mock(), Mock(), Mock()
        self._invalidate_drr_for_background_selection = Mock()


class BackgroundSettingsWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_'
        self.measurement = prefix+'TG_rep01.csv'
        self.bn = prefix+'BNbackground_001.csv'
        for name, stamp, row in ((self.measurement, 1000, '1,1'), (self.bn, 900, '0,0')):
            path = self.root/name
            path.write_text('Vbg,Vtg,740,760,780\n0,0,10,11,12\n'+row+',11,12,13\n')
            os.utime(path, (stamp, stamp))
        self.owner = _Owner(self.root, self.measurement, self.bn)

    def plan(self):
        return build_bn_match_plan(self.root, (self.measurement,))

    def intent(self, draft):
        self.owner._drr_background_intent = draft
        self.owner._drr_background_intent_root = workflow._root_key(self.owner)
        self.owner._drr_background_intent_selection = tuple(self.owner.drr_selected_files)
        return draft

    def state(self):
        owner = self.owner
        return (
            owner.drr_baseline_combo.currentText(), tuple(owner.drr_baseline_files_manual),
            tuple(owner.drr_baseline_files_found), tuple(owner._drr_assignments),
            owner._drr_assignments_automatic, tuple(owner._drr_background_regions),
            owner.drr_pin_baseline_chk.isChecked(), owner.drr_bn_auto_chk.isChecked(),
            dict(owner.settings.values), workflow.source_intent(owner),
        )

    def validate(self, draft, selected=None):
        return workflow._validate(str(self.root), selected or (self.measurement,), draft,
                                  'auto', (5, 2), progress=lambda *_: None, log=lambda *_: None)

    def test_current_draft_infers_ordinary_bn_and_manual_p_only(self):
        self.assertEqual(workflow.current_draft(self.owner).kind, 'bn')
        self.owner.drr_baseline_files_manual = ['YZ383_p2n1_Pbackground.csv']
        self.assertEqual(workflow.current_draft(self.owner).kind, 'p')
        self.owner.drr_baseline_files_manual = ['YZ383_p2n1_Nbackground.csv']
        self.assertEqual(workflow.current_draft(self.owner).kind, 'n')

    def test_current_draft_retains_confirmed_mapping_and_pn_version_provenance(self):
        plan = self.plan()
        self.intent(BackgroundDraft(kind='bn', files=(self.bn,), bn_plan=plan,
                                    bn_candidates=(self.bn,), bn_warnings_acknowledged=True))
        current = workflow.current_draft(self.owner)
        self.assertIs(current.bn_plan, plan)
        self.assertTrue(current.bn_warnings_acknowledged)
        self.assertEqual(current.bn_candidates, (self.bn,))
        version = SimpleNamespace(version_id='pn-version-7', csv_path=self.root/'PN backgrounds'/'version.csv')
        self.intent(BackgroundDraft(kind='pn', files=('PN backgrounds/version.csv',), version=version, allow_stale=True))
        self.owner.drr_baseline_files_manual = ['PN backgrounds/version.csv']
        current = workflow.current_draft(self.owner)
        self.assertEqual(current.kind, 'pn')
        self.assertIs(current.version, version)
        self.assertTrue(current.allow_stale)
        self.owner.current_folder = str(self.root/'different-experiment')
        self.assertIsNone(workflow.source_intent(self.owner))

    def test_current_draft_describes_applied_mapping_and_keeps_pending_preview_separate(self):
        plan = self.plan()
        self.intent(BackgroundDraft(kind='bn', bn_plan=plan, bn_warnings_acknowledged=True))
        replacement = replace(plan, warnings=plan.warnings+('New evidence needs review.',))
        self.owner._drr_pending_bn_plan = replacement
        current = workflow.current_draft(self.owner)
        self.assertIs(current.bn_plan, plan)
        self.assertTrue(current.bn_warnings_acknowledged)
        dialog = Mock()
        dialog.exec.return_value = QDialog.Rejected
        dialog.selection = None
        with patch('ui_qt.drr_background_settings.DrrBackgroundSettings', return_value=dialog) as create, patch.object(
                workflow, 'apply_settings') as apply:
            workflow.edit_settings(self.owner)
        self.assertIs(create.call_args.kwargs['current'].bn_plan, plan)
        dialog.set_bn_plan.assert_called_once_with(replacement)
        apply.assert_not_called()
        self.assertIs(workflow.current_draft(self.owner).bn_plan, plan)

    def test_validate_accepts_only_current_acknowledged_bn_plan(self):
        plan = self.plan()
        draft = BackgroundDraft(kind='bn', bn_plan=plan)
        with self.assertRaisesRegex(ValueError, 'acknowledge'):
            self.validate(draft)
        acknowledged = replace(draft, bn_warnings_acknowledged=True)
        validated, prepared = self.validate(acknowledged)
        self.assertEqual(validated, acknowledged)
        self.assertIs(prepared.plan, plan)
        self.assertTrue(confirmation_is_current(prepared))
        with self.assertRaisesRegex(ValueError, 'changed|incomplete'):
            self.validate(acknowledged, selected=('other.csv',))
        os.utime(self.root/self.bn, (901, 901))
        with self.assertRaisesRegex(ValueError, 'changed|incomplete|stale'):
            self.validate(acknowledged)

    def test_validate_empty_explicit_source_never_invokes_generic_resolution(self):
        with patch('core.drr_sources.resolve_drr_background_assignments', side_effect=AssertionError('No generic fallback.')):
            for kind in ('bn', 'p', 'n', 'pn', 'custom'):
                with self.subTest(kind=kind):
                    with self.assertRaisesRegex(ValueError, 'Choose background files'):
                        self.validate(BackgroundDraft(kind=kind))

    def test_conflicting_bn_mapping_and_calibration_draft_fails_before_either_is_prepared(self):
        draft = BackgroundDraft(kind='bn', bn_plan=self.plan(), bn_warnings_acknowledged=True,
                                calibration={'payload': {'version_id': 'old-calibration'}, 'original': False})
        before = self.state()
        with patch('core.drr_bn_matching.BnMatchCache.prepare_confirm',
                   side_effect=AssertionError('Conflicting source choices require review first.')):
            with self.assertRaisesRegex(ValueError, 'mapping or the calibrated reference'):
                self.validate(draft)
        self.assertEqual(self.state(), before)

    def test_validated_bn_commit_keeps_every_selected_measurement_and_records_confirmation(self):
        plan = self.plan()
        draft = BackgroundDraft(kind='bn', bn_plan=plan, bn_warnings_acknowledged=True)
        before_selected = tuple(self.owner.drr_selected_files)
        with patch('ui_qt.drr_bn_workflow.forget_bn_choice'):
            workflow._commit(self.owner, *self.validate(draft))
        self.assertEqual(tuple(self.owner.drr_selected_files), before_selected)
        self.assertEqual(self.owner.drr_baseline_files_manual, [self.bn])
        self.assertEqual(self.owner._drr_assignments, plan.resolution.assignments)
        self.assertTrue(self.owner._drr_assignments_automatic)
        self.assertEqual(workflow.source_intent(self.owner).kind, 'bn')
        self.owner._start_load.assert_called_once_with('DRR', drr_resolution=plan.resolution)
        persisted = json.loads(self.owner.settings.value('drr/confirmed_bn_matches_v1'))
        restored = BnMatchCache.from_dict(persisted)
        self.assertEqual(restored.find(self.root, before_selected), plan)

    def test_stale_bn_commit_rejects_before_forgetting_previous_calibration(self):
        draft = BackgroundDraft(kind='bn', bn_plan=self.plan(), bn_warnings_acknowledged=True)
        validated, prepared = self.validate(draft)
        before = self.state()
        os.utime(self.root/self.bn, (902, 902))
        with patch('ui_qt.drr_bn_workflow.forget_bn_choice') as forget:
            with self.assertRaisesRegex(ValueError, 'stale|changed'):
                workflow._commit(self.owner, validated, prepared)
        forget.assert_not_called()
        self.assertEqual(self.state(), before)
        self.owner._start_load.assert_not_called()

    def test_unverified_bn_draft_cannot_commit_or_forget_previous_calibration(self):
        draft = BackgroundDraft(kind='bn', bn_plan=self.plan(), bn_warnings_acknowledged=True)
        before = self.state()
        with patch('ui_qt.drr_bn_workflow.forget_bn_choice') as forget:
            with self.assertRaisesRegex(ValueError, 'verification'):
                workflow._commit(self.owner, draft)
        forget.assert_not_called()
        self.assertEqual(self.state(), before)
        self.owner._start_load.assert_not_called()

    def test_pn_validation_prepares_without_persisting_association(self):
        version = SimpleNamespace(version_id='v1', csv_path=self.root/'PN backgrounds'/'v1.csv', fresh=True, status='valid')
        prepared = SimpleNamespace(version=version)
        store = Mock()
        store.get_version.return_value = version
        store.prepare_use.return_value = prepared
        draft = BackgroundDraft(kind='pn', files=('PN backgrounds/v1.csv',), version=version)
        with patch('core.drr_pn_store.DrrPnStore', return_value=store), patch('core.data_io._validate_drr_external_spectral_grids'):
            validated, token = self.validate(draft)
        store.mark_used.assert_not_called()
        store.commit_prepared_use.assert_not_called()
        self.assertIs(validated.version, version)
        self.assertIs(getattr(token, 'version', token), version)

    def test_pn_commit_keeps_version_identity_and_records_prepared_use(self):
        version = SimpleNamespace(version_id='v1', csv_path=self.root/'PN backgrounds'/'v1.csv', fresh=True, status='valid')
        prepared = SimpleNamespace(version=version)
        store = Mock()
        store.commit_prepared_use.return_value = version
        draft = BackgroundDraft(kind='pn', files=('PN backgrounds/v1.csv',), version=version)
        with patch('core.drr_pn_store.DrrPnStore', return_value=store), patch('ui_qt.drr_bn_workflow.forget_bn_choice'):
            workflow._commit(self.owner, draft, prepared)
        store.commit_prepared_use.assert_called_once_with(prepared)
        self.assertIs(workflow.source_intent(self.owner).version, version)
        self.assertEqual(workflow.current_draft(self.owner).version.version_id, 'v1')
        self.assertEqual(self.owner.drr_baseline_files_manual, ['PN backgrounds/v1.csv'])

    def test_cancelled_settings_dialog_keeps_applied_state_and_saved_preferences(self):
        self.intent(BackgroundDraft(kind='bn', files=(self.bn,), fixed=True))
        before = self.state()
        dialog = Mock()
        dialog.exec.return_value = QDialog.Rejected
        # Even a prepared child result remains private when the outer dialog cancels.
        dialog.selection = BackgroundDraft(kind='self', which='first', auto_bn_compare=True)
        with patch('ui_qt.drr_background_settings.DrrBackgroundSettings', return_value=dialog), patch.object(workflow, 'apply_settings') as apply:
            workflow.edit_settings(self.owner)
        apply.assert_not_called()
        self.assertEqual(self.state(), before)
        self.owner._start_load.assert_not_called()
        dialog.deleteLater.assert_called_once()
        self.assertIsNone(self.owner._drr_settings_dialog)

    def test_accepted_dialog_with_changed_selection_does_not_apply_draft(self):
        dialog = Mock()
        dialog.selection = BackgroundDraft(kind='self', which='first')
        def execute():
            self.owner.drr_selected_files = ['another.csv']
            return QDialog.Accepted
        dialog.exec.side_effect = execute
        with patch('ui_qt.drr_background_settings.DrrBackgroundSettings', return_value=dialog), patch.object(workflow, 'apply_settings') as apply:
            workflow.edit_settings(self.owner)
        apply.assert_not_called()
        self.assertIn('Selection changed', self.owner._status.call_args.args[0])

    def test_validation_result_is_ignored_after_context_change_or_cancel(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                draft = BackgroundDraft(kind='self', which='last')
                request = workflow._SettingsApplyRequest(self.owner, draft)
                self.owner._drr_settings_apply_request = request
                if cancel:
                    workflow.cancel_settings_apply(self.owner)
                else:
                    self.owner.drr_sg_poly_spin.number += 1
                with patch.object(workflow, '_commit') as commit:
                    request.complete((draft, None))
                commit.assert_not_called()
                request.deleteLater()

    def test_legacy_row_only_cache_is_not_reused_as_batch_confirmation(self):
        # A previous approval did not include acquisition-batch conflicts.
        # Discard that approval without deleting the saved data or recipe.
        previous = BnMatchCache()
        previous.confirm(self.plan(), warnings_acknowledged=True)
        legacy = previous.to_dict()
        legacy['schema_version'] = 1
        for plan in legacy['plans']:
            plan['schema_version'] = 1
        self.owner.settings.setValue('drr/confirmed_bn_matches_v1', json.dumps(legacy))
        before = self.state()
        cache = workflow._cache(self.owner)
        self.assertIsNone(cache.find(self.root, (self.measurement,)))
        self.assertEqual(self.state(), before)
        self.assertEqual(json.loads(self.owner.settings.value('drr/confirmed_bn_matches_v1')), legacy)
        self.owner._start_load.assert_not_called()

    def test_incomplete_batch_suggestions_remain_pending_without_application(self):
        plan = build_bn_match_plan(self.root, (self.measurement,), candidate_files=())
        self.assertFalse(plan.ready)
        draft = self.intent(BackgroundDraft(kind='bn'))
        request = workflow._FollowBnRequest(self.owner, draft)
        self.owner._drr_external_request = request
        before = self.state()
        request.complete((plan, False, None))
        self.assertIs(self.owner._drr_pending_bn_plan, plan)
        self.assertEqual(self.state(), before)
        self.assertIn('flagged measurements', self.owner._status.call_args.args[0])
        self.owner._start_load.assert_not_called()
        request.deleteLater()

    def test_follow_bn_reuses_confirmed_plan_without_running_matching(self):
        plan = self.plan()
        cache = BnMatchCache()
        cache.confirm(plan, warnings_acknowledged=True)
        with patch('core.drr_bn_matching.build_bn_match_plan', side_effect=AssertionError('Confirmed mapping must be reused.')):
            result, confirmed, prepared = workflow._follow_bn(str(self.root), (self.measurement,),
                BackgroundDraft(kind='bn'), cache, None, progress=lambda *_: None, log=lambda *_: None)
        self.assertIs(result, plan)
        self.assertTrue(confirmed)
        self.assertIs(prepared.plan, plan)
        self.assertTrue(confirmation_is_current(prepared))

    def test_follow_applied_confirmed_bn_needs_no_gui_cache_find_or_matching(self):
        plan = self.plan()
        cache = BnMatchCache()
        cache.confirm(plan, warnings_acknowledged=True)
        self.owner._drr_confirmed_bn_cache = cache
        self.owner._drr_bn_confirmation = cache.prepare_confirm(plan, warnings_acknowledged=True)
        self.owner._drr_assignments = plan.resolution.assignments
        self.intent(BackgroundDraft(kind='bn', bn_plan=plan, bn_warnings_acknowledged=True))
        with patch.object(workflow, '_commit') as commit, patch.object(cache, 'find',
                side_effect=AssertionError('Confirmed cache search must run in a worker.')):
            self.assertTrue(workflow.follow_source_type(self.owner))
        commit.assert_not_called()
        self.assertEqual(self.owner._drr_assignments, plan.resolution.assignments)
        self.owner.thread_pool.start.assert_not_called()
        self.owner.drr_controller._start_auto_external.assert_not_called()

    def test_follow_worker_reuses_verified_mapping_then_commits_on_gui_delivery(self):
        plan = self.plan()
        cache = BnMatchCache()
        cache.confirm(plan, warnings_acknowledged=True)
        self.owner._drr_confirmed_bn_cache = cache
        draft = self.intent(BackgroundDraft(kind='bn', bn_plan=plan, bn_warnings_acknowledged=True))
        request = workflow._FollowBnRequest(self.owner, draft)
        self.owner._drr_external_request = request
        with patch('core.drr_bn_matching.build_bn_match_plan', side_effect=AssertionError('No rematching.')):
            result = workflow._follow_bn(str(self.root), (self.measurement,), draft, cache, None,
                                         progress=lambda *_: None, log=lambda *_: None)
        with patch('ui_qt.drr_bn_workflow.forget_bn_choice'):
            request.complete(result)
        self.assertEqual(self.owner._drr_assignments, plan.resolution.assignments)
        self.assertIsNone(self.owner._drr_external_request)
        self.owner._start_load.assert_called_once_with('DRR', drr_resolution=plan.resolution)
        request.deleteLater()

    def test_follow_worker_stale_delivery_is_reported_without_mutating_applied_recipe(self):
        plan = self.plan()
        cache = BnMatchCache()
        cache.confirm(plan, warnings_acknowledged=True)
        self.owner._drr_confirmed_bn_cache = cache
        draft = self.intent(BackgroundDraft(kind='bn', bn_plan=plan, bn_warnings_acknowledged=True))
        request = workflow._FollowBnRequest(self.owner, draft)
        self.owner._drr_external_request = request
        result = workflow._follow_bn(str(self.root), (self.measurement,), draft, cache, None,
                                     progress=lambda *_: None, log=lambda *_: None)
        before = self.state()
        os.utime(self.root/self.bn, (903, 903))
        with patch('ui_qt.drr_bn_workflow.forget_bn_choice') as forget:
            request.complete(result)
        forget.assert_not_called()
        self.assertEqual(self.state(), before)
        self.owner._start_load.assert_not_called()
        self.assertIn('changed before reuse', self.owner._status.call_args.args[0])
        request.deleteLater()

    def test_changed_group_preserves_bn_intent_and_requires_mapping_review(self):
        self.intent(BackgroundDraft(kind='bn', files=(self.bn,)))
        self.owner._drr_assignments = (DrrMeasurementAssignment(
            self.measurement, 'External', (self.bn,), 'all'),)
        self.owner.drr_selected_files = ['new-group.csv']
        class Request:
            def __init__(self, owner, draft):
                self.selected = tuple(owner.drr_selected_files)
                self.worker = object()
        with patch.object(workflow, '_FollowBnRequest', Request):
            self.assertTrue(workflow.follow_source_type(self.owner, selection_changed=True))
        self.assertEqual(workflow.source_intent(self.owner).kind, 'bn')
        self.assertFalse(self.owner._drr_assignments)
        self.assertFalse(self.owner.drr_baseline_files_manual)
        self.owner.thread_pool.start.assert_called_once()
        self.owner._start_load.assert_not_called()
        self.owner.drr_controller._start_auto_external.assert_not_called()

    def test_follow_manual_p_type_and_fixed_background_do_not_start_bn_or_pn_matching(self):
        for fixed in (False, True):
            with self.subTest(fixed=fixed):
                self.intent(BackgroundDraft(kind='p', files=('Pbackground.csv',), fixed=fixed))
                self.owner.drr_pin_baseline_chk.setChecked(fixed)
                self.owner.drr_baseline_files_manual = ['Pbackground.csv']
                self.assertTrue(workflow.follow_source_type(self.owner, selection_changed=True))
                self.assertEqual(workflow.source_intent(self.owner).kind, 'p')
                # The caller's hint cannot discard a same-group explicit
                # choice. Changed groups first check their saved recipe.
                self.assertEqual(self.owner.drr_baseline_files_manual, ['Pbackground.csv'])
        self.owner.thread_pool.start.assert_not_called()
        self.owner._start_load.assert_not_called()
        self.owner.drr_controller._start_auto_external.assert_not_called()

    def test_controller_bn_entry_opens_review_settings_without_applying(self):
        from ui_qt.controllers_drr import DrrController
        controller = DrrController(self.owner)
        with patch.object(DrrController, '_edit_drr_background_settings') as edit, patch.object(
                DrrController, '_start_auto_external', side_effect=AssertionError('No immediate matching.')):
            controller._auto_match_drr_backgrounds()
        edit.assert_called_once_with(initial_kind='bn')
        self.owner._start_load.assert_not_called()

    def test_controller_cannot_send_explicit_bn_intent_to_generic_external_matcher(self):
        from ui_qt.controllers_drr import DrrController
        controller = DrrController(self.owner)
        self.intent(BackgroundDraft(kind='bn', files=(self.bn,)))
        class Request:
            def __init__(self, owner, draft):
                self.selected = tuple(owner.drr_selected_files)
                self.worker = object()
        with patch.object(workflow, '_FollowBnRequest', Request), patch(
                'ui_qt.drr_auto_external.ExternalMatchRequest', side_effect=AssertionError('No generic matching.')):
            controller._start_auto_external(measured_bn_only=True)
        self.assertIsInstance(self.owner._drr_external_request, Request)
        self.assertEqual(workflow.source_intent(self.owner).kind, 'bn')
        self.owner._start_load.assert_not_called()

    def test_unconfigured_typed_external_load_stops_before_generic_resolution(self):
        from ui_qt.main_window import MainWindow
        self.owner._drr_catalog_is_inventory_preview = Mock(return_value=False)
        self.owner._load_in_progress = False
        self.owner._invalidate_export_move_sources = Mock()
        self.owner._launch_load_options = Mock()
        self.owner.drr_baseline_files_manual = []
        self.owner._drr_assignments = ()
        for kind in ('bn', 'p', 'n', 'pn', 'custom'):
            with self.subTest(kind=kind):
                self.intent(BackgroundDraft(kind=kind))
                with patch('ui_qt.main_window.resolve_drr_background_assignments',
                           side_effect=AssertionError('A missing typed source needs review.')):
                    MainWindow._start_load(self.owner, 'DRR')
                self.assertIn('Background settings', self.owner._invalidate_drr_for_background_selection.call_args.args[0])
        self.owner._launch_load_options.assert_not_called()
        self.owner.thread_pool.start.assert_not_called()

    def test_fifteen_member_apply_survives_real_calibration_forget_and_global_settings_write(self):
        from core.drr_bn_store import DrrBnStore
        from tests.test_drr_auto_external import write_timed_bn_group
        selected, backgrounds = write_timed_bn_group(self.root)
        owner = self.owner
        owner.drr_selected_files = list(selected)
        owner.drr_baseline_files_manual = [backgrounds[0]]
        store = DrrBnStore(self.root)
        store.deactivate(selected, (backgrounds[0],), 'first')
        before_manifest = store.manifest_path.read_bytes()
        before_sources = {name: (self.root/name).read_bytes() for name in (*selected, *backgrounds)}
        settings_dir = tempfile.TemporaryDirectory()
        self.addCleanup(settings_dir.cleanup)
        settings_path = Path(settings_dir.name)/'global-settings.json'
        class PersistedSettings(_Settings):
            def setValue(self, key, value):
                super().setValue(key, value)
                settings_path.write_text(json.dumps(self.values), encoding='utf-8')
        owner.settings = PersistedSettings()
        plan = build_bn_match_plan(self.root, selected)
        draft = BackgroundDraft(kind='bn', bn_plan=plan, bn_warnings_acknowledged=True)
        validated, prepared = self.validate(draft, selected=selected)
        readiness_during_load = []
        owner._start_load.side_effect = lambda *_args, **_kwargs: readiness_during_load.append(
            workflow.background_ready_for_load(owner))
        workflow._commit(owner, validated, prepared)
        self.assertEqual(readiness_during_load, [True])
        self.assertTrue(confirmation_is_current(prepared))
        self.assertTrue(workflow.background_ready_for_load(owner))
        self.assertNotEqual(store.manifest_path.read_bytes(), before_manifest)
        self.assertTrue(settings_path.is_file())
        self.assertEqual(owner.drr_selected_files, selected)
        self.assertEqual(len(owner._drr_assignments), 15)
        self.assertEqual([item.baseline_files for item in owner._drr_assignments],
                         [(backgrounds[0],)]*9+[(backgrounds[1],)]*6)
        self.assertEqual({name: (self.root/name).read_bytes() for name in before_sources}, before_sources)

    def test_changed_group_load_guard_preserves_all_fifteen_and_cancel_keeps_review_state(self):
        from tests.test_drr_auto_external import write_timed_bn_group
        selected, _ = write_timed_bn_group(self.root)
        next_selected = []
        # Create the second group before confirmation, so file inventory
        # signatures stay current when only the UI selection changes later.
        for name in selected:
            new_name = name.replace('RotIn94deg', 'RotIn49deg')
            (self.root/new_name).write_bytes((self.root/name).read_bytes())
            next_selected.append(new_name)
        owner = self.owner
        owner.drr_selected_files = list(selected)
        plan = build_bn_match_plan(self.root, selected)
        draft = BackgroundDraft(kind='bn', bn_plan=plan, bn_warnings_acknowledged=True)
        with patch('ui_qt.drr_bn_workflow.forget_bn_choice'):
            workflow._commit(owner, *self.validate(draft, selected=selected))
        self.assertTrue(confirmation_is_current(owner._drr_bn_confirmation))
        owner._start_load.reset_mock()
        owner.drr_selected_files = list(next_selected)
        class Request:
            def __init__(self, request_owner, _draft):
                self.selected = tuple(request_owner.drr_selected_files)
                self.worker = object()
        with patch.object(workflow, '_FollowBnRequest', Request):
            self.assertFalse(workflow.background_ready_for_load(owner))
        self.assertEqual(owner.drr_selected_files, next_selected)
        self.assertFalse(owner._drr_assignments)
        self.assertFalse(owner.drr_baseline_files_manual)
        self.assertEqual(workflow.source_intent(owner).kind, 'bn')
        owner._start_load.assert_not_called()
        owner.thread_pool.start.assert_called_once()
        before = self.state()
        dialog = Mock()
        dialog.exec.return_value = QDialog.Rejected
        dialog.selection = BackgroundDraft(kind='self', which='first')
        with patch('ui_qt.drr_background_settings.DrrBackgroundSettings', return_value=dialog), patch.object(
                workflow, 'apply_settings') as apply:
            workflow.edit_settings(owner)
        apply.assert_not_called()
        self.assertEqual(self.state(), before)
        self.assertEqual(owner.drr_selected_files, next_selected)

    def test_load_guard_rejects_confirmation_for_another_plan_even_if_files_are_current(self):
        plan = self.plan()
        draft = BackgroundDraft(kind='bn', bn_plan=plan, bn_warnings_acknowledged=True)
        with patch('ui_qt.drr_bn_workflow.forget_bn_choice'):
            workflow._commit(self.owner, *self.validate(draft))
        self.assertTrue(workflow.background_ready_for_load(self.owner))
        self.owner._drr_bn_confirmation = BnMatchCache().prepare_confirm(
            build_bn_match_plan(self.root, (self.measurement,), candidate_files=(self.bn,)),
            warnings_acknowledged=True)
        self.assertTrue(confirmation_is_current(self.owner._drr_bn_confirmation))
        class Request:
            def __init__(self, owner, _draft):
                self.selected = tuple(owner.drr_selected_files)
                self.worker = object()
        with patch.object(workflow, '_FollowBnRequest', Request):
            self.assertFalse(workflow.background_ready_for_load(self.owner))

    def test_legacy_bn_calibration_and_original_restore_sync_actual_source_intent(self):
        from ui_qt.drr_bn_workflow import _select_recipe
        plan = self.plan()
        original_payload = {
            'version_id': 'reviewed-v1',
            'source_identity': {'baseline_files': [{'path': self.bn}], 'baseline_which': 'all'},
        }
        cases = (
            (True, ('.drr_bn_calibration/reviewed-v1.csv',), 'last', original_payload, False),
            (False, (self.bn,), 'all', {**original_payload, 'original_chosen': True}, True),
            (False, (self.bn,), 'first', original_payload, None),
        )
        for calibrated, files, which, payload, original in cases:
            with self.subTest(calibrated=calibrated, original=original):
                self.intent(BackgroundDraft(kind='bn', files=(self.bn,), bn_plan=plan,
                    bn_candidates=(self.bn,), bn_warnings_acknowledged=True))
                self.owner._drr_pending_bn_plan = plan
                self.owner._drr_bn_confirmation = object()
                self.owner.drr_bn_auto_chk.setChecked(True)
                self.owner.drr_pin_baseline_chk.setChecked(True)
                self.owner._start_load.reset_mock()
                with patch('ui_qt.drr_bn_workflow._cached_payload', return_value=payload):
                    _select_recipe(self.owner, files, which, calibrated=calibrated)
                current = workflow.current_draft(self.owner)
                self.assertEqual(current.kind, 'bn')
                self.assertEqual(current.files, files)
                self.assertEqual(current.which, which)
                self.assertEqual(current.bn_candidates, (self.bn,))
                self.assertTrue(current.auto_bn_compare)
                self.assertFalse(current.fixed)
                self.assertIsNone(current.bn_plan)
                self.assertIsNone(self.owner._drr_bn_confirmation)
                self.assertIsNone(self.owner._drr_pending_bn_plan)
                if original is None:
                    self.assertIsNone(current.calibration)
                else:
                    self.assertIs(current.calibration['payload'], payload)
                    self.assertIs(current.calibration['original'], original)
                self.assertTrue(workflow.background_ready_for_load(self.owner))
                self.owner._start_load.assert_called_once_with('DRR')
                self.assertEqual(self.owner.drr_selected_files, [self.measurement])


if __name__ == '__main__':
    unittest.main()
