"""A group BN change publishes only a complete, current DRR result."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QWidget

from core.data_io import load_drr_resolved_cube
from core.drr_bn_matching import BnMatchCache, build_bn_match_plan
from core.drr_sources import discover_drr_sources
from tests.test_drr_background_settings_workflow import _Owner
from ui_qt.common import LoadedState
from ui_qt.drr_background_settings import BackgroundDraft
from ui_qt import drr_background_settings_workflow as settings_workflow
from ui_qt import drr_bn_group_choice as group


class GroupBnChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_polH_1mW_'
        self.selected = tuple(self.csv(f'TG_rep{i+1:02}.csv', stamp, 24 if i < 3 else 96, True)
                              for i, stamp in enumerate((1000, 1010, 1020, 1500, 1510, 1520)))
        self.bn1 = self.csv('BNbackground_001.csv', 900, (10, 14), False)
        self.bn2 = self.csv('BNbackground_002.csv', 1600, (20, 28), False)
        events = [dict(event='capture_started', output={'file': name},
                       started_utc=datetime.fromtimestamp((self.root/name).stat().st_mtime, timezone.utc).isoformat(),
                       run_id='six-measurement-run' if name in self.selected else 'reference-run',
                       experiment_id='experiment-A') for name in (*self.selected, self.bn1, self.bn2)]
        (self.root/'capture.experiment.events.jsonl').write_text('\n'.join(map(json.dumps, events))+'\n')
        self.plan = build_bn_match_plan(self.root, self.selected)
        self.assertTrue(self.plan.ready, self.plan.resolution.reason)
        self.assertEqual([a.baseline_files for a in self.plan.resolution.assignments],
                         [(self.bn1,)]*3+[(self.bn2,)]*3)
        self.assertEqual(len(self.plan.batches), 1)
        self.assertEqual(self.plan.batches[0].status, 'conflict')
        self.draft = BackgroundDraft(kind='bn', files=(self.bn1, self.bn2), bn_plan=self.plan,
                                     bn_warnings_acknowledged=True)
        self.owner = _Owner(self.root, self.selected[0], self.bn1)
        self.owner.drr_selected_files = list(self.selected)
        self.owner._load_in_progress = False
        self.owner.drr_bn_group_row = QWidget()
        self.owner.drr_bn_group_combo = QComboBox(self.owner.drr_bn_group_row)
        self.addCleanup(self.owner.drr_bn_group_row.deleteLater)
        self.owner._drr_derivative_cache = {}
        self.owner._drr_background_intent = self.draft
        self.owner._drr_background_intent_root = settings_workflow._root_key(self.owner)
        self.owner._drr_background_intent_selection = self.selected
        self.owner._drr_assignments = self.plan.resolution.assignments
        self.owner._drr_assignments_automatic = True
        self.owner.loaded = self.load(self.options(group.AUTO))
        self.owner._load_task_data = self.load
        self.owner._on_loaded = Mock(side_effect=self.publish)
        # Refreshing during _commit must not mistake its short transition
        # between the old LoadedState and new adapter controls for a new group.
        self.owner.drr_controller._update_drr_selection_labels.side_effect = lambda: group.refresh_group_choices(self.owner)
        group.refresh_group_choices(self.owner)

    def csv(self, suffix, stamp, intensity, varying):
        name = self.prefix+suffix
        intensities = intensity if isinstance(intensity, tuple) else (intensity, intensity)
        rows = ['Vbg,Vtg,'+','.join(str(value) for value in range(740, 771))]
        rows.extend(f'{i if varying else 0},{i if varying else 0},'+','.join([str(value)]*31)
                    for i, value in enumerate(intensities))
        (self.root/name).write_text('\n'.join(rows)+'\n')
        os.utime(self.root/name, (stamp, stamp))
        return name

    def state(self):
        return group.GroupBnState(str(self.root.resolve()).casefold(), self.selected, self.plan,
                                  self.draft, (self.bn1, self.bn2), applied_intent=self.draft)

    def options(self, choice):
        return group._options(self.owner, self.state(), choice)

    def load(self, options, *, progress=None, log=None):
        cube = load_drr_resolved_cube(options.folder, options.selected_files, options.drr_assignments,
                                      y_axis=options.y_axis_spec, derivative=None)
        return LoadedState(mode='DRR', folder=options.folder, selected_files=list(options.selected_files),
                           primary_file=options.selected_files[0], baseline_files=list(options.baseline_files),
                           cube=cube, y_axis_spec=options.y_axis_spec,
                           drr_background_selection=options.drr_background_selection,
                           drr_assignments=options.drr_assignments)

    def publish(self, loaded, *, preserve_drr_display=False, drr_derivatives=None, drr_display_snapshot=None):
        self.assertTrue(preserve_drr_display)
        self.owner.loaded = loaded
        self.owner._drr_assignments = loaded.drr_assignments
        self.owner._drr_assignments_automatic = loaded.drr_background_selection['selection_method'] == 'automatic_per_measurement'
        self.owner._drr_derivative_cache = dict(drr_derivatives)
        group.refresh_group_choices(self.owner)

    def prepared(self, choice):
        return group.prepare_group_choice(self.state(), choice, self.options(choice),
                                          group._transform_specs(self.owner), self.load,
                                          progress=Mock(), log=Mock())

    def fingerprint(self):
        owner = self.owner
        return (id(owner.loaded), owner.loaded.cube.Z.copy(), tuple(owner.drr_baseline_files_manual),
                tuple(owner._drr_assignments), settings_workflow.source_intent(owner),
                dict(owner.settings.values), owner.drr_sg_window_spin.value(), owner.drr_sg_poly_spin.value())

    def assert_unchanged(self, before):
        after = self.fingerprint()
        self.assertEqual(after[0], before[0])
        np.testing.assert_array_equal(after[1], before[1])
        self.assertEqual(after[2:], before[2:])

    def run_request(self):
        request = self.owner._drr_bn_group_request
        request.worker.run()
        self.app.processEvents()
        self.app.processEvents()
        return request

    def test_all_six_equal_weights_and_both_derivatives_prepared_before_publish(self):
        before = self.fingerprint()
        source_bytes = {name: (self.root/name).read_bytes() for name in (*self.selected, self.bn1, self.bn2)}
        group.request_group_choice(self.owner, self.bn1)
        self.assert_unchanged(before)
        self.assertEqual(self.owner.drr_bn_group_combo.currentData(), self.bn1)
        self.assertIn('Calculating', self.owner.drr_bn_group_combo.toolTip())
        self.run_request()
        np.testing.assert_allclose(self.owner.loaded.cube.Z, 4.0, atol=1e-12)
        self.assertEqual(self.owner.loaded.selected_files, list(self.selected))
        self.assertEqual([a.baseline_files for a in self.owner.loaded.drr_assignments], [(self.bn1,)]*6)
        self.assertEqual({key[1] for key in self.owner._drr_derivative_cache}, {None, 1, 2})
        self.assertEqual(self.owner._drr_bn_group_state.choice, self.bn1)
        self.assertEqual(self.owner.settings.values, {})
        self.owner._start_load.assert_not_called()
        self.assertEqual({name: (self.root/name).read_bytes() for name in source_bytes}, source_bytes)
        self.assertEqual(self.owner.drr_sg_window_spin.value(), before[-2])
        self.assertEqual(self.owner.drr_sg_poly_spin.value(), before[-1])

    def test_restore_auto_keeps_original_three_plus_three_mapping(self):
        for choice, expected in ((self.bn2, 1.5), (group.AUTO, 2.0)):
            group.request_group_choice(self.owner, choice)
            self.run_request()
            np.testing.assert_allclose(self.owner.loaded.cube.Z, expected, atol=1e-12)
        self.assertEqual(self.owner.loaded.drr_assignments, self.plan.resolution.assignments)
        self.assertIs(settings_workflow.source_intent(self.owner).bn_plan, self.plan)
        self.assertIs(self.owner._drr_bn_confirmation.plan, self.plan)
        self.assertEqual(self.owner.drr_bn_group_combo.currentData(), group.AUTO)
        self.assertEqual(self.owner.settings.values, {})

    def test_load_failure_keeps_applied_recipe_loaded_state_and_settings(self):
        before = self.fingerprint()
        self.owner._load_task_data = Mock(side_effect=ValueError('Unreadable measurement'))
        group.request_group_choice(self.owner, self.bn1)
        self.run_request()
        self.assert_unchanged(before)
        self.owner._on_loaded.assert_not_called()
        self.assertIn('Group BN was not changed', self.owner._status.call_args.args[0])

    def test_failed_second_derivative_never_commits_successful_raw_load(self):
        from ui_qt.drr_display_compute import compute
        before = self.fingerprint()
        def failing(cube, key, **kwargs):
            if key[1] == 2:
                raise ValueError('Cannot calculate d2E')
            return compute(cube, key, **kwargs)
        with patch('ui_qt.drr_display_compute.compute', side_effect=failing):
            group.request_group_choice(self.owner, self.bn1)
            self.run_request()
        self.assert_unchanged(before)
        self.owner._on_loaded.assert_not_called()
        self.assertIn('Cannot calculate d2E', self.owner._status.call_args.args[0])
        self.assertEqual(self.owner.drr_bn_group_combo.currentData(), group.AUTO)

    def test_newer_request_wins_and_auto_cancels_pending_override(self):
        first = self.prepared(self.bn1)
        group.request_group_choice(self.owner, self.bn1)
        stale = self.owner._drr_bn_group_request
        group.request_group_choice(self.owner, self.bn2)
        current = self.owner._drr_bn_group_request
        self.assertTrue(stale.cancellation.cancelled)
        stale.complete(first)
        self.owner._on_loaded.assert_not_called()
        self.run_request()
        self.assertEqual(self.owner._drr_bn_group_state.choice, self.bn2)
        self.assertIsNot(current, stale)
        group.request_group_choice(self.owner, self.bn1)
        pending = self.owner._drr_bn_group_request
        group.request_group_choice(self.owner, self.bn2)
        self.assertTrue(pending.cancellation.cancelled)
        self.assertIsNone(self.owner._drr_bn_group_request)
        self.assertEqual(self.owner._drr_bn_group_state.choice, self.bn2)

    def test_changed_group_processing_or_active_tab_rejects_old_delivery(self):
        prepared = self.prepared(self.bn1)
        for change in ('selection', 'sg', 'tab', 'folder'):
            with self.subTest(change=change):
                self.owner.current_folder = str(self.root)
                self.owner.drr_selected_files = list(self.selected)
                self.owner.drr_sg_window_spin.number = 5
                self.owner.tabs = SimpleNamespace(currentIndex=lambda: 0, tabText=lambda _: 'DRR')
                group.refresh_group_choices(self.owner)
                group.request_group_choice(self.owner, self.bn1)
                request = self.owner._drr_bn_group_request
                if change == 'selection':
                    self.owner.drr_selected_files.reverse()
                elif change == 'sg':
                    self.owner.drr_sg_window_spin.number = 7
                elif change == 'folder':
                    self.owner.current_folder = str(self.root / 'different_measurements')
                else:
                    self.owner.tabs.tabText = lambda _: 'PL'
                request.complete(prepared)
                self.owner._on_loaded.assert_not_called()
                self.assertEqual(self.owner.loaded.drr_assignments, self.plan.resolution.assignments)
                group.cancel_group_choice(self.owner)

    def use_short_folder_alias(self):
        if os.name != 'nt':
            self.skipTest('Windows short directory names are required')
        import ctypes
        short = ctypes.create_unicode_buffer(32768)
        length = ctypes.windll.kernel32.GetShortPathNameW(str(self.root), short, len(short))
        if not length or length >= len(short) or short.value.casefold() == str(self.root).casefold():
            self.skipTest('The test volume does not provide an 8.3 directory alias')
        self.assertEqual(Path(short.value).resolve(), self.root.resolve())
        self.root = Path(short.value)

    def test_short_folder_alias_keeps_real_precomputed_publication(self):
        self.use_short_folder_alias()
        # Exercise both the folder picker and previous loaded result using an
        # alias while the BN plan and freshly loaded result use the long name.
        self.owner.loaded.folder = str(self.root)
        self.test_real_main_window_accepts_precomputed_result_without_reload_or_settings_write()

    def test_short_folder_alias_keeps_real_render_failure_rollback(self):
        self.use_short_folder_alias()
        self.test_real_second_product_render_failure_restores_old_meshes_and_toolbar_zoom()

    def test_stale_sources_rejected_again_immediately_before_publication(self):
        prepared = self.prepared(self.bn1)
        group.request_group_choice(self.owner, self.bn1)
        request = self.owner._drr_bn_group_request
        before = self.fingerprint()
        os.utime(self.root/self.bn1, (901, 901))
        request.complete(prepared)
        self.assert_unchanged(before)
        self.owner._on_loaded.assert_not_called()

    def test_publication_failure_rolls_back_recipe_and_old_loaded_result(self):
        prepared = self.prepared(self.bn1)
        before = self.fingerprint()
        def publish_or_fail(loaded, **kwargs):
            self.publish(loaded, **kwargs)
            if loaded is prepared.loaded:
                raise ValueError('Plot publication failed')
        self.owner._on_loaded.side_effect = publish_or_fail
        group.request_group_choice(self.owner, self.bn1)
        self.owner._drr_bn_group_request.complete(prepared)
        self.assert_unchanged(before)
        self.assertEqual(self.owner._drr_bn_group_state.choice, group.AUTO)
        self.assertIn('Plot publication failed', self.owner._status.call_args.args[0])

    def test_explicit_non_bn_intent_clears_scope_and_candidates_require_all_members(self):
        state = self.owner._drr_bn_group_state
        batch = replace(self.plan.batches[0], candidate_files=(self.bn1,))
        restricted = replace(self.plan, batches=(batch,))
        self.owner._drr_background_intent = replace(self.draft, bn_plan=restricted)
        self.owner._drr_bn_group_state = None
        group.refresh_group_choices(self.owner)
        self.assertEqual(self.owner._drr_bn_group_state.candidates, (self.bn1,))
        self.owner._drr_background_intent = BackgroundDraft(kind='p', files=('Pback.csv',))
        group.refresh_group_choices(self.owner)
        self.assertTrue(self.owner.drr_bn_group_row.isHidden())
        self.assertIsNone(self.owner._drr_bn_group_state)
        with self.assertRaisesRegex(ValueError, 'validated'):
            group._options(self.owner, state, 'unvalidated_BNbackground_003.csv')

    def test_refresh_has_no_discovery_and_duplicate_short_labels_are_distinct(self):
        duplicate = 'another/'+Path(self.bn1).name
        state = replace(self.owner._drr_bn_group_state, candidates=(self.bn1, duplicate))
        with patch.object(group, '_eligible_state', return_value=state), patch(
                'core.drr_bn_matching.plan_is_current', side_effect=AssertionError('No GUI discovery')):
            group.refresh_group_choices(self.owner)
        combo = self.owner.drr_bn_group_combo
        self.assertNotEqual(combo.itemText(1), combo.itemText(2))
        self.assertIn(duplicate, combo.itemText(2))

    def test_real_main_window_accepts_precomputed_result_without_reload_or_settings_write(self):
        from ui_qt.main_window import MainWindow
        from tests.ui_test_helpers import dispose_owned_window
        with tempfile.TemporaryDirectory() as settings_root:
            store = QSettings(str(Path(settings_root)/'settings.ini'), QSettings.IniFormat)
            with patch('ui_qt.main_window.QSettings', return_value=store), patch.object(
                    MainWindow, '_restore_last_folder', lambda _: None), patch.object(
                    MainWindow, '_schedule_automatic_update_check', lambda _: None):
                window = MainWindow()
            try:
                window._suspend_drr_autoplot = True
                window.tabs.setCurrentIndex(next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == 'DRR'))
                window.current_folder = str(self.root)
                window.drr_selected_files = list(self.selected)
                window.drr_available_sources = discover_drr_sources(self.root)
                prepared_confirmation = BnMatchCache().prepare_confirm(self.plan, warnings_acknowledged=True)
                settings_workflow._commit(window, self.draft, prepared_confirmation, start_load=False, persist=False)
                window.loaded = self.owner.loaded
                window.drr_sg_window_spin.setValue(5)
                window.drr_sg_poly_spin.setValue(2)
                group.refresh_group_choices(window)
                settings_before = {key: store.value(key) for key in store.allKeys()}
                state = window._drr_bn_group_state
                prepared = group.prepare_group_choice(state, self.bn1, group._options(window, state, self.bn1),
                    group._transform_specs(window), window._load_task_data, progress=Mock(), log=Mock())
                jobs = []
                with patch.object(window.thread_pool, 'start', jobs.append), patch.object(window, '_start_load') as reload, patch.object(
                        window, '_plot_mode') as plot:
                    group.request_group_choice(window, self.bn1)
                    request = window._drr_bn_group_request
                    request.complete(prepared)
                self.assertIs(window.loaded, prepared.loaded)
                self.assertEqual(window.drr_selected_files, list(self.selected))
                self.assertEqual(window._drr_bn_group_state.choice, self.bn1)
                np.testing.assert_allclose(window.loaded.cube.Z, 4.0, atol=1e-12)
                self.assertEqual({key[1] for key in window._drr_derivative_cache}, {None, 1, 2})
                self.assertEqual({key: store.value(key) for key in store.allKeys()}, settings_before)
                reload.assert_not_called()
                plot.assert_called_once_with('DRR', auto=False)

                # The real signal path must also reject a successful raw
                # load if its required d2E calculation fails.
                from ui_qt.drr_display_compute import compute
                old_loaded, old_intent = window.loaded, settings_workflow.source_intent(window)
                def failing(cube, key, **kwargs):
                    if key[1] == 2:
                        raise ValueError('Rejected d2E')
                    return compute(cube, key, **kwargs)
                with patch.object(window.thread_pool, 'start', jobs.append), patch(
                        'ui_qt.drr_display_compute.compute', side_effect=failing):
                    group.request_group_choice(window, self.bn2)
                    window._drr_bn_group_request.worker.run()
                    self.app.processEvents()
                self.assertIs(window.loaded, old_loaded)
                self.assertIs(settings_workflow.source_intent(window), old_intent)
                self.assertEqual(window.drr_bn_group_combo.currentData(), self.bn1)

                # A completed older choice cannot publish after a newer Auto
                # request, including through MainWindow's normal callbacks.
                state = window._drr_bn_group_state
                old_prepared = group.prepare_group_choice(state, self.bn2, group._options(window, state, self.bn2),
                    group._transform_specs(window), window._load_task_data, progress=Mock(), log=Mock())
                with patch.object(window.thread_pool, 'start', jobs.append), patch.object(window, '_plot_mode'):
                    group.request_group_choice(window, self.bn2)
                    superseded = window._drr_bn_group_request
                    self.assertIsNotNone(superseded)
                    group.request_group_choice(window, group.AUTO)
                    newest = window._drr_bn_group_request
                    superseded.complete(old_prepared)
                    self.assertIs(window.loaded, old_loaded)
                    newest.worker.run()
                    self.app.processEvents()
                self.assertEqual(window.loaded.drr_assignments, self.plan.resolution.assignments)
                self.assertEqual(window._drr_bn_group_state.choice, group.AUTO)
                np.testing.assert_allclose(window.loaded.cube.Z, 2.0, atol=1e-12)

                # Opening higher-level settings cancels the quick operation;
                # cancelling that settings dialog leaves Auto applied.
                dialog = QDialog(window)
                dialog.exec = Mock(return_value=QDialog.Rejected)
                dialog.selection = None
                before_cancel = window.loaded
                with patch.object(window.thread_pool, 'start', jobs.append), patch(
                        'ui_qt.drr_background_settings.DrrBackgroundSettings', return_value=dialog), patch(
                        'ui_qt.drr_background_settings_workflow._InlineCatalogBridge'):
                    group.request_group_choice(window, self.bn1)
                    cancelled = window._drr_bn_group_request
                    settings_workflow.edit_settings(window)
                self.assertTrue(cancelled.cancellation.cancelled)
                self.assertIs(window.loaded, before_cancel)
                self.assertEqual(window._drr_bn_group_state.choice, group.AUTO)
                self.assertEqual({key: store.value(key) for key in store.allKeys()}, settings_before)
            finally:
                dispose_owned_window(window)

    def test_real_second_product_render_failure_restores_old_meshes_and_toolbar_zoom(self):
        from ui_qt.main_window import MainWindow, plot_drr
        from tests.ui_test_helpers import dispose_owned_window
        with tempfile.TemporaryDirectory() as settings_root:
            store = QSettings(str(Path(settings_root)/'settings.ini'), QSettings.IniFormat)
            with patch('ui_qt.main_window.QSettings', return_value=store), patch.object(
                    MainWindow, '_restore_last_folder', lambda _: None), patch.object(
                    MainWindow, '_schedule_automatic_update_check', lambda _: None):
                window = MainWindow()
            try:
                window._suspend_drr_autoplot = True
                window.tabs.setCurrentIndex(next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == 'DRR'))
                window.current_folder = str(self.root)
                window.drr_selected_files = list(self.selected)
                window.drr_available_sources = discover_drr_sources(self.root)
                confirmation = BnMatchCache().prepare_confirm(self.plan, warnings_acknowledged=True)
                window.loaded = self.owner.loaded
                settings_workflow._commit(window, self.draft, confirmation, start_load=False, persist=False)
                window.drr_sg_auto_chk.setChecked(False)
                window.drr_sg_window_spin.setValue(5)
                window.drr_sg_poly_spin.setValue(2)
                window.drr_second_auto_scale = False
                window._drr_side_by_side, window._drr_plot_view = True, 'both'
                for name, value in dict(xmin=1.62, xmax=1.67, ymin=0., ymax=1., gate=.4, vmin=1.4, vmax=2.4).items():
                    window.drr_spins[name].setValue(value)
                window.drr_second_vmin_spin.setValue(-.001)
                window.drr_second_vmax_spin.setValue(.001)
                group.refresh_group_choices(window)
                state = window._drr_bn_group_state
                auto = group.prepare_group_choice(state, group.AUTO, group._options(window, state, group.AUTO),
                    group._transform_specs(window), window._load_task_data, progress=Mock(), log=Mock())
                window.loaded = auto.loaded
                window._drr_derivative_cache.update(auto.derivatives)
                with patch.object(window, '_start_load') as reload, patch.object(window, '_show_error') as error:
                    window._plot_mode('DRR', auto=False)
                    error.assert_not_called()
                    reload.assert_not_called()
                self.assertEqual(set(window._drr_heatmap_axes), {'raw', 'second'})
                for axis in window._drr_heatmap_axes.values():
                    axis.set_xlim(1.625, 1.655)
                    axis.set_ylim(.1, .8)
                before_meshes = {key: [np.asarray(mesh.get_array()).copy() for mesh in axis.collections]
                                 for key, axis in window._drr_heatmap_axes.items()}
                old_loaded, old_intent = window.loaded, settings_workflow.source_intent(window)
                values_before = {name: widget.value() for name, widget in window.drr_spins.items()}
                sg_before = (window.drr_sg_window_spin.value(), window.drr_sg_poly_spin.value())
                settings_before = {key: store.value(key) for key in store.allKeys()}
                prepared = group.prepare_group_choice(state, self.bn1, group._options(window, state, self.bn1),
                    group._transform_specs(window), window._load_task_data, progress=Mock(), log=Mock())
                new_plots = []
                def fail_second_new_product(*args, **kwargs):
                    if window.loaded is prepared.loaded:
                        new_plots.append(args[1])
                        if len(new_plots) == 2:
                            raise ValueError('Second product render failed')
                    return plot_drr(*args, **kwargs)
                jobs = []
                with patch.object(window.thread_pool, 'start', jobs.append), patch(
                        'ui_qt.drr_plot_reuse.try_update', return_value=False), patch(
                        'ui_qt.main_window.plot_drr', side_effect=fail_second_new_product), patch.object(
                        window, '_start_load') as reload:
                    group.request_group_choice(window, self.bn1)
                    window._drr_bn_group_request.complete(prepared)
                self.assertEqual(len(new_plots), 2)
                self.assertIs(window.loaded, old_loaded)
                self.assertIs(settings_workflow.source_intent(window), old_intent)
                self.assertEqual(window._drr_bn_group_state.choice, group.AUTO)
                self.assertEqual({name: widget.value() for name, widget in window.drr_spins.items()}, values_before)
                self.assertEqual((window.drr_sg_window_spin.value(), window.drr_sg_poly_spin.value()), sg_before)
                self.assertEqual({key: store.value(key) for key in store.allKeys()}, settings_before)
                reload.assert_not_called()
                for key, axis in window._drr_heatmap_axes.items():
                    np.testing.assert_allclose(axis.get_xlim(), (1.625, 1.655))
                    np.testing.assert_allclose(axis.get_ylim(), (.1, .8))
                    self.assertEqual(len(axis.collections), len(before_meshes[key]))
                    for mesh, before in zip(axis.collections, before_meshes[key]):
                        np.testing.assert_array_equal(np.asarray(mesh.get_array()), before)
            finally:
                dispose_owned_window(window)


if __name__ == '__main__':
    unittest.main()
