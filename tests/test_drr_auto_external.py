from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from core.drr_sources import discover_drr_sources


class AutoExternalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.measurement = 'sample_REF_760nm.csv'
        self.background = 'sample_REF_760nm_background.csv'
        self.write(self.measurement, varying=True)
        self.write(self.background)

    def write(self, name, varying=False, axis='740,760,780'):
        (self.root / name).write_text(
            f'Vbg,{axis}\n0,10,11,12\n{1 if varying else 0},12,13,14\n')

    def resolve(self):
        from core.drr_auto_external import resolve_auto_external
        return resolve_auto_external(self.root, discover_drr_sources(self.root), [self.measurement])

    def test_selects_recommendation_even_without_unique_high_confidence_match(self):
        self.write('other_REF_760nm_background.csv')
        os.utime(self.root / 'other_REF_760nm_background.csv', (1000, 1000))
        result = self.resolve()
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual(result.assignments[0].baseline_files, (self.background,))
        self.assertEqual(result.assignments[0].baseline_which, 'all')

    def test_never_selects_measurement_itself_or_incompatible_background(self):
        self.write(self.background, axis='800,810,820')
        result = self.resolve()
        self.assertFalse(result.resolved)
        self.assertIn(self.measurement, result.unresolved_measurements)

    def test_recommendation_loads_whole_compatible_background_group(self):
        (self.root / self.background).unlink()
        names = ['sample_REF_760nm_background_rep1_1.csv',
                 'sample_REF_760nm_background_rep1_2.csv']
        for name in names:
            self.write(name)
            os.utime(self.root / name, (1700000100, 1700000100))
        incompatible = 'sample_REF_760nm_background_rep1_3.csv'
        self.write(incompatible, axis='800,810,820')
        other_gate = 'sample_REF_760nm_background_rep1_4.csv'
        (self.root / other_gate).write_text('Vbg,740,760,780\n5,10,11,12\n5,12,13,14\n')
        os.utime(self.root / other_gate, (1700000000, 1700000000))
        result = self.resolve()
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual(result.assignments[0].baseline_files, tuple(names))
        self.assertEqual(result.assignments[0].baseline_which, 'all')

    def test_valid_saved_external_wins_over_recommendation(self):
        import json
        older = 'other_REF_760nm_background.csv'
        self.write(older)
        folder = self.root / 'Processed Data' / 'DRR'
        folder.mkdir(parents=True)
        (folder / 'saved.metadata.json').write_text(json.dumps({
            'operation': 'DR/R',
            'sources': [
                {'role': 'measurement', 'source_path': str(self.root / self.measurement)},
                {'role': 'background', 'source_path': str(self.root / older)},
            ],
            'processing': {'baseline_selection': 'External', 'baseline_which': 'first'},
        }))
        result = self.resolve()
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual(result.assignments[0].baseline_files, (older,))
        self.assertEqual(result.assignments[0].baseline_which, 'first')

    def test_cached_catalog_does_not_rescan_folder(self):
        from core.drr_auto_external import resolve_auto_external
        catalog = discover_drr_sources(self.root)
        with patch('core.drr_auto_external.discover_drr_sources', side_effect=AssertionError('rescan')):
            self.assertTrue(resolve_auto_external(self.root, catalog, [self.measurement]).resolved)

    def test_saved_recipe_does_not_inspect_unrelated_candidates(self):
        import json
        from core.drr_auto_external import resolve_auto_external
        other = 'unused_REF_760nm.csv'
        self.write(other)
        history = self.root / 'Processed Data/DRR'
        history.mkdir(parents=True)
        (history / 'saved.metadata.json').write_text(json.dumps({
            'operation': 'DR/R', 'sources': [
                {'role': 'measurement', 'source_path': str(self.root / self.measurement)},
                {'role': 'background', 'source_path': str(self.root / self.background)}],
            'processing': {'baseline_selection': 'External'}}))
        catalog = discover_drr_sources(self.root)
        from core.drr_sources import resolve_source_path
        inspected = []
        def resolve(root, name):
            inspected.append(str(name))
            return resolve_source_path(root, name)
        with patch('core.drr_auto_external.resolve_source_path', side_effect=resolve):
            result = resolve_auto_external(self.root, catalog, [self.measurement])
        self.assertTrue(result.resolved)
        self.assertNotIn(other, inspected)

    def test_multiple_measurements_share_one_history_scan(self):
        from core.drr_auto_external import resolve_auto_external
        from core.drr_sources import _read_drr_metadata
        second = 'second_REF_760nm.csv'
        self.write(second, varying=True)
        catalog = discover_drr_sources(self.root)
        with patch('core.drr_auto_external._read_drr_metadata', wraps=_read_drr_metadata) as read:
            result = resolve_auto_external(self.root, catalog, [self.measurement, second])
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual(len(result.assignments), 2)
        self.assertEqual(read.call_count, 1)


class AutoExternalUiTests(AutoExternalTests):
    def setUp(self):
        super().setUp()
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
        from ui_qt.main_window import MainWindow
        self.app = QApplication.instance() or QApplication([])
        settings = QSettings(str(self.root / 'settings.ini'), QSettings.IniFormat)
        with patch.object(MainWindow, '_restore_last_folder', lambda self: None), patch(
            'ui_qt.main_window.QSettings', return_value=settings
        ), patch.object(MainWindow, '_schedule_automatic_update_check', lambda self: None):
            self.window = MainWindow()
        self.addCleanup(self.window.close)
        self.window.current_folder = str(self.root)
        self.window.drr_selected_files = [self.measurement]
        self.window.drr_available_sources = discover_drr_sources(self.root)

    def test_switch_external_matches_in_worker_then_loads_once(self):
        w = self.window
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append):
            w.drr_baseline_combo.setCurrentText('External')
            self.assertEqual(len(jobs), 1)
            self.assertEqual(w.drr_baseline_files_manual, [])
            self.assertFalse(w.drr_baseline_combine_combo.isEnabled())
            jobs[0].run()
            self.app.processEvents()
        self.assertEqual(w.drr_baseline_files_manual, [self.background])
        self.assertEqual(len(jobs), 2)
        self.assertTrue(w.drr_baseline_combine_combo.isEnabled())
        self.assertEqual(jobs[1].args[0].drr_baseline_text, 'External')
        self.assertIn('recommendation', w.drr_baseline_summary.toolTip().lower())
        self.assertIn('recommendation', jobs[1].args[0].drr_assignments[0].selection_reason.lower())

    def test_real_worker_resolves_off_gui_and_applies_on_gui(self):
        import threading
        import time
        from PySide6.QtTest import QTest
        from core.drr_auto_external import resolve_auto_external
        w = self.window
        gui_thread = threading.get_ident()
        resolver_threads, load_threads = [], []
        def resolve(*args, **kwargs):
            resolver_threads.append(threading.get_ident())
            return resolve_auto_external(*args, **kwargs)
        def load(*args, **kwargs):
            load_threads.append(threading.get_ident())
        with patch('ui_qt.drr_auto_external.resolve_auto_external', side_effect=resolve), \
             patch.object(w, '_start_load', side_effect=load):
            w.drr_baseline_combo.setCurrentText('External')
            deadline = time.monotonic() + 5
            while not load_threads and time.monotonic() < deadline:
                QTest.qWait(10)
            w.thread_pool.waitForDone(5000)
        self.assertEqual(w.drr_baseline_files_manual, [self.background])
        self.assertEqual(load_threads, [gui_thread])
        self.assertEqual(len(resolver_threads), 1)
        self.assertNotEqual(resolver_threads[0], gui_thread)

    def test_switch_self_discards_pending_external_result(self):
        w = self.window
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append):
            w.drr_baseline_combo.setCurrentText('External')
            self.assertEqual(len(jobs), 1)
            w.drr_baseline_combo.setCurrentText('Self (first frame)')
            count = len(jobs)
            jobs[0].run()
            self.app.processEvents()
        self.assertEqual(len(jobs), count)
        self.assertEqual(w.drr_baseline_files_manual, [])
        self.assertEqual(w.drr_baseline_combo.currentText(), 'Self (first frame)')

    def test_existing_manual_background_is_loaded_without_rematching(self):
        w = self.window
        w.drr_baseline_files_manual = [self.background]
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append):
            w.drr_baseline_combo.setCurrentText('External')
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].args[0].baseline_files, [self.background])
        self.assertIsNone(getattr(w, '_drr_external_request', None))

    def test_auto_load_receives_every_file_in_recommended_group(self):
        w = self.window
        (self.root / self.background).unlink()
        names = ['sample_REF_760nm_background_rep1_1.csv',
                 'sample_REF_760nm_background_rep1_2.csv']
        for name in names:
            self.write(name)
        w.drr_available_sources = discover_drr_sources(self.root)
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append):
            w.drr_baseline_combo.setCurrentText('External')
            jobs[0].run()
            self.app.processEvents()
        self.assertEqual(len(jobs), 2)
        self.assertEqual(w.drr_baseline_files_manual, names)
        self.assertEqual(jobs[1].args[0].baseline_files, names)
        self.assertEqual(jobs[1].args[0].drr_assignments[0].baseline_files, tuple(names))

    def test_no_match_keeps_external_and_opens_picker(self):
        from ui_qt.controllers_drr import DrrController
        w = self.window
        (self.root / self.background).unlink()
        w.drr_available_sources = discover_drr_sources(self.root)
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append), patch.object(
            DrrController, '_open_drr_source_dialog', return_value=[]
        ) as picker:
            w.drr_baseline_combo.setCurrentText('External')
            self.assertEqual(len(jobs), 1)
            jobs[0].run()
            self.app.processEvents()
        self.assertEqual(w.drr_baseline_combo.currentText(), 'External')
        self.assertEqual(w.drr_baseline_files_manual, [])
        self.assertEqual(len(jobs), 1)
        self.assertTrue(picker.called)

    def test_measurement_change_rematches_unless_background_is_pinned(self):
        from ui_qt.controllers_drr import DrrController
        w = self.window
        second = 'second_REF_760nm.csv'
        self.write(second, varying=True)
        w.drr_available_sources = discover_drr_sources(self.root)
        w.drr_baseline_files_manual = [self.background]
        w.drr_baseline_combo.blockSignals(True)
        w.drr_baseline_combo.setCurrentText('External')
        w.drr_baseline_combo.blockSignals(False)
        for pinned in (True, False):
            w.drr_selected_files = [self.measurement]
            w.drr_baseline_files_manual = [self.background]
            w.drr_pin_baseline_chk.setChecked(pinned)
            jobs = []
            with patch.object(w.thread_pool, 'start', jobs.append), patch.object(
                DrrController, '_open_drr_source_dialog', return_value=[second]
            ):
                w.drr_controller._edit_drr_measurements()
                self.assertEqual(len(jobs), 1)
                if pinned:
                    self.assertEqual(jobs[0].args[0].baseline_files, [self.background])
                    w._on_load_finished(request_token=w._active_load_token)
                else:
                    self.assertEqual(w.drr_baseline_files_manual, [])
                    jobs[0].run()
                    self.app.processEvents()
                    self.assertEqual(len(jobs), 2)
            self.assertEqual(w.drr_baseline_combo.currentText(), 'External')

    def test_manual_background_choice_supersedes_pending_match(self):
        from ui_qt.controllers_drr import DrrController
        w = self.window
        other = 'manual_REF_background.csv'
        self.write(other)
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append), patch.object(
            DrrController, '_open_drr_source_dialog', return_value=[other]
        ):
            w.drr_baseline_combo.setCurrentText('External')
            w.drr_controller._edit_drr_baselines_dialog()
            count = len(jobs)
            jobs[0].run()
            self.app.processEvents()
        self.assertEqual(w.drr_baseline_files_manual, [other])
        self.assertEqual(len(jobs), count)

    def test_catalog_refresh_preserves_heterogeneous_recommendations(self):
        from core.drr_sources import DrrSourceCache
        w = self.window
        other = 'second_REF_810nm.csv'
        bg = 'second_REF_810nm_background.csv'
        self.write(other, varying=True, axis='800,810,820')
        self.write(bg, axis='800,810,820')
        w.drr_available_sources = discover_drr_sources(self.root)
        w.drr_selected_files.append(other)
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append):
            w.drr_baseline_combo.setCurrentText('External')
            jobs[0].run()
            self.app.processEvents()
            assignments = w._drr_assignments
            self.assertTrue(w._drr_assignments_automatic)
            with patch('ui_qt.main_window.resolve_drr_background_assignments',
                       side_effect=AssertionError('old resolver used')):
                w._on_drr_catalog_refresh_result(
                    (str(self.root), w.drr_available_sources, DrrSourceCache(load_on_init=False)),
                    w._drr_refresh_generation, True, set(),
                )
        self.assertEqual(w._drr_assignments, assignments)
        self.assertEqual(set(w.drr_baseline_files_manual), {self.background, bg})

    def test_catalog_selection_change_replaces_pending_request(self):
        w = self.window
        second = 'second_REF_760nm.csv'
        self.write(second, varying=True)
        w.drr_available_sources = discover_drr_sources(self.root)
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append):
            w.drr_baseline_combo.setCurrentText('External')
            first = w._drr_external_request
            w.drr_selected_files.append(second)
            self.assertTrue(w.drr_controller._refresh_auto_external(selection_changed=True))
            self.assertTrue(first.cancelled.is_set())
            self.assertEqual(len(jobs), 2)
            jobs[0].run()
            jobs[1].run()
            self.app.processEvents()
        self.assertEqual(len(jobs), 3)
        self.assertEqual([a.measurement_file for a in w._drr_assignments], [self.measurement, second])

    def test_catalog_filter_does_not_reload_valid_selected_background(self):
        w = self.window
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append):
            w.drr_baseline_combo.setCurrentText('External')
            jobs[0].run()
            self.app.processEvents()
            w.drr_available_sources = [s for s in w.drr_available_sources if s.source == self.measurement]
            assignments = w._drr_assignments
            self.assertTrue(w.drr_controller._refresh_auto_external())
        self.assertEqual(len(jobs), 2)
        self.assertEqual(w._drr_assignments, assignments)
