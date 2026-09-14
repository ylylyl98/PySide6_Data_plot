from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from ui_qt.main_window import MainWindow, _scan_folder_sources_worker, _cached_folder_sources_worker


class LazyCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def window(self):
        with patch.object(MainWindow, '_restore_last_folder', lambda self: None):
            w = MainWindow()
        self.addCleanup(w.close)
        return w

    def test_pl_scan_does_not_discover_power_mcd_or_shg(self):
        with tempfile.TemporaryDirectory() as folder, patch('ui_qt.main_window.discover_mcd_processing_status') as mcd, patch('ui_qt.main_window.scan_shg_history') as shg, patch('core.data_io.get_power_series_sources') as power:
            result = _scan_folder_sources_worker(folder, mode='PL', progress=None, log=None)
        self.assertEqual(result[0], folder)
        mcd.assert_not_called()
        shg.assert_not_called()
        power.assert_not_called()

    def test_active_pl_refresh_does_not_start_drr(self):
        w = self.window()
        with tempfile.TemporaryDirectory() as folder, patch.object(w.thread_pool, 'start') as start, patch.object(w, '_queue_drr_catalog_refresh') as drr:
            w.current_folder = folder
            w._refresh_file_lists()
        drr.assert_not_called()
        self.assertEqual(start.call_count, 1)

    def test_partial_catalog_keeps_inactive_mcd_choices(self):
        w = self.window()
        w.mcd_available_files = ['kept.csv']
        w.mcd_files.addItem('kept.csv')
        w.mcd_files.item(0).setSelected(True)
        with tempfile.TemporaryDirectory() as folder:
            w.current_folder = folder
            result = _scan_folder_sources_worker(folder, mode='PL', progress=None, log=None)
            w._on_file_lists_result(result, w._file_refresh_generation, False, set(), set(), set(), set(), [])
        self.assertEqual(w.mcd_available_files, ['kept.csv'])
        self.assertEqual(w._selected(w.mcd_files), ['kept.csv'])

    def test_restart_publishes_cache_before_validation_without_discovery(self):
        with tempfile.TemporaryDirectory() as folder:
            args = dict(mode='PL', power_include_legacy=False, force=False, progress=None, log=None)
            _cached_folder_sources_worker(folder, publish_cached=lambda _: None, **args)
            events = []
            from core.source_catalog_cache import SourceCatalogCache
            inventory = SourceCatalogCache.inventory
            def inspect(cache):
                events.append('validate')
                return inventory(cache)
            with patch.object(SourceCatalogCache, 'inventory', inspect), patch('ui_qt.main_window._scan_folder_sources_worker') as scan:
                _cached_folder_sources_worker(folder, publish_cached=lambda _: events.append('preview'), **args)
            self.assertEqual(events, ['preview', 'validate'])
            scan.assert_not_called()

    def test_preview_does_not_finish_worker_and_current_selection_survives(self):
        w = self.window()
        with tempfile.TemporaryDirectory() as folder:
            w.current_folder = folder
            result = list(_scan_folder_sources_worker(folder, mode='PL', progress=None, log=None))
            result[3] = ['a.csv', 'b.csv']
            result[-1] = dict(result[-1], preview=True)
            w._file_refresh_running = True
            args = (w._file_refresh_generation, False, set(), set(), set(), set(), [])
            w._on_file_lists_result(tuple(result), *args)
            self.assertTrue(w._file_refresh_running)
            self.assertNotIn('PL', w._catalog_ready_modes)
            w.pl_files.item(1).setSelected(True)
            result[-1] = dict(result[-1], preview=False)
            w._on_file_lists_result(tuple(result), *args)
            self.assertEqual(w._selected(w.pl_files), ['b.csv'])
            self.assertFalse(w._file_refresh_running)

    def test_slides_reentry_does_not_repeat_scan(self):
        from ui_qt.presentation_widget import PresentationBuilderWidget
        widget = PresentationBuilderWidget()
        self.addCleanup(widget.close)
        with tempfile.TemporaryDirectory() as folder, patch.object(widget, 'refresh_plots') as refresh:
            widget.set_experiment_folder(folder)
            widget.set_experiment_folder(folder)
            refresh.assert_called_once()

    def test_pending_manual_refresh_retains_its_mode_without_rebuilding(self):
        w = self.window()
        with tempfile.TemporaryDirectory() as folder:
            w.current_folder = folder
            w._file_refresh_running = True
            w._refresh_file_lists(auto=False, mode='MCD')
            w._refresh_file_lists(auto=True, mode='MCD')
            self.assertTrue(w._catalog_pending_requests['MCD'])
            w._file_refresh_running = False
            with patch.object(w.thread_pool, 'start') as start:
                w._run_pending_catalog_refresh()
            worker = start.call_args.args[0]
            self.assertEqual(worker.kwargs['mode'], 'MCD')
            self.assertFalse(worker.kwargs['force'])

    def test_explicit_rebuild_survives_queued_normal_refresh(self):
        w = self.window()
        with tempfile.TemporaryDirectory() as folder:
            w.current_folder = folder
            w._file_refresh_running = True
            w._refresh_file_lists(mode='MCD', force=True)
            w._refresh_file_lists(mode='MCD')
            w._file_refresh_running = False
            with patch.object(w.thread_pool, 'start') as start:
                w._run_pending_catalog_refresh()
            self.assertEqual(start.call_args.args[0].kwargs['mode'], 'MCD')
            self.assertTrue(start.call_args.args[0].kwargs['force'])

    def test_refresh_button_reuses_catalog_and_rebuild_action_forces_discovery(self):
        w = self.window()
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as cache:
            with patch.dict(os.environ, {'LOCALAPPDATA': cache}):
                w.current_folder = folder
                _cached_folder_sources_worker(folder, mode='PL', power_include_legacy=False,
                    force=False, publish_cached=None, progress=None, log=None)
                with patch.object(w.thread_pool, 'start', side_effect=lambda worker: worker.run()), \
                     patch('ui_qt.main_window._scan_folder_sources_worker', wraps=_scan_folder_sources_worker) as scan:
                    w.refresh_btn.click()
                    self.assertEqual(scan.call_count, 0)
                    w.rebuild_catalog_action.trigger()
                    self.assertEqual(scan.call_count, 1)

    def test_drr_rebuild_survives_queued_normal_refresh(self):
        w = self.window()
        with tempfile.TemporaryDirectory() as folder:
            w.current_folder = folder
            w._drr_refresh_running = True
            w._queue_drr_catalog_refresh(auto=False, old_source_files=set(), force=True)
            w._queue_drr_catalog_refresh(auto=True, old_source_files=set())
            w._drr_refresh_running = False
            with patch.object(w.thread_pool, 'start') as start:
                w._finish_drr_catalog_refresh()
            self.assertTrue(start.call_args.args[0].kwargs['force'])

    def test_old_folder_preview_does_not_replace_new_sources(self):
        w = self.window()
        with tempfile.TemporaryDirectory() as old, tempfile.TemporaryDirectory() as new:
            result = list(_scan_folder_sources_worker(old, mode='PL', progress=None, log=None))
            result[3] = ['old.csv']
            result[-1] = dict(result[-1], preview=True)
            w.current_folder = new
            w.pl_available_files = ['new.csv']
            w._on_file_lists_result(tuple(result), w._file_refresh_generation, False, set(), set(), set(), set(), [])
            self.assertEqual(w.pl_available_files, ['new.csv'])

    def test_stale_preview_does_not_remove_newer_displayed_selection(self):
        w = self.window()
        with tempfile.TemporaryDirectory() as folder:
            w.current_folder = folder
            result = list(_scan_folder_sources_worker(folder, mode='PL', progress=None, log=None))
            result[3] = ['old.csv', 'new.csv']
            args = (w._file_refresh_generation, False, set(), set(), set(), set(), [])
            w._on_file_lists_result(tuple(result), *args)
            w.pl_files.item(1).setSelected(True)
            result[3] = ['old.csv']
            result[-1] = dict(result[-1], preview=True)
            w._on_file_lists_result(tuple(result), *args)
            self.assertEqual(w._selected(w.pl_files), ['new.csv'])

    def test_compare_preview_preserves_newer_shared_pl_selection(self):
        w = self.window()
        with tempfile.TemporaryDirectory() as folder:
            w.current_folder = folder
            result = list(_scan_folder_sources_worker(folder, mode='PL', progress=None, log=None))
            result[3] = ['old.csv', 'new.csv']
            args = (w._file_refresh_generation, False, set(), set(), set(), set(), [])
            w._on_file_lists_result(tuple(result), *args)
            w.pl_files.item(1).setSelected(True)
            result[3] = ['old.csv']
            result[-1] = dict(result[-1], mode='Compare', preview=True)
            w._on_file_lists_result(tuple(result), *args)
            self.assertEqual(w._selected(w.pl_files), ['new.csv'])

    def test_drr_manual_refresh_remains_manual_behind_auto(self):
        w = self.window()
        w._drr_refresh_running = True
        w._queue_drr_catalog_refresh(auto=True, old_source_files=set())
        w._queue_drr_catalog_refresh(auto=False, old_source_files=set())
        w._drr_refresh_running = False
        with patch.object(w, '_start_drr_catalog_refresh') as start:
            w._finish_drr_catalog_refresh()
        self.assertFalse(start.call_args.kwargs['auto'])

    def test_slides_restart_reuses_catalog_and_publishes_preview(self):
        from ui_qt.presentation_widget import _DiscoveryWorker
        with tempfile.TemporaryDirectory() as folder:
            first = _DiscoveryWorker(folder, force=False)
            first.run()
            second = _DiscoveryWorker(folder, force=False)
            published = []
            second.signals.result.connect(published.append)
            with patch('ui_qt.presentation_widget.discover_plot_images') as discover:
                second.run()
            discover.assert_not_called()
            self.assertEqual(len(published), 2)


if __name__ == '__main__':
    unittest.main()
