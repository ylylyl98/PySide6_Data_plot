import importlib.util
import os
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from core import processing_run as P
from core.drr_sources import DrrMeasurementAssignment
from core.raw_spectrum_cache import RawSpectrumCache
from core.workflow_request import LatestRequest
from ui_qt.common import LoadOptions, LoadedState, Worker
from ui_qt.main_window import MainWindow


class LoadCancellationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('core.load_cancellation'),
                             'Missing cooperative load cancellation')
        from core import load_cancellation
        return load_cancellation

    def test_cancel_after_read_stops_remaining_files_and_scope_does_not_leak(self):
        api = self.api()
        token = api.LoadCancellation()
        with tempfile.TemporaryDirectory() as folder:
            files = ['a.csv', 'b.csv', 'c.csv']
            for name in files:
                Path(folder, name).write_text('Vbg,Vtg,600,700\n0,0,10,20\n1,1,20,40\n')
            parsed = []
            original = P._parse_raw_spectrum
            def parse(path):
                parsed.append(Path(path).name)
                result = original(path)
                token.cancel()
                return result
            kwargs = dict(bg_mode='self_last', plot_interactive=False, save_png=False,
                          save_dat_file=False, move_original=False)
            with patch.object(P, '_RAW_SPECTRUM_CACHE', RawSpectrumCache(max_bytes=0)), \
                    patch.object(P, '_parse_raw_spectrum', new=parse), api.load_scope(token):
                with self.assertRaises(api.LoadCancelled):
                    P.process_ref_avg(folder, files, **kwargs)
            self.assertEqual(parsed, ['a.csv'])
            result = P.process_ref_avg(folder, files, **kwargs)
            self.assertEqual(result['count'], 3)

    def test_cancelled_worker_finishes_silently_without_publishing_result(self):
        api = self.api()
        token = api.LoadCancellation()
        def work(*, progress, log):
            token.cancel()
            return 'obsolete'
        worker = Worker(work, cancellation=token)
        results, errors, finished = [], [], []
        worker.signals.result.connect(results.append)
        worker.signals.error.connect(errors.append)
        worker.signals.finished.connect(lambda: finished.append(True))
        worker.run()
        self.assertEqual(results, [])
        self.assertEqual(errors, [])
        self.assertEqual(finished, [True])

    def test_cancellation_skips_history_phase_after_numerical_result(self):
        api = self.api()
        token = api.LoadCancellation()
        options = LoadOptions('DRR', '', ['a.csv'], [], False, 'Self (last frame)', 'last', False)
        def numerical(_options, **_signals):
            token.cancel()
            return LoadedState(mode='DRR', folder='', selected_files=['a.csv'])
        owner = SimpleNamespace(_load_task_data=numerical)
        with api.load_scope(token), patch('core.drr_display_settings.find_candidates',
                                         side_effect=AssertionError('Obsolete history queried')):
            with self.assertRaises(api.LoadCancelled):
                MainWindow._load_task(owner, options, progress=None, log=None)

    def test_late_failure_from_cancelled_load_is_not_shown(self):
        api = self.api()
        cancellation = api.LoadCancellation()
        tracker = LatestRequest()
        token, _ = tracker.submit('DRR', 'folder', 'a.csv')
        errors = []
        owner = SimpleNamespace(_active_load_cancel=cancellation, _is_closing=False,
            _load_requests={'DRR': tracker}, current_folder='folder', _invalidated_load_modes=set(),
            _show_error=errors.append)
        cancellation.cancel()
        MainWindow._on_load_error(owner, 'late failure', token)
        self.assertEqual(errors, [])

    def test_cancel_during_history_scan_stops_after_current_record(self):
        api = self.api()
        from core import drr_display_settings as settings
        cancellation = api.LoadCancellation()
        visited = []
        def read(_folder, path, _stamp):
            visited.append(path.name)
            cancellation.cancel()
            return None
        with api.load_scope(cancellation), patch.object(settings, '_legacy_record', new=read):
            with self.assertRaises(api.LoadCancelled):
                settings._parse_legacy_catalog(Path('unused'), [('a.json', ()), ('b.json', ())])
        self.assertEqual(visited, ['a.json'])

    @contextmanager
    def blocked_load(self):
        """Keep a real first-file read active while the GUI changes selection."""
        with tempfile.TemporaryDirectory() as folder:
            with patch('ui_qt.main_window.QSettings', return_value=QSettings(
                    str(Path(folder, 'settings.ini')), QSettings.IniFormat)), \
                    patch.object(MainWindow, '_restore_last_folder', autospec=True), \
                    patch.object(MainWindow, '_schedule_automatic_update_check', autospec=True):
                window = MainWindow()
            started, release = threading.Event(), threading.Event()
            parsed, errors = [], []
            try:
                window.current_folder = folder
                blocked = window.tabs.blockSignals(True)
                window.tabs.setCurrentIndex(next(index for index in range(window.tabs.count())
                                                if window.tabs.tabText(index) == 'DRR'))
                window.tabs.blockSignals(blocked)
                blocked = window.drr_baseline_combo.blockSignals(True)
                window.drr_baseline_combo.setCurrentText('Self (last frame)')
                window.drr_baseline_combo.blockSignals(blocked)
                window.drr_memory_files_chk.setChecked(False)
                window.drr_memory_centers_chk.setChecked(False)
                for name in ['a.csv', 'a2.csv', 'b.csv']:
                    Path(folder, name).write_text(
                        'Vbg,Vtg,600,620,640,660,680,700,720\n'
                        '0,0,10,12,14,16,18,20,22\n1,1,20,24,28,32,36,40,44\n')
                original = P._parse_raw_spectrum
                def parse(path):
                    name = Path(path).name
                    parsed.append(name)
                    if name == 'a.csv':
                        started.set()
                        release.wait(5)
                    return original(path)
                def options(names):
                    window.drr_selected_files = list(names)
                    return LoadOptions('DRR', folder, names, [], False, 'Self (last frame)', 'last', False,
                        drr_assignments=tuple(DrrMeasurementAssignment(name) for name in names),
                        y_axis_spec=window._selected_y_axis_spec('drr'),
                        drr_restore_files=False, drr_restore_centers=False)
                with patch.object(window, '_provenance_records_for_load', return_value=()), \
                        patch.object(P, '_RAW_SPECTRUM_CACHE', RawSpectrumCache(max_bytes=0)), \
                        patch.object(P, '_parse_raw_spectrum', new=parse), \
                        patch.object(window, '_show_error', side_effect=errors.append):
                    window._launch_load_options(options(['a.csv', 'a2.csv']))
                    self.assertTrue(started.wait(3))
                    yield window, release, parsed, errors, options
            finally:
                release.set()
                window.thread_pool.waitForDone(5000)
                window.close()
                window.deleteLater()
                self.app.processEvents()

    def settle_loads(self, window):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            self.app.processEvents()
            if not window._load_in_progress and window._pending_load_mode is None:
                # Include deferred launch callbacks before calling this settled.
                self.app.processEvents()
                if not window._load_in_progress:
                    return
            time.sleep(.01)
        self.fail('Load did not finish')

    def test_clear_cancels_active_and_pending_drr_loads(self):
        for queued in [False, True]:
            with self.subTest(queued=queued), self.blocked_load() as (
                    window, release, parsed, errors, options):
                if queued:
                    window._launch_load_options(options(['b.csv']))
                window.drr_controller._clear_drr_measurements()
                self.assertTrue(window._active_load_cancel.cancelled)
                self.assertIsNone(window._pending_load_options)
                release.set()
                self.settle_loads(window)
                self.assertEqual(parsed, ['a.csv'])
                self.assertEqual(errors, [])
                self.assertEqual(window.drr_selected_files, [])
                self.assertIsNone(window.loaded)
                self.assertIsNone(window._drr_heatmap_ax)
                self.assertIsNone(window._data_state_error)

    def test_invalid_newest_capture_drops_older_pending_request(self):
        with self.blocked_load() as (window, release, parsed, errors, options):
            window._launch_load_options(options(['b.csv']))
            with patch.object(type(window.power_controller), '_power_selected_group_key', return_value=''):
                window._start_load('Power Dependent')
            self.assertIsNone(window._pending_load_options)
            self.assertIsNone(window._pending_load_mode)
            release.set()
            self.settle_loads(window)
            self.assertEqual(parsed, ['a.csv'])
            self.assertEqual(errors, ['Select a power sweep before loading.'])
            self.assertIsNone(window.loaded)

    def test_folder_reset_cancels_the_current_read_and_drops_pending(self):
        for queued in [False, True]:
            with self.subTest(queued=queued), self.blocked_load() as (
                    window, release, parsed, errors, options):
                if queued:
                    window._launch_load_options(options(['b.csv']))
                window._reset_workflow_state_for_folder_change()
                self.assertTrue(window._active_load_cancel.cancelled)
                self.assertIsNone(window._pending_load_options)
                release.set()
                self.settle_loads(window)
                self.assertEqual(parsed, ['a.csv'])
                self.assertEqual(errors, [])
                self.assertIsNone(window.loaded)

    def test_newest_global_load_supersedes_older_mcd_continuations(self):
        for continuation in ['reload', 'reapply']:
            with self.subTest(continuation=continuation), tempfile.TemporaryDirectory() as folder:
                with patch('ui_qt.main_window.QSettings', return_value=QSettings(
                        str(Path(folder, 'settings.ini')), QSettings.IniFormat)), \
                        patch.object(MainWindow, '_restore_last_folder', autospec=True), \
                        patch.object(MainWindow, '_schedule_automatic_update_check', autospec=True):
                    window = MainWindow()
                started, release = threading.Event(), threading.Event()
                ran = []
                try:
                    window.current_folder = folder
                    def options(mode, name):
                        return LoadOptions(mode, folder, [name], [], False,
                            'Self (last frame)', 'last', False,
                            drr_restore_files=False, drr_restore_centers=False)
                    older = options('MCD', 'b.csv')
                    newest = options('DRR', 'c.csv')
                    def work(request, *, progress, log):
                        ran.append((request.mode, request.selected_files[0]))
                        if request.selected_files == ['a.csv']:
                            started.set()
                            release.wait(5)
                        return LoadedState(mode=request.mode, folder=folder,
                            primary_file=request.selected_files[0], selected_files=request.selected_files)
                    def loaded(result, *_args, **_kwargs):
                        window.loaded = result
                        window._active_load_succeeded = True
                    # Represent the reapply timer's eventual new load without
                    # involving independent MCD numerical settings/angle logic.
                    timer = window.mcd_controller._mcd_auto_apply_timer
                    timer.timeout.disconnect()
                    timer.timeout.connect(lambda: window._launch_load_options(older))
                    with patch.object(window, '_load_task', new=work), \
                            patch.object(window, '_on_loaded', new=loaded), \
                            patch.object(type(window.mcd_controller), '_request_mcd_load',
                                         new=lambda _controller: window._launch_load_options(older)):
                        window._launch_load_options(options('MCD', 'a.csv'))
                        self.assertTrue(started.wait(3))
                        if continuation == 'reload':
                            window._mcd_reload_pending = True
                        else:
                            window.mcd_controller._mcd_reapply_pending = True
                        window._launch_load_options(newest)
                        release.set()
                        self.settle_loads(window)
                        self.assertEqual(ran, [('MCD', 'a.csv'), ('DRR', 'c.csv')])
                        self.assertEqual(window.loaded.primary_file, 'c.csv')
                        self.assertFalse(window._mcd_reload_pending)
                        self.assertFalse(window.mcd_controller._mcd_reapply_pending)
                finally:
                    release.set()
                    window.thread_pool.waitForDone(5000)
                    window.close()
                    window.deleteLater()
                    self.app.processEvents()

    def test_newest_pending_load_runs_after_cancel_and_keeps_manual_fix(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('ui_qt.main_window.QSettings', return_value=QSettings(
                    str(Path(folder, 'settings.ini')), QSettings.IniFormat)), \
                    patch.object(MainWindow, '_restore_last_folder', autospec=True), \
                    patch.object(MainWindow, '_schedule_automatic_update_check', autospec=True):
                window = MainWindow()
            started, release = threading.Event(), threading.Event()
            try:
                window.current_folder = folder
                blocked = window.tabs.blockSignals(True)
                window.tabs.setCurrentIndex(next(index for index in range(window.tabs.count())
                                                if window.tabs.tabText(index) == 'DRR'))
                window.tabs.blockSignals(blocked)
                blocked = window.drr_baseline_combo.blockSignals(True)
                window.drr_baseline_combo.setCurrentText('Self (last frame)')
                window.drr_baseline_combo.blockSignals(blocked)
                window.drr_memory_files_chk.setChecked(False)
                window.drr_memory_centers_chk.setChecked(False)
                window.drr_spins['ymin'].setValue(-.4)
                window.drr_fix_checks['ymin'].setChecked(True)
                files = ['a.csv', 'a2.csv', 'b.csv', 'c.csv']
                for name in files:
                    Path(folder, name).write_text(
                        'Vbg,Vtg,600,620,640,660,680,700,720\n'
                        '0,0,10,12,14,16,18,20,22\n1,1,20,24,28,32,36,40,44\n')
                parsed = []
                errors = []
                original = P._parse_raw_spectrum
                def parse(path):
                    name = Path(path).name
                    parsed.append(name)
                    if name == 'a.csv':
                        started.set()
                        release.wait(5)
                    return original(path)
                def options(names):
                    window.drr_selected_files = list(names)
                    return LoadOptions('DRR', folder, names, [], False, 'Self (last frame)', 'last', False,
                        drr_assignments=tuple(DrrMeasurementAssignment(name) for name in names),
                        y_axis_spec=window._selected_y_axis_spec('drr'),
                        drr_restore_files=False, drr_restore_centers=False)
                with patch.object(window, '_provenance_records_for_load', return_value=()), \
                        patch.object(P, '_RAW_SPECTRUM_CACHE', RawSpectrumCache(max_bytes=0)), \
                        patch.object(P, '_parse_raw_spectrum', new=parse), \
                        patch.object(window, '_show_error', side_effect=errors.append):
                    window._launch_load_options(options(['a.csv', 'a2.csv']))
                    self.assertTrue(started.wait(3))
                    window._launch_load_options(options(['b.csv']))
                    window._launch_load_options(options(['c.csv']))
                    release.set()
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline:
                        self.app.processEvents()
                        if errors:
                            break
                        if (window.loaded and window.loaded.primary_file == 'c.csv'
                                and not window._load_in_progress and window._drr_heatmap_ax is not None):
                            break
                        time.sleep(.01)
                    self.assertEqual(parsed, ['a.csv', 'c.csv'])
                    self.assertEqual(errors, [], 'Load or cancellation was shown as an error')
                    self.assertEqual(window.loaded.primary_file, 'c.csv')
                    self.assertIsNotNone(window._drr_heatmap_ax)
                    self.assertIsNone(window._data_state_error)
                    self.assertEqual(window.drr_spins['ymin'].value(), -.4)
                    self.assertTrue(window.drr_fix_checks['ymin'].isChecked())
            finally:
                release.set()
                window.thread_pool.waitForDone(5000)
                window.close()
                window.deleteLater()
                self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
