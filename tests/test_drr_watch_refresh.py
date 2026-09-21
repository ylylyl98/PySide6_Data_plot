import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialog
from ui_qt.main_window import MainWindow


class WatchRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / 'Initial Data').mkdir()
        (self.root / 'MCD').mkdir()
        settings = QSettings(str(self.root / 'settings.ini'), QSettings.IniFormat)
        with patch('ui_qt.main_window.QSettings', return_value=settings), patch.object(
                MainWindow, '_restore_last_folder'), patch.object(MainWindow, '_schedule_automatic_update_check'):
            self.w = MainWindow()
        self.addCleanup(self.w.close)
        self.w.current_folder = str(self.root)

    def test_same_folder_preserves_watcher_registrations(self):
        with patch.object(self.w.thread_pool, 'start'):
            self.w._watch_current_folder()
            with patch.object(self.w.folder_watcher, 'removePaths', wraps=self.w.folder_watcher.removePaths) as remove:
                self.w._watch_current_folder()
                remove.assert_not_called()

    def test_mcd_event_does_not_invalidate_drr_catalog(self):
        self.w._catalog_ready_modes = {'DRR', 'MCD', 'PL'}
        with patch.object(self.w, '_active_mode', return_value='DRR'), patch.object(
                self.w, '_refresh_file_lists') as refresh, patch.object(self.w, '_watch_current_folder'):
            self.w._on_watched_folder_changed(str(self.root / 'MCD'))
            self.w.folder_refresh_timer.stop()
            self.w._refresh_watched_folder()
        self.assertEqual(self.w._catalog_ready_modes, {'DRR', 'PL'})
        refresh.assert_not_called()

    def test_new_history_directory_gets_watch_when_capacity_is_full(self):
        w = self.w
        initial = self.root / 'Initial Data'
        watched = {str(self.root), str(initial), str(self.root / 'MCD')}
        watched.update(str(initial / str(i)) for i in range(253))
        history = self.root / 'Processed Data/DRR'
        history.mkdir(parents=True)
        w._watched_folder = str(self.root)
        with patch.object(w.thread_pool, 'start'), patch.object(w.folder_watcher, 'directories', side_effect=lambda:list(watched)), patch.object(
                w.folder_watcher, 'removePaths', side_effect=lambda paths:watched.difference_update(paths)), patch.object(
                w.folder_watcher, 'addPaths', side_effect=lambda paths:watched.update(paths)):
            w._watch_current_folder()
        self.assertIn(str(history), watched)
        self.assertIn(str(history.parent), watched)
        self.assertLessEqual(len(watched), 256)

    def test_pending_manual_refresh_keeps_full_behavior_after_picker_validation(self):
        w = self.w
        w._drr_refresh_running = True
        w._queue_drr_catalog_refresh(auto=True, old_source_files=set(), catalog_only=True)
        w._queue_drr_catalog_refresh(auto=False, old_source_files=set())
        w._drr_refresh_running = False
        with patch.object(w, '_start_drr_catalog_refresh') as start:
            w._finish_drr_catalog_refresh()
        self.assertFalse(start.call_args.kwargs.get('catalog_only', False))

    def test_picker_open_requests_async_validation_without_force(self):
        def execute(dialog):
            from PySide6.QtTest import QTest
            QTest.qWait(200)
            return QDialog.Rejected
        with patch.object(self.w, '_queue_drr_catalog_refresh') as refresh, patch.object(QDialog, 'exec', execute):
            self.w.drr_controller._open_drr_source_dialog(title='test', selected=[], baseline_mode=False)
        refresh.assert_called_once()
        self.assertFalse(refresh.call_args.kwargs.get('force', False))
        self.assertTrue(refresh.call_args.kwargs['catalog_only'])

    def test_recent_picker_does_not_refresh_until_event_or_expiry(self):
        from ui_qt.drr_picker_cache import freshness
        from PySide6.QtTest import QTest
        fresh = freshness(self.w)
        fresh.clock = lambda: 100.
        fresh.complete(fresh.begin(str(self.root), False))
        def execute(dialog):
            QTest.qWait(200)
            return QDialog.Rejected
        def open_picker():
            self.w.drr_controller._open_drr_source_dialog(title='test', selected=[], baseline_mode=False)
        with patch.object(self.w, '_queue_drr_catalog_refresh') as refresh, patch.object(QDialog, 'exec', execute):
            open_picker()
            refresh.assert_not_called()
            self.w._on_watched_folder_changed(str(self.root / 'Initial Data'))
            self.w.folder_refresh_timer.stop()
            open_picker()
            refresh.assert_called_once()
            fresh.complete(fresh.begin(str(self.root), False))
            fresh.clock = lambda: 131.
            open_picker()
            self.assertEqual(refresh.call_count, 2)

    def test_manual_refresh_bypasses_recent_lease(self):
        from ui_qt.drr_picker_cache import freshness
        from PySide6.QtWidgets import QPushButton
        fresh = freshness(self.w)
        fresh.complete(fresh.begin(str(self.root), False))
        def execute(dialog):
            next(b for b in dialog.findChildren(QPushButton) if b.text() == 'Refresh').click()
            return QDialog.Rejected
        with patch.object(self.w, '_queue_drr_catalog_refresh') as refresh, patch.object(QDialog, 'exec', execute):
            self.w.drr_controller._open_drr_source_dialog(title='test', selected=[], baseline_mode=False)
        refresh.assert_called_once()
        self.assertFalse(refresh.call_args.kwargs.get('catalog_only', False))

    def test_picker_validation_preserves_selection_and_does_not_load(self):
        from core.drr_sources import DrrSource, DrrSourceCache
        from ui_qt.controllers_drr import DrrController
        w = self.w
        old = DrrSource('sample_1.csv', 'sample_1.csv', 'g', '', 1., False)
        new = DrrSource('sample_2.csv', 'sample_2.csv', 'g', '', 2., False)
        w.drr_available_sources = [old]
        w.drr_selected_files = [old.source]
        with patch.object(DrrController, '_refresh_auto_external') as match, patch.object(w, '_start_load') as load:
            w._on_drr_catalog_refresh_result((str(self.root), [old, new], DrrSourceCache(load_on_init=False)),
                                            w._drr_refresh_generation, True, {old.source}, catalog_only=True)
        self.assertEqual(w.drr_selected_files, [old.source])
        self.assertEqual(w.drr_available_sources, [old, new])
        match.assert_not_called()
        load.assert_not_called()

    def test_new_unwatched_file_is_published_even_after_picker_cancel(self):
        from core.drr_sources import discover_drr_sources
        first = self.root / 'Initial Data/sample_REF_760nm_1.csv'
        first.write_text('Vbg,740,760,780\n0,10,11,12\n1,12,13,14\n')
        w = self.w
        w.drr_available_sources = discover_drr_sources(self.root)
        w.drr_selected_files = [w.drr_available_sources[0].source]
        selected = list(w.drr_selected_files)
        second = first.with_name('sample_REF_760nm_2.csv')
        second.write_text(first.read_text())
        jobs = []
        def execute(dialog):
            from PySide6.QtTest import QTest
            # Other queued UI work can delay the 150 ms timer in a full suite.
            # Wait for the observable dispatch, with a bounded timeout.
            from PySide6.QtCore import QElapsedTimer
            timer = QElapsedTimer()
            timer.start()
            while not jobs and timer.elapsed() < 2000:
                QTest.qWait(20)
            return QDialog.Rejected
        with patch.object(w.thread_pool, 'start', jobs.append), patch.object(QDialog, 'exec', execute), patch.object(w, '_start_load') as load:
            w.drr_controller._open_drr_source_dialog(title='test', selected=selected, baseline_mode=False)
            self.assertEqual(len(jobs), 1)
            jobs[0].run()
            self.app.processEvents()
            load.assert_not_called()
        self.assertEqual(w.drr_selected_files, selected)
        self.assertTrue(any(s.filename == second.name for s in w.drr_available_sources))
