from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QDialog, QListWidget, QPushButton, QComboBox
from core.drr_sources import DrrSource, DrrSourceCache
from ui_qt.main_window import MainWindow


class DrrCachedPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_open_picker_receives_preview_without_finishing_refresh(self):
        with patch.object(MainWindow, '_restore_last_folder', lambda self: None):
            window = MainWindow()
        self.addCleanup(window.close)
        with tempfile.TemporaryDirectory() as folder:
            window.current_folder = folder
            window._drr_refresh_generation = 1
            window._drr_refresh_running = True
            source = DrrSource('sample_REF_760nmc.csv', 'sample_REF_760nmc.csv', 'g', '2026-09-12', 1.0, False,
                               gate_varies=True, frame_count=2, wavelength_center_nm=760.0)
            Path(folder, source.source).write_text('Gate,740,760\n0,1,2\n1,3,4\n')
            observed = {}
            def execute(dialog):
                groups = dialog.findChild(QListWidget, 'drr_source_group_list')
                observed['before'] = groups.count()
                window._on_drr_catalog_refresh_result((folder, [source], DrrSourceCache(load_on_init=False), True), 1, True, set())
                observed['after'] = groups.count()
                observed['running'] = window._drr_refresh_running
                observed['refreshing'] = any(b.text() == 'Refreshing...' for b in dialog.findChildren(QPushButton))
                return QDialog.Rejected
            with patch.object(QDialog, 'exec', execute):
                window.drr_controller._open_drr_source_dialog(title='Choose', selected=[], baseline_mode=False)
            self.assertEqual(observed['before'], 0)
            self.assertEqual(observed['after'], 1)
            self.assertTrue(observed['running'])
            self.assertTrue(observed['refreshing'])

    def test_all_data_switch_requests_expanded_catalog(self):
        with patch.object(MainWindow, '_restore_last_folder', lambda self: None):
            window = MainWindow()
        self.addCleanup(window.close)
        with tempfile.TemporaryDirectory() as folder:
            window.current_folder = folder
            def execute(dialog):
                combo = next(c for c in dialog.findChildren(QComboBox) if c.findText('All data') >= 0)
                combo.setCurrentIndex(combo.findText('All data'))
                return QDialog.Rejected
            with patch.object(window, '_queue_drr_catalog_refresh') as queue, patch.object(QDialog, 'exec', execute):
                window.drr_controller._open_drr_source_dialog(title='Choose', selected=[], baseline_mode=False)
                self.assertTrue(getattr(window, '_drr_include_all_sources', False))
                queue.assert_called_once()

    def test_old_scope_result_does_not_suppress_all_data_preview(self):
        with patch.object(MainWindow, '_restore_last_folder', lambda self: None):
            window = MainWindow()
        self.addCleanup(window.close)
        with tempfile.TemporaryDirectory() as folder:
            window.current_folder = folder
            window._drr_refresh_generation = 1
            window._drr_refresh_running = True
            window._drr_include_all_sources = True
            source = DrrSource('old_REF.csv', 'old_REF.csv', 'g', 'date', 1.0, False)
            cache = DrrSourceCache(load_on_init=False)
            window._on_drr_catalog_refresh_result((folder, [source], cache, True, False), 1, True, set())
            self.assertNotIn('DRR', window._catalog_displayed_modes)
            self.assertTrue(window._drr_refresh_running)
            with patch.object(window, '_finish_drr_catalog_refresh') as finish:
                window._on_drr_catalog_refresh_result((folder, [source], cache, False, False), 1, True, set())
            finish.assert_called_once()
            self.assertEqual(window.drr_available_sources, [])


if __name__ == '__main__':
    unittest.main()
