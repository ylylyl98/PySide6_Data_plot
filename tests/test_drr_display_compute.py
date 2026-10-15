import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import time
import threading
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import QObject, QThreadPool, QSettings
from PySide6.QtWidgets import QApplication
from core.loader import DataCube
from core.processing import apply_sg_derivative_energy


class DisplayComputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_worker_result_and_stale_source(self):
        from ui_qt.drr_display_compute import DrrDisplayCompute
        x = np.linspace(1, 2, 101)
        cube = DataCube(x, np.array([0.]), np.array([x*x]), 'Gate', 'Data', 'DR/R')
        owner = QObject()
        owner.loaded = SimpleNamespace(cube=cube)
        owner.thread_pool = QThreadPool(owner)
        owner._drr_derivative_cache = {}
        coordinator = DrrDisplayCompute(owner)
        self.assertFalse(coordinator.request(cube, 2, 9, 2))
        self.assertTrue(owner.thread_pool.waitForDone(5000))
        self.app.processEvents()
        self.assertTrue(coordinator.request(cube, 2, 9, 2))
        expected, _ = apply_sg_derivative_energy(cube, derivative=2, window_length=9, polyorder=2)
        np.testing.assert_array_equal(owner._drr_derivative_cache[(id(cube),2,9,2)][0].Z, expected.Z)
        owner._drr_derivative_cache.clear()
        self.assertFalse(coordinator.request(cube, 2, 11, 2))
        owner.loaded = SimpleNamespace(cube=object())
        owner.thread_pool.waitForDone(5000);self.app.processEvents()
        self.assertEqual(owner._drr_derivative_cache, {})

    def test_changed_request_drops_previous_result(self):
        from ui_qt.drr_display_compute import DrrDisplayCompute
        x = np.linspace(1, 2, 101)
        cube = DataCube(x, np.array([0.]), np.array([x*x]), 'Gate', 'Data', 'DR/R')
        owner = QObject(); owner.loaded = SimpleNamespace(cube=cube)
        owner.thread_pool = QThreadPool(owner);owner._drr_derivative_cache = {}
        coordinator = DrrDisplayCompute(owner)
        coordinator.request(cube,2,9,2)
        coordinator.request(cube,2,11,2)
        owner.thread_pool.waitForDone(5000);self.app.processEvents()
        self.assertNotIn((id(cube),2,9,2),owner._drr_derivative_cache)

    def test_obsolete_failure_allows_newest_request_to_resume(self):
        from ui_qt.drr_display_compute import DrrDisplayCompute
        cube = object()
        owner = QObject();owner.loaded = SimpleNamespace(cube=cube)
        owner.thread_pool = QThreadPool(owner);owner._drr_derivative_cache = {}
        coordinator = DrrDisplayCompute(owner)
        ready = [];errors = []
        coordinator.ready.connect(lambda: ready.append(True))
        coordinator.failed.connect(errors.append)
        with patch('ui_qt.drr_display_compute.compute', side_effect=ValueError('old request failed')):
            coordinator.request(cube,2,9,2)
            coordinator.request(cube,2,11,2)
            owner.thread_pool.waitForDone(5000);self.app.processEvents()
        self.assertEqual(ready, [True])
        self.assertEqual(errors, [])

    def test_cached_display_does_not_cancel_pending_second_split(self):
        from ui_qt.drr_display_compute import DrrDisplayCompute
        x = np.linspace(1, 2, 101)
        cube = DataCube(x, np.array([0.]), np.array([x*x]), 'Gate', 'Data', 'DR/R')
        owner = QObject();owner.loaded = SimpleNamespace(cube=cube)
        owner.thread_pool = QThreadPool(owner)
        owner._drr_derivative_cache = {(id(cube),1,9,2): (cube,9)}
        coordinator = DrrDisplayCompute(owner)
        coordinator.request(cube,2,9,2)
        self.assertTrue(coordinator.request(cube,1,9,2))
        owner.thread_pool.waitForDone(5000);self.app.processEvents()
        self.assertIn((id(cube),2,9,2),owner._drr_derivative_cache)

    def test_first_and_second_cache_misses_both_make_progress(self):
        from ui_qt.drr_display_compute import DrrDisplayCompute
        x = np.linspace(1, 2, 101)
        cube = DataCube(x, np.array([0.]), np.array([x*x]), 'Gate', 'Data', 'DR/R')
        owner = QObject();owner.loaded = SimpleNamespace(cube=cube)
        owner.thread_pool = QThreadPool(owner);owner._drr_derivative_cache = {}
        coordinator = DrrDisplayCompute(owner)
        coordinator.request(cube,1,9,2)
        coordinator.request(cube,2,9,2)
        owner.thread_pool.waitForDone(5000);self.app.processEvents()
        self.assertIn((id(cube),1,9,2),owner._drr_derivative_cache)
        coordinator.request(cube,2,9,2)
        owner.thread_pool.waitForDone(5000);self.app.processEvents()
        self.assertIn((id(cube),2,9,2),owner._drr_derivative_cache)

    def test_current_oversized_pair_remains_ready_without_repeated_workers(self):
        from core.drr_derivative_cache import DrrDerivativeCache
        from ui_qt.drr_display_compute import DrrDisplayCompute
        x = np.linspace(1, 2, 101)
        cube = DataCube(x, np.array([0.]), np.array([x*x]), 'Gate', 'Data', 'DR/R')
        owner = QObject(); owner.loaded = SimpleNamespace(cube=cube)
        owner.thread_pool = QThreadPool(owner)
        owner._drr_derivative_cache = DrrDerivativeCache(max_bytes=8)
        coordinator = DrrDisplayCompute(owner)
        for derivative in (1, 2):
            self.assertFalse(coordinator.request(cube, derivative, 9, 2))
            self.assertTrue(owner.thread_pool.waitForDone(5000))
            self.app.processEvents()
        for derivative in (1, 2):
            self.assertTrue(coordinator.request(cube, derivative, 9, 2))
            expected, _ = apply_sg_derivative_energy(cube, derivative=derivative, window_length=9, polyorder=2)
            np.testing.assert_array_equal(owner._drr_derivative_cache[(id(cube), derivative, 9, 2)][0].Z,
                                          expected.Z)
        self.assertIsNone(coordinator.worker)
        self.assertEqual(owner._drr_derivative_cache.history_bytes, 0)

    def test_superseded_sg_stops_after_current_batch_and_the_latest_result_is_ready(self):
        from core import processing_run as P
        from ui_qt.drr_display_compute import DrrDisplayCompute
        x = np.linspace(1, 2, 41) ** 1.1
        cube = DataCube(x, np.arange(70.), np.broadcast_to(x*x, (70, 41)).copy(), 'Gate', 'Data', 'DR/R')
        owner = QObject(); owner.loaded = SimpleNamespace(cube=cube)
        owner.thread_pool = QThreadPool(owner); owner._drr_derivative_cache = {}
        coordinator = DrrDisplayCompute(owner)
        started, release = threading.Event(), threading.Event()
        visited, errors = [], []
        coordinator.failed.connect(errors.append)
        coordinator.ready.connect(lambda: coordinator.request(cube, 2, 11, 2))
        original = P.savgol_filter
        def batch(*args, **kwargs):
            window = kwargs['window_length']
            visited.append(window)
            if window == 9 and len(visited) == 1:
                started.set()
                release.wait(3)
            return original(*args, **kwargs)
        try:
            with patch.object(P, 'savgol_filter', new=batch):
                coordinator.request(cube, 2, 9, 2)
                self.assertTrue(started.wait(2))
                coordinator.request(cube, 2, 11, 2)
                release.set()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    self.app.processEvents()
                    if (id(cube), 2, 11, 2) in owner._drr_derivative_cache and coordinator.worker is None:
                        break
                    time.sleep(.005)
                self.assertEqual(visited, [9] + [11] * 3)
                self.assertEqual(errors, [])
                self.assertIn((id(cube), 2, 11, 2), owner._drr_derivative_cache)
        finally:
            release.set()
            owner.thread_pool.waitForDone(5000)
            self.app.processEvents()

    def test_clear_cancels_current_derivative_without_republishing_or_errors(self):
        from core import processing_run as P
        from ui_qt.drr_display_compute import DrrDisplayCompute
        x = np.linspace(1, 2, 41) ** 1.1
        cube = DataCube(x, np.arange(7.), np.broadcast_to(x*x, (7, 41)).copy(), 'Gate', 'Data', 'DR/R')
        owner = QObject(); owner.loaded = SimpleNamespace(cube=cube)
        owner.thread_pool = QThreadPool(owner); owner._drr_derivative_cache = {}
        coordinator = DrrDisplayCompute(owner)
        started, release = threading.Event(), threading.Event()
        visited, errors, ready = [], [], []
        coordinator.failed.connect(errors.append)
        coordinator.ready.connect(lambda: ready.append(True))
        original = P.savgol_filter
        def batch(*args, **kwargs):
            visited.append(True)
            started.set()
            release.wait(3)
            return original(*args, **kwargs)
        try:
            with patch.object(P, 'savgol_filter', new=batch):
                coordinator.request(cube, 2, 9, 2)
                self.assertTrue(started.wait(2))
                clear = getattr(coordinator, 'clear', lambda: None)
                clear()
                release.set()
                owner.thread_pool.waitForDone(5000); self.app.processEvents()
                self.assertEqual(len(visited), 1)
                self.assertEqual(owner._drr_derivative_cache, {})
                self.assertEqual(errors, [])
                self.assertEqual(ready, [])
        finally:
            release.set()
            owner.thread_pool.waitForDone(5000)
            self.app.processEvents()

    def test_folder_reset_cancels_inflight_display_derivative(self):
        from core import processing_run as P
        from ui_qt.common import LoadedState
        from ui_qt.main_window import MainWindow
        from ui_qt.drr_display_compute import DrrDisplayCompute
        with tempfile.TemporaryDirectory() as folder:
            with patch('ui_qt.main_window.QSettings', return_value=QSettings(
                    str(Path(folder, 'test.ini')), QSettings.IniFormat)), \
                    patch.object(MainWindow, '_restore_last_folder', new=lambda _self: None), \
                    patch.object(MainWindow, '_schedule_automatic_update_check', new=lambda _self: None):
                window = MainWindow()
            started, release = threading.Event(), threading.Event()
            visited = []
            x = np.linspace(1, 2, 41) ** 1.1
            cube = DataCube(x, np.arange(7.), np.broadcast_to(x*x, (7, 41)).copy(), 'Gate', 'Data', 'DR/R')
            window.loaded = LoadedState(mode='DRR', folder=folder, cube=cube)
            window._drr_display_compute = DrrDisplayCompute(window)
            original = P.savgol_filter
            def batch(*args, **kwargs):
                visited.append(True)
                started.set()
                release.wait(3)
                return original(*args, **kwargs)
            try:
                with patch.object(P, 'savgol_filter', new=batch):
                    window._drr_display_compute.request(cube, 2, 9, 2)
                    self.assertTrue(started.wait(2))
                    window._reset_workflow_state_for_folder_change()
                    release.set()
                    window.thread_pool.waitForDone(5000); self.app.processEvents()
                    self.assertEqual(len(visited), 1)
                    self.assertIsNone(window.loaded)
                    self.assertEqual(len(window._drr_derivative_cache), 0)
            finally:
                release.set()
                window.thread_pool.waitForDone(5000)
                window.close(); window.deleteLater(); self.app.processEvents()

    def test_hidden_second_product_is_cancelled_on_parameter_edits_even_if_validation_fails(self):
        from core import processing_run as P
        from ui_qt.common import LoadedState
        from ui_qt.main_window import MainWindow
        for setting in ['window', 'poly', 'auto_window', 'invalid_mixed_kind']:
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as folder:
                with patch('ui_qt.main_window.QSettings', return_value=QSettings(
                        str(Path(folder, 'test.ini')), QSettings.IniFormat)), \
                        patch.object(MainWindow, '_restore_last_folder', new=lambda _self: None), \
                        patch.object(MainWindow, '_schedule_automatic_update_check', new=lambda _self: None):
                    window = MainWindow()
                started, release = threading.Event(), threading.Event()
                visited = []
                x = np.linspace(1, 2, 41) ** 1.1
                cube = DataCube(x, np.zeros(7), np.broadcast_to(x*x, (7, 41)).copy(),
                                'Gate', 'Data', 'DR/R')
                window.drr_sg_auto_chk.setChecked(False)
                window.drr_sg_window_spin.setValue(9)
                window.loaded = LoadedState(mode='DRR', folder=folder, cube=cube)
                original = P.savgol_filter
                def batch(*args, **kwargs):
                    visited.append(True)
                    started.set()
                    release.wait(3)
                    return original(*args, **kwargs)
                try:
                    with patch.object(P, 'savgol_filter', new=batch):
                        self.assertFalse(window._prepare_drr_display_derivative(2))
                        self.assertTrue(started.wait(2))
                        cancellation = window._drr_display_compute.worker.cancellation
                        if setting == 'window':
                            window.drr_sg_window_spin.setValue(11)
                        elif setting == 'poly':
                            window.drr_sg_poly_spin.setValue(3)
                        elif setting == 'auto_window':
                            window.drr_sg_auto_chk.setChecked(True)
                        else:
                            # Constant Y makes mixed validation return before
                            # any replacement derivative request is launched.
                            window.drr_second_kind_combo.setCurrentIndex(1)
                        self.assertTrue(cancellation.cancelled)
                        release.set()
                        window.thread_pool.waitForDone(5000); self.app.processEvents()
                        self.assertEqual(len(visited), 1)
                        self.assertNotIn((id(cube), 2, 9, 2), window._drr_derivative_cache)
                finally:
                    release.set()
                    window.thread_pool.waitForDone(5000)
                    window.close(); window.deleteLater(); self.app.processEvents()

    def test_hidden_mixed_product_is_cancelled_when_auto_x_changes_its_effective_window(self):
        from ui_qt.common import LoadedState
        from ui_qt.main_window import MainWindow
        with tempfile.TemporaryDirectory() as folder:
            with patch('ui_qt.main_window.QSettings', return_value=QSettings(
                    str(Path(folder, 'test.ini')), QSettings.IniFormat)), \
                    patch.object(MainWindow, '_restore_last_folder', new=lambda _self: None), \
                    patch.object(MainWindow, '_schedule_automatic_update_check', new=lambda _self: None):
                window = MainWindow()
            started, release = threading.Event(), threading.Event()
            visited = []
            x, y = np.linspace(1, 2, 41), np.linspace(-1, 1, 7)
            cube = DataCube(x, y, x[None, :] ** 2 + y[:, None], 'Gate', 'Data', 'DR/R')
            window.drr_sg_auto_chk.setChecked(False)
            window.drr_sg_window_spin.setValue(9)
            window.drr_second_kind_combo.setCurrentIndex(1)
            window.drr_sg_y_auto_chk.setChecked(False)
            window.drr_sg_y_window_spin.setValue(5)
            window.loaded = LoadedState(mode='DRR', folder=folder, cube=cube)
            original = np.linalg.pinv
            def fit(*args, **kwargs):
                visited.append(True)
                started.set()
                release.wait(3)
                return original(*args, **kwargs)
            try:
                with patch('core.drr_mixed_derivative.np.linalg.pinv', new=fit):
                    self.assertFalse(window._prepare_drr_display_derivative(2))
                    self.assertTrue(started.wait(2))
                    cancellation = window._drr_display_compute.worker.cancellation
                    window.drr_sg_auto_chk.setChecked(True)
                    self.assertTrue(cancellation.cancelled)
                    release.set()
                    window.thread_pool.waitForDone(5000); self.app.processEvents()
                    self.assertEqual(len(visited), 1)
                    self.assertNotIn((id(cube), ('mixed', 5), 9, 2), window._drr_derivative_cache)
            finally:
                release.set()
                window.thread_pool.waitForDone(5000)
                window.close(); window.deleteLater(); self.app.processEvents()
