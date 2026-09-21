import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import QObject, QThreadPool
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
