from __future__ import annotations

import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QThreadPool, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QTableWidget, QWidget

from core import mcd
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope
from ui_qt.controllers_mcd import McdController, _McdBackgroundSuggestions
from ui_qt.mcd_unified_page import McdUnifiedView


def source(rows=5, columns=64):
    energy = np.linspace(1.5, 1.8, columns)
    spectra = np.tile(100 + 3 * np.sin(energy * 3), (rows, 1))
    return SimpleNamespace(
        wavelength_nm=mcd.EV_NM / energy, energy_ev=energy,
        pair_corrected_pos=spectra, pair_raw_neg=spectra * 1.01,
        dark_neg=np.zeros(columns), gain=np.ones(columns),
        pair_mcd_corrected=np.arange(rows * columns, dtype=float).reshape(rows, columns),
        pair_b=np.arange(rows, dtype=float), pair_labels=np.full(rows, 'B increasing'),
    )


class CapturedPool:
    def __init__(self):
        self.workers = []

    def start(self, worker):
        self.workers.append(worker)


class McdBackgroundLatencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.owner = QWidget()

    def tearDown(self):
        self.owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.app.processEvents()

    def test_protection_typing_commits_once_and_waits_for_latest_result(self):
        result = source()
        suggestion = mcd.suggest_mcd_background_ranges(
            result, protected_ranges_ev=((1.64, 1.66),),
            auto_detect_features=False, use_all_unprotected_bands=True,
        )
        controller = SimpleNamespace(
            _owner=self.owner, windowIcon=self.owner.windowIcon,
            _format_mcd_background_ranges=McdController._format_mcd_background_ranges,
        )
        pool = CapturedPool()

        def edit(dialog):
            spin = dialog.findChild(QTableWidget).cellWidget(0, 3)
            runner = dialog.findChild(_McdBackgroundSuggestions)
            buttons = dialog.findChild(QDialogButtonBox)
            accept = next(button for button in buttons.buttons()
                          if buttons.buttonRole(button) == QDialogButtonBox.AcceptRole)
            self.assertFalse(spin.keyboardTracking())
            spin.lineEdit().selectAll()
            QTest.keyClicks(spin.lineEdit(), '1.654321')
            self.assertIsNone(runner.pending)
            calculate.assert_not_called()
            QTest.keyClick(spin.lineEdit(), Qt.Key_Return)
            self.assertIsNotNone(runner.pending)
            self.assertFalse(accept.isEnabled())
            self.assertEqual(pool.workers, [])
            runner.timer.stop()
            runner._launch()
            pool.workers[0].run()
            calculate.assert_called_once()
            self.assertEqual(calculate.call_args.kwargs['protected_ranges_ev'], ((1.654321, 1.66),))
            self.assertTrue(accept.isEnabled())
            spin.setValue(1.655)
            runner.timer.stop()
            runner._launch()
            calculate.side_effect = ValueError('invalid protection')
            pool.workers[1].run()
            self.assertFalse(accept.isEnabled())
            dialog.reject()
            return QDialog.Rejected

        with patch('ui_qt.controllers_mcd.suggest_mcd_background_ranges', return_value=suggestion) as calculate, \
                patch('ui_qt.controllers_mcd.QThreadPool.globalInstance', return_value=pool), \
                patch.object(QDialog, 'exec', new=edit):
            McdController._show_mcd_background_suggestion_dialog(controller, result, suggestion)

    def test_edits_coalesce_and_obsolete_results_and_errors_are_ignored(self):
        pool = CapturedPool()
        runner = _McdBackgroundSuggestions(source(), self.owner, pool=pool)
        completed, errors = [], []
        runner.completed.connect(completed.append)
        runner.failed.connect(errors.append)
        runner.request(((1.60, 1.61),))
        runner.request(((1.62, 1.63),))
        runner.timer.stop()
        runner._launch()
        first = pool.workers[0]
        self.assertEqual(first.args[2], ((1.62, 1.63),))
        runner.request(((1.64, 1.65),))
        runner.request(((1.66, 1.67),))
        runner.timer.stop()
        self.assertTrue(first.cancellation.cancelled)
        first.signals.result.emit((first.args[1], 'obsolete'))
        first.signals.error.emit('obsolete error')
        first.signals.finished.emit()
        self.assertEqual(len(pool.workers), 2)
        self.assertEqual(pool.workers[1].args[2], ((1.66, 1.67),))
        with patch('ui_qt.controllers_mcd.suggest_mcd_background_ranges', return_value='latest'):
            pool.workers[1].run()
        self.assertEqual(completed, ['latest'])
        self.assertEqual(errors, [])
        runner.close()

    def test_close_cancels_pending_and_active_requests(self):
        pool = CapturedPool()
        runner = _McdBackgroundSuggestions(source(), self.owner, pool=pool)
        completed, errors = [], []
        runner.completed.connect(completed.append)
        runner.failed.connect(errors.append)
        runner.request(((1.64, 1.66),))
        runner.timer.stop()
        runner._launch()
        runner.request(((1.64, 1.67),))
        runner.close()
        worker = pool.workers[0]
        self.assertTrue(worker.cancellation.cancelled)
        worker.signals.result.emit((runner.generation, 'late'))
        worker.signals.error.emit('late error')
        worker.signals.finished.emit()
        self.assertEqual(completed, [])
        self.assertEqual(errors, [])
        self.assertIsNone(runner.pending)
        self.assertEqual(len(pool.workers), 1)

    def test_destroying_dialog_during_real_worker_is_safe(self):
        pool = QThreadPool()
        dialog = QWidget(self.owner)
        runner = _McdBackgroundSuggestions(source(), dialog, pool=pool)
        entered, release = threading.Event(), threading.Event()
        threads, completed = [], []
        runner.completed.connect(completed.append)

        def calculate(*args, **kwargs):
            threads.append(threading.get_ident())
            entered.set()
            if not release.wait(3):
                raise TimeoutError('test did not release worker')
            return 'late'

        try:
            with patch('ui_qt.controllers_mcd.suggest_mcd_background_ranges', side_effect=calculate):
                runner.request(((1.64, 1.66),))
                runner.timer.stop()
                runner._launch()
                self.assertTrue(entered.wait(2))
                dialog.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                release.set()
                self.assertTrue(pool.waitForDone(3000))
                self.app.processEvents()
        finally:
            release.set()
            pool.waitForDone(3000)
        self.assertNotEqual(threads, [threading.get_ident()])
        self.assertEqual(completed, [])

    def test_initial_suggestion_is_dispatched_and_rejects_changed_source(self):
        result = source()
        pool = CapturedPool()
        controller = SimpleNamespace(
            _owner=self.owner, loaded=SimpleNamespace(mode='MCD', mcd_result=result, mcd_settings=None),
            _mcd_background_suggestion=None, _mcd_background_request=None, _is_closing=False,
            mcd_window_center_spin=SimpleNamespace(value=lambda: 1.65),
            mcd_window_width_spin=SimpleNamespace(value=lambda: 10.),
            _status=lambda _: None,
        )
        with patch('ui_qt.controllers_mcd.QThreadPool.globalInstance', return_value=pool), \
                patch('ui_qt.controllers_mcd.suggest_mcd_background_ranges', return_value='obsolete') as calculate:
            controller._show_mcd_background_suggestion_dialog = lambda *_: self.fail('stale dialog opened')
            McdController._suggest_mcd_background_ranges(controller)
            calculate.assert_not_called()
            request = controller._mcd_background_request
            request.timer.stop()
            request._launch()
            controller.loaded.mcd_result = source()
            pool.workers[0].run()
            self.assertIsNone(controller._mcd_background_request)

    def test_initial_suggestion_restarts_if_window_changed_before_completion(self):
        result = source()
        suggestion = mcd.suggest_mcd_background_ranges(
            result, protected_ranges_ev=((1.64, 1.66),),
            auto_detect_features=False, use_all_unprotected_bands=True,
        )
        for first_result in (suggestion, ValueError('obsolete ranges')):
            with self.subTest(failed=isinstance(first_result, Exception)):
                center, shown, pool = [1.65], [], CapturedPool()
                controller = SimpleNamespace(
                    _owner=self.owner, loaded=SimpleNamespace(mode='MCD', mcd_result=result, mcd_settings=None),
                    _mcd_background_suggestion=None, _mcd_background_request=None, _is_closing=False,
                    mcd_window_center_spin=SimpleNamespace(value=lambda: center[0]),
                    mcd_window_width_spin=SimpleNamespace(value=lambda: 10.),
                    _status=lambda _: None,
                    _show_error=lambda _: self.fail('obsolete error shown'),
                    _show_mcd_background_suggestion_dialog=lambda *args: shown.append(args),
                )
                controller._suggest_mcd_background_ranges = lambda: McdController._suggest_mcd_background_ranges(controller)
                with patch('ui_qt.controllers_mcd.QThreadPool.globalInstance', return_value=pool), \
                        patch('ui_qt.controllers_mcd.suggest_mcd_background_ranges',
                              side_effect=(first_result, suggestion)):
                    controller._suggest_mcd_background_ranges()
                    first = controller._mcd_background_request
                    first.timer.stop()
                    first._launch()
                    center[0] = 1.69
                    pool.workers[0].run()
                    self.assertEqual(shown, [])
                    latest = controller._mcd_background_request
                    self.assertIsNot(latest, first)
                    latest.timer.stop()
                    latest._launch()
                    pool.workers[1].run()
                self.assertEqual(len(shown), 1)
                np.testing.assert_allclose(shown[0][1].requested_protected_ranges, ((1.68, 1.70),))


class McdTraceRoiCacheTests(unittest.TestCase):
    def setUp(self):
        self.result = source(rows=6, columns=301)
        self.view = SimpleNamespace(
            _result=self.result, _window_trace_cache=None,
            state=SimpleNamespace(window_metric='mean', window_center_ev=1.65, window_width_mev=3.),
        )

    def trace(self):
        return McdUnifiedView.window_traces(self.view)[1]

    def test_cache_validates_only_roi_and_preserves_in_place_updates(self):
        with patch('core.mcd.pair_window_trace_by_branch', wraps=mcd.pair_window_trace_by_branch) as reduce:
            original = self.trace()
            self.assertIs(self.trace(), original)
            self.result.pair_mcd_corrected[:, 0] += 100
            self.assertIs(self.trace(), original)
            self.assertEqual(reduce.call_count, 1)
            self.result.pair_mcd_corrected[:, 150] += 90
            changed = self.trace()
            self.assertEqual(reduce.call_count, 2)
            self.assertFalse(np.array_equal(original['B increasing']['corrected_mean'][1],
                                           changed['B increasing']['corrected_mean'][1]))
            snapshot = self.view._window_trace_cache[2][1]
            self.assertLess(snapshot.nbytes, self.result.pair_mcd_corrected.nbytes / 10)

    def test_fields_labels_axis_and_window_changes_invalidate(self):
        with patch('core.mcd.pair_window_trace_by_branch', wraps=mcd.pair_window_trace_by_branch) as reduce:
            self.trace()
            self.result.pair_b[0] = -5
            self.assertEqual(self.trace()['B increasing']['corrected_mean'][0][0], -5)
            self.result.pair_labels[0] = 'B decreasing'
            self.assertEqual(self.trace()['B decreasing']['corrected_mean'][0].tolist(), [-5])
            self.result.wavelength_nm[150] = mcd.EV_NM / 1.7
            self.trace()
            self.view.state.window_center_ev = 1.72
            actual = self.trace()
            expected = mcd.pair_window_trace_by_branch(self.result, 1.72, 3., metrics=('mean',), include_raw=False)
            np.testing.assert_allclose(actual['B increasing']['corrected_mean'][1],
                                       expected['B increasing']['corrected_mean'][1])
            self.assertEqual(reduce.call_count, 6)  # five view requests plus expected reference

    def test_energy_only_source_and_nearest_point_fallback(self):
        del self.result.wavelength_nm
        self.view.state.window_width_mev = 0.
        self.view.state.window_center_ev = 1.6501
        original = self.trace()
        self.result.pair_mcd_corrected[:, 150] += 3
        changed = self.trace()
        np.testing.assert_allclose(changed['B increasing']['corrected_mean'][1],
                                   original['B increasing']['corrected_mean'][1] + 3)
        self.result.energy_ev[150] = 1.75
        self.assertIsNot(self.trace(), changed)


class McdSuggestionCancellationTests(unittest.TestCase):
    def test_cancel_during_smoothing_propagates_without_reaching_fits(self):
        result = source(rows=10)
        cancel = LoadCancellation()
        original = mcd.savgol_filter
        calls = []

        def smooth(*args, **kwargs):
            calls.append(1)
            value = original(*args, **kwargs)
            if len(calls) == 2:  # full-sweep average, then the first spectrum
                cancel.cancel()
            return value

        with load_scope(cancel), patch('core.mcd.savgol_filter', side_effect=smooth), \
                patch('core.mcd._cross_validated_spectral_rms') as fit:
            with self.assertRaises(LoadCancelled):
                mcd.suggest_mcd_background_ranges(result)
            fit.assert_not_called()
        self.assertEqual(len(calls), 2)

    def test_cancel_between_validation_rows_stops_remaining_fits(self):
        result = source(rows=10)
        cancel = LoadCancellation()
        original = mcd._robust_log_ratio_coefficients
        calls = []

        def fit(*args, **kwargs):
            calls.append(1)
            value = original(*args, **kwargs)
            cancel.cancel()
            return value

        with load_scope(cancel), patch('core.mcd._robust_log_ratio_coefficients', side_effect=fit):
            with self.assertRaises(LoadCancelled):
                mcd.suggest_mcd_background_ranges(
                    result, protected_ranges_ev=((1.64, 1.66),),
                    auto_detect_features=False, use_all_unprotected_bands=True,
                )
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
