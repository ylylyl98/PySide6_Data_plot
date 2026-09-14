from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.mcd import pair_window_trace_by_branch
from tests.test_mcd_unified_workflow import _result as base_result
from ui_qt.common import LoadedState
from ui_qt.main_window import MainWindow


def _result() -> SimpleNamespace:
    value = base_result()
    value.wavelength_nm = 1239.841984 / np.asarray(value.energy_ev, dtype=float)
    value.pair_b_pos = np.asarray(value.pair_b, dtype=float)
    value.pair_b_neg = np.asarray(value.pair_b, dtype=float)
    return value


def _window(window, *, ident="retained-window", slopes=None):
    return {
        "id": ident,
        "identity": {"source": "synthetic.csv", "kind": "mcd-window"},
        "source_generation": int(window.mcd_unified_view.state.source_generation),
        "original_b": np.asarray(window.loaded.mcd_result.pair_b, dtype=float).copy(),
        "branches": np.asarray(window.loaded.mcd_result.pair_labels, dtype=str).copy(),
        "trace_values": np.array([10.0, 20.0, 30.0]),
        "metric": "Signed integral",
        "center_ev": 1.64,
        "width_mev": 5.0,
        "slopes": slopes or {"fits": [{"region": "low", "slope": 2.0}], "differences": []},
        "settings": {"window_metric": "Signed integral"},
        "status": "complete",
    }


class McdRetentionExportIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _window_with_result(self):
        window = MainWindow()
        window.loaded = LoadedState(mode="MCD", folder="", mcd_result=_result())
        window._plot_mode("MCD")
        window.mcd_unified_view.set_source_generation(7)
        return window

    def test_current_window_uses_branch_metric_helper_exactly(self):
        window = self._window_with_result()
        try:
            window.mcd_window_metric_combo.setCurrentText("Signed integral")
            window.mcd_unified_view.set_window(1.64, 5.0)
            snapshot = window._current_unified_window_snapshot()
            expected_by_branch = pair_window_trace_by_branch(
                window.loaded.mcd_result, 1.64, 5.0,
                metrics=("integral",), include_raw=False,
            )
            expected = np.full(len(window.loaded.mcd_result.pair_b), np.nan)
            labels = np.asarray(window.loaded.mcd_result.pair_labels, dtype=str)
            for branch, branch_data in expected_by_branch.items():
                expected[labels == branch] = branch_data["corrected_integral"][1]
            np.testing.assert_allclose(snapshot["trace_values"], expected)
        finally:
            window.close()

    def test_export_uses_only_checked_retained_windows_and_retained_slopes(self):
        window = self._window_with_result()
        try:
            retained = _window(window)
            window.mcd_retention.store.add_window(retained, included=True)
            window._mcd_unified_track_payload = {"tracks": [{"id": "current"}], "status": "ok"}
            window._mcd_unified_slopes = SimpleNamespace(to_dict=lambda: {"fit": "current"})
            window.mcd_unified_view.set_window(1.67, 8.0)
            captured = []
            messages = []
            with tempfile.TemporaryDirectory() as output, \
                    patch("core.mcd_unified_export.build_mcd_export_snapshot", side_effect=lambda result, **kwargs: kwargs), \
                    patch.object(window.thread_pool, "start", side_effect=lambda worker: captured.append(worker)), \
                    patch.object(window, "_status", side_effect=messages.append):
                window._queue_unified_mcd_export(output, scope="retained")
            self.assertEqual(len(captured), 1)
            kwargs = captured[0].args[0]
            self.assertEqual(kwargs["analysis_results"], [])
            self.assertEqual(kwargs["slopes"][0]["slope"], 2.0)
            self.assertEqual(kwargs["slopes"][0]["window_id"], "retained-window")
            self.assertEqual(kwargs["windows"][0]["center_ev"], 1.64)
            self.assertEqual(kwargs["windows"][0]["trace_values"].tolist(), [10.0, 20.0, 30.0])
        finally:
            window.close()

    def test_export_retained_feature_does_not_duplicate_current_payload(self):
        window = self._window_with_result()
        try:
            retained_payload = {"status": "ok", "tracks": [{"id": "retained"}]}
            window.mcd_retention.store.add_feature({
                "id": "retained-feature", "source_generation": 7,
                "identity": {"label": "retained"}, "track_payload": retained_payload,
                "status": "complete",
            }, included=True)
            window._mcd_unified_track_payload = {"status": "ok", "tracks": [{"id": "current"}]}
            captured = []
            with tempfile.TemporaryDirectory() as output, \
                    patch("core.mcd_unified_export.build_mcd_export_snapshot", side_effect=lambda result, **kwargs: kwargs), \
                    patch.object(window.thread_pool, "start", side_effect=lambda worker: captured.append(worker)):
                window._queue_unified_mcd_export(output, scope="retained")
            kwargs = captured[0].args[0]
            self.assertEqual([item["tracks"][0]["id"] for item in kwargs["analysis_results"]], ["retained"])
        finally:
            window.close()

    def test_current_items_include_completed_matching_feature(self):
        window = self._window_with_result()
        try:
            window._plot_mcd_unified()
            window.mcd_unified_view.set_candidates([{
                "id": "current-feature", "domain": "mcd", "source": "mcd", "kind": "peak",
                "center_ev": 1.64,
            }])
            window._mcd_unified_track_payload = {"status": "ok", "tracks": [{"id": "current"}]}
            window._mcd_unified_track_payload_key = (window.mcd_unified_view.state.source_generation, window._unified_feature_analysis_key())
            self.assertIsNotNone(window._current_unified_feature_snapshot())
            captured = []
            with tempfile.TemporaryDirectory() as output, \
                    patch("core.mcd_unified_export.build_mcd_export_snapshot", side_effect=lambda result, **kwargs: kwargs), \
                    patch.object(window.thread_pool, "start", side_effect=lambda worker: captured.append(worker)):
                window._queue_unified_mcd_export(output)
            kwargs = captured[0].args[0]
            self.assertEqual([item["tracks"][0]["id"] for item in kwargs["analysis_results"]], ["current"])
            self.assertEqual(len(kwargs["windows"]), 1)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
