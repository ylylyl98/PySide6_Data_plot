from __future__ import annotations

import os
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from ui_qt.mcd_unified_page import McdUnifiedView, UnifiedMcdState
from ui_qt.main_window import LoadedState, MainWindow
from ui_qt.mcd_result_retention import StaleRetainedResultError


def _result() -> SimpleNamespace:
    energy = np.linspace(1.60, 1.68, 81)
    fields = np.array([-1.0, 0.0, 1.0])
    spectra = np.array([
        1.0 + np.exp(-((energy - (1.64 + 0.002 * b)) / 0.004) ** 2)
        for b in fields
    ])
    return SimpleNamespace(
        energy_ev=energy,
        pair_b=fields,
        pair_labels=np.array(["B decreasing", "B increasing", "B increasing"]),
        pair_corrected_pos=spectra,
        pair_corrected_neg=spectra * 0.98,
        pair_raw_pos=spectra,
        pair_raw_neg=spectra * 0.98,
        pair_mcd_corrected=np.array([spectra[i] * 0.01 for i in range(3)]),
        pair_mcd_raw=np.array([spectra[i] * 0.02 for i in range(3)]),
        source_file="synthetic.csv",
    )


class UnifiedMcdViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_unified_view_has_four_fixed_regions_and_no_internal_tabs(self):
        view = McdUnifiedView()
        view.render(_result())
        self.assertEqual(set(view.axes), {"mcd_map", "spectra", "mcd_vs_b", "mcd_spectra"})
        self.assertTrue(view.axes["mcd_map"].get_shared_x_axes().joined(view.axes["mcd_map"], view.axes["spectra"]))
        self.assertTrue(view.axes["mcd_map"].get_shared_x_axes().joined(view.axes["mcd_map"], view.axes["mcd_spectra"]))
        self.assertFalse(view.findChildren(__import__("PySide6.QtWidgets", fromlist=["QTabWidget"]).QTabWidget))

    def test_candidate_filter_keeps_numbered_peaks_dips_and_mcd_candidates(self):
        view = McdUnifiedView()
        view.set_candidates(
            [
                {"id": "P1", "kind": "peak", "source": "spectrum"},
                {"id": "D1", "kind": "dip", "source": "spectrum"},
                {"id": "M+1", "kind": "peak", "source": "mcd"},
                {"id": "M-1", "kind": "dip", "source": "mcd"},
            ]
        )
        view.set_candidate_filter("MCD")
        self.assertEqual([item["id"] for item in view.visible_candidates], ["M+1", "M-1"])
        view.set_candidate_filter("Spectrum")
        self.assertEqual([item["id"] for item in view.visible_candidates], ["P1", "D1"])
        view.set_candidate_filter("All")
        self.assertEqual([item["id"] for item in view.visible_candidates], ["P1", "D1", "M+1", "M-1"])

    def test_default_mcd_filter_keeps_full_catalog_and_marks_recommendations(self):
        view = McdUnifiedView()
        candidates = [
            {"id": f"M{i}", "domain": "mcd", "kind": "peak", "energy_ev": energy, "prominence": prominence}
            for i, (energy, prominence) in enumerate(
                ((1.68, 1.0), (1.62, 6.0), (1.66, 2.0), (1.64, 5.0), (1.60, 3.0), (1.70, 4.0)), 1
            )
        ]
        view.set_candidates(candidates)
        self.assertEqual([item["id"] for item in view.visible_candidates], ["M5", "M2", "M4", "M3", "M1", "M6"])
        view.set_candidate_filter("All")
        self.assertEqual(len(view.visible_candidates), 6)

    def test_filter_and_selection_draw_all_candidates_with_selected_outline(self):
        view = McdUnifiedView()
        view.set_candidates([
            {"id": "M1", "domain": "mcd", "kind": "mcd_max", "center_ev": 1.62, "recommended": True},
            {"id": "M2", "domain": "mcd", "kind": "mcd_min", "center_ev": 1.65},
            {"id": "P1", "domain": "spectrum", "kind": "dip", "center_ev": 1.64},
        ])
        view.render(_result(), candidates=view._candidates)
        view.set_candidate_filter("MCD")
        self.assertGreaterEqual(len(view._artists["feature_selection"]), 4)
        self.assertTrue(any(getattr(item, "_mcd_candidate_id", None) == "M1" for item in view._artists["feature_selection"]))
        view.set_candidate_filter("Spectrum")
        self.assertGreaterEqual(len(view._artists["feature_selection"]), 2)
        view.set_candidate_filter("All")
        self.assertEqual(len(view.visible_candidates), 3)

    def test_window_marker_moves_with_window_without_rebuilding_catalog(self):
        view = McdUnifiedView()
        view.set_candidates([{"id": "M1", "domain": "mcd", "kind": "mcd_max", "center_ev": 1.64}])
        view.state.window_center_ev = 1.63
        view.render(_result(), candidates=view._candidates)
        lines = [artist for artist in view._artists["feature_selection"] if getattr(artist, "_mcd_window_label", "") == "Window"]
        self.assertTrue(lines)
        render_count = view.render_count
        view.set_window(1.65, 5.0)
        self.assertEqual(view.render_count, render_count)
        self.assertAlmostEqual(float(lines[0].get_xdata()[0]), 1.65)

    def test_b_cursor_and_window_are_independent_display_state(self):
        view = McdUnifiedView()
        view.render(_result())
        view.state.window_center_ev = 1.64
        view.state.window_width_mev = 5.0
        render_count = view.render_count
        view.set_selected_b(2)
        self.assertEqual(view.state.selected_b_index, 2)
        self.assertEqual(view.state.window_center_ev, 1.64)
        self.assertEqual(view.state.window_width_mev, 5.0)
        self.assertEqual(view.render_count, render_count)
        self.assertGreater(view.draw_only_updates, 0)

    def test_map_click_resolves_physical_field_against_pair_grid(self):
        view = McdUnifiedView()
        view.render(_result())
        view.set_selected_b(2)
        view._on_map_click(SimpleNamespace(inaxes=view.axes["mcd_map"], ydata=-1.0, button=1, key="control"))
        self.assertEqual(view.state.selected_b_index, 0)

    def test_unified_map_honors_selected_colormap(self):
        window = MainWindow()
        try:
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=_result())
            window.mcd_cmap.setCurrentText("viridis")
            window._plot_mode("MCD")
            self.assertTrue(window.mcd_unified_view.axes["mcd_map"].collections)
            self.assertEqual(window.mcd_unified_view.axes["mcd_map"].collections[0].cmap.name, "viridis")
        finally:
            window.close()
            window.deleteLater()

    def test_unified_slope_readout_keeps_all_fits_and_correct_difference_label(self):
        view = McdUnifiedView()
        fits = [
            {"region": region, "branch": branch, "status": "ok", "slope": 1.0, "intercept": 0.0,
             "field_min_t": -0.2 if region == "low" else (1.5 if region == "high_positive" else -2.0),
             "field_max_t": 0.2 if region == "low" else (2.0 if region == "high_positive" else -1.5),
             "slope_se": 0.1}
            for region in ("low", "high_positive", "high_negative")
            for branch in ("B increasing", "B decreasing")
        ]
        payload = {
            "slopes": {
                "fits": fits,
                "differences": [{"region": "high_positive", "reference_region": "low", "branch_a": "B increasing", "branch_b": "B increasing", "comparison": "high_minus_low", "slope_difference": 2.0, "status": "ok"}],
            }
        }
        view.render(_result(), analysis=payload)
        text = getattr(view, "_slope_details_text", "")
        self.assertIn("high_negative/B decreasing", text)
        self.assertIn("high_positive−low / B increasing", text)
        self.assertEqual(len(view._artists["mcd_fit_lines"]), 6)
        line = view._artists["mcd_fit_lines"][("low", "B increasing")]
        view.update_mcd_slope_readout({"fits": [{**fits[0], "slope": 2.0}]})
        self.assertEqual(list(line.get_ydata()), [-0.4, 0.4])

    def test_reversed_explicit_search_bounds_are_rejected_before_worker(self):
        window = MainWindow()
        try:
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=_result())
            window._plot_mode("MCD")
            window.mcd_unified_view.set_candidate_filter("Spectrum")
            with patch.object(window.thread_pool, "start", side_effect=lambda worker: worker.run()):
                window._queue_mcd_unified_analysis()
            window.mcd_unified_view.set_candidates([{"id": "P1", "center_ev": 1.64, "domain": "spectrum", "kind": "peak"}])
            window.mcd_unified_controls.feature_search_low_spin.setValue(1.66)
            window.mcd_unified_controls.feature_search_high_spin.setValue(1.65)
            with patch.object(window.thread_pool, "start") as start:
                window._queue_unified_selected_feature_analysis()
            start.assert_not_called()
            self.assertIn("increasing", window.mcd_unified_view.candidate_status.text())
        finally:
            window.close()
            window.deleteLater()

    def test_pair_spectra_are_independent_of_trace_branch_visibility(self):
        view = McdUnifiedView()
        view.render(_result())
        self.assertEqual(len(view._artists["spectrum_lines"]), 4)
        view.show_inc_chk.setChecked(False)
        self.assertEqual(len(view._artists["spectrum_lines"]), 4)
        self.assertEqual([branch for branch, line in view._artists['mcd_trace_lines'] if line.get_visible()], ['B decreasing'])

    def test_stale_source_generation_is_rejected(self):
        view = McdUnifiedView()
        view.set_source_generation(4)
        self.assertFalse(view.publish_analysis({"generation": 3, "value": "old"}))
        self.assertIsNone(view.analysis_payload)
        self.assertTrue(view.publish_analysis({"generation": 4, "value": "new"}))
        self.assertEqual(view.analysis_payload["value"], "new")

    def test_state_retains_independent_window_snapshots(self):
        state = UnifiedMcdState()
        state.retain_window("MCD", 1.64, 5.0)
        state.retain_window("MCD", 1.67, 8.0)
        self.assertEqual([(x.center_ev, x.width_mev) for x in state.retained_windows], [(1.64, 5.0), (1.67, 8.0)])

    def test_main_window_retained_update_is_explicit_and_keeps_source_settings(self):
        window = MainWindow()
        try:
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=_result())
            window._plot_mode("MCD")
            window.mcd_unified_view.set_source_generation(7)
            window.mcd_unified_view.set_window(1.64, 5.0)
            window._retain_unified_mcd_window()
            window.mcd_unified_controls.retained_window_list.setCurrentRow(0)
            window.mcd_unified_view.set_window(1.67, 8.0)
            window._update_unified_mcd_window()
            retained = window.mcd_unified_view.state.retained_windows[0]
            self.assertEqual((retained.center_ev, retained.width_mev, retained.source_generation), (1.67, 8.0, 7))
            self.assertTrue(dict(retained.settings)["feature_method"])
        finally:
            window.close()
            window.deleteLater()

    def test_main_window_exposes_one_visible_mcd_entry_and_unified_canvas(self):
        window = MainWindow()
        try:
            visible = [window.workflow_tabs.tabText(i) for i in range(window.workflow_tabs.count()) if window.workflow_tabs.isTabVisible(i)]
            self.assertEqual(visible.count("MCD"), 1)
            self.assertNotIn("MCD Peak Shift", visible)
            self.assertIsInstance(window.mcd_unified_view, McdUnifiedView)
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=_result())
            window.tabs.setCurrentIndex(next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == "MCD"))
            window._plot_mode("MCD")
            self.assertEqual(set(window.mcd_unified_view.axes), {"mcd_map", "spectra", "mcd_vs_b", "mcd_spectra"})
        finally:
            window.close()
            window.deleteLater()

    def test_main_window_save_results_dispatches_snapshot_export(self):
        window = MainWindow()
        try:
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=_result())
            window._plot_mode("MCD")
            window.mcd_unified_view.set_window(1.64, 5.0)
            with tempfile.TemporaryDirectory() as output:
                window.mcd_window_center_spin.setValue(1.64)
                window.mcd_window_width_spin.setValue(5.0)
                window._plot_mcd_unified()
                with patch.object(window.thread_pool, "start", side_effect=lambda worker: worker.run()):
                    window._queue_unified_mcd_export(output)
                self.assertIsNone(window._mcd_unified_export_worker)
                exported = list(__import__("pathlib").Path(output).rglob("*.json"))
                self.assertTrue(exported)
        finally:
            window.close()
            window.deleteLater()

    def test_main_window_retention_exports_checked_frozen_snapshots_and_rejects_stale(self):
        window = MainWindow()
        try:
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=_result())
            window._plot_mode("MCD")
            window.mcd_unified_view.set_source_generation(7)
            window.mcd_unified_view.set_window(1.64, 5.0)
            window._retain_unified_mcd_window()
            retained = window.mcd_retention.store.items("window")
            self.assertEqual(len(retained), 1)
            self.assertTrue(retained[0].included)
            with self.assertRaises(ValueError):
                retained[0].trace_values[0] = 99.0
            window.mcd_retention.add_window(
                {**window._current_unified_window_snapshot(), "id": "old", "source_generation": 6},
                included=True,
            )
            with self.assertRaises(StaleRetainedResultError):
                window.mcd_retention.validated_selection(source_generation=7)
            window.mcd_retention.store.set_included("window", "old", False)
            selected = window.mcd_retention.validated_selection(source_generation=7)
            self.assertEqual([item.id for item in selected["window"]], [retained[0].id])
        finally:
            window.close()
            window.deleteLater()

    def test_main_toolbar_save_routes_loaded_mcd_to_unified_export(self):
        window = MainWindow()
        try:
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=_result())
            window.tabs.setCurrentIndex(next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == "MCD"))
            window.last_plotted_mode = "MCD"
            with patch.object(window, "_queue_unified_mcd_export") as save:
                window._toolbar_save()
            save.assert_called_once_with(scope="current")
        finally:
            window.close()
            window.deleteLater()

    def test_main_window_queues_real_feature_worker_and_publishes_candidates(self):
        window = MainWindow()
        try:
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=_result())
            with patch.object(window.thread_pool, "start", side_effect=lambda worker: worker.run()):
                window._queue_mcd_unified_analysis()
            self.assertIsNone(window._mcd_unified_analysis_worker)
            self.assertGreater(len(window._mcd_unified_features), 0)
            self.assertGreater(len(window._unified_mcd_candidates()), 0)
        finally:
            window.close()
            window.deleteLater()

    def test_catalog_completion_automatically_queues_selected_tracking(self):
        window = MainWindow()
        try:
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=_result())
            window._plot_mode("MCD")
            window.mcd_unified_view.candidate_filter_combo.setCurrentText("Spectrum")
            with patch.object(window.thread_pool, "start", side_effect=lambda worker: worker.run()):
                window._queue_mcd_unified_analysis()
                for _ in range(5):
                    self.app.processEvents()
            self.assertIsNotNone(window.mcd_unified_view.selected_candidate)
            self.assertIsNotNone(window._mcd_unified_track_payload)
            self.assertIn(window._mcd_unified_track_payload.get("status"), {"ok", "unmatched", "ambiguous", "unsupported"})
        finally:
            window.close()
            window.deleteLater()

    def test_unified_center_refresh_updates_existing_artists_without_full_render(self):
        window = MainWindow()
        try:
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=_result())
            window._plot_mode("MCD")
            view = window.mcd_unified_view
            view.set_window(1.64, 5.0)
            render_count = view.render_count
            with patch.object(window, "_plot_mode") as full_plot:
                window.mcd_window_center_spin.setValue(1.65)
                window.mcd_controller._apply_pending_mcd_center_refresh()
            full_plot.assert_not_called()
            self.assertEqual(view.render_count, render_count)
            self.assertGreater(view.draw_only_updates, 0)
            self.assertAlmostEqual(view.state.window_center_ev, 1.65)
        finally:
            window.close()
            window.deleteLater()

    def test_selected_spectrum_candidate_uses_core_tracking_worker(self):
        window = MainWindow()
        try:
            window.loaded = LoadedState(mode="MCD", folder="", mcd_result=_result())
            with patch.object(window.thread_pool, "start", side_effect=lambda worker: worker.run()):
                window._queue_mcd_unified_analysis()
            window.mcd_unified_view.set_candidate_filter("Spectrum")
            self.assertIsNotNone(window.mcd_unified_view.selected_candidate)
            with patch.object(window.thread_pool, "start", side_effect=lambda worker: worker.run()):
                window._on_unified_candidate_selected(0)
            self.assertIsNotNone(window._mcd_unified_track_payload)
            self.assertEqual(window._mcd_unified_track_payload.get("status"), "ok")
            tracks = window._mcd_unified_track_payload.get("tracks", ())
            self.assertTrue(tracks)
            self.assertTrue(all(track.get("points") for track in tracks))
        finally:
            window.close()
            window.deleteLater()


if __name__ == "__main__":
    unittest.main()
