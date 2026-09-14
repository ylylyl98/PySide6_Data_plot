from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ui_qt.mcd_unified_page import McdUnifiedView, fit_display_segments


def _result() -> SimpleNamespace:
    energy = np.linspace(1.60, 1.68, 9)
    fields = np.array([-1.0, 0.0, 1.0])
    spectra = np.array([1.0 + 0.1 * row + energy * 0.01 for row in range(3)])
    return SimpleNamespace(
        energy_ev=energy,
        pair_b=fields,
        pair_b_pos=np.array([-1.02, 0.01, 1.03]),
        pair_b_neg=np.array([-0.98, -0.01, 0.97]),
        pair_labels=np.array(["B decreasing", "B increasing", "B increasing"]),
        pair_corrected_pos=spectra,
        pair_corrected_neg=spectra * 0.98,
        pair_raw_pos=spectra,
        pair_raw_neg=spectra * 0.98,
        pair_mcd_corrected=spectra * 0.01,
        pair_mcd_raw=spectra * 0.02,
    )


class MCDDisplayStyleTests(unittest.TestCase):
    def test_mcd_y_limits_follow_data_and_shrink_after_window_change(self):
        view = McdUnifiedView()
        self.addCleanup(view.close)
        result = _result()
        result.wavelength_nm = 1239.841984 / result.energy_ev
        result.pair_mcd_corrected = np.tile(np.array([-.1, 0., .1])[:, None], (1, 9))
        result.pair_mcd_corrected[:, 6] *= .1
        view.state.window_center_ev = 1.62
        view.state.window_width_mev = 5.
        view.render(result, analysis={'slopes': {'fits': [
            {'region': 'low', 'branch': 'B increasing', 'status': 'ok',
             'slope': 100., 'intercept': 0., 'field_min_t': 0., 'field_max_t': 1.}]}})
        np.testing.assert_allclose(view.axes['mcd_vs_b'].get_ylim(), [-.11, .11])
        view.set_window(1.66, 5.)
        from PySide6.QtTest import QTest
        QTest.qWait(220)
        np.testing.assert_allclose(view.axes['mcd_vs_b'].get_ylim(), [-.011, .011])

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_fit_display_extends_half_span_without_changing_fit_subset(self):
        fit = {"field_min_t": 1.0, "field_max_t": 3.0, "slope": 2.0, "intercept": 1.0}
        segments = fit_display_segments(fit)
        np.testing.assert_allclose(segments["actual_x"], [1.0, 3.0])
        np.testing.assert_allclose(segments["extension_x"], [0.0, 4.0])
        np.testing.assert_allclose(segments["actual_y"], [3.0, 7.0])
        np.testing.assert_allclose(segments["extension_y"], [1.0, 9.0])

    def test_fit_display_spans_visible_axis_without_changing_fit_subset(self):
        fit = {"field_min_t": .1, "field_max_t": .2, "slope": 2., "intercept": 1.}
        segments = fit_display_segments(fit, visible_min=-2., visible_max=2.)
        np.testing.assert_allclose(segments['extension_x'], [-2., 2.])
        np.testing.assert_allclose(segments['actual_x'], [.1, .2])

    def test_slope_labels_remain_inside_compact_axes(self):
        view = McdUnifiedView()
        fits = []
        for region, low, high in (("low", -.2, .2), ("high_positive", .4, .8), ("high_negative", -.8, -.4)):
            for branch in ("B increasing", "B decreasing"):
                # Keep this layout fixture inside the measured Y range;
                # off-range fit annotations are intentionally clipped.
                fits.append({"region": region, "branch": branch, "status": "ok", "slope": .001,
                             "intercept": .011, "field_min_t": low, "field_max_t": high, "slope_se": .003})
        view.render(_result(), analysis={"slopes": {"fits": fits}})
        view.canvas.draw()
        renderer = view.canvas.get_renderer()
        bbox = view.axes["mcd_vs_b"].bbox
        for annotation in view._artists["slope_annotations"].values():
            extent = annotation.get_window_extent(renderer)
            assert extent.x0 >= bbox.x0 - 1
            assert extent.x1 <= bbox.x1 + 1

    def test_spectra_use_channel_colors_correction_styles_and_measured_b_labels(self):
        view = McdUnifiedView()
        view.render(_result())
        lines = [entry[0] for entry in view._artists["spectrum_lines"]]
        assert len(lines) == 4
        assert {line.get_color() for line in lines} == {"#2563eb", "#f59e0b"}
        assert {line.get_linestyle() for line in lines} == {"-", "--"}
        assert all("B=" in line.get_label() for line in lines)
        assert all(line.get_marker() in ("", "None", None) for line in lines)

    def test_spectrum_legend_toggle_survives_measured_field_refresh(self):
        view = McdUnifiedView()
        view.render(_result())
        sample, source = next(iter(view._legend_line_map.items()))
        view._on_legend_pick(SimpleNamespace(artist=sample))
        assert source.get_visible() is False
        view.set_selected_b(2)
        assert source.get_visible() is False
        assert "B=" in source.get_label()

    def test_spectrum_legend_is_one_dynamic_artist_for_blit_updates(self):
        view = McdUnifiedView()
        view.render(_result())
        legend = view._artists["spectra_legend"]
        dynamic = view._dynamic_artists_by_axis()["spectra"]
        assert legend in dynamic
        # Only background capture temporarily marks overlays animated; normal
        # canvas redraws must include the legend without callback assistance.
        assert legend.get_animated() is False
        assert all(text not in dynamic for text in legend.get_texts())
        view.set_selected_b(2)
        assert all("B=" in text.get_text() for text in legend.get_texts())

    def test_metric_buttons_render_cached_payload_without_starting_worker(self):
        view = McdUnifiedView()
        view.result_panel_combo.setCurrentIndex(1)
        payload = {
            "tracks": [{"id": "K / Inc", "channel": "pos", "branch": "B increasing",
                        "points": [{"field_t": -1.0, "energy_ev": 1.63, "delta_energy_ev": 0.001},
                                   {"field_t": 1.0, "energy_ev": 1.64, "delta_energy_ev": 0.002}]}],
            "status": "ok",
        }
        view.render(_result(), analysis=payload)
        with patch.object(view, "render", wraps=view.render) as render:
            view.energy_metric_btn.click()
            view.overlay_btn.click()
        assert view.state.feature_metric == "energy"
        assert view.state.feature_overlay is True
        assert render.call_count == 0
        assert view.axes["feature_vs_b"].get_ylabel() == "Energy (eV)"
        assert view._feature_energy_axis is not None
        assert view._feature_energy_axis.get_ylabel() == "Shift (meV)"
        secondary = view._feature_energy_axis
        with patch.object(secondary, 'draw', wraps=secondary.draw) as draw:
            view._prepare_blit(full_draw=False)
            draw.assert_called_once()

    def test_splitting_uses_cached_channel_energies_with_correct_units(self):
        view = McdUnifiedView()
        payload = {
            "splitting": [{"pair_id": "K|Kp", "selected_channel": "raw pos", "branch": "B increasing",
                           "points": [{"field_t": 1.0, "energy_k_ev": 1.640, "energy_kp_ev": 1.638, "splitting_ev": .002}]}],
            "mapping": {"status": "fixed", "k_channel": "pos"}, "status": "ok",
        }
        view.render(_result(), analysis=payload)
        view.splitting_metric_btn.click()
        view.overlay_btn.click()
        assert view.axes["feature_vs_b"].get_ylabel() == "Splitting (meV)"
        assert view._artists["feature_lines"][0].get_label().startswith("E_K−E_K′")
        assert view._feature_energy_axis is not None
        assert view._feature_energy_axis.get_ylabel() == "Energy (eV)"
        np.testing.assert_allclose(view._feature_energy_axis.lines[0].get_ydata(), [1.640])

    def test_measured_points_keep_branch_marker_identity_and_no_interpolation(self):
        view = McdUnifiedView()
        view.result_panel_combo.setCurrentIndex(1)
        payload = {
            "tracks": [
                {"id": "Inc", "channel": "pos", "branch": "B increasing",
                 "points": [{"field_t": -1.0, "energy_ev": 1.63, "delta_energy_ev": .001}, {"field_t": 1.0, "energy_ev": 1.64, "delta_energy_ev": .002}]},
                {"id": "Dec", "channel": "neg", "branch": "B decreasing",
                 "points": [{"field_t": -0.8, "energy_ev": 1.62, "delta_energy_ev": -.001}, {"field_t": 0.8, "energy_ev": 1.61, "delta_energy_ev": -.002}]},
            ],
            "status": "ok",
        }
        view.render(_result(), analysis=payload)
        lines = view._artists["feature_lines"]
        assert [line.get_marker() for line in lines] == ["o", "s"]
        assert [line.get_linestyle() for line in lines] == ["-", "--"]
        assert len(lines[0].get_xdata()) == 2
        assert len(lines[1].get_xdata()) == 2
        assert lines[1].get_markerfacecolor() in ("none", "None", (0.0, 0.0, 0.0, 0.0))
