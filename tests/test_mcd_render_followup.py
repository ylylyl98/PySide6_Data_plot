import unittest
from unittest.mock import patch

import numpy as np
from tests.test_mcd_unified_workflow import _result
from PySide6.QtWidgets import QApplication
from ui_qt.mcd_unified_page import McdUnifiedView


class McdRenderFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_full_draw_paints_animated_curves_before_next_event(self):
        view = McdUnifiedView()
        view.render(_result())
        line = view._artists['spectrum_lines'][0][0]
        with patch.object(line, 'draw', wraps=line.draw) as draw:
            view.canvas.draw()
            draw.assert_called()
        view.close()

    def test_cached_window_is_present_in_full_draw_without_draw_callback(self):
        view = McdUnifiedView()
        self.addCleanup(view.close)
        view.state.window_center_ev = 1.64
        view.render(_result())
        view.set_window(1.65, 5.)
        view._prepare_blit()
        label = view._artists['window_status'][0]
        # A full renderer pass must contain the label without needing a
        # draw-event callback to add animated overlays back afterward.
        with view.canvas.callbacks.blocked(signal='draw_event'):
            view.canvas.draw()
            visible = np.asarray(view.canvas.buffer_rgba()).copy()
            label.set_visible(False)
            view.canvas.draw()
            hidden = np.asarray(view.canvas.buffer_rgba()).copy()
        self.assertGreater(np.count_nonzero(visible != hidden), 30)

    def test_pending_qt_draw_during_cache_publication_keeps_map_label(self):
        view = McdUnifiedView()
        self.addCleanup(view.close)
        view.state.window_center_ev = 1.64
        view.render(_result())
        label = view._artists['window_status'][0]
        original_blit = view.canvas.blit
        pending = [True]
        def publish(*args, **kwargs):
            # Qt paintEvent can service an already queued draw_idle when a
            # blit requests repaint (rapid selection / initial Load).
            if pending[0]:
                pending[0] = False
                view.canvas.draw()
            return original_blit(*args, **kwargs)
        with patch.object(view.canvas, 'blit', side_effect=publish):
            view._prepare_blit()
        actual = np.asarray(view.canvas.buffer_rgba()).copy()
        bbox = label.get_window_extent(view.canvas.get_renderer())
        with view.canvas.callbacks.blocked(signal='draw_event'):
            view.canvas.draw()
        expected = np.asarray(view.canvas.buffer_rgba()).copy()
        h = expected.shape[0]
        crop = (slice(h-int(bbox.y1)-2, h-int(bbox.y0)+2), slice(int(bbox.x0)-2, int(bbox.x1)+2))
        self.assertLess(np.count_nonzero(actual[crop] != expected[crop]), 5)

    def test_window_update_expands_y_limits_without_replacing_map(self):
        result = _result()
        result.wavelength_nm = 1239.841984 / result.energy_ev
        view = McdUnifiedView()
        view.state.window_center_ev = 1.64
        view.state.window_width_mev = 5
        view.render(result)
        map_artist = view.axes['mcd_map'].collections[0]
        result.pair_mcd_corrected *= 100
        view.set_window(1.645, 5)
        ys = np.concatenate([line.get_ydata() for _, line in view._artists['mcd_trace_lines']])
        low, high = view.axes['mcd_vs_b'].get_ylim()
        self.assertLessEqual(low, np.nanmin(ys))
        self.assertGreaterEqual(high, np.nanmax(ys))
        self.assertIs(view.axes['mcd_map'].collections[0], map_artist)
        view.close()

    def test_hidden_branch_has_no_fit_segments(self):
        view = McdUnifiedView()
        view.render(_result())
        view.show_dec_chk.setChecked(False)
        fits = [dict(region='low', branch=branch, status='ok', slope=1., intercept=0.,
                     field_min_t=-.2, field_max_t=.2)
                for branch in ('B increasing', 'B decreasing')]
        view.update_mcd_slope_readout({'fits': fits})
        self.assertEqual(set(view._artists['mcd_fit_lines']), {('low', 'B increasing')})
        view.close()

    def test_resized_canvas_keeps_pixel_space_for_labels(self):
        view = McdUnifiedView()
        view.render(_result())
        view.figure.set_size_inches(9.4, 3.44)
        view.canvas.draw()
        self.app.processEvents()
        top = view.axes['spectra'].bbox
        bottom = view.axes['mcd_spectra'].bbox
        self.assertGreaterEqual(bottom.y0, 46)
        self.assertGreaterEqual(top.y0 - bottom.y1, 60)
        self.assertGreaterEqual(view.figure.bbox.height - top.y1, 30)
        view.close()
