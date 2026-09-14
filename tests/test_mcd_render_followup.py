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
        axis = view.axes['spectra']
        with patch.object(axis, 'draw_artist', wraps=axis.draw_artist) as draw:
            view.canvas.draw()
            lines = [item[0] for item in view._artists['spectrum_lines']]
            self.assertTrue(all(any(call.args[0] is line for call in draw.call_args_list) for line in lines))
        view.close()

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
