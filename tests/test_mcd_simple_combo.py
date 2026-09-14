import unittest
from types import SimpleNamespace

import numpy as np
from matplotlib.backend_bases import MouseEvent
from PySide6.QtWidgets import QApplication

from tests.test_mcd_unified_workflow import _result
from ui_qt.mcd_unified_page import McdUnifiedView
from ui_qt.main_window import MainWindow, LoadedState


class SimpleComboTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_view(self):
        view = McdUnifiedView()
        self.addCleanup(view.close)
        view.state.window_center_ev = 1.64
        result = _result()
        result.summary = {'correction_mode': 'pair_spectral',
                          'background_ranges_ev': ((1.60, 1.61), (1.67, 1.68))}
        result.acquisition_conditions = {'T': (4., 4.)}
        result.pair_raw_pos = result.pair_corrected_pos * 2
        view.render(result)
        return view, result

    def test_pair_spectra_show_raw_and_corrected_for_exact_selected_pair(self):
        view, result = self.make_view()
        view.set_selected_b(2)
        lines = [entry[0] for entry in view._artists['spectrum_lines']]
        self.assertEqual(len(lines), 4)
        for line, name in zip(lines, ('pair_raw_pos', 'pair_raw_neg',
                                     'pair_corrected_pos', 'pair_corrected_neg')):
            np.testing.assert_allclose(line.get_ydata(), getattr(result, name)[2])

    def test_corrected_linecut_and_window_follow_selected_b_and_map(self):
        view, result = self.make_view()
        self.assertIn('mcd_spectra', view.axes)
        view.set_selected_b(2)
        axis = view.axes['mcd_spectra']
        line = next(line for line in axis.lines if line.get_label() == 'Corrected MCD')
        np.testing.assert_allclose(line.get_ydata(), result.pair_mcd_corrected[2])
        view.set_window(1.65, 10.)
        np.testing.assert_allclose([line.get_xdata()[0] for line in view._artists['window_edges']],
                                   [1.645, 1.655, 1.645, 1.655])
        for key in ('mcd_map', 'mcd_spectra'):
            patches = [patch for patch in view._artists['window'] if patch.axes is view.axes[key]]
            self.assertEqual(len(patches), 1)
            self.assertAlmostEqual(patches[0].get_x(), 1.645)
            self.assertAlmostEqual(patches[0].get_width(), .01)
        view.axes['mcd_map'].set_xlim(1.62, 1.67)
        np.testing.assert_allclose(axis.get_xlim(), [1.62, 1.67])

    def test_center_update_preserves_other_panel_caches_and_skips_spectra(self):
        from unittest.mock import patch
        view, result = self.make_view()
        result.wavelength_nm = 1239.841984 / result.energy_ev
        before = dict(view._blit_backgrounds)
        old_limits = view.axes['mcd_vs_b'].get_ylim()
        spectrum = view._artists['spectrum_lines'][0][0]
        with patch.object(view.canvas, 'draw', wraps=view.canvas.draw) as full, patch.object(spectrum, 'draw', wraps=spectrum.draw) as curve:
            result.pair_mcd_corrected[:] = np.arange(result.pair_mcd_corrected.shape[0])[:, None] + 3.
            view.set_window(1.65, 10.)
            self.app.processEvents()
            self.assertEqual(full.call_count, 0)
            self.assertEqual(curve.call_count, 0)
        self.assertNotEqual(view.axes['mcd_vs_b'].get_ylim(), old_limits)
        for key in ('mcd_map', 'spectra', 'mcd_spectra'):
            self.assertIs(view._blit_backgrounds[key], before[key])

    def test_background_shading_uses_result_ranges(self):
        view, _ = self.make_view()
        bands = [p for p in view.axes['spectra'].patches if p.get_gid() == 'mcd-background-fit']
        self.assertEqual(len(bands), 2)
        np.testing.assert_allclose([p.get_x() for p in bands], [1.60, 1.67])
        np.testing.assert_allclose([p.get_width() for p in bands], [.01, .01])

    def test_window_labels_remain_readable_in_dark_and_publication_themes(self):
        from ui_qt.matplotlib_theme import apply_display_theme, MatplotlibDisplayTheme, LIGHT_PUBLICATION_THEME
        view, _ = self.make_view()
        dark = MatplotlibDisplayTheme('#202020', '#202020', '#eeeeee', '#cccccc', '#777777', '#555555', '#202020')
        apply_display_theme(view.figure, dark)
        for text in view._artists['window_status']:
            self.assertEqual(text.get_color(), '#eeeeee')
        apply_display_theme(view.figure, LIGHT_PUBLICATION_THEME)
        for text in view._artists['window_status']:
            self.assertEqual(text.get_color(), LIGHT_PUBLICATION_THEME.text)

    def test_map_mouse_sequence_moves_energy_only(self):
        view, _ = self.make_view()
        view.set_selected_b(1)
        view.canvas.draw()
        changes = []
        view.window_center_changed.connect(changes.append)
        for name, x, y in [('button_press_event', 1.63, -.8),
                           ('motion_notify_event', 1.65, .8),
                           ('button_release_event', 1.65, .8)]:
            px, py = view.axes['mcd_map'].transData.transform((x, y))
            event = MouseEvent(name, view.canvas, px, py, button=1)
            view.canvas.callbacks.process(name, event)
        self.assertAlmostEqual(view.state.window_center_ev, 1.65, places=3)
        self.assertEqual(view.state.selected_b_index, 1)
        self.assertEqual(view.state.window_width_mev, 5.)
        self.assertAlmostEqual(changes[-1], 1.65, places=3)

    def test_drag_previews_window_without_full_draws_and_commits_trace_on_release(self):
        view, result = self.make_view()
        result.wavelength_nm = 1239.841984 / result.energy_ev
        view.render(result)
        self.app.processEvents()
        before = [line.get_ydata().copy() for _, line in view._artists['mcd_trace_lines']]
        draws = []
        view.canvas.mpl_connect('draw_event', lambda event: draws.append(event))
        view._on_map_click(SimpleNamespace(inaxes=view.axes['mcd_map'], xdata=1.63,
                                            ydata=0, button=1, key=None))
        for x in (1.631, 1.632, 1.633):
            view._on_map_motion(SimpleNamespace(inaxes=view.axes['mcd_map'], xdata=x))
            self.app.processEvents()
        self.assertEqual(len(draws), 0, 'Window preview must not redraw the full figure')
        self.assertAlmostEqual(view.state.window_center_ev, 1.633)
        for original, (_, line) in zip(before, view._artists['mcd_trace_lines']):
            np.testing.assert_allclose(original, line.get_ydata())
        view._on_map_release(SimpleNamespace())
        self.assertTrue(any(not np.allclose(original, line.get_ydata())
                            for original, (_, line) in zip(before, view._artists['mcd_trace_lines'])))

    def test_spectral_columns_follow_wavelength_order_for_both_right_panels(self):
        view, result = self.make_view()
        result.wavelength_nm = (1239.841984 / result.energy_ev)[::-1]
        for name in ('pair_raw_pos', 'pair_raw_neg', 'pair_corrected_pos',
                     'pair_corrected_neg', 'pair_mcd_corrected', 'pair_mcd_raw'):
            setattr(result, name, getattr(result, name)[:, ::-1])
        view.render(result)
        np.testing.assert_allclose(view._artists['spectrum_lines'][0][0].get_ydata(), _result().pair_raw_pos[0] * 2)
        np.testing.assert_allclose(view._artists['mcd_spectrum_lines'][0][0].get_ydata(), _result().pair_mcd_corrected[0])

    def test_toolbar_pan_and_outside_motion_do_not_change_selection(self):
        view, _ = self.make_view()
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        toolbar = NavigationToolbar2QT(view.canvas, view)
        toolbar.pan()
        view._on_map_click(SimpleNamespace(inaxes=view.axes['mcd_map'], button=1, xdata=1.66, ydata=1, key=None))
        self.assertEqual(view.state.window_center_ev, 1.64)
        self.assertFalse(view._window_drag_active)
        toolbar.pan()
        view._on_map_click(SimpleNamespace(inaxes=view.axes['mcd_map'], button=1, xdata=1.65, ydata=1, key=None))
        view._on_map_motion(SimpleNamespace(inaxes=view.axes['mcd_vs_b'], xdata=-1))
        self.assertAlmostEqual(view.state.window_center_ev, 1.65)

    def test_production_canvas_updates_sidebar_without_changing_b(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window.loaded = LoadedState(mode='MCD', folder='', mcd_result=_result())
        window.mcd_window_center_spin.setValue(1.64)
        window._plot_mode('MCD')
        view = window.mcd_unified_view
        view.set_selected_b(1)
        view.axes['mcd_map'].set_xlim(1.60, 1.68)
        view.axes['mcd_map'].set_ylim(-1., 1.)
        view.canvas.draw()
        for name, x, y in [('button_press_event', 1.63, -.8),
                           ('motion_notify_event', 1.65, .8),
                           ('button_release_event', 1.65, .8)]:
            px, py = view.axes['mcd_map'].transData.transform((x, y))
            view.canvas.callbacks.process(name, MouseEvent(name, view.canvas, px, py, button=1))
        self.assertAlmostEqual(window.mcd_window_center_spin.value(), 1.65, places=3)
        self.assertEqual(view.state.selected_b_index, 1)
        self.assertFalse(window.mcd_controller._mcd_window_dragging)

    def test_queued_center_refresh_waits_for_active_drag(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window.loaded = LoadedState(mode='MCD', folder='', mcd_result=_result())
        window.mcd_window_center_spin.setValue(1.64)
        window._plot_mode('MCD')
        view = window.mcd_unified_view
        view._on_map_click(SimpleNamespace(inaxes=view.axes['mcd_map'], xdata=1.65,
                                            ydata=0, button=1, key=None))
        window.mcd_controller._apply_pending_mcd_center_refresh()
        self.assertAlmostEqual(view.state.window_center_ev, 1.65)
        view._on_map_release(SimpleNamespace())
        window.mcd_controller._apply_pending_mcd_center_refresh()
        self.assertAlmostEqual(window.mcd_window_center_spin.value(), 1.65)
        self.assertAlmostEqual(view.state.window_center_ev, 1.65)

    def test_feature_fold_preserves_values_and_result_switch_restores_energy_axis(self):
        from ui_qt.mcd_unified_page import McdUnifiedControls
        from PySide6.QtWidgets import QToolButton, QWidget
        controls = McdUnifiedControls()
        self.addCleanup(controls.close)
        content = controls.findChild(QWidget, 'mcdFeatureAnalysisContent')
        self.assertTrue(content.isHidden())
        head = controls.feature_expander.findChild(QToolButton)
        controls.feature_search_low_spin.setValue(1.61)
        head.click()
        self.assertFalse(content.isHidden())
        head.click()
        self.assertTrue(content.isHidden())
        self.assertEqual(controls.feature_search_low_spin.value(), 1.61)
        view, _ = self.make_view()
        view.result_panel_combo.setCurrentIndex(1)
        self.assertIn('feature_vs_b', view.axes)
        view.result_panel_combo.setCurrentIndex(0)
        self.assertIn('mcd_spectra', view.axes)
        np.testing.assert_allclose(view.axes['mcd_spectra'].get_xlim(), view.axes['mcd_map'].get_xlim())


if __name__ == '__main__':
    unittest.main()
