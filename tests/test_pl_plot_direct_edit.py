import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from matplotlib.figure import Figure

from ui_qt.quick.peak_analysis_plot import PeakAnalysisPlotMixin


class Plot(PeakAnalysisPlotMixin):
    def __init__(self):
        self._plot_init()
        self._canvas = SimpleNamespace(figure=Figure())
        self._spectrum = self._canvas.figure.subplots()
        self._navigation = SimpleNamespace(mode='')
        self._controls_page = 'branches'
        self._busy = self._pick_seed = False
        self._selected_branch = ''
        self.dataset = dict(kind='PL', key='one', row=0, plot_view='spectra',
            cube=SimpleNamespace(energy=np.array([1., 2., 3.]), Z=np.array([[1., 10., 1.]])),
            branches=[dict(id='p', enabled=True)],
            result=dict(points=[dict(row_index=0, branch_id='p', energy=2.)]))
        self._redraw_dynamic = Mock()
        self._start_seed = Mock()
        self.invoke = Mock()
        self._publish = Mock()

    def setParameter(self, key, value):
        assert key == 'manual.target'
        self._selected_branch = value


def event(plot, energy, height, **kwargs):
    x, y = plot._spectrum.transData.transform((energy, height))
    return SimpleNamespace(button=1, xdata=energy, ydata=height, x=x, y=y,
                           inaxes=plot._spectrum, dblclick=kwargs.get('dblclick', False))


class DirectPeakEditTests(unittest.TestCase):
    def test_select_and_click_do_not_move(self):
        plot = Plot()
        click = event(plot, 2., 10.)
        plot._plot_clicked(click)
        assert plot._selected_branch == 'p'
        plot._peak_release(click)
        assert plot._peak_drag is None
        plot._start_seed.assert_not_called()


    def test_double_click_empty_adds_once_and_navigation_blocks(self):
        plot = Plot()
        click = event(plot, 1., 1., dblclick=True)
        plot._plot_clicked(click)
        plot._start_seed.assert_called_once_with(1.)
        plot._navigation.mode = 'pan'
        plot._plot_clicked(click)
        plot._start_seed.assert_called_once()


    def test_drag_preview_escape_and_delete(self):
        plot = Plot()
        click = event(plot, 2., 10.)
        plot._plot_clicked(click)
        moved = event(plot, 2.1, 10.)
        plot._peak_motion(moved)
        assert plot._peak_drag['energy'] == 2.1
        assert len(plot._peak_preview) == 2
        plot._peak_key(SimpleNamespace(key='escape'))
        assert plot._peak_drag is None and not plot._peak_preview
        plot._peak_key(SimpleNamespace(key='delete'))
        plot.invoke.assert_called_once_with('manual-exclude')


    def test_drag_commits_once_to_current_row(self):
        from unittest.mock import patch
        plot = Plot()
        plot._start = Mock()
        plot._manual_peak_draft = lambda dataset: dict(width_mev=5., move_limit_mev=3., fixed=False)
        click = event(plot, 2., 10.)
        plot._plot_clicked(click)
        moved = event(plot, 2.1, 10.)
        plot._peak_motion(moved)
        with patch('ui_qt.quick.peak_analysis_workflow.capture_request', return_value=plot.dataset), \
                patch('core.pl_manual_peaks.edit_pl_peak', return_value='edited') as edit:
            plot._peak_release(moved)
            plot._peak_release(moved)
            plot._start.assert_called_once()
            action, operation = plot._start.call_args.args
            assert action == 'seed' and plot._start.call_args.kwargs['dataset_key'] == 'one'
            assert operation(None, None) == 'edited'
            assert edit.call_args.args[1:3] == (0, 2.1)
            assert edit.call_args.kwargs['branch_id'] == 'p'

    def test_dirty_and_page_change_cancel_and_block(self):
        plot = Plot()
        click = event(plot, 2., 10.)
        plot._plot_clicked(click)
        plot._peak_motion(event(plot, 2.1, 10.))
        plot._controls_page = 'find'
        plot._refresh_peak_selection()
        self.assertIsNone(plot._peak_drag)
        self.assertFalse(plot._peak_preview)
        self.assertIsNone(plot._peak_selection_line)
        plot._controls_page = 'branches'
        for flag in ('detection_dirty', 'filters_pending'):
            plot.dataset[flag] = True
            plot._pick_seed = True
            plot._plot_clicked(click)
            plot._start_seed.assert_not_called()
            plot.dataset[flag] = False

    def test_escape_exits_pick_mode_and_selection_is_visible(self):
        plot = Plot()
        plot._plot_clicked(event(plot, 2., 10.))
        self.assertIsNotNone(plot._peak_selection_line)
        plot._pick_seed = True
        plot._peak_key(SimpleNamespace(key='escape'))
        self.assertFalse(plot._pick_seed)
        plot._publish.assert_called_once()

    def test_qt_surface_forwards_mouse_and_keys(self):
        from PySide6.QtCore import QEvent, QPointF, Qt
        from PySide6.QtGui import QGuiApplication, QKeyEvent, QMouseEvent
        from ui_qt.quick.figure_canvas import QuickFigureCanvas
        from ui_qt.quick.plot_surface import PlotSurface
        app = QGuiApplication.instance() or QGuiApplication([])
        plot = Plot()
        canvas = QuickFigureCanvas(plot._canvas.figure)
        plot._canvas = canvas
        plot._spectrum.set_xlim(1, 3)
        plot._spectrum.set_ylim(0, 12)
        canvas.draw()
        surface = PlotSurface()
        surface.set_canvas(canvas)
        canvas.mpl_connect('button_press_event', plot._plot_clicked)
        canvas.mpl_connect('button_release_event', plot._peak_release)
        canvas.mpl_connect('key_press_event', plot._peak_key)
        x, y = plot._spectrum.transData.transform((2., 10.))
        pos = QPointF(x / canvas.device_pixel_ratio,
                      (canvas.figure.bbox.height-y) / canvas.device_pixel_ratio)
        press = QMouseEvent(QEvent.Type.MouseButtonPress, pos, pos,
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        release = QMouseEvent(QEvent.Type.MouseButtonRelease, pos, pos,
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        surface.mousePressEvent(press)
        surface.mouseReleaseEvent(release)
        self.assertEqual(plot._selected_branch, 'p')
        key = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
        surface.keyPressEvent(key)
        plot.invoke.assert_called_once_with('manual-exclude')
        surface.set_canvas(None)
        canvas.close()
        app.processEvents()


if __name__ == '__main__':
    unittest.main()
