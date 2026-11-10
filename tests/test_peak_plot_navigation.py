from types import SimpleNamespace

import numpy as np
import unittest
from matplotlib.backend_bases import MouseEvent, NavigationToolbar2, KeyEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ui_qt.quick.peak_analysis_plot import PeakAnalysisPlotMixin


class Plot(PeakAnalysisPlotMixin):
    def __init__(self, log=False, inverted=False):
        self._plot_init()
        self._canvas = FigureCanvasAgg(Figure())
        self._navigation = NavigationToolbar2(self._canvas)
        self._heat, self._spectrum = self._canvas.figure.subplots(2, 1)
        self._heat.set_xlim((10, 1) if inverted else (1, 10))
        self._heat.set_ylim((100, 1) if inverted else (1, 100))
        if log:
            self._heat.set_xscale('log')
            self._heat.set_yscale('log')
        self._spectrum.set_xlim(self._heat.get_xlim())
        self.dataset = {'cube': SimpleNamespace(energy=np.array([1., 10.]), gate=np.array([1., 100.])), 'view': {}, 'scope': 0}
        self._busy = self._building_plot = False
        self._heat.callbacks.connect('xlim_changed', self._remember_view)
        self._heat.callbacks.connect('ylim_changed', self._remember_view)
        self._heat.callbacks.connect('xlim_changed', self._sync_x_limits)
        self._spectrum.callbacks.connect('xlim_changed', self._sync_x_limits)
        self._capture_home_limits()
        self._canvas.mpl_connect('scroll_event', self._plot_scroll)
        self._canvas.mpl_connect('button_press_event', self._middle_press)
        self._canvas.mpl_connect('motion_notify_event', self._middle_motion)
        self._canvas.mpl_connect('button_release_event', self._middle_release)
        self._canvas.mpl_connect('key_press_event', self._peak_key)
        self._canvas.draw()

    def event(self, name, xy, button=None, step=0):
        event = MouseEvent(name, self._canvas, *xy, button=button, step=step)
        event._process()
        return event


class PeakPlotNavigationTests(unittest.TestCase):
    def test_scroll_center_history_and_home(self):
        for log, inverted in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(log=log, inverted=inverted):
                plot = Plot(log, inverted)
                axis = plot._heat
                xy = (int(axis.bbox.x0 + axis.bbox.width * .3), int(axis.bbox.y0 + axis.bbox.height * .4))
                before = axis.transData.inverted().transform(xy)
                old = axis.get_xlim(), axis.get_ylim()
                plot.event('scroll_event', xy, button='up', step=1)
                self.assertTrue(np.allclose(axis.transData.inverted().transform(xy), before))
                self.assertTrue(np.allclose(axis.get_xlim(), plot._spectrum.get_xlim()))
                self.assertTrue(np.allclose(plot.dataset['view']['xlim'], axis.get_xlim()))
                plot._navigation.back()
                self.assertTrue(np.allclose(axis.get_xlim(), old[0]))
                plot.event('scroll_event', xy, button='up', step=1)
                KeyEvent('key_press_event', plot._canvas, key='home')._process()
                self.assertTrue(np.allclose(axis.get_xlim(), old[0]))
                self.assertTrue(np.allclose(axis.get_ylim(), old[1]))

    def test_middle_drag_and_blocked_navigation(self):
        plot = Plot()
        axis = plot._spectrum
        xy = (axis.bbox.x0 + axis.bbox.width / 2, axis.bbox.y0 + axis.bbox.height / 2)
        old = axis.get_xlim()
        plot.event('button_press_event', xy, button=2)
        plot.event('motion_notify_event', (xy[0] + 30, xy[1] + 10), button=2)
        plot.event('button_release_event', (xy[0] + 30, xy[1] + 10), button=2)
        self.assertFalse(np.allclose(axis.get_xlim(), old))
        self.assertIsNone(plot._peak_drag)
        self.assertTrue(np.allclose(axis.get_xlim(), plot._heat.get_xlim()))
        old = axis.get_xlim()
        plot._busy = True
        plot.event('scroll_event', xy, button='up', step=1)
        plot.event('button_press_event', xy, button=2)
        self.assertTrue(np.allclose(axis.get_xlim(), old))
        self.assertIsNone(plot._middle_pan)

    def test_home_restores_full_data_from_initial_crop_and_ignores_colorbar(self):
        plot = Plot()
        plot._heat.set_xlim(3, 5)
        plot._heat.set_ylim(20, 40)
        plot._capture_home_limits()
        colorbar = plot._canvas.figure.add_axes([.92, .2, .03, .6])
        plot._canvas.draw()
        xy = (int(colorbar.bbox.x0 + colorbar.bbox.width / 2), int(colorbar.bbox.y0 + colorbar.bbox.height / 2))
        limits = colorbar.get_xlim(), colorbar.get_ylim()
        plot.event('scroll_event', xy, button='up', step=1)
        self.assertEqual((colorbar.get_xlim(), colorbar.get_ylim()), limits)
        plot._plot_home()
        self.assertTrue(np.allclose(plot._heat.get_xlim(), [1, 10]))
        self.assertTrue(np.allclose(plot._heat.get_ylim(), [1, 100]))
        self.assertTrue(np.allclose(plot._spectrum.get_xlim(), [1, 10]))

    def test_scroll_blocked_during_toolbar_gesture(self):
        plot = Plot()
        axis = plot._heat
        xy = (axis.bbox.x0 + axis.bbox.width / 2, axis.bbox.y0 + axis.bbox.height / 2)
        old = axis.get_xlim(), axis.get_ylim()
        for attribute in ('_zoom_info', '_pan_info'):
            setattr(plot._navigation, attribute, object())
            plot.event('scroll_event', xy, button='up', step=1)
            self.assertEqual((axis.get_xlim(), axis.get_ylim()), old)
            setattr(plot._navigation, attribute, None)

    def test_home_ignores_nonpositive_values_on_log_energy(self):
        plot = Plot(log=True)
        plot.dataset['cube'].energy = np.array([-1., 0., 1., 10.])
        plot._heat.set_xlim(2, 4)
        plot._capture_home_limits()
        plot._plot_home()
        self.assertTrue(np.allclose(plot._heat.get_xlim(), (1, 10)))

    def test_dataset_change_cancels_middle_drag(self):
        plot = Plot()
        axis = plot._heat
        xy = (axis.bbox.x0 + axis.bbox.width / 2, axis.bbox.y0 + axis.bbox.height / 2)
        plot.event('button_press_event', xy, button=2)
        plot.dataset = dict(plot.dataset)
        plot.event('motion_notify_event', (xy[0] + 20, xy[1]), button=2)
        self.assertIsNone(plot._middle_pan)
        self.assertIsNone(plot._middle_pan_dataset)

    def test_qt_surface_forwards_wheel_middle_and_home(self):
        from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
        from PySide6.QtGui import QGuiApplication, QKeyEvent, QMouseEvent, QWheelEvent
        from ui_qt.quick.figure_canvas import QuickFigureCanvas
        from ui_qt.quick.plot_surface import PlotSurface
        app = QGuiApplication.instance() or QGuiApplication([])
        plot = Plot()
        canvas = QuickFigureCanvas(plot._canvas.figure)
        plot._canvas = canvas
        plot._navigation = NavigationToolbar2(canvas)
        for name, callback in [('scroll_event', plot._plot_scroll), ('button_press_event', plot._middle_press),
                               ('motion_notify_event', plot._middle_motion), ('button_release_event', plot._middle_release),
                               ('key_press_event', plot._peak_key)]:
            canvas.mpl_connect(name, callback)
        canvas.draw()
        surface = PlotSurface()
        surface.set_canvas(canvas)
        axis = plot._heat
        x, y = axis.transData.transform((4., 40.))
        pos = QPointF(x / canvas.device_pixel_ratio, (canvas.figure.bbox.height - y) / canvas.device_pixel_ratio)
        original = axis.get_xlim()
        wheel = QWheelEvent(pos, pos, QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        surface.wheelEvent(wheel)
        self.assertFalse(np.allclose(axis.get_xlim(), original))
        zoomed = axis.get_xlim()
        shifted = pos + QPointF(30, 10)
        surface.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, pos, pos, Qt.MouseButton.MiddleButton,
                                            Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier))
        surface.mouseMoveEvent(QMouseEvent(QEvent.Type.MouseMove, shifted, shifted, Qt.MouseButton.NoButton,
                                           Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier))
        surface.mouseReleaseEvent(QMouseEvent(QEvent.Type.MouseButtonRelease, shifted, shifted, Qt.MouseButton.MiddleButton,
                                              Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))
        self.assertFalse(np.allclose(axis.get_xlim(), zoomed))
        self.assertIsNone(plot._peak_drag)
        surface.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Home, Qt.KeyboardModifier.NoModifier))
        self.assertTrue(np.allclose(axis.get_xlim(), original))
        surface.set_canvas(None)
        canvas.close()
        app.processEvents()


if __name__ == '__main__':
    unittest.main()
