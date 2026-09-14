import unittest
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ui_qt.axes_region_blitter import AxesRegionBlitter
from ui_qt.controllers_pl import PlController


class RegionBlitterTests(unittest.TestCase):
    def test_reuses_artist_and_updates_pixels(self):
        figure = Figure(figsize=(4, 3), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        line, = axis.plot([0, 1], [0, 1])
        axis.set_title("Spectrum")
        canvas.draw()
        helper = AxesRegionBlitter(canvas)
        helper.configure(axis, [line])
        helper.restore_interactive_drawing()
        identity = line
        line.set_ydata([1, 0])
        self.assertTrue(helper.draw())
        self.assertIs(axis.lines[0], identity)
        self.assertEqual(list(line.get_ydata()), [1, 0])
        local = bytes(canvas.buffer_rgba())
        helper.prepare_full_redraw()
        canvas.draw()
        full = bytes(canvas.buffer_rgba())
        self.assertEqual(len(local), len(full))
        self.assertLessEqual(sum(a != b for a, b in zip(local, full)), 8)

    def test_invalidation_falls_back_and_disconnect_restores_flags(self):
        figure = Figure(figsize=(3, 2), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        line, = axis.plot([0, 1], [0, 1])
        helper = AxesRegionBlitter(canvas)
        helper.configure(axis, [line])
        self.assertTrue(line.get_animated())
        helper.invalidate()
        self.assertFalse(helper.draw())
        helper.disconnect()
        self.assertFalse(line.get_animated())

    def test_full_redraw_handles_title_limits_and_resize(self):
        figure = Figure(figsize=(3, 2), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        line, = axis.plot([0, 1], [0, 1])
        axis.set_title("Before")
        canvas.draw()
        helper = AxesRegionBlitter(canvas)
        helper.configure(axis, [line])
        helper.prepare_full_redraw()
        axis.set_title("After")
        axis.set_ylim(-1, 2)
        figure.set_size_inches(5, 3)
        canvas.draw()
        helper.restore_interactive_drawing()
        self.assertEqual(axis.get_title(), "After")
        self.assertEqual(tuple(round(v, 6) for v in axis.get_ylim()), (-1.0, 2.0))

    def test_prepare_draw_restore_move_captures_background_before_dynamic_line(self):
        figure = Figure(figsize=(4, 3), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        line, = axis.plot([0, 1], [0, 1], color="tab:blue", linewidth=5)
        canvas.draw()
        helper = AxesRegionBlitter(canvas)
        helper.configure(axis, [line])
        helper.prepare_full_redraw()
        line.set_ydata([1, 0])
        canvas.draw()
        helper.restore_interactive_drawing()

        # Restore only the captured static region and compare it with a clean
        # full render where the dynamic line is disabled.
        canvas.restore_region(helper._background)
        canvas.blit(axis.bbox)
        restored = np.asarray(canvas.buffer_rgba()).copy()
        line.set_animated(True)
        helper._full_redraw_prepared = True
        canvas.draw()
        helper._full_redraw_prepared = False
        expected = np.asarray(canvas.buffer_rgba()).copy()
        x0, y0, width, height = (int(round(v)) for v in axis.bbox.bounds)
        y0 = restored.shape[0] - y0 - height
        crop = (slice(y0, y0 + height), slice(x0, x0 + width))
        self.assertLessEqual(np.count_nonzero(restored[crop] != expected[crop]), 8)

    def test_real_pl_controller_second_gate_uses_blit_without_canvas_draw(self):
        figure = Figure(figsize=(4, 3), dpi=100)
        canvas = FigureCanvasAgg(figure)
        heatmap = figure.add_subplot(211)
        spectrum = figure.add_subplot(212)
        line, = spectrum.plot([0, 1], [0, 1])
        gate = heatmap.axhline(0.5)
        owner = SimpleNamespace(canvas=canvas, _pl_spectrum_ax=spectrum,
                                _pl_heatmap_ax=heatmap, _pl_spectrum_line=line,
                                _pl_gate_line=gate, _pl_region_blitters=None)
        controller = PlController(owner)
        calls = []
        original_draw = canvas.draw
        canvas.draw = lambda *args, **kwargs: (calls.append(1), original_draw(*args, **kwargs))[1]
        controller._draw_pl_regions()
        first = len(calls)
        line.set_ydata([1, 0]); gate.set_ydata([0.7, 0.7])
        controller._draw_pl_regions()
        self.assertEqual(first, 2)  # first capture per dynamic axes
        self.assertEqual(len(calls), first)

    def test_real_pl_canvas_draw_keeps_dynamic_curve_and_rebinds_same_bbox_source(self):
        figure = Figure(figsize=(4, 3), dpi=100)
        canvas = FigureCanvasAgg(figure)
        heatmap = figure.add_subplot(211)
        spectrum = figure.add_subplot(212)
        line, = spectrum.plot([0, 1], [0, 1], color="tab:blue")
        gate = heatmap.axhline(0.5)
        owner = SimpleNamespace(canvas=canvas, _pl_spectrum_ax=spectrum,
                                _pl_heatmap_ax=heatmap, _pl_spectrum_line=line,
                                _pl_gate_line=gate, _pl_region_blitters=None)
        controller = PlController(owner)
        controller._draw_pl_regions()
        line.set_ydata([1, 0])
        spectrum.set_title("changed")
        canvas.draw()  # ordinary production draw event
        pixels_after_draw = bytes(canvas.buffer_rgba())
        frame = np.frombuffer(pixels_after_draw, dtype=np.uint8).reshape((-1, 4))
        # The animated spectrum remains visible after a normal full draw;
        # checking colored pixels catches the old disappearing-curve failure.
        self.assertGreater(np.count_nonzero((frame[:, 2] > frame[:, 0] + 20)), 100)

        # A newly loaded source can create a new Axes at the same pixel bbox.
        replacement = figure.add_axes(spectrum.get_position())
        replacement_line, = replacement.plot([0, 1], [0.25, 0.75], color="tab:red")
        owner._pl_spectrum_ax = replacement
        owner._pl_spectrum_line = replacement_line
        controller._draw_pl_regions()
        self.assertIs(controller._pl_region_blitters[0].axes, replacement)
        self.assertIs(controller._pl_region_blitters[0].artists[0], replacement_line)
        self.assertIsNot(controller._pl_region_blitters[0].artists[0], line)

    def test_drr_paired_helpers_keep_distinct_artist_ownership(self):
        from ui_qt.controllers_drr import DrrController
        figure = Figure(figsize=(4, 3), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        raw, = axis.plot([0, 1], [0.1, 0.8], color="tab:blue")
        second, = axis.plot([0, 1], [0.8, 0.2], color="tab:orange")
        owner = type("Owner", (), {})()
        controller = DrrController.__new__(DrrController)
        object.__setattr__(controller, "_owner", owner)
        object.__setattr__(controller, "canvas", canvas)
        object.__setattr__(controller, "_drr_spectrum_axes", {"raw": axis, "second": axis})
        object.__setattr__(controller, "_drr_spectrum_lines", {"raw": raw, "second": second})
        object.__setattr__(controller, "_drr_gate_lines", {})
        object.__setattr__(controller, "_drr_region_blitters", {})
        controller._draw_drr_regions()
        self.assertIs(controller._drr_region_blitters["spectrum:raw"].artists[0], raw)
        self.assertIs(controller._drr_region_blitters["spectrum:second"].artists[0], second)
        self.assertIsNot(controller._drr_region_blitters["spectrum:raw"],
                         controller._drr_region_blitters["spectrum:second"])
        raw.set_ydata([0.2, 0.9]); second.set_ydata([0.7, 0.1])
        controller._draw_drr_regions()
        self.assertGreater(np.count_nonzero(np.asarray(canvas.buffer_rgba())[:, :, :3]), 100)

    def test_power_helper_rebinds_same_bbox_artist(self):
        from ui_qt.controllers_power import PowerController
        figure = Figure(figsize=(4, 3), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        old, = axis.plot([0, 1], [0.1, 0.8])
        controller = PowerController.__new__(PowerController)
        object.__setattr__(controller, "_owner", type("Owner", (), {})())
        object.__setattr__(controller, "canvas", canvas)
        object.__setattr__(controller, "_power_spectrum_ax", axis)
        object.__setattr__(controller, "_power_spectrum_lines", {"old": old})
        object.__setattr__(controller, "_power_gate_lines", {})
        object.__setattr__(controller, "_power_region_blitters", {})
        controller._draw_power_regions()
        replacement, = axis.plot([0, 1], [0.8, 0.2], color="tab:red")
        object.__setattr__(controller, "_power_spectrum_lines", {"new": replacement})
        controller._draw_power_regions()
        helper = next(value for value in controller._power_region_blitters.values()
                      if value.artists == (replacement,))
        self.assertIs(helper.artists[0], replacement)

    def test_power_shared_axes_uses_one_helper_for_two_moving_lines(self):
        from ui_qt.controllers_power import PowerController
        figure = Figure(figsize=(4, 3), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        first, = axis.plot([0, 1], [0.1, 0.8], color="tab:blue")
        second, = axis.plot([0, 1], [0.8, 0.2], color="tab:orange")
        controller = PowerController.__new__(PowerController)
        object.__setattr__(controller, "_owner", SimpleNamespace())
        object.__setattr__(controller, "canvas", canvas)
        object.__setattr__(controller, "_power_spectrum_ax", axis)
        object.__setattr__(controller, "_power_spectrum_lines", {"first": first, "second": second})
        object.__setattr__(controller, "_power_gate_lines", {})
        object.__setattr__(controller, "_power_region_blitters", {})
        controller._draw_power_regions()
        self.assertEqual(len(controller._power_region_blitters), 1)
        self.assertEqual(set(controller._power_region_blitters.values()).pop().artists,
                         (first, second))
        first.set_ydata([0.2, 0.9]); second.set_ydata([0.7, 0.1])
        controller._draw_power_regions()
        local = np.asarray(canvas.buffer_rgba()).copy()
        for artist in (first, second): artist.set_animated(False)
        helper = next(iter(controller._power_region_blitters.values()))
        helper._full_redraw_prepared = True
        canvas.draw()
        helper._full_redraw_prepared = False
        full = np.asarray(canvas.buffer_rgba()).copy()
        # Agg's region restore can differ at a handful of antialiased edge
        # channels; the shared-axes corruption previously changed thousands.
        self.assertLessEqual(np.count_nonzero(local != full), 64)


if __name__ == "__main__":
    unittest.main()
