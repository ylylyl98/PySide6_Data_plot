"""Partial renderers use Arial for artists created after the last full frame."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np

from core.typography import FONT_FAMILY, matplotlib_font_context
from ui_qt.axes_region_blitter import AxesRegionBlitter
from ui_qt.drr_gate_blitter import DrrGateBlitter


class PartialPlotTypographyTests(unittest.TestCase):
    def figure(self):
        figure = Figure(figsize=(6, 5), dpi=100)
        canvas = FigureCanvasAgg(figure)
        heat, spectrum = figure.subplots(2, 1)
        gate = heat.axhline(.5)
        spectrum.plot([0, 1], [0, 1])
        with matplotlib_font_context(figure):
            canvas.draw()
        return figure, canvas, heat, spectrum, gate

    def probe(self, axis):
        return axis.text(.5, .5, r"New $\Delta E+\mathbf{x}$", fontfamily="serif",
                         math_fontfamily="cm", fontsize=12, fontweight="bold")

    def assert_partial_arial(self, canvas, probe, draw):
        """Observe real rendering, including the rc settings mathtext reads."""
        rendered = []
        original = Text.draw
        weight = probe.get_fontweight()

        def record(text, renderer):
            if text is probe:
                rendered.append(text.get_text())
                self.assertEqual(text.get_fontfamily()[0], FONT_FAMILY)
                self.assertEqual(text.get_math_fontfamily(), "custom")
                self.assertEqual(mpl.rcParams["mathtext.fontset"], "custom")
                self.assertTrue(mpl.rcParams["mathtext.rm"].startswith(FONT_FAMILY))
            return original(text, renderer)

        previous = dict(mpl.rcParams)
        with patch.object(Text, "draw", new=record), patch.object(canvas, "draw", wraps=canvas.draw) as full:
            draw()
            full.assert_not_called()
        self.assertTrue(rendered, "new math annotation must actually be painted")
        self.assertEqual(probe.get_fontweight(), weight)
        self.assertEqual(dict(mpl.rcParams), previous)

    def test_axes_region_blit_paints_new_dynamic_text_in_arial(self):
        figure, canvas, heat, spectrum, gate = self.figure()
        helper = AxesRegionBlitter(canvas)
        line = spectrum.lines[0]
        helper.configure(spectrum, (line,))
        with matplotlib_font_context(figure):
            canvas.draw()
        with mpl.rc_context({"font.family": "serif", "mathtext.fontset": "cm"}):
            probe = self.probe(spectrum)
            probe.set_animated(True)
            helper.artists = (*helper.artists, probe)
            def draw():
                self.assertTrue(helper.draw())
            self.assert_partial_arial(canvas, probe, draw)
        helper.disconnect()

    def test_gate_blit_paints_rebuilt_spectrum_and_new_ticks_in_arial(self):
        figure, canvas, heat, spectrum, gate = self.figure()
        helper = DrrGateBlitter(canvas)
        helper.configure((spectrum,), (gate,))
        with matplotlib_font_context(figure):
            canvas.draw()
        with mpl.rc_context({"font.family": "serif", "mathtext.fontset": "cm"}):
            spectrum.clear()
            spectrum.plot([0, 1], [0, 1e7])
            spectrum.set_title(r"Recreated spectrum $\alpha$")
            probe = self.probe(spectrum)
            def draw():
                self.assertTrue(helper.draw())
            self.assert_partial_arial(canvas, probe, draw)
            self.assertTrue(all(t.get_fontfamily()[0] == FONT_FAMILY
                                for t in spectrum.findobj(match=Text)))
        helper.disconnect()

    def test_quick_partial_frames_paint_axes_recreated_after_full_draw(self):
        from ui_qt.quick.power_plot import PowerPlotMixin
        from ui_qt.quick.peak_analysis_plot import PeakAnalysisPlotMixin
        from ui_qt.quick.compare_bridge import CompareBridge
        from ui_qt.quick.drr_analysis_plot import DrrAnalysisPlotMixin

        for draw in (PowerPlotMixin._draw_dynamic, PeakAnalysisPlotMixin._draw_dynamic,
                     CompareBridge._draw_dynamic, DrrAnalysisPlotMixin._draw_dynamic):
            with self.subTest(renderer=draw.__qualname__):
                figure, canvas, heat, spectrum, gate = self.figure()
                figure.delaxes(spectrum)
                with mpl.rc_context({"font.family": "serif", "mathtext.fontset": "cm"}):
                    spectrum = figure.add_subplot(212)
                    spectrum.plot([0, 1], [0, 2])
                    spectrum.set_animated(True)
                    probe = self.probe(spectrum)
                    owner = SimpleNamespace(_canvas=canvas, _spectrum=spectrum, _residual=None,
                        _gate_lines={"sample": gate}, _maps={"sample": heat},
                        _gate_line=gate, _heat=heat,
                        _linecuts=[(None, heat, spectrum, gate, None)],
                        _peak_selection=[], _p2p_inspection=None, _overlay_artists=[])
                    self.assert_partial_arial(canvas, probe, lambda: draw(owner))

    def test_colorbar_partial_draw_uses_arial_for_new_colorbar_text(self):
        from ui_qt.drr_color_redraw import ColorFrame
        figure = Figure(figsize=(7, 6), dpi=100)
        canvas = FigureCanvasAgg(figure)
        heat = figure.add_axes((.1, .55, .6, .35))
        host = figure.add_axes((.76, .55, .04, .35))
        spectrum = figure.add_axes((.1, .1, .6, .25))
        mesh = heat.imshow(np.arange(16).reshape(4, 4))
        colorbar = figure.colorbar(mesh, cax=host)
        gate = heat.axhline(1)
        spectrum.plot([0, 1], [0, 1])
        helper = DrrGateBlitter(canvas)
        helper.configure((spectrum,), (gate,), ((heat, host),))
        with matplotlib_font_context(figure):
            canvas.draw()
        self.assertIsInstance(helper._color_frame, ColorFrame)
        token = helper.prepare_color_update()
        self.assertIsNotNone(token)
        with mpl.rc_context({"font.family": "serif", "mathtext.fontset": "cm"}):
            mesh.set_clim(0, 100000)
            colorbar.update_normal(mesh)
            probe = host.text(.5, .5, r"New $\Delta$", fontfamily="serif", math_fontfamily="cm")
            def draw():
                self.assertTrue(helper.draw_colors(((heat, host),), token))
            self.assert_partial_arial(canvas, probe, draw)
        helper.disconnect()


if __name__ == "__main__":
    unittest.main()
