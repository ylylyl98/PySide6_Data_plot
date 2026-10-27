"""Spectrum captions remain within the plot after gate-title updates."""
import unittest

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ui_qt.spectrum_caption import set_spectrum_title


class SpectrumCaptionTests(unittest.TestCase):
    def test_gate_title_stays_inside_axes_and_preserves_title_identity(self):
        for size in ((3, 1.6), (6, 3)):
            with self.subTest(size=size):
                figure = Figure(figsize=size, dpi=100)
                canvas = FigureCanvasAgg(figure)
                axis = figure.add_subplot(111)
                axis.plot([1.7, 2.2, 2.7], [0, 1, .2])
                limits = (axis.get_xlim(), axis.get_ylim())
                title = axis.title
                for text in ("Spectrum @ -0.75 V", "Spectrum @ 1.25 V"):
                    set_spectrum_title(axis, text)
                    canvas.draw()
                    bounds = title.get_window_extent(canvas.get_renderer())
                    self.assertIs(axis.title, title)
                    self.assertEqual(axis.get_title(), text)
                    self.assertGreater(bounds.x0, axis.bbox.x0)
                    self.assertLess(bounds.x0, axis.bbox.x0 + .1 * axis.bbox.width)
                    self.assertLessEqual(bounds.x1, axis.bbox.x1)
                    self.assertLessEqual(bounds.y1, axis.bbox.y1)
                    self.assertGreater(bounds.y0, axis.bbox.y0)
                    self.assertGreater(bounds.y1, axis.bbox.y0 + .8 * axis.bbox.height)
                    self.assertEqual((axis.get_xlim(), axis.get_ylim()), limits)


if __name__ == "__main__":
    unittest.main()
