"""Color-drag preview is temporary; normal and exported renders stay exact."""
from io import BytesIO
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.image import PcolorImage

from ui_qt.quick.raster_heatmaps import screen_heatmaps


class ColorPreviewTests(unittest.TestCase):
    def test_preview_is_coarse_and_restores_exact_screen_and_export(self):
        figure = Figure(figsize=(4, 3), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_axes([.1, .1, .8, .8])
        x = np.linspace(0, 1, 801) ** 1.2
        y = np.linspace(0, 1, 401) ** 1.1
        values = np.ma.array(np.arange(800 * 400).reshape(400, 800), mask=False)
        values.mask[100:120, 100:120] = True
        mesh = axis.pcolormesh(x, y, values)
        source, coordinates = mesh.get_array().copy(), mesh.get_coordinates().copy()
        with screen_heatmaps(canvas):
            canvas.draw()
        baseline = np.asarray(canvas.buffer_rgba()).copy()
        before = BytesIO()
        figure.savefig(before, format="png")

        canvas.color_range_preview = True
        with patch("matplotlib.image.PcolorImage", wraps=PcolorImage) as image:
            with screen_heatmaps(canvas):
                canvas.draw()
            args = image.call_args.args
            self.assertLess(args[3].size, source.size // 4)
            self.assertEqual(len(args[1]), args[3].shape[1] + 1)
            self.assertEqual(len(args[2]), args[3].shape[0] + 1)
            self.assertEqual((args[1][0], args[1][-1]), (x[0], x[-1]))
            self.assertEqual((args[2][0], args[2][-1]), (y[0], y[-1]))
        np.testing.assert_array_equal(mesh.get_array(), source)
        np.testing.assert_array_equal(mesh.get_coordinates(), coordinates)
        exported = BytesIO()
        with screen_heatmaps(canvas):
            figure.savefig(exported, format="png")
        self.assertEqual(exported.getvalue(), before.getvalue())

        canvas.color_range_preview = False
        with screen_heatmaps(canvas):
            canvas.draw()
        np.testing.assert_array_equal(np.asarray(canvas.buffer_rgba()), baseline)


if __name__ == "__main__":
    unittest.main()
