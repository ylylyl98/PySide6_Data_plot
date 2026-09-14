"""Guard the single-pass DRR PNG export against layout/image regressions."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from matplotlib.figure import Figure

from core import export
from core.loader import DataCube
from core.plotting import HeatmapParams, SplitColorScale


class DrrPngRenderTests(unittest.TestCase):
    def test_heatmap_png_uses_fixed_canvas_with_one_figure_draw(self):
        cube = DataCube(
            np.array([1., 1.3, 1.8, 2.4]), np.array([3., 1., 0.2]),
            np.array([[1., 2., np.nan, 4.], [2., 3., 4., 1.], [4., 1., 2., 3.]]),
            "Gate", "test", "DR/R",
        )
        params = HeatmapParams(
            title="A long measurement title that wraps across multiple lines",
            xlabel="Photon energy (eV)", ylabel="Gate (V)", cbar_label="DR/R",
            vmin=0., vmax=4., xlim=(1.1, 2.3), ylim=(3., 0.2),
        )
        variants = [params, replace(params, y_axis_log=True),
                    replace(params, log_scale=True, vmin=0.1),
                    replace(params, cbar_label="d2(DR/R)/dE2", center_zero=True, vmin=-4.),
                    replace(params, split_scale=SplitColorScale(1.5, 0., 2., 1., 4.)),
                    replace(params, title="", xlim=(2.5, 0.8), ylim=(0.1, 3.5))]
        with tempfile.TemporaryDirectory() as directory:
            for drr in (True, False):
                for index, variant in enumerate(variants):
                    with self.subTest(drr=drr, index=index):
                        actual = Path(directory) / f"actual-{drr}-{index}.png"
                        draws = []
                        original = Figure.draw

                        def counted(figure, renderer):
                            draws.append(figure)
                            return original(figure, renderer)

                        with patch.object(Figure, "draw", counted):
                            export._save_heatmap_png(actual, cube, variant, drr=drr)
                        with Image.open(actual) as result:
                            self.assertEqual(
                                result.size,
                                (1200, 930),
                                "Heatmap exports should retain the common canvas",
                            )
                        self.assertEqual(len(draws), 1, "Export should render the figure only once")


if __name__ == "__main__":
    unittest.main()
