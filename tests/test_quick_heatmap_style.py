"""Scientific values and compact bar labels survive native color-only edits."""
import unittest

import numpy as np
from matplotlib.figure import Figure

from core.loader import DataCube
from core.plotting import HeatmapParams, SplitColorScale, plot_heatmap
from ui_qt.inspection_colorbar import add_inspection_colorbar
from ui_qt.quick.heatmap_style import update_heatmap_colors


class QuickHeatmapStyleTests(unittest.TestCase):
    def setUp(self):
        self.cube = DataCube(np.linspace(1., 2., 5), np.arange(3.),
            np.arange(15., dtype=float).reshape(3, 5) - 3., "Gate", "Sample", "counts")
        self.params = HeatmapParams("Sample", "Energy", "Gate", "counts", 0., 5., (1., 2.), (0., 2.), clip_outliers=True)
        self.figure = Figure()
        self.axis = self.figure.add_subplot()

    def test_raising_vmax_restores_source_values_and_preserves_mesh_and_zoom(self):
        render = plot_heatmap(self.axis, self.cube, self.params)
        original = render.primary
        self.axis.set_xlim(1.2, 1.6)
        self.params.vmax = 12.
        update_heatmap_colors(render, self.cube, self.params)
        self.assertIs(render.primary, original)
        self.assertEqual(self.axis.get_xlim(), (1.2, 1.6))
        np.testing.assert_array_equal(original.get_array(), self.cube.Z)
        self.assertEqual(original.get_clim(), (0., 12.))

    def test_split_bounds_and_cmap_update_each_original_region(self):
        self.params.split_scale = SplitColorScale(1.5, 0., 4., 0., 5.)
        render = plot_heatmap(self.axis, self.cube, self.params)
        self.params.split_scale = SplitColorScale(1.5, -3., 20., -2., 21.)
        self.params.cmap = "magma"
        update_heatmap_colors(render, self.cube, self.params)
        self.assertEqual([image.get_clim() for image in render.images], [(-3., 20.), (-2., 21.)])
        self.assertTrue(all(image.cmap.name == "magma" for image in render.images))
        np.testing.assert_array_equal(np.concatenate([image.get_array() for image in render.images], axis=1), self.cube.Z)

    def test_small_positive_log_limits_match_inputs_in_render_and_recolor(self):
        self.params.log_scale = True
        self.params.vmin, self.params.vmax = 1e-16, 1e-14
        render = plot_heatmap(self.axis, self.cube, self.params)
        self.assertEqual(render.primary.get_clim(), (1e-16, 1e-14))
        self.params.vmin, self.params.vmax = 2e-17, 3e-15
        update_heatmap_colors(render, self.cube, self.params)
        self.assertEqual(render.primary.get_clim(), (2e-17, 3e-15))

    def test_vp_range_crossing_zero_keeps_compact_colorbar_endpoints(self):
        self.params.center_zero = True
        self.params.vmin, self.params.vmax = -.8, .9
        render = plot_heatmap(self.axis, self.cube, self.params)
        host = add_inspection_colorbar(self.figure, render, self.axis, label="VP")
        bar = render.primary.colorbar
        locator, formatter = bar.locator, bar.formatter
        self.params.vmin = .1
        update_heatmap_colors(render, self.cube, self.params)
        self.assertIs(bar.locator, locator)
        self.assertIs(bar.formatter, formatter)
        self.assertEqual(list(bar.get_ticks()), [.1, .9])
        self.assertEqual(len(host.child_axes), 1)


if __name__ == "__main__":
    unittest.main()
