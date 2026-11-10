"""Regression tests for safe heatmap array reuse."""
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure

from core.loader import DataCube
from core.plotting import HeatmapParams, SplitColorScale, plot_heatmap
from ui_qt.quick.heatmap_style import update_heatmap_colors


class HeatmapArrayReuseTests(unittest.TestCase):
    def setUp(self):
        self.cube = DataCube(np.linspace(1., 2., 5), np.arange(3.),
            np.arange(15., dtype=float).reshape(3, 5) - 3., "Gate", "Sample", "counts")
        self.params = HeatmapParams("Sample", "Energy", "Gate", "counts", 0., 5., (1., 2.), (0., 2.), clip_outliers=True)
        self.figure = Figure()
        self.axis = self.figure.add_subplot()

    def test_same_colormap_does_not_trigger_colorbar_rebuild(self):
        render = plot_heatmap(self.axis, self.cube, self.params)
        with patch.object(render.primary, "set_cmap", wraps=render.primary.set_cmap) as setter:
            update_heatmap_colors(render, self.cube, self.params)
            setter.assert_not_called()
            self.params.cmap = "magma"
            update_heatmap_colors(render, self.cube, self.params)
            setter.assert_called_once()
        self.assertEqual(render.primary.get_cmap().name, "magma")

    def test_unchanged_data_skips_copy_but_in_place_mutation_is_applied(self):
        self.params.clip_outliers = False
        render = plot_heatmap(self.axis, self.cube, self.params)
        with patch.object(render.primary, "set_array", wraps=render.primary.set_array) as setter:
            self.params.vmax = 8.
            update_heatmap_colors(render, self.cube, self.params)
            setter.assert_not_called()
            self.cube.Z[0, 0] = 123.
            update_heatmap_colors(render, self.cube, self.params)
            setter.assert_called_once()
        np.testing.assert_array_equal(render.primary.get_array(), self.cube.Z)

    def test_split_reuses_only_unchanged_region_and_restores_clipped_values(self):
        self.params.split_scale = SplitColorScale(1.5, 0., 4., 0., 5.)
        render = plot_heatmap(self.axis, self.cube, self.params)
        with patch.object(render.images[0], "set_array", wraps=render.images[0].set_array) as left, \
             patch.object(render.images[1], "set_array", wraps=render.images[1].set_array) as right:
            self.params.split_scale = SplitColorScale(1.5, 0., 4., 0., 20.)
            update_heatmap_colors(render, self.cube, self.params)
            left.assert_not_called()
            right.assert_called_once()
        np.testing.assert_array_equal(render.images[1].get_array(), self.cube.Z[:, 2:])

    def test_masked_values_are_rebuilt(self):
        self.params.clip_outliers = False
        self.cube.Z[0, 0] = np.nan
        render = plot_heatmap(self.axis, self.cube, self.params)
        self.cube.Z[0, 0] = 3.
        update_heatmap_colors(render, self.cube, self.params)
        self.assertFalse(np.ma.is_masked(render.primary.get_array()))
        np.testing.assert_array_equal(render.primary.get_array(), self.cube.Z)
