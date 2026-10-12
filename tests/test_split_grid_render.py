"""Split meshes must preserve original edges, pixels and point resolution."""
import unittest
from dataclasses import replace

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure

from core.loader import DataCube
from core.plotting import HeatmapParams, SplitColorScale, plot_heatmap, _norm_from_bounds


class SplitGridRenderTests(unittest.TestCase):
    def fixture(self, log=False, triple=True):
        cube = DataCube(np.array([1., 1.3, 2., 3.2, 4., 6.]), np.array([1., 2., 7.]),
                        np.array([[-2., .1, 9., 0., 50., 1.], [1., np.nan, 3., 4., 5., 6.],
                                  [2., 4., 6., 8., 10., 12.]]), 'Gate', 'Grid', 'Signal')
        bounds = (.01, 10.) if log else (-3., 13.)
        split = SplitColorScale(1.65, *bounds, *bounds, split_x2=3.6 if triple else None,
                                middle_vmin=bounds[0], middle_vmax=bounds[1])
        params = HeatmapParams('Grid', 'Energy', 'Gate', 'Signal', *bounds, (1.1, 5.8), (1., 7.),
                               split_scale=split, log_scale=log, y_axis_log=log,
                               center_zero=not log, clip_outliers=True)
        return cube, params

    def test_sliced_mesh_pixels_match_full_masked_reference(self):
        # The reference draws the former full grids with disjoint masks.
        # Different center spacing requires slicing original edges, not recomputing them.
        for log in (False, True):
            for triple in (False, True):
                with self.subTest(log=log, triple=triple):
                    cube, params = self.fixture(log, triple)
                    fig = Figure(figsize=(5., 3.), dpi=80)
                    canvas = FigureCanvasAgg(fig)
                    ax = fig.subplots()
                    render = plot_heatmap(ax, cube, params)
                    canvas.draw()
                    actual = np.asarray(canvas.buffer_rgba()).copy()
                    ref = Figure(figsize=(5., 3.), dpi=80)
                    refcanvas = FigureCanvasAgg(ref)
                    refax = ref.subplots()
                    whole = plot_heatmap(refax, cube, replace(params, split_scale=None)).primary
                    coords = whole.get_coordinates()
                    whole.remove()
                    cuts = [0, 2, 4, 6] if triple else [0, 2, 6]
                    for a, b, (lo, hi) in zip(cuts, cuts[1:], params.split_scale.bounds):
                        mask = np.ones(cube.Z.shape, bool)
                        mask[:, a:b] = False
                        values = np.ma.array(np.minimum(cube.Z, hi), mask=mask)
                        refax.pcolormesh(coords[0, :, 0], coords[:, 0, 1], values,
                            shading='flat', cmap=params.cmap,
                            norm=_norm_from_bounds(values, vmin=lo, vmax=hi,
                                log_scale=log, center_zero=not log))
                    for x in ([1.65, 3.6] if triple else [1.65]):
                        refax.axvline(x, color='#202020', linewidth=.9, linestyle='--', alpha=.8, zorder=15)
                    refcanvas.draw()
                    np.testing.assert_array_equal(actual, np.asarray(refcanvas.buffer_rgba()))
                    self.assertEqual(sum(m.get_array().size for m in render.images), 18)

    def test_cursor_returns_original_narrow_peak_from_its_region(self):
        cube, params = self.fixture()
        params.clip_outliers = False
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.subplots()
        render = plot_heatmap(ax, cube, params)
        canvas.draw()
        x, y = ax.transData.transform((4., 1.))
        event = MouseEvent('motion_notify_event', canvas, x, y)
        values = [m.get_cursor_data(event) for m in render.images]
        self.assertEqual([float(v) for v in values if v is not None], [50.])


if __name__ == '__main__':
    unittest.main()
