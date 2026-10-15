import unittest
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from core.loader import DataCube
from core.plotting import HeatmapParams, SplitColorScale, plot_drr
from ui_qt.drr_viewport import visible_drr_meshes


class DrrViewportTests(unittest.TestCase):
    def test_visible_cells_match_full_render_and_model_is_restored(self):
        for reverse in (False, True):
            for split in (None, SplitColorScale(1.45, -1, 1, -2, 2)):
                with self.subTest(reverse=reverse, split=split is not None):
                    figure = Figure(figsize=(5, 4), dpi=100)
                    canvas = FigureCanvasAgg(figure)
                    axis = figure.subplots()
                    x = np.linspace(1, 1.4, 1340)**2
                    y = np.linspace(-1, 1, 361)**3
                    z = np.sin(x[None, :]*32 + y[:, None]*5)
                    z[150, 671] = np.nan
                    if reverse:
                        y, z = y[::-1], z[::-1]
                    cube = DataCube(x, y, z, 'Gate', 'test', 'DR/R')
                    params = HeatmapParams('test', 'Energy', 'Gate', 'DR/R', -1, 1,
                                           (1.3, 1.6), (-.2, .4), split_scale=split)
                    render = plot_drr(axis, cube, params)
                    original = [(mesh.get_array(), mesh.get_coordinates()) for mesh in render.images]
                    canvas.draw()
                    expected = np.asarray(canvas.buffer_rgba()).copy()
                    with visible_drr_meshes(canvas):
                        self.assertLess(sum(m.get_array().size for m in render.images), z.size)
                        canvas.draw()
                    np.testing.assert_array_equal(expected, np.asarray(canvas.buffer_rgba()))
                    for mesh, (values, coords) in zip(render.images, original):
                        self.assertIs(mesh.get_array(), values)
                        self.assertIs(mesh.get_coordinates(), coords)

    def test_exception_and_saving_preserve_full_arrays(self):
        figure = Figure(); canvas = FigureCanvasAgg(figure); axis = figure.subplots()
        cube = DataCube(np.arange(100.), np.arange(100.), np.ones((100, 100)), 'Gate', '', '')
        params = HeatmapParams('', '', '', '', 0, 2, (20, 40), (20, 40))
        mesh = plot_drr(axis, cube, params).primary
        values, coords = mesh.get_array(), mesh.get_coordinates()
        with self.assertRaisesRegex(RuntimeError, 'draw failed'):
            with visible_drr_meshes(canvas):
                raise RuntimeError('draw failed')
        self.assertIs(mesh.get_array(), values)
        self.assertIs(mesh.get_coordinates(), coords)
        canvas._is_saving = True
        try:
            with visible_drr_meshes(canvas):
                self.assertIs(mesh.get_array(), values)
        finally:
            canvas._is_saving = False
