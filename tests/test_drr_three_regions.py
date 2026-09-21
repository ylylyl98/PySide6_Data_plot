import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from core.loader import DataCube
from core.plotting import HeatmapParams, SplitColorScale, plot_heatmap
from core.export import _build_streamlit_style_heatmap_fig, EXPORT_FIGSIZE, EXPORT_AXES_RECT


def example():
    cube = DataCube(np.arange(1., 7.), np.arange(3.),
                    np.array([[1, 2, 10, 20, 100, 200]] * 3, float), 'Gate', 'Three regions', 'DR/R')
    params = HeatmapParams('Three regions', 'Energy (eV)', 'Gate', 'DR/R', 0, 200,
        (1, 6), (0, 2), split_scale=SplitColorScale(2.5, 0, 3, 0, 300,
            split_x2=4.5, middle_vmin=0, middle_vmax=30))
    return cube, params


class ThreeRegionTests(unittest.TestCase):
    def test_three_masks_cover_each_cell_once_and_use_independent_norms(self):
        cube, params = example()
        render = plot_heatmap(Figure().subplots(), cube, params)
        self.assertEqual(len(render.images), 3)
        np.testing.assert_array_equal(sum(~np.ma.getmaskarray(m.get_array()) for m in render.images),
                                      np.ones(cube.Z.shape))
        self.assertEqual([m.norm.vmax for m in render.images], [3, 30, 300])
        self.assertEqual(render.boundaries, (2.5, 4.5))

    def test_rejects_reversed_or_coincident_cell_boundaries(self):
        cube, params = example()
        for boundary in (2., 2.6, 6.):
            params.split_scale.split_x2 = boundary
            with self.assertRaises(ValueError):
                plot_heatmap(Figure().subplots(), cube, params)

    def test_export_compact_top_bars_preserve_canvas_and_heatmap(self):
        cube, params = example()
        fig = _build_streamlit_style_heatmap_fig(cube, params, drr=True)
        FigureCanvasAgg(fig).draw()
        self.assertEqual(len(fig.axes), 4)
        np.testing.assert_allclose(fig.get_size_inches(), EXPORT_FIGSIZE)
        np.testing.assert_allclose(fig.axes[0].get_position().bounds, EXPORT_AXES_RECT)
        width = fig.axes[0].get_position().width * fig.get_figwidth()
        self.assertAlmostEqual(width, EXPORT_AXES_RECT[2] * EXPORT_FIGSIZE[0])
        bars = fig.axes[1:]
        self.assertTrue(all(a.get_position().y0 > fig.axes[0].get_position().y1 for a in bars))
        self.assertTrue(all(a.get_position().width > a.get_position().height for a in bars))
        renderer = fig.canvas.get_renderer()
        boxes = [a.get_tightbbox(renderer) for a in bars]
        self.assertTrue(all(not a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]))
        self.assertTrue(all(fig.bbox.contains(b.x0, b.y0) and fig.bbox.contains(b.x1, b.y1) for b in boxes))

    def test_export_preserves_complete_long_title_without_overlaps(self):
        cube, params = example()
        params.title = 'YZ365 p5n2 | 5 T | 1.67 K | REF 760 nm | Rot 100 deg | TG sweep | BG 24 V | External background | Repeat 03'
        fig = _build_streamlit_style_heatmap_fig(cube, params, drr=True)
        FigureCanvasAgg(fig).draw()
        title = fig.texts[0]
        self.assertEqual(' '.join(title.get_text().split()),
                         'YZ365 p5n2 5 T 1.67 K REF 760 nm Rot 100 deg TG sweep BG 24 V Repeat 03')
        renderer = fig.canvas.get_renderer()
        box = title.get_window_extent(renderer)
        self.assertGreater(box.y0, fig.axes[0].bbox.y1)
        for axis in fig.axes[1:]:
            self.assertFalse(box.overlaps(axis.get_tightbbox(renderer)))

    def test_compact_export_scientific_endpoint_labels_do_not_overlap(self):
        cube, params = example()
        for lo, hi in ((-.000123, .000234), (-12345, 23456), (-.0001, .0001)):
            for side in ('left', 'middle', 'right'):
                setattr(params.split_scale, side+'_vmin', lo)
                setattr(params.split_scale, side+'_vmax', hi)
            fig = _build_streamlit_style_heatmap_fig(cube, params, drr=True)
            FigureCanvasAgg(fig).draw()
            renderer = fig.canvas.get_renderer()
            labels = [label.get_window_extent(renderer) for axis in fig.axes[1:]
                      for label in axis.get_xticklabels() if label.get_visible()]
            self.assertEqual(len(labels), 6)
            self.assertTrue(all(not a.overlaps(b) for i,a in enumerate(labels) for b in labels[i+1:]))

    def test_preview_bars_remain_separate_after_resize(self):
        from core.region_colorbars import three_preview_axes, add_three_colorbars
        cube, params = example()
        fig = Figure(figsize=(10, 6), dpi=100)
        canvas = FigureCanvasAgg(fig)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, .035], hspace=.28, wspace=.12)
        ax, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        render = plot_heatmap(ax, cube, params)
        bars = add_three_colorbars(fig, render, three_preview_axes(cax), fontsize=7)
        for size in ((10, 6), (6.4, 4.8)):
            fig.set_size_inches(*size)
            canvas.draw()
            boxes = [b.ax.get_tightbbox(canvas.get_renderer()) for b in bars]
            self.assertTrue(all(not a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]))

    def test_color_only_export_change_preserves_dat_and_serializes_regions(self):
        import json
        import tempfile
        from dataclasses import replace
        from core.export import export_drr_png_and_dat
        cube, params = example()
        with tempfile.TemporaryDirectory() as folder:
            single = replace(params, split_scale=None)
            first = export_drr_png_and_dat(folder, cube=cube, params=single, export_base='regions')
            original = first['dat'].read_bytes()
            modified = first['dat'].stat().st_mtime_ns
            triple = export_drr_png_and_dat(folder, cube=cube, params=params, export_base='regions')
            self.assertEqual(first['dat'], triple['dat'])
            self.assertEqual(triple['dat'].read_bytes(), original)
            self.assertEqual(triple['dat'].stat().st_mtime_ns, modified)
            metadata = json.loads(triple['dat'].with_suffix('.metadata.json').read_text())
            self.assertEqual(metadata['plot']['split_scale']['split_x2'], 4.5)
            self.assertEqual(metadata['plot']['split_scale']['middle_vmax'], 30)


if __name__ == '__main__':
    unittest.main()
