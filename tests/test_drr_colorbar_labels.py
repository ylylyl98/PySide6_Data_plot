import unittest
from dataclasses import replace
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from core.export import _build_streamlit_style_heatmap_fig, EXPORT_FIGSIZE, EXPORT_AXES_RECT
from tests.test_drr_three_regions import example


class DrrColorbarLabelTests(unittest.TestCase):
    def test_shared_quantity_label_is_above_bars_for_all_products_and_regions(self):
        cube, base = example()
        for quantity in ('DR/R', 'd2(DR/R)/dE2'):
            for split in (None, replace(base.split_scale, split_x2=None), base.split_scale):
                with self.subTest(quantity=quantity, split=split):
                    params = replace(base, cbar_label=quantity, split_scale=split)
                    fig = _build_streamlit_style_heatmap_fig(cube, params, drr=True)
                    FigureCanvasAgg(fig).draw()
                    labels = [t for t in fig.texts if t.get_gid() == 'drr-colorbar-quantity']
                    self.assertEqual(len(labels), 1)
                    renderer = fig.canvas.get_renderer()
                    box = labels[0].get_window_extent(renderer)
                    first = fig.axes[1].bbox
                    self.assertGreater(box.y0, first.y1)
                    self.assertEqual(labels[0].get_text(), quantity)
                    self.assertEqual(labels[0].get_fontsize(), 16 if split is None else 9)
                    for axis in fig.axes[1:]:
                        for tick in axis.get_xticklabels():
                            bounds = tick.get_window_extent(renderer)
                            self.assertGreater(bounds.y0, axis.bbox.y1)
                            self.assertGreater(box.y0, bounds.y1)
                    self.assertGreater(box.y0, fig.axes[0].bbox.y1)
                    expected = [] if split is None else (['L','R'] if split.split_x2 is None else ['L','M','R'])
                    self.assertEqual([a.get_xticklabels()[0].get_text().split(':')[0]
                                      for a in fig.axes[1:]] if split else [], expected)
                    np.testing.assert_allclose(fig.get_size_inches(), EXPORT_FIGSIZE)
                    np.testing.assert_allclose(fig.axes[0].get_position().bounds, EXPORT_AXES_RECT)
                    boxes = [box] + [a.get_tightbbox(renderer) for a in fig.axes[1:]]
                    self.assertTrue(all(not a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]))

    def test_derivative_scientific_ticks_remain_separate(self):
        cube, params = example()
        for lo, hi in ((-.000123, .000234), (-12345, 23456)):
            for side in ('left', 'middle', 'right'):
                setattr(params.split_scale, side+'_vmin', lo)
                setattr(params.split_scale, side+'_vmax', hi)
            fig = _build_streamlit_style_heatmap_fig(cube, replace(params, cbar_label='d2(DR/R)/dE2'), drr=True)
            FigureCanvasAgg(fig).draw()
            renderer = fig.canvas.get_renderer()
            boxes = [t.get_window_extent(renderer) for a in fig.axes[1:] for t in a.get_xticklabels()]
            self.assertTrue(all(not a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]))
