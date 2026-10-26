"""Compact inspection headers remain readable in narrow paired map panels."""
from __future__ import annotations

import pickle
import unittest

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from core.plotting import HeatmapRender
from ui_qt.inspection_colorbar import add_inspection_colorbar


def _render_regions(axis, count):
    edges = np.linspace(1.7, 2.7, count + 1)
    limits = ((-.000123, .000234), (-12345., 23456.), (-.25, .5))
    images = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        low, high = limits[index]
        images.append(axis.pcolormesh(
            [left, right], [-1, 0, 1], [[low], [high]],
            norm=Normalize(low, high), shading="flat",
        ))
    return HeatmapRender(
        primary=images[0],
        secondary=images[1] if count > 1 else None,
        split_x=float(edges[1]) if count > 1 else None,
        tertiary=images[2] if count > 2 else None,
        split_x2=float(edges[2]) if count > 2 else None,
    )


def _paired_headers(count):
    figure = Figure(figsize=(8.8, 4.8), dpi=100)
    canvas = FigureCanvasAgg(figure)
    maps, renders = [], []
    for index, (left, title, label) in enumerate((
        (.06, "Linear", "DR/R"),
        (.555, "Derivative", "d2(DR/R)/dE2"),
    )):
        axis = figure.add_axes([left, .18, 350 / 880, .58], label=f"inspection-map-{index}")
        axis.set_title(title)
        render = _render_regions(axis, count)
        add_inspection_colorbar(figure, render, axis, label=label, title=title)
        maps.append(axis)
        renders.append(render)
    canvas.draw()
    return figure, maps, renders


def _header_texts(map_axis, host):
    texts = [map_axis.title, *host.texts]
    for axis in host.child_axes:
        texts.extend((axis.title, axis.xaxis.label, axis.yaxis.label))
        texts.extend(axis.texts)
        texts.extend(axis.get_xticklabels())
        texts.extend(axis.get_yticklabels())
        texts.extend((axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()))
    return [text for text in texts if text.get_visible() and text.get_text().strip()]


class InspectionColorbarTests(unittest.TestCase):
    def assert_header_readable(self, figure, map_axis, host):
        renderer = figure.canvas.get_renderer()
        texts = _header_texts(map_axis, host)
        boxes = [(text.get_text(), text.get_window_extent(renderer)) for text in texts]
        for text, box in boxes:
            with self.subTest(text=text):
                self.assertGreaterEqual(box.x0, figure.bbox.x0 - 1)
                self.assertLessEqual(box.x1, figure.bbox.x1 + 1)
                self.assertGreaterEqual(box.y0, map_axis.bbox.y1 - 1)
                self.assertLessEqual(box.y1, figure.bbox.y1 + 1)
        for index, (first_text, first) in enumerate(boxes):
            for second_text, second in boxes[index + 1:]:
                self.assertFalse(first.overlaps(second), (first_text, second_text))
        for text, box in boxes:
            for colorbar_axis in host.child_axes:
                self.assertFalse(box.overlaps(colorbar_axis.bbox), text)

    def test_one_two_and_three_region_headers_fit_narrow_paired_maps(self):
        for count in (1, 2, 3):
            with self.subTest(regions=count):
                figure, maps, renders = _paired_headers(count)
                for axis, render, title, label in zip(
                    maps, renders, ("Linear", "Derivative"), ("DR/R", "d2(DR/R)/dE2"),
                ):
                    host = render.colorbar_host
                    self.assertIn(host, figure.axes)
                    self.assertTrue(host._dptk_inspection_header)
                    self.assertNotIn(host, axis.child_axes)
                    self.assertEqual(len(host.child_axes), count)
                    self.assertAlmostEqual(axis.bbox.width, 350)
                    self.assertAlmostEqual(host.bbox.x1, axis.bbox.x1, places=6)
                    self.assertGreater(host.bbox.y0, axis.bbox.y1)
                    self.assertEqual(axis.get_title(), title)
                    title_box = axis.title.get_window_extent(figure.canvas.get_renderer())
                    self.assertAlmostEqual(title_box.x0, axis.bbox.x0, delta=1)
                    quantity = [text for text in host.texts if text.get_text() == label]
                    self.assertEqual(len(quantity), 1)
                    for colorbar_axis in host.child_axes:
                        self.assertGreater(colorbar_axis.bbox.width, colorbar_axis.bbox.height)
                        self.assertGreater(colorbar_axis.bbox.y0, axis.bbox.y1)
                        self.assertEqual(len(colorbar_axis.get_xticks()), 2)
                        self.assertEqual(len([text for text in colorbar_axis.get_xticklabels()
                                              if text.get_visible() and text.get_text()]), 2)
                    labels = [text.get_text() for text in _header_texts(axis, host)]
                    expected = () if count == 1 else ("L", "R") if count == 2 else ("L", "M", "R")
                    for region in expected:
                        self.assertTrue(any(text == region or text.startswith(region + ":")
                                            for text in labels), (region, labels))
                    self.assert_header_readable(figure, axis, host)

    def test_endpoint_ticks_follow_each_independent_norm_after_updates(self):
        figure, maps, renders = _paired_headers(3)
        for render in renders:
            for index, image in enumerate(render.images):
                low, high = (-.000046 * (index + 1), .000087 * (index + 1))
                image.set_clim(low, high)
        figure.canvas.draw()
        for axis, render in zip(maps, renders):
            for image, colorbar_axis in zip(render.images, render.colorbar_host.child_axes):
                np.testing.assert_allclose(colorbar_axis.get_xticks(), [image.norm.vmin, image.norm.vmax])
                self.assertEqual(len([text for text in colorbar_axis.get_xticklabels()
                                      if text.get_visible() and text.get_text()]), 2)
                offset = colorbar_axis.xaxis.get_offset_text()
                self.assertFalse(offset.get_visible() and offset.get_text())
            self.assert_header_readable(figure, axis, render.colorbar_host)

    def test_header_and_colorbar_locators_follow_map_position_changes(self):
        figure, maps, renders = _paired_headers(3)
        axis, render = maps[0], renders[0]
        host = render.colorbar_host
        original_host = host.get_position().bounds
        original_bars = [bar.get_position().bounds for bar in host.child_axes]
        position = axis.get_position()
        axis.set_position([position.x0 + .015, position.y0, position.width * .94, position.height * .88])
        figure.canvas.draw()
        self.assertNotEqual(host.get_position().bounds, original_host)
        self.assertAlmostEqual(host.bbox.x1, axis.bbox.x1, places=6)
        self.assertGreater(host.bbox.y0, axis.bbox.y1)
        for colorbar_axis, original in zip(host.child_axes, original_bars):
            self.assertNotEqual(colorbar_axis.get_position().bounds, original)
            self.assertGreater(colorbar_axis.bbox.y0, axis.bbox.y1)
        self.assert_header_readable(figure, axis, host)

    def test_figure_roundtrip_preserves_picklable_locators_and_can_draw(self):
        for count in (1, 2, 3):
            with self.subTest(regions=count):
                figure, _, renders = _paired_headers(count)
                for render in renders:
                    for axis in (render.colorbar_host, *render.colorbar_host.child_axes):
                        self.assertTrue(callable(axis.get_axes_locator()))
                        pickle.dumps(axis.get_axes_locator())
                restored = pickle.loads(pickle.dumps(figure))
                canvas = FigureCanvasAgg(restored)
                canvas.draw()
                maps = [axis for axis in restored.axes if axis.get_label().startswith("inspection-map-")]
                hosts = [axis for axis in restored.axes if getattr(axis, "_dptk_inspection_header", False)]
                self.assertEqual(len(maps), 2)
                self.assertEqual(len(hosts), 2)
                for map_axis, host in zip(maps, hosts):
                    self.assertEqual(len(host.child_axes), count)
                    self.assertAlmostEqual(host.bbox.x1, map_axis.bbox.x1, places=6)
                    self.assert_header_readable(restored, map_axis, host)

    def test_supplied_host_is_reused_for_a_changed_region_count(self):
        figure, maps, renders = _paired_headers(1)
        axis, previous = maps[0], renders[0]
        host = previous.colorbar_host
        updated = _render_regions(axis, 3)
        result = add_inspection_colorbar(figure, updated, axis, label="DR/R", host=host)
        figure.canvas.draw()
        self.assertIs(result, host)
        self.assertIs(updated.colorbar_host, host)
        self.assertEqual(len(host.child_axes), 3)
        self.assertEqual(sum(getattr(item, "_dptk_inspection_header", False)
                             for item in figure.axes), 2)
        self.assert_header_readable(figure, axis, host)


if __name__ == "__main__":
    unittest.main()
