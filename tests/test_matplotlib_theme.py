from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib as mpl
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication

from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg


class _Theme:
    def __init__(self, name: str):
        self.name = name
        values = {
            "light": {
                "canvas_background": "#ffffff", "surface_secondary": "#f7f7f7",
                "text_primary": "#1a1a1a", "text_secondary": "#4a4a4a",
                "border_primary": "#777777", "border_subtle": "#c8c8c8",
            },
            "dark": {
                "canvas_background": "#202020", "surface_secondary": "#2a2a2a",
                "text_primary": "#f4f4f4", "text_secondary": "#c8c8c8",
                "border_primary": "#9b9b9b", "border_subtle": "#5b5b5b",
            },
        }[name]
        self.aliases = values

    def value(self, key: str) -> str:
        return self.aliases[key]


class MatplotlibThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _canvas(self, theme: str = "dark"):
        figure = Figure(figsize=(3, 2), dpi=80)
        canvas = ThemeAwareFigureCanvasQTAgg(figure, resolved_theme=_Theme(theme))
        return figure, canvas

    def test_dark_theme_updates_surface_and_presentation_artists(self) -> None:
        figure, canvas = self._canvas("dark")
        axis = figure.add_subplot(111)
        axis.plot([0, 1], [0, 1], label="signal")
        axis.set_title("Title")
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.grid(True)
        axis.legend(title="Series")
        canvas.draw()

        self.assertEqual(axis.get_facecolor()[:3], (42 / 255, 42 / 255, 42 / 255))
        self.assertEqual(figure.get_facecolor()[:3], (32 / 255, 32 / 255, 32 / 255))
        self.assertEqual(axis.title.get_color(), "#f4f4f4")
        self.assertEqual(axis.xaxis.label.get_color(), "#f4f4f4")
        self.assertEqual(axis.spines["left"].get_edgecolor(), mpl.colors.to_rgba("#9b9b9b"))
        self.assertEqual(axis.get_legend().get_frame().get_facecolor()[:3], (42 / 255, 42 / 255, 42 / 255))
        self.assertEqual(axis.get_legend().get_title().get_color(), "#f4f4f4")

    def test_explicit_axis_annotation_color_is_preserved(self) -> None:
        figure, canvas = self._canvas("dark")
        axis = figure.add_subplot(111)
        annotation = axis.text(0.5, 0.5, "peak", color="#ff00ff")

        canvas.draw()

        self.assertEqual(annotation.get_color(), "#ff00ff")

    def test_dark_theme_updates_figure_title_and_figure_legend(self) -> None:
        figure, canvas = self._canvas("dark")
        axis = figure.add_subplot(111)
        (line,) = axis.plot([0, 1], [0, 1], color="#d62728", label="signal")
        figure.suptitle("Figure title")
        figure.legend([line], ["signal"], title="Series")

        canvas.draw()

        self.assertEqual(figure._suptitle.get_color(), "#f4f4f4")
        legend = figure.legends[0]
        self.assertEqual(legend.get_frame().get_facecolor()[:3], (42 / 255, 42 / 255, 42 / 255))
        self.assertEqual(legend.get_frame().get_edgecolor()[:3], mpl.colors.to_rgba("#9b9b9b")[:3])
        self.assertEqual(legend.get_texts()[0].get_color(), "#f4f4f4")
        self.assertEqual(legend.get_title().get_color(), "#f4f4f4")
        self.assertEqual(line.get_color(), "#d62728")

    def test_axes_created_after_clear_receive_active_presentation(self) -> None:
        figure, canvas = self._canvas("dark")
        figure.add_subplot(111)
        canvas.draw()
        figure.clear()
        axis = figure.add_subplot(111)
        axis.set_title("new")
        canvas.draw()
        self.assertEqual(axis.get_facecolor()[:3], (42 / 255, 42 / 255, 42 / 255))
        self.assertEqual(axis.title.get_color(), "#f4f4f4")

    def test_data_artists_and_rcparams_remain_unchanged(self) -> None:
        figure, canvas = self._canvas("dark")
        axis = figure.add_subplot(111)
        line, = axis.plot([0, 1], [1, 0], color="#d62728", marker="o")
        points = axis.scatter([0.2, 0.8], [0.3, 0.7], c=[1, 2], cmap="viridis", norm=Normalize(0, 3))
        image = axis.imshow(np.arange(4).reshape(2, 2), cmap="magma", norm=Normalize(0, 3))
        rc_before = dict(mpl.rcParams)
        line_color, line_marker = line.get_color(), line.get_marker()
        points_cmap, points_norm = points.get_cmap(), points.norm
        image_cmap, image_norm = image.get_cmap(), image.norm
        canvas.draw()
        self.assertEqual(dict(mpl.rcParams), rc_before)
        self.assertEqual(line.get_color(), line_color)
        self.assertEqual(line.get_marker(), line_marker)
        self.assertIs(points.get_cmap(), points_cmap)
        self.assertIs(points.norm, points_norm)
        self.assertIs(image.get_cmap(), image_cmap)
        self.assertIs(image.norm, image_norm)

    def test_publication_context_is_light_and_restores_after_exception(self) -> None:
        figure, canvas = self._canvas("dark")
        axis = figure.add_subplot(111)
        canvas.draw()
        with self.assertRaisesRegex(RuntimeError, "save failed"):
            with canvas.publication_context():
                self.assertEqual(figure.get_facecolor()[:3], (1.0, 1.0, 1.0))
                self.assertEqual(axis.get_facecolor()[:3], (1.0, 1.0, 1.0))
                raise RuntimeError("save failed")
        self.assertEqual(figure.get_facecolor()[:3], (32 / 255, 32 / 255, 32 / 255))
        self.assertEqual(axis.get_facecolor()[:3], (42 / 255, 42 / 255, 42 / 255))


if __name__ == "__main__":
    unittest.main()
