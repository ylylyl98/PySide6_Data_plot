"""Exercise the real Qt divider against retained Matplotlib artists."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from io import BytesIO
from types import SimpleNamespace
import unittest

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QTabWidget, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from tests.ui_test_helpers import dispose_owned_window
from ui_qt.axes_region_blitter import AxesRegionBlitter
from ui_qt.inspection_splitter import bind_inspection_splitter
from ui_qt.spectrum_caption import set_spectrum_title
from ui_qt.theme import install_theme


class InspectionSplitterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion")

    def setUp(self):
        install_theme(self.app, mode="light")
        owner = self.owner = QWidget()
        self.addCleanup(dispose_owned_window, owner)
        owner.figure = Figure(figsize=(7, 5), dpi=100)
        owner.canvas = FigureCanvasQTAgg(owner.figure)
        layout = QVBoxLayout(owner)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(owner.canvas)
        owner.tabs = QTabWidget(owner)
        owner.tabs.hide()
        for name in ("PL", "DRR", "Compare"):
            owner.tabs.addTab(QWidget(), name)
        owner.pl_controller = SimpleNamespace(_pl_region_blitters=None)
        owner._drr_region_blitters = {}
        owner.resize(700, 500)
        owner.show()
        self.app.processEvents()
        self.grid = owner.figure.add_gridspec(
            2, 2, width_ratios=[1, .035], height_ratios=[.7, .3],
            hspace=.20, top=.94, bottom=.13,
        )
        self.heatmap = owner.figure.add_subplot(self.grid[0, 0])
        self.colorbar = owner.figure.add_subplot(self.grid[0, 1])
        self.spectrum = owner.figure.add_subplot(self.grid[1, 0], sharex=self.heatmap)
        x = np.linspace(1, 2, 51)
        self.heatmap.imshow(np.outer(np.arange(4), x), extent=(1, 2, -1, 1), aspect="auto")
        self.line, = self.spectrum.plot(x, np.sin(x * 5))
        self.spectrum.set_xlabel("Photon Energy (eV)")
        set_spectrum_title(self.spectrum, "Spectrum @ 0 V")
        self.divider = bind_inspection_splitter(
            owner, "PL", grids=(self.grid,), maps=(self.heatmap,),
            spectra=(self.spectrum,), outer_grid=self.grid,
        )
        owner.canvas.draw()
        self.app.processEvents()

    def test_drag_and_keyboard_retain_axes_zoom_and_numerical_data(self):
        self.heatmap.set_xlim(1.2, 1.7)
        self.heatmap.set_ylim(-.4, .6)
        axes = tuple(self.owner.figure.axes)
        values = np.asarray(self.line.get_ydata()).copy()
        self.assertAlmostEqual(self.heatmap.bbox.height /
                               (self.heatmap.bbox.height + self.spectrum.bbox.height), .7)
        start = QPoint(20, self.divider.height() // 2)
        target = self.divider.mapTo(self.owner.canvas, start) + QPoint(0, 38)
        QTest.mousePress(self.divider, Qt.LeftButton, pos=start)
        QTest.mouseMove(self.divider, self.divider.mapFrom(self.owner.canvas, target))
        QTest.mouseRelease(self.divider, Qt.LeftButton,
                           pos=self.divider.mapFrom(self.owner.canvas, target))
        self.app.processEvents()
        self.assertGreater(self.divider.heatmap_fraction, .7)
        self.assertEqual(tuple(self.owner.figure.axes), axes)
        self.assertEqual(self.heatmap.get_xlim(), (1.2, 1.7))
        self.assertEqual(self.heatmap.get_ylim(), (-.4, .6))
        np.testing.assert_array_equal(self.line.get_ydata(), values)
        QTest.keyClick(self.divider, Qt.Key_Home)
        self.assertAlmostEqual(self.divider.heatmap_fraction, .7)
        QTest.keyClick(self.divider, Qt.Key_Up)
        self.assertAlmostEqual(self.divider.heatmap_fraction, .67)

    def test_hide_and_show_button_restores_ratio_and_avoids_hidden_blit_ghosts(self):
        helper = AxesRegionBlitter(self.owner.canvas)
        helper.configure(self.spectrum, [self.line])
        self.owner.pl_controller._pl_region_blitters = (helper,)
        self.owner.canvas.draw()
        self.divider.set_heatmap_fraction(.62)
        QTest.mouseClick(self.divider.toggle_button, Qt.LeftButton)
        self.owner.canvas.draw()
        self.assertTrue(self.divider.spectrum_collapsed)
        self.assertFalse(self.spectrum.get_visible())
        self.assertEqual(self.heatmap.get_xlabel(), "Photon Energy (eV)")
        self.assertAlmostEqual(self.heatmap.get_position().y0,
                               .13 + 24 / self.owner.canvas.height())
        before = np.asarray(self.owner.canvas.buffer_rgba()).copy()
        self.line.set_ydata(np.asarray(self.line.get_ydata()) + 10)
        self.assertFalse(helper.draw())
        self.owner.canvas.draw()
        np.testing.assert_array_equal(before, np.asarray(self.owner.canvas.buffer_rgba()))
        QTest.mouseClick(self.divider.toggle_button, Qt.LeftButton)
        self.owner.canvas.draw()
        self.assertFalse(self.divider.spectrum_collapsed)
        self.assertTrue(self.spectrum.get_visible())
        self.assertEqual(self.heatmap.get_xlabel(), "")
        self.assertAlmostEqual(self.divider.heatmap_fraction, .62)
        self.assertTrue(helper.draw())

    def test_resize_and_noninspection_mode_preserve_ownership(self):
        self.divider.set_heatmap_fraction(.55)
        self.owner.resize(830, 620)
        self.app.processEvents()
        self.owner.canvas.draw()
        self.assertAlmostEqual(self.heatmap.bbox.height /
                               (self.heatmap.bbox.height + self.spectrum.bbox.height), .55)
        self.assertTrue(self.divider.isVisible())
        self.owner.tabs.setCurrentIndex(2)
        self.assertFalse(self.divider.isVisible())
        self.owner.tabs.setCurrentIndex(0)
        self.assertTrue(self.divider.isVisible())
        buffer = BytesIO()
        self.owner.figure.savefig(buffer, format="png")
        self.assertGreater(buffer.tell(), 1000)


if __name__ == "__main__":
    unittest.main()
