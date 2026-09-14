import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from unittest.mock import patch

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QWidget, QDoubleSpinBox, QCheckBox, QPushButton, QLabel
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from ui_qt.axes_region_blitter import AxesRegionBlitter
from ui_qt.dense_form_layout import DenseFormRowLayout


class LoadLayoutOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_qt_region_initialization_coalesces_with_final_draw(self):
        figure = Figure(figsize=(4, 3))
        canvas = FigureCanvasQTAgg(figure)
        helpers = []
        for slot in (211, 212):
            axis = figure.add_subplot(slot)
            line, = axis.plot([0, 1], [0, 1])
            helper = AxesRegionBlitter(canvas)
            helper.configure(axis, [line]); helpers.append(helper)
        with patch.object(canvas, "draw", wraps=canvas.draw) as draw:
            for helper in helpers:
                helper.restore_interactive_drawing()
            canvas.draw_idle()
            self.assertEqual(draw.call_count, 0, "initialization must not render synchronously")
            self.app.processEvents()
            self.assertEqual(draw.call_count, 1)
            for helper in helpers:
                self.assertIsNotNone(helper._background)
                helper.artists[0].set_ydata([1, 0])
                self.assertTrue(helper.draw())
            self.assertEqual(draw.call_count, 1)
        for helper in helpers:
            helper.disconnect()
        canvas.close()

    def test_detached_axes_do_not_repaint_another_page(self):
        figure = Figure(figsize=(4, 3)); canvas = FigureCanvasQTAgg(figure)
        axis = figure.add_subplot(111); line, = axis.plot([0, 1], [0, 1])
        helper = AxesRegionBlitter(canvas); helper.configure(axis, [line])
        figure.clear()
        figure.add_subplot(111)
        with patch.object(axis, "draw_artist", wraps=axis.draw_artist) as paint:
            canvas.draw()
            self.assertEqual(paint.call_count, 0)
        self.assertIsNone(helper.axes)
        canvas.close()

    def _row(self):
        host = QWidget(); layout = DenseFormRowLayout(host, stable_spin_width=True)
        spins = []
        for _ in range(2):
            spin = QDoubleSpinBox(); spin.setRange(-1e9, 1e9); spin.setDecimals(4)
            spin.setMaximumWidth(130)
            spins.append(spin)
            layout.add_group((spin, QCheckBox("F")), role="range")
        layout.add_group((QPushButton("Auto"),), role="action")
        host.ensurePolished()
        for child in host.findChildren(QWidget): child.ensurePolished()
        return host, layout, spins

    def test_bounded_range_row_height_is_independent_of_loaded_values(self):
        host, layout, spins = self._row()
        try:
            before = [layout.heightForWidth(width) for width in range(250, 601, 10)]
            for spin, value in zip(spins, (-123456.789, 987654.321)):
                spin.setValue(value)
            after = [layout.heightForWidth(width) for width in range(250, 601, 10)]
            self.assertEqual(before, after)
            self.assertGreater(before[0], before[-1], "narrow windows still wrap")
        finally:
            host.close()

    def test_unchanged_layout_queries_reuse_style_measurements(self):
        host, layout, spins = self._row()
        try:
            layout.heightForWidth(380)
            with patch.object(layout, "_spin_width", wraps=layout._spin_width) as measure:
                for _ in range(20):
                    layout.heightForWidth(380); layout.minimumSize(); layout.sizeHint()
                self.assertEqual(measure.call_count, 0)
            old = layout.safe_min_width(spins[0])
            # The application QSS owns control fonts; change the resolved style
            # rather than assuming setFont overrides a stylesheet font.
            spins[0].setStyleSheet("QDoubleSpinBox, QLineEdit { font-size: 30pt; }")
            self.assertGreater(layout.safe_min_width(spins[0]), old)
        finally:
            host.close()

    def test_child_resize_does_not_reissue_parent_geometry_request(self):
        host, layout, spins = self._row()
        try:
            with patch.object(host, "updateGeometry", wraps=host.updateGeometry) as update:
                layout.eventFilter(spins[0], QEvent(QEvent.Resize))
                self.assertEqual(update.call_count, 0)
        finally:
            host.close()

    def test_editor_font_and_margins_invalidate_width_without_resize(self):
        host, layout, spins = self._row()
        try:
            spin = spins[0]
            before = layout.safe_min_width(spin)
            spin.lineEdit().setTextMargins(40, 0, 40, 0)
            self.assertGreater(layout.safe_min_width(spin), before)
            font = spin.lineEdit().font(); font.setPointSize(30)
            spin.lineEdit().setFont(font)
            self.assertEqual(layout.safe_min_width(spin), layout._measure_widget_min_width(spin))
            label = QLabel("Range")
            layout.addWidget(label)
            before = layout.safe_min_width(label)
            label.setContentsMargins(40, 0, 40, 0)
            self.assertEqual(layout.safe_min_width(label), before + 80)
        finally:
            host.close()

    def test_empty_guidance_disappears_without_changing_canvas_geometry(self):
        from scripts.preview_ui import build_preview_window
        window, app, _ = build_preview_window(
            workflow="DRR", theme="light", scale="1", width=1200,
            height=800, sidebar_width=380, demo_data=False)
        try:
            window.show()
            for _ in range(4): app.processEvents()
            before = window.canvas.size()
            window.figure.add_subplot(111).plot([0, 1], [0, 1])
            window.canvas.draw()
            for _ in range(4): app.processEvents()
            self.assertFalse(window.empty_canvas_overlay.isVisible())
            self.assertEqual(window.canvas.size(), before)
        finally:
            window.close(); window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


if __name__ == "__main__":
    unittest.main()
