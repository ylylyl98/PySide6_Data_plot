"""Plot text follows the Qt canvas font without changing plot typography."""
import io
import os
import unittest
import warnings

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.text import Text
from PySide6.QtGui import QFont, QFontInfo
from PySide6.QtWidgets import QApplication

from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg


class PlotSystemFontTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_all_plot_text_uses_qt_font_and_keeps_sizes_and_math(self):
        figure = Figure(figsize=(5, 4))
        canvas = ThemeAwareFigureCanvasQTAgg(figure)
        self.addCleanup(canvas.close)
        canvas.setFont(QFont('Arial', 10))
        axis = figure.add_subplot()
        axis.plot([-1, 0, 1], [1, 0, 1], label='σ+ corrected')
        axis.set_title('MCD spectra', fontsize=15, fontweight='bold')
        axis.set_xlabel('Energy (eV)')
        annotation = axis.text(0, .5, r'$\Delta E = \alpha B^2$', fontsize=8)
        axis.legend(title='Raw / Corrected')
        figure.suptitle('T = 4 K')
        rc_before = dict(mpl.rcParams)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            canvas.draw()
        expected = QFontInfo(canvas.font()).family() or canvas.font().family()
        for text in figure.findobj(match=Text):
            if text.get_text():
                self.assertEqual(text.get_fontfamily()[0], expected)
        self.assertEqual(axis.title.get_fontsize(), 15)
        self.assertEqual(axis.title.get_fontweight(), 'bold')
        self.assertEqual(annotation.get_fontsize(), 8)
        self.assertEqual(dict(mpl.rcParams), rc_before)
        self.assertFalse([w for w in caught if 'Glyph' in str(w.message)])

    def test_new_axes_and_publication_use_current_canvas_font(self):
        figure = Figure(figsize=(3, 2))
        canvas = ThemeAwareFigureCanvasQTAgg(figure)
        self.addCleanup(canvas.close)
        figure.add_subplot().set_title('First')
        canvas.draw()
        canvas.setFont(QFont('Arial', 10))
        figure.clear()
        title = figure.add_subplot().set_title('New plot')
        with canvas.publication_context():
            canvas.print_figure(io.BytesIO(), format='png')
            self.assertEqual(title.get_fontfamily()[0], QFontInfo(canvas.font()).family() or canvas.font().family())


if __name__ == '__main__':
    unittest.main()
