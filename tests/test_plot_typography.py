"""Arial policy across Quick/Agg, title measurement and headless PNG saves."""
import io
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.mathtext import MathTextParser
from matplotlib.text import Text
import numpy as np

from core.typography import FONT_FAMILY, matplotlib_font_context, save_figure


ROOT = Path(__file__).resolve().parents[1]


def _double(values):
    return values * 2


def _half(values):
    return values / 2


class PlotTypographyTests(unittest.TestCase):
    def figure(self):
        figure = Figure(figsize=(6.4, 4.8), dpi=100, layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        mesh = axis.imshow(np.arange(16).reshape(4, 4), cmap="viridis")
        line, = axis.plot([0, 1, 2, 3], [0, 2, 1, 3], color="#d03050", label="σ+ signal")
        axis.set(title="DR/R · Arial normal and bold", xlabel="Energy (eV)", ylabel="Gate (V)")
        axis.title.set_fontweight("bold")
        axis.set_xlabel("Energy (eV)", fontsize=13)
        axis.legend(title="Measured spectrum")
        annotation = axis.annotate(r"$\Delta E=\alpha B^2+\mathbf{x}+\oiint$", (1, 2),
                                   color="#302060", fontsize=12, fontstyle="italic")
        figure.colorbar(mesh, ax=axis, label="Intensity (a.u.)")
        inset = axis.inset_axes((.64, .08, .3, .25))
        inset.set_title("Inset", fontsize=8)
        secondary = axis.secondary_xaxis("top", functions=(_double, _half))
        secondary.set_xlabel("Secondary axis")
        figure.suptitle("Arial publication export · T = 4 K", fontsize=16, fontweight="bold")
        return figure, axis, annotation, line

    def assert_arial(self, figure):
        for text in figure.findobj(match=Text):
            self.assertEqual(text.get_fontfamily()[0], FONT_FAMILY)
            self.assertEqual(text.get_math_fontfamily(), "custom")

    def test_shared_font_import_does_not_load_matplotlib_or_qt(self):
        result = subprocess.run([sys.executable, "-c", "import sys; from core.typography import FONT_FAMILY; "
                                 "assert FONT_FAMILY == 'Arial'; "
                                 "assert not any(n.startswith(('matplotlib', 'PySide6', 'numpy')) for n in sys.modules)"],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_headless_save_overrides_existing_fonts_and_preserves_plot_style(self):
        figure, axis, annotation, line = self.figure()
        annotation.set_fontproperties(FontProperties(fname=findfont("DejaVu Serif"), size=12,
                                                     style="italic", weight="bold"))
        expected = (annotation.get_fontsize(), annotation.get_fontweight(), annotation.get_fontstyle(),
                    annotation.get_color(), annotation.get_text(), axis.get_xlim(), axis.get_ylim(),
                    line.get_color(), line.get_ydata().copy())
        rc_before = dict(mpl.rcParams)
        output = io.BytesIO()
        save_figure(figure, output, format="png")
        self.assertTrue(output.getvalue().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assert_arial(figure)
        self.assertIsNone(annotation.get_fontproperties().get_file())
        actual = (annotation.get_fontsize(), annotation.get_fontweight(), annotation.get_fontstyle(),
                  annotation.get_color(), annotation.get_text(), axis.get_xlim(), axis.get_ylim(), line.get_color())
        self.assertEqual(actual, expected[:-1])
        np.testing.assert_array_equal(line.get_ydata(), expected[-1])
        self.assertEqual(dict(mpl.rcParams), rc_before)

    def test_save_context_overrides_caller_fonts_and_restores_even_after_failure(self):
        figure, *_ = self.figure()
        with mpl.rc_context({"font.family": "serif", "mathtext.fontset": "cm", "mathtext.rm": "DejaVu Serif"}):
            previous = dict(mpl.rcParams)
            def fail(*_args, **_kwargs):
                self.assertEqual(mpl.rcParams["font.family"][0], FONT_FAMILY)
                self.assertEqual(mpl.rcParams["mathtext.fontset"], "custom")
                self.assert_arial(figure)
                raise OSError("failed output")
            with patch.object(figure, "savefig", side_effect=fail), self.assertRaisesRegex(OSError, "failed output"):
                save_figure(figure, io.BytesIO(), format="png")
            self.assertEqual(dict(mpl.rcParams), previous)

    def test_math_uses_installed_arial_variants_and_symbol_fallback(self):
        try:
            findfont(FONT_FAMILY, fallback_to_default=False)
        except ValueError:
            self.skipTest("Arial is not installed on this test platform")
        with matplotlib_font_context():
            result = MathTextParser("path").parse(
                r"$\mathrm{R}+x+\mathbf{B}+\mathbfit{z}+\Delta+\alpha+\oiint$", dpi=100)
        faces = {chr(code): (font.family_name, font.style_name.lower())
                 for font, _size, code, _x, _y in result.glyphs}
        for character in ("R", "x", "B", "z", "Δ", "α"):
            self.assertEqual(faces[character][0], FONT_FAMILY)
        self.assertIn("italic", faces["x"][1])
        self.assertIn("bold", faces["B"][1])
        self.assertIn("bold", faces["z"][1])
        self.assertIn("italic", faces["z"][1])
        self.assertTrue(any("STIX" in family for family, _style in faces.values()))

    def test_title_fitting_measures_arial_before_save(self):
        from core.compact_region_export import complete_title
        figure = Figure(figsize=(5, 3), dpi=100)
        canvas = FigureCanvasAgg(figure)
        renderer = canvas.get_renderer()
        measured = []
        original = renderer.get_text_width_height_descent
        def measure(text, properties, ismath):
            measured.append(properties.get_family()[0])
            return original(text, properties, ismath)
        with mpl.rc_context({"font.family": "serif"}), patch.object(renderer, "get_text_width_height_descent", side_effect=measure):
            title = complete_title(figure, "Complete long Arial title with α and ΔE that wraps across the publication header", .1, .9)
        self.assertTrue(measured)
        self.assertEqual(set(measured), {FONT_FAMILY})
        self.assertEqual(title.get_fontfamily()[0], FONT_FAMILY)
        self.assertLessEqual(title.get_window_extent(renderer).width, figure.bbox.width * .8 + .5)

    def test_timed_png_canvas_direct_save_uses_arial(self):
        from core.png_render_timing import TimedPngCanvas, collect_png_timings
        figure, *_ = self.figure()
        TimedPngCanvas(figure)
        previous = dict(mpl.rcParams)
        with collect_png_timings() as timing:
            figure.savefig(io.BytesIO(), format="png")
        self.assert_arial(figure)
        self.assertIn("draw", timing)
        self.assertIn("encode_write", timing)
        self.assertEqual(dict(mpl.rcParams), previous)

    def test_async_snapshot_uses_arial_after_generic_agg_replacement(self):
        from ui_qt.async_figure_save import _render_snapshot_file
        figure, *_ = self.figure()
        snapshot = pickle.dumps(figure)
        captured = []
        original = Figure.savefig
        def save(figure, *args, **kwargs):
            self.assert_arial(figure)
            captured.append(figure)
            return original(figure, *args, **kwargs)
        previous = dict(mpl.rcParams)
        with tempfile.TemporaryDirectory() as temporary, patch.object(Figure, "savefig", new=save):
            path = Path(temporary) / "snapshot.png"
            _render_snapshot_file(snapshot, path, {"format": "png", "dpi": 100}, width_scale=1.2)
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(list(Path(temporary).glob("*.tmp*")), [])
        self.assertEqual(len(captured), 1)
        self.assertEqual(dict(mpl.rcParams), previous)

    def test_quick_canvas_draw_and_direct_save_in_isolated_gui_process(self):
        code = r'''
import io, sys
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.text import Text
from PySide6.QtGui import QGuiApplication
from ui_qt.quick.figure_canvas import QuickFigureCanvas
app = QGuiApplication([])
figure = Figure(figsize=(3, 2), dpi=100)
canvas = QuickFigureCanvas(figure)
axis = figure.subplots()
axis.plot([0, 1], [1, 0], label='signal')
axis.set_title(r'Arial $\Delta+\mathbf{x}$', fontfamily='serif', fontsize=13)
axis.legend()
with mpl.rc_context({'font.family': 'serif', 'mathtext.fontset': 'cm'}):
    previous = dict(mpl.rcParams)
    canvas.draw()
    assert not canvas.image.isNull()
    assert dict(mpl.rcParams) == previous
    figure.clear()
    axis = figure.subplots()
    axis.set_title('New plot', fontfamily='serif', fontweight='bold')
    output = io.BytesIO()
    figure.savefig(output, format='png')
    assert output.getvalue().startswith(bytes([137, 80, 78, 71]))
    assert all(t.get_fontfamily()[0] == 'Arial' for t in figure.findobj(match=Text))
    assert axis.title.get_fontweight() == 'bold'
    assert dict(mpl.rcParams) == previous
assert 'PySide6.QtWidgets' not in sys.modules
canvas.close()
'''
        environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "QT_QUICK_BACKEND": "software", "MPLBACKEND": "agg"}
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
