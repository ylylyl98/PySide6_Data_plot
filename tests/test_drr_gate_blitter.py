import pickle
import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ui_qt.drr_gate_blitter import DrrGateBlitter


class DrrGateBlitterTests(unittest.TestCase):
    def setUp(self):
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasAgg(self.figure)
        axes = self.figure.subplots(2, 2)
        self.heat = tuple(axes[0])
        self.spectra = tuple(axes[1])
        x = np.linspace(1, 2, 31)
        self.lines, self.gates = [], []
        for heat, spectrum in zip(self.heat, self.spectra):
            heat.pcolormesh(x, [0, 1, 2], np.outer([1, 2, 3], x))
            self.gates.append(heat.axhline(.5, color='black', linestyle='--'))
            self.lines.append(spectrum.plot(x, np.sin(x*30))[0])
            spectrum.set_title('Spectrum @ 0.5 V')
            spectrum.set_xlabel('Energy (eV)')
            spectrum.set_ylabel('Signal')
            spectrum.grid(alpha=.25)
        self.helper = DrrGateBlitter(self.canvas)
        self.helper.configure(self.spectra, self.gates)
        self.canvas.draw()

    def assert_matches_full_draw(self):
        actual = np.asarray(self.canvas.buffer_rgba()).copy()
        self.helper.prepare_full_redraw()
        self.canvas.draw()
        delta = np.abs(actual.astype(int) - np.asarray(self.canvas.buffer_rgba()).astype(int))
        try:
            # Gate/spine intersections can differ by one antialiasing level;
            # all linecut text, ticks and curve pixels must otherwise match.
            self.assertLessEqual(np.count_nonzero(delta), 16)
            self.assertLessEqual(delta.max(), 1)
        finally:
            self.helper.restore_interactive_drawing()

    def test_gate_updates_titles_ticks_and_offsets_without_drawing_heatmaps(self):
        for step, scale in enumerate((1e-7, 5e4, 2.0)):
            with self.subTest(scale=scale):
                for spectrum, line, gate in zip(self.spectra, self.lines, self.gates):
                    line.set_ydata(np.cos(line.get_xdata()*20)*scale)
                    spectrum.set_title(f'Spectrum @ {step + 1} V')
                    spectrum.set_ylim(-scale, scale*1.1)
                    gate.set_ydata([step*.5, step*.5])
                with patch.object(self.canvas, 'draw', wraps=self.canvas.draw) as draw, \
                        patch.object(self.heat[0], 'draw', wraps=self.heat[0].draw) as heat_draw:
                    self.assertTrue(self.helper.draw())
                    draw.assert_not_called()
                    heat_draw.assert_not_called()
                self.assert_matches_full_draw()

    def test_resize_zoom_and_removed_axes_invalidate_background(self):
        self.heat[0].set_xlim(1.2, 1.8)
        self.assertFalse(self.helper.draw())
        self.canvas.draw()
        self.assertTrue(self.helper.draw())
        self.figure.set_size_inches(9, 6)
        self.assertFalse(self.helper.draw())
        self.canvas.draw()
        self.assertTrue(self.helper.draw())
        self.figure.delaxes(self.spectra[0])
        self.assertFalse(self.helper.draw())
        self.assertFalse(self.lines[0].get_animated())

    def test_save_and_async_snapshot_include_spectra_and_do_not_poison_background(self):
        for fmt in ('png', 'pdf', 'svg'):
            with self.subTest(format=fmt):
                # The async toolbar freezes the figure with the dynamic
                # children still animated. Saving must include every child.
                snapshot = pickle.loads(pickle.dumps(self.figure))
                FigureCanvasAgg(snapshot)
                metadata = {'Creator': 'test'}
                if fmt == 'pdf':
                    metadata.update(CreationDate=None, ModDate=None)
                elif fmt == 'svg':
                    metadata.update(Date=None)
                with mpl.rc_context({'svg.hashsalt': 'drr-test'}):
                    dynamic = BytesIO()
                    snapshot.savefig(dynamic, format=fmt, metadata=metadata)
                    for artist in snapshot.findobj():
                        artist.set_animated(False)
                    ordinary = BytesIO()
                    snapshot.savefig(ordinary, format=fmt, metadata=metadata)
                self.assertEqual(dynamic.getvalue(), ordinary.getvalue())
                self.figure.savefig(BytesIO(), format=fmt)
                self.assertIsNone(self.helper._background)
                self.canvas.draw()
                self.lines[0].set_ydata(np.zeros(31))
                self.assertTrue(self.helper.draw())
                self.assert_matches_full_draw()

    def test_overlay_and_color_changes_refresh_the_retained_heatmap(self):
        self.heat[0].collections[0].set_clim(0, 20)
        overlay = self.heat[0].scatter([1.4], [1.1], c='red', zorder=10)
        self.helper.configure(self.spectra, (*self.gates, overlay))
        self.canvas.draw()
        overlay.set_offsets([[1.7, 1.4]])
        self.assertTrue(self.helper.draw())
        self.assert_matches_full_draw()
        self.helper.disconnect()
        self.assertFalse(overlay.get_animated())
