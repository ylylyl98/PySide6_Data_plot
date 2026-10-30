"""Bound display work per field, independent of the number of peak markers."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickMcdPeakResponsivenessTests(unittest.TestCase):
    def test_isolated_field_updates(self):
        with TemporaryDirectory() as directory:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                               MPLCONFIGDIR=directory, LOCALAPPDATA=directory, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                                    cwd=Path(__file__).resolve().parents[1], env=environment,
                                    capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from types import SimpleNamespace
    from time import perf_counter
    from unittest.mock import patch
    import numpy as np
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QGuiApplication
    from core.mcd_peak_shift import PeakPoint, PeakTrack, PeakShiftResult
    from ui_qt.quick.mcd_peak_bridge import McdPeakBridge
    from ui_qt.quick import mcd_peak_plot as plotting

    app = QGuiApplication([])

    class Cases(unittest.TestCase):
        def setUp(self):
            self.directory = TemporaryDirectory()
            self.addCleanup(self.directory.cleanup)
            self.bridge = McdPeakBridge(settings=QSettings(str(Path(self.directory.name)/"test.ini"), QSettings.IniFormat))
            self.addCleanup(self.bridge.close)
            energy = np.linspace(1.5, 1.9, 1501)
            fields = np.linspace(-2, 2, 401)
            branch = "B increasing"
            centers = np.linspace(1.54, 1.86, 6)
            signal = sum(100*np.exp(-((energy-center)/.006)**2) for center in centers)
            spectra = 10000 + signal[None, :] + fields[:, None]*signal[None, :]*.01
            result = SimpleNamespace(energy_ev=energy, wavelength_nm=1239.841984/energy,
                pair_b=fields, pair_b_pos=fields, pair_b_neg=fields,
                pair_interpolated_pos=np.zeros(fields.size, bool), pair_interpolated_neg=np.zeros(fields.size, bool),
                pair_labels=np.repeat(branch, fields.size), pos_angle=0., neg_angle=90.,
                pair_raw_pos=spectra, pair_raw_neg=spectra, pair_corrected_pos=spectra*1.01, pair_corrected_neg=spectra*1.01)
            request = object()
            self.bridge._products = SimpleNamespace(result=result, request=request)
            self.bridge._requested = request
            self.bridge._branch = branch
            self.bridge._values.update(xmin=1.5, xmax=1.9)
            tracks = tuple(PeakTrack(i, branch, tuple(PeakPoint(float(field), branch, float(center), 0., "tracked")
                         for field in fields), float(center), 0., "exact 0 T") for i, center in enumerate(centers, 1))
            analysis = PeakShiftResult(fields, result.pair_labels, (), tracks, "raw pos")
            self.bridge._methods = {"Raw spectrum": {"pos": analysis, "neg": analysis}}
            self.bridge._peak_selected = ("pos", 1, branch, "peak")
            self.bridge._render()
            self.bridge._canvas.draw()

        def test_one_derivative_per_channel_and_marker_reuse(self):
            bridge = self.bridge
            with patch.object(plotting, "second_derivative_map", wraps=plotting.second_derivative_map) as derivative:
                bridge._peak_field = 250
                started = perf_counter()
                plotting.update_field(bridge, draw=False)
                elapsed = perf_counter()-started
                self.assertEqual(derivative.call_count, 2, "Each marker repeated the channel derivative")
                marks = tuple(bridge._candidate_marks)
                plotting.update_field(bridge, draw=False)
                self.assertEqual(derivative.call_count, 2, "Unchanged field repeated display derivatives")
                self.assertEqual(tuple(bridge._candidate_marks), marks, "Cursor updates recreated marker artists")
                print(f"MCD_PEAK_FIELD_UPDATE_MS={elapsed*1000:.2f}")
            for line in bridge._d2_lines.values():
                grid, expected, _ = plotting.second_derivative_map(bridge._products.result.energy_ev,
                    bridge._products.result.pair_raw_pos[250:251], bridge._values["derivativeWindow"])
                np.testing.assert_array_equal(line.get_xdata(), grid)
                np.testing.assert_allclose(line.get_ydata(), -expected[0], equal_nan=True)

        def test_structural_redraw_reuses_source_and_invalidates_for_new_source(self):
            bridge = self.bridge
            with patch.object(plotting, "source_spectra", wraps=plotting.source_spectra) as source:
                bridge._values["showPeakMaps"] = True
                bridge._render()
                bridge._render()
                self.assertEqual(source.call_count, 0, "Redrawing copied both full source matrices")
                bridge._values["peakSource"] = "MCD-corrected R"
                bridge._render()
                self.assertEqual(source.call_count, 2)

        def test_derivative_maps_are_not_recomputed_on_feature_redraw(self):
            bridge = self.bridge
            bridge._values.update(showPeakMaps=True, mapMode="Second derivative")
            with patch.object(plotting, "second_derivative_map", wraps=plotting.second_derivative_map) as derivative:
                bridge._render()
                self.assertEqual(derivative.call_count, 2)
                bridge._peak_selected = ("pos", 2, bridge._branch, "peak")
                bridge._render()
                self.assertEqual(derivative.call_count, 2)
                bridge._values["derivativeWindow"] = 25
                bridge._render()
                self.assertEqual(derivative.call_count, 4)

    outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    return 0 if outcome.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(isolated())
    unittest.main()
