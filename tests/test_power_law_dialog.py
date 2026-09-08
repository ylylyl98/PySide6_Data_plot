import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6.QtWidgets import QApplication, QDialogButtonBox
from core.power_peaks import PeakFit, export_peak_analysis
from core.power_multi_peaks import SpectrumFit, MultiPeakSettings, PeakSeed
from core.power_law_analysis import evaluate_ranges
from ui_qt.power_law_dialog import PowerLawDialog


class PowerLawDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.settings = MultiPeakSettings((PeakSeed(1.4), PeakSeed(1.5)))
        powers = np.geomspace(1, 100, 25)
        self.results = {'Power': tuple(SpectrumFit(float(p), 'ok',
            (PeakFit(float(p), 'ok', area=3*p, height=p, fwhm_mev=10, fwhm_error_mev=1),
             PeakFit(float(p), 'ok', area=2*p**1.7, height=p, fwhm_mev=12, fwhm_error_mev=1)), self.settings) for p in powers)}

    def test_multiselect_drag_suggest_and_invalid_range(self):
        dialog = PowerLawDialog(None, self.results, {})
        self.assertEqual(len(dialog.fits), 1)
        dialog.controls[1][0].setChecked(True)
        self.assertEqual(len(dialog.fits), 2)
        dialog.span_selected(2, 50)
        self.assertEqual(dialog.controls[1][1].value(), 2)
        self.assertAlmostEqual(dialog.fits[1]['alpha'], 1.7)
        dialog.suggest_ranges()
        self.assertIn('No reliable', dialog.message.text())
        self.assertFalse(dialog.shared.isChecked())
        self.assertTrue(dialog.ranges[0]['suggested'])
        dialog.local.setChecked(True)
        dialog.canvas.draw()
        self.assertEqual(len(dialog.figure.axes), 2)
        dialog.controls[1][2].setValue(1)
        self.assertFalse(dialog.buttons.button(QDialogButtonBox.Apply).isEnabled())
        dialog.reject()
        dialog.deleteLater()

    def test_export_contains_ranges_results_and_overlay(self):
        ranges = [dict(channel='Power', peak=1, lower=2., upper=50., suggested=True)]
        fits = evaluate_ranges(self.results, {}, ranges)
        payload = dict(results=self.results, records={'Power': ()}, settings=self.settings,
            background=0., metric='Integrated area', log_power=True,
            power_limits=(1., 100.), power_laws=fits, power_law_ranges=ranges)
        with tempfile.TemporaryDirectory() as folder:
            path = export_peak_analysis(folder, payload)
            metadata = json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(metadata['power_law_ranges'], ranges)
            self.assertAlmostEqual(metadata['power_law_fits'][0]['alpha'], 1.)
            self.assertTrue(path.with_name(path.stem + '_power_law.csv').exists())
            self.assertTrue(path.with_suffix('.png').exists())


if __name__ == '__main__':
    unittest.main()
