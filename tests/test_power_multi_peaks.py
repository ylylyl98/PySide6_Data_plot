import unittest
import numpy as np

from core.power_multi_peaks import (
    MultiPeakSettings, PeakSeed, fit_multi_spectrum,
    multi_peak_model, multi_peak_jacobian,
)


class MultiPeakTests(unittest.TestCase):
    def test_overlapping_components_full_spectrum_and_crossing_heights(self):
        x = np.linspace(1.34, 1.56, 701)
        for model in ("Lorentzian", "Gaussian"):
            settings = MultiPeakSettings((PeakSeed(1.412, .03), PeakSeed(1.438, .02)), model)
            for heights in ((300, 800), (900, 200)):
                parameters = [600, 50, heights[0], 1.41, .035, heights[1], 1.44, .018]
                y = multi_peak_model(x, parameters, model=model, origin=x.mean())
                fit = fit_multi_spectrum(x, y, 2., settings)
                self.assertEqual(fit.status, "ok")
                np.testing.assert_allclose(fit.x, x)
                np.testing.assert_allclose([p.center_ev for p in fit.components], [1.41, 1.44], atol=1e-7)
                np.testing.assert_allclose([p.fwhm_mev for p in fit.components], [35, 18], rtol=1e-5)
                np.testing.assert_allclose([p.height for p in fit.components], heights, rtol=1e-5)
                np.testing.assert_allclose(fit.baseline + sum(fit.component_curves), y, atol=1e-5)
                self.assertLess(fit.rms, 1e-5)

    def test_analytic_derivatives(self):
        x = np.linspace(1.34, 1.56, 101)
        p = np.array([4., 10., 30., 1.41, .035, 80., 1.44, .018])
        for model in ("Lorentzian", "Gaussian"):
            jac = multi_peak_jacobian(x, p, model=model, origin=x.mean())
            for i in range(len(p)):
                step = np.zeros_like(p)
                step[i] = 1e-7
                difference = (multi_peak_model(x, p + step, model=model, origin=x.mean()) -
                              multi_peak_model(x, p - step, model=model, origin=x.mean())) / 2e-7
                np.testing.assert_allclose(jac[:, i], difference, rtol=1e-5, atol=2e-6)
