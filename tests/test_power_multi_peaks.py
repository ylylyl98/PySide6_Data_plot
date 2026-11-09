import unittest
import numpy as np

from core.power_multi_peaks import (
    MultiPeakSettings, PeakSeed, fit_multi_spectrum,
    multi_peak_model, multi_peak_jacobian, detect_peak_seeds, detect_power_peak_seeds,
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


class DetectionTests(unittest.TestCase):
    @staticmethod
    def cube(x, rows):
        from types import SimpleNamespace
        return SimpleNamespace(energy=x, Z=np.asarray(rows))

    def test_weak_peak_visible_only_at_low_power(self):
        x = np.linspace(1.34, 1.56, 701)
        powers = np.geomspace(1, 100, 13)
        rows = [50*p/(1+(2*(x-1.40)/.010)**2) +
                40/(1+(2*(x-1.46)/.008)**2) for p in powers]
        seeds = detect_peak_seeds(self.cube(x, rows))
        np.testing.assert_allclose([s.center_ev for s in seeds], [1.40, 1.46], atol=.001)

    def test_close_resolved_peaks_and_nonuniform_energy(self):
        x = 1.34 + .22 * np.linspace(0, 1, 701)**1.3
        y = 100*np.exp(-4*np.log(2)*((x-1.42)/.006)**2) + 70*np.exp(-4*np.log(2)*((x-1.43)/.006)**2)
        seeds = detect_peak_seeds(self.cube(x, [y, 2*y, 3*y]))
        np.testing.assert_allclose([s.center_ev for s in seeds], [1.42, 1.43], atol=.001)
        self.assertEqual(len(detect_peak_seeds(self.cube(x, [y]), max_peaks=1)), 1)

    def test_flat_noise_and_invalid_axis(self):
        x = np.linspace(1.34, 1.56, 701)
        rng = np.random.default_rng(44)
        for rows in ([np.ones(len(x))], rng.normal(size=(8, len(x)))):
            with self.assertRaisesRegex(ValueError, "No reliable peaks"):
                detect_peak_seeds(self.cube(x, rows))
        with self.assertRaisesRegex(ValueError, "increasing"):
            detect_peak_seeds(self.cube(x[::-1], [x]))
        with self.assertRaisesRegex(ValueError, "one and eight"):
            detect_peak_seeds(self.cube(x, [x]), max_peaks=9)

    def test_second_channel_unique_peak(self):
        x = np.linspace(1.34, 1.56, 701)
        first = 100 / (1 + (2 * (x - 1.40) / .01)**2)
        second = first + 30 / (1 + (2 * (x - 1.46) / .008)**2)
        cubes = {"KK": self.cube(x, [first] * 5), "KKp": self.cube(x, [second] * 5)}
        seeds = detect_power_peak_seeds(cubes)
        np.testing.assert_allclose([s.center_ev for s in seeds], [1.40, 1.46], atol=.001)

    def test_transient_peak_and_single_row_second_channel(self):
        x = np.linspace(1.34, 1.56, 701)
        first = 100 / (1 + (2 * (x - 1.40) / .01)**2)
        extra = 45 / (1 + (2 * (x - 1.46) / .008)**2)
        rows = [first + (extra if i in (1, 2) else 0) for i in range(13)]
        for cubes in ({"KK": self.cube(x, rows)},
                      {"KK": self.cube(x, [first] * 16), "KKp": self.cube(x, [first + extra])}):
            seeds = detect_power_peak_seeds(cubes)
            np.testing.assert_allclose([s.center_ev for s in seeds], [1.40, 1.46], atol=.001)

    def test_small_drift_does_not_duplicate_seed(self):
        x = np.linspace(1.34, 1.56, 701)
        rows = [100 / (1 + (2 * (x - center) / .012)**2)
                for center in np.linspace(1.3995, 1.4005, 13)]
        seeds = detect_peak_seeds(self.cube(x, rows))
        self.assertEqual(len(seeds), 1)
        self.assertAlmostEqual(seeds[0].center_ev, 1.4, places=3)

    def test_helper_preserves_close_peaks_from_same_channel(self):
        x = np.linspace(1.34, 1.56, 701)
        y = 100*np.exp(-4*np.log(2)*((x-1.42)/.006)**2) + 70*np.exp(-4*np.log(2)*((x-1.43)/.006)**2)
        coarse = np.linspace(1.34, 1.56, 41)
        # A flat coarse channel has no suggestion but must not merge the fine peaks.
        cubes = {"fine": self.cube(x, [y]), "coarse": self.cube(coarse, [np.ones(len(coarse))])}
        for source in ({"fine": cubes["fine"]}, cubes):
            seeds = detect_power_peak_seeds(source)
            np.testing.assert_allclose([s.center_ev for s in seeds], [1.42, 1.43], atol=.001)
        # Limiting a single channel must retain its strongest rather than lowest peak.
        y2 = 30*np.exp(-4*np.log(2)*((x-1.42)/.006)**2) + 100*np.exp(-4*np.log(2)*((x-1.43)/.006)**2)
        seed = detect_power_peak_seeds({"fine": self.cube(x, [y2])}, max_peaks=1)[0]
        self.assertAlmostEqual(seed.center_ev, 1.43, places=3)
