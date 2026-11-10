import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from core.power_peaks import covariance_in_units, full_peak_area
from core.power_multi_peaks import PeakSeed, MultiPeakSettings, fit_multi_spectrum, multi_peak_model
from core.power_analysis_store import save_analysis, load_analysis

class PeakUncertaintyTests(unittest.TestCase):
    def data(self, model):
        x = np.linspace(1.3, 1.5, 501)
        y = multi_peak_model(x, [2, 3, 5, 1.4, .02], model=model, origin=x.mean())
        y = y + np.random.default_rng(13).normal(0, .006, len(x))
        return x, y

    def test_models_cross_covariance_and_roi(self):
        for model in ('Gaussian', 'Lorentzian'):
            x, y = self.data(model)
            fit = fit_multi_spectrum(x, y, 1, MultiPeakSettings((PeakSeed(1.4, .02),), model))
            c = fit.components[0]; self.assertEqual(c.uncertainty_status, 'ok')
            expected, factor = full_peak_area(c.height, c.fwhm_mev/1000, model)
            self.assertAlmostEqual(c.fit_area_full, expected)
            grad = np.array([0, 0, factor*c.fwhm_mev/1000, 0, factor*c.height])
            self.assertAlmostEqual(c.fit_area_full_error ** 2, grad @ fit.covariance @ grad, places=15)
            self.assertGreater(abs(fit.covariance[2, 4]), 0)
            self.assertAlmostEqual(fit.spectrum_area_roi, np.trapezoid(y-fit.baseline, x))
            self.assertAlmostEqual(c.area, np.trapezoid(fit.component_curves[0], x))
            self.assertEqual(fit.spectrum_area_roi_domain, (x[0], x[-1]))
            self.assertGreater(c.area_error, 0)

    def test_invalid_covariance_and_constraints(self):
        p = [0, 0, 1, 1.4, .02]; lower = [-np.inf, -np.inf, 0, 1.3, .001]; upper = [np.inf, np.inf, np.inf, 1.5, .1]
        for cov, params, status in [(np.full((5,5), np.inf), p, 'nonfinite covariance'),
                (np.ones((5,5)), p, 'singular covariance'),
                (np.eye(5), [0,0,0,1.4,.02], 'parameter at constraint')]:
            value, reason = covariance_in_units(cov, [1]*5, params, lower, upper)
            self.assertIsNone(value); self.assertEqual(reason, status)
        x,y=self.data('Lorentzian')
        from core import power_multi_peaks as module
        original = module.curve_fit
        def invalid(*args, **kwargs):
            params, cov = original(*args, **kwargs); return params, np.full_like(cov, np.inf)
        with patch.object(module, 'curve_fit', side_effect=invalid):
            fit=fit_multi_spectrum(x,y,1,MultiPeakSettings((PeakSeed(1.4,.02),)))
        self.assertIsNone(fit.covariance)
        self.assertTrue(np.isnan(fit.components[0].center_error_ev))
        self.assertNotEqual(fit.components[0].uncertainty_status, 'ok')

    def test_trend_uses_full_area_when_selected(self):
        from matplotlib.figure import Figure
        from core.power_peaks import draw_peak_trends
        x, y = self.data('Gaussian')
        fit = fit_multi_spectrum(x, y, 1, MultiPeakSettings((PeakSeed(1.4, .02),), 'Gaussian'))
        axes = Figure().subplots(1, 2)
        draw_peak_trends(axes, {'Power': [fit]}, log_power=False, metric='Full fitted area',
                         log_intensity=False, log_linewidth=False)
        self.assertEqual(axes[0].lines[0].get_ydata()[0], fit.components[0].fit_area_full)
        self.assertIn('Full fitted area', axes[0].get_ylabel())

    def test_store_v3_and_legacy_v2(self):
        x,y=self.data('Gaussian'); cube=SimpleNamespace(energy=x,gate=np.array([1.]),Z=np.array([y])); cubes={'P':cube}
        rows=[dict(center=1.4,width=20.,height=None,lower=None,upper=None)]
        fit=fit_multi_spectrum(x,y,1,MultiPeakSettings((PeakSeed(1.4,.02),),'Gaussian'))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fit.npz'; save_analysis(path,cubes,0,rows,'Gaussian',{'P':(fit,)})
            loaded=load_analysis(path,cubes,0)['results']['P'][0]
            np.testing.assert_allclose(loaded.covariance,fit.covariance)
            self.assertEqual(loaded.spectrum_area_roi,fit.spectrum_area_roi)
            self.assertEqual(loaded.components[0].fit_area_full,fit.components[0].fit_area_full)
            with np.load(path, allow_pickle=False) as archive: arrays={key:archive[key] for key in archive.files}
            meta=json.loads(arrays['metadata'].tobytes()); meta['version']=2
            item=meta['results']['P'][0]
            for key in ('covariance','spectrum_area_roi','spectrum_area_roi_domain','spectrum_area_roi_method'): item.pop(key)
            for component in item['components']:
                for key in ('center_error_ev','height_error','fit_area_full','fit_area_full_error','area_error','uncertainty_status'): component.pop(key)
            arrays['metadata']=np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8); np.savez(path,**arrays)
            legacy=load_analysis(path,cubes,0)['results']['P'][0]
            self.assertEqual(legacy.components[0].area, fit.components[0].area)
            self.assertTrue(np.isnan(legacy.components[0].fit_area_full)); self.assertIsNone(legacy.covariance)

if __name__ == '__main__': unittest.main()
