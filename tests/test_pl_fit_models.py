import unittest
import numpy as np

from core.pl_fit_models import PLSeed, evaluate_component, evaluate_fit, fit_pl_model
from core.pl_fit_quality import _bic, fit_pl_spectrum
from core.power_multi_peaks import multi_peak_model


class PLFitModelsTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(1., 1.1, 501)
        self.record = dict(model='Asymmetric Lorentzian', origin=1.05,
            parameters=[1., .02, 3., 1.04, .01, 1.5, 1.075, .008, .35], asymmetric_index=0)
        self.y = evaluate_fit(self.x, self.record) + np.random.default_rng(3).normal(0., .003, len(self.x))

    def test_old_records_match_existing_evaluator(self):
        record = dict(model='Gaussian', origin=1.05, parameters=[1., 2., 3., 1.04, .01])
        np.testing.assert_allclose(evaluate_fit(self.x, record),
            multi_peak_model(self.x, record['parameters'], model='Gaussian', origin=1.05))
        self.assertAlmostEqual(evaluate_component(np.array([1.04]), record, 0)[0], 3.)

    def test_asymmetry_and_fixed_center_recover_data(self):
        seeds = [PLSeed(1.04, .01, center_min_ev=1.03, center_max_ev=1.05, center_fixed=True),
                 PLSeed(1.075, .008, center_min_ev=1.06, center_max_ev=1.09)]
        result, meta = fit_pl_model(self.x, self.y, 0., seeds, 'Asymmetric Lorentzian', asymmetric_index=0)
        self.assertEqual(len(result.parameters), 9)
        self.assertEqual(meta['n_vary'], 8)
        self.assertEqual(result.parameters[3], 1.04)
        self.assertAlmostEqual(result.parameters[-1], .35, places=2)
        self.assertEqual(result.components[0].uncertainty_status, 'user fixed center')
        self.assertTrue(np.all(result.covariance[3, :] == 0))
        self.assertTrue(np.all(result.covariance[:, 3] == 0))
        self.assertAlmostEqual(result.components[0].center_error_ev, 0.)
        fit = dict(model='Asymmetric Lorentzian', origin=float(self.x.mean()),
                   parameters=result.parameters, asymmetric_index=0)
        np.testing.assert_allclose(evaluate_fit(self.x, fit), result.fitted)
        # The full area is invariant to side imbalance at fixed total FWHM.
        self.assertAlmostEqual(result.components[0].fit_area_full,
            np.pi * result.parameters[2] * result.parameters[4] / 2, places=8)

    def test_strong_manual_peak_not_leftmost_gets_asymmetry(self):
        record = dict(model='Asymmetric Lorentzian', origin=1.05,
                      parameters=[1., 0., 1.5, 1.025, .008, 3., 1.075, .01, .3],
                      asymmetric_index=1)
        y = evaluate_fit(self.x, record) + np.random.default_rng(7).normal(0., .002, len(self.x))
        points = [dict(branch_id='auto_weak', energy=1.025, width_mev=8., prominence=1.5),
                  dict(branch_id='manual_strong', energy=1.075, width_mev=10., prominence=0., manual=True)]
        fit, selected, _ = fit_pl_spectrum(self.x, y, 0., points, 'Asymmetric Lorentzian', {'manual_strong'})
        self.assertEqual(selected[fit.pl_meta['asymmetric_index']]['branch_id'], 'manual_strong')

    def test_true_fixed_center_for_symmetric_model_and_bic_count(self):
        points = [dict(branch_id='manual', energy=1.04, width_mev=10., prominence=3.,
                       fit_center_min=1.035, fit_center_max=1.045, fit_center_fixed=True)]
        result, selected, quality = fit_pl_spectrum(self.x, self.y, 0, points, 'Lorentzian', {'manual'})
        self.assertEqual(result.parameters[3], 1.04)
        self.assertEqual(result.pl_meta['n_vary'], len(result.parameters)-1)
        n = len(result.residual)
        expected = n*np.log(np.dot(result.residual, result.residual)/n) + result.pl_meta['n_vary']*np.log(n)
        self.assertAlmostEqual(_bic(result), expected)
        self.assertEqual(len(selected), 1)

    def test_adapter_persists_fixed_center_and_asymmetric_metadata(self):
        from core.loader import DataCube
        from core.peak_workspace import create_dataset
        from core.pl_peak_analysis import fit
        cube = DataCube(self.x, np.array([0.]), self.y[None, :], 'Gate', 'PL', 'Intensity')
        dataset = create_dataset(cube, 'PL', 'PL', 'source', 'PL')
        dataset['branches'] = [dict(id='manual', manual=True), dict(id='auto', manual=False)]
        dataset['settings']['model'] = 'Asymmetric Lorentzian'
        points = [dict(branch_id='manual', row_index=0, status='accepted', energy=1.04,
                       width_mev=10., prominence=3., fit_center_fixed=True,
                       fit_center_min=1.03, fit_center_max=1.05),
                  dict(branch_id='auto', row_index=0, status='accepted', energy=1.075,
                       width_mev=8., prominence=1.5)]
        result = dict(dataset_key=dataset['key'], points=points, fits=[],
                      settings=dataset['settings'], method='Synthetic')
        output = fit(dataset, result, dataset['settings'])
        record = output['fits'][0]
        self.assertIn('manual', record['branch_ids'])
        self.assertIn('manual', record['fixed_center_branch_ids'])
        self.assertEqual(record['parameters'][3], 1.04)
        self.assertEqual(record['asymmetric_branch_id'], 'manual')
        self.assertEqual(record['n_vary'], len(record['parameters'])-1)
        self.assertEqual(output['pl_fit_version'], 5)

    def test_bounds_reject_invalid_and_cancel_is_interruptible(self):
        with self.assertRaises(ValueError):
            fit_pl_model(self.x, self.y, 0, [PLSeed(1.04, .01, center_min_ev=1.05)], 'Lorentzian')
        with self.assertRaises(InterruptedError):
            fit_pl_model(self.x, self.y, 0, [PLSeed(1.04, .01)], 'Lorentzian', cancelled=lambda: True)


if __name__ == '__main__': unittest.main()
