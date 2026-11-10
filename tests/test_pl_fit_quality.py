"""PL-only constrained fitting and bounded model-comparison checks."""
import unittest

import numpy as np

from core.pl_fit_quality import fit_pl_spectrum
from core.power_multi_peaks import multi_peak_model


class PLFitQualityTests(unittest.TestCase):
    def spectrum(self):
        x = np.linspace(1., 1.1, 501)
        y = multi_peak_model(x, [1., 0., 3., 1.025, .009, 2., 1.075, .012],
                             model='Lorentzian', origin=1.05)
        y += np.random.default_rng(13).normal(0., .01, len(x))
        points = [dict(branch_id=k, energy=e, width_mev=9., prominence=h)
                  for k, e, h in [('a', 1.025, 3.), ('fake', 1.05, .01), ('b', 1.075, 2.)]]
        return x, y, points

    def test_redundant_auto_component_removed_with_bic(self):
        x, y, points = self.spectrum()
        fit, selected, quality = fit_pl_spectrum(x, y, 0, points, 'Lorentzian', set())
        self.assertEqual([p['branch_id'] for p in selected], ['a', 'b'])
        self.assertEqual(len(fit.parameters), 2 + 3 * len(selected))
        self.assertEqual(quality['excluded_count'], 1)
        self.assertLessEqual(quality['alternatives_tested'], 3)
        self.assertEqual(quality['status'], 'no large residual flag')

    def test_shoulder_seed_gets_bounded_second_initialization(self):
        x, y, points = self.spectrum()
        points[1].update(candidate_origin='shoulder', detection_contrast=.02)
        _, _, quality = fit_pl_spectrum(x, y, 0, points, 'Lorentzian', set())
        self.assertEqual(quality['initializations_tested'], 2)
        self.assertLessEqual(quality['alternatives_tested'] + quality['initializations_tested'], 5)

    def test_manual_seed_is_never_pruned(self):
        x, y, points = self.spectrum()
        _, selected, _ = fit_pl_spectrum(x, y, 0, points, 'Lorentzian', {'fake'})
        self.assertIn('fake', [p['branch_id'] for p in selected])

    def test_automatic_center_is_limited_and_reviewed_at_bound(self):
        x, y, points = self.spectrum()
        # The dominant real feature sits well outside the automatic seed's
        # allowed ten-meV neighborhood; fitting cannot silently drift to it.
        seed = [dict(branch_id='a', energy=1.045, width_mev=50., prominence=3.)]
        fit, _, _ = fit_pl_spectrum(x, y, 0, seed, 'Lorentzian', set())
        self.assertLessEqual(abs(fit.components[0].center_ev - 1.045), .0100001)
        self.assertNotEqual(fit.components[0].status, 'ok')

    def test_adapter_preserves_rejected_point_and_parameter_branch_mapping(self):
        from core.loader import DataCube
        from core.peak_workspace import create_dataset
        from core.pl_peak_analysis import fit
        x, y, seeds = self.spectrum()
        cube = DataCube(x, np.array([0.]), y[None, :], 'Gate', 'PL', 'Intensity')
        dataset = create_dataset(cube, 'PL', 'PL', 'source', 'PL')
        dataset['branches'] = [dict(id=p['branch_id'], manual=False) for p in seeds]
        points = [dict(p, row_index=0, status='accepted', fit_center=1.05, height=999.,
                       area=999., area_error=1., rms=1.) for p in seeds]
        result = dict(dataset_key=dataset['key'], points=points, fits=[],
                      settings=dataset['settings'], method='Synthetic')
        output = fit(dataset, result, dataset['settings'])
        by_id = {p['branch_id']: p for p in output['points']}
        self.assertEqual(by_id['fake']['fit_status'], 'model_rejected')
        self.assertIsNone(by_id['fake']['fit_center'])
        for key in ('height', 'area', 'area_error', 'rms'):
            self.assertNotIn(key, by_id['fake'])
        record = output['fits'][0]
        self.assertEqual(record['branch_ids'], ['a', 'b'])
        self.assertEqual(len(record['parameters']), 2 + 3 * len(record['branch_ids']))
        self.assertEqual(record['fit_quality']['excluded_count'], 1)
        self.assertEqual(output['pl_fit_version'], 5)
        self.assertEqual(result['points'][1]['height'], 999.)
        # A moved automatic branch can carry manual ownership on the point,
        # even when the original branch metadata was not marked manual.
        result['points'][1]['manual'] = True
        manual_output = fit(dataset, result, dataset['settings'])
        self.assertIn('fake', manual_output['fits'][0]['branch_ids'])
        self.assertNotEqual(next(p for p in manual_output['points'] if p['branch_id'] == 'fake')['fit_status'],
                            'model_rejected')

    def test_at_most_three_alternatives_and_cancel_between_fits(self):
        x, y, points = self.spectrum()
        points = [dict(branch_id=str(i), energy=1.01 + .01*i, width_mev=4., prominence=i+1.)
                  for i in range(8)]
        _, _, quality = fit_pl_spectrum(x, y, 0, points, 'Gaussian', set())
        self.assertEqual(quality['alternatives_tested'], 3)
        self.assertEqual(quality['alternatives_skipped'], 5)
        with self.assertRaisesRegex(RuntimeError, 'cancel'):
            fit_pl_spectrum(x, y, 0, points, 'Gaussian', set(), lambda: True)


if __name__ == '__main__':
    unittest.main()
