"""Post-fit shrinkage, disjoint-band context, and selector discontinuities."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_bn_calibration import (
    _fit, _raw_fit, _mask, _support, _METHODS, _select_candidate, _selection_sensitivity,
)
from core.drr_bn_d2_objective import _detrender, audit_training_influence, fit_regions, ranked_influence_probes
from core.drr_bn_joint import production_d2
from tests.test_drr_bn_calibration import WIDE_TRUSTED, wide_synthetic


class ShrinkageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = wide_synthetic()
        cls.e, cls.z, cls.b, cls.g, _ = cls.values
        cls.support = _support(cls.e, WIDE_TRUSTED)
        cls.fit_mask = _mask(cls.e, fit_regions(WIDE_TRUSTED)) & (cls.support >= 1-1e-12)
        cls.gates = np.arange(len(cls.g))

    def fit(self, method, settings=WIDE_TRUSTED, mask=None, data=None):
        return _fit(self.z if data is None else data, self.b, [0, 1], self.gates, self.e,
                    self.fit_mask if mask is None else mask, self.support, settings, method, None)

    def test_strength_is_fixed_after_the_same_raw_fit_and_alpha_one_is_compatible(self):
        full, raw = _raw_fit(self.z, self.b, [0, 1], self.gates, self.e, self.fit_mask,
                            self.support, WIDE_TRUSTED, 'local_shift_gain', None)
        for method, alpha in [('local_shift_gain_s25', .25), ('local_shift_gain_s50', .5),
                              ('local_shift_gain_s75', .75), ('local_shift_gain', 1.)]:
            candidate, params = self.fit(method)
            np.testing.assert_allclose(candidate, self.b*np.exp(alpha*np.log(full/self.b)), rtol=2e-15)
            self.assertEqual(params['shift_pixel'], raw['shift_pixel'])
            self.assertEqual(params['fine_gain'], raw['fine_gain'])
            self.assertEqual(params['correction_strength'], alpha)
            self.assertFalse(params['shrinkage']['alpha_is_fitted'])
            self.assertTrue(params['shrinkage']['post_fit'])
            self.assertEqual(params['unshrunk_parameters']['shift_pixel'], raw['shift_pixel'])
            if alpha == 1:
                np.testing.assert_array_equal(candidate, full)
            np.testing.assert_array_equal(candidate[self.support == 0], self.b[self.support == 0])

    def test_fixed_alpha_reduces_influence_without_cancelling_in_jacobian_inverse(self):
        audits = []
        for method in ('local_shift_gain_s25', 'local_shift_gain'):
            candidate, params = self.fit(method)
            audit = audit_training_influence(self.z, self.b, [0, 1], self.gates, self.e,
                self.fit_mask, self.support, WIDE_TRUSTED, method, candidate, params,
                lambda: None, lambda injected, frozen=method: self.fit(frozen, data=injected))
            self.assertEqual(audit['actual_refits']['count'], 3)
            self.assertTrue(all(row['correction_strength'] == params['correction_strength']
                                for row in audit['actual_refits']['probes']))
            self.assertEqual(audit['fit_jacobian_background'], 'unshrunk fitted BN')
            self.assertEqual(audit['forcing_background'], 'unshrunk fitted BN')
            refinement = audit['refinement']
            self.assertTrue(refinement['evaluated'])
            self.assertTrue(refinement['passed'])
            self.assertEqual(refinement['probe_count'], refinement['expected_probe_count'])
            self.assertLessEqual(refinement['probe_count'], 168)
            self.assertEqual(refinement['sigma_native_spacings'], [1., 1.25, 1.5, 2., 2.5, 3.])
            references = audit['actual_refits']['chosen_probe_references']
            ranked = ranked_influence_probes(audit)
            self.assertEqual(references, [dict(source=row['probe_source'], probe_index=row['probe_index']) for row in ranked])
            audits.append(audit)
        for product in ('drr', 'd2'):
            ratio = audits[0]['screening'][f'max_{product}_error_fraction']/audits[1]['screening'][f'max_{product}_error_fraction']
            self.assertGreater(ratio, .20)
            self.assertLess(ratio, .30)

    def test_batched_probe_sg_matches_individual_full_axis_outputs(self):
        rng = np.random.default_rng(2791)
        profiles = rng.normal(0, 2e-4, (8, 2, 9, len(self.e)))
        for energy in (self.e, self.e+.014*(self.e-1.81)**2):
            for descending in (False, True):
                axis = energy[::-1] if descending else energy
                rows = profiles[..., ::-1] if descending else profiles
                batched, used = production_d2(axis, rows, WIDE_TRUSTED)
                individual = np.stack([production_d2(axis, probe, WIDE_TRUSTED)[0] for probe in rows])
                self.assertEqual(used, WIDE_TRUSTED.derivative_window)
                np.testing.assert_allclose(batched, individual, rtol=1e-12, atol=1e-8)

    def test_native_refinement_rejects_coarse_only_pass_and_preserves_safe_family(self):
        from tests.test_drr_bn_d2_priority_independent import SETTINGS, independent_input
        values = independent_input(nonuniform=True)
        e, z, b, gates = (values[key] for key in ('energy', 'measurements', 'background', 'gates'))
        selected_gates = np.arange(len(gates))
        support = _support(e, SETTINGS)
        mask = _mask(e, fit_regions(SETTINGS)) & (support >= 1-1e-12)
        audits = []
        for method in ('local_shift_gain_s25', 'local_shift_s25'):
            candidate, params = _fit(z, b, [0, 1], selected_gates, e, mask, support, SETTINGS, method, None)
            audit = audit_training_influence(z, b, [0, 1], selected_gates, e, mask, support,
                SETTINGS, method, candidate, params, lambda: None,
                lambda injected, frozen=method: _fit(injected, b, [0, 1], selected_gates, e,
                    mask, support, SETTINGS, frozen, None))
            self.assertTrue(audit['screening']['passed'])
            self.assertEqual(audit['refinement']['probe_count'], audit['refinement']['expected_probe_count'])
            self.assertLessEqual(audit['refinement']['probe_count'], 168)
            self.assertTrue(any(probe['probe_source'] == 'refinement' for probe in audit['actual_refits']['probes']))
            audits.append(audit)
        self.assertFalse(audits[0]['passed'])
        self.assertGreater(audits[0]['refinement']['max_d2_error_fraction'], .05)
        self.assertGreater(audits[0]['actual_refits']['max_d2_error_fraction'], .05)
        self.assertTrue(audits[1]['passed'])
        self.assertLess(audits[1]['refinement']['max_d2_error_fraction'], .05)

    def test_disjoint_bands_remove_each_quadratic_independently_and_keep_full_sg_context(self):
        settings = replace(WIDE_TRUSTED, train_regions=((1.900, 2.080), (2.180, 2.400)))
        mask = _mask(self.e, settings.train_regions) & (self.support >= 1-1e-12)
        q = _detrender(self.e, mask, fit_regions(settings))
        values = np.where(self.e < 2.1, 4+3*self.e+2*self.e**2,
                          -7-5*self.e+8*self.e**2)[mask]
        np.testing.assert_allclose(values-(values@q)@q.T, 0, atol=5e-13)
        _, params = self.fit('local_shift_gain_s50', settings, mask)
        evidence = params['training_objective']
        self.assertEqual(evidence['channel_indices'], list(range(len(self.e))))
        self.assertEqual(len(evidence['fit_bands']), 2)
        self.assertTrue(evidence['band_consistency']['diagnostic_only'])
        self.assertEqual(evidence['bandwidth_check']['contiguous_scoring_segment_count'], 2)
        self.assertLess(evidence['bandwidth_check']['scoring_energy_span'], .23)
        for band in evidence['fit_bands']:
            self.assertNotEqual(band['channel_count'], int(mask.sum()))
            self.assertEqual(len(band['channel_indices']), band['channel_count'])
            self.assertEqual(band['basis_rank'], 2)
        # The band complement supplies real SG context, so perturbing a nearby
        # unscored native channel is permitted to change the training estimate.
        injected = self.z.copy()
        outside = int(np.flatnonzero(self.e > 2.080)[0])
        injected[:2, :, outside] *= np.exp(.002)
        _, changed = self.fit('local_shift_gain_s50', settings, mask, injected)
        self.assertNotAlmostEqual(params['shift_pixel'], changed['shift_pixel'], places=7)

    def test_training_qualification_precedes_selection_even_when_full_strength_scores_best(self):
        items = {}
        for name in _METHODS[1:]:
            items[name] = dict(parameters=dict(optimizer_success=True),
                selection_joint=dict(passed=False, ranking_objective=.9),
                training_influence_audit=dict(passed=True))
        items['local_shift_gain']['selection_joint'] = dict(passed=True, ranking_objective=.05)
        items['local_shift_gain']['training_influence_audit']['passed'] = False
        items['local_shift_gain_s25']['selection_joint'] = dict(passed=True, ranking_objective=.7)
        self.assertEqual(_select_candidate(items), 'local_shift_gain_s25')
        items['local_shift_gain_s25']['training_influence_audit']['passed'] = False
        self.assertEqual(_select_candidate(items), 'original')

    def test_selector_audit_counts_original_jump_and_physically_excludes_final_spectra(self):
        candidate, params = self.fit('local_shift_gain')
        probes = [dict(probe_index=i, center_energy=1.98+.05*i, sigma_energy=.00075,
                       sign=1 if i % 2 else -1, drr_error_fraction=.03-i*.001,
                       d2_error_fraction=.02+i*.001) for i in range(3)]
        records = {name: dict(parameters=dict(optimizer_success=True),
            selection_joint=dict(passed=True), training_influence_audit=dict(passed=True))
            for name in _METHODS[1:]}
        records['local_shift_gain']['parameters'] = params
        records['local_shift_gain']['training_influence_audit']['screening'] = dict(probes=probes)
        alternatives = dict(records, original=dict(parameters={}))
        with patch('core.drr_bn_calibration._candidate_stage',
                   return_value=({'original': self.b}, alternatives, 'original', 21)) as rerun:
            audit = _selection_sensitivity(self.z, self.b, [0, 1], [2], self.gates,
                self.e, self.fit_mask, _mask(self.e, [WIDE_TRUSTED.validation_region]),
                self.support, WIDE_TRUSTED, None, 'local_shift_gain',
                {'local_shift_gain': candidate}, records)
        self.assertFalse(audit['passed'])
        self.assertTrue(audit['selection_aware'])
        self.assertFalse(audit['final_repeats_read'])
        self.assertEqual(audit['count'], 3)
        self.assertTrue(all(row['original_fallback_jump_included'] for row in audit['probes']))
        self.assertGreater(audit['max_drr_error_fraction'], .05)
        for call in rerun.call_args_list:
            received = call.args[0]
            self.assertEqual(received.shape[0], 3)
            np.testing.assert_array_equal(received[2], self.z[2])
            self.assertFalse(np.array_equal(received[:2], self.z[:2]))


if __name__ == '__main__':
    unittest.main()
