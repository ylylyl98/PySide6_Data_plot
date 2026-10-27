"""Independent physical-output QA for fixed post-fit BN shrinkage.

The clean analytic generator is frozen in the preceding independent QA. New
Lorentzian, odd and asymmetric sample features deliberately differ from the
Gaussian family used by the production influence audit. All comparisons use
raw measurement ratios and the real SG pipeline; fallback is never counted as
successful correction. Multiple declared fit bands are synthetic evidence only.
"""
from dataclasses import replace
import json
import unittest

import numpy as np

from core.drr_bn_store import joint_validation_passed

from tests.test_drr_bn_d2_priority_independent import (
    SETTINGS, WIDE_SETTINGS, d2, evaluate, independent_input, wide_input,
)


MULTIBAND_SETTINGS = replace(
    WIDE_SETTINGS, train_regions=((1.885, 2.015), (2.085, 2.235)))


def inject_unseen(inputs, amplitude, center, width, *, shape='lorentzian',
                  mode='common', repeat_ids=None):
    result = {key: array.copy() for key, array in inputs.items()}
    energy = inputs['energy']
    repeat_ids = list(range(len(inputs['measurements']))) if repeat_ids is None else list(repeat_ids)
    for gate_index, gate in enumerate(inputs['gates']):
        if mode == 'single_gate' and gate_index != len(inputs['gates'])-1:
            continue
        position = center+.00167*gate if mode == 'dispersive' else center
        x = (energy-position)/width
        if shape == 'lorentzian':
            feature = 1/(1+x*x)
        elif shape == 'odd':
            feature = x*np.exp((1-x*x)/2)
        elif shape == 'asymmetric':
            feature = .7*np.exp(-.5*x*x)+.3*np.exp(-.5*((x-1.15)/1.7)**2)
            feature /= np.max(feature)
        else:
            raise ValueError(shape)
        result['measurements'][repeat_ids, gate_index] *= np.exp(amplitude*feature)
    return result


def physical_counterfactual(clean, injected, before_background, after_background,
                            *, window=21, repeat_ids=None):
    ids = list(range(len(clean['measurements']))) if repeat_ids is None else list(repeat_ids)
    baseline_raw = clean['measurements'][ids].mean(0)
    altered_raw = injected['measurements'][ids].mean(0)
    expected = (altered_raw-baseline_raw)/before_background
    observed = altered_raw/after_background-baseline_raw/before_background
    expected_d2 = d2(clean['energy'], clean['gates'], expected, window)
    observed_d2 = d2(clean['energy'], clean['gates'], observed, window)
    errors = {}
    for label, known, recovered in (('drr', expected, observed),
                                    ('d2', expected_d2, observed_d2)):
        scale = max(float(np.max(abs(known))), 1e-15)
        errors[label+'_fractional_adaptation_error'] = float(np.max(abs(recovered-known))/scale)
    return dict(**errors, known_drr=expected, recovered_drr=observed,
                known_d2=expected_d2, recovered_d2=observed_d2)


def preview_background(result):
    """The frozen pre-final winner, including when final confirmation vetoes."""
    return result.candidate_backgrounds[result.selected_method]


class IndependentShrinkageTests(unittest.TestCase):
    def assert_contract(self, result):
        report = result.report
        self.assertEqual(report['schema_version'], 4)
        self.assertEqual(report['optimization_policy'], 'd2_shrinkage_v1')
        self.assertEqual(report['joint_validation']['schema_version'], 3)
        self.assertTrue(report['selection_frozen'])
        json.dumps(report, allow_nan=False)
        if result.eligible:
            self.assertNotEqual(result.applied_method, 'original')
            self.assertTrue(report['joint_validation']['passed'])
            self.assertTrue(report['joint_validation']['influence_audit']['passed'])
            influence = report['joint_validation']['influence_audit']
            self.assertEqual(influence['schema_version'], 3)
            self.assertEqual(influence['policy'], 'training_feature_influence_shrinkage_v2')
            refinement = influence['refinement']
            self.assertTrue(refinement['evaluated'])
            self.assertTrue(refinement['passed'])
            self.assertEqual(refinement['policy'], 'native_feature_grid_v1')
            self.assertEqual(refinement['center_offsets_native_spacings'], [-1., -.5, 0., .5, 1.])
            self.assertEqual(refinement['sigma_native_spacings'], [1., 1.25, 1.5, 2., 2.5, 3.])
            self.assertEqual(refinement['probe_count'], refinement['expected_probe_count'])
            self.assertGreater(refinement['probe_count'], 0)
            self.assertLessEqual(refinement['probe_count'], 168)
            # Both products' most influential probe must feed actual raw
            # refits and the complete pre-final selector, including refinement.
            combined = [('coarse', probe) for probe in influence['screening']['probes']]
            combined += [('refinement', probe) for probe in refinement['probes']]
            actual_refs = {(probe['probe_source'], probe['probe_index'])
                           for probe in influence['actual_refits']['probes']}
            selector_refs = {(probe['source'], probe['probe_index'])
                for probe in report['joint_validation']['selection_sensitivity_audit']['chosen_probe_references']}
            for metric in ('drr_error_fraction', 'd2_error_fraction'):
                source, probe = max(combined, key=lambda item: item[1][metric])
                self.assertIn((source, probe['probe_index']), actual_refs)
                self.assertIn((source, probe['probe_index']), selector_refs)
            self.assertTrue(report['joint_validation']['selection_sensitivity_audit']['passed'])
            self.assertGreaterEqual(report['final_confirmation_d2']['mean_improvement_fraction'], .10)
            self.assertTrue(joint_validation_passed(report, report['settings']),
                            'A real numerical accepted report must pass persistence evidence validation')
        else:
            np.testing.assert_array_equal(result.applied_background, result.original_background)

    def test_postfit_strength_uses_geometric_denominator_not_scaled_parameter_fit(self):
        # Both contracts read the same deterministic fit; evaluate it once.
        inputs = independent_input(nonuniform=True)
        result = evaluate(inputs)
        with self.subTest(contract='geometric_denominator_not_scaled_parameter_fit'):
            self.assert_contract(result)
            expected = {'original'} | {family+suffix for family in ('local_shift', 'local_shift_gain')
                                       for suffix in ('_s25', '_s50', '_s75', '')}
            self.assertEqual(set(result.candidate_backgrounds), expected)
            background = inputs['background']
            for family in ('local_shift', 'local_shift_gain'):
                full = result.candidate_backgrounds[family]
                self.assertGreater(np.max(abs(np.log(full/background))), 1e-6)
                for suffix, strength in (('s25', .25), ('s50', .5), ('s75', .75)):
                    name = family+'_'+suffix
                    actual = result.candidate_backgrounds[name]
                    expected = background*np.exp(strength*np.log(full/background))
                    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)
                    parameters = result.report['candidates'][name]['parameters']
                    self.assertEqual(parameters['correction_strength'], strength)
                    self.assertEqual(parameters['family'], family)
                    self.assertEqual(parameters['unshrunk_parameters']['shift_pixel'],
                                     result.report['candidates'][family]['parameters']['shift_pixel'])

        with self.subTest(contract='alpha_does_not_cancel_from_fixed_family_influence_screen'):
            for family in ('local_shift', 'local_shift_gain'):
                candidates = result.report['candidates']
                full = candidates[family]['training_influence_audit']['screening']
                low = candidates[family+'_s25']['training_influence_audit']['screening']
                # A pseudoinverse of the shrunk output Jacobian would incorrectly
                # restore the full absorption response and make these ~equal.
                self.assertGreater(full['max_d2_error_fraction'], .05)
                self.assertLess(low['max_d2_error_fraction'], .36*full['max_d2_error_fraction'])
                self.assertGreater(low['max_d2_error_fraction'], .15*full['max_d2_error_fraction'])

    def test_unseen_shapes_use_complete_selection_and_actual_nonuniform_sg(self):
        clean = wide_input(nonuniform=True)
        before = evaluate(clean, WIDE_SETTINGS)
        self.assert_contract(before)
        self.assertTrue(before.eligible, before.reasons)
        cases = (
            (2e-4, 1.92437, .00093, 'lorentzian', 'common'),
            (-2e-4, 2.12173, .00127, 'odd', 'dispersive'),
            (1.4e-3, 2.17891, .00183, 'asymmetric', 'single_gate'),
        )
        for amplitude, center, width, shape, mode in cases:
            with self.subTest(shape=shape, mode=mode):
                injected = inject_unseen(clean, amplitude, center, width, shape=shape, mode=mode)
                after = evaluate(injected, WIDE_SETTINGS)
                self.assert_contract(after)
                # This is a declared, unchanged positive information domain.
                # Requiring accepted outputs prevents fallback-zero success.
                self.assertTrue(after.eligible, after.reasons)
                metrics = physical_counterfactual(clean, injected, before.applied_background,
                                                   after.applied_background)
                self.assertLess(metrics['drr_fractional_adaptation_error'], .05)
                self.assertLess(metrics['d2_fractional_adaptation_error'], .05)

    def test_explicit_multiple_bands_preserve_actual_signal_and_frozen_scope(self):
        clean = wide_input(nonuniform=True)
        before = evaluate(clean, MULTIBAND_SETTINGS)
        self.assert_contract(before)
        self.assertTrue(before.eligible, before.reasons)
        score_ids = before.report['scope']['fit_channel_indices']
        expected = ((clean['energy'] >= 1.885) & (clean['energy'] <= 2.015))
        expected |= ((clean['energy'] >= 2.085) & (clean['energy'] <= 2.235))
        np.testing.assert_array_equal(score_ids, np.flatnonzero(expected))
        injected = inject_unseen(clean, 2e-4, 2.14831, .00113, shape='odd', mode='common')
        after = evaluate(injected, MULTIBAND_SETTINGS)
        self.assert_contract(after)
        self.assertTrue(after.eligible, after.reasons)
        self.assertEqual(before.report['scope']['fit_channel_indices'],
                         after.report['scope']['fit_channel_indices'])
        metrics = physical_counterfactual(clean, injected, before.applied_background,
                                           after.applied_background)
        self.assertLess(metrics['drr_fractional_adaptation_error'], .05)
        self.assertLess(metrics['d2_fractional_adaptation_error'], .05)

    def test_final_only_damage_keeps_all_candidates_and_pre_final_winner_frozen(self):
        clean = wide_input(nonuniform=True)
        before = evaluate(clean, WIDE_SETTINGS)
        altered = {key: array.copy() for key, array in clean.items()}
        for repeat in before.report['split']['final']:
            altered['measurements'][repeat] *= (clean['background']/clean['true_references'][repeat])**2
        after = evaluate(altered, WIDE_SETTINGS)
        self.assert_contract(before)
        self.assert_contract(after)
        self.assertTrue(before.eligible)
        self.assertFalse(after.eligible)
        self.assertEqual(before.selected_method, after.selected_method)
        for name, candidate in before.candidate_backgrounds.items():
            np.testing.assert_array_equal(candidate, after.candidate_backgrounds[name])
            for key in ('parameters', 'selection', 'selection_d2', 'selection_joint',
                        'training_influence_audit'):
                if key in before.report['candidates'][name]:
                    self.assertEqual(before.report['candidates'][name][key],
                                     after.report['candidates'][name][key])
        self.assertEqual(before.report['joint_validation']['selection_sensitivity_audit'],
                         after.report['joint_validation']['selection_sensitivity_audit'])

    def test_nonzero_strengths_cannot_override_no_drift_or_adverse_gate(self):
        cases = (independent_input(instrument_shift=0., fine_gain=1.),
                 independent_input(opposite_gate=True))
        for inputs in cases:
            with self.subTest(no_drift=np.array_equal(inputs['true_references'][0, 0], inputs['background'])):
                result = evaluate(inputs, replace(SETTINGS, gate_count=5))
                self.assert_contract(result)
                self.assertFalse(result.eligible)
                for name, candidate in result.report['candidates'].items():
                    if name != 'original':
                        self.assertFalse(candidate['selection_joint']['passed'] and
                            candidate['training_influence_audit']['passed'])

    def test_trust_and_final_availability_veto_application_without_hiding_pre_final_winner(self):
        clean = wide_input(nonuniform=True, repeats=4)
        before = evaluate(clean, WIDE_SETTINGS)
        without_trust = evaluate(clean, replace(WIDE_SETTINGS, trusted_regions=()))
        two_repeats = {key: array.copy() for key, array in clean.items()}
        two_repeats['measurements'] = two_repeats['measurements'][:2]
        without_final = evaluate(two_repeats, WIDE_SETTINGS)
        self.assertNotEqual(before.selected_method, 'original')
        for altered in (without_trust, without_final):
            self.assertFalse(altered.eligible)
            self.assertEqual(before.selected_method, altered.selected_method)
            for name, background in before.candidate_backgrounds.items():
                np.testing.assert_array_equal(background, altered.candidate_backgrounds[name])
            np.testing.assert_array_equal(altered.applied_background, clean['background'])

    def test_unseen_neighboring_common_peak_cannot_be_accepted_above_five_percent(self):
        """Frozen independent failure discovered after the 4.993% boundary.

        No masks or thresholds change in response to the injection. The old
        selector accepted gain_s25 with 5.2501988% full-axis D2 adaptation;
        either actual accepted transfer must improve or application must be
        conservatively rejected. Original-BN fallback is not correction gain.
        """
        from tests.test_drr_bn_d2_priority_independent import inject
        clean = independent_input(nonuniform=True)
        injected = inject(clean, .0002, 1.91603, .00068)
        before, after = evaluate(clean), evaluate(injected)
        self.assert_contract(before)
        self.assert_contract(after)
        if after.eligible:
            self.assertTrue(before.eligible, 'A selection mode change must also pass actual output transfer')
            self.assertNotEqual(after.applied_method, 'original')
            metrics = physical_counterfactual(clean, injected, before.applied_background,
                                               after.applied_background)
            self.assertLess(metrics['drr_fractional_adaptation_error'], .05)
            self.assertLess(metrics['d2_fractional_adaptation_error'], .05)
        else:
            np.testing.assert_array_equal(after.applied_background, clean['background'])


if __name__ == '__main__':
    unittest.main()
