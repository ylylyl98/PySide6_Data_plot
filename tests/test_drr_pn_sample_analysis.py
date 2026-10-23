"""Experimental array contracts, not scientific validation of a background."""
import builtins
import unittest
from unittest.mock import patch

import numpy as np
from scipy.interpolate import PPoly

from core import drr_pn_sample_analysis as S
from core.drr_pn_generator import generate_pn_arrays, PnGeneratorSettings
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope


class SampleAssistedArrayTests(unittest.TestCase):
    def setUp(self):
        # Different domain/size from the historical instrument measurements.
        self.e = np.linspace(.9, 1.35, 901)
        self.logbase = 7 + .2*(self.e-1.1) - .4*(self.e-1.1)**2
        self.base = np.exp(self.logbase)
        self.gates = np.array([-1.7, .2, 3.1])

    def texture_sources(self):
        phase = 2*np.pi*(self.e-self.e[0])/.012
        return self.base*np.exp(.004*np.sin(phase)), self.base*np.exp(.004*np.cos(phase))

    def test_matching_sample_texture_actually_selects_source_and_never_enters_intensity(self):
        p, n = self.texture_sources()
        samples = np.broadcast_to(n, (2, 3, len(self.e))).copy()
        arrays, report = S.generate_sample_assisted_arrays(self.e, p, n, samples, self.gates)
        self.assertGreater(arrays['sample_N_witness'].sum(), 50)
        self.assertGreater(report['pn_only_comparison']['changed_weight_points'], 50)
        self.assertTrue(np.all(arrays['source_weights'][1] == 1))
        np.testing.assert_allclose(arrays['background'], n, rtol=3e-14)
        self.assertEqual(report['algorithm_version'], S.ALGORITHM_VERSION)
        self.assertTrue(report['sample_used'])
        self.assertTrue(report['samples_analyzed'])
        self.assertTrue(report['sample_matching_cost_applied'])
        self.assertTrue(report['assistance_applied'])
        self.assertFalse(report['sample_intensity_in_background'])
        self.assertFalse(report['scientifically_validated'])
        self.assertFalse(report['v66_equivalent'])
        pn, _ = generate_pn_arrays(self.e, p, n)
        np.testing.assert_array_equal(arrays['pn_only_background'], pn['background'])
        self.assertTrue(set(pn).issubset(arrays))

    def test_repeated_weak_moving_signal_freezes_pn_only_pure_choices(self):
        p, n = self.texture_sources()
        centers = (1.08, 1.12, 1.16)
        moving = np.array([.003*np.exp(-((self.e-c)/.003)**2) for c in centers])
        samples = np.broadcast_to(n*np.exp(moving), (2, 3, len(self.e))).copy()
        arrays, report = S.generate_sample_assisted_arrays(self.e, p, n, samples, self.gates)
        for c in centers:
            i = np.argmin(abs(self.e-c))
            self.assertTrue(arrays['sample_dynamic_mask'][i])
            self.assertTrue(arrays['sample_protected_mask'][i])
            np.testing.assert_array_equal(arrays['source_weights'][:, i], arrays['pn_only_source_weights'][:, i])
            self.assertFalse(arrays['sample_selection_cost'][:, i].any())
        # Output never contains a fitted/averaged moving Gaussian component.
        for record in report['source_segments']:
            a, b = record['indices']
            source = p if record['source_key'] == 'P' else n
            delta = arrays['background_log'][a:b+1]-np.log(source[a:b+1])
            self.assertLess(np.ptp(delta), 2e-11)

    def test_common_static_peak_remains_unresolved_even_if_samples_match(self):
        raw = self.base*np.exp(.2*np.exp(-((self.e-1.12)/.004)**2))
        samples = np.broadcast_to(raw, (2, 3, len(self.e))).copy()
        arrays, report = S.generate_sample_assisted_arrays(self.e, raw, raw, samples, self.gates)
        i = np.argmin(abs(self.e-1.12))
        self.assertTrue(arrays['source_suspect_mask'][:, i].all())
        self.assertTrue(arrays['uncertain_mask'][i])
        np.testing.assert_allclose(arrays['background'], raw, rtol=3e-14)
        self.assertFalse(arrays['sample_P_witness'].any())
        self.assertFalse(arrays['sample_N_witness'].any())
        self.assertTrue(report['unresolved_regions'])

    def test_conflicting_repeats_abstain_and_single_repeat_is_explicitly_degraded(self):
        p, n = self.texture_sources()
        samples = np.array([np.broadcast_to(p, (3, len(self.e))), np.broadcast_to(n, (3, len(self.e)))])
        arrays, report = S.generate_sample_assisted_arrays(self.e, p, n, samples, self.gates)
        self.assertFalse(arrays['sample_selection_cost'].any())
        self.assertTrue(arrays['sample_repeat_unstable_mask'].any())
        single, single_report = S.generate_sample_assisted_arrays(self.e, p, n, samples[:1, :1], [4.2])
        self.assertFalse(single['sample_selection_cost'].any())
        self.assertFalse(single_report['sample_selection_enabled'])
        self.assertFalse(single_report['sample_protection_applied'])
        self.assertFalse(single_report['assistance_applied'])
        self.assertEqual(len(single_report['degradation_reasons']), 2)
        self.assertTrue(single['uncertain_mask'].all())
        # A single repeat cannot change the output even with a different
        # gate-dependent signal; diagnostics are still allowed to change.
        moved = np.array([n*np.exp(.03*np.exp(-((self.e-c)/.004)**2))
                          for c in (1.06, 1.12, 1.18)])[None]
        other, _ = S.generate_sample_assisted_arrays(self.e, p, n, moved, self.gates)
        np.testing.assert_array_equal(other['background'], single['background'])
        np.testing.assert_array_equal(single['background'], single['pn_safety_only_background'])

    def test_opposing_band_preferences_never_gain_a_compatibility_witness(self):
        fine = .003*np.sin(2*np.pi*(self.e-.9)/.012)
        wide = .006*np.sin(2*np.pi*(self.e-.9)/.09)
        p, n = np.exp(self.logbase+fine+wide), np.exp(self.logbase-fine-wide)
        samples = np.broadcast_to(np.exp(self.logbase+fine-wide), (2, 3, len(self.e))).copy()
        arrays, _ = S.generate_sample_assisted_arrays(self.e, p, n, samples, self.gates)
        margins = arrays['sample_compatibility_margin'][0]
        opposed = ((margins[0] > 1e-5) & (margins[1] < -1e-5)) | ((margins[0] < -1e-5) & (margins[1] > 1e-5))
        opposed &= arrays['sample_supported_mask']
        self.assertGreater(opposed.sum(), 25)
        self.assertFalse(arrays['sample_selection_cost'][:, opposed].any())

    def test_exclusive_cores_are_preserved_and_no_bridge_crosses_their_interiors(self):
        p = self.base*np.exp(-.3*np.exp(-((self.e-1.04)/.003)**2))
        n = self.base*np.exp(.3*np.exp(-((self.e-1.23)/.004)**2))
        samples = np.broadcast_to(self.base, (2, 3, len(self.e))).copy()
        arrays, report = S.generate_sample_assisted_arrays(self.e, p, n, samples, self.gates)
        for source in (0, 1):
            mask = arrays['source_exclusive_core_mask'][source]
            self.assertTrue(mask.any())
            self.assertTrue(np.all(arrays['source_weights'][source, mask] == 1))
        for a, b in arrays['transition_endpoint_indices']:
            self.assertFalse(arrays['source_exclusive_core_mask'][:, a+1:b].any())
        poly = PPoly(arrays['background_log_polynomial_coefficients'], arrays['polynomial_knots'])
        np.testing.assert_allclose(poly(self.e), arrays['background_log'], atol=1e-12, rtol=0)
        self.assertLess(max(report['C2_join_error']), 1e-6)
        self.assertLess(max(r['shape_residual_max'] for r in report['source_segments']), 1e-10)

    def test_impossible_protected_bridge_fails_instead_of_hard_seam(self):
        e = np.linspace(1., 1.1, 9)
        chosen = np.array([0]*4+[1]*5)
        with self.assertRaisesRegex(ValueError, 'Unresolved protected source cores'):
            S._protected_weights(e, chosen, chosen.copy(), .02)

    def test_nonuniform_axis_role_and_repeat_gate_permutations(self):
        e = .8+.45*np.linspace(0, 1, 901)**1.05
        base = np.exp(8+.1*e)
        p, n = base*np.exp(.004*np.sin(e*500)), base*np.exp(.004*np.cos(e*500))
        samples = np.broadcast_to(n, (2, 3, len(e))).copy()
        a, _ = S.generate_sample_assisted_arrays(e, p, n, samples, self.gates)
        b, _ = S.generate_sample_assisted_arrays(e, n, p, samples[::-1, ::-1], self.gates[::-1])
        np.testing.assert_allclose(a['background'], b['background'], rtol=3e-14)
        np.testing.assert_array_equal(a['source_weights'], b['source_weights'][::-1])
        np.testing.assert_array_equal(a['energy'], e)

    def test_sample_gain_and_slow_quadratic_drift_do_not_supply_output_envelope(self):
        p, n = self.texture_sources()
        samples = np.broadcast_to(n, (2, 3, len(self.e))).copy()
        a, report = S.generate_sample_assisted_arrays(self.e, p, n, samples, self.gates)
        gain = np.exp(np.array([1., -2., .3])[None, :, None]+.8*(self.e-1.1)+.9*(self.e-1.1)**2)
        b, _ = S.generate_sample_assisted_arrays(self.e, p, n, samples*gain, self.gates)
        np.testing.assert_allclose(a['background'], b['background'], rtol=3e-14)
        np.testing.assert_array_equal(a['source_weights'], b['source_weights'])
        cell = np.gradient(self.e)
        self.assertLess(abs(cell@(a['background_log']-a['reference_log'])), 1e-12)
        self.assertIn('reference_log_difference_range', report['level_drift'])

    def test_readonly_no_io_and_scoped_cancellation(self):
        raw = self.base.copy()
        sample = np.broadcast_to(raw, (2, 1, len(self.e))).copy()
        items = (self.e, raw, sample, self.gates[:1].copy())
        before = [v.copy() for v in items]
        for v in items:
            v.setflags(write=False)
        with patch.object(builtins, 'open', side_effect=AssertionError('No IO')):
            S.generate_sample_assisted_arrays(items[0], items[1], items[1], items[2], items[3], settings=PnGeneratorSettings())
        for old, new in zip(before, items):
            np.testing.assert_array_equal(old, new)
        cancellation = LoadCancellation()
        cancellation.cancel()
        with self.assertRaises(LoadCancelled), load_scope(cancellation):
            S.generate_sample_assisted_arrays(self.e, raw, raw, sample, items[3])

    def test_bad_shapes_axes_intensities_and_settings_are_rejected(self):
        sample = np.broadcast_to(self.base, (2, 3, len(self.e))).copy()
        cases = [
            (self.e, self.base, self.base, sample[:, :, :-1], self.gates, None),
            (self.e, self.base, self.base, sample[0], self.gates, None),
            (self.e, self.base, self.base, sample, [0, 0, 1], None),
            (self.e, self.base, self.base, sample, [0, np.nan, 1], None),
            (self.e[::-1], self.base, self.base, sample, self.gates, None),
            (self.e, self.base*0, self.base, sample, self.gates, None),
            (self.e, self.base, self.base, sample*0, self.gates, None),
            (self.e, self.base, self.base, sample*np.nan, self.gates, None),
            (self.e, self.base, self.base, sample, self.gates, {'transition_halfwidth_ev': .5}),
        ]
        for e, p, n, s, g, settings in cases:
            with self.subTest(settings=settings, shape=s.shape), self.assertRaises(ValueError):
                S.generate_sample_assisted_arrays(e, p, n, s, g, settings=settings)
        # Minimum supported array length is allowed, with honest unsupported evidence.
        e = np.linspace(.5, .8, 7)
        arrays, report = S.generate_sample_assisted_arrays(e, np.ones(7), np.ones(7), np.ones((1, 1, 7)), [0])
        np.testing.assert_allclose(arrays['background'], 1., atol=1e-14)
        self.assertFalse(arrays['sample_eligible_mask'].any())
        self.assertTrue(report['degradation_reasons'])


if __name__ == '__main__':
    unittest.main()
