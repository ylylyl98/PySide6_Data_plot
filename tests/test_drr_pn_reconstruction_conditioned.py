"""Fixed conditional routing preserves reliable v1 and bounds local changes."""
import inspect
import unittest
from unittest.mock import patch

import numpy as np
from scipy.interpolate import PPoly

from core.drr_pn_reconstruction.render_conditioned import (
    plan_conditioned_repairs, render_conditioned,
)
from core.drr_pn_reconstruction.render_generalized import render_generalized
from tests.test_drr_pn_reconstruction_generalized import fixture, swap


class ConditionedRepairTests(unittest.TestCase):
    def test_safe_multiple_sources_are_exact_unchanged_v1(self):
        e, b = fixture()
        old = render_generalized(e, baseline=b)
        new = render_conditioned(e, baseline=b)
        for key in ('background', 'background_log', 'integrated_log_polynomial_coefficients',
                    'source_weights', 'original_aligned_source_log', 'alignment_q_log'):
            np.testing.assert_array_equal(new['arrays'][key], old['arrays'][key])
        self.assertEqual(new['diagnostics']['regional_applied_count'], 2)
        self.assertEqual(new['diagnostics']['local_applied_count'], 0)
        self.assertEqual(str(inspect.signature(render_generalized)),
                         '(energy, *, baseline, cancelled=None, progress=None)')
        self.assertFalse(new['diagnostics']['scientifically_validated'])

    def test_boundary_with_short_opposite_evidence_uses_only_compact_repair(self):
        e, b = fixture()
        b['state'][:] = 3
        b['state'][680:720] = 1
        b['state'][780:783] = 2
        r = render_conditioned(e, baseline=b)
        a, d = r['arrays'], r['diagnostics']
        parent = r['stages']['conditioned_reliable_parent']['arrays']
        self.assertEqual(d['regional_applied_count'], 0)
        self.assertEqual(d['local_applied_count'], 1)
        self.assertEqual(d['regional_plan'][0]['route'], 'compact_only')
        self.assertEqual(d['local_plan'][0]['core_indices'], [680, 719])
        self.assertLess(d['local_plan'][0]['support_indices'][1], 780)
        np.testing.assert_array_equal(a['background'][780:], parent['background'][780:])
        np.testing.assert_array_equal(a['background'][~a['conditioned_local_interior_mask']],
                                      parent['background'][~a['conditioned_local_interior_mask']])
        self.assertTrue(d['partial'])
        self.assertEqual(d['opposite_exclusive_count'], 3)
        self.assertTrue(a['conditioned_opposite_exclusive_mask'][780:783].all())

    def test_bounded_conflict_redirects_and_does_not_shrink_collar(self):
        e, b = fixture()
        b['state'][:] = 3
        b['state'][160:190] = 1
        b['state'][140:142] = 2
        r = render_conditioned(e, baseline=b)
        self.assertEqual(r['diagnostics']['regional_applied_count'], 0)
        self.assertEqual(r['diagnostics']['local_applied_count'], 0)
        self.assertIn('opposite_exclusive_inside_pure_region',
                      r['diagnostics']['regional_plan'][0]['route_conditions'])
        self.assertIn('fixed_collar_crosses_opposite_exclusive_evidence',
                      r['diagnostics']['local_plan'][0]['reasons'])
        np.testing.assert_array_equal(r['arrays']['background'],
                                      r['stages']['conditioned_reliable_parent']['arrays']['background'])

    def test_reliable_support_is_protected_in_combined_plan(self):
        e, b = fixture()
        b['state'][:] = 3
        b['state'][160:190] = 1
        b['state'][680:720] = 1
        # Exercise cross-family priority independently of whether the default
        # geometry happened to reject a requested regional bridge earlier.
        plan = plan_conditioned_repairs(e, b['source_weights'], b['state'])
        supplied = plan['regional_plan']
        supplied[0]['affected_bridge_support'] = [90, 650]
        plan = plan_conditioned_repairs(e, b['source_weights'], b['state'], regional_plan=supplied)
        self.assertTrue(plan['regional_plan'][0]['apply'])
        self.assertFalse(plan['local_plan'][0]['apply'])
        self.assertIn('local_support_contacts_protected_regional_support', plan['local_plan'][0]['reasons'])

    def test_reliable_then_local_exterior_means_reliable_parent(self):
        e, b = fixture()
        b['state'][:] = 3
        b['state'][160:190] = 1
        b['state'][680:720] = 1
        r = render_conditioned(e, baseline=b)
        a, d = r['arrays'], r['diagnostics']
        self.assertEqual((d['regional_applied_count'], d['local_applied_count']), (1, 1))
        p = r['stages']['conditioned_reliable_parent']['arrays']
        m = a['conditioned_protected_support_mask']
        np.testing.assert_array_equal(a['background'][m], p['background'][m])
        np.testing.assert_array_equal(a['integrated_log_polynomial_coefficients'][:, a['conditioned_protected_cell_mask']],
                                      p['integrated_log_polynomial_coefficients'][:, a['conditioned_protected_cell_mask']])
        self.assertTrue(np.any(p['background'] != r['stages']['excursion_closure']['arrays']['background']))
        np.testing.assert_array_equal(a['parent_background'], p['background'])

    def test_source_symmetry_and_half_plan_independence(self):
        e, b = fixture()
        b['state'][:] = 3
        b['state'][160:190] = 1
        b['state'][680:720] = 1
        b['state'][780:783] = 2
        a = render_conditioned(e, baseline=b)
        s = render_conditioned(e, baseline=swap(b))
        np.testing.assert_allclose(a['arrays']['background_log'], s['arrays']['background_log'], rtol=0, atol=3e-12)
        np.testing.assert_array_equal(a['arrays']['conditioned_regional_route'], s['arrays']['conditioned_regional_route'])
        np.testing.assert_array_equal(a['arrays']['conditioned_local_route'], s['arrays']['conditioned_local_route'])
        half = {k: v.copy() for k, v in b.items()}
        half['original_corrected_source_log'] += .0002 * np.sin(e * 55)
        half['original_aligned_source_log'] = half['original_corrected_source_log'] + np.array([-.5, .5])[:, None] * half['alignment_q_log']
        half['background'] = np.exp(np.sum(half['source_weights'] * half['original_aligned_source_log'], axis=0))
        half['baseline_reference_background'] = half['background'].copy()
        half['raw_geometric_mean'] = np.exp(half['original_corrected_source_log'].mean(0))
        h = render_conditioned(e, baseline=half)
        self.assertEqual(a['diagnostics']['regional_plan'], h['diagnostics']['regional_plan'])
        self.assertEqual(a['diagnostics']['local_plan'], h['diagnostics']['local_plan'])
        self.assertNotEqual(a['diagnostics']['local_applied_repairs'][0]['local_log_constant'],
                            h['diagnostics']['local_applied_repairs'][0]['local_log_constant'])

    def test_edge_and_local_overlap_defer_without_shortened_bridge(self):
        e, b = fixture()
        b['state'][:] = 3
        b['state'][10:40] = 2
        r = render_conditioned(e, baseline=b)
        self.assertIn('fixed_collar_exceeds_native_domain', r['diagnostics']['local_plan'][0]['reasons'])
        b['state'][:] = 3
        b['state'][650:680] = 1
        b['state'][700:730] = 1
        r = render_conditioned(e, baseline=b)
        self.assertEqual(r['diagnostics']['local_applied_count'], 0)
        self.assertTrue(all('overlapping_requested_supports' in row['reasons'] for row in r['diagnostics']['local_plan']))

    def test_metadata_uses_final_polynomial_and_reports_short_unmet(self):
        e, b = fixture()
        b['state'][:] = 3
        b['state'][680:720] = 1
        b['state'][780:783] = 2
        b['state'][110:112] = 1
        r = render_conditioned(e, baseline=b)
        a, d = r['arrays'], r['diagnostics']
        p = PPoly(a['integrated_log_polynomial_coefficients'], e)
        np.testing.assert_array_equal(a['final_curvature_polynomial_coefficients'], p.derivative(2).c)
        np.testing.assert_array_equal(a['repair_log_delta'], a['background_log'] - np.log(a['raw_geometric_mean']))
        np.testing.assert_array_equal(a['delta_background_log'], a['background_log'] - a['parent_background_log'])
        np.testing.assert_array_equal(a['local_support_mask'], a['conditioned_local_support_mask'])
        self.assertEqual(d['mixed_exclusive_count'], 2)
        self.assertTrue(a['uncertain_mask'][110:112].all())
        self.assertTrue(a['uncertain_mask'][780:783].all())

    def test_readonly_no_io_and_cancellation(self):
        e, b = fixture()
        before = {k: v.tobytes() for k, v in b.items()}
        for v in b.values():
            v.setflags(write=False)
        with patch('builtins.open', side_effect=AssertionError('unexpected IO')), \
             patch('io.open', side_effect=AssertionError('unexpected IO')):
            render_conditioned(e, baseline=b)
        self.assertEqual(before, {k: v.tobytes() for k, v in b.items()})
        with self.assertRaises(InterruptedError):
            render_conditioned(e, baseline=b, cancelled=lambda: True)


if __name__ == '__main__':
    unittest.main()
