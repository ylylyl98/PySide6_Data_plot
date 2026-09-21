import unittest
import numpy as np
from core.cw_method_comparison import compare_cw_methods


class MethodComparisonTests(unittest.TestCase):
    def test_exact_cw_and_equal_errors(self):
        t=np.array([2.,3.,4.,6.,9.,15.])
        rows=compare_cw_methods(t,2/(t+4),np.full(6,.01))
        self.assertEqual(len(rows),3)
        for row in rows:
            self.assertAlmostEqual(row['theta_k'],-4,places=3)
        self.assertEqual(rows[2]['weights'],'1/slope_se^2')

    def test_precise_data_has_finite_profile_interval(self):
        t=np.array([2.,3.,4.,6.,9.,15.])
        row=compare_cw_methods(t,2/(t+4),np.full(6,1e-5))[2]
        self.assertIsNotNone(row['ci95_low_k'])
        self.assertLess(row['ci95_low_k'],-4)
        self.assertGreater(row['ci95_high_k'],-4)

    def test_missing_errors_does_not_drop_points_or_fallback(self):
        rows=compare_cw_methods([2,3,4,5],[.3,.25,.2,.18],[.01,None,.01,.01])
        self.assertEqual(rows[2]['status'],'unavailable')
        self.assertIsNone(rows[2]['theta_k'])
        self.assertEqual(rows[2]['n'],4)
        self.assertIsNotNone(rows[0]['theta_k'])

    def test_real_group2_separates_weighting_and_space(self):
        t=[1.67,2.2,2.7,3.2,4,5]
        s=[.011520155841,.004762562328,.02192692756,.01125377372,.008597500294,.001082407126]
        e=[.004015908781,.003156751252,.004406672627,.005383290519,.003014737869,.007253377878]
        rows=compare_cw_methods(t,s,e)
        self.assertAlmostEqual(rows[0]['theta_k'],1.90859,places=4)
        self.assertEqual(rows[0]['status'],'diagnostic only')
        self.assertAlmostEqual(rows[2]['theta_k'],-9.9573,places=2)
        self.assertIsNone(rows[2]['ci95_low_k'])
        self.assertIn('unbounded',rows[2]['diagnostics'])

    def test_free_background_not_silently_compared(self):
        rows=compare_cw_methods([2,3,4],[.3,.2,.1],[.01]*3,background_mode='fit')
        self.assertTrue(all(r['status']=='unavailable' for r in rows))

    def test_constant_response_not_reported_as_huge_theta(self):
        rows=compare_cw_methods([2,3,4,5],[.2]*4,[.01]*4)
        self.assertIsNone(rows[2]['theta_k'])
