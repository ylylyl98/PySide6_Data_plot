"""Final diagnostic aliases follow the final curve without changing its math."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.interpolate import PPoly,PchipInterpolator

from core.drr_pn_reconstruction.render_generalized import render_generalized
from tests.test_drr_pn_reconstruction_generalized import fixture,swap


class GeneralizedMetadataTests(unittest.TestCase):
    def test_final_aliases_and_constructed_values_are_separate(self):
        e,b=fixture()
        with patch('core.drr_pn_reconstruction.render_generalized._refresh_final_metadata',side_effect=lambda e,a:dict(a)):
            previous=render_generalized(e,baseline=b)
        current=render_generalized(e,baseline=b);a=current['arrays']
        for key in ('background','background_log','integrated_log_polynomial_coefficients',
                    'state','source_weights','original_corrected_source_log','original_aligned_source_log','alignment_q_log'):
            np.testing.assert_array_equal(a[key],previous['arrays'][key])
        p=PPoly(a['integrated_log_polynomial_coefficients'],e)
        pp=PPoly(a['parent_integrated_log_polynomial_coefficients'],e)
        d=PPoly(p.c-pp.c,e)
        np.testing.assert_array_equal(a['final_curvature_polynomial_coefficients'],p.derivative(2).c)
        np.testing.assert_array_equal(a['parent_final_curvature_polynomial_coefficients'],pp.derivative(2).c)
        np.testing.assert_array_equal(a['delta_closed_curvature_polynomial_coefficients'],d.derivative(2).c)
        np.testing.assert_array_equal(a['delta_log_second_derivative'],d(e,2))
        np.testing.assert_array_equal(a['repair_log_delta'],a['background_log']-np.log(a['raw_geometric_mean']))
        np.testing.assert_array_equal(a['background_delta_log'],a['background_log']-np.log(a['baseline_reference_background']))
        np.testing.assert_array_equal(a['delta_background_log'],a['background_log']-a['parent_background_log'])
        np.testing.assert_array_equal(a['weight_pchip_coefficients'],PchipInterpolator(e,a['source_weights'][0]).c)
        self.assertNotIn('curvature_polynomial_coefficients',a)
        self.assertIn('generalized_inherited_curvature_polynomial_coefficients',a)
        self.assertNotIn('1.84-eV',str(current['diagnostics']))
        self.assertNotIn('third P',str(current['diagnostics']))

    def test_no_repair_and_source_relabelled_metadata_contract(self):
        for mode in ('no_repair','swapped'):
            with self.subTest(mode=mode):
                e,b=fixture()
                if mode=='no_repair':b['state'][:]=3
                else:b=swap(b)
                result=render_generalized(e,baseline=b);a=result['arrays']
                p=PPoly(a['integrated_log_polynomial_coefficients'],e)
                np.testing.assert_array_equal(a['final_curvature_polynomial_coefficients'],p.derivative(2).c)
                np.testing.assert_array_equal(a['source_curvature_weights'],a['source_weights'])
                np.testing.assert_array_equal(a['actual_change_mask'],a['background']!=a['parent_background'])
                self.assertEqual(result['diagnostics']['metadata_revision'],2)


if __name__=='__main__':unittest.main()
