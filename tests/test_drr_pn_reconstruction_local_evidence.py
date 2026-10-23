"""Local evidence repair does not propagate source levels into remote tails."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.interpolate import PPoly

from core.drr_pn_reconstruction.render_local_evidence import (
    plan_local_repairs, render_local_evidence,
)


def fixture():
    e=np.linspace(1.3,2.3,801)
    c=np.array([10+.13*(e-1.7)+.001*np.sin(e*130),
                10.002+.12*(e-1.7)+.0012*np.cos(e*110)])
    q=.002+.001*(e-1.7)
    a=c+np.array([-.5,.5])[:,None]*q
    w=np.zeros(len(e));w[125:226]=1;w[375:451]=1;w[625:]=1
    for l,r,start in ((100,125,0),(225,250,1),(350,375,0),(450,475,1),(600,625,0)):
        t=np.linspace(0,1,r-l+1)
        w[l:r+1]=start+(1-2*start)*(t**3*(10-15*t+6*t*t))
        w[l],w[r]=start,1-start
    weights=np.array([w,1-w])
    state=np.full(len(e),3,np.int8);state[160:190]=1;state[520:550]=2
    b=np.exp(np.sum(weights*a,axis=0))
    return e,dict(energy=e,original_corrected_source_log=c,original_aligned_source_log=a,
        alignment_q_log=q,state=state,source_weights=weights,
        source_usable_masks=np.array([(state==1)|(state==3),(state==2)|(state==3)]),
        reference_source_weights=weights.copy(),baseline_reference_background=b.copy(),
        background=b,raw_geometric_mean=np.exp(c.mean(0)))


def swap(b):
    x={k:v.copy() for k,v in b.items()}
    for k in ('original_corrected_source_log','original_aligned_source_log','source_weights',
              'reference_source_weights','source_usable_masks'):
        x[k]=x[k][::-1].copy()
    x['alignment_q_log']=-x['alignment_q_log']
    s=x['state'];x['state']=np.where(s==1,2,np.where(s==2,1,s)).astype(s.dtype)
    return x


class LocalEvidenceTests(unittest.TestCase):
    def test_multiple_sources_compact_closure_and_readonly_no_io(self):
        e,b=fixture();before={k:v.tobytes() for k,v in b.items()}
        for v in b.values():v.setflags(write=False)
        with patch('builtins.open',side_effect=AssertionError('unexpected IO')), \
             patch('io.open',side_effect=AssertionError('unexpected IO')):
            r=render_local_evidence(e,baseline=b)
        a=r['arrays'];p=r['stages']['local_parent']['arrays']
        self.assertEqual(r['diagnostics']['applied_count'],2)
        self.assertEqual([x['source'] for x in r['diagnostics']['applied_repairs']],['P','N'])
        self.assertFalse(r['diagnostics']['scientifically_validated'])
        np.testing.assert_array_equal(a['state'],b['state'])
        np.testing.assert_array_equal(a['background'][~a['local_evidence_support_mask']],p['background'][~a['local_evidence_support_mask']])
        np.testing.assert_array_equal(a['background_log'][~a['local_evidence_support_mask']],p['background_log'][~a['local_evidence_support_mask']])
        active=np.zeros(len(e)-1,bool)
        for row in r['diagnostics']['applied_repairs']:
            lo,hi=row['support_indices'];left,right=row['core_indices'];active[lo:hi]=True
            s=0 if row['source']=='P' else 1
            self.assertLess(np.ptp(a['background_log'][left:right+1]-b['original_aligned_source_log'][s,left:right+1]),3e-12)
            self.assertTrue(row['exterior_background_bitwise'])
            self.assertTrue(row['exterior_coefficients_bitwise'])
            self.assertLess(max(row['C0_C1_C2_join_max']),1e-6)
        np.testing.assert_array_equal(a['integrated_log_polynomial_coefficients'][:,~active],p['integrated_log_polynomial_coefficients'][:,~active])
        np.testing.assert_allclose(PPoly(a['integrated_log_polynomial_coefficients'],e)(e),a['background_log'],atol=1e-11,rtol=0)
        self.assertEqual(before,{k:v.tobytes() for k,v in b.items()})

    def test_source_relabeling_is_symmetric(self):
        e,b=fixture();r=render_local_evidence(e,baseline=b);s=render_local_evidence(e,baseline=swap(b))
        np.testing.assert_allclose(r['arrays']['background_log'],s['arrays']['background_log'],atol=3e-12,rtol=0)
        np.testing.assert_array_equal(r['arrays']['local_evidence_support_mask'],s['arrays']['local_evidence_support_mask'])
        self.assertEqual([x['source'] for x in s['diagnostics']['applied_repairs']],['N','P'])

    def test_long_pure_tail_only_repairs_evidence_neighborhood(self):
        e,b=fixture();b['state'][:]=3;b['state'][680:720]=1;b['state'][780:783]=2
        r=render_local_evidence(e,baseline=b);a=r['arrays'];parent=r['stages']['local_parent']['arrays']
        self.assertEqual(r['diagnostics']['applied_count'],1)
        self.assertEqual(r['diagnostics']['repair_plan'][0]['core_indices'],[680,719])
        self.assertLess(r['diagnostics']['repair_plan'][0]['support_indices'][1],780)
        np.testing.assert_array_equal(a['background'][780:],parent['background'][780:])
        np.testing.assert_array_equal(a['background_log'][:630],parent['background_log'][:630])
        np.testing.assert_array_equal(a['state'],b['state'])

    def test_opposite_short_evidence_defers_instead_of_crossing(self):
        e,b=fixture();b['state'][140:142]=2
        r=render_local_evidence(e,baseline=b)
        self.assertEqual(r['diagnostics']['applied_count'],1)
        self.assertIn('fixed_collar_crosses_opposite_exclusive_evidence',r['diagnostics']['repair_plan'][0]['reasons'])
        self.assertTrue(r['diagnostics']['partial'])
        np.testing.assert_array_equal(r['arrays']['background'][:300],r['stages']['local_parent']['arrays']['background'][:300])

    def test_boundary_collar_is_not_silently_shrunk(self):
        e,b=fixture();b['state'][:]=3;b['state'][10:40]=2
        r=render_local_evidence(e,baseline=b)
        self.assertEqual(r['diagnostics']['applied_count'],0)
        self.assertIn('fixed_collar_exceeds_native_domain',r['diagnostics']['repair_plan'][0]['reasons'])
        np.testing.assert_array_equal(r['arrays']['background'],r['stages']['local_parent']['arrays']['background'])

    def test_overlapping_supports_defer_without_source_order_preference(self):
        e,b=fixture();s=np.full(len(e),3,np.int8);s[160:185]=1;s[205:230]=1
        w=np.array([np.ones(len(e)),np.zeros(len(e))])
        plan=plan_local_repairs(e,w,s)
        self.assertEqual(len(plan),2)
        self.assertTrue(all(not x['apply'] for x in plan))
        self.assertTrue(all('overlapping_requested_supports' in x['reasons'] for x in plan))

    def test_no_evidence_invalid_state_and_cancellation(self):
        e,b=fixture();b['state'][:]=3
        r=render_local_evidence(e,baseline=b)
        self.assertEqual(r['diagnostics']['repair_plan'],[])
        self.assertTrue(r['diagnostics']['no_supported_repairs'])
        with self.assertRaises(InterruptedError):render_local_evidence(e,baseline=b,cancelled=lambda:True)
        b['state'][0]=4
        with self.assertRaises(ValueError):plan_local_repairs(e,b['source_weights'],b['state'])

    def test_diagnostic_aliases_describe_current_not_parent_curve(self):
        e,b=fixture();r=render_local_evidence(e,baseline=b);a=r['arrays']
        p=PPoly(a['integrated_log_polynomial_coefficients'],e)
        np.testing.assert_array_equal(a['final_curvature_polynomial_coefficients'],p.derivative(2).c)
        np.testing.assert_array_equal(a['repair_log_delta'],a['background_log']-np.log(a['raw_geometric_mean']))
        np.testing.assert_array_equal(a['delta_background_log'],a['background_log']-a['parent_background_log'])
        np.testing.assert_array_equal(a['local_support_mask'],a['local_evidence_support_mask'])
        self.assertNotIn('compensation_curvature_polynomial_coefficients',a)
        self.assertIn('local_evidence_parent_compensation_curvature_polynomial_coefficients',a)
        self.assertEqual(r['diagnostics']['metadata_revision'],3)
        text=str(r['diagnostics'])
        self.assertNotIn('1.84-eV',text);self.assertNotIn('third P',text)

    def test_unmet_short_exclusive_evidence_is_partial_and_uncertain(self):
        e,b=fixture();b['state'][:]=3
        # Both runs are too short for a requested shape repair. One requests
        # the opposite pure source, the other is only partially selected.
        b['state'][650:652]=2;b['state'][110:112]=1
        r=render_local_evidence(e,baseline=b);a=r['arrays'];d=r['diagnostics']
        self.assertEqual(d['repair_plan'],[])
        self.assertFalse(d['requested_repairs_deferred'])
        self.assertTrue(d['partial'])
        self.assertEqual(d['unmet_exclusive_count'],4)
        self.assertEqual(d['mixed_exclusive_count'],2)
        self.assertEqual(d['opposite_exclusive_count'],2)
        self.assertTrue(a['uncertain_mask'][110:112].all())
        self.assertTrue(a['uncertain_mask'][650:652].all())
        self.assertTrue(a['local_evidence_mixed_exclusive_mask'][110:112].all())
        self.assertTrue(a['local_evidence_opposite_exclusive_mask'][650:652].all())
        self.assertEqual([x['inherited_selection'] for x in d['unmet_exclusive_intervals']],['mixed','opposite'])
        np.testing.assert_array_equal(a['state'],b['state'])
        np.testing.assert_array_equal(a['background'],r['stages']['local_parent']['arrays']['background'])


if __name__=='__main__':unittest.main()
