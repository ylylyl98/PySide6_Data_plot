"""Source symmetry, multiplicity and honest limits of generalized rendering."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.interpolate import PPoly

from core.drr_pn_reconstruction.render_generalized import render_generalized, select_repair_regions


def fixture():
    e=np.linspace(1.3,2.3,801)
    c=np.array([10+.13*(e-1.7)+.001*np.sin(e*130),
                10.002+.12*(e-1.7)+.0012*np.cos(e*110)])
    q=.002+.001*(e-1.7)
    a=c+np.array([-.5,.5])[:,None]*q
    w=np.zeros(len(e))
    w[125:226]=1;w[375:451]=1;w[625:]=1
    for l,r,start in ((100,125,0),(225,250,1),(350,375,0),(450,475,1),(600,625,0)):
        t=np.linspace(0,1,r-l+1)
        w[l:r+1]=start+(1-2*start)*(t**3*(10-15*t+6*t*t))
        w[l],w[r]=start,1-start
    weights=np.array([w,1-w])
    state=np.full(len(e),3,np.int8)
    state[160:190]=1;state[520:550]=2
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


class GeneralizedRegionTests(unittest.TestCase):
    def test_multiple_P_N_repairs_are_source_symmetric_and_readonly(self):
        e,b=fixture()
        before={k:v.tobytes() for k,v in b.items()}
        for v in b.values():v.setflags(write=False)
        with patch('builtins.open',side_effect=AssertionError('unexpected IO')), \
             patch('io.open',side_effect=AssertionError('unexpected IO')):
            result=render_generalized(e,baseline=b)
        other=render_generalized(e,baseline=swap(b))
        self.assertEqual(result['diagnostics']['applied_count'],2)
        self.assertEqual([r['source'] for r in result['diagnostics']['applied_repairs']],['P','N'])
        self.assertEqual([r['source'] for r in other['diagnostics']['applied_repairs']],['N','P'])
        np.testing.assert_allclose(result['arrays']['background_log'],other['arrays']['background_log'],rtol=0,atol=3e-12)
        np.testing.assert_allclose(result['arrays']['source_weights'],other['arrays']['source_weights'][::-1],rtol=0,atol=3e-15)
        np.testing.assert_array_equal(result['arrays']['original_aligned_source_log'],b['original_aligned_source_log'])
        self.assertEqual(before,{k:v.tobytes() for k,v in b.items()})
        for row in result['diagnostics']['applied_repairs']:
            l,r=row['core_indices'];s=0 if row['source']=='P' else 1
            delta=result['arrays']['background_log'][l:r+1]-b['original_aligned_source_log'][s,l:r+1]
            self.assertLess(np.ptp(delta),3e-12)

    def test_one_sided_tail_repair_and_swapped_source(self):
        for first,last,state,core,missing in ((680,720,1,[625,800],'right'),(40,75,2,[0,100],'left')):
            with self.subTest(boundary=missing):
                e,b=fixture()
                b['state'][:]=3;b['state'][first:last]=state
                result=render_generalized(e,baseline=b)
                other=render_generalized(e,baseline=swap(b))
                self.assertEqual(result['diagnostics']['applied_count'],1)
                row=result['diagnostics']['applied_repairs'][0]
                self.assertTrue(row['boundary_core'])
                self.assertIsNone(row[missing+'_bridge_indices'])
                self.assertEqual(row['core_indices'],core)
                np.testing.assert_allclose(result['arrays']['background_log'],other['arrays']['background_log'],rtol=0,atol=3e-12)
                l,r=core
                delta=result['arrays']['background_log'][l:r+1]-b['original_aligned_source_log'][state-1,l:r+1]
                self.assertLess(np.ptp(delta),3e-12)

    def test_two_symmetric_unknown_excursions_close_independently(self):
        e,b=fixture()
        b['state'][:]=3
        for l,r,base in ((155,175,1.),(290,310,0.)):
            t=np.linspace(0,np.pi,r-l+1)
            selected=base+(1-2*base)*.8*np.sin(t)**2
            selected[0]=selected[-1]=base
            b['source_weights'][0,l:r+1]=selected
            b['source_weights'][1,l:r+1]=1-selected
            b['state'][l+1:r]=0
        b['reference_source_weights']=b['source_weights'].copy()
        b['background']=np.exp(np.sum(b['source_weights']*b['original_aligned_source_log'],axis=0))
        b['baseline_reference_background']=b['background'].copy()
        result=render_generalized(e,baseline=b)
        other=render_generalized(e,baseline=swap(b))
        self.assertEqual(len(result['stages']['excursion_closure']['diagnostics']['support_records']),2)
        self.assertEqual(result['diagnostics']['applied_count'],0)
        np.testing.assert_allclose(result['arrays']['background_log'],other['arrays']['background_log'],rtol=0,atol=3e-12)

    def test_shared_bridge_repairs_defer_both_without_preference(self):
        e,b=fixture()
        b['state'][:]=3;b['state'][160:190]=1;b['state'][285:320]=2
        result=render_generalized(e,baseline=b)
        self.assertTrue(result['diagnostics']['partial'])
        self.assertEqual(result['diagnostics']['applied_count'],0)
        self.assertEqual(len(result['diagnostics']['repair_plan']),2)
        self.assertTrue(all('shared_or_overlapping_repair_bridge' in r['reasons'] for r in result['diagnostics']['repair_plan']))
        np.testing.assert_array_equal(result['arrays']['background'],result['stages']['excursion_closure']['arrays']['background'])
        self.assertFalse(result['diagnostics']['scientifically_validated'])

    def test_no_exclusive_evidence_does_not_request_shape_repair(self):
        e,b=fixture();b['state'][:]=3
        result=render_generalized(e,baseline=b)
        self.assertEqual(result['diagnostics']['repair_plan'],[])
        self.assertTrue(result['diagnostics']['no_supported_repairs'])

    def test_opposite_short_evidence_remains_explicit(self):
        e,b=fixture();b['state'][:]=3;b['state'][680:720]=1;b['state'][760:763]=2
        result=render_generalized(e,baseline=b)
        self.assertEqual(result['diagnostics']['repair_plan'][0]['opposite_exclusive_indices'],[760,761,762])
        np.testing.assert_array_equal(result['arrays']['state'],b['state'])

    def test_cancellation_and_invalid_state_do_not_fallback(self):
        e,b=fixture()
        with self.assertRaises(InterruptedError):
            render_generalized(e,baseline=b,cancelled=lambda:True)
        b['state'][100]=4
        with self.assertRaises(ValueError):
            select_repair_regions(e,b['source_weights'],b['state'])


if __name__=='__main__':unittest.main()
