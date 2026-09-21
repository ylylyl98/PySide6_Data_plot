import unittest
import numpy as np
from core.loader import DataCube
from core.drr_peak_analysis import PeakAnalysisSettings,analyze_drr_peaks


class AutoCandidateTests(unittest.TestCase):
    def test_noise_reduction_preserves_broad_peak_and_inputs(self):
        x=np.linspace(1,1.1,501);rng=np.random.default_rng(71)
        z=np.array([np.exp(-((x-(1.05+i*.0001))/.004)**2)+rng.normal(0,.025,len(x)) for i in range(5)])
        cube=DataCube(x,np.arange(5.),z.copy(),'Y','test','DRR')
        s=PeakAnalysisSettings(1,1.1,0,4,source='raw')
        r=analyze_drr_peaks(cube,s,retain_candidates=True,auto_candidates=True)
        points=r['products']['raw']['points'];good=[p for p in points if p['confidence']=='supported']
        self.assertLess(len(good),len(points)//5)
        self.assertTrue(all(any(p['row_index']==row and p['polarity']=='peak' and abs(p['energy']-(1.05+row*.0001))<.0015 for p in good) for row in range(5)))
        self.assertTrue(any(p['confidence']=='low' for p in points))
        np.testing.assert_array_equal(cube.Z,z)

    def test_constant_and_missing_segments_are_not_promoted(self):
        x=np.linspace(1,1.1,101);z=np.ones((2,101));z[:,40:60]=np.nan
        r=analyze_drr_peaks(DataCube(x,np.arange(2.),z,'Y','test','DRR'),PeakAnalysisSettings(1,1.1,0,1),retain_candidates=True,auto_candidates=True)
        self.assertFalse(any(p['confidence']=='supported' for product in r['products'].values() for p in product['points']))

    def test_low_confidence_filter_is_reversible(self):
        from core.drr_result_filter import filtered_result
        point={'energy':1.05,'y':0.,'row_index':0,'polarity':'peak','prominence':1.,'track_id':1,'status':'accepted','confidence':'low'}
        result={'candidate_pool':True,'products':{'raw':{'points':[point]}}}
        bounds=dict(x_min=1,x_max=1.1,y_min=0,y_max=1,show_low_confidence=False,link_tracks=False)
        self.assertEqual(filtered_result(result,bounds)['products']['raw']['points'],[])
        bounds['show_low_confidence']=True
        self.assertEqual(len(filtered_result(result,bounds)['products']['raw']['points']),1)
        self.assertEqual(len(result['products']['raw']['points']),1)

    def test_pure_noise_is_mostly_low_confidence(self):
        x=np.linspace(1,1.1,501);rng=np.random.default_rng(8)
        z=rng.normal(0,.02,(5,len(x)))
        r=analyze_drr_peaks(DataCube(x,np.arange(5.),z,'Y','noise','DRR'),PeakAnalysisSettings(1,1.1,0,4),auto_candidates=True)
        for product in r['products'].values():
            points=product['points']
            self.assertLess(sum(p['confidence']=='supported' for p in points),len(points)*.02)

    def test_derivative_noise_floor_is_local_to_finite_segment(self):
        x=np.linspace(1,1.1,1001);rng=np.random.default_rng(8)
        z=rng.normal(0,.0001,(5,len(x)))
        z[:,750:770]=np.nan
        z[:,770:]=rng.normal(0,.1,(5,231))
        r=analyze_drr_peaks(DataCube(x,np.arange(5.),z,'Y','noise','DRR'),PeakAnalysisSettings(1,1.1,0,4),auto_candidates=True)
        points=[p for p in r['products']['second']['points'] if p['energy']>=x[770]]
        self.assertGreater(len(points),100)
        self.assertLess(sum(p['confidence']=='supported' for p in points),len(points)*.03)
