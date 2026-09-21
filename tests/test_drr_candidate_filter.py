import unittest
from copy import deepcopy
import numpy as np
from core.loader import DataCube
from core.drr_peak_analysis import PeakAnalysisSettings,analyze_drr_peaks
from core.drr_result_filter import filtered_result


class CandidateFilterTests(unittest.TestCase):
    def test_display_filter_avoids_copying_discarded_pool_and_matches_export(self):
        from unittest.mock import patch
        point=dict(energy=1.05,y=0,row_index=0,polarity='peak',prominence=1.,track_id=1,status='accepted')
        raw=dict(candidate_pool=True,products={'raw':{'points':[point,dict(point,energy=1.09)]}})
        bounds=dict(limit_x=True,x_min=1.04,x_max=1.06,link_tracks=False)
        exported=filtered_result(raw,bounds)
        with patch('core.drr_result_filter.deepcopy',side_effect=AssertionError('full pool copied')):
            view=filtered_result(raw,bounds,for_display=True)
        self.assertEqual(view['products'],exported['products'])
        self.assertNotIn('filtered_out_points',view)
        view['products']['raw']['points'][0]['track_id']=9
        self.assertEqual(point['track_id'],1)
        self.assertEqual(len(raw['products']['raw']['points']),2)

    def test_display_branch_linking_matches_export_without_mutating_candidates(self):
        point=dict(energy=1.05,y=0,row_index=0,polarity='peak',prominence=1.,track_id=7,status='accepted')
        raw=dict(candidate_pool=True,products={'raw':{'points':[point,dict(point,row_index=1,y=1,track_id=8)]}})
        before=deepcopy(raw)
        bounds=dict(limit_x=False,link_tracks=True,max_shift_mev=3)
        view=filtered_result(raw,bounds,for_display=True)
        self.assertEqual(view['products'],filtered_result(raw,bounds)['products'])
        self.assertEqual(raw,before)
        self.assertEqual(len({p['track_id'] for p in view['products']['raw']['points']}),1)

    def test_thresholds_are_reversible_without_redetection(self):
        x=np.linspace(1,1.1,501)
        z=np.exp(-((x-1.03)/.004)**2)+.1*np.exp(-((x-1.07)/.002)**2)
        cube=DataCube(x,np.arange(3.),np.array([z]*3),'Y','Test','DRR')
        s=PeakAnalysisSettings(1,1.1,0,2,source='raw',polarity='peaks',prominence=.5,max_peaks=1)
        raw=analyze_drr_peaks(cube,s,retain_candidates=True);before=deepcopy(raw)
        points=raw['products']['raw']['points']
        self.assertEqual(len(points),6)
        self.assertTrue(all(p['width_mev']>0 for p in points))
        bounds=dict(x_min=1,x_max=1.1,y_min=0,y_max=2,min_prominence=.2,min_width_mev=0,max_width_mev=0,min_distance_mev=0,max_peaks=0,max_shift_mev=3)
        strong=filtered_result(raw,bounds)
        self.assertEqual(len(strong['products']['raw']['points']),3)
        bounds['min_prominence']=.01
        self.assertEqual(len(filtered_result(raw,bounds)['products']['raw']['points']),6)
        self.assertEqual(raw,before)
        bounds['min_width_mev']=5
        wide=filtered_result(raw,bounds)
        self.assertEqual(len(wide['products']['raw']['points']),3)
        self.assertEqual(len({p['track_id'] for p in wide['products']['raw']['points']}),1)
