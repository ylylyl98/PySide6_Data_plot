import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.peak_workspace import create_dataset, save_workspace, load_workspace
from tests.test_peak_workspace import sample


class CandidateTests(unittest.TestCase):
    def api(self):
        from core import peak_tracking
        self.assertTrue(hasattr(peak_tracking, 'detect_heatmap'), 'Whole-map candidate detection is missing')
        return peak_tracking

    def dataset(self, kind='DRR', missing=False):
        return create_dataset(sample(kind, missing), kind, 'raw', 'a.csv', kind)

    def test_full_map_pool_filters_reversibly_without_detection(self):
        api=self.api();d=self.dataset()
        pool=api.detect_heatmap(d, d['settings'])
        self.assertEqual(len(pool['points']),18)
        self.assertEqual({p['row_index'] for p in pool['points']},set(range(9)))
        d['candidates']=pool
        before=copy.deepcopy(pool['points'])
        with patch('core.drr_peak_analysis.find_peaks',side_effect=AssertionError('Detector reran during filtering')):
            result=api.filter_heatmap(d,d['settings'])
            self.assertEqual(len(result['points']),18)
            d['result']=result;d['branches']=result['branches']
            d['branches'][0]['name']='Exciton';identity=d['branches'][0]['id']
            tight=api.filter_heatmap(d,{**d['settings'],'prominence':1.})
            self.assertLess(len(tight['points']),18)
            d['result']=tight
            restored=api.filter_heatmap(d,d['settings'])
        self.assertEqual(len(restored['points']),18)
        self.assertEqual(next(b['name'] for b in restored['branches'] if b['id']==identity),'Exciton')
        self.assertEqual(pool['points'],before)
        self.assertEqual({p['polarity'] for p in restored['points']},{'peak','dip'})
        with tempfile.TemporaryDirectory() as folder:
            d['result']=restored;d['branches']=restored['branches']
            path=Path(folder)/'cache.npz';save_workspace(path,[d],d['key'])
            loaded,_=load_workspace(path)
            self.assertEqual(loaded[0]['candidates']['points'],before)

    def test_range_polarity_width_spacing_and_limit_are_post_filters(self):
        api=self.api();d=self.dataset();d['candidates']=api.detect_heatmap(d,d['settings'])
        result=api.filter_heatmap(d,{**d['settings'],'polarity':'dips','y_min':-.5,'y_max':.5})
        self.assertEqual(len(result['points']),3)
        self.assertTrue(all(p['polarity']=='dip' for p in result['points']))
        result=api.filter_heatmap(d,{**d['settings'],'min_width_mev':100})
        self.assertEqual(result['points'],[])
        result=api.filter_heatmap(d,{**d['settings'],'max_per_row':1})
        self.assertEqual(len(result['points']),9)

    def test_sg_is_explicit_and_does_not_modify_or_differentiate_product(self):
        api=self.api();d=self.dataset(missing=True);before=d['cube'].Z.copy()
        pool=api.detect_heatmap(d,{**d['settings'],'detection_method':'sg','smoothing_window':11})
        self.assertIn('Savitzky',pool['method'])
        self.assertNotIn(6,{p['row_index'] for p in pool['points']})
        self.assertTrue(all(abs(p['energy']-(1.65+p['y']*.001 if p['polarity']=='peak' else 1.75-p['y']*.001))<.0005 for p in pool['points']))
        np.testing.assert_array_equal(d['cube'].Z,before)
        with self.assertRaisesRegex(RuntimeError,'cancel'):
            api.detect_heatmap(d,d['settings'],cancelled=lambda:True)

    def test_detection_has_no_hidden_eight_peak_cap_and_floor_is_recorded(self):
        api=self.api();cube=sample();cube.Z=np.tile(np.sin(np.linspace(0,30*np.pi,501)),(9,1))
        d=create_dataset(cube,'PL','PL','many.csv','Many peaks')
        pool=api.detect_heatmap(d,{**d['settings'],'candidate_floor':.01})
        self.assertEqual(len(pool['points']),135)
        self.assertEqual(pool['settings']['candidate_floor'],.01)

    def test_temporary_branch_split_restores_original_identity_and_color(self):
        api=self.api();d=self.dataset('PL');d['candidates']=api.detect_heatmap(d,d['settings'])
        d['candidates']['points']=[p for p in d['candidates']['points'] if p['energy']<1.7]
        for p in d['candidates']['points']:p['prominence_fraction']=.1 if p['row_index']==2 else .8
        def accept(settings):
            r=api.filter_heatmap(d,settings);d.update(result=r,branches=r['branches']);return r
        first=accept(d['settings']);d['branches'][0]['name']='Exciton'
        identity=d['branches'][0]['id'];color=d['branches'][0]['color']
        split=accept({**d['settings'],'prominence':.2})
        self.assertEqual(len(split['branches']),2)
        split_ids=[b['id'] for b in split['branches']]
        restored=accept(d['settings'])
        self.assertEqual([(b['id'],b['name'],b['color']) for b in restored['branches']],[(identity,'Exciton',color)])
        again=accept({**d['settings'],'prominence':.2})
        self.assertEqual([b['id'] for b in again['branches']],split_ids)

    def test_connected_overlay_keeps_ambiguous_candidates_as_isolated_markers(self):
        from matplotlib.figure import Figure
        from core.peak_plotting import draw_overlay
        api=self.api();cube=sample();x=cube.energy
        cube.Z=np.tile(np.exp(-((x-1.65)/.0005)**2)+np.exp(-((x-1.653)/.0005)**2),(9,1))
        d=create_dataset(cube,'PL','PL','ambiguous.csv','Close peaks')
        d['candidates']=api.detect_heatmap(d,d['settings']);d['result']=api.filter_heatmap(d,d['settings'])
        d['branches']=d['result']['branches'];d['result']['candidate_mode']=False
        expected=sum(p['status']=='uncertain' for p in d['result']['points'])
        self.assertGreater(expected,0)
        axis=Figure().add_subplot();draw_overlay(axis,d)
        self.assertEqual(sum(len(a.get_offsets()) for a in axis.collections if a.get_gid()=='peak-uncertain'),expected)

    def test_fit_records_actual_cached_method_despite_pending_detector_change(self):
        from core.pl_peak_analysis import fit
        api=self.api();d=self.dataset('PL');d['settings']['detection_method']='local'
        d['candidates']=api.detect_heatmap(d,d['settings'])
        result=api.filter_heatmap(d,d['settings'])
        fitted=fit(d,result,{**d['settings'],'detection_method':'sg'})
        self.assertEqual(len(fitted['fits']),9)
        self.assertEqual(fitted['settings']['detection_method'],'local')
        self.assertIn('Local extrema',fitted['method'])

    def test_main_overlay_packet_omits_candidate_cache_and_keeps_visible_positions(self):
        from core import peak_plotting
        from core.peak_workspace import validate_result
        from matplotlib.figure import Figure
        self.assertTrue(hasattr(peak_plotting,'overlay_payload'))
        api=self.api();d=self.dataset();d['candidates']=api.detect_heatmap(d,d['settings'])
        d['result']=api.filter_heatmap(d,d['settings']);d['branches']=d['result']['branches']
        packet=peak_plotting.overlay_payload(d)
        self.assertNotIn('candidates',packet)
        self.assertNotIn('branch_catalog',packet['result'])
        self.assertNotIn('assignments',packet['result'])
        validate_result(d,packet['result'])
        axis=Figure().add_subplot();peak_plotting.draw_overlay(axis,packet)
        self.assertEqual(sum(len(a.get_offsets()) for a in axis.collections),18)

    def test_fresh_defaults_suppress_noise_but_retain_both_moving_features(self):
        from core.loader import DataCube
        api=self.api();x=np.linspace(1.6,1.8,1201);y=np.linspace(-10,10,30)
        z=np.array([np.exp(-((x-1.65-g*.001)/.004)**2)-.6*np.exp(-((x-1.75+g*.001)/.005)**2) for g in y])
        z+=np.random.default_rng(41).normal(0,.02,z.shape)
        d=create_dataset(DataCube(x,y,z,'Gate','Noisy map','DR/R'),'DRR','raw','noise','Noise')
        d['candidates']=api.detect_heatmap(d,d['settings']);r=api.filter_heatmap(d,d['settings'])
        self.assertLessEqual(len(r['points']),63,'Default filtering should reject most noise extrema')
        # A noisy detected extremum is a sampled position, not a fitted center.
        # Allow 1.2 meV against features with roughly 7–8 meV FWHM.
        matches=[p for p in r['points'] if abs(p['energy']-(1.65+p['y']*.001 if p['polarity']=='peak' else 1.75-p['y']*.001))<.0012]
        self.assertEqual(len(matches),60)
        self.assertLess(len(d['candidates']['points']),10000)
