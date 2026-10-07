import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.loader import DataCube


def sample(kind='PL', missing=False):
    x = np.linspace(1.60, 1.80, 501)
    y = np.linspace(-2, 2, 9)
    z = np.array([.1 + 1 / (1 + (2 * (x - (1.65 + g * .001)) / .008) ** 2)
                  + (-.6 if kind == 'DRR' else .6) / (1 + (2 * (x - (1.75 - g * .001)) / .012) ** 2)
                  for g in y])
    if missing:
        z[6] = np.nan
    return DataCube(x, y, z, 'Gate (V)', 'Two branches', 'Intensity')


class WorkspaceTests(unittest.TestCase):
    def api(self):
        from core import peak_workspace
        self.assertTrue(hasattr(peak_workspace, 'create_dataset'))
        return peak_workspace

    def test_snapshot_copy_and_identity_include_values_and_channel(self):
        api = self.api(); cube = sample()
        d = api.create_dataset(cube, 'PL', 'PL', 'source.csv', 'PL')
        key = d['key']
        cube.Z[0, 0] += 1
        self.assertNotEqual(key, api.create_dataset(cube, 'PL', 'PL', 'source.csv', 'PL')['key'])
        self.assertFalse(d['cube'].Z.flags.writeable)
        self.assertNotEqual(key, api.create_dataset(sample(), 'PL', 'KK', 'source.csv', 'PL')['key'])

    def test_default_detection_method_depends_on_product(self):
        api=self.api()
        for kind,channel,method in [('PL','PL','sg'),('PL','KK','sg'),('PL','KKp','sg'),
                                    ('DRR','raw','sg'),('DRR','second','local')]:
            with self.subTest(kind=kind,channel=channel):
                d=api.create_dataset(sample(kind),kind,channel,'a.csv','Sample')
                self.assertEqual(d['settings']['detection_method'],method)

    def test_fresh_handoff_recommends_method_but_workspace_restores_user_choice(self):
        api=self.api()
        for channel,label in [('second','2nd derivative'),('second','Mixed derivative'),('raw','1st derivative')]:
            with self.subTest(channel=channel,label=label),tempfile.TemporaryDirectory() as folder:
                d=api.create_dataset(sample('DRR'),'DRR',channel,'a.csv','DRR')
                # A previous main-app process may still send its old SG default.
                d['settings'].update(detection_method='sg',smoothing_window=17,candidate_floor=.023)
                d['product_label']=label
                path=Path(folder)/'snapshot.npz';api.save_workspace(path,[d],d['key'])
                saved,_=api.load_workspace(path)
                self.assertEqual(saved[0]['settings'],d['settings'])
                fresh,_=api.load_workspace(path,fresh_snapshot=True)
                self.assertEqual(fresh[0]['settings']['detection_method'],'local')
                self.assertEqual(fresh[0]['settings']['smoothing_window'],17)
                self.assertEqual(fresh[0]['settings']['candidate_floor'],.023)

    def test_two_pl_branches_keep_ids_across_missing_row_and_round_trip(self):
        api = self.api()
        from core.pl_peak_analysis import detect, track
        d = api.create_dataset(sample(missing=True), 'PL', 'PL', 'a.csv', 'PL')
        seeds = detect(d, 4, d['settings'])
        self.assertEqual(len(seeds), 2)
        seeds[0].update(id='branch-a', name='Exciton', color=0)
        seeds[1].update(id='branch-b', name='Trion', color=1)
        result = track(d, seeds, d['settings'])
        points = result['points']
        self.assertEqual({p['branch_id'] for p in points}, {'branch-a', 'branch-b'})
        self.assertNotIn(6, {p['row_index'] for p in points})
        for p in points:
            expected = 1.65 + p['y'] * .001 if p['branch_id'] == 'branch-a' else 1.75 - p['y'] * .001
            self.assertAlmostEqual(p['energy'], expected, delta=.0005)
        d['branches'] = seeds; d['result'] = result
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.npz'
            api.save_workspace(path, [d], d['key'])
            restored, active = api.load_workspace(path)
            self.assertEqual(active, d['key'])
            self.assertEqual(restored[0]['branches'][0]['name'], 'Exciton')
            self.assertEqual(restored[0]['result'], result)

    def test_drr_detects_and_tracks_peak_and_dip_separately(self):
        api = self.api()
        from core.drr_peak_adapter import detect, track
        d = api.create_dataset(sample('DRR'), 'DRR', 'raw', 'a.csv', 'DRR')
        seeds = detect(d, 4, d['settings'])
        self.assertEqual({s['polarity'] for s in seeds}, {'peak', 'dip'})
        result = track(d, seeds, d['settings'])
        self.assertEqual(len(result['points']), 18)
        self.assertEqual({p['polarity'] for p in result['points']}, {'peak', 'dip'})

    def test_redetection_uses_tracked_position_to_preserve_branch_identity(self):
        from core.pl_peak_analysis import detect,track
        cube=sample();x=cube.energy
        cube.Z=np.array([.1+1/(1+(2*(x-(1.65+g*.007))/.008)**2) for g in cube.gate])
        d=self.api().create_dataset(cube,'PL','PL','moving.csv','Moving peak')
        d['branches']=detect(d,0,d['settings']);d['branches'][0]['name']='Exciton'
        d['result']=track(d,d['branches'],d['settings'])
        found=detect(d,8,d['settings'])
        self.assertEqual(len(found),1)
        self.assertEqual(found[0]['id'],d['branches'][0]['id'])
        self.assertEqual(found[0]['name'],'Exciton')
        self.assertEqual(found[0]['color'],d['branches'][0]['color'])

    def test_fit_uses_raw_intensity_and_reports_centers_widths(self):
        api = self.api()
        from core.pl_peak_analysis import detect, track, fit
        d = api.create_dataset(sample(), 'PL', 'PL', 'a.csv', 'PL')
        result = fit(d, track(d, detect(d, 4, d['settings']), d['settings']), d['settings'])
        self.assertEqual(len(result['fits']), 9)
        self.assertTrue(all(p['fit_status'] == 'ok' for p in result['points']))
        for p in result['points']:
            expected = 1.65 + p['y'] * .001 if p['energy'] < 1.7 else 1.75 - p['y'] * .001
            self.assertAlmostEqual(p['fit_center'], expected, delta=1e-5)
            self.assertAlmostEqual(p['fwhm_mev'], 8 if p['energy'] < 1.7 else 12, delta=.1)

    def test_cancellation_has_no_partial_result_or_source_mutation(self):
        api = self.api()
        from core.pl_peak_analysis import detect, track
        d = api.create_dataset(sample(), 'PL', 'PL', 'a.csv', 'PL')
        before = d['cube'].Z.copy()
        with self.assertRaisesRegex(RuntimeError, 'cancel'):
            track(d, detect(d, 4, d['settings']), d['settings'], cancelled=lambda: True)
        np.testing.assert_array_equal(d['cube'].Z, before)
        self.assertIsNone(d['result'])

    def test_gaussian_fit_recovers_fwhm_in_ev_units(self):
        api=self.api()
        from core.pl_peak_analysis import detect, track, fit
        cube=sample();x=cube.energy
        cube.Z=np.array([.2+np.exp(-4*np.log(2)*((x-1.66-g*.001)/.009)**2) for g in cube.gate])
        d=api.create_dataset(cube,'PL','PL','gauss.csv','Gaussian')
        d['settings']['model']='Gaussian'
        result=fit(d,track(d,detect(d,4,d['settings']),d['settings']),d['settings'])
        self.assertEqual(len(result['points']),9)
        for p in result['points']:
            self.assertEqual(p['fit_status'],'ok')
            self.assertAlmostEqual(p['fit_center'],1.66+p['y']*.001,delta=1e-5)
            self.assertAlmostEqual(p['fwhm_mev'],9,delta=.01)

    def test_rejects_stale_result_and_duplicate_scan_coordinates(self):
        api = self.api()
        cube = sample(); cube.gate[1] = cube.gate[0]
        with self.assertRaisesRegex(ValueError, 'Repeated|unique'):
            api.create_dataset(cube, 'PL', 'PL', 'a.csv', 'PL')
        d = api.create_dataset(sample(), 'PL', 'PL', 'a.csv', 'PL')
        d['result'] = {'dataset_key':'wrong', 'points':[]}
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, 'identity|match'):
            api.save_workspace(Path(folder)/'bad.npz', [d])

    def test_export_keeps_named_branches_and_separate_detected_fitted_columns(self):
        import csv
        import json
        from PIL import Image
        from core.peak_export import export_dataset
        from core.pl_peak_analysis import detect,track,fit
        d=self.api().create_dataset(sample(),'PL','PL','a.csv','PL sample')
        d['branches']=detect(d,4,d['settings']);d['branches'][0]['name']='Exciton'
        d['result']=fit(d,track(d,d['branches'],d['settings']),d['settings'])
        with tempfile.TemporaryDirectory() as folder:
            target=Path(export_dataset(folder,d))
            with (target/'peaks.csv').open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),18)
            self.assertEqual(rows[0]['branch'],'Exciton')
            self.assertTrue(rows[0]['energy']);self.assertTrue(rows[0]['fit_center'])
            self.assertEqual(json.loads((target/'analysis.json').read_text(encoding='utf-8'))['key'],d['key'])
            with Image.open(target/'peaks.png') as picture:
                self.assertEqual(picture.size,(1440,1260))


if __name__ == '__main__':
    unittest.main()
