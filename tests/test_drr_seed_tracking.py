import unittest
from dataclasses import replace
import numpy as np
from core.loader import DataCube
from core.drr_peak_analysis import PeakAnalysisSettings, analyze_drr_peaks


class SeedTrackingTests(unittest.TestCase):
    def run_case(self, rows, source='raw', **kw):
        x=np.linspace(1,1.1,1001)
        cube=DataCube(x,np.arange(len(rows),dtype=float),np.asarray(rows),'Y','','')
        s=PeakAnalysisSettings(1,1.1,0,len(rows)-1,source=source,polarity='peaks')
        self.assertIn('mode',s.__dataclass_fields__, 'Seeded tracking settings are missing')
        s=replace(s,mode='seed',seed_energy=1.05,seed_y=5,**kw)
        return analyze_drr_peaks(cube,s)['products'][source]['points']

    def test_tracks_seed_not_stronger_neighbor(self):
        x=np.linspace(1,1.1,1001);rng=np.random.default_rng(8)
        rows=[np.exp(-((x-(1.049+.0002*i))/.0015)**2)+3*np.exp(-((x-1.065)/.002)**2)+rng.normal(0,.015,len(x)) for i in range(11)]
        pts=self.run_case(rows)
        self.assertEqual(len(pts),11)
        self.assertTrue(all(abs(p['energy']-(1.049+.0002*p['row_index']))<.0005 for p in pts))

    def test_blank_rows_not_filled_and_long_gap_stops(self):
        x=np.linspace(1,1.1,1001);rows=[np.exp(-((x-1.05)/.002)**2) for _ in range(11)]
        rows[1:4]=[np.zeros(len(x)) for _ in range(3)]
        pts=self.run_case(rows)
        self.assertEqual([p['row_index'] for p in pts],list(range(4,11)))

    def test_noise_only_seed_is_rejected(self):
        rng=np.random.default_rng(22)
        with self.assertRaisesRegex(ValueError,'seed'):
            self.run_case(rng.normal(0,.01,(11,1001)))

    def test_two_close_competing_branches_leave_gap(self):
        x=np.linspace(1,1.1,1001);rows=[np.exp(-((x-1.05)/.0005)**2) for _ in range(11)]
        rows[6]=np.exp(-((x-1.049)/.0005)**2)+np.exp(-((x-1.051)/.0005)**2)
        pts=self.run_case(rows,min_width_mev=.4)
        self.assertNotIn(6,[p['row_index'] for p in pts])

    def test_correlated_derivative_noise_seed_is_rejected(self):
        rng=np.random.default_rng(22)
        with self.assertRaisesRegex(ValueError,'seed'):
            self.run_case(rng.normal(0,.01,(11,1001)),source='second',sg_window=31)
