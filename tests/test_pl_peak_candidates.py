import pickle
import unittest
from pathlib import Path

import numpy as np

from core.drr_peak_analysis import PeakAnalysisSettings, _detect
from core.peak_candidates import detection_spectrum
from core.pl_peak_candidates import suggest_pl_candidates


def extrema(x, y):
    settings = PeakAnalysisSettings(float(x.min()), float(x.max()), 0, 1, polarity='peaks')
    signal = detection_spectrum(x, y, {'detection_method':'sg', 'smoothing_window':11})
    return _detect(x, signal, settings, True)


class ShoulderCandidateTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(1.30, 1.40, 1001)
        self.rng = np.random.default_rng(2)

    def spectrum(self, shoulder):
        x=self.x
        y=np.exp(-.5*((x-1.35)/.008)**2)
        if shoulder:
            y += .1*np.exp(-.5*((x-1.365)/.0025)**2)
            y += .1*np.exp(-.5*((x-1.375)/.0025)**2)
        return y+self.rng.normal(0,.00005,len(x))

    def test_shoulder_and_existing_peak_are_distinct(self):
        x=self.x; y=self.spectrum(True); original=y.copy(); base=extrema(x,y)
        pts=suggest_pl_candidates(x,y,base,{})
        self.assertTrue(any(abs(p['energy']-1.365)<.001 for p in pts))
        self.assertTrue(any(abs(p['energy']-1.375)<.001 for p in base))
        self.assertFalse(any(abs(p['energy']-1.375)<.0015 for p in pts))
        for p in pts:
            self.assertTrue(p['provisional'])
            self.assertEqual(p['candidate_origin'], 'curvature')
            self.assertEqual(p['prominence'], 0.)
            self.assertGreater(p['detection_contrast'],0.)
            self.assertIn('not FWHM',p['width_definition'])
        np.testing.assert_array_equal(y,original)

    def test_single_broad_peak_with_noise_has_no_shoulder(self):
        y=self.spectrum(False)
        self.assertEqual(suggest_pl_candidates(self.x,y,extrema(self.x,y),{}),[])

    def test_descending_uneven_axis_nan_gap_and_filter_config_independent(self):
        x=self.x+0.000005*np.sin(np.arange(len(self.x))/10);y=self.spectrum(True)
        y[:20]=np.nan
        base=extrema(x,y)
        a=suggest_pl_candidates(x,y,base,{'min_width_mev':.1})
        b=suggest_pl_candidates(x[::-1],y[::-1],base,{'min_width_mev':50.})
        self.assertTrue(a)
        self.assertEqual(a,b)
        self.assertTrue(all(p['energy'] in x for p in a))

    def test_real_p02_representative_shoulders_if_audit_data_available(self):
        path=Path('build/peak-analysis-review/pl-audit-P02-detect.pkl')
        if not path.exists(): self.skipTest('optional local audit data unavailable')
        with path.open('rb') as f: cube,_=pickle.load(f)
        for gate, expected in [(18., (1.34924,1.35679)), (20., (1.34952,1.35820))]:
            i=int(np.argmin(abs(cube.gate-gate)));x=cube.energy;y=cube.Z[i];base=extrema(x,y)
            found=suggest_pl_candidates(x,y,base,{})
            energies=[p['energy'] for p in found]+[p['energy'] for p in base if p['prominence']>0]
            for energy in expected:
                self.assertTrue(any(abs(e-energy)<.0015 for e in energies),(gate,energy,energies))
            self.assertTrue(any(abs(p['energy']-expected[0])<.0015 for p in found))


if __name__=='__main__': unittest.main()
