import unittest
import numpy as np
from core.loader import DataCube
from core.drr_peak_analysis import PeakAnalysisSettings,analyze_drr_peaks


class AnalysisProgressTests(unittest.TestCase):
    def test_progress_reports_intermediate_rows_and_completion(self):
        import inspect
        self.assertIn('progress',inspect.signature(analyze_drr_peaks).parameters)
        x=np.linspace(1,1.1,101);z=np.array([np.exp(-((x-1.05)/.003)**2)]*8)
        cube=DataCube(x,np.arange(8.),z,'Y','','')
        seen=[]
        analyze_drr_peaks(cube,PeakAnalysisSettings(1,1.1,0,7),progress=seen.append)
        self.assertEqual(seen[0],0);self.assertEqual(seen[-1],100)
        self.assertTrue(any(0<v<100 for v in seen));self.assertEqual(seen,sorted(seen))
