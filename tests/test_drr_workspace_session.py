import tempfile
import unittest
from pathlib import Path
import numpy as np
from core.loader import DataCube
from core.drr_analysis_workspace import create_dataset,analyze_dataset


class SessionTests(unittest.TestCase):
    def test_roundtrip_preserves_inputs_results_and_view(self):
        import core.drr_workspace_session as session
        x=np.linspace(1,1.1,101);z=np.array([np.exp(-((x-1.05)/.003)**2)]*3)
        d=create_dataset(DataCube(x,np.arange(3.),z,'Y','title','DR/R'),'sample',{'measurement_files':['file.csv']})
        d.result=analyze_dataset(d)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'workspace.npz';views={d.key:((1.02,1.08),(0,2),1,False,1)}
            session.save_session(path,[d],views,d.key)
            datasets,restored,key=session.load_session(path)
        self.assertEqual(key,d.key);self.assertEqual(datasets[0].result,d.result)
        self.assertEqual(restored[d.key][0],[1.02,1.08])
        np.testing.assert_array_equal(datasets[0].cube.Z,z)

    def test_invalid_file_is_rejected(self):
        import core.drr_workspace_session as session
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.npz';path.write_bytes(b'not a workspace')
            with self.assertRaises(ValueError):session.load_session(path)
