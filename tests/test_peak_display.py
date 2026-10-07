import tempfile
import unittest
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from core.colormaps import RDBU_R_P0P45_ID, register_colormaps
from core.plotting import HeatmapParams, SplitColorScale, plot_heatmap
from core.peak_plotting import build_figure
from core.peak_workspace import create_dataset, save_workspace, load_workspace
from tests.test_peak_workspace import sample


class PeakDisplayTests(unittest.TestCase):
    def test_descending_energy_keeps_each_regions_physical_colors(self):
        from dataclasses import replace
        from core.peak_display import capture_display
        cube=sample('DRR');cube=replace(cube,energy=cube.energy[::-1],Z=cube.Z[:,::-1])
        axis=Figure().add_subplot()
        original=plot_heatmap(axis,cube,HeatmapParams('Main','Energy','Gate','DR/R',-1,1,
            (1.6,1.8),(-2,2),cmap='RdBu_r',split_scale=SplitColorScale(1.7,0,10,-1,1)))
        d=create_dataset(cube,'DRR','raw','descending.csv','DRR',{'display':capture_display(axis)})
        heat,*_=build_figure(d)
        def colors(meshes):
            output=np.zeros((*cube.Z.shape,4))
            for mesh in meshes:
                values=mesh.get_array();valid=~np.ma.getmaskarray(values)
                output[valid]=mesh.to_rgba(values)[valid]
            return output
        np.testing.assert_allclose(colors(heat.collections),colors(original.images)[:,::-1])

    def test_snapshot_preserves_custom_colors_norms_and_split_regions(self):
        from core.peak_display import capture_display
        register_colormaps()
        cube=sample('DRR')
        for split,log,center in [(None,False,True),
            (SplitColorScale(1.70,-.2,1.2,-.8,.3),False,True),
            (SplitColorScale(1.66,.01,1.2,.03,.7,split_x2=1.74,middle_vmin=.02,middle_vmax=.8),True,False)]:
            with self.subTest(split=split,log=log):
                axis=Figure().add_subplot()
                params=HeatmapParams('Main','Energy','Gate','DR/R',-.8 if not log else .01,1.2,
                    (1.60,1.8),(-2,2),cmap=RDBU_R_P0P45_ID,log_scale=log,center_zero=center,split_scale=split)
                original=plot_heatmap(axis,cube,params)
                view={'display':capture_display(axis),'xlim':[1.61,1.68],'ylim':[-1,1]}
                d=create_dataset(cube,'DRR','raw','a.csv','DRR',view)
                key=d['key']
                with tempfile.TemporaryDirectory() as folder:
                    path=Path(folder)/'snapshot.npz';save_workspace(path,[d])
                    restored=load_workspace(path)[0][0]
                heat,*_=build_figure(restored)
                self.assertEqual(restored['key'],key)
                self.assertEqual(len(heat.collections),len(original.images))
                values=np.array([.01,.03,.1,.2,.7,1.])
                for expected,actual in zip(original.images,heat.collections):
                    self.assertEqual(type(actual.norm),type(expected.norm))
                    np.testing.assert_allclose(actual.cmap(actual.norm(values)),expected.cmap(expected.norm(values)))
                    np.testing.assert_array_equal(np.ma.getmaskarray(actual.get_array()),np.ma.getmaskarray(expected.get_array()))
                np.testing.assert_allclose(heat.get_xlim(),[1.61,1.68])


if __name__=='__main__':unittest.main()
