import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from core.megasweep.workflow import load_dataset, load_background, analyze, default_parameters

class MegasweepCoreTests(unittest.TestCase):
    def test_frame_weight_and_zero_denominator(self):
        with TemporaryDirectory() as d:
            p=Path(d); pd.DataFrame({'x':[0,1], 'y':[0,0], '700':[4,4], '710':[8,8]}).to_csv(p/'sample.csv',index=False)
            pd.DataFrame({'700':[0], '710':[2]}).to_csv(p/'a.csv',index=False)
            pd.DataFrame({'700':[0,0,0], '710':[6,6,6]}).to_csv(p/'b.csv',index=False)
            ds=load_dataset(p/'sample.csv',mode='Reflection')
            bg=load_background([p/'a.csv',p/'b.csv'],ds['wavelength'])
            np.testing.assert_allclose(bg,[0,5]); self.assertEqual(default_parameters()['background_average'],'all_frames')
            result=analyze(ds,{'view':'slice','auto_background_scale':False,'cut_type':'y','cut_value':0},[p/'a.csv',p/'b.csv'])
            self.assertTrue(np.isnan(result['z'][:,0]).all()); np.testing.assert_allclose(result['z'][:,1],0.6)
            pd.DataFrame({'700':[1], '711':[1]}).to_csv(p/'wrong.csv',index=False)
            with self.assertRaises(ValueError): load_background([p/'wrong.csv'],ds['wavelength'])
    def test_map_preserves_missing(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'sample.csv'; pd.DataFrame({'x':[0,0,1], 'y':[0,1,0], '700':[10,11,12], '710':[9,10,11]}).to_csv(p,index=False)
            ds=load_dataset(p); result=analyze(ds,{'view':'intensity','energy_min':1.7,'energy_max':1.8,'baseline':0,'auto_baseline':False})
            self.assertEqual(result['z'].shape,(2,2)); self.assertTrue(np.isnan(result['z'][1,1]))
            np.testing.assert_array_equal(ds['x_data'],[0,0,1])

    def test_transport_and_transformed_topology(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'t.csv'; pd.DataFrame({'Vbg':[0,0,1,1], 'Vtg':[0,1,0,1], 'current':[2,3,4,5]}).to_csv(p,index=False)
            ds=load_dataset(p,mode='Transport')
            result=analyze(ds,{'view':'transport_map','x_column':'Vbg','y_column':'Vtg','channel':'current'})
            np.testing.assert_array_equal(result['z'],[[2,3],[4,5]])
            curve=analyze(ds,{'view':'slice','x_column':'Vbg','y_column':'Vtg','channel':'current','cut_type':'Vbg','cut_value':0})
            np.testing.assert_array_equal(curve['values'],[2,3])

    def test_auto_baseline_and_background_scaling(self):
        from core.megasweep import megasweep_analysis as upstream
        with TemporaryDirectory() as d:
            p=Path(d); wavelengths=[780,790,800,810,820,830,840,850,860]
            data={'x':[1,0], 'y':[1,0]}
            for n,w in enumerate(wavelengths): data[str(w)]=[10+n, 20+2*n]
            pd.DataFrame(data).to_csv(p/'data.csv',index=False)
            pd.DataFrame({str(w):[5+n/2] for n,w in enumerate(wavelengths)}).to_csv(p/'bg.csv',index=False)
            ds=load_dataset(p/'data.csv'); params={'view':'intensity','energy_min':1.49,'energy_max':1.53}
            result=analyze(ds,params)
            oracle,info=upstream.estimate_global_baseline(ds['Intensity'],ds['energy'],1.49,1.53,return_info=True)
            self.assertEqual(result['diagnostics']['baseline'],oracle)
            ds['mode']='Reflection'
            result=analyze(ds,{'view':'rc_energy','energy_target':1.5,'background_scale_windows':[(1.45,1.6)]},[p/'bg.csv'])
            self.assertEqual(result['diagnostics']['background_scale_frame'],1)
            self.assertAlmostEqual(result['diagnostics']['background_scale'],4.0)

    def test_upstream_numeric_oracles(self):
        from core.megasweep import megasweep_analysis as a
        # Source algorithms at 2612f6f: duplicate coordinates stay point-cloud;
        # absent grid points remain NaN; background zero is explicitly masked.
        payload=a.build_map_payload([0,0,1],[0,0,1],[2,3,4])
        self.assertEqual(payload['grid_kind'],'irregular')
        np.testing.assert_array_equal(payload['z_flat'],[2,3,4])
        np.testing.assert_allclose(a.compute_transformed_coords(np.array([1.,2.]),np.array([3.,4.]),.5),[[3.5,5],[2.5,3]])
        np.testing.assert_allclose(a.compute_rc_spectra([[3.,0.,8.]],[1.,0.,4.]),[[2.,np.nan,1.]],equal_nan=True)
        self.assertAlmostEqual(a.compute_rc_at_energy_map(np.array([[0.,2.]]),np.array([1.,2.]),1.25)[0],.5)
        from core.megasweep.spectral_processing import process_spectral_slice
        energy=np.array([1.,1.1,1.4,1.8,2.]); cut={'energy':energy,'spectra':np.array([energy**2])}
        np.testing.assert_allclose(process_spectral_slice(cut,'second_derivative',5)['spectra'],2,atol=1e-10)

    def test_optical_axes_guess_before_metadata_and_validate(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'sample.csv'
            pd.DataFrame({'Ibias':[1,2], 'Vtg':[3,4], 'Vbg':[5,6], '700':[10,11]}).to_csv(p,index=False)
            ds=load_dataset(p)
            self.assertEqual((ds['x_name'],ds['y_name']),('Vbg','Vtg'))
            np.testing.assert_array_equal(ds['x_data'],[5,6])
            with self.assertRaisesRegex(ValueError,'swapped'):
                load_dataset(p,x_col='Vtg',y_col='Vbg')

    def test_transport_labels_use_only_supplied_units(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'transport.csv'
            p.write_text('Vbg,Vtg,Current,Extra\nV,V,nA,\n0,0,2,1\n0,1,3,2\n1,0,4,3\n1,1,5,4\n',encoding='utf8')
            data=load_dataset(p,mode='Transport')
            params={'x_column':'Vbg','y_column':'Vtg','channel':'Current'}
            result=analyze(data,dict(params,view='transport_map'))
            self.assertEqual((result['x_label'],result['y_label'],result['signal_label']),('Vbg (V)','Vtg (V)','Current (nA)'))
            curve=analyze(data,dict(params,view='transport_curve'))
            self.assertEqual(curve['y_label'],'Current (nA)')
            cut=analyze(data,dict(params,view='slice',cut_type='Vbg',cut_value=0))
            self.assertEqual((cut['x_label'],cut['y_label']),('Vtg (V)','Current (nA)'))
            unlabeled=analyze(data,dict(params,view='transport_curve',channel='Extra'))
            self.assertEqual(unlabeled['y_label'],'Extra')

if __name__=='__main__': unittest.main()
