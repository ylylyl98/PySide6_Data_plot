"""DRR export names and metadata through the native Quick export path."""
import json
from dataclasses import replace
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
os.environ.setdefault('MPLBACKEND','agg')

from core.drr_sources import DrrMeasurementAssignment
from ui_qt.quick.drr_workflow import DrrRequest, load_drr, export_drr
from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft, prepare_application

class QuickDrrExportNamesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.m=self.write('sample_TG_rep01.csv',varying=True)
        self.m2=self.write('sample_TG_rep02.csv',varying=True,offset=3)
        self.p=self.write('Pbackground.csv',offset=5,gate_value=-10)
        self.n=self.write('Nbackground.csv',offset=10,gate_value=10)
        self.bn=self.write('BNbackground.csv',offset=15)
        self.pn=self.write('composite.csv',offset=20)
        self.custom=self.write('P_looks_like_background.csv',offset=25)
    def write(self,name,varying=False,offset=0,gate_value=0):
        path=self.root/name;wave=np.linspace(630,690,41);gate=np.linspace(-2,2,9) if varying else np.full(4,gate_value)
        data=150+offset+5*gate[:,None]+np.linspace(-1,1,41)[None,:]**2
        np.savetxt(path,np.column_stack((gate,gate,data)),delimiter=',',header='Vbg,Vtg,'+','.join(map(str,wave)),comments='')
        return path
    def products(self,kind,files,*,measurements=None,assignments=(),raw=None,second=2,window=9):
        samples=tuple(str(x) for x in (measurements or (self.m,)))
        context=BackgroundContext(str(self.root),samples,'tg',window,2)
        draft=BackgroundDraft(kind=kind,files=tuple(str(x) for x in files),which='all',saved_assignments=assignments)
        prepared=prepare_application(context,draft)
        request=DrrRequest(samples,background_mode='external',backgrounds=prepared.recipe.files,background_frame='all',
            y_axis='tg',window=window,polyorder=2,raw_derivative=raw,second_derivative=second,
            experiment_folder=str(self.root),background_recipe=prepared.recipe)
        return load_drr(request,prepared_background=prepared)
    def params(self,cube):
        from core.plotting import HeatmapParams
        finite=cube.Z[np.isfinite(cube.Z)];low,high=float(finite.min()),float(finite.max())
        if low==high:high=low+1
        return HeatmapParams(cube.title,'Energy',cube.gate_label,cube.cbar_label,low,high,
            (float(min(cube.energy)),float(max(cube.energy))),(float(min(cube.gate)),float(max(cube.gate))),cmap='RdBu_r')
    def export(self,products,view='raw'):
        with patch('core.export._save_heatmap_png',side_effect=lambda path,*args,**kwargs:Path(path).write_bytes(b'PNG')) as renderer:
            result=export_drr(products,folder=str(self.root),raw_params=self.params(products.raw),
                second_params=self.params(products.second) if products.second is not None else None,view=view)
            self.assertTrue(result['raw_png'].exists())
        return result
    def metadata(self,result,key='raw_dat'):
        return json.loads(result[key].with_suffix('.metadata.json').read_text(encoding='utf8'))

    def test_background_labels_and_metadata_do_not_guess_unknown_names(self):
        for kind, background, expected in (('p',self.p,'DR_R_P'),('n',self.n,'DR_R_N'),
                                           ('bn',self.bn,'DR_R_BN'),('pn',self.pn,'DR_R_PN'),
                                           ('custom',self.custom,'DR_R_External')):
            with self.subTest(kind=kind):
                result=self.export(self.products(kind,(background,)))
                self.assertIn('_'+expected,result['raw_png'].stem)
                self.assertTrue(result['raw_dat'].exists())
                meta=self.metadata(result)['processing']
                self.assertEqual(meta['mode'],'DR/R External')
                self.assertEqual(meta['baseline_selection'],'External')
                self.assertEqual(meta['sg_window'],9)
                self.assertEqual(meta['sg_polyorder'],2)
                self.assertEqual(meta['derivative_grid'],'More correct (regrid)')
                self.assertEqual(len(meta['drr_background_assignments']),1)

    def test_real_saved_pn_version_name(self):
        from core.drr_pn_pairing import DrrPnPair
        from ui_qt.quick.drr_background_workflow import generate_pn
        context=BackgroundContext(str(self.root),(self.m.name,),'tg',9,2)
        version, _document, _report=generate_pn(context,DrrPnPair((self.p.name,),(self.n.name,)),mode='pn-only')
        prepared=prepare_application(context,BackgroundDraft(kind='pn',which='all',version=version))
        request=DrrRequest((str(self.m),),background_mode='external',backgrounds=prepared.recipe.files,
            background_frame='all',y_axis='tg',window=9,polyorder=2,experiment_folder=str(self.root),
            background_recipe=prepared.recipe)
        result=self.export(load_drr(request,prepared_background=prepared))
        self.assertTrue(result['raw_png'].stem.endswith('_DR_R_PN'))
        metadata=self.metadata(result)['processing']
        self.assertEqual(metadata['background_kind'],'pn')
        self.assertEqual(metadata['mode'],'DR/R External')

    def test_real_batch_generated_pn_uses_match_label(self):
        from core.drr_pn_batch_plan import build_pn_batch_plan
        from core.drr_pn_reconstruction import pipeline
        from ui_qt.quick.drr_pn_batch_workflow import generate_pn_batch, prepare_pn_batch_application
        p2=self.write('Pbackground_2.csv',offset=8,gate_value=-10)
        n2=self.write('Nbackground_2.csv',offset=12,gate_value=10)
        for path,stamp in ((self.m,1000),(self.m2,2000),(self.p,900),(self.n,905),(p2,1900),(n2,1905)):
            os.utime(path,(stamp,stamp))
        context=BackgroundContext(str(self.root),(self.m.name,self.m2.name),'tg',9,2)
        plan=build_pn_batch_plan(self.root,context.measurements,p_candidates=(self.p.name,p2.name),
                                 n_candidates=(self.n.name,n2.name))
        self.assertTrue(plan.ready,plan.warnings)
        def stub(energy,pn_mean,frames,samples,**kwargs):
            if kwargs['method']=='conditioned_single':
                algorithm,protocol=pipeline.CONDITIONED_SINGLE_ALGORITHM_VERSION,pipeline.CONDITIONED_SINGLE_PROTOCOL
            else:
                algorithm,protocol=pipeline.CONDITIONED_BOUNDED_ALGORITHM_VERSION,pipeline.CONDITIONED_BOUNDED_PROTOCOL
            return {'energy':energy.copy(),'background':pn_mean.mean(0)}, {'algorithm_version':algorithm,'settings':dict(protocol),'BN_used':False}
        with patch.object(pipeline,'reconstruct_background',side_effect=stub):
            batch=generate_pn_batch(context,plan)
        prepared=prepare_pn_batch_application(context,batch,warnings_acknowledged=True)
        request=DrrRequest(tuple(str(self.root/n) for n in context.measurements),background_mode='external',
            backgrounds=prepared.recipe.files,background_frame='all',y_axis='tg',window=9,polyorder=2,
            experiment_folder=str(self.root),background_recipe=prepared.recipe)
        result=self.export(load_drr(request,prepared_background=prepared))
        self.assertIn('_DR_R_PN-Match',result['raw_png'].stem)
        self.assertEqual(len(self.metadata(result)['processing']['drr_background_assignments']),2)

    def test_pn_heterogeneous_assignment_uses_match_label(self):
        assignments=(DrrMeasurementAssignment(self.m.name,'External',(str(self.p),),'all'),
                     DrrMeasurementAssignment(self.m2.name,'External',(str(self.n),),'all'))
        products=self.products('pn',(),measurements=(self.m,self.m2),assignments=assignments)
        result=self.export(products)
        self.assertIn('_DR_R_PN-Match',result['raw_png'].stem)
        processing=self.metadata(result)['processing']
        self.assertEqual(processing['mode'],'DR/R External')
        self.assertEqual(len(processing['drr_background_assignments']),2)
        self.assertEqual(processing['numerical_path'],'heterogeneous')

    def test_same_reference_matched_pn_remains_pn(self):
        assignments=(DrrMeasurementAssignment(self.m.name,'External',(str(self.pn),),'all'),
                     DrrMeasurementAssignment(self.m2.name,'External',(str(self.pn),),'all'))
        products=self.products('pn',(),measurements=(self.m,self.m2),assignments=assignments)
        result=self.export(products)
        self.assertTrue(result['raw_png'].stem.endswith('_DR_R_PN'))
        self.assertNotIn('PN-Match',result['raw_png'].stem)

    def test_missing_recipe_draft_is_external_even_if_filename_looks_like_p(self):
        products=self.products('p',(self.p,))
        request=replace(products.request,background_recipe=replace(products.request.background_recipe,draft=None))
        products=replace(products,request=request)
        result=self.export(products)
        self.assertIn('_DR_R_External',result['raw_png'].stem)
        self.assertEqual(self.metadata(result)['processing']['mode'],'DR/R External')

    def assert_anonymous_kind_round_trip(self, kind, expected):
        from ui_qt.quick.drr_background_restore import restore_saved_background
        original=self.products(kind,(self.pn,))
        self.export(original)
        fresh=DrrRequest((str(self.m),),y_axis='tg',window=9,polyorder=2,experiment_folder=str(self.root))
        restored=restore_saved_background(fresh)
        self.assertIsNotNone(restored.background_recipe)
        self.assertEqual(restored.background_recipe.draft.kind,kind)
        result=self.export(load_drr(restored))
        self.assertIn('_'+expected,result['raw_png'].stem)
        self.assertEqual(self.metadata(result)['processing']['background_kind'],kind)

    def test_anonymous_p_background_kind_survives_restore_and_export(self):
        self.assert_anonymous_kind_round_trip('p','DR_R_P')

    def test_anonymous_pn_background_kind_survives_restore_and_export(self):
        self.assert_anonymous_kind_round_trip('pn','DR_R_PN')

    def test_raw_first_second_and_mixed_derivatives_use_short_names(self):
        for derivative, expected in ((None,None),(1,'_dE'),(2,'_d2E')):
            with self.subTest(derivative=derivative):
                result=self.export(self.products('p',(self.p,),raw=derivative))
                name=result['raw_png'].stem
                if expected is not None:self.assertTrue(name.endswith(expected),name)
                self.assertNotIn('_W9_',name);self.assertNotIn('_O2_',name)
                processing=self.metadata(result)['processing']
                self.assertEqual(processing['sg_window'],9)
                self.assertEqual(processing['sg_polyorder'],2)
                self.assertEqual(processing['derivative_order'],derivative)
        result=self.export(self.products('p',(self.p,),second=('mixed',9)),view='second')
        self.assertTrue(result['second_png'].stem.endswith('_dXdY'))
        metadata=self.metadata(result,'second_dat')['processing']
        self.assertEqual(metadata['derivative_axes'],'XY')
        self.assertEqual(metadata['sg_window'],9)
        self.assertEqual(metadata['savgol_window_y'],9)

    def test_old_long_name_metadata_remains_restorable(self):
        from core.drr_saved_background import resolve_saved_group_background
        from core.drr_sources import discover_drr_sources
        products=self.products('p',(self.p,),raw=2,window=9)
        old_base='sample_TG_rep01_avg1_DR_R_External_d2E_W9_O2_regrid'
        with patch('core.export.build_drr_export_base',return_value=old_base):
            legacy=self.export(products)
        self.assertIn('_W9_O2_',legacy['raw_dat'].stem)
        restored=resolve_saved_group_background(self.root,discover_drr_sources(self.root),[self.m.name])
        self.assertIsNotNone(restored)
        self.assertTrue(restored.resolved,restored.reason)
        self.assertEqual(restored.assignments[0].measurement_file,self.m.name)
        compact=self.export(products)
        self.assertNotEqual(compact['raw_dat'],legacy['raw_dat'])
        self.assertTrue(compact['raw_dat'].stem.endswith('_DR_R_P_d2E'),compact['raw_dat'].name)
        self.assertTrue(legacy['raw_dat'].exists())

    def test_different_sg_windows_do_not_overwrite_and_identical_reuses(self):
        first=self.export(self.products('p',(self.p,),raw=2,window=9))
        original=first['raw_dat'].read_bytes()
        second=self.export(self.products('p',(self.p,),raw=2,window=7))
        self.assertNotEqual(first['raw_dat'],second['raw_dat'])
        self.assertEqual(first['raw_dat'].read_bytes(),original)
        self.assertEqual(self.metadata(first)['processing']['sg_window'],9)
        self.assertEqual(self.metadata(second)['processing']['sg_window'],7)
        repeated=self.export(self.products('p',(self.p,),raw=2,window=9))
        self.assertEqual(repeated['raw_dat'],first['raw_dat'])

if __name__=='__main__':unittest.main()
