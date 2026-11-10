import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from core.drr_bn_store import DrrBnStore, source_records, validate_bn_assignment_usage
from core.drr_sources import DrrMeasurementAssignment
from core.data_io import load_drr_resolved_cube
from tests.bn_joint_fixtures import accepted_joint_report

class BnAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.store=DrrBnStore(self.root)
        self.energy=np.linspace(1.8,2.04,121); self.columns=np.round(1240/self.energy,9); self.energy=1240/self.columns
        self.names=['sample0.csv','sample1.csv','sample2.csv']; self.raw='sample_BNback.csv'
        for index,name in enumerate([*self.names,self.raw]):
            with (self.root/name).open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f); w.writerow(['Vbg','Vtg',*self.columns])
                for gate in (0,1): w.writerow([gate,gate,*np.full(len(self.energy),100+index*10+gate)])
    def make_version(self,names,level):
        baseline=np.full(len(self.energy),100.); candidate=np.full(len(self.energy),level)
        identity=dict(schema_version=1,root=str(self.root),measurements=source_records(self.root,names),
            baseline_files=source_records(self.root,[self.raw]),baseline_which='all',spectral_columns=self.columns.tolist(),
            y_axis='BG',repeat_order=names,ordering={})
        payload=dict(source_identity=identity,report=accepted_joint_report(),
            settings={'model':1,'derivative_window':21,'derivative_polyorder':2},energy=self.energy,gates=np.array([0.,1.]),
            original_background=baseline,candidate_background=candidate,candidate_backgrounds={'original':baseline,'local_shift':candidate},
            mean_raw=np.ones((2,len(self.energy)))*110,input_snapshots={},eligible=True,status='accepted',
            selected_candidate='local_shift',summary='test',reasons=[],candidates=[])
        return self.store.save_result(payload)
    def assignment(self,measurement,background):
        return DrrMeasurementAssignment(measurement,'External',(str(background),),'all')
    def test_two_singletons_and_original_correct_before_mean(self):
        a=self.make_version([self.names[0]],90.); b=self.make_version([self.names[1]],120.)
        self.store.activate_many((a,b),original_choices=(((self.names[2],),(self.raw,),'all'),))
        assignments=(self.assignment(self.names[0],a.csv_path),self.assignment(self.names[1],b.csv_path),self.assignment(self.names[2],self.raw))
        validate_bn_assignment_usage(self.root,self.names,assignments,y_axis='BG')
        cube=load_drr_resolved_cube(str(self.root),self.names,assignments,y_axis='BG')
        expected=np.mean(np.array([(np.array([100.,101.])/90-1), (np.array([110.,111.])/120-1),
                                   (np.array([120.,121.])/130.5-1)]),axis=0)
        np.testing.assert_allclose(cube.Z, np.broadcast_to(expected[:,None],cube.Z.shape),atol=1e-12)
        swapped=(self.assignment(self.names[0],b.csv_path),self.assignment(self.names[1],a.csv_path),assignments[2])
        with self.assertRaisesRegex(ValueError,'exact measurement group'):
            validate_bn_assignment_usage(self.root,self.names,swapped,y_axis='BG')
        with self.assertRaisesRegex(ValueError,'gate axis'):
            validate_bn_assignment_usage(self.root,self.names,assignments,y_axis='TG')
    def test_group_scope_cannot_expand_or_contract(self):
        v=self.make_version(self.names[:2],90.); self.store.activate(v)
        good=tuple(self.assignment(name,v.csv_path) for name in self.names[:2])
        validate_bn_assignment_usage(self.root,self.names[:2],good,y_axis='BG')
        with self.assertRaisesRegex(ValueError,'exact measurement group'):
            validate_bn_assignment_usage(self.root,self.names[:1],good[:1],y_axis='BG')
        with self.assertRaisesRegex(ValueError,'exact measurement group'):
            validate_bn_assignment_usage(self.root,self.names,(*good,self.assignment(self.names[2],v.csv_path)),y_axis='BG')
    def test_batch_is_atomic_on_failure_and_staleness(self):
        a=self.make_version([self.names[0]],90.); b=self.make_version([self.names[1]],120.)
        before=self.store.manifest_path.read_bytes()
        with self.assertRaisesRegex(ValueError,'same measurement'):
            self.store.activate_many((a,b),original_choices=(((self.names[0],),(self.raw,),'all'),))
        self.assertEqual(before,self.store.manifest_path.read_bytes())
        with (self.root/self.names[1]).open('a',encoding='utf-8') as f: f.write('\n')
        with self.assertRaisesRegex(ValueError,'fresh'):
            self.store.activate_many((a,b))
        self.assertEqual(before,self.store.manifest_path.read_bytes())
    def test_managed_averaging_and_unactivated_are_rejected(self):
        v=self.make_version([self.names[0]],90.)
        with self.assertRaisesRegex(ValueError,'not been activated'):
            validate_bn_assignment_usage(self.root,self.names[:1],(self.assignment(self.names[0],v.csv_path),),y_axis='BG')
        bad=DrrMeasurementAssignment(self.names[0],'External',(str(v.csv_path),self.raw),'all')
        with self.assertRaisesRegex(ValueError,'averaged or spliced'):
            validate_bn_assignment_usage(self.root,self.names[:1],(bad,),y_axis='BG')

    def test_original_only_commit_and_write_failure_preserve_manifest(self):
        version=self.make_version([self.names[0]],90.)
        before=self.store.manifest_path.read_bytes()
        with patch.object(self.store,'_atomic_json',side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError,'disk full'):
                self.store.activate_many((version,),original_choices=(((self.names[1],),(self.raw,),'all'),))
        self.assertEqual(before,self.store.manifest_path.read_bytes())
        self.store.activate_many((),original_choices=(((self.names[1],),(self.raw,),'all'),))
        self.assertEqual(self.store.original_recipe((self.names[1],))['baseline_which'],'all')

    def test_group_may_coexist_with_unrelated_raw_assignment(self):
        v=self.make_version(self.names[:2],90.); self.store.activate(v)
        assignments=tuple(self.assignment(name,v.csv_path) for name in self.names[:2])+(self.assignment(self.names[2],self.raw),)
        validate_bn_assignment_usage(self.root,self.names,assignments,y_axis='BG')
