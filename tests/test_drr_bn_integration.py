"""Real native CSV round trips and external-route isolation for group BN."""
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core import data_io, processing_run
from core.drr_auto_external import resolve_auto_external
from core.drr_bn_calibration import BnCalibrationResult
from core.drr_bn_service import compare_bn_group
from core.drr_bn_store import DrrBnStore, file_digest, comparison_key
from core.drr_sources import DrrMeasurementAssignment, discover_drr_sources, portable_source_name
from core.loader import load_drr_resolved_avg
from tests.bn_joint_fixtures import accepted_joint_report


class BnIntegrationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.store=DrrBnStore(self.root)
        energy=np.linspace(1.80,2.04,241)
        # Intentionally neither wavelength-ascending nor descending.
        order=np.roll(np.arange(len(energy)),55)
        self.columns=np.round((1240/energy)[order],9)
        self.bg='YZ123_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_BNbackground.csv'
        self.names=[f'YZ123_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_measurement_rep{i}.csv' for i in range(4)]
        native_base=100+2*np.sin(np.arange(len(energy))*.7)
        self.write(self.bg,np.stack([native_base+v for v in (0,.1,-.1,0)]),constant=True)
        for i,name in enumerate(self.names):
            self.write(name,np.stack([native_base*(1.1+j*.01)+i*.1 for j in range(3)]))
            os.utime(self.root/name,(1000+i,1000+i))

    def write(self,name,values,constant=False):
        out=io.StringIO(newline='');writer=csv.writer(out)
        writer.writerow(['Vbg','Vtg','Vbias',*self.columns])
        for j,row in enumerate(values):writer.writerow([0 if constant else j*.1,0 if constant else j*.1,0,*row])
        (self.root/name).write_text(out.getvalue(),encoding='utf-8')

    @staticmethod
    def fake_engine(energy,measurements,background,**kwargs):
        candidate=background.copy();candidate[(energy>=1.9)&(energy<=1.97)]*=1.0001
        settings = kwargs['settings']
        return BnCalibrationResult(background.copy(),{'original':background.copy(),'local_shift':candidate},
            'local_shift','local_shift',candidate.copy(),True,'accepted',(),
            accepted_joint_report(settings.derivative_window, settings.derivative_polyorder))

    def compare(self,**kwargs):
        with patch('core.drr_bn_service.evaluate_bn_group',side_effect=self.fake_engine):
            return compare_bn_group(self.root,self.names,[self.bg],y_axis='BG',**kwargs)

    def saved(self):
        payload=self.compare();version=self.store.save_result(payload);self.store.activate(version)
        return payload,version

    def assignments(self,files,background):
        return tuple(DrrMeasurementAssignment(n,'External',(background,),'last') for n in files)

    def test_standard_single_frame_native_column_roundtrip_all_external_routes(self):
        payload,version=self.saved();name=portable_source_name(self.root,version.csv_path)
        for which in ('first','last','all'):
            cube=data_io.load_drr_external_cube(str(self.root),self.names,[name],baseline_which=which,y_axis='BG')
            np.testing.assert_allclose(cube.Z,payload['mean_raw']/payload['candidate_background']-1,rtol=0,atol=1e-14)
            np.testing.assert_array_equal(cube.energy,payload['energy'])
        assignments=self.assignments(self.names,name)
        for loader in (data_io.load_drr_resolved_cube,load_drr_resolved_avg):
            cube=loader(str(self.root),self.names,assignments=assignments,y_axis='BG')
            np.testing.assert_allclose(cube.Z,payload['mean_raw']/payload['candidate_background']-1,rtol=0,atol=1e-14)

    def test_all_public_external_routes_reject_subset_even_with_saved_assignment(self):
        _,version=self.saved();name=portable_source_name(self.root,version.csv_path);subset=self.names[:2]
        with self.assertRaisesRegex(ValueError,'exact measurement group'):
            data_io.load_drr_external_cube(str(self.root),subset,[name],baseline_which='all')
        for loader in (data_io.load_drr_resolved_cube,load_drr_resolved_avg):
            with self.assertRaisesRegex(ValueError,'exact measurement group'):
                loader(str(self.root),subset,assignments=self.assignments(subset,name))

    def test_saved_calibration_requires_original_gate_axis_in_every_load_route(self):
        _,version=self.saved();name=portable_source_name(self.root,version.csv_path)
        with self.assertRaisesRegex(ValueError,'different gate axis'):
            data_io.load_drr_external_cube(str(self.root),self.names,[name],baseline_which='all',y_axis='TG')
        for loader in (data_io.load_drr_resolved_cube,load_drr_resolved_avg):
            with self.assertRaisesRegex(ValueError,'different gate axis'):
                loader(str(self.root),self.names,assignments=self.assignments(self.names,name),y_axis='TG')

    def test_copied_measurements_do_not_count_as_independent_repeats(self):
        raw=(self.root/self.names[0]).read_bytes()
        for name in self.names[1:]:
            (self.root/name).write_bytes(raw)
        with patch('core.drr_bn_service.evaluate_bn_group') as engine:
            with self.assertRaisesRegex(ValueError,'Duplicate measurement contents'):
                compare_bn_group(self.root,self.names,[self.bg],y_axis='BG')
        engine.assert_not_called()
        self.assertFalse(self.store.directory.exists())

    def test_changed_algorithm_identity_requires_fresh_comparison(self):
        payload=self.compare()
        version=self.store.save_result(payload)
        self.store.activate(version)
        # Model a historical version produced by an older numerical pipeline.
        doc=json.loads(version.report_path.read_text(encoding='utf-8'))
        doc['source_identity']['algorithm']={'engine_sha256':'older-implementation'}
        version.report_path.write_text(json.dumps(doc),encoding='utf-8')
        manifest=self.store._read()
        manifest['versions'][0].update(report_sha256=file_digest(version.report_path),
            comparison_key=comparison_key(doc['source_identity'],doc['settings']))
        self.store._atomic_json(self.store.manifest_path,manifest)
        with self.assertRaisesRegex(ValueError,'stale'):
            self.store.find_applied(self.names)
        with patch('core.drr_bn_service.evaluate_bn_group',side_effect=self.fake_engine) as engine:
            refreshed=compare_bn_group(self.root,self.names,[self.bg],y_axis='BG')
        engine.assert_called_once()
        self.assertFalse(refreshed['cached'])

    def test_derivative_settings_key_finds_exact_older_version(self):
        first = self.compare(settings={'derivative_window': 21})
        version = self.store.save_result(first)
        second = self.compare(settings={'derivative_window': 31})
        self.assertFalse(second['cached'])
        self.store.save_result(second)
        with patch('core.drr_bn_service.evaluate_bn_group', side_effect=AssertionError('Older exact comparison refitted')):
            cached = compare_bn_group(self.root, self.names[::-1], [self.bg],
                                      settings={'derivative_window': 21}, y_axis='BG')
        self.assertTrue(cached['cached'])
        self.assertEqual(cached['version_id'], version.version_id)
        self.assertEqual(cached['report']['derivative_settings']['window_length'], 21)

    def test_service_rejects_acceptance_without_joint_validation(self):
        def incomplete(*args, **kwargs):
            result = self.fake_engine(*args, **kwargs)
            result.report.pop('final_confirmation_d2')
            return result
        with patch('core.drr_bn_service.evaluate_bn_group', side_effect=incomplete):
            with self.assertRaisesRegex(ValueError, 'joint DR/R and D2E evidence'):
                compare_bn_group(self.root, self.names, [self.bg], y_axis='BG')
        self.assertFalse(self.store.directory.exists())

    def test_service_requires_actual_d2_training_not_only_posthoc_acceptance(self):
        def posthoc_only(*args, **kwargs):
            result = self.fake_engine(*args, **kwargs)
            result.report['joint_validation']['d2_training_evaluated'] = False
            return result
        with patch('core.drr_bn_service.evaluate_bn_group', side_effect=posthoc_only):
            with self.assertRaisesRegex(ValueError, 'joint DR/R and D2E evidence'):
                compare_bn_group(self.root, self.names, [self.bg], y_axis='BG')
        self.assertFalse(self.store.directory.exists())

    def test_service_requires_rerun_selection_not_only_fixed_candidate_audit(self):
        def incomplete(*args, **kwargs):
            result = self.fake_engine(*args, **kwargs)
            result.report['joint_validation'].pop('selection_sensitivity_audit')
            return result
        with patch('core.drr_bn_service.evaluate_bn_group', side_effect=incomplete):
            with self.assertRaisesRegex(ValueError, 'joint DR/R and D2E evidence'):
                compare_bn_group(self.root, self.names, [self.bg], y_axis='BG')
        self.assertFalse(self.store.directory.exists())

    def test_multiple_fit_bands_are_part_of_comparison_cache_identity(self):
        def preview(*args, **kwargs):
            result = self.fake_engine(*args, **kwargs)
            result.eligible = False
            result.status = 'exploratory'
            return result
        bands = ((1.9, 1.92), (1.925, 1.935))
        with patch('core.drr_bn_service.evaluate_bn_group', side_effect=preview) as engine:
            first = compare_bn_group(self.root, self.names, [self.bg], y_axis='BG',
                                     settings={'train_regions': bands})
            version = self.store.save_result(first)
            changed = compare_bn_group(self.root, self.names, [self.bg], y_axis='BG',
                                       settings={'train_regions': ((1.9, 1.915), (1.925, 1.935))})
            self.assertFalse(changed['cached'])
            cached = compare_bn_group(self.root, self.names, [self.bg], y_axis='BG',
                                      settings={'train_regions': bands})
        self.assertEqual(engine.call_count, 2)
        self.assertTrue(cached['cached'])
        self.assertEqual(cached['version_id'], version.version_id)

    def test_unselected_preview_explains_failed_weak_feature_audit(self):
        def rejected(*args, **kwargs):
            result = self.fake_engine(*args, **kwargs)
            result.eligible = False
            result.selected_method = result.applied_method = 'original'
            result.report['selected_method'] = 'original'
            result.report['candidates']['local_shift']['training_influence_audit'].update(
                passed=False, reasons=['training_feature_influence_refit_failed'])
            return result
        with patch('core.drr_bn_service.evaluate_bn_group', side_effect=rejected):
            payload = compare_bn_group(self.root, self.names, [self.bg], y_axis='BG')
        preview = next(item for item in payload['candidates'] if item['name'] == 'local_shift')
        self.assertFalse(preview['eligible'])
        self.assertEqual(preview['reason'], 'training_feature_influence_refit_failed')

    def test_service_exposes_drr_constraint_separately_from_benefit(self):
        def drr_neutral(*args, **kwargs):
            result = self.fake_engine(*args, **kwargs)
            result.report['candidates']['local_shift']['selection']['mean_improvement_fraction'] = 0.
            result.report['joint_validation']['selection']['drr_benefit_passed'] = False
            return result
        with patch('core.drr_bn_service.evaluate_bn_group', side_effect=drr_neutral):
            payload = compare_bn_group(self.root, self.names, [self.bg], y_axis='BG')
        selected = next(item for item in payload['candidates'] if item['name'] == 'local_shift')
        self.assertTrue(payload['eligible'])
        self.assertEqual(selected['improvement_fraction'], 0.)
        self.assertTrue(selected['drr_noninferiority_passed'])

    def test_unselected_preview_retains_training_bandwidth_reason(self):
        def narrow_band(*args, **kwargs):
            result = self.fake_engine(*args, **kwargs)
            result.eligible = False
            result.selected_method = result.applied_method = 'original'
            result.report['selected_method'] = 'original'
            candidate = result.report['candidates']['local_shift']
            candidate['parameters']['reason'] = 'insufficient_d2_training_bandwidth'
            candidate['selection_joint'].update(passed=False, reasons=['insufficient_d2_training_bandwidth'])
            return result
        with patch('core.drr_bn_service.evaluate_bn_group', side_effect=narrow_band):
            payload = compare_bn_group(self.root, self.names, [self.bg], y_axis='BG')
        preview = next(item for item in payload['candidates'] if item['name'] == 'local_shift')
        self.assertFalse(preview['eligible'])
        self.assertFalse(preview['joint_passed'])
        self.assertEqual(preview['reason'], 'insufficient_d2_training_bandwidth')

    def test_exact_auto_restoration_and_managed_candidate_never_ranked_for_subset(self):
        _,version=self.saved();catalog=discover_drr_sources(self.root,include_all=True)
        result=resolve_auto_external(self.root,catalog,self.names[::-1])
        self.assertTrue(result.resolved,result.reason)
        self.assertEqual(result.assignments[0].baseline_files,(portable_source_name(self.root,version.csv_path),))
        from core.drr_auto_external import rank_baseline_candidates
        seen=[]
        def rank(measurements,candidates):
            seen.extend(candidates)
            return rank_baseline_candidates(measurements,candidates)
        with patch('core.drr_auto_external.rank_baseline_candidates',side_effect=rank):
            resolve_auto_external(self.root,catalog,self.names[:1])
        self.assertFalse(any('.drr_bn_calibration' in item.source for item in seen))

    def export_member_recipe(self, background, which='last'):
        output=self.root/'Processed Data'/'DRR';output.mkdir(parents=True)
        (output/'old.metadata.json').write_text(json.dumps({
            'operation':'DR/R','sources':[
                *[{'role':'measurement','source_path':str(self.root/n)} for n in self.names],
                {'role':'background','source_path':str(self.root/background)}],
            'processing':{'baseline_selection':'External','baseline_which':which,
                          'drr_assignments':[DrrMeasurementAssignment(
                              name,'External',(background,),which).to_dict() for name in self.names]}}),encoding='utf-8')

    def test_original_tombstone_beats_exported_member_recipe(self):
        _,version=self.saved();derived=portable_source_name(self.root,version.csv_path)
        self.export_member_recipe(derived)
        self.store.deactivate(self.names,[self.bg],'all')
        result=resolve_auto_external(self.root,discover_drr_sources(self.root,include_all=True),self.names)
        self.assertTrue(result.resolved,result.reason)
        self.assertTrue(all(a.baseline_files==(self.bg,) and a.baseline_which=='all' for a in result.assignments))

    def test_original_tombstone_preserves_exported_measured_background(self):
        self.saved()
        other=self.bg.replace('BNbackground','BNbackground_other')
        (self.root/other).write_bytes((self.root/self.bg).read_bytes())
        self.export_member_recipe(other, 'first')
        self.store.deactivate(self.names,[self.bg],'all')
        result=resolve_auto_external(self.root,discover_drr_sources(self.root,include_all=True),self.names)
        self.assertTrue(result.resolved,result.reason)
        self.assertTrue(all(a.baseline_files==(other,) and a.baseline_which=='first' for a in result.assignments))

    def test_explicit_original_restore_survives_missing_candidate_and_precedes_pn(self):
        _,version=self.saved()
        self.store.deactivate(self.names,[self.bg],'all',restore_original=True)
        version.csv_path.unlink()
        with patch('core.drr_auto_external.DrrPnStore.find_applied',side_effect=AssertionError('old PN association won')), \
                patch('core.drr_auto_external.rank_baseline_candidates',side_effect=AssertionError('ranking won')):
            result=resolve_auto_external(self.root,(),self.names)
        self.assertTrue(result.resolved,result.reason)
        self.assertTrue(all(a.baseline_files==(self.bg,) and a.baseline_which=='all' for a in result.assignments))
        self.store.deactivate(self.names,[self.bg],'all')
        self.assertIsNone(self.store.original_recipe(self.names))
        with patch('core.drr_auto_external.DrrPnStore.find_applied',return_value=None) as pn:
            resolve_auto_external(self.root,(),self.names)
        pn.assert_called_once()

    def test_unavailable_explicit_original_reference_stops_fallback(self):
        self.store.deactivate(self.names,[self.bg],'all',restore_original=True)
        (self.root/self.bg).unlink()
        with patch('core.drr_auto_external.DrrPnStore.find_applied',side_effect=AssertionError('old PN association won')), \
                patch('core.drr_auto_external.rank_baseline_candidates',side_effect=AssertionError('ranking won')):
            result=resolve_auto_external(self.root,(),self.names)
        self.assertFalse(result.resolved)
        self.assertIn('original BN reference needs review',result.reason)

    def test_saved_result_reused_without_refitting_and_managed_review_unwraps(self):
        payload,version=self.saved()
        with patch('core.drr_bn_service.evaluate_bn_group',side_effect=AssertionError('refit')):
            cached=compare_bn_group(self.root,self.names[::-1],[self.bg],y_axis='BG')
            reviewed=compare_bn_group(self.root,self.names,[str(version.csv_path)],y_axis='BG')
        self.assertTrue(cached['cached']);self.assertEqual(cached['version_id'],version.version_id)
        self.assertEqual(reviewed['source_identity']['baseline_files'][0]['path'],self.bg)
        np.testing.assert_array_equal(reviewed['original_background'],payload['original_background'])

    def test_raw_source_and_equal_file_weighted_background_halves_reach_engine(self):
        observed={}
        def engine(e,z,b,**kwargs):
            observed.update(e=e,z=z,b=b,**kwargs)
            return self.fake_engine(e,z,b,**kwargs)
        with patch('core.drr_bn_service.evaluate_bn_group',side_effect=engine):
            compare_bn_group(self.root,self.names,[self.bg],y_axis='BG')
        self.assertGreater(observed['z'].min(),90)  # raw intensity, not normalized DR/R
        self.assertEqual(observed['background_repeats'].shape,(2,241))
        np.testing.assert_allclose(observed['background_repeats'].mean(0),observed['b'],rtol=1e-14)
        np.testing.assert_array_equal(observed['repeat_order'],np.arange(4))

    def test_matching_missing_optional_bias_is_not_a_coordinate_mismatch(self):
        for name in self.names:
            path=self.root/name
            rows=[row for row in csv.reader(io.StringIO(path.read_text(encoding='utf-8'))) if row]
            for row in rows[1:]:row[2]=''
            out=io.StringIO(newline='');csv.writer(out).writerows(rows)
            path.write_text(out.getvalue(),encoding='utf-8')
        payload=self.compare()
        self.assertEqual(payload['mean_raw'].shape,(3,241))
        self.assertTrue(np.isnan(payload['input_snapshots']['measurement_0_coordinates'][2]).all())

    def test_bn_recorded_bias_must_be_constant_but_unknown_bias_is_allowed(self):
        path=self.root/self.bg
        rows=[row for row in csv.reader(io.StringIO(path.read_text(encoding='utf-8'))) if row]
        for row in rows[1:]:row[2]=''
        out=io.StringIO(newline='');csv.writer(out).writerows(rows)
        path.write_text(out.getvalue(),encoding='utf-8')
        self.compare()
        for i,row in enumerate(rows[1:]):row[2]=str(i*.1)
        out=io.StringIO(newline='');csv.writer(out).writerows(rows)
        path.write_text(out.getvalue(),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'constant bias'):
            self.compare()

    def test_capture_time_ordering_precedes_modification_proxy(self):
        logfile=self.root/'run.experiment.events.jsonl'
        logfile.write_text('\n'.join(json.dumps({'event':'capture_started','output':{'file':str(self.root/n)},
            'started_utc':f'2026-01-01T00:00:0{3-i}Z'}) for i,n in enumerate(self.names)),encoding='utf-8')
        payload=self.compare()
        self.assertEqual(payload['source_identity']['ordering']['basis'],'capture_started_utc')
        self.assertEqual(payload['source_identity']['repeat_order'],self.names[::-1])

    def test_unknown_background_and_mismatched_rotation_rejected(self):
        unknown='Pback.csv';self.write(unknown,np.ones((4,241))*100,constant=True)
        with self.assertRaisesRegex(ValueError,'BNback'):
            compare_bn_group(self.root,self.names,[unknown])
        wrong=self.bg.replace('RotIn94','RotIn49');self.write(wrong,np.ones((4,241))*100,constant=True)
        with self.assertRaisesRegex(ValueError,'rotation'):
            compare_bn_group(self.root,self.names,[wrong])

    def test_first_frame_cannot_invent_independent_bn_halves(self):
        observed={}
        def engine(e,z,b,**kwargs):
            observed.update(kwargs)
            return self.fake_engine(e,z,b,**kwargs)
        with patch('core.drr_bn_service.evaluate_bn_group',side_effect=engine):
            payload=compare_bn_group(self.root,self.names,[self.bg],which='first',y_axis='BG')
        self.assertIsNone(observed['background_repeats'])
        self.assertFalse(payload['report']['background_frames']['independent_halves_available'])

    def test_composite_wrapping_cannot_remove_group_scope(self):
        from core.drr_composite_background import CompositeBackground, CompositeSource, save_composite_background
        payload,version=self.saved()
        source=CompositeSource('cal','Saved group BN',(str(version.csv_path),),'last',
            payload['energy'],tuple(str(v) for v in self.columns),payload['candidate_background'])
        wrapped=self.root/'wrapped.csv'
        save_composite_background(wrapped,CompositeBackground((source,),'cal'))
        with self.assertRaisesRegex(ValueError,'composite wrapping'):
            data_io.load_drr_external_cube(str(self.root),self.names[:1],[str(wrapped)],baseline_which='last')


if __name__=='__main__':unittest.main()
