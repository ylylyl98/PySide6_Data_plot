"""Version publication and whole-group scope are independent of fit success."""
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_bn_store import (DrrBnStore, source_records, validate_bn_background_usage,
                               file_digest, optimization_policy_current, joint_validation_passed)
from tests.bn_joint_fixtures import accepted_joint_report


class BnStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = DrrBnStore(self.root)
        self.energy = np.linspace(1.80, 2.04, 121)
        self.columns = np.round(1240 / self.energy, 9)
        self.energy = 1240 / self.columns
        self.measurements = [f'sample_{i}.csv' for i in range(4)]
        self.background = 'sample_BNback.csv'
        for index, name in enumerate([*self.measurements, self.background]):
            self.write_source(name, offset=index * .1)

    def write_source(self, name, offset=0.):
        stream = io.StringIO(newline=''); writer = csv.writer(stream)
        writer.writerow(['Vbg', 'Vtg', *self.columns])
        writer.writerow([0, 0, *np.full(len(self.columns), 100. + offset)])
        writer.writerow([1, 1, *np.full(len(self.columns), 110. + offset)])
        (self.root / name).write_text(stream.getvalue(), encoding='utf-8')

    def payload(self, *, eligible=True):
        baseline = np.full(len(self.energy), 100.)
        candidate = baseline.copy(); candidate[40:70] += .25
        identity = dict(schema_version=1, root=str(self.root),
                        measurements=source_records(self.root, self.measurements),
                        baseline_files=source_records(self.root, [self.background]),
                        baseline_which='all', spectral_columns=self.columns.tolist(), y_axis='BG',
                        repeat_order=self.measurements, ordering={})
        return dict(source_identity=identity, report=accepted_joint_report(),
                    settings={'model': 1, 'derivative_window': 21, 'derivative_polyorder': 2},
                    energy=self.energy, gates=np.array([0., 1.]), original_background=baseline,
                    candidate_background=candidate, candidate_backgrounds={'original': baseline, 'local_shift': candidate},
                    mean_raw=np.ones((2,len(self.energy)))*110, input_snapshots={'measurement_0':np.ones((2,len(self.energy)))*110},
                    eligible=eligible, status='accepted' if eligible else 'exploratory',
                    selected_candidate='local_shift', summary='Comparison test', reasons=[], candidates=[])

    def test_save_is_immutable_and_does_not_activate(self):
        version=self.store.save_result(self.payload())
        self.assertIsNone(self.store.find_applied(self.measurements))
        again=self.store.save_result(self.payload())
        self.assertNotEqual(version.csv_path,again.csv_path)
        restored=self.store.load_result(version)
        np.testing.assert_array_equal(restored['candidate_background'],self.payload()['candidate_background'])
        self.assertEqual(restored['version_id'],version.version_id)

    def test_new_eligible_publication_requires_both_product_evidence(self):
        for missing in ('joint_validation', 'final_confirmation_d2', 'derivative_settings'):
            with self.subTest(missing=missing):
                payload = self.payload()
                payload['report'].pop(missing)
                with self.assertRaisesRegex(ValueError, 'joint DR/R and D2E'):
                    self.store.save_result(payload)
        self.assertFalse(self.store.directory.exists())

    def test_d2_priority_acceptance_does_not_require_drr_benefit(self):
        payload = self.payload()
        report = payload['report']
        for role in ('selection', 'final'):
            report['joint_validation'][role]['drr_benefit_passed'] = False
        report['final_confirmation'].update(candidate_rms=[1., 1.], mean_candidate_rms=1.,
                                            mean_improvement_fraction=0.)
        version = self.store.save_result(payload)
        self.assertTrue(version.eligible)
        self.store.activate(version)
        self.assertEqual(self.store.find_applied(self.measurements).version_id, version.version_id)

    def test_drr_damage_cannot_be_overridden_by_d2_improvement(self):
        payload = self.payload()
        payload['report']['joint_validation']['final']['drr_noninferiority_passed'] = False
        with self.assertRaisesRegex(ValueError, 'joint DR/R and D2E'):
            self.store.save_result(payload)

    def test_policy_identity_is_distinct_from_acceptance(self):
        report = accepted_joint_report()
        report['joint_validation']['passed'] = False
        self.assertTrue(optimization_policy_current(report))
        self.assertFalse(joint_validation_passed(report))
        for old in ({}, {'schema_version': 2}, None):
            self.assertFalse(optimization_policy_current(old))

    def test_old_dual_benefit_policy_cannot_publish_as_current(self):
        payload = self.payload()
        payload['report'].update(schema_version=2, optimization_policy='joint_drr_d2_v1')
        payload['report']['joint_validation'].update(schema_version=1, policy='joint_drr_d2_v1')
        with self.assertRaisesRegex(ValueError, 'joint DR/R and D2E'):
            self.store.save_result(payload)

    def test_previous_d2_policy_cannot_publish_without_shrinkage_selection_audit(self):
        payload = self.payload()
        payload['report'].update(schema_version=3, optimization_policy='d2_priority_v1')
        payload['report']['joint_validation'].update(schema_version=2, policy='d2_priority_v1')
        self.assertFalse(optimization_policy_current(payload['report']))
        with self.assertRaisesRegex(ValueError, 'joint DR/R and D2E'):
            self.store.save_result(payload)

    def test_all_fixed_strengths_retain_publishable_evidence(self):
        for family in ('local_shift', 'local_shift_gain'):
            for suffix in ('_s25', '_s50', '_s75', ''):
                with self.subTest(method=family + suffix):
                    self.assertTrue(joint_validation_passed(accepted_joint_report(method=family + suffix)))

    def test_post_fit_shrinkage_cannot_be_replaced_by_fitted_alpha(self):
        for mutation in ('alpha', 'family', 'fitted_alpha', 'not_post_fit', 'raw_parameters',
                         'raw_failed', 'raw_bound', 'digest', 'point_count', 'amplitude'):
            with self.subTest(mutation=mutation):
                report = accepted_joint_report(method='local_shift_s25')
                parameters = report['candidates']['local_shift_s25']['parameters']
                if mutation == 'alpha':
                    parameters['correction_strength'] = .5
                elif mutation == 'family':
                    parameters['family'] = 'local_shift_gain'
                elif mutation == 'fitted_alpha':
                    parameters['shrinkage']['alpha_is_fitted'] = True
                elif mutation == 'not_post_fit':
                    parameters['shrinkage']['post_fit'] = False
                elif mutation == 'raw_parameters':
                    parameters['unshrunk_parameters']['shift_pixel'] = .2
                elif mutation == 'raw_failed':
                    parameters['unshrunk_parameters']['optimizer_success'] = False
                elif mutation == 'raw_bound':
                    parameters['unshrunk_parameters']['near_parameter_bound'] = True
                elif mutation == 'digest':
                    parameters['unshrunk_background']['sha256'] = 'invalid'
                elif mutation == 'point_count':
                    parameters['unshrunk_background']['point_count'] = 1
                else:
                    parameters['unshrunk_background']['log_correction_rms'] = .1
                self.assertFalse(joint_validation_passed(report))

    def test_withdrawn_trial_strength_cannot_be_reused_or_compete(self):
        for family in ('local_shift', 'local_shift_gain'):
            name = family + '_s12p5'
            self.assertFalse(joint_validation_passed(accepted_joint_report(method=name)))
            report = accepted_joint_report()
            report['joint_validation']['selection_sensitivity_audit']['baseline_qualified_candidates'].append(name)
            self.assertFalse(joint_validation_passed(report))

    def test_full_selector_audit_is_required_and_cannot_hide_original_fallback(self):
        for mutation in ('missing', 'failed', 'fixed_only', 'final_read', 'train_scope',
                         'selection_scope', 'duplicate_probe', 'peak_mismatch', 'alpha_mismatch',
                         'fallback_not_counted', 'unaggregated_error', 'omitted_baseline'):
            with self.subTest(mutation=mutation):
                report = accepted_joint_report(method='local_shift_s50')
                audit = report['joint_validation']['selection_sensitivity_audit']
                if mutation == 'missing':
                    report['joint_validation'].pop('selection_sensitivity_audit')
                elif mutation == 'failed':
                    audit['passed'] = False
                elif mutation == 'fixed_only':
                    audit['selection_aware'] = False
                elif mutation == 'final_read':
                    audit['final_repeats_read'] = True
                elif mutation == 'train_scope':
                    audit['train_repeat_indices'] = [0, 2]
                elif mutation == 'selection_scope':
                    audit['selection_repeat_indices'] = [2]
                elif mutation == 'duplicate_probe':
                    audit['chosen_probe_references'][1] = dict(audit['chosen_probe_references'][0])
                elif mutation == 'peak_mismatch':
                    audit['probes'][0]['center_energy'] += .001
                elif mutation == 'alpha_mismatch':
                    audit['probes'][0]['perturbed_correction_strength'] = .25
                elif mutation == 'fallback_not_counted':
                    audit['probes'][0].update(perturbed_selected_method='original',
                                              perturbed_family='original', perturbed_correction_strength=0.)
                elif mutation == 'unaggregated_error':
                    audit['probes'][0]['d2_error_fraction'] = .051
                else:
                    audit['baseline_qualified_candidates'] = ['local_shift_s25']
                self.assertFalse(joint_validation_passed(report))

    def test_multiple_fit_bands_bind_saved_evidence_to_explicit_settings(self):
        report = accepted_joint_report()
        training = report['candidates']['local_shift']['parameters']['training_objective']
        training['fit_bands'].append(dict(region=[1.95, 1.96], channel_indices=list(range(50, 61)),
                                          channel_count=11, energy_span=.01))
        training['score_channel_indices'].extend(range(50, 61))
        training['bandwidth_check']['total_score_channel_count'] = 32
        settings = dict(derivative_window=21, derivative_polyorder=2,
                        train_regions=[band['region'][:] for band in training['fit_bands']])
        self.assertTrue(joint_validation_passed(report, settings))
        settings['train_regions'][1] = [1.96, 1.97]
        self.assertFalse(joint_validation_passed(report, settings))
        training['fit_bands'][1]['channel_indices'][0] = 20
        self.assertFalse(joint_validation_passed(report))

    def test_refinement_evidence_is_required_before_publication(self):
        for mutation in ('legacy', 'missing', 'failed', 'not_evaluated', 'policy', 'widths',
                         'offsets', 'seeds', 'bounds', 'missing_probe', 'fake_count', 'coordinates',
                         'source', 'error', 'refit_source', 'refit_coordinate', 'selector_source',
                         'omitted_worst'):
            with self.subTest(mutation=mutation):
                payload = self.payload()
                report = payload['report']
                audit = report['joint_validation']['influence_audit']
                refinement = audit['refinement']
                if mutation == 'legacy':
                    audit.update(schema_version=2, policy='training_feature_influence_shrinkage_v1')
                elif mutation == 'missing':
                    audit.pop('refinement')
                elif mutation == 'failed':
                    refinement['passed'] = False
                elif mutation == 'not_evaluated':
                    refinement['evaluated'] = False
                elif mutation == 'policy':
                    refinement['policy'] = 'coarse_only'
                elif mutation == 'widths':
                    refinement['sigma_native_spacings'] = [1.5, 3.]
                elif mutation == 'offsets':
                    refinement['center_offsets_native_spacings'] = [0.]
                elif mutation == 'seeds':
                    refinement['seed_center_indices'] = [1, 2, 3]
                elif mutation == 'bounds':
                    refinement['center_bounds_energy'] = [2., 1.]
                elif mutation == 'missing_probe':
                    refinement['probes'].pop()
                    refinement['probe_count'] -= 1
                    refinement['expected_probe_count'] -= 1
                elif mutation == 'fake_count':
                    refinement['expected_probe_count'] = 0
                elif mutation == 'coordinates':
                    refinement['probes'][0]['sigma_energy'] *= 1.01
                elif mutation == 'source':
                    refinement['probes'][0]['probe_source'] = 'coarse'
                elif mutation == 'error':
                    refinement['probes'][0]['d2_error_fraction'] = .051
                elif mutation == 'refit_source':
                    audit['actual_refits']['chosen_probe_references'][1]['source'] = 'coarse'
                elif mutation == 'refit_coordinate':
                    audit['actual_refits']['probes'][1]['center_energy'] += .001
                elif mutation == 'selector_source':
                    report['joint_validation']['selection_sensitivity_audit']['chosen_probe_references'][1]['source'] = 'coarse'
                else:
                    refinement['probes'][-1]['d2_error_fraction'] = .02
                    refinement['max_d2_error_fraction'] = .02
                from copy import deepcopy
                report['candidates']['local_shift']['training_influence_audit'] = deepcopy(audit)
                self.assertFalse(joint_validation_passed(report))
                with self.assertRaisesRegex(ValueError, 'joint DR/R and D2E'):
                    self.store.save_result(payload)
        self.assertFalse(self.store.directory.exists())

    def test_d2_training_evidence_requires_bn_basis_and_actual_read_scope(self):
        for mutation in ('top_level', 'sample_basis', 'rank', 'context', 'training_repeat',
                         'channels', 'score_channels', 'sg', 'train_selection_overlap'):
            with self.subTest(mutation=mutation):
                payload = self.payload()
                report = payload['report']
                training = report['candidates']['local_shift']['parameters']['training_objective']
                if mutation == 'top_level':
                    report['joint_validation']['d2_training_evaluated'] = False
                elif mutation == 'sample_basis':
                    training['fixed_bn_basis'] = False
                elif mutation == 'rank':
                    training['basis_rank'] = 0
                elif mutation == 'context':
                    training['context'] = 'zero extension proxy'
                elif mutation == 'training_repeat':
                    training['repeat_indices'] = [2]
                elif mutation == 'channels':
                    training['channel_indices'] = [1, 2, 3]
                elif mutation == 'score_channels':
                    training['score_channel_indices'] = [999]
                elif mutation == 'sg':
                    training['used_window_length'] = 101
                else:
                    report['split']['selection'] = [0]
                with self.assertRaisesRegex(ValueError, 'joint DR/R and D2E'):
                    self.store.save_result(payload)

    def test_two_parameter_family_requires_two_resolved_bn_responses(self):
        report = accepted_joint_report(method='local_shift_gain')
        evidence = report['candidates']['local_shift_gain']['parameters']['training_objective']
        self.assertTrue(joint_validation_passed(report))
        evidence['basis_rank'] = 1
        self.assertFalse(joint_validation_passed(report))

    def test_training_bandwidth_failure_cannot_be_activated(self):
        for mutation in ('missing', 'failed', 'span', 'count', 'window'):
            with self.subTest(mutation=mutation):
                report = self.payload()['report']
                evidence = report['candidates']['local_shift']['parameters']['training_objective']
                bandwidth = evidence['bandwidth_check']
                if mutation == 'missing':
                    evidence.pop('bandwidth_check')
                elif mutation == 'failed':
                    bandwidth['passed'] = False
                elif mutation == 'span':
                    bandwidth['sg_footprint_energy'] = 2*bandwidth['scoring_energy_span']
                elif mutation == 'count':
                    bandwidth['score_channel_count'] = 1
                else:
                    bandwidth['required_channel_count'] = 101
                self.assertFalse(joint_validation_passed(report))

    def test_missing_or_failed_feature_influence_evidence_cannot_activate(self):
        for mutation in ('missing', 'protocol', 'heldout', 'method', 'no_real_refit',
                         'unchecked_probe', 'drr_damage', 'd2_damage', 'relaxed_tolerance'):
            with self.subTest(mutation=mutation):
                report = self.payload()['report']
                audit = report['joint_validation']['influence_audit']
                if mutation == 'missing':
                    report['joint_validation'].pop('influence_audit')
                elif mutation == 'protocol':
                    audit['policy'] = 'unqualified'
                elif mutation == 'heldout':
                    audit['train_repeat_indices'] = [2]
                elif mutation == 'method':
                    audit['selected_method'] = 'local_shift_gain'
                elif mutation == 'no_real_refit':
                    audit['actual_refits']['count'] = 0
                elif mutation == 'unchecked_probe':
                    audit['screening']['probes'].pop()
                elif mutation == 'drr_damage':
                    audit['actual_refits']['max_drr_error_fraction'] = .051
                elif mutation == 'd2_damage':
                    audit['screening']['max_d2_error_fraction'] = .051
                else:
                    audit['max_relative_error'] = .1
                self.assertFalse(joint_validation_passed(report))

    def test_saved_legacy_acceptance_is_preview_only_without_rewriting_history(self):
        payload = self.payload(eligible=False)
        payload['report'] = {'schema_version': 1}
        version = self.store.save_result(payload)
        doc = json.loads(version.report_path.read_text(encoding='utf-8'))
        doc.update(eligible=True, status='accepted')
        version.report_path.write_text(json.dumps(doc), encoding='utf-8')
        manifest = self.store._read()
        manifest['versions'][0]['report_sha256'] = file_digest(version.report_path)
        self.store._atomic_json(self.store.manifest_path, manifest)
        before = version.report_path.read_bytes()
        restored = self.store.get(version.version_id)
        self.assertFalse(restored.eligible)
        self.assertEqual(restored.status, 'needs_joint_validation')
        preview = self.store.load_result(restored)
        self.assertFalse(preview['eligible'])
        self.assertIn('D2E-priority joint DR/R and D2E validation required', preview['summary'])
        with self.assertRaisesRegex(ValueError, 'validated'):
            self.store.activate(restored)
        self.assertEqual(version.report_path.read_bytes(), before)

    def test_inconsistent_derivative_settings_cannot_publish_acceptance(self):
        payload = self.payload()
        payload['settings']['derivative_window'] = 101
        with self.assertRaisesRegex(ValueError, 'joint DR/R and D2E'):
            self.store.save_result(payload)

    def test_forged_top_level_pass_cannot_replace_protection_or_disjoint_holdout(self):
        for missing in ('protection', 'final', 'selection', 'derivative_checked'):
            with self.subTest(missing=missing):
                payload = self.payload()
                payload['report']['joint_validation'].pop(missing)
                with self.assertRaisesRegex(ValueError, 'joint DR/R and D2E'):
                    self.store.save_result(payload)
        payload = self.payload()
        payload['report']['split']['train'] = [0, 2]
        with self.assertRaisesRegex(ValueError, 'joint DR/R and D2E'):
            self.store.save_result(payload)

    def test_duplicate_content_cannot_publish_as_eligible(self):
        (self.root/self.measurements[1]).write_bytes((self.root/self.measurements[0]).read_bytes())
        with self.assertRaisesRegex(ValueError,'Duplicate measurement contents'):
            self.store.save_result(self.payload())
        self.assertFalse(self.store.directory.exists())

    def test_legacy_eligible_duplicate_content_is_rejected_on_verification(self):
        (self.root/self.measurements[1]).write_bytes((self.root/self.measurements[0]).read_bytes())
        version=self.store.save_result(self.payload(eligible=False))
        doc=json.loads(version.report_path.read_text(encoding='utf-8'))
        doc.update(eligible=True,status='accepted')
        version.report_path.write_text(json.dumps(doc),encoding='utf-8')
        manifest=json.loads(self.store.manifest_path.read_text(encoding='utf-8'))
        manifest['versions'][0]['report_sha256']=file_digest(version.report_path)
        self.store.manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Duplicate measurement contents'):
            self.store.get(version.version_id)

    def test_exact_group_order_independent_subset_and_superset_rejected(self):
        version=self.store.save_result(self.payload());self.store.activate(version)
        self.assertEqual(self.store.find_applied(self.measurements[::-1]).version_id,version.version_id)
        validate_bn_background_usage(self.root,self.measurements[::-1],[str(version.csv_path)])
        self.assertIsNone(self.store.find_applied(self.measurements[:1]))
        for other in (self.measurements[:1],[*self.measurements,'another.csv']):
            with self.assertRaisesRegex(ValueError,'exact measurement group'):
                validate_bn_background_usage(self.root,other,[str(version.csv_path)])

    def test_same_size_same_mtime_changed_bytes_are_stale(self):
        version=self.store.save_result(self.payload());self.store.activate(version)
        path=self.root/self.measurements[0];stat=path.stat()
        path.write_bytes(path.read_bytes().replace(b'110.0',b'111.0'))
        os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        self.assertFalse(self.store.get(version.version_id).fresh)
        with self.assertRaisesRegex(ValueError,'stale'):
            self.store.find_applied(self.measurements)

    def test_changed_bn_blocks_saved_candidate(self):
        version=self.store.save_result(self.payload());self.store.activate(version)
        path=self.root/self.background;path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'stale'):
            validate_bn_background_usage(self.root,self.measurements,[version.csv_path])

    def test_exploratory_candidate_cannot_activate_or_load(self):
        version=self.store.save_result(self.payload(eligible=False))
        with self.assertRaisesRegex(ValueError,'validated'):
            self.store.activate(version)
        with self.assertRaisesRegex(ValueError,'exploratory'):
            validate_bn_background_usage(self.root,self.measurements,[version.csv_path])

    def test_candidate_and_report_tampering_detected(self):
        for suffix in ('csv','metadata'):
            with self.subTest(suffix=suffix):
                version=self.store.save_result(self.payload())
                path=version.csv_path if suffix=='csv' else version.report_path
                path.write_bytes(path.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError,'modified'):
                    self.store.get(version.version_id)

    def test_copied_csv_cannot_escape_guard(self):
        version=self.store.save_result(self.payload());self.store.activate(version)
        copy=self.root/'copied_background.csv';copy.write_bytes(version.csv_path.read_bytes())
        with self.assertRaisesRegex(ValueError,'verified version'):
            validate_bn_background_usage(self.root,self.measurements,[copy])

    def test_original_choice_deactivates_only_exact_group(self):
        version=self.store.save_result(self.payload());self.store.activate(version)
        self.store.deactivate(self.measurements,[self.background],'all')
        self.assertEqual(self.store.selection_state(self.measurements),'original')
        self.assertTrue(self.store.has_original_choice(self.measurements,[self.background],'all'))
        self.assertFalse(self.store.has_original_choice(self.measurements,[self.background],'first'))
        self.assertIsNone(self.store.selection_state(self.measurements[:1]))
        self.assertIsNone(self.store.find_applied(self.measurements))

    def test_automatic_fallback_clears_activation_without_an_original_choice(self):
        version = self.store.save_result(self.payload())
        self.store.activate(version)
        self.assertTrue(self.store.clear_activation(self.measurements))
        self.assertIsNone(self.store.selection_state(self.measurements))
        self.assertFalse(self.store.has_original_choice(self.measurements))
        self.store.deactivate(self.measurements, [self.background], 'all', restore_original=True)
        self.assertFalse(self.store.clear_activation(self.measurements))
        self.assertTrue(self.store.has_original_choice(self.measurements))
        self.assertIsNotNone(self.store.original_recipe(self.measurements))

    def test_verified_activation_and_deactivation_do_no_spectrum_reads(self):
        version=self.store.save_result(self.payload())
        with patch('core.drr_bn_store.file_digest',side_effect=AssertionError('GUI hashed a spectrum')):
            self.store.activate_verified(version,measurements=self.measurements)
            self.store.deactivate(self.measurements,[self.background],'all')

    def test_source_changed_after_verification_still_rejected_at_load(self):
        version=self.store.save_result(self.payload())
        path=self.root/self.background;path.write_bytes(path.read_bytes()+b'\n')
        self.store.activate_verified(version,measurements=self.measurements)
        with self.assertRaisesRegex(ValueError,'stale'):
            validate_bn_background_usage(self.root,self.measurements,[version.csv_path])

    def test_failed_publication_has_no_candidate_or_association(self):
        with patch.object(self.store,'_atomic_json',side_effect=OSError('disk error')):
            with self.assertRaisesRegex(OSError,'disk error'):
                self.store.save_result(self.payload())
        self.assertEqual(list(self.store.directory.glob('*.csv')),[])
        self.assertIsNone(self.store.find_applied(self.measurements))


if __name__=='__main__': unittest.main()
