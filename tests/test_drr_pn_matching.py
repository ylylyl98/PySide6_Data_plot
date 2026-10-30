"""Existing P/N references stay independent through matching and review."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from core.drr_composite_background import (
    CompositeBackground, CompositeSource, load_composite_source, save_composite_background,
    read_composite_background,
)
from core.drr_pn_matching import (
    build_pn_match_plan, override_pn_match, plan_is_current,
    prepare_pn_confirmation, confirmation_is_current,
)
from core.load_cancellation import LoadCancelled


class PnMatchingTests(unittest.TestCase):
    prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_'

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.measurement = self.write('TG_rep01', 1000, varying=True)

    def write(self, token, stamp, *, gate=-10, varying=False, prefix=None, columns='740,760,780'):
        name = (self.prefix if prefix is None else prefix) + token + '.csv'
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'Vbg,Vtg,{columns}\n{gate},{gate},10,11,12\n'
                        f'{gate+1 if varying else gate},{gate},11,12,13\n')
        os.utime(path, (stamp, stamp))
        return name

    def composite(self, token, p, n, *, stamp=5000, draft=False):
        first = load_composite_source(self.root, (p,), 'all', key='P')
        second = load_composite_source(self.root, (n,), 'all', key='N')
        generated = CompositeSource('generated', 'Saved P/N', first.files + second.files,
            'all', first.energy, first.spectral_columns, (first.intensity + second.intensity) / 2,
            source_hashes=first.source_hashes + second.source_hashes,
            generation_json=json.dumps({'scientifically_validated': not draft}))
        path = self.root / 'Processed Data' / 'DRR' / 'PN Backgrounds' / (token + '.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        save_composite_background(path, CompositeBackground((first, second, generated), 'generated'))
        os.utime(path, (stamp, stamp))
        return path.relative_to(self.root).as_posix()

    def plan(self, kind='p', selected=None, **kwargs):
        return build_pn_match_plan(self.root, selected or (self.measurement,), kind=kind, **kwargs)

    def test_raw_assignments_choose_nearest_for_each_measurement_without_pooling(self):
        early = self.write('Pbackground_001', 900)
        late = self.write('Pbackground_002', 1900)
        second = self.write('TG_rep02', 2000, varying=True)
        plan = self.plan(selected=(self.measurement, second))
        self.assertTrue(plan.ready)
        self.assertEqual([row.baseline_file for row in plan.rows], [early, late])
        self.assertEqual(plan.resolution.numerical_path, 'heterogeneous')
        self.assertTrue(all(len(item.baseline_files) == 1 for item in plan.resolution.assignments))

    def test_verified_run_conflict_preserves_choices_and_rechecks_override(self):
        early = self.write('Pbackground_001', 900)
        late = self.write('Pbackground_002', 1900)
        second = self.write('TG_rep02', 2000, varying=True)
        times = {str(self.root / name).casefold(): stamp for name, stamp in
                 ((self.measurement, 1000), (second, 2000), (early, 900), (late, 1900))}
        clock = SimpleNamespace(capture_context=lambda source: SimpleNamespace(
            verified_run=True, run_id='run', experiment_id='experiment'))
        with patch('core.drr_pn_matching._timing', return_value=(times, 'capture_log', '', clock)):
            plan = self.plan(selected=(self.measurement, second))
        self.assertEqual([row.baseline_file for row in plan.rows], [early, late])
        self.assertEqual(plan.batches[0][1], 'conflict')
        self.assertTrue(all(any('Verified capture batch:' in w for w in row.warnings) for row in plan.rows))
        changed = override_pn_match(plan, second, early)
        self.assertEqual(changed.batches[0][1], 'conflict')  # joint timing is tied
        self.assertTrue(changed.ready)

    def test_role_grid_and_known_condition_conflicts_are_excluded(self):
        valid = self.write('Pbackground_good', 800)
        opposite = self.write('Nbackground', 1000, gate=10)
        wrong_role_gate = self.write('Pbackground_wrong_gate', 1000, gate=10)
        wrong_grid = self.write('Pbackground_wrong_grid', 1000, columns='740,760,790')
        wrong_condition = self.write('Pbackground_wrong_point', 1000, prefix=self.prefix.replace('p2n1', 'p3n1'))
        variable = self.write('Pbackground_varying', 1000, varying=True)
        plan = self.plan(candidate_files=(valid, opposite, wrong_role_gate, wrong_grid, wrong_condition, variable))
        self.assertTrue(plan.ready)
        self.assertEqual(plan.rows[0].candidate_files, (valid,))

    def test_n_role_and_frame_method_are_preserved(self):
        background = self.write('Nbackground', 900, gate=10)
        plan = self.plan('n', which='first')
        self.assertEqual(plan.rows[0].baseline_file, background)
        self.assertEqual(plan.rows[0].assignment.baseline_which, 'first')
        self.assertTrue(plan.rows[0].assignment.selection_reason.startswith('P/N match (n): '))

    def test_automatic_raw_inventory_does_not_warn_about_other_background_types(self):
        valid = self.write('Pbackground', 900)
        self.write('Nbackground', 950, gate=10)
        self.write('BNbackground', 960, gate=0)
        plan = self.plan()
        self.assertEqual(plan.candidate_files, (valid,))
        self.assertFalse(plan.inventory_warnings)

    def test_equal_evidence_remains_unresolved_then_manual_override_resolves(self):
        left = self.write('Pbackground_left', 900)
        right = self.write('Pbackground_right', 1100)
        plan = self.plan()
        self.assertFalse(plan.ready)
        self.assertEqual(set(plan.rows[0].candidate_files), {left, right})
        chosen = override_pn_match(plan, (self.measurement,), right)
        self.assertTrue(chosen.ready)
        self.assertEqual(chosen.rows[0].baseline_file, right)
        self.assertIn('Manual assignment', chosen.rows[0].evidence)
        self.assertFalse(plan.ready)

    def test_manual_override_cannot_assign_an_incompatible_candidate(self):
        self.write('Pbackground_good', 900)
        bad = self.write('Nbackground', 950, gate=10)
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            override_pn_match(plan, self.measurement, bad)

    def test_explicit_empty_pool_does_not_fall_back_to_discovery(self):
        self.write('Pbackground', 900)
        plan = self.plan(candidate_files=())
        self.assertFalse(plan.ready)
        self.assertEqual(plan.candidate_files, ())

    def test_selected_raw_pool_never_uses_a_closer_unselected_reference(self):
        for kind, gate in (('p', -10), ('n', 10)):
            with self.subTest(kind=kind):
                selected = self.write(kind.upper() + 'background_selected', 800, gate=gate)
                unselected = self.write(kind.upper() + 'background_unselected', 999, gate=gate)
                with patch('core.drr_pn_matching.drr_source_paths',
                           side_effect=AssertionError('An explicit pool must not discover other references.')):
                    plan = self.plan(kind, candidate_files=(selected,))
                    self.assertTrue(plan_is_current(plan))
                self.assertTrue(plan.ready)
                self.assertFalse(plan.automatic_candidates)
                self.assertEqual(plan.candidate_files, (selected,))
                self.assertEqual(plan.rows[0].candidate_files, (selected,))
                self.assertEqual(plan.rows[0].baseline_file, selected)
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    override_pn_match(plan, self.measurement, unselected)

    def test_selected_composite_pool_never_uses_a_closer_unselected_version(self):
        chosen_p = self.write('Pbackground_selected', 800)
        chosen_n = self.write('Nbackground_selected', 820, gate=10)
        other_p = self.write('Pbackground_unselected', 990)
        other_n = self.write('Nbackground_unselected', 995, gate=10)
        chosen = self.composite('selected_version', chosen_p, chosen_n)
        unselected = self.composite('unselected_version', other_p, other_n)
        with patch('core.drr_pn_matching.drr_source_paths',
                   side_effect=AssertionError('An explicit pool must not discover other versions.')):
            plan = self.plan('pn', candidate_files=(chosen,))
            self.assertTrue(plan_is_current(plan))
        self.assertTrue(plan.ready)
        self.assertEqual(plan.candidate_files, (chosen,))
        self.assertEqual(plan.rows[0].candidate_files, (chosen,))
        self.assertEqual(plan.rows[0].baseline_file, chosen)
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            override_pn_match(plan, self.measurement, unselected)

    def test_capture_times_take_precedence_over_file_modification_times(self):
        first = self.write('Pbackground_first', 999)
        second = self.write('Pbackground_second', 3000)
        events = [dict(event='capture_started',
                       started_utc=datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
                       output={'file': name})
                  for name, stamp in ((self.measurement, 1000), (first, 500), (second, 990))]
        (self.root / 'run.experiment.events.jsonl').write_text('\n'.join(map(json.dumps, events)))
        plan = self.plan()
        self.assertEqual(plan.rows[0].baseline_file, second)
        self.assertEqual(plan.rows[0].timing_basis, 'capture_started_utc')

    def test_real_capture_runs_group_p_and_n_and_keep_conflicting_choices(self):
        second = self.write('TG_rep02', 1001, varying=True)
        for kind, gate in (('p', -10), ('n', 10)):
            with self.subTest(kind=kind):
                early = self.write(kind.upper() + 'background_early', 1999, gate=gate)
                late = self.write(kind.upper() + 'background_late', 999, gate=gate)
                events = [dict(event='capture_started', output={'file': name},
                    experiment_id='experiment-A', run_id=run,
                    started_utc=datetime.fromtimestamp(stamp, timezone.utc).isoformat())
                    for name, stamp, run in ((self.measurement, 1000, 'run-A'),
                        (second, 2000, 'run-A'), (early, 900, 'reference'), (late, 1900, 'reference'))]
                (self.root / 'run.experiment.events.jsonl').write_text('\n'.join(map(json.dumps, events)))
                plan = self.plan(kind, selected=(self.measurement, second), candidate_files=(early, late))
                self.assertTrue(plan.ready)
                self.assertEqual([row.baseline_file for row in plan.rows], [early, late])
                self.assertEqual([row.timing_basis for row in plan.rows], ['capture_started_utc'] * 2)
                self.assertEqual(plan.batches[0][0], (self.measurement, second))
                self.assertEqual(plan.batches[0][1], 'conflict')
                self.assertTrue(all(any('Verified capture batch:' in w for w in row.warnings) for row in plan.rows))

    def test_unverified_and_incompatible_measurements_do_not_form_batch(self):
        second = self.write('TG_rep02', 1010, varying=True)
        other = self.write('TG_other', 1020, varying=True, prefix=self.prefix.replace('p2n1', 'p3n1'))
        grid = self.write('TG_grid', 1030, varying=True, columns='740,760,781')
        background = self.write('Pbackground', 900)
        names = (self.measurement, second, other, grid, background)
        def write_log(verified):
            events = [dict(event='capture_started', output={'file': name}, experiment_id='experiment-A',
                started_utc=datetime.fromtimestamp(1000 + i * 10, timezone.utc).isoformat(),
                **({'run_id': 'run-A'} if verified else {})) for i, name in enumerate(names)]
            (self.root / 'run.experiment.events.jsonl').write_text('\n'.join(map(json.dumps, events)))
        write_log(False)
        plan = self.plan(selected=names[:-1], candidate_files=(background,))
        self.assertEqual(plan.batches, ())
        write_log(True)
        plan = self.plan(selected=names[:-1], candidate_files=(background,))
        self.assertEqual(len(plan.batches), 1)
        self.assertEqual(plan.batches[0][0], (self.measurement, second))
        self.assertEqual([row.measurement_file for row in plan.rows], list(names[:-1]))

    def test_bad_capture_log_for_incompatible_candidate_does_not_block_match(self):
        valid = self.write('Pbackground_good', 900)
        incompatible = self.write('Pbackground_wrong_point', 1000, prefix=self.prefix.replace('p2n1', 'p3n1'))
        event = dict(event='capture_started', started_utc='bad timestamp', output={'file': incompatible})
        (self.root / 'run.experiment.events.jsonl').write_text(json.dumps(event))
        plan = self.plan()
        self.assertTrue(plan.ready)
        self.assertEqual(plan.rows[0].baseline_file, valid)

    def test_composite_uses_verified_raw_times_instead_of_generated_file_time(self):
        p1, n1 = self.write('Pbackground_1', 900), self.write('Nbackground_1', 920, gate=10)
        p2, n2 = self.write('Pbackground_2', 3000), self.write('Nbackground_2', 3020, gate=10)
        close = self.composite('near_sources', p1, n1, stamp=10000)
        self.composite('near_generated_timestamp', p2, n2, stamp=1000)
        plan = self.plan('pn')
        self.assertTrue(plan.ready)
        self.assertEqual(plan.rows[0].baseline_file, close)
        self.assertEqual(plan.rows[0].time_gap_seconds, 100)

    def test_missing_originals_do_not_block_verified_frozen_composite(self):
        p, n = self.write('Pbackground', 900), self.write('Nbackground', 920, gate=10)
        composite = self.composite('saved', p, n)
        (self.root / p).unlink()
        (self.root / n).unlink()
        plan = self.plan('pn')
        self.assertTrue(plan.ready)
        self.assertEqual(plan.rows[0].baseline_file, composite)
        self.assertEqual(plan.rows[0].timing_basis, 'unavailable')
        self.assertTrue(any('frozen' in warning for warning in plan.warnings))
        self.assertTrue(plan_is_current(plan))

    def test_two_frozen_candidates_without_original_timing_require_manual_choice(self):
        p, n = self.write('Pbackground', 900), self.write('Nbackground', 920, gate=10)
        self.composite('older_save', p, n, stamp=100)
        newer = self.composite('newer_save', p, n, stamp=2000)
        (self.root / p).unlink()
        (self.root / n).unlink()
        plan = self.plan('pn')
        self.assertFalse(plan.ready)
        self.assertTrue(override_pn_match(plan, self.measurement, newer).ready)

    def test_composite_metadata_tamper_is_not_eligible(self):
        p, n = self.write('Pbackground', 900), self.write('Nbackground', 920, gate=10)
        name = self.composite('saved', p, n)
        (self.root / name).with_suffix('.metadata.json').write_text('{}')
        self.assertFalse(self.plan('pn').ready)

    def test_composite_sidecar_change_during_matching_invalidates_plan(self):
        p, n = self.write('Pbackground', 900), self.write('Nbackground', 920, gate=10)
        self.composite('saved', p, n)
        def changing_reader(path):
            result = read_composite_background(path)
            with Path(path).with_suffix('.metadata.json').open('a') as stream:
                stream.write(' ')
            return result
        with patch('core.drr_pn_matching.read_composite_background', side_effect=changing_reader):
            plan = self.plan('pn')
        self.assertFalse(plan.ready)
        self.assertIn('changed during matching', plan.rows[0].evidence)

    def test_reconstruction_training_records_are_not_background_acquisition_times(self):
        p, n = self.write('Pbackground', 900), self.write('Nbackground', 920, gate=10)
        training = self.write('TG_training', 90000, varying=True)
        name = self.composite('reconstructed', p, n, draft=True)
        document = read_composite_background(self.root / name)
        generated = document.sources[-1]
        report = {'algorithm_version': 'pn_raw_conditioned_bounded_v1',
                  'scientifically_validated': False, 'input_source_keys': ['P', 'N'],
                  'input_records': [{'role': 'P', 'path': str(self.root / p)},
                                    {'role': 'N', 'path': str(self.root / n)},
                                    {'role': 'training', 'path': str(self.root / training)}]}
        generated = replace(generated, generation_json=json.dumps(report))
        save_composite_background(self.root / name,
            CompositeBackground((*document.sources[:-1], generated), generated.key), overwrite=True)
        plan = self.plan('pn')
        self.assertTrue(plan.ready)
        self.assertEqual(plan.rows[0].time_gap_seconds, 100)
        self.assertTrue(any('experimental draft' in warning for warning in plan.warnings))

    def test_confirmation_requires_warning_review_and_detects_changed_file(self):
        background = self.write('Pbackground', 900)
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, 'acknowledge'):
            prepare_pn_confirmation(plan)
        prepared = prepare_pn_confirmation(plan, warnings_acknowledged=True)
        self.assertTrue(confirmation_is_current(prepared))
        with (self.root / background).open('a') as stream:
            stream.write('\n')
        self.assertFalse(confirmation_is_current(prepared))
        self.assertFalse(plan_is_current(plan))

    def test_new_automatic_candidate_invalidates_confirmation(self):
        self.write('Pbackground_1', 900)
        prepared = prepare_pn_confirmation(self.plan(), warnings_acknowledged=True)
        self.write('Pbackground_2', 990)
        self.assertFalse(confirmation_is_current(prepared))

    def test_saving_export_package_does_not_invalidate_input_confirmation(self):
        output = self.root / 'Processed Data' / 'DRR'
        output.mkdir(parents=True)
        self.write('Pbackground', 900)
        prepared = prepare_pn_confirmation(self.plan(), warnings_acknowledged=True)
        (output / 'new-export-package').mkdir()
        self.assertTrue(confirmation_is_current(prepared))

    def test_changed_source_or_sidecar_invalidates_composite_confirmation(self):
        p, n = self.write('Pbackground', 900), self.write('Nbackground', 920, gate=10)
        name = self.composite('saved', p, n)
        prepared = prepare_pn_confirmation(self.plan('pn'), warnings_acknowledged=True)
        with (self.root / name).with_suffix('.metadata.json').open('a') as stream:
            stream.write(' ')
        self.assertFalse(confirmation_is_current(prepared))

    def test_explicit_external_raw_background_is_allowed(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        external = Path(temporary.name) / (self.prefix + 'Pbackground.csv')
        external.write_text('Vbg,Vtg,740,760,780\n-10,-10,10,11,12\n-10,-10,11,12,13\n')
        plan = self.plan(candidate_files=(str(external),))
        self.assertTrue(plan.ready)
        self.assertEqual(Path(plan.rows[0].baseline_file), external.resolve())

    def test_cancelled_worker_does_not_return_an_applicable_plan(self):
        stopped = Event()
        stopped.set()
        with self.assertRaises(LoadCancelled):
            self.plan(cancelled=stopped)

    def test_duplicate_measurements_and_forged_rows_are_not_confirmed(self):
        self.write('Pbackground', 900)
        self.assertFalse(self.plan(selected=(self.measurement, self.measurement)).ready)
        plan = self.plan()
        forged = replace(plan, rows=(replace(plan.rows[0], evidence='changed'),))
        with self.assertRaises(ValueError):
            prepare_pn_confirmation(forged, warnings_acknowledged=True)


if __name__ == '__main__':
    unittest.main()
