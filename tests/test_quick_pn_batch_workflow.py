"""Exact, transactional Quick P/N batch generation and application."""
from dataclasses import replace
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QSettings
from core.drr_pn_batch_plan import build_pn_batch_plan
from core.drr_pn_reconstruction import pipeline
from core.drr_pn_store import DrrPnStore
from ui_qt.quick.drr_background_workflow import BackgroundContext, commit_application, load_with_recipe
from ui_qt.quick.drr_pn_batch_workflow import generate_pn_batch, prepare_pn_batch_application, require_pn_batch_current


class PnBatchWorkflowTests(unittest.TestCase):
    prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_'
    def setUp(self):
        temp = TemporaryDirectory(prefix='pn-batch-workflow-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.measurements = (self.write('TG_rep01', 1000), self.write('TG_rep02', 2000))
        self.p = self.write('Pbackground_001', 900, gate=-10)
        self.n = self.write('Nbackground_001', 905, gate=10)
        self.context = BackgroundContext(str(self.root), self.measurements)
        self.settings = QSettings(str(self.root/'settings.ini'), QSettings.IniFormat)
        self.plan = build_pn_batch_plan(self.root, self.measurements, p_candidates=(self.p,), n_candidates=(self.n,))
        self.assertTrue(self.plan.ready)

    def write(self, token, stamp, gate=None):
        path = self.root/(self.prefix + token + '.csv')
        wave = np.linspace(630., 690., 41)
        gates = np.linspace(-2, 2, 9) if gate is None else np.full(4, gate)
        values = 150 + np.linspace(-1, 1, len(wave))[None, :]**2 + np.zeros((len(gates), len(wave)))
        if gate is None:
            values += 5*gates[:, None]
        np.savetxt(path, np.column_stack((gates, gates, values)), delimiter=',',
            header='Vbg,Vtg,' + ','.join(map(str, wave)), comments='')
        os.utime(path, (stamp, stamp))
        return path.name

    @staticmethod
    def stub(energy, pn_mean, frames, samples, **kwargs):
        if kwargs['method'] == 'conditioned_single':
            algorithm, protocol = pipeline.CONDITIONED_SINGLE_ALGORITHM_VERSION, pipeline.CONDITIONED_SINGLE_PROTOCOL
            assert samples.shape[0] == 1
        else:
            algorithm, protocol = pipeline.CONDITIONED_BOUNDED_ALGORITHM_VERSION, pipeline.CONDITIONED_BOUNDED_PROTOCOL
            assert samples.shape[0] == 2
        return {'energy': energy.copy(), 'background': pn_mean.mean(0)}, {
            'algorithm_version': algorithm, 'settings': dict(protocol), 'BN_used': False}

    def generate(self):
        with patch.object(pipeline, 'reconstruct_background', side_effect=self.stub):
            return generate_pn_batch(self.context, self.plan)

    def test_exact_assignments_generated_and_applied_without_associations(self):
        result = self.generate()
        self.assertEqual(len(result.versions), 2)
        self.assertEqual(tuple(a.measurement_file for a in result.assignments), self.measurements)
        self.assertNotEqual(result.assignments[0].baseline_files, result.assignments[1].baseline_files)
        store = DrrPnStore(self.root)
        self.assertIsNone(store.find_applied(self.measurements))
        with self.assertRaisesRegex(ValueError, 'acknowledg'):
            prepare_pn_batch_application(self.context, result)
        prepared = prepare_pn_batch_application(self.context, result, warnings_acknowledged=True)
        load_with_recipe(self.context, prepared.recipe, prepared=prepared)
        with patch('core.drr_pn_store._file_digest', side_effect=AssertionError('GUI must not hash')), \
                patch('ui_qt.quick.drr_pn_batch_workflow.pn_batch_plan_is_current', side_effect=AssertionError('GUI must not rematch')), \
                patch('core.drr_pn_store.DrrPnStore.get_version', side_effect=AssertionError('GUI must not load versions')):
            recipe = commit_application(prepared, current_context=self.context, settings=self.settings)
        self.assertEqual(recipe.assignments, result.assignments)
        self.assertIsNone(store.find_applied(self.measurements))

    def test_tamper_and_source_change_cannot_apply(self):
        result = self.generate()
        with self.assertRaises(ValueError):
            require_pn_batch_current(self.context, replace(result, assignments=tuple(reversed(result.assignments))))
        prepared = prepare_pn_batch_application(self.context, result, warnings_acknowledged=True)
        path = self.root/self.p
        stat = path.stat()
        contents = path.read_bytes()
        path.write_bytes(contents.replace(b'150', b'151', 1))
        self.assertEqual(path.stat().st_size, stat.st_size)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaises(ValueError):
            commit_application(prepared, current_context=self.context, settings=self.settings)
        with self.assertRaises(ValueError):
            prepare_pn_batch_application(self.context, result, warnings_acknowledged=True)

    def test_batch_review_export_and_exact_group_restore(self):
        from core.export import write_export_metadata
        from ui_qt.quick.drr_workflow import DrrRequest, export_drr, load_drr
        from ui_qt.quick.drr_background_restore import restore_saved_background
        result = self.generate()
        prepared = prepare_pn_batch_application(self.context, result, warnings_acknowledged=True)
        recipe = commit_application(prepared, current_context=self.context, settings=self.settings)
        request = DrrRequest(tuple(str(self.root/name) for name in self.measurements),
            experiment_folder=str(self.root), background_mode='external', background_frame='all',
            background_recipe=recipe, window=9)
        products = load_drr(request)
        captured = {}
        def capture(folder, **kwargs):
            captured.update(kwargs)
            return {}
        with patch('core.export.export_drr_png_and_dat', side_effect=capture), \
                patch('core.export._validate_drr_pair_product'):
            export_drr(products, folder=str(self.root), raw_params=None, second_params=None, view='raw')
        policy = captured['metadata_processing']['background_matching_policy']
        data = policy['pn_batch']
        self.assertEqual(data['p_candidates'], [self.p])
        self.assertEqual(data['n_candidates'], [self.n])
        self.assertEqual([row['measurement_file'] for row in data['rows']], list(self.measurements))
        self.assertTrue(all(row['p_file'] == self.p and row['n_file'] == self.n for row in data['rows']))
        self.assertEqual([s['version_id'] for s in data['segments']], [v.version_id for v in result.versions])
        output = self.root/'Processed Data'/'DRR'/'batch.dat'
        metadata = write_export_metadata(str(self.root), [output], operation='DR/R',
            input_files=[*(('measurement', name) for name in self.measurements),
                         *(('background', a.baseline_files[0]) for a in result.assignments)],
            processing=captured['metadata_processing'])
        restored = restore_saved_background(DrrRequest(request.files, experiment_folder=str(self.root), window=9))
        draft = restored.background_recipe.draft
        self.assertEqual([(a.measurement_file, a.baseline_files) for a in draft.saved_assignments], [(a.measurement_file, a.baseline_files) for a in result.assignments])
        self.assertEqual(draft.saved_matching_review['pn_batch'], data)
        self.assertIsNone(draft.batch_result)
        self.assertIsNone(draft.pn_plan)
        self.assertFalse(draft.pn_warnings_acknowledged)
        import copy
        import json
        original = json.loads(Path(metadata).read_text(encoding='utf-8'))
        mutations = (
            lambda batch: batch['rows'][0].update(manual='yes'),
            lambda batch: batch['segments'][0].update(p_file='wrong.csv'),
            lambda batch: batch['segments'][0].update(training_files=batch['segments'][0]['training_files'] * 2),
            lambda batch: batch['segments'][0].update(version_id='wrong-version'),
            lambda batch: batch['rows'][0].update(measurement_file='wrong.csv'),
        )
        for mutate in mutations:
            payload = copy.deepcopy(original)
            mutate(payload['processing']['background_matching_policy']['pn_batch'])
            Path(metadata).write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                restore_saved_background(DrrRequest(request.files, experiment_folder=str(self.root), window=9))


    def test_exact_reuse_and_cancellation(self):
        first = self.generate()
        with patch.object(pipeline, 'reconstruct_background', side_effect=AssertionError('unexpected regeneration')):
            again = generate_pn_batch(self.context, self.plan)
        self.assertEqual([v.version_id for v in first.versions], [v.version_id for v in again.versions])
        from core.load_cancellation import LoadCancelled
        with self.assertRaises(LoadCancelled):
            generate_pn_batch(self.context, self.plan, cancelled=lambda: True)

    def test_partial_failure_saves_only_draft_no_application(self):
        calls = []
        def failure(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise ValueError('synthetic generation failure')
            return self.stub(*args, **kwargs)
        with patch.object(pipeline, 'reconstruct_background', side_effect=failure):
            with self.assertRaisesRegex(ValueError, 'synthetic generation failure'):
                generate_pn_batch(self.context, self.plan)
        store = DrrPnStore(self.root)
        self.assertEqual(len(store.list_versions()), 1)
        self.assertIsNone(store.find_applied(self.measurements))


if __name__ == '__main__':
    unittest.main()
