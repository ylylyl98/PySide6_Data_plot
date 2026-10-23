"""Saved group exports restore their numerical background before rematching."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_background_regions import DrrBackgroundRegion
from core.drr_composite_background import (
    CompositeBackground, CompositeSource, composite_metadata_path,
    load_composite_source, save_composite_background,
)
from core.drr_pn_pairing import DrrPnPair
from core.drr_pn_store import DrrPnStore
from core.drr_saved_background import resolve_saved_group_background
from core.drr_sources import DrrMeasurementAssignment, portable_source_name
from core.export import write_export_metadata
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope


class SavedGroupBackgroundTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'Processed Data' / 'DRR'
        self.output.mkdir(parents=True)
        self.columns = np.linspace(700., 760., 31)
        for name, intensity in (('a.csv', 30), ('b.csv', 40), ('old.csv', 10),
                                ('latest.csv', 20), ('other.csv', 15)):
            self.csv(name, intensity)

    def csv(self, name, intensity, columns=None):
        columns = self.columns if columns is None else columns
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Vbg,Vtg,Vbias,' + ','.join(f'{value:.8f}' for value in columns) + '\n'
            + '\n'.join(f'{gate},0,0,' + ','.join([str(intensity + gate)] * len(columns))
                        for gate in (0, 1)) + '\n', encoding='utf-8')
        return path

    def export(self, name, selected, backgrounds=(), *, assignments=None, stamp=1000,
               baseline='External', which='all', derivative=None):
        processing = {'baseline_selection': baseline, 'baseline_which': which,
                      'derivative_order': derivative}
        if assignments is not None:
            processing['drr_background_assignments'] = [item.to_dict() for item in assignments]
        path = write_export_metadata(str(self.root), [self.output / f'{name}.dat'],
            operation='DR/R', input_files=[*(('measurement', item) for item in selected),
                                         *(('background', item) for item in backgrounds)],
            processing=processing)
        os.utime(path, (stamp, stamp))
        return path

    def resolve(self, selected=('a.csv',)):
        return resolve_saved_group_background(self.root, (), selected)

    def test_latest_exact_group_wins_over_older_export_and_newer_other_group(self):
        self.export('old', ('a.csv', 'b.csv'), ('old.csv',), stamp=1000)
        self.export('latest', ('b.csv', 'a.csv'), ('latest.csv',), stamp=2000)
        self.export('other-group', ('a.csv',), ('other.csv',), stamp=3000)
        resolved = self.resolve(('a.csv', 'b.csv'))
        self.assertTrue(resolved.resolved, resolved.reason)
        self.assertEqual([item.baseline_files for item in resolved.assignments],
                         [('latest.csv',), ('latest.csv',)])
        self.assertEqual([item.measurement_file for item in resolved.assignments], ['a.csv', 'b.csv'])

    def test_no_exact_export_returns_none_without_guessing_or_using_member_history(self):
        self.export('single', ('a.csv',), ('old.csv',))
        with patch('core.drr_saved_background.resolve_drr_background_assignments') as resolver:
            self.assertIsNone(self.resolve(('a.csv', 'b.csv')))
        resolver.assert_not_called()

    def test_empty_and_non_csv_selection_skip_history_io(self):
        with patch('core.drr_saved_background._read_drr_metadata') as read:
            self.assertIsNone(self.resolve(()))
            self.assertIsNone(self.resolve(('map.xlsx',)))
        read.assert_not_called()

    def test_worker_signals_and_callable_callbacks_receive_log_and_progress(self):
        self.export('latest', ('a.csv',), ('latest.csv',))

        class Signal:
            def __init__(self, values):
                self.values = values

            def emit(self, value):
                self.values.append(value)

        for signal_like in (False, True):
            with self.subTest(signal_like=signal_like):
                messages, percentages = [], []
                log = Signal(messages) if signal_like else messages.append
                progress = Signal(percentages) if signal_like else percentages.append
                resolved = resolve_saved_group_background(self.root, (), ('a.csv',),
                    log=log, progress=progress)
                self.assertTrue(resolved.resolved, resolved.reason)
                self.assertEqual(percentages, [100])
                self.assertEqual(len(messages), 1)
                self.assertIn('Restoring the latest saved DRR background', messages[0])

    def test_raw_and_derivative_records_from_same_export_agree(self):
        assignments = (DrrMeasurementAssignment('a.csv', 'External', ('latest.csv',), 'all'),)
        self.export('raw', ('a.csv',), ('latest.csv',), assignments=assignments, derivative=None)
        self.export('second', ('a.csv',), ('latest.csv',), assignments=assignments, derivative=2)
        resolved = self.resolve()
        self.assertTrue(resolved.resolved, resolved.reason)
        self.assertEqual(resolved.assignments[0].baseline_files, ('latest.csv',))
        self.assertEqual(resolved.assignments[0].baseline_which, 'all')

    def test_equal_time_conflicting_latest_records_remain_unresolved(self):
        self.export('first', ('a.csv',), ('old.csv',))
        self.export('second', ('a.csv',), ('latest.csv',))
        resolved = self.resolve()
        self.assertFalse(resolved.resolved)
        self.assertIn('conflicting saved assignments', resolved.reason)
        self.assertFalse(resolved.self_fallback_allowed)

    def test_heterogeneous_member_backgrounds_keep_their_native_grids(self):
        self.csv('b.csv', 40, self.columns + 1)
        self.csv('other.csv', 15, self.columns + 1)
        assignments = (DrrMeasurementAssignment('a.csv', 'External', ('old.csv',), 'first'),
                       DrrMeasurementAssignment('b.csv', 'External', ('other.csv',), 'last'))
        self.export('mixed', ('a.csv', 'b.csv'), ('old.csv', 'other.csv'), assignments=assignments)
        resolved = self.resolve(('a.csv', 'b.csv'))
        self.assertTrue(resolved.resolved, resolved.reason)
        self.assertEqual(resolved.numerical_path, 'heterogeneous')
        self.assertEqual([(item.baseline_files, item.baseline_which) for item in resolved.assignments],
                         [(('old.csv',), 'first'), (('other.csv',), 'last')])

    def test_missing_latest_background_does_not_restore_older_export(self):
        self.export('old', ('a.csv',), ('old.csv',), stamp=1000)
        self.export('latest', ('a.csv',), ('latest.csv',), stamp=2000)
        (self.root / 'latest.csv').unlink()
        resolved = self.resolve()
        self.assertFalse(resolved.resolved)
        self.assertIn('recorded baseline missing', resolved.reason)
        self.assertFalse(resolved.self_fallback_allowed)

    def test_saved_regions_validate_every_override_and_keep_shared_grid_requirement(self):
        region = DrrBackgroundRegion(1.65, 1.70, ('other.csv',), 'last')
        assignments = (DrrMeasurementAssignment('a.csv', 'External', ('old.csv',), 'all', regions=(region,)),
                       DrrMeasurementAssignment('b.csv', 'External', ('latest.csv',), 'last'))
        self.export('regions', ('a.csv', 'b.csv'), ('old.csv', 'other.csv', 'latest.csv'),
                    assignments=assignments)
        resolved = self.resolve(('a.csv', 'b.csv'))
        self.assertTrue(resolved.resolved, resolved.reason)
        self.assertEqual(resolved.assignments[0].regions, (region,))
        # Matching each member's baseline is insufficient for regional recipes.
        self.csv('b.csv', 40, self.columns + 1)
        self.csv('latest.csv', 20, self.columns + 1)
        resolved = self.resolve(('a.csv', 'b.csv'))
        self.assertFalse(resolved.resolved)
        self.assertIn('exact native spectral grid', resolved.reason)

    def test_malformed_latest_assignment_does_not_restore_older_or_flat_union(self):
        self.export('old', ('a.csv',), ('old.csv',), stamp=1000)
        path = self.export('latest', ('a.csv',), ('latest.csv',), stamp=2000)
        payload = json.loads(path.read_text())
        payload['processing']['drr_background_assignments'] = None
        path.write_text(json.dumps(payload), encoding='utf-8')
        os.utime(path, (2000, 2000))
        resolved = self.resolve()
        self.assertFalse(resolved.resolved)
        self.assertIn('malformed saved assignment', resolved.reason)
        self.assertFalse(resolved.self_fallback_allowed)

    def test_native_grid_mismatch_in_saved_background_is_reported(self):
        self.export('latest', ('a.csv',), ('latest.csv',))
        self.csv('latest.csv', 20, self.columns[::-1])
        resolved = self.resolve()
        self.assertFalse(resolved.resolved)
        self.assertIn('exact native spectral grid', resolved.reason)

    def test_missing_measurement_is_reported_before_applying_recipe(self):
        self.export('latest', ('a.csv',), ('latest.csv',))
        (self.root / 'a.csv').unlink()
        resolved = self.resolve()
        self.assertFalse(resolved.resolved)
        self.assertIn('source is unavailable', resolved.reason)

    def test_saved_self_first_restores_without_external_background(self):
        self.export('self', ('a.csv',), baseline='Self (first frame)', which='first')
        resolved = self.resolve()
        self.assertTrue(resolved.resolved, resolved.reason)
        self.assertEqual(resolved.assignments[0].baseline_mode, 'Self (first frame)')
        self.assertEqual(resolved.assignments[0].baseline_files, ())
        self.assertEqual(resolved.assignments[0].baseline_which, 'first')

    def composite(self):
        source = load_composite_source(self.root, ('latest.csv',), key='background')
        return save_composite_background(self.root / 'frozen.csv', CompositeBackground((source,), source.key))

    def test_broken_frozen_csv_or_metadata_is_unresolved(self):
        path = self.composite()
        metadata = composite_metadata_path(path)
        original_csv, original_metadata = path.read_bytes(), metadata.read_bytes()
        self.export('latest', ('a.csv',), ('frozen.csv',))
        for failure in ('missing', 'json', 'changed_csv', 'changed_snapshot'):
            with self.subTest(failure=failure):
                path.write_bytes(original_csv)
                metadata.write_bytes(original_metadata)
                if failure == 'missing':
                    metadata.unlink()
                elif failure == 'json':
                    metadata.write_text('{', encoding='utf-8')
                elif failure == 'changed_csv':
                    path.write_bytes(original_csv + b'\n')
                else:
                    payload = json.loads(original_metadata)
                    payload['sources'][0]['intensity'][0] += 1
                    metadata.write_text(json.dumps(payload), encoding='utf-8')
                resolved = self.resolve()
                self.assertFalse(resolved.resolved)
                self.assertIn('composite', resolved.reason.casefold())
                self.assertFalse(resolved.self_fallback_allowed)

    def test_frozen_export_restores_despite_stale_code_or_missing_original_pn_sources(self):
        self.csv('Pbackground.csv', 10)
        self.csv('Nbackground.csv', 20)
        dependency = self.root / 'algorithm.py'
        dependency.write_text('version = 1\n', encoding='utf-8')
        store = DrrPnStore(self.root)
        pair = DrrPnPair(('Pbackground.csv',), ('Nbackground.csv',))
        key = store.prepare(pair, algorithm_version='test-v1', settings={}, dependencies=('algorithm.py',))
        provenance = key.provenance
        p = load_composite_source(self.root, pair.p_files, provenance['frame_choice']['P'], key='P')
        n = load_composite_source(self.root, pair.n_files, provenance['frame_choice']['N'], key='N')
        report = {'algorithm_version': provenance['algorithm_version'], 'settings': provenance['settings'],
                  'input_source_keys': ['P', 'N']}
        generated = CompositeSource('generated', 'Saved generated result', p.files + n.files, 'all',
            p.energy, p.spectral_columns, .7 * p.intensity + .3 * n.intensity,
            source_hashes=p.source_hashes + n.source_hashes, generation_json=json.dumps(report))
        version = store.save(CompositeBackground((p, n, generated), 'generated'), key)
        store.mark_used(version.version_id, measurements=('a.csv',))
        frozen = portable_source_name(self.root, version.csv_path)
        self.export('latest', ('a.csv',), (frozen,))
        dependency.write_text('version = 2\n', encoding='utf-8')
        self.assertEqual(store.get_version(version.version_id).status, 'stale')
        with self.assertRaisesRegex(ValueError, 'stale'):
            store.find_applied(('a.csv',))
        resolved = self.resolve()
        self.assertTrue(resolved.resolved, resolved.reason)
        self.assertEqual(resolved.assignments[0].baseline_files, (frozen,))
        (self.root / 'Pbackground.csv').unlink()
        (self.root / 'Nbackground.csv').unlink()
        self.assertEqual(store.get_version(version.version_id).status, 'missing_sources')
        resolved = self.resolve()
        self.assertTrue(resolved.resolved, resolved.reason)

    def test_cancellation_is_propagated(self):
        self.export('latest', ('a.csv',), ('latest.csv',))
        cancellation = LoadCancellation()
        with self.assertRaises(LoadCancelled), load_scope(cancellation):
            cancellation.cancel()
            self.resolve()


if __name__ == '__main__':
    unittest.main()
