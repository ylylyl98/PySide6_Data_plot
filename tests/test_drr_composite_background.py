"""Frozen references retain editable snapshots, exact grids and seam provenance."""
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from core import drr_composite_background as C, loader
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope


class CompositeBackgroundTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.energy = np.linspace(1., 2., 21)
        self.write_csv('p.csv', [np.arange(21.) + 10, np.arange(21.) + 30])
        self.write_csv('n.csv', [np.arange(21.) + 50, np.arange(21.) + 70])
        self.p = C.load_composite_source(self.root, ['p.csv'], key='p', label='p background')
        self.n = C.load_composite_source(self.root, ['n.csv'], key='n', label='n background')
        self.energy = self.p.energy
        self.document = C.CompositeBackground((self.p, self.n), 'p')
        self.output = self.root / 'combined.csv'

    def write_csv(self, name, rows, *, columns=None):
        columns = columns or tuple(format(v, '.17g') for v in self.energy)
        with (self.root / name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('Vbg', 'Vtg', 'Vbias', *columns))
            for index, values in enumerate(rows):
                writer.writerow((index, 0, 0, *values))
        return self.root / name

    def region(self, lo=5, hi=13, key='n', document=None):
        return C.assign_region(document or self.document, self.energy[lo], self.energy[hi], key)

    def save(self, document=None):
        return C.save_composite_background(self.output, document or self.region())

    def metadata(self):
        return json.loads(C.composite_metadata_path(self.output).read_text(encoding='utf-8'))

    def write_metadata(self, payload):
        C.composite_metadata_path(self.output).write_text(json.dumps(payload), encoding='utf-8')

    def test_source_frame_selection_and_equal_file_weighting(self):
        self.write_csv('extra.csv', [np.full(21, 100.)])
        for which, expected in [('first', np.arange(21.) + 10),
                                ('last', np.arange(21.) + 30),
                                ('all', np.arange(21.) + 20)]:
            with self.subTest(which=which):
                source = C.load_composite_source(self.root, ['p.csv'], which)
                np.testing.assert_array_equal(source.intensity, expected)
                grouped = C.load_composite_source(self.root, ['p.csv', 'extra.csv'], which)
                np.testing.assert_array_equal(grouped.intensity, (expected + 100.) / 2)
                self.assertEqual(len(grouped.source_hashes), 2)

    def test_compose_uses_default_gaps_half_open_intervals_and_final_sample(self):
        document = self.region()
        document = self.region(17, 20, document=document)
        energy, actual, boundaries = C.compose_background(document)
        expected = self.p.intensity.copy()
        expected[5:13], expected[17:] = self.n.intensity[5:13], self.n.intensity[17:]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(energy, self.energy)
        self.assertEqual(boundaries, tuple(self.energy[[5, 13, 17]]))
        np.testing.assert_array_equal(self.document.sources[0].intensity, self.p.intensity)
        with self.assertRaises(ValueError):
            self.p.intensity[0] = 0
        with self.assertRaises(ValueError):
            self.p.energy[0] = 0

    def test_assign_splits_merges_and_removes_default_overrides(self):
        first = self.region(3, 18)
        split = self.region(7, 11, key='p', document=first)
        self.assertEqual([(r.xmin_ev, r.xmax_ev) for r in split.regions],
                         [(self.energy[3], self.energy[7]), (self.energy[11], self.energy[18])])
        restored = self.region(7, 11, document=split)
        self.assertEqual(restored.regions, first.regions)
        cleared = self.region(0, 20, key='p', document=restored)
        self.assertEqual(cleared.regions, ())
        self.assertEqual(len(first.regions), 1)
        changed = C.assign_region(replace(first, regions=()), self.energy[5], self.energy[9], 'n')
        self.assertEqual(changed.regions, self.region(5, 9).regions)

    def test_invalid_region_source_and_grid_are_rejected(self):
        for lo, hi, key in [(.9, 1.5, 'n'), (1.5, 2.1, 'n'), (1.001, 1.002, 'n'),
                            (1.5, 1.5, 'n'), (np.nan, 2, 'n'), (True, 2, 'n'), (1.2, 1.5, 'missing')]:
            with self.subTest(lo=lo, hi=hi, key=key), self.assertRaises(ValueError):
                C.assign_region(self.document, lo, hi, key)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            C.CompositeBackground(self.document.sources, 'p',
                (C.CompositeRegion(1.1, 1.6, 'n'), C.CompositeRegion(1.5, 1.8, 'n')))
        reversed_source = replace(self.n, spectral_columns=self.n.spectral_columns[::-1])
        with self.assertRaisesRegex(ValueError, 'original spectral grid'):
            C.CompositeBackground((self.p, reversed_source), 'p')
        shifted = tuple(str(float(v) + .001) for v in self.p.spectral_columns)
        self.write_csv('shifted.csv', [np.ones(21)], columns=shifted)
        with self.assertRaisesRegex(ValueError, 'original spectral grid'):
            C.load_composite_source(self.root, ['p.csv', 'shifted.csv'])

    def test_original_wavelength_header_order_and_precision_are_preserved(self):
        columns = ('620.123456789123', '650.987654321123', '600.333333333333')
        path = self.write_csv('wavelength.csv', [[7.123456789012345, 9., 11.]], columns=columns)
        source = C.load_composite_source(self.root, [path], key='wl')
        document = C.CompositeBackground((source,), 'wl')
        self.save(document)
        with self.output.open(newline='') as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(tuple(rows[0][4:]), columns)
        np.testing.assert_array_equal(np.asarray(rows[1][4:], float), [7.123456789012345, 9., 11.])
        frozen = C.read_composite_background(self.output)
        np.testing.assert_array_equal(frozen.sources[0].intensity, source.intensity)
        for which in ('first', 'last', 'all'):
            baseline = loader.build_external_baseline(str(self.root), [self.output.name], which=which)
            # The legacy CSV parser can round a 17-digit value by one ULP.
            np.testing.assert_allclose(baseline['I0'], source.intensity, rtol=1e-15, atol=0)
            np.testing.assert_array_equal(baseline['energy'], source.energy)

    def test_missing_values_outside_active_regions_do_not_invalidate_source(self):
        p, n = np.arange(21., dtype=float) + 10, np.full(21, np.nan)
        p[5:13], n[5:13] = np.nan, np.arange(8.) + 100
        self.write_csv('p_missing.csv', [p])
        self.write_csv('n_missing.csv', [n])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            psource = C.load_composite_source(self.root, ['p_missing.csv'], key='p')
            nsource = C.load_composite_source(self.root, ['n_missing.csv'], key='n')
        document = self.region(document=C.CompositeBackground((psource, nsource), 'p'))
        expected = p.copy()
        expected[5:13] = n[5:13]
        np.testing.assert_array_equal(C.compose_background(document)[1], expected)
        self.save(document)
        reopened = C.read_composite_background(self.output)
        np.testing.assert_array_equal(C.compose_background(reopened)[1], expected)
        self.assertTrue(np.isnan(reopened.sources[1].intensity[0]))
        # NaN in a genuinely selected location is retained, never interpolated.
        selected_nan = self.region(3, 13, document=reopened)
        self.assertTrue(np.isnan(C.compose_background(selected_nan)[1][3]))

    def test_frozen_snapshot_can_load_reassign_and_resave_without_originals(self):
        self.save()
        before = self.metadata()
        for source in (self.p, self.n):
            Path(source.files[0]).unlink()
        reopened = C.read_composite_background(self.output)
        np.testing.assert_array_equal(C.compose_background(reopened)[1], C.compose_background(self.region())[1])
        changed = self.region(2, 17, document=reopened)
        other = C.save_composite_background(self.root / 'other.csv', changed)
        np.testing.assert_array_equal(C.compose_background(C.read_composite_background(other))[1],
                                      C.compose_background(changed)[1])
        self.assertEqual(reopened.sources[0].source_hashes, self.p.source_hashes)
        self.assertEqual(before['sources'][0]['files'], ['p.csv'])
        self.assertEqual(before['operation'], 'Composite background')
        baseline = loader.build_external_baseline(str(self.root), [self.output.name])
        np.testing.assert_array_equal(baseline['I0'], C.compose_background(reopened)[1])

    def test_inherited_seams_only_follow_active_source_intervals(self):
        p = replace(self.p, boundaries_ev=tuple(self.energy[[2, 8, 16]]))
        n = replace(self.n, boundaries_ev=tuple(self.energy[[3, 9, 18]]))
        document = self.region(5, 13, document=C.CompositeBackground((p, n), 'p'))
        expected = tuple(self.energy[[2, 5, 9, 13, 16]])
        self.assertEqual(C.compose_background(document)[2], expected)
        self.save(document)
        nested = C.load_composite_source(self.root, [self.output.name], key='nested')
        self.assertEqual(nested.boundaries_ev, expected)
        self.assertEqual(C.composite_background_boundaries(self.output), expected)

    def test_marker_missing_sidecar_and_ordinary_csv_behavior(self):
        self.assertFalse(C.is_composite_background(self.root / 'p.csv'))
        (self.root / 'p.metadata.json').write_text('invalid', encoding='utf-8')
        self.assertEqual(C.composite_background_boundaries(self.root / 'p.csv'), ())
        self.save()
        C.composite_metadata_path(self.output).unlink()
        self.assertTrue(C.is_composite_background(self.output))
        self.assertIsNone(C.composite_metadata_signature(self.output)[-1])
        with self.assertRaisesRegex(ValueError, 'metadata is missing'):
            C.composite_background_boundaries(self.output)

    def test_hash_version_and_all_snapshot_validation(self):
        self.save()
        original = self.metadata()
        for version in (0, 2, True, '1'):
            with self.subTest(version=version):
                self.write_metadata({**original, 'schema_version': version})
                with self.assertRaisesRegex(ValueError, 'version'):
                    C.read_composite_background(self.output)
        self.write_metadata(original)
        # An unused source must be authenticated too, for later editing.
        document = self.document
        C.save_composite_background(self.output, document, overwrite=True)
        corrupt = self.metadata()
        corrupt['sources'][1]['intensity'][0] += 1
        self.write_metadata(corrupt)
        with self.assertRaisesRegex(ValueError, 'snapshot hash'):
            C.read_composite_background(self.output)
        self.write_metadata(original)
        self.output.write_bytes(self.output.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'CSV hash'):
            C.read_composite_background(self.output)

    def test_marker_cache_and_sidecar_signature_invalidation_are_independent(self):
        self.save()
        with C._MARKER_LOCK:
            C._MARKER_CACHE.clear()
        with patch.object(C, '_header', wraps=C._header) as read:
            for path in (self.root / 'p.csv', self.output):
                self.assertEqual(C.is_composite_background(path), path == self.output)
                self.assertEqual(C.is_composite_background(path), path == self.output)
            self.assertEqual(read.call_count, 2)
            before = C.composite_metadata_signature(self.output)
            C.composite_metadata_path(self.output).write_text('{}', encoding='utf-8')
            after = C.composite_metadata_signature(self.output)
            self.assertNotEqual(before, after)
            self.assertEqual(read.call_count, 2)
            self.output.write_text('Vbg,Vtg,1,2\n0,0,3,4\n', encoding='utf-8')
            self.assertFalse(C.is_composite_background(self.output))
            self.assertEqual(read.call_count, 3)
        with patch.object(C, '_digest_signature', return_value=None), patch.object(C, '_header', wraps=C._header) as read:
            C.is_composite_background(self.output)
            C.is_composite_background(self.output)
            self.assertEqual(read.call_count, 2)

    def test_nested_metadata_changed_during_source_read_is_rejected(self):
        self.save()
        original = loader.build_external_baseline
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            payload = self.metadata()
            payload['note'] = 'changed while loading'
            self.write_metadata(payload)
            return result
        with patch.object(loader, 'build_external_baseline', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'changed while'):
                C.load_composite_source(self.root, [self.output.name])

    def test_save_refuses_sources_plain_csv_or_broken_existing_pairs(self):
        for path in (self.root / 'p.csv', self.root / 'n.csv'):
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'source CSV'):
                C.save_composite_background(path, self.document, overwrite=True)
            self.assertEqual(path.read_bytes(), before)
        unrelated = self.write_csv('other.csv', [np.ones(21)])
        with self.assertRaisesRegex(ValueError, 'ordinary CSV'):
            C.save_composite_background(unrelated, self.document, overwrite=True)
        self.save()
        with self.assertRaises(FileExistsError):
            C.save_composite_background(self.output, self.document)
        C.composite_metadata_path(self.output).unlink()
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, 'metadata is missing'):
            C.save_composite_background(self.output, self.document, overwrite=True)
        self.assertEqual(self.output.read_bytes(), before)
        orphan = self.root / 'orphan.csv'
        C.composite_metadata_path(orphan).write_text('unrelated', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'without its composite CSV'):
            C.save_composite_background(orphan, self.document, overwrite=True)

    def test_explicit_valid_pair_overwrite_updates_frozen_data(self):
        self.save()
        updated = self.region(2, 18)
        C.save_composite_background(self.output, updated, overwrite=True)
        np.testing.assert_array_equal(C.compose_background(C.read_composite_background(self.output))[1],
                                      C.compose_background(updated)[1])
        self.assertFalse(list(self.root.glob('.*.tmp')))
        self.assertFalse(list(self.root.glob('.*.backup')))

    def test_mid_publish_failure_restores_both_original_artifacts(self):
        self.save()
        metadata = C.composite_metadata_path(self.output)
        before = self.output.read_bytes(), metadata.read_bytes()
        original = os.replace
        def fail_metadata(source, target):
            if str(source).endswith('.tmp') and Path(target).resolve() == metadata.resolve():
                raise OSError('simulated metadata promotion failure')
            return original(source, target)
        with patch.object(C.os, 'replace', side_effect=fail_metadata), self.assertRaisesRegex(OSError, 'simulated'):
            C.save_composite_background(self.output, self.document, overwrite=True)
        self.assertEqual((self.output.read_bytes(), metadata.read_bytes()), before)
        self.assertFalse(list(self.root.glob('.*.tmp')))
        self.assertFalse(list(self.root.glob('.*.backup')))
        C.read_composite_background(self.output)

    def test_new_pair_failure_leaves_no_half_published_background(self):
        metadata = C.composite_metadata_path(self.output)
        original = os.replace
        def fail_metadata(source, target):
            if Path(target).resolve() == metadata.resolve():
                raise OSError('simulated failure')
            return original(source, target)
        with patch.object(C.os, 'replace', side_effect=fail_metadata), self.assertRaises(OSError):
            self.save()
        self.assertFalse(self.output.exists())
        self.assertFalse(metadata.exists())
        self.assertFalse(list(self.root.glob('.*.tmp')))

    def test_cancellation_after_staging_preserves_original_pair(self):
        self.save()
        metadata = C.composite_metadata_path(self.output)
        before = self.output.read_bytes(), metadata.read_bytes()
        cancellation = LoadCancellation()
        original_open = Path.open
        staged = 0
        def cancel_after_staging(path, *args, **kwargs):
            nonlocal staged
            result = original_open(path, *args, **kwargs)
            if path.suffix == '.tmp':
                staged += 1
                if staged == 2:
                    cancellation.cancel()
            return result
        with load_scope(cancellation), patch.object(Path, 'open', cancel_after_staging):
            with self.assertRaises(LoadCancelled):
                C.save_composite_background(self.output, self.document, overwrite=True)
        self.assertEqual((self.output.read_bytes(), metadata.read_bytes()), before)
        self.assertFalse(list(self.root.glob('.*.tmp')))


if __name__ == '__main__':
    unittest.main()
