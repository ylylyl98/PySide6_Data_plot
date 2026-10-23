"""Generated references keep immutable provenance and only effective diagnostics."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core import drr_composite_background as C


class CompositeGenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.energy = np.linspace(1., 2., 11)
        self.columns = tuple(format(value, '.17g') for value in self.energy)
        for name, value in (('p.csv', 10.), ('n.csv', 20.)):
            (self.root / name).write_text('Vbg,Vtg,Vbias,' + ','.join(self.columns)
                + '\n0,0,0,' + ','.join([str(value)] * self.energy.size) + '\n', encoding='utf-8')
        self.p = C.load_composite_source(self.root, ['p.csv'], key='p')
        self.n = C.load_composite_source(self.root, ['n.csv'], key='n')
        self.report = dict(schema_version=1, report_kind='auto_composite_suggestion',
            algorithm='adaptive_masked_regions_v1', original_default_source='p',
            settings={'smooth_window': 5, 'optional_threshold': None}, input_source_keys=['p', 'n'],
            source_segments=[{'xmin_ev': 1., 'xmax_ev': 1.5, 'source_key': 'p'},
                             {'xmin_ev': 1.5, 'xmax_ev': 2., 'source_key': 'n'}],
            estimated_regions=[[1.1, 1.9]], unresolved_regions=[[1.7, 1.9]],
            unsupported_regions=[[1., 1.1]], texture_missing_regions=[[1.7, 1.8]],
            texture_source_segments=[{'xmin_ev': 1.1, 'xmax_ev': 1.5, 'source_key': 'p'},
                                     {'xmin_ev': 1.3, 'xmax_ev': 1.9, 'source_key': 'n'}],
            scores={'confidence': None, 'residual': 0.125})
        self.generated = replace(self.p, key='generated', label='Estimated background',
            files=self.p.files + self.n.files, source_hashes=self.p.source_hashes + self.n.source_hashes,
            intensity=np.linspace(12., 18., self.energy.size), boundaries_ev=(1.5,),
            generation_json=json.dumps(self.report))
        self.document = C.CompositeBackground((self.p, self.n, self.generated), 'generated')
        self.output = self.root / 'generated.csv'

    def test_generation_is_canonical_and_deeply_isolated_from_returned_reports(self):
        expected = json.dumps(self.report, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        self.assertEqual(self.generated.generation_json, expected)
        returned = C.source_generation(self.generated)
        returned['settings']['smooth_window'] = 999
        returned['estimated_regions'][0][0] = 0
        returned['source_segments'][0]['source_key'] = 'missing'
        self.assertEqual(C.source_generation(self.generated), self.report)
        self.assertEqual(C.source_generation(self.p), {})
        self.assertEqual(replace(self.generated, generation_json='{}').generation_json, '{}')

    def test_invalid_json_shapes_nonfinite_values_and_schema_fields_fail_fast(self):
        invalid = ['[]', 'null', '1', '"text"', '{',
                   '{"settings":{"score":NaN}}', '{"score":Infinity}', '{"score":1e999}',
                   '{"schema_version":2}', '{"schema_version":true}', '{"settings":[]}',
                   '{"algorithm":false}', '{"report_kind":""}',
                   '{"input_source_keys":["p","p"]}', '{"input_source_keys":[null]}']
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.p, generation_json=value)
        with self.assertRaises(ValueError):
            replace(self.p, generation_json=self.report)

    def test_invalid_diagnostic_intervals_are_rejected(self):
        invalid = [None, [[1.2]], [[1.4, 1.3]], [[1.3, 1.3]], [[0.9, 1.5]], [[1.5, 2.1]],
                   [[True, 1.4]], [[None, 1.4]], [[1.5, 1.8], [1.1, 1.3]],
                   [[1.1, 1.5], [1.4, 1.8]], [{'xmin_ev': 1.2}],
                   [{'xmin_ev': 1.2, 'xmax_ev': 1.3, 'source_key': False}]]
        for field in ('estimated_regions', 'unresolved_regions', 'unsupported_regions', 'source_segments',
                      'texture_missing_regions', 'texture_source_segments'):
            for regions in invalid:
                with self.subTest(field=field, regions=regions), self.assertRaises(ValueError):
                    replace(self.p, generation_json=json.dumps({field: regions}))

    def test_manual_overrides_clip_generated_diagnostics_and_restore_them_when_removed(self):
        edited = C.assign_region(self.document, 1.3, 1.6, 'p')
        self.assertEqual(C.effective_generation_regions(edited), ((1.1, 1.3), (1.6, 1.9)))
        edited = C.assign_region(edited, 1.8, 2., 'n')
        self.assertEqual(C.effective_generation_regions(edited), ((1.1, 1.3), (1.6, 1.8)))
        self.assertEqual(C.effective_generation_regions(edited, 'unresolved_regions'), ((1.7, 1.8),))
        self.assertEqual(C.effective_generation_regions(edited, 'unsupported_regions'), ((1., 1.1),))
        restored = C.assign_region(edited, 1., 2., 'generated')
        self.assertEqual(C.effective_generation_regions(restored), ((1.1, 1.9),))

    def test_inactive_sources_are_hidden_and_active_region_intersection_is_exact(self):
        inactive = replace(self.document, default_source='p')
        self.assertEqual(C.effective_generation_regions(inactive), ())
        clipped = C.assign_region(inactive, 1.4, 1.6, 'generated')
        self.assertEqual(C.effective_generation_regions(clipped), ((1.4, 1.6),))
        self.assertEqual(C.effective_generation_regions(clipped, 'unresolved_regions'), ())
        self.assertEqual(C.effective_generation_regions(clipped, 'unsupported_regions'), ())
        with self.assertRaises(ValueError):
            C.effective_generation_regions(clipped, 'settings')

    def test_dictionary_intervals_and_adjacent_active_generated_sources_are_unioned(self):
        second = replace(self.generated, key='second', generation_json=json.dumps({
            'estimated_regions': [{'xmin_ev': 1.3, 'xmax_ev': 1.8}],
            'unsupported_regions': [{'xmin_ev': 1.8, 'xmax_ev': 2.}]}))
        document = C.CompositeBackground((*self.document.sources, second), 'generated')
        document = C.assign_region(document, 1.3, 1.8, 'second')
        self.assertEqual(C.effective_generation_regions(document), ((1.1, 1.9),))
        # The second source's unsupported interval begins where it stops being used.
        self.assertEqual(C.effective_generation_regions(document, 'unsupported_regions'), ((1., 1.1),))

    def test_texture_diagnostics_follow_manual_overrides_and_overlapping_carrier_contributions(self):
        self.assertEqual(C.effective_generation_regions(self.document, 'texture_missing_regions'), ((1.7, 1.8),))
        self.assertEqual(C.effective_generation_regions(self.document, 'texture_source_segments'), ((1.1, 1.9),))
        edited = C.assign_region(self.document, 1.3, 1.6, 'p')
        self.assertEqual(C.effective_generation_regions(edited, 'texture_source_segments'), ((1.1, 1.3), (1.6, 1.9)))
        edited = C.assign_region(edited, 1.6, 1.9, 'n')
        self.assertEqual(C.effective_generation_regions(edited, 'texture_missing_regions'), ())
        self.assertEqual(C.effective_generation_regions(edited, 'texture_source_segments'), ((1.1, 1.3),))

    def test_texture_source_segments_reject_ambiguous_or_duplicate_source_contributions(self):
        invalid = [
            [[1.1, 1.5]],
            [{'xmin_ev': 1.1, 'xmax_ev': 1.5}],
            [{'xmin_ev': 1.1, 'xmax_ev': 1.5, 'source_key': 'p'},
             {'xmin_ev': 1.3, 'xmax_ev': 1.9, 'source_key': 'p'}],
            [{'xmin_ev': 1.3, 'xmax_ev': 1.9, 'source_key': 'p'},
             {'xmin_ev': 1.1, 'xmax_ev': 1.5, 'source_key': 'n'}],
        ]
        for segments in invalid:
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                replace(self.p, generation_json=json.dumps({'texture_source_segments': segments}))

    def test_generated_save_reopen_preserves_parents_reports_data_and_seams_without_originals(self):
        edited = C.assign_region(self.document, 1.3, 1.4, 'p')
        C.save_composite_background(self.output, edited)
        payload = json.loads(C.composite_metadata_path(self.output).read_text(encoding='utf-8'))
        generated = next(source for source in payload['sources'] if source['key'] == 'generated')
        self.assertEqual(generated['files'], ['p.csv', 'n.csv'])
        self.assertEqual(generated['generation'], self.report)
        self.assertEqual(len(generated['source_hashes']), 2)
        self.assertNotIn('no normalization or smoothing', payload['processing']['formula'])
        self.assertIn('generation report', payload['processing']['formula'])
        for filename in ('p.csv', 'n.csv'):
            (self.root / filename).unlink()
        restored = C.read_composite_background(self.output)
        self.assertEqual([source.key for source in restored.sources], ['p', 'n', 'generated'])
        self.assertEqual(C.source_generation(restored.sources[-1]), self.report)
        self.assertEqual(C.effective_generation_regions(restored), C.effective_generation_regions(edited))
        self.assertEqual(C.compose_background(restored)[2], C.compose_background(edited)[2])
        np.testing.assert_array_equal(C.compose_background(restored)[1], C.compose_background(edited)[1])

    def test_generation_report_is_covered_by_the_existing_snapshot_hash(self):
        C.save_composite_background(self.output, self.document)
        metadata = C.composite_metadata_path(self.output)
        payload = json.loads(metadata.read_text(encoding='utf-8'))
        payload['sources'][-1]['generation']['settings']['smooth_window'] = 7
        metadata.write_text(json.dumps(payload), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'snapshot hash'):
            C.read_composite_background(self.output)

    def test_rehashed_but_invalid_generation_metadata_is_rejected(self):
        C.save_composite_background(self.output, self.document)
        metadata = C.composite_metadata_path(self.output)
        original = metadata.read_text(encoding='utf-8')
        for invalid in ([], None, {'estimated_regions': [[0.5, 1.5]]}, {'schema_version': 2}):
            payload = json.loads(original)
            item = payload['sources'][-1]
            item['generation'] = invalid
            item['snapshot_sha256'] = C._snapshot_digest(item)
            metadata.write_text(json.dumps(payload), encoding='utf-8')
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'Invalid composite'):
                C.read_composite_background(self.output)

    def test_generation_does_not_change_csv_schema_and_legacy_manual_snapshots_still_load(self):
        manual = replace(self.generated, generation_json='')
        document = replace(self.document, sources=(self.p, self.n, manual))
        intensity = C.compose_background(document)[1]
        self.assertEqual(C._csv_bytes(document, intensity), C._csv_bytes(self.document, intensity))
        C.save_composite_background(self.output, document)
        payload = json.loads(C.composite_metadata_path(self.output).read_text(encoding='utf-8'))
        self.assertEqual(payload['schema_version'], 1)
        self.assertTrue(all('generation' not in source for source in payload['sources']))
        self.assertIn('no normalization or smoothing', payload['processing']['formula'])
        restored = C.read_composite_background(self.output)
        self.assertTrue(all(source.generation_json == '' for source in restored.sources))


if __name__ == '__main__':
    unittest.main()
