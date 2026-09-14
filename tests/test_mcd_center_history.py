import json
import tempfile
import unittest
from pathlib import Path

from core.mcd_center_history import read_center_history, merge_center_candidates


class CenterHistoryTests(unittest.TestCase):
    def test_unpublished_staging_metadata_is_not_saved_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'a.csv').touch()
            stage = root / 'Processed Data' / 'MCD' / '.staging-interrupted'
            stage.mkdir(parents=True)
            (stage / 'result_MCD_settings_1.json').write_text(json.dumps({
                'workflow': 'MCD', 'source_file': 'a.csv',
                'mcd_b': {'center_ev': 1.64, 'width_mev': 5}}))
            self.assertEqual(read_center_history(root, 'a.csv'), [])

    def test_reads_all_revisions_and_retained_windows_without_merging_widths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'mcd').mkdir()
            (root / 'mcd' / 'a.csv').touch()
            out = root / 'Processed Data' / 'MCD'
            out.mkdir(parents=True)
            for i, windows in enumerate(([(1.62, 5.)], [(1.62, 5.), (1.66, 5.), (1.62, 10.)])):
                payload = {'workflow': 'MCD', 'source_relative_path': 'mcd/a.csv',
                           'created_utc': f'2026-09-1{i}T12:00:00Z',
                           'mcd_b': {'center_ev': windows[0][0], 'width_mev': windows[0][1]},
                           'windows': [{'center_ev': e, 'width_mev': w} for e, w in windows]}
                (out / f'result_MCD_settings_{i}.json').write_text(json.dumps(payload))
            history = read_center_history(root, 'mcd/a.csv')
            self.assertEqual({(h['center_ev'], h['width_mev']) for h in history}, {(1.62, 5.), (1.66, 5.), (1.62, 10.)})
            item = next(h for h in history if (h['center_ev'], h['width_mev']) == (1.62, 5.))
            self.assertEqual(item['uses'], 2)
            self.assertEqual(item['last_used'], '2026-09-11T12:00:00Z')

    def test_other_source_and_ambiguous_basename_are_not_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder in ('mcd/one', 'mcd/two'):
                (root / folder).mkdir(parents=True)
                (root / folder / 'a.csv').touch()
            out = root / 'Processed Data' / 'MCD'
            out.mkdir(parents=True)
            for i, source in enumerate(({'source_file': 'a.csv'}, {'source_relative_path': 'mcd/two/a.csv'})):
                payload = {'workflow': 'MCD', **source, 'mcd_b': {'center_ev': 1.64, 'width_mev': 5}}
                (out / f'result_MCD_settings_{i}.json').write_text(json.dumps(payload))
            self.assertEqual(read_center_history(root, 'mcd/one/a.csv'), [])

    def test_overlap_keeps_both_origins_without_duplicate_marker(self):
        recommendations = [{'id': 'rec1', 'label': '1', 'display_id': 'R1', 'kind': 'window',
                            'domain': 'mcd', 'center_ev': 1.62, 'width_mev': 5., 'recommended': True}]
        history = [{'center_ev': 1.62, 'width_mev': 5., 'last_used': '2026-09-14', 'uses': 2}]
        merged = merge_center_candidates(recommendations, history)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['display_id'], 'H1/R1')
        self.assertTrue(merged[0]['history'])
        self.assertTrue(merged[0]['recommended'])
