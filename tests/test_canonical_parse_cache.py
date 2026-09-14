import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core import processing_run as P
from core.raw_spectrum_cache import RawSpectrumCache


class CanonicalParseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'scan.csv'
        self.path.write_text('Vbg,Vtg,700,710\n0,2,10,20\n1,3,30,40\n', encoding='utf-8')

    def load(self, **kwargs):
        return P._load_canonical(self.tmp.name, self.path.name, **kwargs)

    def test_numeric_columns_skip_coercion(self):
        calls = []
        original = P.pd.to_numeric
        def convert(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        with patch.object(P.pd, 'to_numeric', convert):
            result = self.load(y_axis='BG')
        self.assertEqual(len(calls), 0)
        np.testing.assert_array_equal(result['Z'], [[20, 10], [40, 30]])

    def test_repeated_load_reuses_parse_and_isolates_returned_arrays(self):
        with patch.object(P.pd, 'read_csv', wraps=P.pd.read_csv) as read:
            first = self.load(y_axis='BG')
            first['Z'][:] = -99
            first['energy'][:] = -99
            first['gate_axis'][:] = -99
            second = self.load(y_axis='TG')
        self.assertEqual(read.call_count, 1)
        np.testing.assert_array_equal(second['Z'], [[20, 10], [40, 30]])
        np.testing.assert_array_equal(second['gate_axis'], [2, 3])
        np.testing.assert_array_equal(second['energy'], 1240 / np.array([710, 700]))

    def test_mixed_duplicate_columns_preserve_finite_selection_and_nan(self):
        self.path.write_text('Vbg,Vtg,700.0,700.0,710.0\n0,2,bad,10,inf\n1,3,30,40,bad\n', encoding='utf-8')
        result = self.load(y_axis='BG')
        np.testing.assert_array_equal(result['Z'], [[np.inf, 10], [np.nan, 40]])

    def test_changed_file_invalidates_cache(self):
        with patch.object(P.pd, 'read_csv', wraps=P.pd.read_csv) as read:
            self.load()
            self.path.write_text('Vbg,Vtg,700,710\n0,2,100,200\n1,3,300,400\n', encoding='utf-8')
            result = self.load()
        self.assertEqual(read.call_count, 2)
        np.testing.assert_array_equal(result['Z'], [[200, 100], [400, 300]])

    def test_file_changed_during_read_is_not_cached(self):
        original = P.pd.read_csv
        def changing_read(*args, **kwargs):
            result = original(*args, **kwargs)
            stat = self.path.stat()
            os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
            return result
        with patch.object(P.pd, 'read_csv', side_effect=changing_read) as read:
            self.load()
            self.load()
        self.assertEqual(read.call_count, 2)

    def test_legacy_format_matches_previous_conversion(self):
        self.path.write_text('0,0,0,0,700,710\n0,2,0,0,10,bad\n1,3,0,0,30,40\n', encoding='utf-8')
        result = self.load(y_axis='BG')
        np.testing.assert_array_equal(result['Z'], [[np.nan, 10], [40, 30]])

    def test_lru_budget_version_and_oversized_entries(self):
        other = self.path.with_name('other.csv')
        other.write_text('unused', encoding='utf-8')
        third = self.path.with_name('third.csv')
        third.write_text('unused', encoding='utf-8')
        cache = RawSpectrumCache(max_bytes=32)
        calls = []
        def parse(path):
            calls.append(path)
            return (np.ones(2), None)
        cache.load(self.path, 1, parse)
        cache.load(other, 1, parse)
        cache.load(self.path, 1, parse)  # refresh LRU
        cache.load(third, 1, parse)  # evict other
        cache.load(self.path, 1, parse)
        self.assertEqual(len(calls), 3)
        cache.load(other, 1, parse)
        self.assertEqual(len(calls), 4)
        cache.load(other, 2, parse)
        self.assertEqual(len(calls), 5)
        self.assertLessEqual(cache._bytes, 32)
        cache = RawSpectrumCache(max_bytes=8)
        cache.load(self.path, 1, parse)
        cache.load(self.path, 1, parse)
        self.assertEqual(len(calls), 7)
        self.assertEqual(cache._bytes, 0)

    def test_same_size_timestamp_change_invalidates(self):
        self.load()
        stat = self.path.stat()
        self.path.write_text(self.path.read_text().replace('10,20', '11,21'), encoding='utf-8')
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
        np.testing.assert_array_equal(self.load(y_axis='BG')['Z'], [[21, 11], [40, 30]])

    def test_background_changes_recompute_using_one_raw_parse(self):
        kwargs = dict(user_folder=self.tmp.name, files=[self.path.name], y_axis='BG',
                      plot_interactive=False, save_png=False, save_dat_file=False,
                      move_original=False)
        with patch.object(P.pd, 'read_csv', wraps=P.pd.read_csv) as read:
            first = P.process_ref_avg(**kwargs, bg_mode='self_first')
            last = P.process_ref_avg(**kwargs, bg_mode='self_last')
        self.assertEqual(read.call_count, 1)
        np.testing.assert_array_equal(first['Z_out'][0], [0, 0])
        np.testing.assert_array_equal(last['Z_out'][-1], [0, 0])
        self.assertFalse(np.array_equal(first['Z_out'], last['Z_out']))

    def test_parse_failure_does_not_poison_cache(self):
        with patch.object(P.pd, 'read_csv', side_effect=ValueError('incomplete CSV')):
            with self.assertRaisesRegex(ValueError, 'incomplete CSV'):
                self.load()
        np.testing.assert_array_equal(self.load(y_axis='BG')['Z'], [[20, 10], [40, 30]])


if __name__ == '__main__':
    unittest.main()
