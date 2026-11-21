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

    def test_scientific_notation_spectrum_headers_are_parsed(self):
        from core.drr_sources import inspect_csv_spectral_grid
        self.path.write_text(
            'Vbg,Vtg,1.0174e+03,1.1e+03,1.1825E3\n0,2,10,20,30\n1,3,40,50,60\n', encoding='utf-8')
        result = self.load(y_axis='BG')
        np.testing.assert_allclose(result['energy'], 1240 / np.array([1182.5, 1100.0, 1017.4]))
        np.testing.assert_array_equal(result['Z'], [[30, 20, 10], [60, 50, 40]])
        self.assertEqual(inspect_csv_spectral_grid(self.path), (1017.4, 1100.0, 1182.5))

    def test_multi_detector_acquisition_headers_are_parsed(self):
        from core.drr_sources import inspect_csv_spectral_grid, inspect_csv_wavelength_center
        self.path.write_text(
            'run_id,point_id,backend,started_at,center_nm,exposure_ms,frames,'
            'Vbg_set,Vbg_meas,Vtg_set,Vtg_meas,Ibg,Itg,Ibias,nm_1180.5,nm_1100.25,nm_1020.0\n'
            'abc,1,andor_ingaas,2026-11-20T16:17:41+00:00,1100.0,1000.0,4,'
            '-7.0,-7.0,2.0,2.0,-5e-10,-9e-10,-2e-11,10,20,30\n'
            'abc,2,andor_ingaas,2026-11-20T16:17:48+00:00,1100.0,1000.0,4,'
            '-6.5,-6.5,2.0,2.0,-4e-10,-9e-10,-3e-11,40,50,60\n', encoding='utf-8')
        result = self.load(y_axis='BG')
        np.testing.assert_allclose(result['energy'], 1240 / np.array([1180.5, 1100.25, 1020.0]))
        np.testing.assert_array_equal(result['Z'], [[10, 20, 30], [40, 50, 60]])
        np.testing.assert_array_equal(result['gate_axis'], [-7.0, -6.5])
        self.assertEqual(inspect_csv_spectral_grid(self.path), (1180.5, 1100.25, 1020.0))
        self.assertAlmostEqual(inspect_csv_wavelength_center(self.path), 1100.25)
        from core.drr_sources import inspect_csv_gate_profile
        profile = inspect_csv_gate_profile(self.path)
        self.assertEqual(profile.spectral_grid, (1180.5, 1100.25, 1020.0))
        self.assertEqual(profile.gate_labels, ('Vbg_set', 'Vbg_meas', 'Vtg_set', 'Vtg_meas'))
        self.assertTrue(profile.grid_complete)

    def test_spectrum_column_names_match_float_forms(self):
        cases = {'703.177': 703.177, '703.177.1': 703.177, '1.0486e+00': 1.0486,
                 '1.0486e+00.1': 1.0486, '7E2': 700.0, '703.': 703.0, '.5': 0.5,
                 '1.742 eV': 1.742, '1.742 EV': 1.742, '703 NM': 703.0,
                 'nm_1182.53759765625': 1182.53759765625, 'nm_703.177.1': 703.177,
                 'eV_1.742': 1.742, 'NM 703': 703.0}
        for name, expected in cases.items():
            self.assertEqual(P._parse_spec_axis_from_colname(name), expected, name)
        for name in ('Vbg', 'nan', 'inf', 'e5', '1e', 'T (K)', '', 'center_nm', 'nm_', 'run_1'):
            self.assertIsNone(P._parse_spec_axis_from_colname(name), name)

    def test_missing_spectrum_columns_error_lists_found_columns(self):
        self.path.write_text('Vbg,Vtg,px0,px1\n0,2,10,20\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, r"no spectrum columns found[\s\S]*'px0', 'px1'"):
            self.load()

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

    def test_same_size_rewrite_with_restored_mtime_invalidates(self):
        self.load()
        stat = self.path.stat()
        self.path.write_text(self.path.read_text().replace('10,20', '11,21'), encoding='utf-8')
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        np.testing.assert_array_equal(self.load(y_axis='BG')['Z'], [[21, 11], [40, 30]])

    def test_unavailable_strong_stamp_does_not_cache_parsed_arrays(self):
        cache = RawSpectrumCache()
        with patch('core.raw_spectrum_cache._digest_signature', return_value=None), \
                patch.object(P, '_parse_raw_spectrum', wraps=P._parse_raw_spectrum) as parse:
            cache.load(self.path, 1, parse)
            cache.load(self.path, 1, parse)
        self.assertEqual(parse.call_count, 2)

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
