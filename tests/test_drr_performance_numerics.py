"""Numerical and input-validation contracts for the batched DRR paths."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core import processing_run as processing
from core.drr_sources import inspect_csv_gate_profile


class BatchedDerivativeTests(unittest.TestCase):
    def test_batch_matches_row_reference_with_gaps_and_all_options(self):
        energy = 1 + np.linspace(0, 1, 53) ** 1.2
        values = np.random.default_rng(9).normal(size=(69, 53))
        values[2, 10:13] = np.nan
        values[33, :3] = np.inf
        values[-1] = np.nan
        before = values.copy()
        for order in (1, 2):
            for poly, window, oversample in ((2, 10, 1), (3, 11, 1.5), (4, 101, 2)):
                with self.subTest(order=order, poly=poly, oversample=oversample):
                    args = dict(deriv=order, window_pts=window, polyorder=poly, oversample=oversample)
                    expected = np.array([processing.sg_derivative_origin_1d(energy, row, **args) for row in values])
                    actual = processing.sg_derivative_origin_rows(values, energy, **args)
                    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)
                    np.testing.assert_array_equal(values, before)
                    self.assertFalse(np.shares_memory(actual, values))

    def test_complete_rows_share_bounded_filter_calls(self):
        energy = np.linspace(1, 2, 1340) ** 1.1
        values = np.broadcast_to(energy ** 2, (97, energy.size)).copy()
        sizes = []
        original = processing.savgol_filter
        def filter(values, *args, **kwargs):
            sizes.append(values.shape[0])
            return original(values, *args, **kwargs)
        with patch.object(processing, 'savgol_filter', new=filter):
            result = processing.sg_derivative_origin_rows(values, energy, window_pts=21)
        self.assertEqual(sum(sizes), 97)
        self.assertLess(len(sizes), 10)
        self.assertLessEqual(max(sizes), 32)
        np.testing.assert_allclose(result, 2, atol=1e-7)

    def test_legacy_and_linear_modes_preserve_row_results(self):
        energy = np.linspace(1, 2, 19)
        values = np.random.default_rng(12).normal(size=(3, 19))
        for options in (dict(origin_like=True), dict(interp_kind='linear')):
            expected = np.array([processing.sg_derivative_origin_1d(energy, row, window_pts=7, **options) for row in values])
            np.testing.assert_array_equal(processing.sg_derivative_origin_rows(values, energy, window_pts=7, **options), expected)


class StreamingProfileTests(unittest.TestCase):
    def profile(self, header, rows, delimiter=',', max_rows=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.csv'
            with path.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream, delimiter=delimiter)
                writer.writerow(header)
                writer.writerows(rows)
            return inspect_csv_gate_profile(path, max_rows=max_rows)

    def test_spectral_validation_retains_malformed_rows_and_gate_summary(self):
        for invalid in ('', 'NaN', 'Inf', '-Inf', 'bad'):
            with self.subTest(invalid=invalid):
                result = self.profile(['Vbg', 'Vbias', '700', '701'],
                                      [[0, '', 1, 2], [1, '', invalid, 3], [2, '', 4, 5]])
                self.assertEqual(result.frame_count, 3)
                self.assertTrue(result.varies)
                self.assertFalse(result.grid_complete)
                self.assertEqual(result.gate_grid, ((0.,), (2.,)))
                self.assertEqual(result.gate_labels, ('Vbg',))

    def test_disjoint_spectra_quoted_metadata_and_numeric_formats(self):
        for delimiter in (',', ';', '\t'):
            result = self.profile(['Vbg', '700', 'comment', '701'],
                                  [[0, ' 1_000 ', 'comma, tab\t semicolon;', '+2e2'],
                                   [1, 3, 'multi\nline', 4]], delimiter=delimiter)
            self.assertTrue(result.grid_complete)
            self.assertEqual(result.gate_grid, ((0.,), (1.,)))
            self.assertEqual(result.spectral_grid, (700., 701.))

    def test_ragged_rows_partially_populated_gate_and_preview_limit(self):
        header = ['Vbg', 'Vbias', '700', '701']
        rows = [[0, '', 1, 2], [1, 0, 3], [2, 0, 4, 5]]
        full = self.profile(header, rows)
        self.assertFalse(full.grid_complete)
        self.assertEqual(full.gate_grid, ((2., 0.),))
        self.assertEqual(full.frame_count, 3)
        preview = self.profile(header, rows, max_rows=1)
        self.assertTrue(preview.grid_complete)
        self.assertEqual(preview.gate_labels, ('Vbg',))
        self.assertEqual(preview.frame_count, 1)

    def test_gate_columns_after_spectra_keep_original_labels_and_order(self):
        result = self.profile(['700', 'Vbg', '701', 'Vbias', 'Vtg'],
                              [[1, 0, 2, '', 4], [2, 1, 3, '', 4]])
        self.assertEqual(result.gate_labels, ('Vbg', 'Vtg'))
        self.assertEqual(result.gate_grid, ((0., 4.), (1., 4.)))
        self.assertTrue(result.grid_complete)

    def test_numeric_header_empty_file_and_header_only(self):
        result = self.profile([0, 0, 700, 701], [[0, 0, 1, 2], [1, 0, 3, 4]])
        self.assertTrue(result.grid_complete)
        self.assertEqual(result.gate_labels, ('gate 1', 'gate 2'))
        self.assertEqual(result.spectral_grid, (700., 701.))
        self.assertEqual(self.profile(['Vbg', '700'], []).frame_count, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.csv'
            path.touch()
            self.assertEqual(inspect_csv_gate_profile(path).frame_count, 0)


if __name__ == '__main__':
    unittest.main()
