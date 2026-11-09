# Adapted from Megasweep Analysis commit 2612f6f.
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core.megasweep import transport_analysis as transport
from core.megasweep.megasweep_analysis import plot_map


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'scan.csv'
        self.path.write_text(
            'Gate,Bias,Signal,Unused,PassIndex,FastDirection\n'
            'V,V,A,V,#,\n'
            '0,1,10,,0,forward\n0,2,20,,0,forward\n'
            '1,2,40,,1,reverse\n', encoding='utf-8')

    def test_units_row_and_header_order_without_optical_channels(self):
        data = transport.load_transport_csv(self.path)
        self.assertEqual(len(data['df']), 3)
        self.assertEqual(data['columns'], ['Gate', 'Bias', 'Signal'])
        self.assertEqual(data['units']['Signal'], 'A')
        self.assertEqual(data['df']['Signal'].tolist(), [10, 20, 40])
        self.assertTrue(data['df']['Unused'].isna().all())

    def test_no_units_row_keeps_first_measurement(self):
        self.path.write_text('B,T,Response\n1,3,4\n2,3,5\n', encoding='utf-8')
        data = transport.load_transport_csv(self.path)
        self.assertEqual(data['columns'], ['B', 'T', 'Response'])
        self.assertEqual(data['df']['Response'].tolist(), [4, 5])

    def test_partial_snake_grid_keeps_missing_cell_and_filters_direction(self):
        data = transport.load_transport_csv(self.path)
        result = transport.transport_map(data, 'Gate', 'Bias', 'Signal')
        np.testing.assert_allclose(result['Z2D'], [[10, 20], [np.nan, 40]], equal_nan=True)
        reverse = transport.transport_map(data, 'Gate', 'Bias', 'Signal', direction='reverse')
        self.assertEqual(reverse['samples']['Signal'].tolist(), [40])

    def test_metadata_extends_grid_only_for_matching_headers(self):
        self.path.with_name('scan_metadata.json').write_text(json.dumps({'params': {
            'axis_slow': 'Gate', 'axis_fast': 'Bias',
            'gate_start': 0, 'gate_stop': 2, 'gate_step': 1,
            'bias_start': 1, 'bias_stop': 2, 'bias_step': 1}}))
        data = transport.load_transport_csv(self.path)
        result = transport.transport_map(data, 'Gate', 'Bias', 'Signal')
        self.assertEqual(result['Z2D'].shape, (3, 2))
        self.assertTrue(np.isnan(result['Z2D'][2]).all())

    def test_duplicate_coordinates_require_pass_selection(self):
        with self.path.open('a') as f:
            f.write('0,1,99,,2,reverse\n')
        data = transport.load_transport_csv(self.path)
        with self.assertRaisesRegex(ValueError, 'duplicate|Duplicate'):
            transport.transport_map(data, 'Gate', 'Bias', 'Signal')
        selected = transport.transport_map(data, 'Gate', 'Bias', 'Signal', pass_index=2)
        self.assertEqual(selected['samples']['Signal'].tolist(), [99])

    def test_cuts_in_both_directions_preserve_values(self):
        result = transport.transport_map(transport.load_transport_csv(self.path), 'Gate', 'Bias', 'Signal')
        cut = transport.transport_cut(result, 'Gate', 0)
        self.assertEqual(cut['axis'], 'Bias')
        np.testing.assert_array_equal(cut['values'], [10, 20])
        cut = transport.transport_cut(result, 'Bias', 2)
        np.testing.assert_array_equal(cut['coordinates'], [0, 1])
        np.testing.assert_array_equal(cut['values'], [20, 40])

    def test_invalid_axes_and_empty_signal_fail_explicitly(self):
        data = transport.load_transport_csv(self.path)
        for axes in [('Gate', 'Gate', 'Signal'), ('Gate', 'Bias', 'Unused')]:
            with self.assertRaises(ValueError):
                transport.transport_map(data, *axes)

    def test_nearby_distinct_coordinates_do_not_collapse_to_metadata_cell(self):
        self.path.write_text('X,Y,Z\n0,0,10\n0.0000001,0,20\n')
        self.path.with_name('scan_metadata.json').write_text(json.dumps({'params': {
            'axis_fast': 'X', 'x_start': 0, 'x_stop': 1, 'x_step': 1}}))
        result = transport.transport_map(transport.load_transport_csv(self.path), 'X', 'Y', 'Z')
        np.testing.assert_array_equal(result['Z2D'][:, 0], [10, 20])

    def test_picoamp_auto_scale_preserves_signal_range(self):
        _, ax = plot_map(np.array([[0, 0], [1, 1]]), np.array([[0, 1], [0, 1]]),
                         np.array([[1e-12, 2e-12], [3e-12, 4e-12]]))
        self.assertEqual(ax.collections[0].get_clim(), (1e-12, 4e-12))


if __name__ == '__main__':
    unittest.main()
