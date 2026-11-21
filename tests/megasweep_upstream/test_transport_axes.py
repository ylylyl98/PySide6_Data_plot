# Adapted from Megasweep Analysis commit 2612f6f.
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from core.megasweep.transport_analysis import load_transport_csv
from core.megasweep import transport_analysis as analysis


class TransportAxisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'scan.csv'

    def load(self, frame, params=None):
        frame.to_csv(self.path, index=False)
        meta = self.path.with_name('scan_metadata.json')
        if params is not None:
            meta.write_text(json.dumps({'params': params}))
        elif meta.exists():
            meta.unlink()
        return load_transport_csv(self.path)

    def suggestion(self, data):
        self.assertIn('axis_suggestion', data, 'Loading should compute an axis suggestion')
        return data['axis_suggestion']

    def test_metadata_sets_orientation_using_only_present_headers(self):
        data = self.load(pd.DataFrame({'Vtg': [0, 0, 1, 1], 'Vbg': [0, 0, 1, 1],
                                      'Vds': [1, 2, 2, 1], 'Doping': [0, 0, 2, 2],
                                      'Ids_X': [10, 20, 30, 40]}),
                         {'axis_fast': 'Vds', 'axis_slow': 'Doping', 'plot_x_axis': 'Vds'})
        result = self.suggestion(data)
        self.assertEqual((result['kind'], result['x'], result['y']), ('map', 'Vds', 'Doping'))
        self.assertEqual(result['source'], 'metadata')
        self.assertIn('plot_x_axis', result['reason'])

    def _optical_frame(self, spectral=10):
        frame = pd.DataFrame({
            'point_id': [1, 2, 3, 4],
            'Vbias_axis_a_set': [-0.7, -0.7, -0.75, -0.75],
            'Doping_axis_b_set': [-5.0, -4.95, -4.95, -5.0],
            'Vtg_set': [-2.5, -2.475, -2.475, -2.5],
            'Vbias_meas': [-0.7, -0.7, -0.75, -0.75],
            'Ibg_A': [1e-10, 2e-10, 3e-10, 4e-10],
            'Ibias_A': [-1e-10, -2e-10, -3e-10, -4e-10],
        })
        for index in range(spectral):
            frame[f'nm_{1180.5 - index}'] = [1.0, 2.0, 3.0, 4.0]
        return frame

    def test_optical_file_hides_spectra_and_uses_acquisition_axes(self):
        data = self.load(self._optical_frame())
        self.assertEqual(data['columns'], ['point_id', 'Vbias_axis_a_set', 'Doping_axis_b_set',
                                           'Vtg_set', 'Vbias_meas', 'Ibg_A', 'Ibias_A'])
        self.assertEqual(len(data['spectral_columns']), 10)
        self.assertIn('nm_1180.5', data['df'])
        result = self.suggestion(data)
        self.assertEqual((result['kind'], result['x'], result['y'], result['source']),
                         ('map', 'Doping_axis_b_set', 'Vbias_axis_a_set', 'acquisition'))
        self.assertEqual(result['channel'], 'Ibias_A')
        grid = analysis.transport_map(data, result['x'], result['y'], 'Ibg_A')
        self.assertEqual(grid['Z2D'].shape, (2, 2))

    def test_few_numeric_named_columns_remain_scalar_channels(self):
        data = self.load(self._optical_frame(spectral=2))
        self.assertIn('nm_1180.5', data['columns'])
        self.assertEqual(data['spectral_columns'], [])

    def test_metadata_preserves_planned_slow_axis_on_partial_first_line(self):
        data = self.load(pd.DataFrame({'Doping': [-1, -1], 'Vds': [.9, .902], 'Ids_X': [2, 3]}),
                         {'axis_fast': 'Vds', 'axis_slow': 'Doping'})
        result = self.suggestion(data)
        self.assertEqual((result['kind'], result['x'], result['y']), ('map', 'Doping', 'Vds'))

    def test_fallback_recognizes_snake_grid_and_excludes_signal_and_readback(self):
        data = self.load(pd.DataFrame({'Bias_measured': [1.01, 2.01, 2.02, 1.02],
                                      'Gate': [0, 0, 1, 1], 'Bias': [1, 2, 2, 1],
                                      'Current': [3, 4, 5, 6], 'PassIndex': [0, 0, 1, 1]}))
        result = self.suggestion(data)
        self.assertEqual((result['x'], result['y']), ('Gate', 'Bias'))
        self.assertEqual(result['source'], 'data')

    def test_stale_metadata_falls_back_with_explanation(self):
        data = self.load(pd.DataFrame({'Gate': [0, 0, 1], 'Bias': [1, 2, 2], 'Signal': [2, 4, 6]}),
                         {'axis_fast': 'Missing', 'axis_slow': 'Absent'})
        result = self.suggestion(data)
        self.assertEqual((result['x'], result['y']), ('Gate', 'Bias'))
        self.assertIn('metadata', result['reason'].lower())

    def test_equivalent_gate_parameterizations_require_manual_choice(self):
        data = self.load(pd.DataFrame({'Vtg': [0, 0, 1, 1], 'Doping': [0, 0, 2, 2],
                                      'Vds': [1, 2, 2, 1], 'Ids_X': [1, 2, 3, 4]}))
        result = self.suggestion(data)
        self.assertIsNone(result['x'])
        self.assertEqual(result['source'], 'ambiguous')
        self.assertEqual(len(result['candidates']), 2)

    def test_linked_gates_are_not_a_two_dimensional_grid(self):
        data = self.load(pd.DataFrame({'Vtg': [0, 1, 2], 'Vbg': [0, 1, 2], 'Ids_X': [4, 5, 7]}))
        result = self.suggestion(data)
        self.assertIsNone(result['x'])

    def test_single_scan_column_suggests_curve_excluding_dc_and_ac_signals(self):
        data = self.load(pd.DataFrame({'Vds': [0, 1, 2], 'E-field': [0, 0, 0],
                                      'Ids_X': [1, 2, 3], 'Ids_DC': [2, 4, 6], 'raw_X': [1, 2, 3]}))
        result = self.suggestion(data)
        self.assertEqual((result['kind'], result['x'], result['y']), ('curve', 'Vds', None))

    def test_unknown_columns_with_no_grid_are_not_guessed(self):
        data = self.load(pd.DataFrame({'Alpha': [1, 2, 3], 'Beta': [5, 6, 7], 'Signal': [7, 8, 9]}))
        self.assertIsNone(self.suggestion(data)['x'])

    def test_curve_preserves_scan_order_and_pass_direction(self):
        data = self.load(pd.DataFrame({'Vds': [0, 1, 1, 0], 'Ids_DC': [2, 3, 5, 4],
                                      'PassIndex': [0, 0, 1, 1],
                                      'FastDirection': ['forward', 'forward', 'reverse', 'reverse']}))
        self.assertTrue(hasattr(analysis, 'transport_curve'))
        result = analysis.transport_curve(data, 'Vds', 'Ids_DC', direction='reverse')
        self.assertEqual(result['samples']['Vds'].tolist(), [1, 0])
        self.assertEqual(result['samples']['Ids_DC'].tolist(), [5, 4])
        self.assertEqual(result['kind'], 'curve')


if __name__ == '__main__':
    unittest.main()
