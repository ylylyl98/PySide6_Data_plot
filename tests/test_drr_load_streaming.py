import tempfile
import unittest
import warnings
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core import loader, processing_run as P
from core.drr_sources import DrrMeasurementAssignment
from core.raw_spectrum_cache import RawSpectrumCache


class DrrLoadStreamingTests(unittest.TestCase):
    def test_raw_and_corrected_averages_preserve_weights_missing_pixels_and_infinities(self):
        inputs = {
            'a.csv': np.array([[2., 0., np.nan, np.inf, 4.], [4., 2., np.nan, 6., -np.inf]]),
            'b.csv': np.array([[4., 2., np.nan, 8., 8.], [4., 4., np.nan, 12., np.inf]]),
        }
        def load(_folder, name, **_kwargs):
            return {'energy': np.arange(1., 6.), 'gate_axis': np.array([0., 1.]),
                    'Z': inputs[name], 'gate_label': 'TG', 'title_name': name, 'stem': name}
        expected = {
            'self_first': [[0., 0., np.nan, 0., 0.], [.5, 1., np.nan, .5, np.nan]],
            'external': [[.5, -.5, np.nan, np.inf, 2.], [1., .5, np.nan, 3.5, np.nan]],
        }
        for mode, want in expected.items():
            with self.subTest(mode=mode), warnings.catch_warnings(), patch.object(P, '_load_canonical', new=load):
                warnings.simplefilter('ignore', RuntimeWarning)
                result = P.process_ref_avg(
                    'unused', ['a.csv', 'b.csv'], bg_mode=mode,
                    external_vector=np.full(5, 2.) if mode == 'external' else None,
                    plot_interactive=False, save_png=False, save_dat_file=False, move_original=False)
                np.testing.assert_array_equal(result['Z_avg'], want)
                np.testing.assert_array_equal(result['R_avg'],
                                              [[3., 1., np.nan, np.inf, 6.], [4., 3., np.nan, 9., np.nan]])
                np.testing.assert_array_equal(inputs['a.csv'],
                                              [[2., 0., np.nan, np.inf, 4.], [4., 2., np.nan, 6., -np.inf]])

    def test_common_and_heterogeneous_loads_keep_measured_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'a.csv').write_text(
                'Vbg,Vtg,Vbias,600,700\n2,2,1.015,30,60\n1,1,.990,20,40\n0,0,.965,10,20\n')
            Path(folder, 'b.csv').write_text(
                'Vbg,Vtg,Vbias,600,700\n0,0,.965,20,40\n1,1,.990,30,60\n2,2,1.015,40,80\n')
            for second_mode in ['Self (last frame)', 'Self (first frame)']:
                with self.subTest(second_mode=second_mode):
                    assignments = (DrrMeasurementAssignment('a.csv'),
                                   DrrMeasurementAssignment('b.csv', second_mode))
                    with patch.object(P, '_RAW_SPECTRUM_CACHE', RawSpectrumCache(max_bytes=0)), \
                            patch.object(P, '_parse_raw_spectrum', wraps=P._parse_raw_spectrum) as parse:
                        cube = loader.load_drr_resolved_avg(
                            folder, ['a.csv', 'b.csv'], assignments=assignments, y_axis='linear:1,1,0')
                    self.assertEqual(parse.call_count, 2)
                    self.assertTrue(hasattr(cube, 'gate_coordinates'), 'Loaded cube discarded measured coordinates')
                    np.testing.assert_allclose(cube.gate_coordinates['Bias'], [.965, .990, 1.015])
                    np.testing.assert_array_equal(cube.gate_coordinates['TG+BG'], [0., 2., 4.])

    def test_load_rejects_conflicting_coordinate_between_reference_knots(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'a.csv').write_text(
                'Vbg,Vtg,Vbias,600,700\n0,0,0,10,20\n2,2,2,20,40\n')
            Path(folder, 'b.csv').write_text(
                'Vbg,Vtg,Vbias,600,700\n0,0,0,10,20\n1,1,.5,15,30\n2,2,2,20,40\n')
            cube = loader.load_drr_avg(folder, ['a.csv', 'b.csv'], bg_mode='self_last', y_axis='tg')
        self.assertTrue(hasattr(cube, 'gate_coordinates'), 'Loaded cube discarded measured coordinates')
        self.assertNotIn('Bias', cube.gate_coordinates)
        np.testing.assert_array_equal(cube.gate_coordinates['TG+BG'], [0., 4.])

    def test_average_does_not_stack_entire_file_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            for index in range(6):
                Path(folder, f'{index}.csv').write_text(
                    f'Vbg,Vtg,600,700\n0,0,{10 + index},{20 + index}\n1,1,20,40\n')
            files = [f'{index}.csv' for index in range(6)]
            for heterogeneous in [False, True]:
                with self.subTest(heterogeneous=heterogeneous), patch.object(
                        np, 'stack', side_effect=AssertionError('Whole selection stacked')):
                    assignments = tuple(DrrMeasurementAssignment(
                        name, 'Self (first frame)' if heterogeneous and index % 2 else 'Self (last frame)')
                        for index, name in enumerate(files))
                    cube = loader.load_drr_resolved_avg(folder, files, assignments=assignments, y_axis='tg')
                    self.assertEqual(cube.Z.shape, (2, 2))
                    self.assertTrue(np.isfinite(cube.Z).all())

    def test_measurement_arrays_are_released_as_files_are_processed(self):
        for heterogeneous in [False, True]:
            with self.subTest(heterogeneous=heterogeneous):
                references = []
                def load(_folder, name, **_kwargs):
                    self.assertLessEqual(sum(ref() is not None for ref in references), 2,
                                         'Prior measurement matrices retained across the selection')
                    values = np.full((2, 3), float(name.split('.')[0]) + 10.)
                    references.append(weakref.ref(values))
                    return {'energy': np.array([1., 2., 3.]), 'gate_axis': np.array([0., 1.]),
                            'Z': values, 'gate_label': 'TG', 'title_name': name, 'stem': name}
                files = [f'{index}.csv' for index in range(16)]
                assignments = tuple(DrrMeasurementAssignment(
                    name, 'Self (first frame)' if heterogeneous and index % 2 else 'Self (last frame)')
                    for index, name in enumerate(files))
                with patch.object(P, '_load_canonical', new=load):
                    cube = loader.load_drr_resolved_avg('unused', files, assignments=assignments)
                self.assertFalse(any(ref() is not None for ref in references))
                np.testing.assert_array_equal(cube.Z, np.zeros((2, 3)))


if __name__ == '__main__':
    unittest.main()
