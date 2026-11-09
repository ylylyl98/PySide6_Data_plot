# Adapted from Megasweep Analysis commit 2612f6f.
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from core.megasweep import megasweep_analysis as analysis

from core.megasweep import spectral_movie as movie


def cut(value):
    return dict(energy=np.array([1., 2., 3.]), axis_values=np.array([0., 2.]),
                spectra=np.array([[1., 2., 3.], [4., 5., 6.]]) * (value + 1),
                axis_label='Efield (V)', fixed_axis_label='Doping (V)',
                c_value_used=value, cut_type='x', processing='spectrum',
                signal_label='PL Intensity (a.u.)')


class MovieTests(unittest.TestCase):
    def test_titles_are_short_without_rounding_scientific_coordinates(self):
        result = cut(.123456789)
        result.update(cut_type='doping', fixed_axis_label='TG + 0.9 BG (V)')
        self.assertEqual(analysis.spectral_slice_title(result), 'D = 0.123457 V')
        self.assertEqual(result['c_value_used'], .123456789)
        result['c_value_used'] = -.010000
        self.assertEqual(analysis.spectral_slice_title(result), 'D = -0.01 V')
        result['c_value_used'] = .123456789
        result.update(processing='second_derivative', derivative_window=9)
        self.assertEqual(analysis.spectral_slice_title(result), 'D = 0.123457 V')
        self.assertEqual(result['derivative_window'], 9)
        precision = analysis.spectral_title_precision([1., 1.00000001])
        first, second = cut(1.), cut(1.00000001)
        self.assertNotEqual(analysis.spectral_slice_title(first, precision),
                            analysis.spectral_slice_title(second, precision))

    def test_sequence_uses_numeric_coordinates_and_inclusive_range(self):
        self.assertEqual(movie.select_values([2., -.3, 0., .01], -.3, 2., 2), [-.3, .01])
        with self.assertRaises(ValueError):
            movie.select_values([0., 1.], 1., 0., 1)

    def test_timing_respects_direction_speed_rounding_and_endpoint_holds(self):
        options = movie.MovieOptions(hold=.25, speed=2., fps=20, head=.1, tail=.2,
                                     direction='pingpong', repeats=1)
        self.assertEqual(movie.timeline(3, options), [(0, 5), (1, 2), (2, 3), (1, 2), (0, 7)])
        self.assertEqual(movie.timeline(3, movie.MovieOptions(hold=.1, fps=10, direction='reverse')),
                         [(2, 1), (1, 1), (0, 1)])
        with self.assertRaises(ValueError):
            movie.timeline(2, movie.MovieOptions(hold=.001, fps=30))

    def test_global_view_uses_visible_signal_and_fixed_sequence_axes(self):
        ranges = {'x': dict(auto=False, min=1.5, max=3.), 'y': dict(auto=True, min=0, max=1)}
        view = movie.resolve_sequence_view(cut, [0., 1.], ranges, 'global', {})
        self.assertEqual(view['xlim'], (1.5, 3.))
        self.assertEqual(view['ylim'], (0., 2.))
        self.assertEqual(view['clim'], (2., 12.))
        self.assertEqual(view['values'], [0., 1.])

    def test_manual_and_per_frame_modes_and_derivative_symmetry(self):
        manual = movie.resolve_sequence_view(cut, [0., 1.], {}, 'manual', dict(min=-10, max=20))
        self.assertEqual(manual['clim'], (-10., 20.))
        per_frame = movie.resolve_sequence_view(cut, [0., 1.], {}, 'frame', {})
        self.assertIsNone(per_frame['clim'])
        def derivative(value):
            result = cut(value)
            result.update(processing='second_derivative', derivative_window=3)
            result['spectra'][0, 0] = -20
            return result
        view = movie.resolve_sequence_view(derivative, [0., 1.], {}, 'global', {})
        self.assertEqual(view['clim'], (-20., 20.))
        with self.assertRaises(ValueError):
            movie.resolve_sequence_view(cut, [0.], {}, 'manual', dict(min=2, max=1))

    def test_invalid_slices_report_skips_without_inventing_measurements(self):
        def partial(value):
            result = cut(value)
            if value == 1:
                result['axis_values'] = np.array([0.])
                result['spectra'] = result['spectra'][:1]
            return result
        view = movie.resolve_sequence_view(partial, [0., 1.], {}, 'global', {})
        self.assertEqual(view['values'], [0.])
        self.assertEqual(view['skipped'][0]['value'], 1.)

    def test_frames_keep_canvas_and_axes_layout_with_long_titles(self):
        options = movie.MovieOptions(width=420, height=288)
        view = movie.resolve_sequence_view(cut, [0., 12345.678], {}, 'global', {})
        first = movie.render_frame(cut(0), view, options)
        last = movie.render_frame(cut(12345.678), view, options)
        self.assertEqual(first.shape, (288, 420, 3))
        self.assertEqual(last.shape, first.shape)
        self.assertTrue(np.all(first[0, 0] == 255))

    def test_small_frames_keep_titles_axis_labels_and_colorbar_inside_canvas(self):
        bounds = []
        class RecordingCanvas(FigureCanvasAgg):
            def draw(self):
                super().draw()
                bounds.clear()
                renderer = self.get_renderer()
                for axes in self.figure.axes:
                    for text in (axes.title, axes.xaxis.label, axes.yaxis.label):
                        if text.get_text():
                            bounds.append((text.get_text(), text.get_window_extent(renderer).bounds))
        result = cut(0)
        result.update(processing='second_derivative', derivative_window=9,
                      fixed_axis_label='Back gate compensated electron density divided by relative dielectric permittivity using calibrated top and bottom capacitance (10^12 cm^-2)',
                      axis_label='Back gate compensated electron density divided by relative dielectric permittivity using calibrated top and bottom capacitance (10^12 cm^-2)',
                      signal_label='d²PL/dE² (a.u./eV²)')
        view = movie.resolve_sequence_view(lambda value: result, [0], {}, 'global', {})
        for output_width, output_height in ((320, 240), (420, 288), (1400, 240), (1400, 960)):
            with self.subTest(size=(output_width, output_height)):
                bounds.clear()
                with patch('core.megasweep.spectral_movie.FigureCanvasAgg', RecordingCanvas):
                    movie.render_frame(result, view, movie.MovieOptions(width=output_width, height=output_height))
                for name, (x, y, width, height) in bounds:
                    self.assertGreaterEqual(x, 0, name)
                    self.assertGreaterEqual(y, 0, name)
                    self.assertLessEqual(x + width, output_width, name)
                    self.assertLessEqual(y + height, output_height, name)

    def test_display_between_sample_centers_keeps_intersecting_heatmap_cells(self):
        ranges = {'x': dict(auto=False, min=1.2, max=1.8),
                  'y': dict(auto=False, min=.2, max=.8)}
        view = movie.resolve_sequence_view(cut, [0., 1.], ranges, 'global', {})
        self.assertEqual(view['values'], [0., 1.])
        self.assertEqual(view['clim'], (1., 4.))

    def test_readable_default_movie_keeps_standard_labels_inside_after_final_draw(self):
        bounds = []
        class RecordingCanvas(FigureCanvasAgg):
            def draw(self):
                super().draw()
                bounds.clear()
                for axes in self.figure.axes:
                    bounds.append(axes.yaxis.label.get_window_extent(self.get_renderer()).bounds)
        result = cut(0)
        result['energy'] = np.linspace(1.1, 1.5, 101)
        result['axis_values'] = np.linspace(-1., 1., 41)
        result['spectra'] = np.ones((41, 101)) * 1000
        result['axis_label'] = 'E = TG - 0.9 BG (V)'
        view = movie.resolve_sequence_view(lambda value: result, [0.])
        with patch('core.megasweep.spectral_movie.FigureCanvasAgg', RecordingCanvas):
            movie.render_frame(result, view, movie.MovieOptions())
        for x, y, width, height in bounds:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x + width, 1400)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y + height, 960)

    def test_real_mp4_decodes_with_expected_frames_duration_and_numbered_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'sequence.mp4'
            options = movie.MovieOptions(hold=.2, fps=10, width=420, height=288)
            first = movie.export_movie(cut, [0., 1.], destination, options,
                                       metadata={'source_csv': 'sample.csv'})
            reader = imageio_ffmpeg.read_frames(str(first['path']), pix_fmt='rgb24')
            try:
                info = next(reader)
                frames = list(reader)
            finally:
                reader.close()
            self.assertEqual(len(frames), 4)
            self.assertAlmostEqual(info['duration'], .4, delta=.11)
            self.assertEqual(info['size'], (420, 288))
            sidecar = json.loads(Path(first['sidecar']).read_text(encoding='utf-8'))
            self.assertEqual(sidecar['values'], [0., 1.])
            self.assertEqual(sidecar['metadata']['source_csv'], 'sample.csv')
            second = movie.export_movie(cut, [0., 1.], destination, options)
            self.assertEqual(Path(second['path']).name, 'sequence_002.mp4')
            self.assertEqual(Path(second['sidecar']).name, 'sequence_002.settings.json')
            self.assertEqual(len(list(Path(directory).glob('*.mp4'))), 2)

    def test_cancellation_during_encoding_leaves_no_partial_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            cancelled = threading.Event()
            def progress(current, total, message):
                if message.startswith('Encoding'):
                    cancelled.set()
            with self.assertRaises(movie.MovieCancelled):
                movie.export_movie(cut, [0., 1.], Path(directory) / 'cancel.mp4',
                                   movie.MovieOptions(width=420, height=288),
                                   progress=progress, cancel=cancelled)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cancellation_while_publishing_removes_reserved_video_and_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            cancelled = threading.Event()
            def progress(current, total, message):
                if message.startswith('Saving validated MP4') and current > 0:
                    cancelled.set()
            with self.assertRaises(movie.MovieCancelled):
                movie.export_movie(cut, [0., 1.], Path(directory) / 'cancel.mp4',
                                   movie.MovieOptions(width=420, height=288),
                                   progress=progress, cancel=cancelled)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
