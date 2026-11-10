import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.export import _build_streamlit_style_heatmap_fig
from core.loader import DataCube
from core.peak_source_figure import render_source_figure
from core.plotting import HeatmapParams, RightAxis, SplitColorScale, plot_heatmap
from ui_qt.quick.peak_source_adapters import capture_source_export


class SourceFigureTests(unittest.TestCase):
    def setUp(self):
        self.cube = DataCube(np.array([1., 2., 3.]), np.array([1., 2.]), np.array([[1., 2., 3.], [2., 3., 4.]]), 'Gate', 'Sample', 'Intensity')
        self.params = HeatmapParams('Sample', 'Energy', 'Gate', 'Intensity', 1., 4., (1., 3.), (1., 2.),
                                    split_scale=SplitColorScale(2., 1., 3., 2., 4.),
                                    right_axis=RightAxis('Other', (1., 2.), (10., 20.)))
        self.dataset = dict(cube=self.cube, kind='PL', branches=[dict(id='b', name='B', enabled=True, color=0)],
                            result=dict(candidate_mode=True, points=[dict(energy=2., y=1., polarity='peak', branch_id='b')]),
                            source_export=dict(params=asdict(self.params), drr=False, png_title=None))

    def test_export_geometry_and_marker(self):
        normal = _build_streamlit_style_heatmap_fig(self.cube, self.params, drr=False)
        overlay = render_source_figure(self.dataset)
        try:
            self.assertEqual(tuple(normal.get_size_inches()), tuple(overlay.get_size_inches()))
            self.assertEqual(normal.dpi, overlay.dpi)
            self.assertEqual(len(normal.axes), len(overlay.axes))
            self.assertEqual(normal.axes[0].get_position().bounds, overlay.axes[0].get_position().bounds)
            markers = [a for a in overlay.axes[0].collections if a.get_gid() == 'peak-candidates']
            self.assertEqual(len(markers), 1)
            np.testing.assert_array_equal(markers[0].get_offsets(), [[2., 1.]])
        finally:
            plt.close(normal); plt.close(overlay)

    def test_capture_does_not_change_controls(self):
        owner = SimpleNamespace(_values={'scale': 'linear'}, _params=lambda scale: self.params,
                                _png_export_width=lambda: 1.5)
        axis = SimpleNamespace(get_xlim=lambda: (1.1, 2.9), get_ylim=lambda: (1., 2.))
        before = asdict(self.params)
        with patch('core.peak_display.capture_display', return_value=dict(cmap='plasma', vmin=0., vmax=9., split_scale=None)):
            captured = capture_source_export(owner, 'pl', self.cube, axis)
        self.assertEqual(asdict(self.params), before)
        self.assertEqual(captured['params']['png_width_scale'], 1.5)
        self.assertEqual(captured['params']['xlim'], (1., 3.))
        self.assertEqual((captured['params']['cmap'], captured['params']['vmin'], captured['params']['vmax']), ('plasma', 0., 9.))

    def test_old_workspace_falls_back_to_export_geometry(self):
        self.dataset.pop('source_export')
        self.dataset['view'] = {'xlim': [1., 3.], 'ylim': [1., 2.], 'display': {'cmap': 'plasma', 'vmin': 1., 'vmax': 4.}}
        fig = render_source_figure(self.dataset)
        try:
            self.assertEqual(fig.axes[0].get_xlim(), (1., 3.))
            self.assertTrue(any(a.get_gid() == 'peak-candidates' for a in fig.axes[0].collections))
        finally:
            plt.close(fig)

    def test_current_two_x_width_reaches_saved_overlay(self):
        from io import BytesIO
        from PIL import Image
        from core.typography import save_figure
        owner = SimpleNamespace(_values={'scale': 'linear'}, _params=lambda scale: self.params,
                                _png_export_width=lambda: 2.)
        with patch('core.peak_display.capture_display', return_value=dict(cmap='plasma', vmin=1., vmax=4., split_scale=None)):
            self.dataset['source_export'] = capture_source_export(owner, 'pl', self.cube, None)
        normal = render_source_figure(dict(self.dataset, source_export=dict(params=asdict(self.params))))
        wide = render_source_figure(self.dataset)
        try:
            self.assertEqual(wide.get_figwidth(), normal.get_figwidth() * 2)
            self.assertEqual(wide.get_figheight(), normal.get_figheight())
            self.assertEqual(wide.dpi, normal.dpi)
            output = BytesIO()
            save_figure(wide, output, format='png', dpi=wide.dpi,
                        bbox_inches=wide.bbox_inches, pad_inches=0)
            output.seek(0)
            with Image.open(output) as saved:
                self.assertEqual(saved.size, (2400, 930))
        finally:
            plt.close(normal)
            plt.close(wide)

    def test_drr_capture_uses_export_background_title_and_viewport(self):
        request = SimpleNamespace(background_mode='self-first', background_recipe=None)
        products = SimpleNamespace(request=request, source=self.cube)
        display = SimpleNamespace(heatmap_params=lambda cube, channel: self.params)
        owner = SimpleNamespace(_drr_display=display, _drr_products=products, _png_export_width=lambda: 1.)
        axis = SimpleNamespace(get_xlim=lambda: (1.1, 2.9), get_ylim=lambda: (1.2, 1.8))
        with patch('core.peak_display.capture_display', return_value=dict(cmap='plasma', vmin=0., vmax=9., split_scale=None)):
            captured = capture_source_export(owner, 'drr', self.cube, axis, 'second')
        self.assertEqual(captured['png_title'], 'Sample · First')
        self.assertEqual(captured['params']['xlim'], (1.1, 2.9))
        self.assertEqual(self.params.xlim, (1., 3.))

    def test_actual_mesh_norm_overrides_stale_controls(self):
        fig, axis = plt.subplots()
        try:
            plot_heatmap(axis, self.cube, self.params)
            axis.collections[0].set_clim(1.2, 2.8)
            axis.collections[1].set_clim(2.2, 3.8)
            owner = SimpleNamespace(_values={'scale': 'linear'}, _params=lambda scale: self.params)
            captured = capture_source_export(owner, 'pl', self.cube, axis)
            split = captured['params']['split_scale']
            self.assertEqual((split['left_vmin'], split['left_vmax']), (1.2, 2.8))
            self.assertEqual((split['right_vmin'], split['right_vmax']), (2.2, 3.8))
            self.assertEqual(self.params.split_scale.left_vmin, 1.)
        finally:
            plt.close(fig)

    def test_missing_color_settings_are_rejected(self):
        self.dataset.pop('source_export')
        self.dataset['view'] = {'display': {'cmap': 'viridis'}}
        with self.assertRaisesRegex(ValueError, 'Reopen.*main window'):
            render_source_figure(self.dataset)

    def test_saved_markers_follow_applied_peak_dip_filters_at_both_widths(self):
        from copy import deepcopy
        from core.peak_workspace import create_dataset
        from core.peak_candidates import detect_heatmap, filter_heatmap
        from tests.test_peak_workspace import sample
        dataset = create_dataset(sample('DRR'), 'DRR', 'second', 'source_d2E.dat', 'Source')
        dataset['settings'].update(candidate_floor=0., prominence=0., min_snr=0.,
                                   min_width_mev=0., min_distance_mev=0., neighbor_support=0)
        dataset['candidates'] = detect_heatmap(dataset, dataset['settings'])
        dataset['view'] = dict(display=dict(cmap='RdBu_r', vmin=-1., vmax=1.))
        dataset['show_rejected'] = True
        for selection, labels in [('peaks', {'Peaks'}), ('dips', {'Dips'}), ('both', {'Peaks', 'Dips'})]:
            dataset['settings']['polarity'] = selection
            dataset['result'] = filter_heatmap(dataset, dataset['settings'])
            dataset['branches'] = dataset['result']['branches']
            before = deepcopy(dataset['result'])
            # An unapplied selection must not silently change saved positions.
            dataset['settings']['polarity'] = 'both' if selection != 'both' else 'peaks'
            for width in (1., 2.):
                with self.subTest(selection=selection, width=width):
                    with patch('core.peak_filter_preview.excluded_candidates', side_effect=AssertionError('Preview leaked into PNG')):
                        figure = render_source_figure(dataset, width_scale=width)
                    try:
                        markers = [artist for artist in figure.axes[0].collections if artist.get_gid() == 'peak-candidates']
                        self.assertEqual({artist.get_label() for artist in markers}, labels)
                        actual = sorted(tuple(xy) for artist in markers for xy in artist.get_offsets())
                        expected = sorted((point['energy'], point['y']) for point in before['points'])
                        self.assertEqual(actual, expected)
                        self.assertTrue(all(artist.get_sizes()[0] == 4 for artist in markers))
                        self.assertEqual(dataset['result'], before)
                        self.assertTrue(dataset['show_rejected'])
                    finally:
                        plt.close(figure)

    def test_three_region_actual_ranges_are_captured(self):
        from dataclasses import replace
        cube = replace(self.cube, energy=np.arange(1., 7.), Z=np.arange(12.).reshape(2, 6))
        split = SplitColorScale(2.5, 0., 3., 7., 11., split_x2=4.5, middle_vmin=3., middle_vmax=7.)
        params = replace(self.params, xlim=(1., 6.), split_scale=split)
        fig, axis = plt.subplots()
        try:
            plot_heatmap(axis, cube, params)
            for mesh, bounds in zip(axis.collections, [(0., 2.), (3., 6.), (8., 11.)]):
                mesh.set_clim(*bounds)
            owner = SimpleNamespace(_values={'scale': 'linear'}, _params=lambda scale: params)
            captured = capture_source_export(owner, 'pl', cube, axis)['params']['split_scale']
            self.assertEqual([(captured[s + '_vmin'], captured[s + '_vmax']) for s in ('left', 'middle', 'right')],
                             [(0., 2.), (3., 6.), (8., 11.)])
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
