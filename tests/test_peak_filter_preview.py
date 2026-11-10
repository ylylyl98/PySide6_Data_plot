"""Applied filter explanations, drawing parity, and coordinate-aware inspection."""
from copy import deepcopy
import unittest

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from core.peak_candidates import filter_heatmap
from core.peak_filter_preview import excluded_candidates
from core.peak_plotting import draw_overlay, draw_spectrum, overlay_payload
from tests.test_peak_auto_selection import _dataset
from ui_qt.quick.peak_analysis_preview import CandidatePreviewIndex


class FilterPreviewTests(unittest.TestCase):
    def fixture(self):
        dataset = _dataset(np.arange(9.))
        dataset['settings'].update(min_snr=0., prominence=0., min_width_mev=0.,
            min_distance_mev=0., neighbor_support=0, max_width_mev=0., max_per_row=0)
        return dataset

    def apply(self, dataset):
        dataset['result'] = filter_heatmap(dataset, dataset['settings'])
        dataset['branches'] = dataset['result']['branches']
        return dataset['result']['filter_preview']

    def test_primary_reasons_cover_quality_support_spacing_and_limit(self):
        for key, value, reason in [('min_snr', 5., 'snr'), ('prominence', 2., 'prominence'),
                ('min_width_mev', 3., 'min_width'), ('max_width_mev', 1., 'max_width'),
                ('max_per_row', 1, 'limit'), ('min_distance_mev', 100., 'spacing'),
                ('neighbor_support', 5, 'neighbor')]:
            with self.subTest(reason=reason):
                dataset = self.fixture()
                dataset['settings'][key] = value
                dataset['settings']['neighbor_tolerance_mev'] = .2
                before = deepcopy(dataset['candidates'])
                diagnostics = self.apply(dataset)
                self.assertIn(reason, diagnostics['counts'])
                self.assertEqual(set(diagnostics['reasons'].values()), {reason})
                self.assertEqual(sum(diagnostics['counts'].values()), len(diagnostics['reasons']))
                self.assertEqual(dataset['candidates'], before)
                self.assertEqual(dataset['result']['filter_counts']['kept'], len(dataset['result']['points']))

    def test_preview_uses_applied_range_and_type_and_does_not_change_results(self):
        dataset = self.fixture()
        dataset['settings'].update(min_snr=20., x_min=1.015, x_max=1.025, y_min=2., y_max=5., polarity='peaks')
        self.apply(dataset)
        dataset['show_rejected'] = True
        preview = excluded_candidates(dataset)
        self.assertEqual(len(preview), 4)
        dataset['settings'].update(x_min=1., x_max=1.1, polarity='both', min_snr=0.)
        self.assertEqual(excluded_candidates(dataset), preview, 'Unapplied drafts must not move the preview')
        axis = Figure().add_subplot()
        artists = draw_overlay(axis, dataset)
        self.assertEqual(len(artists[0].get_offsets()), 4)
        packet = overlay_payload(dataset)
        self.assertEqual(len(draw_overlay(Figure().add_subplot(), packet)[0].get_offsets()), 4)
        self.assertEqual(dataset['result']['points'], [])

    def test_cursor_lookup_tracks_zoom_and_reports_applied_reason(self):
        dataset = self.fixture()
        dataset['settings']['min_snr'] = 5.
        self.apply(dataset)
        index = CandidatePreviewIndex(dataset)
        figure = Figure(figsize=(8, 5)); FigureCanvasAgg(figure)
        axis = figure.add_subplot()
        axis.set(xlim=(1., 1.1), ylim=(-1., 10.))
        figure.canvas.draw()
        weak = next(p for p in dataset['candidates']['points'] if p['candidate_id'] == 'weak:4')
        point, reason = index.nearest(axis, weak['energy'], weak['y'])
        self.assertEqual((point['candidate_id'], reason), ('weak:4', 'snr'))
        axis.set(xlim=(1.05, 1.065), ylim=(3., 5.))
        self.assertEqual(index.nearest(axis, weak['energy'], weak['y'])[1], 'snr')
        self.assertIsNone(index.nearest(axis, 1.055, 3.5))
        axis.set(ylim=(-1., 1.))
        self.assertEqual(index.nearest(axis, weak['energy'], 0., spectrum_row=4)[1], 'snr')

    def test_spectrum_preview_is_bounded_and_matches_export_positions(self):
        from ui_qt.quick.peak_analysis_spectrum import RetainedPeakSpectrum
        dataset = self.fixture(); dataset['settings']['min_snr'] = 5.
        self.apply(dataset); dataset['show_rejected'] = True
        figure = Figure(); FigureCanvasAgg(figure); axis = figure.add_subplot()
        draw_spectrum(dataset, 4, axis)
        expected = next(a for a in axis.collections if a.get_gid() == 'peak-rejected').get_offsets().copy()
        palette = dict(inset='white', border='gray', text='black')
        retained = RetainedPeakSpectrum(axis, None, dataset, 4, palette)
        np.testing.assert_allclose(retained.excluded_markers.get_offsets(), expected)
        marker = retained.excluded_markers
        for row in (0, 4, 8, 2):
            retained.update(dataset, row, palette)
            self.assertIs(retained.excluded_markers, marker)
            self.assertEqual(len(marker.get_offsets()), 1)
        dataset['show_rejected'] = False
        retained.update(dataset, 4, palette)
        self.assertFalse(marker.get_visible())

    def test_legacy_preview_does_not_claim_no_exclusions_and_hidden_points_do_not_hover(self):
        from core.peak_filter_preview import preview_summary
        dataset = self.fixture(); dataset['settings']['min_snr'] = 5.
        self.apply(dataset)
        point = next(p for p in dataset['result']['points'] if p['candidate_id'] == 'strong:4')
        for branch in dataset['branches']:
            if branch['id'] == point['branch_id']:
                branch['enabled'] = False
        index = CandidatePreviewIndex(dataset)
        figure = Figure(); FigureCanvasAgg(figure); axis = figure.add_subplot()
        axis.set(xlim=(1., 1.1), ylim=(-1., 10.)); figure.canvas.draw()
        self.assertIsNone(index.nearest(axis, point['energy'], point['y']))
        dataset['result'].pop('filter_preview')
        self.assertTrue(excluded_candidates(dataset))
        self.assertIn('unavailable', preview_summary(dataset['result']))
