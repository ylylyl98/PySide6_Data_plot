import tempfile
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from matplotlib.figure import Figure
from core.mcd_unified_export import _plot_pngs
from tests.test_mcd_unified_export_luna import _snapshot


class FinalExportAcceptanceTests(unittest.TestCase):
    def test_window_title_has_only_energy_and_width_for_signed_mean(self):
        snapshot = _snapshot()
        snapshot['windows'][0].update(id='window-0-1.641582', center_ev=1.641582, metric='Signed mean')
        snapshot['mcd']['energy_ev'] = __import__('numpy').array([1.6, 1.65, 1.7])
        fig = self.render(snapshot)['MCD_vs_B_r01.png']
        titles = [text.get_text() for text in fig.texts]
        self.assertIn('E = 1.641582 eV · W = 5 meV', titles)
        self.assertFalse(any('window-0-' in text or 'Signed mean' in text for text in titles))

    def test_inc_markers_are_filled_and_dec_markers_are_hollow(self):
        axis = self.render(_snapshot())['MCD_vs_B_r01.png'].axes[0]
        # This fixture draws decreasing first, increasing second.
        dec, inc = axis.collections[:2]
        self.assertEqual(len(dec.get_facecolors()), 0)
        self.assertGreater(len(inc.get_facecolors()), 0)
        self.assertGreater(len(dec.get_edgecolors()), 0)

    def test_fit_spans_full_x_without_expanding_data_y_limits(self):
        snapshot = _snapshot()
        snapshot['slopes'] = []
        baseline = self.render(snapshot)['MCD_vs_B_r01.png'].axes[0]
        snapshot['slopes'] = [{'branch': 'B increasing', 'region': 'high_positive',
                               'slope': 100., 'intercept': 0., 'field_min_t': .1,
                               'field_max_t': .2, 'status': 'ok'}]
        axis = self.render(snapshot)['MCD_vs_B_r01.png'].axes[0]
        self.assertEqual(axis.get_ylim(), baseline.get_ylim())
        self.assertEqual(axis.get_xlim(), baseline.get_xlim())
        lines = [line for line in axis.lines if line.get_zorder() == 5]
        self.assertEqual(min(min(line.get_xdata()) for line in lines), axis.get_xlim()[0])
        self.assertEqual(max(max(line.get_xdata()) for line in lines), axis.get_xlim()[1])
        self.assertEqual({line.get_color() for line in lines}, {'#7950a3'})
        self.assertEqual(sorted(line.get_alpha() for line in lines), [.32, .32, 1.])

    def test_production_high_field_region_names_show_all_six_slopes(self):
        snapshot = _snapshot()
        snapshot['slopes'] = [
            {'branch': branch, 'region': region, 'slope': value,
             'intercept': 0., 'field_min_t': -1., 'field_max_t': 1., 'status': 'ok'}
            for region, branch, value in (
                ('low', 'B increasing', .111), ('low', 'B decreasing', .112),
                ('high_negative', 'B increasing', .211), ('high_negative', 'B decreasing', .212),
                ('high_positive', 'B increasing', .311), ('high_positive', 'B decreasing', .312))
        ]
        axis = self.render(snapshot)['MCD_vs_B_r01.png'].axes[0]
        text = [item.get_text() for item in axis.texts]
        self.assertNotIn('N/A', text)
        for value in ('0.111', '0.112', '0.211', '0.212', '0.311', '0.312'):
            self.assertIn(value, text)
        fit_colors = {line.get_color() for line in axis.lines if line.get_zorder() == 5}
        self.assertEqual(fit_colors, {'#806015', '#bd3651', '#7950a3'})

    def render(self, snapshot):
        rendered = {}
        with tempfile.TemporaryDirectory() as directory, patch.object(
            Figure, "savefig", lambda fig, path, **kwargs: rendered.update({Path(path).name: fig})
        ):
            _plot_pngs(Path(directory), snapshot["mcd"], snapshot, "r01")
        return rendered

    def test_integral_and_mean_have_separate_figures_with_owned_slopes(self):
        snapshot = _snapshot()
        snapshot["windows"] = [
            {"id": "mean-window", "center_ev": 1.6, "width_mev": 5., "metric": "Signed mean", "values": [.1, .2, .3]},
            {"id": "area-window", "center_ev": 1.6, "width_mev": 5., "metric": "Signed integral", "values": [.001, .002, .003]},
        ]
        snapshot["slopes"] = [
            {"window_id": "mean-window", "branch": "B increasing", "region": "high+", "slope": .1, "intercept": .2, "actual_range_t": [0., 1.], "status": "ok"},
            {"window_id": "area-window", "branch": "B increasing", "region": "high-", "slope": .001, "intercept": .002, "actual_range_t": [0., 1.], "status": "ok"},
        ]
        figures = [fig for name, fig in self.render(snapshot).items() if name.startswith("MCD_vs_B")]
        self.assertEqual(len(figures), 2)
        self.assertEqual([fig.axes[0].get_ylabel() for fig in figures], ["MCD", "Integrated MCD (eV)"])
        for fig, identity, slope_unit in zip(figures, ("mean-window", "area-window"), ("T⁻¹", "eV T⁻¹")):
            text = " ".join(t.get_text() for t in fig.texts + fig.axes[0].texts)
            self.assertIn('E = 1.6 eV · W = 5 meV', text)
            self.assertIn(slope_unit, text)
            labels = fig.axes[0].get_legend_handles_labels()[1]
            self.assertEqual(labels, ["Dec", "Inc"])
        self.assertNotIn("Corrected mean", " ".join(t.get_text() for t in figures[1].texts))

    def test_single_integral_does_not_claim_mean_or_dimensionless_slopes(self):
        snapshot = _snapshot()
        snapshot["windows"][0]["metric"] = "Signed integral"
        fig = self.render(snapshot)["MCD_vs_B_r01.png"]
        self.assertEqual(fig.axes[0].get_ylabel(), "Integrated MCD (eV)")
        self.assertIn("eV T⁻¹", " ".join(t.get_text() for t in fig.axes[0].texts))
        self.assertNotIn("mean", " ".join(t.get_text() for t in fig.texts).casefold())

    def test_four_channel_tracks_have_compact_legend_clear_of_measured_points(self):
        snapshot = _snapshot()
        base = snapshot["analysis_results"][0]["tracks"]
        tracks = []
        for index, (channel, branch) in enumerate((
            ("raw pos", "B increasing"), ("raw pos", "B decreasing"),
            ("raw neg", "B increasing"), ("raw neg", "B decreasing"),
        )):
            track = copy.deepcopy(base[0 if "pos" in channel else 1])
            track.update(feature_id=f"long-unique-feature-identifier-{index}", channel=channel, branch=branch, reference_method="interpolated near-zero")
            tracks.append(track)
        snapshot["analysis_results"][0]["tracks"] = tracks
        fig = self.render(snapshot)["feature_energy_r01.png"]
        fig.canvas.draw()
        axis = fig.axes[0]
        legend = axis.get_legend()
        box = legend.get_window_extent(fig.canvas.get_renderer())
        self.assertLess(box.width, axis.bbox.width * .6)
        self.assertLess(box.height, axis.bbox.height * .16)
        self.assertTrue(all(t.get_fontsize() == 16 for t in legend.get_texts()))
        for line in axis.lines:
            self.assertFalse(any(box.contains(x, y) for x, y in axis.transData.transform(line.get_xydata())))
