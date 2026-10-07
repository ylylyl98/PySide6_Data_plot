import unittest

import numpy as np
from matplotlib.figure import Figure

from core.peak_plotting import draw_spectrum
from core.peak_workspace import create_dataset
from tests.test_peak_workspace import sample


class CountingPoints(list):
    visits = 0

    def __iter__(self):
        for point in super().__iter__():
            self.visits += 1
            yield point


class SpectrumPlotTests(unittest.TestCase):
    def test_small_marker_size_is_shared_by_overlay_packet_and_plot(self):
        from core.peak_candidates import detect_heatmap,filter_heatmap
        from core.peak_plotting import draw_overlay,overlay_payload
        d=create_dataset(sample(),'PL','PL','a.csv','Map')
        d['candidates']=detect_heatmap(d,d['settings'])
        d['result']=filter_heatmap(d,d['settings']);d['branches']=d['result']['branches']
        axis=Figure().add_subplot();artists=draw_overlay(axis,d)
        self.assertEqual(float(artists[0].get_sizes()[0]),9)
        d['marker_size']=4
        packet=overlay_payload(d)
        self.assertEqual(packet['marker_size'],4)
        artists=draw_overlay(Figure().add_subplot(),packet)
        self.assertEqual(float(artists[0].get_sizes()[0]),4)

    def test_many_branches_do_not_rescan_map_for_each_spectrum_marker(self):
        d = create_dataset(sample(), 'PL', 'PL', 'a.csv', 'Map')
        d['branches'] = [dict(id=str(i), name=f'P{i}', color=i, enabled=i != 2,
                              seed_row=1, seed_energy=1.65) for i in range(60)]
        d['branches'].append(dict(id='manual', name='Manual', color=60,
                                  enabled=True, seed_row=0, seed_energy=1.72))
        points = CountingPoints(dict(branch_id=str(i), row_index=0 if i < 4 else 1,
                                     energy=1.64 + i * .001) for i in range(60))
        # The first point for a branch and row remains the displayed point.
        points.append(dict(branch_id='1', row_index=0, energy=1.79))
        d['result'] = dict(points=points)
        ax = Figure().add_subplot()

        draw_spectrum(d, 0, ax)

        self.assertEqual([label.get_text() for label in ax.texts], ['P0', 'P1', 'P3', 'Manual'])
        np.testing.assert_allclose([artist.get_offsets()[0, 0] for artist in ax.collections],
                                   [1.64, 1.641, 1.643, 1.72])
        self.assertLessEqual(points.visits, 2 * len(points),
                             'Changing spectrum must not scan all map points for every branch')
