from types import SimpleNamespace
import numpy as np
from matplotlib.figure import Figure
from core.mcd_energy_groups import energy_group_colors, energy_record_colors, draw_temperature_slope_panels


def rows():
    result = []
    for g in (1, 2):
        for t in (2., 3., 5.):
            result.append(SimpleNamespace(record_id=f'{g}-{t}', condition_value=lambda key,t=t: t if key=='T' else 0., slope=lambda b,m,t=t,g=g:g/t))
    return result


import unittest


class TemperatureGroupColorTests(unittest.TestCase):
    def test_group_hue_survives_subsets_and_new_groups(self):
        full = energy_group_colors({'a':'Group 1','b':'Group 2','c':'Custom'})
        for name in full:
            assert energy_group_colors({'x':name})[name] == full[name]
        assert full['Group 1'] != full['Group 2']


    def test_temperature_shades_and_group_slope_panels(self):
        records = rows()
        groups = {r.record_id:f'Group {r.record_id[0]}' for r in records}
        bases = energy_group_colors(groups)
        colors = energy_record_colors(records, groups, bases, variable='Temperature')
        assert sum(colors['1-2.0'][:3]) > sum(colors['1-5.0'][:3])
        assert colors['1-2.0'] != colors['2-2.0']
        fig = Figure()
        artists = draw_temperature_slope_panels(fig, records, groups, ('near_zero',), ('B increasing','B decreasing'))
        assert len(artists) == 4
        for artist,(members,metric,branch) in artists.items():
            assert len({groups[r.record_id] for r in members}) == 1
            np.testing.assert_allclose(artist.get_offsets()[:,0], [2,3,5])
            np.testing.assert_allclose(artist.get_edgecolors()[0], bases[groups[members[0].record_id]])
    def test_large_legend_fits_preview_canvas(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from core.mcd_energy_groups import temperature_curve_legend
        records = []
        for g in range(3):
            for t in range(10):
                records.append(SimpleNamespace(record_id=f'{g}-{t}', condition_value=lambda key,t=t: t))
        groups = {r.record_id:f'Group {r.record_id[0]}' for r in records}
        colors = energy_record_colors(records, groups, energy_group_colors(groups), variable='Temperature')
        fig = Figure(figsize=(9,5))
        FigureCanvasAgg(fig)
        fig.subplots()
        temperature_curve_legend(fig, records, groups, colors)
        fig.canvas.draw()
        box = fig.legends[0].get_window_extent(fig.canvas.get_renderer())
        assert box.x0 >= 0 and box.x1 <= fig.bbox.width
        assert box.y0 >= 0 and box.y1 <= fig.bbox.height

    def test_organizer_preview_export_keep_full_series_shades(self):
        import tempfile
        from pathlib import Path
        from dataclasses import replace
        from tests.test_curie_weiss_ui import records as fixture_records
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        from ui_qt.mcd_organizer_window import McdOrganizerWindow
        with tempfile.TemporaryDirectory() as folder:
            r = fixture_records(Path(folder))
            r += [replace(x, record_id='second-'+x.record_id, center_ev=1.58) for x in r]
            w = McdOrganizerWindow(folder, auto_scan=False)
            try:
                w.records = r
                w.compare_combo.setCurrentIndex(w.compare_combo.findData('Temperature'))
                w._regroup()
                s = w._current_series()
                expected = w._preview_energy_colors(s)
                assert expected == w._export_record_colors([s])
                w._selected_record_ids[s.series_id].discard(r[0].record_id)
                assert expected == w._preview_energy_colors(s)
                w._update_preview()
                assert r[0].record_id not in w._plot_artists
                for key, lines in w._plot_artists.items():
                    for line in lines:
                        np.testing.assert_allclose(line.get_color(), expected[key])
                assert w.slope_figure.axes[-1].get_xlabel() == 'Temperature (K)'
            finally:
                w.close()
