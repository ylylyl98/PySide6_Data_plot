"""Landscape exports preserve the data, height and original 1x layout."""
import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolBar

from core import export
from core.loader import DataCube
from core.plotting import HeatmapParams, RightAxis, SplitColorScale
from core.png_geometry import widened_figsize
from ui_qt.common import ExportOptions
from ui_qt.png_export_controls import apply_export_width, export_width, install_png_width_control


def sample():
    cube = DataCube(np.linspace(1., 2., 9), np.linspace(-1., 1., 5),
                    np.arange(45.).reshape(5, 9) / 45, 'Gate', 'Sample', 'DR/R')
    params = HeatmapParams('Sample', 'Photon energy (eV)', 'Gate (V)', 'DR/R',
                           0., 1., (1., 2.), (-1., 1.))
    return cube, params


class PngWidthTests(unittest.TestCase):
    def test_wide_header_has_a_small_title_gap_and_no_overlaps(self):
        cube, base = sample()
        split = SplitColorScale(1.3, 0., 1., .1, .9,
                                split_x2=1.7, middle_vmin=.2, middle_vmax=.8)
        for drr in (False, True):
            for regions in (None, replace(split, split_x2=None), split):
                for text in ('Sample', 'Sample 720nmc custom conditions repeat03 ' * 4, ''):
                    with self.subTest(drr=drr, regions=regions, title=text):
                        params = replace(base, title=text, split_scale=regions, png_width_scale=2.)
                        fig = export._build_streamlit_style_heatmap_fig(cube, params, drr=drr)
                        try:
                            fig.canvas.draw()
                            renderer = fig.canvas.get_renderer()
                            title = fig.texts[0].get_window_extent(renderer)
                            gap = title.y0 - fig.axes[0].bbox.y1
                            self.assertGreaterEqual(gap, 6.)
                            self.assertLessEqual(gap, 10.)
                            headers = [axis.get_tightbbox(renderer) for axis in fig.axes[1:]]
                            headers += [artist.get_window_extent(renderer) for artist in fig.texts[1:]]
                            self.assertTrue(all(not title.overlaps(box) for box in headers))
                            for box in [title, *headers]:
                                self.assertGreaterEqual(box.y0, fig.axes[0].bbox.y1)
                                self.assertLessEqual(box.y1, fig.bbox.height - 19.)
                                self.assertGreaterEqual(box.x0, 0.)
                                self.assertLessEqual(box.x1, fig.bbox.width)
                            if text:
                                self.assertEqual(' '.join(fig.texts[0].get_text().split()), text.strip())
                        finally:
                            plt.close(fig)

    def test_old_wide_render_is_refreshed_without_rewriting_dat(self):
        cube, params = sample()
        params = replace(params, png_width_scale=2.)
        fingerprint = export._fingerprint_json

        def old_header_fingerprint(payload):
            if isinstance(payload, dict) and 'wide_header_version' in payload:
                payload = {**payload, 'wide_header_version': 1}
            return fingerprint(payload)

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'sample.csv'
            source.write_text('source', encoding='utf-8')
            kwargs = dict(folder=folder, cube=cube, params=params, export_base='sample',
                          metadata_input_files=(('measurement', source.name),))
            with patch.object(export, '_fingerprint_json', side_effect=old_header_fingerprint):
                first = export.export_drr_png_and_dat(**kwargs)
            modified = first['dat'].stat().st_mtime_ns
            updated = export.export_drr_png_and_dat(**kwargs)
            self.assertEqual(updated.save_status, 'updated')
            self.assertEqual(updated['dat'], first['dat'])
            self.assertEqual(updated['dat'].stat().st_mtime_ns, modified)
            self.assertNotEqual(updated['png'], first['png'])
            self.assertTrue(first['png'].exists())

    def test_rendered_sizes_and_preserved_axes_for_split_and_right_axis(self):
        cube, params = sample()
        right = RightAxis('Top gate (V)', (-1., 0., 1.), (-2., 0., 2.))
        variants = [params, replace(params, right_axis=right),
                    replace(params, split_scale=SplitColorScale(1.4, 0., 1., .1, .9)),
                    replace(params, split_scale=SplitColorScale(
                        1.3, 0., 1., .1, .9, split_x2=1.7, middle_vmin=.2, middle_vmax=.8))]
        with tempfile.TemporaryDirectory() as folder:
            for drr in (False, True):
                for variant in variants:
                    for scale in (1., 1.5, 2.):
                        with self.subTest(drr=drr, regions=variant.split_scale, scale=scale):
                            fig = export._build_streamlit_style_heatmap_fig(
                                cube, replace(variant, png_width_scale=scale), drr=drr)
                            try:
                                np.testing.assert_allclose(fig.axes[0].get_xlim(), params.xlim)
                                np.testing.assert_allclose(fig.axes[0].get_ylim(), params.ylim)
                                bounds = fig.axes[0].get_position().bounds
                                if scale == 1.:
                                    np.testing.assert_allclose(bounds, export.EXPORT_AXES_RECT)
                                else:
                                    self.assertLess(bounds[0], export.EXPORT_AXES_RECT[0])
                                    self.assertGreater(bounds[2], export.EXPORT_AXES_RECT[2])
                                    self.assertGreater(bounds[3], export.EXPORT_AXES_RECT[3])
                                self.assertEqual(fig.axes[0].xaxis.label.get_fontsize(), 16)
                                path = Path(folder) / 'plot.png'
                                fig.savefig(path, dpi=fig.dpi, bbox_inches=fig.bbox_inches)
                                if scale > 1.:
                                    renderer = fig.canvas.get_renderer()
                                    boxes = [axis.get_tightbbox(renderer) for axis in fig.axes]
                                    boxes += [text.get_window_extent(renderer) for text in fig.texts]
                                    self.assertAlmostEqual(min(box.x0 for box in boxes),
                                                           10 * fig.dpi / 72, delta=1.)
                                    self.assertAlmostEqual(fig.bbox.width - max(box.x1 for box in boxes),
                                                           10 * fig.dpi / 72, delta=1.)
                                with Image.open(path) as image:
                                    self.assertEqual(image.size, (int(1200 * scale), 930))
                            finally:
                                plt.close(fig)

    def test_drr_width_change_keeps_dat_and_original_png_and_reuses_wide_png(self):
        cube, params = sample()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'sample.csv'
            source.write_text('source', encoding='utf-8')
            kwargs = dict(folder=folder, cube=cube, export_base='sample',
                          metadata_input_files=(('measurement', source.name),))
            first = export.export_drr_png_and_dat(params=params, **kwargs)
            dat_bytes = first['dat'].read_bytes()
            modified = first['dat'].stat().st_mtime_ns
            wide = export.export_drr_png_and_dat(params=replace(params, png_width_scale=2.), **kwargs)
            self.assertEqual(wide['dat'], first['dat'])
            self.assertEqual(wide['dat'].read_bytes(), dat_bytes)
            self.assertEqual(wide['dat'].stat().st_mtime_ns, modified)
            self.assertNotEqual(wide['png'], first['png'])
            for path, size in ((first['png'], (1200, 930)), (wide['png'], (2400, 930))):
                with Image.open(path) as image:
                    self.assertEqual(image.size, size)
            metadata = json.loads(wide['dat'].with_suffix('.metadata.json').read_text())
            self.assertEqual(metadata['plot']['png_width_scale'], 2.)
            repeated = export.export_drr_png_and_dat(params=replace(params, png_width_scale=2.), **kwargs)
            self.assertEqual(repeated['png'], wide['png'])
            self.assertEqual(repeated.save_status, 'reused')

    def test_pl_pair_compare_vp_and_power_exports_propagate_width(self):
        cube, params = sample()
        params = replace(params, png_width_scale=1.5)
        with tempfile.TemporaryDirectory() as folder:
            pl = export.export_pl_pngs_and_dat(folder, 'sample.csv', cube_linear=cube,
                cube_log=cube, params_linear=params, params_log=replace(params, log_scale=True, vmin=.01),
                include_pair=True)
            compare = export.export_compare_panels(folder, cubes={'KK': cube, 'KKp': cube},
                source_files={'KK': 'KK.csv', 'KKp': 'KKp.csv'}, params=params,
                scale_tag='linear', clip_outliers=False)
            power = export.export_power_vp_pngs_and_dat(folder, kk_cube=cube, kkp_cube=cube,
                vp_cube=cube, params=params, kk_group_key='KK', kkp_group_key='KKp',
                kk_records=(), kkp_records=(), y_axis_log=False, background=0.)
            for path in [*pl.values(), *compare, *power.values()]:
                if path.suffix != '.png':
                    continue
                with Image.open(path) as image:
                    self.assertEqual(image.size, (3600 if path == pl['png_pair'] else 1800, 930))

    def test_unified_mcd_figures_use_saved_width(self):
        from core.mcd_unified_export import _mcd_data, _plot_pngs
        from tests.test_mcd_unified_export import _snapshot
        snapshot = _snapshot()
        snapshot.setdefault('plot_state', {})['png_width_scale'] = 1.5
        with tempfile.TemporaryDirectory() as folder:
            paths = _plot_pngs(Path(folder), _mcd_data(snapshot), snapshot, 'wide')
            self.assertGreater(len(paths), 1)
            for path in paths:
                with Image.open(path) as image:
                    self.assertEqual(image.size, (1800, 900))

    def test_all_export_products_freeze_width_without_mutating_live_params(self):
        _, params = sample()
        fields = ('params', 'params_linear', 'params_log', 'params_intensity',
                  'drr_raw_params', 'drr_second_params')
        options = ExportOptions(mode='DRR', **dict.fromkeys(fields, params))
        wide = apply_export_width(options, 2.)
        for name in fields:
            self.assertEqual(getattr(wide, name).png_width_scale, 2.)
            self.assertEqual(getattr(options, name).png_width_scale, 1.)
        for bad in (0., -1., 3.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                widened_figsize((8, 6.2), bad)


class PngWidthUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_control_commits_keyboard_value_and_remembers_it(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = QSettings(str(Path(folder) / 'settings.ini'), QSettings.IniFormat)
            owner = SimpleNamespace(settings=settings)
            toolbar = QToolBar()
            install_png_width_control(owner, toolbar)
            toolbar.show()
            spin = owner.png_width_spin
            self.assertEqual(spin.value(), 2.)
            self.assertEqual(export_width(owner), 1.)
            self.assertFalse(spin.isEnabled())
            owner.png_layout_combo.setFocus()
            QTest.keyClick(owner.png_layout_combo, Qt.Key_Down)
            self.assertTrue(spin.isEnabled())
            self.assertEqual(export_width(owner), 2.)
            spin.setFocus()
            spin.selectAll()
            QTest.keyClicks(spin, '1.5')
            self.assertEqual(spin.value(), 2.)
            QTest.keyClick(spin, Qt.Key_Return)
            self.assertEqual(spin.value(), 1.5)
            second = SimpleNamespace(settings=settings)
            other_toolbar = QToolBar()
            install_png_width_control(second, other_toolbar)
            self.assertEqual(second.png_width_spin.value(), 1.5)
            self.assertEqual(export_width(second), 1.5)
            self.assertTrue(second.png_width_spin.isEnabled())
            owner.png_layout_combo.setCurrentIndex(0)
            self.assertEqual(export_width(owner), 1.)
            self.assertEqual(spin.value(), 1.5)
            third = SimpleNamespace(settings=QSettings(str(Path(folder) / 'settings.ini'), QSettings.IniFormat))
            third_toolbar = QToolBar()
            install_png_width_control(third, third_toolbar)
            self.assertEqual(export_width(third), 1.)
            self.assertEqual(third.png_width_spin.value(), 1.5)
            self.assertFalse(third.png_width_spin.isEnabled())
            toolbar.close()
            other_toolbar.close()
            third_toolbar.close()

    def test_invalid_saved_layout_uses_original_without_losing_width(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = QSettings(str(Path(folder) / 'settings.ini'), QSettings.IniFormat)
            settings.setValue('export/png_layout', 'invalid')
            settings.setValue('export/png_width_scale', 2.5)
            owner = SimpleNamespace(settings=settings)
            toolbar = QToolBar()
            install_png_width_control(owner, toolbar)
            self.assertEqual(export_width(owner), 1.)
            self.assertFalse(owner.png_width_spin.isEnabled())
            self.assertEqual(owner.png_width_spin.value(), 2.5)
            toolbar.close()

    def test_old_or_invalid_width_never_enables_wide_export_at_startup(self):
        for stored, expected in ((1., 2.), (2.5, 2.5), ('invalid', 2.), ('nan', 2.)):
            with self.subTest(stored=stored), tempfile.TemporaryDirectory() as folder:
                settings = QSettings(str(Path(folder) / 'settings.ini'), QSettings.IniFormat)
                settings.setValue('export/png_width_scale', stored)
                owner = SimpleNamespace(settings=settings)
                toolbar = QToolBar()
                install_png_width_control(owner, toolbar)
                self.assertEqual(export_width(owner), 1.)
                owner.png_layout_combo.setCurrentIndex(1)
                self.assertEqual(export_width(owner), expected)
                toolbar.close()

    def test_async_save_widens_snapshot_without_resizing_live_figure(self):
        from ui_qt.async_figure_save import save_figure_async
        with tempfile.TemporaryDirectory() as folder:
            fig = Figure(figsize=(4, 3), dpi=100)
            ax = fig.subplots()
            ax.plot([0, 1], [2, 3])
            path = Path(folder) / 'wide.png'
            job = save_figure_async(fig, path, width_scale=2., dpi=100)
            np.testing.assert_array_equal(fig.get_size_inches(), (4, 3))
            fig.clear()
            deadline = time.monotonic() + 15
            while not job.done and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.005)
            self.assertTrue(job.done)
            self.assertIsNone(job.error)
            with Image.open(path) as image:
                self.assertEqual(image.size, (800, 300))

    def test_main_toolbar_control_is_visible_and_saves_a_wide_png(self):
        from ui_qt.main_window import MainWindow
        with tempfile.TemporaryDirectory() as folder:
            settings = QSettings(str(Path(folder) / 'settings.ini'), QSettings.IniFormat)
            with patch('ui_qt.main_window.QSettings', return_value=settings), \
                    patch.object(MainWindow, '_restore_last_folder', lambda s: None), \
                    patch.object(MainWindow, '_schedule_automatic_update_check', lambda s: None):
                window = MainWindow()
            try:
                window.resize(1180, 800)
                window.last_plotted_mode = 'PL'
                window.figure.subplots().plot([0, 1], [2, 3])
                window.toolbar.locLabel.setText('Photon energy = 1.234567, Gate = -123.456789')
                window.show()
                self.app.processEvents()
                self.assertTrue(window.png_layout_combo.isVisible())
                self.assertTrue(window.png_width_spin.isVisible())
                original_size = window.figure.get_size_inches().copy()
                dpi = window.figure.dpi
                original_bounds = window.figure.axes[0].get_position().bounds
                for index, (layout, scale) in enumerate(((0, 1), (1, 2), (0, 1))):
                    window.png_layout_combo.setCurrentIndex(layout)
                    target = Path(folder) / f'toolbar-{index}.png'
                    with patch('ui_qt.main_window.QFileDialog.getSaveFileName',
                               return_value=(str(target), 'PNG (*.png)')):
                        window.toolbar.save_figure()
                    job = getattr(window.toolbar, '_png_save_job', None)
                    if job is not None:
                        deadline = time.monotonic() + 15
                        while not job.done and time.monotonic() < deadline:
                            self.app.processEvents()
                            time.sleep(.005)
                        self.assertTrue(job.done)
                        self.assertIsNone(job.error)
                    np.testing.assert_array_equal(window.figure.get_size_inches(), original_size)
                    np.testing.assert_array_equal(window.figure.axes[0].get_position().bounds, original_bounds)
                    with Image.open(target) as image:
                        self.assertEqual(image.size, (int(original_size[0] * dpi * scale),
                                                      int(original_size[1] * dpi)))
            finally:
                window.close()
                window.deleteLater()
                self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
