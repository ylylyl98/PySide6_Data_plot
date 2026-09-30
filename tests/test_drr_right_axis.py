import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from core import plotting
from core.loader import DataCube


class RightAxisTests(unittest.TestCase):
    def test_mapping_rejects_constant_and_folded_coordinates(self):
        for values in [(0., 0., 0.), (0., 1., 0.)]:
            with self.assertRaisesRegex(ValueError, 'one-to-one'):
                plotting.RightAxis('Bias', (-1., 0., 1.), values)

    def test_inconsistent_bias_across_files_is_unavailable(self):
        from core.drr_axis_coordinates import load_coordinates
        with tempfile.TemporaryDirectory() as tmp:
            for name, bias in [('a.csv', '1'), ('b.csv', '2')]:
                Path(tmp, name).write_text(f'Vbg,Vtg,Vbias,600,700\n0,0,0,10,20\n1,1,{bias},11,21\n')
            values = load_coordinates(tmp, ['a.csv', 'b.csv'], 'tg', np.array([0., 1.]))
        self.assertNotIn('Bias', values)
        np.testing.assert_allclose(values['TG+BG'], [0., 2.])

    def test_finer_scan_cannot_hide_folded_or_conflicting_bias(self):
        from core.drr_axis_coordinates import load_coordinates
        for middle in [3., .5]:
            with tempfile.TemporaryDirectory() as tmp:
                Path(tmp, 'a.csv').write_text('Vbg,Vtg,Vbias,600,700\n0,0,0,10,20\n2,2,2,11,21\n')
                Path(tmp, 'b.csv').write_text(f'Vbg,Vtg,Vbias,600,700\n0,0,0,10,20\n1,1,{middle},11,21\n2,2,2,12,22\n')
                values = load_coordinates(tmp, ['a.csv', 'b.csv'], 'tg', np.array([0., 2.]))
            self.assertNotIn('Bias', values)

    def test_mapping_tracks_zoom_and_export(self):
        self.assertTrue(hasattr(plotting, 'RightAxis'), 'Missing measured right-axis mapping')
        mapping = plotting.RightAxis('Bias (V)', (-1.5, 0., 1.5), (.965, .990, 1.015))
        cube = DataCube(np.array([1., 2.]), np.array([-1.5, 0., 1.5]), np.ones((3, 2)), 'TG+BG (V)', 'map', 'DR/R')
        params = plotting.HeatmapParams('map', 'Energy', cube.gate_label, 'DR/R', 0, 2, (1, 2), (-1.5, 1.5), right_axis=mapping)
        fig = Figure(); FigureCanvasAgg(fig)
        ax = fig.subplots(); plotting.plot_heatmap(ax, cube, params)
        ax.set_ylim(-.75, .75); fig.canvas.draw()
        np.testing.assert_allclose(ax.child_axes[0].get_ylim(), [.9775, 1.0025])
        from core.export import _build_streamlit_style_heatmap_fig
        exported = _build_streamlit_style_heatmap_fig(cube, params, drr=True)
        exported.canvas.draw()
        self.assertEqual(exported.axes[0].child_axes[0].get_ylabel(), 'Bias (V)')
        np.testing.assert_allclose(exported.axes[0].child_axes[0].get_ylim(), [.965, 1.015])
        from matplotlib import pyplot as plt
        plt.close(exported)

    def test_reversed_sweep_keeps_bias_paired_with_left_axis(self):
        from core import drr_axis_coordinates as axes
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'scan.csv').write_text('Vbg,Vtg,Vbias,600,700\n.75,.75,.965,10,20\n0,0,.990,11,21\n-.75,-.75,1.015,12,22\n')
            values = axes.load_coordinates(tmp, ['scan.csv'], 'linear:1,1,0', np.array([-1.5, 0, 1.5]))
        np.testing.assert_allclose(values['Bias'], [1.015, .990, .965])


class RightAxisUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_axis_reload_uses_new_range_and_right_axis_control(self):
        from PySide6.QtCore import QSettings
        from ui_qt.main_window import MainWindow
        from ui_qt.common import LoadedState
        with tempfile.TemporaryDirectory() as tmp:
            with patch('ui_qt.main_window.QSettings', return_value=QSettings(str(Path(tmp)/'settings.ini'), QSettings.IniFormat)), patch.object(MainWindow, '_restore_last_folder', autospec=True), patch.object(MainWindow, '_schedule_automatic_update_check', autospec=True):
                w = MainWindow()
            try:
                from ui_qt.common import LoadOptions, WorkerSignals
                from core.drr_sources import DrrMeasurementAssignment
                Path(tmp, 'scan.csv').write_text('Vbg,Vtg,Vbias,600,650,700\n-.75,-.75,.965,10,20,30\n0,0,.990,11,21,31\n.75,.75,1.015,12,22,32\n')
                options = LoadOptions('DRR', tmp, ['scan.csv'], [], False, 'Self (last frame)', 'last', False,
                                      y_axis_spec='linear:1,1,0', drr_assignments=(DrrMeasurementAssignment('scan.csv'),))
                signals = WorkerSignals()
                loaded = w._load_task(options, progress=signals.progress, log=signals.log)
                np.testing.assert_allclose(loaded.cube.gate, [-1.5, 0., 1.5])
                np.testing.assert_allclose(loaded.drr_axis_coordinates['Bias'], [.965, .990, 1.015])
                w._ensure_loaded_matches_ui_params = lambda mode: False
                for axis, gates in [('tg', [-.75, 0, .75]), ('bias', [.965, .990, 1.015])]:
                    cube = DataCube(np.linspace(1, 2, 21), np.array(gates), np.ones((3, 21)), axis, 'map', 'DR/R')
                    w._on_loaded(LoadedState(mode='DRR', folder=tmp, primary_file='scan.csv', selected_files=['scan.csv'], cube=cube, y_axis_spec=axis))
                np.testing.assert_allclose(w._drr_heatmap_ax.get_ylim(), [.965, 1.015])
                self.assertTrue(hasattr(w, 'drr_right_yaxis_combo'))
                self.assertEqual(w.drr_right_yaxis_combo.currentText(), 'Off')
                w.drr_yaxis_combo.setCurrentText('TG+BG')
                self.assertEqual(w._selected_y_axis_spec('drr'), 'linear:1,1,0')
                w.loaded.drr_axis_coordinates = {'TG+BG': np.array([-1.5, 0., 1.5])}
                w._update_drr_right_axis_options()
                w.drr_right_yaxis_combo.setCurrentText('TG+BG')
                w._plot_mode('DRR')
                w.canvas.draw()
                np.testing.assert_allclose(w._drr_heatmap_ax.child_axes[0].get_ylim(), [-1.5, 1.5])
                # Color updates reuse the map without duplicating the secondary axis.
                w.drr_spins['vmax'].setValue(3)
                w._plot_mode('DRR')
                self.assertEqual(len(w._drr_heatmap_ax.child_axes), 1)
                w.drr_view_side_btn.setChecked(True)
                self.assertTrue(w.thread_pool.waitForDone(5000))
                with patch.object(w, '_active_mode', return_value='DRR'):
                    self.app.processEvents()
                    w._plot_mode('DRR')
                    w.canvas.draw()
                self.assertEqual(len(w._drr_heatmap_axes), 2)
                for ax in w._drr_heatmap_axes.values():
                    np.testing.assert_allclose(ax.child_axes[0].get_ylim(), [-1.5, 1.5])
                w.drr_right_yaxis_combo.setCurrentText('Off')
                w._plot_mode('DRR')
                self.assertEqual(len(w._drr_heatmap_ax.child_axes), 0)
            finally:
                w.close(); w.deleteLater(); self.app.processEvents()
