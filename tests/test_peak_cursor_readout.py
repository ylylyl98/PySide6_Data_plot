import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from matplotlib.backend_bases import MouseEvent, LocationEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.loader import DataCube
from core.peak_workspace import create_dataset
from ui_qt.peak_analysis_window import PeakAnalysisWindow


class CursorReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.window = PeakAnalysisWindow(Path(self.temp.name) / 'workspace.npz')
        # Descending, nonuniform energy; unsorted scan rows, with a missing cell.
        cube = DataCube(np.array([1.4, 1.1, 1.0]), np.array([8., -4., 1.]),
                        np.array([[80., 81., 82.], [-40., -41., -42.], [10., 11., np.nan]]),
                        'Gate (V)', 'Cursor fixture', 'DR/R')
        self.window.add_datasets([create_dataset(cube, 'DRR', 'raw', 'cursor.csv', 'Cursor fixture')])
        self.window.show(); self.app.processEvents(); self.window.canvas.draw()

    def tearDown(self):
        self.window.close()
        deadline = time.monotonic() + 10
        while self.window.busy and time.monotonic() < deadline:
            self.app.processEvents(); time.sleep(.01)
        self.assertFalse(self.window.busy)
        self.window.deleteLater(); self.app.processEvents(); self.temp.cleanup()

    def move(self, axes, x, y, wait=True):
        w = self.window
        px, py = axes.transData.transform((x, y))
        event = MouseEvent('motion_notify_event', w.canvas, px, py)
        w.canvas.callbacks.process('motion_notify_event', event)
        if wait:
            self.wait_for_readout()

    def wait_for_readout(self):
        # Earlier windows can leave expensive deferred Qt cleanup. Wait for
        # the throttled update, not a fixed 60 ms that may be consumed entirely
        # by that cleanup before Qt delivers the due timer.
        timer = self.window.cursor_readout._timer
        deadline = time.monotonic() + 2
        while timer.isActive() and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertFalse(timer.isActive(), 'Cursor update did not finish')

    def test_heatmap_reads_original_cell_without_changing_analysis_or_drawing(self):
        w = self.window
        w.dataset['view']['display'] = dict(vmin=-1., vmax=1., clip_outliers=True)
        w.redraw(); w.canvas.draw(); self.app.processEvents()
        row, generation, result = w.row_slider.value(), w.generation, w.dataset['result']
        draws = []
        connection = w.canvas.mpl_connect('draw_event', lambda event: draws.append(event))
        self.move(w.heat_ax, 1.08, 5.2)
        text = w.cursor_readout.text()
        self.assertIn('1.08000 eV', text)
        self.assertIn('5.2', text)
        self.assertIn('Gate (V)', text)
        self.assertIn('81', text)  # Original unsorted row; no color clipping to 1.
        self.assertIn('nearest', text)
        self.assertIn('DR/R', w.cursor_readout.toolTip())
        self.assertIn('8', w.cursor_readout.toolTip())  # Actual cell coordinate.
        self.assertEqual(w.row_slider.value(), row)
        self.assertEqual(w.generation, generation)
        self.assertIs(w.dataset['result'], result)
        self.assertEqual(draws, [])
        w.canvas.mpl_disconnect(connection)

    def test_missing_cells_and_outside_mesh_never_borrow_neighbor_values(self):
        w = self.window
        self.move(w.heat_ax, 1.001, 1.)
        self.assertIn('—', w.cursor_readout.text())
        self.assertIn('1.00100 eV', w.cursor_readout.text())
        w.heat_ax.set_xlim(.5, 2.); w.canvas.draw(); self.app.processEvents()
        self.move(w.heat_ax, .7, 1.)
        self.assertIn('0.70000 eV', w.cursor_readout.text())
        self.assertIn('—', w.cursor_readout.text())

    def test_log_scan_uses_visible_cell_and_physical_coordinate(self):
        w = self.window
        cube = DataCube(np.array([1., 1.1, 1.4]), np.array([100., -1., 1.]),
                        np.array([[99., 100., 101.], [-8., -9., -10.], [1., 2., 3.]]),
                        'Power (mW)', 'Log sweep', 'Intensity')
        w.add_datasets([create_dataset(cube, 'PL', 'PL', 'power.csv', 'Log sweep',
                                      dict(y_axis_log=True, ylim=[1., 100.]))])
        w.canvas.draw(); self.app.processEvents()
        # Log cell boundary is 10 mW. Above it belongs to the 100 mW cell.
        self.move(w.heat_ax, 1.1, 20.)
        self.assertIn('20', w.cursor_readout.text())
        self.assertIn('Power (mW)', w.cursor_readout.text())
        self.assertIn('100', w.cursor_readout.text())

    def test_spectrum_and_residual_report_mouse_y_instead_of_snapping_to_signal(self):
        w = self.window
        self.move(w.spectrum_ax, 1.08, 80.5)
        self.assertIn('Spectrum', w.cursor_readout.text())
        self.assertIn('80.5', w.cursor_readout.text())
        self.assertNotIn('nearest', w.cursor_readout.text())
        cube = w.dataset['cube']
        w.add_datasets([create_dataset(cube, 'PL', 'PL', 'pl.csv', 'PL')])
        w.controls.residual.setChecked(True)
        w.residual_ax.set_ylim(-1e-5, 1e-5); w.canvas.draw(); self.app.processEvents()
        self.move(w.residual_ax, 1.08, 2.5e-6)
        self.assertIn('Residual', w.cursor_readout.text())
        self.assertIn('2.5e-06', w.cursor_readout.text())

    def test_leave_colorbar_row_and_product_changes_cancel_pending_readings(self):
        w = self.window
        self.move(w.heat_ax, 1.08, 5.2, wait=False)
        w.canvas.callbacks.process('figure_leave_event', LocationEvent('figure_leave_event', w.canvas, 0, 0))
        QTest.qWait(60)
        self.assertNotIn('1.08000', w.cursor_readout.text())
        self.assertIn('—', w.cursor_readout.text())
        self.move(w.heat_ax, 1.08, 5.2)
        colorbar = next(ax for ax in w.figure.axes if ax not in (w.heat_ax, w.spectrum_ax, w.residual_ax))
        self.move(colorbar, .5, 0.)
        self.assertNotIn('1.08000', w.cursor_readout.text())
        self.move(w.heat_ax, 1.08, 5.2, wait=False)
        w.row_slider.setValue(1); QTest.qWait(60)
        self.assertNotIn('1.08000', w.cursor_readout.text())
        self.move(w.heat_ax, 1.08, 5.2, wait=False)
        d = create_dataset(w.dataset['cube'], 'DRR', 'second', 'cursor.csv', 'Second')
        w.add_datasets([d]); QTest.qWait(60)
        self.assertNotIn('1.08000', w.cursor_readout.text())

    def test_motion_bursts_coalesce_to_latest_value(self):
        w = self.window
        for y in (2., 3., 4., 5.2):
            self.move(w.heat_ax, 1.08, y, wait=False)
        self.wait_for_readout()
        self.assertIn('5.2', w.cursor_readout.text())
        self.assertIn('81', w.cursor_readout.text())

    def test_scan_units_from_metadata_are_displayed_without_duplication(self):
        from dataclasses import replace
        w = self.window
        for label in ('Power', 'Power (mW)'):
            cube = replace(w.dataset['cube'], gate_label=label, gate_unit='mW')
            w.add_datasets([create_dataset(cube, 'PL', 'PL', 'units.csv', 'Units')])
            self.app.processEvents(); w.canvas.draw(); self.app.processEvents()
            self.move(w.heat_ax, 1.08, 5.2)
            self.assertEqual(w.cursor_readout.text().count('mW'), 1)
            self.assertIn('Power (mW)', w.cursor_readout.toolTip())

    def test_last_cell_details_remain_accessible_after_leaving_plot(self):
        w = self.window
        self.move(w.heat_ax, 1.08, 5.2)
        w.canvas.callbacks.process('axes_leave_event', LocationEvent('axes_leave_event', w.canvas, 0, 0))
        w.canvas.callbacks.process('figure_leave_event', LocationEvent('figure_leave_event', w.canvas, 0, 0))
        QTest.qWait(60)
        self.assertNotIn('1.08000', w.cursor_readout.text())
        self.assertIn('—', w.cursor_readout.text())
        self.assertIn('Last cursor reading', w.cursor_readout.toolTip())
        self.assertIn('DR/R', w.cursor_readout.toolTip())
        self.assertIn('Nearest cell: Energy = 1.1 eV; Gate (V) = 8', w.cursor_readout.toolTip())
        w.row_slider.setValue(1); self.app.processEvents()
        self.assertNotIn('Nearest cell:', w.cursor_readout.toolTip())


if __name__ == '__main__':
    unittest.main()
