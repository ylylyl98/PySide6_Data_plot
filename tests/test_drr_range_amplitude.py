import unittest
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
from core.loader import DataCube
from core.drr_range_amplitude import range_amplitude


class RangeAmplitudeTests(unittest.TestCase):
    def test_points_window_is_same_count_across_sampling_intervals(self):
        from core.drr_range_amplitude import smooth_energy_cube
        for step in (.0001,.0003):
            cube=DataCube(1+np.arange(101)*step,np.array([0.]),np.sin(np.arange(101))[None,:],'Y','A','DRR')
            _,meta=smooth_energy_cube(cube,21,2,unit='points')
            self.assertEqual(meta['window_points_min'],21)
            self.assertAlmostEqual(meta['actual_span_mev_min'],20*step*1000)
    def test_energy_smoothing_preserves_gaps_input_and_reduces_spike(self):
        from core.drr_range_amplitude import smooth_energy_cube
        x=np.linspace(1.,1.1,101);z=np.zeros((2,101));z[:,50]=1;z[:,70:75]=np.nan
        cube=DataCube(x,np.arange(2.),z.copy(),'Y','A','DRR')
        smooth,meta=smooth_energy_cube(cube,9.,2)
        self.assertLess(np.nanmax(smooth.Z),.5)
        self.assertTrue(np.isnan(smooth.Z[:,70:75]).all())
        np.testing.assert_array_equal(cube.Z,z)
        self.assertEqual(meta['method'],'SG')

    def test_range_extrema_positions_and_missing_rows(self):
        cube=DataCube(np.array([1.,2.,3.,4.]),np.array([0.,1.,2.]),
                      np.array([[99.,-2.,5.,99.],[0.,np.nan,3.,0.],[0.,np.nan,np.nan,0.]]),'Y','A','DRR')
        result=range_amplitude(cube,2,3,0,2)
        np.testing.assert_allclose(result[:,1],[7.,np.nan,np.nan],equal_nan=True)
        np.testing.assert_allclose(result[0,2:6],[-2.,5.,2.,3.])
        np.testing.assert_array_equal(result[:,-1],[2,1,0])
        self.assertEqual(len(range_amplitude(cube,2,3,1,2)),2)

    def test_empty_or_invalid_window(self):
        cube=DataCube(np.array([1.,2.]),np.array([0.]),np.array([[1.,2.]]),'Y','A','DRR')
        for bounds in [(3,4,0,1),(2,1,0,1),(1,2,2,3)]:
            with self.assertRaises(ValueError):range_amplitude(cube,*bounds)

    def test_dialog_range_edit_invalidates_export_and_csv_records_window(self):
        import csv
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from PySide6.QtWidgets import QApplication
        from ui_qt.drr_range_amplitude import RangeAmplitudeDialog
        app=QApplication.instance() or QApplication([])
        cube=DataCube(np.array([1.,2.,3.]),np.array([0.,1.]),np.array([[1.,2.,4.],[2.,3.,8.]]),'Gate V','test','DRR')
        dialog=RangeAmplitudeDialog(None,cube,'group',(1.,3.,0.,1.))
        try:
            np.testing.assert_array_equal(dialog.result[:,1],[3.,6.])
            dialog.select_y(1.)
            np.testing.assert_array_equal(dialog.spectrum_line.get_ydata(),cube.Z[1])
            np.testing.assert_allclose(dialog.extrema.get_offsets(),[[1.,2.],[3.,8.]])
            dialog.bounds[0].setValue(2.)
            self.assertFalse(dialog.csv_button.isEnabled())
            dialog.calculate()
            self.assertTrue(dialog.csv_button.isEnabled())
            with TemporaryDirectory() as folder:
                path=Path(folder)/'amplitude.csv';dialog.write_csv(path)
                with path.open(encoding='utf-8-sig',newline='') as handle:rows=list(csv.DictReader(handle))
                self.assertEqual(float(rows[0]['DRR_peak_to_peak']),2.)
                self.assertEqual(float(rows[0]['Energy_min_eV']),2.)
                self.assertEqual(rows[0]['Y_label'],'Gate V')
        finally:dialog.close()
