import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from types import SimpleNamespace
import numpy as np
from core.loader import DataCube
from core.drr_analysis_workspace import create_dataset
from PySide6.QtWidgets import QApplication,QWidget
from ui_qt.drr_p2p_batch import BatchRangeAmplitudePage


class BatchP2PTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_sg_records_use_full_spectrum_and_keep_raw_comparison(self):
        x=np.linspace(1.,1.1,101);z=np.zeros((2,101));z[:,50]=1.
        d=create_dataset(DataCube(x,np.arange(2.),z,'Y','A','DRR'),'A',{})
        settings=dict(method='SG',window_mev=9.,polyorder=2)
        full=BatchRangeAmplitudePage.compute_records([d],(1.,1.1,0,1),False,smoothing=settings)[0]
        crop=BatchRangeAmplitudePage.compute_records([d],(1.045,1.055,0,1),False,smoothing=settings)[0]
        np.testing.assert_array_equal(full['processed'].Z,crop['processed'].Z)
        self.assertLess(crop['result'][0,1],crop['raw_result'][0,1])
        owner=QWidget();page=BatchRangeAmplitudePage(owner,d,(1.045,1.055,0,1))
        try:
            self.assertEqual(page.sg_unit.currentText(),'points')
            self.assertEqual(page.sg_width.value(),21)
            page.receive_records([crop])
            self.assertTrue(page.smoothed_line.get_visible())
            np.testing.assert_array_equal(page.smoothed_line.get_ydata(),crop['processed'].Z[0])
        finally:page.close();owner.close()

    def test_shared_energy_all_y_export_and_picked_file(self):
        x=np.array([1.,2.,3.,4.]);y=np.array([0.,1.,2.])
        a=create_dataset(DataCube(x,y,np.array([[0.,-2.,4.,0.]]*3),'Y','A','DRR'),'A',{})
        b=create_dataset(DataCube(x,y+5,np.array([[0.,-1.,8.,0.]]*3),'Y','B','DRR'),'B',{})
        owner=QWidget();page=BatchRangeAmplitudePage(owner,a,(2.,3.,0.,2.))
        try:
            records=page.compute_records([a,b],(2.,3.,0.,2.),False)
            self.assertEqual(records[1]['bounds'],(2.,3.,5.,7.))
            np.testing.assert_array_equal(records[1]['result'][:,1],9.)
            page.receive_records(records);page.view_mode.setCurrentIndex(1);page.canvas.draw()
            line=next(line for line in page.amplitude_axis.lines if getattr(line,'_p2p_record',None)==1)
            pos=line.axes.transData.transform([6.,9.])
            page.pick_curve(SimpleNamespace(artist=line,ind=[1],mouseevent=SimpleNamespace(x=pos[0],y=pos[1])))
            self.assertEqual(page.name,'B');self.assertEqual(page.row_spin.value(),2)
            np.testing.assert_array_equal(page.spectrum_line.get_ydata(),b.cube.Z[1])
            page.view_mode.setCurrentIndex(0)
            page.select_window(1.9,3.1);self.assertFalse(page.csv_button.isEnabled())
            page.inspect_file.setCurrentIndex(0);self.assertFalse(page.csv_button.isEnabled())
            import tempfile,csv
            with tempfile.TemporaryDirectory() as folder:
                page.write_csv(folder+'/results.csv')
                with open(folder+'/results.csv',encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
                self.assertEqual(len(rows),6);self.assertEqual(rows[-1]['Dataset'],'B')
        finally:page.close();owner.close()

    def test_view_switch_preserves_result_file_row_and_does_not_calculate(self):
        from unittest.mock import patch
        x=np.array([1.,2.,3.]);y=np.array([0.,1.,2.])
        a=create_dataset(DataCube(x,y,np.array([[0.,1.,0.]]*3),'Y','A','DRR'),'A',{})
        b=create_dataset(DataCube(x,y,np.array([[0.,2.,0.]]*3),'Y','B','DRR'),'B',{})
        owner=QWidget();page=BatchRangeAmplitudePage(owner,a,(1.,3.,0.,2.))
        try:
            page.receive_records(page.compute_records([a,b],(1.,3.,0.,2.),False))
            page.inspect_file.setCurrentIndex(1);page.select_y(2.)
            records=page.records
            page.display_bounds[0].setValue(1.5);page.display_bounds[1].setValue(2.5)
            page.apply_display_range()
            self.assertEqual(page.map_axis.get_xlim(),(1.5,2.5))
            self.assertEqual(page.spectrum_axis.get_xlim(),(1.5,2.5))
            self.assertGreater(page.spectrum_axis.get_ylim()[0],1.)
            page.map_axis.set_xlim(1.7,2.3)
            self.assertEqual(page.spectrum_axis.get_xlim(),(1.7,2.3))
            page.apply_display_range()
            page.select_y(1.)
            self.assertEqual(page.spectrum_axis.get_xlim(),(1.5,2.5))
            page.select_y(2.)
            page.fixed_display.setChecked(True)
            page.inspect_file.setCurrentIndex(0)
            self.assertEqual(page.map_axis.get_xlim(),(1.5,2.5))
            page.inspect_file.setCurrentIndex(1)
            with patch.object(page,'compute_records',side_effect=AssertionError('recalculated')):
                page.view_mode.setCurrentIndex(1)
                self.assertIsNone(page.map_axis)
                self.assertEqual(page.spectrum_axis.get_xlim(),(1.5,2.5))
                self.assertEqual(len([l for l in page.amplitude_axis.lines if hasattr(l,'_p2p_record')]),2)
                page.view_mode.setCurrentIndex(0)
                self.assertIsNotNone(page.map_axis)
                self.assertEqual(len([l for l in page.amplitude_axis.lines if hasattr(l,'_p2p_record')]),1)
            self.assertIs(page.records,records)
            self.assertEqual(page.inspect_file.currentIndex(),1)
            self.assertEqual(page.row_spin.value(),3)
            np.testing.assert_array_equal(page.spectrum_line.get_ydata(),b.cube.Z[2])
        finally:page.close();owner.close()

    def test_empty_file_does_not_break_other_results(self):
        d=create_dataset(DataCube(np.array([1.,2.]),np.array([0.]),np.array([[1.,2.]]),'Y','A','DRR'),'A',{})
        records=BatchRangeAmplitudePage.compute_records([d],(3.,4.,0.,1.),False)
        self.assertIsNone(records[0]['result']);self.assertTrue(records[0]['error'])
