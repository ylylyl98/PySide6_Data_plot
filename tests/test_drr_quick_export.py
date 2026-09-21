import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import time
import numpy as np
from PySide6.QtWidgets import QApplication,QWidget
from core.loader import DataCube
from core.drr_analysis_workspace import create_dataset
from ui_qt.drr_p2p_batch import BatchRangeAmplitudePage


class QuickExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_default_export_no_dialog_versions_and_origin_csv(self):
        with TemporaryDirectory() as folder:
            owner=QWidget();owner.start_folder=folder;owner.comparison_key='group1'
            owner.comparison_catalog={'group1':{'label':'YZ365 · p5n2 · TG-1.087BG=18\n1.67 K · 720 nm'}}
            d=create_dataset(DataCube(np.array([1.,2.,3.]),np.array([0.,1.]),np.array([[0.,2.,0.],[0.,3.,0.]]),'Y','A','DRR'),'A',{})
            page=BatchRangeAmplitudePage(owner,d,(1.,3.,0.,1.))
            try:
                with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName',side_effect=AssertionError('Unexpected dialog')):
                    a=page.save_csv();b=page.save_csv();png=page.save_png()
                deadline=time.monotonic()+10
                while not page._png_save_job.done and time.monotonic()<deadline:
                    self.app.processEvents();time.sleep(.005)
                self.assertTrue(page._png_save_job.done)
                self.assertIsNone(page._png_save_job.error)
                self.assertNotEqual(a,b);self.assertTrue(a.is_file());self.assertTrue(b.is_file());self.assertTrue(png.is_file())
                self.assertIn('p5n2',a.name);self.assertIn('1.0000-3.0000eV',a.name)
                self.assertIn('raw',a.name);self.assertIn('single',png.name)
                self.assertIn('Dataset,Y_label,Y,P2P',a.read_text(encoding='utf-8-sig'))
                self.assertEqual(png.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
                self.assertEqual(page.last_export_path,png)
                self.assertTrue(page.open_export_folder_button.isEnabled())
                with patch('ui_qt.drr_quick_export.QDesktopServices.openUrl') as open_folder:
                    page.open_export_folder()
                    self.assertEqual(Path(open_folder.call_args[0][0].toLocalFile()),png.parent.resolve())
                page.view_mode.setCurrentIndex(1)
                _,name=page.export_destination('png');self.assertIn('compare',name)
                page.smoothing.setCurrentText('SG');page.auto_timer.stop()
                _,name=page.export_destination('csv');self.assertIn('SG21points_p2',name)
                page.invalidate()
                self.assertIsNone(page.save_csv())
            finally:page.close();owner.close()

    def test_explicit_save_as_keeps_default_directory(self):
        with TemporaryDirectory() as folder:
            owner=QWidget();owner.start_folder=folder
            d=create_dataset(DataCube(np.array([1.,2.,3.]),np.array([0.]),np.array([[0.,2.,0.]]),'Y','A','DRR'),'A',{})
            page=BatchRangeAmplitudePage(owner,d,(1.,3.,0.,0.))
            try:
                target=Path(folder)/'custom.csv'
                with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName',return_value=(str(target),'')):
                    self.assertEqual(page.save_csv_as(),target)
                self.assertTrue(target.exists())
                self.assertNotEqual(page.save_csv().parent,target.parent)
            finally:page.close();owner.close()
