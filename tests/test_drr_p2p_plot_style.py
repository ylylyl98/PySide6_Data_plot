import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication,QWidget
from core.loader import DataCube
from core.drr_analysis_workspace import create_dataset
from ui_qt.drr_p2p_batch import BatchRangeAmplitudePage


class P2PStyleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_clean_comparison_and_state_restore(self):
        owner=QWidget();x=np.array([1.,2.,3.]);y=np.arange(3.)
        datasets=[create_dataset(DataCube(x,y,np.array([[0.,b+1.,0.]]*3),'Y','A','DRR'),str(b),{'measurement_files':[f'YZ365_p5n2_{b}T_1.67KREF_720nmc_Rot49deg_TG-1.087BG=18.csv']}) for b in (0,1)]
        owner.datasets={d.key:d for d in datasets}
        page=BatchRangeAmplitudePage(owner,datasets[0],(1.,3.,0.,2.))
        try:
            page.receive_records(page.compute_records(datasets,(1.,3.,0.,2.),False));page.view_mode.setCurrentIndex(1)
            figure=page.comparison_export_figure()
            self.assertEqual(len(figure.axes),1)
            lines=figure.axes[0].lines
            self.assertEqual(len(lines),2)
            self.assertTrue(page.style_controls['markers'].isChecked())
            self.assertTrue(all(line.get_marker()=='o' and line.get_linestyle()=='-' for line in lines))
            self.assertEqual([l.get_label() for l in lines],['0 T','1 T'])
            self.assertEqual(lines[0].get_linewidth(),lines[1].get_linewidth())
            self.assertEqual(lines[0].get_alpha(),lines[1].get_alpha())
            self.assertFalse(page.spectrum_axis.get_visible())
            page.show_inspection.setChecked(True);self.assertTrue(page.spectrum_axis.get_visible())
            page.style_controls['width'].setValue(7.5);page.style_controls['label_size'].setValue(14)
            state=page.snapshot_state()
            page.style_controls['width'].setValue(8.)
            page.restore_state(state)
            figure=page.comparison_export_figure()
            self.assertEqual(figure.get_figwidth(),7.5)
            self.assertEqual(figure.axes[0].xaxis.label.get_fontsize(),14)
            from tempfile import TemporaryDirectory
            from pathlib import Path
            from PIL import Image
            with TemporaryDirectory() as folder:
                path=Path(folder)/'compare.png';page.write_png(path)
                with Image.open(path) as exported:self.assertEqual(exported.size,(2250,1500))
            page.records.append(page.records[0])
            self.assertEqual(page.comparison_labels(),['0 T · #1','1 T','0 T · #2'])
            styles=page.comparison_curve_styles()
            self.assertEqual(styles[0]['color'],styles[2]['color'])
            self.assertNotEqual(styles[0]['marker'],styles[2]['marker'])
            color=styles[1]['color'];page.records.pop(0)
            self.assertEqual(page.comparison_curve_styles()[0]['color'],color)
            page.style_controls['offset'].setChecked(True)
            page.style_controls['offset_step'].setValue(.01)
            before=[r['result'].copy() for r in page.records]
            figure=page.comparison_export_figure()
            self.assertTrue(figure.axes[0].lines[0].get_label().startswith('0 T'))
            np.testing.assert_allclose(figure.axes[0].lines[1].get_ydata(),before[0][:,1]+.01)
            for record,original in zip(page.records,before):np.testing.assert_array_equal(record['result'],original)
        finally:page.close();owner.close()
