"""Shared selection must retain Megasweep identities without DRR filtering."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PySide6.QtCore import QCoreApplication, QSettings, QUrl
from core.megasweep_catalog import scan, inventory
from ui_qt.quick.source_catalog import SourceCatalogBridge


class MegasweepCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Match the project suite's Widgets-capable test application. A bare
        # QCoreApplication here would prevent later GUI tests creating widgets.
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_candidates_include_scalar_and_optical_but_not_exports(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'optical.csv').write_text('Vbg,Vtg,700,701\n0,0,3,4\n')
            (root / 'transport.csv').write_text('# acquisition\nVbg,Vtg,Current\n0,0,1e-9\n')
            output = root / 'Processed Data' / 'Megasweep'
            output.mkdir(parents=True)
            (output / 'map.csv').write_text('x,y,z\n0,0,1\n')
            rows = scan(root)['files']
            self.assertEqual({row['kind'] for row in rows}, {'Optical', 'Transport'})
            self.assertEqual(len(inventory(root)), 2)
            self.assertTrue(all(Path(row['id']).is_absolute() for row in rows))

    def test_picker_single_selection_and_no_pl_or_drr_filter_leak(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            first, second = root / 'first.csv', root / 'second.csv'
            first.write_text('X,Y,700,701\n0,0,1,2\n')
            second.write_text('X,Y,I\n0,0,3\n')
            settings = QSettings(str(root / 'test.ini'), QSettings.IniFormat)
            picker = SourceCatalogBridge(settings=settings)
            try:
                self.assertTrue(picker._configure('megasweep', str(root), []))
                picker._catalog = scan(root)
                picker._catalog_validated = True
                picker._refresh_models()
                self.assertEqual(picker.filesModel.rowCount(), 2)
                picker.selectFile(str(first), True)
                picker.selectFile(str(second), True)
                self.assertEqual(picker.snapshot['selected'], [str(second.resolve())])
                picker.addExternal([QUrl.fromLocalFile(str(first))])
                self.assertEqual(picker.snapshot['selected'], [str(first.resolve())])
                self.assertTrue(picker.snapshot['canAccept'])
            finally:
                picker.close()


if __name__ == '__main__':
    unittest.main()
