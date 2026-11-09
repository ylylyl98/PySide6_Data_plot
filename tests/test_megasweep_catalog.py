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
            root = Path(folder).resolve()
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
            root = Path(folder).resolve()
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


    def test_scope_selection_mode_isolation_and_reopen(self):
        from time import monotonic, sleep
        with TemporaryDirectory() as folder:
            root=Path(folder).resolve(); current=root/'measurement'; nested=current/'nested'; other=root/'other'
            nested.mkdir(parents=True); other.mkdir()
            paths=[current/'current.csv',nested/'nested.csv',other/'other.csv']
            for path in paths: path.write_text('Vbg,Vtg,700,701\n0,0,1,2\n')
            settings=QSettings(str(root/'picker.ini'),QSettings.IniFormat)
            picker=SourceCatalogBridge(settings=settings)
            def wait():
                until=monotonic()+10
                while picker.snapshot['busy'] and monotonic()<until:
                    self.app.processEvents(); sleep(.005)
                self.app.processEvents()
                self.assertFalse(picker.snapshot['busy']); self.assertFalse(picker.snapshot['error'])
            def shown(): return {row['id'] for row in picker.filesModel.rows}
            try:
                picker.openMegasweep(str(current),str(root),'PL',[str(paths[0])]); wait()
                self.assertEqual(picker.snapshot['megasweepScope'],'folder'); self.assertEqual(shown(),{str(paths[0])})
                picker.setParameter('query','does not exist')
                picker.openMegasweep(str(current),str(root),'PL',[str(paths[0])]); wait()
                self.assertEqual(picker.snapshot['query'],''); self.assertEqual(picker.snapshot['focused'],str(paths[0])); self.assertEqual(shown(),{str(paths[0])})
                picker.setParameter('megasweepScope','subfolders'); wait()
                self.assertEqual(shown(),{str(paths[0]),str(paths[1])})
                picker.setParameter('megasweepScope','experiment'); wait()
                self.assertEqual(shown(),{str(p) for p in paths})
                picker.openMegasweep(str(current),str(root),'Transport',[str(paths[0])]); wait()
                self.assertEqual(picker.snapshot['megasweepScope'],'folder'); self.assertEqual(shown(),{str(paths[0])})
                picker.openMegasweep(str(current),str(root),'PL',[str(paths[1])]); wait()
                self.assertEqual(picker.snapshot['megasweepScope'],'experiment'); self.assertEqual(shown(),{str(p) for p in paths})
                self.assertEqual(picker.snapshot['focused'],str(paths[1]))
                picker.open('pl',str(current),[],False); wait()
                self.assertEqual(picker.snapshot['workflow'],'pl'); self.assertFalse(picker.snapshot['includeAll'])
            finally: picker.close()

    def test_scanner_defaults_remain_recursive_for_backgrounds(self):
        from core.megasweep_catalog import source_paths
        with TemporaryDirectory() as folder:
            root=Path(folder).resolve(); (root/'child').mkdir(); (root/'a.csv').write_text('a,b\n1,2\n'); (root/'child'/'b.csv').write_text('a,b\n1,2\n')
            self.assertEqual(len(list(source_paths(root))),2)
            self.assertEqual(len(list(source_paths(root,recursive=False))),1)
            self.assertEqual(len(inventory(root,recursive=False)),1)

    def test_mode_default_filter_and_selected_exception(self):
        from time import monotonic, sleep
        with TemporaryDirectory() as folder:
            root=Path(folder).resolve(); optical=root/'optical.csv'; transport=root/'transport.csv'
            optical.write_text('Vbg,Vtg,700,701\n0,0,1,2\n'); transport.write_text('Vbg,Vtg,I\n0,0,1\n')
            picker=SourceCatalogBridge(settings=QSettings(str(root/'test.ini'),QSettings.IniFormat))
            def wait():
                limit=monotonic()+10
                while picker.snapshot['busy'] and monotonic()<limit: self.app.processEvents(); sleep(.005)
                self.app.processEvents(); self.assertFalse(picker.snapshot['busy'])
            try:
                picker.openMegasweep(str(root),str(root),'PL',[]); wait()
                self.assertEqual(picker.snapshot['megasweepType'],'Optical')
                self.assertEqual({row['id'] for row in picker.filesModel.rows},{str(optical)})
                picker.openMegasweep(str(root),str(root),'Transport',[]); wait()
                self.assertEqual(picker.snapshot['megasweepType'],'Transport')
                self.assertEqual({row['id'] for row in picker.filesModel.rows},{str(transport)})
                picker.openMegasweep(str(root),str(root),'PL',[str(transport)]); wait()
                self.assertEqual(picker.snapshot['focused'],str(transport))
                self.assertIn(str(transport),{row['id'] for row in picker.filesModel.rows})
                picker.setParameter('megasweepType','All')
                self.assertEqual(picker.filesModel.rowCount(),2)
            finally: picker.close()

    def test_rapid_scope_change_discards_stale_result(self):
        from threading import Event
        from time import monotonic, sleep
        with TemporaryDirectory() as folder:
            root=Path(folder).resolve(); nested=root/'nested'; nested.mkdir()
            (root/'a.csv').write_text('X,Y,700,701\n0,0,1,2\n'); (nested/'b.csv').write_text('X,Y,700,701\n0,0,1,2\n')
            picker=SourceCatalogBridge(settings=QSettings(str(root/'test.ini'),QSettings.IniFormat))
            entered=Event(); release=Event(); original=picker._scan_catalog
            def slow(folder,workflow,**kwargs):
                if workflow=='megasweep' and not kwargs['include_all']:
                    entered.set(); release.wait(5)
                return original(folder,workflow,**kwargs)
            picker._scan_catalog=slow
            try:
                picker.openMegasweep(str(root),str(root),'PL',[])
                self.assertTrue(entered.wait(5))
                picker.setParameter('megasweepScope','subfolders'); release.set()
                limit=monotonic()+10
                while picker.snapshot['busy'] and monotonic()<limit: self.app.processEvents(); sleep(.005)
                self.app.processEvents()
                self.assertFalse(picker.snapshot['busy']); self.assertEqual(picker.snapshot['megasweepScope'],'subfolders')
                self.assertEqual({row['name'] for row in picker.filesModel.rows},{'a.csv','b.csv'})
                self.assertIn(str(nested),picker._watcher.directories())
            finally: release.set(); picker.close()

if __name__ == '__main__':
    unittest.main()
