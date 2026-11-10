"""The real two-process handoff refreshes colors and rejects mismatched data."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtGui import QGuiApplication
from tests.test_peak_workspace import sample
from ui_qt.quick.peak_analysis_workflow import prepare_sources, refresh_source_export


class LiveExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QGuiApplication.instance() or QGuiApplication([])

    def test_live_display_refresh_and_stale_numeric_source_rejection(self):
        code = '''
import json,sys
from pathlib import Path
from PySide6.QtCore import QObject,QSettings,QTimer
from PySide6.QtGui import QGuiApplication
from tests.test_peak_workspace import sample
from ui_qt.quick.peak_analysis_launcher import QuickPeakAnalysisLauncher
root=Path(sys.argv[1]);app=QGuiApplication([]);owner=QObject()
source=dict(cube=sample('DRR'),kind='DRR',channel='second',source='test.dat',name='test',
            source_export=dict(params=dict(cmap='RdBu_r',vmin=-1.,vmax=1.),drr=True))
def capture():
    config=json.loads((root/'config.json').read_text())
    source['source_export']['params'].update(vmin=config['vmin'])
    source['cube']=sample('DRR')
    source['cube'].Z[0,0]+=config.get('change',0.)
    return [source]
owner._capture_peak_sources=capture
launcher=QuickPeakAnalysisLauncher(settings=QSettings(str(root/'settings.ini'),QSettings.Format.IniFormat),session_path=root/'workspace.npz',parent=owner)
(root/'ready.json').write_text(json.dumps(dict(server=launcher.server_name,listening=launcher.server.listening)))
timer=QTimer();timer.timeout.connect(lambda: app.quit() if (root/'stop').exists() else None);timer.start(30)
QTimer.singleShot(30000,app.quit)
app.exec();launcher.close()
'''
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config.json').write_text(json.dumps(dict(vmin=-5000.)))
            process = subprocess.Popen([sys.executable, '-c', code, str(root)],
                cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, QT_QPA_PLATFORM='offscreen'),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 20
                while not (root/'ready.json').exists() and time.monotonic() < deadline:
                    self.assertIsNone(process.poll())
                    time.sleep(.02)
                ready = json.loads((root/'ready.json').read_text())
                self.assertTrue(ready['listening'])
                source = dict(cube=sample('DRR'), kind='DRR', channel='second', source='test.dat', name='test',
                              source_export=dict(params=dict(cmap='RdBu_r', vmin=-1., vmax=1.), drr=True))
                dataset = prepare_sources([source], reply_server=ready['server'])[0]
                fresh = refresh_source_export(dataset, root/'workspace.npz')
                self.assertEqual(fresh['source_export']['params']['vmin'], -5000.)
                self.assertEqual(dataset['source_export']['params']['vmin'], -1.)
                (root/'config.json').write_text(json.dumps(dict(vmin=-5000.,change=1.)))
                with self.assertRaisesRegex(ValueError, 'data changed'):
                    refresh_source_export(dataset, root/'workspace.npz')
                self.assertFalse(list((root/'display_requests').glob('*.json')))
            finally:
                (root/'stop').touch()
                try:
                    _, error = process.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill();process.communicate();raise
                self.assertEqual(process.returncode, 0, error.decode(errors='replace'))
