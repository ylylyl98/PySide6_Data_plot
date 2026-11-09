import os
import subprocess
import sys
import unittest

class MegasweepQuickTests(unittest.TestCase):
    def test_modes_session_and_async_plot(self):
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
        completed = subprocess.run([sys.executable, '-c', RUNNER], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

RUNNER = r'''
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtCore import QSettings, QUrl
from PySide6.QtGui import QGuiApplication
from ui_qt.quick.megasweep_bridge import MegasweepBridge
app = QGuiApplication([])
def wait_for(predicate, seconds=15):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        if predicate(): return True
        time.sleep(.01)
    return False
with TemporaryDirectory() as directory:
    root = Path(directory); source = root / 'sample.csv'
    source.write_text('x,y,700,710\n0,0,10,9\n0,1,11,10\n1,0,12,11\n1,1,13,12\n')
    settings = QSettings(str(root / 'test.ini'), QSettings.IniFormat)
    bridge = MegasweepBridge(settings=settings, session_directory=root / 'sessions')
    try:
        bridge.setExperimentFolder(str(root)); bridge.selectFiles([QUrl.fromLocalFile(str(source))])
        bridge.setParameter('auto_baseline', False); bridge.setParameter('baseline', 0)
        bridge.setParameter('energy_min', 1.7); bridge.setParameter('energy_max', 1.8)
        assert wait_for(lambda: bool(bridge.snapshot['canExport'])), bridge.snapshot['error']
        assert bridge.plotBackend is not None
        bridge.setParameter('mode', 'Reflection'); assert bridge.snapshot['sourceFile'] == ''
        bridge.setParameter('mode', 'PL'); assert bridge.snapshot['values']['baseline'] == 0
        assert wait_for(lambda: bool(bridge.snapshot['canExport'])), bridge.snapshot['error']
        bridge.setParameter('movie.fps', 24); assert bridge.snapshot['movieOptions']['fps'] == 24
        bridge.setParameter('movie.fps', 24.5); assert bridge.snapshot['movieOptions']['fps'] == 24
        assert bridge.snapshot['movieError']
        bridge._busy = True; bridge._operation = 'export'
        bridge.setParameter('movie.fps', 30); assert bridge.snapshot['movieOptions']['fps'] == 24
        bridge._busy = False; bridge._operation = ''
        bridge.setParameter('auto_update', False); bridge.setParameter('baseline', 5)
        assert not bridge.snapshot['canExport']
        bridge.invoke('update'); assert wait_for(lambda: bool(bridge.snapshot['canExport'])), bridge.snapshot['error']
        transport = root / 'transport.csv'
        transport.write_text('Vbg,Vtg,current\n2,4,10\n2,5,11\n3,4,12\n3,5,13\n')
        bridge.setParameter('mode', 'Transport'); bridge.selectFiles([str(transport)])
        assert wait_for(lambda: bool(bridge.snapshot['canExport'])), bridge.snapshot['error']
        assert bridge.snapshot['values']['cut_type'] in ('Vbg','Vtg')
        assert bridge.snapshot['values']['cut_value'] in (2.,3.,4.,5.)
        bridge.setParameter('pass_index', 1); bridge.setParameter('pass_index', ''); assert bridge.snapshot['values']['pass_index'] is None
        bridge.setParameter('mode', 'PL'); bridge.setParameter('auto_update', True)
        assert wait_for(lambda: bool(bridge.snapshot['canExport'])), bridge.snapshot['error']
        import core.megasweep.workflow as workflow
        original = workflow.analyze
        def slow(dataset, parameters, background_paths=(), cancelled=None):
            if parameters['baseline'] == 6: time.sleep(.5)
            return original(dataset, parameters, background_paths, cancelled)
        workflow.analyze = slow
        bridge.setParameter('baseline', 6); assert wait_for(lambda: bridge.snapshot['busy'])
        bridge.setParameter('baseline', 7)
        assert wait_for(lambda: bool(bridge.snapshot['canExport']) and bridge._result['parameters']['baseline'] == 7), bridge.snapshot['error']
        time.sleep(.6); app.processEvents()
        assert bridge._result['parameters']['baseline'] == 7
        workflow.analyze = original
        # Reference spectra are a distinct inspection state: never export the hidden map.
        import numpy as np
        preview = dict(energy=np.array([1.,2.]), raw=np.array([2.,3.]), background=np.array([1.,1.]), scaled_background=np.array([1.,1.]), rc=np.array([1.,2.]), frame=0)
        bridge._finish(bridge._sequence, 'reflection-preview', preview, '', 0.)
        assert bridge.snapshot['reflectionPreviewVisible'] and not bridge.snapshot['canExport']
        bridge.invoke('return-to-map')
        assert not bridge.snapshot['reflectionPreviewVisible'] and bridge.snapshot['canExport']
        # Keep the export receipt and the background chooser alive through shortcuts.
        import threading
        gate = threading.Event(); original_export = workflow.export_result; exported_parameters = {}
        bridge._canvas.figure.axes[0].set_xlim(-.25, .75)
        bridge._canvas.figure.axes[0].set_ylim(-.5, .5)
        def held_export(result, directory, *, parameters=None):
            exported_parameters.update(parameters)
            assert gate.wait(5)
            output = root / 'receipt.csv'; output.write_text('saved')
            return [str(output)]
        workflow.export_result = held_export
        bridge.invoke('export'); assert bridge.snapshot['busy']
        seq = bridge._sequence; before_source = bridge.snapshot['sourceFile']; before_mode = bridge.snapshot['mode']
        for action in ('update','reload','choose-background','clear-background','slice-next','reflection-preview','movie-preview'):
            bridge.invoke(action)
        bridge.setParameter('mode', 'Reflection'); bridge.openFiles('background', [str(source)]); bridge.selectFiles([str(transport)])
        assert bridge._sequence == seq and bridge.snapshot['busy'] and bridge.snapshot['sourceFile'] == before_source and bridge.snapshot['mode'] == before_mode
        assert not bridge._background._editing_enabled
        gate.set(); assert wait_for(lambda: not bridge.snapshot['busy'])
        assert bridge._background._editing_enabled and bridge.snapshot['resultPaths'] == [str(root / 'receipt.csv')]
        assert not exported_parameters['x_auto'] and not exported_parameters['y_auto']
        assert (exported_parameters['xmin'], exported_parameters['xmax']) == (-.25, .75)
        assert (exported_parameters['ymin'], exported_parameters['ymax']) == (-.5, .5)
        workflow.export_result = original_export
        bridge.invoke('export'); assert wait_for(lambda: not bridge.snapshot['busy'])
        import json
        paths = [Path(p) for p in bridge.snapshot['resultPaths']]
        assert any(p.suffix == '.png' and p.is_file() for p in paths)
        metadata = json.loads(next(p for p in paths if p.suffix == '.json').read_text(encoding='utf8'))
        exported = metadata['parameters']
        assert not exported['x_auto'] and (exported['xmin'], exported['xmax']) == (-.25, .75)
        assert not exported['y_auto'] and (exported['ymin'], exported['ymax']) == (-.5, .5)
        # A source outside the browser root exports beside its own experiment.
        browse = root / 'browse'; external = root / 'external'; browse.mkdir(); external.mkdir()
        other = external / 'other.csv'; other.write_text(source.read_text())
        bridge.setExperimentFolder(str(browse)); bridge.selectFiles([str(other)])
        assert bridge._state['source_root'] == str(external)
        from core.megasweep_store import load_session
        assert load_session(root / 'sessions', other, 'PL')['source_root'] == str(external)
    finally: bridge.close()
'''
