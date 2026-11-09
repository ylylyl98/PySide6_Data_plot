"""Per-mode picker and background locations are isolated from analysis state."""
import os
import subprocess
import sys
import unittest

class MegasweepLocationTests(unittest.TestCase):
    def test_locations_in_isolated_process(self):
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
        result = subprocess.run([sys.executable, '-c', RUNNER], env=env, capture_output=True, text=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

RUNNER = r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtCore import QSettings, QUrl
from PySide6.QtGui import QGuiApplication
from ui_qt.quick.megasweep_bridge import MegasweepBridge
app = QGuiApplication([])
with TemporaryDirectory() as directory:
    root = Path(directory)
    paths = {name: root / name for name in ('experiment','pl','reflection','transport','background','manual')}
    for path in paths.values(): path.mkdir()
    files = {name: paths[name] / (name + '.csv') for name in ('pl','reflection','transport','background')}
    for path in files.values(): path.write_text('x,y,700,710\n0,0,1,2\n')
    settings_file = root / 'test.ini'
    settings = QSettings(str(settings_file), QSettings.IniFormat)
    bridge = MegasweepBridge(settings=settings, session_directory=root/'sessions')
    try:
        bridge.setExperimentFolder(str(paths['experiment']))
        bridge.selectFiles([str(files['pl'])])
        assert bridge.snapshot['pickerFolder'] == str(paths['pl'])
        assert bridge.snapshot['experimentRoot'] == str(paths['pl'])
        assert bridge.snapshot['pickerSelected'] == [str(files['pl'])]
        assert bridge._state['source_root'] == str(paths['pl'])
        source_root = bridge._state['source_root']; source = bridge.snapshot['sourceFile']
        bridge._result = {'title':'test', 'diagnostics':{}}; bridge._result_sequence = bridge._sequence
        bridge.setExperimentFolder(str(paths['manual']))
        assert bridge.snapshot['pickerFolder'] == str(paths['manual'])
        assert bridge.snapshot['experimentRoot'] == str(paths['manual'])
        assert bridge.snapshot['sourceFile'] == source and bridge._state['source_root'] == source_root
        assert bridge.snapshot['canExport']
        bridge.setParameter('mode','Reflection')
        assert bridge.snapshot['experimentRoot'] == str(paths['manual'])
        assert bridge.snapshot['pickerFolder'] == str(paths['manual'])
        bridge.setExperimentFolder(str(paths['experiment']))
        bridge.selectFiles([str(files['reflection'])])
        bridge.openFiles('background',[QUrl.fromLocalFile(str(files['background']))])
        assert bridge.snapshot['backgroundFolder'] == str(paths['background'])
        bridge.setExperimentFolder(str(paths['manual']))
        assert bridge.snapshot['backgroundFolder'] == str(paths['background'])
        bridge.invoke('choose-background')
        assert bridge.backgroundSelectionBridge.snapshot['folder'] == str(paths['background'])
        bridge.setParameter('mode','Transport'); bridge.setExperimentFolder(str(paths['experiment']))
        bridge.selectFiles([str(files['transport'])])
        assert bridge.snapshot['pickerFolder'] == str(paths['transport'])
        assert bridge.snapshot['experimentRoot'] == str(paths['transport'])
        bridge.setParameter('mode','PL'); assert bridge.snapshot['pickerFolder'] == str(paths['manual'])
        bridge.setParameter('mode','Reflection'); assert bridge.snapshot['pickerFolder'] == str(paths['manual'])
    finally: bridge.close(); settings.sync()
    # A fresh bridge offers last files but does not load or scan them automatically.
    second = MegasweepBridge(settings=QSettings(str(settings_file), QSettings.IniFormat), session_directory=root/'sessions')
    try:
        assert second.snapshot['sourceFile'] == '' and second.snapshot['pickerSelected'] == [str(files['pl'])]
        assert second.snapshot['pickerFolder'] == str(paths['manual'])
        second.setParameter('mode','Reflection')
        assert second.snapshot['sourceFile'] == '' and second.snapshot['pickerSelected'] == [str(files['reflection'])]
        assert second.snapshot['backgroundFolder'] == str(paths['background'])
        second.setParameter('mode','Transport')
        assert second.snapshot['pickerFolder'] == str(paths['transport'])
        assert second.snapshot['pickerSelected'] == [str(files['transport'])]
    finally: second.close()
    files['transport'].unlink(); paths['transport'].rmdir()
    third = MegasweepBridge(settings=QSettings(str(settings_file), QSettings.IniFormat), session_directory=root/'sessions')
    try:
        third.setParameter('mode','Transport')
        assert third.snapshot['pickerFolder'] == str(paths['manual'])
        assert third.snapshot['pickerSelected'] == []
    finally: third.close()
'''
