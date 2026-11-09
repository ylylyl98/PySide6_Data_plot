"""Role-isolated reference selection and session round trips, without Widgets."""
import os
import subprocess
import sys
import unittest

class MegasweepRoleTests(unittest.TestCase):
    def test_role_selection_session_and_tokens(self):
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
        run = subprocess.run([sys.executable, '-c', RUNNER], env=env, capture_output=True, text=True, timeout=40)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

RUNNER = r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtCore import QSettings, QUrl
from PySide6.QtGui import QGuiApplication
from ui_qt.quick.megasweep_bridge import MegasweepBridge
app = QGuiApplication([])
with TemporaryDirectory() as directory:
    root = Path(directory); source = root/'sample.csv'; other = root/'other.csv'
    for p in (source, other): p.write_text('x,y,700,710\n0,0,1,2\n')
    folders = {role:root/role for role in ('custom','p','n','bn')}
    for folder in folders.values(): folder.mkdir()
    files = {'custom':folders['custom']/'reference.csv', 'p':folders['p']/'Pback.csv', 'n':folders['n']/'Nback.csv', 'bn':folders['bn']/'BNback.csv'}
    for path in files.values(): path.write_text('700,710\n1,2\n')
    settings_path = root/'settings.ini'; sessions = root/'sessions'
    bridge = MegasweepBridge(settings=QSettings(str(settings_path), QSettings.IniFormat), session_directory=sessions)
    try:
        bridge.setExperimentFolder(str(root)); bridge.setParameter('mode','Reflection'); bridge.selectFiles([str(source)])
        assert bridge.snapshot['backgroundRole'] == 'custom'
        requests=[]; bridge.sourceDialogRequested.connect(lambda purpose, multiple: requests.append((purpose,multiple)))
        bridge.openFiles('background',[str(files['custom'])]); bridge.invoke('choose-background')
        picker=bridge.backgroundSelectionBridge; picker.invoke('browse'); stale=requests[-1][0]
        bridge.setParameter('background_role','p')
        assert bridge.snapshot['backgroundFiles'] == [] and picker.snapshot['selected'] == []
        bridge.openFiles(stale,[QUrl.fromLocalFile(str(files['custom']))])
        assert bridge.snapshot['backgroundFiles'] == []
        picker.selectFile(str(files['p']),True)
        assert bridge.snapshot['backgroundFiles'] == [str(files['p'])]
        assert bridge.snapshot['backgroundFolder'] == str(folders['p'])
        bridge._result = {'title':'valid', 'diagnostics':{}}; bridge._result_sequence = bridge._sequence
        assert bridge.snapshot['canExport']
        picker.selectFile(str(files['n']),True)
        assert bridge.snapshot['backgroundSelectionError'] and bridge.snapshot['canExport']
        assert bridge.snapshot['backgroundFiles'] == [str(files['p'])]
        assert picker.snapshot['selected'] == [str(files['p'])]
        assert bridge.snapshot['status']
        bridge.setParameter('background_role','n'); assert bridge.snapshot['backgroundFiles'] == []
        assert not bridge.snapshot['backgroundSelectionError']
        picker.selectFile(str(files['n']),True)
        bridge.setParameter('background_role','bn'); picker.selectFile(str(files['bn']),True)
        bridge.setParameter('background_role','custom')
        assert bridge.snapshot['backgroundFiles'] == [str(files['custom'])]
        assert bridge.snapshot['backgroundFolder'] == str(folders['custom'])
        picker.selectFile(str(files['p']),True)
        picker.selectFile(str(files['n']),True)
        assert bridge.snapshot['backgroundSelectionError']
        assert bridge.snapshot['backgroundFiles'] == [str(files['custom']), str(files['p'])]
        picker.selectFile(str(files['p']),False)
        assert not bridge.snapshot['backgroundSelectionError']
        groups=bridge.snapshot['backgroundGroups']
        assert groups == {role:[str(path)] for role,path in files.items()}
        effective=bridge._effective_values()
        assert effective['background_role'] == 'custom' and effective['background_groups'] == groups
        picker.invoke('browse'); stale=requests[-1][0]
        bridge.selectFiles([str(other)])
        bridge.openFiles(stale,[str(files['p'])])
        assert all(not v for v in bridge.snapshot['backgroundGroups'].values())
        bridge.selectFiles([str(source)])
        assert bridge.snapshot['backgroundGroups'] == groups and bridge.snapshot['backgroundRole'] == 'custom'
        bridge.setParameter('background_role','p')
    finally: bridge.close()
    restored = MegasweepBridge(settings=QSettings(str(settings_path), QSettings.IniFormat), session_directory=sessions)
    try:
        restored.setParameter('mode','Reflection'); restored.selectFiles([str(source)])
        assert restored.snapshot['backgroundRole'] == 'p'
        assert restored.snapshot['backgroundFiles'] == [str(files['p'])]
        assert restored.snapshot['backgroundFolder'] == str(folders['p'])
        restored.setParameter('background_role','n'); assert restored.snapshot['backgroundFolder'] == str(folders['n'])
        assert restored.snapshot['backgroundGroups'] == {role:[str(path)] for role,path in files.items()}
    finally: restored.close()
'''
