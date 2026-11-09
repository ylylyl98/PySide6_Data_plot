"""Regression tests for Megasweep's inline reference picker through the shell."""
import os
import subprocess
import sys
import unittest

class MegasweepBackgroundTests(unittest.TestCase):
    def test_reference_picker_routes_and_context(self):
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
        result = subprocess.run([sys.executable, '-c', RUNNER], env=env, capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

RUNNER = r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtCore import QSettings, QUrl
from PySide6.QtGui import QGuiApplication
from ui_qt.quick.application_bridge import QuickApplicationBridge
from ui_qt.quick.background_source_catalog import BackgroundSourceCatalogBridge

app = QGuiApplication([])
with TemporaryDirectory() as directory:
    root = Path(directory).resolve(); nested = root / 'nested'; nested.mkdir()
    a, b, bg1, bg2 = [root / name for name in ('a.csv', 'b.csv', 'bg1.csv', 'bg2.csv')]
    nested_bg = nested / 'reference.csv'
    for path in (a, b, bg1, bg2, nested_bg):
        path.write_text('x,y,700,710\n0,0,1,2\n')
    settings = QSettings(str(root / 'test.ini'), QSettings.IniFormat)
    shell = QuickApplicationBridge(settings=settings)
    try:
        shell.selectWorkflow('megasweep'); shell.setSourcePath(str(root))
        child = shell.workflowBridge
        child.setParameter('mode', 'Reflection'); child.selectFiles([str(a)])
        child.openFiles('background', [str(bg1)]); child.invoke('choose-background')
        chooser = child.backgroundSelectionBridge
        # Megasweep scans nested reference directories without widening DRR's inventory.
        found = {row['id'] for row in chooser._scan_catalog(root, 'drr')['files']}
        assert str(nested_bg.resolve()) in found
        default = BackgroundSourceCatalogBridge(settings=settings, inline=True)
        try:
            assert str(nested_bg.resolve()) not in {row['id'] for row in default._scan_catalog(root, 'drr')['files']}
            cache = root / 'cache' / 'catalog.json'
            assert chooser._display_cache(root, 'drr', False, cache).path != default._display_cache(root, 'drr', False, cache).path
            assert any(row[0] == 'nested/reference.csv' for row in chooser._source_inventory(root, 'drr'))
            assert not any(row[0] == 'nested/reference.csv' for row in default._source_inventory(root, 'drr'))
        finally: default.close()
        requests = []
        shell.fileDialogRequested.connect(lambda purpose, multiple: requests.append((purpose, multiple)))
        chooser.invoke('browse'); purpose, multiple = requests[-1]
        assert purpose.startswith('inline-catalog:') and multiple
        shell.acceptFiles('megasweep', purpose, [QUrl.fromLocalFile(str(bg2))])
        assert child.snapshot['backgroundFiles'] == [str(bg1), str(bg2)]
        # Switch sources while a dialog is outstanding; the old response is rejected,
        # and a new click cannot restore the previous source's references.
        chooser.invoke('browse'); stale = requests[-1][0]
        child.selectFiles([str(b)])
        assert child.snapshot['backgroundFiles'] == [] and chooser.snapshot['selected'] == []
        shell.acceptFiles('megasweep', stale, [QUrl.fromLocalFile(str(bg1))])
        assert child.snapshot['backgroundFiles'] == []
        chooser.selectFile(str(bg2), True)
        assert child.snapshot['backgroundFiles'] == [str(bg2)]
        # Measurement-folder changes invalidate outstanding requests but preserve the reference folder.
        chooser.invoke('browse'); stale = requests[-1][0]
        shell.setSourcePath(str(nested))
        assert chooser.snapshot['folder'] == str(root.resolve())
        shell.acceptFiles('megasweep', stale, [QUrl.fromLocalFile(str(bg1))])
        assert child.snapshot['backgroundFiles'] == [str(bg2)]
        # Switching modes also invalidates the former picker, even after reopening it.
        chooser.invoke('browse'); stale = requests[-1][0]
        child.setParameter('mode', 'PL'); child.setParameter('mode', 'Reflection')
        child.invoke('choose-background')
        shell.acceptFiles('megasweep', stale, [QUrl.fromLocalFile(str(bg1))])
        assert child.snapshot['backgroundFiles'] == [str(bg2)]
        # Leaving and re-entering the workflow invalidates its former dialog token.
        chooser.invoke('browse'); stale = requests[-1][0]
        shell.selectWorkflow('pl'); shell.selectWorkflow('megasweep')
        shell.acceptFiles('megasweep', stale, [QUrl.fromLocalFile(str(bg1))])
        assert child.snapshot['backgroundFiles'] == [str(bg2)]
    finally: shell.close()
'''
