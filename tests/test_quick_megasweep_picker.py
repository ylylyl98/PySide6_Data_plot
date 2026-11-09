"""Real shell regression tests for Megasweep picker locations and scopes."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class MegasweepPickerTests(unittest.TestCase):
    def test_real_shell_picker(self):
        with tempfile.TemporaryDirectory(prefix='dptk-picker-test-') as temporary:
            env = dict(os.environ, QT_QPA_PLATFORM='offscreen', QT_QUICK_BACKEND='software',
                       MPLBACKEND='agg', MPLCONFIGDIR=temporary, XDG_CONFIG_HOME=temporary,
                       XDG_CACHE_HOME=temporary, PYTHONIOENCODING='utf-8')
            run = subprocess.run([sys.executable, __file__, '--child', temporary], cwd=ROOT,
                                 env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


def child(tmp):
    sys.path.insert(0, str(ROOT))
    import time
    import numpy as np
    import pandas as pd
    from PySide6.QtCore import QObject, QSettings, QUrl, Qt, QMetaObject, QEvent, QCoreApplication, qInstallMessageHandler
    from PySide6.QtGui import QGuiApplication, QFont, QFontDatabase
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from PySide6.QtTest import QTest
    from ui_qt.quick.application_bridge import QuickApplicationBridge
    from ui_qt.quick.plot_surface import register_plot_surface

    base = Path(tmp); experiment = base / 'experiment'; a = experiment / 'A'; nested = a / 'nested'; b = experiment / 'B'; external = base / 'external'
    for folder in (a, nested, b, external): folder.mkdir(parents=True)
    wavelengths = np.linspace(800, 1000, 31)
    cols = ['Vbg', 'Vtg', *map(str, wavelengths)]
    spectrum = pd.DataFrame([[x, y, *(1000 + np.exp(-((wavelengths - 925) / 25)**2) * (100+x+y))] for x in (-1,0,1) for y in (-1,0,1)], columns=cols)
    for path in (a/'pl.csv', nested/'nested.csv', b/'reflection.csv', external/'outside.csv'):
        spectrum.to_csv(path, index=False)
    pd.DataFrame([(x,y,x+y+5) for x in (-1,0,1) for y in (-1,0,1)], columns=['Vbg','Vtg','Current']).to_csv(b/'transport.csv', index=False)
    background_dir = base/'background'; background_dir.mkdir(); spectrum.iloc[:1].to_csv(background_dir/'reference.csv',index=False)
    QQuickStyle.setStyle('Basic'); app = QGuiApplication([])
    if sys.platform == 'win32': QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf'); app.setFont(QFont('Segoe UI',9))
    errors=[]
    def handler(kind, context, message):
        if any(x in message for x in ('ReferenceError','TypeError','Binding loop','Cannot assign','Unable to assign')): errors.append(message)
    old_handler=qInstallMessageHandler(handler)
    settings=QSettings(str(base/'settings.ini'),QSettings.IniFormat)
    bridge=QuickApplicationBridge(settings=settings); register_plot_surface(); engine=QQmlApplicationEngine()
    engine.setInitialProperties({'appBridge':bridge}); engine.load(QUrl.fromLocalFile(str(ROOT/'ui_qt/quick/qml/Application.qml')))
    assert engine.rootObjects(), 'shell failed to load'
    window=engine.rootObjects()[0]; window.setWidth(1000); window.setHeight(700)
    def wait(predicate, label, timeout=25):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            app.processEvents()
            if predicate(): app.processEvents(); return
            time.sleep(.01)
        raise AssertionError(label)
    def find(name):
        def walk(item):
            if item.objectName()==name: return item
            for obj in item.childItems() if hasattr(item,'childItems') else item.children():
                found=walk(obj)
                if found is not None: return found
        result=walk(window.contentItem()) or window.findChild(QObject,name)
        assert result is not None, name
        return result
    def click(name):
        obj=find(name); assert obj.isVisible() and obj.isEnabled(), name
        pt=obj.mapToScene(obj.boundingRect().center()).toPoint()
        assert 0 <= pt.x() < window.width() and 0 <= pt.y() < window.height(), (name,pt)
        QTest.mouseClick(window,Qt.LeftButton,Qt.NoModifier,pt); app.processEvents()
    def open_picker():
        click('app_open_file'); wait(lambda: find('quick_source_catalog').property('visible'), 'source picker')
        wait(lambda: not catalog.snapshot['busy'], 'catalog inventory')
    def close_picker():
        QMetaObject.invokeMethod(find('quick_source_catalog'),'reject'); app.processEvents()
        assert not find('quick_source_catalog').property('visible')
    def paths(): return {str(Path(row['id']).resolve()) for row in catalog.filesModel.rows}
    def is_path(actual, expected): return Path(actual).resolve()==Path(expected).resolve()
    def wait_child(): wait(lambda: not child.snapshot['busy'], 'analysis')
    try:
        bridge.selectWorkflow('megasweep'); child=bridge.workflowBridge; catalog=bridge.sourceCatalogBridge
        bridge.setSourcePath(str(experiment)); child.selectFiles([str(a/'pl.csv')]); wait_child()
        open_picker(); assert is_path(catalog.snapshot['folder'], a), catalog.snapshot
        assert catalog.snapshot['selected']==[str((a/'pl.csv').resolve())]
        assert str((a/'pl.csv').resolve()) in paths() and str((nested/'nested.csv').resolve()) not in paths()
        scope=find('megasweep_picker_scope'); click('megasweep_picker_scope'); QTest.keyClick(window,Qt.Key_Down); QTest.keyClick(window,Qt.Key_Return); app.processEvents()
        wait(lambda: not catalog.snapshot['busy'], 'subfolder scope')
        assert catalog.snapshot['megasweepScope']=='subfolders', catalog.snapshot
        assert str((nested/'nested.csv').resolve()) in paths() and str((b/'reflection.csv').resolve()) not in paths()
        assert catalog.snapshot['selected']==[str((a/'pl.csv').resolve())]
        click('megasweep_picker_scope'); QTest.keyClick(window,Qt.Key_Down); QTest.keyClick(window,Qt.Key_Return); app.processEvents(); wait(lambda: not catalog.snapshot['busy'], 'experiment scope')
        assert str((b/'reflection.csv').resolve()) in paths()
        catalog.setParameter('query','nested'); close_picker(); open_picker()
        assert catalog.snapshot['query']=='' and catalog.snapshot['megasweepScope']=='experiment'
        assert catalog.snapshot['focused']==str((a/'pl.csv').resolve())
        for width in (640,1000):
            window.setWidth(width); window.setHeight(560); app.processEvents()
            for name in ('source_catalog_search','megasweep_picker_scope','source_catalog_browse','source_catalog_load'):
                obj=find(name); pt=obj.mapToScene(obj.boundingRect().center())
                assert obj.isVisible() and 0 <= pt.x() < width and 0 <= pt.y() < 560, (width,name,pt)
            output=ROOT/'build/megasweep/picker-location'; output.mkdir(parents=True,exist_ok=True)
            assert window.grabWindow().save(str(output/f'picker-{width}.png'))
        close_picker()
        bridge.setSourcePath(str(b)); assert is_path(child.snapshot['sourceFile'],a/'pl.csv')
        open_picker(); assert is_path(catalog.snapshot['folder'],b); close_picker()
        child.setParameter('mode','Reflection'); child.selectFiles([str(b/'reflection.csv')]); wait_child()
        child.openFiles('background',[str(background_dir/'reference.csv')]); wait_child()
        assert is_path(child.snapshot['backgroundFolder'],background_dir)
        open_picker(); assert is_path(catalog.snapshot['folder'],b); assert catalog.snapshot['megasweepScope']=='folder'
        click('source_catalog_browse'); chooser=find('app_file_dialog'); wait(lambda: chooser.property('visible'),'external file dialog')
        QMetaObject.invokeMethod(chooser,'reject'); app.processEvents(); assert find('quick_source_catalog').property('visible')
        bridge.acceptFiles('megasweep','catalog',[QUrl.fromLocalFile(str(external/'outside.csv'))]); app.processEvents()
        assert catalog.snapshot['selected']==[str((external/'outside.csv').resolve())]
        click('source_catalog_load'); wait_child(); assert is_path(child.snapshot['sourceFile'],external/'outside.csv')
        assert is_path(child.snapshot['backgroundFolder'],background_dir)
        child.setParameter('mode','Transport'); child.selectFiles([str(b/'transport.csv')]); wait_child()
        open_picker(); assert is_path(catalog.snapshot['folder'],b); assert catalog.snapshot['megasweepScope']=='folder'; close_picker()
        child.setParameter('mode','PL'); wait_child(); open_picker()
        assert catalog.snapshot['megasweepScope']=='experiment'
        assert is_path(catalog.snapshot['folder'],b)
        close_picker()
        for workflow in ('pl','drr'):
            bridge.selectWorkflow(workflow); bridge.invoke('select-files'); app.processEvents(); wait(lambda: catalog.snapshot['workflow']==workflow, 'workflow catalog')
            assert catalog.snapshot['workflow']==workflow
            assert find('megasweep_picker_scope').property('visible') is False
            if find('quick_source_catalog').property('visible'): close_picker()
        assert not errors, errors
    finally:
        bridge.close(); window.close(); engine.deleteLater(); QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete); app.processEvents(); qInstallMessageHandler(old_handler)

if __name__=='__main__':
    if '--child' in sys.argv: child(sys.argv[-1])
    else: unittest.main()
