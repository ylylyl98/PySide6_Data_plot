"""Real shell regression for Megasweep reference-role controls."""
from pathlib import Path
import os, subprocess, sys, tempfile, unittest
ROOT = Path(__file__).resolve().parents[1]

class RoleUiTests(unittest.TestCase):
    def test_real_role_selector(self):
        with tempfile.TemporaryDirectory(prefix='dptk-role-ui-') as directory:
            env = dict(os.environ, QT_QPA_PLATFORM='offscreen', QT_QUICK_BACKEND='software', MPLBACKEND='agg', MPLCONFIGDIR=directory, XDG_CONFIG_HOME=directory, XDG_CACHE_HOME=directory, PYTHONIOENCODING='utf-8')
            result = subprocess.run([sys.executable,__file__,'--child',directory], cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

def child(directory):
    sys.path.insert(0,str(ROOT))
    import time
    import numpy as np
    import pandas as pd
    from PySide6.QtCore import QObject,QSettings,QUrl,QEvent,QCoreApplication,Qt,qInstallMessageHandler
    from PySide6.QtGui import QGuiApplication,QFont,QFontDatabase
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from PySide6.QtTest import QTest
    from ui_qt.quick.application_bridge import QuickApplicationBridge
    from ui_qt.quick.plot_surface import register_plot_surface
    root=Path(directory).resolve(); wavelength=np.linspace(800,1000,31); columns=['Vbg','Vtg',*map(str,wavelength)]
    rows=[[x,y,*(1000+100*np.exp(-((wavelength-925)/25)**2))] for x in (-1,0,1) for y in (-1,0,1)]
    source=root/'measurement.csv'; pd.DataFrame(rows,columns=columns).to_csv(source,index=False)
    files={}
    for role,name in {'custom':'mystery.csv','p':'Pback.csv','n':'Nback.csv','bn':'BNback.csv'}.items():
        files[role]=root/name; pd.DataFrame([rows[0]],columns=columns).to_csv(files[role],index=False)
    QQuickStyle.setStyle('Basic'); app=QGuiApplication([])
    if sys.platform=='win32': QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf'); app.setFont(QFont('Segoe UI',9))
    errors=[]
    def handler(kind,context,message):
        if any(x in message for x in ('ReferenceError','TypeError','Binding loop','Cannot assign','Unable to assign')): errors.append(message)
    old=qInstallMessageHandler(handler); bridge=QuickApplicationBridge(settings=QSettings(str(root/'settings.ini'),QSettings.IniFormat)); register_plot_surface()
    engine=QQmlApplicationEngine(); engine.setInitialProperties({'appBridge':bridge}); engine.load(QUrl.fromLocalFile(str(ROOT/'ui_qt/quick/qml/Application.qml'))); assert engine.rootObjects()
    window=engine.rootObjects()[0]
    def wait(predicate,label):
        end=time.monotonic()+25
        while time.monotonic()<end:
            app.processEvents()
            if predicate(): app.processEvents(); return
            time.sleep(.01)
        raise AssertionError(label)
    def find(name):
        def walk(item):
            if item.objectName()==name:return item
            for obj in item.childItems() if hasattr(item,'childItems') else item.children():
                result=walk(obj)
                if result is not None:return result
        result=walk(window.contentItem()) or window.findChild(QObject,name)
        assert result is not None,name
        return result
    def click(name):
        item=find(name); wait(lambda:item.isVisible() and item.isEnabled(),name+" enabled")
        point=item.mapToScene(item.boundingRect().center()).toPoint()
        assert 0<=point.x()<window.width() and 0<=point.y()<window.height(),(name,point)
        QTest.mouseClick(window,Qt.LeftButton,Qt.NoModifier,point); app.processEvents()
    def settled():
        # Parameter changes debounce analysis on a single-shot timer. A false
        # busy flag alone can precede the queued load and its QML update.
        wait(lambda:not child_bridge._timer.isActive() and not child_bridge.snapshot['busy'],'analysis and debounce')
        app.processEvents()
    try:
        bridge.selectWorkflow('megasweep'); child_bridge=bridge.workflowBridge
        bridge.setSourcePath(str(root)); child_bridge.setParameter('mode','Reflection'); child_bridge.selectFiles([str(source)]); settled()
        child_bridge.setParameter('auto_background_scale',False)
        for role in ('custom','p','n','bn'):
            child_bridge.setParameter('background_role',role); child_bridge.openFiles('background',[str(files[role])]); settled()
        child_bridge.setParameter('background_role','custom'); settled()
        for width in (640,1000):
            window.setWidth(width); window.setHeight(560); app.processEvents(); click('megasweep_background')
            wait(lambda:find('megasweep_background_browser').isVisible(),'background dialog')
            role=find('megasweep_background_role'); assert role.property('currentText')=='Custom · 1'
            for name in ('megasweep_background_role','megasweep_background_role_help','drr_background_inline_browse','megasweep_background_close'):
                item=find(name); pt=item.mapToScene(item.boundingRect().center()); assert 0<=pt.x()<width and 0<=pt.y()<560,(width,name,pt)
            output=ROOT/'build/megasweep/background-roles'; output.mkdir(parents=True,exist_ok=True); assert window.grabWindow().save(str(output/f'roles-{width}.png'))
            click('megasweep_background_role'); QTest.keyClick(window,Qt.Key_Down); QTest.keyClick(window,Qt.Key_Return)
            wait(lambda:child_bridge.snapshot['backgroundRole']=='p','P role'); assert role.property('currentText')=='P · 1'
            assert child_bridge.snapshot['backgroundGroups']['custom']==[str(files['custom'])]
            chooser=child_bridge.backgroundSelectionBridge; assert chooser.snapshot['selected']==[str(files['p'])]
            wait(lambda:not chooser.snapshot['busy'],'P inventory')
            names={row['name'] for row in chooser.filesModel.rows}
            assert files['p'].name in names and files['n'].name not in names and files['bn'].name not in names, names
            chooser.selectFile(str(files['n']),True); app.processEvents()
            assert child_bridge.snapshot['backgroundGroups']['p']==[str(files['p'])]
            assert chooser.snapshot['selected']==[str(files['p'])]
            assert 'not P' in child_bridge.snapshot['backgroundSelectionError']
            assert 'not P' in find('megasweep_background_status').property('text')
            chooser.addExternal([QUrl.fromLocalFile(str(files['custom']))]); app.processEvents()
            assert child_bridge.snapshot['backgroundGroups']['p']==[str(files['p']),str(files['custom'])]
            assert not child_bridge.snapshot['backgroundSelectionError']
            chooser.selectFile(str(files['custom']),False); app.processEvents()
            settled()
            for next_role in ('n','bn'):
                click('megasweep_background_role'); QTest.keyClick(window,Qt.Key_Down); QTest.keyClick(window,Qt.Key_Return)
                wait(lambda:child_bridge.snapshot['backgroundRole']==next_role,next_role+' role')
                assert chooser.snapshot['selected']==[str(files[next_role])]
                assert role.property('currentText')==next_role.upper()+' · 1'
                settled()
            click('megasweep_background_role'); QTest.keyClick(window,Qt.Key_Home); QTest.keyClick(window,Qt.Key_Return)
            wait(lambda:child_bridge.snapshot['backgroundRole']=='custom','custom role'); assert chooser.snapshot['selected']==[str(files['custom'])]
            click('megasweep_background_close'); settled()
        assert not errors,errors
    finally:
        bridge.close(); window.close(); engine.deleteLater(); QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete); app.processEvents(); qInstallMessageHandler(old)

if __name__=='__main__':
    if '--child' in sys.argv: child(sys.argv[-1])
    else: unittest.main()
