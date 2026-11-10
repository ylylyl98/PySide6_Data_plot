"""Exercise real range handles, final frames and footer geometry in native Quick."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class ColorRangeBarTests(unittest.TestCase):
    def test_native_color_controls(self):
        with TemporaryDirectory(prefix="color-bar-") as folder:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                       MPLBACKEND="agg", MPLCONFIGDIR=folder, XDG_CONFIG_HOME=folder,
                       PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                                    cwd=Path(__file__).resolve().parents[1], env=env,
                                    text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout)


def isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import time
    from threading import Event
    from unittest.mock import patch
    import numpy as np
    from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QSettings, Qt, QUrl, qInstallMessageHandler
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication, QKeyEvent
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from PySide6.QtTest import QTest
    from core.loader import DataCube
    from ui_qt.quick.application_bridge import QuickApplicationBridge
    from ui_qt.quick.plot_surface import register_plot_surface
    from ui_qt.quick.drr_workflow import DrrProducts, DrrRequest

    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    if sys.platform == 'win32':
        QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
        app.setFont(QFont('Segoe UI', 9))
    register_plot_surface()
    errors = []
    def message(kind, context, text):
        if any(marker in text for marker in ("ReferenceError", "TypeError", "Cannot assign", "Unable to assign", "Binding loop", "Error:")):
            errors.append(text)
            print(text, file=sys.stderr)
    old_handler = qInstallMessageHandler(message)
    def settle(predicate=lambda: True, timeout=20):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            app.processEvents()
            if predicate():
                for _ in range(4):
                    app.processEvents(); time.sleep(.005)
                return
            time.sleep(.002)
        raise AssertionError("Timed out waiting for color range frame")

    with TemporaryDirectory(prefix="color-bar-case-") as folder:
        bridge = QuickApplicationBridge(settings=QSettings(str(Path(folder)/"settings.ini"), QSettings.IniFormat))
        engine = QQmlApplicationEngine()
        engine.setInitialProperties({"appBridge": bridge})
        engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/"ui_qt/quick/qml/Application.qml")))
        assert engine.rootObjects(), errors
        window = engine.rootObjects()[0]
        window.setWidth(1500); window.setHeight(980); window.show()
        x = np.linspace(1., 3., 301); y = np.linspace(-2., 2., 61)
        z = 1 + np.exp(-((x[None,:]-2-y[:,None]*.05)/.1)**2)
        cube = DataCube(x,y,z,"Gate (V)","Color controls","PL")
        bridge.set_cube(cube)
        def idle():
            settle(lambda: bridge._canvas is not None and not bridge._plot_timer.isActive()
                   and not bridge._publish_timer.isActive() and not bridge._canvas.backend._timer.isActive())
        def find(name):
            found = window.findChild(QObject, name)
            if found is None:
                def visit(item):
                    if item.objectName() == name: return item
                    for child in item.childItems():
                        result = visit(child)
                        if result is not None: return result
                    return None
                found = visit(window.contentItem())
            assert found is not None, name
            return found
        def click(item):
            p = item.mapToScene(QPointF(item.width()/2,item.height()/2))
            QTest.mouseClick(window, Qt.LeftButton, Qt.NoModifier, QPoint(round(p.x()),round(p.y())))
            idle()
        def type_value(item, text):
            item.forceActiveFocus()
            settle()
            QTest.keyClick(window, Qt.Key_A, Qt.ControlModifier)
            for kind in (QEvent.KeyPress, QEvent.KeyRelease):
                app.sendEvent(window, QKeyEvent(kind, 0, Qt.NoModifier, text))
        def assert_spectrum_button_clear():
            toggle = find('app_toggle_spectrum')
            bottom = toggle.mapToScene(QPointF(0, toggle.height())).y()
            for group in bridge.colorRangeController.snapshot['groups']:
                for prefix in ('color_overall_', 'color_weak_', 'color_auto_'):
                    control = find(prefix + group['id'])
                    if control.isVisible():
                        top = control.mapToScene(QPointF(0, 0)).y()
                        assert bottom <= top, (toggle.property('text'), prefix, bottom, top)
        idle()
        controller = bridge.colorRangeController
        assert len(controller.snapshot['groups']) == 1
        bar = find('color_range_bar'); assert bar.isVisible()
        assert_spectrum_button_clear()
        spectra = bridge._spectra[0][1]
        spectrum_values = np.array(spectra.get_ydata(),copy=True)
        axes = tuple(bridge._canvas.figure.axes)
        group = controller.snapshot['groups'][0]
        slider = find('color_slider_'+group['id'])
        sidebar_min = find('color_sidebar_min_all')
        sidebar_max = find('color_sidebar_max_all')
        assert window.findChild(QObject,'color_min_all') is None  # footer has readouts, not duplicate editors
        assert sidebar_min.property('text') == group['lowText']
        # Exercise the real Save action with a blocked worker, both success and
        # failure. The visible range must neither collapse nor commit a stale
        # pending drag while the export owns the current context.
        source = Path(folder) / 'measurement.csv'
        source.write_text('fixture', encoding='utf-8')
        old_source, old_sources, old_folder, old_valid = bridge._source, bridge._sources, bridge._folder, bridge._export_valid
        bridge._source, bridge._sources, bridge._folder, bridge._export_valid = source, (str(source),), folder, True
        for fail in (False, True):
            entered, release = Event(), Event()
            def operation():
                entered.set()
                if not release.wait(10): raise RuntimeError('test export timed out')
                if fail: raise RuntimeError('test export failure')
                return {}
            before = [(item['low'], item['high']) for item in controller.snapshot['groups']]
            height = bar.height()
            controller._last = time.monotonic()
            controller.moveRange('all', .25, .75, False)
            assert controller._pending
            try:
                with patch.object(bridge, '_pl_export_request', return_value=operation), \
                     patch.object(bridge, '_pl_export_completed'), \
                     patch('ui_qt.quick.data_maintenance.run_export_with_maintenance', side_effect=lambda op, **kw: op()):
                    bridge.invoke('export')
                    settle(lambda: entered.is_set() and bridge._busy)
                    state = controller.snapshot
                    assert state['groups'] and all(not item['enabled'] for item in state['groups'])
                    assert bar.isVisible() and abs(bar.height()-height) < 2
                    assert not sidebar_min.isEnabled() and not sidebar_max.isEnabled()
                    assert not controller._pending and not controller.preview['groups']
                    assert controller.adapter.apply_pair('all', before[0][0]-.1, before[0][1]) is False
                    assert controller.adapter.auto('all') is False
                    assert controller.adapter.set_fixed('all', 'low', True) is False
                    assert controller.adapter.set_product('log') is False
                    assert [(item['low'], item['high']) for item in controller.snapshot['groups']] == before
                    busy_screenshot = os.environ.get('DPTK_COLOR_BUSY_SCREENSHOT')
                    if busy_screenshot and not fail:
                        window.grabWindow().save(busy_screenshot)
                    release.set()
                    settle(lambda: not bridge._busy)
                    idle()
                    assert bar.isVisible() and abs(bar.height()-height) < 2
                    assert all(item['enabled'] for item in controller.snapshot['groups'])
                    assert [(item['low'], item['high']) for item in controller.snapshot['groups']] == before
            finally:
                release.set()
        bridge._source, bridge._sources, bridge._folder, bridge._export_valid = old_source, old_sources, old_folder, old_valid
        # Display shortening alone must not commit or round the underlying
        # double when the user only focuses, presses Enter, or leaves.
        exact_low = group['low'] + (group['high'] - group['low']) * .123456789123
        assert controller.commitNumber('all', 'low', repr(exact_low)); idle()
        assert controller.snapshot['groups'][0]['low'] == exact_low
        assert sidebar_min.property('text') != repr(exact_low)
        sidebar_min.forceActiveFocus(); settle()
        QTest.keyClick(window, Qt.Key_Return); idle()
        sidebar_max.forceActiveFocus(); idle()
        assert controller.snapshot['groups'][0]['low'] == exact_low
        typed_low = group['low'] + (group['high'] - group['low']) * .234567891234
        type_value(sidebar_min, repr(typed_low))
        QTest.keyClick(window, Qt.Key_Return); idle()
        assert controller.snapshot['groups'][0]['low'] == typed_low
        assert controller.commitNumber('all', 'low', repr(group['low'])); idle()
        # Lightweight value notifications reach the sidebar before any timer or
        # image refresh runs. The expensive canonical update remains throttled.
        before_draw = bridge._canvas.draw_count
        controller.moveRange('all', .3, .8, False)
        assert sidebar_min.property('text') == controller.preview['groups']['all']['lowText']
        assert bridge._canvas.draw_count == before_draw
        assert controller.snapshot['groups'][0]['low'] == group['low']
        controller.moveRange('all',group['first'],group['second'],True)
        idle()
        # Drag the lower handle with real window events, then verify the final
        # frame/state and unchanged spectrum rather than only calling a slot.
        track = slider.width()-slider.property('leftPadding')-slider.property('rightPadding')-16
        start = slider.mapToScene(QPointF(slider.property('leftPadding')+group['first']*track+8,slider.height()/2))
        end = slider.mapToScene(QPointF(slider.property('leftPadding')+(group['first']*.7+group['second']*.3)*track+8,slider.height()/2))
        QTest.mousePress(window,Qt.LeftButton,Qt.NoModifier,start.toPoint())
        QTest.mouseMove(window,end.toPoint(),30)
        settle(lambda: getattr(bridge._canvas, 'color_range_preview', False), timeout=3)
        assert getattr(bridge._canvas, 'color_range_preview', False)
        assert sidebar_min.property('text') == controller.preview['groups']['all']['lowText']
        QTest.mouseRelease(window,Qt.LeftButton,Qt.NoModifier,end.toPoint())
        idle()
        updated = controller.snapshot['groups'][0]
        assert updated['low'] > group['low'], (group,updated)
        assert not controller._pending
        assert not bridge._canvas.color_range_preview
        assert tuple(bridge._canvas.figure.axes) == axes
        assert bridge._spectra[0][1] is spectra
        np.testing.assert_array_equal(spectrum_values,spectra.get_ydata())
        mesh = bridge._maps[0][0]._dptk_quick_render[0].primary
        np.testing.assert_allclose(mesh.get_clim(),(updated['low'],updated['high']))
        assert sidebar_min.property('text') == updated['lowText']
        # Keyboard edits flow back into the same range; legal values beyond the
        # previous domain expand it. Fix still permits an explicit numeric edit.
        type_value(sidebar_min, '0.5')
        QTest.keyClick(window, Qt.Key_Return); idle()
        assert controller.snapshot['groups'][0]['low'] == .5
        type_value(sidebar_max, '5e0')
        QTest.keyClick(window, Qt.Key_Return); idle()
        assert controller.snapshot['groups'][0]['high'] == 5.
        assert controller.snapshot['groups'][0]['domainHigh'] >= 5.
        np.testing.assert_allclose(mesh.get_clim(),(.5,5.))
        sidebar_min.forceActiveFocus(); settle()
        click(find('color_sidebar_fix_min_all'))
        assert controller.snapshot['groups'][0]['fixedLow']
        type_value(sidebar_min, '0.25')
        QTest.keyClick(window, Qt.Key_Return); idle()
        assert controller.snapshot['groups'][0]['low'] == .25
        controller.moveRange('all',.4,.9,True); idle()
        assert controller.snapshot['groups'][0]['low'] == .25
        # Pending text survives snapshot and live-preview updates. It commits
        # on focus leave and cancels queued older slider values.
        type_value(sidebar_max, '6')
        controller.refresh()
        controller.moveRange('all',.1,.7,False)
        assert sidebar_max.property('text') == '6'
        sidebar_min.forceActiveFocus(); idle()
        assert controller.snapshot['groups'][0]['high'] == 6.
        assert not controller._pending and not controller.preview['groups']
        QTest.qWait(100); idle()
        assert controller.snapshot['groups'][0]['high'] == 6.
        # Invalid input remains editable and Escape restores the current value.
        type_value(sidebar_min, '100')
        QTest.keyClick(window, Qt.Key_Return); idle()
        assert sidebar_min.property('text') == '100'
        assert controller.snapshot['groups'][0]['low'] == .25
        QTest.keyClick(window, Qt.Key_Escape)
        assert sidebar_min.property('text') == controller.snapshot['groups'][0]['lowText']
        assert not controller.snapshot['error']
        controller.setFixed('all','low',False); idle()
        updated = controller.snapshot['groups'][0]
        # A release from an old source must not apply its positions to a newly
        # loaded measurement, even when the region ID is still "all".
        start = slider.mapToScene(QPointF(slider.property('leftPadding')+updated['first']*track+8,slider.height()/2))
        QTest.mousePress(window,Qt.LeftButton,Qt.NoModifier,start.toPoint())
        bridge.set_cube(DataCube(x,y,z*2,"Gate (V)","Replacement data","PL"))
        idle()
        replacement = [(g['low'],g['high']) for g in controller.snapshot['groups']]
        QTest.mouseRelease(window,Qt.LeftButton,Qt.NoModifier,end.toPoint())
        idle()
        assert [(g['low'],g['high']) for g in controller.snapshot['groups']] == replacement
        # One/two/three regions use the same footer height and separate groups.
        heights = []
        for count in (1,2,3):
            bridge.plDisplayBridge.setChoice('regionCount',count-1); idle()
            assert len(controller.snapshot['groups']) == count
            heights.append(bar.height())
            for item in controller.snapshot['groups']:
                assert find('color_slider_'+item['id']).isVisible()
        assert max(heights)-min(heights) < 3, heights
        old_colors = [(g['low'],g['high']) for g in controller.snapshot['groups']]
        click(find('app_toggle_spectrum'))
        assert_spectrum_button_clear()
        assert [(g['low'],g['high']) for g in controller.snapshot['groups']] == old_colors
        assert bar.isVisible()
        plot = find('app_plot_surface')
        assert plot.mapToScene(QPointF(0,plot.height())).y() <= bar.mapToScene(QPointF(0,0)).y()
        # DRR recommendations have moved out of the sidebar into each region.
        bridge.selectWorkflow('drr')
        raw = DataCube(x,y,z-1.3,"Gate (V)","DRR color controls","ΔR/R")
        bridge._drr_products = DrrProducts(DrrRequest(()),None,raw,raw,21,21)
        bridge.set_cube(raw); idle()
        bridge.drrDisplayBridge.setChoice('regionCount',2); idle()
        # Editing tabs choose the same product for colormap, sidebar numbers
        # and slider even when only the raw heatmap is displayed.
        bridge.drrDisplayBridge.setChoice('colorProduct',1); idle()
        assert bridge._drr_processing['drrView'] == 'raw'
        assert controller.snapshot['productId'] == 'second'
        original_raw = bridge.drrDisplayBridge._numbers['raw.left_vmax']
        target_high = controller.snapshot['groups'][0]['high'] + 1.123456789123
        type_value(find('color_sidebar_max_left'),str(target_high))
        QTest.keyClick(window,Qt.Key_Return); idle()
        assert bridge.drrDisplayBridge._numbers['second.left_vmax'] == target_high
        assert bridge.drrDisplayBridge._numbers['raw.left_vmax'] == original_raw
        bridge.drrDisplayBridge.setChoice('colorProduct',0); idle()
        for g in controller.snapshot['groups']:
            assert find('color_overall_'+g['id']).isVisible()
            assert find('color_weak_'+g['id']).isVisible()
        first_group = controller.snapshot['groups'][0]
        click(find('color_weak_'+first_group['id']))
        assert not controller.snapshot['error'], controller.snapshot['error']
        assert controller.snapshot['groups'][0]['status']
        if not bridge._values['spectrumVisible']:
            click(find('app_toggle_spectrum'))
        QTest.mouseMove(window,QPoint(20,20)); QTest.qWait(50)
        find('color_sidebar_min_left').forceActiveFocus(); settle()
        screenshot = os.environ.get('DPTK_COLOR_SCREENSHOT')
        if screenshot:
            window.grabWindow().save(screenshot)
        bridge._appearance.setMode('dark'); idle()
        assert_spectrum_button_clear()
        if screenshot:
            window.grabWindow().save(str(Path(screenshot).with_stem(Path(screenshot).stem+'-dark')))
        window.setWidth(920); idle()
        assert_spectrum_button_clear()
        click(find('app_toggle_spectrum'))
        assert_spectrum_button_clear()
        assert bar.height() < 190, bar.height()
        if screenshot:
            window.grabWindow().save(str(Path(screenshot).with_stem(Path(screenshot).stem+'-narrow')))
        assert not errors, errors
        print('COLOR_BAR_QML_OK regions=1,2,3; mouse drag; sidebar live sync; keyboard edits; draft protection; Fix; final frame; spectrum; recommendations; narrow layout')
        window.close(); bridge.close(); engine.deleteLater(); app.processEvents()
    qInstallMessageHandler(old_handler)


if __name__ == '__main__':
    if '--isolated' in sys.argv: isolated()
    else: unittest.main()
