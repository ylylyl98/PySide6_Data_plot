"""Exercise real Quick shell, three Megasweep modes, exports and isolation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'build/megasweep/integration')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('QT_QUICK_BACKEND', 'software')
    from PySide6.QtCore import QSettings, QUrl, QEvent, QCoreApplication, qInstallMessageHandler, QObject, QMetaObject
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from ui_qt.quick.application_bridge import QuickApplicationBridge
    from ui_qt.quick.plot_surface import register_plot_surface

    QQuickStyle.setStyle('Basic')
    app = QGuiApplication([])
    from ui_qt.typography import install_application_font
    install_application_font(app)
    errors = []
    def message(kind, context, text):
        if any(word in text for word in ('ReferenceError', 'TypeError', 'Binding loop', 'Cannot assign', 'Unable to assign', 'Error:')):
            errors.append(text)
    old_handler = qInstallMessageHandler(message)
    with tempfile.TemporaryDirectory(prefix='dptk-megasweep-integration-') as temporary:
        root = Path(temporary)
        settings = QSettings(str(root / 'settings.ini'), QSettings.IniFormat)
        bridge = QuickApplicationBridge(settings=settings)
        register_plot_surface()
        engine = QQmlApplicationEngine()
        engine.setInitialProperties({'appBridge': bridge})
        engine.load(QUrl.fromLocalFile(str(ROOT / 'ui_qt/quick/qml/Application.qml')))
        assert engine.rootObjects(), 'Application QML did not load'
        window = engine.rootObjects()[0]
        window.setWidth(1440); window.setHeight(940)
        def wait(predicate, label, timeout=25):
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                app.processEvents()
                if predicate():
                    return
                time.sleep(.01)
            raise AssertionError(label + ': ' + str(bridge.workflowBridge.snapshot))
        def settle():
            end = time.monotonic() + .4
            while time.monotonic() < end:
                app.processEvents(); time.sleep(.01)
        def loaded(child):
            wait(lambda: child.snapshot['canExport'] or bool(child.snapshot['error']), 'analysis')
            assert not child.snapshot['error'], child.snapshot['error']
            settle()
        captures = []
        def capture(name):
            settle()
            destination = args.output / (name + '.png')
            assert window.grabWindow().save(str(destination)), 'capture failed'
            captures.append(str(destination))
        try:
            settle()
            assert 'PySide6.QtWidgets' not in sys.modules
            bridge.selectWorkflow('megasweep')
            child = bridge.workflowBridge
            bridge.setSourcePath(str(root))
            import numpy as np
            import pandas as pd
            wavelength = np.linspace(660, 820, 41)
            rows = []
            for x in (-1., 0., 1.):
                for y in (-2., 0., 2.):
                    spectrum = 1000 + (200 + 20*x + 10*y) * np.exp(-((wavelength - 745 - x*3-y)/12)**2)
                    rows.append([x, y, *spectrum])
            optical = root / 'optical.csv'
            columns = ['Vbg', 'Vtg', *map(str, wavelength)]
            pd.DataFrame(rows, columns=columns).to_csv(optical, index=False)
            background = root / 'reference.csv'
            pd.DataFrame([[0, 0, *np.full(41, 1000.)]], columns=columns).to_csv(background, index=False)
            transport = root / 'transport.csv'
            pd.DataFrame([(x,y,1e-9*(x+2)*(y+3)) for x in (-1,0,1) for y in (-2,0,2)], columns=['Vbg','Vtg','Current']).to_csv(transport,index=False)
            bridge.selectFiles([QUrl.fromLocalFile(str(optical))])
            child.setParameter("energy_min", 1.6); child.setParameter("energy_max", 1.8)
            loaded(child)
            capture('pl-light')
            child.invoke('select-files'); settle()
            assert bridge.sourceCatalogBridge.snapshot['workflow'] == 'megasweep'
            popup = window.findChild(QObject, 'quick_source_catalog')
            assert popup is not None
            QMetaObject.invokeMethod(popup, 'reject')
            settle()
            assert not popup.property('visible')
            child.setParameter('cut_type', 'x'); child.setParameter('cut_value', 0.)
            child.setParameter('view', 'slice'); loaded(child)
            child.setParameter('derivative', 'second_derivative'); loaded(child)
            assert child._result['kind'] == 'slice'
            capture('slice-light')
            elsewhere = root / 'another-experiment'; elsewhere.mkdir()
            bridge.setSourcePath(str(elsewhere))
            child.invoke('export')
            wait(lambda: not child.snapshot['busy'], 'export')
            paths = child.snapshot['resultPaths']
            assert paths and all(Path(p).is_file() and not Path(p).is_relative_to(elsewhere) for p in paths), paths
            child.setParameter('mode', 'Reflection')
            child.setParameter('auto_background_scale', False)
            child.openFiles('background', [str(background)])
            child.selectFiles([str(optical)])
            # A new source restores its own saved backgrounds; set the explicit reference afterwards.
            child.openFiles('background', [str(background)])
            child.setParameter('energy_min', 1.6); child.setParameter('energy_max', 1.8)
            child.setParameter('auto_background_scale', False)
            loaded(child)
            bridge.invoke('theme-dark'); capture('reflection-dark')
            child.invoke('reflection-preview')
            wait(lambda: not child.snapshot['busy'], 'reference preview')
            assert not child.snapshot['error'], child.snapshot['error']
            assert len(child._canvas.figure.axes) == 2
            capture('reflection-reference-dark')
            child.invoke('update'); loaded(child)
            child.setParameter('mode', 'Transport')
            child.selectFiles([str(transport)])
            child.setParameter('x_column', 'Vbg'); child.setParameter('y_column', 'Vtg'); child.setParameter('channel', 'Current')
            loaded(child); capture('transport-dark')
            child.setParameter('mode', 'PL'); loaded(child)
            assert child.snapshot['sourceFile'] == str(optical)
            assert child.snapshot['values']['view'] == 'slice'
            for key, value in {'movie.width': 800, 'movie.height': 600, 'movie.fps': 5, 'movie.hold': .2}.items():
                child.setParameter(key, value)
            child.invoke('movie-preview')
            wait(lambda: not child.snapshot['busy'], 'movie preview')
            assert not child.snapshot.get('movieError'), child.snapshot.get('movieError')
            assert child.snapshot['moviePreviewFrameCount'] > 0
            last_frame = child.snapshot['moviePreviewFrameCount'] - 1
            child.setParameter('movie.previewFrame', last_frame)
            wait(lambda: not child.snapshot['busy'] and child.snapshot['moviePreviewFrameIndex'] == last_frame, 'movie seek')
            movie_dialog = window.findChild(QObject, 'megasweep_movie_dialog')
            assert movie_dialog is not None
            QMetaObject.invokeMethod(movie_dialog, 'open'); capture('movie-preview-dark')
            QMetaObject.invokeMethod(movie_dialog, 'close')
            child.invoke('export-video')
            wait(lambda: not child.snapshot['busy'], 'movie export', timeout=45)
            assert not child.snapshot.get('movieError'), child.snapshot.get('movieError')
            movie_paths = child.snapshot['resultPaths']
            assert any(str(p).endswith('.mp4') and Path(p).is_file() for p in movie_paths), movie_paths
            bridge.selectWorkflow('pl'); bridge.selectWorkflow('megasweep'); settle()
            assert bridge.workflowBridge is child
            assert child.snapshot['sourceFile'] == str(optical)
            assert 'PySide6.QtWidgets' not in sys.modules
            assert not errors, errors
            report = dict(status='ok', captures=captures, exports=paths, movie_exports=movie_paths, errors=errors,
                          modes=['PL','Reflection','Transport'], widgets_imported=False)
            (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report, indent=2))
        finally:
            bridge.close(); window.close(); engine.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
            qInstallMessageHandler(old_handler)


if __name__ == '__main__':
    main()
