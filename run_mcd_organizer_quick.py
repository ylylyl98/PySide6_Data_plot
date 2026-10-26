"""Standalone native Qt Quick MCD Organizer, without Qt Widgets imports."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path,nargs="?")
    parser.add_argument("--settings",type=Path)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--capture",type=Path)
    args=parser.parse_args(argv)
    from PySide6.QtCore import QCoreApplication,QEvent,QSettings,QTimer,QUrl
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from ui_qt.quick.organizer_bridge import OrganizerBridge
    from ui_qt.quick.plot_surface import register_plot_surface
    from ui_qt.typography import install_application_font
    from tempfile import TemporaryDirectory
    temporary=TemporaryDirectory(prefix="dptk-quick-organizer-") if args.check else None
    if args.check and args.folder is None:args.folder=Path(temporary.name)
    settings=QSettings(str(args.settings or Path(temporary.name)/"quick.ini"),QSettings.Format.IniFormat) if args.settings or temporary else QSettings("DPTK","PlotWorkbench")
    QQuickStyle.setStyle("Basic");app=QGuiApplication([sys.argv[0]]);app.setApplicationName("DPTK MCD Organizer");app.setOrganizationName("DPTK")
    install_application_font(app)
    register_plot_surface();bridge=OrganizerBridge(settings=settings);engine=QQmlApplicationEngine();engine.setInitialProperties(dict(organizerBridge=bridge))
    engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parent/"ui_qt"/"quick"/"qml"/"Organizer.qml")))
    if not engine.rootObjects():bridge.close();return 2
    window=engine.rootObjects()[0]
    if args.folder:QTimer.singleShot(0,lambda:bridge.setExperimentFolder(str(args.folder)))
    if args.check:
        def finish():
            if bridge.snapshot["busy"] or bridge.snapshot["analysisBusy"] or bridge._preview_timer.isActive() or bridge._plot_timer.isActive():QTimer.singleShot(80,finish);return
            if args.capture:args.capture.parent.mkdir(parents=True,exist_ok=True);window.grabWindow().save(str(args.capture))
            window.close();app.quit()
        QTimer.singleShot(600,finish);QTimer.singleShot(30000,lambda:app.exit(4))
    result=app.exec()
    engine.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete);app.processEvents()
    bridge.close()
    if temporary:temporary.cleanup()
    return result


if __name__=="__main__":raise SystemExit(main())
