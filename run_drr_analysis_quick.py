"""Standalone native Qt Quick DRR Analysis, compatible with schema-one IPC."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def server_name(session):
    return "DPTK-DRR-"+hashlib.sha256(str(Path(session).resolve()).casefold().encode()).hexdigest()[:24]


def main(argv=None):
    from PySide6.QtCore import QCoreApplication, QEvent, QLockFile, QSettings, QTimer, QUrl
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtNetwork import QLocalServer, QLocalSocket
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from ui_qt.typography import install_application_font

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--settings-file", type=Path)
    parser.add_argument("--check", action="store_true", help="Load and render, write optional report, then close.")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    from ui_qt.quick.settings_storage import analysis_session_path
    temporary=tempfile.TemporaryDirectory(prefix="dptk-quick-drr-") if args.check else None
    if args.folder is None:
        args.folder=temporary.name if temporary else str(Path.cwd())
    if args.session is None:
        injected_settings=QSettings(str(args.settings_file.resolve()),QSettings.IniFormat) if args.settings_file else None
        args.session=Path(temporary.name)/"workspace.npz" if temporary else analysis_session_path(injected_settings,"drr")
    args.session=args.session.resolve()
    if args.settings_file is None and temporary:
        args.settings_file=Path(temporary.name)/"quick.ini"
    os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")
    QQuickStyle.setStyle("Basic")
    app = QGuiApplication.instance() or QGuiApplication([sys.argv[0]])
    install_application_font(app)
    app.setApplicationName("DPTK DRR Analysis")
    app.setOrganizationName("ylylyl98")
    name = server_name(args.session)
    message = dict(snapshot=str(args.snapshot.resolve()) if args.snapshot else None, folder=str(Path(args.folder).resolve()))
    socket = QLocalSocket()
    socket.connectToServer(name)
    if socket.waitForConnected(500):
        socket.write((json.dumps(message)+"\n").encode())
        socket.flush()
        if socket.bytesToWrite(): socket.waitForBytesWritten(3000)
        socket.disconnectFromServer()
        return 0
    # This lock closes the simultaneous new-process startup race. The shared
    # server name also prevents a legacy Widgets process owning this session.
    args.session.parent.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(args.session.resolve())+".quick.lock")
    lock.setStaleLockTime(30000)
    if not lock.tryLock(0):
        print("Another DRR Analysis process is starting or owns this session.", file=sys.stderr)
        return 1
    server = QLocalServer()
    server.setSocketOptions(QLocalServer.UserAccessOption)
    if not server.listen(name):
        print("Cannot claim the DRR Analysis session: "+server.errorString(), file=sys.stderr)
        lock.unlock()
        return 1

    from ui_qt.quick.drr_analysis_bridge import DrrAnalysisBridge
    from ui_qt.quick.plot_surface import register_plot_surface
    register_plot_surface()
    settings = QSettings(str(args.settings_file.resolve()), QSettings.IniFormat) if args.settings_file else QSettings()
    bridge = DrrAnalysisBridge(settings=settings, session_path=args.session, start_folder=args.folder)
    engine = QQmlApplicationEngine()
    warnings = []
    engine.warnings.connect(lambda errors: warnings.extend(error.toString() for error in errors))
    engine.rootContext().setContextProperty("drrAnalysisBridge", bridge)
    engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parent/"ui_qt"/"quick"/"qml"/"DrrAnalysisApplication.qml")))
    if not engine.rootObjects():
        bridge.close(); server.close(); lock.unlock()
        return 2
    window = engine.rootObjects()[0]
    app.aboutToQuit.connect(bridge.close)
    queue, connections = [], []
    restoring = args.session.exists()

    def deliver(payload):
        if not isinstance(payload, dict): return
        if restoring:
            queue.append(payload); return
        folder = payload.get("folder")
        if folder and str(Path(bridge.start_folder).resolve()) != folder: bridge.setExperimentFolder(folder)
        if payload.get("snapshot"): bridge.receiveSnapshot(str(payload["snapshot"]))
        window.showNormal(); window.raise_(); window.requestActivate()

    def connected():
        while server.hasPendingConnections():
            client = server.nextPendingConnection()
            connections.append(client)
            buffer = bytearray()
            def receive(client=client, buffer=buffer):
                buffer.extend(bytes(client.readAll()))
                if len(buffer) > 65536: client.abort(); return
                if b"\n" in buffer:
                    try: deliver(json.loads(bytes(buffer).split(b"\n", 1)[0]))
                    except (ValueError, TypeError): pass
                    buffer.clear(); client.disconnectFromServer()
            client.readyRead.connect(receive)
            def disconnected(client=client):
                if client in connections: connections.remove(client)
                client.deleteLater()
            client.disconnected.connect(disconnected)
            receive()
    server.newConnection.connect(connected)

    def restore_finished(kind, elapsed):
        nonlocal restoring
        if kind != "restore" or not restoring: return
        restoring = False
        if bridge.snapshot["error"]:
            # Preserve unreadable sessions until the user chooses Save As.
            bridge._session_protected = True
            bridge._dirty = False
            bridge._autosave_timer.stop()
        for payload in list(queue): deliver(payload)
        queue.clear()
        bridge.refreshCatalog()
    bridge.operationCompleted.connect(restore_finished)
    if restoring: bridge.restoreWorkspace(str(args.session))
    else: bridge.refreshCatalog()
    if args.snapshot: QTimer.singleShot(0, lambda: deliver(message))
    check_timer = None
    if args.check:
        check_timer = QTimer()
        check_timer.setInterval(100)
        checks = [0]
        def check():
            checks[0] += 1
            if restoring or bridge.snapshot["busy"] or bridge._plot_timer.isActive() or bridge._autosave_timer.isActive():
                if checks[0] < 600: return
            if bridge._canvas and (bridge._canvas.backend._timer.isActive() or not bridge._canvas.draw_count): return
            report = dict(application=type(app).__name__, qmlRoots=len(engine.rootObjects()), warnings=warnings,
                widgetsImported="PySide6.QtWidgets" in sys.modules, datasets=len(bridge.datasets), error=bridge.snapshot["error"],
                renderedFrames=bridge._canvas.draw_count if bridge._canvas else 0, session=str(args.session.resolve()))
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
            check_timer.stop()
            window.setProperty("allowClose", True)
            window.close()
        check_timer.timeout.connect(check)
        check_timer.start()
    result = app.exec()
    if check_timer: check_timer.stop()
    server.close()
    # QML context properties do not own their Python objects. Destroy every
    # binding while the bridge and its models still have live Python owners.
    engine.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
    bridge.close()
    lock.unlock()
    if temporary:
        temporary.cleanup()
    return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
