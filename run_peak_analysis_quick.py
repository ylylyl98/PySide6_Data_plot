"""Standalone native Qt Quick Peak Analysis, with restartable local IPC."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def legacy_owner_running(session_path):
    from PySide6.QtNetwork import QLocalSocket
    from ui_qt.quick.peak_analysis_launcher import server_name
    socket = QLocalSocket()
    socket.connectToServer(server_name(session_path, legacy=True))
    connected = socket.waitForConnected(150)
    socket.abort()
    return connected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--reply-server", default="")
    parser.add_argument("--settings", "--settings-file", type=Path)
    parser.add_argument("--check", action="store_true", help="Present an isolated native window, then close")
    parser.add_argument("--capture", type=Path)
    args = parser.parse_args(argv)
    from PySide6.QtCore import QSettings, QTimer, QUrl, QProcess
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from ui_qt.peak_analysis_ipc import MessageServer, send_message
    from ui_qt.quick.peak_analysis_launcher import launch_arguments, server_name
    from ui_qt.quick.peak_analysis_bridge import PeakAnalysisBridge
    from ui_qt.quick.plot_surface import register_plot_surface
    from ui_qt.quick.settings_storage import analysis_session_path
    from ui_qt.typography import install_application_font
    import tempfile
    temporary = tempfile.TemporaryDirectory(prefix="dptk-quick-peak-") if args.check else None
    if args.session is None:
        if temporary:
            args.session = Path(temporary.name) / "workspace.npz"
        else:
            injected_settings=QSettings(str(args.settings.resolve()),QSettings.Format.IniFormat) if args.settings else None
            args.session = analysis_session_path(injected_settings,"peak")
    args.session = args.session.resolve()
    settings_path = (args.settings or args.session.with_name("quick.ini")).resolve()
    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([sys.argv[0]])
    install_application_font(app)
    app.setApplicationName("DPTK Peak Analysis")
    app.setOrganizationName("DPTK")
    message = dict(snapshot=str(args.snapshot.resolve()) if args.snapshot else None, reply_server=args.reply_server)
    if send_message(server_name(args.session), message):
        if temporary:
            temporary.cleanup()
        return 0
    if legacy_owner_running(args.session):
        print("The legacy Peak Analysis window owns this workspace. Close it before opening the same workspace in Quick.", file=sys.stderr)
        if temporary:
            temporary.cleanup()
        return 3
    server = MessageServer(server_name(args.session))
    if not server.listening:
        return 1
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
    bridge = PeakAnalysisBridge(settings=settings, session_path=args.session, reply_server=args.reply_server)
    engine = QQmlApplicationEngine()
    register_plot_surface()
    engine.setInitialProperties({"peakBridge": bridge})
    engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parent / "ui_qt/quick/qml/PeakAnalysis.qml")))
    if not engine.rootObjects():
        bridge.close()
        server.server.close()
        return 2
    window = engine.rootObjects()[0]
    def deliver(payload):
        if payload.get("snapshot"):
            bridge.loadSnapshot(payload["snapshot"], True, payload.get("reply_server", ""))
        elif payload.get("reply_server"):
            bridge.reply_server = str(payload["reply_server"])
        window.showNormal()
        window.raise_()
        window.requestActivate()
    server.received.connect(deliver)
    bridge.closeReady.connect(window.close)
    initialized = False
    def start():
        nonlocal initialized
        if initialized:
            return
        initialized = True
        bridge.restore()
        if args.snapshot:
            deliver(message)
        if args.check:
            QTimer.singleShot(500, check_ready)
    def check_ready():
        if bridge.snapshot["busy"] or bridge._plot_timer.isActive():
            QTimer.singleShot(100, check_ready)
            return
        if args.capture:
            args.capture.parent.mkdir(parents=True, exist_ok=True)
            window.grabWindow().save(str(args.capture))
        bridge.requestClose()
    window.frameSwapped.connect(start)
    QTimer.singleShot(30000, lambda: app.exit(4)) if args.check else None
    result = app.exec()
    restart = bridge._restart_requested
    bridge.close()
    server.server.close()
    if restart:
        executable, arguments, cwd = launch_arguments(session_path=args.session, reply_server=bridge.reply_server, settings_path=settings_path)
        launched, _ = QProcess.startDetached(executable, arguments, cwd)
        if not launched:
            result = 1
    if temporary:
        temporary.cleanup()
    return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
