"""Run the native Qt Quick frontend with isolated check/demo profiles.

The production launcher supplies persistent application settings; this direct
development entry point stays isolated unless an INI profile is specified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter


def main(argv=None, *, production=False):
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--check", action="store_true", help="Render and close with isolated settings")
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--settings-file", type=Path)
    parser.add_argument("--open", type=Path, dest="sources", action="append", help="Measurement path; repeat for multiple channels")
    parser.add_argument("--workflow", choices=("pl", "drr", "cmp", "power", "mcd", "mcd_peak_shift", "shg", "slides", "tools"), default="pl")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--theme", choices=("light", "dark", "system", "high-contrast"))
    parser.add_argument("--width", type=int, default=1320)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args(argv)
    from PySide6.QtCore import QCoreApplication, QEvent, QObject, QSettings, QTimer, QUrl, qInstallMessageHandler
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from ui_qt.quick.application_bridge import QuickApplicationBridge
    from ui_qt.quick.plot_surface import register_plot_surface
    from ui_qt.typography import install_application_font

    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([sys.argv[0]])
    install_application_font(app)
    app.setApplicationName("DPTK Desktop" if production else "DPTK Quick Preview")
    app.setOrganizationName("DPTK")
    from app_version import __version__
    app.setApplicationVersion(__version__)
    if production:
        from run_qt import load_app_icon, set_windows_app_user_model_id
        set_windows_app_user_model_id()
        icon = load_app_icon()
        if not icon.isNull():
            app.setWindowIcon(icon)
    temporary = tempfile.TemporaryDirectory(prefix="dptk-native-quick-")
    persistent = bool(production and not args.check and not args.demo)
    if args.settings_file is not None:
        settings = QSettings(str(args.settings_file), QSettings.IniFormat)
    elif persistent:
        settings = QSettings("DPTK", "PySide6_Data_Plot")
    else:
        settings = QSettings(str(Path(temporary.name) / "preview.ini"), QSettings.IniFormat)
    if args.theme is not None:
        settings.setValue("quick/appearance", args.theme)
    old_cache = os.environ.get("MPLCONFIGDIR")
    isolated_cache = args.check or args.demo or not production
    if isolated_cache:
        os.environ["MPLCONFIGDIR"] = str(Path(temporary.name) / "matplotlib")
    errors = []
    def qt_message(kind, context, message):
        text = str(message)
        if any(marker in text for marker in ("ReferenceError", "TypeError", "Binding loop", "Cannot assign", "Unable to assign", "Error:")):
            errors.append(text)
        if sys.stderr is not None:
            print(text, file=sys.stderr)
    old_handler = qInstallMessageHandler(qt_message)
    bridge = QuickApplicationBridge(settings=settings)
    engine = QQmlApplicationEngine()
    register_plot_surface()
    engine.setInitialProperties({"appBridge": bridge})
    engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parent / "ui_qt/quick/qml/Application.qml")))
    if not engine.rootObjects():
        bridge.close()
        qInstallMessageHandler(old_handler)
        temporary.cleanup()
        return 2
    window = engine.rootObjects()[0]
    screen = app.primaryScreen()
    available = screen.availableGeometry() if persistent and screen is not None else None
    window.setWidth(min(args.width, available.width()) if available is not None else args.width)
    window.setHeight(min(args.height, available.height()) if available is not None else args.height)
    timings = {"engine_loaded_ms": (perf_counter() - started) * 1000}
    shell_imports = []
    surface = window.findChild(QObject, "app_plot_surface")
    frames = []
    outcome = {"exit": 0}
    def swapped():
        if "first_shell_frame_ms" not in timings:
            timings["first_shell_frame_ms"] = (perf_counter() - started) * 1000
            shell_imports.extend(name for name in ("matplotlib", "numpy", "scipy", "pandas") if name in sys.modules)
            QTimer.singleShot(0, start_content)
        frames.append(perf_counter())
    window.frameSwapped.connect(swapped)
    def start_content():
        if persistent and not args.sources:
            bridge.restoreSourceFolder()
        if not args.check and not args.demo:
            bridge.startServices()
        bridge.selectWorkflow(args.workflow)
        if args.sources:
            bridge.selectFiles([QUrl.fromLocalFile(str(path.resolve())) for path in args.sources])
        elif args.demo:
            bridge.loadDemo()
    def plot_presented(generation):
        if generation > 0 and "first_plot_frame_ms" not in timings:
            timings["first_plot_frame_ms"] = (perf_counter() - started) * 1000
    surface.framePresented.connect(plot_presented)
    background_window = window.findChild(QObject, "drr_background_window")
    background_frames = []
    def background_presented():
        if bridge._background_open and background_window.isVisible():
            background_frames.append(perf_counter())
            timings.setdefault("background_settings_frame_ms", (perf_counter() - started) * 1000)
    background_window.frameSwapped.connect(background_presented)
    timer = QTimer()
    deadline = perf_counter() + 30
    stable_frame = {"state": None, "since": 0.0}
    def active_surface():
        backend = bridge.plotBackend
        def find(item):
            if item.property("backend") is backend and backend is not None and item.property("visible"):
                return item
            for child in item.childItems():
                match = find(child)
                if match is not None:
                    return match
            return None
        return find(window.contentItem())

    def check_finished():
        current_surface, backend = active_surface(), bridge.plotBackend
        generation = current_surface.property("presentedGeneration") if current_surface is not None else 0
        owner = bridge.workflowBridge
        plot_timer = getattr(owner, "_plot_timer", None)
        plot_ready = (backend is not None and generation > 0 and generation == backend.frameGeneration
                      and (plot_timer is None or not plot_timer.isActive()) and not backend._timer.isActive())
        if plot_ready:
            plot_presented(generation)
        # Power and unresolved DRR CSV selections intentionally wait for an
        # explicit user action before producing scientific plots.
        awaiting_background = (args.workflow == "drr" and bridge.snapshot.get("backgroundRequired", False)
            and bridge._background_open and not bridge.snapshot["busy"])
        expected_plot = args.demo or bool(args.sources) and args.workflow != "power" and not awaiting_background
        ready_state = (generation, awaiting_background)
        if ready_state != stable_frame["state"]:
            stable_frame.update(state=ready_state, since=perf_counter())
        if perf_counter() - stable_frame["since"] < .15 and perf_counter() < deadline:
            return
        child_state = owner.snapshot
        busy = bridge.snapshot["busy"] or child_state.get("catalogBusy", False) or child_state.get("analysisBusy", False)
        missing_frame = (not frames or busy or (expected_plot and not plot_ready)
            or awaiting_background and not background_frames)
        if missing_frame and perf_counter() < deadline:
            return
        if missing_frame:
            errors.append("Native Quick smoke did not reach a rendered frame before its deadline")
        if args.sources and child_state.get("error"):
            errors.append(child_state["error"])
        if args.capture:
            args.capture.parent.mkdir(parents=True, exist_ok=True)
            captured = (background_window if awaiting_background else window).grabWindow()
            if captured.isNull() or not captured.save(str(args.capture), "PNG"):
                errors.append("Native Quick window capture failed")
        report = dict(status="ok" if not errors else "error", errors=errors, timing_ms=timings,
                      widgets_imported="PySide6.QtWidgets" in sys.modules,
                      settings_isolated=not persistent and args.settings_file is None,
                      scientific_imports_at_shell_frame=shell_imports,
                      background_required=bridge.snapshot.get("backgroundRequired", False),
                      background_dialog_open=bridge._background_open, can_export=bridge.snapshot.get("canExport", False),
                      background_visible_frames=len(background_frames),
                      workflow=bridge.snapshot["workflow"], visible_frames=len(frames),
                      supported_scope="native shell, PL, DRR, Compare, Power, MCD, MCD Peak Shift, SHG, Slides and separate analysis windows")
        if report["widgets_imported"]:
            report["status"] = "error"
            errors.append("The native Quick path imported QtWidgets")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        outcome["exit"] = 0 if report["status"] == "ok" else 2
        timer.stop()
        app.quit()
    if args.check:
        timer.setInterval(100)
        timer.timeout.connect(check_finished)
        timer.start()
    try:
        code = app.exec()
    finally:
        timer.stop()
        bridge.close()
        window.close()
        engine.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
        qInstallMessageHandler(old_handler)
        if old_cache is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = old_cache
        temporary.cleanup()
    return outcome["exit"] or code


if __name__ == "__main__":
    raise SystemExit(main())
