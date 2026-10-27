"""Isolated source/frozen smoke check for the production DRR Qt Quick page.

``run_qt.py --check-quick-display`` dispatches here before normal startup. Run
this check in its own process: it creates the real shell, bridge and QQuickWidget
while keeping settings, the data folder and Matplotlib's cache in a temporary
directory. Nothing is loaded from the user's last experiment.
"""

from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback


@contextmanager
def _temporary_attributes(changes):
    originals = [(owner, name, getattr(owner, name)) for owner, name, _value in changes]
    try:
        for owner, name, value in changes:
            setattr(owner, name, value)
        yield
    finally:
        for owner, name, value in reversed(originals):
            setattr(owner, name, value)


def _frame_summary(image) -> dict[str, object]:
    """Reject a missing/transparent/uniform framebuffer without saving files."""
    if image.isNull() or image.width() < 32 or image.height() < 32:
        raise RuntimeError("DRR QQuickWidget did not produce a usable framebuffer")
    colors = {
        image.pixelColor(x, y).rgba()
        for x in range(0, image.width(), max(1, image.width() // 80))
        for y in range(0, image.height(), max(1, image.height() // 80))
        if image.pixelColor(x, y).alpha() > 0
    }
    if len(colors) < 2:
        raise RuntimeError("DRR Qt Quick framebuffer is empty or uniform")
    return {"width": image.width(), "height": image.height(), "sampled_colors": len(colors)}


def _run_isolated(root: Path, capture_path: Path | None = None) -> dict[str, object]:
    # Import QtGui before NumPy/SciPy on Windows so their DLL directories cannot
    # cause an incompatible Qt dependency to be selected.
    from PySide6 import QtGui  # noqa: F401
    from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QThread, qInstallMessageHandler
    from PySide6.QtQuickWidgets import QQuickWidget
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is not None:
        raise RuntimeError("The Qt Quick smoke check requires a fresh process")

    import ui_qt.main_window as main_window_module
    import ui_qt.presentation_widget as presentation_module
    from ui_qt.theme import install_theme

    settings_file = root / "settings.ini"
    data_folder = root / "empty-data"
    data_folder.mkdir()
    settings = QSettings(str(settings_file), QSettings.IniFormat)
    settings.setValue("updates/check_automatically", False)
    settings.setValue("data/last_folder", str(data_folder))
    settings.setValue("data/recent_folders", [])
    settings.sync()

    class IsolatedSettings(QSettings):
        def __init__(self, *_args, **_kwargs):
            super().__init__(str(settings_file), QSettings.IniFormat)

    def restore_empty_folder(window):
        # Construction may create catalogs lazily; do not start a scan or read
        # the persisted source cache just to check the renderer.
        window.current_folder = str(data_folder)

    app = QApplication(["dptk-quick-display-smoke"])
    app.setApplicationName("DPTK Qt Quick Smoke Check")
    app.setOrganizationName("DPTK Isolated Checks")
    app.setStyle("Fusion")
    install_theme(app, mode="light")
    window = None
    qml_messages: list[str] = []
    python_errors: list[str] = []

    def qt_message(kind, context, message):
        if ".qml:" in message or message.startswith(("QQml", "QQuick")):
            qml_messages.append(message)
        if old_qt_handler is not None:
            old_qt_handler(kind, context, message)
        elif sys.stderr is not None:
            print(message, file=sys.stderr)

    def python_error(kind, value, tb):
        python_errors.append("".join(traceback.format_exception(kind, value, tb)))

    old_qt_handler = qInstallMessageHandler(qt_message)
    old_excepthook = sys.excepthook
    sys.excepthook = python_error
    try:
        with _temporary_attributes([
            (main_window_module, "QSettings", IsolatedSettings),
            (presentation_module, "QSettings", IsolatedSettings),
            (main_window_module.MainWindow, "_restore_last_folder", restore_empty_folder),
            (main_window_module.MainWindow, "_schedule_automatic_update_check", lambda _self: None),
        ]):
            window = main_window_module.MainWindow(drr_display_backend="quick")
            window.workflow_tabs.setCurrentIndex(1)
            window.tabs.setCurrentIndex(1)
            window.resize(1320, 820)
            window.show()
            host = window.drr_display_host
            if not host.ensure_loaded():
                raise RuntimeError(f"DRR QML load failed: {host.load_error}")
            quick_widget = host.quick_widget
            if not isinstance(quick_widget, QQuickWidget):
                raise RuntimeError("DRR display did not create the production QQuickWidget")
            deadline = time.monotonic() + 10.0
            while quick_widget.status() == QQuickWidget.Loading and time.monotonic() < deadline:
                app.processEvents()
                QThread.msleep(10)
            if not host.ready or quick_widget.status() != QQuickWidget.Ready:
                errors = "\n".join(error.toString() for error in quick_widget.errors())
                raise RuntimeError(f"DRR Qt Quick page is not ready: {host.load_error}\n{errors}")
            if host.bridge is None or quick_widget.rootObject() is None:
                raise RuntimeError("The production DRR bridge or QML root is missing")
            # Let layout, binding evaluation and scene graph updates settle.
            for _ in range(10):
                app.processEvents()
                QThread.msleep(10)
            image = quick_widget.grabFramebuffer()
            frame = _frame_summary(image)
            if capture_path is not None:
                capture_path.parent.mkdir(parents=True, exist_ok=True)
                if not image.save(str(capture_path), "PNG"):
                    raise RuntimeError(f"Could not save Qt Quick smoke capture: {capture_path}")
            if quick_widget.errors():
                raise RuntimeError("\n".join(error.toString() for error in quick_widget.errors()))
            result = {
                "status": "ok",
                "backend": "quick",
                "qml": quick_widget.source().toLocalFile(),
                "settings_isolated": Path(window.settings.fileName()).resolve() == settings_file.resolve(),
                "data_isolated": Path(window.current_folder).resolve() == data_folder.resolve(),
                "frame": frame,
            }
            if not result["settings_isolated"] or not result["data_isolated"]:
                raise RuntimeError("The Qt Quick check escaped its isolated settings/data folder")
    finally:
        if window is not None:
            window.close()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
            window = None
            gc.collect()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
        qInstallMessageHandler(old_qt_handler)
        sys.excepthook = old_excepthook
        app.quit()
    if qml_messages or python_errors:
        raise RuntimeError("Qt Quick smoke emitted errors:\n" + "\n".join(qml_messages + python_errors))
    return result


def check_quick_display(capture_path: str | Path | None = None) -> int:
    """Build/render/destroy the actual DRR page; return a process exit code."""
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        os.environ.setdefault("QT_QUICK_BACKEND", "software")
    old_mpl_cache = os.environ.get("MPLCONFIGDIR")
    try:
        with tempfile.TemporaryDirectory(prefix="dptk-quick-display-smoke-") as directory:
            root = Path(directory)
            os.environ["MPLCONFIGDIR"] = str(root / "mpl-cache")
            result = _run_isolated(root, Path(capture_path).resolve() if capture_path is not None else None)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception:
        traceback.print_exc()
        return 2
    finally:
        if old_mpl_cache is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = old_mpl_cache


def main() -> int:
    return check_quick_display()


if __name__ == "__main__":
    if not getattr(sys, "frozen", False):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
