from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app_version import __version__

APP_USER_MODEL_ID = "com.ylylyl98.dptk_desktop.data_plot"
APP_ICON_PATH = Path("assets") / "icons" / "app_icon.ico"


def resource_path(relative_path: str | Path) -> Path:
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path / relative_path


def set_windows_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass


def load_app_icon() -> QIcon:
    icon_path = resource_path(APP_ICON_PATH)
    if icon_path.exists():
        return QIcon(str(icon_path))
    return QIcon()


def main() -> int:
    if "--mcd-organizer" in sys.argv:
        index = sys.argv.index("--mcd-organizer")
        experiment = sys.argv[index + 1] if index + 1 < len(sys.argv) else str(Path.cwd())
        from run_mcd_organizer import main as organizer_main

        return organizer_main([experiment])

    if "--check-powerpoint-integration" in sys.argv:
        from core.presentation import powerpoint_integration_available

        available, _message = powerpoint_integration_available()
        return 0 if available else 2

    set_windows_app_user_model_id()

    app = QApplication(sys.argv)
    app.setApplicationName("DPTK Desktop")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("ylylyl98")
    app.setStyle("Fusion")
    from ui_qt.theme import install_theme

    install_theme(app)
    app_icon = load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    from ui_qt.main_window import MainWindow

    window = MainWindow()
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
