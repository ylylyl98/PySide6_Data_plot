"""Lightweight standalone launcher for the processed MCD Organizer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app_version import __version__
from ui_qt.mcd_organizer_window import McdOrganizerWindow


def _resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Organize and export processed MCD results.")
    parser.add_argument("experiment", nargs="?", default=str(Path.cwd()))
    args = parser.parse_args(argv)

    app = QApplication([sys.argv[0], *(argv or [])])
    app.setApplicationName("DPTK MCD Organizer")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("ylylyl98")
    app.setStyle("Fusion")
    from ui_qt.theme import install_theme

    install_theme(app)
    icon_path = _resource_path("assets/icons/app_icon.ico")
    icon = QIcon(str(icon_path)) if icon_path.is_file() else QIcon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    window = McdOrganizerWindow(args.experiment)
    if not icon.isNull():
        window.setWindowIcon(icon)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
