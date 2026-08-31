"""Lightweight standalone launcher for the processed MCD Organizer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from app_version import __version__
from ui_qt.mcd_organizer_window import McdOrganizerWindow


ORGANIZER_QSS = """
QMainWindow { background: #f5f5f7; }
QWidget { color: #1d1d1f; }
QPushButton {
    background: #f5f5f7; border: 1px solid #d2d2d7; border-radius: 7px;
    min-height: 30px; padding: 3px 11px;
}
QPushButton:hover { background: #e8e8ed; border-color: #aeb0b5; }
QPushButton:disabled { color: #999; border-color: #e5e5ea; }
QLineEdit, QComboBox, QListWidget, QTableWidget {
    background: white; border: 1px solid #d2d2d7; border-radius: 7px;
    min-height: 29px; padding: 3px 7px;
}
QListWidget::item { padding: 7px 5px; border-bottom: 1px solid #eeeeef; }
QListWidget::item:selected { background: #dceeff; color: #123f69; }
QCheckBox { spacing: 6px; }
QSplitter::handle { background: #e5e5ea; width: 4px; }
"""


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
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(ORGANIZER_QSS)
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
