from __future__ import annotations

import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from ui_qt.main_window import MainWindow


APP_QSS = """
QMainWindow {
    background: #f0f3f7;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #d4dbe5;
    border-radius: 8px;
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
    font-weight: 500;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #3a4a5c;
}
QPushButton {
    background: #eaf1fa;
    border: 1px solid #c2cfdf;
    border-radius: 7px;
    min-height: 28px;
    padding: 3px 10px;
    color: #1f3050;
}
QPushButton:hover {
    background: #d6e8f7;
    border-color: #a8c0d8;
}
QPushButton:pressed {
    background: #c2d9ef;
}
QPushButton:disabled {
    background: #f0f3f7;
    border-color: #d8dde5;
    color: #9aa8b8;
}
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QListWidget, QTextEdit, QPlainTextEdit {
    background: #ffffff;
    border: 1px solid #c8d4e0;
    border-radius: 7px;
    min-height: 28px;
    padding: 3px 6px;
    selection-background-color: #5b9bd5;
    selection-color: #ffffff;
}
QLineEdit:read-only {
    background: #f5f7fa;
    color: #4a5a6a;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #5a7a9a;
    margin-right: 4px;
}
QListWidget {
    outline: none;
}
QListWidget::item:selected {
    background: #5b9bd5;
    color: #ffffff;
}
QListWidget::item:hover:!selected {
    background: #e8f2fb;
}
QToolButton {
    background: transparent;
    border: 1px solid #c8d4e0;
    border-radius: 6px;
    min-height: 24px;
    padding: 1px 8px;
    color: #2a3a50;
}
QToolButton:hover {
    background: #e4eef8;
    border-color: #a8bfd4;
}
QToolButton:checked {
    background: #d0e4f5;
    border-color: #7aacd4;
}
QTabWidget::pane {
    border: 1px solid #c8d4e0;
    border-radius: 6px;
    background: #ffffff;
    top: -1px;
}
QTabBar::tab {
    background: #e4ecf5;
    border: 1px solid #c4d0de;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    min-width: 52px;
    padding: 5px 12px;
    margin-right: 2px;
    color: #4a5a72;
    font-weight: 500;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #1a3060;
    border-color: #c8d4e0;
    font-weight: 600;
}
QTabBar::tab:hover:!selected {
    background: #d8e8f5;
}
QSplitter::handle:horizontal {
    width: 4px;
    background: #dce4ee;
    margin: 2px 0;
}
QSplitter::handle:horizontal:hover {
    background: #a8c0d8;
}
QStatusBar {
    background: #f8fafc;
    border-top: 1px solid #d0d8e2;
    color: #3a4a5c;
    font-size: 11px;
}
QStatusBar QLabel {
    padding: 0 4px;
}
QProgressBar {
    border: 1px solid #b8c8d8;
    border-radius: 5px;
    text-align: center;
    min-height: 14px;
    max-height: 14px;
    background: #e8eef6;
}
QProgressBar::chunk {
    border-radius: 4px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #5b9bd5, stop:1 #4a8ac0);
}
QDockWidget {
    titlebar-close-icon: none;
    font-weight: 600;
    color: #2a3a50;
}
QDockWidget::title {
    background: #e8eef6;
    border-bottom: 1px solid #c8d4e0;
    padding: 4px 8px;
}
QToolBar {
    background: #f0f4f8;
    border-bottom: 1px solid #d0d8e4;
    spacing: 4px;
    padding: 2px 4px;
}
QToolBar::separator {
    width: 1px;
    background: #c4d0de;
    margin: 4px 6px;
}
QCheckBox {
    spacing: 5px;
    color: #2a3a50;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #b0c0d0;
    border-radius: 3px;
    background: #ffffff;
}
QCheckBox::indicator:checked {
    background: #5b9bd5;
    border-color: #4a88c0;
}
QCheckBox::indicator:hover {
    border-color: #7aacd4;
}
"""


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(APP_QSS)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
