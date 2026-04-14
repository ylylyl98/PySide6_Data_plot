from __future__ import annotations

import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from ui_qt.main_window import MainWindow


APP_QSS = """
QMainWindow {
    background: #f5f5f7;
}
QWidget {
    color: #1d1d1f;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #e5e5ea;
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
    font-size: 11px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #1d1d1f;
}
QPushButton {
    background: #f5f5f7;
    border: 1px solid #d2d2d7;
    border-radius: 8px;
    min-height: 30px;
    padding: 4px 12px;
    color: #1d1d1f;
    font-weight: 400;
}
QPushButton:hover {
    background: #e8e8ed;
    border-color: #b8b8bd;
}
QPushButton:pressed {
    background: #deded6;
}
QPushButton:disabled {
    background: #f5f5f7;
    border-color: #e5e5ea;
    color: rgba(0, 0, 0, 0.28);
}
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QListWidget, QTextEdit, QPlainTextEdit {
    background: #ffffff;
    border: 1px solid #d2d2d7;
    border-radius: 8px;
    min-height: 30px;
    padding: 3px 8px;
    color: #1d1d1f;
    selection-background-color: #0071e3;
    selection-color: #ffffff;
}
QLineEdit:read-only {
    background: #f5f5f7;
    color: rgba(0, 0, 0, 0.55);
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
    border-top: 5px solid #6e6e73;
    margin-right: 4px;
}
QListWidget {
    outline: none;
}
QListWidget::item {
    padding: 3px 4px;
}
QListWidget::item:selected {
    background: #0071e3;
    color: #ffffff;
}
QListWidget::item:hover:!selected {
    background: #f5f5f7;
}
QToolButton {
    background: #f5f5f7;
    border: 1px solid #d2d2d7;
    border-radius: 6px;
    min-height: 26px;
    padding: 2px 10px;
    color: #1d1d1f;
}
QToolButton:hover {
    background: #e8e8ed;
    border-color: #b8b8bd;
}
QToolButton:checked {
    background: #daeaf8;
    border-color: #0071e3;
    color: #0071e3;
}
QToolButton[autoRaise="true"] {
    border: 1px solid transparent;
    background: transparent;
}
QToolButton[autoRaise="true"]:hover {
    background: #e8e8ed;
    border-color: #d2d2d7;
}
QTabWidget::pane {
    border: 1px solid #e5e5ea;
    border-radius: 0 6px 6px 6px;
    background: #ffffff;
    top: -1px;
}
QTabBar::tab {
    background: #f5f5f7;
    border: 1px solid #e5e5ea;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    min-width: 52px;
    padding: 6px 14px;
    margin-right: 2px;
    color: rgba(0, 0, 0, 0.55);
    font-weight: 400;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #1d1d1f;
    border-color: #e5e5ea;
    font-weight: 600;
}
QTabBar::tab:hover:!selected {
    background: #e8e8ed;
    color: #1d1d1f;
}
QSplitter::handle:horizontal {
    width: 4px;
    background: #e5e5ea;
    margin: 2px 0;
}
QSplitter::handle:horizontal:hover {
    background: #0071e3;
}
QStatusBar {
    background: #f5f5f7;
    border-top: 1px solid #e5e5ea;
    color: #6e6e73;
    font-size: 11px;
}
QStatusBar QLabel {
    padding: 0 4px;
}
QProgressBar {
    border: 1px solid #d2d2d7;
    border-radius: 5px;
    text-align: center;
    min-height: 14px;
    max-height: 14px;
    background: #f5f5f7;
}
QProgressBar::chunk {
    border-radius: 4px;
    background: #0071e3;
}
QDockWidget {
    font-weight: 600;
    color: #1d1d1f;
}
QDockWidget::title {
    background: #f5f5f7;
    border-bottom: 1px solid #e5e5ea;
    padding: 4px 10px;
    font-size: 12px;
}
QToolBar {
    background: #f5f5f7;
    border-bottom: 1px solid #e5e5ea;
    spacing: 4px;
    padding: 3px 6px;
}
QToolBar::separator {
    width: 1px;
    background: #e5e5ea;
    margin: 4px 6px;
}
QCheckBox {
    spacing: 6px;
    color: #1d1d1f;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #b8b8bd;
    border-radius: 4px;
    background: #ffffff;
}
QCheckBox::indicator:checked {
    background: #0071e3;
    border-color: #0071e3;
}
QCheckBox::indicator:hover {
    border-color: #0071e3;
}
QScrollBar:vertical {
    background: #f5f5f7;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #c7c7cc;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #aeaeb2;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: #f5f5f7;
    height: 8px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #c7c7cc;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #aeaeb2;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
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
