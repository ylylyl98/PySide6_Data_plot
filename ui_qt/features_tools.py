"""Tools feature page.

The page is implemented as a mixin so the main window remains the shared
application context while this feature UI is independently located.
"""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ui_qt.fluent_ui.style import set_fluent_property


class ToolsPageMixin:
    def _build_tools_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        log_box = QGroupBox("Log Panel")
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(8, 8, 8, 8)
        log_layout.setSpacing(6)
        hint = QLabel("The log panel records all load, plot, and export events.")
        hint.setWordWrap(True)
        set_fluent_property(hint, "appRole", "hintText")
        log_layout.addWidget(hint)
        log_btn_row = QHBoxLayout()
        log_btn_row.setSpacing(8)
        self.show_log_btn = QPushButton("Show / Hide Log Panel")
        self.show_log_btn.setToolTip("Toggle the bottom log dock panel")
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.setToolTip("Clear all messages from the log")
        log_btn_row.addWidget(self.show_log_btn)
        log_btn_row.addWidget(self.clear_log_btn)
        log_layout.addLayout(log_btn_row)
        layout.addWidget(log_box)

        file_box = QGroupBox("File Management")
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(8, 8, 8, 8)
        file_layout.setSpacing(6)
        file_hint = QLabel(
            "After export, source files remain in place. Use 'Move Exported Sources' "
            "only when you intentionally want to archive them."
        )
        file_hint.setWordWrap(True)
        set_fluent_property(file_hint, "appRole", "hintText")
        file_layout.addWidget(file_hint)
        layout.addWidget(file_box)

        data_box = QGroupBox("Data Organization")
        data_layout = QVBoxLayout(data_box)
        data_layout.setContentsMargins(8, 8, 8, 8)
        data_layout.setSpacing(6)
        data_hint = QLabel(
            "Organize previously processed MCD measurements into matched E-field, "
            "temperature, doping, or gate-voltage comparison series."
        )
        data_hint.setWordWrap(True)
        set_fluent_property(data_hint, "appRole", "hintText")
        data_layout.addWidget(data_hint)
        self.mcd_extract_btn = QPushButton("Open standalone MCD Organizer…")
        self.mcd_extract_btn.setToolTip(
            "Launch the processed-MCD comparison and export tool in a separate window."
        )
        data_layout.addWidget(self.mcd_extract_btn)
        layout.addWidget(data_box)

        layout.addStretch(1)
        return tab
