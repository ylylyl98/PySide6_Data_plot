"""Compact three-level source identity used by both MCD pages."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ui_qt.status_badge import StatusBadge


class McdSourceSummary(QWidget):
    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("mcdSourceSummary")
        self.status_label = StatusBadge("", self, app_role="sourceBadge")
        self.filename_label = QLabel(self)
        self.filename_label.setMinimumWidth(0)
        self.filename_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.time_label = QLabel(self)
        self.time_label.setMinimumWidth(0)
        self.time_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        font = self.time_label.font(); font.setPointSize(max(1, font.pointSize() - 1)); self.time_label.setFont(font)
        self.time_label.setProperty("secondary", True)
        self.time_label.setStyleSheet("color: palette(mid);")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
        layout.addWidget(self.status_label); layout.addWidget(self.filename_label); layout.addWidget(self.time_label)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        if text:
            self.setText(text)

    def set_source(self, *, status: str, filename: str, saved_at: str = "", tooltip: str = "", badge_state: str | None = None) -> None:
        self.setToolTip(str(tooltip))
        self.status_label.set_status(str(status), tooltip=tooltip, app_role="sourceBadge", badge_state=badge_state)
        self.filename_label.setText(str(filename))
        self.filename_label.setToolTip(str(tooltip or filename))
        self.time_label.setText(str(saved_at))
        self.time_label.setVisible(bool(saved_at))
        self._elide_filename()

    def _elide_filename(self) -> None:
        text = self.filename_label.text()
        width = max(0, self.filename_label.width())
        if width:
            self.filename_label.setText(self.filename_label.fontMetrics().elidedText(text, Qt.TextElideMode.ElideMiddle, width))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event); self._elide_filename()

    def set_status(self, text: str, *, tooltip: str = "", app_role=None, badge_state=None, **kwargs) -> None:
        self.set_source(status=str(text), filename="", saved_at="", tooltip=tooltip, badge_state=badge_state)

    def setText(self, text: str) -> None:  # noqa: N802
        self.set_status(str(text))

    def text(self) -> str:
        fields = [self.status_label.text(), self.filename_label.text(), self.time_label.text()]
        return "\n".join(value for value in fields if value)
