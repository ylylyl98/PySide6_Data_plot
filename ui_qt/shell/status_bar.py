"""Status-bar shell widget for DPTK Desktop.

``StatusBarView`` owns Qt widget construction and layout for the main window's
status region (global message area plus the right/contextual progress and
update-availability widgets). The owning window decides WHAT status to show and
wires the exposed widgets to its own state and handlers; this view deliberately
contains no application or workflow logic.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QProgressBar, QPushButton, QStatusBar

from ui_qt.fluent_ui.style import apply_accessible_identity, set_fluent_property


class StatusBarView(QStatusBar):
    """Composes the main window's global/contextual status widgets."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._cursor_readback = QLabel(self)
        self._cursor_readback.setObjectName("cursorReadback")
        self._cursor_readback.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._cursor_readback.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        set_fluent_property(self._cursor_readback, "appRole", "statusItemPassive")
        apply_accessible_identity(
            self._cursor_readback,
            name="Cursor readback",
            description="Passive readback of the current plot cursor position",
            identifier="status.cursorReadback",
        )
        self.addPermanentWidget(self._cursor_readback)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 0)  # indeterminate spinner
        self._progress.setMaximumWidth(120)
        self._progress.setVisible(False)
        self._progress.setToolTip("Loading data in background…")
        self.addPermanentWidget(self._progress)

        self.showMessage("Ready")

        self._update_button = QPushButton(self)
        self._update_button.setFlat(True)
        self._update_button.setCursor(Qt.PointingHandCursor)
        set_fluent_property(self._update_button, "appRole", "linkButton")
        apply_accessible_identity(
            self._update_button,
            name="Update status",
            description="Open the update dialog",
        )
        self._update_button.setVisible(False)
        self.addPermanentWidget(self._update_button)

    @property
    def progress(self) -> QProgressBar:
        """Indeterminate background-task progress indicator."""
        return self._progress

    @property
    def cursor_readback(self) -> QLabel:
        """Passive plot cursor readback kept outside the persistent message lane."""
        return self._cursor_readback

    def set_cursor_readback(self, text: str) -> None:
        """Update only the passive plot cursor readback label."""
        self._cursor_readback.setText(text)

    @property
    def update_button(self) -> QPushButton:
        """Actionable update-availability button."""
        return self._update_button
