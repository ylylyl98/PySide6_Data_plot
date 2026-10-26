"""Status-bar shell widget for DPTK Desktop.

``StatusBarView`` owns Qt widget construction and layout for the main window's
status region (global message area plus the right/contextual progress and
update-availability widgets). The owning window decides WHAT status to show and
wires the exposed widgets to its own state and handlers; this view deliberately
contains no application or workflow logic.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QLabel, QProgressBar, QPushButton, QSizePolicy, QStatusBar

from ui_qt.fluent_ui.style import apply_accessible_identity, set_fluent_property


class StatusBarView(QStatusBar):
    """Composes the main window's global/contextual status widgets."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("workspaceStatusBar")
        self.setMinimumHeight(max(28, self.fontMetrics().height() + 10))

        self._context = QLabel(self)
        self._context.setObjectName("workflowStatusContext")
        self._context.setProperty("appRole", "statusItemPassive")
        self._context.setFocusPolicy(Qt.NoFocus)
        apply_accessible_identity(self._context, name="Current workspace", identifier="status.workspace")
        self.addPermanentWidget(self._context)

        self._cursor_readback = QLabel(self)
        self._cursor_readback.setObjectName("cursorReadback")
        self._cursor_readback.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._cursor_readback.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._cursor_readback.setMinimumWidth(0)
        self._cursor_readback.setMaximumWidth(480)
        self._cursor_readback.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        set_fluent_property(self._cursor_readback, "appRole", "statusItemPassive")
        apply_accessible_identity(
            self._cursor_readback,
            name="Cursor readback",
            description="Passive readback of the current plot cursor position",
            identifier="status.cursorReadback",
        )
        self.addPermanentWidget(self._cursor_readback)

        self._busy_label = QLabel("Working…", self)
        self._busy_label.setObjectName("backgroundTaskStatus")
        self._busy_label.setProperty("appRole", "busyStatus")
        self._busy_label.setFocusPolicy(Qt.NoFocus)
        self._busy_label.hide()
        apply_accessible_identity(
            self._busy_label, name="Background task status", identifier="status.backgroundTask"
        )
        self.addPermanentWidget(self._busy_label)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 0)  # indeterminate spinner
        self._progress.setMaximumWidth(120)
        self._progress.setMaximumHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        self._progress.setToolTip("Loading data in background…")
        self.addPermanentWidget(self._progress)
        self._progress.installEventFilter(self)

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
        self._cursor_readback.setToolTip(text)

    def set_context(self, text: str) -> None:
        """Set the compact current-workflow context on the status bar's right."""
        self._context.setText(text)
        self._context.setVisible(bool(text))

    def set_busy(self, busy: bool, text: str = "Working…") -> None:
        """Show an explicit task label alongside the existing progress widget.

        The same progress instance remains available to controllers that set a
        determinate range or value. Finishing a task hides both feedback parts.
        """
        self._busy_label.setText(text)
        self._busy_label.setToolTip(text)
        self._progress.setVisible(bool(busy))
        self._sync_busy_feedback()

    def _sync_busy_feedback(self) -> None:
        busy = not self._progress.isHidden()
        self._busy_label.setVisible(busy)
        set_fluent_property(self, "fluentBusy", busy)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if watched is self._progress and event.type() in (
            QEvent.Show, QEvent.Hide, QEvent.ShowToParent, QEvent.HideToParent,
        ):
            self._sync_busy_feedback()
        return super().eventFilter(watched, event)

    @property
    def update_button(self) -> QPushButton:
        """Actionable update-availability button."""
        return self._update_button
