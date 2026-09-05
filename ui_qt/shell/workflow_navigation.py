"""Workflow navigation presentation for the main window shell."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QStyle, QTabBar, QTabWidget, QToolButton

from ui_qt.fluent_ui.style import apply_accessible_identity


class _WorkflowTabBar(QTabBar):
    """Indexed workflow tabs with a purely visual utility-group separator."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._separator_index = -1
        self.utility_separator = QFrame(self)
        self.utility_separator.setObjectName("workflowUtilitySeparator")
        self.utility_separator.setFrameShape(QFrame.VLine)
        self.utility_separator.setFrameShadow(QFrame.Plain)
        self.utility_separator.setProperty("fluentRole", "divider")
        self.utility_separator.setFocusPolicy(Qt.NoFocus)
        self.utility_separator.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.utility_separator.hide()

    def set_separator_index(self, index: int) -> None:
        self._separator_index = int(index)
        self._position_separator()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._position_separator()

    def _position_separator(self) -> None:
        index = self._separator_index
        if index < 0 or index >= self.count() or self.height() <= 0:
            self.utility_separator.hide()
            return
        rect = self.tabRect(index)
        x = max(0, rect.left() - 3)
        separator_width = max(1, self.utility_separator.sizeHint().width())
        self.utility_separator.setGeometry(
            x, 4, separator_width, max(1, self.height() - 8)
        )
        self.utility_separator.show()


class WorkflowNavigation(QFrame):
    """Construct the workflow tab strip and contextual-sidebar toggle.

    The owning window supplies the page-stack tab widget and remains responsible
    for navigation semantics, signal wiring, and sidebar visibility policy.
    """

    def __init__(self, tabs: QTabWidget, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("workflowNavigation")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(4)

        self.sidebar_toggle_btn = QToolButton(self)
        self.sidebar_toggle_btn.setIcon(
            self.style().standardIcon(QStyle.SP_ToolBarHorizontalExtensionButton)
        )
        self.sidebar_toggle_btn.setText("Controls")
        self.sidebar_toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.sidebar_toggle_btn.setCheckable(True)
        self.sidebar_toggle_btn.setChecked(True)
        self.sidebar_toggle_btn.setToolTip("Show or hide the contextual controls sidebar (Ctrl+B)")
        apply_accessible_identity(
            self.sidebar_toggle_btn,
            name="Toggle controls sidebar",
            description="Show or hide the contextual controls sidebar",
        )
        layout.addWidget(self.sidebar_toggle_btn)

        self.workflow_tabs = _WorkflowTabBar(self)
        self.workflow_tabs.setExpanding(True)
        self.workflow_tabs.setUsesScrollButtons(False)
        for index in range(tabs.count()):
            tab_index = self.workflow_tabs.addTab(tabs.tabText(index))
            self.workflow_tabs.setTabToolTip(tab_index, tabs.tabToolTip(index))
        slides_index = next(
            (
                index
                for index in range(self.workflow_tabs.count())
                if self.workflow_tabs.tabText(index) == "Slides"
            ),
            -1,
        )
        self.workflow_tabs.set_separator_index(slides_index)
        self.workflow_separator = self.workflow_tabs.utility_separator
        layout.addWidget(self.workflow_tabs, 1)


__all__ = ["WorkflowNavigation"]
