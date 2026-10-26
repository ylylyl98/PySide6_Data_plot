"""Persistent workflow destinations for the application shell."""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QLabel, QSizePolicy, QTabBar, QTabWidget, QToolButton, QVBoxLayout,
)
from ui_qt.fluent_ui.style import apply_accessible_identity, set_fluent_property


class _WorkflowTabBar(QTabBar):
    """Keep the indexed Qt API while the rail presents its state."""
    metadata_changed = Signal()

    def setTabVisible(self, index: int, visible: bool) -> None:  # noqa: N802
        super().setTabVisible(index, visible)
        self.metadata_changed.emit()

    def setTabEnabled(self, index: int, enabled: bool) -> None:  # noqa: N802
        super().setTabEnabled(index, enabled)
        self.metadata_changed.emit()

    def setTabText(self, index: int, text: str) -> None:  # noqa: N802
        super().setTabText(index, text)
        self.metadata_changed.emit()

    def setTabToolTip(self, index: int, text: str) -> None:  # noqa: N802
        super().setTabToolTip(index, text)
        self.metadata_changed.emit()


class WorkflowNavigation(QFrame):
    """Persistent standard buttons mirroring the existing indexed QTabBar.

    The owning window retains all page and sidebar policy. The hidden QTabBar
    preserves the existing public signal, index and visibility contracts.
    """

    def __init__(self, tabs: QTabWidget, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("workflowNavigation")
        set_fluent_property(self, "appRole", "workflowRail")
        set_fluent_property(self, "fluentRole", "activityBar")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setFixedWidth(104)  # Deliberate shell metric, with horizontal text.
        apply_accessible_identity(self, name="Workflow navigation", identifier="shell.workflows")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 18, 8, 12)
        layout.setSpacing(6)
        self.rail_title = QLabel("ANALYZE", self)
        self.rail_title.setAlignment(Qt.AlignCenter)
        self.rail_title.setProperty("appRole", "workflowRailTitle")
        layout.addWidget(self.rail_title)
        layout.addSpacing(10)

        self.workflow_tabs = _WorkflowTabBar(self)
        self.workflow_tabs.setObjectName("workflowTabs")
        self.workflow_tabs.setExpanding(True)
        self.workflow_tabs.setUsesScrollButtons(False)
        self.workflow_tabs.setFocusPolicy(Qt.NoFocus)
        for index in range(tabs.count()):
            item = self.workflow_tabs.addTab(tabs.tabText(index))
            self.workflow_tabs.setTabToolTip(item, tabs.tabToolTip(index))
            self.workflow_tabs.setTabEnabled(item, tabs.isTabEnabled(index))
            self.workflow_tabs.setTabVisible(item, tabs.isTabVisible(index))
        self.workflow_tabs.setCurrentIndex(tabs.currentIndex())
        self.workflow_tabs.hide()

        self.destination_group = QButtonGroup(self)
        self.destination_group.setExclusive(True)
        self.destination_buttons: dict[int, QToolButton] = {}
        utility_indices = []
        for index in range(self.workflow_tabs.count()):
            if self.workflow_tabs.tabText(index) in {"Slides", "Tools"}:
                utility_indices.append(index)
            else:
                layout.addWidget(self._make_destination(index))
        layout.addStretch(1)
        self.workflow_separator = QFrame(self)
        self.workflow_separator.setObjectName("workflowUtilitySeparator")
        self.workflow_separator.setFrameShape(QFrame.HLine)
        self.workflow_separator.setFrameShadow(QFrame.Plain)
        self.workflow_separator.setProperty("fluentRole", "divider")
        self.workflow_separator.setFocusPolicy(Qt.NoFocus)
        self.workflow_separator.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.workflow_separator)
        self.utility_label = QLabel("WORKSPACE", self)
        self.utility_label.setAlignment(Qt.AlignCenter)
        self.utility_label.setProperty("appRole", "workflowUtilityLabel")
        layout.addWidget(self.utility_label)
        for index in utility_indices:
            layout.addWidget(self._make_destination(index))
        self.workflow_separator.setVisible(bool(utility_indices))
        self.utility_label.setVisible(bool(utility_indices))

        self.sidebar_toggle_btn = QToolButton(self)
        self.sidebar_toggle_btn.setObjectName("sidebarToggleButton")
        self.sidebar_toggle_btn.setText("Controls")
        self.sidebar_toggle_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.sidebar_toggle_btn.setAutoRaise(True)
        self.sidebar_toggle_btn.setProperty("appRole", "sidebarToggle")
        self.sidebar_toggle_btn.setFocusPolicy(Qt.StrongFocus)
        self.sidebar_toggle_btn.setCheckable(True)
        self.sidebar_toggle_btn.setChecked(True)
        self.sidebar_toggle_btn.setToolTip("Show or hide the contextual controls sidebar (Ctrl+B)")
        apply_accessible_identity(
            self.sidebar_toggle_btn, name="Toggle controls sidebar",
            description="Show or hide the contextual controls sidebar",
            identifier="shell.sidebar.toggle",
        )
        # WorkspaceShell moves this same instance to its contextual header.
        layout.addWidget(self.sidebar_toggle_btn)
        self.destination_group.idClicked.connect(self.workflow_tabs.setCurrentIndex)
        self.workflow_tabs.currentChanged.connect(self._sync_selection)
        self.workflow_tabs.metadata_changed.connect(self._sync_destinations)
        self._sync_destinations()
        self._sync_selection(self.workflow_tabs.currentIndex())

    def _make_destination(self, index: int) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName(f"workflowDestination{index}")
        button.setProperty("appRole", "workflowDestination")
        button.setProperty("workflowIndex", index)
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        button.setCheckable(True)
        button.setAutoRaise(True)
        button.setFocusPolicy(Qt.StrongFocus)
        button.setMinimumHeight(max(42, button.fontMetrics().height() + 20))
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button.installEventFilter(self)
        self.destination_group.addButton(button, index)
        self.destination_buttons[index] = button
        return button

    def _sync_destinations(self) -> None:
        for index, button in self.destination_buttons.items():
            text = self.workflow_tabs.tabText(index)
            tooltip = self.workflow_tabs.tabToolTip(index) or f"Open {text} workspace"
            button.setText(text)
            button.setToolTip(tooltip)
            button.setVisible(self.workflow_tabs.isTabVisible(index))
            button.setEnabled(self.workflow_tabs.isTabEnabled(index))
            apply_accessible_identity(
                button, name=f"{text} workspace", description=tooltip,
                identifier=f"workflow.{index}",
            )

    def _sync_selection(self, index: int) -> None:
        self.destination_group.setExclusive(False)
        for item, button in self.destination_buttons.items():
            button.setChecked(item == index)
            button.setFocusPolicy(Qt.StrongFocus if item == index else Qt.ClickFocus)
        self.destination_group.setExclusive(True)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() == QEvent.KeyPress and watched in self.destination_buttons.values():
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                watched.click()
                return True
            keys = (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right, Qt.Key_Home, Qt.Key_End)
            if event.key() in keys:
                available = [
                    button for index, button in self.destination_buttons.items()
                    if self.workflow_tabs.isTabVisible(index) and button.isEnabled()
                ]
                if available:
                    current = available.index(watched) if watched in available else 0
                    if event.key() == Qt.Key_Home:
                        target = 0
                    elif event.key() == Qt.Key_End:
                        target = len(available) - 1
                    else:
                        direction = 1 if event.key() in (Qt.Key_Down, Qt.Key_Right) else -1
                        target = max(0, min(len(available) - 1, current + direction))
                    available[target].setFocus(Qt.ShortcutFocusReason)
                return True
        return super().eventFilter(watched, event)


__all__ = ["WorkflowNavigation"]
