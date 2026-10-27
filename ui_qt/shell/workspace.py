"""Presentational construction for the main workspace shell."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMainWindow, QMenu, QSizePolicy, QSplitter,
    QStackedWidget, QToolButton, QVBoxLayout, QWidget,
)
from ui_qt.common import UI_METRICS
from ui_qt.fluent_ui.style import apply_accessible_identity, set_fluent_property


class _WorkspaceTitle(QLabel):
    """Let a long title contract while its full text remains in the tooltip."""

    def minimumSizeHint(self):  # noqa: N802 - Qt API
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint


class WorkspaceShell:
    """Build a persistent destination rail and contextual analysis workspace."""

    def __init__(
        self, window: QMainWindow, *, navigation: QWidget, left_panel: QWidget,
        plot_panel: QWidget, presentation_widget: QWidget,
        tools_widget: QWidget | None = None,
        plot_toolbar: QWidget | None = None,
    ) -> None:
        self.central_widget = QWidget()
        self.central_widget.setObjectName("workspaceShell")
        window.setCentralWidget(self.central_widget)
        self.navigation = navigation
        self._rail_visible = True
        self.left_panel = left_panel
        set_fluent_property(left_panel, "appRole", "controlsSidebar")
        self.left_panel.setMinimumWidth(UI_METRICS["sidebar_min_width"])
        self.left_panel.setMaximumWidth(UI_METRICS["sidebar_max_width"])
        self.left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        set_fluent_property(plot_panel, "appRole", "workspaceCanvas")
        self.workspace_splitter = QSplitter(Qt.Horizontal)
        self.workspace_splitter.setObjectName("workspaceSplitter")
        self.workspace_splitter.addWidget(left_panel)
        self.workspace_splitter.addWidget(plot_panel)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setSizes([UI_METRICS["left_width"], 980])
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 1)

        self.presentation_widget = presentation_widget
        self.workspace_stack = QStackedWidget()
        self.workspace_stack.setObjectName("workspaceStack")
        self.workspace_stack.setProperty("appRole", "workspaceCanvas")
        self.workspace_stack.addWidget(self.workspace_splitter)
        self.workspace_stack.addWidget(presentation_widget)
        self.tools_widget = tools_widget
        if tools_widget is not None:
            self.workspace_stack.addWidget(tools_widget)

        self.workspace_content = QWidget()
        self.workspace_content.setObjectName("workspaceContent")
        self.workspace_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_layout = QVBoxLayout(self.workspace_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        self.header = QFrame()
        self.header.setObjectName("workspaceHeader")
        self.header.setProperty("appRole", "workspaceHeader")
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(12, 4, 12, 4)
        header_layout.setSpacing(8)
        # Context metadata remains available to callers and accessibility;
        # the visible header is one row of title, plot commands and navigation.
        self.eyebrow_label = QLabel("ANALYSIS WORKSPACE", self.header)
        self.eyebrow_label.setProperty("appRole", "workspaceEyebrow")
        self.eyebrow_label.hide()
        self.title_label = _WorkspaceTitle("Data analysis")
        self.title_label.setProperty("appRole", "workspaceTitle")
        self.title_label.setMinimumWidth(0)
        self.title_label.setMaximumWidth(240)
        self.title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.subtitle_label = QLabel("Select a workflow and choose a data source to begin.", self.header)
        self.subtitle_label.setProperty("appRole", "workspaceSubtitle")
        self.subtitle_label.hide()
        header_layout.addWidget(self.title_label)
        self.plot_toolbar = plot_toolbar
        if plot_toolbar is not None:
            old_parent = plot_toolbar.parentWidget()
            if old_parent is not None and old_parent.layout() is not None:
                old_parent.layout().removeWidget(plot_toolbar)
            plot_toolbar.setMinimumWidth(0)
            plot_toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            header_layout.addWidget(plot_toolbar, 1)
        else:
            header_layout.addStretch(1)

        self.workflow_selector_btn = QToolButton(self.header)
        self.workflow_selector_btn.setObjectName("compactWorkflowSelector")
        self.workflow_selector_btn.setProperty("appRole", "sidebarToggle")
        self.workflow_selector_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.workflow_selector_btn.setPopupMode(QToolButton.InstantPopup)
        self.workflow_selector_btn.setAutoRaise(True)
        self.workflow_selector_btn.setFocusPolicy(Qt.StrongFocus)
        apply_accessible_identity(
            self.workflow_selector_btn, name="Switch workflow",
            description="Select a workflow while the navigation rail is hidden",
            identifier="shell.workflow.selector",
        )
        self.workflow_menu = QMenu(self.workflow_selector_btn)
        self.workflow_selector_btn.setMenu(self.workflow_menu)
        self._workflow_action_group = QActionGroup(self.workflow_menu)
        self._workflow_action_group.setExclusive(True)
        self._workflow_actions = {}
        self._workflow_tabs = getattr(navigation, "workflow_tabs", None)
        if self._workflow_tabs is not None:
            self._workflow_tabs.currentChanged.connect(self._sync_workflow_selector)
            metadata_changed = getattr(self._workflow_tabs, "metadata_changed", None)
            if metadata_changed is not None:
                metadata_changed.connect(self._sync_workflow_selector)
        self.workflow_menu.aboutToShow.connect(self._sync_workflow_selector)
        header_layout.addWidget(self.workflow_selector_btn, 0, Qt.AlignVCenter)

        self.rail_toggle_btn = QToolButton(self.header)
        self.rail_toggle_btn.setObjectName("workflowRailToggleButton")
        self.rail_toggle_btn.setProperty("appRole", "sidebarToggle")
        self.rail_toggle_btn.setText("Workflows")
        self.rail_toggle_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.rail_toggle_btn.setAutoRaise(True)
        self.rail_toggle_btn.setCheckable(True)
        self.rail_toggle_btn.setChecked(True)
        self.rail_toggle_btn.setFocusPolicy(Qt.StrongFocus)
        self.rail_toggle_btn.setToolTip("Show or hide workflow navigation")
        apply_accessible_identity(
            self.rail_toggle_btn, name="Toggle workflow navigation",
            description="Show or hide the workflow rail while keeping the controls sidebar visible",
            identifier="shell.workflow.railToggle",
        )
        self.rail_toggle_btn.toggled.connect(self.set_rail_visible)
        header_layout.addWidget(self.rail_toggle_btn, 0, Qt.AlignVCenter)
        sidebar_toggle = getattr(navigation, "sidebar_toggle_btn", None)
        if sidebar_toggle is not None:
            header_layout.addWidget(sidebar_toggle, 0, Qt.AlignVCenter)
            QWidget.setTabOrder(self.rail_toggle_btn, sidebar_toggle)
        QWidget.setTabOrder(self.workflow_selector_btn, self.rail_toggle_btn)
        content_layout.addWidget(self.header)
        content_layout.addWidget(self.workspace_stack, 1)
        layout = QHBoxLayout(self.central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(navigation)
        layout.addWidget(self.workspace_content, 1)
        self._sync_workflow_selector()

    @property
    def rail_visible(self) -> bool:
        """Requested rail visibility, independent of the window's shown state."""
        return self._rail_visible

    def is_rail_visible(self) -> bool:
        return self._rail_visible

    def set_rail_visible(self, visible: bool) -> None:
        """Reclaim rail width without changing the controls sidebar or workflow.

        Programmatic updates do not emit the toggle button's clicked signal;
        the owning window can use clicked(bool) to remember manual overrides.
        """
        self._rail_visible = bool(visible)
        self.navigation.setVisible(self._rail_visible)
        blocked = self.rail_toggle_btn.blockSignals(True)
        self.rail_toggle_btn.setChecked(self._rail_visible)
        self.rail_toggle_btn.blockSignals(blocked)
        self.workflow_selector_btn.setVisible(
            not self._rail_visible and self._workflow_tabs is not None
        )

    def _sync_workflow_selector(self, *_args) -> None:
        tabs = self._workflow_tabs
        if tabs is None:
            self.workflow_selector_btn.hide()
            return
        for index in tuple(self._workflow_actions):
            if index >= tabs.count():
                action = self._workflow_actions.pop(index)
                self.workflow_menu.removeAction(action)
                action.deleteLater()
        for index in range(tabs.count()):
            action = self._workflow_actions.get(index)
            if action is None:
                action = self.workflow_menu.addAction(tabs.tabText(index))
                action.setCheckable(True)
                self._workflow_action_group.addAction(action)
                action.triggered.connect(
                    lambda checked, target=index: tabs.setCurrentIndex(target) if checked else None
                )
                self._workflow_actions[index] = action
            action.setText(tabs.tabText(index))
            action.setToolTip(tabs.tabToolTip(index))
            action.setVisible(tabs.isTabVisible(index))
            action.setEnabled(tabs.isTabEnabled(index))
            action.setChecked(index == tabs.currentIndex())
        current = tabs.tabText(tabs.currentIndex())
        self.workflow_selector_btn.setText(current or "Workflow")
        self.workflow_selector_btn.setToolTip("Switch workflow · Current: " + current)
        self.workflow_selector_btn.setVisible(not self._rail_visible)

    def set_context(
        self, title: str, subtitle: str = "", *, eyebrow: str = "ANALYSIS WORKSPACE"
    ) -> None:
        """Update the workspace heading without changing navigation or content."""
        self.eyebrow_label.setText(eyebrow)
        self.title_label.setText(title)
        self.subtitle_label.setText(subtitle)
        self.title_label.setToolTip(title + ("\n" + subtitle if subtitle else ""))
        self.header.setToolTip(self.title_label.toolTip())
        apply_accessible_identity(self.title_label, name=title, description=subtitle)
        apply_accessible_identity(
            self.header, name=f"{title} workspace", description=subtitle,
            identifier="shell.workspace.context",
        )


__all__ = ["WorkspaceShell"]
