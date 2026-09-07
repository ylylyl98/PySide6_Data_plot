"""Presentational construction for the main workspace shell."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui_qt.common import UI_METRICS


class WorkspaceShell:
    """Build the central navigation/workspace shell without owning policy."""

    def __init__(
        self,
        window: QMainWindow,
        *,
        navigation: QWidget,
        left_panel: QWidget,
        plot_panel: QWidget,
        presentation_widget: QWidget,
    ) -> None:
        self.central_widget = QWidget()
        window.setCentralWidget(self.central_widget)

        self.left_panel = left_panel
        self.left_panel.setMinimumWidth(UI_METRICS["sidebar_min_width"])
        self.left_panel.setMaximumWidth(UI_METRICS["sidebar_max_width"])
        self.left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.workspace_splitter = QSplitter(Qt.Horizontal)
        self.workspace_splitter.addWidget(left_panel)
        self.workspace_splitter.addWidget(plot_panel)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setSizes([UI_METRICS["left_width"], 980])
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 1)

        self.presentation_widget = presentation_widget
        self.workspace_stack = QStackedWidget()
        self.workspace_stack.addWidget(self.workspace_splitter)
        self.workspace_stack.addWidget(presentation_widget)

        layout = QVBoxLayout(self.central_widget)
        margin = UI_METRICS["main_margin"]
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.addWidget(navigation)
        layout.addWidget(self.workspace_stack, 1)


__all__ = ["WorkspaceShell"]
