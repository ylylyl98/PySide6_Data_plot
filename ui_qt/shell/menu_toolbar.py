"""Presentational construction for the application's menus and main toolbar."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QMainWindow,
    QPushButton,
    QToolBar,
    QWidgetAction,
)
from ui_qt.fluent_ui.style import themed_icon


class MenuToolbarHost:
    """Build the main menus and toolbar without owning application semantics."""

    def __init__(
        self,
        window: QMainWindow,
        *,
        log_dock: QDockWidget,
        results_dock: QDockWidget,
    ) -> None:
        self.view_menu = window.menuBar().addMenu("View")
        self.show_log_action = log_dock.toggleViewAction()
        self.show_log_action.setText("Show Log")
        self.view_menu.addAction(self.show_log_action)
        self.show_results_action = results_dock.toggleViewAction()
        self.show_results_action.setText("Show Analysis Results")
        self.view_menu.addAction(self.show_results_action)
        self.show_sidebar_action = QAction("Show Controls Sidebar", window)
        self.view_menu.addAction(self.show_sidebar_action)

        self.panels_toolbar = QToolBar("Panels", window)
        window.addToolBar(Qt.BottomToolBarArea, self.panels_toolbar)
        self.panels_toolbar.setObjectName("panelsToolbar")
        self.panels_toolbar.setMovable(False)
        self.panels_toolbar.setFloatable(False)
        self.panels_toolbar.setAllowedAreas(Qt.BottomToolBarArea)
        self.panels_toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.panels_toolbar.setIconSize(QSize(16, 16))
        self.panels_toolbar.addAction(self.show_results_action)
        self.panels_toolbar.addAction(self.show_log_action)

        self.data_maintenance_menu = window.menuBar().addMenu("Data Maintenance")

        self.help_menu = window.menuBar().addMenu("Help")
        self.check_updates_action = QAction("Check for Updates...", window)
        self.auto_update_check_action = QAction("Check for Updates Automatically", window)
        self.about_action = QAction("About", window)
        self.help_menu.addAction(self.check_updates_action)
        self.help_menu.addAction(self.auto_update_check_action)
        self.help_menu.addAction(self.about_action)

        self.main_toolbar = window.addToolBar("Main")
        self.main_toolbar.setObjectName("mainToolbar")
        self.main_toolbar.setMovable(False)
        self.main_toolbar.setFloatable(False)
        self.main_toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.main_toolbar.setIconSize(QSize(18, 18))
        self.load_action = QAction("Load", window)
        self.load_action.setToolTip("Load data for the active tab")
        self.plot_action = QAction("Plot / Update", window)
        self.plot_action.setToolTip("Plot/update current state")
        self.save_action = QAction("Save PNG + DAT", window)
        self.save_action.setToolTip("Export for the active tab")
        self.move_now_btn = QPushButton("Move Exported Sources")
        self.move_now_btn.setToolTip("Save first to enable moving exported source files.")
        self.clean_verified_sources_chk = QCheckBox("Clean verified source copies after successful export")
        self.clean_verified_sources_chk.setToolTip(
            "Only verified root-level files matching Initial Data by SHA-256 can be deleted."
        )
        self.clean_verified_sources_action = QWidgetAction(window)
        self.clean_verified_sources_action.setDefaultWidget(self.clean_verified_sources_chk)
        self.data_maintenance_menu.addAction(self.clean_verified_sources_action)
        self.move_now_action = QWidgetAction(window)
        self.move_now_action.setDefaultWidget(self.move_now_btn)
        self.data_maintenance_menu.addAction(self.move_now_action)
        self.main_toolbar.addAction(self.load_action)
        self.main_toolbar.addAction(self.plot_action)
        self.main_toolbar.addAction(self.save_action)
        self.apply_theme()

    def apply_theme(self, theme=None, *, navigation_toolbar=None) -> None:
        """Refresh approved Fluent SVG icons while preserving QAction identity."""
        self.load_action.setIcon(themed_icon("open-folder.svg", theme=theme))
        self.plot_action.setIcon(themed_icon("arrow-sync.svg", theme=theme))
        self.save_action.setIcon(themed_icon("save.svg", theme=theme))
        self.show_results_action.setIcon(themed_icon("panel-results.svg", theme=theme))
        self.show_log_action.setIcon(themed_icon("panel-log.svg", theme=theme))
        if navigation_toolbar is None:
            return
        nav_icons = {
            "Home": "home.svg",
            "Back": "arrow-left.svg",
            "Forward": "arrow-right.svg",
            "Pan": "cursor-move.svg",
            "Zoom": "zoom.svg",
            "Subplots": "layout.svg",
            "Customize": "edit.svg",
            "Save": "save.svg",
        }
        for action in navigation_toolbar.actions():
            filename = nav_icons.get(action.text())
            if filename:
                action.setIcon(themed_icon(filename, theme=theme))


__all__ = ["MenuToolbarHost"]
