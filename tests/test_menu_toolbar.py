import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QMainWindow,
    QPushButton,
    QToolBar,
    QWidgetAction,
    QWidget,
)
from PySide6.QtCore import Qt

from ui_qt.shell.menu_toolbar import MenuToolbarHost


class MenuToolbarHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_constructs_view_and_help_menus_with_exact_action_order(self):
        window = QMainWindow()
        log_dock = QDockWidget("Run Log", window)
        results_dock = QDockWidget("Analysis Results", window)
        host = MenuToolbarHost(
            window,
            log_dock=log_dock,
            results_dock=results_dock,
        )
        try:
            menus = window.menuBar().actions()
            self.assertEqual([action.text() for action in menus], ["View", "Data Maintenance", "Help"])
            self.assertEqual(
                [action.text() for action in host.view_menu.actions()],
                ["Show Log", "Show Analysis Results", "Show Controls Sidebar"],
            )
            self.assertEqual(
                [action.text() for action in host.help_menu.actions()],
                ["Check for Updates...", "Check for Updates Automatically", "About"],
            )
            self.assertEqual(
                len(host.data_maintenance_menu.actions()),
                2,
            )
            self.assertTrue(
                all(
                    isinstance(action, QWidgetAction)
                    for action in host.data_maintenance_menu.actions()
                )
            )
            self.assertIs(host.show_log_action, log_dock.toggleViewAction())
            self.assertIs(host.show_results_action, results_dock.toggleViewAction())
            self.assertFalse(host.show_sidebar_action.isCheckable())
            self.assertFalse(host.show_sidebar_action.isChecked())
            self.assertFalse(host.auto_update_check_action.isCheckable())
            self.assertFalse(host.auto_update_check_action.isChecked())
            self.assertEqual(host.show_sidebar_action.shortcut().toString(), "")
        finally:
            window.close()

    def test_constructs_non_movable_main_toolbar_with_exact_items(self):
        window = QMainWindow()
        log_dock = QDockWidget("Run Log", window)
        results_dock = QDockWidget("Analysis Results", window)
        source = QWidget()
        source.setObjectName("dataSourceContext")
        host = MenuToolbarHost(window, log_dock=log_dock, results_dock=results_dock, data_source_context=source)
        try:
            self.assertEqual(host.main_toolbar.windowTitle(), "Main")
            self.assertFalse(host.main_toolbar.isMovable())
            self.assertEqual(
                [action.text() for action in host.main_toolbar.actions()],
                ["Load", "Plot / Update", "Save PNG + DAT", "", ""],
            )
            self.assertEqual(host.main_toolbar.actions()[3], host.source_separator_action)
            self.assertEqual(host.main_toolbar.actions()[4], host.source_widget_action)
            self.assertIs(host.source_widget_action.defaultWidget(), source)
            self.assertIs(host.main_toolbar.widgetForAction(host.source_widget_action), source)
            self.assertEqual(host.load_action.toolTip(), "Load data for the active tab")
            self.assertEqual(host.plot_action.toolTip(), "Plot/update current state")
            self.assertEqual(host.save_action.toolTip(), "Export for the active tab")
            self.assertIsInstance(host.clean_verified_sources_chk, QCheckBox)
            self.assertIsInstance(host.move_now_btn, QPushButton)
            self.assertTrue(host.move_now_btn.isEnabled())
            self.assertFalse(host.clean_verified_sources_chk.isChecked())
            self.assertIsNone(host.main_toolbar.widgetForAction(host.clean_verified_sources_action))
            self.assertIsNone(host.main_toolbar.widgetForAction(host.move_now_action))
            self.assertIs(host.clean_verified_sources_action.defaultWidget(), host.clean_verified_sources_chk)
            self.assertIs(host.move_now_action.defaultWidget(), host.move_now_btn)
            self.assertIsNotNone(host.data_maintenance_menu)

            state_changes = []
            host.clean_verified_sources_chk.toggled.connect(state_changes.append)
            host.clean_verified_sources_chk.setChecked(True)
            self.assertEqual(state_changes, [True])
            clicked = []
            host.move_now_btn.clicked.connect(lambda: clicked.append(True))
            host.move_now_btn.click()
            self.assertEqual(clicked, [True])
        finally:
            window.close()

    def test_primary_actions_use_fluent_icons_and_compact_panel_buttons(self):
        window = QMainWindow()
        log_dock = QDockWidget("Run Log", window)
        results_dock = QDockWidget("Analysis Results", window)
        source = QWidget()
        host = MenuToolbarHost(window, log_dock=log_dock, results_dock=results_dock, data_source_context=source)
        try:
            self.assertFalse(host.load_action.icon().isNull())
            self.assertFalse(host.plot_action.icon().isNull())
            self.assertFalse(host.save_action.icon().isNull())
            self.assertEqual(host.main_toolbar.toolButtonStyle(), Qt.ToolButtonTextBesideIcon)
            self.assertEqual(host.panels_toolbar.toolButtonStyle(), Qt.ToolButtonTextBesideIcon)
            self.assertEqual(host.panels_toolbar.objectName(), "panelsToolbar")
            self.assertEqual(host.main_toolbar.objectName(), "mainToolbar")
            self.assertFalse(host.show_results_action.icon().isNull())
            self.assertFalse(host.show_log_action.icon().isNull())
        finally:
            window.close()

    def test_main_window_composes_host_and_preserves_compatibility_aliases(self):
        from ui_qt.main_window import MainWindow

        window = MainWindow()
        try:
            self.assertIsInstance(window.menu_toolbar_host, MenuToolbarHost)
            host = window.menu_toolbar_host
            for name in (
                "show_log_action",
                "show_results_action",
                "show_sidebar_action",
                "check_updates_action",
                "auto_update_check_action",
                "about_action",
                "load_action",
                "plot_action",
                "save_action",
                "move_now_btn",
                "clean_verified_sources_chk",
            ):
                self.assertIs(getattr(window, name), getattr(host, name))
            self.assertTrue(window.show_sidebar_action.isCheckable())
            self.assertTrue(window.show_sidebar_action.isChecked())
            self.assertTrue(window.auto_update_check_action.isCheckable())
            self.assertEqual(
                window.auto_update_check_action.isChecked(),
                window._auto_update_check_enabled(),
            )
            self.assertFalse(window.move_now_btn.isEnabled())
            self.assertEqual(window.show_sidebar_action.shortcut().toString(), "Ctrl+B")
        finally:
            window.close()

    def test_panels_toolbar_is_bottom_docked_and_reuses_view_actions(self):
        window = QMainWindow()
        log_dock = QDockWidget("Run Log", window)
        results_dock = QDockWidget("Analysis Results", window)
        window.addDockWidget(Qt.BottomDockWidgetArea, log_dock)
        window.addDockWidget(Qt.BottomDockWidgetArea, results_dock)
        log_dock.hide()
        results_dock.hide()
        host = MenuToolbarHost(window, log_dock=log_dock, results_dock=results_dock)
        try:
            window.show()
            self.app.processEvents()
            self.assertIsInstance(host.panels_toolbar, QToolBar)
            self.assertEqual(host.panels_toolbar.windowTitle(), "Panels")
            self.assertFalse(host.panels_toolbar.isMovable())
            self.assertEqual(host.panels_toolbar.toolButtonStyle(), Qt.ToolButtonTextBesideIcon)
            self.assertEqual(window.toolBarArea(host.panels_toolbar), Qt.BottomToolBarArea)
            self.assertEqual(
                host.panels_toolbar.actions(),
                [host.show_results_action, host.show_log_action],
            )
            self.assertIs(host.panels_toolbar.actions()[0], results_dock.toggleViewAction())
            self.assertIs(host.panels_toolbar.actions()[1], log_dock.toggleViewAction())
            for action in host.panels_toolbar.actions():
                button = host.panels_toolbar.widgetForAction(action)
                self.assertIsNotNone(button)
                self.assertTrue(button.isVisible())
                self.assertTrue(button.text())
            self.assertEqual(
                [action.text() for action in host.view_menu.actions()],
                ["Show Log", "Show Analysis Results", "Show Controls Sidebar"],
            )
            self.assertFalse(log_dock.isVisible())
            self.assertFalse(results_dock.isVisible())
            self.assertFalse(host.show_log_action.isChecked())
            self.assertFalse(host.show_results_action.isChecked())
        finally:
            window.close()

    def test_panels_actions_keep_dock_and_checked_state_synchronized(self):
        window = QMainWindow()
        log_dock = QDockWidget("Run Log", window)
        results_dock = QDockWidget("Analysis Results", window)
        window.addDockWidget(Qt.RightDockWidgetArea, log_dock)
        window.addDockWidget(Qt.RightDockWidgetArea, results_dock)
        host = MenuToolbarHost(window, log_dock=log_dock, results_dock=results_dock)
        try:
            window.show()
            self.app.processEvents()
            log_dock.hide()
            results_dock.hide()
            self.app.processEvents()
            results_dock.show()
            self.app.processEvents()
            self.assertTrue(host.show_results_action.isChecked())
            self.assertTrue(results_dock.isVisible())
            results_dock.hide()
            self.app.processEvents()
            self.assertFalse(host.show_results_action.isChecked())

            host.show_log_action.trigger()
            self.app.processEvents()
            self.assertTrue(log_dock.isVisible())
            self.assertTrue(host.show_log_action.isChecked())
            host.show_log_action.trigger()
            self.app.processEvents()
            self.assertFalse(log_dock.isVisible())
            self.assertFalse(host.show_log_action.isChecked())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
