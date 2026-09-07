import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QSplitter, QSizePolicy, QWidget

from ui_qt.common import UI_METRICS
from ui_qt.shell.workspace import WorkspaceShell


class WorkspaceShellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_constructs_navigation_splitter_and_workspace_stack(self):
        window = QMainWindow()
        navigation = QWidget()
        left_panel = QWidget()
        plot_panel = QWidget()
        presentation_widget = QWidget()
        shell = WorkspaceShell(
            window,
            navigation=navigation,
            left_panel=left_panel,
            plot_panel=plot_panel,
            presentation_widget=presentation_widget,
        )
        try:
            window.resize(1500, 800)
            central = shell.central_widget
            self.assertIs(window.centralWidget(), central)
            self.assertIsInstance(central.layout().itemAt(0).widget(), QWidget)
            self.assertIs(central.layout().itemAt(0).widget(), navigation)
            self.assertIs(central.layout().itemAt(1).widget(), shell.workspace_stack)
            self.assertEqual(
                [central.layout().itemAt(index).widget() for index in range(2)],
                [navigation, shell.workspace_stack],
            )
            self.assertEqual(
                [central.layout().contentsMargins().left(), central.layout().contentsMargins().top(),
                 central.layout().contentsMargins().right(), central.layout().contentsMargins().bottom()],
                [UI_METRICS["main_margin"]] * 4,
            )

            self.assertIsInstance(shell.workspace_splitter, QSplitter)
            self.assertEqual(shell.workspace_splitter.orientation(), Qt.Horizontal)
            self.assertFalse(shell.workspace_splitter.childrenCollapsible())
            self.assertEqual(shell.workspace_splitter.count(), 2)
            self.assertIs(shell.workspace_splitter.widget(0), left_panel)
            self.assertIs(shell.workspace_splitter.widget(1), plot_panel)
            shell.workspace_splitter.resize(UI_METRICS["left_width"] + 980 + 4, 500)
            shell.workspace_splitter.setSizes([UI_METRICS["left_width"], 980])
            self.assertEqual(shell.workspace_splitter.sizes(), [UI_METRICS["left_width"], 980])
            self.assertEqual(left_panel.sizePolicy().horizontalStretch(), 0)
            self.assertEqual(plot_panel.sizePolicy().horizontalStretch(), 1)
            self.assertEqual(left_panel.minimumWidth(), UI_METRICS["sidebar_min_width"])
            self.assertEqual(left_panel.maximumWidth(), UI_METRICS["sidebar_max_width"])
            self.assertEqual(left_panel.sizePolicy().horizontalPolicy(), QSizePolicy.Preferred)

            self.assertIsInstance(shell.workspace_stack, QStackedWidget)
            self.assertEqual(shell.workspace_stack.count(), 2)
            self.assertEqual(shell.workspace_stack.currentIndex(), 0)
            self.assertIs(shell.workspace_stack.widget(0), shell.workspace_splitter)
            self.assertIs(shell.workspace_stack.widget(1), presentation_widget)
        finally:
            window.close()

    def test_main_window_composes_shell_and_preserves_aliases(self):
        from ui_qt.main_window import MainWindow

        window = MainWindow()
        try:
            window.show()
            self.app.processEvents()
            self.assertIsInstance(window.workspace_shell, WorkspaceShell)
            shell = window.workspace_shell
            self.assertIs(window.centralWidget(), shell.central_widget)
            self.assertIs(window.left_panel, shell.left_panel)
            self.assertIs(window.workspace_splitter, shell.workspace_splitter)
            self.assertIs(window.presentation_widget, shell.presentation_widget)
            self.assertIs(window.workspace_stack, shell.workspace_stack)
            self.assertIs(window.workspace_stack.widget(0), window.workspace_splitter)
            self.assertIs(window.workspace_stack.widget(1), window.presentation_widget)
            self.assertEqual(window.centralWidget().layout().count(), 2)
            self.assertIs(window.centralWidget().layout().itemAt(1).widget(), window.workspace_stack)
            self.assertIs(window.data_source_context.parentWidget(), window.menu_toolbar_host.main_toolbar)
            self.assertTrue(window.data_source_context.isVisible())
            self.assertFalse(window.tabs.tabBar().isVisible())
            self.assertFalse(window.workspace_splitter.childrenCollapsible())
        finally:
            window.close()

    def test_toolbar_source_context_is_independent_of_sidebar_and_reuses_source_widgets(self):
        from ui_qt.main_window import MainWindow

        window = MainWindow()
        try:
            window.show()
            self.app.processEvents()
            context = window.data_source_context
            self.assertIs(context.parentWidget(), window.menu_toolbar_host.main_toolbar)
            self.assertTrue(window.menu_toolbar_host.source_widget_action.isVisible())
            self.assertIs(window.recent_folder_combo.lineEdit(), window.folder_edit)
            self.assertEqual(len(context.findChildren(type(window.folder_edit))), 1)
            self.assertEqual(
                len([child for child in window.findChildren(type(window.folder_edit)) if child.objectName() == "folderEdit"]),
                1,
            )
            self.assertEqual(
                len([child for child in window.findChildren(type(window.browse_btn)) if child.objectName() in {"browseFolderButton", "openFileButton", "refreshButton"}]),
                3,
            )
            window.sidebar_toggle_btn.setChecked(False)
            self.app.processEvents()
            self.assertFalse(window.left_panel.isVisible())
            self.assertTrue(window.menu_toolbar_host.source_widget_action.isVisible())

            slides_index = next(
                index for index in range(window.tabs.count())
                if window.tabs.tabText(index) == "Slides"
            )
            window.workflow_tabs.setCurrentIndex(slides_index)
            self.app.processEvents()
            self.assertFalse(window.menu_toolbar_host.source_widget_action.isVisible())
            self.assertFalse(window.menu_toolbar_host.source_separator_action.isVisible())
            self.assertFalse(context.isVisible())
            window.tabs.setCurrentIndex(0)
            self.app.processEvents()
            self.assertTrue(window.menu_toolbar_host.source_widget_action.isVisible())
            self.assertTrue(window.menu_toolbar_host.source_separator_action.isVisible())
        finally:
            window.close()

    def test_sidebar_starts_with_workflow_controls_without_data_source_banner(self):
        from ui_qt.main_window import MainWindow

        window = MainWindow()
        try:
            layout = window.left_panel.layout()
            widgets = [layout.itemAt(i).widget() for i in range(layout.count()) if layout.itemAt(i).widget()]
            self.assertFalse(any(getattr(widget, "title", lambda: "")() == "Data Source" for widget in widgets))
            self.assertFalse(any(getattr(widget, "property", lambda *_: None)("appRole") == "stepBanner" for widget in widgets))
            self.assertIs(widgets[0], window.tabs)
        finally:
            window.close()

    def test_main_window_keeps_bidirectional_workflow_sync_and_slides_policy(self):
        from ui_qt.main_window import MainWindow

        window = MainWindow()
        try:
            slides_index = next(
                index for index in range(window.tabs.count())
                if window.tabs.tabText(index) == "Slides"
            )
            window.workflow_tabs.setCurrentIndex(slides_index)
            self.app.processEvents()
            self.assertEqual(window.tabs.currentIndex(), slides_index)
            self.assertEqual(window.workspace_stack.currentIndex(), 1)
            self.assertFalse(window.sidebar_toggle_btn.isEnabled())
            self.assertFalse(window.show_sidebar_action.isEnabled())

            window.tabs.setCurrentIndex(0)
            self.app.processEvents()
            self.assertEqual(window.workflow_tabs.currentIndex(), 0)
            self.assertEqual(window.workspace_stack.currentIndex(), 0)
            self.assertTrue(window.sidebar_toggle_btn.isEnabled())
            self.assertTrue(window.show_sidebar_action.isEnabled())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
