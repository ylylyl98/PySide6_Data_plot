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
            self.assertIs(window.data_source_context.parentWidget(), window.left_panel)
            self.assertIs(window.left_panel.layout().itemAt(1).widget(), window.data_source_context)
            self.assertFalse(window.tabs.tabBar().isVisible())
            self.assertFalse(window.workspace_splitter.childrenCollapsible())
        finally:
            window.close()

    def test_sidebar_presents_data_source_context_above_workflow_controls(self):
        from ui_qt.main_window import MainWindow

        window = MainWindow()
        try:
            window.show()
            self.app.processEvents()
            context = window.data_source_context
            self.assertTrue(context.isVisible())
            self.assertIs(context.parentWidget(), window.left_panel)
            self.assertIs(window.left_panel.layout().itemAt(2).widget(), window.tabs)
            window.sidebar_toggle_btn.setChecked(False)
            self.app.processEvents()
            self.assertFalse(window.left_panel.isVisible())
            self.assertFalse(context.isVisible())

            slides_index = next(
                index for index in range(window.tabs.count())
                if window.tabs.tabText(index) == "Slides"
            )
            window.workflow_tabs.setCurrentIndex(slides_index)
            self.app.processEvents()
            self.assertFalse(context.isVisible())
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
