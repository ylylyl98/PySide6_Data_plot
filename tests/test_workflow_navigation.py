import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFrame, QTabWidget

from ui_qt.shell.workflow_navigation import WorkflowNavigation


class WorkflowNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(QTabWidget(), "Select")
        self.tabs.setTabToolTip(0, "Choose a data source")
        self.tabs.addTab(QTabWidget(), "Load")
        self.tabs.setTabToolTip(1, "Load selected data")
        self.tabs.addTab(QTabWidget(), "Plot")

    def tearDown(self):
        self.tabs.deleteLater()

    def test_constructs_mirrored_workflow_tabs_and_sidebar_toggle(self):
        navigation = WorkflowNavigation(self.tabs)

        self.assertEqual(navigation.objectName(), "workflowNavigation")
        self.assertEqual(navigation.workflow_tabs.count(), self.tabs.count())
        self.assertEqual(
            [navigation.workflow_tabs.tabText(i) for i in range(navigation.workflow_tabs.count())],
            ["Select", "Load", "Plot"],
        )
        self.assertEqual(navigation.workflow_tabs.tabToolTip(0), "Choose a data source")
        self.assertEqual(navigation.workflow_tabs.tabToolTip(1), "Load selected data")
        self.assertTrue(navigation.sidebar_toggle_btn.isCheckable())
        self.assertTrue(navigation.sidebar_toggle_btn.isChecked())
        self.assertEqual(navigation.sidebar_toggle_btn.text(), "Controls")
        self.assertEqual(
            navigation.sidebar_toggle_btn.toolTip(),
            "Show or hide the contextual controls sidebar (Ctrl+B)",
        )
        self.assertEqual(navigation.sidebar_toggle_btn.accessibleName(), "Toggle controls sidebar")
        self.assertEqual(
            navigation.sidebar_toggle_btn.accessibleDescription(),
            "Show or hide the contextual controls sidebar",
        )

    def test_does_not_own_workflow_or_sidebar_semantics(self):
        navigation = WorkflowNavigation(self.tabs)

        navigation.workflow_tabs.setCurrentIndex(1)
        self.assertEqual(self.tabs.currentIndex(), 0)
        navigation.sidebar_toggle_btn.setChecked(False)
        self.assertFalse(navigation.sidebar_toggle_btn.isChecked())

    def test_separator_before_slides_is_semantic_and_noninteractive(self):
        tabs = QTabWidget()
        for label in ("Select", "Load", "Plot", "Slides", "Tools"):
            tabs.addTab(QTabWidget(), label)
        navigation = WorkflowNavigation(tabs)
        navigation.resize(720, 44)
        navigation.show()
        self.app.processEvents()
        try:
            self.assertEqual(navigation.workflow_tabs.count(), tabs.count())
            self.assertEqual(
                [navigation.workflow_tabs.tabText(i) for i in range(navigation.workflow_tabs.count())],
                ["Select", "Load", "Plot", "Slides", "Tools"],
            )
            separator = navigation.workflow_separator
            self.assertIsInstance(separator, QFrame)
            self.assertEqual(separator.parent(), navigation.workflow_tabs)
            self.assertEqual(separator.objectName(), "workflowUtilitySeparator")
            self.assertEqual(separator.property("fluentRole"), "divider")
            self.assertEqual(separator.frameShape(), QFrame.VLine)
            self.assertEqual(separator.focusPolicy(), Qt.NoFocus)
            self.assertTrue(separator.testAttribute(Qt.WA_TransparentForMouseEvents))
            self.assertTrue(separator.isVisible())

            slides_index = next(
                index
                for index in range(navigation.workflow_tabs.count())
                if navigation.workflow_tabs.tabText(index) == "Slides"
            )
            slides_rect = navigation.workflow_tabs.tabRect(slides_index)
            self.assertLess(separator.geometry().center().x(), slides_rect.center().x())
            self.assertLessEqual(separator.geometry().right(), slides_rect.left())
            self.assertNotEqual(
                navigation.workflow_tabs.tabAt(separator.geometry().center()),
                slides_index,
            )
            navigation.workflow_tabs.setCurrentIndex(slides_index - 1)
            QTest.mouseClick(
                navigation.workflow_tabs,
                Qt.LeftButton,
                pos=separator.geometry().center(),
            )
            self.assertEqual(navigation.workflow_tabs.currentIndex(), slides_index - 1)
        finally:
            navigation.close()
            tabs.deleteLater()

    def test_separator_is_skipped_by_keyboard_navigation(self):
        tabs = QTabWidget()
        for label in ("Select", "Load", "Plot", "Slides", "Tools"):
            tabs.addTab(QTabWidget(), label)
        navigation = WorkflowNavigation(tabs)
        try:
            self.assertEqual(navigation.workflow_tabs.count(), 5)
            navigation.workflow_tabs.setCurrentIndex(2)
            navigation.workflow_tabs.setFocus()
            QTest.keyClick(navigation.workflow_tabs, Qt.Key_Right)
            self.assertEqual(navigation.workflow_tabs.currentIndex(), 3)
            self.assertEqual(navigation.workflow_tabs.tabText(3), "Slides")
            QTest.keyClick(navigation.workflow_tabs, Qt.Key_Right)
            self.assertEqual(navigation.workflow_tabs.currentIndex(), 4)
            self.assertEqual(navigation.workflow_tabs.tabText(4), "Tools")
        finally:
            navigation.deleteLater()
            tabs.deleteLater()

    def test_main_window_composes_component_and_preserves_sync_aliases(self):
        from ui_qt.main_window import MainWindow

        window = MainWindow()
        try:
            self.assertIsInstance(window.workflow_navigation, WorkflowNavigation)
            self.assertEqual(window.workflow_navigation.objectName(), "workflowNavigation")
            self.assertIs(window.workflow_tabs, window.workflow_navigation.workflow_tabs)
            self.assertIs(window.sidebar_toggle_btn, window.workflow_navigation.sidebar_toggle_btn)
            self.assertEqual(window.workflow_tabs.count(), window.tabs.count())

            window.workflow_tabs.setCurrentIndex(1)
            self.assertEqual(window.tabs.currentIndex(), 1)
            window.tabs.setCurrentIndex(0)
            self.assertEqual(window.workflow_tabs.currentIndex(), 0)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
