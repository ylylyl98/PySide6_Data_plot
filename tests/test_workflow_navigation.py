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
        self.assertTrue(navigation.workflow_tabs.isHidden())
        self.assertEqual(navigation.property("appRole"), "workflowRail")
        self.assertEqual(navigation.width(), 104)
        self.assertEqual(
            [button.text() for button in navigation.destination_buttons.values()],
            ["Select", "Load", "Plot"],
        )
        self.assertTrue(navigation.destination_buttons[0].isChecked())
        self.assertTrue(navigation.sidebar_toggle_btn.isCheckable())
        self.assertTrue(navigation.sidebar_toggle_btn.isChecked())
        self.assertTrue(navigation.sidebar_toggle_btn.autoRaise())
        self.assertEqual(navigation.sidebar_toggle_btn.property("appRole"), "sidebarToggle")
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
        navigation.resize(104, 600)
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
            self.assertEqual(separator.parent(), navigation)
            self.assertEqual(separator.objectName(), "workflowUtilitySeparator")
            self.assertEqual(separator.property("fluentRole"), "divider")
            self.assertEqual(separator.frameShape(), QFrame.HLine)
            self.assertEqual(separator.focusPolicy(), Qt.NoFocus)
            self.assertTrue(separator.testAttribute(Qt.WA_TransparentForMouseEvents))
            self.assertTrue(separator.isVisible())

            slides_index = next(
                index
                for index in range(navigation.workflow_tabs.count())
                if navigation.workflow_tabs.tabText(index) == "Slides"
            )
            slides_button = navigation.destination_buttons[slides_index]
            self.assertLess(separator.geometry().bottom(), slides_button.geometry().top())
            navigation.workflow_tabs.setCurrentIndex(slides_index - 1)
            QTest.mouseClick(
                separator,
                Qt.LeftButton,
                pos=separator.rect().center(),
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
        navigation.show()
        self.app.processEvents()
        try:
            self.assertEqual(navigation.workflow_tabs.count(), 5)
            navigation.workflow_tabs.setCurrentIndex(2)
            button = navigation.destination_buttons[2]
            button.setFocus()
            QTest.keyClick(button, Qt.Key_Down)
            # Arrow keys move focus; Enter/Space activates a destination.
            self.assertTrue(navigation.destination_buttons[3].hasFocus())
            self.assertEqual(navigation.workflow_tabs.currentIndex(), 2)
            QTest.keyClick(navigation.destination_buttons[3], Qt.Key_Space)
            self.assertEqual(navigation.workflow_tabs.currentIndex(), 3)
            self.assertEqual(navigation.workflow_tabs.tabText(3), "Slides")
            QTest.keyClick(navigation.destination_buttons[3], Qt.Key_Down)
            QTest.keyClick(navigation.destination_buttons[4], Qt.Key_Return)
            self.assertEqual(navigation.workflow_tabs.currentIndex(), 4)
            self.assertEqual(navigation.workflow_tabs.tabText(4), "Tools")
        finally:
            navigation.deleteLater()
            tabs.deleteLater()

    def test_destination_buttons_preserve_index_selection_and_hidden_legacy_tabs(self):
        navigation = WorkflowNavigation(self.tabs)
        try:
            navigation.destination_buttons[1].click()
            self.assertEqual(navigation.workflow_tabs.currentIndex(), 1)
            self.assertTrue(navigation.destination_buttons[1].isChecked())
            self.assertFalse(navigation.destination_buttons[0].isChecked())
            self.assertEqual(self.tabs.currentIndex(), 0)
            navigation.workflow_tabs.setTabVisible(1, False)
            self.assertTrue(navigation.destination_buttons[1].isHidden())
            self.assertEqual(navigation.workflow_tabs.count(), 3)
            navigation.workflow_tabs.setTabEnabled(2, False)
            self.assertFalse(navigation.destination_buttons[2].isEnabled())
            navigation.workflow_tabs.setTabText(0, "PL")
            self.assertEqual(navigation.destination_buttons[0].text(), "PL")
            self.assertEqual(navigation.destination_buttons[0].accessibleName(), "PL workspace")
        finally:
            navigation.deleteLater()

    def test_rail_keeps_utilities_visible_at_700px_with_larger_text(self):
        from PySide6.QtGui import QFont, QFontDatabase
        from PySide6.QtWidgets import QMainWindow, QWidget
        from ui_qt.shell.workspace import WorkspaceShell

        original_font = self.app.font()
        # Windows' offscreen platform can expose no system fonts. Load one for
        # this geometry check so it measures real glyphs rather than tofu.
        font_id = -1
        if not QFontDatabase.families() and os.path.isfile("C:/Windows/Fonts/segoeui.ttf"):
            font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
            self.app.setFont(QFont("Segoe UI", 9))
        test_font = self.app.font()
        larger_font = QFont(test_font)
        larger_font.setPointSizeF(test_font.pointSizeF() * 1.5)
        self.app.setFont(larger_font)
        window = QMainWindow()
        tabs = QTabWidget()
        for label in ("PL", "DRR", "Compare", "Power", "MCD", "MCD Peak Shift", "SHG", "Slides", "Tools"):
            tabs.addTab(QWidget(), label)
        navigation = WorkflowNavigation(tabs)
        navigation.workflow_tabs.setTabVisible(5, False)
        shell = WorkspaceShell(
            window, navigation=navigation, left_panel=QWidget(), plot_panel=QWidget(),
            presentation_widget=QWidget(), tools_widget=QWidget(),
        )
        try:
            window.resize(1280, 700)
            window.show()
            self.app.processEvents()
            self.assertIs(navigation.sidebar_toggle_btn.parentWidget(), shell.header)
            self.assertTrue(navigation.destination_buttons[8].isVisible())
            self.assertLessEqual(
                navigation.destination_buttons[8].geometry().bottom(), navigation.contentsRect().bottom(),
            )
            self.assertLess(
                navigation.destination_buttons[6].geometry().bottom(), navigation.workflow_separator.geometry().top(),
            )
        finally:
            window.close()
            window.deleteLater()
            tabs.deleteLater()
            self.app.setFont(original_font)
            if font_id >= 0:
                QFontDatabase.removeApplicationFont(font_id)

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
