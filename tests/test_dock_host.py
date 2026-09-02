import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QDockWidget, QMainWindow, QStackedWidget, QTextEdit, QWidget

from ui_qt.shell.dock_host import DockHost


class _RecordingMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize_docks_calls = []

    def resizeDocks(self, docks, sizes, orientation):
        self.resize_docks_calls.append((list(docks), list(sizes), orientation))


class DockHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self):
        for widget in getattr(self, "_widgets", ()):
            widget.deleteLater()

    def _host(self):
        window = _RecordingMainWindow()
        log_content = QTextEdit()
        pl = QTextEdit()
        drr = QTextEdit()
        mcd = QTextEdit()
        host = DockHost(
            window,
            log_content=log_content,
            results_pages=(
                ("PL", "PL peak and fit results", pl),
                ("DRR", "DRR peak and fit results", drr),
                ("MCD", "MCD pair diagnostics", mcd),
            ),
            results_empty_text="This workflow has no separate text results. Use the plot and export controls.",
        )
        self._widgets = (window, log_content, pl, drr, mcd)
        return window, host, log_content, (pl, drr, mcd)

    def test_constructs_hidden_run_log_in_bottom_area(self):
        window, host, log_content, _ = self._host()

        self.assertEqual(host.log_dock.windowTitle(), "Run Log")
        self.assertEqual(host.log_dock.allowedAreas(), Qt.BottomDockWidgetArea)
        self.assertEqual(window.dockWidgetArea(host.log_dock), Qt.BottomDockWidgetArea)
        self.assertTrue(host.log_dock.isHidden())
        self.assertFalse(host.log_dock.isFloating())
        self.assertIsNotNone(host.log_dock.widget())
        self.assertIs(log_content.parentWidget(), host.log_dock.widget())

    def test_constructs_results_pages_and_empty_page(self):
        window, host, _, pages = self._host()

        self.assertEqual(host.results_dock.windowTitle(), "Analysis Results")
        self.assertEqual(
            host.results_dock.allowedAreas(),
            Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea,
        )
        self.assertEqual(window.dockWidgetArea(host.results_dock), Qt.BottomDockWidgetArea)
        self.assertTrue(host.results_dock.isHidden())
        self.assertFalse(host.results_dock.isFloating())
        self.assertIsInstance(host.results_stack, QStackedWidget)
        self.assertIs(host.results_dock.widget(), host.results_stack)
        self.assertEqual(host.results_stack.count(), 4)
        self.assertEqual(host.results_page_indices, {"PL": 0, "DRR": 1, "MCD": 2})
        self.assertEqual(host.results_empty_index, 3)

        expected_titles = ["PL peak and fit results", "DRR peak and fit results", "MCD pair diagnostics"]
        for index, (content, title) in enumerate(zip(pages, expected_titles)):
            page = host.results_stack.widget(index)
            self.assertEqual(page.layout().contentsMargins().left(), 8)
            self.assertEqual(page.layout().contentsMargins().top(), 6)
            self.assertEqual(page.layout().contentsMargins().right(), 8)
            self.assertEqual(page.layout().contentsMargins().bottom(), 6)
            title_label = page.layout().itemAt(0).widget()
            self.assertIsInstance(title_label, QLabel)
            self.assertEqual(title_label.text(), title)
            self.assertEqual(title_label.property("appRole"), "dockSectionTitle")
            self.assertIs(page.layout().itemAt(1).widget(), content)
            self.assertEqual(content.maximumHeight(), 16777215)

        empty_page = host.results_stack.widget(host.results_empty_index)
        self.assertIsInstance(empty_page, QLabel)
        self.assertEqual(
            empty_page.text(),
            "This workflow has no separate text results. Use the plot and export controls.",
        )
        self.assertTrue(empty_page.wordWrap())
        self.assertEqual(empty_page.alignment(), Qt.AlignCenter)

    def test_resizes_results_dock_once_and_preserves_default_features(self):
        window, host, _, _ = self._host()

        self.assertEqual(len(window.resize_docks_calls), 1)
        docks, sizes, orientation = window.resize_docks_calls[0]
        self.assertEqual(docks, [host.results_dock])
        self.assertEqual(sizes, [190])
        self.assertEqual(orientation, Qt.Vertical)
        default_dock = QDockWidget()
        try:
            self.assertEqual(host.results_dock.features(), default_dock.features())
        finally:
            default_dock.deleteLater()

    def test_main_window_composes_dock_host_with_compatibility_aliases(self):
        from ui_qt.main_window import MainWindow

        window = MainWindow()
        try:
            self.assertIsInstance(window.dock_host, DockHost)
            self.assertIs(window.log_dock, window.dock_host.log_dock)
            self.assertIs(window.results_dock, window.dock_host.results_dock)
            self.assertIs(window.results_stack, window.dock_host.results_stack)
            self.assertIs(window._results_page_indices, window.dock_host.results_page_indices)
            self.assertEqual(window._results_empty_index, window.dock_host.results_empty_index)
            self.assertIsInstance(window.log_text, QTextEdit)
            self.assertTrue(window.log_text.isReadOnly())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
