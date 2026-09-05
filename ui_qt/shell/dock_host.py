"""Presentational construction for the application's analysis docks."""

from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QLabel, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from ui_qt.fluent_ui.style import set_fluent_property


class DockHost:
    """Build the analysis docks without owning their application semantics."""

    def __init__(
        self,
        window: QMainWindow,
        *,
        log_content: QWidget,
        results_pages: Sequence[tuple[str, str, QWidget]],
        results_empty_text: str,
    ) -> None:
        self.log_dock = self._build_log_dock(window, log_content)
        (
            self.results_dock,
            self.results_stack,
            self.results_page_indices,
            self.results_empty_index,
        ) = self._build_results_dock(window, results_pages, results_empty_text)

    @staticmethod
    def _build_log_dock(window: QMainWindow, log_content: QWidget) -> QDockWidget:
        dock = QDockWidget("Run Log", window)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(log_content)
        dock.setWidget(container)
        window.addDockWidget(Qt.BottomDockWidgetArea, dock)
        dock.hide()
        return dock

    @staticmethod
    def _build_results_dock(
        window: QMainWindow,
        results_pages: Sequence[tuple[str, str, QWidget]],
        results_empty_text: str,
    ) -> tuple[QDockWidget, QStackedWidget, dict[str, int], int]:
        dock = QDockWidget("Analysis Results", window)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        stack = QStackedWidget()
        page_indices: dict[str, int] = {}

        for mode, title, widget in results_pages:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(8, 6, 8, 6)
            label = QLabel(title)
            set_fluent_property(label, "appRole", "dockSectionTitle")
            page_layout.addWidget(label)
            widget.setMaximumHeight(16777215)
            page_layout.addWidget(widget, 1)
            page_indices[mode] = stack.addWidget(page)

        empty_page = QLabel(results_empty_text)
        empty_page.setAlignment(Qt.AlignCenter)
        empty_page.setWordWrap(True)
        empty_index = stack.addWidget(empty_page)
        dock.setWidget(stack)
        window.addDockWidget(Qt.BottomDockWidgetArea, dock)
        window.resizeDocks([dock], [190], Qt.Vertical)
        dock.hide()
        return dock, stack, page_indices, empty_index
