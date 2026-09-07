"""Reusable shell for the PL, DRR, and MCD source-picker dialogs.

The shell owns only controls and interaction mechanics that are shared by the
workflows.  Catalog discovery, item rendering, filtering rules, and selection
validation remain in the owning controller.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui_qt.common import WrappedFilenameDelegate


class SourcePickerDialog(QDialog):
    """Common single-source picker chrome and list interaction behavior.

    Controllers populate ``source_list`` themselves so workflow-specific
    rules and presentation stay controller-owned.  ``repopulate`` wraps the
    shared clear/populate/selection-preservation mechanics.
    """

    filter_requested = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        title: str,
        hint: str = "",
        selected: str = "",
        filter_placeholder: str = "Search filename...",
        filter_interval: int = 140,
        filter_controls: Sequence[tuple[str | None, QWidget]] = (),
        minimum_size: tuple[int, int] = (820, 520),
        size: tuple[int, int] = (980, 640),
        selection_mode: QAbstractItemView.SelectionMode = QAbstractItemView.SingleSelection,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        if parent is not None and not parent.windowIcon().isNull():
            self.setWindowIcon(parent.windowIcon())
        self.setMinimumSize(*minimum_size)
        self.resize(*size)
        self._initial_selection = str(selected or "")

        layout = QVBoxLayout(self)
        self.hint_label = QLabel(hint)
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        filter_row = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(filter_placeholder)
        self.filter_row = filter_row
        filter_row.addWidget(QLabel("Find"))
        filter_row.addWidget(self.filter_edit, 1)
        for label, widget in filter_controls:
            if label:
                filter_row.addWidget(QLabel(label))
            filter_row.addWidget(widget)
        self.refresh_button = QPushButton("Refresh")
        filter_row.addWidget(self.refresh_button)
        layout.addLayout(filter_row)

        self.source_list = QListWidget()
        self.configure_source_list(self.source_list, selection_mode=selection_mode)
        layout.addWidget(self.source_list, 1)

        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        self.details_label.setMinimumHeight(42)
        layout.addWidget(self.details_label)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.ok_button = self.button_box.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(max(0, int(filter_interval)))
        self._filter_timer.timeout.connect(self.filter_requested)
        self.filter_edit.textChanged.connect(self._on_filter_text_changed)
        self.source_list.currentItemChanged.connect(self._update_ok_state)
        self.source_list.itemDoubleClicked.connect(lambda _item: self.accept())

    @staticmethod
    def configure_source_list(
        widget: QListWidget,
        *,
        selection_mode: QAbstractItemView.SelectionMode = QAbstractItemView.SingleSelection,
        spacing: int | None = None,
    ) -> QListWidget:
        """Apply the wrapped filename-list setup shared by source pickers."""
        widget.setSelectionMode(selection_mode)
        widget.setWordWrap(True)
        widget.setTextElideMode(Qt.ElideNone)
        widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        widget.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        widget.setResizeMode(QListView.Fixed)
        widget.setUniformItemSizes(False)
        widget.setItemDelegate(WrappedFilenameDelegate(widget))
        if spacing is not None:
            widget.setSpacing(int(spacing))
        return widget

    @staticmethod
    def connect_debounced_filter(
        line_edit: QLineEdit,
        callback: Callable[[], Any],
        parent: QWidget,
        *,
        interval: int = 140,
    ) -> QTimer:
        """Connect a line edit to a single-shot, debounced callback."""
        timer = QTimer(parent)
        timer.setSingleShot(True)
        timer.setInterval(max(0, int(interval)))
        timer.timeout.connect(callback)
        line_edit.textChanged.connect(lambda _text: timer.start())
        return timer

    def _on_filter_text_changed(self, _text: str) -> None:
        if self._filter_timer.interval() == 0:
            self.filter_requested.emit()
        else:
            self._filter_timer.start()

    def _update_ok_state(self, _current: Any = None, _previous: Any = None) -> None:
        self.ok_button.setEnabled(self.source_list.currentItem() is not None)

    def selected_source(self) -> str | None:
        """Return the current item's source role, or its display text."""
        item = self.source_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.UserRole)
        return str(value if value is not None else item.text())

    def restore_selection(self, selected: str = "") -> None:
        """Restore a source selection after a controller repopulates the list."""
        wanted = str(selected or "")
        selected_row = -1
        if wanted:
            for index in range(self.source_list.count()):
                item = self.source_list.item(index)
                value = item.data(Qt.UserRole)
                value = str(value if value is not None else item.text())
                if value == wanted:
                    selected_row = index
                    break
        if selected_row >= 0:
            self.source_list.setCurrentRow(selected_row)
        elif self.source_list.count() == 1:
            self.source_list.setCurrentRow(0)
        else:
            self.source_list.clearSelection()
            self.source_list.setCurrentItem(None)
        self._update_ok_state()

    def repopulate(
        self,
        populate: Callable[[QListWidget], Any],
        *,
        fallback_selection: str | None = None,
    ) -> None:
        """Clear and repopulate the list while preserving its source choice."""
        current = self.selected_source()
        wanted = current or (
            self._initial_selection if fallback_selection is None else fallback_selection
        )
        self.source_list.setUpdatesEnabled(False)
        try:
            self.source_list.clear()
            populate(self.source_list)
        finally:
            self.source_list.setUpdatesEnabled(True)
        self.restore_selection(wanted)

    def set_details(self, text: str) -> None:
        self.details_label.setText(text)
