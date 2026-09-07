"""Bounded two-mode layout for dense form rows.

The label always sits above the controls. Controls use one row when their
rendered font/style widths fit; otherwise the trailing action moves to row 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PySide6.QtCore import QEvent, QRect, QSize
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QLabel,
    QLayout,
    QLayoutItem,
    QSizePolicy,
    QStyle,
    QStyleOptionButton,
    QStyleOptionComboBox,
    QStyleOptionSpinBox,
    QStyleOptionToolButton,
    QToolButton,
    QWidget,
    QWidgetItem,
)


@dataclass
class _Group:
    items: list[QLayoutItem]
    role: str
    priority: int
    grow_weight: int


class DenseFormRowLayout(QLayout):
    """Height-for-width layout with ``SINGLE_ROW`` and ``WRAPPED`` modes."""

    SINGLE_ROW = "SINGLE_ROW"
    WRAPPED = "WRAPPED"

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        spacing: int | None = None,
        label: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self._groups: list[_Group] = []
        self._items: list[QLayoutItem] = []
        self._spacing_override = spacing
        self._last_mode = self.SINGLE_ROW
        self._label_widget = label
        self._label_item = QWidgetItem(label) if label is not None else None
        if self._label_item is not None:
            self._items.append(self._label_item)
        if parent is not None:
            parent.installEventFilter(self)
        if label is not None:
            label.setParent(parent)
            label.installEventFilter(self)

    def set_label_widget(self, label: QWidget | None) -> None:
        if label is self._label_widget:
            return
        old = self._label_widget
        if old is not None:
            old.removeEventFilter(self)
            old.hide()
            if self._label_item in self._items:
                self._items.remove(self._label_item)
        self._label_widget = label
        self._label_item = None
        if label is not None:
            parent = self.parentWidget()
            if parent is not None:
                label.setParent(parent)
                if parent.isVisible():
                    label.show()
            label.installEventFilter(self)
            self._label_item = QWidgetItem(label)
            self._items.insert(0, self._label_item)
        self.invalidate()

    def labelWidget(self) -> QWidget | None:  # noqa: N802 - Qt API
        return self._label_widget

    def add_group(
        self,
        widgets: Iterable[QWidget],
        *,
        role: str = "control",
        priority: int = 0,
        grow_weight: int = 1,
    ) -> None:
        items = [QWidgetItem(widget) for widget in widgets]
        if not items:
            return
        group = _Group(items, str(role), int(priority), max(0, int(grow_weight)))
        self._groups.append(group)
        self._items.extend(items)
        for item in items:
            widget = item.widget()
            widget.setParent(self.parentWidget())
            widget.installEventFilter(self)
        self.invalidate()

    def addWidget(self, widget: QWidget) -> None:  # noqa: N802 - Qt API
        self.add_group((widget,))

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API
        if event.type() in {
            QEvent.Resize,
            QEvent.StyleChange,
            QEvent.FontChange,
            QEvent.PaletteChange,
            QEvent.ApplicationFontChange,
            QEvent.LayoutRequest,
            QEvent.DevicePixelRatioChange,
        }:
            self.invalidate()
            parent = self.parentWidget()
            if parent is not None:
                parent.updateGeometry()
        return super().eventFilter(watched, event)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt API
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt API
        if not 0 <= index < len(self._items):
            return None
        item = self._items.pop(index)
        if item is self._label_item:
            widget = item.widget()
            if widget is not None:
                widget.removeEventFilter(self)
                widget.hide()
            self._label_item = None
            self._label_widget = None
            self.invalidate()
        for group in self._groups:
            if item in group.items:
                group.items.remove(item)
                if not group.items:
                    self._groups.remove(group)
                break
        return item

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API
        return True

    def spacing(self) -> int:  # noqa: N802 - Qt API
        if self._spacing_override is not None:
            return max(0, int(self._spacing_override))
        value = super().spacing()
        if value >= 0:
            return value
        parent = self.parentWidget()
        if parent is None:
            return 4
        return max(0, parent.style().pixelMetric(QStyle.PM_LayoutHorizontalSpacing, None, parent))

    @staticmethod
    def _probe_width(widget: QWidget) -> int:
        return max(256, widget.sizeHint().width(), widget.minimumWidth(), 1)

    def _spin_width(self, widget: QAbstractSpinBox) -> int:
        line_edit = widget.lineEdit()
        if line_edit is None:
            return max(1, widget.minimumWidth())

        values = [widget.text()]
        minimum = getattr(widget, "minimum", lambda: 0)()
        maximum = getattr(widget, "maximum", lambda: 0)()
        for representative in (0, -12):
            if minimum <= representative <= maximum:
                values.append(widget.textFromValue(representative))
        text_width = max(QFontMetrics(line_edit.font()).horizontalAdvance(value) for value in values)
        text_margins = line_edit.textMargins()

        option = QStyleOptionSpinBox()
        widget.initStyleOption(option)
        probe_width = self._probe_width(widget)
        probe_height = max(widget.sizeHint().height(), line_edit.fontMetrics().height())
        option.rect = QRect(0, 0, probe_width, probe_height)
        style = widget.style()
        edit_rect = style.subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxEditField, widget)
        up_rect = style.subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxUp, widget)
        down_rect = style.subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxDown, widget)
        stepper_width = max(up_rect.width(), down_rect.width())
        frame_width = style.pixelMetric(QStyle.PM_SpinBoxFrameWidth, option, widget)
        measured_chrome = max(0, probe_width - edit_rect.width())
        required_chrome = max(measured_chrome, stepper_width + frame_width * 2)
        return max(
            1,
            widget.minimumWidth(),
            text_width + text_margins.left() + text_margins.right() + required_chrome,
        )

    def _checkbox_width(self, widget: QCheckBox) -> int:
        option = QStyleOptionButton()
        widget.initStyleOption(option)
        style = widget.style()
        text_width = QFontMetrics(widget.font()).horizontalAdvance(widget.text())
        indicator_width = style.pixelMetric(QStyle.PM_IndicatorWidth, option, widget)
        label_spacing = style.pixelMetric(QStyle.PM_CheckBoxLabelSpacing, option, widget)
        return max(1, widget.minimumWidth(), text_width + indicator_width + label_spacing)

    def _button_width(self, widget: QAbstractButton) -> int:
        if isinstance(widget, QToolButton):
            option = QStyleOptionToolButton()
            widget.initStyleOption(option)
            metrics = QFontMetrics(widget.font())
            text_size = QSize(metrics.horizontalAdvance(widget.text()), metrics.height())
            styled = widget.style().sizeFromContents(QStyle.CT_ToolButton, option, text_size, widget)
            return max(1, widget.minimumWidth(), styled.width())
        option = QStyleOptionButton()
        widget.initStyleOption(option)
        probe_width = self._probe_width(widget)
        probe_height = max(widget.sizeHint().height(), QFontMetrics(widget.font()).height())
        option.rect = QRect(0, 0, probe_width, probe_height)
        contents = widget.style().subElementRect(QStyle.SE_PushButtonContents, option, widget)
        style_padding = max(0, probe_width - contents.width())
        text_width = QFontMetrics(widget.font()).horizontalAdvance(widget.text())
        return max(1, widget.minimumWidth(), text_width + style_padding)

    def _combo_width(self, widget: QComboBox) -> int:
        option = QStyleOptionComboBox()
        widget.initStyleOption(option)
        probe_width = self._probe_width(widget)
        option.rect = QRect(0, 0, probe_width, max(1, widget.sizeHint().height()))
        edit_rect = widget.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, widget)
        chrome = max(0, probe_width - edit_rect.width())
        text_width = QFontMetrics(widget.font()).horizontalAdvance(widget.currentText())
        return max(1, widget.minimumWidth(), text_width + chrome)

    def _widget_min_width(self, widget: QWidget) -> int:
        if isinstance(widget, QAbstractSpinBox):
            return self._spin_width(widget)
        if isinstance(widget, QCheckBox):
            return self._checkbox_width(widget)
        if isinstance(widget, QAbstractButton):
            return self._button_width(widget)
        if isinstance(widget, QComboBox):
            return self._combo_width(widget)
        if isinstance(widget, QLabel):
            margins = widget.contentsMargins()
            return max(
                1,
                widget.minimumWidth(),
                QFontMetrics(widget.font()).horizontalAdvance(widget.text()) + margins.left() + margins.right(),
            )
        return max(1, widget.minimumWidth(), widget.sizeHint().width())

    safe_min_width = _widget_min_width

    def _control_groups(self) -> list[_Group]:
        return [group for group in self._groups if group.items]

    def _row_width(self, items: list[QLayoutItem]) -> int:
        return sum(self._widget_min_width(item.widget()) for item in items) + self.spacing() * max(0, len(items) - 1)

    def _rows_for_width(self, width: int) -> tuple[list[list[QLayoutItem]], str]:
        groups = self._control_groups()
        items = [item for group in groups for item in group.items]
        if not items:
            return [], self.SINGLE_ROW

        left, _top, right, _bottom = self.getContentsMargins()
        available = max(0, int(width) - left - right)
        if self._row_width(items) <= available:
            return [items], self.SINGLE_ROW

        action_index = next((index for index, group in enumerate(groups) if group.role == "action"), len(groups) - 1)
        first_row = [item for group in groups[:action_index] for item in group.items]
        second_row = [item for group in groups[action_index:] for item in group.items]
        if not first_row:
            first_row, second_row = items[:-1], items[-1:]
        return [row for row in (first_row, second_row) if row], self.WRAPPED

    def mode_for_width(self, width: int) -> str:
        return self._rows_for_width(max(0, int(width)))[1]

    @staticmethod
    def _item_height(item: QLayoutItem) -> int:
        widget = item.widget()
        return max(item.sizeHint().height(), widget.minimumHeight())

    def _row_height(self, row: list[QLayoutItem]) -> int:
        return max((self._item_height(item) for item in row), default=0)

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API
        rows, mode = self._rows_for_width(max(0, int(width)))
        self._last_mode = mode
        _left, top, _right, bottom = self.getContentsMargins()
        gap = self.spacing()
        total = top + bottom + sum(self._row_height(row) for row in rows)
        total += gap * max(0, len(rows) - 1)
        if self._label_widget is not None:
            total += max(self._label_widget.sizeHint().height(), self._label_widget.minimumHeight()) + gap
        return total

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        items = [item for group in self._control_groups() for item in group.items]
        left, _top, right, _bottom = self.getContentsMargins()
        width = self._row_width(items) + left + right
        return QSize(width, self.heightForWidth(width))

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt API
        groups = self._control_groups()
        action_index = next((index for index, group in enumerate(groups) if group.role == "action"), len(groups))
        primary = [item for group in groups[:action_index] for item in group.items]
        if not primary:
            primary = [item for group in groups for item in group.items]
        left, _top, right, _bottom = self.getContentsMargins()
        width = self._row_width(primary) + left + right
        return QSize(width, self.heightForWidth(width))

    def _set_row_geometry(self, row: list[QLayoutItem], rect: QRect, y: int) -> int:
        gap = self.spacing()
        row_height = self._row_height(row)
        baseline = self._row_width(row)
        extra = max(0, rect.width() - baseline)
        expandable = [
            item
            for item in row
            if item.widget().sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
            and not isinstance(item.widget(), (QCheckBox, QAbstractButton))
        ]
        x = rect.left()
        remaining_extra = extra
        remaining_expandable = len(expandable)
        for item in row:
            widget = item.widget()
            width = self._widget_min_width(widget)
            if item in expandable:
                share = remaining_extra // max(1, remaining_expandable)
                added = min(share, max(0, widget.maximumWidth() - width))
                width += added
                remaining_extra -= added
                remaining_expandable -= 1
            height = self._item_height(item)
            item.setGeometry(QRect(x, y + (row_height - height) // 2, max(1, width), height))
            x += width + gap
        return row_height

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt API
        super().setGeometry(rect)
        rows, mode = self._rows_for_width(rect.width())
        self._last_mode = mode
        left, top, right, bottom = self.getContentsMargins()
        content = rect.adjusted(left, top, -right, -bottom)
        gap = self.spacing()
        y = content.top()
        if self._label_widget is not None:
            label_height = max(self._label_widget.sizeHint().height(), self._label_widget.minimumHeight())
            label_width = min(content.width(), self._widget_min_width(self._label_widget))
            self._label_widget.setGeometry(QRect(content.left(), y, max(1, label_width), label_height))
            y += label_height + gap
        for row in rows:
            y += self._set_row_geometry(row, content, y) + gap
