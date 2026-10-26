"""Explicit shared-symbol boundary for the DPTK Qt UI layer.

This module owns symbols shared across the application shell, feature pages,
and workflow controllers so they do not depend on ``ui_qt.main_window`` as a
generic symbol provider. Definitions moved here are deliberately kept identical
to their historical definitions; ``ui_qt.main_window`` re-exports them for
compatibility with existing tests and callers.
"""

from __future__ import annotations

import traceback
from core.load_cancellation import LoadCancelled, check_load_cancelled, load_scope
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from collections import OrderedDict
from math import ceil
from typing import Any, Callable, Dict, List

from PySide6.QtCore import QObject, QPointF, QRect, QRunnable, QSize, Qt, Signal, qVersion
from PySide6.QtGui import QTextLayout, QTextOption
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QComboBox as _QComboBox,
    QDoubleSpinBox as _QDoubleSpinBox,
    QListWidget,
    QSpinBox as _QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)

from core.application_models import ExportOptions, LoadOptions, LoadedState

from core.loader import DataCube
from core.data_io import PowerSeriesSource
from core.mcd import McdCenterCandidate, McdResult, McdSettings
from core.plotting import HeatmapParams
from core.provenance import WorkingCopyRecord
from core.shg import ShgProcessResult, ShgSettings, ShgSweepData
from core.shg_fit import ShgAngularFitResult, ShgFitSettings, ShgTwistFitResult

UI_METRICS = {
    "left_width": 320,
    "sidebar_min_width": 320,
    "sidebar_max_width": 560,
    "main_margin": 8,
    "group_margin": 10,
    "row_spacing": 8,
    "label_col_width": 86,
    "input_h": 30,
    "spin_w": 88,
    "short_combo_w": 145,
    "deriv_combo_w": 90,
    "long_combo_min_w": 200,
    "tool_h": 30,
    "tool_w": 62,
}


class QDoubleSpinBox(_QDoubleSpinBox):
    """Commit typed numbers on Return/focus loss and ignore wheel scrolling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setKeyboardTracking(False)

    def wheelEvent(self, event) -> None:
        event.ignore()


class QSpinBox(_QSpinBox):
    """Commit typed numbers on Return/focus loss and ignore wheel scrolling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setKeyboardTracking(False)

    def wheelEvent(self, event) -> None:
        event.ignore()


class QComboBox(_QComboBox):
    """A combo box that leaves the mouse wheel to its enclosing scroll area."""

    def wheelEvent(self, event) -> None:
        event.ignore()


def focused_numeric_input(parent: QWidget) -> QAbstractSpinBox | None:
    """Find the focused spin box, including focus on its child line editor."""
    focused = QApplication.focusWidget()
    while focused is not None and isinstance(parent, QWidget) and parent.isAncestorOf(focused):
        if isinstance(focused, QAbstractSpinBox):
            return focused
        focused = focused.parentWidget()
    return None


def pending_numeric_input(parent: QWidget) -> QAbstractSpinBox | None:
    """Find an uncommitted number in either the Widgets or Quick display view.

    The Quick adapter returns its compatibility control so existing display
    restoration can identify the same field without knowing about QML items.
    """
    focused = focused_numeric_input(parent)
    if focused is not None and focused.lineEdit().isModified():
        return focused
    host = getattr(parent, "drr_display_host", None)
    bridge = getattr(host, "bridge", None)
    if bridge is not None and bridge.editing:
        return bridge.editingWidget()
    return None


class WorkerSignals(QObject):
    progress = Signal(int)
    log = Signal(str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class Worker(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args, cancellation=None, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.cancellation = cancellation
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            with load_scope(self.cancellation):
                result = self.fn(*self.args, progress=self.signals.progress, log=self.signals.log, **self.kwargs)
                check_load_cancelled()
                self.signals.result.emit(result)
        except LoadCancelled:
            pass
        except Exception as exc:
            self.signals.error.emit(f"{exc}\n\n{traceback.format_exc()}")
        finally:
            self.signals.finished.emit()


class WrappedFilenameDelegate(QStyledItemDelegate):
    """Paint complete text as dynamically sized, wrapped list rows."""

    HORIZONTAL_PADDING = 8
    VERTICAL_PADDING = 6

    def __init__(self, parent=None, *, cache_path=None,
                 wrap_mode=QTextOption.WrapAnywhere) -> None:
        super().__init__(parent)
        self._wrap_mode = wrap_mode
        # Wrapped rows are queried repeatedly while QListView lays out and
        # repaints.  Bounding-rect calculation is surprisingly expensive for
        # long paths, so keep the result for each viewport width/text/font pair.
        self._size_hint_cache = OrderedDict()
        self._cache_path = Path(cache_path) if cache_path else None
        self._saved_sizes = OrderedDict()
        self._sizes_dirty = False
        if self._cache_path:
            try:
                payload = json.loads(self._cache_path.read_text(encoding='utf-8'))
                if payload.get('version') == 1 and isinstance(payload.get('sizes'), dict):
                    for key, size in list(payload['sizes'].items())[-8192:]:
                        if (isinstance(key, str) and isinstance(size, list) and len(size) == 2
                                and all(type(value) is int and 0 < value < 100000 for value in size)):
                            self._saved_sizes[key] = size
            except (OSError, ValueError, TypeError, AttributeError):
                pass

    def save_size_cache(self) -> None:
        if not self._cache_path or not self._sizes_dirty:
            return
        temporary = self._cache_path.with_name(f'.row-heights-{uuid4().hex}.tmp')
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps({'version': 1, 'sizes': self._saved_sizes}), encoding='utf-8')
            temporary.replace(self._cache_path)
            self._sizes_dirty = False
        except OSError:
            pass
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _text_width(self, option: QStyleOptionViewItem) -> int:
        view = self.parent()
        if isinstance(view, QListWidget):
            # QListView reduces the paint rect by both spacing margins.  Use
            # that same width for wrapping or a resize can add a line after
            # sizeHint() has already fixed the row height.
            width = view.viewport().width() - 2 * view.spacing()
        else:
            width = option.rect.width()
        return max(80, int(width) - 2 * self.HORIZONTAL_PADDING)

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text = self._normalize_text(index.data(Qt.DisplayRole))
        view = self.parent()
        cache_key = (self._text_width(opt), text, opt.font.toString(),
                     opt.fontMetrics.height(), opt.fontMetrics.averageCharWidth(),
                     view.logicalDpiX(), view.logicalDpiY(), view.devicePixelRatioF(),
                     view.style().metaObject().className(), qVersion(),
                     self.HORIZONTAL_PADDING, self.VERTICAL_PADDING, self._wrap_mode.value)
        cached = self._size_hint_cache.get(cache_key)
        if cached is not None:
            self._size_hint_cache.move_to_end(cache_key)
            return QSize(cached)
        disk_key = hashlib.sha256(repr(cache_key).encode('utf-8')).hexdigest() if self._cache_path else None
        saved = self._saved_sizes.get(disk_key)
        if saved is not None:
            result = QSize(*saved)
            if len(self._size_hint_cache) >= 8192:
                self._size_hint_cache.popitem(last=False)
            self._size_hint_cache[cache_key] = result
            return QSize(result)
        text_layout = self._layout_text(text, opt.font, self._text_width(opt))
        metric_height = opt.fontMetrics.boundingRect(
            QRect(0, 0, self._text_width(opt), 10000),
            Qt.AlignLeft | Qt.TextWrapAnywhere, text,
        ).height()
        base = super().sizeHint(opt, index)
        view = self.parent()
        row_width = (
            view.viewport().width()
            if isinstance(view, QListWidget)
            else self._text_width(opt) + 2 * self.HORIZONTAL_PADDING
        )
        result = QSize(
            row_width,
            max(base.height(), max(metric_height, ceil(text_layout.boundingRect().height()))
                + 2 * self.VERTICAL_PADDING),
        )
        # Keep this bounded in practice; a list normally has only a handful
        # of viewport widths, but a dialog can be resized many times.
        if len(self._size_hint_cache) >= 8192:
            self._size_hint_cache.popitem(last=False)
        self._size_hint_cache[cache_key] = QSize(result)
        if disk_key is not None:
            if len(self._saved_sizes) >= 8192:
                self._saved_sizes.popitem(last=False)
            self._saved_sizes[disk_key] = [result.width(), result.height()]
            self._sizes_dirty = True
        return result

    @staticmethod
    def _normalize_text(value) -> str:
        # QTextLayout treats a literal newline as a soft break on Windows,
        # while Qt's item-style conversion uses U+2028 for a forced row break.
        # Canonicalize both inputs to U+2028 so status text always stays on
        # its own row and sizeHint() measures exactly what paint() draws.
        return (
            str(value or "")
            .replace("\r\n", "\u2028")
            .replace("\r", "\u2028")
            .replace("\n", "\u2028")
            .replace("\u2029", "\u2028")
        )

    def _layout_text(self, text, font, width):
        # Use precisely the same line breaking for measurement and painting.
        # QFontMetrics.boundingRect and QPainter.drawText can disagree at a
        # wrap boundary on Windows, leaving a two-line filename in one row.
        layout = QTextLayout(text, font)
        text_option = QTextOption()
        text_option.setWrapMode(self._wrap_mode)
        layout.setTextOption(text_option)
        layout.beginLayout()
        height = 0.0
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(max(1, width))
            line.setPosition(QPointF(0, height))
            height += line.height()
        layout.endLayout()
        return layout

    def paint(self, painter, option: QStyleOptionViewItem, index) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text = self._normalize_text(opt.text)
        opt.text = ""
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)

        painter.save()
        painter.setFont(opt.font)
        painter.setPen(
            opt.palette.highlightedText().color()
            if opt.state & QStyle.State_Selected
            else opt.palette.text().color()
        )
        text_rect = option.rect.adjusted(
            self.HORIZONTAL_PADDING,
            self.VERTICAL_PADDING,
            -self.HORIZONTAL_PADDING,
            -self.VERTICAL_PADDING,
        )
        text_layout = self._layout_text(text, opt.font, text_rect.width())
        text_layout.draw(painter, QPointF(text_rect.left(), text_rect.top()))
        painter.restore()
