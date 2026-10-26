"""A window-owned divider for the linked PL/DRR map and spectrum axes."""
from __future__ import annotations

import weakref

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget, QToolButton

from ui_qt.theme import alias


class InspectionPlotDivider(QWidget):
    """Move existing subplot geometry without rebuilding scientific artists."""

    DEFAULT_FRACTION = .70
    GAP = .20

    def __init__(self, owner):
        super().__init__(owner.canvas)
        self.setObjectName("inspectionPlotDivider")
        self.setAccessibleName("Heatmap and spectrum divider")
        self.setAccessibleDescription(
            "Drag up or down to resize the heatmap and spectrum. Arrow keys adjust the split; "
            "Home or double-click restores seventy percent for the heatmap."
        )
        self.setCursor(Qt.SizeVerCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self._owner = weakref.ref(owner)
        self._mode = ""
        self._grids = ()
        self._maps = ()
        self._spectra = ()
        self._axes = ()
        self._outer_grid = None
        self._base_bottom = .13
        self._fractions = {"PL": self.DEFAULT_FRACTION, "DRR": self.DEFAULT_FRACTION}
        self._collapsed = {"PL": False, "DRR": False}
        self._dragging = False
        self._hovered = False
        self._applying = False
        self.toggle_button = QToolButton(self)
        self.toggle_button.setObjectName("inspectionSpectrumToggle")
        self.toggle_button.setProperty("fluentAppearance", "subtle")
        self.toggle_button.setProperty("fluentSize", "compact")
        font = self.toggle_button.font()
        font.setPointSize(8)
        self.toggle_button.setFont(font)
        self.toggle_button.setCursor(Qt.ArrowCursor)
        self.toggle_button.clicked.connect(self.toggle_spectrum)
        self.toggle_button.setFocusPolicy(Qt.StrongFocus)
        owner.canvas.mpl_connect("resize_event", self._on_resize)
        owner.canvas.mpl_connect("draw_event", self._on_draw)
        tabs = getattr(owner, "tabs", None)
        if tabs is not None:
            tabs.currentChanged.connect(self._on_mode_changed)
        self.hide()

    @property
    def heatmap_fraction(self):
        return self._fractions.get(self._mode, self.DEFAULT_FRACTION)

    @property
    def spectrum_collapsed(self):
        return self._collapsed.get(self._mode, False)

    def configure(self, mode, grids, maps, spectra, outer_grid):
        self._mode = mode
        self._grids = tuple(grids)
        self._maps = tuple(maps)
        self._spectra = tuple(spectra)
        self._outer_grid = outer_grid
        self._base_bottom = float(outer_grid.get_subplot_params(self.parent().figure).bottom)
        self._axes = tuple(self.parent().figure.axes)
        self._apply_layout(redraw=False)

    def _owns_axes(self):
        figure = self.parent().figure
        return (bool(self._maps and self._spectra)
                and all(axis in figure.axes for axis in (*self._maps, *self._spectra)))

    def _mode_is_active(self):
        owner = self._owner()
        tabs = getattr(owner, "tabs", None)
        return tabs is None or tabs.tabText(tabs.currentIndex()) == self._mode

    def _invalidate_blits(self):
        owner = self._owner()
        if owner is None:
            return
        for helper in (getattr(owner, "_drr_region_blitters", None) or {}).values():
            helper.invalidate()
        controller = getattr(owner, "pl_controller", None)
        for helper in getattr(controller, "_pl_region_blitters", None) or ():
            helper.invalidate()

    def _apply_layout(self, *, redraw=True):
        if self._applying or not self._owns_axes():
            self.hide()
            return
        self._applying = True
        try:
            canvas = self.parent()
            collapsed = self.spectrum_collapsed
            fraction = self.heatmap_fraction
            # Reserve a separate bottom strip for expansion, below the map's
            # restored energy labels, rather than covering measured data.
            bottom = self._base_bottom + (24 / max(1, canvas.height()) if collapsed else 0)
            geometry = {"bottom": bottom}
            header_height = getattr(self._outer_grid, "_dptk_inspection_header_height", None)
            if header_height is not None:
                geometry["top"] = 1 - header_height * canvas.figure.dpi / 100 / max(1, canvas.figure.bbox.height)
            self._outer_grid.update(**geometry)
            for grid in self._grids:
                grid.set_height_ratios([1, 0] if collapsed else [fraction, 1 - fraction])
            for axis in self._axes:
                if axis not in canvas.figure.axes:
                    continue
                spec = axis.get_subplotspec()
                if spec is not None:
                    axis.set_position(spec.get_position(canvas.figure))
            if collapsed:
                map_bottom = min(axis.get_position().y0 for axis in self._maps)
                for axis in self._axes:
                    position = axis.get_position()
                    if (axis in canvas.figure.axes and not getattr(axis, "_dptk_inspection_header", False)
                            and position.y0 >= map_bottom - 1e-6):
                        axis.set_position([position.x0, bottom, position.width, position.y1 - bottom])
            xlabel = self._spectra[0].get_xlabel() or "Photon Energy (eV)"
            for axis in self._spectra:
                axis.set_visible(not collapsed)
            for axis in self._maps:
                axis.set_xlabel(xlabel if collapsed else "")
                axis.tick_params(axis="x", labelbottom=collapsed)
            self._invalidate_blits()
            self._position_handle()
            if redraw:
                canvas.draw_idle()
        finally:
            self._applying = False

    def set_heatmap_fraction(self, fraction):
        # Small spectra remain readable; collapsing is the explicit way to
        # devote the full scientific canvas to the heatmap.
        fraction = max(.30, min(.85, float(fraction)))
        self._fractions[self._mode] = fraction
        if self.spectrum_collapsed:
            self._collapsed[self._mode] = False
        self._apply_layout()

    def set_spectrum_collapsed(self, collapsed):
        self._collapsed[self._mode] = bool(collapsed)
        self._apply_layout()

    def toggle_spectrum(self):
        self.set_spectrum_collapsed(not self.spectrum_collapsed)

    def _position_handle(self):
        if not self._owns_axes() or not self._mode_is_active():
            self.hide()
            return
        canvas = self.parent()
        left = min(axis.get_position().x0 for axis in self._maps)
        right = max(axis.get_position().x1 for axis in (*self._maps, *self._spectra))
        if self.spectrum_collapsed:
            top, height = canvas.height() - 24, 22
        else:
            map_bottom = (1 - min(axis.get_position().y0 for axis in self._maps)) * canvas.height()
            spectrum_top = (1 - max(axis.get_position().y1 for axis in self._spectra)) * canvas.height()
            height = max(16, min(22, round(spectrum_top - map_bottom - 2)))
            top = round((map_bottom + spectrum_top - height) / 2)
        self.setGeometry(round(left * canvas.width()), round(top),
                         max(120, round((right - left) * canvas.width())), height)
        text = "Show spectrum" if self.spectrum_collapsed else "Hide spectrum"
        self.toggle_button.setText(text)
        self.toggle_button.setAccessibleName(text)
        self.toggle_button.setToolTip("Restore the previous split" if self.spectrum_collapsed else "Give the heatmap the full plot area")
        width = self.toggle_button.fontMetrics().horizontalAdvance(text) + 18
        self.toggle_button.setGeometry(max(0, self.width() - width), 0, width, self.height())
        self.setToolTip("Drag to resize · Double-click for 70% heatmap")
        self.show()
        self.raise_()
        self.update()

    def _on_resize(self, _event):
        self._apply_layout(redraw=False)

    def _on_draw(self, _event):
        # Geometry changes happen before rendering. A draw callback only
        # tracks the overlay and never moves axes or recursively draws.
        self._position_handle()

    def _on_mode_changed(self, *_args):
        self._dragging = False
        self._position_handle()

    def _drag_to(self, point):
        canvas = self.parent()
        top = (1 - max(axis.get_position().y1 for axis in self._maps)) * canvas.height()
        bottom = (1 - min(axis.get_position().y0 for axis in self._spectra)) * canvas.height()
        gap = max(0, ((1 - max(axis.get_position().y1 for axis in self._spectra))
                      - (1 - min(axis.get_position().y0 for axis in self._maps))) * canvas.height())
        self.set_heatmap_fraction((point.y() - top - gap / 2) / max(1, bottom - top - gap))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.spectrum_collapsed:
            self._dragging = True
            self.setFocus(Qt.MouseFocusReason)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._drag_to(self.parent().mapFromGlobal(event.globalPosition().toPoint()))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.LeftButton:
            self._drag_to(self.parent().mapFromGlobal(event.globalPosition().toPoint()))
            self._dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.set_heatmap_fraction(self.DEFAULT_FRACTION)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Up, Qt.Key_Down):
            self.set_heatmap_fraction(self.heatmap_fraction + (.03 if event.key() == Qt.Key_Down else -.03))
        elif event.key() == Qt.Key_Home:
            self.set_heatmap_fraction(self.DEFAULT_FRACTION)
        elif event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self.toggle_spectrum()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def focusInEvent(self, event):
        self.update()
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        self.update()
        super().focusOutEvent(event)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        background = self.parent().figure.get_facecolor()
        painter.fillRect(self.rect(), QColor.fromRgbF(*background))
        ink = QColor(alias("brand_foreground" if self._hovered or self.hasFocus() else "border_primary"))
        painter.setPen(QPen(ink, 1))
        right = self.toggle_button.x() - 8
        center = max(12, right / 2)
        y = self.height() / 2
        painter.drawLine(0, round(y), max(0, right), round(y))
        painter.setBrush(QColor.fromRgbF(*background))
        painter.drawRoundedRect(QRectF(center - 18, y - 3, 36, 6), 3, 3)
        if self.hasFocus():
            painter.setPen(QPen(QColor(alias("focus_border")), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(QRectF(1, 1, max(0, right - 2), self.height() - 2), 3, 3)


def bind_inspection_splitter(owner, mode, *, grids, maps, spectra, outer_grid):
    divider = getattr(owner, "_inspection_plot_divider", None)
    if divider is None:
        divider = InspectionPlotDivider(owner)
        owner._inspection_plot_divider = divider
    divider.configure(mode, grids, maps, spectra, outer_grid)
    return divider
