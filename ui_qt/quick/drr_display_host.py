"""Lazy Qt Quick display page, with an operational Widgets fallback.

Only this host owns QML objects. Controllers continue to consume their existing
controls through DrrDisplayBridge while their state boundary is migrated.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QTimer, QUrl, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget

from ui_qt.theme import alias, theme_manager


def display_colors() -> dict[str, str]:
    names = {
        "surface": "sidebar_surface", "text": "heading_text",
        "muted": "section_text", "border": "border_primary",
        "accent": "brand_background", "control": "control_surface",
        "hover": "surface_background_hover", "pressed": "surface_background_pressed",
        "error": "danger_foreground", "disabled": "text_disabled",
        "selectedText": "text_on_brand",
        "inset": "chrome_surface", "divider": "chrome_border",
        "borderHover": "border_hover", "borderDisabled": "border_disabled",
        "controlDisabled": "surface_disabled", "focus": "focus_border",
        "accentText": "brand_foreground", "selected": "brand_background_subtle",
        "selectedBorder": "brand_border_subtle", "errorSurface": "danger_background",
    }
    return {key: alias(value) for key, value in names.items()}


class DrrDisplayHost(QWidget):
    def __init__(self, owner, legacy_display: QWidget, parent=None):
        super().__init__(parent)
        self.setObjectName("drr_quick_display_host")
        self.setAccessibleName("DRR display settings")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.owner = owner
        self.legacy_display = legacy_display
        self.bridge = None
        self.quick_widget = None
        self.load_error = ""
        self._attempted = False
        self._loading = False
        self._stack = QStackedWidget(self)
        self._stack.addWidget(legacy_display)
        self._notice = QLabel(self)
        self._notice.setWordWrap(True)
        self._notice.setAccessibleName("Display view status")
        self._notice.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._notice)
        layout.addWidget(self._stack, 1)
        manager = theme_manager()
        if manager is not None:
            manager.themeChanged.connect(self._apply_theme)

    @property
    def ready(self) -> bool:
        return self.quick_widget is not None and not self.load_error and self.quick_widget.rootObject() is not None

    def ensure_loaded(self) -> bool:
        if self._attempted or self._loading:
            return self.ready
        self._loading = True
        self._attempted = True
        try:
            from PySide6.QtQuickWidgets import QQuickWidget
            from ui_qt.quick.drr_display_bridge import DrrDisplayBridge

            self.bridge = DrrDisplayBridge(self.owner, self)
            self.quick_widget = QQuickWidget(self)
            self.quick_widget.setObjectName("drr_quick_display")
            self.quick_widget.setAccessibleName("DRR display settings")
            self.quick_widget.setMinimumWidth(0)
            self.quick_widget.setResizeMode(QQuickWidget.SizeRootObjectToView)
            # Own the view in the stack even if loading fails. An unmanaged
            # child otherwise becomes visible over the fallback on first show.
            self._stack.addWidget(self.quick_widget)
            self.quick_widget.statusChanged.connect(self._status_changed)
            self.quick_widget.sceneGraphError.connect(self._scene_error)
            self.quick_widget.setInitialProperties({"bridge": self.bridge, "colors": display_colors()})
            self.quick_widget.setClearColor(QColor(display_colors()["surface"]))
            self.quick_widget.setSource(QUrl.fromLocalFile(str(Path(__file__).with_name("qml") / "Display.qml")))
            if self.quick_widget.status() != QQuickWidget.Ready:
                details = "\n".join(error.toString() for error in self.quick_widget.errors())
                self._fallback(details or "The display page did not load.")
            else:
                self._stack.setCurrentWidget(self.quick_widget)
                self.bridge.setActive(self.isVisible())
        except Exception as exc:
            self._fallback(str(exc))
        finally:
            self._loading = False
        return self.ready

    @Slot(object)
    def _apply_theme(self, _theme=None):
        if self.quick_widget is not None and self.quick_widget.rootObject() is not None:
            colors = display_colors()
            self.quick_widget.setClearColor(QColor(colors["surface"]))
            self.quick_widget.rootObject().setProperty("colors", colors)

    def _status_changed(self, status):
        from PySide6.QtQuickWidgets import QQuickWidget
        if status == QQuickWidget.Error:
            self._fallback("\n".join(error.toString() for error in self.quick_widget.errors()))

    def _scene_error(self, _error, message):
        # A rendering callback may run while Qt is updating the backing store.
        self.load_error = str(message)
        QTimer.singleShot(0, self, self._apply_fallback)

    def _fallback(self, message: str):
        self.load_error = message
        self._apply_fallback()

    def _apply_fallback(self):
        self.suspend()
        self._stack.setCurrentWidget(self.legacy_display)
        self._notice.setText("The display view could not start. Standard controls are available below.")
        self._notice.setToolTip(self.load_error)
        self._notice.show()
        logging.getLogger("dptk").warning("Qt Quick display fallback: %s", self.load_error)

    def suspend(self):
        if self.bridge is not None:
            self.bridge.setActive(False)

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()
        if self.ready:
            self.bridge.refresh()
            self.bridge.setActive(True)

    def hideEvent(self, event):
        self.suspend()
        super().hideEvent(event)
