"""Keep wheel scrolling from editing values or changing the active tab."""

from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QApplication, QAbstractScrollArea, QAbstractSpinBox, QComboBox,
    QDial, QSlider, QTabBar, QWidget,
)


class WheelValueGuard(QObject):
    def eventFilter(self, watched, event):  # noqa: N802 - Qt API
        if event.type() != QEvent.Type.Wheel or not isinstance(watched, QWidget):
            return False
        # Editors inside combo/spin boxes receive wheel events too. Stop at a
        # scrolling view so an open combo popup can still scroll its options.
        control = watched
        while control is not None:
            if isinstance(control, QAbstractScrollArea):
                return False
            if isinstance(control, (QComboBox, QAbstractSpinBox, QSlider, QDial, QTabBar)):
                break
            control = control.parentWidget()
        if control is None:
            return False

        parent = control.parentWidget()
        while parent is not None and not isinstance(parent, QAbstractScrollArea):
            parent = parent.parentWidget()
        if parent is not None:
            viewport = parent.viewport()
            forwarded = QWheelEvent(
                viewport.mapFromGlobal(event.globalPosition().toPoint()).toPointF(),
                event.globalPosition(), event.pixelDelta(), event.angleDelta(),
                event.buttons(), event.modifiers(), event.phase(), event.inverted(),
                event.source(), event.pointingDevice(),
            )
            QApplication.sendEvent(viewport, forwarded)
        event.accept()
        return True


def install_wheel_value_guard(app: QApplication | None = None) -> WheelValueGuard | None:
    """Install once for all current/future windows and child editors."""
    app = app or QApplication.instance()
    if app is None:
        return None
    guard = getattr(app, "_wheel_value_guard", None)
    if guard is None:
        guard = WheelValueGuard(app)
        app.installEventFilter(guard)
        app._wheel_value_guard = guard
    return guard
