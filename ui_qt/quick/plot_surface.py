"""Matplotlib Agg figures presented by native Qt Quick, without a widget canvas.

Create :class:`QuickFigureCanvas` on the GUI thread and expose ``canvas.backend``
to QML's ``PlotSurface.backend``. Existing Python plotting code can continue to
use ``figure.canvas.draw_idle()``, Matplotlib events and Agg blitting. Figure
mutation and synchronous rendering belong to the GUI thread; ``draw_idle`` may
be requested from another thread. Quick's render thread only paints an owned,
immutable image snapshot and never enters Matplotlib.
"""
from __future__ import annotations

import sys
import threading
import weakref

from PySide6.QtCore import QEvent, QObject, Property, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QCursor, QGuiApplication, QImage
from PySide6.QtQml import QQmlEngine, qmlRegisterType
from PySide6.QtQuick import QQuickPaintedItem


def _gui_thread() -> None:
    app = QGuiApplication.instance()
    if not isinstance(app, QGuiApplication) or QThread.currentThread() != app.thread():
        raise RuntimeError("Qt Quick plotting requires the QGuiApplication GUI thread")


class _CanvasBackend(QObject):
    """QObject boundary; QML never needs the Python Matplotlib canvas itself."""

    frameReady = Signal(QImage, int)
    frameGenerationChanged = Signal()
    closed = Signal()
    drawFailed = Signal(str)
    cursorChanged = Signal(int)
    cursorTextChanged = Signal()
    _drawRequested = Signal()

    def __init__(self, canvas):
        super().__init__()
        self._canvas_ref = weakref.ref(canvas)
        self._cursor_text = ""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._draw)
        self._drawRequested.connect(self._queue_draw)
        QQmlEngine.setObjectOwnership(self, QQmlEngine.ObjectOwnership.CppOwnership)

    @Property(int, notify=frameGenerationChanged)
    def frameGeneration(self):
        canvas = self._canvas_ref()
        return canvas.frame_generation if canvas is not None else 0

    @Property(str, notify=cursorTextChanged)
    def cursorText(self):
        return self._cursor_text

    def set_cursor_text(self, text):
        if text != self._cursor_text:
            self._cursor_text = text
            self.cursorTextChanged.emit()

    @Slot()
    def _queue_draw(self):
        canvas = self._canvas_ref()
        if canvas is not None and not canvas.closed and not self._timer.isActive():
            self._timer.start(0)

    @Slot()
    def _draw(self):
        canvas = self._canvas_ref()
        if canvas is None or canvas.closed:
            return
        try:
            from core.typography import try_render_lock
            with try_render_lock() as available:
                if available:
                    canvas.draw()
                else:
                    # Export owns a different figure. Keep processing clicks
                    # while it saves, then render the latest live state once.
                    self._timer.start(16)
        except Exception as error:
            canvas.last_error = str(error)
            self.drawFailed.emit(canvas.last_error)


def _buttons():
    from matplotlib.backend_bases import MouseButton
    return {
        Qt.MouseButton.LeftButton: MouseButton.LEFT,
        Qt.MouseButton.MiddleButton: MouseButton.MIDDLE,
        Qt.MouseButton.RightButton: MouseButton.RIGHT,
        Qt.MouseButton.BackButton: MouseButton.BACK,
        Qt.MouseButton.ForwardButton: MouseButton.FORWARD,
    }


_MODIFIERS = (
    (Qt.KeyboardModifier.ControlModifier, Qt.Key.Key_Control, "ctrl"),
    (Qt.KeyboardModifier.AltModifier, Qt.Key.Key_Alt, "alt"),
    (Qt.KeyboardModifier.ShiftModifier, Qt.Key.Key_Shift, "shift"),
    (Qt.KeyboardModifier.MetaModifier, Qt.Key.Key_Meta, "meta"),
)
_SPECIAL_KEYS = {
    Qt.Key.Key_Escape: "escape", Qt.Key.Key_Tab: "tab", Qt.Key.Key_Backtab: "tab",
    Qt.Key.Key_Backspace: "backspace", Qt.Key.Key_Return: "enter", Qt.Key.Key_Enter: "enter",
    Qt.Key.Key_Insert: "insert", Qt.Key.Key_Delete: "delete", Qt.Key.Key_Pause: "pause",
    Qt.Key.Key_SysReq: "sysreq", Qt.Key.Key_Clear: "clear", Qt.Key.Key_Home: "home",
    Qt.Key.Key_End: "end", Qt.Key.Key_Left: "left", Qt.Key.Key_Up: "up",
    Qt.Key.Key_Right: "right", Qt.Key.Key_Down: "down", Qt.Key.Key_PageUp: "pageup",
    Qt.Key.Key_PageDown: "pagedown", Qt.Key.Key_Shift: "shift", Qt.Key.Key_Control: "control",
    Qt.Key.Key_Meta: "meta", Qt.Key.Key_Alt: "alt", Qt.Key.Key_CapsLock: "caps_lock",
    Qt.Key.Key_Super_L: "super", Qt.Key.Key_Super_R: "super",
    **{int(Qt.Key.Key_F1) + index: f"f{index + 1}" for index in range(12)},
}


def _modifiers(event, exclude=None):
    return [name for mask, key, name in _MODIFIERS
            if key != exclude and event.modifiers() & mask]


class _SurfaceOwnership:
    """Destruction hook without a bound-method reference back to the item."""

    def __init__(self):
        self.canvas = None
        self.owned = False

    def dispose(self, *_):
        canvas, owned = self.canvas, self.owned
        self.canvas, self.owned = None, False
        if canvas is not None and not canvas.closed:
            canvas.backend.set_cursor_text("")
        if owned and canvas is not None and not canvas.closed:
            canvas.close()


class PlotSurface(QQuickPaintedItem):
    """Native Quick item accepting a ``QuickFigureCanvas.backend`` QObject.

    Python may instead call ``set_figure`` (surface owns its new canvas) or
    ``set_canvas`` (caller owns the supplied canvas). A canvas has one viewport;
    do not attach it to multiple independently sized surfaces.
    """

    backendChanged = Signal()
    paintedGenerationChanged = Signal()
    presentedGenerationChanged = Signal()
    framePainted = Signal(int)
    framePresented = Signal(int)
    _paintCompleted = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._backend = None
        self._ownership = _SurfaceOwnership()
        self._image = QImage()
        self._image_generation = 0
        self._attachment_epoch = 0
        self._painted_generation = 0
        self._presented_generation = 0
        self._image_lock = threading.Lock()
        self._window = None
        self._size_pending = False
        self._size_timer = QTimer(self)
        self._size_timer.setSingleShot(True)
        self._size_timer.timeout.connect(self._flush_canvas_size)
        self._last_position = None
        self.setAcceptedMouseButtons(Qt.MouseButton.AllButtons)
        self.setAcceptHoverEvents(True)
        self.setFlag(self.Flag.ItemIsFocusScope, True)
        self.setAntialiasing(False)
        self._paintCompleted.connect(self._mark_painted)
        self.windowChanged.connect(self._window_changed)
        self.visibleChanged.connect(self._visibility_changed)
        self.destroyed.connect(self._ownership.dispose)
        # QQuickItem's constructor can attach to a parent's window before our
        # Python signal connections exist. Bind that already-established window
        # as well as future windowChanged notifications.
        if self.window() is not None:
            self._window_changed(self.window())

    @Property(QObject, notify=backendChanged)
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        if backend is not None and not isinstance(backend, _CanvasBackend):
            raise TypeError("PlotSurface.backend must be a QuickFigureCanvas.backend")
        canvas = backend._canvas_ref() if backend is not None else None
        self.set_canvas(canvas)

    @property
    def canvas(self):
        return self._ownership.canvas

    @property
    def figure(self):
        return self.canvas.figure if self.canvas is not None else None

    @Property(int, notify=paintedGenerationChanged)
    def paintedGeneration(self):
        return self._painted_generation

    @Property(int, notify=presentedGenerationChanged)
    def presentedGeneration(self):
        return self._presented_generation

    def set_figure(self, figure):
        from ui_qt.quick.figure_canvas import QuickFigureCanvas
        self.set_canvas(QuickFigureCanvas(figure))
        self._ownership.owned = True
        return self.canvas

    def set_canvas(self, canvas):
        _gui_thread()
        if canvas is not None:
            from ui_qt.quick.figure_canvas import QuickFigureCanvas
            if not isinstance(canvas, QuickFigureCanvas):
                raise TypeError("PlotSurface requires a QuickFigureCanvas")
        if canvas is not None and canvas.closed:
            raise ValueError("Cannot attach a closed canvas")
        if canvas is self.canvas:
            return
        if self._backend is not None:
            self._backend.set_cursor_text("")
            self._backend.frameReady.disconnect(self._accept_frame)
            self._backend.closed.disconnect(self._backend_closed)
            self._backend.cursorChanged.disconnect(self._set_cursor)
        self._ownership.dispose()
        self._ownership.canvas = canvas
        self._size_pending = False
        self._size_timer.stop()
        self._backend = canvas.backend if canvas is not None else None
        with self._image_lock:
            self._attachment_epoch += 1
            self._image = canvas.image if canvas is not None else QImage()
            self._image_generation = canvas.frame_generation if canvas is not None else 0
        self._painted_generation = self._presented_generation = 0
        self.paintedGenerationChanged.emit()
        self.presentedGenerationChanged.emit()
        if self._backend is not None:
            self._backend.set_cursor_text("")
            self._backend.frameReady.connect(self._accept_frame)
            self._backend.closed.connect(self._backend_closed)
            self._backend.cursorChanged.connect(self._set_cursor)
            self._sync_size()
        self.backendChanged.emit()
        self.update()

    @Slot()
    def _visibility_changed(self):
        if not self.isVisible() and self._backend is not None:
            self._backend.set_cursor_text("")

    @Slot()
    def close(self):
        """Detach the backend and close a canvas created by ``set_figure``."""
        self.set_canvas(None)

    @Slot()
    def _backend_closed(self):
        self.set_canvas(None)

    @Slot(QImage, int)
    def _accept_frame(self, image, generation):
        with self._image_lock:
            self._image = QImage(image)
            self._image_generation = generation
        self.update()

    def paint(self, painter):
        # This can run on the scene graph render thread. All mutable Agg state
        # stays on the GUI thread; retain one immutable image for this paint.
        with self._image_lock:
            image, generation, epoch = QImage(self._image), self._image_generation, self._attachment_epoch
        if not image.isNull():
            painter.drawImage(self.boundingRect(), image)
            self._paintCompleted.emit(generation, epoch)

    @Slot(int, int)
    def _mark_painted(self, generation, epoch):
        if epoch != self._attachment_epoch:
            return
        if generation != self._painted_generation:
            self._painted_generation = generation
            self.paintedGenerationChanged.emit()
            self.framePainted.emit(generation)

    @Slot()
    def _frame_swapped(self):
        generation = self._painted_generation
        if generation and generation != self._presented_generation:
            self._presented_generation = generation
            self.presentedGenerationChanged.emit()
            self.framePresented.emit(generation)

    @Slot(object)
    def _window_changed(self, window):
        from shiboken6 import isValid
        # During native teardown the wrapper can remain valid after its signal
        # source is gone. Connection handles remain safe to disconnect then;
        # accessing the old window's SignalInstance does not.
        if self._window is not None and isValid(self._window):
            self._window.removeEventFilter(self)
        for connection in getattr(self, "_window_connections", ()):
            QObject.disconnect(connection)
        self._window_connections = ()
        self._window = window
        if window is not None:
            window.installEventFilter(self)
            self._window_connections = (
                window.frameSwapped.connect(self._frame_swapped),
                window.screenChanged.connect(self._sync_size),
            )
        self._sync_size()

    def eventFilter(self, watched, event):
        if watched is self._window and event.type() == QEvent.Type.DevicePixelRatioChange:
            self._sync_size()
        return super().eventFilter(watched, event)

    def geometryChange(self, new_geometry, old_geometry):
        super().geometryChange(new_geometry, old_geometry)
        if hasattr(self, "_ownership"):
            self._sync_size()

    def _sync_size(self, *_):
        if self.canvas is None or self.canvas.closed:
            return
        if self._window is None:
            self._size_timer.stop()
            self._size_pending = False
            self._apply_canvas_size()
        else:
            # Workflow changes update the backend before the QML layout has
            # settled. Resize during Quick's polish phase, not at the outgoing
            # page's dimensions; queued Agg draws then see the final geometry.
            self._size_pending = True
            self.polish()

    def updatePolish(self):
        # A parent layout may request another polish pass in this frame. Wait
        # until all those passes finish before mutating a retained Agg figure.
        if self._size_pending:
            self._size_timer.start(0)

    def _flush_canvas_size(self):
        self._size_pending = False
        self._apply_canvas_size()

    def _apply_canvas_size(self):
        canvas = self.canvas
        if canvas is not None and not canvas.closed:
            dpr = self._window.devicePixelRatio() if self._window is not None else 1.0
            canvas.set_logical_size(self.width(), self.height(), dpr)
            if canvas.image.isNull():
                canvas.draw_idle()

    @Slot(int)
    def _set_cursor(self, cursor):
        from matplotlib.backend_tools import Cursors
        shapes = {
            Cursors.POINTER: Qt.CursorShape.ArrowCursor,
            Cursors.HAND: Qt.CursorShape.PointingHandCursor,
            Cursors.SELECT_REGION: Qt.CursorShape.CrossCursor,
            Cursors.MOVE: Qt.CursorShape.SizeAllCursor,
            Cursors.WAIT: Qt.CursorShape.WaitCursor,
            Cursors.RESIZE_HORIZONTAL: Qt.CursorShape.SizeHorCursor,
            Cursors.RESIZE_VERTICAL: Qt.CursorShape.SizeVerCursor,
        }
        self.setCursor(QCursor(shapes.get(cursor, Qt.CursorShape.ArrowCursor)))

    def _coordinates(self, event=None):
        if event is not None:
            self._last_position = event.position()
        position = self._last_position
        if position is None:
            return 0.0, 0.0
        canvas = self.canvas
        ratio = canvas.device_pixel_ratio
        return position.x() * ratio, canvas.figure.bbox.height - position.y() * ratio

    def _mouse_event(self, name, event, *, double=False):
        canvas = self.canvas
        if canvas is None or canvas.closed:
            event.ignore()
            return
        from matplotlib.backend_bases import MouseEvent
        button = _buttons().get(event.button())
        if button is None:
            event.ignore()
            return
        MouseEvent(name, canvas, *self._coordinates(event), button=button,
                   dblclick=double, modifiers=_modifiers(event), guiEvent=event)._process()
        event.accept()

    def mousePressEvent(self, event):
        self.forceActiveFocus(Qt.FocusReason.MouseFocusReason)
        self._mouse_event("button_press_event", event)

    def mouseDoubleClickEvent(self, event):
        self._mouse_event("button_press_event", event, double=True)

    def mouseReleaseEvent(self, event):
        self._mouse_event("button_release_event", event)

    def _motion_event(self, event, buttons):
        canvas = self.canvas
        if canvas is not None and not canvas.closed:
            from matplotlib.backend_bases import MouseEvent
            MouseEvent("motion_notify_event", canvas, *self._coordinates(event), buttons=buttons,
                       modifiers=_modifiers(event), guiEvent=event)._process()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.canvas is None or self.canvas.closed:
            event.ignore()
            return
        self._motion_event(event, {_button for qt_button, _button in _buttons().items()
                                   if event.buttons() & qt_button})

    def hoverMoveEvent(self, event):
        self._motion_event(event, set())

    def hoverEnterEvent(self, event):
        canvas = self.canvas
        if canvas is not None and not canvas.closed:
            from matplotlib.backend_bases import LocationEvent
            LocationEvent("figure_enter_event", canvas, *self._coordinates(event),
                          modifiers=_modifiers(event), guiEvent=event)._process()
            event.accept()

    def hoverLeaveEvent(self, event):
        canvas = self.canvas
        if canvas is not None and not canvas.closed:
            from matplotlib.backend_bases import LocationEvent
            LocationEvent("figure_leave_event", canvas, *self._coordinates(event),
                          modifiers=_modifiers(event), guiEvent=event)._process()
            event.accept()

    def wheelEvent(self, event):
        canvas = self.canvas
        if canvas is None or canvas.closed:
            event.ignore()
            return
        pixel_delta = event.pixelDelta()
        app = QGuiApplication.instance()
        steps = (event.angleDelta().y() / 120 if pixel_delta.isNull() or app.platformName() == "xcb"
                 else pixel_delta.y())
        if steps:
            from matplotlib.backend_bases import MouseEvent
            MouseEvent("scroll_event", canvas, *self._coordinates(event), step=steps,
                       modifiers=_modifiers(event), guiEvent=event)._process()
            event.accept()

    def _key_event(self, name, event):
        canvas = self.canvas
        if canvas is None or canvas.closed:
            event.ignore()
            return
        code = event.key()
        modifiers = _modifiers(event, exclude=code)
        key = _SPECIAL_KEYS.get(code)
        if key is None:
            if code > sys.maxunicode:
                event.ignore()
                return
            key = chr(code)
            if "shift" in modifiers:
                modifiers.remove("shift")
            else:
                key = key.lower()
        from matplotlib.backend_bases import KeyEvent
        KeyEvent(name, canvas, "+".join(modifiers + [key]), *self._coordinates(), guiEvent=event)._process()
        event.accept()

    def keyPressEvent(self, event):
        self._key_event("key_press_event", event)

    def keyReleaseEvent(self, event):
        self._key_event("key_release_event", event)


_registered = False


def register_plot_surface():
    """Register ``import Dptk 1.0; PlotSurface { backend: ... }`` once."""
    global _registered
    _gui_thread()
    if not _registered:
        qmlRegisterType(PlotSurface, "Dptk", 1, 0, "PlotSurface")
        _registered = True


def __getattr__(name):
    # Keep the public import compatible while a blank QML PlotSurface remains
    # independent of Matplotlib, NumPy and its font cache.
    if name == "QuickFigureCanvas":
        from ui_qt.quick.figure_canvas import QuickFigureCanvas
        globals()[name] = QuickFigureCanvas
        return QuickFigureCanvas
    raise AttributeError(name)


__all__ = ["PlotSurface", "QuickFigureCanvas", "register_plot_surface"]
