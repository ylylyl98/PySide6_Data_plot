"""Matplotlib canvas loaded only when a native Quick workspace first plots."""
from __future__ import annotations

import math
import time

from matplotlib.backend_bases import CloseEvent, ResizeEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QImage
from core.typography import apply_figure_font, matplotlib_font_context
from ui_qt.quick.plot_surface import _CanvasBackend, _gui_thread


class QuickFigureCanvas(FigureCanvasAgg):
    """Agg canvas with coalesced Qt draws and immutable Quick image frames.

    ``frameReady`` is a Qt signal carrying ``(QImage, generation)``. ``draw`` and
    ``blit`` publish synchronously; ``draw_idle`` combines requests until the next
    GUI event turn. Blit publishes Agg's modified buffer without redrawing any
    artists. A complete buffer copy protects Quick from later Agg writes.
    """

    required_interactive_framework = "qt"

    def __init__(self, figure=None):
        _gui_thread()
        # Existing Qt figures may currently have a physical DPI scaled by their
        # old screen. Start the new viewport at logical DPI rather than applying
        # that previous screen's ratio a second time.
        previous_canvas = getattr(figure, "canvas", None)
        previous_ratio = getattr(previous_canvas, "device_pixel_ratio", 1.0)
        logical_dpi = figure.dpi / previous_ratio if figure is not None else None
        super().__init__(figure)
        if previous_ratio != 1.0:
            figure._original_dpi = logical_dpi
            figure._set_dpi(logical_dpi, forward=False)
        self.closed = False
        self._rendering = False
        self._image = QImage()
        self.frame_generation = 0
        self.draw_count = 0
        self.blit_count = 0
        self.last_render_ms = 0.0
        self.last_error = ""
        self.backend = _CanvasBackend(self)

    @property
    def frameReady(self):
        return self.backend.frameReady

    @property
    def image(self):
        """A shared, read-only snapshot; subsequent frames cannot modify it."""
        return QImage(self._image)

    def _assert_open(self):
        _gui_thread()
        if self.closed:
            raise RuntimeError("The Qt Quick figure canvas is closed")

    def draw(self):
        self._assert_open()
        if self._rendering:
            self.draw_idle()
            return
        self.backend._timer.stop()
        self._rendering = True
        started = time.perf_counter()
        try:
            from ui_qt.quick.raster_heatmaps import screen_heatmaps
            with matplotlib_font_context(self.figure), screen_heatmaps(self):
                super().draw()
            self.draw_count += 1
            self._publish_frame()
            self.last_error = ""
        finally:
            self.last_render_ms = (time.perf_counter() - started) * 1000
            self._rendering = False

    def apply_ui_font(self):
        apply_figure_font(self.figure)

    def print_figure(self, *args, **kwargs):
        # Agg's PNG encoder calls FigureCanvasAgg.draw directly, bypassing
        # this canvas's live-frame draw override.
        with matplotlib_font_context(self.figure):
            return super().print_figure(*args, **kwargs)

    def draw_idle(self):
        if not self.closed:
            # AutoConnection marshals calls from workers to this QObject's
            # GUI thread. The timer itself is only ever touched on that thread.
            self.backend._drawRequested.emit()

    def blit(self, bbox=None):
        self._assert_open()
        if not hasattr(self, "renderer"):
            self.draw()
            return
        self.blit_count += 1
        self._publish_frame()

    def _publish_frame(self):
        buffer = self.buffer_rgba()
        height, width = buffer.shape[:2]
        image = QImage(buffer, width, height, width * 4, QImage.Format.Format_RGBA8888).copy()
        image.setDevicePixelRatio(self.device_pixel_ratio)
        self._image = image
        self.frame_generation += 1
        self.backend.frameGenerationChanged.emit()
        self.backend.frameReady.emit(QImage(image), self.frame_generation)

    def set_logical_size(self, width, height, dpr=1.0):
        """Resize using Quick logical pixels and a screen device pixel ratio."""
        self._assert_open()
        width, height, dpr = float(width), float(height), float(dpr)
        if not all(math.isfinite(v) for v in (width, height, dpr)) or dpr <= 0:
            raise ValueError("Plot dimensions and DPR must be finite; DPR must be positive")
        if width <= 0 or height <= 0:
            return
        physical_size = (max(1, round(width * dpr)), max(1, round(height * dpr)))
        ratio_changed = self._set_device_pixel_ratio(dpr)
        if not ratio_changed and self.get_width_height(physical=True) == physical_size:
            return
        # Agg truncates float bbox sizes. For example, 995 / 100 * 100 can be
        # 994.9999999999999, causing a one-pixel loss and repeat resize requests.
        self.figure.set_size_inches(math.nextafter(physical_size[0] / self.figure.dpi, math.inf),
                                    math.nextafter(physical_size[1] / self.figure.dpi, math.inf), forward=False)
        ResizeEvent("resize_event", self)._process()
        self.draw_idle()

    def flush_events(self):
        self._assert_open()
        QCoreApplication.processEvents()

    def set_cursor(self, cursor):
        if not self.closed:
            self.backend.cursorChanged.emit(int(cursor))

    def close(self):
        """Stop queued draws, notify listeners and release the figure/image."""
        _gui_thread()
        if self.closed:
            return
        self.closed = True
        self.backend._timer.stop()
        try:
            CloseEvent("close_event", self)._process()
        finally:
            figure, self.figure = self.figure, None
            if figure is not None and figure.canvas is self:
                figure.set_canvas(None)
            self._image = QImage()
            if hasattr(self, "renderer"):
                del self.renderer
            self.backend.closed.emit()
