"""Native Quick/Agg checks in a fresh GUI process with no application state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickPlotSurfaceTests(unittest.TestCase):
    def _run_gui_process(self, scale):
        with TemporaryDirectory() as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                               QT_SCALE_FACTOR=str(scale), MPLBACKEND="agg", MPLCONFIGDIR=temporary,
                               XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                                    cwd=Path(__file__).resolve().parents[1], env=environment,
                                    capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_PLOT_MEASUREMENTS=", result.stdout)

    def test_native_quick_surface_in_isolated_gui_process(self):
        self._run_gui_process(1)

    def test_native_quick_surface_at_high_dpi(self):
        self._run_gui_process(2)


def _run_isolated():
    # Import only in the child. Existing unittest discovery may already own a
    # QApplication; it must never select this canvas's GUI process or settings.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import gc
    import threading
    import time
    import weakref
    from unittest.mock import patch

    import numpy as np
    from matplotlib.backend_bases import MouseButton
    from matplotlib.figure import Figure
    from PySide6.QtCore import QByteArray, QCoreApplication, QEvent, QPoint, QPointF, QSize, Qt, QUrl
    from PySide6.QtGui import QGuiApplication, QHoverEvent, QImage, QKeyEvent, QMouseEvent, QPainter, QWheelEvent
    from PySide6.QtQml import QQmlApplicationEngine, QQmlComponent, QQmlEngine
    from PySide6.QtQuick import QQuickWindow
    from shiboken6 import isValid
    from ui_qt.quick.plot_surface import PlotSurface, QuickFigureCanvas, register_plot_surface

    production_imports = set(sys.modules)
    # QtTest has QWidget overloads and imports QtWidgets on this PySide build.
    # Inspect the production dependency boundary before loading that test helper.
    from PySide6.QtTest import QTest

    app = QGuiApplication([])
    measurements = {}

    def settle(milliseconds=15):
        app.processEvents()
        QTest.qWait(milliseconds)
        app.processEvents()

    def pixels(image):
        image = image.convertToFormat(QImage.Format.Format_RGBA8888)
        return np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine())[:, :image.width() * 4].copy()

    class SurfaceCases(unittest.TestCase):
        def make_canvas(self, width=320, height=240):
            figure = Figure(figsize=(width / 100, height / 100), dpi=100)
            axes = figure.add_axes((0.12, 0.15, 0.82, 0.78))
            axes.plot([0, 1, 2, 3], [1, 4, 2, 3], color="#c03060")
            axes.set_title("Native Quick / Agg")
            canvas = QuickFigureCanvas(figure)
            self.addCleanup(canvas.close)
            return canvas, axes

        def make_surface(self, canvas, width=320, height=240):
            surface = PlotSurface()
            surface.setWidth(width)
            surface.setHeight(height)
            surface.set_canvas(canvas)
            self.addCleanup(surface.close)
            return surface

        def test_rgba_fidelity_and_paint_never_calls_matplotlib(self):
            canvas, axes = self.make_canvas()
            axes.imshow(np.arange(64).reshape(8, 8), extent=(0, 3, 0, 5), alpha=.3)
            surface = self.make_surface(canvas)
            canvas.draw()
            expected = np.asarray(canvas.buffer_rgba()).reshape(240, 320 * 4).copy()
            np.testing.assert_array_equal(pixels(canvas.image), expected)
            target = QImage(320, 240, QImage.Format.Format_RGBA8888)
            target.fill(Qt.GlobalColor.transparent)
            painter = QPainter(target)
            with patch.object(canvas.figure, "draw", side_effect=AssertionError("paint rendered Matplotlib")):
                for _ in range(3):
                    surface.paint(painter)
            painter.end()
            np.testing.assert_array_equal(pixels(target), expected)
            self.assertEqual(canvas.draw_count, 1)
            snapshot = canvas.image
            np.asarray(canvas.buffer_rgba())[:] = 0
            np.testing.assert_array_equal(pixels(snapshot), expected)

        def test_draw_idle_coalesces_gui_and_worker_requests(self):
            canvas, _ = self.make_canvas()
            render_threads = []
            canvas.mpl_connect("draw_event", lambda _: render_threads.append(threading.get_ident()))
            started = time.perf_counter()
            for _ in range(250):
                canvas.draw_idle()
            self.assertEqual(canvas.draw_count, 0)
            settle()
            self.assertEqual(canvas.draw_count, 1)
            worker = threading.Thread(target=lambda: [canvas.draw_idle() for _ in range(250)])
            worker.start()
            worker.join()
            canvas.flush_events()
            settle()
            self.assertEqual(canvas.draw_count, 2)
            self.assertEqual(set(render_threads), {threading.get_ident()})
            failures = []
            def wrong_thread():
                try:
                    canvas.draw()
                except RuntimeError as error:
                    failures.append(str(error))
            worker = threading.Thread(target=wrong_thread)
            worker.start()
            worker.join()
            self.assertEqual(len(failures), 1)
            measurements["500_idle_requests_completed_draws"] = canvas.draw_count
            measurements["two_event_turns_ms"] = round((time.perf_counter() - started) * 1000, 3)

        def test_blit_publishes_changed_agg_buffer_without_full_draw(self):
            canvas, axes = self.make_canvas()
            line, = axes.plot([0, 3], [1, 4], animated=True, linewidth=4)
            canvas.draw()
            background = canvas.copy_from_bbox(axes.bbox)
            before = pixels(canvas.image)
            canvas.restore_region(background)
            axes.draw_artist(line)
            full_draws = canvas.draw_count
            started = time.perf_counter()
            canvas.blit(axes.bbox)
            measurements["320x240_blit_publish_ms"] = round((time.perf_counter() - started) * 1000, 3)
            self.assertEqual(canvas.draw_count, full_draws)
            self.assertEqual(canvas.blit_count, 1)
            self.assertEqual(canvas.frame_generation, 2)
            self.assertFalse(np.array_equal(before, pixels(canvas.image)))
            np.testing.assert_array_equal(pixels(canvas.image), np.asarray(canvas.buffer_rgba()).reshape(240, 320 * 4))

        def test_resize_dpi_preserves_physical_and_logical_geometry(self):
            canvas, _ = self.make_canvas()
            resize_events = []
            canvas.mpl_connect("resize_event", resize_events.append)
            canvas.set_logical_size(400, 200, 2)
            settle()
            self.assertEqual(canvas.image.size(), QSize(800, 400))
            self.assertEqual(canvas.image.devicePixelRatio(), 2)
            self.assertEqual(canvas.figure.dpi, 200)
            self.assertEqual(canvas.get_width_height(), (400, 200))
            self.assertEqual(len(resize_events), 1)
            draws = canvas.draw_count
            canvas.set_logical_size(400, 200, 2)
            settle()
            self.assertEqual(canvas.draw_count, draws)
            canvas.set_logical_size(400, 200, 1)
            settle()
            self.assertEqual(canvas.image.size(), QSize(400, 200))
            self.assertEqual(canvas.figure.dpi, 100)
            self.assertEqual(len(resize_events), 2)
            canvas.set_logical_size(0, 0, 1)
            with self.assertRaises(ValueError):
                canvas.set_logical_size(float("nan"), 20)

        def test_attaching_an_existing_hidpi_figure_keeps_logical_dpi_and_callbacks(self):
            previous, _ = self.make_canvas()
            previous.set_logical_size(320, 240, 2)
            figure = previous.figure
            draws = []
            previous.mpl_connect("draw_event", draws.append)
            replacement = QuickFigureCanvas(figure)
            self.addCleanup(replacement.close)
            self.assertIs(figure.canvas, replacement)
            self.assertEqual(figure.dpi, 100)
            self.assertEqual(replacement.get_width_height(), (320, 240))
            previous.close()
            self.assertIs(figure.canvas, replacement)
            replacement.draw()
            self.assertEqual(len(draws), 1)
            self.assertIs(draws[0].canvas, replacement)

        def test_mouse_drag_hover_double_click_and_scroll_coordinates(self):
            canvas, axes = self.make_canvas(320, 200)
            axes.set_position((0, 0, 1, 1))
            axes.set_xlim(0, 10)
            axes.set_ylim(0, 10)
            surface = self.make_surface(canvas, 320, 200)
            canvas.set_logical_size(320, 200, 2)
            canvas.draw()
            draw_count, generation = canvas.draw_count, canvas.frame_generation
            cursor_updates = []
            canvas.backend.cursorTextChanged.connect(lambda: cursor_updates.append(canvas.backend.cursorText))
            received = []
            gui_events = []
            def record(event):
                received.append(event)
                gui_events.append(event.guiEvent)
            for name in ("button_press_event", "motion_notify_event", "button_release_event", "scroll_event", "figure_enter_event", "figure_leave_event"):
                canvas.mpl_connect(name, record)
            point = QPointF(80, 60)
            ctrl = Qt.KeyboardModifier.ControlModifier
            def mouse(kind, button=Qt.MouseButton.LeftButton, buttons=Qt.MouseButton.LeftButton):
                return QMouseEvent(kind, point, point, button, buttons, ctrl)
            press = mouse(QEvent.Type.MouseButtonPress)
            surface.mousePressEvent(press)
            surface.mouseMoveEvent(mouse(QEvent.Type.MouseMove, Qt.MouseButton.NoButton))
            surface.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease, buttons=Qt.MouseButton.NoButton))
            surface.mouseDoubleClickEvent(mouse(QEvent.Type.MouseButtonDblClick))
            hover = QHoverEvent(QEvent.Type.HoverMove, point, point, point, Qt.KeyboardModifier.NoModifier)
            surface.hoverMoveEvent(hover)
            self.assertEqual(canvas.backend.cursorText, axes.format_coord(2.5, 7.0))
            self.assertEqual(cursor_updates, [axes.format_coord(2.5, 7.0)],
                             "Repeated mouse/hover coordinates must not republish the same text")
            surface.hoverEnterEvent(hover)
            surface.hoverLeaveEvent(hover)
            self.assertEqual(canvas.backend.cursorText, "")
            self.assertEqual(cursor_updates[-1], "")
            wheel = QWheelEvent(point, point, QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton,
                                ctrl, Qt.ScrollPhase.NoScrollPhase, False)
            surface.wheelEvent(wheel)
            self.assertEqual([event.name for event in received], ["button_press_event", "motion_notify_event", "button_release_event", "button_press_event", "motion_notify_event", "figure_enter_event", "figure_leave_event", "scroll_event"])
            first = received[0]
            self.assertEqual((first.x, first.y), (160, 280))
            self.assertAlmostEqual(first.xdata, 2.5)
            self.assertAlmostEqual(first.ydata, 7)
            self.assertEqual(first.button, MouseButton.LEFT)
            self.assertEqual(first.modifiers, frozenset({"ctrl"}))
            self.assertEqual(received[1].buttons, frozenset({MouseButton.LEFT}))
            self.assertEqual(received[4].buttons, frozenset())
            self.assertTrue(received[3].dblclick)
            self.assertEqual(received[-1].step, 1)
            self.assertEqual(received[-1].button, "up")
            self.assertIs(gui_events[0], press)
            self.assertEqual(canvas.draw_count, draw_count, "Reading coordinates must not redraw artists")
            self.assertEqual(canvas.frame_generation, generation, "Reading coordinates must not publish image frames")
            self.assertFalse(canvas.backend._timer.isActive())

        def test_cursor_uses_data_axes_and_clears_on_view_lifecycle_changes(self):
            canvas, axes = self.make_canvas()
            axes.set_xscale("log")
            axes.set_yscale("log")
            axes.set_xlim(1, 1000)
            axes.set_ylim(100, .1)
            surface = self.make_surface(canvas)
            canvas.set_logical_size(320, 240, 2)
            canvas.draw()

            def hover_at(x=10, y=1):
                physical_x, physical_y = axes.transData.transform((x, y))
                point = QPointF(physical_x / canvas.device_pixel_ratio,
                    (canvas.figure.bbox.height - physical_y) / canvas.device_pixel_ratio)
                event = QHoverEvent(QEvent.Type.HoverMove, point, point, point, Qt.KeyboardModifier.NoModifier)
                surface.hoverMoveEvent(event)
                self.assertEqual(canvas.backend.cursorText, axes.format_coord(x, y))

            hover_at()
            margin = QPointF(1, 1)
            surface.hoverMoveEvent(QHoverEvent(QEvent.Type.HoverMove, margin, margin, margin, Qt.KeyboardModifier.NoModifier))
            self.assertEqual(canvas.backend.cursorText, "", "Figure margins have no data coordinates")
            hover_at()
            canvas.draw()
            self.assertEqual(canvas.backend.cursorText, "", "A new view must not retain coordinates from its old transform")
            hover_at()
            canvas.set_logical_size(400, 240, 2)
            self.assertEqual(canvas.backend.cursorText, "")
            settle()
            hover_at()
            surface.setVisible(False)
            self.assertEqual(canvas.backend.cursorText, "", "Hidden workflow plots must clear their reading")
            surface.setVisible(True)
            hover_at()
            surface.set_canvas(None)
            self.assertEqual(canvas.backend.cursorText, "")
            replacement, _ = self.make_canvas()
            surface.set_canvas(replacement)
            self.assertEqual(replacement.backend.cursorText, "")
            surface.set_canvas(canvas)
            hover_at()
            canvas.close()
            self.assertEqual(canvas.backend.cursorText, "")
            self.assertIsNone(surface.backend)

        def test_key_events_match_matplotlib_shortcut_conventions(self):
            canvas, axes = self.make_canvas()
            surface = self.make_surface(canvas)
            keys = []
            canvas.mpl_connect("key_press_event", keys.append)
            canvas.mpl_connect("key_release_event", keys.append)
            cases = ((Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier, "a"),
                     (Qt.Key.Key_A, Qt.KeyboardModifier.ShiftModifier, "A"),
                     (Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier, "ctrl+a"),
                     (Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier, "shift+left"),
                     (Qt.Key.Key_Control, Qt.KeyboardModifier.ControlModifier, "control"),
                     (Qt.Key.Key_F10, Qt.KeyboardModifier.NoModifier, "f10"))
            for code, modifiers, expected in cases:
                surface.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, code, modifiers))
                self.assertEqual(keys[-1].key, expected)
                surface.keyReleaseEvent(QKeyEvent(QEvent.Type.KeyRelease, code, modifiers))
                self.assertEqual(keys[-1].name, "key_release_event")
                self.assertEqual(keys[-1].key, expected)

        def test_qml_registration_real_window_and_presented_generation(self):
            register_plot_surface()
            register_plot_surface()
            canvas, _ = self.make_canvas()
            engine = QQmlEngine()
            engine.rootContext().setContextProperty("testBackend", canvas.backend)
            component = QQmlComponent(engine)
            component.setData(QByteArray(b'import QtQuick\nimport Dptk 1.0\nPlotSurface { width: 320; height: 240; backend: testBackend }'), QUrl())
            self.assertTrue(component.isReady(), str(component.errors()))
            surface = component.create()
            self.assertIsInstance(surface, PlotSurface)
            window = QQuickWindow()
            window.resize(320, 240)
            surface.setParentItem(window.contentItem())
            presented = []
            surface.framePresented.connect(lambda generation: presented.append((generation, time.perf_counter())))
            window.show()
            started = time.perf_counter()
            canvas.draw()
            deadline = time.monotonic() + 3
            # Showing the window can queue a resize render after the explicit
            # draw. Wait for that current image, not merely the first swap.
            while (not presented or surface.presentedGeneration != canvas.frame_generation
                   or canvas.backend._timer.isActive() or surface._size_timer.isActive()) and time.monotonic() < deadline:
                settle(10)
            self.assertTrue(presented, "No native Quick frameSwapped followed the painted frame")
            self.assertEqual(surface.presentedGeneration, canvas.frame_generation)
            self.assertEqual(surface.paintedGeneration, canvas.frame_generation)
            image = window.grabWindow()
            self.assertFalse(image.isNull())
            self.assertGreater(float(pixels(image).std()), 10)
            self.assertEqual(canvas.image.size(), QSize(round(320 * window.devicePixelRatio()),
                                                       round(240 * window.devicePixelRatio())))
            np.testing.assert_array_equal(pixels(image), pixels(canvas.image))
            measurements["320x240_draw_to_native_frame_ms"] = round((presented[0][1] - started) * 1000, 3)
            measurements["320x240_agg_draw_and_image_publish_ms"] = round(canvas.last_render_ms, 3)
            measurements["native_window_dpr"] = window.devicePixelRatio()
            routed = []
            for name in ("button_press_event", "motion_notify_event", "button_release_event", "key_press_event", "key_release_event"):
                canvas.mpl_connect(name, lambda event: routed.append(event.name))
            QTest.mousePress(window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(80, 60))
            QTest.mouseMove(window, QPoint(100, 80))
            QTest.mouseRelease(window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(100, 80))
            QTest.keyClick(window, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
            for name in ("button_press_event", "motion_notify_event", "button_release_event", "key_press_event", "key_release_event"):
                self.assertIn(name, routed)
            surface.close()
            surface.setParentItem(None)
            surface.deleteLater()
            window.close()
            window.deleteLater()
            engine.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            settle()
            self.assertFalse(canvas.closed, "An externally supplied canvas belongs to its caller")

        def test_cleanup_cancels_pending_draw_and_releases_figure(self):
            canvas, axes = self.make_canvas()
            surface = self.make_surface(canvas)
            close_events = []
            canvas.mpl_connect("close_event", close_events.append)
            figure_ref = weakref.ref(canvas.figure)
            canvas.draw_idle()
            canvas.close()
            canvas.close()
            settle()
            self.assertEqual(canvas.draw_count, 0)
            self.assertEqual(len(close_events), 1)
            self.assertIsNone(surface.canvas)
            self.assertIsNone(surface.backend)
            self.assertIsNone(canvas.figure)
            self.assertTrue(canvas.image.isNull())
            self.assertFalse(canvas.backend._timer.isActive())
            # Close events retain the canvas, never its detached Figure.
            del axes
            gc.collect()
            self.assertIsNone(figure_ref())
            with self.assertRaises(RuntimeError):
                canvas.draw()
            with self.assertRaises(ValueError):
                surface.set_canvas(canvas)

        def test_owned_canvas_closes_on_native_item_destruction(self):
            surface = PlotSurface()
            figure = Figure()
            canvas = surface.set_figure(figure)
            surface.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            settle()
            self.assertTrue(canvas.closed)
            self.assertIsNone(figure.canvas)

        def test_window_destruction_disposes_its_owned_surface_canvas(self):
            window = QQuickWindow()
            surface = PlotSurface(window.contentItem())
            canvas = surface.set_figure(Figure())
            with patch("sys.excepthook") as unhandled:
                window.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                settle()
            unhandled.assert_not_called()
            self.assertTrue(canvas.closed)

        def test_window_migration_disconnects_old_frame_and_screen_signals(self):
            canvas, _ = self.make_canvas()
            surface = self.make_surface(canvas)
            old_window, new_window = QQuickWindow(), QQuickWindow()
            surface.setParentItem(old_window.contentItem())
            surface._painted_generation = 1
            old_window.frameSwapped.emit()
            self.assertEqual(surface.presentedGeneration, 1)

            surface.setParentItem(new_window.contentItem())
            surface._painted_generation = 2
            old_window.frameSwapped.emit()
            self.assertEqual(surface.presentedGeneration, 1)
            new_window.frameSwapped.emit()
            self.assertEqual(surface.presentedGeneration, 2)
            surface._size_pending = False
            old_window.screenChanged.emit(None)
            self.assertFalse(surface._size_pending)
            new_window.screenChanged.emit(None)
            self.assertTrue(surface._size_pending)

            surface.setParentItem(None)
            old_window.deleteLater()
            new_window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

        def test_detach_handles_deleted_signal_source_with_a_valid_window_wrapper(self):
            canvas, _ = self.make_canvas()
            surface = self.make_surface(canvas)
            window, signal_source = QQuickWindow(), QQuickWindow()
            # Model the teardown interval where the window wrapper is valid but
            # its cached SignalInstances have already lost their native source.
            window.frameSwapped = signal_source.frameSwapped
            window.screenChanged = signal_source.screenChanged
            surface.setParentItem(window.contentItem())
            signal_source.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.assertTrue(isValid(window))
            self.assertFalse(isValid(signal_source))
            with self.assertRaisesRegex(RuntimeError, "Signal source has been deleted"):
                window.frameSwapped.disconnect(surface._frame_swapped)
            with patch("sys.excepthook") as unhandled:
                surface.setParentItem(None)
                settle()
            unhandled.assert_not_called()
            self.assertIsNone(surface._window)
            self.assertEqual(surface._window_connections, ())
            self.assertFalse(canvas.closed)
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

        def test_qml_engine_destruction_disposes_surface_without_callback_errors(self):
            register_plot_surface()
            engine = QQmlApplicationEngine()
            engine.loadData(QByteArray(b'import QtQuick\nimport QtQuick.Window\nimport Dptk 1.0\n'
                b'Window { visible: true; width: 320; height: 240; '
                b'PlotSurface { objectName: "surface"; anchors.fill: parent } }'))
            self.assertTrue(engine.rootObjects())
            window = engine.rootObjects()[0]
            surface = window.findChild(PlotSurface, "surface")
            canvas = surface.set_figure(Figure())
            settle()
            with patch("sys.excepthook") as unhandled:
                engine.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                settle()
            unhandled.assert_not_called()
            self.assertFalse(isValid(window))
            self.assertFalse(isValid(surface))
            self.assertTrue(canvas.closed)

        def test_direct_parent_constructor_reports_native_frame_presentation(self):
            window = QQuickWindow()
            window.resize(320, 240)
            surface = PlotSurface(window.contentItem())
            surface.setWidth(320)
            surface.setHeight(240)
            canvas, _ = self.make_canvas()
            surface.set_canvas(canvas)
            window.show()
            canvas.draw()
            deadline = time.monotonic() + 3
            while surface.presentedGeneration != canvas.frame_generation and time.monotonic() < deadline:
                settle(10)
            self.assertIs(surface._window, window)
            self.assertEqual(surface.presentedGeneration, canvas.frame_generation)
            self.assertGreater(surface.presentedGeneration, 0)
            window.close()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            settle()

        def test_import_does_not_load_a_widget_or_qtagg_backend(self):
            for module in ("PySide6.QtWidgets", "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_qt"):
                self.assertNotIn(module, production_imports)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SurfaceCases))
    print("QUICK_PLOT_MEASUREMENTS=" + json.dumps(measurements, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
