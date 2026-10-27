"""Measure the native Quick PL slice with isolated settings and synthetic data.

Run each policy in a fresh process. This is a same-slice comparison of committed
gate input to Agg completion and a generation-matched native Quick frame. It
excludes loading, analysis and export, and does not benchmark the whole app.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time


def _summary(values):
    if not values:
        return dict(count=0, min_ms=None, p50_ms=None, p95_ms=None, max_ms=None)
    ordered = sorted(values)
    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return dict(count=len(values), min_ms=min(values), p50_ms=percentile(.5),
                p95_ms=percentile(.95), max_ms=max(values))


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("incremental", "full"), required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--platform", choices=("offscreen", "native"), default="offscreen")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 2 <= args.samples <= 157:
        parser.error("samples must be between 2 and the synthetic cube's 157 rows")

    with TemporaryDirectory(prefix="dptk-native-quick-profile-") as temporary:
        root = Path(temporary)
        if args.platform == "offscreen":
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
        elif os.environ.get("QT_QPA_PLATFORM") in {"offscreen", "minimal"}:
            del os.environ["QT_QPA_PLATFORM"]
        os.environ.update(QT_QUICK_BACKEND="software", MPLBACKEND="agg",
                          MPLCONFIGDIR=str(root / "matplotlib"),
                          QML_DISABLE_DISK_CACHE="1", XDG_CONFIG_HOME=str(root / "config"),
                          XDG_CACHE_HOME=str(root / "cache"))

        # Load Qt before the scientific stack so its Windows DLL directories are
        # registered. No QApplication, QWidget or QtTest is needed by this run.
        from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QPointF, QSettings, QTimer, Qt, QUrl, qVersion
        from PySide6.QtGui import QGuiApplication, QImage, QKeyEvent
        from PySide6.QtQml import QQmlApplicationEngine
        from PySide6.QtQuick import QQuickItem
        import numpy as np

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from ui_qt.quick.application_bridge import QuickApplicationBridge
        from ui_qt.quick.plot_surface import PlotSurface, register_plot_surface

        repository = Path(__file__).resolve().parents[1]
        source_paths = ("ui_qt/quick/application_bridge.py", "ui_qt/quick/plot_surface.py",
                        "ui_qt/quick/figure_canvas.py",
                        "ui_qt/axes_region_blitter.py", "ui_qt/quick/qml/Application.qml",
                        "ui_qt/quick/qml/AppParameterPane.qml", "ui_qt/quick/qml/AppNumberField.qml")
        def source_digests():
            return {path: hashlib.sha256((repository / path).read_bytes()).hexdigest() for path in source_paths}
        sources_before = source_digests()
        imports_finished = time.perf_counter()
        app = QGuiApplication([sys.argv[0]])
        app.setQuitOnLastWindowClosed(False)
        settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
        settings.setValue("quick/appearance", "light")
        bridge = QuickApplicationBridge(settings=settings)
        register_plot_surface()
        engine = QQmlApplicationEngine()
        warnings = []
        engine.warnings.connect(lambda messages: warnings.extend(message.toString() for message in messages))
        engine.setInitialProperties({"appBridge": bridge, "width": 1320, "height": 900})
        qml = Path(__file__).resolve().parents[1] / "ui_qt" / "quick" / "qml" / "Application.qml"
        engine.load(QUrl.fromLocalFile(str(qml)))
        if not engine.rootObjects():
            bridge.close()
            raise RuntimeError("Native Quick shell did not load: " + "\n".join(warnings))
        window = engine.rootObjects()[0]
        window.requestActivate()

        def wait_until(predicate, timeout=10):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 5)
                if predicate():
                    return True
                time.sleep(.001)
            return False

        def item_named(name):
            item = window.findChild(QQuickItem, name)
            if item is not None:
                return item
            def visit(parent):
                for child in parent.childItems():
                    if child.objectName() == name:
                        return child
                    found = visit(child)
                    if found is not None:
                        return found
                return None
            return visit(window.contentItem())

        def image_array(image):
            image = image.convertToFormat(QImage.Format.Format_RGBA8888)
            return np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())[:, :image.width() * 4].copy()

        heartbeat = QTimer()
        heartbeat.setInterval(5)
        heartbeat.setTimerType(Qt.TimerType.PreciseTimer)
        heartbeat_times = []
        heartbeat.timeout.connect(lambda: heartbeat_times.append(time.perf_counter()))
        report = None
        try:
            bridge.loadDemo()
            surface = item_named("app_plot_surface")
            if not isinstance(surface, PlotSurface):
                raise RuntimeError("The shell did not instantiate the native PlotSurface")
            if not wait_until(lambda: bridge._canvas is not None
                              and surface.presentedGeneration == bridge._canvas.frame_generation
                              and surface.presentedGeneration > 0):
                raise RuntimeError("The initial synthetic plot did not reach native framePresented")
            canvas = bridge._canvas
            cube = bridge._cube
            if cube.Z.shape != (157, 1340):
                raise RuntimeError(f"Unexpected synthetic dimensions: {cube.Z.shape}")
            initial_presented = time.perf_counter()
            gate_field = item_named("app_parameter_gate")
            if gate_field is None:
                raise RuntimeError("The real QML gate editor was not created")
            # Bring the editor's clipped ScrollView region into view before
            # measuring. Draft typing and scrolling are outside commit latency.
            ancestor = gate_field.parentItem()
            while ancestor is not None:
                if ancestor.metaObject().indexOfProperty("contentY") >= 0:
                    current = float(ancestor.property("contentY"))
                    y = gate_field.mapToItem(ancestor, QPointF()).y()
                    extent = max(0., float(ancestor.property("contentHeight")) - ancestor.height())
                    ancestor.setProperty("contentY", min(extent, max(0., current + y - ancestor.height() * .35)))
                    break
                ancestor = ancestor.parentItem()
            gate_field.forceActiveFocus()
            wait_until(lambda: gate_field.hasActiveFocus(), timeout=2)

            active = None
            def on_agg_frame(_image, generation):
                if active is not None and generation > active["before_generation"]:
                    active["agg_events"].append((generation, time.perf_counter()))
            def on_presented(generation):
                if active is not None and generation > active["before_generation"]:
                    active["presented_events"].append((generation, time.perf_counter()))
            canvas.frameReady.connect(on_agg_frame)
            surface.framePresented.connect(on_presented)

            def send_key(code, text="", modifiers=Qt.KeyboardModifier.NoModifier):
                QCoreApplication.sendEvent(window, QKeyEvent(QEvent.Type.KeyPress, code, modifiers, text))
                QCoreApplication.sendEvent(window, QKeyEvent(QEvent.Type.KeyRelease, code, modifiers, text))

            def commit_gate(value, request_index):
                nonlocal active
                gate_field.forceActiveFocus()
                send_key(Qt.Key.Key_A, modifiers=Qt.KeyboardModifier.ControlModifier)
                draft = f"{value:.6f}"
                for character in draft:
                    send_key(ord(character.upper()), character)
                if str(gate_field.property("text")) != draft:
                    raise RuntimeError("Native key events did not enter the gate draft")
                if args.policy == "full":
                    for blitter in bridge._blitters:
                        # Invalidate only the cached backgrounds. Keep artists
                        # animated so both policies use identical interactive
                        # layer composition when draw_event paints them. The
                        # existing gate updater then queues one full Agg draw.
                        blitter.invalidate()
                active = dict(index=request_index, requested_gate=float(value),
                              before_generation=canvas.frame_generation,
                              agg_events=[], presented_events=[],
                              started=time.perf_counter())
                send_key(Qt.Key.Key_Return, "\r")
                request = active
                request["target_generation"] = request["agg_events"][-1][0] if request["agg_events"] else None
                superseded = False
                def completed():
                    nonlocal superseded
                    if not request["agg_events"] or canvas.backend._timer.isActive() or bridge._plot_timer.isActive():
                        return False
                    if request["target_generation"] is None:
                        request["target_generation"] = request["agg_events"][-1][0]
                    generation = request["target_generation"]
                    if any(value[0] == generation for value in request["presented_events"]):
                        return True
                    superseded = any(value[0] > generation for value in request["presented_events"])
                    return superseded
                finished = wait_until(completed)
                active = None
                if not finished or superseded:
                    return dict(index=request_index, requested_gate=float(value),
                                discarded=True, reason=("Request generation was coalesced before presentation" if superseded
                                                        else "No matching completed Agg generation was presented"),
                                target_generation=request["target_generation"],
                                agg_generations=[event[0] for event in request["agg_events"]],
                                presented_generations=[event[0] for event in request["presented_events"]])
                generation = request["target_generation"]
                agg_time = next(at for candidate, at in request["agg_events"] if candidate == generation)
                presented_time = next(at for candidate, at in request["presented_events"] if candidate == generation)
                actual_gate = float(bridge._values["gate"])
                expected_gate = float(cube.gate[np.argmin(abs(cube.gate - value))])
                if actual_gate != expected_gate:
                    raise RuntimeError(f"Gate input selected {actual_gate}, expected {expected_gate}")
                return dict(index=request_index, requested_gate=float(value), selected_gate=actual_gate,
                            generation=generation, agg_ms=(agg_time - request["started"]) * 1000,
                            presented_ms=(presented_time - request["started"]) * 1000,
                            agg_to_presented_ms=(presented_time - agg_time) * 1000,
                            intermediate_generations=len(request["agg_events"]) - 1,
                            discarded=False)

            warmups = [commit_gate(float(cube.gate[index]), -offset - 1)
                       for offset, index in enumerate((156, 39))]
            if any(row["discarded"] for row in warmups):
                raise RuntimeError("A gate warmup did not reach its matching presented frame")
            counts_before = (canvas.draw_count, canvas.blit_count, canvas.frame_generation)
            heartbeat_times.append(time.perf_counter())
            heartbeat.start()
            indices = np.rint(np.linspace(0, 156, args.samples)).astype(int)
            measured = [commit_gate(float(cube.gate[index]), offset) for offset, index in enumerate(indices)]
            heartbeat.stop()
            heartbeat_times.append(time.perf_counter())
            counts_after = (canvas.draw_count, canvas.blit_count, canvas.frame_generation)
            completed = [row for row in measured if not row["discarded"]]

            # Keep measurements separate from verification. A fresh rebuild
            # exercises the controller's normal complete scientific pipeline.
            def data_digest():
                digest = hashlib.sha256()
                for array in (cube.energy, cube.gate, cube.Z):
                    digest.update(np.asarray(array).tobytes())
                return digest.hexdigest()
            before_digest = data_digest()
            before_image = image_array(canvas.image)
            selected_gate = float(bridge._values["gate"])
            bridge._rebuild_plot()
            canvas.draw()
            reference_generation = canvas.frame_generation
            if not wait_until(lambda: surface.presentedGeneration == reference_generation):
                raise RuntimeError("The full scientific rebuild was not presented")
            reference_image = image_array(canvas.image)
            same_shape = before_image.shape == reference_image.shape
            different = int(np.count_nonzero(before_image != reference_image)) if same_shape else None
            fidelity = dict(reference="fresh QuickApplicationBridge._rebuild_plot plus complete Agg draw",
                            shape_equal=same_shape, differing_rgba_channels=different,
                            rgba_exact=bool(same_shape and different == 0),
                            data_sha256=before_digest, data_unchanged=before_digest == data_digest(),
                            selected_gate_unchanged=selected_gate == float(bridge._values["gate"]))
            heartbeat_gaps = [(right - left) * 1000 for left, right in zip(heartbeat_times, heartbeat_times[1:])]
            report = dict(scope="Native Quick PL inspection only; synthetic already-loaded 157x1340 cube; excludes file loading, analysis and export",
                          comparison="Compare policies only with this same PL slice, window, platform and machine; this is not a whole-application or prior DRR benchmark",
                          policy=args.policy, platform=args.platform, qt_platform=app.platformName(),
                          redraw_policy=("invalidate region backgrounds before gate commit; existing gate updater queues full Agg draw"
                                         if args.policy == "full" else "normal controller region blits"),
                          qt_quick_backend=os.environ["QT_QUICK_BACKEND"],
                          application_type=type(app).__name__, widgets_imported="PySide6.QtWidgets" in sys.modules,
                          runtime=dict(python=sys.version, qt=qVersion(), numpy=np.__version__,
                                       matplotlib=sys.modules["matplotlib"].__version__),
                          source_sha256=sources_before, source_unchanged_during_run=sources_before == source_digests(),
                          window_size=[window.width(), window.height()],
                          plot_logical_size=[surface.width(), surface.height()],
                          plot_physical_size=[canvas.image.width(), canvas.image.height()],
                          device_pixel_ratio=window.devicePixelRatio(), rows=157, columns=1340,
                          settings_isolated=Path(settings.fileName()).resolve().is_relative_to(root.resolve()),
                          caches_isolated=True, qml_warnings=warnings,
                          startup_ms=dict(imports=(imports_finished - started) * 1000,
                                          to_initial_presented=(initial_presented - started) * 1000),
                          gate=dict(control="Real QML app_parameter_gate; native Enter key commits the typed draft",
                                    input_count=args.samples, completed_count=len(completed),
                                    discarded_count=len(measured) - len(completed),
                                    agg_completion=_summary([row["agg_ms"] for row in completed]),
                                    native_frame_presented=_summary([row["presented_ms"] for row in completed]),
                                    agg_to_presented=_summary([row["agg_to_presented_ms"] for row in completed]),
                                    samples=measured),
                          rendering=dict(full_draws=counts_after[0] - counts_before[0],
                                         blits=counts_after[1] - counts_before[1],
                                         published_generations=counts_after[2] - counts_before[2]),
                          gui_heartbeat=dict(interval_ms=5, intervals=_summary(heartbeat_gaps)),
                          fidelity=fidelity)
        finally:
            heartbeat.stop()
            bridge.close()
            window.close()
            engine.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
            settings.sync()
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
        return 0 if (report["fidelity"]["rgba_exact"] and report["fidelity"]["data_unchanged"]
                     and report["source_unchanged_during_run"]
                     and report["fidelity"]["selected_gate_unchanged"] and report["gate"]["completed_count"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
