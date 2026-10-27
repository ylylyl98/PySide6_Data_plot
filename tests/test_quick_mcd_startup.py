"""Cold MCD controls must work before scientific dependencies are imported.

Every case owns a fresh process, temporary INI and experiment folder. The live
cases load Application.qml and wait for a frame after the workflow's controls
and empty workspace become visible; elapsed time alone never passes a case.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickMcdStartupTests(unittest.TestCase):
    def run_case(self, case):
        with TemporaryDirectory(prefix="quick-mcd-startup-") as directory:
            environment = dict(
                os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                QT_SCALE_FACTOR="1", MPLBACKEND="agg", MPLCONFIGDIR=directory,
                LOCALAPPDATA=directory, XDG_CONFIG_HOME=directory,
                XDG_CACHE_HOME=directory, QML_DISK_CACHE_PATH=directory,
                PYTHONIOENCODING="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--case", case],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=90,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_MCD_STARTUP_OK=" + case, result.stdout)

    def test_settings_defaults_and_legacy_import_are_compatible(self):
        self.run_case("settings")

    def test_standalone_mcd_bridge_without_scientific_imports(self):
        self.run_case("bridge-mcd")

    def test_standalone_peak_bridge_without_scientific_imports(self):
        self.run_case("bridge-mcd_peak_shift")

    def test_first_mcd_frame_and_empty_controls_without_scientific_imports(self):
        self.run_case("qml-mcd")

    def test_first_peak_frame_and_empty_controls_without_scientific_imports(self):
        self.run_case("qml-mcd_peak_shift")

    def test_source_catalog_and_processed_metadata_without_scientific_imports(self):
        self.run_case("catalog")

    def test_first_real_load_imports_science_on_a_worker_while_ui_events_continue(self):
        self.run_case("first-load")


def _isolated(case):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import importlib.abc
    import json
    import time

    scientific = {"numpy", "pandas", "scipy", "matplotlib"}
    attempts = []

    class NoScientificImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in scientific or fullname in {
                "PySide6.QtWidgets", "PySide6.QtQuickWidgets",
            }:
                attempts.append(fullname)
                raise AssertionError("Cold MCD controls attempted to import " + fullname)

    blocker = NoScientificImports()
    sys.meta_path.insert(0, blocker)

    def no_science():
        assert not attempts, attempts
        assert not scientific.intersection(sys.modules), scientific.intersection(sys.modules)

    if case == "settings":
        from dataclasses import FrozenInstanceError, asdict, fields, replace
        from core.mcd_settings import McdSettings
        defaults = dict(
            pos_angle=None, neg_angle=None, max_sequence_gap=3, max_delta_b=.1,
            pair_b_alignment="direct", zero_window_t=.02, reference_mode="nearest",
            bin_decimals=3, gain_mode="per_wavelength", gain_smooth_window=15,
            gain_smooth_poly=2, correction_mode="pair_spectral", spectral_order=2,
            background_ranges_ev=(), background_selection="auto",
            suggestion_protected_ranges_ev=(), manual_protected_ranges_ev=(),
            suggestion_linear_validation_rms=None, suggestion_quadratic_validation_rms=None,
            suggestion_algorithm=None, dark_pos_file=None, dark_neg_file=None,
        )
        original = McdSettings()
        assert asdict(original) == defaults
        assert [field.name for field in fields(McdSettings)] == list(defaults)
        updated = replace(original, spectral_order=1, background_ranges_ev=((1.5, 1.6),))
        assert updated.spectral_order == 1 and original.spectral_order == 2
        try:
            original.spectral_order = 1
        except FrozenInstanceError:
            pass
        else:
            raise AssertionError("McdSettings must remain frozen")
        no_science()
        sys.meta_path.remove(blocker)
        from core.mcd import McdSettings as LegacyMcdSettings
        from core.mcd import discover_mcd_processing_status
        from core.data_io import list_mcd_csv_files
        from core import mcd_catalog
        import pickle
        assert LegacyMcdSettings is McdSettings
        assert discover_mcd_processing_status is mcd_catalog.discover_mcd_processing_status
        assert list_mcd_csv_files is mcd_catalog.list_mcd_csv_files
        assert pickle.loads(b"ccore.mcd\nMcdSettings\n.") is McdSettings
        assert pickle.loads(pickle.dumps(updated)) == updated
        print("QUICK_MCD_STARTUP_OK=" + case)
        return

    if case == "catalog":
        from ui_qt.quick.mcd_workflow import catalog_mcd
        with TemporaryDirectory(prefix="quick-mcd-catalog-") as temporary:
            root = Path(temporary)
            (root / "McD/nested").mkdir(parents=True)
            for name in ("new.csv", "nested/processed.csv"):
                # Cataloging must use names and metadata, not parse sweep data.
                (root / "McD" / name).write_text("not a measurement", encoding="utf-8")
            (root / "ignored-root.csv").write_text("not a measurement", encoding="utf-8")
            destination = root / "Processed Data/MCD"
            destination.mkdir(parents=True)
            (destination / "processed_MCD_settings.json").write_text(json.dumps({
                "workflow": "mcd", "source_relative_path": "McD/nested/processed.csv",
                "created_utc": "2026-10-24T12:00:00+00:00",
            }), encoding="utf-8")
            rows = {row["name"]: row for row in catalog_mcd(str(root))}
            assert set(rows) == {"McD/new.csv", "McD/nested/processed.csv"}, rows
            assert rows["McD/new.csv"]["status"] == "new", rows
            assert rows["McD/nested/processed.csv"]["status"] == "processed", rows
            assert rows["McD/nested/processed.csv"]["processedAt"] == "2026-10-24T12:00:00+00:00"
            custom = root / "custom-export"
            custom.mkdir()
            (custom / "custom_MCD_settings.json").write_text(json.dumps({
                "workflow": "mcd", "source_relative_path": "McD/new.csv",
                "created_utc": "2026-10-24T13:00:00+00:00",
                "mcd_b": {"center_ev": 1.7, "width_mev": 5.},
            }), encoding="utf-8")
            rows = {row["name"]: row for row in catalog_mcd(str(root), history_roots=(str(custom),))}
            assert rows["McD/new.csv"]["status"] == "processed", rows
            assert rows["McD/new.csv"]["processedAt"] == "2026-10-24T13:00:00+00:00"
            legacy = root / "legacy"
            legacy.mkdir()
            (legacy / "sweep2.csv").write_text("not a measurement", encoding="utf-8")
            (legacy / "sweep10.csv").write_text("not a measurement", encoding="utf-8")
            from core.mcd_catalog import list_mcd_csv_files
            assert list_mcd_csv_files(str(legacy)) == ["sweep2.csv", "sweep10.csv"]
            assert {row["name"] for row in catalog_mcd(str(legacy))} == {"sweep2.csv", "sweep10.csv"}
        no_science()
        print("QUICK_MCD_STARTUP_OK=" + case)
        return

    from PySide6.QtCore import (Q_ARG, QCoreApplication, QEvent, QMetaObject,
                               QObject, QSettings, Qt, QUrl, qInstallMessageHandler)
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from ui_qt.quick.plot_surface import register_plot_surface
    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    if sys.platform == "win32":
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
        app.setFont(QFont("Segoe UI", 9))
    register_plot_surface()
    errors = []

    def qt_message(kind, context, message):
        if any(marker in message for marker in (
            "ReferenceError", "TypeError", "Binding loop", "Cannot assign", "Unable to assign", "Error:",
        )):
            errors.append(str(message))

    def python_error(kind, value, trace):
        import traceback
        errors.append("".join(traceback.format_exception(kind, value, trace)))
        traceback.print_exception(kind, value, trace)

    qInstallMessageHandler(qt_message)
    sys.excepthook = python_error

    def drain():
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def wait(predicate, timeout=15, *, allow_science=False):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            drain()
            assert not errors, errors
            if not allow_science:
                no_science()
            if predicate():
                return
            time.sleep(.001)
        raise AssertionError("Timed out waiting for cold MCD control/frame state")

    def idle(child):
        timers = (child._publish_timer, child._load_timer, child._window_timer,
                  child._feature_timer, child._review_timer, child._plot_timer)
        wait(lambda: not any(timer.isActive() for timer in timers)
             and not any(child.snapshot[key] for key in ("busy", "catalogBusy", "analysisBusy")))
        assert not child.snapshot["error"], child.snapshot["error"]
        assert child.plotBackend is None
        assert child.reviewBackend is None
        assert not child.snapshot["inspectionVisible"]

    workflow = "mcd" if case == "first-load" else case.split("-", 1)[1]
    peak = workflow == "mcd_peak_shift"
    with TemporaryDirectory(prefix="quick-mcd-empty-") as temporary:
        root = Path(temporary)
        experiment = root / "empty-experiment"
        experiment.mkdir()
        settings = QSettings(str(root / "settings.ini"), QSettings.IniFormat)
        engine = window = None
        if case.startswith("bridge-") or case == "first-load":
            if peak:
                from ui_qt.quick.mcd_peak_bridge import McdPeakBridge
                bridge = child = McdPeakBridge(settings=settings)
            else:
                from ui_qt.quick.mcd_bridge import McdBridge
                bridge = child = McdBridge(settings=settings)
            child.setExperimentFolder(str(experiment))
        else:
            from ui_qt.quick.application_bridge import QuickApplicationBridge
            bridge = QuickApplicationBridge(settings=settings)
            bridge.setSourcePath(str(experiment))
            engine = QQmlApplicationEngine()
            engine.setInitialProperties({"appBridge": bridge})
            engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / "ui_qt/quick/qml/Application.qml")))
            assert engine.rootObjects(), errors
            window = engine.rootObjects()[0]
            window.setWidth(1320)
            window.setHeight(900)
            frames = []
            window.frameSwapped.connect(lambda: frames.append(time.perf_counter()))
            wait(lambda: bool(frames))
            bridge.selectWorkflow(workflow)
            child = bridge._workflow_bridges[workflow]
            prefix = "quick_mcd_peak" if peak else "quick_mcd"

            def find_item(name):
                # Repeater delegates can have a different QObject ownership
                # tree; inspect the actual visual descendants displayed here.
                pending = [window.contentItem()]
                while pending:
                    item = pending.pop()
                    if item.objectName() == name:
                        return item
                    pending.extend(item.childItems())
                return None

            def controls_ready():
                items = [find_item(prefix + suffix) for suffix in ("_parameters", "_workspace")]
                return all(item is not None and item.property("visible")
                           and item.property("width") > 0 and item.property("height") > 0 for item in items) and find_item("app_parameter_historyFilter") is not None

            wait(controls_ready)
            count = len(frames)
            window.update()
            wait(lambda: len(frames) > count)
            assert not window.grabWindow().isNull()

            # Emit the actual QML editor signals so their handlers and snapshot
            # bindings participate in the test, including processing choices.
            def choose(key, index):
                control = find_item("app_parameter_" + key)
                assert control is not None, key
                assert QMetaObject.invokeMethod(control, "activated", Qt.DirectConnection, Q_ARG(int, index))

            choose("historyFilter", 1)
            choose("gain_mode", 2)
            choose("spectral_order", 0)
            assert child.snapshot["values"]["historyFilter"] == "new"
            assert child.snapshot["values"]["gain_mode"] == "scalar"
            assert child.snapshot["values"]["spectral_order"] == 1
            if peak:
                choose("trackerMethod", 2)
                assert child.snapshot["values"]["trackerMethod"] == "Local mixed fit"

        if case == "first-load":
            import csv
            import math
            import threading
            from PySide6.QtCore import QTimer
            idle(child)
            source = experiment / "mcd/synthetic.csv"
            source.parent.mkdir()
            energy = [1.5 + index * .003 for index in range(121)]
            fields = [-2 + index * .5 for index in range(9)] + [2 - index * .5 for index in range(1, 9)]
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["B_T", "angle_deg", *[1239.841984 / value for value in energy]])
                for field in fields:
                    for angle, sign, gain in ((90, 1, 1.), (0, -1, 1.08)):
                        writer.writerow([field, angle, *[
                            gain * (10000 + 1800 * math.exp(-((value - 1.7 - sign * .001 * field) / .009) ** 2)
                                    + 300 * math.exp(-((value - 1.78) / .014) ** 2)) for value in energy]])
            main_thread = threading.get_ident()
            imported = []
            products_ready = []
            heartbeats = []

            class ImportThreads(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname in scientific:
                        imported.append((fullname, threading.get_ident(), time.perf_counter()))

            no_science()
            sys.meta_path.remove(blocker)
            recorder = ImportThreads()
            sys.meta_path.insert(0, recorder)
            heartbeat = QTimer()
            heartbeat.setInterval(10)
            heartbeat.timeout.connect(lambda: heartbeats.append((time.perf_counter(), child.snapshot["busy"])))
            child.productsChanged.connect(lambda value: products_ready.append(time.perf_counter()))
            heartbeat.start()
            try:
                child.openFiles("sources", [QUrl.fromLocalFile(str(source))])
                wait(lambda: child._products is not None and child.snapshot["windowReady"]
                     and not any(child.snapshot[key] for key in ("busy", "catalogBusy", "analysisBusy")),
                     timeout=60, allow_science=True)
                assert not child.snapshot["error"], child.snapshot["error"]
                assert {entry[0] for entry in imported} == scientific, imported
                assert all(thread != main_thread for name, thread, when in imported), imported
                assert products_ready, "The first MCD products never completed"
                first_import = min(entry[2] for entry in imported)
                assert any(first_import <= when < products_ready[0] and busy for when, busy in heartbeats), heartbeats
                assert child._products.result.pair_b.size > 0
            finally:
                heartbeat.stop()
                bridge.close()
                drain()
                sys.meta_path.remove(recorder)
            print("QUICK_MCD_STARTUP_OK=" + case)
            return

        try:
            no_science()
            idle(child)
            assert child._request().settings.background_ranges_ev == ()
            child._values["background_ranges_ev"] = "1.5 1.6; 1.8 1.9"
            assert child._request().settings.background_ranges_ev == ((1.5, 1.6), (1.8, 1.9))
            for invalid in ("nan 2", "1 inf", "2 1", "1 1", "1 2 3"):
                child._values["background_ranges_ev"] = invalid
                try:
                    child._request()
                except ValueError:
                    pass
                else:
                    raise AssertionError("Accepted invalid background ranges: " + invalid)
            child._values["background_ranges_ev"] = ""
            child.setPalette({"inset": "#111827", "text": "#f9fafb"})
            child.setParameter("center", 1.7)
            child.setParameter("width", 7.)
            child.setParameter("background_ranges_ev", "1.5 1.6; 1.8 1.9")
            child.setParameter("gain_mode", "scalar")
            child.setParameter("heatmapFraction", .6)
            child.setParameter("spectrumVisible", False)
            if peak:
                for method in ("Second derivative", "Local mixed fit", "Raw spectrum"):
                    child.setParameter("trackerMethod", method)
                child.setParameter("peakProminence", .05)
                child.setParameter("mapMode", "Second derivative")
                child.setParameter("showPeakMaps", True)
                child.invoke("analyze-peaks")
            else:
                child.setParameter("featureMethod", "Second derivative")
                child.setParameter("featureMethod", "Local mixed fit")
                child.invoke("analyze-feature")
            idle(child)
            for action in ("clear-files", "clear-dark-pos", "clear-dark-neg", "recalculate", "refresh-catalog"):
                child.invoke(action)
            idle(child)
            if window is not None:
                count = len(frames)
                window.update()
                wait(lambda: len(frames) > count)
                assert controls_ready()
                assert not window.grabWindow().isNull()
            no_science()
            assert not errors, errors
        finally:
            if window is not None:
                window.close()
            bridge.close()
            if engine is not None:
                engine.deleteLater()
            drain()
        assert not errors, errors
        no_science()
    print("QUICK_MCD_STARTUP_OK=" + case)


if __name__ == "__main__":
    if "--case" in sys.argv:
        _isolated(sys.argv[sys.argv.index("--case") + 1])
    else:
        unittest.main()
