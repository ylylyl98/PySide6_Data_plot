"""Cold SHG controls and request capture must leave scientific imports to workers."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickShgStartupTests(unittest.TestCase):
    def run_case(self, case):
        with TemporaryDirectory(prefix="quick-shg-startup-") as temporary:
            environment = dict(
                os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                MPLBACKEND="agg", MPLCONFIGDIR=temporary, LOCALAPPDATA=temporary,
                XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary,
                PYTHONIOENCODING="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--case", case],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=90,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_SHG_STARTUP_OK=" + case, result.stdout)

    def test_settings_defaults_serialization_and_legacy_imports(self):
        self.run_case("settings")

    def test_empty_processing_edits_do_not_import_scientific_libraries(self):
        self.run_case("empty")

    def test_load_request_capture_does_not_import_scientific_libraries(self):
        self.run_case("request")

    def test_first_source_imports_numpy_and_pandas_off_the_gui_thread(self):
        self.run_case("first-load")


def _isolated(case):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import importlib.abc
    import time

    scientific = {"numpy", "pandas", "scipy", "matplotlib"}
    attempts = []

    class NoScientificImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in scientific:
                attempts.append(fullname)
                raise AssertionError("Cold SHG controls imported " + fullname)

    blocker = NoScientificImports()
    sys.meta_path.insert(0, blocker)

    def no_science():
        assert not attempts, attempts
        assert not scientific.intersection(sys.modules), scientific.intersection(sys.modules)

    if case == "settings":
        from dataclasses import FrozenInstanceError, asdict, replace
        from core.shg_settings import ShgSettings, ShgFitSettings
        defaults = dict(
            peak_center_nm=515., gate_min_nm=513., gate_max_nm=517.,
            left_min_nm=508., left_max_nm=512., right_min_nm=518., right_max_nm=522.,
            background_method="local_linear", sigma_clip=3., remove_cosmic_rays=True,
            cosmic_threshold_mad=8., cosmic_window_points=7, cosmic_max_width_points=3,
            angle_scale=1., angle_offset_deg=0., angle_wrap_deg=None, include_failed_rows=False,
        )
        settings = ShgSettings()
        assert asdict(settings) == defaults
        assert settings.to_dict() == dict(defaults, integration_wavelength_nm=515.,
            integration_half_range_nm=2., integration_interval_nm=[513., 517.],
            integration_source="cosmic_cleaned_background_subtracted_spectrum",
            left_sideband_gap_nm=1., right_sideband_gap_nm=1.,
            left_sideband_width_nm=4., right_sideband_width_nm=4.)
        assert asdict(ShgFitSettings()) == dict(enabled=True, angle_min_deg=None,
            angle_max_deg=None, use_uncertainty_weights=True, include_excluded_rows=False,
            minimum_points=6, minimum_span_deg=45., phase_branch=0)
        assert ShgFitSettings().to_dict() == asdict(ShgFitSettings())
        assert replace(settings, remove_cosmic_rays=False).to_dict()["integration_source"] == "background_subtracted_spectrum"
        try:
            settings.peak_center_nm = 520.
        except FrozenInstanceError:
            pass
        else:
            raise AssertionError("SHG request settings must remain immutable")
        no_science()
        sys.meta_path.remove(blocker)
        from core.shg import ShgSettings as LegacySettings
        from core.shg_fit import ShgFitSettings as LegacyFitSettings
        import pickle
        assert LegacySettings is ShgSettings
        assert LegacyFitSettings is ShgFitSettings
        assert pickle.loads(b"ccore.shg\nShgSettings\n.") is ShgSettings
        assert pickle.loads(b"ccore.shg_fit\nShgFitSettings\n.") is ShgFitSettings
        assert pickle.loads(pickle.dumps(settings)) == settings
        print("QUICK_SHG_STARTUP_OK=" + case)
        return

    from PySide6.QtCore import QSettings, QTimer, QUrl
    from PySide6.QtGui import QGuiApplication
    from ui_qt.quick.shg_bridge import ShgBridge
    app = QGuiApplication([])

    with TemporaryDirectory(prefix="quick-shg-controls-") as temporary:
        root = Path(temporary)
        bridge = ShgBridge(settings=QSettings(str(root / "test.ini"), QSettings.IniFormat))
        try:
            # Exercise each setting family that captures a processing request.
            for key, value in (("peakCenter", 516.), ("halfRange", 2.),
                    ("backgroundMethod", "none"), ("cosmicEnabled", False),
                    ("angleScale", 1.1), ("fitEnabled", False), ("mode", "compare")):
                bridge.setParameter(key, value)
                app.processEvents()
                assert not bridge.snapshot["busy"], bridge.snapshot
                no_science()
            bridge.setParameter("mode", "single")

            if case == "request":
                bridge._sources["single"] = str(root / "measurement.csv")
                bridge.setParameter("peakCenter", 515.)
                no_science()
                request = bridge._requested
                assert request.settings.peak_center_nm == 515.
                assert request.settings.gate_min_nm == 513.
                assert request.settings.gate_max_nm == 517.
                assert not request.fit_settings.enabled
                assert request.sources == (("single", str(root / "measurement.csv")),)
            elif case == "first-load":
                import csv
                import math
                import threading
                source = root / "measurement.csv"
                wave = [507. + index * .05 for index in range(321)]
                with source.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["target_value", "measured_value", "move_ok", "acquisition_ok", *wave])
                    for index in range(19):
                        angle = index * 10.
                        area = 120 + 900 * math.cos(math.radians(2 * (angle - 17.5))) ** 2
                        writer.writerow([angle, angle, 1, 1, *[
                            800 + area * math.exp(-.5 * ((value - 515.) / .35) ** 2) for value in wave]])
                main_thread = threading.get_ident()
                imported, heartbeats, completed = [], [], []

                class ImportThreads(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname in {"numpy", "pandas"}:
                            imported.append((fullname, threading.get_ident(), time.perf_counter()))

                no_science()
                sys.meta_path.remove(blocker)
                recorder = ImportThreads()
                sys.meta_path.insert(0, recorder)
                heartbeat = QTimer()
                heartbeat.setInterval(10)
                heartbeat.timeout.connect(lambda: heartbeats.append((time.perf_counter(), bridge._busy)))
                bridge.operationCompleted.connect(lambda kind, elapsed: completed.append((kind, time.perf_counter())))
                heartbeat.start()
                try:
                    bridge.openFiles("single", [QUrl.fromLocalFile(str(source))])
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline:
                        app.processEvents()
                        if bridge._products is not None and not bridge._busy and not bridge._pending:
                            break
                        time.sleep(.001)
                    assert bridge._products is not None, bridge.snapshot["status"]
                    assert not bridge.snapshot["error"], bridge.snapshot["error"]
                    assert len(bridge._products.results[0].integrated_area) == 19
                    assert {name for name, thread, when in imported} == {"numpy", "pandas"}, imported
                    assert all(thread != main_thread for name, thread, when in imported), imported
                    ready = next(when for kind, when in completed if kind == "load")
                    first_import = min(when for name, thread, when in imported)
                    assert any(first_import <= when < ready and busy for when, busy in heartbeats), heartbeats
                finally:
                    heartbeat.stop()
                    sys.meta_path.remove(recorder)
        finally:
            bridge.close()
        if case != "first-load":
            no_science()
    print("QUICK_SHG_STARTUP_OK=" + case)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        _isolated(sys.argv[2])
    else:
        unittest.main()
