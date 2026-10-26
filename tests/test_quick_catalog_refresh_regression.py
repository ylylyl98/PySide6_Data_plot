"""Source picker refresh stability in a separate Qt process and temp folder."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickCatalogRefreshRegressionTests(unittest.TestCase):
    def test_isolated_catalog_refresh_regressions(self):
        with TemporaryDirectory(prefix="catalog-refresh-process-") as directory:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                MPLBACKEND="agg", MPLCONFIGDIR=directory, XDG_CONFIG_HOME=directory,
                XDG_CACHE_HOME=directory, LOCALAPPDATA=directory, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_CATALOG_REFRESH_TESTS=", result.stdout)
        print(result.stdout.strip())


def _run_isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import json
    import time
    from unittest.mock import patch
    from PySide6.QtCore import QCoreApplication, QSettings
    from ui_qt.quick import source_catalog
    import core.drr_sources as sources
    import core.drr_catalog as layers

    app = QCoreApplication([])

    def wait(predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(.003)
        raise AssertionError("Catalog state did not settle")

    def settle(seconds=.85):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.003)

    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="catalog-refresh-case-")
            self.addCleanup(self.temporary.cleanup)
            self.base = Path(self.temporary.name)
            self.root = self.base / "experiment"
            self.raw = self.root / "Initial Data" / "REF"
            self.history = self.root / "Processed Data" / "DRR"
            self.raw.mkdir(parents=True)
            self.history.mkdir(parents=True)
            self.measurement = self.write_measurement("measurement_700nmc_rep01.csv")
            self.bridge = source_catalog.SourceCatalogBridge(settings=QSettings(
                str(self.base / "settings.ini"), QSettings.IniFormat))
            self.addCleanup(self.bridge.close)

        def write_measurement(self, name):
            path = self.raw / name
            path.write_text("Vbg,700,701\n0,1,2\n1,2,3\n", encoding="utf-8")
            return path

        def open(self):
            self.bridge.open("drr", str(self.root), [], False)
            wait(lambda: not self.bridge.snapshot["busy"])
            self.assertEqual(self.bridge.snapshot["error"], "")

        def test_cache_writes_and_generated_outputs_do_not_invalidate_raw_sources(self):
            # Deliberately put caches within the experiment so their atomic
            # writes cannot accidentally be treated as acquisition changes.
            cache_path = self.root / "cache" / "inspection.json"
            before = source_catalog._catalog_inventory(self.root, "drr")
            with patch.object(sources, "inspect_csv_gate", wraps=sources.inspect_csv_gate) as inspect, \
                    patch.object(layers, "discover_drr_sources", wraps=layers.discover_drr_sources) as discover:
                original = source_catalog.scan_catalog(self.root, "drr", cache_path=cache_path)
                self.assertEqual(inspect.call_count, 1)
                self.assertEqual(discover.call_count, 1)
                (self.history / "preview.png").write_bytes(b"generated PNG")
                (self.history / "unlinked.dat").write_text("export without metadata", encoding="utf-8")
                refreshed = source_catalog.scan_catalog(self.root, "drr", cache_path=cache_path)
                self.assertEqual(refreshed, original)
                self.assertEqual(inspect.call_count, 1)
                self.assertEqual(discover.call_count, 1)
            self.assertEqual(source_catalog._catalog_inventory(self.root, "drr"), before)

        def test_idle_picker_and_generated_png_event_do_not_restart_validation(self):
            self.open()
            generation = self.bridge._generation
            (self.history / "preview.png").write_bytes(b"generated PNG")
            self.bridge._directory_changed(str(self.history))
            settle()
            self.assertEqual(self.bridge._generation, generation)
            self.assertFalse(self.bridge.snapshot["busy"])
            self.assertTrue(self.bridge._catalog_validated)

        def test_failed_watch_inventory_keeps_last_validated_catalog(self):
            self.open()
            self.bridge._watch_timer.stop()
            generation = self.bridge._generation
            rows = list(self.bridge.filesModel.rows)
            with patch.object(self.bridge, "_source_inventory", side_effect=PermissionError("transient watch read")):
                self.bridge._watch_refresh()
                wait(lambda: not self.bridge._watch_probe_running)
                self.assertEqual(self.bridge._generation, generation,
                    "An unreadable watch probe must not start another visible refresh")
                self.assertFalse(self.bridge.snapshot["busy"])
                self.assertTrue(self.bridge._catalog_validated)
                self.assertEqual(self.bridge.filesModel.rows, rows)

        def test_new_measurement_refreshes_once_and_then_settles(self):
            self.open()
            generation = self.bridge._generation
            path = self.write_measurement("measurement_700nmc_rep02.csv")
            self.bridge._directory_changed(str(self.raw))
            wait(lambda: any(row["id"] == str(path) for row in self.bridge._catalog["files"])
                and not self.bridge.snapshot["busy"])
            self.assertEqual(self.bridge._generation, generation + 1)
            settle()
            self.assertEqual(self.bridge._generation, generation + 1)

        def test_acquisition_appends_wait_for_settling_without_losing_selection(self):
            self.open()
            self.bridge._watch_timer.stop()
            self.bridge._watch_settle_ms = 600
            self.bridge.selectFile(str(self.measurement), True)
            generation = self.bridge._generation
            busy_transitions = []
            self.bridge.snapshotChanged.connect(lambda: busy_transitions.append(self.bridge.snapshot["busy"]))
            with patch.object(self.bridge, "_scan_catalog", wraps=self.bridge._scan_catalog) as scan:
                # Each acquisition burst is farther apart than the original
                # 350ms directory debounce. The picker must remain usable.
                for index in range(7):
                    with self.measurement.open("a", encoding="utf-8") as handle:
                        handle.write(f"{index + 2},2,3\n")
                    self.bridge._directory_changed(str(self.raw))
                    settle(.4)
                    self.assertEqual(self.bridge._generation, generation,
                        "An active acquisition must not restart full catalog inspection")
                    self.assertFalse(self.bridge.snapshot["busy"])
                    self.assertTrue(self.bridge._catalog_validated)
                    self.assertTrue(self.bridge.snapshot["canAccept"])
                    self.assertEqual(scan.call_count, 0)
                self.assertNotIn(True, busy_transitions)
                wait(lambda: self.bridge._generation == generation + 1 and not self.bridge.snapshot["busy"])
                self.assertEqual(scan.call_count, 1)
                self.assertEqual(self.bridge.snapshot["selected"], [str(self.measurement)])
                self.assertTrue(self.bridge.snapshot["canAccept"])
                settle()
                self.assertEqual(self.bridge._generation, generation + 1)
                self.assertEqual(scan.call_count, 1)

        def test_reopening_warm_picker_does_not_read_measurement_contents(self):
            self.open()
            self.bridge.invoke("cancel")
            with patch.object(sources, "inspect_csv_gate", side_effect=AssertionError("reopened source contents")):
                self.open()
                self.assertEqual(len(self.bridge._catalog["sources"]), 1)

        def test_manual_refresh_bypasses_pending_acquisition_settling(self):
            self.open()
            self.bridge._watch_timer.stop()
            self.bridge._watch_settle_ms = 5000
            generation = self.bridge._generation
            with self.measurement.open("a", encoding="utf-8") as handle:
                handle.write("2,3,4\n")
            self.bridge._watch_refresh()
            wait(lambda: self.bridge._watch_candidate is not None and not self.bridge._watch_probe_running)
            self.assertEqual(self.bridge._generation, generation)
            with patch.object(self.bridge, "_scan_catalog", wraps=self.bridge._scan_catalog) as scan:
                self.bridge.invoke("refresh")
                self.assertEqual(self.bridge._generation, generation + 1)
                wait(lambda: not self.bridge.snapshot["busy"])
                self.assertEqual(scan.call_count, 1)
                self.assertIsNone(self.bridge._watch_candidate)
                self.assertEqual(self.bridge._catalog["sources"][0].frame_count, 3)
                settle()
                self.assertEqual(self.bridge._generation, generation + 1)
                self.assertEqual(scan.call_count, 1)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("QUICK_CATALOG_REFRESH_TESTS=" + json.dumps(dict(tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors))))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
