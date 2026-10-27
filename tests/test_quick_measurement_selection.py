"""Measurement catalogs reuse inspections and keep explicit choices responsive."""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class QuickMeasurementSelectionProcessTests(unittest.TestCase):
    def test_measurement_selection_in_isolated_gui_process(self):
        with TemporaryDirectory(prefix="quick-measurement-process-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                MPLCONFIGDIR=temporary, XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("MEASUREMENT_SELECTION_TESTS=", result.stdout)
        print(result.stdout.strip())


def _run_isolated():
    from PySide6.QtCore import QSettings, QUrl
    from PySide6.QtGui import QGuiApplication

    from core.drr_sources import DrrSourceCache
    from core.source_catalog_cache import SourceCatalogCache
    from ui_qt.quick.source_catalog import SourceCatalogBridge, scan_catalog

    class QuickMeasurementSelectionTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.app = QGuiApplication.instance() or QGuiApplication([])

        def setUp(self):
            self.temporary = TemporaryDirectory(prefix="quick-measurement-selection-")
            self.addCleanup(self.temporary.cleanup)
            self.base = Path(self.temporary.name).resolve()
            self.root = self.base / "experiment"
            self.raw = self.root / "Initial Data" / "REF"
            self.raw.mkdir(parents=True)
            self.cache = self.base / "cache" / "details.json"
            self.bridges = []
            self.addCleanup(self.close_bridges)

        def close_bridges(self):
            for bridge in self.bridges:
                bridge.close()

        def wait(self, predicate, timeout=15):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                self.app.processEvents()
                if predicate():
                    self.app.processEvents()
                    return
                time.sleep(.003)
            self.fail("Timed out waiting for measurement catalog")

        def csv(self, name="sample_730nmc_TG_rep01.csv", folder=None):
            path = (folder or self.raw) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Vbg,Vtg,730,731\n0,0,1,2\n1,2,3,4\n", encoding="utf-8")
            return path.resolve()

        def seed_inspections(self, *paths):
            cache = DrrSourceCache(self.cache, load_on_init=False)
            metadata = dict(gate_varies=True, frame_count=2, gate_grid=((0., 0.), (1., 2.)),
                spectral_grid=(730., 731.), gate_direction="Vbg increasing, Vtg increasing",
                gate_ranges=((0., 1.), (0., 2.)), gate_labels=("Vbg", "Vtg"), grid_complete=True,
                wavelength_center_nm=730., wavelength_center_source="filename")
            for path in paths:
                stat = path.stat()
                cache.put(str(path).casefold(), modified_ns=stat.st_mtime_ns, size_bytes=stat.st_size, metadata=metadata)
            cache.save()

        def scan(self, **options):
            return scan_catalog(self.root, "drr", cache_path=self.cache, **options)

        def bridge(self):
            bridge = SourceCatalogBridge(settings=QSettings(str(self.base / "settings.ini"), QSettings.IniFormat))
            self.bridges.append(bridge)
            return bridge

        def test_warm_scan_does_not_decode_or_reload_inspections_or_history(self):
            paths = [self.csv(f"sample_730nmc_TG_rep{rep:02}.csv") for rep in range(3)]
            self.seed_inspections(*paths)
            with patch("core.drr_sources.inspect_csv_gate", side_effect=AssertionError("Cached CSV inspected")):
                expected = self.scan()
                self.scan()  # Populate the existing decoded DrrSource-list LRU.
            inventory = self.cache.parent / "inventory"
            forbidden = {self.cache, *(SourceCatalogCache(str(self.root), namespace, cache_root=inventory).path
                for namespace in ("DRR-raw-v3-composite", "DRR-history-v3-composite"))}
            original = Path.read_text
            def read(path, *args, **kwargs):
                self.assertNotIn(path, forbidden, "Warm catalog reread full acquisition grids")
                return original(path, *args, **kwargs)
            with patch.object(Path, "read_text", autospec=True, side_effect=read), \
                 patch.object(DrrSourceCache, "load", side_effect=AssertionError("Inspection cache loaded")), \
                 patch("core.drr_catalog.discover_drr_sources", side_effect=AssertionError("Raw catalog rebuilt")), \
                 patch("core.drr_catalog.refresh_drr_source_history", side_effect=AssertionError("History rebuilt")), \
                 patch("core.drr_processed_groups.discover_groups", side_effect=AssertionError("Saved metadata reread")):
                actual = self.scan()
            self.assertEqual(actual, expected)
            self.assertEqual(len(actual["groups"]), 1)
            self.assertTrue(actual["groups"][0]["complete"])

        def test_metadata_and_dat_updates_do_not_reinspect_raw_sources(self):
            measurement, background = self.csv(), self.csv("sample_730nmc_BNback.csv")
            self.seed_inspections(measurement, background)
            history = self.root / "Processed Data" / "DRR"
            history.mkdir(parents=True)
            metadata, saved = history / "group.metadata.json", history / "group.dat"
            payload = dict(operation="DR/R", processing=dict(derivative_order=0, baseline_selection="External", baseline_which="all"),
                sources=[dict(role="measurement", source_path=str(measurement)), dict(role="background", source_path=str(background))])
            metadata.write_text(json.dumps(payload), encoding="utf-8")
            saved.write_text("saved matrix", encoding="utf-8")
            initial = self.scan()
            self.assertTrue(next(source for source in initial["sources"] if source.filename == measurement.name).processed)
            with patch.object(DrrSourceCache, "load", side_effect=AssertionError("Raw cache loaded for export update")), \
                 patch("core.drr_catalog.discover_drr_sources", side_effect=AssertionError("Raw sources reinspected")):
                saved.write_text("a longer saved matrix", encoding="utf-8")
                with patch("core.drr_catalog.refresh_drr_source_history", side_effect=AssertionError("DAT changed raw history")):
                    dat_updated = self.scan()
                self.assertEqual(next(row for row in dat_updated["files"] if row["saved"])["size"], saved.stat().st_size)
                payload["sources"] = []
                metadata.write_text(json.dumps(payload), encoding="utf-8")
                updated = self.scan()
                source = next(source for source in updated["sources"] if source.filename == measurement.name)
                self.assertFalse(source.processed)
                self.assertEqual(source.linked_backgrounds, ())
                self.assertEqual([group["members"] for group in updated["groups"]], [group["members"] for group in initial["groups"]])
                metadata.unlink()
                removed = self.scan()
            self.assertFalse(any(row["saved"] for row in removed["files"]))

        def test_unchanged_cached_preview_and_final_result_prepare_rows_once(self):
            self.seed_inspections(self.csv())
            self.scan()
            previews = []
            from ui_qt.quick import source_catalog
            with patch.object(source_catalog, "_drr_catalog_rows", wraps=source_catalog._drr_catalog_rows) as prepare:
                result = self.scan(publish_cached=previews.append)
            self.assertEqual(prepare.call_count, 1)
            self.assertIs(result, previews[0])
            self.assertTrue(result["groups"][0]["complete"])

        def test_saved_rows_are_revalidated_after_raw_scan(self):
            self.seed_inspections(self.csv())
            history = self.root / "Processed Data" / "DRR"
            history.mkdir(parents=True)
            metadata, saved = history / "group.metadata.json", history / "group.dat"
            for change in ("delete-dat", "modify-dat", "modify-metadata"):
                with self.subTest(change=change):
                    payload = dict(operation="DR/R", processing=dict(derivative_order=0))
                    metadata.write_text(json.dumps(payload), encoding="utf-8")
                    saved.write_text("initial matrix", encoding="utf-8")
                    previews = []
                    def preview(value):
                        previews.append(value)
                        if len(previews) != 1:
                            return
                        self.assertTrue(any(row["saved"] for row in value["files"]))
                        if change == "delete-dat":
                            saved.unlink()
                        elif change == "modify-dat":
                            saved.write_text("the matrix changed during the raw scan", encoding="utf-8")
                        else:
                            payload["processing"]["derivative_order"] = 2
                            metadata.write_text(json.dumps(payload), encoding="utf-8")
                    result = self.scan(publish_cached=preview)
                    final_saved = [row for row in result["files"] if row["saved"]]
                    if change == "modify-dat":
                        self.assertEqual(len(final_saved), 1)
                        self.assertEqual(final_saved[0]["size"], saved.stat().st_size)
                    else:
                        self.assertEqual(final_saved, [])
                    self.assertTrue(result["groups"][0]["complete"])

        def test_include_all_is_scoped_and_new_raw_files_invalidate_inspections(self):
            regular = self.csv()
            legacy = self.csv("legacy_730nmc_TG.csv", folder=self.root / "Initial Data" / "PL")
            self.seed_inspections(regular, legacy)
            self.assertEqual({row["id"] for row in self.scan()["files"]}, {str(regular)})
            self.assertEqual({row["id"] for row in self.scan(include_all=True)["files"]}, {str(regular), str(legacy)})
            self.assertEqual({row["id"] for row in self.scan()["files"]}, {str(regular)})
            added = self.csv("new_730nmc_TG.csv")
            updated = self.scan()
            self.assertEqual({row["id"] for row in updated["files"]}, {str(regular), str(added)})
            self.assertTrue(all(source.grid_complete for source in updated["sources"]))
            added.unlink()
            self.assertEqual({row["id"] for row in self.scan()["files"]}, {str(regular)})

        def test_inventory_is_visible_before_inspection_and_external_choice_can_finish(self):
            self.csv()
            external = self.csv("outside.csv", folder=self.base / "external")
            bridge = self.bridge()
            entered, release = Event(), Event()
            from core.drr_sources import inspect_csv_gate
            def slow(*args, **kwargs):
                entered.set()
                if not release.wait(15):
                    raise AssertionError("Inspection test was not released")
                return inspect_csv_gate(*args, **kwargs)
            received = []
            bridge.selectionAccepted.connect(lambda workflow, files: received.append((workflow, files)))
            with patch("core.drr_sources.inspect_csv_gate", side_effect=slow):
                try:
                    bridge.open("drr", str(self.root), [], False)
                    self.wait(lambda: entered.is_set() and bridge.filesModel.rowCount() > 0)
                    self.assertTrue(bridge.snapshot["busy"])
                    self.assertFalse(bridge.snapshot["canAddGroup"])
                    self.assertFalse(any(source.grid_complete for source in bridge._catalog["sources"]))
                    self.assertEqual(bridge.snapshot["selected"], [])
                    bridge.setParameter("selectedGroup", bridge.groupsModel.rows[0]["id"])
                    bridge.invoke("add-group")
                    self.assertEqual(bridge.snapshot["selected"], [])
                    bridge.addExternal([QUrl.fromLocalFile(str(external))])
                    self.assertTrue(bridge.snapshot["canAccept"])
                    bridge.invoke("compatible")
                    self.assertEqual(bridge.snapshot["selected"], [str(external)])
                    generation = bridge._generation
                    bridge.invoke("accept")
                    self.assertEqual(received, [("drr", [str(external)])])
                    self.assertEqual(bridge._generation, generation)
                    self.assertFalse(bridge._cancel.cancelled)
                    self.assertTrue(bridge.snapshot["busy"])
                finally:
                    release.set()
                    # Owner teardown cancels an inspection paused for the accepted load.
                    bridge.close()
                    if bridge._future:
                        bridge._future.result(timeout=15)
                    self.app.processEvents()
            self.assertEqual(bridge.snapshot["selected"], [str(external)])
            self.assertFalse(any(source.grid_complete for source in bridge._catalog["sources"]))

        def test_cached_preview_waits_for_validation_before_group_selection(self):
            self.csv()
            bridge = self.bridge()
            bridge.open("drr", str(self.root), [], False)
            self.wait(lambda: not bridge.snapshot["busy"])
            entered, release = Event(), Event()
            from core.drr_catalog import drr_history_inventory
            def slow(*args, **kwargs):
                entered.set()
                if not release.wait(15):
                    raise AssertionError("Inventory test was not released")
                return drr_history_inventory(*args, **kwargs)
            with patch("core.drr_catalog.drr_history_inventory", side_effect=slow):
                try:
                    bridge.refresh()
                    self.wait(lambda: entered.is_set() and "inventory is available" in bridge.snapshot["status"])
                    self.assertTrue(bridge.snapshot["busy"])
                    bridge.setParameter("selectedGroup", bridge.groupsModel.rows[0]["id"])
                    bridge.invoke("add-group")
                    self.assertEqual(bridge.snapshot["selected"], [])
                finally:
                    release.set()
                    self.wait(lambda: not bridge.snapshot["busy"])
            bridge.invoke("add-group")
            self.assertEqual(len(bridge.snapshot["selected"]), 1)

        def test_saved_dat_can_be_chosen_while_raw_csv_inspection_is_blocked(self):
            self.csv()
            history = self.root / "Processed Data" / "DRR"
            history.mkdir(parents=True)
            (history / "group.metadata.json").write_text(json.dumps(dict(operation="DR/R", processing=dict(derivative_order=0))), encoding="utf-8")
            saved = history / "group.dat"
            saved.write_text("X\t0\n1\t2\n", encoding="utf-8")
            bridge = self.bridge()
            entered, release = Event(), Event()
            from core.drr_sources import inspect_csv_gate
            def slow(*args, **kwargs):
                entered.set()
                if not release.wait(15):
                    raise AssertionError("Saved-result test was not released")
                return inspect_csv_gate(*args, **kwargs)
            received = []
            bridge.selectionAccepted.connect(lambda workflow, files: received.append((workflow, files)))
            with patch("core.drr_sources.inspect_csv_gate", side_effect=slow):
                try:
                    bridge.open("drr", str(self.root), [], True)
                    self.wait(lambda: entered.is_set() and bridge.filesModel.rowCount() == 1)
                    self.assertTrue(bridge.snapshot["busy"])
                    self.assertFalse(bridge.snapshot["canAddGroup"])
                    self.assertTrue(bridge.filesModel.rows[0]["saved"])
                    bridge.selectFile(str(saved), True)
                    self.assertTrue(bridge.snapshot["canAccept"])
                    bridge.invoke("accept")
                    self.assertEqual(received, [("drr", [str(saved)])])
                    self.assertFalse(bridge._cancel.cancelled)
                    self.assertTrue(bridge.snapshot["busy"])
                finally:
                    release.set()
                    # Owner teardown cancels an inspection paused for the accepted load.
                    bridge.close()
                    if bridge._future:
                        bridge._future.result(timeout=15)
                    self.app.processEvents()
            self.assertEqual(bridge.snapshot["selected"], [str(saved)])

        def test_single_toggle_updates_one_file_row_and_snapshot_reuses_counts(self):
            paths = [self.csv(f"g{index}_730nmc_TG.csv") for index in range(206)]
            self.seed_inspections(*paths)
            bridge = self.bridge()
            bridge._workflow, bridge._folder, bridge._only_new = "drr", str(self.root), False
            bridge._catalog = self.scan()
            bridge._refresh_models()
            notifications = []
            bridge.filesModel.dataChanged.connect(lambda first, last, roles: notifications.append((first.row(), last.row())))
            bridge.selectFile(str(paths[0]), True)
            self.assertEqual(len(notifications), 1)
            self.assertEqual(notifications[0][0], notifications[0][1])
            class NoIteration(list):
                def __iter__(self):
                    raise AssertionError("Snapshot rescanned every file")
            bridge._catalog["files"] = NoIteration(bridge._catalog["files"])
            self.assertEqual(bridge.snapshot["counts"]["new"], 206)
            self.assertEqual(bridge.snapshot["selected"], [str(paths[0])])

        def test_failed_inspection_does_not_certify_preview_groups(self):
            path = self.csv()
            bridge = self.bridge()
            with patch("core.drr_sources.inspect_csv_gate", side_effect=ValueError("Synthetic inspection failure")):
                bridge.open("drr", str(self.root), [], False)
                self.wait(lambda: not bridge.snapshot["busy"])
            self.assertIn("Synthetic inspection failure", bridge.snapshot["error"])
            self.assertTrue(bridge.groupsModel.rows)
            self.assertFalse(bridge.snapshot["canAddGroup"])
            bridge.setParameter("selectedGroup", bridge.groupsModel.rows[0]["id"])
            bridge.invoke("add-group")
            self.assertEqual(bridge.snapshot["selected"], [])
            bridge.selectFile(str(path), True)
            self.assertTrue(bridge.snapshot["canAccept"])
            bridge.invoke("compatible")
            self.assertEqual(bridge.snapshot["selected"], [str(path)])

        def test_failed_refresh_does_not_accept_previous_nonexplicit_group(self):
            self.csv()
            bridge = self.bridge()
            bridge.open("drr", str(self.root), [], False)
            self.wait(lambda: not bridge.snapshot["busy"])
            bridge.setParameter("selectedGroup", bridge.groupsModel.rows[0]["id"])
            bridge.invoke("add-group")
            self.assertTrue(bridge.snapshot["canAccept"])
            received = []
            bridge.selectionAccepted.connect(lambda *args: received.append(args))
            with patch("ui_qt.quick.source_catalog.scan_catalog", side_effect=ValueError("Failed catalog refresh")):
                bridge.refresh()
                self.wait(lambda: not bridge.snapshot["busy"])
            self.assertFalse(bridge.snapshot["canAccept"])
            bridge.invoke("accept")
            self.assertEqual(received, [])
            self.assertEqual(len(bridge.snapshot["selected"]), 1)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(QuickMeasurementSelectionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("MEASUREMENT_SELECTION_TESTS=" + json.dumps(dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors))))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
