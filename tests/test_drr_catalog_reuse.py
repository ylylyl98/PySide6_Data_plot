from __future__ import annotations

import os
import json
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog, QListWidget
from PySide6.QtCore import QSettings

from core.drr_sources import DrrSourceCache, discover_drr_sources, inspect_csv_gate


class DrrCatalogReuseTests(unittest.TestCase):
    def inspection_cache(self, root):
        cache = DrrSourceCache(root / "drr-inspection.json", load_on_init=False)
        identity = 'C:/data/测量 "one".csv'
        metadata = dict(gate_varies=True, frame_count=2, gate_grid=((-0., 1e-12), (2.5, 3.)),
            spectral_grid=(700.1, 701.2), gate_direction="Vbg increasing · Δ",
            gate_ranges=((0., 2.5), (1e-12, 3.)), gate_labels=("Vbg", "Vtg"), grid_complete=True,
            wavelength_center_nm=700.65, wavelength_center_source="spectral axis")
        cache.put(identity, modified_ns=3, size_bytes=4, metadata=metadata)
        return cache, identity, metadata

    def test_save_preserves_streaming_bytes_and_round_trips_inspections(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, identity, metadata = self.inspection_cache(Path(tmp))
            record = dict(schema_version=1, entries={identity:dict(modified_ns=3, size_bytes=4, metadata=metadata)})
            reference = StringIO()
            json.dump(record, reference, separators=(",", ":"))
            cache.save()
            self.assertEqual(cache.path.read_bytes(), reference.getvalue().encode("utf-8"))
            restored = DrrSourceCache(cache.path)
            self.assertEqual(restored.get(identity, modified_ns=3, size_bytes=4), metadata)

    def test_save_memory_fallback_preserves_streaming_bytes_and_readback(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, identity, metadata = self.inspection_cache(Path(tmp))
            cache.save()
            expected = cache.path.read_bytes()
            with patch("core.drr_sources.json.dumps", side_effect=MemoryError):
                cache.save()
            self.assertEqual(cache.path.read_bytes(), expected)
            self.assertEqual(DrrSourceCache(cache.path).get(identity, modified_ns=3, size_bytes=4), metadata)

    def test_save_replace_failure_keeps_old_cache_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, identity, metadata = self.inspection_cache(Path(tmp))
            cache.save()
            expected = cache.path.read_bytes()
            cache.put("new.csv", modified_ns=5, size_bytes=6, metadata=metadata)
            with patch.object(Path, "replace", side_effect=OSError("Synthetic locked destination")):
                cache.save()
            self.assertEqual(cache.path.read_bytes(), expected)
            self.assertEqual(DrrSourceCache(cache.path).get(identity, modified_ns=3, size_bytes=4), metadata)
            self.assertEqual(list(cache.path.parent.glob(".*.tmp")), [])

    def test_persistent_cache_reuses_unchanged_csv_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "Initial Data"
            data.mkdir()
            source = data / "sample_760nmc_data.csv"
            source.write_text(
                "Vbg,Vtg,740,760\n0,0,1,2\n1,0,3,4\n", encoding="utf-8"
            )
            cache_path = root / "app-cache" / "drr-inspection.json"

            first_cache = DrrSourceCache(cache_path)
            with patch("core.drr_sources.inspect_csv_gate", wraps=inspect_csv_gate) as inspect:
                discover_drr_sources(root, cache=first_cache)
                first_cache.save()
                first_reads = inspect.call_count

            restarted_cache = DrrSourceCache(cache_path)
            with patch("core.drr_sources.inspect_csv_gate") as inspect:
                discover_drr_sources(root, cache=restarted_cache)
                self.assertEqual(inspect.call_count, 0)
            self.assertGreater(first_reads, 0)

    def test_corrupt_persistent_cache_falls_back_to_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "drr-inspection.json"
            cache_path.write_text("{not-json", encoding="utf-8")
            cache = DrrSourceCache(cache_path)
            self.assertIsNone(cache.get("missing", modified_ns=1, size_bytes=1))

    def test_changed_file_is_reinspected_and_deleted_file_is_evicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "Initial Data"
            data.mkdir()
            source = data / "sample_760nmc_data.csv"
            source.write_text(
                "Vbg,Vtg,740,760\n0,0,1,2\n1,0,3,4\n", encoding="utf-8"
            )
            cache_path = root / "app-cache" / "drr-inspection.json"
            cache = DrrSourceCache(cache_path)
            discover_drr_sources(root, cache=cache)
            cache.save()

            source.write_text(
                "Vbg,Vtg,740,760\n0,0,1,2\n1,0,3,4\n2,0,5,6\n", encoding="utf-8"
            )
            restarted = DrrSourceCache(cache_path)
            with patch("core.drr_sources.inspect_csv_gate") as inspect:
                inspect.side_effect = inspect_csv_gate
                discover_drr_sources(root, cache=restarted)
                self.assertGreater(inspect.call_count, 0)

            identity = str(source.resolve()).casefold()
            source.unlink()
            self.assertEqual(discover_drr_sources(root, cache=restarted), [])
            self.assertIsNone(restarted.get(identity, modified_ns=0, size_bytes=0))

    def test_incomplete_valid_cache_entry_is_discarded_and_reinspected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "Initial Data"
            data.mkdir()
            source = data / "sample_760nmc_data.csv"
            source.write_text(
                "Vbg,Vtg,740,760\n0,0,1,2\n1,0,3,4\n", encoding="utf-8"
            )
            stat = source.stat()
            identity = str(source.resolve()).casefold()
            cache_path = root / "app-cache" / "drr-inspection.json"
            cache_path.parent.mkdir()
            cache_path.write_text(json.dumps({
                "schema_version": 1,
                "entries": {identity: {
                    "modified_ns": int(stat.st_mtime_ns),
                    "size_bytes": int(stat.st_size),
                    "metadata": {"gate_varies": False},
                }},
            }), encoding="utf-8")
            cache = DrrSourceCache(cache_path)
            with patch("core.drr_sources.inspect_csv_gate", wraps=inspect_csv_gate) as inspect:
                discover_drr_sources(root, cache=cache)
                self.assertGreater(inspect.call_count, 0)

    def test_load_preserves_memory_entries_when_disk_cache_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = DrrSourceCache(Path(tmp) / "missing.json", load_on_init=False)
            cache.put(
                "C:/data/sample.csv", modified_ns=3, size_bytes=4,
                metadata={"gate_varies": True},
            )
            cache.load()
            self.assertIsNotNone(cache.get("C:/data/sample.csv", modified_ns=3, size_bytes=4))

    def test_save_does_not_clobber_a_stale_fixed_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "drr-inspection.json"
            stale_temp = cache_path.with_name(cache_path.name + ".tmp")
            stale_temp.write_text("sentinel", encoding="utf-8")
            cache = DrrSourceCache(cache_path, load_on_init=False)
            cache.put(
                "C:/data/sample.csv", modified_ns=3, size_bytes=4,
                metadata={"gate_varies": True},
            )
            cache.save()
            self.assertEqual(stale_temp.read_text(encoding="utf-8"), "sentinel")

    def test_dialog_applies_catalog_completion_without_refresh_button(self) -> None:
        app = QApplication.instance() or QApplication([])
        from core.drr_sources import DrrSource
        from ui_qt.main_window import MainWindow
        temporary = tempfile.TemporaryDirectory(prefix="drr-catalog-dialog-")
        self.addCleanup(temporary.cleanup)
        settings = QSettings(str(Path(temporary.name) / "settings.ini"), QSettings.IniFormat)

        source = DrrSource(
            source="Initial Data/new_REF_760nmc_data.csv",
            filename="new_REF_760nmc_data.csv",
            group_key="new_REF_760nmc_data",
            session_date="2026-09-12",
            modified_time=1.0,
            is_background=False,
        )
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None), \
                patch("ui_qt.main_window.QSettings", return_value=settings):
            window = MainWindow()
        window.current_folder = tempfile.gettempdir()
        window.drr_available_sources = []
        observed = {}

        def fake_exec(dialog):
            window.drr_available_sources = [source]
            window.drr_catalog_refresh_finished.emit(window.current_folder, True)
            observed["files"] = dialog.findChild(QListWidget, "drr_source_file_list").count()
            return QDialog.Rejected

        try:
            with patch.object(QDialog, "exec", fake_exec):
                window.drr_controller._open_drr_source_dialog(
                    title="Choose DRR Measurement Group",
                    selected=[],
                    baseline_mode=False,
                )
            self.assertEqual(observed["files"], 1)
        finally:
            window.close()
            window.deleteLater()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
