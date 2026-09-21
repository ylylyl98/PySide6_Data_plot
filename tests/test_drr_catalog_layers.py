import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from core.drr_sources import discover_drr_sources, refresh_drr_source_history, drr_source_paths

class DrrCatalogLayerTests(unittest.TestCase):
    def test_ref_partition_keeps_legacy_initial_data_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);initial=root/'Initial Data';ref=initial/'REF'
            ref.mkdir(parents=True)
            old=initial/'old_REF.csv';new=ref/'new_REF.csv'
            for path in (old,new):path.write_text('Vbg,700,701\n0,1,2\n1,2,3\n')
            self.assertEqual(set(drr_source_paths(root)),{old,new})
            self.assertEqual(len(discover_drr_sources(root,include_history=False)),2)

    def test_ref_partition_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pl = root / "Initial Data" / "PL"
            pl.mkdir(parents=True)
            (pl / "sample.csv").write_text("Vbg,700,701\n0,1,2\n")
            legacy = root / "legacy.xlsx"
            legacy.touch()
            self.assertEqual(len(drr_source_paths(root)), 2)
            ref = pl.parent / "ref"
            ref.mkdir()
            (ref / "sample.csv").write_text("Vbg,700,701\n0,1,2\n")
            self.assertEqual(set(drr_source_paths(root)), {legacy, ref / "sample.csv"})
            self.assertEqual(set(drr_source_paths(root, include_all=True)), {legacy, ref / "sample.csv", pl / "sample.csv"})
            self.assertEqual(len(discover_drr_sources(root, include_all=True)), 3)

    def test_history_add_change_remove_without_source_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample_back.csv"
            source.write_text("Vbg,700,701\n0,1,2\n1,2,3\n")
            with patch("core.drr_sources._read_drr_metadata", side_effect=AssertionError("history read")):
                base = discover_drr_sources(root, include_history=False)
            self.assertEqual(base[0].classification, "background")
            history = root / "Processed Data" / "DRR"
            history.mkdir(parents=True)
            meta = history / "saved.metadata.json"
            meta.write_text(json.dumps({"operation": "DR/R", "sources": [
                {"source_path": "sample_back.csv", "role": "measurement"},
                {"source_path": "Initial Data/PL/external.csv", "role": "background"}
            ], "processing": {"baseline_selection": "External"}}))
            with patch("core.drr_sources.inspect_csv_gate", side_effect=AssertionError("source read")):
                overlay = refresh_drr_source_history(root, base)
                self.assertTrue(overlay[0].processed)
                self.assertEqual(overlay[0].classification, "measurement")
                self.assertEqual(overlay[0].linked_backgrounds, ("Initial Data/PL/external.csv",))
                meta.write_text(json.dumps({"sources": [{"source_path": "sample_back.csv", "role": "background"}]}))
                corrected = refresh_drr_source_history(root, overlay)
                self.assertFalse(corrected[0].processed)
                self.assertEqual(corrected[0].metadata_role, "background")
                meta.unlink()
                self.assertEqual(refresh_drr_source_history(root, overlay), base)

class DrrCatalogPersistenceTests(unittest.TestCase):
    def test_all_data_and_ref_caches_are_isolated(self):
        import core.drr_catalog as catalog
        from core.drr_sources import DrrSourceCache
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            ref = root / "Initial Data" / "REF"
            pl = ref.parent / "PL"
            ref.mkdir(parents=True)
            pl.mkdir()
            content = "Vbg,700,701\n0,1,2\n1,2,3\n"
            (ref / "measurement.csv").write_text(content)
            (pl / "measurement.csv").write_text(content)
            inspection = DrrSourceCache(Path(tmp) / "inspection.json", load_on_init=False)
            with patch.dict("os.environ", {"LOCALAPPDATA": str(Path(tmp) / "cache")}):
                selected = catalog.load_drr_catalog(root, inspection)
                all_data = catalog.load_drr_catalog(root, inspection, include_all=True)
                self.assertEqual(len(selected), 1)
                self.assertEqual(len(all_data), 2)
                with patch.object(catalog, "discover_drr_sources", side_effect=AssertionError("rebuild")):
                    selected_preview, all_preview = [], []
                    self.assertEqual(catalog.load_drr_catalog(root, inspection, publish_cached=selected_preview.append), selected)
                    self.assertEqual(catalog.load_drr_catalog(root, inspection, include_all=True, publish_cached=all_preview.append), all_data)
                    self.assertEqual(selected_preview, [selected])
                    self.assertEqual(all_preview, [all_data])

    def test_new_ref_acquisition_invalidates_both_layers_but_pl_does_not(self):
        import core.drr_catalog as catalog
        from core.drr_sources import DrrSourceCache
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            ref = root / "Initial Data" / "REF"
            pl = ref.parent / "PL"
            ref.mkdir(parents=True)
            pl.mkdir()
            content = "Vbg,700,701\n0,1,2\n1,2,3\n"
            (ref / "measurement.csv").write_text(content)
            inspection = DrrSourceCache(Path(tmp) / "inspection.json", load_on_init=False)
            with patch.dict("os.environ", {"LOCALAPPDATA": str(Path(tmp) / "cache")}):
                original = catalog.load_drr_catalog(root, inspection)
                with patch.object(catalog, "discover_drr_sources", wraps=catalog.discover_drr_sources) as discover:
                    (pl / "pl.csv").write_text(content)
                    self.assertEqual(catalog.load_drr_catalog(root, inspection), original)
                    self.assertEqual(discover.call_count, 0)
                    (ref / "new.csv").write_text(content)
                    refreshed = catalog.load_drr_catalog(root, inspection)
                    self.assertEqual(len(refreshed), 2)
                    self.assertEqual(discover.call_count, 1)
                    self.assertEqual(refreshed, discover_drr_sources(root))

    def test_save_only_refreshes_history_and_preserves_cached_preview(self):
        import core.drr_catalog as catalog
        from core.drr_sources import DrrSourceCache
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            root.mkdir()
            (root / "measurement.csv").write_text("Vbg,700,701\n0,1,2\n1,2,3\n")
            inspection = DrrSourceCache(Path(tmp) / "inspection.json", load_on_init=False)
            with patch.dict("os.environ", {"LOCALAPPDATA": str(Path(tmp) / "cache")}):
                with patch.object(catalog, "discover_drr_sources", wraps=catalog.discover_drr_sources) as discover:
                    original = catalog.load_drr_catalog(root, inspection)
                    self.assertEqual(discover.call_count, 1)
                    history = root / "Processed Data" / "DRR"
                    history.mkdir(parents=True)
                    (history / "saved.dat").write_text("saved data")
                    (history / "saved.png").write_bytes(b"image")
                    with patch.object(catalog, "refresh_drr_source_history", wraps=catalog.refresh_drr_source_history) as overlay:
                        preview = []
                        self.assertEqual(catalog.load_drr_catalog(root, inspection, publish_cached=preview.append), original)
                        self.assertEqual(preview, [original])
                        self.assertEqual(overlay.call_count, 0)
                        meta = history / "saved.metadata.json"
                        meta.write_text(json.dumps({"sources": [{"source_path": "measurement.csv", "role": "measurement"}]}))
                        saved = catalog.load_drr_catalog(root, inspection)
                        self.assertTrue(saved[0].processed)
                        self.assertEqual(overlay.call_count, 1)
                        self.assertEqual(discover.call_count, 1)
                        meta.unlink()
                        self.assertEqual(catalog.load_drr_catalog(root, inspection), original)
                        self.assertEqual(discover.call_count, 1)
                    self.assertEqual(catalog.load_drr_catalog(root, inspection, force=True), original)
                    self.assertEqual(discover.call_count, 2)


if __name__ == "__main__":
    unittest.main()
