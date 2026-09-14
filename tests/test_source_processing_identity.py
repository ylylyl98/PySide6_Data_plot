import json
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.data_io import discover_pl_processing_status
from core.mcd import discover_mcd_processing_status
from core.source_identity import match_source_identity


class SourceIdentityTests(unittest.TestCase):
    def test_exact_relative_path_wins_over_duplicate_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = ["Initial Data/a/same_PL.csv", "Initial Data/b/same_PL.csv"]
            for source in sources:
                (root / source).parent.mkdir(parents=True, exist_ok=True)
                (root / source).write_text("x", encoding="utf-8")
            out = root / "Processed Data" / "PL"
            out.mkdir(parents=True)
            (out / "result.metadata.json").write_text(json.dumps({
                "workflow": "PL", "created_utc": "2026-09-01T00:00:00+00:00",
                "sources": [{"name": sources[0]}],
            }), encoding="utf-8")
            self.assertEqual(discover_pl_processing_status(root, sources), {sources[0]: "2026-09-01T00:00:00+00:00"})
            unknown = set()
            (out / "result.metadata.json").write_text(json.dumps({
                "workflow": "PL", "created_utc": "2026-09-01T00:00:00+00:00",
                "sources": [{"name": "same_PL.csv"}],
            }), encoding="utf-8")
            self.assertEqual(discover_pl_processing_status(root, sources, ambiguous_sources=unknown), {})
            self.assertEqual(unknown, set(sources))

    def test_identity_strong_missing_does_not_fallback(self):
        resolved, unknown = match_source_identity(
            "/experiment", ["a/same.csv", "b/same.csv"],
            relative_path="missing/same.csv", legacy_name="same.csv",
        )
        self.assertIsNone(resolved)
        self.assertEqual(unknown, ())

    def test_unique_and_ambiguous_legacy_names(self):
        self.assertEqual(match_source_identity("/experiment", ["a.csv"], legacy_name="a.csv"), ("a.csv", ()))
        self.assertEqual(match_source_identity("/experiment", ["a/a.csv", "b/a.csv"], legacy_name="a.csv"), (None, ("a/a.csv", "b/a.csv")))

    def test_normalizes_slashes_and_case(self):
        self.assertEqual(match_source_identity("/experiment", ["Initial Data/A.CSV"], relative_path="initial data\\a.csv")[0], "Initial Data/A.CSV")

    def test_mcd_exact_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = ["mcd/a/same.csv", "mcd/b/same.csv"]
            for source in sources:
                (root / source).parent.mkdir(parents=True, exist_ok=True)
                (root / source).write_text("x", encoding="utf-8")
            out = root / "Processed Data" / "MCD"
            out.mkdir(parents=True)
            (out / "a_MCD_settings_test.json").write_text(json.dumps({
                "workflow": "MCD", "created_utc": "2026-09-01", "source_file": "mcd/a/same.csv",
            }), encoding="utf-8")
            self.assertEqual(discover_mcd_processing_status(root, sources), {sources[0]: "2026-09-01"})
            unknown = set()
            (out / "a_MCD_settings_test.json").write_text(json.dumps({
                "workflow": "MCD", "created_utc": "2026-09-01", "source_file": "same.csv",
            }), encoding="utf-8")
            self.assertEqual(discover_mcd_processing_status(root, sources, ambiguous_sources=unknown), {})
            self.assertEqual(unknown, set(sources))

    def test_mcd_malformed_sources_list_uses_legacy_top_level_fields(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = "mcd/a.csv"
            (root / source).parent.mkdir(parents=True)
            (root / source).write_text("x", encoding="utf-8")
            out = root / "Processed Data" / "MCD"
            out.mkdir(parents=True)
            (out / "a_MCD_settings_test.json").write_text(
                json.dumps({
                    "workflow": "MCD",
                    "created_utc": "2026-09-02",
                    "sources": ["broken"],
                    "source_file": source,
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                discover_mcd_processing_status(root, [source]),
                {source: "2026-09-02"},
            )


if __name__ == "__main__":
    unittest.main()
