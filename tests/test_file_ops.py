from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.file_ops import archive_all, list_root_csvs, restore_all


class FileOpsTests(unittest.TestCase):
    def test_list_root_csvs_empty_for_missing_folder(self) -> None:
        missing = Path(tempfile.gettempdir()) / "definitely_missing_csv_root"
        self.assertEqual(list_root_csvs(str(missing)), [])

    def test_list_root_csvs_uses_natural_sort(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "run_10.csv").write_text("x")
            (p / "run_2.csv").write_text("x")
            (p / "run_1.csv").write_text("x")
            (p / "note.txt").write_text("ignored")

            self.assertEqual(
                list_root_csvs(str(p)),
                ["run_1.csv", "run_2.csv", "run_10.csv"],
            )

    def test_archive_and_restore_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "a.csv").write_text("a")
            (p / "b.csv").write_text("b")

            moved = archive_all(str(p), "archive")
            self.assertEqual(moved, 2)
            self.assertEqual(list_root_csvs(str(p)), [])
            self.assertEqual(sorted(x.name for x in (p / "archive").glob("*.csv")), ["a.csv", "b.csv"])

            restored = restore_all(str(p), "archive")
            self.assertEqual(restored, 2)
            self.assertEqual(list_root_csvs(str(p)), ["a.csv", "b.csv"])


if __name__ == "__main__":
    unittest.main()
