import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class PickerIOBoundaryTests(unittest.TestCase):
    def test_power_validation_key_uses_worker_catalog_signature(self):
        from ui_qt.power_group_dialog import PowerGroupDialog
        dialog = PowerGroupDialog.__new__(PowerGroupDialog)
        source = SimpleNamespace(file_name="sample.csv", records=())
        dialog._folder = "folder"
        dialog._sources = {"g": source}
        dialog.controller = SimpleNamespace(
            _power_catalog_signatures=(("sample.csv", 12, 345),)
        )
        dialog.pair_combo = SimpleNamespace(currentText=lambda: "Pair by Stage")
        with patch("pathlib.Path.stat", side_effect=AssertionError("GUI stat")):
            key = dialog._validation_key("Single intensity", ["g"])
        self.assertIn(("sample.csv", 12, 345), key[2][0][1])

if __name__ == "__main__":
    unittest.main()
