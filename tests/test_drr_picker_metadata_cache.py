import unittest
from dataclasses import replace
from unittest.mock import patch

from PySide6.QtWidgets import QDialog, QListWidget
from tests import test_drr_picker_stability as fixture
from ui_qt.controllers_drr import format_drr_source_summary


class DrrPickerMetadataCacheTests(unittest.TestCase):
    setUpClass = classmethod(fixture.DrrPickerStabilityTests.setUpClass.__func__)
    setUp = fixture.DrrPickerStabilityTests.setUp
    open_picker = fixture.DrrPickerStabilityTests.open_picker

    def test_revisiting_group_reuses_metadata_until_catalog_publication(self):
        w = self.window
        w.drr_available_sources.append(replace(w.drr_available_sources[0],
            group_key="other", source="other_REF_760nmc.csv", filename="other_REF_760nmc.csv"))

        def execute(dialog):
            groups = dialog.findChild(QListWidget, "drr_source_group_list")
            self.assertEqual(groups.count(), 2)
            groups.setCurrentRow(1)
            with patch("ui_qt.controllers_drr.format_drr_source_summary",
                       wraps=format_drr_source_summary) as summary:
                groups.setCurrentRow(0)
                self.assertEqual(summary.call_count, 0)
                w.drr_available_sources = [replace(s, frame_count=42) for s in w.drr_available_sources]
                w.drr_catalog_refresh_finished.emit(w.current_folder, True)
                files = dialog.findChild(QListWidget, "drr_source_file_list")
                self.assertIn("42 frames", files.item(0).text())
                self.assertGreater(summary.call_count, 0)
            return QDialog.Rejected

        self.open_picker(execute)
