"""Saved P/N creation times use local display without changing provenance."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QLabel

from tests import test_drr_pn_background_ui as _fixtures


class PnTimestampUiTests(unittest.TestCase):
    setUpClass = classmethod(_fixtures.PnBackgroundDialogTests.setUpClass.__func__)
    setUp = _fixtures.PnBackgroundDialogTests.setUp
    tearDown = _fixtures.PnBackgroundDialogTests.tearDown
    write = _fixtures.PnBackgroundDialogTests.write
    save = _fixtures.PnBackgroundDialogTests.save
    dialog = _fixtures.PnBackgroundDialogTests.dialog
    wait = _fixtures.PnBackgroundDialogTests.wait
    choose = _fixtures.PnBackgroundDialogTests.choose

    def test_saved_utc_history_uses_local_time_without_rewriting_artifacts(self):
        stamp = "2026-10-21T17:28:23.355892+00:00"
        with patch("core.drr_pn_store.datetime") as clock:
            clock.now.return_value.isoformat.return_value = stamp
            version = self.save()
        artifacts = (self.store.manifest_path, version.metadata_path, version.csv_path)
        before = {path: path.read_bytes() for path in artifacts}

        dialog = self.dialog()
        self.choose(dialog)
        expected = datetime.fromtimestamp(
            datetime(2026, 10, 21, 17, 28, 23, 355892, tzinfo=timezone.utc).timestamp()
        ).strftime("%Y-%m-%d %H:%M")
        index = dialog.history.findData(version.version_id)
        self.assertGreater(index, 0)
        self.assertTrue(dialog.history.itemText(index).endswith(" · " + expected))
        self.assertNotIn(stamp, dialog.history.itemText(index))
        self.assertIn("local time", dialog.history.toolTip())
        self.assertIn("Saved version (local time)",
                      [label.text() for label in dialog.findChildren(QLabel)])
        self.assertEqual(dialog._current.created_at, stamp)
        self.assertEqual(self.store.get_version(version.version_id).created_at, stamp)
        self.assertEqual({path: path.read_bytes() for path in artifacts}, before)

    def test_history_accepts_iso_offsets_z_and_legacy_local_values(self):
        version = self.save()
        dialog = self.dialog()
        expected = datetime.fromtimestamp(
            datetime(2026, 10, 21, 17, 28, tzinfo=timezone.utc).timestamp()
        ).strftime("%Y-%m-%d %H:%M")
        for stamp, shown in (
            ("2026-10-21T17:28:23.355892+00:00", expected),
            ("2026-10-21T13:28:23.355892-04:00", expected),
            ("2026-10-21T17:28:23.355892Z", expected),
            ("2026-10-21T13:28:23.355892", "2026-10-21 13:28"),
            ("unknown", "unknown"),
        ):
            with self.subTest(stamp=stamp):
                historical = replace(version, created_at=stamp)
                dialog._loaded_history(((historical,), None))
                self.assertTrue(dialog.history.itemText(1).endswith(" · " + shown))
                self.assertEqual(dialog.history.itemData(1), historical.version_id)
                self.assertEqual(historical.created_at, stamp)
