"""Focused tests for the Phase 4.2 status badge presentation component."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ui_qt.status_badge import StatusBadge


class StatusBadgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_status_badge_owns_text_tooltip_semantics_and_accessibility(self) -> None:
        badge = StatusBadge(
            "● NEW — No saved analysis",
            tooltip="raw/sweep.csv",
            app_role="sourceBadge",
            badge_state="new",
        )

        self.assertEqual(badge.text(), "● NEW — No saved analysis")
        self.assertEqual(badge.toolTip(), "raw/sweep.csv")
        self.assertTrue(badge.wordWrap())
        self.assertEqual(badge.property("appRole"), "sourceBadge")
        self.assertEqual(badge.property("badgeState"), "new")
        self.assertIsNone(badge.property("fluentSeverity"))
        self.assertEqual(badge.accessibleDescription(), badge.text())

    def test_workflow_states_do_not_become_generic_severity(self) -> None:
        badge = StatusBadge()

        for state in ("new", "processed", "saved"):
            badge.set_status(f"{state} text", badge_state=state)
            self.assertEqual(badge.property("badgeState"), state)
            self.assertIsNone(badge.property("fluentSeverity"))

    def test_set_status_updates_owned_presentation_metadata(self) -> None:
        badge = StatusBadge()
        badge.set_status(
            "✓ MCD plots are from the same folder (run).",
            tooltip="run",
            app_role="statusBadge",
            fluent_severity="success",
            accessible_description="MCD plots are paired.",
        )

        self.assertEqual(badge.text(), "✓ MCD plots are from the same folder (run).")
        self.assertEqual(badge.toolTip(), "run")
        self.assertEqual(badge.property("appRole"), "statusBadge")
        self.assertEqual(badge.property("fluentSeverity"), "success")
        self.assertEqual(badge.accessibleDescription(), "MCD plots are paired.")


if __name__ == "__main__":
    unittest.main()
