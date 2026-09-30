from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget
from shiboken6 import isValid

from tests.ui_test_helpers import dispose_owned_window


class OwnedWindowCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_cleanup_destroys_owned_tree_and_preserves_other_windows(self) -> None:
        owned = QWidget()
        child = QWidget(owned)
        unrelated = QWidget()
        self.addCleanup(dispose_owned_window, unrelated)
        dispose_owned_window(owned)
        self.assertFalse(isValid(owned))
        self.assertFalse(isValid(child))
        self.assertTrue(isValid(unrelated))
        # Test cleanup can also be called after an explicit disposal in a test.
        dispose_owned_window(owned)
