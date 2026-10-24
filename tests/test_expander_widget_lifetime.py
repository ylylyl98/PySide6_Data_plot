"""Moving an existing widget must retire its old native layout-item wrapper."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFormLayout, QPlainTextEdit, QVBoxLayout, QWidget
from shiboken6 import isValid

from tests.ui_test_helpers import dispose_owned_window
from ui_qt.main_window import MainWindow


class ExpanderWidgetLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_moving_form_content_invalidates_old_item_and_preserves_widget(self):
        self._assert_existing_form_content_moves(nested=False)

    def test_moving_nested_form_content_preserves_unrelated_fields(self):
        self._assert_existing_form_content_moves(nested=True, label_text="Analysis")

    def test_moving_spanning_form_content_preserves_widget(self):
        self._assert_existing_form_content_moves(nested=False, label_text=None)

    def _assert_existing_form_content_moves(self, *, nested, label_text=""):
        owner = QWidget()
        source = QWidget()
        self.addCleanup(dispose_owned_window, owner)
        self.addCleanup(dispose_owned_window, source)
        if nested:
            outer_layout = QVBoxLayout(source)
            source_layout = QFormLayout()
            outer_layout.addLayout(source_layout)
        else:
            source_layout = QFormLayout(source)
        content = QPlainTextEdit("Existing analysis")
        if label_text is None:
            source_layout.addRow(content)
        else:
            source_layout.addRow(label_text, content)
        unrelated = QPlainTextEdit("Other result")
        source_layout.addRow("Other", unrelated)
        # Sidebar traversal has already wrapped this native item before the
        # real MainWindow moves its DRR analysis text into the results dock.
        role = QFormLayout.SpanningRole if label_text is None else QFormLayout.FieldRole
        old_item = source_layout.itemAt(0, role)
        label_item = source_layout.itemAt(0, QFormLayout.LabelRole)
        label = label_item.widget() if label_item is not None else None
        old_count = source_layout.count()

        expander = MainWindow._make_expander(owner, "Analysis", content, expanded=True)
        self.addCleanup(dispose_owned_window, expander)

        self.assertFalse(isValid(old_item), "The removed native item must not remain in Shiboken's wrapper map")
        self.assertTrue(isValid(content))
        self.assertEqual(content.toPlainText(), "Existing analysis")
        self.assertEqual(source_layout.count(), old_count - 1)
        if label is not None:
            self.assertIs(source_layout.itemAt(0, QFormLayout.LabelRole).widget(), label)
            self.assertTrue(isValid(label))
        self.assertTrue(isValid(unrelated))
        self.assertEqual(unrelated.toPlainText(), "Other result")
        self.assertIs(source_layout.itemAt(1, QFormLayout.FieldRole).widget(), unrelated)
        self.assertIs(content.parentWidget(), expander)
        self.assertGreaterEqual(expander.layout().indexOf(content), 0)

    def test_expander_accepts_content_without_an_existing_layout(self):
        owner = QWidget()
        self.addCleanup(dispose_owned_window, owner)
        content = QPlainTextEdit("New analysis")
        expander = MainWindow._make_expander(owner, "Analysis", content, expanded=False)
        self.addCleanup(dispose_owned_window, expander)

        self.assertTrue(isValid(content))
        self.assertIs(content.parentWidget(), expander)
        self.assertTrue(content.isHidden())
        self.assertEqual(content.toPlainText(), "New analysis")


if __name__ == "__main__":
    unittest.main()
