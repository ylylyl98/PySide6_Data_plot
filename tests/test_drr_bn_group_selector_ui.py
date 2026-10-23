"""The current-group BN selector stays compact in the pinned Data controls."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect, QSettings, Qt
from PySide6.QtWidgets import QApplication, QLabel, QSizePolicy

from tests.ui_test_helpers import dispose_owned_window
from ui_qt.main_window import MainWindow
from ui_qt.theme import install_theme


class BnGroupSelectorLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.theme = install_theme(cls.app, mode="light")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = QSettings(str(Path(self.tmp.name) / "ui.ini"), QSettings.IniFormat)
        with (patch("ui_qt.main_window.QSettings", return_value=self.settings),
              patch.object(MainWindow, "_restore_last_folder", lambda self: None),
              patch.object(MainWindow, "_schedule_automatic_update_check", lambda self: None)):
            self.window = MainWindow()
        self.window.setAttribute(Qt.WA_DontShowOnScreen)
        self.window._suspend_drr_autoplot = True
        self.window.resize(1180, 700)
        self.window.show()
        self.window.tabs.setCurrentIndex(1)
        self.app.processEvents()

    def tearDown(self):
        dispose_owned_window(self.window)
        self.theme.set_mode("light")
        self.tmp.cleanup()

    def test_selector_starts_hidden_without_a_matching_context(self):
        window = self.window
        self.assertTrue(window.drr_bn_group_row.isHidden())
        self.assertEqual(window.drr_bn_group_row.objectName(), "drr_bn_group_row")
        self.assertEqual(window.drr_bn_group_combo.objectName(), "drr_bn_group_choice")
        self.assertEqual(window.drr_bn_group_combo.accessibleName(), "Group BN")
        self.assertIn("immediately", window.drr_bn_group_combo.accessibleDescription())
        labels = window.drr_bn_group_row.findChildren(QLabel)
        self.assertEqual(len(labels), 1)
        self.assertIs(labels[0].buddy(), window.drr_bn_group_combo)

    def test_long_runtime_choices_fit_the_existing_sidebar_in_both_themes(self):
        window = self.window
        combo = window.drr_bn_group_combo
        prefix = "YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn49deg_BNbackground_PIXIS_"
        # Workflow owns the actual model. Long labels must not establish the
        # sidebar's minimum width even when a future source has no short alias.
        previous = combo.blockSignals(True)
        combo.clear()
        combo.addItem("Automatic · 3 BN001 + 3 BN002", "automatic")
        for number in (1, 2):
            filename = prefix + f"{number:03}.csv"
            combo.addItem("All use " + filename, filename)
            combo.setItemData(combo.count() - 1, filename, Qt.ToolTipRole)
        combo.setCurrentIndex(1)
        combo.blockSignals(previous)
        window.drr_bn_group_row.show()
        for mode in ("light", "dark"):
            with self.subTest(mode=mode):
                self.theme.set_mode(mode)
                for _ in range(3):
                    self.app.processEvents()
                self.assertEqual(combo.minimumWidth(), 0)
                self.assertEqual(combo.sizePolicy().horizontalPolicy(), QSizePolicy.Ignored)
                self.assertEqual(window.drr_bn_group_row.sizePolicy().horizontalPolicy(), QSizePolicy.Ignored)
                viewport = window.drr_tab_scroll.viewport()
                for widget in (window.drr_bn_group_row, combo, window.drr_edit_baselines_btn):
                    rect = QRect(widget.mapTo(viewport, QPoint(0, 0)), widget.size())
                    self.assertTrue(widget.isVisible(), widget.objectName())
                    self.assertTrue(viewport.rect().contains(rect), (widget.objectName(), rect, viewport.rect()))
                    self.assertTrue(widget.parentWidget().contentsRect().contains(widget.geometry()))
                self.assertLess(combo.width(), 400)
                self.assertEqual(window.drr_tab_scroll.horizontalScrollBar().maximum(), 0)
                self.assertEqual(window.drr_tab_scroll.verticalScrollBar().maximum(), 0)
                self.assertEqual(combo.itemData(1, Qt.ToolTipRole), prefix + "001.csv")
                self.assertEqual(combo.styleSheet(), "")


if __name__ == "__main__":
    unittest.main()
