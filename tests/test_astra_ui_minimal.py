import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QSettings
from PySide6.QtWidgets import QApplication, QToolButton

from ui_qt.main_window import MainWindow
from ui_qt.theme import install_theme


class AstraUiMinimalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode="light")

    def setUp(self):
        settings = QSettings("DPTK", "PySide6_Data_Plot")
        for prefix, titles in {
            "pl": ("Manual plot ranges",),
            "cmp": ("Display", "Manual channel assignments", "Angle Rules"),
            "mcd": ("Analysis", "Correction", "Diagnostics", "Plot"),
        }.items():
            for title in titles:
                settings.remove(f"ui/expanders/{prefix}/{title.replace(' ', '_')}")
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            self.window = MainWindow()
        self.window.resize(1180, 820)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    @staticmethod
    def _expander(page, title):
        return next(
            button for button in page.findChildren(QToolButton)
            if button.text() == title
        )

    @staticmethod
    def _section_widget(page, button):
        current = button
        while current.parentWidget() is not page:
            current = current.parentWidget()
        return current

    def test_toolbar_uses_short_action_names(self):
        self.assertEqual(self.window.menu_toolbar_host.load_action.text(), "Reload")
        self.assertEqual(self.window.menu_toolbar_host.plot_action.text(), "Update now")

    def test_pl_common_display_controls_are_visible_without_manual_ranges(self):
        self.assertFalse(self._expander(self.window.pl_tab_scroll.widget(), "Manual plot ranges").isChecked())
        self.assertTrue(self.window.pl_auto_v_btn.isVisible())
        self.assertTrue(self.window.pl_spins["gate"].isVisible())
        self.assertFalse(self.window.pl_split_scale_chk.isVisible())
        self.assertEqual(
            sum(button is self.window.pl_auto_v_btn for button in self.window.pl_tab_scroll.widget().findChildren(QToolButton)),
            1,
        )

    def test_compare_display_is_open_and_group_actions_precede_source_filter(self):
        page = self.window.cmp_tab_scroll.widget()
        self.window.tabs.setCurrentIndex(
            next(index for index in range(self.window.tabs.count()) if self.window.tabs.tabText(index) == "Compare")
        )
        self.app.processEvents()
        self.assertTrue(self._expander(page, "Display").isChecked())
        self.assertFalse(self._expander(page, "Manual channel assignments").isChecked())
        self.assertFalse(self._expander(page, "Angle Rules").isChecked())
        group_row = self.window.cmp_group_selection_summary
        source_row = self.window.cmp_source_filter_combo
        self.assertLess(
            group_row.mapTo(page, QPoint(0, 0)).y(),
            source_row.mapTo(page, QPoint(0, 0)).y(),
        )

    def test_mcd_plot_precedes_collapsed_processing_sections_and_analysis_is_advanced(self):
        page = self.window.mcd_tab_scroll.widget()
        self.window.tabs.setCurrentIndex(
            next(index for index in range(self.window.tabs.count()) if self.window.tabs.tabText(index) == "MCD")
        )
        self.app.processEvents()
        headers = {title: next(
            (button for button in page.findChildren(QToolButton) if button.text() == title),
            None,
        ) for title in ("Plot", "Analysis", "Correction", "Diagnostics")}
        self.assertTrue(all(headers.values()))
        self.assertTrue(headers["Plot"].isChecked())
        self.assertFalse(headers["Analysis"].isChecked())
        self.assertFalse(headers["Correction"].isChecked())
        self.assertFalse(headers["Diagnostics"].isChecked())
        self.assertLess(
            page.layout().indexOf(self._section_widget(page, headers["Plot"])),
            page.layout().indexOf(self._section_widget(page, headers["Correction"])),
        )
        self.assertLess(
            page.layout().indexOf(self._section_widget(page, headers["Correction"])),
            page.layout().indexOf(self._section_widget(page, headers["Diagnostics"])),
        )
        self.assertFalse(self.window.mcd_window_metric_combo.isVisible())
        self.assertFalse(self.window.mcd_fit_zero_chk.isVisible())


if __name__ == "__main__":
    unittest.main()
