from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyleOptionSpinBox,
    QToolButton,
)

from ui_qt.presentation_widget import PresentationBuilderWidget
from ui_qt.shell.status_bar import StatusBarView
from ui_qt.shell.workflow_navigation import WorkflowNavigation
from PySide6.QtWidgets import QTabWidget


class Phase7AccessibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_presentation_advanced_toggle_has_accessible_identity(self) -> None:
        widget = PresentationBuilderWidget()
        try:
            button = widget.advanced_btn
            self.assertEqual(button.accessibleName(), "Advanced presentation options")
            self.assertEqual(
                button.accessibleDescription(),
                "Show or hide optional presentation copy and recovery settings",
            )
            self.assertIn("Advanced", button.toolTip())
        finally:
            widget.close()

    def test_shell_actionable_controls_have_names_and_passive_status_is_not_focusable(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(QTabWidget(), "Select")
        navigation = WorkflowNavigation(tabs)
        status = StatusBarView()
        try:
            self.assertTrue(navigation.sidebar_toggle_btn.accessibleName())
            self.assertTrue(navigation.sidebar_toggle_btn.toolTip())
            self.assertTrue(status.update_button.accessibleName())
            self.assertEqual(status.cursor_readback.focusPolicy().value, 0)
        finally:
            navigation.close()
            status.close()
            tabs.close()

    def test_minimum_window_keeps_scrollable_controls_inside_viewport(self) -> None:
        from ui_qt.main_window import MainWindow

        window = MainWindow()
        try:
            pl_index = next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == "PL")
            window.tabs.setCurrentIndex(pl_index)
            window.resize(window.minimumSize())
            window.show()
            self.app.processEvents()
            scroll = window.pl_tab_scroll
            self.assertLessEqual(scroll.widget().width(), scroll.viewport().width())
            cmap = window.pl_cmap
            self.assertLessEqual(cmap.geometry().right(), cmap.parentWidget().rect().right())
        finally:
            window.close()

    def test_source_toolbar_reuses_source_controls_with_accessible_order(self) -> None:
        from ui_qt.main_window import MainWindow

        window = MainWindow()
        try:
            self.assertIsInstance(window.recent_folder_combo, QComboBox)
            self.assertIs(window.recent_folder_combo.lineEdit(), window.folder_edit)
            self.assertEqual(window.data_source_context.objectName(), "dataSourceContext")
            self.assertEqual(window.folder_edit.objectName(), "folderEdit")
            for control in (window.browse_btn, window.open_file_btn, window.refresh_btn):
                self.assertIsInstance(control, QPushButton)
                self.assertTrue(control.accessibleName())
                self.assertTrue(control.toolTip())
            self.assertTrue(window.folder_edit.toolTip())
            self.assertTrue(window.recent_folder_combo.toolTip())
            host = window.menu_toolbar_host
            self.assertIs(host.source_widget_action.defaultWidget(), window.data_source_context)
            self.assertIs(host.main_toolbar.widgetForAction(host.source_widget_action), window.data_source_context)
            order = [window.recent_folder_combo, window.browse_btn, window.open_file_btn, window.refresh_btn]
            positions = [window.data_source_context.mapFromGlobal(widget.mapToGlobal(widget.rect().topLeft())).x() for widget in order]
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(window.data_source_context.layout().count(), 5)
            self.assertEqual(window.data_source_context.layout().itemAt(1).widget(), window.recent_folder_combo)
        finally:
            window.close()

    def test_source_toolbar_geometry_fits_at_minimum_window_and_programmatic_refresh_is_silent(self) -> None:
        from ui_qt.main_window import MainWindow

        window = MainWindow()
        try:
            window.resize(window.minimumSize())
            window.show()
            self.app.processEvents()
            source = window.data_source_context
            self.assertGreater(source.width(), 0)
            self.assertTrue(source.isVisible())
            self.assertFalse(window.menu_toolbar_host.source_separator_action.isSeparator() and not window.menu_toolbar_host.source_widget_action.isVisible())
            before = window.current_folder
            window.recent_folders = []
            window._populate_recent_folder_combo()
            self.assertEqual(window.current_folder, before)
        finally:
            window.close()

    def test_source_context_is_compact_single_row_with_selector_contraction_priority(self) -> None:
        from ui_qt.main_window import MainWindow

        window = MainWindow()
        try:
            window.resize(window.minimumSize())
            window.show()
            self.app.processEvents()
            context = window.data_source_context
            self.assertEqual(context.metaObject().className(), "QWidget")
            self.assertEqual(context.sizePolicy().horizontalPolicy(), QSizePolicy.Expanding)
            self.assertEqual(context.layout().count(), 5)
            self.assertEqual(context.height(), context.sizeHint().height())
            combo = window.recent_folder_combo
            self.assertEqual(combo.sizePolicy().horizontalPolicy(), QSizePolicy.Ignored)
            self.assertGreater(context.layout().stretch(1), 0)
            for button in (window.browse_btn, window.open_file_btn, window.refresh_btn):
                self.assertGreaterEqual(button.width(), button.sizeHint().width())
            self.assertEqual(window.centralWidget().layout().count(), 2)
        finally:
            window.close()

    def test_recent_folder_refresh_keeps_full_current_path_and_tooltip(self) -> None:
        from ui_qt.main_window import MainWindow

        window = MainWindow()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                folder = str(Path(tmp).resolve())
                window.current_folder = folder
                window.recent_folders = [folder]
                window._populate_recent_folder_combo()
                self.assertEqual(window.folder_edit.text(), folder)
                self.assertEqual(window.recent_folder_combo.itemData(1, Qt.ToolTipRole), folder)
        finally:
            window.close()

    def test_spinbox_edit_field_reserves_text_space_from_steppers(self) -> None:
        """The rendered app style keeps numeric text clear of both steppers."""
        from PySide6.QtWidgets import QStyle, QStyleOptionSpinBox, QToolButton

        from ui_qt.main_window import MainWindow, UI_METRICS
        from ui_qt.theme import install_theme

        # Exercise the same generated application QSS used by the desktop app.
        install_theme(self.app, mode="light")
        window = MainWindow()
        try:
            window.resize(1180, 820)
            window.show()
            window.workspace_splitter.setSizes([UI_METRICS["left_width"], 900])
            pl_index = next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == "PL")
            window.tabs.setCurrentIndex(pl_index)
            manual_head = next(
                button
                for button in window.pl_tab_scroll.widget().findChildren(QToolButton)
                if button.text() == "Manual plot ranges"
            )
            manual_head.setChecked(True)
            self.app.processEvents()

            spin = window.pl_spins["vmin"]
            original_read_only = spin.isReadOnly()
            original_property = spin.property("readOnly")
            try:
                for read_only in (False, True):
                    spin.setReadOnly(read_only)
                    spin.setProperty("readOnly", read_only)
                    spin.style().unpolish(spin)
                    spin.style().polish(spin)
                    self.app.processEvents()

                    option = QStyleOptionSpinBox()
                    spin.initStyleOption(option)
                    edit = spin.style().subControlRect(
                        QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxEditField, spin
                    )
                    up = spin.style().subControlRect(
                        QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxUp, spin
                    )
                    down = spin.style().subControlRect(
                        QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxDown, spin
                    )
                    self.assertFalse(edit.intersects(up))
                    self.assertFalse(edit.intersects(down))
                    self.assertGreaterEqual(up.left() - edit.right() - 1, 2)
                    self.assertGreaterEqual(down.left() - edit.right() - 1, 2)
                    self.assertTrue(spin.rect().contains(up))
                    self.assertTrue(spin.rect().contains(down))
            finally:
                spin.setReadOnly(original_read_only)
                spin.setProperty("readOnly", original_property)
                spin.style().unpolish(spin)
                spin.style().polish(spin)
                self.app.processEvents()
        finally:
            window.close()

    def test_production_spinboxes_keep_safe_edit_capacity_in_all_workflows(self) -> None:
        """Real workflow spinboxes retain text space beside both steppers."""
        from ui_qt.main_window import MainWindow, UI_METRICS
        from ui_qt.theme import install_theme

        install_theme(self.app, mode="light")
        window = MainWindow()
        try:
            window.resize(1180, 820)
            window.show()
            window.workspace_splitter.setSizes([UI_METRICS["left_width"], 900])
            self.app.processEvents()
            for tab_name, head_text in (("PL", "Manual plot ranges"), ("DRR", "Manual plot ranges"), ("Compare", "VP"), ("MCD", "Correction")):
                index = next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == tab_name)
                window.tabs.setCurrentIndex(index)
                head = next(
                    button
                    for button in window.tabs.widget(index).findChildren(QToolButton)
                    if button.text() == head_text
                )
                head.setChecked(True)
                self.app.processEvents()

            controls = (
                ("PL", window.pl_spins["vmin"]),
                ("DRR", window.drr_spins["vmin"]),
                ("Compare", window.cmp_vp_background_spin),
                ("MCD", window.mcd_zero_spin),
            )
            for workflow, spin in controls:
                self.assertGreater(spin.width(), 0, workflow)
                original_read_only = spin.isReadOnly()
                original_property = spin.property("readOnly")
                try:
                    for read_only in (False, True):
                        spin.setReadOnly(read_only)
                        spin.setProperty("readOnly", read_only)
                        spin.style().unpolish(spin)
                        spin.style().polish(spin)
                        self.app.processEvents()
                        option = QStyleOptionSpinBox()
                        spin.initStyleOption(option)
                        edit = spin.style().subControlRect(
                            QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxEditField, spin
                        )
                        up = spin.style().subControlRect(
                            QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxUp, spin
                        )
                        down = spin.style().subControlRect(
                            QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxDown, spin
                        )
                        self.assertGreaterEqual(edit.width(), spin.width() - 34, workflow)
                        self.assertGreaterEqual(up.width(), 20, workflow)
                        self.assertEqual(up.width(), 20, workflow)
                        self.assertEqual(down.width(), 20, workflow)
                        self.assertFalse(edit.intersects(up), workflow)
                        self.assertFalse(edit.intersects(down), workflow)
                        self.assertGreaterEqual(up.left() - edit.right() - 1, 2, workflow)
                        self.assertGreaterEqual(down.left() - edit.right() - 1, 2, workflow)
                        self.assertTrue(spin.rect().contains(up), workflow)
                        self.assertTrue(spin.rect().contains(down), workflow)
                        self.assertGreater(spin.lineEdit().fontMetrics().horizontalAdvance(spin.text()), 0)
                finally:
                    spin.setReadOnly(original_read_only)
                    spin.setProperty("readOnly", original_property)
                    spin.style().unpolish(spin)
                    spin.style().polish(spin)
                    self.app.processEvents()
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
