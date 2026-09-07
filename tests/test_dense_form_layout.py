from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent, QRect
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox, QLabel, QPushButton, QToolButton, QWidget, QSizePolicy, QStyle, QStyleOptionSpinBox, QStyleOptionButton

from ui_qt.main_window import MainWindow, UI_METRICS
from ui_qt.theme import install_theme


class DenseFormRowLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode="light")

    def test_layout_uses_only_single_row_and_wrapped_modes(self) -> None:
        from ui_qt.dense_form_layout import DenseFormRowLayout

        host = QWidget()
        layout = DenseFormRowLayout(host, spacing=4)
        host.setLayout(layout)
        first = QDoubleSpinBox(); first.setDecimals(4); first.setRange(-100, 100); first.setValue(0)
        first_fix = QCheckBox("F")
        second = QDoubleSpinBox(); second.setDecimals(4); second.setRange(-100, 100); second.setValue(-12)
        second_fix = QCheckBox("F")
        auto = QPushButton("Auto")
        layout.add_group((first, first_fix), role="range", priority=10, grow_weight=1)
        layout.add_group((second, second_fix), role="range", priority=10, grow_weight=1)
        layout.add_group((auto,), role="action", priority=1, grow_weight=0)
        host.resize(600, 120)
        host.show(); self.app.processEvents()
        layout.setGeometry(host.contentsRect())
        self.assertEqual(layout.mode_for_width(host.width()), "SINGLE_ROW")
        self.assertEqual(first.geometry().y(), second.geometry().y())
        self.assertLess(abs(first.geometry().center().y() - first_fix.geometry().center().y()), 12)
        self.assertLess(abs(second.geometry().center().y() - second_fix.geometry().center().y()), 12)
        self.assertLess(abs(auto.geometry().center().y() - first.geometry().center().y()), 2)

        pair_width = sum(layout.safe_min_width(widget) for widget in (first, first_fix, second, second_fix)) + layout.spacing() * 3
        host.resize(pair_width + 12, 160)
        self.app.processEvents(); layout.setGeometry(host.contentsRect())
        self.assertEqual(layout.mode_for_width(host.width()), "WRAPPED")
        self.assertEqual(first.geometry().y(), second.geometry().y())
        self.assertGreater(auto.geometry().y(), first.geometry().y())
        self.assertFalse(first.geometry().intersects(first_fix.geometry()))
        self.assertFalse(second.geometry().intersects(second_fix.geometry()))

    def test_height_for_width_recalculates_when_width_changes(self) -> None:
        from ui_qt.dense_form_layout import DenseFormRowLayout

        host = QWidget(); layout = DenseFormRowLayout(host, spacing=4); host.setLayout(layout)
        for text in ("First", "Second"):
            layout.add_group((QPushButton(text),), role="control", priority=1, grow_weight=1)
        layout.add_group((QPushButton("Auto"),), role="action", priority=0, grow_weight=0)
        wide = layout.heightForWidth(420)
        narrow = layout.heightForWidth(120)
        self.assertGreater(narrow, wide)
        self.assertTrue(layout.hasHeightForWidth())

    def _make_range_row(self, width: int = 380):
        from ui_qt.dense_form_layout import DenseFormRowLayout

        host = QWidget()
        label = QLabel("vmin / vmax")
        layout = DenseFormRowLayout(host, spacing=4, label=label)
        host.setLayout(layout)
        first = QDoubleSpinBox(); first.setDecimals(4); first.setRange(-100, 100); first.setValue(0)
        first_fix = QCheckBox("F")
        second = QDoubleSpinBox(); second.setDecimals(4); second.setRange(-100, 100); second.setValue(-12)
        second_fix = QCheckBox("F")
        auto = QPushButton("Auto")
        for spin in (first, second):
            spin.setMinimumWidth(0)
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for fix in (first_fix, second_fix):
            fix.setMinimumWidth(0)
        auto.setFlat(False)
        layout.add_group((first, first_fix), role="range", priority=10, grow_weight=1)
        layout.add_group((second, second_fix), role="range", priority=10, grow_weight=1)
        layout.add_group((auto,), role="action", priority=1, grow_weight=0)
        host.resize(width, 120); host.show(); self.app.processEvents(); layout.setGeometry(host.contentsRect())
        return host, layout, (first, first_fix, second, second_fix, auto)

    def test_drr_range_controls_share_one_control_row_at_380(self) -> None:
        host, layout, controls = self._make_range_row()
        try:
            measured_width = sum(layout.safe_min_width(widget) for widget in controls)
            measured_width += layout.spacing() * (len(controls) - 1)
            expected_mode = "SINGLE_ROW" if measured_width <= host.width() else "WRAPPED"
            self.assertEqual(layout.mode_for_width(host.width()), expected_mode)
            first_row_centers = [widget.geometry().center().y() for widget in controls[:4]]
            self.assertLess(max(first_row_centers) - min(first_row_centers), 12, controls)
            if expected_mode == "SINGLE_ROW":
                self.assertLess(
                    abs(controls[4].geometry().center().y() - controls[0].geometry().center().y()),
                    2,
                )
            else:
                self.assertGreater(controls[4].geometry().top(), controls[0].geometry().bottom())
            self.assertTrue(layout.labelWidget().isVisible())
            self.assertLess(layout.labelWidget().geometry().bottom(), min(widget.geometry().top() for widget in controls))
            self.assertTrue(all(host.contentsRect().contains(widget.geometry()) for widget in controls))
            self.assertFalse(any(a.geometry().intersects(b.geometry()) for i, a in enumerate(controls) for b in controls[i + 1:]))
            self.assertFalse(controls[4].isFlat())
        finally:
            host.close(); host.deleteLater(); self.app.processEvents()

    def test_rendered_representative_text_fits_style_content_rects(self) -> None:
        host, layout, controls = self._make_range_row()
        try:
            for spin, text in ((controls[0], "0.0000"), (controls[2], "-12.0000")):
                option = QStyleOptionSpinBox(); spin.initStyleOption(option)
                option.rect = QRect(0, 0, spin.width(), spin.height())
                edit = spin.style().subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxEditField, spin)
                margins = spin.lineEdit().textMargins()
                available = edit.width() - margins.left() - margins.right()
                self.assertGreaterEqual(available, spin.lineEdit().fontMetrics().horizontalAdvance(text))

            auto = controls[4]
            option_btn = QStyleOptionButton(); auto.initStyleOption(option_btn)
            option_btn.rect = QRect(0, 0, auto.width(), auto.height())
            content = auto.style().subElementRect(QStyle.SE_PushButtonContents, option_btn, auto)
            self.assertGreaterEqual(content.width(), auto.fontMetrics().horizontalAdvance(auto.text()))

            clicked = QSignalSpy(auto.clicked)
            auto.click()
            self.assertEqual(clicked.count(), 1)
        finally:
            host.close(); host.deleteLater(); self.app.processEvents()

    def test_width_measurement_is_bounded_and_independent_of_live_geometry(self) -> None:
        host, layout, controls = self._make_range_row()
        try:
            spin = controls[0]
            measured_before = layout.safe_min_width(spin)
            spin.resize(spin.width() + 80, spin.height())
            self.assertEqual(layout.safe_min_width(spin), measured_before)
            self.assertLess(measured_before, spin.minimumSizeHint().width())
        finally:
            host.close(); host.deleteLater(); self.app.processEvents()

    def test_set_label_widget_replaces_and_removes_owned_layout_item(self) -> None:
        from ui_qt.dense_form_layout import DenseFormRowLayout

        host = QWidget(); first = QLabel("first"); second = QLabel("second"); button = QPushButton("Go")
        layout = DenseFormRowLayout(host, spacing=4, label=first); host.setLayout(layout); layout.addWidget(button)
        self.assertEqual(layout.count(), 2)
        layout.set_label_widget(second)
        self.assertEqual(layout.count(), 2)
        self.assertIs(layout.labelWidget(), second)
        self.assertIs(layout.itemAt(0).widget(), second)
        self.assertNotIn(first, [layout.itemAt(i).widget() for i in range(layout.count())])
        layout.set_label_widget(None)
        self.assertEqual(layout.count(), 1)
        self.assertIsNone(layout.labelWidget())
        self.assertIs(layout.itemAt(0).widget(), button)
        third = QLabel("third")
        layout.set_label_widget(third)
        taken = layout.takeAt(0)
        self.assertIs(taken.widget(), third)
        self.assertIsNone(layout.labelWidget())
        self.assertFalse(third.isVisible())

    def test_layout_request_and_dpr_events_invalidate(self) -> None:
        from ui_qt.dense_form_layout import DenseFormRowLayout

        host = QWidget(); layout = DenseFormRowLayout(host); host.setLayout(layout)
        with patch.object(layout, "invalidate", wraps=layout.invalidate) as invalidate:
            layout.eventFilter(host, QEvent(QEvent.LayoutRequest))
            layout.eventFilter(host, QEvent(QEvent.DevicePixelRatioChange))
            self.assertGreaterEqual(invalidate.call_count, 2)

    def _assert_real_axis_rows_use_dense_layout(self, mode: str, prefix: str) -> None:
        from ui_qt.dense_form_layout import DenseFormRowLayout

        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            window = MainWindow()
        try:
            window.resize(1180, 820); window.show(); self.app.processEvents()
            window.workspace_splitter.setSizes([UI_METRICS["left_width"], 900]); self.app.processEvents()
            index = next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == mode)
            window.tabs.setCurrentIndex(index)
            manual = next(button for button in window.tabs.widget(index).findChildren(QToolButton) if button.text() == "Manual plot ranges")
            manual.setChecked(True)
            self.app.processEvents()
            for axis in ("vmin", "xmin", "ymin"):
                spins = getattr(window, f"{prefix}_spins")
                fixes = getattr(window, f"{prefix}_fix_checks")
                row = spins[axis].parentWidget()
                controls = [spins[axis], fixes[axis], spins[axis.replace("min", "max")], fixes[axis.replace("min", "max")], getattr(window, f"{prefix}_auto_{axis[0]}_btn")]
                self.assertIsInstance(row.layout(), DenseFormRowLayout)
                self.assertTrue(row.layout().labelWidget().isVisible())
                direct = [row.layout().itemAt(i).widget() for i in range(row.layout().count()) if row.layout().itemAt(i).widget() is not row.layout().labelWidget()]
                target_width = 380
                row.resize(target_width, row.layout().heightForWidth(target_width))
                row.layout().setGeometry(row.contentsRect())
                width = row.contentsRect().width()
                measured_width = sum(row.layout().safe_min_width(widget) for widget in direct[:5])
                measured_width += row.layout().spacing() * 4
                expected_mode = "SINGLE_ROW" if measured_width <= width else "WRAPPED"
                self.assertEqual(window.workspace_splitter.sizes()[0], 380)
                self.assertEqual(row.layout().mode_for_width(width), expected_mode)
                self.assertLess(row.layout().labelWidget().geometry().bottom(), min(widget.geometry().top() for widget in direct[:5]))
                first_row_centers = [widget.geometry().center().y() for widget in direct[:4]]
                self.assertLess(max(first_row_centers) - min(first_row_centers), 12)
                if expected_mode == "SINGLE_ROW":
                    self.assertLess(abs(direct[4].geometry().center().y() - direct[0].geometry().center().y()), 2)
                else:
                    self.assertGreater(direct[4].geometry().top(), direct[0].geometry().bottom())
                self.assertLessEqual(max(w.geometry().right() for w in direct[:5]), row.contentsRect().right())
                self.assertTrue(all(row.contentsRect().contains(widget.geometry()) for widget in direct[:5]))
                self.assertFalse(any(a.geometry().intersects(b.geometry()) for i, a in enumerate(direct[:5]) for b in direct[i + 1:5]))
                for spin, text in ((direct[0], "0.0000"), (direct[2], "-12.0000")):
                    option = QStyleOptionSpinBox(); spin.initStyleOption(option)
                    option.rect = QRect(0, 0, spin.width(), spin.height())
                    edit = spin.style().subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxEditField, spin)
                    margins = spin.lineEdit().textMargins()
                    available = edit.width() - margins.left() - margins.right()
                    self.assertGreaterEqual(available, spin.lineEdit().fontMetrics().horizontalAdvance(text))
                self.assertIsInstance(controls[1], QCheckBox)
                self.assertIsInstance(controls[3], QCheckBox)
                self.assertIsInstance(controls[4], (QPushButton, QToolButton))
                original_checked = controls[1].isChecked()
                controls[1].click()
                self.assertIs(controls[1].isChecked(), not original_checked)
                controls[1].click()
                self.assertIs(controls[1].isChecked(), original_checked)
            spin = getattr(window, f"{prefix}_spins")["vmin"]
            measured_before = row.layout()._widget_min_width(spin)
            spin.resize(spin.width() + 40, spin.height())
            measured_after = row.layout()._widget_min_width(spin)
            self.assertEqual(measured_before, measured_after)
            self.assertEqual(getattr(window, f"{prefix}_auto_v_btn").text(), "Auto")
        finally:
            window.close(); window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            self.app.processEvents()

    def test_real_pl_axis_rows_use_dense_layout_at_sidebar_width(self) -> None:
        self._assert_real_axis_rows_use_dense_layout("PL", "pl")

    def test_real_drr_axis_rows_use_dense_layout_at_sidebar_width(self) -> None:
        self._assert_real_axis_rows_use_dense_layout("DRR", "drr")

    def test_real_compare_axis_rows_use_dense_layout_at_sidebar_width(self) -> None:
        self._assert_real_axis_rows_use_dense_layout("Compare", "cmp")

    def test_real_widget_ownership_roots_and_mcd_descendants_teardown(self) -> None:
        import shiboken6

        install_theme(self.app, mode="light")
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            window = MainWindow()

        roots = [
            window.pl_yaxis_controls,
            window.drr_yaxis_controls,
            window.cmp_yaxis_controls,
            window._tools_tab_placeholder,
            window.mcd_split_scale_panel,
            window.mcd_split_scale_chk,
        ]
        mcd_descendants = [
            *window.mcd_split_spins.values(),
            *window.mcd_split_fix_checks.values(),
            window.mcd_split_boundary_chk,
            window.mcd_split_auto_left_btn,
            window.mcd_split_auto_right_btn,
        ]
        retained = [*roots, *mcd_descendants]
        destroyed_spies = [(widget, QSignalSpy(widget.destroyed)) for widget in retained]

        def lifetime_descends(widget: QWidget, ancestor: QWidget) -> bool:
            current = widget.parent()
            while current is not None:
                if current is ancestor:
                    return True
                current = current.parent()
            return False

        try:
            for root in roots:
                self.assertTrue(lifetime_descends(root, window), root.objectName())
            self.assertTrue(lifetime_descends(window.mcd_split_scale_panel, window))
            for descendant in mcd_descendants:
                self.assertTrue(lifetime_descends(descendant, window), type(descendant).__name__)

            window.resize(1180, 820)
            window.show()
            self.app.processEvents()
            for label in ("PL", "DRR", "Compare", "Power", "MCD", "MCD Peak Shift", "SHG", "Tools", "PL"):
                index = next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == label)
                window.tabs.setCurrentIndex(index)
                self.app.processEvents()
                for prefix in ("pl", "drr", "cmp"):
                    self.assertFalse(getattr(window, f"{prefix}_yaxis_controls").isVisible())
                    self.assertFalse(getattr(window, f"{prefix}_yaxis_advanced_box").isVisible())
                self.assertFalse(window.mcd_split_scale_panel.isVisible())
                self.assertFalse(window.mcd_split_scale_chk.isVisible())
                if label != "Tools":
                    self.assertFalse(window._tools_tab_placeholder.isVisible())
        finally:
            window.close()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            self.app.processEvents()
            for widget, spy in destroyed_spies:
                self.assertFalse(shiboken6.isValid(widget), type(widget).__name__)
                self.assertGreaterEqual(spy.count(), 1, type(widget).__name__)

    def test_real_power_axis_rows_use_dense_layout_at_sidebar_width(self) -> None:
        self._assert_real_axis_rows_use_dense_layout("Power", "power")


if __name__ == "__main__":
    unittest.main()
