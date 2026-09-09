from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QStyle,
    QStyleOptionButton,
    QStyleOptionComboBox,
    QStyleOptionSpinBox,
    QToolButton,
)

from ui_qt.main_window import MainWindow
from ui_qt.theme import install_theme
from scripts.preview_ui import build_preview_window


class SidebarLayoutRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode="light")

    def _window(self, workflow: str, width: int) -> MainWindow:
        window, self.app, _settings_dir = build_preview_window(
            workflow=workflow,
            theme="light",
            scale="1",
            width=1180,
            height=820,
            sidebar_width=width,
            demo_data=False,
        )
        window.show()
        self.app.processEvents()
        return window

    def _close_window(self, window: MainWindow) -> None:
        window.close()
        window.deleteLater()
        self.app.processEvents()
        for name in ("_preview_settings_temp", "_preview_mpl_cache_temp"):
            temporary = getattr(window, name, None)
            if temporary is not None:
                temporary.cleanup()

    def _expand(self, window: MainWindow, *titles: str) -> None:
        wanted = set(titles)
        for button in window.findChildren(QToolButton):
            if button.text() in wanted and not button.isChecked():
                button.setChecked(True)
        self.app.processEvents()

    @staticmethod
    def _form_labels(form: QFormLayout) -> list[QLabel]:
        labels: list[QLabel] = []
        for row in range(form.rowCount()):
            item = form.itemAt(row, QFormLayout.LabelRole)
            if item is not None and isinstance(item.widget(), QLabel):
                labels.append(item.widget())
        return labels

    def test_shg_long_labels_wrap_inside_expanded_forms_at_compact_width(self) -> None:
        window = self._window("SHG", 320)
        try:
            self._expand(window, "Peak Integration", "Cosmic Rays", "Angle")
            for width in (320, 380):
                window.workspace_splitter.setSizes([width, max(1, window.workspace_splitter.width() - width)])
                self.app.processEvents()
                forms = [
                    window.shg_peak_center_spin.parentWidget().parentWidget().layout(),
                    window.shg_cosmic_threshold_spin.parentWidget().layout(),
                    window.shg_angle_scale_spin.parentWidget().layout(),
                ]
                forms = [form for form in forms if isinstance(form, QFormLayout)]
                labels = [label for form in forms for label in self._form_labels(form)]
                expected = {
                    "Integration wavelength",
                    "Integration range",
                    "Threshold (MAD)",
                    "Detection window",
                }
                found = {label.text() for label in labels}
                self.assertTrue(
                    all(any(text.startswith(prefix) for text in found) for prefix in expected),
                    found,
                )
                for label in labels:
                    if not any(label.text().startswith(prefix) for prefix in expected):
                        continue
                    self.assertTrue(label.wordWrap(), label.text())
                    self.assertGreaterEqual(label.height(), label.heightForWidth(label.width()), label.text())
                    for word in label.text().split():
                        self.assertLessEqual(
                            label.fontMetrics().horizontalAdvance(word), label.width(), label.text()
                        )
        finally:
            self._close_window(window)

    def test_power_selected_combos_retain_edit_area_at_compact_width(self) -> None:
        window = self._window("Power", 320)
        try:
            for width in (320, 380):
                window.workspace_splitter.setSizes([width, max(1, window.workspace_splitter.width() - width)])
                for axis_scale in ("Linear", "Log"):
                    for pair_mode in ("Stage", "Power Interpolation"):
                        window.power_axis_scale_combo.setCurrentText(axis_scale)
                        window.power_pair_mode_combo.setCurrentText(pair_mode)
                        self.app.processEvents()
                        for combo in (window.power_axis_scale_combo, window.power_pair_mode_combo):
                            option = QStyleOptionComboBox()
                            combo.initStyleOption(option)
                            option.rect = QRect(0, 0, combo.width(), combo.height())
                            edit = combo.style().subControlRect(
                                QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, combo
                            )
                            required = combo.fontMetrics().horizontalAdvance(combo.currentText())
                            self.assertGreaterEqual(edit.width(), required, combo.currentText())
                            self.assertGreaterEqual(combo.width(), combo.sizeHint().height())
                auto = window.power_background_auto_chk
                option = QStyleOptionButton()
                auto.initStyleOption(option)
                auto_required = auto.fontMetrics().horizontalAdvance(auto.text())
                auto_required += auto.style().pixelMetric(QStyle.PM_IndicatorWidth, option, auto)
                auto_required += auto.style().pixelMetric(QStyle.PM_CheckBoxLabelSpacing, option, auto)
                self.assertGreaterEqual(auto.width(), auto_required, auto.text())
                self.assertTrue(auto.parentWidget().contentsRect().contains(auto.geometry()))
        finally:
            self._close_window(window)

    def test_mcd_numeric_inputs_use_content_width_at_compact_width(self) -> None:
        window = self._window("MCD Peak Shift", 380)
        try:
            self._expand(window, "Advanced detection")
            controls = (
                window.mcd_peak_prom_spin,
                window.mcd_peak_dist_spin,
                window.mcd_peak_smooth_spin,
                window.mcd_peak_jump_spin,
                window.mcd_peak_max_spin,
            )
            viewport = window.mcd_peak_shift_tab_scroll.viewport()
            for sidebar_width in (320, 380, 480, 320):
                window.workspace_splitter.setSizes(
                    [sidebar_width, max(1, window.workspace_splitter.width() - sidebar_width)]
                )
                self.app.processEvents()
                for spin in controls:
                    window.mcd_peak_shift_tab_scroll.ensureWidgetVisible(spin)
                    self.app.processEvents()
                    self.assertTrue(spin.isVisible(), spin.text())
                    self.assertLessEqual(spin.width(), 240, spin.objectName() or spin.text())
                    self.assertGreaterEqual(spin.width(), spin.minimumSizeHint().width(), spin.text())
                    viewport_rect = QRect(spin.mapTo(viewport, QPoint(0, 0)), spin.size())
                    self.assertTrue(viewport.rect().contains(viewport_rect), f"{spin.text()} {viewport_rect}")
                    original = spin.value()
                    for value in (spin.minimum(), spin.maximum(), original):
                        spin.setValue(value)
                        option = QStyleOptionSpinBox()
                        spin.initStyleOption(option)
                        option.rect = QRect(0, 0, spin.width(), spin.height())
                        edit = spin.style().subControlRect(
                            QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxEditField, spin
                        )
                        self.assertGreaterEqual(
                            edit.width(), spin.lineEdit().fontMetrics().horizontalAdvance(spin.text()), spin.text()
                        )
        finally:
            self._close_window(window)

    def test_drr_flags_keep_checkbox_content_inside_compact_form(self) -> None:
        window = self._window("DRR", 320)
        try:
            self._expand(window, "Manual plot ranges")
            for width in (320, 380):
                window.workspace_splitter.setSizes([width, max(1, window.workspace_splitter.width() - width)])
                self.app.processEvents()
                flags = window.drr_center_zero_chk.parentWidget()
                self.assertIsNotNone(flags)
                window.drr_tab_scroll.ensureWidgetVisible(flags)
                self.app.processEvents()
                checkboxes = flags.findChildren(QCheckBox, options=Qt.FindDirectChildrenOnly)
                self.assertEqual({check.text() for check in checkboxes}, {"Log Scale", "Clip Outliers", "Center Zero"})
                contents = flags.contentsRect()
                for check in checkboxes:
                    option = QStyleOptionButton()
                    check.initStyleOption(option)
                    required = check.fontMetrics().horizontalAdvance(check.text())
                    required += check.style().pixelMetric(QStyle.PM_IndicatorWidth, option, check)
                    required += check.style().pixelMetric(QStyle.PM_CheckBoxLabelSpacing, option, check)
                    self.assertGreaterEqual(check.width(), required, check.text())
                    self.assertTrue(
                        contents.contains(check.geometry()),
                        f"{check.text()} rect={check.geometry()} contents={contents}",
                    )
                for button in (
                    window.drr_edit_measurements_btn,
                    window.drr_clear_measurements_btn,
                    window.drr_edit_baselines_btn,
                    window.drr_baseline_autofind_btn,
                ):
                    parent = button.parentWidget()
                    self.assertIsNotNone(parent)
                    self.assertTrue(
                        parent.contentsRect().contains(button.geometry()),
                        f"{button.text()} rect={button.geometry()} parent={parent.geometry()} contents={parent.contentsRect()}",
                    )
                    self.assertGreaterEqual(button.height(), button.sizeHint().height(), button.text())
                for control in (
                    window.drr_baseline_combo,
                    window.drr_cmap,
                    window.drr_derivative_combo,
                ):
                    if not control.isVisible():
                        continue
                    parent = control.parentWidget()
                    self.assertIsNotNone(parent)
                    self.assertTrue(
                        parent.contentsRect().contains(control.geometry()),
                        f"{type(control).__name__} rect={control.geometry()} parent={parent.contentsRect()}",
                    )
        finally:
            self._close_window(window)

    def test_drr_external_and_sg_controls_fit_conditionally_at_compact_width(self) -> None:
        window = self._window("DRR", 320)
        try:
            window.drr_baseline_combo.setCurrentText("External")
            window.drr_derivative_combo.setCurrentText("dE")
            self.app.processEvents()
            pin = window.drr_pin_baseline_chk
            self.assertLessEqual(pin.fontMetrics().horizontalAdvance(pin.text()), pin.width())
            self.assertEqual(window.drr_sg_window_spin.prefix(), "W ")
            self.assertEqual(window.drr_sg_poly_spin.prefix(), "O ")
            for sidebar_width in (320, 380):
                window.workspace_splitter.setSizes(
                    [sidebar_width, max(1, window.workspace_splitter.width() - sidebar_width)]
                )
                self.app.processEvents()
                for control in (
                    pin,
                    window.drr_derivative_combo,
                    window.drr_sg_window_spin,
                    window.drr_sg_poly_spin,
                ):
                    description = control.objectName() or getattr(control, "text", lambda: "")()
                    self.assertTrue(control.isVisible(), description)
                    parent = control.parentWidget()
                    self.assertIsNotNone(parent)
                    self.assertTrue(parent.contentsRect().contains(control.geometry()))
        finally:
            self._close_window(window)

    def test_mcd_source_buttons_fit_their_row_at_compact_width(self) -> None:
        window = self._window("MCD", 320)
        try:
            source_row = window.mcd_select_source_btn.parentWidget()
            self.assertIsNotNone(source_row)
            for width in (320, 380):
                window.workspace_splitter.setSizes([width, max(1, window.workspace_splitter.width() - width)])
                self.app.processEvents()
                self.assertTrue(source_row.contentsRect().contains(window.mcd_select_source_btn.geometry()))
                self.assertTrue(source_row.contentsRect().contains(window.mcd_clear_source_btn.geometry()))
                for button in (window.mcd_select_source_btn, window.mcd_clear_source_btn):
                    self.assertGreaterEqual(button.height(), button.sizeHint().height(), button.text())
        finally:
            self._close_window(window)
