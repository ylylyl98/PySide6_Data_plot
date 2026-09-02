"""Workflow feature page builders.

These builders own the feature-specific UI. The main window remains the
shared application context for state and event handlers during the next
refactoring stage.
"""

from __future__ import annotations

import csv
from typing import Dict

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.mcd_peak_shift import analyze_peak_shift, valley_quantities
from core.plotting import COMPARE_PANEL_ORDER
from ui_qt.common import UI_METRICS, QComboBox, QDoubleSpinBox, QSpinBox
from ui_qt.fluent_ui.style import set_fluent_property
from ui_qt.status_badge import StatusBadge


class FeatureTabsMixin:
    def _build_pl_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        files = QGroupBox("File")
        files_layout = QVBoxLayout(files)
        files_layout.setContentsMargins(6, 6, 6, 6)
        self.pl_files = QListWidget(files)
        self.pl_files.setSelectionMode(QAbstractItemView.SingleSelection)
        self.pl_files.hide()
        self.pl_files.itemSelectionChanged.connect(self.pl_controller._on_pl_selection_changed)
        source_row = QWidget()
        source_grid = QGridLayout(source_row)
        source_grid.setContentsMargins(0, 0, 0, 0)
        source_grid.setHorizontalSpacing(6)
        source_grid.setVerticalSpacing(4)
        self.pl_selection_summary = StatusBadge("No PL file selected.", app_role=None)
        self.pl_selection_summary.setMinimumWidth(0)
        self.pl_selection_summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.pl_select_source_btn = QPushButton("Select...")
        self.pl_select_source_btn.setFixedHeight(30)
        self.pl_select_source_btn.setMaximumWidth(92)
        self.pl_clear_source_btn = QPushButton("Clear")
        self.pl_clear_source_btn.setFixedHeight(30)
        self.pl_clear_source_btn.setMaximumWidth(72)
        source_grid.addWidget(self.pl_selection_summary, 0, 0, 1, 3)
        source_grid.setColumnStretch(0, 1)
        source_grid.addWidget(self.pl_select_source_btn, 1, 1)
        source_grid.addWidget(self.pl_clear_source_btn, 1, 2)
        files_layout.addWidget(source_row)
        self.pl_auto_next_chk = QCheckBox("Auto-load next new file after Save")
        auto_next_value = self.settings.value(self.SETTINGS_PL_AUTO_NEXT, True)
        self.pl_auto_next_chk.setChecked(
            str(auto_next_value).strip().casefold() not in {"0", "false", "no", "off"}
        )
        self.pl_auto_next_chk.setToolTip(
            "After a successful first save, automatically load the next newest unprocessed PL source."
        )
        files_layout.addWidget(self.pl_auto_next_chk)
        self.pl_select_source_btn.clicked.connect(self.pl_controller._edit_pl_source)
        self.pl_clear_source_btn.clicked.connect(self.pl_controller._clear_pl_source)
        self.pl_auto_next_chk.toggled.connect(
            lambda checked: self.settings.setValue(self.SETTINGS_PL_AUTO_NEXT, bool(checked))
        )
        layout.addWidget(self._make_expander("Measurement File", files, expanded=True))

        params = QGroupBox("Plot Options")
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(6, 6, 6, 4)
        params_layout.setSpacing(4)
        cfg = QFormLayout()
        cfg.setHorizontalSpacing(6)
        cfg.setVerticalSpacing(4)
        _grid, spins, _, _, cmap, fix_checks = self._build_common_range_grid("pl", "turbo")
        self.pl_yaxis_controls = self._build_y_axis_controls("pl")
        _pl_yc_row = QWidget()
        _pl_yc_h = QHBoxLayout(_pl_yc_row)
        _pl_yc_h.setContentsMargins(0, 0, 0, 0)
        _pl_yc_h.setSpacing(6)
        _pl_yc_h.addWidget(self.pl_yaxis_combo, 1)
        _pl_yc_h.addWidget(QLabel("Cmap"))
        _pl_yc_h.addWidget(cmap)
        cfg.addRow("Y-axis", _pl_yc_row)
        cfg.addRow("", self.pl_yaxis_advanced_box)
        self.pl_dat_yaxis_label_edit = QLineEdit()
        self.pl_dat_yaxis_label_edit.setPlaceholderText("Custom Y-axis label")
        self.pl_dat_yaxis_unit_edit = QLineEdit()
        self.pl_dat_yaxis_unit_edit.setPlaceholderText("Optional unit")
        dat_y_row = QWidget()
        dat_y_layout = QHBoxLayout(dat_y_row)
        dat_y_layout.setContentsMargins(0, 0, 0, 0)
        dat_y_layout.setSpacing(6)
        dat_y_layout.addWidget(self.pl_dat_yaxis_label_edit)
        dat_y_layout.addWidget(self.pl_dat_yaxis_unit_edit)
        cfg.addRow("Imported DAT Y", dat_y_row)
        self.pl_dat_yaxis_label_edit.setVisible(False)
        self.pl_dat_yaxis_unit_edit.setVisible(False)
        params_layout.addLayout(cfg)
        for s in spins.values():
            s.setFixedWidth(UI_METRICS["spin_w"])
            s.setFixedHeight(UI_METRICS["input_h"])

        self.pl_auto_v_btn = QToolButton()
        self.pl_auto_x_btn = QToolButton()
        self.pl_auto_y_btn = QToolButton()

        basic = QGroupBox("Axis Ranges")
        basic_form = QFormLayout(basic)
        basic_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        basic_form.setHorizontalSpacing(4)
        basic_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        basic_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        basic_form.addRow(
            "vmin / vmax",
            self._make_axis_range_row(spins["vmin"], spins["vmax"], fix_checks["vmin"], fix_checks["vmax"], self.pl_auto_v_btn, "Auto V"),
        )
        basic_form.addRow("Color scale", self.pl_split_scale_chk)
        basic_form.addRow(self.pl_split_scale_panel)
        basic_form.addRow(
            "xmin / xmax",
            self._make_axis_range_row(spins["xmin"], spins["xmax"], fix_checks["xmin"], fix_checks["xmax"], self.pl_auto_x_btn, "Auto X"),
        )
        basic_form.addRow(
            "ymin / ymax",
            self._make_axis_range_row(spins["ymin"], spins["ymax"], fix_checks["ymin"], fix_checks["ymax"], self.pl_auto_y_btn, "Auto Y"),
        )
        basic_form.addRow("Cursor Gate", spins["gate"])
        flags = QWidget()
        flags_h = QHBoxLayout(flags)
        flags_h.setContentsMargins(0, 0, 0, 0)
        flags_h.setSpacing(10)
        flags_h.addWidget(self.pl_log_chk)
        flags_h.addWidget(self.pl_clip_chk)
        flags_h.addStretch(1)
        basic_form.addRow("Scale / Clip", flags)
        self._set_form_label_width(basic_form, UI_METRICS["label_col_width"])

        params_layout.addWidget(self._make_expander("Manual plot ranges", basic, expanded=False))

        analysis = QGroupBox("Spectrum Analysis Controls")
        analysis_form = QFormLayout(analysis)
        analysis_form.setContentsMargins(6, 6, 6, 6)
        analysis_form.setHorizontalSpacing(6)
        analysis_form.setVerticalSpacing(4)
        self.pl_peak_find_btn = QPushButton("Find Peaks")
        self.pl_peak_show_chk = QCheckBox("Show Peaks")
        self.pl_peak_show_chk.setChecked(True)
        self.pl_peak_mode_combo = QComboBox()
        self.pl_peak_mode_combo.addItems(["Peaks", "Dips"])
        row1 = QWidget()
        row1h = QHBoxLayout(row1)
        row1h.setContentsMargins(0, 0, 0, 0)
        row1h.setSpacing(6)
        row1h.addWidget(self.pl_peak_find_btn)
        row1h.addWidget(self.pl_peak_mode_combo)
        row1h.addWidget(self.pl_peak_show_chk)
        row1h.addStretch(1)
        analysis_form.addRow("", row1)
        self.pl_peak_prom_spin = QDoubleSpinBox()
        self.pl_peak_prom_spin.setDecimals(3)
        self.pl_peak_prom_spin.setRange(0.0, 1.0)
        self.pl_peak_prom_spin.setSingleStep(0.01)
        self.pl_peak_prom_spin.setValue(0.05)
        self.pl_peak_dist_spin = QSpinBox()
        self.pl_peak_dist_spin.setRange(1, 500)
        self.pl_peak_dist_spin.setValue(5)
        self.pl_peak_max_spin = QSpinBox()
        self.pl_peak_max_spin.setRange(1, 20)
        self.pl_peak_max_spin.setValue(6)
        row2 = QWidget()
        row2h = QHBoxLayout(row2)
        row2h.setContentsMargins(0, 0, 0, 0)
        row2h.setSpacing(6)
        row2h.addWidget(QLabel("Prom"))
        row2h.addWidget(self.pl_peak_prom_spin)
        row2h.addWidget(QLabel("Dist"))
        row2h.addWidget(self.pl_peak_dist_spin)
        row2h.addWidget(QLabel("Top"))
        row2h.addWidget(self.pl_peak_max_spin)
        analysis_form.addRow("", row2)
        self.pl_fit_btn = QPushButton("Fit Multi-Lorentz")
        self.pl_fit_clear_btn = QPushButton("Clear Fit")
        self.pl_fit_show_chk = QCheckBox("Show Fit")
        self.pl_fit_show_chk.setChecked(True)
        self.pl_fit_n_spin = QSpinBox()
        self.pl_fit_n_spin.setRange(1, 8)
        self.pl_fit_n_spin.setValue(3)
        row3 = QWidget()
        row3h = QHBoxLayout(row3)
        row3h.setContentsMargins(0, 0, 0, 0)
        row3h.setSpacing(6)
        row3h.addWidget(self.pl_fit_btn)
        row3h.addWidget(self.pl_fit_clear_btn)
        row3h.addWidget(self.pl_fit_show_chk)
        row3h.addWidget(QLabel("N"))
        row3h.addWidget(self.pl_fit_n_spin)
        row3h.addStretch(1)
        analysis_form.addRow("", row3)
        self.pl_fit_status = QLabel("")
        set_fluent_property(self.pl_fit_status, "appRole", "fitStatus")
        analysis_form.addRow("", self.pl_fit_status)
        self.pl_analysis_text = QPlainTextEdit()
        self.pl_analysis_text.setReadOnly(True)
        self.pl_analysis_text.setMinimumHeight(60)
        self.pl_analysis_text.setMaximumHeight(100)
        self.pl_analysis_text.setPlaceholderText("Peak/fit results will appear here after detection.")
        analysis_form.addRow("", self.pl_analysis_text)
        layout.addWidget(self._make_expander("Parameters", params, expanded=True))
        layout.addWidget(self._make_expander("Spectrum Analysis", analysis, expanded=False))
        layout.addStretch(1)
        return tab

    def _build_drr_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        files = QGroupBox("Files")
        files_layout = QVBoxLayout(files)
        files_layout.setContentsMargins(6, 6, 6, 6)
        files_layout.setSpacing(6)

        meas_row = QWidget()
        meas_grid = QGridLayout(meas_row)
        meas_grid.setContentsMargins(0, 0, 0, 0)
        meas_grid.setHorizontalSpacing(6)
        meas_grid.setVerticalSpacing(4)
        self.drr_measurement_summary = QLabel("Measurement: 0 files")
        self.drr_measurement_summary.setWordWrap(True)
        self.drr_edit_measurements_btn = QPushButton("Select...")
        self.drr_edit_measurements_btn.setFixedHeight(30)
        self.drr_edit_measurements_btn.setMaximumWidth(92)
        self.drr_clear_measurements_btn = QPushButton("Clear")
        self.drr_clear_measurements_btn.setFixedHeight(30)
        self.drr_clear_measurements_btn.setMaximumWidth(72)
        meas_grid.addWidget(self.drr_measurement_summary, 0, 0, 1, 3)
        meas_grid.setColumnStretch(0, 1)
        meas_grid.addWidget(self.drr_edit_measurements_btn, 1, 1)
        meas_grid.addWidget(self.drr_clear_measurements_btn, 1, 2)
        files_layout.addWidget(meas_row)

        base_row = QWidget()
        self.drr_external_baseline_row = base_row
        base_grid = QGridLayout(base_row)
        base_grid.setContentsMargins(0, 0, 0, 0)
        base_grid.setHorizontalSpacing(6)
        base_grid.setVerticalSpacing(4)
        self.drr_baseline_summary = QLabel("Baselines: 0 files")
        self.drr_baseline_summary.setWordWrap(True)
        self.drr_edit_baselines_btn = QPushButton("Select...")
        self.drr_edit_baselines_btn.setFixedHeight(30)
        self.drr_edit_baselines_btn.setMaximumWidth(92)
        self.drr_baseline_autofind_btn = QPushButton("Clear")
        self.drr_baseline_autofind_btn.setToolTip(
            "Clear the selected external background files."
        )
        self.drr_baseline_autofind_btn.setFixedHeight(30)
        self.drr_baseline_autofind_btn.setMaximumWidth(104)
        base_grid.addWidget(self.drr_baseline_summary, 0, 0, 1, 3)
        base_grid.setColumnStretch(0, 1)
        base_grid.addWidget(self.drr_edit_baselines_btn, 1, 1)
        base_grid.addWidget(self.drr_baseline_autofind_btn, 1, 2)
        files_layout.addWidget(base_row)

        self.drr_baseline_combine_combo = QComboBox()
        self.drr_baseline_combine_combo.addItems(
            [
                "Last frame from each file, then average",
                "First frame from each file, then average",
                "Average all frames in each file, then average files",
            ]
        )
        self.drr_baseline_combine_combo.setMaximumWidth(320)
        self._style_combo_popup(self.drr_baseline_combine_combo)
        files_layout.addWidget(self.drr_baseline_combine_combo)
        self.drr_pin_baseline_chk = QCheckBox("Pin background when measurement changes")
        self.drr_pin_baseline_chk.setToolTip(
            "Keep this manually selected background for another measurement group. "
            "The wavelength center and spectral grid are still validated."
        )
        files_layout.addWidget(self.drr_pin_baseline_chk)
        self.drr_external_baseline_row.setVisible(False)
        self.drr_baseline_combine_combo.setVisible(False)
        self.drr_pin_baseline_chk.setVisible(False)
        layout.addWidget(self._make_expander("Data", files, expanded=True))

        params = QGroupBox("Plot Options")
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(6, 6, 6, 4)
        params_layout.setSpacing(6)

        self.drr_baseline_combo = QComboBox()
        self.drr_baseline_combo.addItems(["Self (last frame)", "Self (first frame)", "External"])
        self.drr_baseline_combo.setToolTip("Background strategy for DRR")
        self._style_combo_popup(self.drr_baseline_combo)
        self.drr_derivative_combo = QComboBox()
        self.drr_derivative_combo.addItems(["None", "dE", "d2E"])
        self.drr_derivative_combo.setToolTip("Apply derivative transform to DRR")
        self._style_combo_popup(self.drr_derivative_combo)
        _grid, spins, log_chk, clip_chk, cmap, fix_checks = self._build_common_range_grid("drr", "RdBu_r")

        for s in spins.values():
            s.setFixedWidth(UI_METRICS["spin_w"])
            s.setFixedHeight(UI_METRICS["input_h"])
        self.drr_baseline_combo.setMinimumWidth(120)
        self.drr_baseline_combo.setMaximumWidth(180)
        self.drr_baseline_combo.setFixedHeight(UI_METRICS["input_h"])
        self.drr_baseline_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.drr_yaxis_controls = self._build_y_axis_controls("drr")
        cmap.setMinimumWidth(90)
        cmap.setMaximumWidth(115)
        cmap.setFixedHeight(UI_METRICS["input_h"])
        cmap.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.drr_derivative_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.drr_derivative_combo.setMinimumContentsLength(3)
        self.drr_derivative_combo.setFixedWidth(UI_METRICS["deriv_combo_w"])
        self.drr_derivative_combo.setFixedHeight(UI_METRICS["input_h"])
        self.drr_sg_window_spin = QSpinBox()
        self.drr_sg_window_spin.setRange(5, 401)
        self.drr_sg_window_spin.setSingleStep(2)
        self.drr_sg_window_spin.setValue(20)
        self.drr_sg_window_spin.setToolTip("Savitzky-Golay window length (odd).")
        self.drr_sg_window_spin.setFixedWidth(UI_METRICS["spin_w"])
        self.drr_sg_window_spin.setFixedHeight(UI_METRICS["input_h"])
        self.drr_sg_poly_spin = QSpinBox()
        self.drr_sg_poly_spin.setRange(1, 6)
        self.drr_sg_poly_spin.setValue(2)
        self.drr_sg_poly_spin.setToolTip("Savitzky-Golay polynomial order.")
        self.drr_sg_poly_spin.setFixedWidth(UI_METRICS["spin_w"])
        self.drr_sg_poly_spin.setFixedHeight(UI_METRICS["input_h"])
        self.drr_sg_window_spin.setVisible(False)
        self.drr_sg_poly_spin.setVisible(False)

        deriv_row = QWidget()
        deriv_h = QHBoxLayout(deriv_row)
        deriv_h.setContentsMargins(0, 0, 0, 0)
        deriv_h.setSpacing(4)
        deriv_h.addWidget(self.drr_derivative_combo)
        self.drr_sg_window_spin.setPrefix("W ")
        deriv_h.addWidget(self.drr_sg_window_spin)
        self.drr_sg_poly_spin.setPrefix("O ")
        deriv_h.addWidget(self.drr_sg_poly_spin)
        deriv_h.addStretch(1)

        self.drr_auto_v_btn = QToolButton()
        self.drr_auto_x_btn = QToolButton()
        self.drr_auto_y_btn = QToolButton()
        self.drr_center_zero_chk = QCheckBox("Center Zero")
        self.drr_center_zero_chk.setToolTip("When enabled, DRR colormap is centered at zero.")
        self.drr_center_zero_chk.setChecked(False)

        # Config rows outside Axis Ranges — mirrors PL tab structure
        cfg = QFormLayout()
        cfg.setHorizontalSpacing(6)
        cfg.setVerticalSpacing(UI_METRICS["row_spacing"])
        cfg.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        baseline_cmap_row = QWidget()
        baseline_cmap_h = QHBoxLayout(baseline_cmap_row)
        baseline_cmap_h.setContentsMargins(0, 0, 0, 0)
        baseline_cmap_h.setSpacing(6)
        baseline_cmap_h.addWidget(self.drr_baseline_combo, 1)
        baseline_cmap_h.addWidget(QLabel("Cmap"))
        baseline_cmap_h.addWidget(cmap)
        _drr_yc_row = QWidget()
        _drr_yc_h = QHBoxLayout(_drr_yc_row)
        _drr_yc_h.setContentsMargins(0, 0, 0, 0)
        _drr_yc_h.setSpacing(6)
        _drr_yc_h.addWidget(self.drr_yaxis_combo, 1)
        cfg.addRow("DRR Baseline", baseline_cmap_row)
        cfg.addRow("Y-axis", _drr_yc_row)
        cfg.addRow("", self.drr_yaxis_advanced_box)
        cfg.addRow("Derivative / SG", deriv_row)
        self._set_form_label_width(cfg, UI_METRICS["label_col_width"])
        params_layout.addLayout(cfg)

        basic = QGroupBox("Axis Ranges")
        basic_form = QFormLayout(basic)
        basic_form.setContentsMargins(
            4,
            UI_METRICS["group_margin"],
            4,
            UI_METRICS["group_margin"],
        )
        basic_form.setHorizontalSpacing(4)
        basic_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        basic_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        basic_form.addRow(
            "vmin / vmax",
            self._make_axis_range_row(spins["vmin"], spins["vmax"], fix_checks["vmin"], fix_checks["vmax"], self.drr_auto_v_btn, "Auto V"),
        )
        basic_form.addRow("Color scale", self.drr_split_scale_chk)
        basic_form.addRow(self.drr_split_scale_panel)
        basic_form.addRow(
            "xmin / xmax",
            self._make_axis_range_row(spins["xmin"], spins["xmax"], fix_checks["xmin"], fix_checks["xmax"], self.drr_auto_x_btn, "Auto X"),
        )
        basic_form.addRow(
            "ymin / ymax",
            self._make_axis_range_row(spins["ymin"], spins["ymax"], fix_checks["ymin"], fix_checks["ymax"], self.drr_auto_y_btn, "Auto Y"),
        )
        basic_form.addRow("Cursor Gate", spins["gate"])
        flags = QWidget()
        flags_h = QHBoxLayout(flags)
        flags_h.setContentsMargins(0, 0, 0, 0)
        flags_h.setSpacing(10)
        flags_h.addWidget(log_chk)
        flags_h.addWidget(clip_chk)
        flags_h.addWidget(self.drr_center_zero_chk)
        flags_h.addStretch(1)
        basic_form.addRow("Scale / Clip", flags)
        self._set_form_label_width(basic_form, UI_METRICS["label_col_width"])

        analysis_box = QGroupBox("")
        analysis_form = QFormLayout(analysis_box)
        analysis_form.setContentsMargins(6, 6, 6, 6)
        analysis_form.setHorizontalSpacing(6)
        analysis_form.setVerticalSpacing(4)

        self.drr_peak_show_chk = QCheckBox("Show Peaks")
        self.drr_peak_show_chk.setChecked(True)
        self.drr_peak_find_btn = QPushButton("Find Peaks")
        self.drr_peak_mode_combo = QComboBox()
        self.drr_peak_mode_combo.addItems(["Peaks", "Dips"])
        self.drr_peak_find_btn.setToolTip("Find peaks in current bottom spectrum and overlay on both plots.")
        self.drr_peak_prom_spin = QDoubleSpinBox()
        self.drr_peak_prom_spin.setDecimals(3)
        self.drr_peak_prom_spin.setRange(0.0, 1.0)
        self.drr_peak_prom_spin.setSingleStep(0.01)
        self.drr_peak_prom_spin.setValue(0.05)
        self.drr_peak_prom_spin.setToolTip("Prominence as fraction of visible Y-range.")
        self.drr_peak_dist_spin = QSpinBox()
        self.drr_peak_dist_spin.setRange(1, 500)
        self.drr_peak_dist_spin.setValue(5)
        self.drr_peak_dist_spin.setToolTip("Minimum peak spacing in points.")
        self.drr_peak_max_spin = QSpinBox()
        self.drr_peak_max_spin.setRange(1, 20)
        self.drr_peak_max_spin.setValue(6)
        self.drr_peak_max_spin.setToolTip("Maximum number of strongest peaks to keep.")

        self.drr_fit_show_chk = QCheckBox("Show Fit")
        self.drr_fit_show_chk.setChecked(True)
        self.drr_fit_btn = QPushButton("Fit Multi-Lorentz")
        self.drr_fit_btn.setToolTip("Fit multiple Lorentz peaks to current bottom spectrum.")
        self.drr_fit_clear_btn = QPushButton("Clear Fit")
        self.drr_fit_n_spin = QSpinBox()
        self.drr_fit_n_spin.setRange(1, 8)
        self.drr_fit_n_spin.setValue(3)
        self.drr_fit_n_spin.setToolTip("Number of Lorentz peaks in fit.")
        self.drr_fit_status = QLabel("")

        drr_row1 = QWidget()
        drr_row1h = QHBoxLayout(drr_row1)
        drr_row1h.setContentsMargins(0, 0, 0, 0)
        drr_row1h.setSpacing(6)
        drr_row1h.addWidget(self.drr_peak_find_btn)
        drr_row1h.addWidget(self.drr_peak_mode_combo)
        drr_row1h.addWidget(self.drr_peak_show_chk)
        drr_row1h.addStretch(1)
        analysis_form.addRow("", drr_row1)

        drr_row2 = QWidget()
        drr_row2h = QHBoxLayout(drr_row2)
        drr_row2h.setContentsMargins(0, 0, 0, 0)
        drr_row2h.setSpacing(6)
        drr_row2h.addWidget(QLabel("Prom"))
        drr_row2h.addWidget(self.drr_peak_prom_spin)
        drr_row2h.addWidget(QLabel("Dist"))
        drr_row2h.addWidget(self.drr_peak_dist_spin)
        drr_row2h.addWidget(QLabel("Top"))
        drr_row2h.addWidget(self.drr_peak_max_spin)
        analysis_form.addRow("", drr_row2)

        drr_row3 = QWidget()
        drr_row3h = QHBoxLayout(drr_row3)
        drr_row3h.setContentsMargins(0, 0, 0, 0)
        drr_row3h.setSpacing(6)
        drr_row3h.addWidget(self.drr_fit_btn)
        drr_row3h.addWidget(self.drr_fit_clear_btn)
        drr_row3h.addWidget(self.drr_fit_show_chk)
        drr_row3h.addWidget(QLabel("N"))
        drr_row3h.addWidget(self.drr_fit_n_spin)
        drr_row3h.addStretch(1)
        analysis_form.addRow("", drr_row3)
        analysis_form.addRow("", self.drr_fit_status)
        self.drr_analysis_text = QPlainTextEdit()
        self.drr_analysis_text.setReadOnly(True)
        self.drr_analysis_text.setMinimumHeight(60)
        self.drr_analysis_text.setMaximumHeight(100)
        self.drr_analysis_text.setPlaceholderText("Peak/fit results will appear here after detection.")
        analysis_form.addRow("", self.drr_analysis_text)
        set_fluent_property(self.drr_fit_status, "appRole", "fitStatus")
        params_layout.addWidget(self._make_expander("Manual plot ranges", basic, expanded=False))
        layout.addWidget(self._make_expander("Parameters", params, expanded=True))
        layout.addWidget(self._make_expander("Spectrum Analysis", analysis_box, expanded=False))
        layout.addStretch(1)
        return tab

    def _build_compare_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        params = QGroupBox("Plot Options")
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(6, 6, 6, 4)
        params_layout.setSpacing(4)

        assignment = QGroupBox("Assignment")
        assignment_layout = QVBoxLayout(assignment)
        assignment_layout.setContentsMargins(6, 8, 6, 6)
        assignment_layout.setSpacing(6)
        assignment_form = QFormLayout()
        assignment_form.setContentsMargins(0, 0, 0, 0)
        assignment_form.setHorizontalSpacing(6)
        assignment_form.setVerticalSpacing(4)
        def _angle_spin(default: float = 0.0) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(-360.0, 360.0)
            spin.setValue(default)
            spin.setSuffix(" deg")
            spin.setToolTip("Approximate reference angle; matching uses the tolerance below.")
            return spin

        self.cmp_in_k_angle_spin = _angle_spin()
        self.cmp_in_kp_angle_spin = _angle_spin(45.0)
        self.cmp_out_k_angle_spin = _angle_spin()
        self.cmp_out_kp_angle_spin = _angle_spin(45.0)
        self.cmp_angle_tolerance_spin = QDoubleSpinBox()
        self.cmp_angle_tolerance_spin.setDecimals(2)
        self.cmp_angle_tolerance_spin.setRange(0.1, 180.0)
        self.cmp_angle_tolerance_spin.setValue(15.0)
        self.cmp_angle_tolerance_spin.setSuffix(" deg")
        self.cmp_angle_tolerance_spin.setToolTip(
            "Maximum distance from the nearest K or Kp reference. Equal-distance matches are rejected."
        )
        self.cmp_infer_angles_btn = QPushButton("Infer Angles")
        self.cmp_infer_angles_btn.setToolTip(
            "Suggest editable K/Kp references when exactly two filename-angle clusters are detected."
        )
        self.cmp_auto_assign_btn = QPushButton("Auto Detect")
        angle_box = QWidget()
        angle_grid = QGridLayout(angle_box)
        angle_grid.setContentsMargins(0, 0, 0, 0)
        angle_grid.setHorizontalSpacing(6)
        angle_grid.setVerticalSpacing(4)
        angle_grid.addWidget(QLabel("In K"), 0, 0)
        angle_grid.addWidget(self.cmp_in_k_angle_spin, 0, 1)
        angle_grid.addWidget(QLabel("In Kp"), 0, 2)
        angle_grid.addWidget(self.cmp_in_kp_angle_spin, 0, 3)
        angle_grid.addWidget(QLabel("Out K"), 1, 0)
        angle_grid.addWidget(self.cmp_out_k_angle_spin, 1, 1)
        angle_grid.addWidget(QLabel("Out Kp"), 1, 2)
        angle_grid.addWidget(self.cmp_out_kp_angle_spin, 1, 3)
        angle_grid.addWidget(QLabel("Tolerance"), 2, 0)
        angle_grid.addWidget(self.cmp_angle_tolerance_spin, 2, 1)
        angle_grid.addWidget(self.cmp_infer_angles_btn, 2, 2)
        angle_grid.addWidget(self.cmp_auto_assign_btn, 2, 3)
        angle_grid.setColumnStretch(4, 1)
        assignment_form.addRow("Angle Rules", angle_box)
        assignment_layout.addLayout(assignment_form)
        self.cmp_channel_combos: dict[str, QComboBox] = {}
        channels_box = QWidget()
        channels_grid = QGridLayout(channels_box)
        channels_grid.setContentsMargins(0, 0, 0, 0)
        channels_grid.setHorizontalSpacing(8)
        channels_grid.setVerticalSpacing(4)
        for idx, key in enumerate(COMPARE_PANEL_ORDER):
            combo = QComboBox()
            combo.setEditable(False)
            self._style_combo_popup(combo)
            self.cmp_channel_combos[key] = combo
            row = idx // 2
            col = idx % 2
            label = QLabel(key)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            channels_grid.addWidget(label, row, col * 2)
            channels_grid.addWidget(combo, row, col * 2 + 1)
        assignment_layout.addWidget(channels_box)
        self.cmp_assignment_summary = QPlainTextEdit()
        self.cmp_assignment_summary.setReadOnly(True)
        self.cmp_assignment_summary.setMaximumHeight(88)
        summary_form = QFormLayout()
        summary_form.setContentsMargins(0, 0, 0, 0)
        summary_form.setHorizontalSpacing(6)
        summary_form.setVerticalSpacing(4)
        summary_form.addRow("Summary", self.cmp_assignment_summary)
        assignment_layout.addLayout(summary_form)
        layout.addWidget(self._make_expander("Assignment", assignment, expanded=True))

        display = QGroupBox("Display Preset")
        display_form = QFormLayout(display)
        self.cmp_display_preset_combo = QComboBox()
        self.cmp_display_preset_combo.addItems(["KK + KKp", "KpK + KpKp", "All four", "Custom"])
        self._style_combo_popup(self.cmp_display_preset_combo)
        display_form.addRow("Preset", self.cmp_display_preset_combo)
        checks_row = QWidget()
        checks_h = QHBoxLayout(checks_row)
        checks_h.setContentsMargins(0, 0, 0, 0)
        checks_h.setSpacing(10)
        self.cmp_show_checks: dict[str, QCheckBox] = {}
        for key in COMPARE_PANEL_ORDER:
            chk = QCheckBox(key)
            self.cmp_show_checks[key] = chk
            checks_h.addWidget(chk)
        checks_h.addStretch(1)
        display_form.addRow("Channels", checks_row)
        params_layout.addWidget(self._make_expander("Display", display, expanded=False))

        vp_box = QGroupBox("Valley Polarization")
        vp_form = QFormLayout(vp_box)
        vp_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        vp_form.setHorizontalSpacing(6)
        vp_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        self.cmp_vp_background_spin = QDoubleSpinBox()
        self.cmp_vp_background_spin.setDecimals(6)
        self.cmp_vp_background_spin.setRange(-1.0e12, 1.0e12)
        self.cmp_vp_background_spin.setSingleStep(100.0)
        self.cmp_vp_background_spin.setFixedWidth(UI_METRICS["spin_w"] + 18)
        self.cmp_vp_auto_background_chk = QCheckBox("Auto")
        self.cmp_vp_auto_background_chk.setChecked(True)
        self.cmp_vp_auto_background_chk.setToolTip("Estimate one constant background from KK and KKp.")
        bkg_row = QWidget()
        bkg_h = QHBoxLayout(bkg_row)
        bkg_h.setContentsMargins(0, 0, 0, 0)
        bkg_h.setSpacing(8)
        bkg_h.addWidget(self.cmp_vp_background_spin)
        bkg_h.addWidget(self.cmp_vp_auto_background_chk)
        bkg_h.addStretch(1)
        self.cmp_vp_filename_preview = QLineEdit()
        self.cmp_vp_filename_preview.setReadOnly(True)
        self.cmp_vp_filename_preview.setMinimumWidth(200)
        self.cmp_kk_title_preview = QLineEdit()
        self.cmp_kk_title_preview.setReadOnly(True)
        self.cmp_kkp_title_preview = QLineEdit()
        self.cmp_kkp_title_preview.setReadOnly(True)
        self.cmp_vp_title_preview = QLineEdit()
        self.cmp_vp_title_preview.setReadOnly(True)
        vp_form.addRow("Background", bkg_row)
        vp_form.addRow("VP filename", self.cmp_vp_filename_preview)
        vp_form.addRow("KK title", self.cmp_kk_title_preview)
        vp_form.addRow("KKp title", self.cmp_kkp_title_preview)
        vp_form.addRow("VP title", self.cmp_vp_title_preview)
        params_layout.addWidget(self._make_expander("VP", vp_box, expanded=False))

        cfg = QFormLayout()
        cfg.setHorizontalSpacing(6)
        cfg.setVerticalSpacing(4)
        _grid, spins, _, _, cmap, fix_checks = self._build_common_range_grid("cmp", "turbo")
        self.cmp_yaxis_controls = self._build_y_axis_controls("cmp")
        _cmp_yc_row = QWidget()
        _cmp_yc_h = QHBoxLayout(_cmp_yc_row)
        _cmp_yc_h.setContentsMargins(0, 0, 0, 0)
        _cmp_yc_h.setSpacing(6)
        _cmp_yc_h.addWidget(self.cmp_yaxis_combo, 1)
        _cmp_yc_h.addWidget(QLabel("Cmap"))
        _cmp_yc_h.addWidget(cmap)
        cfg.addRow("Y-axis / Cmap", _cmp_yc_row)
        cfg.addRow("", self.cmp_yaxis_advanced_box)
        params_layout.addLayout(cfg)
        for s in spins.values():
            s.setFixedWidth(UI_METRICS["spin_w"])
            s.setFixedHeight(UI_METRICS["input_h"])

        self.cmp_auto_v_btn = QToolButton()
        self.cmp_auto_x_btn = QToolButton()
        self.cmp_auto_y_btn = QToolButton()
        basic = QGroupBox("Axis Ranges")
        basic_form = QFormLayout(basic)
        basic_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        basic_form.setHorizontalSpacing(4)
        basic_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        basic_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        basic_form.addRow(
            "vmin / vmax",
            self._make_axis_range_row(spins["vmin"], spins["vmax"], fix_checks["vmin"], fix_checks["vmax"], self.cmp_auto_v_btn, "Auto V"),
        )
        basic_form.addRow("Color scale", self.cmp_split_scale_chk)
        basic_form.addRow(self.cmp_split_scale_panel)
        basic_form.addRow(
            "xmin / xmax",
            self._make_axis_range_row(spins["xmin"], spins["xmax"], fix_checks["xmin"], fix_checks["xmax"], self.cmp_auto_x_btn, "Auto X"),
        )
        basic_form.addRow(
            "ymin / ymax",
            self._make_axis_range_row(spins["ymin"], spins["ymax"], fix_checks["ymin"], fix_checks["ymax"], self.cmp_auto_y_btn, "Auto Y"),
        )
        basic_form.addRow("Cursor Gate", spins["gate"])
        flags = QWidget()
        flags_h = QHBoxLayout(flags)
        flags_h.setContentsMargins(0, 0, 0, 0)
        flags_h.setSpacing(10)
        flags_h.addWidget(self.cmp_log_chk)
        flags_h.addWidget(self.cmp_clip_chk)
        flags_h.addStretch(1)
        basic_form.addRow("Scale / Clip", flags)
        self._set_form_label_width(basic_form, UI_METRICS["label_col_width"])
        params_layout.addWidget(self._make_expander("Manual plot ranges", basic, expanded=False))
        layout.addWidget(self._make_expander("Parameters", params, expanded=True))
        layout.addStretch(1)
        return tab

    def _build_mcd_tab(self) -> QWidget:
        tab = QWidget()
        tab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        source = QGroupBox("B-sweep source")
        source_form = QFormLayout(source)
        source_form.setContentsMargins(6, 4, 6, 4)
        source_form.setHorizontalSpacing(6)
        source_form.setVerticalSpacing(3)
        # Keep the selection in a list model so the existing load/reprocess
        # path remains unchanged, but expose it through a focused file chooser.
        self.mcd_files = QListWidget(source)
        self.mcd_files.setSelectionMode(QAbstractItemView.SingleSelection)
        self.mcd_files.hide()
        source_row = QWidget()
        source_grid = QGridLayout(source_row)
        source_grid.setContentsMargins(0, 0, 0, 0)
        source_grid.setHorizontalSpacing(6)
        source_grid.setVerticalSpacing(4)
        self.mcd_selection_summary = StatusBadge("No MCD CSV selected.", app_role=None)
        self.mcd_selection_summary.setMinimumWidth(0)
        # Keep the action row anchored when the status changes between NEW,
        # PROCESSED, and long-filename states.  The status text is allowed to
        # wrap inside this reserved area; it must not change the panel height.
        self.mcd_selection_summary.setMinimumHeight(56)
        self.mcd_selection_summary.setMaximumHeight(56)
        self.mcd_selection_summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.mcd_select_source_btn = QPushButton("Select...")
        self.mcd_select_source_btn.setFixedHeight(30)
        self.mcd_select_source_btn.setMaximumWidth(92)
        self.mcd_clear_source_btn = QPushButton("Clear")
        self.mcd_clear_source_btn.setFixedHeight(30)
        self.mcd_clear_source_btn.setMaximumWidth(72)
        source_grid.addWidget(self.mcd_selection_summary, 0, 0, 1, 3)
        source_grid.setColumnStretch(0, 1)
        source_grid.setRowMinimumHeight(0, 56)
        source_grid.setRowMinimumHeight(1, 30)
        source_grid.addWidget(self.mcd_select_source_btn, 1, 1)
        source_grid.addWidget(self.mcd_clear_source_btn, 1, 2)
        source_row.setFixedHeight(90)
        source_form.addRow(source_row)
        self.mcd_source_summary = QLabel("Select a B-sweep CSV.")
        self.mcd_source_summary.setWordWrap(True)
        self.mcd_source_summary.setMinimumWidth(0)
        # Keep the following correction controls from shifting when the
        # detected-angle/reference diagnostic wraps to two lines.
        self.mcd_source_summary.setFixedHeight(36)
        self.mcd_source_summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        source_form.addRow("Format", self.mcd_source_summary)
        layout.addWidget(self._make_expander("Source", source, expanded=True))

        correction = QGroupBox("Angle background correction")
        correction_layout = QVBoxLayout(correction)
        correction_layout.setContentsMargins(6, 4, 6, 4)
        correction_layout.setSpacing(3)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(3)
        correction_layout.addLayout(form)
        self.mcd_auto_angles_chk = QCheckBox("Auto-assign sigma+ / sigma-")
        self.mcd_auto_angles_chk.setChecked(True)
        self.mcd_auto_angles_chk.setToolTip(
            "Detect the available angles in the selected CSV. When enabled, the largest angle is assigned to sigma+ and the smallest to sigma-. "
            "Turn this off to choose the sigma+ and sigma- assignments yourself."
        )
        self.mcd_sigma_plus_combo = QComboBox(); self.mcd_sigma_minus_combo = QComboBox()
        for combo in (self.mcd_sigma_plus_combo, self.mcd_sigma_minus_combo):
            combo.addItem("-- Select source CSV first --", None)
            self._style_combo_popup(combo)
        self.mcd_reference_mode_combo = QComboBox()
        self.mcd_reference_mode_combo.addItems(["Nearest paired B (recommended)", "Median near-zero window"])
        self._style_combo_popup(self.mcd_reference_mode_combo)
        self.mcd_zero_spin = QDoubleSpinBox(); self.mcd_zero_spin.setRange(0.0, 10.0); self.mcd_zero_spin.setDecimals(4); self.mcd_zero_spin.setValue(0.02); self.mcd_zero_spin.setSuffix(" T")
        self.mcd_zero_spin.setEnabled(False)
        self.mcd_gap_spin = QSpinBox(); self.mcd_gap_spin.setRange(1, 50); self.mcd_gap_spin.setValue(3)
        self.mcd_delta_b_spin = QDoubleSpinBox(); self.mcd_delta_b_spin.setRange(0.0001, 10.0); self.mcd_delta_b_spin.setDecimals(4); self.mcd_delta_b_spin.setValue(0.1); self.mcd_delta_b_spin.setSuffix(" T")
        self.mcd_pair_alignment_combo = QComboBox(); self.mcd_pair_alignment_combo.addItems(["Direct measured pair", "Interpolate both angles to Bpair"])
        self.mcd_bin_spin = QSpinBox(); self.mcd_bin_spin.setRange(0, 6); self.mcd_bin_spin.setValue(3)
        self.mcd_gain_combo = QComboBox(); self.mcd_gain_combo.addItems(["Per wavelength", "Smoothed per wavelength", "Scalar (diagnostic only)"])
        self.mcd_correction_mode_combo = QComboBox(); self.mcd_correction_mode_combo.addItems([
            "Global reference gain (current)", "Global gain + per-pair scale", "Global gain + per-pair scale/offset",
            "Global gain + per-pair spectral baseline",
        ])
        self.mcd_correction_mode_combo.setCurrentIndex(3)
        self.mcd_spectral_order_combo = QComboBox(); self.mcd_spectral_order_combo.addItems([
            "Linear", "Quadratic (default)",
        ])
        self.mcd_spectral_order_combo.setCurrentIndex(1)
        self.mcd_spectral_order_combo.setEnabled(True)
        self.mcd_background_ranges_edit = QLineEdit()
        self.mcd_background_ranges_edit.setPlaceholderText("Auto outer 15%, or e.g. 1.50-1.58, 1.73-1.79")
        self.mcd_suggest_background_btn = QPushButton("Select protected regions")
        self.mcd_suggest_background_btn.setToolTip(
            "Draw persistent feature-protection windows on the full-sweep reflection plot. "
            "Every sufficiently wide unprotected interval updates automatically as a background band."
        )
        self.mcd_background_preview = QLabel("Auto outer 15% ranges are shown after loading an MCD sweep.")
        self.mcd_background_preview.setWordWrap(True)
        self.mcd_background_preview.setMinimumWidth(0)
        self.mcd_background_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.mcd_apply_correction_btn = QPushButton("Recalculate now")
        self.mcd_apply_correction_btn.setToolTip(
            "MCD processing updates automatically after settings settle. Click to recalculate immediately or retry after an error."
        )
        self.mcd_dark_pos_combo = QComboBox(); self.mcd_dark_neg_combo = QComboBox()
        for combo in (self.mcd_dark_pos_combo, self.mcd_dark_neg_combo):
            combo.addItem("-- No dark / offset file --", "")
            self._style_combo_popup(combo)
        self.mcd_sigma_plus_combo.setToolTip(
            "Measured waveplate/analyser angle to interpret as sigma+. Changing the sigma+ and sigma- assignments reverses the MCD sign."
        )
        self.mcd_sigma_minus_combo.setToolTip(
            "Measured waveplate/analyser angle to interpret as sigma-. It must be different from the sigma+ choice."
        )
        self.mcd_zero_spin.setToolTip(
            "Only used with Median near-zero window. Paired spectra with |B_pair| at or below this value "
            "are combined into each angle's reference spectrum."
        )
        self.mcd_reference_mode_combo.setToolTip(
            "Nearest paired B uses the one valid sigma+/sigma- pair with the smallest |B_pair|. "
            "This is the normal choice when an exact B = 0 spectrum was not acquired. "
            "Median near-zero window combines all pairs inside the reference window; use it only when several "
            "near-zero pairs are available and their physical MCD is negligible."
        )
        self.mcd_gap_spin.setToolTip(
            "Maximum separation, in CSV rows, allowed between opposite-angle spectra in one pair. "
            "A value of 3 accepts an opposite-angle partner within three acquired frames."
        )
        self.mcd_delta_b_spin.setToolTip(
            "Maximum allowed difference between the two raw B-field values in a pair. "
            "The paired B shown in the app is their average. Reduce this to reject poorly matched field pairs."
        )
        self.mcd_pair_alignment_combo.setToolTip(
            "Direct measured pair uses the two acquired spectra as-is. Interpolate both angle channels to Bpair using neighbours from the same sweep branch, "
            "which reduces artifacts when the two angles were measured at different B. It never interpolates across a detected field reversal."
        )
        self.mcd_bin_spin.setToolTip(
            "Round each paired B field to this many decimal places before averaging repeated points into the colormap. "
            "3 means bins spaced by 0.001 T; use fewer decimals only when the field readings are noisy."
        )
        self.mcd_gain_combo.setToolTip(
            "Angle-throughput correction derived from the two near-zero-field reference spectra. "
            "Per wavelength is the normal choice. Smoothed per wavelength suppresses noisy gain ripples. "
            "Scalar applies one number to the whole spectrum and is intended only as a diagnostic."
        )
        self.mcd_correction_mode_combo.setToolTip(
            "Global reference gain applies one reference-derived wavelength correction to every pair. "
            "Per-pair scale additionally corrects intensity drift using only the background energy ranges. "
            "Scale/offset also removes an additive offset. Spectral baseline robustly fits a smooth energy-dependent "
            "sigma+/sigma- ratio in the selected background regions and is the default correction. Review protected regions because an overly broad fit can remove real MCD structure."
        )
        self.mcd_spectral_order_combo.setToolTip(
            "Polynomial order for the per-pair spectral baseline. Quadratic is the default and corrects a broad curved mismatch. "
            "Linear remains available when only a wavelength-dependent tilt is justified; orders above two are intentionally not offered to avoid unstable overfitting."
        )
        self.mcd_background_ranges_edit.setToolTip(
            "Energy intervals used to fit per-pair scale, scale/offset, or the spectral baseline. Separate intervals with commas, for example "
            "1.50-1.58, 1.73-1.79. Leave blank to use both spectrum ends: the lowest 15% and highest 15% of the measured energy range (30% total). "
            "Exclude exciton peaks and the MCD feature of interest. Spectral correction requires separated regions spanning at least 25% of the energy range. "
            "Use Select protected regions to draw resonances that must be excluded; all unprotected background bands recalculate automatically."
        )
        self.mcd_dark_pos_combo.setToolTip(
            "Optional CSV containing the additive dark/stray-light spectrum measured with the positive angle. "
            "It is subtracted before reference normalization and must have the same wavelength columns."
        )
        self.mcd_dark_neg_combo.setToolTip(
            "Optional CSV containing the additive dark/stray-light spectrum measured with the negative angle. "
            "It is subtracted before reference normalization and must have the same wavelength columns."
        )
        mcd_compact_combos = (
            self.mcd_sigma_plus_combo, self.mcd_sigma_minus_combo, self.mcd_reference_mode_combo,
            self.mcd_pair_alignment_combo, self.mcd_gain_combo, self.mcd_correction_mode_combo, self.mcd_spectral_order_combo,
            self.mcd_dark_pos_combo, self.mcd_dark_neg_combo,
        )
        for combo in mcd_compact_combos:
            # Keep the selected long filename/mode readable by tooltip while
            # allowing the closed combo box to elide in the narrow sidebar.
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        for widget in (*mcd_compact_combos, self.mcd_zero_spin, self.mcd_gap_spin, self.mcd_delta_b_spin,
                       self.mcd_bin_spin, self.mcd_background_ranges_edit):
            widget.setFixedHeight(UI_METRICS["input_h"])
        form.addRow("Angles", self.mcd_auto_angles_chk)
        form.addRow("Sigma+ angle", self.mcd_sigma_plus_combo)
        form.addRow("Sigma- angle", self.mcd_sigma_minus_combo)
        form.addRow("Reference method", self.mcd_reference_mode_combo)
        form.addRow("Reference window", self.mcd_zero_spin)
        form.addRow("Sequence gap", self.mcd_gap_spin)
        form.addRow("Pair dB", self.mcd_delta_b_spin)
        form.addRow("Gain", self.mcd_gain_combo)
        form.addRow("Drift correction", self.mcd_correction_mode_combo)
        form.addRow("", self.mcd_apply_correction_btn)
        advanced_correction = QGroupBox("Advanced correction")
        advanced_form = QFormLayout(advanced_correction)
        advanced_form.setContentsMargins(4, 3, 4, 3)
        advanced_form.setHorizontalSpacing(6)
        advanced_form.setVerticalSpacing(3)
        advanced_form.addRow("Pair B alignment", self.mcd_pair_alignment_combo)
        advanced_form.addRow("B bin decimals", self.mcd_bin_spin)
        advanced_form.addRow("Fit bg E (eV)", self.mcd_background_ranges_edit)
        advanced_form.addRow("", self.mcd_suggest_background_btn)
        advanced_form.addRow("Background", self.mcd_background_preview)
        advanced_form.addRow("Spectral fit", self.mcd_spectral_order_combo)
        advanced_form.addRow("Dark sigma+", self.mcd_dark_pos_combo)
        advanced_form.addRow("Dark sigma-", self.mcd_dark_neg_combo)
        correction_layout.addWidget(self._make_expander("Advanced", advanced_correction, expanded=False))
        layout.addWidget(self._make_expander("Correction", correction, expanded=False))

        diagnostics = QGroupBox("Pair diagnostics")
        diagnostics_layout = QVBoxLayout(diagnostics)
        self.mcd_diagnostics_text = QPlainTextEdit()
        self.mcd_diagnostics_text.setReadOnly(True)
        self.mcd_diagnostics_text.setMinimumHeight(125)
        self.mcd_diagnostics_text.setToolTip(
            "One row per sigma+/sigma- pair. Large |dB|, a large relative RMS residual, or a rapidly changing fitted scale "
            "indicates a pair that may not be corrected reliably. Spectral mode also reports log-gain slope/curvature, "
            "the applied gain range, and background RMS before/after fitting. The full table is exported with MCD results."
        )
        diagnostics_layout.addWidget(self.mcd_diagnostics_text)
        self.mcd_diagnostics_expander = self._make_expander("Diagnostics", diagnostics, expanded=False)
        layout.addWidget(self.mcd_diagnostics_expander)

        display = QGroupBox("Display and MCD(B)")
        display_form = QFormLayout(display)
        display_form.setContentsMargins(6, 4, 6, 4)
        display_form.setHorizontalSpacing(6)
        display_form.setVerticalSpacing(3)
        self.mcd_map_combo = QComboBox(); self.mcd_map_combo.addItem("Combo")
        _grid, self.mcd_spins, _a, _b, self.mcd_cmap, _mcd_unused_fix_checks = self._build_common_range_grid("mcd", "RdBu_r")
        # MCD does not expose fixed-range controls. Keeping only the visible
        # widgets avoids retaining unparented QCheckBoxes after construction.
        self.mcd_fix_checks: Dict[str, QCheckBox] = {}
        unused_mcd_cursor = self.mcd_spins.pop("gate")
        unused_mcd_cursor.deleteLater()
        for spin in self.mcd_spins.values():
            # The shared range helper uses generous numerical spin-box hints
            # for the main plot tabs.  Here they appear in compact pairs, so
            # cap their width while retaining a usable minimum.  They must not
            # be Ignored: the trailing stretch in _pair_row would then shrink
            # the Energy/B-field inputs to zero width.
            spin.setMinimumWidth(72)
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mcd_center_zero_chk = QCheckBox("Zero-centered"); self.mcd_center_zero_chk.setChecked(True)
        self.mcd_auto_v_btn = QPushButton("Auto color")
        self.mcd_pair_b_combo = QComboBox(); self._style_combo_popup(self.mcd_pair_b_combo)
        self.mcd_pair_b_combo.setToolTip(
            "Choose the paired measurement used for the spectra and MCD linecut. "
            "Ctrl+click a B field on the colormap to select its nearest pair."
        )
        self.mcd_window_center_spin = QDoubleSpinBox(); self.mcd_window_center_spin.setRange(0.0, 10.0); self.mcd_window_center_spin.setDecimals(6)
        self.mcd_window_center_spin.setToolTip("Center energy. You can also drag the highlighted band on the MCD map to move this value.")
        self.mcd_window_width_spin = QDoubleSpinBox(); self.mcd_window_width_spin.setRange(0.01, 1000); self.mcd_window_width_spin.setDecimals(3); self.mcd_window_width_spin.setValue(5.0); self.mcd_window_width_spin.setSuffix(" meV")
        self.mcd_window_width_spin.setToolTip("Fixed energy-window width. Dragging the map band moves its center and never changes this width.")
        self.mcd_window_metric_combo = QComboBox(); self.mcd_window_metric_combo.addItems(["Field-signed absolute mean", "Signed mean", "Signed integral", "Unsigned absolute mean (diagnostic)"])
        self.mcd_window_metric_combo.setCurrentText("Signed mean")
        self.mcd_show_raw_chk = QCheckBox("Raw"); self.mcd_show_raw_chk.setChecked(False)
        self.mcd_show_signed_mean_chk = QCheckBox("Signed mean"); self.mcd_show_signed_mean_chk.setChecked(True)
        self.mcd_show_absolute_mean_chk = QCheckBox("B*|MCD|"); self.mcd_show_absolute_mean_chk.setChecked(False)
        self.mcd_show_unsigned_absolute_mean_chk = QCheckBox("|MCD|"); self.mcd_show_unsigned_absolute_mean_chk.setChecked(False)
        self.mcd_show_integral_chk = QCheckBox("Integral"); self.mcd_show_integral_chk.setChecked(False)
        trace_visibility = QWidget(); trace_visibility_layout = QGridLayout(trace_visibility)
        trace_visibility_layout.setContentsMargins(0, 0, 0, 0)
        trace_visibility_layout.setHorizontalSpacing(6); trace_visibility_layout.setVerticalSpacing(2)
        trace_visibility_layout.addWidget(self.mcd_show_raw_chk, 0, 0)
        trace_visibility_layout.addWidget(self.mcd_show_signed_mean_chk, 0, 1)
        trace_visibility_layout.addWidget(self.mcd_show_absolute_mean_chk, 1, 0)
        trace_visibility_layout.addWidget(self.mcd_show_unsigned_absolute_mean_chk, 1, 1)
        trace_visibility_layout.addWidget(self.mcd_show_integral_chk, 2, 0)
        trace_visibility_layout.setColumnStretch(0, 1)
        trace_visibility_layout.setColumnStretch(1, 1)
        self.mcd_window_metric_combo.setToolTip("Selects the primary MCD(B) metric recorded in export settings. The Origin-ready MCD(B) CSV contains corrected signed mean, field-signed absolute mean, and signed integral for both B-sweep directions.")
        self.mcd_show_raw_chk.setToolTip("Add dashed raw-MCD curves for comparison with the corrected curves.")
        self.mcd_show_signed_mean_chk.setToolTip("Average signed MCD inside the selected energy window.")
        self.mcd_show_absolute_mean_chk.setToolTip("MCD magnitude multiplied by the sign of Bpair. It avoids cancellation between opposite spectral lobes while following the positive/negative field sides.")
        self.mcd_show_unsigned_absolute_mean_chk.setToolTip("Pure MCD magnitude, always non-negative. Keep this off unless you need a correction/noise diagnostic.")
        self.mcd_show_integral_chk.setToolTip("Signed energy integral of MCD. It uses the right axis because its unit is MCD eV and changes with the selected window width.")
        self.mcd_fit_zero_chk = QCheckBox("Near-zero fit"); self.mcd_fit_zero_chk.setChecked(True)
        self.mcd_fit_zero_chk.setToolTip(
            "Fit and save separate low-field MCD slopes for the increasing and decreasing branches."
        )
        self.mcd_fit_b_window_spin = QDoubleSpinBox(); self.mcd_fit_b_window_spin.setRange(0.001, 10.0); self.mcd_fit_b_window_spin.setDecimals(3); self.mcd_fit_b_window_spin.setValue(0.2); self.mcd_fit_b_window_spin.setSuffix(" T")
        for combo in (self.mcd_map_combo, self.mcd_pair_b_combo, self.mcd_cmap, self.mcd_window_metric_combo):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        display_form.addRow("Map", self.mcd_map_combo)
        display_form.addRow("Color", self._pair_row(self.mcd_spins["vmin"], self.mcd_spins["vmax"], self.mcd_auto_v_btn))
        display_form.addRow("Energy", self._pair_row(self.mcd_spins["xmin"], self.mcd_spins["xmax"]))
        display_form.addRow("B field", self._pair_row(self.mcd_spins["ymin"], self.mcd_spins["ymax"]))
        display_form.addRow("Selected pair", self.mcd_pair_b_combo)
        display_form.addRow("Color map", self.mcd_cmap)
        display_form.addRow("Scale", self.mcd_center_zero_chk)
        display_form.addRow("MCD(B) E0", self.mcd_window_center_spin)
        display_form.addRow("MCD(B) width", self.mcd_window_width_spin)
        display_form.addRow("MCD(B) traces", trace_visibility)
        display_form.addRow("Primary export metric", self.mcd_window_metric_combo)
        display_form.addRow("Near-zero fit", self._pair_row(self.mcd_fit_zero_chk, self.mcd_fit_b_window_spin))
        layout.addWidget(self._make_expander("Plot", display, expanded=True))
        layout.addStretch(1)
        return tab

    def _build_mcd_peak_shift_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        source = QGroupBox("Reflection peak source")
        form = QFormLayout(source)
        self.mcd_peak_source_summary = QLabel("No MCD result loaded. Load an MCD sweep to begin.")
        self.mcd_peak_source_summary.setWordWrap(True)
        self.mcd_peak_source_combo = QComboBox()
        self.mcd_peak_source_combo.addItems(["Corrected average", "Corrected pos", "Corrected neg", "Raw average", "Raw pos", "Raw neg"])
        self.mcd_peak_source_combo.setToolTip("Reflection spectrum used for peak detection; this is not the raw K-K' intensity difference.")
        self.mcd_peak_display_combo = QComboBox()
        self.mcd_peak_display_combo.addItems(["Delta E", "Absolute E"])
        self.mcd_peak_display_combo.setToolTip("Choose zero-field-referenced energy shift or absolute peak energy.")
        form.addRow("Loaded MCD", self.mcd_peak_source_summary)
        form.addRow("Spectrum", self.mcd_peak_source_combo)
        form.addRow("Display", self.mcd_peak_display_combo)
        layout.addWidget(source)
        controls = QGroupBox("Detection and tracking")
        cform = QFormLayout(controls)
        self.mcd_peak_prom_spin = QDoubleSpinBox(); self.mcd_peak_prom_spin.setRange(0.0, 1.0); self.mcd_peak_prom_spin.setDecimals(3); self.mcd_peak_prom_spin.setValue(0.03)
        self.mcd_peak_dist_spin = QSpinBox(); self.mcd_peak_dist_spin.setRange(1, 500); self.mcd_peak_dist_spin.setValue(5)
        self.mcd_peak_smooth_spin = QSpinBox(); self.mcd_peak_smooth_spin.setRange(3, 101); self.mcd_peak_smooth_spin.setSingleStep(2); self.mcd_peak_smooth_spin.setValue(7)
        self.mcd_peak_jump_spin = QDoubleSpinBox(); self.mcd_peak_jump_spin.setRange(0.0001, 2.0); self.mcd_peak_jump_spin.setDecimals(4); self.mcd_peak_jump_spin.setValue(0.04); self.mcd_peak_jump_spin.setSuffix(" eV")
        self.mcd_peak_max_spin = QSpinBox(); self.mcd_peak_max_spin.setRange(1, 12); self.mcd_peak_max_spin.setValue(6)
        cform.addRow("Prominence fraction", self.mcd_peak_prom_spin)
        cform.addRow("Minimum separation", self.mcd_peak_dist_spin)
        cform.addRow("Smoothing points", self.mcd_peak_smooth_spin)
        cform.addRow("Maximum jump", self.mcd_peak_jump_spin)
        cform.addRow("Maximum peaks", self.mcd_peak_max_spin)
        row = QHBoxLayout()
        self.mcd_peak_analyze_btn = QPushButton("Analyze")
        self.mcd_peak_analyze_btn.setAccessibleName("Analyze MCD reflection peak shifts")
        self.mcd_peak_analyze_btn.setToolTip("Detect multiple reflection peaks and track them through each sweep branch.")
        self.mcd_peak_export_btn = QPushButton("Export CSV")
        self.mcd_peak_export_btn.setAccessibleName("Export MCD peak shift CSV")
        self.mcd_peak_export_btn.setEnabled(False)
        row.addWidget(self.mcd_peak_analyze_btn); row.addWidget(self.mcd_peak_export_btn); row.addStretch(1)
        cform.addRow(row)
        layout.addWidget(controls)
        self.mcd_peak_status = QLabel("Ready when an MCD result is loaded.")
        self.mcd_peak_status.setWordWrap(True)
        layout.addWidget(self.mcd_peak_status)
        self.mcd_peak_table = QTableWidget(0, 7)
        self.mcd_peak_table.setHorizontalHeaderLabels(["Peak", "B (T)", "Branch", "E (eV)", "Delta E (eV)", "Status", "Reference"])
        self.mcd_peak_table.setAlternatingRowColors(True)
        self.mcd_peak_table.setToolTip("Tracked reflection peak energies. Missing and ambiguous points are retained explicitly.")
        self.mcd_peak_table.setAccessibleName("MCD peak shift results")
        layout.addWidget(self.mcd_peak_table, 1)
        self.mcd_valley_table = QTableWidget(0, 11)
        self.mcd_valley_table.setHorizontalHeaderLabels([
            "B (T)", "Branch", "E_K", "E_Kp", "Delta E_K", "Delta E_Kp",
            "Delta E_Kp-K", "Average E", "Odd average", "Even average", "Status",
        ])
        self.mcd_valley_table.setToolTip("K/K' labels use the documented energy-order convention, not waveplate-angle calibration.")
        self.mcd_valley_table.setAccessibleName("MCD valley quantities")
        layout.addWidget(self.mcd_valley_table, 1)
        pair_row = QWidget(); pair_form = QHBoxLayout(pair_row); pair_form.setContentsMargins(0, 0, 0, 0)
        pair_form.addWidget(QLabel("Valley pair"))
        self.mcd_peak_k_combo = QComboBox(); self.mcd_peak_kp_combo = QComboBox()
        for combo in (self.mcd_peak_k_combo, self.mcd_peak_kp_combo):
            combo.setToolTip("Select two tracked optical peak IDs for derived K/K' quantities; labels use energy ordering, not waveplate angles.")
            pair_form.addWidget(combo)
        pair_form.addStretch(1); cform.addRow(pair_row)
        self.mcd_peak_result = None
        self.mcd_peak_analyze_btn.setEnabled(False)
        self.mcd_peak_analyze_btn.clicked.connect(self._analyze_mcd_peak_shift)
        self.mcd_peak_export_btn.clicked.connect(self._export_mcd_peak_shift)
        self.mcd_peak_display_combo.currentTextChanged.connect(self._refresh_mcd_peak_plot)
        self.mcd_peak_k_combo.currentIndexChanged.connect(self._on_mcd_valley_pair_changed)
        self.mcd_peak_kp_combo.currentIndexChanged.connect(self._on_mcd_valley_pair_changed)
        return tab

    def _update_mcd_peak_shift_source(self, result) -> None:
        if not hasattr(self, "mcd_peak_source_summary"):
            return
        if result is None:
            self.mcd_peak_source_summary.setText("No MCD result loaded. Load an MCD sweep to begin.")
            self.mcd_peak_analyze_btn.setEnabled(False)
            self.mcd_peak_export_btn.setEnabled(False)
            self.mcd_peak_status.setText("Empty: load an MCD result first.")
            return
        n = int(np.asarray(result.pair_b).size)
        self.mcd_peak_source_summary.setText(f"{n} paired spectra; {result.source_file}. K/K' labels follow energy ordering: lower branch is K for B > 0; labels are not waveplate-angle calibration.")
        self.mcd_peak_analyze_btn.setEnabled(True)
        self.mcd_peak_status.setText("Loaded. Choose a reflection source and analyze.")
        self.mcd_peak_result = None
        self.mcd_peak_export_btn.setEnabled(False)
        self.mcd_peak_table.setRowCount(0)
        self.mcd_valley_table.setRowCount(0)

    def _analyze_mcd_peak_shift(self) -> None:
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            self.mcd_peak_status.setText("Error: no MCD result is loaded."); return
        self.mcd_peak_analyze_btn.setEnabled(False); self.mcd_peak_status.setText("Analyzing reflection peaks…")
        try:
            self.mcd_peak_result = analyze_peak_shift(self.loaded.mcd_result, source=self.mcd_peak_source_combo.currentText().casefold(), prominence_fraction=self.mcd_peak_prom_spin.value(), min_distance_points=self.mcd_peak_dist_spin.value(), smoothing_points=self.mcd_peak_smooth_spin.value(), max_jump_ev=self.mcd_peak_jump_spin.value(), max_peaks=self.mcd_peak_max_spin.value())
            self._populate_mcd_peak_table()
            self._refresh_mcd_peak_plot()
            self.mcd_peak_export_btn.setEnabled(bool(self.mcd_peak_result.tracks))
            refs = ", ".join(dict.fromkeys(t.reference_method for t in self.mcd_peak_result.tracks)) or "none"
            self.mcd_peak_status.setText(f"Complete: {len(self.mcd_peak_result.tracks)} branch-local peak track(s). Valley pair IDs: {self.mcd_peak_k_combo.currentText()} / {self.mcd_peak_kp_combo.currentText()}. Zero-field reference: {refs}. Missing/ambiguous points are retained.")
        except Exception as exc:
            self.mcd_peak_result = None; self.mcd_peak_export_btn.setEnabled(False); self.mcd_peak_status.setText(f"Error: {exc}")
        finally:
            self.mcd_peak_analyze_btn.setEnabled(True)

    def _populate_mcd_peak_table(self) -> None:
        result = self.mcd_peak_result
        self.mcd_peak_table.setRowCount(0)
        self.mcd_valley_table.setRowCount(0)
        if result is None: return
        ids = sorted({track.peak_id for track in result.tracks})
        for combo in (self.mcd_peak_k_combo, self.mcd_peak_kp_combo):
            previous = combo.currentData(); combo.blockSignals(True); combo.clear()
            for value in ids: combo.addItem(f"Peak {value}", value)
            if previous in ids:
                combo.setCurrentIndex(ids.index(previous))
            elif len(ids) >= 2:
                combo.setCurrentIndex(0 if combo is self.mcd_peak_k_combo else 1)
            combo.blockSignals(False)
        for track in result.tracks:
            for point in track.points:
                row = self.mcd_peak_table.rowCount(); self.mcd_peak_table.insertRow(row)
                reference = track.reference_method
                if track.reference_energy_ev is not None:
                    reference += f": {track.reference_energy_ev:.8g} eV"
                if track.reference_field_t is not None:
                    reference += f" at {track.reference_field_t:.6g} T"
                values = [
                    track.peak_id, f"{point.field_t:.6g}", point.branch,
                    "" if point.energy_ev is None else f"{point.energy_ev:.8g}",
                    "" if point.delta_energy_ev is None else f"{point.delta_energy_ev:.8g}",
                    point.status, reference,
                ]
                for col, value in enumerate(values): self.mcd_peak_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.mcd_peak_table.resizeColumnsToContents()
        for value in valley_quantities(result, (self.mcd_peak_k_combo.currentData(), self.mcd_peak_kp_combo.currentData())):
            row = self.mcd_valley_table.rowCount(); self.mcd_valley_table.insertRow(row)
            b = float(value["B_T"])
            fields = [
                f"{b:.6g}",
                "B > 0: lower=K" if b > 0 else "B < 0: upper=K" if b < 0 else "B = 0: ambiguous",
                value.get("E_K"), value.get("E_Kp"),
                value.get("delta_E_K"), value.get("delta_E_Kp"),
                value.get("splitting_E_Kp_minus_E_K"), value.get("average_E"),
                value.get("odd_average_E"), value.get("even_average_E"), value.get("status"),
            ]
            for col, item in enumerate(fields):
                text = "" if item is None else f"{float(item):.8g}" if isinstance(item, (float, np.floating)) else str(item)
                self.mcd_valley_table.setItem(row, col, QTableWidgetItem(text))
        self.mcd_valley_table.resizeColumnsToContents()

    def _refresh_mcd_peak_plot(self) -> None:
        self._plot_mode("MCD Peak Shift")

    def _on_mcd_valley_pair_changed(self) -> None:
        if self.mcd_peak_result is not None:
            self._populate_mcd_peak_table()
            self._refresh_mcd_peak_plot()

    def _export_mcd_peak_shift(self) -> None:
        if self.mcd_peak_result is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Export MCD peak shifts", "mcd_peak_shift.csv", "CSV files (*.csv)")
        if not path: return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow([
                "peak_id", "B_T", "branch", "E_peak_eV", "delta_E_eV", "status",
                "reference_method", "reference_field_T", "selected_K_peak_id",
                "selected_Kp_peak_id", "E_K_eV", "E_Kp_eV", "delta_E_K_eV",
                "delta_E_Kp_eV", "delta_E_Kp_minus_K_eV", "average_E_eV",
                "odd_average_E_eV", "even_average_E_eV", "odd_splitting_eV",
                "even_splitting_eV",
            ])
            valleys = {(round(float(value["B_T"]), 9), str(value["branch"])): value for value in valley_quantities(self.mcd_peak_result, (self.mcd_peak_k_combo.currentData(), self.mcd_peak_kp_combo.currentData()))}
            for track in self.mcd_peak_result.tracks:
                for point in track.points:
                    valley = valleys.get((round(point.field_t, 9), point.branch), {})
                    writer.writerow([
                        track.peak_id, point.field_t, point.branch, point.energy_ev,
                        point.delta_energy_ev, point.status, track.reference_method,
                        track.reference_field_t, self.mcd_peak_k_combo.currentData(),
                        self.mcd_peak_kp_combo.currentData(), valley.get("E_K"),
                        valley.get("E_Kp"), valley.get("delta_E_K"),
                        valley.get("delta_E_Kp"), valley.get("splitting_E_Kp_minus_E_K"),
                        valley.get("average_E"), valley.get("odd_average_E"),
                        valley.get("even_average_E"), valley.get("odd_splitting"),
                        valley.get("even_splitting"),
                    ])
        self.mcd_peak_status.setText(f"Exported {path}")

    def _build_shg_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.shg_workflow_tabs = QTabWidget()
        single_page = QWidget()
        single_layout = QVBoxLayout(single_page)
        single_layout.setContentsMargins(4, 4, 4, 4)
        single_layout.setSpacing(6)

        data_box = QGroupBox("SHG Sweep Table")
        data_layout = QVBoxLayout(data_box)
        data_layout.setContentsMargins(6, 6, 6, 6)
        data_layout.setSpacing(6)
        self.shg_files = QListWidget()
        self.shg_files.setSelectionMode(QAbstractItemView.SingleSelection)
        self.shg_files.setMinimumHeight(60)
        self.shg_files.setMaximumHeight(120)
        self.shg_files.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.shg_files.setToolTip("Select an SHG sweep CSV to load")
        self.shg_background_combo = QComboBox()
        self._style_combo_popup(self.shg_background_combo)
        self.shg_summary = QPlainTextEdit()
        self.shg_summary.setReadOnly(True)
        self.shg_summary.setMaximumHeight(105)
        background_row = QWidget()
        background_layout = QHBoxLayout(background_row)
        background_layout.setContentsMargins(0, 0, 0, 0)
        background_layout.setSpacing(6)
        background_layout.addWidget(QLabel("External background"))
        background_layout.addWidget(self.shg_background_combo, 1)
        data_layout.addWidget(self.shg_files)
        data_layout.addWidget(background_row)
        data_layout.addWidget(self.shg_summary)
        single_layout.addWidget(self._make_expander("Data", data_box, expanded=True))
        single_layout.addStretch(1)
        self.shg_workflow_tabs.addTab(single_page, "Single File")

        compare_page = QWidget()
        compare_layout = QVBoxLayout(compare_page)
        compare_layout.setContentsMargins(4, 4, 4, 4)
        compare_layout.setSpacing(6)
        compare_box = QGroupBox("SHG Twist Comparison")
        compare_form = QFormLayout(compare_box)
        compare_form.setContentsMargins(6, 6, 6, 6)
        compare_form.setSpacing(6)
        self.shg_compare_reference_combo = QComboBox()
        self.shg_compare_sample_combo = QComboBox()
        self.shg_compare_background_a_combo = QComboBox()
        self.shg_compare_background_b_combo = QComboBox()
        self.shg_compare_display_combo = QComboBox()
        self.shg_compare_display_combo.addItems(["Raw area", "Normalized"])
        for combo in (
            self.shg_compare_reference_combo,
            self.shg_compare_sample_combo,
            self.shg_compare_background_a_combo,
            self.shg_compare_background_b_combo,
            self.shg_compare_display_combo,
        ):
            self._style_combo_popup(combo)
        self.shg_compare_summary = QPlainTextEdit()
        self.shg_compare_summary.setReadOnly(True)
        self.shg_compare_summary.setMaximumHeight(150)
        compare_form.addRow("Reference A", self.shg_compare_reference_combo)
        compare_form.addRow("Sample B", self.shg_compare_sample_combo)
        compare_form.addRow("Background A", self.shg_compare_background_a_combo)
        compare_form.addRow("Background B", self.shg_compare_background_b_combo)
        compare_form.addRow("Plot", self.shg_compare_display_combo)
        compare_form.addRow(self.shg_compare_summary)
        self._set_form_label_width(compare_form, UI_METRICS["label_col_width"])
        compare_layout.addWidget(compare_box)
        compare_layout.addStretch(1)
        self.shg_workflow_tabs.addTab(compare_page, "Compare / Twist Angle")
        layout.addWidget(self.shg_workflow_tabs)

        def wavelength_spin(value: float) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(0.0, 5000.0)
            spin.setSingleStep(0.1)
            spin.setValue(value)
            spin.setFixedWidth(UI_METRICS["spin_w"] + 12)
            return spin

        integration = QGroupBox("SHG Peak Integration")
        integration_form = QFormLayout(integration)
        integration_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        integration_form.setSpacing(6)
        self.shg_peak_center_spin = wavelength_spin(515.0)
        self.shg_gate_half_range_spin = wavelength_spin(3.0)
        self.shg_gate_half_range_spin.setMinimum(0.0001)
        self.shg_sideband_gap_spin = wavelength_spin(1.0)
        self.shg_sideband_width_spin = wavelength_spin(4.0)
        self.shg_sideband_width_spin.setMinimum(0.0001)
        self.shg_background_method_combo = QComboBox()
        self.shg_background_method_combo.addItems(
            ["Local linear", "Local quadratic", "External + local residual", "None"]
        )
        self._style_combo_popup(self.shg_background_method_combo)
        self.shg_sigma_clip_spin = QDoubleSpinBox()
        self.shg_sigma_clip_spin.setDecimals(1)
        self.shg_sigma_clip_spin.setRange(1.0, 10.0)
        self.shg_sigma_clip_spin.setSingleStep(0.5)
        self.shg_sigma_clip_spin.setValue(3.0)
        self.shg_sigma_clip_spin.setFixedWidth(UI_METRICS["spin_w"])

        def value_row(spin: QDoubleSpinBox, suffix: str = "nm") -> QWidget:
            widget = QWidget()
            row_layout = QHBoxLayout(widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            row_layout.addWidget(spin)
            row_layout.addWidget(QLabel(suffix))
            row_layout.addStretch(1)
            return widget

        integration_form.addRow("Integration wavelength", value_row(self.shg_peak_center_spin))
        integration_form.addRow("Integration range (±)", value_row(self.shg_gate_half_range_spin, "nm on each side"))
        integration_form.addRow("Sideband gap", value_row(self.shg_sideband_gap_spin, "nm from peak gate"))
        integration_form.addRow("Sideband width", value_row(self.shg_sideband_width_spin, "nm on each side"))
        integration_form.addRow("Background", self.shg_background_method_combo)
        integration_form.addRow("Sigma clip", self.shg_sigma_clip_spin)
        self._set_form_label_width(integration_form, UI_METRICS["label_col_width"])
        layout.addWidget(self._make_expander("Peak Integration", integration, expanded=True))

        cosmic_box = QGroupBox("Cosmic Ray Removal")
        cosmic_form = QFormLayout(cosmic_box)
        cosmic_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        cosmic_form.setSpacing(6)
        self.shg_cosmic_enable_chk = QCheckBox("Remove narrow positive spikes")
        self.shg_cosmic_enable_chk.setChecked(True)
        self.shg_cosmic_threshold_spin = QDoubleSpinBox()
        self.shg_cosmic_threshold_spin.setDecimals(1)
        self.shg_cosmic_threshold_spin.setRange(3.0, 30.0)
        self.shg_cosmic_threshold_spin.setSingleStep(0.5)
        self.shg_cosmic_threshold_spin.setValue(8.0)
        self.shg_cosmic_window_spin = QSpinBox()
        self.shg_cosmic_window_spin.setRange(3, 51)
        self.shg_cosmic_window_spin.setSingleStep(2)
        self.shg_cosmic_window_spin.setValue(7)
        self.shg_cosmic_max_width_spin = QSpinBox()
        self.shg_cosmic_max_width_spin.setRange(1, 15)
        self.shg_cosmic_max_width_spin.setValue(3)
        self.shg_spectrum_view_combo = QComboBox()
        self.shg_spectrum_view_combo.addItems(["Raw + cleaned", "Raw", "Cosmic-cleaned"])
        self._style_combo_popup(self.shg_spectrum_view_combo)
        cosmic_form.addRow(self.shg_cosmic_enable_chk)
        cosmic_form.addRow("Threshold (MAD)", self.shg_cosmic_threshold_spin)
        cosmic_form.addRow("Detection window", self.shg_cosmic_window_spin)
        cosmic_form.addRow("Maximum width", self.shg_cosmic_max_width_spin)
        cosmic_form.addRow("Spectrum view", self.shg_spectrum_view_combo)
        self._set_form_label_width(cosmic_form, UI_METRICS["label_col_width"])
        layout.addWidget(self._make_expander("Cosmic Rays", cosmic_box, expanded=False))

        angle_box = QGroupBox("Measured Angle")
        angle_form = QFormLayout(angle_box)
        angle_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        angle_form.setSpacing(6)
        self.shg_angle_scale_spin = QDoubleSpinBox()
        self.shg_angle_scale_spin.setDecimals(6)
        self.shg_angle_scale_spin.setRange(-1.0e6, 1.0e6)
        self.shg_angle_scale_spin.setValue(1.0)
        self.shg_angle_offset_spin = QDoubleSpinBox()
        self.shg_angle_offset_spin.setDecimals(6)
        self.shg_angle_offset_spin.setRange(-1.0e6, 1.0e6)
        self.shg_angle_offset_spin.setValue(0.0)
        self.shg_angle_wrap_combo = QComboBox()
        self.shg_angle_wrap_combo.addItems(["None", "0-180°", "0-360°"])
        self._style_combo_popup(self.shg_angle_wrap_combo)
        self.shg_include_failed_chk = QCheckBox("Include move/acquisition failures")
        self.shg_angle_cursor_spin = QDoubleSpinBox()
        self.shg_angle_cursor_spin.setDecimals(6)
        self.shg_angle_cursor_spin.setRange(-1.0e9, 1.0e9)
        self.shg_angle_cursor_spin.setSingleStep(1.0)
        angle_form.addRow("Scale", self.shg_angle_scale_spin)
        angle_form.addRow("Offset (deg)", self.shg_angle_offset_spin)
        angle_form.addRow("Wrap", self.shg_angle_wrap_combo)
        angle_form.addRow("Rows", self.shg_include_failed_chk)
        angle_form.addRow("Selected angle", self.shg_angle_cursor_spin)
        self._set_form_label_width(angle_form, UI_METRICS["label_col_width"])
        layout.addWidget(self._make_expander("Angle", angle_box, expanded=False))

        fit_box = QGroupBox("Angular Fit")
        fit_form = QFormLayout(fit_box)
        fit_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        fit_form.setSpacing(6)
        self.shg_fit_enable_chk = QCheckBox("Fit I(θ) = I₀ + A cos²[2(θ-xc)]")
        self.shg_fit_enable_chk.setChecked(True)
        self.shg_fit_min_spin = QDoubleSpinBox()
        self.shg_fit_max_spin = QDoubleSpinBox()
        for spin, value in ((self.shg_fit_min_spin, 0.0), (self.shg_fit_max_spin, 180.0)):
            spin.setDecimals(4)
            spin.setRange(-1.0e6, 1.0e6)
            spin.setValue(value)
        fit_range_row = QWidget()
        fit_range_layout = QHBoxLayout(fit_range_row)
        fit_range_layout.setContentsMargins(0, 0, 0, 0)
        fit_range_layout.setSpacing(6)
        fit_range_layout.addWidget(self.shg_fit_min_spin)
        fit_range_layout.addWidget(QLabel("to"))
        fit_range_layout.addWidget(self.shg_fit_max_spin)
        fit_range_layout.addWidget(QLabel("deg"))
        self.shg_fit_weighted_chk = QCheckBox("Use area uncertainty weights")
        self.shg_fit_weighted_chk.setChecked(True)
        self.shg_fit_include_excluded_chk = QCheckBox("Include excluded acquisition rows")
        self.shg_fit_branch_spin = QSpinBox()
        self.shg_fit_branch_spin.setRange(-3, 3)
        self.shg_fit_branch_spin.setValue(0)
        self.shg_fit_branch_spin.setEnabled(False)
        self.shg_fit_branch_spin.setToolTip("Adds 90° per branch to Δxc, equivalent to 60° per twist branch")
        self.shg_fit_summary = QPlainTextEdit()
        self.shg_fit_summary.setReadOnly(True)
        self.shg_fit_summary.setMaximumHeight(150)
        fit_form.addRow(self.shg_fit_enable_chk)
        fit_form.addRow("Fit angle range", fit_range_row)
        fit_form.addRow(self.shg_fit_weighted_chk)
        fit_form.addRow(self.shg_fit_include_excluded_chk)
        fit_form.addRow("Phase branch", self.shg_fit_branch_spin)
        fit_form.addRow(self.shg_fit_summary)
        self._set_form_label_width(fit_form, UI_METRICS["label_col_width"])
        layout.addWidget(self._make_expander("Angular Fit", fit_box, expanded=True))
        self.shg_controller._shg_update_cosmic_controls()
        layout.addStretch(1)
        return tab
