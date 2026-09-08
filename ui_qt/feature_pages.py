"""Workflow feature page builders.

These builders own the feature-specific UI. The main window remains the
shared application context for state and event handlers during the next
refactoring stage.
"""

from __future__ import annotations

import csv
import copy
import threading
from dataclasses import replace
from typing import Dict

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLayout,
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

from core.mcd_local_fit import FIT_OK, local_fit_cache_key
from core.mcd_peak_shift import BOUNDARY_UNRELIABLE, analyze_local_peak_shift, analyze_peak_shift, format_mcd_angle, spectrum_energy_order, valley_quantities
from core.mcd_valley_split import compute_valley_splitting
from core.plotting import COMPARE_PANEL_ORDER
from ui_qt.common import UI_METRICS, QComboBox, QDoubleSpinBox, QSpinBox, Worker
from ui_qt.fluent_ui.style import set_fluent_property
from ui_qt.status_badge import StatusBadge
from ui_qt.dense_form_layout import DenseFormRowLayout


def _mcd_local_fit_worker(
    source,
    *,
    seed_energy_ev: float,
    locator_energy_ev: float,
    feature_kind: str,
    peak_id: int,
    spectrum_source: str,
    background_model: str,
    window_ev: tuple[float, float],
    max_starts: int = 150,
    cancel_event=None,
    progress=None,
    log=None,
):
    """Fit the selected local resonance away from the Qt GUI thread."""
    output = {}
    pair_fields = np.asarray(source.pair_b, dtype=float)
    for number, channel in enumerate(("pos", "neg"), start=1):
        if cancel_event is not None and cancel_event.is_set():
            return {}
        field = np.asarray(getattr(source, f"pair_b_{channel}", source.pair_b), dtype=float)
        interpolated = np.asarray(getattr(source, f"pair_interpolated_{channel}", np.zeros(field.size, dtype=bool)), bool)
        effective = np.where(interpolated, pair_fields, field) if interpolated.size == field.size else field
        adapted = copy.copy(source)
        adapted.pair_b = effective
        fit_source = _mcd_fit_source_for_channel(spectrum_source, channel)
        output[channel] = analyze_local_peak_shift(
            adapted,
            source=fit_source,
            seed_energy_ev=float(seed_energy_ev),
            locator_energy_ev=float(locator_energy_ev),
            feature_kind=str(feature_kind),
            peak_id=int(peak_id),
            window_ev=window_ev,
            background_model=str(background_model),
            max_starts=int(max_starts),
            cancel_check=(cancel_event.is_set if cancel_event is not None else None),
        )
        if progress is not None:
            progress.emit(int(number * 50))
    return output


def _mcd_fit_source_for_channel(spectrum_source: str, channel: str) -> str:
    """Map the shared selector to one physical channel per local fit."""
    normalized = str(spectrum_source).casefold().strip()
    base = "corrected" if normalized.startswith(("corrected", "mcd-corrected")) else "raw"
    return f"{base} {str(channel).casefold()}"


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
        source_grid.setSizeConstraint(QLayout.SetMinimumSize)
        source_grid.setContentsMargins(0, 0, 0, 0)
        source_grid.setHorizontalSpacing(6)
        source_grid.setVerticalSpacing(4)
        self.pl_selection_summary = StatusBadge("No PL file selected.", app_role=None)
        self.pl_selection_summary.setMinimumWidth(0)
        self.pl_selection_summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.pl_select_source_btn = QPushButton("Select...")
        self.pl_select_source_btn.setMinimumWidth(110)
        self.pl_select_source_btn.setMaximumWidth(110)
        self.pl_clear_source_btn = QPushButton("Clear")
        self.pl_clear_source_btn.setMaximumWidth(72)
        self.pl_saved_results_btn = QPushButton("Saved results")
        self.pl_saved_results_btn.setMinimumWidth(110)
        self.pl_saved_results_btn.setMaximumWidth(110)
        source_grid.addWidget(self.pl_selection_summary, 0, 0, 1, 3)
        source_grid.setColumnStretch(0, 1)
        source_grid.addWidget(self.pl_select_source_btn, 1, 1)
        source_grid.addWidget(self.pl_clear_source_btn, 1, 2)
        source_grid.addWidget(self.pl_saved_results_btn, 1, 0)
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
        self.pl_saved_results_btn.clicked.connect(self.pl_controller._open_pl_saved_results)
        self.pl_clear_source_btn.clicked.connect(self.pl_controller._clear_pl_source)
        self.pl_auto_next_chk.toggled.connect(
            lambda checked: self.settings.setValue(self.SETTINGS_PL_AUTO_NEXT, bool(checked))
        )
        layout.addWidget(self._make_expander("Measurement File", files, expanded=True))

        params = QGroupBox("Plot Options")
        params.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(6, 6, 6, 4)
        params_layout.setSpacing(4)
        cfg = QFormLayout()
        cfg.setRowWrapPolicy(QFormLayout.WrapLongRows)
        cfg.setHorizontalSpacing(6)
        cfg.setVerticalSpacing(4)
        _grid, spins, _, _, cmap, fix_checks = self._build_common_range_grid("pl", "turbo")
        self.pl_yaxis_controls = self._build_y_axis_controls("pl")
        self.pl_yaxis_combo.setMinimumWidth(200)
        self.pl_yaxis_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        _pl_yc_row = QWidget()
        _pl_yc_h = QVBoxLayout(_pl_yc_row)
        _pl_yc_h.setContentsMargins(0, 0, 0, 0)
        _pl_yc_h.setSpacing(6)
        _pl_yc_h.addWidget(self.pl_yaxis_combo)
        _pl_cmap_row = QWidget()
        _pl_cmap_h = QHBoxLayout(_pl_cmap_row)
        _pl_cmap_h.setContentsMargins(0, 0, 0, 0)
        _pl_cmap_h.setSpacing(4)
        _pl_cmap_h.addWidget(QLabel("Cmap"))
        cmap.setMinimumWidth(112)
        cmap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        _pl_cmap_h.addWidget(cmap, 1)
        _pl_yc_h.addWidget(_pl_cmap_row)
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
            s.setMinimumWidth(116)
            s.setMaximumWidth(120)
            s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            s.setMinimumHeight(UI_METRICS["input_h"])

        self.pl_auto_v_btn = QToolButton()
        self.pl_auto_x_btn = QToolButton()
        self.pl_auto_y_btn = QToolButton()

        basic = QGroupBox("Axis Ranges")
        basic_form = QFormLayout(basic)
        basic_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        basic_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        basic_form.setHorizontalSpacing(4)
        basic_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        basic_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        basic_form.addRow(
            self._make_axis_range_row(spins["vmin"], spins["vmax"], fix_checks["vmin"], fix_checks["vmax"], self.pl_auto_v_btn, "Auto V", dense=True, label_text="vmin / vmax"),
        )
        basic_form.addRow("Color scale", self.pl_split_scale_chk)
        basic_form.addRow(self.pl_split_scale_panel)
        basic_form.addRow(
            self._make_axis_range_row(spins["xmin"], spins["xmax"], fix_checks["xmin"], fix_checks["xmax"], self.pl_auto_x_btn, "Auto X", dense=True, label_text="xmin / xmax"),
        )
        basic_form.addRow(
            self._make_axis_range_row(spins["ymin"], spins["ymax"], fix_checks["ymin"], fix_checks["ymax"], self.pl_auto_y_btn, "Auto Y", dense=True, label_text="ymin / ymax"),
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
        row1h = QGridLayout(row1)
        row1h.setContentsMargins(0, 0, 0, 0)
        row1h.setSpacing(6)
        row1h.addWidget(self.pl_peak_find_btn, 0, 0)
        row1h.addWidget(self.pl_peak_mode_combo, 0, 1)
        row1h.addWidget(self.pl_peak_show_chk, 1, 0, 1, 2)
        row1h.setColumnStretch(1, 1)
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
        row2h = QGridLayout(row2)
        row2h.setContentsMargins(0, 0, 0, 0)
        row2h.setHorizontalSpacing(6)
        row2h.setVerticalSpacing(4)
        for spin, width in ((self.pl_peak_prom_spin, 80), (self.pl_peak_dist_spin, 56), (self.pl_peak_max_spin, 56)):
            spin.setMinimumWidth(width)
            spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        row2h.addWidget(QLabel("Prom"), 0, 0)
        row2h.addWidget(self.pl_peak_prom_spin, 0, 1)
        row2h.addWidget(QLabel("Dist"), 0, 2)
        row2h.addWidget(self.pl_peak_dist_spin, 0, 3)
        row2h.addWidget(QLabel("Top"), 1, 0)
        row2h.addWidget(self.pl_peak_max_spin, 1, 1)
        row2h.setColumnStretch(1, 1)
        row2h.setColumnStretch(3, 1)
        analysis_form.addRow("", row2)
        self.pl_fit_btn = QPushButton("Fit Multi-Lorentz")
        self.pl_fit_clear_btn = QPushButton("Clear Fit")
        self.pl_fit_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.pl_fit_clear_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.pl_fit_show_chk = QCheckBox("Show Fit")
        self.pl_fit_show_chk.setChecked(True)
        self.pl_fit_n_spin = QSpinBox()
        self.pl_fit_n_spin.setRange(1, 8)
        self.pl_fit_n_spin.setValue(3)
        row3 = QWidget()
        row3h = QGridLayout(row3)
        row3h.setContentsMargins(0, 0, 0, 0)
        row3h.setSpacing(6)
        row3h.addWidget(self.pl_fit_btn, 0, 0, 1, 3)
        row3h.addWidget(self.pl_fit_clear_btn, 1, 0, 1, 3)
        row3h.addWidget(self.pl_fit_show_chk, 2, 0)
        row3h.addWidget(QLabel("N"), 2, 1)
        row3h.addWidget(self.pl_fit_n_spin, 2, 2)
        row3h.setColumnStretch(2, 1)
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
        self.drr_edit_measurements_btn.setMinimumWidth(110)
        self.drr_edit_measurements_btn.setMaximumWidth(110)
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
        self.drr_edit_baselines_btn.setMinimumWidth(110)
        self.drr_edit_baselines_btn.setMaximumWidth(110)
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
        params.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
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
            s.setMinimumWidth(116)
            s.setMaximumWidth(120)
            s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            s.setMinimumHeight(UI_METRICS["input_h"])
        self.drr_baseline_combo.setMinimumWidth(210)
        self.drr_baseline_combo.setMaximumWidth(320)
        self.drr_baseline_combo.setFixedHeight(UI_METRICS["input_h"])
        self.drr_baseline_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.drr_yaxis_controls = self._build_y_axis_controls("drr")
        self.drr_yaxis_combo.setMinimumWidth(200)
        self.drr_yaxis_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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

        # DRR range actions are real push buttons so their readable content
        # width participates in the dense row's metric-based packing.
        self.drr_auto_v_btn = QPushButton()
        self.drr_auto_x_btn = QPushButton()
        self.drr_auto_y_btn = QPushButton()
        self.drr_center_zero_chk = QCheckBox("Center Zero")
        self.drr_center_zero_chk.setToolTip("When enabled, DRR colormap is centered at zero.")
        self.drr_center_zero_chk.setChecked(False)

        # Config rows outside Axis Ranges — mirrors PL tab structure
        cfg = QFormLayout()
        cfg.setRowWrapPolicy(QFormLayout.WrapLongRows)
        cfg.setHorizontalSpacing(6)
        cfg.setVerticalSpacing(UI_METRICS["row_spacing"])
        cfg.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        baseline_cmap_row = QWidget()
        baseline_cmap_h = QVBoxLayout(baseline_cmap_row)
        baseline_cmap_h.setContentsMargins(0, 0, 0, 0)
        baseline_cmap_h.setSpacing(4)
        baseline_cmap_h.addWidget(self.drr_baseline_combo)
        drr_cmap_row = QWidget()
        drr_cmap_h = QHBoxLayout(drr_cmap_row)
        drr_cmap_h.setContentsMargins(0, 0, 0, 0)
        drr_cmap_h.setSpacing(4)
        drr_cmap_h.addWidget(QLabel("Cmap"))
        cmap.setMinimumWidth(0)
        cmap.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        drr_cmap_h.addWidget(cmap, 1)
        baseline_cmap_h.addWidget(drr_cmap_row)
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
        basic.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        basic_form = QFormLayout(basic)
        basic_form.setContentsMargins(
            4,
            UI_METRICS["group_margin"],
            4,
            UI_METRICS["group_margin"],
        )
        basic_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        basic_form.setHorizontalSpacing(4)
        basic_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        basic_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        basic_form.addRow(
            self._make_axis_range_row(spins["vmin"], spins["vmax"], fix_checks["vmin"], fix_checks["vmax"], self.drr_auto_v_btn, "Auto V", dense=True, label_text="vmin / vmax"),
        )
        basic_form.addRow("Color scale", self.drr_split_scale_chk)
        basic_form.addRow(self.drr_split_scale_panel)
        basic_form.addRow(
            self._make_axis_range_row(spins["xmin"], spins["xmax"], fix_checks["xmin"], fix_checks["xmax"], self.drr_auto_x_btn, "Auto X", dense=True, label_text="xmin / xmax"),
        )
        basic_form.addRow(
            self._make_axis_range_row(spins["ymin"], spins["ymax"], fix_checks["ymin"], fix_checks["ymax"], self.drr_auto_y_btn, "Auto Y", dense=True, label_text="ymin / ymax"),
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
        self.drr_fit_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.drr_fit_clear_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.drr_fit_n_spin = QSpinBox()
        self.drr_fit_n_spin.setRange(1, 8)
        self.drr_fit_n_spin.setValue(3)
        self.drr_fit_n_spin.setToolTip("Number of Lorentz peaks in fit.")
        self.drr_fit_status = QLabel("")

        drr_row1 = QWidget()
        drr_row1h = QGridLayout(drr_row1)
        drr_row1h.setContentsMargins(0, 0, 0, 0)
        drr_row1h.setSpacing(6)
        drr_row1h.addWidget(self.drr_peak_find_btn, 0, 0)
        drr_row1h.addWidget(self.drr_peak_mode_combo, 0, 1)
        drr_row1h.addWidget(self.drr_peak_show_chk, 1, 0, 1, 2)
        drr_row1h.setColumnStretch(1, 1)
        analysis_form.addRow("", drr_row1)

        drr_row2 = QWidget()
        drr_row2h = QGridLayout(drr_row2)
        drr_row2h.setContentsMargins(0, 0, 0, 0)
        drr_row2h.setHorizontalSpacing(6)
        drr_row2h.setVerticalSpacing(4)
        for spin, width in ((self.drr_peak_prom_spin, 80), (self.drr_peak_dist_spin, 56), (self.drr_peak_max_spin, 56)):
            spin.setMinimumWidth(width)
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        drr_row2h.addWidget(QLabel("Prom"), 0, 0)
        drr_row2h.addWidget(self.drr_peak_prom_spin, 0, 1)
        drr_row2h.addWidget(QLabel("Dist"), 0, 2)
        drr_row2h.addWidget(self.drr_peak_dist_spin, 0, 3)
        drr_row2h.addWidget(QLabel("Top"), 1, 0)
        drr_row2h.addWidget(self.drr_peak_max_spin, 1, 1)
        drr_row2h.setColumnStretch(1, 1)
        drr_row2h.setColumnStretch(3, 1)
        analysis_form.addRow("", drr_row2)

        drr_row3 = QWidget()
        drr_row3h = QGridLayout(drr_row3)
        drr_row3h.setContentsMargins(0, 0, 0, 0)
        drr_row3h.setSpacing(6)
        drr_row3h.addWidget(self.drr_fit_btn, 0, 0, 1, 3)
        drr_row3h.addWidget(self.drr_fit_clear_btn, 1, 0, 1, 3)
        drr_row3h.addWidget(self.drr_fit_show_chk, 2, 0)
        drr_row3h.addWidget(QLabel("N"), 2, 1)
        drr_row3h.addWidget(self.drr_fit_n_spin, 2, 2)
        drr_row3h.setColumnStretch(2, 1)
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
        params.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(6, 6, 6, 4)
        params_layout.setSpacing(4)

        assignment = QGroupBox("Channel Files")
        assignment_layout = QVBoxLayout(assignment)
        assignment_layout.setContentsMargins(6, 8, 6, 6)
        assignment_layout.setSpacing(6)
        angle_rules_form = QFormLayout()
        angle_rules_form.setContentsMargins(0, 0, 0, 0)
        angle_rules_form.setHorizontalSpacing(6)
        angle_rules_form.setVerticalSpacing(4)
        source_filter_row = QHBoxLayout()
        source_filter_row.setContentsMargins(0, 0, 0, 0)
        source_filter_row.setSpacing(6)
        source_filter_row.addWidget(QLabel("Source filter"))
        self.cmp_source_filter_combo = QComboBox()
        self.cmp_source_filter_combo.addItem("PL raw sources", "pl")
        self.cmp_source_filter_combo.addItem("All raw data", "all")
        self.cmp_source_filter_combo.setToolTip(
            "Auto Detect uses files with recognizable rotation angles or channel labels "
            "(KK, KKp, KpK, KpKp). Other files remain available for manual assignment. "
            "Manual assignments are retained when a filter changes."
        )
        self._style_combo_popup(self.cmp_source_filter_combo)
        source_filter_row.addWidget(self.cmp_source_filter_combo, 1)
        assignment_layout.addLayout(source_filter_row)

        group_row = QWidget()
        group_grid = QGridLayout(group_row)
        group_grid.setContentsMargins(0, 0, 0, 0)
        group_grid.setHorizontalSpacing(6)
        group_grid.setVerticalSpacing(4)
        self.cmp_group_selection_summary = StatusBadge("No compare group selected.", app_role=None)
        self.cmp_group_selection_summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.cmp_select_group_btn = QPushButton("Select...")
        self.cmp_select_group_btn.setMinimumWidth(110)
        self.cmp_select_group_btn.setMaximumWidth(110)
        self.cmp_clear_group_btn = QPushButton("Clear")
        self.cmp_clear_group_btn.setMaximumWidth(72)
        group_grid.addWidget(self.cmp_group_selection_summary, 0, 0, 1, 3)
        group_grid.setColumnStretch(0, 1)
        group_grid.addWidget(self.cmp_select_group_btn, 1, 1)
        group_grid.addWidget(self.cmp_clear_group_btn, 1, 2)
        assignment_layout.addWidget(group_row)
        def _angle_spin(default: float = 0.0) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(-360.0, 360.0)
            spin.setValue(default)
            spin.setSuffix(" deg")
            spin.setMinimumWidth(164)
            spin.setMaximumWidth(180)
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        self.cmp_auto_assign_btn.setToolTip(
            "Assign channels from filename rotation angles or explicit KK / KKp / KpK / KpKp labels. "
            "Files without either are skipped; select them manually below."
        )
        for button in (self.cmp_infer_angles_btn, self.cmp_auto_assign_btn):
            button.setMinimumWidth(button.fontMetrics().horizontalAdvance(button.text()) + 12)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        angle_box = QWidget()
        angle_grid = QGridLayout(angle_box)
        angle_grid.setContentsMargins(0, 0, 0, 0)
        angle_grid.setHorizontalSpacing(6)
        angle_grid.setVerticalSpacing(4)
        angle_grid.addWidget(QLabel("In K"), 0, 0)
        angle_grid.addWidget(self.cmp_in_k_angle_spin, 0, 1, 1, 3)
        angle_grid.addWidget(QLabel("In Kp"), 1, 0)
        angle_grid.addWidget(self.cmp_in_kp_angle_spin, 1, 1, 1, 3)
        angle_grid.addWidget(QLabel("Out K"), 2, 0)
        angle_grid.addWidget(self.cmp_out_k_angle_spin, 2, 1, 1, 3)
        angle_grid.addWidget(QLabel("Out Kp"), 3, 0)
        angle_grid.addWidget(self.cmp_out_kp_angle_spin, 3, 1, 1, 3)
        angle_grid.addWidget(QLabel("Tolerance"), 4, 0)
        angle_grid.addWidget(self.cmp_angle_tolerance_spin, 4, 1, 1, 3)
        # Keep the long action on a dedicated row so it retains its full
        # label at the 380 px sidebar width.
        angle_grid.addWidget(self.cmp_infer_angles_btn, 5, 0, 1, 4)
        angle_grid.setColumnStretch(1, 1)
        angle_grid.setColumnStretch(3, 1)
        angle_rules_form.addRow(angle_box)
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
            label = QLabel(key)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            channels_grid.addWidget(label, idx, 0)
            channels_grid.addWidget(combo, idx, 1)
        channels_grid.setColumnStretch(1, 1)
        manual_assignment = self._make_expander(
            "Manual channel assignments", channels_box, expanded=False
        )
        assignment_layout.addWidget(manual_assignment)
        assignment_layout.addWidget(self.cmp_auto_assign_btn)
        self.cmp_swap_kk_btn = QPushButton("Swap KK / KKp")
        self.cmp_swap_kk_btn.setToolTip("Swap the KK and KKp assignments without rerunning Auto Detect.")
        assignment_layout.addWidget(self.cmp_swap_kk_btn)
        self.cmp_assignment_summary = QPlainTextEdit()
        self.cmp_assignment_summary.setReadOnly(True)
        self.cmp_assignment_summary.setMaximumHeight(88)
        summary_form = QFormLayout()
        summary_form.setContentsMargins(0, 0, 0, 0)
        summary_form.setHorizontalSpacing(6)
        summary_form.setVerticalSpacing(4)
        summary_form.addRow("Summary", self.cmp_assignment_summary)
        assignment_layout.addLayout(summary_form)
        self.cmp_group_power_tolerance_percent = 5.0
        self.cmp_select_group_btn.clicked.connect(self.compare_controller._cmp_open_group_dialog)
        self.cmp_clear_group_btn.clicked.connect(self.compare_controller._cmp_clear_group)
        self.cmp_swap_kk_btn.clicked.connect(self.compare_controller._cmp_swap_kk_channels)
        layout.addWidget(self._make_expander("Data Selection", assignment, expanded=True))
        angle_rules = QWidget()
        angle_rules.setLayout(angle_rules_form)
        layout.addWidget(self._make_expander("Angle Rules", angle_rules, expanded=False))

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
        self.cmp_vp_background_spin.setFixedWidth(116)
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
        # Valley polarization has its own color limits.  Keeping these
        # separate from the intensity limits means switching views does not
        # overwrite a carefully chosen intensity scale.
        self.cmp_vp_spins = {}
        self.cmp_vp_fix_checks = {}
        for key, default, label in (
            ("vmin", -1.0, "VP color minimum"),
            ("vmax", 1.0, "VP color maximum"),
        ):
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1.0, 1.0)
            spin.setSingleStep(0.05)
            spin.setValue(default)
            spin.setMinimumWidth(116)
            spin.setMaximumWidth(130)
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            spin.setToolTip(f"{label}; VP limits must stay within -1 and 1.")
            self.cmp_vp_spins[key] = spin
            check = QCheckBox("F")
            check.setToolTip(f"Keep the VP {key} when VP Auto is used.")
            self.cmp_vp_fix_checks[key] = check
        self.cmp_vp_auto_v_btn = QToolButton()
        self.cmp_vp_auto_v_btn.setText("Auto VP")
        self.cmp_vp_auto_v_btn.setToolTip("Set unlocked VP color limits from finite values in the current ROI.")
        self.cmp_vp_auto_v_btn.setAutoRaise(True)
        # Use a stacked layout here because the VP expander sits in the
        # narrow controls sidebar; a horizontal dense row clips the vmax/F
        # controls at the supported minimum width.
        vp_range_row = QWidget()
        vp_range_grid = QGridLayout(vp_range_row)
        vp_range_grid.setContentsMargins(0, 0, 0, 0)
        vp_range_grid.setHorizontalSpacing(4)
        vp_range_grid.setVerticalSpacing(3)
        vp_range_grid.addWidget(QLabel("vmin"), 0, 0)
        vp_range_grid.addWidget(self.cmp_vp_spins["vmin"], 0, 1)
        vp_range_grid.addWidget(self.cmp_vp_fix_checks["vmin"], 0, 2)
        vp_range_grid.addWidget(QLabel("vmax"), 1, 0)
        vp_range_grid.addWidget(self.cmp_vp_spins["vmax"], 1, 1)
        vp_range_grid.addWidget(self.cmp_vp_fix_checks["vmax"], 1, 2)
        vp_range_grid.addWidget(self.cmp_vp_auto_v_btn, 2, 0, 1, 3, Qt.AlignLeft)
        vp_range_grid.setColumnStretch(1, 1)
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
        vp_form.addRow("Color range", vp_range_row)
        vp_form.addRow("VP filename", self.cmp_vp_filename_preview)
        vp_form.addRow("KK title", self.cmp_kk_title_preview)
        vp_form.addRow("KKp title", self.cmp_kkp_title_preview)
        vp_form.addRow("VP title", self.cmp_vp_title_preview)
        self.cmp_vp_expander = self._make_expander("VP", vp_box, expanded=False)
        params_layout.addWidget(self.cmp_vp_expander)

        cfg = QFormLayout()
        cfg.setRowWrapPolicy(QFormLayout.WrapLongRows)
        cfg.setHorizontalSpacing(6)
        cfg.setVerticalSpacing(4)
        _grid, spins, _, _, cmap, fix_checks = self._build_common_range_grid("cmp", "turbo")
        self.cmp_yaxis_controls = self._build_y_axis_controls("cmp")
        self.cmp_yaxis_combo.setMinimumWidth(200)
        self.cmp_yaxis_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        _cmp_yc_row = QWidget()
        _cmp_yc_h = QVBoxLayout(_cmp_yc_row)
        _cmp_yc_h.setContentsMargins(0, 0, 0, 0)
        _cmp_yc_h.setSpacing(6)
        _cmp_yc_h.addWidget(self.cmp_yaxis_combo, 1)
        _cmp_yc_h.addWidget(QLabel("Cmap"))
        _cmp_yc_h.addWidget(cmap)
        cfg.addRow("Y-axis / Cmap", _cmp_yc_row)
        cfg.addRow("", self.cmp_yaxis_advanced_box)
        params_layout.addLayout(cfg)
        for s in spins.values():
            s.setMinimumWidth(116)
            s.setMaximumWidth(120)
            s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            s.setMinimumHeight(UI_METRICS["input_h"])

        self.cmp_auto_v_btn = QToolButton()
        self.cmp_auto_x_btn = QToolButton()
        self.cmp_auto_y_btn = QToolButton()
        basic = QGroupBox("Axis Ranges")
        basic_form = QFormLayout(basic)
        basic_form.setContentsMargins(4, UI_METRICS["group_margin"], 4, UI_METRICS["group_margin"])
        basic_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        basic_form.setHorizontalSpacing(4)
        basic_form.setVerticalSpacing(UI_METRICS["row_spacing"])
        basic_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        basic_form.addRow(
            self._make_axis_range_row(spins["vmin"], spins["vmax"], fix_checks["vmin"], fix_checks["vmax"], self.cmp_auto_v_btn, "Auto V", dense=True, label_text="vmin / vmax"),
        )
        basic_form.addRow("Color scale", self.cmp_split_scale_chk)
        basic_form.addRow(self.cmp_split_scale_panel)
        basic_form.addRow(
            self._make_axis_range_row(spins["xmin"], spins["xmax"], fix_checks["xmin"], fix_checks["xmax"], self.cmp_auto_x_btn, "Auto X", dense=True, label_text="xmin / xmax"),
        )
        basic_form.addRow(
            self._make_axis_range_row(spins["ymin"], spins["ymax"], fix_checks["ymin"], fix_checks["ymax"], self.cmp_auto_y_btn, "Auto Y", dense=True, label_text="ymin / ymax"),
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
        self.mcd_select_source_btn.setMinimumWidth(110)
        self.mcd_select_source_btn.setMaximumWidth(110)
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
        self.mcd_sigma_plus_combo = QComboBox(); self.mcd_sigma_plus_combo.setObjectName("mcd_sigma_plus_combo")
        self.mcd_sigma_minus_combo = QComboBox(); self.mcd_sigma_minus_combo.setObjectName("mcd_sigma_minus_combo")
        for combo in (self.mcd_sigma_plus_combo, self.mcd_sigma_minus_combo):
            combo.addItem("-- Select source CSV first --", None)
            self._style_combo_popup(combo)
        self.mcd_reference_mode_combo = QComboBox(); self.mcd_reference_mode_combo.setObjectName("mcd_reference_mode_combo")
        self.mcd_reference_mode_combo.addItems(["Nearest paired B (recommended)", "Median near-zero window"])
        self._style_combo_popup(self.mcd_reference_mode_combo)
        self.mcd_zero_spin = QDoubleSpinBox(); self.mcd_zero_spin.setRange(0.0, 10.0); self.mcd_zero_spin.setDecimals(4); self.mcd_zero_spin.setValue(0.02); self.mcd_zero_spin.setSuffix(" T")
        self.mcd_zero_spin.setEnabled(False)
        self.mcd_gap_spin = QSpinBox(); self.mcd_gap_spin.setRange(1, 50); self.mcd_gap_spin.setValue(3)
        self.mcd_delta_b_spin = QDoubleSpinBox(); self.mcd_delta_b_spin.setRange(0.0001, 10.0); self.mcd_delta_b_spin.setDecimals(4); self.mcd_delta_b_spin.setValue(0.1); self.mcd_delta_b_spin.setSuffix(" T")
        self.mcd_pair_alignment_combo = QComboBox(); self.mcd_pair_alignment_combo.setObjectName("mcd_pair_alignment_combo"); self.mcd_pair_alignment_combo.addItems(["Direct measured pair", "Interpolate both angles to Bpair"])
        self.mcd_bin_spin = QSpinBox(); self.mcd_bin_spin.setRange(0, 6); self.mcd_bin_spin.setValue(3)
        self.mcd_gain_combo = QComboBox(); self.mcd_gain_combo.setObjectName("mcd_gain_combo"); self.mcd_gain_combo.addItems(["Per wavelength", "Smoothed per wavelength", "Scalar (diagnostic only)"])
        self.mcd_correction_mode_combo = QComboBox(); self.mcd_correction_mode_combo.setObjectName("mcd_correction_mode_combo"); self.mcd_correction_mode_combo.addItems([
            "Global reference gain (current)", "Global gain + per-pair scale", "Global gain + per-pair scale/offset",
            "Global gain + per-pair spectral baseline",
        ])
        self.mcd_correction_mode_combo.setCurrentIndex(3)
        self.mcd_spectral_order_combo = QComboBox(); self.mcd_spectral_order_combo.setObjectName("mcd_spectral_order_combo"); self.mcd_spectral_order_combo.addItems([
            "Linear", "Quadratic (default)",
        ])
        self.mcd_spectral_order_combo.setCurrentIndex(1)
        self.mcd_spectral_order_combo.setEnabled(True)
        self.mcd_background_ranges_edit = QLineEdit()
        self.mcd_background_ranges_edit.setPlaceholderText("Auto outer 15%, or e.g. 1.50-1.58, 1.73-1.79")
        self.mcd_suggest_background_btn = QPushButton("Select protected regions")
        self.mcd_suggest_background_btn.setAccessibleName("Select protected regions")
        self.mcd_suggest_background_btn.setToolTip(
            "Draw persistent feature-protection windows on the full-sweep reflection plot. "
            "Every sufficiently wide unprotected interval updates automatically as a background band."
        )
        self.mcd_background_preview = QLabel("Auto outer 15% ranges are shown after loading an MCD sweep.")
        self.mcd_background_preview.setWordWrap(True)
        self.mcd_background_preview.setMinimumWidth(0)
        self.mcd_background_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.mcd_apply_correction_btn = QPushButton("Recalculate now")
        self.mcd_apply_correction_btn.setAccessibleName("Recalculate now")
        self.mcd_apply_correction_btn.setToolTip(
            "MCD processing updates automatically after settings settle. Click to recalculate immediately or retry after an error."
        )
        self.mcd_dark_pos_combo = QComboBox(); self.mcd_dark_pos_combo.setObjectName("mcd_dark_pos_combo")
        self.mcd_dark_neg_combo = QComboBox(); self.mcd_dark_neg_combo.setObjectName("mcd_dark_neg_combo")
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
        form.addRow(self.mcd_apply_correction_btn)
        advanced_correction = QGroupBox("Advanced correction")
        advanced_form = QFormLayout(advanced_correction)
        advanced_form.setContentsMargins(4, 3, 4, 3)
        advanced_form.setHorizontalSpacing(6)
        advanced_form.setVerticalSpacing(3)
        advanced_form.addRow("Pair B alignment", self.mcd_pair_alignment_combo)
        advanced_form.addRow("B bin decimals", self.mcd_bin_spin)
        advanced_form.addRow("Fit bg E (eV)", self.mcd_background_ranges_edit)
        advanced_form.addRow(self.mcd_suggest_background_btn)
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
        display_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
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
            spin.setMinimumWidth(116)
            spin.setMaximumWidth(130)
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mcd_center_zero_chk = QCheckBox("Zero-centered"); self.mcd_center_zero_chk.setChecked(True)
        self.mcd_auto_v_btn = QPushButton("Auto color")
        self.mcd_auto_v_btn.setMinimumWidth(136)
        self.mcd_auto_v_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
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
            combo.setMinimumWidth(120)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        source_row = QWidget()
        source_layout = QHBoxLayout(source_row)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(4)
        self.mcd_peak_source_selection_summary = StatusBadge("No MCD CSV selected.", app_role=None)
        self.mcd_peak_source_selection_summary.setMinimumWidth(0)
        self.mcd_peak_source_selection_summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.mcd_peak_select_source_btn = QPushButton("Select...")
        self.mcd_peak_select_source_btn.setMinimumWidth(88)
        self.mcd_peak_clear_source_btn = QPushButton("Clear")
        self.mcd_peak_clear_source_btn.setMaximumWidth(64)
        source_layout.addWidget(self.mcd_peak_source_selection_summary, 1)
        source_layout.addWidget(self.mcd_peak_select_source_btn)
        source_layout.addWidget(self.mcd_peak_clear_source_btn)
        form.addRow("MCD CSV", source_row)
        self.mcd_peak_source_summary = QLabel("No MCD result loaded. Load an MCD sweep to begin.")
        self.mcd_peak_source_summary.setWordWrap(True)
        self.mcd_peak_source_combo = QComboBox()
        self.mcd_peak_source_combo.addItems(["Raw R", "MCD-corrected R"])
        self.mcd_peak_source_combo.setMinimumWidth(210)
        self.mcd_peak_source_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mcd_peak_source_combo.setToolTip("Reflection source for the local fit. Both physical channels are fitted separately; this is not the K-K' intensity difference.")
        self.mcd_peak_display_combo = QComboBox()
        self.mcd_peak_display_combo.addItems(["Absolute E", "Delta E"])
        self.mcd_peak_display_combo.setToolTip("Choose zero-field-referenced energy shift or absolute peak energy.")
        form.addRow("Loaded MCD", self.mcd_peak_source_summary)
        form.addRow("Display", self.mcd_peak_display_combo)
        self.mcd_peak_display_combo.hide()
        for hidden_combo in (self.mcd_peak_display_combo,):
            label = form.labelForField(hidden_combo)
            if label is not None:
                label.hide()
        layout.addWidget(source)
        self.mcd_peak_select_source_btn.clicked.connect(self.mcd_controller._edit_mcd_source)
        self.mcd_peak_clear_source_btn.clicked.connect(self.mcd_controller._clear_mcd_source)
        preview_controls = QGroupBox("Peak inspection")
        preview_form = QFormLayout(preview_controls)
        self.mcd_peak_branch_combo = QComboBox()
        self.mcd_peak_branch_combo.addItem("All sweep directions")
        self.mcd_peak_selector_combo = QComboBox()
        self.mcd_peak_field_combo = QComboBox()
        self.mcd_peak_field_combo.setToolTip("Selected actual raw field; map clicks choose the nearest measured field.")
        self.mcd_peak_field_prev_btn = QToolButton(); self.mcd_peak_field_prev_btn.setText("‹"); self.mcd_peak_field_prev_btn.setToolTip("Select previous measured field")
        self.mcd_peak_field_next_btn = QToolButton(); self.mcd_peak_field_next_btn.setText("›"); self.mcd_peak_field_next_btn.setToolTip("Select next measured field")
        self.mcd_peak_field_prev_btn.setAutoRaise(True); self.mcd_peak_field_next_btn.setAutoRaise(True)
        self.mcd_peak_prev_btn = QToolButton(); self.mcd_peak_prev_btn.setText("‹"); self.mcd_peak_prev_btn.setToolTip("Select previous reliable feature")
        self.mcd_peak_next_btn = QToolButton(); self.mcd_peak_next_btn.setText("›"); self.mcd_peak_next_btn.setToolTip("Select next reliable feature")
        self.mcd_peak_prev_btn.setAutoRaise(True); self.mcd_peak_next_btn.setAutoRaise(True)
        feature_row = QWidget(); feature_layout = QHBoxLayout(feature_row); feature_layout.setContentsMargins(0, 0, 0, 0); feature_layout.setSpacing(2)
        feature_layout.addWidget(self.mcd_peak_selector_combo, 1); feature_layout.addWidget(self.mcd_peak_prev_btn); feature_layout.addWidget(self.mcd_peak_next_btn)
        field_row = QWidget(); field_layout = QHBoxLayout(field_row); field_layout.setContentsMargins(0, 0, 0, 0); field_layout.setSpacing(4)
        field_layout.addWidget(self.mcd_peak_field_combo, 1); field_layout.addWidget(self.mcd_peak_field_prev_btn); field_layout.addWidget(self.mcd_peak_field_next_btn)
        candidate_row = QWidget()
        candidate_grid = QGridLayout(candidate_row)
        candidate_grid.setContentsMargins(0, 0, 0, 0)
        candidate_grid.setHorizontalSpacing(3)
        candidate_grid.setVerticalSpacing(2)
        self.mcd_peak_candidate_buttons: list[QToolButton] = []
        self._mcd_peak_candidate_keys: list[tuple] = []
        for candidate_index in range(5):
            button = QToolButton()
            button.setCheckable(True)
            button.setAutoRaise(True)
            button.setVisible(False)
            button.clicked.connect(
                lambda _checked=False, index=candidate_index: self._on_mcd_peak_candidate_clicked(index)
            )
            self.mcd_peak_candidate_buttons.append(button)
            candidate_grid.addWidget(button, candidate_index // 2, candidate_index % 2)
        preview_form.addRow("Sweep direction", self.mcd_peak_branch_combo)
        preview_form.addRow("Selected feature", feature_row)
        preview_form.addRow("Nearby features", candidate_row)
        # Feature selection is rendered in the plot panel's MCD-style bar.
        # Keep the combo/buttons as the authoritative model and for keyboard/
        # workflow compatibility, but remove their sidebar rows.
        for hidden_widget in (
            feature_row,
            candidate_row,
            preview_form.labelForField(feature_row),
            preview_form.labelForField(candidate_row),
        ):
            if hidden_widget is not None:
                hidden_widget.hide()
        self.mcd_peak_selector_combo.hide()
        self.mcd_peak_prev_btn.hide()
        self.mcd_peak_next_btn.hide()
        for candidate_button in self.mcd_peak_candidate_buttons:
            candidate_button.hide()
        preview_form.addRow("Selected field", field_row)
        self.mcd_peak_map_mode_combo = QComboBox(); self.mcd_peak_map_mode_combo.addItems(["Raw R", "Second derivative"])
        self.mcd_peak_tracker_method_combo = QComboBox(); self.mcd_peak_tracker_method_combo.addItems(["Local mixed fit", "Raw spectrum", "Second derivative"])
        self.mcd_peak_tracker_method_combo.setCurrentText("Raw spectrum")
        self.mcd_peak_tracker_method_combo.setToolTip("By default, show Raw spectrum and Second derivative peak-finding results together. This selector chooses the candidate list; Local mixed fit is optional.")
        self.mcd_peak_result_mode_combo = QComboBox(); self.mcd_peak_result_mode_combo.addItems(["Valley splitting", "Single peak shift"])
        self.mcd_peak_result_mode_combo.setToolTip("Choose absolute Kp−K valley splitting or the selected per-channel peak shift.")
        self.mcd_peak_deriv_window_spin = QSpinBox(); self.mcd_peak_deriv_window_spin.setRange(7, 101); self.mcd_peak_deriv_window_spin.setSingleStep(2); self.mcd_peak_deriv_window_spin.setValue(35)
        self.mcd_peak_show_tracks_chk = QCheckBox("Show peak tracks"); self.mcd_peak_show_tracks_chk.setChecked(True)
        self.mcd_peak_show_maps_chk = QCheckBox("Inspect tracking maps"); self.mcd_peak_show_maps_chk.setChecked(False)
        self.mcd_peak_show_derivative_chk = QCheckBox("Show derivative panel"); self.mcd_peak_show_derivative_chk.setChecked(True)
        preview_form.addRow("Map display", self.mcd_peak_map_mode_combo)
        self._mcd_peak_map_mode_label = preview_form.labelForField(self.mcd_peak_map_mode_combo)
        preview_form.addRow("Peak tracker", self.mcd_peak_tracker_method_combo)
        preview_form.addRow("Result", self.mcd_peak_result_mode_combo)
        preview_form.addRow("Derivative SG window", self.mcd_peak_deriv_window_spin)
        preview_form.addRow(self.mcd_peak_show_tracks_chk)
        preview_form.addRow(self.mcd_peak_show_maps_chk)
        preview_form.addRow(self.mcd_peak_show_derivative_chk)
        self.mcd_peak_note = QLabel("Displayed spectra and SG second derivative use the selected reflection source. Positive B assigns K to the lower-energy member.")
        self.mcd_peak_note.setWordWrap(True)
        preview_form.addRow(self.mcd_peak_note)
        layout.addWidget(preview_controls)
        controls = QGroupBox("Detection and tracking")
        cform = QFormLayout(controls)
        cform.setHorizontalSpacing(6)
        cform.setVerticalSpacing(4)
        self.mcd_peak_prom_spin = QDoubleSpinBox(); self.mcd_peak_prom_spin.setRange(0.0, 1.0); self.mcd_peak_prom_spin.setDecimals(3); self.mcd_peak_prom_spin.setValue(0.03)
        self.mcd_peak_dist_spin = QSpinBox(); self.mcd_peak_dist_spin.setRange(1, 500); self.mcd_peak_dist_spin.setValue(5)
        self.mcd_peak_smooth_spin = QSpinBox(); self.mcd_peak_smooth_spin.setRange(3, 101); self.mcd_peak_smooth_spin.setSingleStep(2); self.mcd_peak_smooth_spin.setValue(7)
        self.mcd_peak_jump_spin = QDoubleSpinBox(); self.mcd_peak_jump_spin.setRange(0.0001, 2.0); self.mcd_peak_jump_spin.setDecimals(4); self.mcd_peak_jump_spin.setValue(0.04); self.mcd_peak_jump_spin.setSuffix(" eV")
        self.mcd_peak_max_spin = QSpinBox(); self.mcd_peak_max_spin.setRange(1, 12); self.mcd_peak_max_spin.setValue(6)
        for spin in (self.mcd_peak_prom_spin, self.mcd_peak_dist_spin, self.mcd_peak_smooth_spin, self.mcd_peak_jump_spin, self.mcd_peak_max_spin):
            spin.setMinimumWidth(128 if isinstance(spin, QDoubleSpinBox) else 72)
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mcd_peak_prom_spin.setToolTip("Prominence fraction used for reflection-peak detection.")
        self.mcd_peak_dist_spin.setToolTip("Minimum separation between detected peaks, in points.")
        self.mcd_peak_smooth_spin.setToolTip("Smoothing window length, in points.")
        self.mcd_peak_jump_spin.setToolTip("Maximum allowed peak jump in eV.")
        self.mcd_peak_max_spin.setToolTip("Maximum number of peaks to track.")
        def _peak_control_row(label: str, spin: QAbstractSpinBox) -> QWidget:
            row = QWidget()
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(2)
            row_layout.addWidget(QLabel(label))
            row_layout.addWidget(spin)
            return row

        cform.addRow(_peak_control_row("Prominence fraction", self.mcd_peak_prom_spin))
        cform.addRow(_peak_control_row("Minimum separation", self.mcd_peak_dist_spin))
        cform.addRow(_peak_control_row("Smoothing points", self.mcd_peak_smooth_spin))
        cform.addRow(_peak_control_row("Maximum jump", self.mcd_peak_jump_spin))
        cform.addRow(_peak_control_row("Maximum peaks", self.mcd_peak_max_spin))
        self.mcd_peak_background_combo = QComboBox(); self.mcd_peak_background_combo.addItems(["Linear background", "Quadratic background"])
        self.mcd_peak_background_combo.setToolTip("Local mixed fit background. Quadratic is an optional comparison model.")
        cform.addRow("Reflection source", self.mcd_peak_source_combo)
        cform.addRow("Local background", self.mcd_peak_background_combo)
        row = QHBoxLayout()
        self.mcd_peak_analyze_btn = QPushButton("Analyze")
        self.mcd_peak_analyze_btn.setAccessibleName("Analyze MCD reflection peak shifts")
        self.mcd_peak_analyze_btn.setToolTip("Detect multiple reflection peaks and track them through each sweep branch.")
        self.mcd_peak_export_btn = QPushButton("Export CSV")
        self.mcd_peak_export_btn.setAccessibleName("Export MCD peak shift CSV")
        self.mcd_peak_export_btn.setEnabled(False)
        row.addWidget(self.mcd_peak_analyze_btn); row.addWidget(self.mcd_peak_export_btn); row.addStretch(1)
        self.mcd_peak_controls_expander = self._make_expander("Advanced detection", controls, expanded=False)
        layout.addWidget(self.mcd_peak_controls_expander)
        layout.addLayout(row)
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
        self.mcd_peak_table.hide()
        self.mcd_valley_table.hide()
        pair_row = QWidget(); pair_form = QHBoxLayout(pair_row); pair_form.setContentsMargins(0, 0, 0, 0)
        pair_form.addWidget(QLabel("Valley pair"))
        self.mcd_peak_k_combo = QComboBox(); self.mcd_peak_kp_combo = QComboBox()
        for combo in (self.mcd_peak_k_combo, self.mcd_peak_kp_combo):
            combo.setToolTip("Select two tracked optical peak IDs for derived K/K' quantities; labels use energy ordering, not waveplate angles.")
            pair_form.addWidget(combo)
        pair_form.addStretch(1); cform.addRow(pair_row)
        pair_row.hide()
        self.mcd_peak_result = None
        self.mcd_peak_channel_results = {}
        self._mcd_peak_local_fit_cache = {}
        self._mcd_peak_fit_generation = 0
        self._mcd_peak_fit_worker = None
        self.mcd_peak_map_axes = []
        self.mcd_peak_spectrum_ax = None
        self._mcd_peak_track_lines = []
        self._mcd_peak_candidate_artists = {}
        self._mcd_peak_manual_center_ev: float | None = None
        self._mcd_peak_manual_center_channel: str | None = None
        self._mcd_peak_manual_center_field_index: int | None = None
        self.mcd_peak_selected_field = None
        self.mcd_peak_analyze_btn.setEnabled(False)
        self.mcd_peak_analyze_btn.clicked.connect(self._analyze_mcd_peak_shift)
        self.mcd_peak_export_btn.clicked.connect(self._export_mcd_peak_shift)
        self.mcd_peak_export_btn.hide()
        self.mcd_peak_display_combo.currentTextChanged.connect(self._refresh_mcd_peak_plot)
        self.mcd_peak_source_combo.currentTextChanged.connect(self._on_mcd_peak_source_changed)
        self.mcd_peak_map_mode_combo.currentTextChanged.connect(self._refresh_mcd_peak_plot)
        self.mcd_peak_tracker_method_combo.currentTextChanged.connect(self._reanalyze_mcd_peak_shift)
        self.mcd_peak_background_combo.currentTextChanged.connect(self._reanalyze_mcd_peak_shift)
        self.mcd_peak_result_mode_combo.currentTextChanged.connect(self._refresh_mcd_peak_plot)
        self.mcd_peak_deriv_window_spin.valueChanged.connect(self._on_mcd_peak_deriv_window_changed)
        self.mcd_peak_show_tracks_chk.toggled.connect(self._refresh_mcd_peak_plot)
        self.mcd_peak_branch_combo.currentTextChanged.connect(self._on_mcd_peak_branch_changed)
        self.mcd_peak_selector_combo.currentIndexChanged.connect(self._refresh_mcd_peak_plot)
        self.mcd_peak_selector_combo.currentIndexChanged.connect(self._on_mcd_local_selection_changed)
        self.mcd_peak_field_combo.currentIndexChanged.connect(self._on_mcd_peak_field_changed)
        self.mcd_peak_prev_btn.clicked.connect(lambda: self._step_mcd_peak_feature(-1))
        self.mcd_peak_next_btn.clicked.connect(lambda: self._step_mcd_peak_feature(1))
        self.mcd_peak_field_prev_btn.clicked.connect(lambda: self._step_mcd_peak_field(-1))
        self.mcd_peak_field_next_btn.clicked.connect(lambda: self._step_mcd_peak_field(1))
        self.mcd_peak_show_maps_chk.toggled.connect(self._refresh_mcd_peak_plot)
        self.mcd_peak_show_maps_chk.toggled.connect(self._set_mcd_inspect_controls)
        self.mcd_peak_show_derivative_chk.toggled.connect(self._refresh_mcd_peak_plot)
        self._set_mcd_inspect_controls(False)
        self.mcd_peak_k_combo.currentIndexChanged.connect(self._on_mcd_valley_pair_changed)
        self.mcd_peak_kp_combo.currentIndexChanged.connect(self._on_mcd_valley_pair_changed)
        return tab

    def _set_mcd_inspect_controls(self, visible: bool) -> None:
        self.mcd_peak_map_mode_combo.setVisible(bool(visible))
        if getattr(self, "_mcd_peak_map_mode_label", None) is not None:
            self._mcd_peak_map_mode_label.setVisible(bool(visible))
        self.mcd_peak_show_tracks_chk.setVisible(bool(visible))

    def _step_mcd_peak_feature(self, delta: int) -> None:
        combo = getattr(self, "mcd_peak_selector_combo", None)
        if combo is None or combo.count() == 0:
            return
        index = combo.currentIndex()
        combo.setCurrentIndex((index + int(delta)) % combo.count())

    def _step_mcd_peak_field(self, delta: int) -> None:
        combo = getattr(self, "mcd_peak_field_combo", None)
        if combo is None or combo.count() == 0:
            return
        index = combo.currentIndex()
        combo.setCurrentIndex((index + int(delta)) % combo.count())

    def _update_mcd_peak_shift_source(self, result) -> None:
        if not hasattr(self, "mcd_peak_source_summary"):
            return
        if result is not None and result is getattr(self, "_mcd_peak_analysis_source", None):
            return
        self._mcd_peak_analysis_source = result
        self._mcd_peak_analysis_key = None
        self._mcd_peak_local_pending_key = None
        cancel = getattr(self, "_mcd_peak_fit_cancel_event", None)
        if cancel is not None:
            cancel.set()
        self._mcd_peak_fit_generation = int(getattr(self, "_mcd_peak_fit_generation", 0)) + 1
        self._mcd_peak_local_fit_cache = {}
        self._mcd_peak_locator_results = {}
        if result is None:
            self.mcd_peak_result = None
            self.mcd_peak_source_summary.setText("No MCD result loaded. Load an MCD sweep to begin.")
            self.mcd_peak_analyze_btn.setEnabled(False)
            self.mcd_peak_export_btn.setEnabled(False)
            self.mcd_peak_status.setText("Empty: load an MCD result first.")
            self.mcd_peak_channel_results = {}
            self.mcd_peak_method_results = {}
            self.mcd_peak_map_axes = []
            self._mcd_peak_candidate_artists = {}
            self._clear_mcd_peak_manual_center()
            self.mcd_peak_selected_field = None
            self.mcd_peak_selector_combo.clear()
            self.mcd_peak_field_combo.clear()
            self.mcd_peak_table.setRowCount(0)
            self.mcd_valley_table.setRowCount(0)
            return
        n = int(np.asarray(result.pair_b).size)
        self.mcd_peak_source_summary.setText(f"{n} paired spectra; {result.source_file}. K/K' labels follow energy ordering: lower branch is K for B > 0; labels are not waveplate-angle calibration.")
        self.mcd_peak_analyze_btn.setEnabled(True)
        self.mcd_peak_status.setText("Loaded. Choose a reflection source and analyze.")
        self.mcd_peak_result = None
        self.mcd_peak_channel_results = {}
        self.mcd_peak_method_results = {}
        self.mcd_peak_map_axes = []
        self._mcd_peak_candidate_artists = {}
        self._clear_mcd_peak_manual_center()
        self.mcd_peak_selected_field = None
        self.mcd_peak_selector_combo.clear()
        self.mcd_peak_field_combo.clear()
        self.mcd_peak_export_btn.setEnabled(False)
        self.mcd_peak_table.setRowCount(0)
        self.mcd_valley_table.setRowCount(0)

    def _mcd_peak_computation_key(self) -> tuple:
        """Cheap freshness check; display controls do not affect computation."""
        return (
            id(self.loaded.mcd_result),
            self.mcd_peak_source_combo.currentText(),
            self.mcd_peak_tracker_method_combo.currentText(),
            self.mcd_peak_background_combo.currentText(),
            self.mcd_peak_prom_spin.value(), self.mcd_peak_dist_spin.value(),
            self.mcd_peak_smooth_spin.value(), self.mcd_peak_jump_spin.value(),
            self.mcd_peak_max_spin.value(), self.mcd_peak_deriv_window_spin.value(),
        )

    def _ensure_mcd_peak_analysis(self) -> None:
        if self.loaded is None or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            return
        if self._load_in_progress or self.mcd_controller._mcd_auto_apply_timer.isActive():
            return
        key = self._mcd_peak_computation_key()
        cancel = getattr(self, "_mcd_peak_fit_cancel_event", None)
        pending = getattr(self, "_mcd_peak_local_pending_key", None) is not None and cancel is not None and not cancel.is_set()
        if key == getattr(self, "_mcd_peak_analysis_key", None) and (self.mcd_peak_result is not None or pending):
            return
        self._analyze_mcd_peak_shift()

    def _analyze_mcd_peak_shift(self) -> None:
        if self.loaded is not None and self.loaded.mode == "MCD" and self.loaded.mcd_result is not None:
            self._mcd_peak_analysis_key = self._mcd_peak_computation_key()
        if self.mcd_peak_tracker_method_combo.currentText() == "Local mixed fit":
            self._request_mcd_local_fit()
            return
        cancel = getattr(self, "_mcd_peak_fit_cancel_event", None)
        if cancel is not None:
            cancel.set()
        self._mcd_peak_local_pending_key = None
        self._mcd_peak_fit_generation = int(getattr(self, "_mcd_peak_fit_generation", 0)) + 1
        self._analyze_mcd_peak_shift_legacy()

    def _on_mcd_peak_source_changed(self, _source_text: str = "") -> None:
        """Invalidate source-dependent locators while retaining reusable fits."""
        if getattr(self, "_mcd_peak_source_change_guard", False):
            return
        self._mcd_peak_source_change_guard = True
        try:
            self._mcd_peak_fit_generation = int(getattr(self, "_mcd_peak_fit_generation", 0)) + 1
            cancel_event = getattr(self, "_mcd_peak_fit_cancel_event", None)
            if cancel_event is not None:
                cancel_event.set()
            # Locator candidates belong to the selected reflection source;
            # cached local model results are keyed by source and can be reused
            # when the user returns to a prior source.
            self._mcd_peak_locator_results = {}
            self._mcd_peak_local_requested_key = None
            self.mcd_peak_result = None
            self.mcd_peak_channel_results = {}
            self.mcd_peak_method_results = {}
            self.mcd_peak_export_btn.setEnabled(False)
            self.mcd_peak_table.setRowCount(0)
            self.mcd_valley_table.setRowCount(0)
            self.mcd_peak_selector_combo.blockSignals(True)
            self.mcd_peak_selector_combo.clear()
            self.mcd_peak_selector_combo.blockSignals(False)
            self._mcd_peak_candidate_keys = []
            for button in getattr(self, "mcd_peak_candidate_buttons", ()):
                button.setVisible(False)
            self._refresh_mcd_peak_plot()
            if self.loaded is not None and self.loaded.mcd_result is not None:
                self._analyze_mcd_peak_shift()
        finally:
            self._mcd_peak_source_change_guard = False

    def _request_mcd_local_fit(
        self,
        *,
        seed_energy_ev: float | None = None,
        locator_energy_ev: float | None = None,
        feature_kind: str | None = None,
        selection_key: tuple | None = None,
    ) -> None:
        """Queue one selected raw-R local fit and ignore stale completions."""
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            self.mcd_peak_status.setText("Error: no MCD result is loaded.")
            return
        source = self.loaded.mcd_result
        selected = selection_key if selection_key is not None else self.mcd_peak_selector_combo.currentData()
        selected_track = None
        locator_results = getattr(self, "_mcd_peak_locator_results", {})
        if selected is not None and len(selected) == 4:
            selected_channel, selected_id, selected_branch, selected_kind = selected
            current_analysis = locator_results.get(str(selected_channel)) or self.mcd_peak_channel_results.get(str(selected_channel))
            if current_analysis is not None:
                selected_track = next((track for track in current_analysis.tracks if int(track.peak_id) == int(selected_id) and str(track.branch) == str(selected_branch) and str(track.feature_kind) == str(selected_kind)), None)
        if seed_energy_ev is None and selected_track is not None:
            seed_energy_ev = selected_track.locator_energy_ev or selected_track.reference_energy_ev
        if seed_energy_ev is None:
            # The raw detector supplies a locator only; no raw/D2 centre is
            # promoted into the local model result.
            field = np.asarray(getattr(source, "pair_b_pos", source.pair_b), dtype=float)
            effective = np.where(np.asarray(getattr(source, "pair_interpolated_pos", np.zeros(field.size, dtype=bool)), bool), np.asarray(source.pair_b, dtype=float), field)
            adapted = copy.copy(source); adapted.pair_b = effective
            locator = analyze_peak_shift(
                adapted, source=_mcd_fit_source_for_channel(self.mcd_peak_source_combo.currentText(), "pos"), prominence_fraction=self.mcd_peak_prom_spin.value(), min_distance_points=self.mcd_peak_dist_spin.value(),
                smoothing_points=self.mcd_peak_smooth_spin.value(), max_jump_ev=self.mcd_peak_jump_spin.value(), max_peaks=self.mcd_peak_max_spin.value(), tracking_method="Raw spectrum",
            )
            locator_results = {"pos": locator}
            for locator_channel in ("neg",):
                locator_field = np.asarray(getattr(source, f"pair_b_{locator_channel}", source.pair_b), dtype=float)
                locator_interp = np.asarray(getattr(source, f"pair_interpolated_{locator_channel}", np.zeros(locator_field.size, dtype=bool)), bool)
                locator_adapted = copy.copy(source)
                locator_adapted.pair_b = np.where(locator_interp, np.asarray(source.pair_b, dtype=float), locator_field)
                locator_results[locator_channel] = analyze_peak_shift(
                    locator_adapted, source=_mcd_fit_source_for_channel(self.mcd_peak_source_combo.currentText(), locator_channel), prominence_fraction=self.mcd_peak_prom_spin.value(), min_distance_points=self.mcd_peak_dist_spin.value(),
                    smoothing_points=self.mcd_peak_smooth_spin.value(), max_jump_ev=self.mcd_peak_jump_spin.value(), max_peaks=self.mcd_peak_max_spin.value(), tracking_method="Raw spectrum",
                )
            self._mcd_peak_locator_results = locator_results
            usable = [track for track in locator.tracks if track.quality != BOUNDARY_UNRELIABLE and track.reference_energy_ev is not None]
            if not usable:
                self.mcd_peak_status.setText(
                    f"No reliable {self.mcd_peak_source_combo.currentText()} locator was found for a local fit."
                )
                return
            floor = float(np.nanmin(np.asarray(source.energy_ev, dtype=float))) + 0.01
            candidates = [track for track in usable if float(track.reference_energy_ev) > floor] or usable
            selected_track = min(candidates, key=lambda track: float(track.reference_energy_ev))
            seed_energy_ev = float(selected_track.reference_energy_ev)
            locator_energy_ev = seed_energy_ev
            feature_kind = selected_track.feature_kind
        if selected is not None and len(selected) == 4:
            requested_channel = str(selected_channel)
            requested_id = int(selected_id)
            requested_branch = str(selected_branch)
            requested_kind = str(selected_kind)
        elif selected_track is not None:
            requested_channel = "pos"
            requested_id = int(selected_track.peak_id)
            requested_branch = str(selected_track.branch)
            requested_kind = str(selected_track.feature_kind)
        else:
            requested_channel = "pos"
            requested_id = 1
            requested_branch = str(np.asarray(source.pair_labels, dtype=str)[0])
            requested_kind = feature_kind
        self._mcd_peak_local_requested_key = (requested_channel, requested_id, requested_branch, requested_kind)
        seed_energy_ev = float(seed_energy_ev)
        locator_energy_ev = float(locator_energy_ev if locator_energy_ev is not None else seed_energy_ev)
        feature_kind = str(feature_kind or (selected_track.feature_kind if selected_track is not None else "peak"))
        background_model = "quadratic" if self.mcd_peak_background_combo.currentText().startswith("Quadratic") else "linear"
        window_ev = (seed_energy_ev - 0.0136, seed_energy_ev + 0.0164)
        spectrum_source = str(self.mcd_peak_source_combo.currentText())
        settings = (
            self.mcd_peak_prom_spin.value(), self.mcd_peak_dist_spin.value(),
            self.mcd_peak_smooth_spin.value(), self.mcd_peak_jump_spin.value(),
            self.mcd_peak_deriv_window_spin.value(), feature_kind,
            requested_id, requested_branch,
        )
        cache_key = local_fit_cache_key(source, source=spectrum_source, background_model=background_model, window_ev=window_ev, seed_energy_ev=seed_energy_ev, settings=settings)
        previous_cancel = getattr(self, "_mcd_peak_fit_cancel_event", None)
        if (
            cache_key == getattr(self, "_mcd_peak_local_pending_key", None)
            and previous_cancel is not None and not previous_cancel.is_set()
        ):
            return
        self._mcd_peak_fit_generation = int(getattr(self, "_mcd_peak_fit_generation", 0)) + 1
        generation = self._mcd_peak_fit_generation
        if previous_cancel is not None:
            previous_cancel.set()
        cancel_event = threading.Event()
        self._mcd_peak_fit_cancel_event = cancel_event
        self._mcd_peak_local_pending_key = None
        self._mcd_peak_analysis_key = self._mcd_peak_computation_key()
        cache = getattr(self, "_mcd_peak_local_fit_cache", {})
        cached = cache.get(cache_key)
        if cached is not None:
            self._mcd_peak_fit_cancel_event = None
            self.mcd_peak_analyze_btn.setEnabled(True)
            self._apply_mcd_local_fit(cached, source=source, seed_energy_ev=seed_energy_ev, locator_energy_ev=locator_energy_ev, feature_kind=feature_kind, cache_key=cache_key)
            return
        # Keep the detector catalog and measured-field controls available while
        # the selected local model runs.  The fitted result itself is cleared
        # so the result pane cannot present stale centers as current values.
        self.mcd_peak_result = None
        self.mcd_peak_channel_results = {}
        self.mcd_peak_method_results = {}
        self.mcd_peak_export_btn.setEnabled(False)
        self._populate_mcd_peak_preview_controls()
        self._select_mcd_peak_key_blocked(self._mcd_peak_local_requested_key)
        self._refresh_mcd_peak_plot()
        self.mcd_peak_analyze_btn.setEnabled(False)
        self.mcd_peak_status.setText(f"Fitting local mixed line shape near E = {seed_energy_ev:.6g} eV…")
        worker = Worker(
            _mcd_local_fit_worker,
            source,
            seed_energy_ev=seed_energy_ev,
            locator_energy_ev=locator_energy_ev,
            feature_kind=feature_kind,
            peak_id=int(self._mcd_peak_local_requested_key[1]),
            spectrum_source=spectrum_source,
            background_model=background_model,
            window_ev=window_ev,
            max_starts=150,
            cancel_event=cancel_event,
        )
        self._mcd_peak_fit_worker = worker
        self._mcd_peak_local_pending_key = cache_key
        worker.signals.result.connect(lambda results, w=worker, g=generation, key=cache_key: self._finish_mcd_local_fit(w, g, results, source, seed_energy_ev, locator_energy_ev, feature_kind, key))
        worker.signals.error.connect(lambda message, g=generation: self._mcd_local_fit_error(g, message))
        worker.signals.finished.connect(lambda w=worker, g=generation: self._mcd_local_fit_finished(w, g))
        self.thread_pool.start(worker)

    def _finish_mcd_local_fit(self, worker, generation, results, source, seed_energy_ev, locator_energy_ev, feature_kind, cache_key) -> None:
        if generation != int(getattr(self, "_mcd_peak_fit_generation", -1)) or self.loaded is None or self.loaded.mcd_result is not source:
            return
        self._mcd_peak_local_fit_cache = getattr(self, "_mcd_peak_local_fit_cache", {})
        self._mcd_peak_local_fit_cache[cache_key] = results
        self._apply_mcd_local_fit(results, source=source, seed_energy_ev=seed_energy_ev, locator_energy_ev=locator_energy_ev, feature_kind=feature_kind, cache_key=cache_key)

    def _mcd_local_fit_error(self, generation: int, message: str) -> None:
        if generation != int(getattr(self, "_mcd_peak_fit_generation", -1)):
            return
        self.mcd_peak_status.setText(f"Local fit error: {str(message).splitlines()[0]}")

    def _mcd_local_fit_finished(self, worker, generation: int) -> None:
        if generation == int(getattr(self, "_mcd_peak_fit_generation", -1)):
            self._mcd_peak_local_pending_key = None
            self._mcd_peak_fit_worker = None
            self.mcd_peak_analyze_btn.setEnabled(True)

    def _select_mcd_peak_key_blocked(self, key: tuple | None) -> None:
        if key is None or len(key) != 4:
            return
        combo = self.mcd_peak_selector_combo
        self._mcd_peak_local_applying = True
        blocked = combo.blockSignals(True)
        try:
            wanted = (str(key[0]), int(key[1]), str(key[2]), str(key[3]))
            for index in range(combo.count()):
                value = combo.itemData(index)
                if value is not None and (str(value[0]), int(value[1]), str(value[2]), str(value[3])) == wanted:
                    combo.setCurrentIndex(index)
                    break
        finally:
            combo.blockSignals(blocked)
            self._mcd_peak_local_applying = False

    def _apply_mcd_local_fit(self, results, *, source, seed_energy_ev: float, locator_energy_ev: float, feature_kind: str, cache_key) -> None:
        self.mcd_peak_method_results = {"Local mixed fit": dict(results)}
        self.mcd_peak_channel_results = dict(results)
        self.mcd_peak_result = self.mcd_peak_channel_results.get("pos")
        self._mcd_peak_selected_result_method = "Local mixed fit"
        self._mcd_peak_local_fit_key = cache_key
        self._mcd_peak_preferred_selection = None
        self._mcd_peak_local_pending_selection = getattr(self, "_mcd_peak_local_requested_key", None)
        self._populate_mcd_peak_preview_controls()
        pending = getattr(self, "_mcd_peak_local_pending_selection", None)
        if pending is not None:
            self._select_mcd_peak_key_blocked(pending)
            self._mcd_peak_local_pending_selection = None
        self._populate_mcd_peak_table()
        self._refresh_mcd_peak_plot()
        # A branch can contain a few unavailable rows while still providing a
        # useful fitted segment.  Keep those gaps in the result, but count a
        # track as usable when it has at least one tracked point.
        valid_count = sum(
            1 for analysis in results.values()
            for track in analysis.tracks
            if any(point.status == "tracked" and point.energy_ev is not None for point in track.points)
        )
        self.mcd_peak_export_btn.setEnabled(valid_count > 0)
        self.mcd_peak_status.setText(
            f"Complete: local mixed fit; Locator E0 = {locator_energy_ev:.6g} eV; "
            f"{valid_count} usable branch track(s). Fit centres are model values; raw/D2 remain locator diagnostics."
        )

    def _analyze_mcd_peak_shift_legacy(self) -> None:
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            self.mcd_peak_status.setText("Error: no MCD result is loaded."); return
        self.mcd_peak_analyze_btn.setEnabled(False); self.mcd_peak_status.setText("Analyzing reflection peaks…")
        try:
            source = self.loaded.mcd_result
            previous_selection = self.mcd_peak_selector_combo.currentData()
            preferred = None
            if previous_selection is not None:
                old_channel, old_peak_id, old_branch, old_kind = previous_selection if len(previous_selection) == 4 else (*previous_selection, "peak")
                old_analysis = self.mcd_peak_channel_results.get(old_channel)
                old_track = next((track for track in old_analysis.tracks if track.peak_id == old_peak_id and track.branch == old_branch and track.feature_kind == old_kind), None) if old_analysis else None
                if old_track is not None:
                    preferred = (
                        old_channel,
                        old_branch,
                        old_kind,
                        None if old_track.reference_energy_ev is None else float(old_track.reference_energy_ev),
                    )
            analyses = {}
            method_analyses = {}
            selected_method = self.mcd_peak_tracker_method_combo.currentText()
            for method in ("Raw spectrum", "Second derivative"):
                method_analyses[method] = {}
            for channel, field_name in (("pos", "pair_b_pos"), ("neg", "pair_b_neg")):
                field = np.asarray(getattr(source, field_name, source.pair_b), float)
                interpolated = np.asarray(getattr(source, f"pair_interpolated_{channel}", np.zeros(field.size, dtype=bool)), bool)
                effective = np.where(interpolated, np.asarray(source.pair_b, float), field)
                adapted = copy.copy(source); adapted.pair_b = effective
                for method in method_analyses:
                    method_analyses[method][channel] = analyze_peak_shift(
                        adapted, source=_mcd_fit_source_for_channel(self.mcd_peak_source_combo.currentText(), channel),
                        prominence_fraction=self.mcd_peak_prom_spin.value(), min_distance_points=self.mcd_peak_dist_spin.value(),
                        smoothing_points=self.mcd_peak_smooth_spin.value(), max_jump_ev=self.mcd_peak_jump_spin.value(), max_peaks=self.mcd_peak_max_spin.value(),
                        tracking_method=method, derivative_window_points=self.mcd_peak_deriv_window_spin.value())
            analyses = method_analyses[selected_method]
            self.mcd_peak_method_results = method_analyses
            self.mcd_peak_channel_results = analyses
            self.mcd_peak_result = analyses["pos"]
            self._mcd_peak_preferred_selection = preferred
            self._populate_mcd_peak_preview_controls()
            self._populate_mcd_peak_table()
            self._refresh_mcd_peak_plot()
            self.mcd_peak_export_btn.setEnabled(any(track.quality != BOUNDARY_UNRELIABLE for analysis in analyses.values() for track in analysis.tracks))
            refs = ", ".join(dict.fromkeys(t.reference_method for t in self.mcd_peak_result.tracks if t.quality != BOUNDARY_UNRELIABLE)) or "none"
            valid_count = sum(t.quality != BOUNDARY_UNRELIABLE for analysis in analyses.values() for t in analysis.tracks)
            method = self.mcd_peak_tracker_method_combo.currentText()
            source_label = self.mcd_peak_source_combo.currentText()
            if self.mcd_peak_result_mode_combo.currentText() == "Valley splitting":
                status = f"Complete: {valid_count} usable {method.casefold()} feature tracks from {source_label}. Valley splitting uses absolute energies."
            else:
                status = f"Complete: {valid_count} usable {method.casefold()} feature tracks from {source_label}. Reference: {refs}."
            if getattr(self, "_mcd_peak_selection_unavailable", False):
                status += " Previous selected energy was not found within 5 meV; choose a peak."
            self.mcd_peak_status.setText(status)
        except Exception as exc:
            self.mcd_peak_result = None; self.mcd_peak_channel_results = {}; self.mcd_peak_method_results = {}; self.mcd_peak_export_btn.setEnabled(False)
            self.mcd_peak_selector_combo.clear(); self.mcd_peak_field_combo.clear(); self.mcd_peak_k_combo.clear(); self.mcd_peak_kp_combo.clear()
            self.mcd_peak_table.setRowCount(0); self.mcd_valley_table.setRowCount(0)
            self.mcd_peak_status.setText(f"Error: {exc}")
            self._refresh_mcd_peak_plot()
        finally:
            self.mcd_peak_analyze_btn.setEnabled(True)

    def _reanalyze_mcd_peak_shift(self) -> None:
        if self.loaded is not None and self.loaded.mcd_result is not None:
            self._analyze_mcd_peak_shift()

    def _on_mcd_local_selection_changed(self, index: int) -> None:
        if getattr(self, "_mcd_peak_local_applying", False) or int(index) < 0 or self.mcd_peak_tracker_method_combo.currentText() != "Local mixed fit":
            return
        selected = self.mcd_peak_selector_combo.itemData(int(index))
        catalog = getattr(self, "_mcd_peak_locator_results", {})
        if selected is None or len(selected) != 4 or not catalog:
            return
        channel, peak_id, branch, feature_kind = selected
        analysis = catalog.get(str(channel))
        track = next((item for item in analysis.tracks if int(item.peak_id) == int(peak_id) and str(item.branch) == str(branch) and str(item.feature_kind) == str(feature_kind)), None) if analysis is not None else None
        if track is not None and track.reference_energy_ev is not None:
            self._mcd_peak_local_requested_key = (str(channel), int(peak_id), str(branch), str(feature_kind))
            self._request_mcd_local_fit(seed_energy_ev=float(track.reference_energy_ev), locator_energy_ev=float(track.reference_energy_ev), feature_kind=str(feature_kind))

    def _on_mcd_peak_deriv_window_changed(self) -> None:
        self._reanalyze_mcd_peak_shift()

    def _populate_mcd_peak_preview_controls(self) -> None:
        result = self.mcd_peak_result
        catalog = getattr(self, "_mcd_peak_locator_results", {}) if self.mcd_peak_tracker_method_combo.currentText() == "Local mixed fit" else self.mcd_peak_channel_results
        if not catalog:
            catalog = self.mcd_peak_channel_results
        self.mcd_peak_branch_combo.blockSignals(True)
        current = self.mcd_peak_branch_combo.currentText()
        self.mcd_peak_branch_combo.clear()
        source_labels = np.asarray(getattr(self.loaded.mcd_result, "pair_labels", []), str) if self.loaded and self.loaded.mcd_result is not None else np.array([], str)
        branches = sorted(set(source_labels.tolist())) if source_labels.size else sorted({str(point.branch) for track in result.tracks for point in track.points}) if result else []
        self.mcd_peak_branch_combo.addItems(branches)
        self.mcd_peak_branch_combo.setCurrentText(current if current in branches else (branches[0] if branches else ""))
        self.mcd_peak_branch_combo.blockSignals(False)
        self.mcd_peak_selector_combo.blockSignals(True); self.mcd_peak_selector_combo.clear()
        branch = self.mcd_peak_branch_combo.currentText()
        entries = []
        for channel, analysis in catalog.items():
            angle = float(getattr(self.loaded.mcd_result, "pos_angle" if channel == "pos" else "neg_angle", np.nan))
            for track in analysis.tracks:
                if track.branch == branch and track.quality != BOUNDARY_UNRELIABLE:
                    quality = f"{track.quality} · " if track.quality != "OK" else ""
                    kind_label = "P" if track.feature_kind == "peak" else "D"
                    if track.reference_energy_ev is None or track.reference_method == "unavailable":
                        reference = "E0 unavailable"
                        sort_energy = float("inf")
                    else:
                        method_label = "exact" if track.reference_method == "exact 0 T" else "interpolated"
                        reference = f"E0={track.reference_energy_ev:.4f} eV ({method_label})"
                        sort_energy = float(track.reference_energy_ev)
                    label = f"{quality}{format_mcd_angle(angle)}° · {kind_label}{track.peak_id} · {reference}"
                    entries.append((sort_energy, channel, track.peak_id, track.feature_kind, label, track))
        entries.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        for _, channel, peak_id, feature_kind, label, _track in entries:
            item_index = self.mcd_peak_selector_combo.count()
            self.mcd_peak_selector_combo.addItem(label, (channel, peak_id, branch, feature_kind))
            self.mcd_peak_selector_combo.setItemData(item_index, label, Qt.ToolTipRole)
        preferred = getattr(self, "_mcd_peak_preferred_selection", None)
        self._mcd_peak_selection_unavailable = False
        if preferred is not None and self.mcd_peak_selector_combo.count():
            channel, preferred_branch, preferred_kind, preferred_energy = preferred
            matches = []
            if preferred_energy is not None and np.isfinite(preferred_energy):
                for index in range(self.mcd_peak_selector_combo.count()):
                    item_channel, _, item_branch, item_kind = self.mcd_peak_selector_combo.itemData(index)
                    if item_channel != channel or item_branch != preferred_branch or item_kind != preferred_kind:
                        continue
                    track = next((item for item in catalog[channel].tracks if item.peak_id == self.mcd_peak_selector_combo.itemData(index)[1] and item.branch == item_branch and item.feature_kind == item_kind), None)
                    if track is not None and track.reference_energy_ev is not None and abs(float(track.reference_energy_ev) - preferred_energy) <= 0.005:
                        matches.append(index)
            if len(matches) == 1:
                self.mcd_peak_selector_combo.setCurrentIndex(matches[0])
            else:
                self.mcd_peak_selector_combo.setCurrentIndex(-1)
                self._mcd_peak_selection_unavailable = True
        elif self.mcd_peak_selector_combo.count() and self.loaded and self.loaded.mcd_result is not None:
            floor = float(np.nanmin(np.asarray(self.loaded.mcd_result.energy_ev, float))) + 0.01
            for index in range(self.mcd_peak_selector_combo.count()):
                channel, peak_id, peak_branch, peak_kind = self.mcd_peak_selector_combo.itemData(index)
                track = next((item for item in catalog[channel].tracks if item.peak_id == peak_id and item.branch == peak_branch and item.feature_kind == peak_kind), None)
                if track is not None and track.quality == "OK" and track.reference_energy_ev is not None and float(track.reference_energy_ev) > floor:
                    self.mcd_peak_selector_combo.setCurrentIndex(index)
                    break
        self._mcd_peak_preferred_selection = None
        self.mcd_peak_selector_combo.blockSignals(False)
        previous_field = self.mcd_peak_selected_field
        self.mcd_peak_field_combo.blockSignals(True); self.mcd_peak_field_combo.clear()
        source = self.loaded.mcd_result
        bpos_all = np.asarray(getattr(source, "pair_b_pos", source.pair_b), float)
        bneg_all = np.asarray(getattr(source, "pair_b_neg", source.pair_b), float)
        ipos_all = np.asarray(getattr(source, "pair_interpolated_pos", np.zeros(len(bpos_all), dtype=bool)), bool)
        ineg_all = np.asarray(getattr(source, "pair_interpolated_neg", np.zeros(len(bneg_all), dtype=bool)), bool)
        branch = self.mcd_peak_branch_combo.currentText()
        labels = np.asarray(getattr(source, "pair_labels", np.full(len(bpos_all), branch)), str)
        for index, (bpos, bneg) in enumerate(zip(bpos_all, bneg_all)):
            if branch and branch != "All sweep directions" and labels[index] != branch:
                continue
            pos_label = f"B+ {bpos:.6g} T" + (" (interpolated)" if ipos_all[index] else "")
            neg_label = f"B− {bneg:.6g} T" + (" (interpolated)" if ineg_all[index] else "")
            self.mcd_peak_field_combo.addItem(f"{pos_label} / {neg_label}", index)
        self.mcd_peak_field_combo.blockSignals(False)
        if self.mcd_peak_field_combo.count():
            field_values = np.asarray(source.pair_b, float)
            target = float(previous_field) if previous_field is not None and np.isfinite(previous_field) else 0.0
            candidates = [self.mcd_peak_field_combo.itemData(i) for i in range(self.mcd_peak_field_combo.count())]
            selected = min(candidates, key=lambda idx: abs(float(field_values[int(idx)]) - target))
            self.mcd_peak_field_combo.setCurrentIndex(self.mcd_peak_field_combo.findData(selected))
            self.mcd_peak_selected_field = float(source.pair_b[self.mcd_peak_field_combo.currentData()])

    def _on_mcd_peak_field_changed(self) -> None:
        index = self.mcd_peak_field_combo.currentData()
        if index is not None and self.loaded and self.loaded.mcd_result is not None:
            manual_index = getattr(self, "_mcd_peak_manual_center_field_index", None)
            if manual_index is None or int(manual_index) != int(index):
                self._clear_mcd_peak_manual_center()
            self.mcd_peak_selected_field = float(np.asarray(self.loaded.mcd_result.pair_b)[int(index)])
            self._refresh_mcd_peak_plot()

    def _on_mcd_peak_branch_changed(self) -> None:
        if self.mcd_peak_channel_results:
            self._clear_mcd_peak_manual_center()
            selected = self.mcd_peak_selector_combo.currentData()
            target_branch = self.mcd_peak_branch_combo.currentText()
            if selected is not None:
                channel, peak_id, old_branch, feature_kind = selected if len(selected) == 4 else (*selected, "peak")
                analysis = self.mcd_peak_channel_results.get(channel)
                track = next((item for item in analysis.tracks if item.peak_id == peak_id and item.branch == old_branch and item.feature_kind == feature_kind), None) if analysis else None
                self._mcd_peak_preferred_selection = (
                    channel,
                    target_branch,
                    feature_kind,
                    None if track is None or track.reference_energy_ev is None else float(track.reference_energy_ev),
                )
            self._populate_mcd_peak_preview_controls()
            self._refresh_mcd_peak_plot()

    def _populate_mcd_peak_table(self) -> None:
        result = self.mcd_peak_result
        self.mcd_peak_table.setRowCount(0)
        self.mcd_valley_table.setRowCount(0)
        if result is None: return
        visible_tracks = tuple(track for track in result.tracks if track.quality != BOUNDARY_UNRELIABLE)
        peak_tracks = tuple(track for track in visible_tracks if track.feature_kind == "peak")
        visible_result = replace(result, tracks=peak_tracks)
        ids = sorted({track.peak_id for track in peak_tracks})
        for combo in (self.mcd_peak_k_combo, self.mcd_peak_kp_combo):
            previous = combo.currentData(); combo.blockSignals(True); combo.clear()
            for value in ids: combo.addItem(f"Peak {value}", value)
            if previous in ids:
                combo.setCurrentIndex(ids.index(previous))
            elif len(ids) >= 2:
                combo.setCurrentIndex(0 if combo is self.mcd_peak_k_combo else 1)
            combo.blockSignals(False)
        for track in visible_tracks:
            for point in track.points:
                row = self.mcd_peak_table.rowCount(); self.mcd_peak_table.insertRow(row)
                reference = track.reference_method
                if track.reference_energy_ev is not None:
                    reference += f": {track.reference_energy_ev:.8g} eV"
                if track.reference_field_t is not None:
                    reference += f" at {track.reference_field_t:.6g} T"
                values = [
                    f"{'P' if track.feature_kind == 'peak' else 'D'}{track.peak_id}", f"{point.field_t:.6g}", point.branch,
                    "" if point.energy_ev is None else f"{point.energy_ev:.8g}",
                    "" if point.delta_energy_ev is None else f"{point.delta_energy_ev:.8g}",
                    point.status, reference,
                ]
                for col, value in enumerate(values): self.mcd_peak_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.mcd_peak_table.resizeColumnsToContents()
        selected_ids = (self.mcd_peak_k_combo.currentData(), self.mcd_peak_kp_combo.currentData())
        valley_rows = valley_quantities(visible_result, selected_ids) if len(ids) >= 2 else ()
        for value in valley_rows:
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
        self._update_mcd_peak_candidate_buttons()
        self._plot_mode("MCD Peak Shift")

    def _mcd_peak_candidate_track(self, key: tuple):
        if len(key) != 4:
            return None
        channel, peak_id, branch, feature_kind = key
        catalog = getattr(self, "_mcd_peak_locator_results", {}) if self.mcd_peak_tracker_method_combo.currentText() == "Local mixed fit" else self.mcd_peak_channel_results
        if not catalog:
            catalog = self.mcd_peak_channel_results
        analysis = catalog.get(str(channel))
        if analysis is None:
            return None
        return next(
            (
                track for track in analysis.tracks
                if int(track.peak_id) == int(peak_id)
                and str(track.branch) == str(branch)
                and str(track.feature_kind) == str(feature_kind)
                and track.quality != BOUNDARY_UNRELIABLE
            ),
            None,
        )

    def _update_mcd_peak_candidate_buttons(self) -> None:
        buttons = getattr(self, "mcd_peak_candidate_buttons", ())
        if not buttons:
            return
        for button in buttons:
            button.setVisible(False)
            button.setChecked(False)
        self._mcd_peak_candidate_keys = []
        selected = self.mcd_peak_selector_combo.currentData() if hasattr(self, "mcd_peak_selector_combo") else None
        catalog = getattr(self, "_mcd_peak_locator_results", {}) if self.mcd_peak_tracker_method_combo.currentText() == "Local mixed fit" else self.mcd_peak_channel_results
        if not catalog:
            catalog = self.mcd_peak_channel_results
        if selected is None or not catalog:
            return
        selected = tuple(selected)
        channel, _peak_id, branch, _feature_kind = selected
        analysis = catalog.get(str(channel))
        if analysis is None:
            return
        selected_track = self._mcd_peak_candidate_track(selected)
        selected_energy = (
            float(selected_track.reference_energy_ev)
            if selected_track is not None and selected_track.reference_energy_ev is not None and np.isfinite(selected_track.reference_energy_ev)
            else None
        )
        candidates = []
        for track in analysis.tracks:
            if str(track.branch) != str(branch) or track.quality == BOUNDARY_UNRELIABLE:
                continue
            key = (str(channel), int(track.peak_id), str(track.branch), str(track.feature_kind))
            energy = track.reference_energy_ev
            finite_energy = energy is not None and np.isfinite(energy)
            distance = abs(float(energy) - selected_energy) if finite_energy and selected_energy is not None else float("inf")
            candidates.append((0 if finite_energy else 1, distance, float(energy) if finite_energy else float("inf"), key, track))
        if len(candidates) > len(buttons) and selected_energy is not None:
            candidates.sort(key=lambda item: item[:3])
            candidates = candidates[: len(buttons)]
        candidates.sort(key=lambda item: (item[0], item[2], item[3]))
        for index, (_available, _distance, _energy, key, track) in enumerate(candidates[: len(buttons)]):
            button = buttons[index]
            kind_label = "P" if track.feature_kind == "peak" else "D"
            full_kind_label = "Peak" if track.feature_kind == "peak" else "Dip"
            if track.reference_energy_ev is None or track.reference_method == "unavailable":
                label = f"{kind_label}{track.peak_id} · E0 unavailable"
                tooltip = f"{full_kind_label} {track.peak_id}; E0 unavailable"
            else:
                label = f"{kind_label}{track.peak_id} · {float(track.reference_energy_ev):.4f} eV"
                method_label = "exact 0 T" if track.reference_method == "exact 0 T" else "interpolated"
                tooltip = f"{full_kind_label} {track.peak_id}; E0 = {float(track.reference_energy_ev):.4f} eV ({method_label})"
            button.setText(label)
            button.setToolTip(f"{tooltip}. Select this tracked reflection feature.")
            button.setVisible(True)
            button.setChecked(tuple(selected) == key)
            self._mcd_peak_candidate_keys.append(key)

    def _on_mcd_peak_candidate_clicked(self, index: int) -> None:
        if index < 0 or index >= len(getattr(self, "_mcd_peak_candidate_keys", ())):
            return
        key = self._mcd_peak_candidate_keys[index]
        self._clear_mcd_peak_manual_center()
        combo = self.mcd_peak_selector_combo
        for item_index in range(combo.count()):
            value = combo.itemData(item_index)
            if value is None or len(value) != 4:
                continue
            if (str(value[0]), int(value[1]), str(value[2]), str(value[3])) == tuple(key):
                combo.setCurrentIndex(item_index)
                self._update_mcd_peak_candidate_buttons()
                return

    def _clear_mcd_peak_manual_center(self) -> None:
        """Forget the free energy cursor and remove its transient artists."""
        self._mcd_peak_manual_center_ev = None
        self._mcd_peak_manual_center_channel = None
        self._mcd_peak_manual_center_field_index = None
        if hasattr(self, "_stop_mcd_peak_manual_center_blit"):
            self._stop_mcd_peak_manual_center_blit()
        for artist in getattr(self, "_mcd_peak_manual_center_artists", {}).values():
            try:
                artist.remove()
            except (AttributeError, ValueError):
                pass
        self._mcd_peak_manual_center_artists = {}

    def _commit_mcd_peak_feature_center(self, drag: dict) -> None:
        """Reseek once at release and merge local tracks into the view."""
        center = getattr(self, "_mcd_peak_manual_center_ev", None)
        channel = str(getattr(self, "_mcd_peak_manual_center_channel", None) or drag.get("channel", ""))
        index = getattr(self, "_mcd_peak_manual_center_field_index", None)
        source = self.loaded.mcd_result if self.loaded is not None else None
        if source is None or center is None or channel not in {"pos", "neg"} or index is None:
            return
        selected_method = self.mcd_peak_tracker_method_combo.currentText()
        branch = self.mcd_peak_branch_combo.currentText()
        selected = self.mcd_peak_selector_combo.currentData()
        selected_kind = str(selected[3]) if selected is not None and len(selected) == 4 else "peak"
        index = int(index)
        source_fields = np.asarray(source.pair_b, float)
        self.mcd_peak_status.setText(f"Searching near E = {float(center):.6g} eV…")
        if selected_method == "Local mixed fit":
            selected_id = int(selected[1]) if selected is not None and len(selected) == 4 else 1
            pending = (channel, selected_id, branch, selected_kind)
            self._mcd_peak_local_pending_selection = pending
            self._clear_mcd_peak_manual_center()
            self._request_mcd_local_fit(
                seed_energy_ev=float(center), locator_energy_ev=float(center),
                feature_kind=selected_kind, selection_key=pending,
            )
            return
        def effective_fields(target_channel: str) -> np.ndarray:
            values = np.asarray(getattr(source, f"pair_b_{target_channel}", source.pair_b), float)
            interpolated = np.asarray(
                getattr(source, f"pair_interpolated_{target_channel}", np.zeros(values.size, dtype=bool)),
                bool,
            )
            return np.where(interpolated, source_fields, values) if interpolated.size == values.size else values

        def local_by_branch(seeded: Any, target_channel: str) -> dict[str, Any]:
            fields = effective_fields(target_channel)
            if index < 0 or index >= fields.size or not np.isfinite(fields[index]):
                return {}
            target_field = float(fields[index])
            candidates: dict[str, list[tuple[float, Any]]] = {}
            for track in seeded.tracks:
                if str(track.feature_kind) != selected_kind or track.quality == BOUNDARY_UNRELIABLE:
                    continue
                points = [
                    item for item in track.points
                    if item.status == "tracked"
                    and item.energy_ev is not None
                    and np.isfinite(item.energy_ev)
                    and np.isfinite(item.field_t)
                ]
                point = min(points, key=lambda item: abs(float(item.field_t) - target_field)) if points else None
                if point is not None and abs(float(point.energy_ev) - float(center)) <= 0.005 + 1e-9:
                    candidates.setdefault(str(track.branch), []).append((abs(float(point.energy_ev) - float(center)), track))
            return {name: min(items, key=lambda item: item[0])[1] for name, items in candidates.items()}

        seeded_results: dict[tuple[str, str], dict[str, Any]] = {}
        seed_errors: list[str] = []
        for method in ("Raw spectrum", "Second derivative"):
            for target_channel in ("pos", "neg"):
                fields = effective_fields(target_channel)
                if index < 0 or index >= fields.size or not np.isfinite(fields[index]):
                    continue
                try:
                    adapted = copy.copy(source)
                    adapted.pair_b = fields
                    seeded = analyze_peak_shift(
                        adapted,
                        source=_mcd_fit_source_for_channel(self.mcd_peak_source_combo.currentText(), target_channel),
                        prominence_fraction=self.mcd_peak_prom_spin.value(),
                        min_distance_points=self.mcd_peak_dist_spin.value(),
                        smoothing_points=self.mcd_peak_smooth_spin.value(),
                        max_jump_ev=self.mcd_peak_jump_spin.value(),
                        max_peaks=self.mcd_peak_max_spin.value(),
                        tracking_method=method,
                        derivative_window_points=self.mcd_peak_deriv_window_spin.value(),
                        seed_energy_ev=float(center),
                        seed_half_width_ev=0.005,
                        seed_field_t=float(fields[index]),
                    )
                except Exception as exc:
                    seed_errors.append(f"{method} {target_channel}: {exc}")
                    continue
                seeded_results[(method, target_channel)] = local_by_branch(seeded, target_channel)

        selected_local = seeded_results.get((selected_method, channel), {}).get(branch)
        if selected_local is None:
            selected_error = next(
                (item for item in seed_errors if item.startswith(f"{selected_method} {channel}:")),
                None,
            )
            message = f"No reliable local feature near E = {float(center):.6g} eV."
            if selected_error:
                message += f" Search error: {selected_error.split(': ', 1)[1]}"
            self.mcd_peak_status.setText(message)
            self.mcd_peak_selector_combo.blockSignals(True)
            self.mcd_peak_selector_combo.setCurrentIndex(-1)
            self.mcd_peak_selector_combo.blockSignals(False)
            self._refresh_mcd_peak_plot()
            return

        chosen_keys: dict[tuple[str, str, str], tuple] = {}
        for (method, target_channel), by_branch in seeded_results.items():
            current = self.mcd_peak_method_results.setdefault(method, {}).get(target_channel)
            if current is None:
                continue
            fields = effective_fields(target_channel)
            target_field = float(fields[index])
            merged_tracks = list(current.tracks)
            next_id = max((int(track.peak_id) for track in merged_tracks), default=0) + 1
            energy_grid = np.sort(np.asarray(source.energy_ev, float))
            finite_grid = energy_grid[np.isfinite(energy_grid)]
            grid_step = float(np.nanmedian(np.diff(finite_grid))) if finite_grid.size > 1 else 0.001
            match_tolerance = min(0.003, max(0.001, 2.0 * abs(grid_step)))
            for target_branch, local_track in by_branch.items():
                matches = []
                local_points = [
                    point for point in local_track.points
                    if point.energy_ev is not None and np.isfinite(point.energy_ev) and np.isfinite(point.field_t)
                ]
                local_point = min(local_points, key=lambda point: abs(float(point.field_t) - target_field), default=None)
                for existing_index, existing in enumerate(merged_tracks):
                    if str(existing.branch) != target_branch or str(existing.feature_kind) != selected_kind:
                        continue
                    existing_points = [
                        point for point in existing.points
                        if point.energy_ev is not None and np.isfinite(point.energy_ev) and np.isfinite(point.field_t)
                    ]
                    existing_point = min(existing_points, key=lambda point: abs(float(point.field_t) - target_field), default=None)
                    ref_distance = (
                        abs(float(existing.reference_energy_ev) - float(local_track.reference_energy_ev))
                        if existing.reference_energy_ev is not None and local_track.reference_energy_ev is not None
                        else float("inf")
                    )
                    point_distance = (
                        abs(float(existing_point.energy_ev) - float(local_point.energy_ev))
                        if existing_point is not None and local_point is not None
                        else float("inf")
                    )
                    distance = min(ref_distance, point_distance)
                    if distance <= match_tolerance:
                        matches.append((distance, existing_index, ref_distance, point_distance))
                replacement_index = None
                if matches:
                    matches.sort(key=lambda item: (item[0], item[1]))
                    best_distance = matches[0][0]
                    tied = [item for item in matches if abs(item[0] - best_distance) <= 1e-9]
                    # An exact point match identifies the previously merged
                    # manual track. Otherwise a numerical tie is ambiguous;
                    # append a distinct track instead of overwriting a nearby
                    # physical resonance.
                    exact = [item for item in tied if item[3] <= 1e-9]
                    if len(exact) == 1:
                        replacement_index = exact[0][1]
                    elif len(tied) == 1:
                        replacement_index = tied[0][1]
                peak_id = int(merged_tracks[replacement_index].peak_id) if replacement_index is not None else next_id
                if replacement_index is None:
                    next_id += 1
                merged_track = replace(local_track, peak_id=peak_id)
                if replacement_index is None:
                    merged_tracks.append(merged_track)
                else:
                    merged_tracks[replacement_index] = merged_track
                if method == selected_method and target_channel == channel and target_branch == branch:
                    chosen_keys[(method, target_channel, target_branch)] = (target_channel, peak_id, target_branch, selected_kind)
            self.mcd_peak_method_results[method][target_channel] = replace(current, tracks=tuple(merged_tracks))

        self.mcd_peak_channel_results = self.mcd_peak_method_results[selected_method]
        self.mcd_peak_result = self.mcd_peak_channel_results.get("pos")
        self._populate_mcd_peak_preview_controls()
        self._populate_mcd_peak_table()
        self.mcd_peak_export_btn.setEnabled(
            any(
                track.quality != BOUNDARY_UNRELIABLE
                for analysis in self.mcd_peak_channel_results.values()
                for track in analysis.tracks
            )
        )
        key = chosen_keys.get((selected_method, channel, branch))
        if key is None:
            key = (channel, int(selected_local.peak_id), branch, selected_kind)
        combo_index = -1
        for item_index in range(self.mcd_peak_selector_combo.count()):
            value = self.mcd_peak_selector_combo.itemData(item_index)
            if value is not None and len(value) == 4 and tuple(value) == tuple(key):
                combo_index = item_index
                break
        if combo_index >= 0:
            self.mcd_peak_selector_combo.setCurrentIndex(combo_index)
        else:
            self.mcd_peak_selector_combo.setCurrentIndex(-1)
        status = (
            f"Center E = {float(center):.6g} eV · local {selected_kind} tracked; "
            "tracked E remains separate from the manual center."
        )
        if seed_errors:
            status += " Some paired searches failed: " + "; ".join(seed_errors)
        self.mcd_peak_status.setText(status)
        self._refresh_mcd_peak_plot()

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
                "peak_id", "feature_kind", "B_T", "branch", "E_peak_eV", "delta_E_eV", "status",
                "reference_method", "reference_field_T", "selected_K_peak_id",
                "selected_Kp_peak_id", "E_K_eV", "E_Kp_eV", "delta_E_K_eV",
                "delta_E_Kp_eV", "delta_E_Kp_minus_K_eV", "average_E_eV",
                "odd_average_E_eV", "even_average_E_eV", "odd_splitting_eV",
                "even_splitting_eV", "channel", "fit_source", "fit_window_low_eV",
                "fit_window_high_eV", "background_model", "locator_energy_eV",
                "model_name", "valley_status",
            ])
            if self.mcd_peak_tracker_method_combo.currentText() == "Local mixed fit":
                selected = self.mcd_peak_selector_combo.currentData()
                if selected is None or len(selected) != 4:
                    self.mcd_peak_status.setText("Local fit export needs a selected resonance.")
                    return
                channel, peak_id, branch, feature_kind = selected
                local_analysis = self.mcd_peak_method_results.get("Local mixed fit", {})
                selected_analysis = local_analysis.get(str(channel))
                local_track = next((track for track in selected_analysis.tracks if track.peak_id == int(peak_id) and track.branch == str(branch) and track.feature_kind == str(feature_kind)), None) if selected_analysis is not None else None
                target_energy = local_track.reference_energy_ev if local_track is not None else None
                if target_energy is None:
                    self.mcd_peak_status.setText("Local fit export needs a fitted E0 reference.")
                    return
                # Use the same per-branch valley routine as the page.  Fix the
                # positive-field K/K' ordering once, then apply it to every
                # branch so exported splitting values match the displayed plot.
                branch_names = [str(value) for value in dict.fromkeys(np.asarray(self.loaded.mcd_result.pair_labels, str).tolist())]
                splits = {
                    direction: compute_valley_splitting(
                        self.mcd_peak_method_results, method="Local mixed fit", selected_channel=str(channel),
                        branch=direction, target_energy_ev=float(target_energy), tolerance_ev=0.005,
                        feature_kind=str(feature_kind), allow_energy_fallback=False,
                    )
                    for direction in branch_names
                }
                fixed_k = next((item.k_channel for item in splits.values() if item.k_channel in {"pos", "neg"}), None)
                if fixed_k in {"pos", "neg"}:
                    splits = {
                        direction: compute_valley_splitting(
                            self.mcd_peak_method_results, method="Local mixed fit", selected_channel=str(channel),
                            branch=direction, target_energy_ev=float(target_energy), tolerance_ev=0.005,
                            fixed_k_channel=fixed_k, feature_kind=str(feature_kind), allow_energy_fallback=False,
                        )
                        for direction in branch_names
                    }
                first_analysis = next(iter(local_analysis.values()), None)
                fit_window = getattr(first_analysis, "fit_window_ev", None)
                window_low = fit_window[0] if fit_window else None
                window_high = fit_window[1] if fit_window else None
                background_model = str(self.mcd_peak_background_combo.currentText()).replace(" background", "").casefold()
                model_name = getattr(first_analysis, "model_name", None)
                for local_channel, analysis in local_analysis.items():
                    for track in analysis.tracks:
                        for point in track.points:
                            split = splits.get(str(point.branch))
                            split_point = next((item for item in split.points if abs(float(item.field_t) - float(point.field_t)) <= 1e-12), None) if split is not None else None
                            writer.writerow([
                                track.peak_id, track.feature_kind, point.field_t, point.branch,
                                point.energy_ev, point.delta_energy_ev, point.status,
                                track.reference_method, track.reference_field_t,
                                self.mcd_peak_k_combo.currentData(), self.mcd_peak_kp_combo.currentData(),
                                None, None, None, None,
                                None if split_point is None else split_point.splitting_ev,
                                None, None, None, None, None,
                                local_channel, self.mcd_peak_source_combo.currentText(), window_low,
                                window_high, background_model, track.locator_energy_ev,
                                track.model_name or model_name,
                                None if split is None else split.status,
                            ])
                self.mcd_peak_status.setText(f"Exported {path}")
                return
            visible_result = replace(self.mcd_peak_result, tracks=tuple(track for track in self.mcd_peak_result.tracks if track.quality != BOUNDARY_UNRELIABLE))
            selected_ids = (self.mcd_peak_k_combo.currentData(), self.mcd_peak_kp_combo.currentData())
            valleys = {(round(float(value["B_T"]), 9), str(value["branch"])): value for value in valley_quantities(visible_result, selected_ids)} if all(value is not None for value in selected_ids) else {}
            for track in self.mcd_peak_result.tracks:
                if track.quality == BOUNDARY_UNRELIABLE:
                    continue
                for point in track.points:
                    valley = valleys.get((round(point.field_t, 9), point.branch), {}) if track.feature_kind == "peak" else {}
                    writer.writerow([
                        track.peak_id, track.feature_kind, point.field_t, point.branch, point.energy_ev,
                        point.delta_energy_ev, point.status, track.reference_method,
                        track.reference_field_t, self.mcd_peak_k_combo.currentData(),
                        self.mcd_peak_kp_combo.currentData(), valley.get("E_K"),
                        valley.get("E_Kp"), valley.get("delta_E_K"),
                        valley.get("delta_E_Kp"), valley.get("splitting_E_Kp_minus_E_K"),
                        valley.get("average_E"), valley.get("odd_average_E"),
                        valley.get("even_average_E"), valley.get("odd_splitting"),
                        valley.get("even_splitting"), "", "", "", "", "", "", "", "",
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
        self.shg_background_combo.setMinimumWidth(124)
        self.shg_background_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
            spin.setMinimumWidth(128)
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        self.shg_sigma_clip_spin.setMinimumWidth(88)
        self.shg_sigma_clip_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        def value_row(spin: QDoubleSpinBox, suffix: str = "nm") -> QWidget:
            widget = QWidget()
            row_layout = QVBoxLayout(widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(3)
            row_layout.addWidget(spin)
            row_layout.addWidget(QLabel(suffix))
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
        angle_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
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
        for spin in (self.shg_angle_scale_spin, self.shg_angle_offset_spin):
            spin.setMinimumWidth(180)
            spin.setMaximumWidth(180)
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        fit_range_layout = QVBoxLayout(fit_range_row)
        fit_range_layout.setContentsMargins(0, 0, 0, 0)
        fit_range_layout.setSpacing(3)
        for spin in (self.shg_fit_min_spin, self.shg_fit_max_spin):
            spin.setMinimumWidth(128)
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        fit_range_layout.addWidget(QLabel("Minimum"))
        fit_range_layout.addWidget(self.shg_fit_min_spin)
        fit_range_layout.addWidget(QLabel("Maximum"))
        fit_range_layout.addWidget(self.shg_fit_max_spin)
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
