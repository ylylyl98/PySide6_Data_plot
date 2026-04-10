from __future__ import annotations

import traceback
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QDialog,
    QDialogButtonBox,
    QProgressBar,
    QSpinBox,
    QStyle,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QPlainTextEdit,
    QToolButton,
    QFrame,
    QVBoxLayout,
    QWidget,
)
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from core import data_io
from core.export_legacy import (
    build_drr_export_base,
    export_compare_panels,
    export_drr_png_and_dat,
    export_pl_pngs_and_dat,
)
from core.loader import DataCube
from core.plotting import COMPARE_PANEL_ORDER, HeatmapParams, plot_compare_panel, plot_drr, plot_pl, render_compare_grid
from core.processing import (
    apply_sg_derivative_energy,
    clamp_sg_window,
    compute_auto_limits,
    group_measurement_files,
    nearest_gate_spectrum,
)

UI_METRICS = {
    "left_max_width": 520,
    "main_margin": 10,
    "group_margin": 8,
    "row_spacing": 6,
    "label_col_width": 82,
    "input_h": 29,
    "spin_w": 88,
    "short_combo_w": 150,
    "deriv_combo_w": 90,
    "long_combo_min_w": 210,
    "tool_h": 26,
    "tool_w": 60,
}


@dataclass
class LoadedState:
    mode: str
    folder: str
    primary_file: str | None = None
    selected_files: List[str] = field(default_factory=list)
    baseline_files: List[str] = field(default_factory=list)
    cube: DataCube | None = None
    compare_cubes: Dict[str, DataCube] | None = None
    compare_sources: Dict[str, str] = field(default_factory=dict)
    drr_mode_label: str = "DR/R Self"
    drr_derivative_label: str = "None"
    drr_baseline_text: str = "Self (last frame)"
    drr_baseline_which: str = "last"
    y_axis_spec: str = "auto"


@dataclass(frozen=True)
class LoadOptions:
    mode: str
    folder: str
    selected_files: List[str]
    baseline_files: List[str]
    pl_log_scale: bool
    drr_baseline_text: str
    drr_baseline_which: str
    compare_log_scale: bool
    y_axis_spec: str = "auto"
    compare_sources: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportOptions:
    mode: str
    params: HeatmapParams
    params_linear: HeatmapParams | None = None
    params_log: HeatmapParams | None = None
    drr_cube: DataCube | None = None
    drr_derivative_label: str = "None"
    compare_scale_tag: str = "linear"
    compare_clip: bool = True
    compare_gate: float = 0.0
    auto_move_sources: bool = False


class WorkerSignals(QObject):
    progress = Signal(int)
    log = Signal(str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class Worker(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.fn(*self.args, progress=self.signals.progress, log=self.signals.log, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(f"{exc}\n\n{traceback.format_exc()}")
        finally:
            self.signals.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DPTK Desktop (PySide6)")
        self.setMinimumSize(1100, 700)
        self.thread_pool = QThreadPool.globalInstance()
        self.current_folder = ""
        self.available_files: List[str] = []
        self.drr_selected_files: List[str] = []
        self.drr_baseline_files_manual: List[str] = []
        self.drr_baseline_files_found: List[str] = []
        self.loaded: LoadedState | None = None
        self.last_plotted_mode: str | None = None
        self.log_lines: deque[str] = deque(maxlen=300)
        self._last_plot_params_key: tuple[Any, ...] | None = None
        self._last_plot_cube: DataCube | None = None
        self._drr_heatmap_ax = None
        self._drr_spectrum_ax = None
        self._pl_heatmap_ax = None
        self._pl_spectrum_ax = None
        self._cmp_heatmap_axes: dict[str, Any] = {}
        self._cmp_gate_lines: dict[str, Any] = {}
        self._cmp_linecut_ax = None
        self._pl_last_plot_cube: DataCube | None = None
        self._gate_line = None
        self._pl_gate_line = None
        self._gate_motion_cid: int | None = None
        self._gate_click_cid: int | None = None
        self._suspend_drr_autoplot = False
        self._pl_peak_gate: float | None = None
        self._pl_peak_indices: np.ndarray | None = None
        self._pl_fit_gate: float | None = None
        self._pl_fit_x: np.ndarray | None = None
        self._pl_fit_y: np.ndarray | None = None
        self._pl_fit_centers: np.ndarray | None = None
        self._pl_heatmap_peak_artist = None
        self._pl_heatmap_fit_artist = None
        self._drr_peak_gate: float | None = None
        self._drr_peak_indices: np.ndarray | None = None
        self._drr_fit_gate: float | None = None
        self._drr_fit_x: np.ndarray | None = None
        self._drr_fit_y: np.ndarray | None = None
        self._drr_fit_centers: np.ndarray | None = None
        self._drr_heatmap_peak_artist = None
        self._drr_heatmap_fit_artist = None

        self._build_ui()
        self.apply_ui_metrics()
        self._wire_actions()
        self._apply_initial_geometry()
        self._set_stage("No data")
        self._update_action_states()

    def _apply_initial_geometry(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1320, 820)
            return
        available = screen.availableGeometry()
        width = min(available.width(), max(self.minimumWidth(), int(available.width() * 0.88)))
        height = min(available.height(), max(self.minimumHeight(), int(available.height() * 0.88)))
        self.resize(width, height)
        self.move(
            available.x() + (available.width() - width) // 2,
            available.y() + (available.height() - height) // 2,
        )

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Horizontal)
        left = self._build_left_panel()
        right = self._build_plot_panel()
        left.setMaximumWidth(UI_METRICS["left_max_width"])
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([UI_METRICS["left_max_width"] - 20, 980])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(
            UI_METRICS["main_margin"],
            UI_METRICS["main_margin"],
            UI_METRICS["main_margin"],
            UI_METRICS["main_margin"],
        )
        layout.addWidget(splitter)

        sb = QStatusBar()
        self._status_progress = QProgressBar()
        self._status_progress.setRange(0, 0)  # indeterminate spinner
        self._status_progress.setMaximumWidth(120)
        self._status_progress.setVisible(False)
        self._status_progress.setToolTip("Loading data in background…")
        sb.addPermanentWidget(self._status_progress)
        self.setStatusBar(sb)
        self.statusBar().showMessage("Ready")
        self._build_log_dock()
        self._build_menu_and_toolbar()

    def _build_left_panel(self) -> QWidget:
        box = QWidget()
        box.setMaximumWidth(UI_METRICS["left_max_width"])
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(UI_METRICS["row_spacing"])

        # Workflow step banner
        steps_label = QLabel("Select  ›  Load  ›  Plot  ›  Export")
        steps_label.setAlignment(Qt.AlignCenter)
        steps_label.setStyleSheet(
            "QLabel { background: #deeaf8; color: #1a4a88; border-radius: 5px; "
            "padding: 5px 10px; font-size: 11px; font-weight: 600; "
            "border: 1px solid #bcd0ec; letter-spacing: 0.3px; }"
        )
        layout.addWidget(steps_label)

        # Data source section
        folder_box = QGroupBox("")
        folder_grid = QGridLayout(folder_box)
        folder_grid.setContentsMargins(6, 4, 6, 6)
        folder_grid.setHorizontalSpacing(6)
        folder_grid.setVerticalSpacing(6)
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        self.folder_edit.setPlaceholderText("No folder selected — click Browse or Open File")
        self.folder_edit.setToolTip("Current working folder for data files")
        self.browse_btn = QPushButton("Browse Folder")
        self.browse_btn.setToolTip("Select a folder containing CSV data files")
        self.open_file_btn = QPushButton("Open File")
        self.open_file_btn.setToolTip("Open a single CSV file and set its folder as the working directory")
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Re-scan the current folder for new or changed CSV files")
        folder_grid.addWidget(self.folder_edit, 0, 0, 1, 3)
        folder_grid.addWidget(self.browse_btn, 1, 0)
        folder_grid.addWidget(self.open_file_btn, 1, 1)
        folder_grid.addWidget(self.refresh_btn, 1, 2)
        layout.addWidget(self._make_expander("Data Source", folder_box, expanded=True))

        self.tabs = QTabWidget()
        # Keep DRR tab non-scrollable: do not wrap this panel or parameters in QScrollArea.
        self.tabs.addTab(self._build_pl_tab(), "PL")
        self.tabs.addTab(self._build_drr_tab(), "DRR")
        self.tabs.addTab(self._build_compare_tab(), "Compare")
        self.tabs.addTab(self._build_tools_tab(), "Log / Tools")
        layout.addWidget(self.tabs, 1)
        return box

    def _build_common_range_grid(
        self, prefix: str
    ) -> tuple[QGridLayout, Dict[str, QDoubleSpinBox], QCheckBox, QCheckBox, QComboBox, Dict[str, QCheckBox]]:
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        def spin() -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setDecimals(4)
            s.setRange(-1e12, 1e12)
            s.setSingleStep(0.1)
            s.setMaximumWidth(130)
            return s

        spins = {
            "vmin": spin(),
            "vmax": spin(),
            "xmin": spin(),
            "xmax": spin(),
            "ymin": spin(),
            "ymax": spin(),
            "gate": spin(),
        }
        fix_checks = {
            "vmin": QCheckBox("F"),
            "vmax": QCheckBox("F"),
            "xmin": QCheckBox("F"),
            "xmax": QCheckBox("F"),
            "ymin": QCheckBox("F"),
            "ymax": QCheckBox("F"),
        }
        spins["vmin"].setToolTip("Lower color scale bound")
        spins["vmax"].setToolTip("Upper color scale bound")
        spins["xmin"].setToolTip("Left energy axis bound")
        spins["xmax"].setToolTip("Right energy axis bound")
        spins["ymin"].setToolTip("Lower gate axis bound")
        spins["ymax"].setToolTip("Upper gate axis bound")
        spins["gate"].setToolTip("Gate value for spectrum extraction")
        fix_checks["vmin"].setToolTip("Fix vmin when loading/updating files")
        fix_checks["vmax"].setToolTip("Fix vmax when loading/updating files")
        fix_checks["xmin"].setToolTip("Fix xmin when loading/updating files")
        fix_checks["xmax"].setToolTip("Fix xmax when loading/updating files")
        fix_checks["ymin"].setToolTip("Fix ymin when loading/updating files")
        fix_checks["ymax"].setToolTip("Fix ymax when loading/updating files")

        def add_pair(row: int, col: int, text: str, widget: QWidget) -> None:
            label = QLabel(text)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setFixedWidth(50)
            base = col * 2
            grid.addWidget(label, row, base)
            grid.addWidget(widget, row, base + 1)

        def spin_with_fix(k: str) -> QWidget:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            h.addWidget(spins[k])
            h.addWidget(fix_checks[k])
            h.addStretch(1)
            return row

        add_pair(0, 0, "vmin", spin_with_fix("vmin"))
        add_pair(0, 1, "vmax", spin_with_fix("vmax"))
        add_pair(1, 0, "xmin", spin_with_fix("xmin"))
        add_pair(1, 1, "xmax", spin_with_fix("xmax"))
        add_pair(2, 0, "ymin", spin_with_fix("ymin"))
        add_pair(2, 1, "ymax", spin_with_fix("ymax"))
        add_pair(3, 0, "gate", spins["gate"])

        log_chk = QCheckBox("Log Scale")
        clip_chk = QCheckBox("Clip Outliers")
        clip_chk.setChecked(True)
        log_chk.setToolTip("Use logarithmic color normalization")
        clip_chk.setToolTip("Clip values above vmax for cleaner contrast")
        cmap = QComboBox()
        cmap.addItems(["turbo", "viridis", "plasma", "inferno", "magma", "cividis", "RdBu_r"])
        cmap.setToolTip("Colormap for heatmap rendering")
        self._style_combo_popup(cmap)

        flags = QWidget()
        flags_layout = QHBoxLayout(flags)
        flags_layout.setContentsMargins(0, 0, 0, 0)
        flags_layout.setSpacing(10)
        flags_layout.addWidget(log_chk)
        flags_layout.addWidget(clip_chk)
        flags_layout.addStretch(1)
        grid.addWidget(flags, 3, 2, 1, 2)

        setattr(self, f"{prefix}_spins", spins)
        setattr(self, f"{prefix}_log_chk", log_chk)
        setattr(self, f"{prefix}_clip_chk", clip_chk)
        setattr(self, f"{prefix}_cmap", cmap)
        setattr(self, f"{prefix}_fix_checks", fix_checks)
        return grid, spins, log_chk, clip_chk, cmap, fix_checks

    def _build_y_axis_controls(self, prefix: str) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        combo = QComboBox()
        combo.addItems(["Auto / Default", "TG", "BG", "Bias", "Advanced..."])
        combo.setToolTip("Choose how the plot y-axis is derived from gate variables.")
        self._style_combo_popup(combo)

        advanced = QGroupBox("Advanced Linear Combination")
        advanced_form = QFormLayout(advanced)
        advanced_form.setContentsMargins(6, 6, 6, 6)
        advanced_form.setHorizontalSpacing(6)
        advanced_form.setVerticalSpacing(4)

        def coeff_spin(default_value: float) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1e6, 1e6)
            spin.setSingleStep(0.1)
            spin.setValue(default_value)
            spin.setFixedWidth(100)
            return spin

        a_spin = coeff_spin(1.0)
        b_spin = coeff_spin(-1.0)
        c_spin = coeff_spin(0.0)
        equation = QLabel("")
        label_preview = QLabel("")
        equation.setWordWrap(True)
        label_preview.setWordWrap(True)

        advanced_form.addRow("a (TG)", a_spin)
        advanced_form.addRow("b (BG)", b_spin)
        advanced_form.addRow("c", c_spin)
        advanced_form.addRow("Equation", equation)
        advanced_form.addRow("Label", label_preview)
        advanced.setVisible(False)

        layout.addWidget(combo)
        layout.addWidget(advanced)

        setattr(self, f"{prefix}_yaxis_combo", combo)
        setattr(self, f"{prefix}_yaxis_advanced_box", advanced)
        setattr(self, f"{prefix}_yaxis_a_spin", a_spin)
        setattr(self, f"{prefix}_yaxis_b_spin", b_spin)
        setattr(self, f"{prefix}_yaxis_c_spin", c_spin)
        setattr(self, f"{prefix}_yaxis_equation_lbl", equation)
        setattr(self, f"{prefix}_yaxis_label_lbl", label_preview)
        return host

    def _build_pl_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        files = QGroupBox("File")
        files_layout = QVBoxLayout(files)
        files_layout.setContentsMargins(6, 6, 6, 6)
        self.pl_files = QListWidget()
        self.pl_files.setSelectionMode(QAbstractItemView.SingleSelection)
        self.pl_files.setMinimumHeight(60)
        self.pl_files.setMaximumHeight(120)
        self.pl_files.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.pl_files.setToolTip("Select a PL measurement file to load")
        files_layout.addWidget(self.pl_files)
        layout.addWidget(self._make_expander("Measurement File", files, expanded=True))

        params = QGroupBox("Plot Options")
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(6, 6, 6, 4)
        params_layout.setSpacing(4)
        cfg = QFormLayout()
        cfg.setHorizontalSpacing(6)
        cfg.setVerticalSpacing(4)
        _grid, spins, _, _, cmap, fix_checks = self._build_common_range_grid("pl")
        cmap.setCurrentText("turbo")
        self.pl_yaxis_controls = self._build_y_axis_controls("pl")
        _pl_yc_row = QWidget()
        _pl_yc_h = QHBoxLayout(_pl_yc_row)
        _pl_yc_h.setContentsMargins(0, 0, 0, 0)
        _pl_yc_h.setSpacing(6)
        _pl_yc_h.addWidget(self.pl_yaxis_combo, 1)
        _pl_yc_h.addWidget(QLabel("Cmap"))
        _pl_yc_h.addWidget(cmap)
        cfg.addRow("Y-axis / Cmap", _pl_yc_row)
        cfg.addRow("", self.pl_yaxis_advanced_box)
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

        params_layout.addWidget(basic)

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
        self.pl_fit_status.setStyleSheet("QLabel { color: #2a6090; font-size: 10px; }")
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
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        files = QGroupBox("Files")
        files_layout = QVBoxLayout(files)
        files_layout.setContentsMargins(6, 6, 6, 6)
        files_layout.setSpacing(6)

        meas_row = QWidget()
        meas_h = QHBoxLayout(meas_row)
        meas_h.setContentsMargins(0, 0, 0, 0)
        meas_h.setSpacing(6)
        self.drr_measurement_summary = QLabel("Measurement: 0 files")
        self.drr_edit_measurements_btn = QPushButton("Select...")
        self.drr_edit_measurements_btn.setFixedHeight(30)
        self.drr_edit_measurements_btn.setMaximumWidth(92)
        self.drr_clear_measurements_btn = QPushButton("Clear")
        self.drr_clear_measurements_btn.setFixedHeight(30)
        self.drr_clear_measurements_btn.setMaximumWidth(72)
        meas_h.addWidget(self.drr_measurement_summary, 1)
        meas_h.addWidget(self.drr_edit_measurements_btn)
        meas_h.addWidget(self.drr_clear_measurements_btn)
        files_layout.addWidget(meas_row)

        base_row = QWidget()
        base_h = QHBoxLayout(base_row)
        base_h.setContentsMargins(0, 0, 0, 0)
        base_h.setSpacing(6)
        self.drr_baseline_summary = QLabel("Baselines: 0 files")
        self.drr_edit_baselines_btn = QPushButton("Select...")
        self.drr_edit_baselines_btn.setFixedHeight(30)
        self.drr_edit_baselines_btn.setMaximumWidth(92)
        self.drr_baseline_autofind_btn = QPushButton("Auto Find...")
        self.drr_baseline_autofind_btn.setFixedHeight(30)
        self.drr_baseline_autofind_btn.setMaximumWidth(104)
        base_h.addWidget(self.drr_baseline_summary, 1)
        base_h.addWidget(self.drr_edit_baselines_btn)
        base_h.addWidget(self.drr_baseline_autofind_btn)
        files_layout.addWidget(base_row)

        self.drr_baseline_combine_combo = QComboBox()
        self.drr_baseline_combine_combo.addItems(
            [
                "Last frame of each baseline file",
                "First frame of each baseline file",
                "Average all baseline files",
            ]
        )
        self.drr_baseline_combine_combo.setMaximumWidth(320)
        self._style_combo_popup(self.drr_baseline_combine_combo)
        files_layout.addWidget(self.drr_baseline_combine_combo)
        layout.addWidget(self._make_expander("Measurement + Baseline Files", files, expanded=True))

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
        _grid, spins, log_chk, clip_chk, cmap, fix_checks = self._build_common_range_grid("drr")

        for s in spins.values():
            s.setFixedWidth(UI_METRICS["spin_w"])
            s.setFixedHeight(UI_METRICS["input_h"])
        self.drr_baseline_combo.setMinimumWidth(150)
        self.drr_baseline_combo.setMaximumWidth(210)
        self.drr_baseline_combo.setFixedHeight(UI_METRICS["input_h"])
        self.drr_baseline_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.drr_yaxis_controls = self._build_y_axis_controls("drr")
        cmap.setMinimumWidth(110)
        cmap.setMaximumWidth(145)
        cmap.setFixedHeight(UI_METRICS["input_h"])
        cmap.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.drr_derivative_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.drr_derivative_combo.setMinimumContentsLength(3)
        self.drr_derivative_combo.setFixedWidth(UI_METRICS["deriv_combo_w"])
        self.drr_derivative_combo.setFixedHeight(UI_METRICS["input_h"])
        cmap.setCurrentText("RdBu_r")
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
        baseline_cmap_row = QWidget()
        baseline_cmap_h = QHBoxLayout(baseline_cmap_row)
        baseline_cmap_h.setContentsMargins(0, 0, 0, 0)
        baseline_cmap_h.setSpacing(6)
        baseline_cmap_h.addWidget(self.drr_baseline_combo, 1)
        baseline_cmap_h.addWidget(QLabel("Colormap"))
        baseline_cmap_h.addWidget(cmap)
        basic_form.addRow("DRR Baseline", baseline_cmap_row)
        basic_form.addRow("Y-axis", self.drr_yaxis_combo)
        basic_form.addRow("", self.drr_yaxis_advanced_box)
        basic_form.addRow("Derivative / SG", deriv_row)
        basic_form.addRow(
            "vmin / vmax",
            self._make_axis_range_row(spins["vmin"], spins["vmax"], fix_checks["vmin"], fix_checks["vmax"], self.drr_auto_v_btn, "Auto V"),
        )
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
        self.drr_fit_status.setStyleSheet("QLabel { color: #2a6090; font-size: 10px; }")
        params_layout.addWidget(basic)
        layout.addWidget(self._make_expander("Parameters", params, expanded=True))
        layout.addWidget(self._make_expander("Spectrum Analysis", analysis_box, expanded=False))
        layout.addStretch(1)
        return tab

    def _set_form_label_width(self, form: QFormLayout, width: int) -> None:
        for row in range(form.rowCount()):
            item = form.itemAt(row, QFormLayout.LabelRole)
            if item and isinstance(item.widget(), QLabel):
                lbl = item.widget()
                lbl.setFixedWidth(width)
                lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def _make_axis_range_row(
        self,
        a: QDoubleSpinBox,
        b: QDoubleSpinBox,
        fa: QCheckBox,
        fb: QCheckBox,
        auto_btn: QToolButton,
        auto_text: str,
    ) -> QWidget:
        """Build a min/max spin pair with Fix checkboxes and an Auto button."""
        auto_btn.setText(auto_text)
        auto_btn.setAutoRaise(True)
        auto_btn.setFixedWidth(UI_METRICS["tool_w"])
        auto_btn.setFixedHeight(UI_METRICS["tool_h"])
        auto_btn.setToolTip(f"Set {auto_text.replace('Auto ', '')} bounds automatically from loaded data")
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        h.addWidget(a)
        h.addWidget(fa)
        h.addWidget(b)
        h.addWidget(fb)
        h.addWidget(auto_btn)
        h.addStretch(1)
        return row

    def _style_combo_popup(self, combo: QComboBox) -> None:
        view = combo.view()
        view.setStyleSheet(
            "QListView::item:selected { background-color: #2d6cdf; color: #ffffff; }"
            "QListView { selection-background-color: #2d6cdf; selection-color: #ffffff; }"
        )

    def _format_axis_coeff(self, value: float) -> str:
        if not np.isfinite(value):
            raise ValueError("Coefficient must be finite.")
        rounded = round(float(value), 12)
        if abs(rounded - round(rounded)) < 1e-12:
            return str(int(round(rounded)))
        return format(rounded, ".12g")

    def _manual_y_axis_label(self, a: float, b: float, c: float) -> str:
        if not all(np.isfinite(v) for v in (a, b, c)):
            raise ValueError("Coefficients must be finite.")
        if abs(c) < 1e-12:
            if abs(a - 1.0) < 1e-12 and abs(b) < 1e-12:
                return "TG (V)"
            if abs(b - 1.0) < 1e-12 and abs(a) < 1e-12:
                return "BG (V)"
            if abs(b + 1.0) < 1e-12 and abs(a - 1.0) < 1e-12:
                return "TG-BG (V)"
            if abs(b + 1.0) < 1e-12:
                ratio = self._format_axis_coeff(a)
                return f"{ratio}TG-BG (V)" if ratio != "1" else "TG-BG (V)"
        terms: list[str] = []
        for coeff, symbol in ((a, "TG"), (b, "BG")):
            if abs(coeff) < 1e-12:
                continue
            sign = "-" if coeff < 0 else "+"
            mag = abs(float(coeff))
            piece = symbol if abs(mag - 1.0) < 1e-12 else f"{self._format_axis_coeff(mag)}*{symbol}"
            if not terms:
                terms.append(piece if sign == "+" else f"-{piece}")
            else:
                terms.append(f"{sign}{piece}")
        if abs(c) >= 1e-12:
            c_piece = self._format_axis_coeff(abs(float(c)))
            sign = "-" if c < 0 else "+"
            if not terms:
                terms.append(c_piece if sign == "+" else f"-{c_piece}")
            else:
                terms.append(f"{sign}{c_piece}")
        return f"y = {''.join(terms)} (V)"

    def _update_y_axis_controls(self, prefix: str) -> None:
        combo: QComboBox = getattr(self, f"{prefix}_yaxis_combo")
        advanced_box: QGroupBox = getattr(self, f"{prefix}_yaxis_advanced_box")
        a_spin: QDoubleSpinBox = getattr(self, f"{prefix}_yaxis_a_spin")
        b_spin: QDoubleSpinBox = getattr(self, f"{prefix}_yaxis_b_spin")
        c_spin: QDoubleSpinBox = getattr(self, f"{prefix}_yaxis_c_spin")
        equation_lbl: QLabel = getattr(self, f"{prefix}_yaxis_equation_lbl")
        label_lbl: QLabel = getattr(self, f"{prefix}_yaxis_label_lbl")
        advanced_on = combo.currentText() == "Advanced..."
        advanced_box.setVisible(advanced_on)
        a = float(a_spin.value())
        b = float(b_spin.value())
        c = float(c_spin.value())
        equation_lbl.setText(
            f"y = {self._format_axis_coeff(a)}*TG + {self._format_axis_coeff(b)}*BG + {self._format_axis_coeff(c)}"
        )
        label_lbl.setText(self._manual_y_axis_label(a, b, c))

    def _selected_y_axis_spec(self, prefix: str) -> str:
        combo: QComboBox = getattr(self, f"{prefix}_yaxis_combo")
        text = combo.currentText()
        if text == "Auto / Default":
            return "auto"
        if text == "TG":
            return "tg"
        if text == "BG":
            return "bg"
        if text == "Bias":
            return "bias"
        if text == "Advanced...":
            a = float(getattr(self, f"{prefix}_yaxis_a_spin").value())
            b = float(getattr(self, f"{prefix}_yaxis_b_spin").value())
            c = float(getattr(self, f"{prefix}_yaxis_c_spin").value())
            if not all(np.isfinite(v) for v in (a, b, c)):
                raise ValueError("Manual linear-combination coefficients must be finite.")
            return f"linear:{self._format_axis_coeff(a)},{self._format_axis_coeff(b)},{self._format_axis_coeff(c)}"
        raise ValueError(f"Unknown y-axis selection: {text}")

    def _make_expander(self, title: str, content: QWidget, *, expanded: bool = True) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 2, 0, 2)
        v.setSpacing(3)
        head = QToolButton()
        head.setCheckable(True)
        head.setChecked(bool(expanded))
        head.setAutoRaise(True)
        head.setToolButtonStyle(Qt.ToolButtonTextOnly)
        head.setStyleSheet(
            "QToolButton { border: none; padding: 2px 4px; font-weight: 600; "
            "color: #2a3a50; font-size: 11px; text-align: left; }"
            "QToolButton:hover { color: #1a5090; }"
        )
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Plain)
        line.setStyleSheet("QFrame { color: #ccd6e2; }")
        row = QWidget()
        row_h = QHBoxLayout(row)
        row_h.setContentsMargins(0, 0, 0, 0)
        row_h.setSpacing(6)
        row_h.addWidget(head)
        row_h.addWidget(line, 1)
        v.addWidget(row)
        content.setVisible(bool(expanded))
        v.addWidget(content)

        def _update(on: bool) -> None:
            arrow = "▼" if on else "▶"
            head.setText(f"{arrow}  {title}")
            content.setVisible(bool(on))

        _update(bool(expanded))
        head.toggled.connect(_update)
        return box

    def _build_compare_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        files = QGroupBox("File")
        files_layout = QVBoxLayout(files)
        files_layout.setContentsMargins(6, 6, 6, 6)
        self.cmp_files = QListWidget()
        self.cmp_files.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.cmp_files.setMinimumHeight(70)
        self.cmp_files.setToolTip("Select files to compare (multi-select with Ctrl/Shift)")
        files_layout.addWidget(self.cmp_files)
        layout.addWidget(self._make_expander("Available Compare Files", files, expanded=True))

        params = QGroupBox("Plot Options")
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(6, 6, 6, 4)
        params_layout.setSpacing(4)

        assignment = QGroupBox("Channel Assignment")
        assignment_layout = QVBoxLayout(assignment)
        assignment_layout.setContentsMargins(6, 8, 6, 6)
        assignment_layout.setSpacing(6)
        assignment_form = QFormLayout()
        assignment_form.setContentsMargins(0, 0, 0, 0)
        assignment_form.setHorizontalSpacing(6)
        assignment_form.setVerticalSpacing(4)
        self.cmp_assign_mode_combo = QComboBox()
        self.cmp_assign_mode_combo.addItems(["Auto by angle", "Manual mapping"])
        self._style_combo_popup(self.cmp_assign_mode_combo)
        self.cmp_in_k_angle_spin = QDoubleSpinBox()
        self.cmp_in_k_angle_spin.setDecimals(3)
        self.cmp_in_k_angle_spin.setRange(-360.0, 360.0)
        self.cmp_out_k_angle_spin = QDoubleSpinBox()
        self.cmp_out_k_angle_spin.setDecimals(3)
        self.cmp_out_k_angle_spin.setRange(-360.0, 360.0)
        self.cmp_auto_assign_btn = QPushButton("Auto Detect")
        angle_row = QWidget()
        angle_h = QHBoxLayout(angle_row)
        angle_h.setContentsMargins(0, 0, 0, 0)
        angle_h.setSpacing(6)
        angle_h.addWidget(QLabel("In K"))
        angle_h.addWidget(self.cmp_in_k_angle_spin)
        angle_h.addWidget(QLabel("Out K"))
        angle_h.addWidget(self.cmp_out_k_angle_spin)
        angle_h.addWidget(self.cmp_auto_assign_btn)
        angle_h.addStretch(1)
        assignment_form.addRow("Mode", self.cmp_assign_mode_combo)
        assignment_form.addRow("Angle Rule", angle_row)
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
        params_layout.addWidget(self._make_expander("Assignment", assignment, expanded=False))

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

        cfg = QFormLayout()
        cfg.setHorizontalSpacing(6)
        cfg.setVerticalSpacing(4)
        _grid, spins, _, _, cmap, fix_checks = self._build_common_range_grid("cmp")
        cmap.setCurrentText("turbo")
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
        params_layout.addWidget(self._make_expander("Plot", basic, expanded=True))
        layout.addWidget(self._make_expander("Parameters", params, expanded=True))
        layout.addStretch(1)
        return tab

    def _cmp_assign_candidate_files(self) -> list[str]:
        chosen = self._selected(self.cmp_files)
        return chosen if chosen else list(self.available_files)

    def _cmp_set_channel_combo_items(self) -> None:
        files = [""] + list(self.available_files)
        for combo in self.cmp_channel_combos.values():
            current = combo.currentText()
            old = combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItems(files)
                if current in files:
                    combo.setCurrentText(current)
            finally:
                combo.blockSignals(old)

    @staticmethod
    def _cmp_angle_distance(a: float, b: float) -> float:
        return abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)

    def _cmp_parse_in_out_angles(self, file_name: str) -> tuple[float | None, float | None]:
        text = str(Path(file_name).stem)
        patterns = (
            r"in[^0-9\-+]*([\-+]?\d+(?:\.\d+)?)\s*degree.*out[^0-9\-+]*([\-+]?\d+(?:\.\d+)?)\s*degree",
            r"out[^0-9\-+]*([\-+]?\d+(?:\.\d+)?)\s*degree.*in[^0-9\-+]*([\-+]?\d+(?:\.\d+)?)\s*degree",
        )
        for idx, pattern in enumerate(patterns):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            if idx == 0:
                return float(match.group(1)), float(match.group(2))
            return float(match.group(2)), float(match.group(1))
        return None, None

    def _cmp_classify_channel(self, file_name: str) -> str | None:
        in_angle, out_angle = self._cmp_parse_in_out_angles(file_name)
        if in_angle is None or out_angle is None:
            return None
        in_k = float(self.cmp_in_k_angle_spin.value())
        out_k = float(self.cmp_out_k_angle_spin.value())
        in_is_k = self._cmp_angle_distance(in_angle, in_k) <= 45.0
        out_is_k = self._cmp_angle_distance(out_angle, out_k) <= 45.0
        if in_is_k and out_is_k:
            return "KK"
        if in_is_k and (not out_is_k):
            return "KKp"
        if (not in_is_k) and out_is_k:
            return "KpK"
        return "KpKp"

    def _cmp_current_mapping(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        used: set[str] = set()
        for key in COMPARE_PANEL_ORDER:
            name = self.cmp_channel_combos[key].currentText().strip()
            if not name or name in used:
                continue
            mapping[key] = name
            used.add(name)
        return mapping

    def _cmp_visible_channels(self, mapping: dict[str, str] | None = None) -> list[str]:
        mapping = mapping or self._cmp_current_mapping()
        preset = self.cmp_display_preset_combo.currentText()
        if preset == "KK + KKp":
            order = ["KK", "KKp"]
        elif preset == "KpK + KpKp":
            order = ["KpK", "KpKp"]
        elif preset == "All four":
            order = list(COMPARE_PANEL_ORDER)
        else:
            order = [key for key in COMPARE_PANEL_ORDER if self.cmp_show_checks[key].isChecked()]
        return [key for key in order if key in mapping]

    def _cmp_apply_display_preset(self) -> None:
        preset = self.cmp_display_preset_combo.currentText()
        enabled = preset == "Custom"
        desired = {
            "KK + KKp": {"KK", "KKp"},
            "KpK + KpKp": {"KpK", "KpKp"},
            "All four": set(COMPARE_PANEL_ORDER),
            "Custom": None,
        }[preset]
        for key, chk in self.cmp_show_checks.items():
            chk.setEnabled(enabled)
            if desired is not None:
                old = chk.blockSignals(True)
                try:
                    chk.setChecked(key in desired)
                finally:
                    chk.blockSignals(old)

    def _cmp_update_assignment_summary(self) -> None:
        mapping = self._cmp_current_mapping()
        visible = self._cmp_visible_channels(mapping)
        lines: list[str] = []
        for key in COMPARE_PANEL_ORDER:
            source = mapping.get(key, "missing")
            lines.append(f"{key} -> {source}")
        if visible:
            lines.append("Visible -> " + ", ".join(visible))
        else:
            lines.append("Visible -> none")
        self.cmp_assignment_summary.setPlainText("\n".join(lines))

    def _cmp_update_assignment_mode(self) -> None:
        auto_mode = self.cmp_assign_mode_combo.currentText() == "Auto by angle"
        self.cmp_in_k_angle_spin.setEnabled(auto_mode)
        self.cmp_out_k_angle_spin.setEnabled(auto_mode)
        self.cmp_auto_assign_btn.setEnabled(auto_mode)

    def _cmp_selection_from_ui(self) -> data_io.CompareSelection:
        mapping = self._cmp_current_mapping()
        visible = self._cmp_visible_channels(mapping)
        if len(visible) < 1:
            raise ValueError("Assign at least one compare channel.")
        if len(visible) < 2:
            raise ValueError("Select at least two visible compare channels.")
        return data_io.CompareSelection.from_mapping(
            mapping,
            visible_order=visible,
        )

    def _cmp_auto_assign_channels(self) -> None:
        candidates = self._cmp_assign_candidate_files()
        found: dict[str, str] = {}
        duplicates: dict[str, list[str]] = {}
        for file_name in candidates:
            key = self._cmp_classify_channel(file_name)
            if key is None:
                continue
            if key in found:
                duplicates.setdefault(key, [found[key]]).append(file_name)
                continue
            found[key] = file_name
        for key, combo in self.cmp_channel_combos.items():
            old = combo.blockSignals(True)
            try:
                combo.setCurrentText(found.get(key, ""))
            finally:
                combo.blockSignals(old)
        self._cmp_update_assignment_summary()
        if duplicates:
            dup_text = "; ".join(f"{k}: {', '.join(v)}" for k, v in duplicates.items())
            self._append_log(f"Compare auto-detect found duplicate matches -> {dup_text}")
        self._on_cmp_plot_param_changed()

    def _build_tools_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        log_box = QGroupBox("Log Panel")
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(8, 8, 8, 8)
        log_layout.setSpacing(6)
        hint = QLabel("The log panel records all load, plot, and export events.")
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #5a6a7a; font-size: 10px; }")
        log_layout.addWidget(hint)
        log_btn_row = QHBoxLayout()
        log_btn_row.setSpacing(8)
        self.show_log_btn = QPushButton("Show / Hide Log Panel")
        self.show_log_btn.setToolTip("Toggle the bottom log dock panel")
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.setToolTip("Clear all messages from the log")
        log_btn_row.addWidget(self.show_log_btn)
        log_btn_row.addWidget(self.clear_log_btn)
        log_layout.addLayout(log_btn_row)
        layout.addWidget(log_box)

        file_box = QGroupBox("File Management")
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(8, 8, 8, 8)
        file_layout.setSpacing(6)
        file_hint = QLabel(
            "After export, source CSV files can be moved to an archive folder "
            "('Initial data after processing') to keep the workspace clean."
        )
        file_hint.setWordWrap(True)
        file_hint.setStyleSheet("QLabel { color: #5a6a7a; font-size: 10px; }")
        file_layout.addWidget(file_hint)
        layout.addWidget(file_box)

        layout.addStretch(1)
        return tab

    def apply_ui_metrics(self) -> None:
        h = UI_METRICS["input_h"]
        for w in self.findChildren(QLineEdit):
            if w.minimumHeight() < h:
                w.setMinimumHeight(h)
        for w in self.findChildren(QComboBox):
            if w.minimumHeight() < h:
                w.setMinimumHeight(h)
        for w in self.findChildren(QDoubleSpinBox):
            if w.minimumHeight() < h:
                w.setMinimumHeight(h)
        for w in self.findChildren(QSpinBox):
            if w.minimumHeight() < h:
                w.setMinimumHeight(h)
        for btn in self.findChildren(QToolButton):
            if btn.minimumHeight() < UI_METRICS["tool_h"]:
                btn.setMinimumHeight(UI_METRICS["tool_h"])
        for combo in (self.pl_cmap, self.cmp_cmap):
            combo.setMinimumWidth(UI_METRICS["short_combo_w"])

    def _build_plot_panel(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(9, 7), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, box)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        return box

    def _build_log_dock(self) -> None:
        self.log_dock = QDockWidget("Run Log", self)
        self.log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        container = QWidget()
        layout = QVBoxLayout(container)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        self.log_dock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()

    def _build_menu_and_toolbar(self) -> None:
        menu = self.menuBar().addMenu("View")
        self.show_log_action = self.log_dock.toggleViewAction()
        self.show_log_action.setText("Show Log")
        menu.addAction(self.show_log_action)

        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        self.load_action = QAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), "Load", self)
        self.load_action.setToolTip("Load data for the active tab")
        self.plot_action = QAction(self.style().standardIcon(QStyle.SP_BrowserReload), "Plot / Update", self)
        self.plot_action.setToolTip("Plot/update current state")
        self.save_action = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), "Save PNG + DAT", self)
        self.save_action.setToolTip("Export for the active tab")
        self.auto_move_after_export_chk = QCheckBox("Auto Move")
        self.auto_move_after_export_chk.setChecked(False)
        self.auto_move_after_export_chk.setToolTip("After export, move source CSVs to 'Initial data after processing'.")
        self.move_now_btn = QPushButton("Move Now")
        self.move_now_btn.setToolTip("Move current source CSVs to 'Initial data after processing' now.")
        toolbar.addAction(self.load_action)
        toolbar.addAction(self.plot_action)
        toolbar.addAction(self.save_action)
        toolbar.addSeparator()
        toolbar.addWidget(self.move_now_btn)
        toolbar.addWidget(self.auto_move_after_export_chk)

    def _wire_actions(self) -> None:
        self.browse_btn.clicked.connect(self._browse_folder)
        self.open_file_btn.clicked.connect(self._open_file)
        self.refresh_btn.clicked.connect(self._refresh_file_lists)
        self.load_action.triggered.connect(self._toolbar_load)
        self.plot_action.triggered.connect(self._toolbar_plot)
        self.save_action.triggered.connect(self._toolbar_save)
        self.move_now_btn.clicked.connect(self._manual_move_sources)
        self.tabs.currentChanged.connect(lambda _i: self._update_action_states())
        for prefix in ("pl", "drr", "cmp"):
            combo: QComboBox = getattr(self, f"{prefix}_yaxis_combo")
            combo.currentTextChanged.connect(lambda _text, p=prefix: self._update_y_axis_controls(p))
            for key in ("a", "b", "c"):
                spin: QDoubleSpinBox = getattr(self, f"{prefix}_yaxis_{key}_spin")
                spin.valueChanged.connect(lambda _value, p=prefix: self._update_y_axis_controls(p))
        self.pl_auto_v_btn.clicked.connect(self._auto_pl_vrange)
        self.pl_auto_x_btn.clicked.connect(self._auto_pl_xrange)
        self.pl_auto_y_btn.clicked.connect(self._auto_pl_yrange)
        self.pl_yaxis_combo.currentTextChanged.connect(self._on_pl_plot_param_changed)
        self.pl_yaxis_a_spin.valueChanged.connect(self._on_pl_plot_param_changed)
        self.pl_yaxis_b_spin.valueChanged.connect(self._on_pl_plot_param_changed)
        self.pl_yaxis_c_spin.valueChanged.connect(self._on_pl_plot_param_changed)
        for key in ("vmin", "vmax", "xmin", "xmax", "ymin", "ymax"):
            self.pl_spins[key].valueChanged.connect(self._on_pl_plot_param_changed)
        self.pl_spins["gate"].valueChanged.connect(self._on_pl_gate_changed)
        self.pl_cmap.currentTextChanged.connect(self._on_pl_plot_param_changed)
        self.pl_log_chk.toggled.connect(self._on_pl_plot_param_changed)
        self.pl_clip_chk.toggled.connect(self._on_pl_plot_param_changed)
        self.cmp_assign_mode_combo.currentTextChanged.connect(self._on_cmp_assignment_mode_changed)
        self.cmp_in_k_angle_spin.valueChanged.connect(self._on_cmp_assignment_inputs_changed)
        self.cmp_out_k_angle_spin.valueChanged.connect(self._on_cmp_assignment_inputs_changed)
        self.cmp_auto_assign_btn.clicked.connect(self._cmp_auto_assign_channels)
        self.cmp_display_preset_combo.currentTextChanged.connect(self._on_cmp_display_preset_changed)
        for combo in self.cmp_channel_combos.values():
            combo.currentTextChanged.connect(self._on_cmp_plot_param_changed)
        for chk in self.cmp_show_checks.values():
            chk.toggled.connect(self._on_cmp_plot_param_changed)
        self.cmp_yaxis_combo.currentTextChanged.connect(self._on_cmp_plot_param_changed)
        self.cmp_yaxis_a_spin.valueChanged.connect(self._on_cmp_plot_param_changed)
        self.cmp_yaxis_b_spin.valueChanged.connect(self._on_cmp_plot_param_changed)
        self.cmp_yaxis_c_spin.valueChanged.connect(self._on_cmp_plot_param_changed)
        for key in ("vmin", "vmax", "xmin", "xmax", "ymin", "ymax", "gate"):
            self.cmp_spins[key].valueChanged.connect(self._on_cmp_plot_param_changed)
        self.cmp_cmap.currentTextChanged.connect(self._on_cmp_plot_param_changed)
        self.cmp_log_chk.toggled.connect(self._on_cmp_plot_param_changed)
        self.cmp_clip_chk.toggled.connect(self._on_cmp_plot_param_changed)
        self.cmp_auto_v_btn.clicked.connect(self._auto_cmp_vrange)
        self.cmp_auto_x_btn.clicked.connect(self._auto_cmp_xrange)
        self.cmp_auto_y_btn.clicked.connect(self._auto_cmp_yrange)
        self.drr_yaxis_combo.currentTextChanged.connect(self._on_drr_plot_param_changed)
        self.drr_yaxis_a_spin.valueChanged.connect(self._on_drr_plot_param_changed)
        self.drr_yaxis_b_spin.valueChanged.connect(self._on_drr_plot_param_changed)
        self.drr_yaxis_c_spin.valueChanged.connect(self._on_drr_plot_param_changed)
        self.drr_baseline_combo.currentTextChanged.connect(self._on_drr_plot_param_changed)
        self.drr_derivative_combo.currentTextChanged.connect(self._on_drr_derivative_changed)
        self.drr_sg_window_spin.valueChanged.connect(self._on_drr_derivative_changed)
        self.drr_sg_poly_spin.valueChanged.connect(self._on_drr_derivative_changed)
        self.drr_edit_measurements_btn.clicked.connect(self._edit_drr_measurements)
        self.drr_clear_measurements_btn.clicked.connect(self._clear_drr_measurements)
        self.drr_edit_baselines_btn.clicked.connect(self._edit_drr_baselines_dialog)
        self.drr_baseline_autofind_btn.clicked.connect(self._auto_find_back_baselines)
        self.drr_baseline_combine_combo.currentTextChanged.connect(self._on_drr_baseline_mode_changed)
        self.drr_auto_v_btn.clicked.connect(self._auto_drr_vrange)
        self.drr_auto_x_btn.clicked.connect(self._auto_drr_xrange)
        self.drr_auto_y_btn.clicked.connect(self._auto_drr_yrange)
        for key in ("vmin", "vmax", "xmin", "xmax", "ymin", "ymax", "gate"):
            self.drr_spins[key].valueChanged.connect(self._on_drr_plot_param_changed)
        self.drr_cmap.currentTextChanged.connect(self._on_drr_plot_param_changed)
        self.drr_log_chk.toggled.connect(self._on_drr_plot_param_changed)
        self.drr_clip_chk.toggled.connect(self._on_drr_plot_param_changed)
        self.drr_center_zero_chk.toggled.connect(self._on_drr_plot_param_changed)
        self.drr_peak_find_btn.clicked.connect(self._on_drr_find_peaks)
        self.drr_peak_show_chk.toggled.connect(self._on_drr_analysis_view_changed)
        self.drr_peak_mode_combo.currentTextChanged.connect(self._on_drr_analysis_view_changed)
        self.drr_fit_btn.clicked.connect(self._on_drr_fit_lorentz)
        self.drr_fit_clear_btn.clicked.connect(self._on_drr_clear_fit)
        self.drr_fit_show_chk.toggled.connect(self._on_drr_analysis_view_changed)
        self.pl_peak_find_btn.clicked.connect(self._on_pl_find_peaks)
        self.pl_peak_show_chk.toggled.connect(self._on_pl_analysis_view_changed)
        self.pl_peak_mode_combo.currentTextChanged.connect(self._on_pl_analysis_view_changed)
        self.pl_fit_btn.clicked.connect(self._on_pl_fit_lorentz)
        self.pl_fit_clear_btn.clicked.connect(self._on_pl_clear_fit)
        self.pl_fit_show_chk.toggled.connect(self._on_pl_analysis_view_changed)

        self.show_log_btn.clicked.connect(self._toggle_log)
        self.clear_log_btn.clicked.connect(self._clear_log)
        self._gate_motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self._gate_click_cid = self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        for prefix in ("pl", "drr", "cmp"):
            self._update_y_axis_controls(prefix)
        self._cmp_apply_display_preset()
        self._cmp_update_assignment_mode()
        self._cmp_set_channel_combo_items()
        self._cmp_update_assignment_summary()

    def _toggle_log(self) -> None:
        self.log_dock.setVisible(not self.log_dock.isVisible())

    def _clear_log(self) -> None:
        self.log_lines.clear()
        self.log_text.clear()

    def _selected(self, widget: QListWidget) -> List[str]:
        return [i.text() for i in widget.selectedItems()]

    def _manual_move_sources(self) -> None:
        if not self.current_folder:
            self._show_error("Choose a folder first.")
            return
        mode = self._active_mode()
        names: list[str] = []
        if mode == "PL":
            names = self._selected(self.pl_files)
            if not names and self.loaded and self.loaded.mode == "PL" and self.loaded.primary_file:
                names = [self.loaded.primary_file]
        elif mode == "DRR":
            names = list(self.drr_selected_files) + list(self.drr_baseline_files_manual)
            if (not names) and self.loaded and self.loaded.mode == "DRR":
                names = list(self.loaded.selected_files) + list(self.loaded.baseline_files)
        elif mode == "Compare":
            names = self._selected(self.cmp_files)
            if (not names) and self.loaded and self.loaded.mode == "Compare":
                names = list(self.loaded.compare_sources.values()) if self.loaded.compare_sources else list(self.loaded.selected_files)
        if not names:
            self._show_error("No source files selected to move.")
            return
        moved = int(data_io.move_selected_to_archive(self.current_folder, names))
        self._refresh_file_lists()
        self._append_log(f"Moved {moved} source CSV file(s) to 'Initial data after processing'.")
        self._status(f"Moved {moved} file(s).")

    def _status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def _append_log(self, text: str) -> None:
        self.log_lines.append(text)
        self.log_text.setPlainText("\n".join(self.log_lines))
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def _set_stage(self, stage: str) -> None:
        self._status(f"State: {stage}")
        self._status_progress.setVisible(stage == "Loading...")

    def _active_mode(self) -> str | None:
        text = self.tabs.tabText(self.tabs.currentIndex())
        return text if text in {"PL", "DRR", "Compare"} else None

    def _toolbar_load(self) -> None:
        mode = self._active_mode()
        if mode:
            self._start_load(mode)

    def _toolbar_plot(self) -> None:
        mode = self._active_mode()
        if mode:
            self._plot_mode(mode)

    def _toolbar_save(self) -> None:
        mode = self._active_mode()
        if mode:
            self._start_export(mode)

    def _on_drr_derivative_changed(self) -> None:
        self._enforce_drr_sg_constraints(show_status=True)
        if self.loaded and self.loaded.mode == "DRR" and not self._suspend_drr_autoplot:
            self._plot_mode("DRR")

    def _on_drr_plot_param_changed(self) -> None:
        if self.loaded and self.loaded.mode == "DRR" and not self._suspend_drr_autoplot:
            self._plot_mode("DRR")

    def _on_pl_plot_param_changed(self) -> None:
        if self.loaded and self.loaded.mode == "PL":
            self._plot_mode("PL")

    def _on_cmp_assignment_mode_changed(self) -> None:
        self._cmp_update_assignment_mode()
        if self.cmp_assign_mode_combo.currentText() == "Auto by angle":
            self._cmp_auto_assign_channels()
        else:
            self._cmp_update_assignment_summary()
            self._on_cmp_plot_param_changed()

    def _on_cmp_assignment_inputs_changed(self) -> None:
        if self.cmp_assign_mode_combo.currentText() == "Auto by angle":
            self._cmp_auto_assign_channels()
        else:
            self._cmp_update_assignment_summary()

    def _on_cmp_display_preset_changed(self) -> None:
        self._cmp_apply_display_preset()
        self._cmp_update_assignment_summary()
        self._on_cmp_plot_param_changed()

    def _on_cmp_plot_param_changed(self) -> None:
        self._cmp_update_assignment_summary()
        if self.loaded and self.loaded.mode == "Compare":
            self._plot_mode("Compare")

    def _show_error(self, message: str) -> None:
        first = message.splitlines()[0] if message else "Unknown error"
        self._append_log(f"ERROR: {first}")
        self._status(f"Error: {first}")
        QMessageBox.critical(self, "Error", message)

    def _browse_folder(self) -> None:
        start = self.current_folder or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder", start)
        if not folder:
            return
        self.current_folder = folder
        self.folder_edit.setText(folder)
        self._refresh_file_lists()

    def _open_file(self) -> None:
        start = self.current_folder or str(Path.home())
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", start, "CSV (*.csv)")
        if not file_path:
            return
        path = Path(file_path)
        self.current_folder = str(path.parent)
        self.folder_edit.setText(self.current_folder)
        self._refresh_file_lists()
        for lst in (self.pl_files, self.cmp_files):
            matches = lst.findItems(path.name, Qt.MatchExactly)
            if matches:
                lst.clearSelection()
                matches[0].setSelected(True)
        self.drr_selected_files = [path.name]
        self._update_drr_selection_labels()
        self._status(f"Selected {path.name}")

    def _refresh_file_lists(self) -> None:
        for lst in (self.pl_files, self.cmp_files):
            lst.clear()
        if not self.current_folder:
            return
        self.available_files = data_io.list_csv_files(self.current_folder)
        self.pl_files.addItems(self.available_files)
        self.cmp_files.addItems(self.available_files)
        self._cmp_set_channel_combo_items()
        self.drr_selected_files = [f for f in self.drr_selected_files if f in self.available_files]
        self.drr_baseline_files_manual = [f for f in self.drr_baseline_files_manual if f in self.available_files]
        self.drr_baseline_files_found = [f for f in self.drr_baseline_files_found if f in self.available_files]
        self._cmp_update_assignment_summary()
        self._update_drr_selection_labels()
        self._status(f"Loaded file list: {len(self.available_files)}")

    def _update_drr_selection_labels(self) -> None:
        def _brief(names: List[str]) -> str:
            if not names:
                return "none"
            if len(names) == 1:
                s = names[0]
            else:
                s = f"{names[0]}, {names[1]}" if len(names) > 1 else names[0]
                if len(names) > 2:
                    s += f", +{len(names)-2} more"
            return (s[:46] + "...") if len(s) > 49 else s

        mode_map = {
            "Last frame of each baseline file": "last",
            "First frame of each baseline file": "first",
            "Average all baseline files": "avg",
        }
        mode_short = mode_map.get(self.drr_baseline_combine_combo.currentText(), "last")
        self.drr_measurement_summary.setText(f"Measurement: {len(self.drr_selected_files)} files ({_brief(self.drr_selected_files)})")
        self.drr_baseline_summary.setText(
            f"Baselines: {len(self.drr_baseline_files_manual)} files (mode: {mode_short})"
        )

    def _edit_drr_measurements(self) -> None:
        self.drr_selected_files = self._open_dual_list_dialog(
            title="Measurement Files",
            selected=self.drr_selected_files,
            enable_group_auto=True,
            enable_back_auto=False,
        )
        self._update_drr_selection_labels()
        if self.loaded and self.loaded.mode == "DRR":
            self._start_load("DRR")

    def _clear_drr_measurements(self) -> None:
        self.drr_selected_files = []
        self._update_drr_selection_labels()

    def _edit_drr_baselines_dialog(self) -> None:
        self.drr_baseline_files_manual = self._open_dual_list_dialog(
            title="Baseline Files",
            selected=self.drr_baseline_files_manual,
            enable_group_auto=False,
            enable_back_auto=True,
        )
        self._update_drr_selection_labels()
        if self.loaded and self.loaded.mode == "DRR":
            self._start_load("DRR")

    def _open_dual_list_dialog(
        self,
        *,
        title: str,
        selected: List[str],
        enable_group_auto: bool,
        enable_back_auto: bool,
    ) -> List[str]:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(760, 500)
        v = QVBoxLayout(dlg)
        top_row = QHBoxLayout()
        filter_edit = QLineEdit()
        filter_edit.setPlaceholderText("Filter filenames...")
        top_row.addWidget(QLabel("Filter"))
        top_row.addWidget(filter_edit, 1)
        if enable_back_auto:
            auto_back_btn = QPushButton("Find 'back'")
            auto_back_btn.setToolTip("Auto-select files where filename contains 'back'.")
            top_row.addWidget(auto_back_btn)
        if enable_group_auto:
            auto_group_btn = QPushButton("Use Largest Group")
            auto_group_btn.setToolTip("Auto-select the largest measurement group in this folder.")
            top_row.addWidget(auto_group_btn)
        v.addLayout(top_row)

        lists_row = QHBoxLayout()
        available = QListWidget()
        available.setSelectionMode(QAbstractItemView.ExtendedSelection)
        current = QListWidget()
        current.setSelectionMode(QAbstractItemView.ExtendedSelection)
        selected_set = set(selected)
        for f in self.available_files:
            if f not in selected_set:
                available.addItem(f)
        current.addItems(selected)
        lists_row.addWidget(available, 1)

        mid_btns = QVBoxLayout()
        add_btn = QPushButton("Add >")
        remove_btn = QPushButton("< Remove")
        clear_btn = QPushButton("Clear")
        mid_btns.addStretch(1)
        mid_btns.addWidget(add_btn)
        mid_btns.addWidget(remove_btn)
        mid_btns.addWidget(clear_btn)
        mid_btns.addStretch(1)
        lists_row.addLayout(mid_btns)
        lists_row.addWidget(current, 1)
        v.addLayout(lists_row)

        def _move(src: QListWidget, dst: QListWidget) -> None:
            items = [i.text() for i in src.selectedItems()]
            if not items:
                return
            existing = {dst.item(i).text() for i in range(dst.count())}
            for name in items:
                if name not in existing:
                    dst.addItem(name)
                    existing.add(name)
            for it in src.selectedItems():
                src.takeItem(src.row(it))

        def _filter() -> None:
            needle = filter_edit.text().strip().lower()
            selected_now = [current.item(i).text() for i in range(current.count())]
            available.clear()
            for f in self.available_files:
                if f in selected_now:
                    continue
                if not needle or needle in f.lower():
                    available.addItem(f)

        add_btn.clicked.connect(lambda: _move(available, current))
        remove_btn.clicked.connect(lambda: _move(current, available))
        clear_btn.clicked.connect(lambda: (available.addItems([current.item(i).text() for i in range(current.count())]), current.clear(), _filter()))
        filter_edit.textChanged.connect(lambda _t: _filter())
        available.itemDoubleClicked.connect(lambda _i: _move(available, current))
        current.itemDoubleClicked.connect(lambda _i: _move(current, available))

        if enable_back_auto:
            def _auto_back() -> None:
                matches = [f for f in self.available_files if "back" in f.lower()]
                current.clear()
                current.addItems(matches)
                _filter()
                self._status(f"State: Found {len(matches)} baseline files containing 'back'.")
            auto_back_btn.clicked.connect(_auto_back)

        if enable_group_auto:
            def _auto_group() -> None:
                groups = group_measurement_files(self.available_files)
                if not groups:
                    return
                largest_key = max(groups.keys(), key=lambda k: len(groups[k]))
                matches = [f for f in groups[largest_key] if "back" not in f.lower()]
                current.clear()
                current.addItems(matches)
                _filter()
                self._status(f"State: Selected group '{largest_key}' ({len(matches)} files).")
            auto_group_btn.clicked.connect(_auto_group)

        b = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        b.accepted.connect(dlg.accept)
        b.rejected.connect(dlg.reject)
        v.addWidget(b)
        if dlg.exec() != QDialog.Accepted:
            return selected
        return [current.item(i).text() for i in range(current.count())]

    def _on_drr_baseline_mode_changed(self) -> None:
        self._status(f"State: Baseline mode set: {self.drr_baseline_combine_combo.currentText()}.")
        self._update_drr_selection_labels()
        if self.loaded and self.loaded.mode == "DRR" and not self._suspend_drr_autoplot:
            self._plot_mode("DRR")

    def _auto_find_back_baselines(self) -> None:
        matches = [f for f in self.available_files if "back" in f.lower()]
        self.drr_baseline_files_manual = matches
        if matches:
            self.drr_baseline_combo.setCurrentText("External")
        self._update_drr_selection_labels()
        self._status(f"State: Auto-found {len(matches)} baseline files containing 'back'.")
        if self.loaded and self.loaded.mode == "DRR":
            self._start_load("DRR")

    def _update_action_states(self) -> None:
        active_mode = self._active_mode()
        loaded_mode = self.loaded.mode if self.loaded else None
        plotted_mode = self.last_plotted_mode
        self.load_action.setEnabled(active_mode is not None)
        self.plot_action.setEnabled(active_mode is not None and loaded_mode == active_mode)
        self.save_action.setEnabled(active_mode is not None and plotted_mode == active_mode)
        pl_loaded = loaded_mode == "PL"
        drr_loaded = loaded_mode == "DRR"
        cmp_loaded = loaded_mode == "Compare"
        self.pl_auto_v_btn.setEnabled(pl_loaded)
        self.pl_auto_x_btn.setEnabled(pl_loaded)
        self.pl_auto_y_btn.setEnabled(pl_loaded)
        self.drr_auto_v_btn.setEnabled(drr_loaded)
        self.drr_auto_x_btn.setEnabled(drr_loaded)
        self.drr_auto_y_btn.setEnabled(drr_loaded)
        self.cmp_auto_v_btn.setEnabled(cmp_loaded)
        self.cmp_auto_x_btn.setEnabled(cmp_loaded)
        self.cmp_auto_y_btn.setEnabled(cmp_loaded)

    def _reset_params(self, mode: str) -> None:
        if self.loaded and self.loaded.mode == mode:
            self._apply_auto_limits_for_loaded()

    def _auto_drr_vrange(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        cube = self._drr_cube_for_display()
        x = np.asarray(cube.energy, float).ravel()
        y = np.asarray(cube.gate, float).ravel()
        z = np.asarray(cube.Z, float)
        x0, x1 = sorted((float(self.drr_spins["xmin"].value()), float(self.drr_spins["xmax"].value())))
        y0, y1 = sorted((float(self.drr_spins["ymin"].value()), float(self.drr_spins["ymax"].value())))
        x_mask = (x >= x0) & (x <= x1)
        y_mask = (y >= y0) & (y <= y1)
        if np.any(y_mask) and np.any(x_mask):
            z_roi = z[np.ix_(y_mask, x_mask)]
        else:
            z_roi = z
        finite = z_roi[np.isfinite(z_roi)]
        if finite.size == 0:
            self._status("State: Auto vmin/vmax skipped (no finite values in selected x/y range).")
            return
        if self._mode_log("DRR"):
            pos = z_roi[np.isfinite(z_roi) & (z_roi > 0)]
            if pos.size:
                vmin, vmax = np.nanpercentile(pos, [0.01, 99.99])
                vmin = float(max(vmin, 1e-12))
                vmax = float(max(vmax, vmin * 1.01))
            else:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            vmin, vmax = np.nanpercentile(finite, [0.01, 99.99])
            vmin, vmax = float(vmin), float(vmax)
        spins = self.drr_spins
        spins["vmin"].setValue(vmin)
        spins["vmax"].setValue(vmax)
        self._status(f"State: Auto vmin/vmax (ROI) = {vmin:.4g}, {vmax:.4g}")
        self._plot_mode("DRR")

    def _auto_pl_vrange(self) -> None:
        if not self.loaded or self.loaded.mode != "PL" or self.loaded.cube is None:
            return
        cube = self.loaded.cube
        x = np.asarray(cube.energy, float).ravel()
        y = np.asarray(cube.gate, float).ravel()
        z = np.asarray(cube.Z, float)
        x0, x1 = sorted((float(self.pl_spins["xmin"].value()), float(self.pl_spins["xmax"].value())))
        y0, y1 = sorted((float(self.pl_spins["ymin"].value()), float(self.pl_spins["ymax"].value())))
        x_mask = (x >= x0) & (x <= x1)
        y_mask = (y >= y0) & (y <= y1)
        z_roi = z[np.ix_(y_mask, x_mask)] if np.any(y_mask) and np.any(x_mask) else z
        finite = z_roi[np.isfinite(z_roi)]
        if finite.size == 0:
            self._status("State: Auto vmin/vmax skipped (no finite values in selected x/y range).")
            return
        if self._mode_log("PL"):
            pos = z_roi[np.isfinite(z_roi) & (z_roi > 0)]
            if pos.size:
                vmin, vmax = np.nanpercentile(pos, [0.01, 99.99])
                vmin = float(max(vmin, 1e-12))
                vmax = float(max(vmax, vmin * 1.01))
            else:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            vmin, vmax = np.nanpercentile(finite, [0.01, 99.99])
            vmin, vmax = float(vmin), float(vmax)
        self.pl_spins["vmin"].setValue(vmin)
        self.pl_spins["vmax"].setValue(vmax)
        self._status(f"State: Auto vmin/vmax (ROI) = {vmin:.4g}, {vmax:.4g}")
        self._plot_mode("PL")

    def _auto_cmp_vrange(self) -> None:
        if not self.loaded or self.loaded.mode != "Compare" or not self.loaded.compare_cubes:
            return
        x0, x1 = sorted((float(self.cmp_spins["xmin"].value()), float(self.cmp_spins["xmax"].value())))
        y0, y1 = sorted((float(self.cmp_spins["ymin"].value()), float(self.cmp_spins["ymax"].value())))
        vals: list[np.ndarray] = []
        for cube in self.loaded.compare_cubes.values():
            x = np.asarray(cube.energy, float).ravel()
            y = np.asarray(cube.gate, float).ravel()
            z = np.asarray(cube.Z, float)
            x_mask = (x >= x0) & (x <= x1)
            y_mask = (y >= y0) & (y <= y1)
            z_roi = z[np.ix_(y_mask, x_mask)] if np.any(y_mask) and np.any(x_mask) else z
            finite = z_roi[np.isfinite(z_roi)]
            if finite.size:
                vals.append(finite)
        if not vals:
            return
        finite = np.concatenate(vals)
        if self._mode_log("Compare"):
            pos = finite[finite > 0]
            if pos.size:
                vmin, vmax = np.nanpercentile(pos, [0.01, 99.99])
                vmin = float(max(vmin, 1e-12))
                vmax = float(max(vmax, vmin * 1.01))
            else:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            vmin, vmax = np.nanpercentile(finite, [0.01, 99.99])
            vmin, vmax = float(vmin), float(vmax)
        self.cmp_spins["vmin"].setValue(vmin)
        self.cmp_spins["vmax"].setValue(vmax)
        self._plot_mode("Compare")

    def _auto_drr_xrange(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        cube = self._drr_cube_for_display()
        self.drr_spins["xmin"].setValue(float(np.nanmin(cube.energy)))
        self.drr_spins["xmax"].setValue(float(np.nanmax(cube.energy)))
        self._status("State: Auto xmin/xmax set from energy axis.")
        self._plot_mode("DRR")

    def _auto_pl_xrange(self) -> None:
        if not self.loaded or self.loaded.mode != "PL" or self.loaded.cube is None:
            return
        self.pl_spins["xmin"].setValue(float(np.nanmin(self.loaded.cube.energy)))
        self.pl_spins["xmax"].setValue(float(np.nanmax(self.loaded.cube.energy)))
        self._status("State: Auto xmin/xmax set from energy axis.")
        self._plot_mode("PL")

    def _auto_cmp_xrange(self) -> None:
        if not self.loaded or self.loaded.mode != "Compare" or not self.loaded.compare_cubes:
            return
        mins = [float(np.nanmin(c.energy)) for c in self.loaded.compare_cubes.values()]
        maxs = [float(np.nanmax(c.energy)) for c in self.loaded.compare_cubes.values()]
        self.cmp_spins["xmin"].setValue(min(mins))
        self.cmp_spins["xmax"].setValue(max(maxs))
        self._plot_mode("Compare")

    def _auto_drr_yrange(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        cube = self._drr_cube_for_display()
        self.drr_spins["ymin"].setValue(float(np.nanmin(cube.gate)))
        self.drr_spins["ymax"].setValue(float(np.nanmax(cube.gate)))
        self._status("State: Auto ymin/ymax set from gate axis.")
        self._plot_mode("DRR")

    def _auto_pl_yrange(self) -> None:
        if not self.loaded or self.loaded.mode != "PL" or self.loaded.cube is None:
            return
        self.pl_spins["ymin"].setValue(float(np.nanmin(self.loaded.cube.gate)))
        self.pl_spins["ymax"].setValue(float(np.nanmax(self.loaded.cube.gate)))
        self._status("State: Auto ymin/ymax set from gate axis.")
        self._plot_mode("PL")

    def _auto_cmp_yrange(self) -> None:
        if not self.loaded or self.loaded.mode != "Compare" or not self.loaded.compare_cubes:
            return
        mins = [float(np.nanmin(c.gate)) for c in self.loaded.compare_cubes.values()]
        maxs = [float(np.nanmax(c.gate)) for c in self.loaded.compare_cubes.values()]
        self.cmp_spins["ymin"].setValue(min(mins))
        self.cmp_spins["ymax"].setValue(max(maxs))
        self._plot_mode("Compare")

    def _drr_derivative_value(self) -> int | None:
        text = self.drr_derivative_combo.currentText()
        return None if text == "None" else (1 if text == "dE" else 2)

    def _enforce_drr_sg_constraints(self, *, show_status: bool) -> int:
        poly = int(self.drr_sg_poly_spin.value())
        req_win = int(self.drr_sg_window_spin.value())
        n_energy = (
            int(np.asarray(self.loaded.cube.energy).size)
            if self.loaded and self.loaded.mode == "DRR" and self.loaded.cube is not None
            else 401
        )
        used_win = clamp_sg_window(req_win, n_energy=n_energy, polyorder=poly)
        if used_win != req_win:
            self.drr_sg_window_spin.blockSignals(True)
            self.drr_sg_window_spin.setValue(used_win)
            self.drr_sg_window_spin.blockSignals(False)
            if show_status:
                self._status(f"State: SG window clamped to {used_win} (odd, valid for order={poly}).")
        return used_win

    def _drr_cube_for_display(self) -> DataCube:
        if not self.loaded or self.loaded.mode != "DRR" or self.loaded.cube is None:
            raise ValueError("No DRR data loaded.")
        deriv = self._drr_derivative_value()
        poly = int(self.drr_sg_poly_spin.value())
        req_win = self._enforce_drr_sg_constraints(show_status=True)
        cube, used_win = apply_sg_derivative_energy(
            self.loaded.cube,
            derivative=deriv,
            window_length=req_win,
            polyorder=poly,
        )
        if deriv is not None and used_win != req_win:
            self._status(f"State: SG window adjusted to {used_win}.")
        return cube

    def _start_load(self, mode: str) -> None:
        if not self.current_folder:
            self._show_error("Choose a folder first.")
            return

        if mode == "PL":
            selected = self._selected(self.pl_files)
            baselines: list[str] = []
            pl_log = bool(self.pl_log_chk.isChecked())
            cmp_log = False
            drr_baseline = "Self (last frame)"
            drr_baseline_which = "last"
            y_axis_spec = self._selected_y_axis_spec("pl")
        elif mode == "DRR":
            selected = list(self.drr_selected_files)
            baselines = list(self.drr_baseline_files_manual)
            pl_log = False
            cmp_log = False
            drr_baseline = self.drr_baseline_combo.currentText()
            which_map = {
                "Last frame of each baseline file": "last",
                "First frame of each baseline file": "first",
                "Average all baseline files": "all",
            }
            drr_baseline_which = which_map.get(self.drr_baseline_combine_combo.currentText(), "last")
            y_axis_spec = self._selected_y_axis_spec("drr")
        else:
            selection = self._cmp_selection_from_ui()
            selected = list(selection.as_pairs().values())
            compare_sources = selection.as_pairs()
            baselines = []
            pl_log = False
            cmp_log = bool(self.cmp_log_chk.isChecked())
            drr_baseline = "Self (last frame)"
            drr_baseline_which = "last"
            y_axis_spec = self._selected_y_axis_spec("cmp")
        if mode != "Compare":
            compare_sources = {}

        options = LoadOptions(
            mode=mode,
            folder=self.current_folder,
            selected_files=selected,
            baseline_files=baselines,
            pl_log_scale=pl_log,
            drr_baseline_text=drr_baseline,
            drr_baseline_which=drr_baseline_which,
            compare_log_scale=cmp_log,
            y_axis_spec=y_axis_spec,
            compare_sources=compare_sources,
        )

        self._set_stage("Loading...")
        worker = Worker(self._load_task, options)
        worker.signals.log.connect(self._append_log)
        worker.signals.result.connect(self._on_loaded)
        worker.signals.error.connect(self._show_error)
        self.thread_pool.start(worker)

    def _load_task(self, options: LoadOptions, *, progress: Signal, log: Signal) -> LoadedState:
        mode = options.mode
        folder = options.folder
        if not options.selected_files:
            raise ValueError("Select required files before loading.")
        log.emit(f"Loading {mode} ...")

        if mode == "PL":
            cube = data_io.load_pl_cube(
                folder, options.selected_files[0], log_scale=options.pl_log_scale, y_axis=options.y_axis_spec
            )
            return LoadedState(
                mode="PL",
                folder=folder,
                primary_file=options.selected_files[0],
                selected_files=options.selected_files,
                cube=cube,
                y_axis_spec=options.y_axis_spec,
            )

        if mode == "DRR":
            baseline = options.drr_baseline_text
            if baseline == "External":
                if not options.baseline_files:
                    raise ValueError("External DRR mode requires baseline files.")
                cube = data_io.load_drr_external_cube(
                    folder,
                    options.selected_files,
                    options.baseline_files,
                    baseline_which=options.drr_baseline_which,
                    y_axis=options.y_axis_spec,
                    derivative=None,
                )
                drr_mode_label = "DR/R External"
            else:
                cube = data_io.load_drr_self_cube(
                    folder,
                    options.selected_files,
                    use_first_frame=(baseline == "Self (first frame)"),
                    y_axis=options.y_axis_spec,
                    derivative=None,
                )
                drr_mode_label = "DR/R Self"

            return LoadedState(
                mode="DRR",
                folder=folder,
                primary_file=options.selected_files[0],
                selected_files=options.selected_files,
                baseline_files=options.baseline_files,
                cube=cube,
                drr_mode_label=drr_mode_label,
                drr_derivative_label="None",
                drr_baseline_text=baseline,
                drr_baseline_which=options.drr_baseline_which,
                y_axis_spec=options.y_axis_spec,
            )

        selection = data_io.CompareSelection.from_mapping(options.compare_sources)
        cubes = data_io.load_compare_cubes(folder, selection, log_scale=options.compare_log_scale, y_axis=options.y_axis_spec)
        return LoadedState(
            mode="Compare",
            folder=folder,
            selected_files=list(selection.as_pairs().values()),
            compare_cubes=cubes,
            compare_sources=selection.as_pairs(),
            y_axis_spec=options.y_axis_spec,
        )

    def _on_loaded(self, loaded: LoadedState) -> None:
        self.loaded = loaded
        self.last_plotted_mode = None
        self._last_plot_params_key = None
        self._last_plot_cube = None
        self._apply_auto_limits_for_loaded()
        self._set_stage("Loaded")
        self._update_action_states()
        self._status(f"Loaded {loaded.mode}.")
        self._plot_mode(loaded.mode, auto=True)

    def _mode_spins(self, mode: str) -> Dict[str, QDoubleSpinBox]:
        if mode == "PL":
            return self.pl_spins
        if mode == "DRR":
            return self.drr_spins
        return self.cmp_spins

    def _mode_cmap(self, mode: str) -> QComboBox:
        return self.pl_cmap if mode == "PL" else self.drr_cmap if mode == "DRR" else self.cmp_cmap

    def _mode_log(self, mode: str) -> bool:
        return bool(self.pl_log_chk.isChecked()) if mode == "PL" else bool(self.drr_log_chk.isChecked()) if mode == "DRR" else bool(self.cmp_log_chk.isChecked())

    def _mode_clip(self, mode: str) -> bool:
        return bool(self.pl_clip_chk.isChecked()) if mode == "PL" else bool(self.drr_clip_chk.isChecked()) if mode == "DRR" else bool(self.cmp_clip_chk.isChecked())

    def _mode_fix_value(self, mode: str, key: str) -> bool:
        if mode == "PL":
            checks = self.pl_fix_checks
        elif mode == "DRR":
            checks = self.drr_fix_checks
        else:
            checks = self.cmp_fix_checks
        chk = checks.get(key)
        return bool(chk.isChecked()) if chk is not None else False

    def _mode_y_axis_prefix(self, mode: str) -> str:
        return "pl" if mode == "PL" else "drr" if mode == "DRR" else "cmp"

    def _current_y_axis_spec_for_mode(self, mode: str) -> str:
        return self._selected_y_axis_spec(self._mode_y_axis_prefix(mode))

    def _ensure_loaded_matches_ui_params(self, mode: str) -> bool:
        if not self.loaded or self.loaded.mode != mode:
            return False
        current_spec = self._current_y_axis_spec_for_mode(mode)
        if mode == "DRR" and current_spec == getattr(self.loaded, "y_axis_spec", "auto"):
            return self._ensure_loaded_matches_drr_params()
        if mode == "Compare":
            selection = self._cmp_selection_from_ui()
            desired_sources = selection.as_pairs()
            if (
                current_spec == getattr(self.loaded, "y_axis_spec", "auto")
                and desired_sources == (self.loaded.compare_sources or {})
            ):
                return False
        elif current_spec == getattr(self.loaded, "y_axis_spec", "auto"):
            return False

        if mode == "PL":
            if not self.loaded.primary_file:
                raise ValueError("No PL file loaded.")
            cube = data_io.load_pl_cube(
                self.current_folder,
                self.loaded.primary_file,
                log_scale=bool(self.pl_log_chk.isChecked()),
                y_axis=current_spec,
            )
            self.loaded = LoadedState(
                mode="PL",
                folder=self.current_folder,
                primary_file=self.loaded.primary_file,
                selected_files=list(self.loaded.selected_files),
                cube=cube,
                y_axis_spec=current_spec,
            )
        elif mode == "Compare":
            selection = self._cmp_selection_from_ui()
            cubes = data_io.load_compare_cubes(
                self.current_folder,
                selection,
                log_scale=bool(self.cmp_log_chk.isChecked()),
                y_axis=current_spec,
            )
            self.loaded = LoadedState(
                mode="Compare",
                folder=self.current_folder,
                selected_files=list(selection.as_pairs().values()),
                compare_cubes=cubes,
                compare_sources=selection.as_pairs(),
                y_axis_spec=current_spec,
            )
        else:
            return self._ensure_loaded_matches_drr_params()

        self._last_plot_cube = None
        self._last_plot_params_key = None
        self._apply_auto_limits_for_loaded()
        return True

    def _apply_auto_limits_for_loaded(self) -> None:
        if not self.loaded:
            return
        mode = self.loaded.mode
        if mode == "PL" and self.loaded.cube is not None:
            cube = self.loaded.cube
        elif mode == "DRR" and self.loaded.cube is not None:
            cube = self._drr_cube_for_display()
        elif mode == "Compare" and self.loaded.compare_cubes:
            cube = next(iter(self.loaded.compare_cubes.values()))
        else:
            return

        limits = compute_auto_limits(cube, log_scale=self._mode_log(mode))
        spins = self._mode_spins(mode)
        if not self._mode_fix_value(mode, "vmin"):
            spins["vmin"].setValue(limits.vmin)
        if not self._mode_fix_value(mode, "vmax"):
            spins["vmax"].setValue(limits.vmax)
        if not self._mode_fix_value(mode, "xmin"):
            spins["xmin"].setValue(limits.xmin)
        if not self._mode_fix_value(mode, "xmax"):
            spins["xmax"].setValue(limits.xmax)
        if not self._mode_fix_value(mode, "ymin"):
            spins["ymin"].setValue(limits.ymin)
        if not self._mode_fix_value(mode, "ymax"):
            spins["ymax"].setValue(limits.ymax)
        spins["gate"].setValue(float(np.nanmedian(cube.gate)))

    def _make_params(self, mode: str, cube: DataCube) -> HeatmapParams:
        spins = self._mode_spins(mode)
        return HeatmapParams(
            title=cube.title,
            xlabel="Photon Energy (eV)",
            ylabel=cube.gate_label,
            cbar_label=cube.cbar_label,
            vmin=float(spins["vmin"].value()),
            vmax=float(spins["vmax"].value()),
            xlim=(float(spins["xmin"].value()), float(spins["xmax"].value())),
            ylim=(float(spins["ymin"].value()), float(spins["ymax"].value())),
            cmap=self._mode_cmap(mode).currentText(),
            log_scale=self._mode_log(mode),
            center_zero=(mode == "DRR" and bool(self.drr_center_zero_chk.isChecked())),
            clip_outliers=self._mode_clip(mode),
        )

    def _auto_scale_spectrum_y(self, ax, x: np.ndarray, y: np.ndarray, xlim: tuple[float, float]) -> None:
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        lo, hi = sorted((float(xlim[0]), float(xlim[1])))
        mask = (x >= lo) & (x <= hi) & np.isfinite(y)
        region = y[mask] if np.any(mask) else y[np.isfinite(y)]
        if region.size == 0:
            return
        ymin = float(np.nanmin(region))
        ymax = float(np.nanmax(region))
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            return
        if ymin == ymax:
            pad = max(1e-12, abs(ymin) * 0.05, 1.0)
        else:
            pad = max(1e-12, (ymax - ymin) * 0.08)
        ax.set_ylim(ymin - pad, ymax + pad)

    def _safe_spectrum_xlim(self, x: np.ndarray, xlim: tuple[float, float]) -> tuple[float, float]:
        x = np.asarray(x, float).ravel()
        lo, hi = float(xlim[0]), float(xlim[1])
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo = float(np.nanmin(x))
            hi = float(np.nanmax(x))
        if lo == hi:
            pad = max(1e-9, abs(lo) * 1e-6, (float(np.nanmax(x)) - float(np.nanmin(x))) * 1e-3)
            lo -= pad
            hi += pad
        return (lo, hi)

    def _plot_spectrum_with_roi(
        self,
        ax,
        cube: DataCube,
        gate_value: float,
        *,
        ylabel: str,
        xlim: tuple[float, float],
    ) -> float:
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        ax.plot(x, np.asarray(y, float), linewidth=1.3)
        ax.set_title(f"Spectrum @ {gate_used:.6g} V")
        ax.set_xlabel("Photon Energy (eV)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        lo, hi = float(xlim[0]), float(xlim[1])
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo = float(np.nanmin(x))
            hi = float(np.nanmax(x))
        if lo == hi:
            pad = max(1e-9, abs(lo) * 1e-6, (float(np.nanmax(x)) - float(np.nanmin(x))) * 1e-3)
            lo -= pad
            hi += pad
        safe_xlim = (lo, hi)
        ax.set_xlim(safe_xlim)
        self._auto_scale_spectrum_y(ax, x, y, safe_xlim)
        return gate_used

    def _set_drr_gate_spin_value(self, gate_value: float) -> None:
        spin = self.drr_spins["gate"]
        old = spin.blockSignals(True)
        try:
            spin.setValue(float(gate_value))
        finally:
            spin.blockSignals(old)

    def _set_pl_gate_spin_value(self, gate_value: float) -> None:
        spin = self.pl_spins["gate"]
        old = spin.blockSignals(True)
        try:
            spin.setValue(float(gate_value))
        finally:
            spin.blockSignals(old)

    def _set_cmp_gate_spin_value(self, gate_value: float) -> None:
        spin = self.cmp_spins["gate"]
        old = spin.blockSignals(True)
        try:
            spin.setValue(float(gate_value))
        finally:
            spin.blockSignals(old)

    def _current_drr_spectrum(self, cube: DataCube) -> tuple[float, np.ndarray, np.ndarray]:
        gate_value = float(self.drr_spins["gate"].value())
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        return gate_used, x, np.asarray(y, float).ravel()

    def _visible_x_mask(self, x: np.ndarray, spins: Dict[str, QDoubleSpinBox]) -> np.ndarray:
        x0, x1 = sorted((float(spins["xmin"].value()), float(spins["xmax"].value())))
        return (x >= x0) & (x <= x1)

    def _compute_peak_indices(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        spins: Dict[str, QDoubleSpinBox],
        prom_spin: QDoubleSpinBox,
        dist_spin: QSpinBox,
        max_spin: QSpinBox,
        mode: str,
    ) -> np.ndarray:
        mask = self._visible_x_mask(x, spins) & np.isfinite(y)
        idx_all = np.where(mask)[0]
        if idx_all.size < 3:
            return np.asarray([], dtype=int)
        y_sel = y[idx_all]
        rng = float(np.nanmax(y_sel) - np.nanmin(y_sel))
        prom_frac = float(prom_spin.value())
        prom = max(0.0, prom_frac * rng)
        distance = int(dist_spin.value())
        target = -y_sel if str(mode).lower().startswith("dip") else y_sel
        peaks_local, props = find_peaks(target, prominence=prom if prom > 0 else None, distance=max(1, distance))
        if peaks_local.size == 0:
            return np.asarray([], dtype=int)
        peaks = idx_all[peaks_local]
        max_keep = int(max_spin.value())
        if max_keep > 0 and peaks.size > max_keep:
            prom_vals = np.asarray(props.get("prominences", np.zeros(peaks_local.size)), float)
            order = np.argsort(prom_vals)[::-1][:max_keep]
            peaks = peaks[order]
        return np.asarray(np.sort(peaks), dtype=int)

    def _set_fit_n_from_found(self, spin: QSpinBox, count: int) -> None:
        n = max(int(spin.minimum()), min(int(spin.maximum()), int(count)))
        if n <= 0:
            return
        old = spin.blockSignals(True)
        try:
            spin.setValue(n)
        finally:
            spin.blockSignals(old)

    @staticmethod
    def _multi_lorentz_model(x: np.ndarray, *p: float) -> np.ndarray:
        x_arr = np.asarray(x, float)
        out = p[0] + p[1] * x_arr
        n = (len(p) - 2) // 3
        for i in range(n):
            amp = p[2 + 3 * i]
            cen = p[3 + 3 * i]
            gam = max(1e-12, p[4 + 3 * i])
            out = out + amp * (gam * gam) / ((x_arr - cen) * (x_arr - cen) + gam * gam)
        return out

    def _draw_drr_analysis_overlays(self, cube: DataCube, gate_used: float, x: np.ndarray, y: np.ndarray) -> None:
        if self._drr_spectrum_ax is None or self._drr_heatmap_ax is None:
            return
        if self._drr_heatmap_peak_artist is not None:
            try:
                self._drr_heatmap_peak_artist.remove()
            except Exception:
                pass
            self._drr_heatmap_peak_artist = None
        if self._drr_heatmap_fit_artist is not None:
            try:
                self._drr_heatmap_fit_artist.remove()
            except Exception:
                pass
            self._drr_heatmap_fit_artist = None
        if self._drr_peak_gate is not None and abs(float(gate_used) - float(self._drr_peak_gate)) > 1e-9:
            self._drr_peak_gate = None
            self._drr_peak_indices = None
        if self._drr_fit_gate is not None and abs(float(gate_used) - float(self._drr_fit_gate)) > 1e-9:
            self._drr_fit_gate = None
            self._drr_fit_x = None
            self._drr_fit_y = None
            self._drr_fit_centers = None
            self.drr_fit_status.setText("")

        if (
            self.drr_peak_show_chk.isChecked()
            and self._drr_peak_indices is not None
            and self._drr_peak_gate is not None
            and abs(float(gate_used) - float(self._drr_peak_gate)) <= 1e-9
            and self._drr_peak_indices.size > 0
        ):
            pidx = np.asarray(self._drr_peak_indices, dtype=int)
            self._drr_spectrum_ax.scatter(x[pidx], y[pidx], s=26, marker="o", facecolor="#ffd84d", edgecolor="#222", zorder=30)
            self._drr_heatmap_peak_artist = self._drr_heatmap_ax.scatter(
                x[pidx],
                np.full(pidx.size, float(gate_used)),
                s=28,
                marker="o",
                facecolor="#ffd84d",
                edgecolor="#111",
                zorder=35,
            )

        if (
            self.drr_fit_show_chk.isChecked()
            and self._drr_fit_gate is not None
            and self._drr_fit_x is not None
            and self._drr_fit_y is not None
            and abs(float(gate_used) - float(self._drr_fit_gate)) <= 1e-9
        ):
            self._drr_spectrum_ax.plot(self._drr_fit_x, self._drr_fit_y, color="#f28e2b", linewidth=1.6, zorder=28)
            if self._drr_fit_centers is not None and self._drr_fit_centers.size:
                self._drr_heatmap_fit_artist = self._drr_heatmap_ax.scatter(
                    np.asarray(self._drr_fit_centers, float),
                    np.full(int(self._drr_fit_centers.size), float(gate_used)),
                    s=34,
                    marker="x",
                    color="#f28e2b",
                    linewidths=1.5,
                    zorder=36,
                )

    def _on_drr_find_peaks(self) -> None:
        if self.last_plotted_mode != "DRR" or self._last_plot_cube is None:
            return
        gate_used, x, y = self._current_drr_spectrum(self._last_plot_cube)
        peaks = self._compute_peak_indices(
            x,
            y,
            spins=self.drr_spins,
            prom_spin=self.drr_peak_prom_spin,
            dist_spin=self.drr_peak_dist_spin,
            max_spin=self.drr_peak_max_spin,
            mode=self.drr_peak_mode_combo.currentText(),
        )
        self._drr_peak_gate = float(gate_used)
        self._drr_peak_indices = peaks
        if peaks.size > 0:
            self._set_fit_n_from_found(self.drr_fit_n_spin, int(peaks.size))
        self.drr_fit_status.setText(f"Peaks: {int(peaks.size)}")
        self._update_drr_spectrum_and_gate_line(self._last_plot_cube)

    def _on_drr_fit_lorentz(self) -> None:
        if self.last_plotted_mode != "DRR" or self._last_plot_cube is None:
            return
        gate_used, x, y = self._current_drr_spectrum(self._last_plot_cube)
        mask = self._visible_x_mask(x, self.drr_spins) & np.isfinite(y)
        if np.count_nonzero(mask) < 8:
            self.drr_fit_status.setText("Fit failed: not enough points in x-range.")
            return
        x_sel = x[mask]
        y_sel = y[mask]
        n_peaks = int(self.drr_fit_n_spin.value())
        if self._drr_peak_indices is None or self._drr_peak_gate is None or abs(float(gate_used) - float(self._drr_peak_gate)) > 1e-9:
            peak_idx = self._compute_peak_indices(
                x,
                y,
                spins=self.drr_spins,
                prom_spin=self.drr_peak_prom_spin,
                dist_spin=self.drr_peak_dist_spin,
                max_spin=self.drr_peak_max_spin,
                mode=self.drr_peak_mode_combo.currentText(),
            )
        else:
            peak_idx = np.asarray(self._drr_peak_indices, dtype=int)
        peak_idx = peak_idx[(peak_idx >= 0) & (peak_idx < x.size)]
        if peak_idx.size < n_peaks:
            order = np.argsort(np.abs(y_sel - np.nanmedian(y_sel)))[::-1]
            idx_sel = np.where(mask)[0]
            extra = idx_sel[order[: max(1, n_peaks - peak_idx.size)]]
            peak_idx = np.unique(np.concatenate([peak_idx, extra]))
        if peak_idx.size == 0:
            self.drr_fit_status.setText("Fit failed: no peak candidates.")
            return
        centers0 = np.asarray(np.sort(x[peak_idx])[:n_peaks], float)
        while centers0.size < n_peaks:
            centers0 = np.append(centers0, float(np.nanmean(x_sel)))
        base0 = float(np.nanmedian(y_sel))
        slope0 = 0.0
        y_amp = float(np.nanmax(np.abs(y_sel - base0))) if np.isfinite(np.nanmax(np.abs(y_sel - base0))) else 1.0
        x_rng = max(1e-9, float(np.nanmax(x_sel) - np.nanmin(x_sel)))
        dx = float(np.nanmedian(np.diff(x_sel))) if x_sel.size > 2 else x_rng / 100.0
        g0 = max(abs(dx) * 2.0, x_rng / 80.0, 1e-6)

        p0: list[float] = [base0, slope0]
        lo: list[float] = [float(np.nanmin(y_sel) - 3 * y_amp), -1e9]
        hi: list[float] = [float(np.nanmax(y_sel) + 3 * y_amp), 1e9]
        for c0 in centers0:
            p0.extend([y_amp * 0.7, float(c0), g0])
            lo.extend([-5 * y_amp, float(np.nanmin(x_sel)), max(abs(dx) * 0.25, 1e-8)])
            hi.extend([5 * y_amp, float(np.nanmax(x_sel)), x_rng])
        try:
            popt, _ = curve_fit(
                self._multi_lorentz_model,
                x_sel,
                y_sel,
                p0=np.asarray(p0, float),
                bounds=(np.asarray(lo, float), np.asarray(hi, float)),
                maxfev=50000,
            )
        except Exception as exc:
            self.drr_fit_status.setText(f"Fit failed: {exc}")
            return
        y_fit = self._multi_lorentz_model(x, *popt)
        centers_fit = np.asarray([popt[3 + 3 * i] for i in range(n_peaks)], float)
        self._drr_fit_gate = float(gate_used)
        self._drr_fit_x = x.copy()
        self._drr_fit_y = np.asarray(y_fit, float)
        self._drr_fit_centers = np.asarray(np.sort(centers_fit), float)
        self.drr_fit_status.setText(
            "Fit centers: " + ", ".join(f"{c:.4f}" for c in np.asarray(self._drr_fit_centers, float)[:4])
        )
        self._update_drr_spectrum_and_gate_line(self._last_plot_cube)

    def _on_drr_clear_fit(self) -> None:
        self._drr_fit_gate = None
        self._drr_fit_x = None
        self._drr_fit_y = None
        self._drr_fit_centers = None
        self.drr_fit_status.setText("")
        if self.last_plotted_mode == "DRR" and self._last_plot_cube is not None:
            self._update_drr_spectrum_and_gate_line(self._last_plot_cube)

    def _on_drr_analysis_view_changed(self) -> None:
        if self.last_plotted_mode == "DRR" and self._last_plot_cube is not None:
            self._update_drr_spectrum_and_gate_line(self._last_plot_cube)

    def _update_drr_analysis_text(self, gate_used: float, x: np.ndarray, y: np.ndarray) -> None:
        lines: list[str] = []
        if (
            self._drr_peak_indices is not None
            and self._drr_peak_gate is not None
            and abs(float(gate_used) - float(self._drr_peak_gate)) <= 1e-9
            and self._drr_peak_indices.size > 0
        ):
            pidx = np.asarray(self._drr_peak_indices, dtype=int)
            pairs = [f"({float(x[i]):.5f}, {float(y[i]):.4g})" for i in pidx]
            lines.append("Found: " + "; ".join(pairs))
        else:
            lines.append("Found: none")
        if (
            self._drr_fit_centers is not None
            and self._drr_fit_centers.size > 0
            and self._drr_fit_gate is not None
            and abs(float(gate_used) - float(self._drr_fit_gate)) <= 1e-9
            and self._drr_fit_x is not None
            and self._drr_fit_y is not None
        ):
            yc = np.interp(np.asarray(self._drr_fit_centers, float), np.asarray(self._drr_fit_x, float), np.asarray(self._drr_fit_y, float))
            fit_pairs = [f"({float(cx):.5f}, {float(cy):.4g})" for cx, cy in zip(np.asarray(self._drr_fit_centers, float), np.asarray(yc, float))]
            lines.append("Fit: " + "; ".join(fit_pairs))
        else:
            lines.append("Fit: none")
        self.drr_analysis_text.setPlainText("\n".join(lines))

    def _current_pl_spectrum(self, cube: DataCube) -> tuple[float, np.ndarray, np.ndarray]:
        gate_value = float(self.pl_spins["gate"].value())
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        return gate_used, x, np.asarray(y, float).ravel()

    def _draw_pl_analysis_overlays(self, gate_used: float, x: np.ndarray, y: np.ndarray) -> None:
        if self._pl_spectrum_ax is None or self._pl_heatmap_ax is None:
            return
        if self._pl_heatmap_peak_artist is not None:
            try:
                self._pl_heatmap_peak_artist.remove()
            except Exception:
                pass
            self._pl_heatmap_peak_artist = None
        if self._pl_heatmap_fit_artist is not None:
            try:
                self._pl_heatmap_fit_artist.remove()
            except Exception:
                pass
            self._pl_heatmap_fit_artist = None
        if self._pl_peak_gate is not None and abs(float(gate_used) - float(self._pl_peak_gate)) > 1e-9:
            self._pl_peak_gate = None
            self._pl_peak_indices = None
        if self._pl_fit_gate is not None and abs(float(gate_used) - float(self._pl_fit_gate)) > 1e-9:
            self._pl_fit_gate = None
            self._pl_fit_x = None
            self._pl_fit_y = None
            self._pl_fit_centers = None
            self.pl_fit_status.setText("")
        if (
            self.pl_peak_show_chk.isChecked()
            and self._pl_peak_indices is not None
            and self._pl_peak_gate is not None
            and abs(float(gate_used) - float(self._pl_peak_gate)) <= 1e-9
            and self._pl_peak_indices.size > 0
        ):
            pidx = np.asarray(self._pl_peak_indices, dtype=int)
            self._pl_spectrum_ax.scatter(x[pidx], y[pidx], s=26, marker="o", facecolor="#ffd84d", edgecolor="#222", zorder=30)
            self._pl_heatmap_peak_artist = self._pl_heatmap_ax.scatter(
                x[pidx],
                np.full(pidx.size, float(gate_used)),
                s=28,
                marker="o",
                facecolor="#ffd84d",
                edgecolor="#111",
                zorder=35,
            )
        if (
            self.pl_fit_show_chk.isChecked()
            and self._pl_fit_gate is not None
            and self._pl_fit_x is not None
            and self._pl_fit_y is not None
            and abs(float(gate_used) - float(self._pl_fit_gate)) <= 1e-9
        ):
            self._pl_spectrum_ax.plot(self._pl_fit_x, self._pl_fit_y, color="#f28e2b", linewidth=1.6, zorder=28)
            if self._pl_fit_centers is not None and self._pl_fit_centers.size:
                self._pl_heatmap_fit_artist = self._pl_heatmap_ax.scatter(
                    np.asarray(self._pl_fit_centers, float),
                    np.full(int(self._pl_fit_centers.size), float(gate_used)),
                    s=34,
                    marker="x",
                    color="#f28e2b",
                    linewidths=1.5,
                    zorder=36,
                )

    def _update_pl_spectrum_with_analysis(self, cube: DataCube) -> None:
        if self._pl_spectrum_ax is None:
            return
        gate_value = float(self.pl_spins["gate"].value())
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        self._pl_spectrum_ax.clear()
        self._pl_spectrum_ax.plot(x, np.asarray(y, float), linewidth=1.3)
        self._pl_spectrum_ax.set_title(f"Spectrum @ {gate_used:.6g} V")
        self._pl_spectrum_ax.set_xlabel("Photon Energy (eV)")
        self._pl_spectrum_ax.set_ylabel(cube.cbar_label)
        self._pl_spectrum_ax.grid(alpha=0.25)
        xlim = (float(self.pl_spins["xmin"].value()), float(self.pl_spins["xmax"].value()))
        self._pl_spectrum_ax.set_xlim(xlim)
        self._auto_scale_spectrum_y(self._pl_spectrum_ax, x, y, xlim)
        self._draw_pl_analysis_overlays(gate_used, x, np.asarray(y, float))
        self._update_pl_analysis_text(gate_used, x, np.asarray(y, float))
        self.canvas.draw_idle()

    def _ensure_pl_gate_line(self, cube: DataCube, gate_value: float) -> None:
        if self._pl_heatmap_ax is None:
            return
        gate = np.asarray(cube.gate, float).ravel()
        gate_clamped = float(np.clip(gate_value, float(np.nanmin(gate)), float(np.nanmax(gate))))
        if self._pl_gate_line is None or getattr(self._pl_gate_line, "axes", None) is not self._pl_heatmap_ax:
            self._pl_gate_line = self._pl_heatmap_ax.axhline(
                y=gate_clamped,
                lw=1.2,
                alpha=0.9,
                color="#222",
                linestyle="--",
                zorder=20,
            )
        else:
            self._pl_gate_line.set_ydata([gate_clamped, gate_clamped])
            self._pl_gate_line.set_linestyle("--")

    def _update_pl_spectrum_and_gate_line(self, cube: DataCube) -> None:
        if self._pl_spectrum_ax is None:
            return
        gate_value = float(self.pl_spins["gate"].value())
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        self._pl_spectrum_ax.clear()
        self._pl_spectrum_ax.plot(x, np.asarray(y, float), linewidth=1.3)
        self._pl_spectrum_ax.set_title(f"Spectrum @ {gate_used:.6g} V")
        self._pl_spectrum_ax.set_xlabel("Photon Energy (eV)")
        self._pl_spectrum_ax.set_ylabel(cube.cbar_label)
        self._pl_spectrum_ax.grid(alpha=0.25)
        xlim = self._safe_spectrum_xlim(
            x,
            (float(self.pl_spins["xmin"].value()), float(self.pl_spins["xmax"].value())),
        )
        self._pl_spectrum_ax.set_xlim(xlim)
        self._auto_scale_spectrum_y(self._pl_spectrum_ax, x, y, xlim)
        self._set_pl_gate_spin_value(gate_used)
        self._ensure_pl_gate_line(cube, gate_used)
        self._draw_pl_analysis_overlays(gate_used, x, np.asarray(y, float))
        self._update_pl_analysis_text(gate_used, x, np.asarray(y, float))
        self.canvas.draw_idle()

    def _on_pl_find_peaks(self) -> None:
        if self.last_plotted_mode != "PL" or self._pl_last_plot_cube is None:
            return
        gate_used, x, y = self._current_pl_spectrum(self._pl_last_plot_cube)
        peaks = self._compute_peak_indices(
            x,
            y,
            spins=self.pl_spins,
            prom_spin=self.pl_peak_prom_spin,
            dist_spin=self.pl_peak_dist_spin,
            max_spin=self.pl_peak_max_spin,
            mode=self.pl_peak_mode_combo.currentText(),
        )
        self._pl_peak_gate = float(gate_used)
        self._pl_peak_indices = peaks
        if peaks.size > 0:
            self._set_fit_n_from_found(self.pl_fit_n_spin, int(peaks.size))
        self.pl_fit_status.setText(f"Peaks: {int(peaks.size)}")
        self._update_pl_spectrum_with_analysis(self._pl_last_plot_cube)

    def _on_pl_fit_lorentz(self) -> None:
        if self.last_plotted_mode != "PL" or self._pl_last_plot_cube is None:
            return
        gate_used, x, y = self._current_pl_spectrum(self._pl_last_plot_cube)
        mask = self._visible_x_mask(x, self.pl_spins) & np.isfinite(y)
        if np.count_nonzero(mask) < 8:
            self.pl_fit_status.setText("Fit failed: not enough points in x-range.")
            return
        x_sel = x[mask]
        y_sel = y[mask]
        n_peaks = int(self.pl_fit_n_spin.value())
        if self._pl_peak_indices is None or self._pl_peak_gate is None or abs(float(gate_used) - float(self._pl_peak_gate)) > 1e-9:
            peak_idx = self._compute_peak_indices(
                x,
                y,
                spins=self.pl_spins,
                prom_spin=self.pl_peak_prom_spin,
                dist_spin=self.pl_peak_dist_spin,
                max_spin=self.pl_peak_max_spin,
                mode=self.pl_peak_mode_combo.currentText(),
            )
        else:
            peak_idx = np.asarray(self._pl_peak_indices, dtype=int)
        peak_idx = peak_idx[(peak_idx >= 0) & (peak_idx < x.size)]
        if peak_idx.size < n_peaks:
            order = np.argsort(np.abs(y_sel - np.nanmedian(y_sel)))[::-1]
            idx_sel = np.where(mask)[0]
            extra = idx_sel[order[: max(1, n_peaks - peak_idx.size)]]
            peak_idx = np.unique(np.concatenate([peak_idx, extra]))
        if peak_idx.size == 0:
            self.pl_fit_status.setText("Fit failed: no peak candidates.")
            return
        centers0 = np.asarray(np.sort(x[peak_idx])[:n_peaks], float)
        while centers0.size < n_peaks:
            centers0 = np.append(centers0, float(np.nanmean(x_sel)))
        base0 = float(np.nanmedian(y_sel))
        slope0 = 0.0
        y_amp = float(np.nanmax(np.abs(y_sel - base0))) if np.isfinite(np.nanmax(np.abs(y_sel - base0))) else 1.0
        x_rng = max(1e-9, float(np.nanmax(x_sel) - np.nanmin(x_sel)))
        dx = float(np.nanmedian(np.diff(x_sel))) if x_sel.size > 2 else x_rng / 100.0
        g0 = max(abs(dx) * 2.0, x_rng / 80.0, 1e-6)
        p0: list[float] = [base0, slope0]
        lo: list[float] = [float(np.nanmin(y_sel) - 3 * y_amp), -1e9]
        hi: list[float] = [float(np.nanmax(y_sel) + 3 * y_amp), 1e9]
        for c0 in centers0:
            p0.extend([y_amp * 0.7, float(c0), g0])
            lo.extend([-5 * y_amp, float(np.nanmin(x_sel)), max(abs(dx) * 0.25, 1e-8)])
            hi.extend([5 * y_amp, float(np.nanmax(x_sel)), x_rng])
        try:
            popt, _ = curve_fit(
                self._multi_lorentz_model,
                x_sel,
                y_sel,
                p0=np.asarray(p0, float),
                bounds=(np.asarray(lo, float), np.asarray(hi, float)),
                maxfev=50000,
            )
        except Exception as exc:
            self.pl_fit_status.setText(f"Fit failed: {exc}")
            return
        y_fit = self._multi_lorentz_model(x, *popt)
        centers_fit = np.asarray([popt[3 + 3 * i] for i in range(n_peaks)], float)
        self._pl_fit_gate = float(gate_used)
        self._pl_fit_x = x.copy()
        self._pl_fit_y = np.asarray(y_fit, float)
        self._pl_fit_centers = np.asarray(np.sort(centers_fit), float)
        self.pl_fit_status.setText("Fit centers: " + ", ".join(f"{c:.4f}" for c in np.asarray(self._pl_fit_centers, float)[:4]))
        self._update_pl_spectrum_with_analysis(self._pl_last_plot_cube)

    def _on_pl_clear_fit(self) -> None:
        self._pl_fit_gate = None
        self._pl_fit_x = None
        self._pl_fit_y = None
        self._pl_fit_centers = None
        self.pl_fit_status.setText("")
        if self.last_plotted_mode == "PL" and self._pl_last_plot_cube is not None:
            self._update_pl_spectrum_with_analysis(self._pl_last_plot_cube)

    def _on_pl_analysis_view_changed(self) -> None:
        if self.last_plotted_mode == "PL" and self._pl_last_plot_cube is not None:
            self._update_pl_spectrum_and_gate_line(self._pl_last_plot_cube)

    def _on_pl_gate_changed(self) -> None:
        if self.last_plotted_mode == "PL" and self._pl_last_plot_cube is not None:
            self._update_pl_spectrum_and_gate_line(self._pl_last_plot_cube)

    def _plot_compare_linecut(
        self,
        ax: Any,
        cubes: Dict[str, DataCube],
        *,
        gate_value: float,
        xlim: tuple[float, float],
    ) -> float:
        gate_used_values: list[float] = []
        for key in [label for label in COMPARE_PANEL_ORDER if label in cubes]:
            cube = cubes[key]
            gate_used, y = nearest_gate_spectrum(cube, gate_value)
            x = np.asarray(cube.energy, float).ravel()
            ax.plot(x, np.asarray(y, float), linewidth=1.3, label=key)
            gate_used_values.append(float(gate_used))
        gate_used = float(np.median(gate_used_values)) if gate_used_values else float(gate_value)
        ax.set_title(f"Compare Spectra @ {gate_used:.6g} V")
        ax.set_xlabel("Photon Energy (eV)")
        ax.set_ylabel("PL (a.u.)")
        ax.grid(alpha=0.25)
        safe_xlim = self._safe_spectrum_xlim(np.asarray(next(iter(cubes.values())).energy, float), xlim)
        ax.set_xlim(safe_xlim)
        finite_lines = [np.asarray(line.get_ydata(), float) for line in ax.lines if len(line.get_ydata())]
        if finite_lines:
            y_all = np.concatenate([line[np.isfinite(line)] for line in finite_lines if np.any(np.isfinite(line))])
            if y_all.size:
                ymin = float(np.nanmin(y_all))
                ymax = float(np.nanmax(y_all))
                pad = max(1e-12, (ymax - ymin) * 0.08) if ymax != ymin else max(1e-12, abs(ymin) * 0.05, 1.0)
                ax.set_ylim(ymin - pad, ymax + pad)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        return gate_used

    def _update_pl_analysis_text(self, gate_used: float, x: np.ndarray, y: np.ndarray) -> None:
        lines: list[str] = []
        if (
            self._pl_peak_indices is not None
            and self._pl_peak_gate is not None
            and abs(float(gate_used) - float(self._pl_peak_gate)) <= 1e-9
            and self._pl_peak_indices.size > 0
        ):
            pidx = np.asarray(self._pl_peak_indices, dtype=int)
            pairs = [f"({float(x[i]):.5f}, {float(y[i]):.4g})" for i in pidx]
            lines.append("Found: " + "; ".join(pairs))
        else:
            lines.append("Found: none")
        if (
            self._pl_fit_centers is not None
            and self._pl_fit_centers.size > 0
            and self._pl_fit_gate is not None
            and abs(float(gate_used) - float(self._pl_fit_gate)) <= 1e-9
            and self._pl_fit_x is not None
            and self._pl_fit_y is not None
        ):
            yc = np.interp(np.asarray(self._pl_fit_centers, float), np.asarray(self._pl_fit_x, float), np.asarray(self._pl_fit_y, float))
            fit_pairs = [f"({float(cx):.5f}, {float(cy):.4g})" for cx, cy in zip(np.asarray(self._pl_fit_centers, float), np.asarray(yc, float))]
            lines.append("Fit: " + "; ".join(fit_pairs))
        else:
            lines.append("Fit: none")
        self.pl_analysis_text.setPlainText("\n".join(lines))

    def _remove_nearest_drr_peak(self, x_click: float) -> bool:
        if self._last_plot_cube is None or self._drr_peak_indices is None or self._drr_peak_indices.size == 0:
            return False
        gate_used, _y = nearest_gate_spectrum(self._last_plot_cube, float(self.drr_spins["gate"].value()))
        if self._drr_peak_gate is None or abs(float(gate_used) - float(self._drr_peak_gate)) > 1e-9:
            return False
        x = np.asarray(self._last_plot_cube.energy, float).ravel()
        pidx = np.asarray(self._drr_peak_indices, dtype=int)
        j = int(np.argmin(np.abs(x[pidx] - float(x_click))))
        self._drr_peak_indices = np.delete(pidx, j)
        self.drr_fit_status.setText(f"Peaks: {int(self._drr_peak_indices.size)}")
        self._update_drr_spectrum_and_gate_line(self._last_plot_cube)
        return True

    def _remove_nearest_pl_peak(self, x_click: float) -> bool:
        if self._pl_last_plot_cube is None or self._pl_peak_indices is None or self._pl_peak_indices.size == 0:
            return False
        gate_used, _y = nearest_gate_spectrum(self._pl_last_plot_cube, float(self.pl_spins["gate"].value()))
        if self._pl_peak_gate is None or abs(float(gate_used) - float(self._pl_peak_gate)) > 1e-9:
            return False
        x = np.asarray(self._pl_last_plot_cube.energy, float).ravel()
        pidx = np.asarray(self._pl_peak_indices, dtype=int)
        j = int(np.argmin(np.abs(x[pidx] - float(x_click))))
        self._pl_peak_indices = np.delete(pidx, j)
        self.pl_fit_status.setText(f"Peaks: {int(self._pl_peak_indices.size)}")
        self._update_pl_spectrum_with_analysis(self._pl_last_plot_cube)
        return True

    def _remove_peak_from_drr_heatmap_click(self, x_click: float, y_click: float) -> bool:
        if (
            self._last_plot_cube is None
            or self._drr_peak_indices is None
            or self._drr_peak_indices.size == 0
            or self._drr_peak_gate is None
        ):
            return False
        gate_axis = np.asarray(self._last_plot_cube.gate, float).ravel()
        if gate_axis.size == 0:
            return False
        gmin, gmax = float(np.nanmin(gate_axis)), float(np.nanmax(gate_axis))
        grng = max(1e-12, gmax - gmin)
        dg = float(np.nanmedian(np.abs(np.diff(gate_axis)))) if gate_axis.size > 1 else grng * 0.02
        tol = max(0.02 * grng, 0.75 * max(dg, 1e-9))
        if abs(float(y_click) - float(self._drr_peak_gate)) > tol:
            return False
        return self._remove_nearest_drr_peak(float(x_click))

    def _remove_peak_from_pl_heatmap_click(self, x_click: float, y_click: float) -> bool:
        if (
            self._pl_last_plot_cube is None
            or self._pl_peak_indices is None
            or self._pl_peak_indices.size == 0
            or self._pl_peak_gate is None
        ):
            return False
        gate_axis = np.asarray(self._pl_last_plot_cube.gate, float).ravel()
        if gate_axis.size == 0:
            return False
        gmin, gmax = float(np.nanmin(gate_axis)), float(np.nanmax(gate_axis))
        grng = max(1e-12, gmax - gmin)
        dg = float(np.nanmedian(np.abs(np.diff(gate_axis)))) if gate_axis.size > 1 else grng * 0.02
        tol = max(0.02 * grng, 0.75 * max(dg, 1e-9))
        if abs(float(y_click) - float(self._pl_peak_gate)) > tol:
            return False
        return self._remove_nearest_pl_peak(float(x_click))

    def _plot_mode(self, mode: str, *, auto: bool = False) -> None:
        try:
            if not self.loaded or self.loaded.mode != mode:
                self._show_error("Load data for this tab before plotting.")
                return

            self._ensure_loaded_matches_ui_params(mode)
            plot_key = self._current_plot_params_key(mode)
            gate_only_update = mode == "DRR" and self._is_drr_gate_only_change(plot_key)
            if gate_only_update and self._last_plot_cube is not None:
                self._update_drr_spectrum_and_gate_line(self._last_plot_cube)
                self._last_plot_params_key = plot_key
                self.last_plotted_mode = mode
                self._set_stage("Plotted")
                self._update_action_states()
                self._status(f"Plotted DRR (gate update).")
                return

            self.figure.clear()
            self._cmp_heatmap_axes = {}
            self._cmp_gate_lines = {}
            self._cmp_linecut_ax = None
            if mode == "PL" and self.loaded.cube is not None:
                plot_cube = self.loaded.cube
                params = self._make_params(mode, plot_cube)
                gs = self.figure.add_gridspec(
                    nrows=2,
                    ncols=2,
                    width_ratios=[1.0, 0.035],
                    height_ratios=[1.0, 1.0],
                    wspace=0.12,
                    hspace=0.28,
                )
                ax1 = self.figure.add_subplot(gs[0, 0])
                cax = self.figure.add_subplot(gs[0, 1])
                ax2 = self.figure.add_subplot(gs[1, 0], sharex=ax1)
                im = plot_pl(ax1, plot_cube, params)
                self.figure.colorbar(im, cax=cax, label=params.cbar_label)
                self._pl_heatmap_ax = ax1
                self._pl_spectrum_ax = ax2
                self._pl_last_plot_cube = plot_cube
                self._pl_gate_line = None
                self._pl_heatmap_peak_artist = None
                self._pl_heatmap_fit_artist = None
                self._update_pl_spectrum_and_gate_line(plot_cube)
            elif mode == "DRR" and self.loaded.cube is not None:
                self._pl_heatmap_ax = None
                self._pl_spectrum_ax = None
                self._pl_last_plot_cube = None
                self._pl_gate_line = None
                plot_cube = self._drr_cube_for_display()
                self.loaded.drr_derivative_label = self.drr_derivative_combo.currentText()
                params = self._make_params(mode, plot_cube)
                gs = self.figure.add_gridspec(
                    nrows=2,
                    ncols=2,
                    width_ratios=[1.0, 0.035],
                    height_ratios=[1.0, 1.0],
                    wspace=0.12,
                    hspace=0.28,
                )
                ax1 = self.figure.add_subplot(gs[0, 0])
                cax = self.figure.add_subplot(gs[0, 1])
                ax2 = self.figure.add_subplot(gs[1, 0], sharex=ax1)
                im = plot_drr(ax1, plot_cube, params)
                self.figure.colorbar(im, cax=cax, label=params.cbar_label)
                gate_val = float(self._mode_spins(mode)["gate"].value())
                gate_used = self._plot_spectrum_with_roi(
                    ax2, plot_cube, gate_val, ylabel=params.cbar_label, xlim=params.xlim
                )
                self._drr_heatmap_ax = ax1
                self._drr_spectrum_ax = ax2
                self._drr_heatmap_peak_artist = None
                self._drr_heatmap_fit_artist = None
                self._set_drr_gate_spin_value(gate_used)
                self._update_drr_spectrum_and_gate_line(plot_cube)
            elif mode == "Compare" and self.loaded.compare_cubes:
                self._pl_heatmap_ax = None
                self._pl_spectrum_ax = None
                self._pl_last_plot_cube = None
                cubes = {
                    key: self.loaded.compare_cubes[key]
                    for key in COMPARE_PANEL_ORDER
                    if key in self.loaded.compare_cubes
                }
                if len(cubes) < 2:
                    raise ValueError("Compare mode needs at least two visible channels.")
                first = next(iter(cubes.values()))
                params = self._make_params(mode, first)
                n = len(cubes)
                if n <= 2:
                    gs = self.figure.add_gridspec(
                        nrows=2,
                        ncols=n + 1,
                        width_ratios=([1.0] * n) + [0.035],
                        height_ratios=[1.0, 0.95],
                        wspace=0.12,
                        hspace=0.30,
                    )
                    heat_axes = [self.figure.add_subplot(gs[0, idx]) for idx in range(n)]
                    line_ax = self.figure.add_subplot(gs[1, :n], sharex=heat_axes[0])
                    cax = self.figure.add_subplot(gs[0, n])
                else:
                    gs = self.figure.add_gridspec(
                        nrows=3,
                        ncols=3,
                        width_ratios=[1.0, 1.0, 0.035],
                        height_ratios=[1.0, 1.0, 0.95],
                        wspace=0.12,
                        hspace=0.30,
                    )
                    heat_axes = [
                        self.figure.add_subplot(gs[0, 0]),
                        self.figure.add_subplot(gs[0, 1]),
                        self.figure.add_subplot(gs[1, 0]),
                        self.figure.add_subplot(gs[1, 1]),
                    ]
                    line_ax = self.figure.add_subplot(gs[2, :2], sharex=heat_axes[0])
                    cax = self.figure.add_subplot(gs[0:2, 2])
                images = []
                for ax, key in zip(heat_axes, cubes.keys()):
                    im = plot_compare_panel(ax, key, cubes[key], params)
                    images.append(im)
                    self._cmp_heatmap_axes[key] = ax
                if images:
                    self.figure.colorbar(images[0], cax=cax, label=params.cbar_label)
                gate_used = self._plot_compare_linecut(
                    line_ax,
                    cubes,
                    gate_value=float(self.cmp_spins["gate"].value()),
                    xlim=(float(self.cmp_spins["xmin"].value()), float(self.cmp_spins["xmax"].value())),
                )
                self._cmp_linecut_ax = line_ax
                self._set_cmp_gate_spin_value(gate_used)
                self._ensure_cmp_gate_lines(cubes, gate_used)
            else:
                raise ValueError("No loaded data to plot.")

            if mode == "Compare":
                self.figure.tight_layout()
            self.canvas.draw_idle()
            self._last_plot_params_key = plot_key
            self._last_plot_cube = plot_cube if mode == "DRR" else None
            self.last_plotted_mode = mode
            self._set_stage("Plotted")
            self._update_action_states()
            if mode == "DRR":
                self._status(
                    f"Plot: baseline={self._drr_baseline_key()}, deriv={self.drr_derivative_combo.currentText()}, "
                    f"SG(win={int(self.drr_sg_window_spin.value())}, order={int(self.drr_sg_poly_spin.value())})"
                )
            else:
                self._status(f"Plotted {mode}.")
            if not auto:
                self._append_log(f"Plotted {mode}.")
        except Exception as exc:
            self._show_error(str(exc))

    def _drr_baseline_key(self) -> str:
        text = self.drr_baseline_combo.currentText()
        if text == "Self (first frame)":
            return "self_first"
        if text == "External":
            return f"external_{self.drr_baseline_combine_combo.currentText()}"
        return "self_last"

    def _read_drr_params(self) -> Dict[str, Any]:
        s = self.drr_spins
        return {
            "baseline_mode": self.drr_baseline_combo.currentText(),
            "baseline_which": self.drr_baseline_combine_combo.currentText(),
            "baseline_files": tuple(self.drr_baseline_files_manual),
            "selected_files": tuple(self.drr_selected_files),
            "y_axis_spec": self._selected_y_axis_spec("drr"),
            "derivative": self.drr_derivative_combo.currentText(),
            "sg_window": int(self.drr_sg_window_spin.value()),
            "sg_poly": int(self.drr_sg_poly_spin.value()),
            "cmap": self.drr_cmap.currentText(),
            "vmin": float(s["vmin"].value()),
            "vmax": float(s["vmax"].value()),
            "xmin": float(s["xmin"].value()),
            "xmax": float(s["xmax"].value()),
            "ymin": float(s["ymin"].value()),
            "ymax": float(s["ymax"].value()),
            "gate": float(s["gate"].value()),
            "log": bool(self.drr_log_chk.isChecked()),
            "clip": bool(self.drr_clip_chk.isChecked()),
            "center_zero": bool(self.drr_center_zero_chk.isChecked()),
        }

    def _current_plot_params_key(self, mode: str) -> tuple[Any, ...]:
        if mode == "Compare":
            return (
                mode,
                tuple(self._cmp_current_mapping().items()),
                tuple(self._cmp_visible_channels()),
                self._current_y_axis_spec_for_mode(mode),
                self.cmp_cmap.currentText(),
                float(self.cmp_spins["vmin"].value()),
                float(self.cmp_spins["vmax"].value()),
                float(self.cmp_spins["xmin"].value()),
                float(self.cmp_spins["xmax"].value()),
                float(self.cmp_spins["ymin"].value()),
                float(self.cmp_spins["ymax"].value()),
                float(self.cmp_spins["gate"].value()),
                bool(self.cmp_log_chk.isChecked()),
                bool(self.cmp_clip_chk.isChecked()),
            )
        if mode != "DRR":
            return (mode, int(self.tabs.currentIndex()), self.last_plotted_mode, self._current_y_axis_spec_for_mode(mode))
        p = self._read_drr_params()
        return (
            "DRR", p["baseline_mode"], p["baseline_which"], p["baseline_files"], p["selected_files"],
            p["y_axis_spec"], p["derivative"], p["sg_window"], p["sg_poly"], p["cmap"], p["vmin"], p["vmax"],
            p["xmin"], p["xmax"], p["ymin"], p["ymax"], p["gate"], p["log"], p["clip"], p["center_zero"],
        )

    def _is_drr_gate_only_change(self, new_key: tuple[Any, ...]) -> bool:
        if self._last_plot_params_key is None or self._last_plot_cube is None:
            return False
        if len(new_key) != len(self._last_plot_params_key):
            return False
        gate_idx = 16
        return (
            new_key[:gate_idx] == self._last_plot_params_key[:gate_idx]
            and new_key[gate_idx + 1 :] == self._last_plot_params_key[gate_idx + 1 :]
            and new_key[gate_idx] != self._last_plot_params_key[gate_idx]
        )

    def _ensure_loaded_matches_drr_params(self) -> bool:
        if not self.loaded or self.loaded.mode != "DRR":
            return False
        p = self._read_drr_params()
        selected = list(p["selected_files"])
        baselines = list(p["baseline_files"])
        baseline_text = p["baseline_mode"]
        y_axis_spec = str(p["y_axis_spec"])
        which_map = {
            "Last frame of each baseline file": "last",
            "First frame of each baseline file": "first",
            "Average all baseline files": "all",
        }
        baseline_which = which_map.get(str(p["baseline_which"]), "last")
        needs_reload = (
            selected != list(self.loaded.selected_files)
            or baseline_text != self.loaded.drr_baseline_text
            or baseline_which != self.loaded.drr_baseline_which
            or baselines != list(self.loaded.baseline_files)
            or y_axis_spec != getattr(self.loaded, "y_axis_spec", "auto")
        )
        if needs_reload:
            if baseline_text == "External":
                if not baselines:
                    raise ValueError("External DRR mode requires baseline files.")
                cube = data_io.load_drr_external_cube(
                    self.current_folder,
                    selected,
                    baselines,
                    baseline_which=baseline_which,
                    y_axis=y_axis_spec,
                    derivative=None,
                )
                mode_label = "DR/R External"
            else:
                cube = data_io.load_drr_self_cube(
                    self.current_folder,
                    selected,
                    use_first_frame=(baseline_text == "Self (first frame)"),
                    y_axis=y_axis_spec,
                    derivative=None,
                )
                mode_label = "DR/R Self"
            self.loaded = LoadedState(
                mode="DRR",
                folder=self.current_folder,
                primary_file=(selected[0] if selected else None),
                selected_files=selected,
                baseline_files=baselines,
                cube=cube,
                drr_mode_label=mode_label,
                drr_derivative_label=self.drr_derivative_combo.currentText(),
                drr_baseline_text=baseline_text,
                drr_baseline_which=baseline_which,
                y_axis_spec=y_axis_spec,
            )
            self._last_plot_cube = None
            self._last_plot_params_key = None
            self._apply_auto_limits_for_loaded()
            return True
        return False

    def _ensure_gate_line(self, cube: DataCube, gate_value: float) -> None:
        if self._drr_heatmap_ax is None:
            return
        gate = np.asarray(cube.gate, float).ravel()
        gate_clamped = float(np.clip(gate_value, float(np.nanmin(gate)), float(np.nanmax(gate))))
        if self._gate_line is None or getattr(self._gate_line, "axes", None) is not self._drr_heatmap_ax:
            self._gate_line = self._drr_heatmap_ax.axhline(
                y=gate_clamped,
                lw=1.2,
                alpha=0.9,
                color="#222",
                linestyle="--",
                zorder=20,
            )
        else:
            self._gate_line.set_ydata([gate_clamped, gate_clamped])
            self._gate_line.set_linestyle("--")

    def _update_drr_spectrum_and_gate_line(self, cube: DataCube) -> None:
        if self._drr_spectrum_ax is None:
            return
        gate_value = float(self.drr_spins["gate"].value())
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        self._drr_spectrum_ax.clear()
        self._drr_spectrum_ax.plot(x, np.asarray(y, float), linewidth=1.3)
        self._drr_spectrum_ax.set_title(f"Spectrum @ {gate_used:.6g} V")
        self._drr_spectrum_ax.set_xlabel("Photon Energy (eV)")
        self._drr_spectrum_ax.set_ylabel(cube.cbar_label)
        self._drr_spectrum_ax.grid(alpha=0.25)
        xlim = self._safe_spectrum_xlim(
            x,
            (float(self.drr_spins["xmin"].value()), float(self.drr_spins["xmax"].value())),
        )
        self._drr_spectrum_ax.set_xlim(xlim)
        self._auto_scale_spectrum_y(self._drr_spectrum_ax, x, y, xlim)
        self._set_drr_gate_spin_value(gate_used)
        self._ensure_gate_line(cube, gate_used)
        self._draw_drr_analysis_overlays(cube, gate_used, x, np.asarray(y, float))
        self._update_drr_analysis_text(gate_used, x, np.asarray(y, float))
        self.canvas.draw_idle()

    def _ensure_cmp_gate_lines(self, cubes: Dict[str, DataCube], gate_value: float) -> None:
        active_keys = set(cubes.keys())
        for key in list(self._cmp_gate_lines.keys()):
            if key in active_keys and key in self._cmp_heatmap_axes:
                continue
            line = self._cmp_gate_lines.pop(key, None)
            if line is not None:
                try:
                    line.remove()
                except Exception:
                    pass
        for key, cube in cubes.items():
            ax = self._cmp_heatmap_axes.get(key)
            if ax is None:
                continue
            gate = np.asarray(cube.gate, float).ravel()
            gate_clamped = float(np.clip(gate_value, float(np.nanmin(gate)), float(np.nanmax(gate))))
            line = self._cmp_gate_lines.get(key)
            if line is None or getattr(line, "axes", None) is not ax:
                self._cmp_gate_lines[key] = ax.axhline(
                    y=gate_clamped,
                    lw=1.2,
                    alpha=0.95,
                    color="#222",
                    linestyle="--",
                    zorder=50,
                )
            else:
                line.set_ydata([gate_clamped, gate_clamped])
                line.set_linestyle("--")

    def _on_canvas_motion(self, event: Any) -> None:
        if self.last_plotted_mode == "DRR":
            heatmap_ax = self._drr_heatmap_ax
            cube = self._last_plot_cube
        elif self.last_plotted_mode == "PL":
            heatmap_ax = self._pl_heatmap_ax
            cube = self._pl_last_plot_cube
        elif self.last_plotted_mode == "Compare":
            heatmap_ax = event.inaxes if event.inaxes in set(self._cmp_heatmap_axes.values()) else None
            cube = None
            for key, ax in self._cmp_heatmap_axes.items():
                if ax is heatmap_ax and self.loaded and self.loaded.compare_cubes:
                    cube = self.loaded.compare_cubes.get(key)
                    break
        else:
            return
        if heatmap_ax is None or cube is None or event.inaxes is not heatmap_ax or event.ydata is None:
            return
        ygrid = np.asarray(cube.gate, float).ravel()
        y = float(np.clip(float(event.ydata), float(np.nanmin(ygrid)), float(np.nanmax(ygrid))))
        self.statusBar().showMessage(f"Hover gate: {y:.3f} V")

    def _on_canvas_click(self, event: Any) -> None:
        if event.button != 1:
            return
        # Manual peak delete by clicking near a marker on the bottom spectrum.
        if event.xdata is not None and self.last_plotted_mode == "DRR" and event.inaxes is self._drr_spectrum_ax:
            if self._remove_nearest_drr_peak(float(event.xdata)):
                return
        if (
            event.xdata is not None
            and event.ydata is not None
            and self.last_plotted_mode == "DRR"
            and event.inaxes is self._drr_heatmap_ax
        ):
            if self._remove_peak_from_drr_heatmap_click(float(event.xdata), float(event.ydata)):
                return
        if event.xdata is not None and self.last_plotted_mode == "PL" and event.inaxes is self._pl_spectrum_ax:
            if self._remove_nearest_pl_peak(float(event.xdata)):
                return
        if (
            event.xdata is not None
            and event.ydata is not None
            and self.last_plotted_mode == "PL"
            and event.inaxes is self._pl_heatmap_ax
        ):
            if self._remove_peak_from_pl_heatmap_click(float(event.xdata), float(event.ydata)):
                return

        if self.last_plotted_mode == "DRR":
            if self._drr_heatmap_ax is None or self._last_plot_cube is None:
                return
            if event.inaxes is not self._drr_heatmap_ax or event.ydata is None:
                return
            ygrid = np.asarray(self._last_plot_cube.gate, float).ravel()
            idx = int(np.argmin(np.abs(ygrid - float(event.ydata))))
            gate = float(ygrid[idx])
            self.drr_spins["gate"].setValue(gate)
            self._update_drr_spectrum_and_gate_line(self._last_plot_cube)
            return

        if self.last_plotted_mode == "Compare":
            if not self.loaded or not self.loaded.compare_cubes or event.ydata is None:
                return
            clicked_key = None
            for key, ax in self._cmp_heatmap_axes.items():
                if event.inaxes is ax:
                    clicked_key = key
                    break
            if clicked_key is None:
                return
            cube = self.loaded.compare_cubes.get(clicked_key)
            if cube is None:
                return
            ygrid = np.asarray(cube.gate, float).ravel()
            idx = int(np.argmin(np.abs(ygrid - float(event.ydata))))
            self._set_cmp_gate_spin_value(float(ygrid[idx]))
            self._plot_mode("Compare")
            return

        if self.last_plotted_mode != "PL" or self._pl_heatmap_ax is None or self._pl_last_plot_cube is None:
            return
        if event.inaxes is not self._pl_heatmap_ax or event.ydata is None:
            return
        ygrid = np.asarray(self._pl_last_plot_cube.gate, float).ravel()
        idx = int(np.argmin(np.abs(ygrid - float(event.ydata))))
        gate = float(ygrid[idx])
        self.pl_spins["gate"].setValue(gate)
        self._update_pl_spectrum_and_gate_line(self._pl_last_plot_cube)

    def _start_export(self, mode: str) -> None:
        if not self.loaded or self.loaded.mode != mode:
            self._show_error("Load and plot data before exporting.")
            return
        if self.last_plotted_mode != mode:
            self._show_error("Plot/Update before exporting.")
            return

        if mode in {"PL", "DRR"} and self.loaded.cube is not None:
            if mode == "DRR":
                export_cube = self._drr_cube_for_display()
                params = self._make_params(mode, export_cube)
            else:
                export_cube = self.loaded.cube
                params = self._make_params(mode, export_cube)
        elif mode == "Compare" and self.loaded.compare_cubes:
            first = next(iter(self.loaded.compare_cubes.values()))
            params = self._make_params("Compare", first)
            export_cube = None
        else:
            self._show_error("Nothing to export for this mode.")
            return

        if mode == "PL":
            options = ExportOptions(
                mode=mode,
                params=params,
                params_linear=HeatmapParams(**{**params.__dict__, "log_scale": False}),
                params_log=HeatmapParams(**{**params.__dict__, "log_scale": True}),
                auto_move_sources=bool(self.auto_move_after_export_chk.isChecked()),
            )
        elif mode == "DRR":
            options = ExportOptions(
                mode=mode,
                params=params,
                drr_cube=export_cube,
                drr_derivative_label=self.drr_derivative_combo.currentText(),
                auto_move_sources=bool(self.auto_move_after_export_chk.isChecked()),
            )
        else:
            options = ExportOptions(
                mode=mode,
                params=params,
                compare_scale_tag=("log" if self.cmp_log_chk.isChecked() else "linear"),
                compare_clip=bool(self.cmp_clip_chk.isChecked()),
                compare_gate=float(self.cmp_spins["gate"].value()),
                auto_move_sources=bool(self.auto_move_after_export_chk.isChecked()),
            )

        worker = Worker(self._export_task, self.loaded, options)
        worker.signals.log.connect(self._append_log)
        worker.signals.result.connect(self._on_export_done)
        worker.signals.error.connect(self._show_error)
        self._set_stage("Exporting...")
        self.thread_pool.start(worker)

    def _export_task(self, loaded: LoadedState, options: ExportOptions, *, progress: Signal, log: Signal) -> dict:
        mode = options.mode
        folder = loaded.folder
        out_folder: str | None = None
        files_to_move: list[str] = []
        if mode == "PL" and loaded.primary_file and options.params_linear is not None and options.params_log is not None:
            linear_cube = data_io.load_pl_cube(folder, loaded.primary_file, log_scale=False, y_axis=loaded.y_axis_spec)
            log_cube = data_io.load_pl_cube(folder, loaded.primary_file, log_scale=True, y_axis=loaded.y_axis_spec)
            paths = export_pl_pngs_and_dat(
                folder,
                loaded.primary_file,
                cube_linear=linear_cube,
                cube_log=log_cube,
                params_linear=options.params_linear,
                params_log=options.params_log,
            )
            log.emit(f"Exported PNG: {paths['png_linear'].name}, {paths['png_log'].name}")
            log.emit(f"Exported DAT: {paths['dat'].name}")
            out_folder = str(paths["png_linear"].parent)
            files_to_move = [loaded.primary_file]
        elif mode == "DRR" and options.drr_cube is not None and loaded.primary_file:
            base = build_drr_export_base(
                loaded.primary_file,
                len(loaded.selected_files),
                loaded.drr_mode_label,
                options.drr_derivative_label,
                "More correct (regrid)",
            )
            paths = export_drr_png_and_dat(folder, cube=options.drr_cube, params=options.params, export_base=base)
            log.emit(f"Exported PNG: {paths['png'].name}")
            log.emit(f"Exported DAT: {paths['dat'].name}")
            out_folder = str(paths["png"].parent)
            files_to_move = list(loaded.selected_files) + list(loaded.baseline_files)
        elif mode == "Compare" and loaded.compare_cubes:
            paths = export_compare_panels(
                folder,
                cubes=loaded.compare_cubes,
                source_files=loaded.compare_sources,
                params=options.params,
                scale_tag=options.compare_scale_tag,
                clip_outliers=options.compare_clip,
                gate_value=options.compare_gate,
            )
            log.emit(f"Exported {len(paths)} compare files.")
            out_folder = str(Path(paths[0]).parent) if paths else str(Path(folder))
            files_to_move = list(loaded.compare_sources.values()) if loaded.compare_sources else list(loaded.selected_files)
        else:
            raise ValueError("Nothing to export for this mode.")

        moved = 0
        if options.auto_move_sources and files_to_move:
            moved = int(data_io.move_selected_to_archive(folder, files_to_move))
            log.emit(f"Moved {moved} source CSV file(s) to 'Initial data after processing'.")
        return {"out_folder": out_folder or str(Path(folder)), "moved": moved}

    def _on_export_done(self, result: object) -> None:
        if isinstance(result, dict):
            out_folder = str(result.get("out_folder", self.current_folder or ""))
            moved = int(result.get("moved", 0))
        else:
            out_folder = str(result)
            moved = 0
        if moved > 0:
            self._refresh_file_lists()
        self._set_stage("Exported")
        self._status(f"Export completed: {out_folder}")

