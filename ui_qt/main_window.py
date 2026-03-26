from __future__ import annotations

import traceback
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
from core.plotting import HeatmapParams, plot_drr, plot_pl, render_compare_grid
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
        self._pl_last_plot_cube: DataCube | None = None
        self._gate_line = None
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

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")
        self._build_log_dock()
        self._build_menu_and_toolbar()

    def _build_left_panel(self) -> QWidget:
        box = QWidget()
        box.setMaximumWidth(UI_METRICS["left_max_width"])
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(UI_METRICS["row_spacing"])
        steps_label = QLabel("1) Select -> 2) Load -> 3) Plot -> 4) Export")
        steps_label.setWordWrap(False)
        layout.addWidget(steps_label)

        folder_box = QGroupBox("Data Source")
        folder_grid = QGridLayout(folder_box)
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        self.browse_btn = QPushButton("Browse Folder")
        self.open_file_btn = QPushButton("Open File")
        self.refresh_btn = QPushButton("Refresh")
        folder_grid.addWidget(self.folder_edit, 0, 0, 1, 3)
        folder_grid.addWidget(self.browse_btn, 1, 0)
        folder_grid.addWidget(self.open_file_btn, 1, 1)
        folder_grid.addWidget(self.refresh_btn, 1, 2)
        layout.addWidget(folder_box)

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

    def _build_pl_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        files = QGroupBox("")
        files_layout = QVBoxLayout(files)
        self.pl_files = QListWidget()
        self.pl_files.setSelectionMode(QAbstractItemView.SingleSelection)
        files_layout.addWidget(self.pl_files)
        layout.addWidget(self._make_expander("Measurement File", files, expanded=True))

        params = QGroupBox("")
        params_layout = QVBoxLayout(params)
        cfg = QFormLayout()
        grid, _, _, _, cmap, _ = self._build_common_range_grid("pl")
        cmap.setCurrentText("turbo")
        cfg.addRow("Colormap", cmap)
        params_layout.addLayout(cfg)
        params_layout.addLayout(grid)

        analysis = QGroupBox("")
        analysis_form = QFormLayout(analysis)
        analysis_form.setContentsMargins(6, 6, 6, 6)
        analysis_form.setHorizontalSpacing(6)
        analysis_form.setVerticalSpacing(6)
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
        analysis_form.addRow("", self.pl_fit_status)
        self.pl_analysis_text = QPlainTextEdit()
        self.pl_analysis_text.setReadOnly(True)
        self.pl_analysis_text.setMaximumHeight(86)
        self.pl_analysis_text.setPlaceholderText("Detected/Fit points will appear here.")
        analysis_form.addRow("", self.pl_analysis_text)
        analysis_section = self._make_expander("Spectrum Analysis", analysis, expanded=False)
        layout.addWidget(self._make_expander("Parameters", params, expanded=True))
        layout.addWidget(analysis_section)
        layout.addStretch(1)
        return tab

    def _build_drr_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        files = QGroupBox("")
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

        params = QGroupBox("")
        params_layout = QVBoxLayout(params)
        params_layout.setContentsMargins(8, 8, 8, 8)
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

        def pair_auto_row(
            a: QDoubleSpinBox,
            b: QDoubleSpinBox,
            fa: QCheckBox,
            fb: QCheckBox,
            auto_btn: QToolButton,
            auto_text: str,
        ) -> QWidget:
            auto_btn.setText(auto_text)
            auto_btn.setAutoRaise(True)
            auto_btn.setFixedWidth(UI_METRICS["tool_w"])
            auto_btn.setFixedHeight(UI_METRICS["tool_h"])
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

        basic = QGroupBox("Basic")
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
        basic_form.addRow("Derivative / SG", deriv_row)
        basic_form.addRow(
            "vmin / vmax",
            pair_auto_row(spins["vmin"], spins["vmax"], fix_checks["vmin"], fix_checks["vmax"], self.drr_auto_v_btn, "Auto V"),
        )
        basic_form.addRow(
            "xmin / xmax",
            pair_auto_row(spins["xmin"], spins["xmax"], fix_checks["xmin"], fix_checks["xmax"], self.drr_auto_x_btn, "Auto X"),
        )
        basic_form.addRow(
            "ymin / ymax",
            pair_auto_row(spins["ymin"], spins["ymax"], fix_checks["ymin"], fix_checks["ymax"], self.drr_auto_y_btn, "Auto Y"),
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
        analysis_form.setVerticalSpacing(6)

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
        self.drr_analysis_text.setMaximumHeight(86)
        self.drr_analysis_text.setPlaceholderText("Detected/Fit points will appear here.")
        analysis_form.addRow("", self.drr_analysis_text)
        analysis_section = self._make_expander("Spectrum Analysis", analysis_box, expanded=False)
        params_layout.addWidget(basic)
        layout.addWidget(self._make_expander("Parameters", params, expanded=True))
        layout.addWidget(analysis_section)
        layout.addStretch(1)
        return tab

    def _set_form_label_width(self, form: QFormLayout, width: int) -> None:
        for row in range(form.rowCount()):
            item = form.itemAt(row, QFormLayout.LabelRole)
            if item and isinstance(item.widget(), QLabel):
                lbl = item.widget()
                lbl.setFixedWidth(width)
                lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def _style_combo_popup(self, combo: QComboBox) -> None:
        view = combo.view()
        view.setStyleSheet(
            "QListView::item:selected { background-color: #2d6cdf; color: #ffffff; }"
            "QListView { selection-background-color: #2d6cdf; selection-color: #ffffff; }"
        )

    def _make_expander(self, title: str, content: QWidget, *, expanded: bool = True) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        head = QToolButton()
        head.setCheckable(True)
        head.setChecked(bool(expanded))
        head.setAutoRaise(True)
        head.setToolButtonStyle(Qt.ToolButtonTextOnly)
        head.setStyleSheet("QToolButton { border: none; padding: 0px 4px; }")
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        row = QWidget()
        row_h = QHBoxLayout(row)
        row_h.setContentsMargins(0, 0, 0, 0)
        row_h.setSpacing(4)
        row_h.addWidget(head)
        row_h.addWidget(line, 1)
        v.addWidget(row)
        content.setVisible(bool(expanded))
        v.addWidget(content)

        def _update(on: bool) -> None:
            head.setText(f"{'[-]' if on else '[+]'} {title}")
            content.setVisible(bool(on))

        _update(bool(expanded))
        head.toggled.connect(_update)
        return box

    def _build_compare_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        files = QGroupBox("")
        files_layout = QVBoxLayout(files)
        self.cmp_files = QListWidget()
        self.cmp_files.setSelectionMode(QAbstractItemView.ExtendedSelection)
        files_layout.addWidget(self.cmp_files)
        layout.addWidget(self._make_expander("Compare Files (2-4)", files, expanded=True))

        params = QGroupBox("")
        params_layout = QVBoxLayout(params)
        cfg = QFormLayout()
        grid, _, _, _, cmap, _ = self._build_common_range_grid("cmp")
        cmap.setCurrentText("turbo")
        cfg.addRow("Colormap", cmap)
        params_layout.addLayout(cfg)
        params_layout.addLayout(grid)
        layout.addWidget(self._make_expander("Parameters", params, expanded=True))
        layout.addStretch(1)
        return tab

    def _build_tools_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        box = QGroupBox("")
        b = QVBoxLayout(box)
        self.show_log_btn = QPushButton("Show/Hide Log")
        self.clear_log_btn = QPushButton("Clear Log")
        b.addWidget(self.show_log_btn)
        b.addWidget(self.clear_log_btn)
        layout.addWidget(self._make_expander("Tools", box, expanded=True))
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
        self.drr_selected_files = [f for f in self.drr_selected_files if f in self.available_files]
        self.drr_baseline_files_manual = [f for f in self.drr_baseline_files_manual if f in self.available_files]
        self.drr_baseline_files_found = [f for f in self.drr_baseline_files_found if f in self.available_files]
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
        drr_loaded = loaded_mode == "DRR"
        self.drr_auto_v_btn.setEnabled(drr_loaded)
        self.drr_auto_x_btn.setEnabled(drr_loaded)
        self.drr_auto_y_btn.setEnabled(drr_loaded)

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

    def _auto_drr_xrange(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        cube = self._drr_cube_for_display()
        self.drr_spins["xmin"].setValue(float(np.nanmin(cube.energy)))
        self.drr_spins["xmax"].setValue(float(np.nanmax(cube.energy)))
        self._status("State: Auto xmin/xmax set from energy axis.")
        self._plot_mode("DRR")

    def _auto_drr_yrange(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        cube = self._drr_cube_for_display()
        self.drr_spins["ymin"].setValue(float(np.nanmin(cube.gate)))
        self.drr_spins["ymax"].setValue(float(np.nanmax(cube.gate)))
        self._status("State: Auto ymin/ymax set from gate axis.")
        self._plot_mode("DRR")

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
        else:
            selected = self._selected(self.cmp_files)
            baselines = []
            pl_log = False
            cmp_log = bool(self.cmp_log_chk.isChecked())
            drr_baseline = "Self (last frame)"
            drr_baseline_which = "last"

        options = LoadOptions(
            mode=mode,
            folder=self.current_folder,
            selected_files=selected,
            baseline_files=baselines,
            pl_log_scale=pl_log,
            drr_baseline_text=drr_baseline,
            drr_baseline_which=drr_baseline_which,
            compare_log_scale=cmp_log,
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
            cube = data_io.load_pl_cube(folder, options.selected_files[0], log_scale=options.pl_log_scale)
            return LoadedState(
                mode="PL",
                folder=folder,
                primary_file=options.selected_files[0],
                selected_files=options.selected_files,
                cube=cube,
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
                    derivative=None,
                )
                drr_mode_label = "DR/R External"
            else:
                cube = data_io.load_drr_self_cube(
                    folder,
                    options.selected_files,
                    use_first_frame=(baseline == "Self (first frame)"),
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
            )

        if len(options.selected_files) < 2:
            raise ValueError("Compare mode needs at least 2 files.")
        kpk = options.selected_files[2] if len(options.selected_files) > 2 else None
        kpkp = options.selected_files[3] if len(options.selected_files) > 3 else None
        selection = data_io.CompareSelection(kk=options.selected_files[0], kkp=options.selected_files[1], kpk=kpk, kpkp=kpkp)
        cubes = data_io.load_compare_cubes(folder, selection, log_scale=options.compare_log_scale)
        return LoadedState(
            mode="Compare",
            folder=folder,
            selected_files=options.selected_files,
            compare_cubes=cubes,
            compare_sources=selection.as_pairs(),
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
            self._update_pl_spectrum_with_analysis(self._pl_last_plot_cube)

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

            self._ensure_loaded_matches_drr_params()
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
                gate_val = float(self._mode_spins(mode)["gate"].value())
                self._plot_spectrum_with_roi(ax2, plot_cube, gate_val, ylabel=params.cbar_label, xlim=params.xlim)
                self._pl_heatmap_ax = ax1
                self._pl_spectrum_ax = ax2
                self._pl_last_plot_cube = plot_cube
                self._pl_heatmap_peak_artist = None
                self._pl_heatmap_fit_artist = None
                self._update_pl_spectrum_with_analysis(plot_cube)
            elif mode == "DRR" and self.loaded.cube is not None:
                self._pl_heatmap_ax = None
                self._pl_spectrum_ax = None
                self._pl_last_plot_cube = None
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
                first = next(iter(self.loaded.compare_cubes.values()))
                params = self._make_params(mode, first)
                images = render_compare_grid(self.figure, self.loaded.compare_cubes, params)
                if images:
                    self.figure.colorbar(images[0], ax=self.figure.axes, label=params.cbar_label, shrink=0.82)
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
        if mode != "DRR":
            return (mode, int(self.tabs.currentIndex()), self.last_plotted_mode)
        p = self._read_drr_params()
        return (
            "DRR", p["baseline_mode"], p["baseline_which"], p["baseline_files"], p["selected_files"],
            p["derivative"], p["sg_window"], p["sg_poly"], p["cmap"], p["vmin"], p["vmax"],
            p["xmin"], p["xmax"], p["ymin"], p["ymax"], p["gate"], p["log"], p["clip"], p["center_zero"],
        )

    def _is_drr_gate_only_change(self, new_key: tuple[Any, ...]) -> bool:
        if self._last_plot_params_key is None or self._last_plot_cube is None:
            return False
        if len(new_key) != len(self._last_plot_params_key):
            return False
        gate_idx = 15
        return (
            new_key[:gate_idx] == self._last_plot_params_key[:gate_idx]
            and new_key[gate_idx + 1 :] == self._last_plot_params_key[gate_idx + 1 :]
            and new_key[gate_idx] != self._last_plot_params_key[gate_idx]
        )

    def _ensure_loaded_matches_drr_params(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        p = self._read_drr_params()
        selected = list(p["selected_files"])
        baselines = list(p["baseline_files"])
        baseline_text = p["baseline_mode"]
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
                    derivative=None,
                )
                mode_label = "DR/R External"
            else:
                cube = data_io.load_drr_self_cube(
                    self.current_folder,
                    selected,
                    use_first_frame=(baseline_text == "Self (first frame)"),
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
            )
            self._last_plot_cube = None
            self._last_plot_params_key = None

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
        xlim = (float(self.drr_spins["xmin"].value()), float(self.drr_spins["xmax"].value()))
        self._drr_spectrum_ax.set_xlim(xlim)
        self._auto_scale_spectrum_y(self._drr_spectrum_ax, x, y, xlim)
        self._set_drr_gate_spin_value(gate_used)
        self._ensure_gate_line(cube, gate_used)
        self._draw_drr_analysis_overlays(cube, gate_used, x, np.asarray(y, float))
        self._update_drr_analysis_text(gate_used, x, np.asarray(y, float))
        self.canvas.draw_idle()

    def _on_canvas_motion(self, event: Any) -> None:
        if self._drr_heatmap_ax is None or self.last_plotted_mode != "DRR":
            return
        if event.inaxes is not self._drr_heatmap_ax or event.ydata is None or self._last_plot_cube is None:
            return
        ygrid = np.asarray(self._last_plot_cube.gate, float).ravel()
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

        if self._drr_heatmap_ax is None or self.last_plotted_mode != "DRR":
            return
        if event.inaxes is not self._drr_heatmap_ax or event.ydata is None or self._last_plot_cube is None:
            return
        ygrid = np.asarray(self._last_plot_cube.gate, float).ravel()
        idx = int(np.argmin(np.abs(ygrid - float(event.ydata))))
        gate = float(ygrid[idx])
        self.drr_spins["gate"].setValue(gate)
        self._update_drr_spectrum_and_gate_line(self._last_plot_cube)

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
            linear_cube = data_io.load_pl_cube(folder, loaded.primary_file, log_scale=False)
            log_cube = data_io.load_pl_cube(folder, loaded.primary_file, log_scale=True)
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

