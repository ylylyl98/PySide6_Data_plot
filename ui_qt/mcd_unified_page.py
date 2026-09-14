"""Unified MCD presentation surface.

The source and processing controls remain owned by :class:`MainWindow` and
``McdController``.  This module owns the small amount of presentation state
needed to show MCD correction, spectra, and feature results together.  It is
deliberately independent of the legacy peak-shift page so existing worker and
cache lifetimes remain valid while the visible workflow is consolidated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Signal, QRect, QSize
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QInputDialog,
    QLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSlider,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg


def fit_display_segments(fit: Mapping[str, Any], *, visible_min: float | None = None,
                         visible_max: float | None = None) -> dict[str, np.ndarray]:
    """Return measured fit and its display-only visible-axis extrapolation.

    ``actual_x``/``actual_y`` always remain the exact fit subset.  The
    extension is a presentation aid and spans the supplied visible
    field range; no scientific fit values are changed.
    """
    low = float(fit["field_min_t"])
    high = float(fit["field_max_t"])
    if high < low:
        low, high = high, low
    span = abs(high - low)
    pad = 0.5 * span if span else max(abs(low) * 0.05, 1e-6)
    ext_low, ext_high = low - pad, high + pad
    if visible_min is not None:
        ext_low = float(visible_min)
    if visible_max is not None:
        ext_high = float(visible_max)
    slope = float(fit["slope"])
    intercept = float(fit["intercept"])
    actual_x = np.asarray([low, high], dtype=float)
    extension_x = np.asarray([ext_low, ext_high], dtype=float)
    return {
        "actual_x": actual_x,
        "actual_y": slope * actual_x + intercept,
        "extension_x": extension_x,
        "extension_y": slope * extension_x + intercept,
    }


@dataclass(frozen=True)
class RetainedMcdWindow:
    """An independently retained MCD window snapshot."""

    label: str
    center_ev: float
    width_mev: float
    source_generation: int = 0
    settings: tuple[tuple[str, Any], ...] = ()


@dataclass
class UnifiedMcdState:
    """Presentation state that is safe to retain across redraws."""

    window_center_ev: float = 0.0
    window_width_mev: float = 5.0
    window_metric: str = "mean"
    selected_b_index: int = 0
    selected_feature_id: str | None = None
    feature_metric: str = "shift"
    feature_overlay: bool = False
    detection_mode: str = "raw"
    source_generation: int = 0
    retained_windows: list[RetainedMcdWindow] = field(default_factory=list)
    retained_features: list[dict[str, Any]] = field(default_factory=list)

    def retain_window(self, label: str, center_ev: float, width_mev: float, *, source_generation: int = 0,
                      settings: Mapping[str, Any] | None = None) -> RetainedMcdWindow:
        snapshot = RetainedMcdWindow(
            str(label), float(center_ev), float(width_mev), int(source_generation),
            tuple(sorted((str(key), value) for key, value in (settings or {}).items())),
        )
        self.retained_windows.append(snapshot)
        return snapshot


class McdControlFlowLayout(QLayout):
    """Small wrapping layout for center controls at laptop widths."""
    def __init__(self, parent=None):
        super().__init__(parent); self._items = []
    def addItem(self, item): self._items.append(item)
    def count(self): return len(self._items)
    def itemAt(self, index): return self._items[index] if 0 <= index < len(self._items) else None
    def takeAt(self, index): return self._items.pop(index) if 0 <= index < len(self._items) else None
    def expandingDirections(self): return Qt.Orientation.Horizontal
    def hasHeightForWidth(self): return True
    def heightForWidth(self, width):
        return self._do_layout(self.contentsRect().adjusted(0, 0, int(width - self.contentsRect().width()), 0), True)
    def minimumSize(self):
        margins = self.contentsMargins(); widest = max((item.minimumSize().width() for item in self._items), default=0)
        height = max((item.minimumSize().height() for item in self._items), default=0)
        return QSize(widest + margins.left() + margins.right(), height + margins.top() + margins.bottom())
    def sizeHint(self): return self.minimumSize()
    def setGeometry(self, rect):
        super().setGeometry(rect); self._do_layout(rect, False)
    def _do_layout(self, rect, test_only):
        x = rect.x(); y = rect.y(); line_h = 0; spacing = self.spacing(); right = rect.right(); lines = 1
        for item in self._items:
            hint = item.sizeHint(); next_x = x + hint.width() + (spacing if x > rect.x() else 0)
            if next_x - spacing > right and x > rect.x(): x = rect.x(); y += line_h + spacing; next_x = x + hint.width(); line_h = 0; lines += 1
            if not test_only: item.setGeometry(QRect(x, y, hint.width(), hint.height()))
            x = next_x; line_h = max(line_h, hint.height())
        return (y - rect.y()) + line_h


class McdUnifiedView(QWidget):
    """Candidate bar and four fixed MCD plot regions.

    ``figure``/``canvas`` may be supplied by the production shell.  In that
    mode the view contributes only its candidate bar and draws into the
    existing toolbar canvas.  Standalone construction embeds its own canvas,
    which keeps the component easy to exercise in isolated Qt tests.
    """

    manual_candidate_added = Signal(object)
    candidate_selection_changed = Signal(object)
    detection_mode_changed = Signal(str)
    window_center_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None, *, figure: Figure | None = None,
                 canvas: ThemeAwareFigureCanvasQTAgg | None = None,
                 embed_canvas: bool | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mcdUnifiedView")
        self.state = UnifiedMcdState()
        self.figure = figure or Figure(figsize=(9, 7), dpi=100)
        self.canvas = canvas or ThemeAwareFigureCanvasQTAgg(self.figure)
        if embed_canvas is None:
            embed_canvas = canvas is None
        self._embed_canvas = bool(embed_canvas)
        self.axes: dict[str, Any] = {}
        self._feature_energy_axis = None
        self._map_fields = np.array([], dtype=float)
        self._blit_backgrounds: dict[str, Any] = {}
        self._blit_preparing = False
        self._blit_rebuild_pending = False
        self._rendering = False
        self._overlay_syncing = False
        self._xlim_cids: list[tuple[Any, int]] = []
        self._displayed_spectrum_rows: dict[tuple[str, str], int] = {}
        self._draw_event_cid = self.canvas.mpl_connect("draw_event", self._on_canvas_draw)
        self._resize_event_cid = self.canvas.mpl_connect("resize_event", self._update_subplot_spacing)
        self._map_click_cid = self.canvas.mpl_connect("button_press_event", self._on_map_click)
        self._map_motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_map_motion)
        self._map_release_cid = self.canvas.mpl_connect("button_release_event", self._on_map_release)
        self._window_drag_active = False
        self._legend_pick_cid = self.canvas.mpl_connect("pick_event", self._on_legend_pick)
        self._legend_line_map: dict[Any, Any] = {}
        self._spectrum_legend_texts: dict[Any, Any] = {}
        self._artists: dict[str, Any] = {}
        self._result: Any | None = None
        self._candidates: list[dict[str, Any]] = []
        self._manual_candidates: list[dict[str, Any]] = []
        self._manual_candidate_number = 0
        self.visible_candidates: list[dict[str, Any]] = []
        self.analysis_payload: Mapping[str, Any] | None = None
        self._source_generation = 0
        self.render_count = 0
        self.draw_only_updates = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.candidate_bar = QWidget(self)
        self.candidate_bar.setObjectName("mcdUnifiedCandidateBar")
        bar = McdControlFlowLayout(self.candidate_bar)
        bar.setContentsMargins(6, 3, 6, 3)
        bar.setSpacing(2)
        def add_widget(widget: QWidget, stretch: int = 0) -> None:
            bar.addWidget(widget)
        def finish_top_row() -> None:
            return None
        add_widget(QLabel("Feature centers:"))
        self.candidate_filter_combo = QComboBox(self.candidate_bar)
        self.candidate_filter_combo.addItems(["MCD", "History", "Recommended", "Spectrum", "All"])
        self.candidate_filter_combo.setToolTip("H: saved center history · R: current recommendation · H/R: both")
        self.candidate_filter_combo.setCurrentText("MCD")
        self.candidate_filter_combo.currentTextChanged.connect(self.set_candidate_filter)
        add_widget(self.candidate_filter_combo)
        self.recommended_only_chk = QCheckBox("Recommended only", self.candidate_bar)
        self.recommended_only_chk.setObjectName("recommended_only_chk")
        self.recommended_only_chk.setToolTip("Show the six highest-ranked MCD windows; turn off to see all qualified candidates")
        self.recommended_only_chk.toggled.connect(lambda _checked: self.set_candidate_filter(self.candidate_filter_combo.currentText()))
        self.detection_mode_combo = QComboBox(self.candidate_bar)
        self.detection_mode_combo.setObjectName("Centers")
        self.detection_mode_combo.addItem("Raw extrema", "raw")
        self.detection_mode_combo.addItem("Residual extrema", "residual")
        self.detection_mode_combo.currentIndexChanged.connect(self._detection_mode_index_changed)
        self.detection_mode_combo.setToolTip("Choose the center locator catalog")
        self.previous_candidate_btn = QToolButton(self.candidate_bar)
        self.previous_candidate_btn.setText("‹")
        self.previous_candidate_btn.setToolTip("Select previous feature candidate")
        self.next_candidate_btn = QToolButton(self.candidate_bar)
        self.next_candidate_btn.setText("›")
        self.next_candidate_btn.setToolTip("Select next feature candidate")
        self.previous_candidate_btn.clicked.connect(lambda: self.step_candidate(-1))
        self.next_candidate_btn.clicked.connect(lambda: self.step_candidate(1))
        self.candidate_combo = QComboBox(self.candidate_bar)
        self.candidate_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.candidate_combo.setMinimumWidth(160)
        self.candidate_combo.setMaximumWidth(220)
        self.candidate_combo.currentIndexChanged.connect(self._candidate_changed)
        add_widget(self.candidate_combo, 1)
        self.candidate_nav_group = QWidget(self.candidate_bar)
        self.candidate_nav_group.setObjectName("centerCandidateNavigation")
        candidate_nav_layout = QHBoxLayout(self.candidate_nav_group)
        candidate_nav_layout.setContentsMargins(0, 0, 0, 0)
        candidate_nav_layout.setSpacing(2)
        candidate_nav_layout.addWidget(self.previous_candidate_btn)
        candidate_nav_layout.addWidget(self.next_candidate_btn)
        add_widget(self.candidate_nav_group)
        add_widget(self.recommended_only_chk)
        self.settings_btn = QPushButton("Settings…", self.candidate_bar)
        self.settings_btn.setObjectName("mcdCentersSettingsButton")
        self.settings_btn.clicked.connect(self._show_settings_dialog)
        add_widget(self.settings_btn)
        self.manual_seed_btn = QPushButton("+ Manual", self.candidate_bar)
        self.manual_seed_btn.setToolTip("Add a manually seeded feature candidate")
        self.manual_seed_btn.clicked.connect(self._add_manual_candidate_dialog)
        add_widget(self.manual_seed_btn)
        finish_top_row()
        self.spectra_source_combo = QComboBox(self.candidate_bar)
        self.spectra_source_combo.addItems(["Raw", "Corrected"])
        self.spectra_source_combo.setToolTip("Change the displayed spectra without re-running feature analysis")
        add_widget(self.spectra_source_combo)
        self.spectra_source_combo.currentTextChanged.connect(self._display_source_changed)
        self.channel_mapping_combo = QComboBox(self.candidate_bar)
        self.channel_mapping_combo.addItems([
            "Auto channel labels", "K = + / K′ = −", "K = − / K′ = +",
            "Infer mapping from energy",
        ])
        self.channel_mapping_combo.setToolTip("Use K/K′ names only when the experiment mapping is established")
        self.channel_mapping_combo.currentTextChanged.connect(self._display_source_changed)
        add_widget(self.channel_mapping_combo)
        self.spectra_b_label = QLabel("B:", self.candidate_bar)
        add_widget(self.spectra_b_label)
        self.spectra_b_spin = QDoubleSpinBox(self.candidate_bar)
        self.spectra_b_spin.setRange(-1e6, 1e6)
        self.spectra_b_spin.setDecimals(5)
        self.spectra_b_spin.setSuffix(" T")
        self.spectra_b_spin.setFixedWidth(92)
        self.spectra_b_spin.valueChanged.connect(self._set_b_from_value)
        add_widget(self.spectra_b_spin)
        self.spectra_b_slider = QSlider(Qt.Horizontal, self.candidate_bar)
        self.spectra_b_slider.setToolTip("Select the B slice shown in the spectra panel")
        self.spectra_b_slider.setFixedWidth(110)
        self.spectra_b_slider.valueChanged.connect(self.set_selected_b)
        add_widget(self.spectra_b_slider)
        self.show_inc_chk = QPushButton("Inc sweep", self.candidate_bar)
        self.show_inc_chk.setCheckable(True)
        self.show_inc_chk.setChecked(True)
        self.show_dec_chk = QPushButton("Dec sweep", self.candidate_bar)
        self.show_dec_chk.setCheckable(True)
        self.show_dec_chk.setChecked(True)
        add_widget(self.show_inc_chk)
        add_widget(self.show_dec_chk)
        self.show_inc_chk.toggled.connect(self._display_source_changed)
        self.show_dec_chk.toggled.connect(self._display_source_changed)
        self.candidate_status = QLabel("No MCD feature candidates")
        # The status is attached to the result row after the canvas controls
        # are assembled, so long explanations never widen the top bar.
        self.feature_metric_combo = QComboBox(self.candidate_bar)
        self.feature_metric_combo.addItems(["Shift (meV)", "Energy (eV)", "Splitting (meV)"])
        self.feature_metric_combo.setToolTip("Choose the quantity shown in the feature-versus-B panel")
        self.feature_metric_combo.currentTextChanged.connect(self._display_source_changed)
        # Retain the combo for saved-state and older callers, while exposing
        # compact plot-local controls for fast cached display changes.
        self.feature_metric_combo.setVisible(False)
        self.display_metric_group = QButtonGroup(self)
        self.display_metric_group.setExclusive(True)
        self.shift_metric_btn = QToolButton(self.candidate_bar)
        self.energy_metric_btn = QToolButton(self.candidate_bar)
        self.splitting_metric_btn = QToolButton(self.candidate_bar)
        for button, label, tip in (
            (self.shift_metric_btn, "Shift", "Show measured feature shift in meV"),
            (self.energy_metric_btn, "Energy", "Show measured feature energy in eV"),
            (self.splitting_metric_btn, "Splitting", "Show measured same-branch channel splitting in meV"),
        ):
            button.setText(label)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.setAccessibleName(f"Feature {label} display")
            self.display_metric_group.addButton(button)
            add_widget(button)
        self.shift_metric_btn.setChecked(True)
        self.display_metric_group.buttonClicked.connect(self._fast_metric_clicked)
        self.overlay_btn = QToolButton(self.candidate_bar)
        self.overlay_btn.setText("Overlay")
        self.overlay_btn.setCheckable(True)
        self.overlay_btn.setToolTip("Overlay a linked secondary quantity on the feature-versus-B plot")
        self.overlay_btn.setAccessibleName("Feature overlay")
        self.overlay_btn.toggled.connect(self._display_overlay_changed)
        add_widget(self.overlay_btn)
        self.show_energy_chk = QCheckBox("E", self.candidate_bar)
        self.show_energy_chk.setChecked(True)
        self.show_energy_chk.setToolTip("Show absolute feature energy on the secondary axis")
        self.show_splitting_chk = QCheckBox("ΔK", self.candidate_bar)
        self.show_splitting_chk.setChecked(False)
        self.show_splitting_chk.setToolTip("Overlay measured K/K′ splitting when mapping is established")
        self.show_energy_chk.toggled.connect(self._display_source_changed)
        self.show_splitting_chk.toggled.connect(self._display_source_changed)
        # Compatibility switches remain available to old integrations but do
        # not duplicate the fast plot controls in the normal surface.
        self.show_energy_chk.setVisible(False)
        self.show_splitting_chk.setVisible(False)
        self._control_bar_layout = bar
        self._build_compact_plot_controls(layout)
        layout.addWidget(self.candidate_bar)
        if self._embed_canvas:
            layout.addWidget(self.canvas, 1)
        # Let complete control groups share the available toolbar width.
        # The flow layout wraps a group only when it no longer fits.
        bar.addWidget(self.spectra_controls_host)
        bar.addWidget(self.feature_controls_host)

    def _remove_from_control_bar(self, widget: QWidget) -> None:
        for index in range(self._control_bar_layout.count()):
            item = self._control_bar_layout.itemAt(index)
            if item is not None and item.widget() is widget:
                self._control_bar_layout.takeAt(index)
                widget.setParent(None)
                return

    def _build_compact_plot_controls(self, outer_layout: QVBoxLayout) -> None:
        """Place graph-local controls outside the Matplotlib export canvas."""
        self.spectra_controls_host = QWidget(self)
        self.spectra_controls_host.setObjectName("mcdSpectraControls")
        spectra_row = QHBoxLayout(self.spectra_controls_host)
        spectra_row.setContentsMargins(6, 0, 6, 0); spectra_row.setSpacing(4)
        spectra_row.addWidget(QLabel("B slice:"))
        self._plot_local_spectra_controls = (self.spectra_b_label, self.spectra_b_spin, self.spectra_b_slider, self.show_inc_chk, self.show_dec_chk)
        for widget in self._plot_local_spectra_controls:
            if widget is self.show_inc_chk:
                spectra_row.addWidget(QLabel("MCD(B):"))
            self._remove_from_control_bar(widget); spectra_row.addWidget(widget)
        spectra_row.addStretch(1)
        self.feature_controls_host = QWidget(self)
        self.feature_controls_host.setObjectName("mcdFeatureControls")
        feature_row = QHBoxLayout(self.feature_controls_host)
        feature_row.setContentsMargins(6, 0, 6, 0); feature_row.setSpacing(4)
        feature_row.addWidget(QLabel("Feature result:"))
        self._remove_from_control_bar(self.candidate_status)
        self.candidate_status.setMinimumWidth(0)
        self.candidate_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        feature_row.addWidget(self.candidate_status, 1)
        # Keep the compatibility label object for callers, but do not reserve
        # a visible row for long catalog/status text.
        self.candidate_status.setVisible(False)
        for widget in (self.shift_metric_btn, self.energy_metric_btn, self.splitting_metric_btn):
            self._remove_from_control_bar(widget); feature_row.addWidget(widget)
        feature_row.addStretch(1)
        self.result_panel_combo = QComboBox(self.candidate_bar)
        self.result_panel_combo.addItems(["MCD spectra", "Peak shift vs B"])
        self.result_panel_combo.setToolTip("Choose the lower-right plot")
        self._control_bar_layout.addWidget(self.result_panel_combo)
        self.result_panel_combo.currentIndexChanged.connect(self._result_panel_changed)
        self.feature_controls_host.setVisible(False)

        self.settings_dialog = QDialog(self)
        self.settings_dialog.setObjectName("mcdCentersSettingsDialog")
        self.settings_dialog.setWindowTitle("Center settings")
        settings_layout = QVBoxLayout(self.settings_dialog)
        settings_layout.addWidget(QLabel("Center locator and display settings"))
        settings_widgets = (self.detection_mode_combo, self.recommended_only_chk, self.channel_mapping_combo,
                            self.manual_seed_btn, self.overlay_btn, self.show_energy_chk, self.show_splitting_chk)
        for widget in settings_widgets:
            self._remove_from_control_bar(widget); widget.setVisible(True); settings_layout.addWidget(widget)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self.settings_dialog)
        buttons.rejected.connect(self.settings_dialog.hide); buttons.accepted.connect(self.settings_dialog.hide)
        self._remove_from_control_bar(self.spectra_source_combo)
        self.spectra_source_combo.setParent(self.settings_dialog)
        self.spectra_source_combo.hide()
        settings_layout.addWidget(buttons)

    def _result_panel_changed(self, _index: int) -> None:
        self.feature_controls_host.setVisible(self.result_panel_combo.currentIndex() == 1)
        if self._result is not None:
            self.render(self._result, candidates=self._candidates, analysis=self.analysis_payload)

    def _show_settings_dialog(self) -> None:
        self.settings_dialog.adjustSize()
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def _on_legend_pick(self, event: Any) -> None:
        """Toggle a measured spectrum by clicking its legend sample."""
        line = self._legend_line_map.get(getattr(event, "artist", None))
        if line is None:
            return
        line.set_visible(not line.get_visible())
        sample = getattr(event, "artist", None)
        try:
            sample.set_alpha(1.0 if line.get_visible() else 0.25)
        except AttributeError:
            pass
        self.canvas.draw_idle()

    def _set_b_from_value(self, value: float) -> None:
        if self._result is None:
            return
        fields = np.asarray(getattr(self._result, "pair_b", ()), dtype=float).ravel()
        if fields.size:
            self.set_selected_b(int(np.nanargmin(np.abs(fields - float(value)))))

    def _detection_mode_index_changed(self, _index: int) -> None:
        mode = str(self.detection_mode_combo.currentData() or "raw")
        if mode == self.state.detection_mode:
            return
        self.state.detection_mode = mode
        self.detection_mode_changed.emit(mode)

    def set_detection_mode(self, mode: str) -> None:
        mode = str(mode).casefold()
        index = self.detection_mode_combo.findData(mode)
        if index >= 0:
            self.detection_mode_combo.setCurrentIndex(index)

    @property
    def selected_candidate(self) -> dict[str, Any] | None:
        index = self.candidate_combo.currentIndex()
        identifier = self.candidate_combo.currentData(Qt.ItemDataRole.UserRole)
        if identifier is not None:
            for item in self.visible_candidates:
                if str(item.get("id", "")) == str(identifier):
                    return item
        return self.visible_candidates[index] if 0 <= index < len(self.visible_candidates) else None

    def set_source_generation(self, generation: int) -> None:
        self._source_generation = int(generation)
        self.state.source_generation = int(generation)
        self.analysis_payload = None

    def deactivate(self) -> None:
        """Release the shared Figure when another workflow owns the canvas."""
        self._window_drag_active = False
        self._blit_rebuild_pending = False
        self._blit_backgrounds = {}
        self._rendering = False
        if self.axes:
            self.figure.clear()
            self.axes = {}
        self._artists = {}
        self._result = None

    def publish_analysis(self, payload: Mapping[str, Any] | None) -> bool:
        if payload is None:
            return False
        if not isinstance(payload, Mapping) and hasattr(payload, "to_dict"):
            payload = payload.to_dict()
        generation = payload.get("generation", self._source_generation)
        if int(generation) != self._source_generation:
            return False
        self.analysis_payload = dict(payload)
        return True

    def _display_source_changed(self, _value: str) -> None:
        value = str(_value).casefold()
        if "shift" in value or "energy" in value or "splitting" in value:
            metric = "splitting" if "splitting" in value else "energy" if "energy" in value else "shift"
            self.state.feature_metric = metric
            button = {"shift": self.shift_metric_btn, "energy": self.energy_metric_btn, "splitting": self.splitting_metric_btn}[metric]
            button.setChecked(True)
        if self._result is not None:
            self.render(self._result, candidates=self._candidates, analysis=self.analysis_payload)

    def _fast_metric_clicked(self, button: QToolButton) -> None:
        metric = {
            self.shift_metric_btn: "shift",
            self.energy_metric_btn: "energy",
            self.splitting_metric_btn: "splitting",
        }.get(button)
        if metric is None:
            return
        self.state.feature_metric = metric
        self.feature_metric_combo.blockSignals(True)
        self.feature_metric_combo.setCurrentText({
            "shift": "Shift (meV)", "energy": "Energy (eV)", "splitting": "Splitting (meV)",
        }[metric])
        self.feature_metric_combo.blockSignals(False)
        if self.result_panel_combo.currentIndex() != 1:
            self.result_panel_combo.setCurrentIndex(1)
            return
        self._refresh_feature_display()

    def _display_overlay_changed(self, checked: bool) -> None:
        self.state.feature_overlay = bool(checked)
        self._refresh_feature_display()

    def _refresh_feature_display(self, *, queue_draw: bool = True) -> None:
        """Refresh only the cached feature panel, retaining user ranges."""
        if self._result is None or "feature_vs_b" not in self.axes:
            return
        axis = self.axes["feature_vs_b"]
        xlim, ylim = axis.get_xlim(), axis.get_ylim()
        preserve_y = getattr(self, "_feature_axis_metric", self.state.feature_metric) == self.state.feature_metric
        if self._feature_energy_axis is not None:
            try:
                self._feature_energy_axis.remove()
            except (AttributeError, ValueError):
                pass
            self._feature_energy_axis = None
        axis.clear()
        axis.grid(alpha=.25)
        fields = np.asarray(getattr(self._result, "pair_b", ()), dtype=float).ravel()
        self._draw_feature_vs_b(fields)
        if np.all(np.isfinite(xlim)) and xlim[1] > xlim[0]:
            axis.set_xlim(*xlim)
        if preserve_y and np.all(np.isfinite(ylim)) and ylim[1] > ylim[0]:
            axis.set_ylim(*ylim)
        if queue_draw:
            self.canvas.draw_idle()
        else:
            # The caller will repaint the existing blit layers after this
            # panel-only rebuild; avoid queuing a full-figure draw here.
            self._blit_backgrounds = {}

    @property
    def display_metric_label(self) -> str:
        return {"shift": "Shift (meV)", "energy": "Energy (eV)", "splitting": "Splitting (meV)"}.get(
            self.state.feature_metric, "Shift (meV)"
        )

    def set_candidates(self, candidates: Iterable[Mapping[str, Any]]) -> None:
        self._candidates = [dict(item) for item in candidates] + [dict(item) for item in self._manual_candidates]
        self._assign_candidate_display_ids()
        self.set_candidate_filter(self.candidate_filter_combo.currentText())

    def _assign_candidate_display_ids(self) -> None:
        """Assign stable short IDs to the complete catalog before filtering."""
        counters: dict[tuple[str, str], int] = {}
        ordered = sorted(self._candidates, key=lambda item: (
            str(item.get("domain", "")).casefold(), str(item.get("kind", "")).casefold(),
            float(self._candidate_center(item) if self._candidate_center(item) is not None else np.inf),
            str(item.get("id", ""))))
        for item in ordered:
            domain = str(item.get("domain", item.get("source", "spectrum"))).casefold()
            kind = str(item.get("kind", "feature")).casefold()
            key = (domain, "max" if kind in {"peak", "mcd_max", "max"} else "min" if kind in {"dip", "mcd_min", "min"} else kind)
            counters[key] = counters.get(key, 0) + 1
            if domain == "mcd":
                item["display_id"] = str(item.get("display_id") or item.get("label") or counters[key]) if key[1] == "window" else f"M{'+' if key[1] == 'max' else '-'}{counters[key]}"
            elif domain == "spectrum":
                item["display_id"] = f"{'P' if key[1] == 'peak' else 'D'}{counters[key]}"
            else:
                item["display_id"] = f"F{counters[key]}"

    def set_candidate_filter(self, value: str) -> None:
        mode = str(value).casefold()
        if mode == "all":
            self.visible_candidates = list(self._candidates)
        elif mode == "history":
            self.visible_candidates = [item for item in self._candidates if item.get("history")]
        elif mode == "recommended":
            self.visible_candidates = [item for item in self._candidates if item.get("recommended")]
        elif mode == "spectrum":
            self.visible_candidates = [
                item for item in self._candidates
                if str(item.get("domain", "")).casefold() == "spectrum"
                or str(item.get("source", "spectrum")).casefold() in {"spectrum", "raw", "corrected"}
            ]
        else:
            self.visible_candidates = [
                item for item in self._candidates
                if str(item.get("domain", "")).casefold() == "mcd"
                or str(item.get("source", "mcd")).casefold().startswith("mcd")
            ]
        self.recommended_only_chk.setEnabled(mode in {"mcd", "all"})
        if self.recommended_only_chk.isChecked() and mode in {"mcd", "all"}:
            self.visible_candidates = [item for item in self.visible_candidates
                                       if str(item.get("domain", "")).casefold() != "mcd"
                                       or bool(item.get("recommended", item.get("metadata", {}).get("recommended", False)
                                                if isinstance(item.get("metadata", {}), Mapping) else False))]
        # Keep every detected candidate available in every filter.  The pure
        # detector marks one recommendation representative per energy group;
        # presentation must not turn that review hint into data loss.
        if any(self._candidate_center(item) is not None for item in self.visible_candidates):
            self.visible_candidates = sorted(
                self.visible_candidates,
                key=lambda item: (float(self._candidate_center(item) if self._candidate_center(item) is not None else np.inf), str(item.get("id", ""))),
            )
        current_id = self.state.selected_feature_id
        self.candidate_combo.blockSignals(True)
        self.candidate_combo.clear()
        ranks: dict[tuple[str, str], int] = {}
        for item in self.visible_candidates:
            identifier = str(item.get("id", item.get("label", "feature")))
            kind = str(item.get("kind", "feature")).title()
            domain = str(item.get("domain", item.get("source", "spectrum"))).casefold()
            key = (domain, kind.casefold())
            ranks[key] = ranks.get(key, 0) + 1
            display_id = str(item.get("display_id") or item.get("label") or identifier)
            center = self._candidate_center(item)
            locator = str(metadata.get("detection_mode", self.state.detection_mode)) if isinstance(metadata := item.get("metadata", {}), Mapping) else self.state.detection_mode
            suffix = (f" · {float(center):.4f} eV · {float(item.get('width_mev', 0.0)):.4g} meV window"
                      if center is not None and kind.casefold() == "window" else
                      f" · Ref {float(center):.4f} eV · {'Raw extrema' if locator == 'raw' else 'Residual extrema'}" if center is not None else "")
            metadata = item.get("metadata", {})
            recommended = bool(item.get("recommended", metadata.get("recommended", False) if isinstance(metadata, Mapping) else False))
            origin = "Saved + Recommended" if item.get("history") and recommended else "Saved" if item.get("history") else kind
            self.candidate_combo.addItem(f"{display_id}{' ★' if recommended else ''} · {origin}{suffix}", identifier)
            tooltip = str(item.get("id", ""))
            if kind.casefold() == "window" and isinstance(metadata, Mapping):
                tooltip = (f"Window {float(center):.6f} eV · {float(item.get('width_mev', 0.0)):.4g} meV\n"
                           f"SNR {float(metadata.get('snr', 0.0)):.3g}; score {float(metadata.get('score', 0.0)):.3g}; "
                           "quality gate SNR >= 3")
            if item.get("history"):
                history = item["history"]
                tooltip = (f"Saved center {float(center):.6f} eV · window {float(item['width_mev']):g} meV\n"
                           f"Last saved: {history.get('last_used', '')}\nSaved records: {history.get('uses', 1)}\n"
                           + ("Also recommended for the current data\n" if recommended else "")
                           + "\n".join(history.get('records', ())))
            self.candidate_combo.setItemData(self.candidate_combo.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole)
        if self.visible_candidates:
            index = next((i for i, item in enumerate(self.visible_candidates)
                          if str(item.get("id", "")) == str(current_id)), None)
            if index is None:
                # Start at the candidate nearest the independent MCD window;
                # this keeps the initial view useful without moving the
                # scientific window to follow a feature.
                centers = np.asarray([
                    float(self._candidate_center(item)) if self._candidate_center(item) is not None else np.nan
                    for item in self.visible_candidates
                ], dtype=float)
                valid = np.flatnonzero(np.isfinite(centers))
                if valid.size and mode in {"mcd", "spectrum"} and self.state.window_center_ev:
                    index = int(valid[np.argmin(np.abs(centers[valid] - self.state.window_center_ev))])
                elif valid.size:
                    prominence = np.asarray([float(self.visible_candidates[i].get("prominence", 0.0)) for i in valid])
                    index = int(valid[np.argmax(prominence)])
                else:
                    index = 0
            self.candidate_combo.setCurrentIndex(index)
            # Signals are blocked while the catalog is rebuilt, therefore the
            # selection state must be updated explicitly for auto tracking.
            self.state.selected_feature_id = str(self.visible_candidates[index].get("id", ""))
            mcd_count = sum(str(item.get("domain", "")).casefold() == "mcd" for item in self._candidates)
            spectrum_count = sum(str(item.get("domain", "")).casefold() == "spectrum" for item in self._candidates)
            recommended_count = sum(bool(item.get("recommended", item.get("metadata", {}).get("recommended", False) if isinstance(item.get("metadata", {}), Mapping) else False)) for item in self._candidates if str(item.get("domain", "")).casefold() == "mcd")
            self.candidate_status.setText(f"{mcd_count} MCD · {spectrum_count} Spectrum · {recommended_count} recommended")
        else:
            self.state.selected_feature_id = None
            self.candidate_status.setText("No MCD region passes the response threshold" if self.recommended_only_chk.isChecked() else "No candidates in this filter")
        self.candidate_combo.blockSignals(False)
        if self.state.selected_feature_id != current_id:
            self.candidate_selection_changed.emit(self.state.selected_feature_id)
        if self.axes:
            self._draw_feature_overlay()

    def refresh_catalog(self, candidates: Iterable[Mapping[str, Any]],
                        analysis: Mapping[str, Any] | None = None) -> None:
        """Refresh candidate/result panels while preserving the four plots.

        Worker completion changes the catalog and feature result, but the
        loaded MCD arrays and map are unchanged.  Keep their artists alive so
        a background detector result cannot replace an interactive map.
        """
        self.set_candidates(candidates)
        if analysis is not None and self.publish_analysis(analysis) and self.axes:
            self._refresh_feature_display(queue_draw=False)
        if self.axes:
            self._draw_feature_overlay()
            if not self._blit_update():
                self.canvas.draw_idle()

    def _resolve_current_measurement(self, candidate: Mapping[str, Any] | None) -> tuple[Any | None, str]:
        if candidate is None:
            return None, "No candidate selected"
        points = candidate.get("measured_points", ())
        if not points:
            return None, "Manual reference has no measured locator" if candidate.get("manual") else "No measured locator at displayed B"
        domain = str(candidate.get("domain", "")).casefold()
        source = str(candidate.get("source", ""))
        branch = str(candidate.get("branch", ""))
        if domain == "spectrum" and self._result is not None:
            key = (source, branch)
            row_index = self._displayed_spectrum_rows.get(key)
            if row_index is None:
                return None, "No measured locator at displayed B"
            point = next((p for p in points if int(p.get("row_index", -1) if isinstance(p, Mapping) else getattr(p, "row_index", -1)) == row_index), None)
            return (point, "") if point is not None else (None, "No measured locator at displayed B")
        if self._result is not None and branch:
            fields = np.asarray(getattr(self._result, "pair_b", ()), dtype=float).ravel()
            labels = np.asarray(getattr(self._result, "pair_labels", ()), dtype=object).ravel()
            if labels.size == fields.size and not np.any(np.asarray([str(x) for x in labels]) == branch):
                return None, "Branch hidden"
            valid = [p for p in points if isinstance(p, Mapping) or hasattr(p, "row_index")]
            if valid:
                target_b = float(fields[self.state.selected_b_index]) if self.state.selected_b_index < fields.size else np.nan
                point = next((p for p in valid if np.isfinite(target_b) and abs(float(p.get("field_t", np.nan) if isinstance(p, Mapping) else p.field_t) - target_b) <= 1e-10), None)
                if point is not None:
                    return point, ""
        return None, "No measured locator at displayed B"

    @staticmethod
    def _candidate_center(candidate: Mapping[str, Any]) -> float | None:
        value = candidate.get("center_ev", candidate.get("energy_ev"))
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def _candidate_changed(self, index: int) -> None:
        identifier = self.candidate_combo.currentData(Qt.ItemDataRole.UserRole)
        item = next((candidate for candidate in self.visible_candidates if str(candidate.get("id", "")) == str(identifier)), None)
        if item is None:
            item = self.visible_candidates[index] if 0 <= index < len(self.visible_candidates) else None
        self.state.selected_feature_id = None if item is None else str(item.get("id", ""))
        if self.axes:
            self._draw_feature_overlay()

    def _add_manual_candidate_dialog(self) -> None:
        center, accepted = QInputDialog.getDouble(
            self, "Manual feature center", "Center energy (eV):",
            float(self.state.window_center_ev), -1e6, 1e6, 6,
        )
        if not accepted:
            return
        self._manual_candidate_number += 1
        item = {
            "id": f"manual-{self._manual_candidate_number}",
            "label": f"M{self._manual_candidate_number}",
            "domain": "spectrum", "source": "manual", "kind": "peak",
            "center_ev": float(center), "energy_ev": float(center), "manual": True,
        }
        self._manual_candidates.append(item)
        self.set_candidate_filter(self.candidate_filter_combo.currentText())
        index = self.candidate_combo.findData(item["id"])
        if index >= 0:
            self.candidate_combo.setCurrentIndex(index)
        self.manual_candidate_added.emit(dict(item))

    def step_candidate(self, delta: int) -> None:
        count = self.candidate_combo.count()
        if count:
            self.candidate_combo.setCurrentIndex((self.candidate_combo.currentIndex() + int(delta)) % count)

    def set_selected_b(self, index: int) -> None:
        if self._result is None:
            return
        fields = np.asarray(getattr(self._result, "pair_b", ()), dtype=float).ravel()
        if not fields.size:
            return
        index = max(0, min(int(index), fields.size - 1))
        self.state.selected_b_index = index
        fields_value = float(fields[index])
        blocked = self.spectra_b_spin.blockSignals(True)
        self.spectra_b_spin.setValue(fields_value)
        self.spectra_b_spin.blockSignals(blocked)
        blocked = self.spectra_b_slider.blockSignals(True)
        self.spectra_b_slider.setValue(index)
        self.spectra_b_slider.blockSignals(blocked)
        for line, values, row_indices, row_fields in self._artists.get("spectrum_lines", ()):
            try:
                row = index
                line.set_ydata(values[row])
                measured_b = float(row_fields[row])
                old_label = str(line.get_label())
                prefix = old_label.split(" · B=", 1)[0]
                updated_label = f"{prefix} · B={measured_b:+.4g} T"
                line.set_label(updated_label)
                for legend_sample, source_line in self._legend_line_map.items():
                    if source_line is line:
                        legend_sample.set_label(updated_label)
                        text = self._spectrum_legend_texts.get(legend_sample)
                        if text is not None:
                            text.set_text(updated_label)
            except (AttributeError, IndexError, TypeError):
                pass
        if self.axes.get("spectra") is not None:
            self.axes["spectra"].relim()
            self.axes["spectra"].autoscale_view(scalex=False, scaley=True)
        for line, values in self._artists.get("mcd_spectrum_lines", ()):
            line.set_ydata(values[index])
        if "mcd_spectra" in self.axes:
            self.axes["mcd_spectra"].relim()
            self.axes["mcd_spectra"].autoscale_view(scalex=False, scaley=True)
        # B changes can change both spectral y ranges; cached ticks must be rebuilt.
        self._blit_backgrounds = {}
        map_value = float(fields[index])
        if self._map_fields.size:
            map_value = float(self._map_fields[np.argmin(np.abs(self._map_fields - map_value))])
        for artist in self._artists.get("b_map_cursor", ()):
            try:
                artist.set_ydata([map_value, map_value])
            except (AttributeError, TypeError):
                pass
        for artist in self._artists.get("b_trace_cursor", ()):
            try:
                artist.set_xdata([float(fields[index]), float(fields[index])])
            except (AttributeError, TypeError):
                pass
        for artist in self._artists.get("b_status", ()):
            try:
                artist.set_text(f"B={fields[index]:+.5g} T")
            except (AttributeError, TypeError):
                pass
        self.draw_only_updates += 1
        if self._rendering:
            return
        if not self._blit_update():
            self.canvas.draw_idle()

    def _on_map_click(self, event: Any) -> None:
        """Normal map gestures select energy; Ctrl-click explicitly selects B."""
        if (event is None or not self._owns_current_figure()
                or event.inaxes is not self.axes.get("mcd_map")
                or getattr(event, "button", None) != 1 or self._navigation_active()):
            return
        key = str(getattr(event, "key", "") or "").casefold()
        if "control" in key or "ctrl" in key:
            if event.ydata is not None:
                self._set_b_from_value(float(event.ydata))
            return
        if event.xdata is None:
            return
        self._window_drag_active = True
        self._move_window_to_energy(float(event.xdata))

    def _navigation_active(self) -> bool:
        return bool(getattr(getattr(self.canvas, "toolbar", None), "mode", ""))

    def _move_window_to_energy(self, value: float) -> None:
        energy = np.asarray(getattr(self._result, "energy_ev", ()), dtype=float)
        energy = energy[np.isfinite(energy)]
        if not energy.size or not np.isfinite(value):
            return
        half = self.state.window_width_mev * .0005
        low, high = float(energy.min()), float(energy.max())
        center = float(np.clip(value, low + half, high - half)) if high - low >= 2 * half else (low + high) / 2
        self.set_window(center, self.state.window_width_mev)

    def _on_map_motion(self, event: Any) -> None:
        if (self._window_drag_active and not self._navigation_active()
                and self._owns_current_figure() and event.inaxes is self.axes.get("mcd_map")
                and event.xdata is not None):
            self._move_window_to_energy(float(event.xdata))

    def _on_map_release(self, event: Any) -> None:
        if not self._window_drag_active:
            return
        self._window_drag_active = False
        if self._owns_current_figure():
            # Commit the numerical trace once, after the lightweight gesture.
            self.set_window(self.state.window_center_ev, self.state.window_width_mev)
            self.window_center_changed.emit(self.state.window_center_ev)

    def set_window(self, center_ev: float, width_mev: float, *, redraw: bool = True) -> None:
        self.state.window_center_ev = float(center_ev)
        self.state.window_width_mev = float(width_mev)
        for patch in self._artists.get("window", ()):
            try:
                patch.set_x(float(center_ev) - float(width_mev) * 0.0005)
                patch.set_width(float(width_mev) * 0.001)
            except (AttributeError, TypeError):
                pass
        for index, edge in enumerate(self._artists.get("window_edges", ())):
            direction = -1 if index % 2 == 0 else 1
            value = float(center_ev) + direction * float(width_mev) * .0005
            edge.set_xdata([value, value])
        for artist in self._artists.get("window_cursor", ()):
            try:
                artist.set_xdata([float(center_ev), float(center_ev)])
            except (AttributeError, TypeError):
                pass
        for label in self._artists.get("window_status", ()):
            label.set_text(self._window_text())
        for artist in self._artists.get("feature_selection", ()):
            if getattr(artist, "_mcd_window_label", "") == "Window":
                try:
                    if hasattr(artist, "set_position"):
                        artist.set_position((float(center_ev), .80))
                        artist.set_text(f"Window {float(center_ev):.4f} eV")
                    else:
                        artist.set_xdata([float(center_ev), float(center_ev)])
                except (AttributeError, TypeError):
                    pass
        # Moving the energy overlays must not reduce the full spectral arrays
        # or rescale MCD(B): either operation invalidates the blit backgrounds.
        # The release handler (and ordinary parameter edits) commits the trace.
        if not self._window_drag_active:
            self._refresh_mcd_window_trace()
        self.draw_only_updates += 1
        if not redraw:
            return
        keys = ("mcd_map", "spectra", "mcd_spectra") if self._window_drag_active else None
        if not self._blit_update(axes_keys=keys):
            self.canvas.draw_idle()

    def _refresh_mcd_window_trace(self) -> None:
        """Update the existing MCD-vs-B lines for a center/width edit."""
        if self._result is None or not self._artists.get("mcd_trace_lines"):
            return
        metric = str(self.state.window_metric).casefold().replace(" ", "_")
        if "integral" in metric:
            metric_key = "integral"
        elif "field" in metric and "absolute" in metric:
            metric_key = "field_signed_absolute_mean"
        elif "absolute" in metric:
            metric_key = "absolute_mean"
        else:
            metric_key = "mean"
        try:
            from core.mcd import pair_window_trace_by_branch
            traces = pair_window_trace_by_branch(
                self._result, self.state.window_center_ev, self.state.window_width_mev,
                metrics=(metric_key,), include_raw=False,
            )
        except (ImportError, AttributeError, TypeError, ValueError, FloatingPointError):
            return
        for branch, line in self._artists.get("mcd_trace_lines", ()):
            points = traces.get(branch, {}).get(f"corrected_{metric_key}")
            if points is not None:
                line.set_data(points[0], points[1])
        self._update_mcd_trace_y_limits()

    def _update_mcd_trace_y_limits(self) -> None:
        """Size the trace axis from measured data, excluding fit extrapolations."""
        axis = self.axes.get("mcd_vs_b")
        arrays = [np.asarray(line.get_ydata(), dtype=float) for _, line in self._artists.get("mcd_trace_lines", ()) if line.get_visible()]
        finite = np.concatenate(arrays) if arrays else np.array([])
        finite = finite[np.isfinite(finite)]
        if axis is not None and finite.size:
            axis.set_autoscaley_on(False)
            pad = max(float(np.ptp(finite)) * .05, 1e-12)
            limits = (float(finite.min()) - pad, float(finite.max()) + pad)
            if tuple(axis.get_ylim()) != limits:
                axis.set_ylim(*limits)
                self._blit_backgrounds = {}

    def update_mcd_slope_readout(self, slopes: Mapping[str, Any] | None) -> None:
        """Refresh compact slope results after a window edit."""
        axis = self.axes.get("mcd_vs_b")
        if axis is None:
            return
        self._set_slope_details(slopes, axis)

    def _set_slope_details(self, slopes: Mapping[str, Any] | None, axis: Any) -> None:
        """Keep full fit details accessible while the plot shows one compact line."""
        self._update_mcd_fit_lines(slopes, axis)
        if not isinstance(slopes, Mapping):
            for key in ("slope_readout", "slope_difference_readout"):
                for artist in self._artists.get(key, ()):
                    artist.set_text("")
            self._slope_details_text = "No slope fits available."
            return
        fits = slopes.get("fits", ())
        if isinstance(fits, Mapping):
            fits = list(fits.values())
        fits = list(fits) if isinstance(fits, (list, tuple)) else []
        valid = sum(1 for fit in fits if isinstance(fit, Mapping) and fit.get("status") == "ok")
        details = [f"Slope fits: {valid}/{len(fits)} valid"]
        for fit in fits:
            if not isinstance(fit, Mapping) or fit.get("status") != "ok" or fit.get("slope") is None:
                continue
            error = fit.get("slope_se")
            suffix = f" ± {float(error):.2g}" if error is not None else ""
            details.append(f"{fit.get('region', '')}/{fit.get('branch', '')}: {float(fit['slope']):.3g}{suffix}")
        detail_text = "\n".join(details)
        differences = slopes.get("differences", ())
        if isinstance(differences, Mapping):
            differences = list(differences.values())
        diff_lines = []
        for difference in differences if isinstance(differences, (list, tuple)) else ():
            if not isinstance(difference, Mapping) or difference.get("slope_difference") is None:
                continue
            error = difference.get("standard_error")
            suffix = f" ± {float(error):.2g}" if error is not None else ""
            if str(difference.get("comparison", "")) == "low_minus_high":
                label = f"Δ {difference.get('reference_region', 'low')}−{difference.get('region', '')} / {difference.get('branch_a', '')}"
            elif str(difference.get("comparison", "")) == "high_minus_low":
                label = f"Δ {difference.get('region', '')}−{difference.get('reference_region', 'low')} / {difference.get('branch_a', '')}"
            else:
                label = f"Δ {difference.get('region', '')} / {difference.get('branch_a', '')}−{difference.get('branch_b', '')}"
            diff_lines.append(f"{label}: {float(difference['slope_difference']):.3g}{suffix}")
        self._slope_details_text = "\n".join(details + (["", *diff_lines] if diff_lines else []))
        compact = f"Fits {valid}/{len(fits)}"
        if diff_lines:
            compact += f" · Δ fits {len(diff_lines)}"
        existing = self._artists.get("slope_readout", ())
        if existing:
            existing[0].set_text(compact)
        else:
            self._artists["slope_readout"] = [axis.text(0.98, 0.96, compact, transform=axis.transAxes, ha="right", va="top", fontsize=6.5)]
        for artist in self._artists.get("slope_difference_readout", ()):
            artist.set_text("")
        controls = getattr(self.window(), "mcd_unified_controls", None)
        details_button = getattr(controls, "slope_details_btn", None)
        if details_button is not None:
            details_button.setToolTip(self._slope_details_text)

    def _update_mcd_fit_lines(self, slopes: Mapping[str, Any] | None, axis: Any) -> None:
        existing = self._artists.setdefault("mcd_fit_lines", {})
        extensions = self._artists.setdefault("mcd_fit_extensions", {})
        annotations = self._artists.setdefault("slope_annotations", {})
        fits = slopes.get("fits", ()) if isinstance(slopes, Mapping) else ()
        if isinstance(fits, Mapping):
            fits = list(fits.values())
        desired: dict[tuple[str, str], Mapping[str, Any]] = {}
        for fit in fits if isinstance(fits, (list, tuple)) else ():
            if not isinstance(fit, Mapping) or fit.get("status") != "ok":
                continue
            if fit.get("slope") is None or fit.get("intercept") is None:
                continue
            low, high = fit.get("field_min_t"), fit.get("field_max_t")
            if low is None or high is None:
                continue
            key = (str(fit.get("region", "")), str(fit.get("branch", "")))
            if (key[1] == "B increasing" and not self.show_inc_chk.isChecked()) or (key[1] == "B decreasing" and not self.show_dec_chk.isChecked()):
                continue
            desired[key] = fit
        colors = {"low": "#b45309", "high_positive": "#7c3aed", "high_negative": "#be123c"}
        changed = False
        for key in list(existing):
            if key not in desired:
                try:
                    existing[key].remove()
                except (AttributeError, ValueError):
                    pass
                existing.pop(key, None)
                for artist_group in (extensions, annotations):
                    artist = artist_group.pop(key, None)
                    if isinstance(artist, (tuple, list)):
                        for item in artist:
                            try:
                                item.remove()
                            except (AttributeError, ValueError):
                                pass
                    elif artist is not None:
                        try:
                            artist.remove()
                        except (AttributeError, ValueError):
                            pass
                changed = True
        visible_min, visible_max = axis.get_xlim()
        for fit_index, (key, fit) in enumerate(desired.items()):
            segments = fit_display_segments(fit, visible_min=visible_min, visible_max=visible_max)
            xs, ys = segments["actual_x"], segments["actual_y"]
            ext_xs, ext_ys = segments["extension_x"], segments["extension_y"]
            color = colors.get(key[0], "#a16207")
            style = "-" if "increas" in key[1].casefold() else "--"
            line = existing.get(key)
            if line is None:
                line = axis.plot(xs, ys, ls=style, lw=2.3, alpha=1.0, zorder=5,
                                 color=color,
                                 label=f"Fit {key[0]} · {key[1]}")[0]
                existing[key] = line
                changed = True
            else:
                line.set_data(xs, ys)
            ext = extensions.get(key)
            if ext is None:
                ext = axis.plot(ext_xs, ext_ys, ls=style, lw=1.4, alpha=.32, zorder=3,
                                color=color, label="_nolegend_", clip_on=True)[0]
                extensions[key] = ext
                changed = True
            else:
                ext.set_data(ext_xs, ext_ys)
            error = fit.get("slope_se")
            region_label = {"low": "Low", "high_positive": "High+", "high_negative": "High−"}.get(str(fit.get("region", "")), str(fit.get("region", "")))
            branch_label = "Inc" if "increas" in key[1].casefold() else "Dec" if "decreas" in key[1].casefold() else str(key[1])
            slope_text = f"{region_label} {branch_label}: {float(fit['slope']):.3g}"
            if error is not None:
                slope_text += f" ± {float(error):.2g}"
            annotation = annotations.get(key)
            midpoint = float(np.mean(xs))
            mid_y = float(fit["slope"]) * midpoint + float(fit["intercept"])
            view_low, view_high = (float(value) for value in axis.get_xlim())
            view_span = max(view_high - view_low, 1e-12)
            if key[0] == "high_negative":
                anchor_x, align = max(midpoint, view_low + .18 * view_span), "left"
            elif key[0] == "high_positive":
                anchor_x, align = min(midpoint, view_high - .18 * view_span), "right"
            else:
                anchor_x, align = min(max(midpoint, view_low + .10 * view_span), view_high - .10 * view_span), "center"
            offset = 8 if fit_index % 2 == 0 else -10
            if annotation is None:
                annotation = axis.annotate(
                    slope_text, xy=(anchor_x, mid_y), xytext=(0, offset),
                    textcoords="offset points", ha=align, va="center", fontsize=6,
                    color=color, annotation_clip=True,
                    bbox={"boxstyle": "round,pad=.15", "fc": "white", "ec": "none", "alpha": .72},
                )
                # Keep annotation pixels inside the bottom-left axes so the
                # panel gutter and its blit background never receive labels.
                annotation.set_clip_on(True)
                annotation.set_clip_path(axis.patch)
                annotations[key] = annotation
                changed = True
            else:
                annotation.set_text(slope_text)
                annotation.xy = (anchor_x, mid_y)
                annotation.set_position((0, offset))
                annotation.set_ha(align)
        if changed and self._blit_backgrounds:
            self._blit_backgrounds = {}

    def retain_current_window(self, label: str = "MCD", *, settings: Mapping[str, Any] | None = None) -> RetainedMcdWindow:
        return self.state.retain_window(
            label, self.state.window_center_ev, self.state.window_width_mev,
            source_generation=self._source_generation, settings=settings,
        )

    def retain_selected_feature(
        self, feature: Mapping[str, Any], *, settings: Mapping[str, Any] | None = None,
        analysis_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = dict(feature)
        snapshot["source_generation"] = self._source_generation
        snapshot["settings"] = dict(settings or {})
        # Retention is a numerical snapshot.  Keep the selected measured
        # tracks/links/splitting with the candidate metadata so a later source
        # redraw cannot silently turn Save Results into an inventory-only dump.
        if analysis_payload:
            snapshot["analysis_payload"] = dict(analysis_payload)
        self.state.retained_features.append(snapshot)
        return snapshot

    def render(self, result: Any, *, candidates: Iterable[Mapping[str, Any]] | None = None,
               analysis: Mapping[str, Any] | None = None) -> None:
        """Render all four fixed regions from an immutable result snapshot."""
        self._rendering = True
        self._result = result
        self._artists = {}
        self._feature_energy_axis = None
        self._blit_backgrounds = {}
        # Detection is a cancellable worker concern.  A first render must stay
        # responsive even when the source contains hundreds of B slices.
        self.set_candidates(candidates if candidates is not None else ())
        if analysis is not None:
            self.publish_analysis(analysis)
        self._xlim_cids.clear()
        self.figure.clear()
        # The shared production canvas can be short (laptop/1280×720) or
        # wide/high (publication/export).  Explicit compact margins keep all
        # four panels large while leaving enough row/column gutter for labels.
        self.figure.subplots_adjust(left=0.065, right=0.935, bottom=0.14, top=0.91,
                                    hspace=0.50, wspace=0.27)
        grid = self.figure.add_gridspec(2, 2, hspace=0.50, wspace=0.27)
        self._plot_grid = grid
        self._update_subplot_spacing()
        self.axes = {
            "mcd_map": self.figure.add_subplot(grid[0, 0]),
            "spectra": self.figure.add_subplot(grid[0, 1], sharex=self.figure.axes[0]),
            "mcd_vs_b": self.figure.add_subplot(grid[1, 0]),
        }
        if self.result_panel_combo.currentIndex() == 1:
            self.axes["feature_vs_b"] = self.figure.add_subplot(grid[1, 1], sharex=self.axes["mcd_vs_b"])
        else:
            self.axes["mcd_spectra"] = self.figure.add_subplot(grid[1, 1], sharex=self.axes["mcd_map"])
        from core.mcd import format_mcd_acquisition_conditions
        conditions = format_mcd_acquisition_conditions(getattr(result, "acquisition_conditions", {}))
        mode = str(getattr(result, "summary", {}).get("correction_mode", ""))
        correction = {"global": "Global gain", "pair_scale": "Per-pair gain",
                      "pair_affine": "Per-pair affine", "pair_spectral": "Spectral baseline"}.get(mode, mode)
        subtitle = " | ".join(part for part in (conditions, f"Correction: {correction}" if correction else "") if part)
        self.figure.suptitle(subtitle, fontsize=9, y=.995)
        for axis in (self.axes["mcd_map"], self.axes["spectra"]):
            self._xlim_cids.append((axis, axis.callbacks.connect("xlim_changed", self._on_energy_xlim_changed)))
        for axis in self.axes.values():
            axis.grid(alpha=0.25)
        energy = np.asarray(getattr(result, "energy_ev", ()), dtype=float).ravel()
        try:
            from core.mcd_peak_shift import spectrum_energy_order
            order = np.asarray(spectrum_energy_order(result), dtype=int)
        except (ImportError, AttributeError, TypeError, ValueError):
            order = np.argsort(energy) if energy.size else np.array([], dtype=int)
        fields = np.asarray(getattr(result, "pair_b", ()), dtype=float).ravel()
        self.state.selected_b_index = min(self.state.selected_b_index, max(0, fields.size - 1))
        blocked = self.spectra_b_slider.blockSignals(True)
        self.spectra_b_slider.setRange(0, max(0, fields.size - 1))
        self.spectra_b_slider.setValue(self.state.selected_b_index)
        self.spectra_b_slider.blockSignals(blocked)
        mcd = self._mcd_values(result, order)
        self._draw_map(energy, order, fields, mcd)
        self._draw_spectra(result, energy, order, fields)
        self._draw_mcd_vs_b(result, energy, order, fields, mcd)
        if "feature_vs_b" in self.axes:
            self._draw_feature_vs_b(fields)
        else:
            self._draw_mcd_spectra(result, energy, order, fields)
        self._draw_feature_overlay()
        for axis in self.axes.values():
            axis.tick_params(labelsize=8)
            axis.xaxis.label.set_fontsize(9)
            axis.yaxis.label.set_fontsize(9)
        for text in (*self._artists.get("window_status", ()), *self._artists.get("b_status", ())):
            text._use_theme_text = True
        self.render_count += 1
        self.set_selected_b(self.state.selected_b_index)
        self._prepare_blit()
        self._rendering = False

    def _on_energy_xlim_changed(self, axis: Any) -> None:
        if self._overlay_syncing or axis not in (self.axes.get("mcd_map"), self.axes.get("spectra")):
            return
        self._sync_candidate_overlay()
        self._blit_backgrounds = {}

    @staticmethod
    def _default_candidates(result: Any) -> list[dict[str, Any]]:
        """Build a complete display catalog while pure analysis is optional."""
        try:
            from core.mcd_analysis import detect_analysis_features
            features = detect_analysis_features(result)
            output = []
            for item in features:
                if isinstance(item, Mapping):
                    output.append(dict(item))
                else:
                    values = {name: getattr(item, name) for name in ("id", "kind", "source", "center_ev") if hasattr(item, name)}
                    output.append(values)
            if output:
                return output
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
        # The legacy detector remains a safe fallback for a result produced
        # by older processing code; each finite B slice is intentionally kept
        # separate for MCD extrema rather than averaging signed rows away.
        energy = np.asarray(getattr(result, "energy_ev", ()), dtype=float).ravel()
        if not energy.size:
            return []
        try:
            from core.mcd_peak_shift import detect_reflection_peaks, spectrum_energy_order, source_spectra
            order = np.asarray(spectrum_energy_order(result), dtype=int)
            spectra = source_spectra(result, "raw pos")
            row = np.nanmedian(np.asarray(spectra, dtype=float), axis=0)
            axis = energy
            output = []
            for kind, values in (("peak", row), ("dip", -row)):
                found = detect_reflection_peaks(axis, values, max_peaks=24, feature_kind=kind)
                for index, candidate in enumerate(found, 1):
                    output.append({"id": f"{'P' if kind == 'peak' else 'D'}{index}", "kind": kind, "source": "spectrum", "center_ev": float(candidate.energy_ev)})
            mcd = np.asarray(getattr(result, "pair_mcd_corrected", ()), dtype=float)
            if mcd.ndim == 2:
                extrema = np.nanmedian(mcd, axis=0)
                for index, candidate in enumerate(detect_reflection_peaks(axis, extrema, max_peaks=24, feature_kind="peak"), 1):
                    output.append({"id": f"M+{index}", "kind": "peak", "source": "mcd", "center_ev": float(candidate.energy_ev)})
                for index, candidate in enumerate(detect_reflection_peaks(axis, -extrema, max_peaks=24, feature_kind="dip"), 1):
                    output.append({"id": f"M-{index}", "kind": "dip", "source": "mcd", "center_ev": float(candidate.energy_ev)})
            return output
        except (ImportError, AttributeError, TypeError, ValueError):
            return []

    @staticmethod
    def _mcd_values(result: Any, order: np.ndarray) -> np.ndarray:
        for name in ("pair_mcd_corrected", "pair_mcd_raw"):
            values = getattr(result, name, None)
            if values is not None:
                array = np.asarray(values, dtype=float)
                return array[:, order] if order.size and array.ndim == 2 else array
        try:
            array = np.asarray(result.cube("Combo").Z, dtype=float)
            return array[:, order] if order.size and array.ndim == 2 else array
        except (AttributeError, KeyError, ValueError):
            return np.empty((0, 0), dtype=float)

    def _draw_map(self, energy: np.ndarray, order: np.ndarray, fields: np.ndarray, mcd: np.ndarray) -> None:
        axis = self.axes["mcd_map"]
        map_energy, map_fields, map_values = energy, fields, mcd
        cube_map = False
        map_name = "Corrected"
        owner = self.parentWidget()
        while owner is not None and not hasattr(owner, "mcd_spins"):
            owner = owner.parentWidget()
        cmap = "RdBu_r"
        try:
            from core.colormaps import resolve_cmap
            cmap_name = str(getattr(getattr(owner, "mcd_cmap"), "currentText")())
            cmap = resolve_cmap(cmap_name, "RdBu_r")
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
        try:
            map_name = str(getattr(getattr(owner, "mcd_map_combo"), "currentText")())
            cube = self._result.cube(map_name)
            map_energy = np.asarray(cube.energy, dtype=float).ravel()
            map_fields = np.asarray(cube.gate, dtype=float).ravel()
            map_values = np.asarray(cube.Z, dtype=float)
            cube_map = True
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        self._map_fields = np.asarray(map_fields, dtype=float).ravel()
        if map_energy.size and map_fields.size and map_values.ndim == 2 and map_values.shape[0] == map_fields.size:
            finite = map_values[np.isfinite(map_values)]
            limit = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
            vmin, vmax = -limit, limit
            try:
                spins = owner.mcd_spins
                configured_min = float(spins["vmin"].value())
                configured_max = float(spins["vmax"].value())
                if np.isfinite(configured_min) and np.isfinite(configured_max) and configured_max > configured_min:
                    vmin, vmax = configured_min, configured_max
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = -max(limit, 1e-12), max(limit, 1e-12)
            labels = np.asarray(getattr(self._result, "pair_labels", ()), dtype=object).ravel()
            if cube_map and map_values.shape[1] == map_energy.size:
                axis.pcolormesh(map_energy, map_fields, map_values, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
                labels = np.full(map_fields.size, map_name, dtype=object)
            groups: list[tuple[str, np.ndarray]] = []
            if not axis.collections:
                if labels.size == map_fields.size:
                    groups = [(str(label), np.flatnonzero(labels == label)) for label in dict.fromkeys(map(str, labels))]
                if not groups:
                    groups = [("B sweep", np.arange(map_fields.size))]
                for _label, indices in groups:
                    indices = indices[np.argsort(map_fields[indices], kind="stable")]
                    axis.pcolormesh(map_energy, map_fields[indices], map_values[indices], cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
            if axis.collections:
                try:
                    self._map_colorbar = self.figure.colorbar(axis.collections[-1], ax=axis, fraction=0.046, pad=0.04)
                    self._map_colorbar.ax.set_title("MCD", fontsize=8, pad=4)
                    self._map_colorbar.ax.tick_params(labelsize=7, pad=2)
                except (AttributeError, RuntimeError, ValueError):
                    self._map_colorbar = None
        axis.set_title(f"{map_name} MCD map")
        axis.set_xlabel("Energy (eV)")
        axis.set_ylabel("Magnetic field B (T)")
        try:
            spins = owner.mcd_spins
            xmin, xmax = float(spins["xmin"].value()), float(spins["xmax"].value())
            ymin, ymax = float(spins["ymin"].value()), float(spins["ymax"].value())
            if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                axis.set_xlim(xmin, xmax)
            if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                axis.set_ylim(ymin, ymax)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        center = self.state.window_center_ev
        patch = axis.axvspan(center - self.state.window_width_mev * 0.0005,
                             center + self.state.window_width_mev * 0.0005,
                             ymin=.985, ymax=1.0, color="#4f9d8f", alpha=0.65)
        cursor_value = map_fields[int(np.argmin(np.abs(map_fields - fields[self.state.selected_b_index])))] if map_fields.size and fields.size else 0.0
        cursor = axis.axhline(cursor_value,
                              color="#d97706", ls="--", lw=1.0)
        edges = [axis.axvline(center + direction * self.state.window_width_mev * .0005,
                    color="#666666", ls=(0, (4, 3)), lw=.85, alpha=.55, zorder=7)
                 for direction in (-1, 1)]
        self._artists["window_edges"] = edges
        self._artists["window"] = [patch]
        self._artists["b_map_cursor"] = [cursor]
        label = axis.text(.02, .04, self._window_text(), transform=axis.transAxes, fontsize=7, va="bottom")
        self._artists["window_status"] = [label]
        self._artists["b_status"] = [axis.text(0.98, 0.96, f"B={cursor_value:+.5g} T", transform=axis.transAxes, ha="right", va="top", fontsize=7)]

    def _window_text(self) -> str:
        return f"E = {self.state.window_center_ev:.6f} eV · Window = {self.state.window_width_mev:g} meV"

    def _add_energy_window(self, axis: Any) -> None:
        center, half = self.state.window_center_ev, self.state.window_width_mev * .0005
        patch = axis.axvspan(center - half, center + half, color="#4f9d8f", alpha=.12, zorder=0)
        cursor = axis.axvline(center, color="#0f766e", ls="--", lw=.9)
        self._artists.setdefault("window", []).append(patch)
        self._artists.setdefault("window_cursor", []).append(cursor)
        for direction in (-1, 1):
            edge = axis.axvline(center + direction * half, color="#666666", ls=":", lw=.8, alpha=.6)
            self._artists.setdefault("window_edges", []).append(edge)
        label = axis.text(.02, .97, self._window_text(), transform=axis.transAxes,
                          ha="left", va="top", fontsize=7)
        self._artists.setdefault("window_status", []).append(label)

    def _draw_spectra(self, result: Any, energy: np.ndarray, order: np.ndarray, fields: np.ndarray) -> None:
        """Display the four old-tab reflection curves at one acquisition pair."""
        axis = self.axes["spectra"]
        self._displayed_spectrum_rows = {}
        self._artists["spectrum_lines"] = []
        self._legend_line_map = {}
        self._spectrum_legend_texts = {}
        index = self.state.selected_b_index
        for source in ("raw", "corrected"):
            for channel, label, color in (("pos", "σ+", "#2563eb"), ("neg", "σ−", "#f59e0b")):
                values = np.asarray(getattr(result, f"pair_{source}_{channel}", ()), dtype=float)
                if values.ndim != 2 or index >= values.shape[0] or values.shape[1] != energy.size:
                    continue
                values = values[:, order]
                channel_fields = np.asarray(getattr(result, f"pair_b_{channel}", fields), dtype=float)
                if channel_fields.size != fields.size:
                    channel_fields = fields
                line = axis.plot(energy, values[index], color=color,
                                 ls="--" if source == "raw" else "-",
                                 alpha=.65 if source == "raw" else 1., lw=1.,
                                 label=f"{label} {source.title()} · B={channel_fields[index]:+.4g} T")[0]
                self._artists["spectrum_lines"].append((line, values, np.arange(fields.size), channel_fields))
        summary = getattr(result, "summary", {})
        if summary.get("correction_mode") in {"pair_scale", "pair_affine", "pair_spectral"}:
            from core.mcd import background_fit_regions
            for low, high in background_fit_regions(energy, summary.get("background_ranges_ev", ())):
                patch = axis.axvspan(low, high, color="0.75", alpha=.25, zorder=0)
                patch.set_gid("mcd-background-fit")
        axis.set_title("Paired spectra · Raw / Corrected", fontsize=10, pad=4)
        axis.margins(y=.25)
        axis.set_xlabel("Energy (eV)")
        axis.set_ylabel("Reflection (a.u.)")
        if self._artists["spectrum_lines"]:
            legend = axis.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper right")
            self._artists["spectra_legend"] = legend
            for sample, text, entry in zip(legend.get_lines(), legend.get_texts(), self._artists["spectrum_lines"]):
                sample.set_picker(5)
                self._legend_line_map[sample] = entry[0]
                self._spectrum_legend_texts[sample] = text
        cursor = axis.axvline(self.state.window_center_ev, color="#0f766e", ls="--", lw=.9)
        self._artists.setdefault("window_cursor", []).append(cursor)

    def _draw_mcd_spectra(self, result: Any, energy: np.ndarray, order: np.ndarray, fields: np.ndarray) -> None:
        axis = self.axes["mcd_spectra"]
        values = np.asarray(getattr(result, "pair_mcd_corrected", ()), dtype=float)
        self._artists["mcd_spectrum_lines"] = []
        if values.ndim == 2 and values.shape == (fields.size, energy.size) and fields.size:
            values = values[:, order]
            line = axis.plot(energy, values[self.state.selected_b_index], color="#2563eb", lw=1., label="Corrected MCD")[0]
            self._artists["mcd_spectrum_lines"].append((line, values))
        else:
            axis.text(.5, .5, "Corrected MCD unavailable", transform=axis.transAxes, ha="center")
        axis.set_title("Corrected MCD spectra", fontsize=10, pad=4)
        axis.margins(y=.20)
        axis.set_xlabel("Energy (eV)")
        axis.set_ylabel("MCD")
        axis.axhline(0, color="0.5", lw=.6, alpha=.5)
        self._add_energy_window(axis)
        status = axis.text(.98, .97, "", transform=axis.transAxes, ha="right", va="top", fontsize=7)
        self._artists.setdefault("b_status", []).append(status)

    def _draw_mcd_vs_b(self, result: Any, energy: np.ndarray, order: np.ndarray, fields: np.ndarray, mcd: np.ndarray) -> None:
        axis = self.axes["mcd_vs_b"]
        self._artists["mcd_trace_lines"] = []
        self._artists["mcd_fit_lines"] = {}
        self._artists["slope_readout"] = []
        self._artists["slope_difference_readout"] = []
        metric = str(self.state.window_metric).casefold().replace(" ", "_")
        if "integral" in metric:
            metric_key = "integral"
        elif "field" in metric and "absolute" in metric:
            metric_key = "field_signed_absolute_mean"
        elif "absolute" in metric:
            metric_key = "absolute_mean"
        else:
            metric_key = "mean"
        try:
            from core.mcd import pair_window_trace_by_branch
            traces = pair_window_trace_by_branch(
                result, self.state.window_center_ev, self.state.window_width_mev,
                metrics=(metric_key,), include_raw=False,
            )
        except (ImportError, AttributeError, TypeError, ValueError, FloatingPointError):
            traces = {}
        colors = {"B increasing": "#2563eb", "B decreasing": "#f59e0b"}
        plotted = False
        for branch, values_by_name in traces.items():
            points = values_by_name.get(f"corrected_{metric_key}")
            if points is None:
                continue
            branch_fields, branch_values = points
            if len(branch_fields):
                increasing = "increas" in str(branch).casefold()
                line = axis.plot(
                    branch_fields, branch_values, color=colors.get(branch, "#2563eb"), lw=.75,
                    ls="-" if increasing else "--", marker="o" if increasing else "s",
                    markersize=3.2, markeredgewidth=.65, alpha=.85, zorder=2,
                    markerfacecolor=colors.get(branch, "#2563eb") if increasing else "none",
                    markeredgecolor=colors.get(branch, "#2563eb"),
                    label=f"Corrected MCD · {'Inc' if increasing else 'Dec'}",
                )[0]
                self._artists["mcd_trace_lines"].append((branch, line))
                plotted = True
        if not plotted and fields.size and mcd.ndim == 2 and energy.size:
            mask = (energy >= self.state.window_center_ev - self.state.window_width_mev * 0.0005) & (energy <= self.state.window_center_ev + self.state.window_width_mev * 0.0005)
            values = np.nanmean(mcd[:, mask], axis=1) if np.any(mask) else np.full(fields.shape, np.nan)
            labels = np.asarray(getattr(result, "pair_labels", ()), dtype=str).ravel()
            if labels.size == fields.size and np.any(np.char.find(np.char.lower(labels), "increas") >= 0):
                branch_masks = (("B increasing", np.char.find(np.char.lower(labels), "increas") >= 0), ("B decreasing", np.char.find(np.char.lower(labels), "decreas") >= 0))
            else:
                branch_masks = (("B sweep", np.ones(fields.shape, dtype=bool)),)
            for branch, mask_branch in branch_masks:
                if np.any(mask_branch):
                    increasing = "increas" in branch.casefold()
                    color = colors.get(branch, "#2563eb") if increasing else colors.get(branch, "#f59e0b")
                    line = axis.plot(
                        fields[mask_branch], values[mask_branch], color=color, lw=.75,
                        ls="-" if increasing else "--", marker="o" if increasing else "s",
                        markersize=3.2, markeredgewidth=.65, alpha=.85, zorder=2,
                        markerfacecolor=color if increasing else "none",
                        markeredgecolor=color, label=f"Corrected MCD · {'Inc' if increasing else 'Dec' if branch != 'B sweep' else 'measured'}",
                    )[0]
                    self._artists["mcd_trace_lines"].append((branch, line))
        for branch, line in self._artists["mcd_trace_lines"]:
            visible = (self.show_inc_chk.isChecked() if "increas" in branch.casefold()
                       else self.show_dec_chk.isChecked() if "decreas" in branch.casefold() else True)
            line.set_visible(visible)
        self._update_mcd_trace_y_limits()
        axis.set_title("MCD vs B · corrected", fontsize=9)
        axis.set_xlabel("Paired mean field B (T)")
        axis.set_ylabel("MCD")
        finite_fields = np.asarray(fields, dtype=float)
        finite_fields = finite_fields[np.isfinite(finite_fields)]
        if finite_fields.size > 1:
            axis.set_xlim(float(np.min(finite_fields)), float(np.max(finite_fields)))
        visible_lines = [line for _, line in self._artists["mcd_trace_lines"] if line.get_visible()]
        if visible_lines:
            axis.legend(handles=visible_lines, fontsize=7, frameon=False)
        payload = self.analysis_payload if isinstance(self.analysis_payload, Mapping) else {}
        slopes = payload.get("slopes") if isinstance(payload, Mapping) else None
        if isinstance(slopes, Mapping):
            self._set_slope_details(slopes, axis)
        self._artists.setdefault("b_trace_cursor", []).append(axis.axvline(fields[self.state.selected_b_index] if fields.size else 0.0, color="#d97706", ls="--", lw=0.9))

    def _draw_feature_vs_b(self, fields: np.ndarray) -> None:
        axis = self.axes["feature_vs_b"]
        payload = self.analysis_payload or {}
        metric_key = self.state.feature_metric
        metric_label = self.display_metric_label
        records = payload.get("tracks", ()) if isinstance(payload, Mapping) else ()
        if metric_key == "splitting":
            records = payload.get("splitting", ()) if isinstance(payload, Mapping) else ()
        records = records if isinstance(records, (list, tuple)) else ()
        self._artists["feature_lines"] = []
        self._artists["feature_overlay_lines"] = []
        colors = {"pos": "#2563eb", "neg": "#f59e0b", "k": "#2563eb", "kp": "#f59e0b"}
        plotted: list[tuple[Mapping[str, Any], list[Mapping[str, Any]], Any]] = []
        value_key = {"shift": "delta_energy_ev", "energy": "energy_ev", "splitting": "splitting_ev"}[metric_key]
        scale = 1.0 if metric_key == "energy" else 1000.0
        for index, track in enumerate(records):
            if not isinstance(track, Mapping):
                continue
            points = [point for point in track.get("points", ()) if isinstance(point, Mapping)]
            points = [point for point in points if point.get("field_t") is not None and point.get(value_key) is not None]
            if not points:
                continue
            xs = np.asarray([point["field_t"] for point in points], dtype=float)
            ys = np.asarray([float(point[value_key]) * scale for point in points], dtype=float)
            branch = str(track.get("branch", "")).casefold()
            increasing = "increas" in branch or branch in {"inc", "increasing"}
            channel = str(track.get("channel", track.get("source_channel", track.get("selected_channel", "")))).casefold()
            channel = "neg" if any(token in channel for token in ("neg", "kp", "kprime", "minus")) else "pos" if any(token in channel for token in ("pos", "plus")) else channel
            color = colors.get(channel, "#2563eb" if increasing else "#f59e0b")
            mapping = payload.get("mapping", {}) if isinstance(payload, Mapping) else {}
            mapped_k = str(mapping.get("k_channel", "")).casefold() if isinstance(mapping, Mapping) else ""
            if metric_key == "splitting":
                channel_label = "E_K−E_K′"
            elif mapped_k in {"pos", "neg"} and channel in {"pos", "neg"}:
                channel_label = "K" if channel == mapped_k else "K′"
            else:
                channel_label = "+ channel" if channel == "pos" else "− channel" if channel == "neg" else "ΔK−K′" if metric_key == "splitting" else "measured"
            branch_label = "Inc" if increasing else "Dec" if branch else "measured"
            identity = str(track.get("label") or track.get("id") or track.get("feature_id") or track.get("peak_id") or track.get("pair_id") or f"track {index + 1}")
            label = f"{channel_label} · {branch_label}" if identity.casefold() in {"feature", "track"} else f"{channel_label} · {branch_label} · {identity}"
            line = axis.plot(
                xs, ys, color=color, lw=1.1, ls="-" if increasing else "--",
                marker="o" if increasing else "s",
                markerfacecolor=color if increasing else "none", markeredgecolor=color,
                label=label,
            )[0]
            self._artists["feature_lines"].append(line)
            plotted.append((track, points, line))

        if self.state.feature_overlay:
            secondary_key = "energy_ev" if metric_key == "shift" else "energy_k_ev" if metric_key == "splitting" else "delta_energy_ev"
            secondary_scale = 1.0 if secondary_key in {"energy_ev", "energy_k_ev"} else 1000.0
            overlay_data = []
            for _track, points, source_line in plotted:
                if metric_key == "splitting":
                    valid = [point for point in points if (point.get("energy_k_ev") if point.get("energy_k_ev") is not None else point.get("energy_ev")) is not None]
                else:
                    valid = [point for point in points if point.get(secondary_key) is not None]
                if valid:
                    overlay_data.append((
                        np.asarray([point["field_t"] for point in valid], dtype=float),
                        np.asarray([float(point.get(secondary_key)) if point.get(secondary_key) is not None else float(point.get("energy_ev")) for point in valid], dtype=float) * secondary_scale,
                        source_line,
                    ))
            if overlay_data:
                secondary_label = "Energy (eV)" if secondary_key in {"energy_ev", "energy_k_ev"} else "Shift (meV)"
                self._feature_energy_axis = axis.twinx()
                self._feature_energy_axis.set_ylabel(secondary_label)
                self._feature_energy_axis.grid(False)
                for xs, ys, source_line in overlay_data:
                    overlay = self._feature_energy_axis.plot(xs, ys, color=source_line.get_color(), lw=.8, ls=":", alpha=.85)[0]
                    self._artists["feature_overlay_lines"].append(overlay)
        if metric_key == "splitting" and not self._artists["feature_lines"]:
            reason = str(payload.get("splitting_reason") or payload.get("splitting_status") or "Splitting unavailable") if isinstance(payload, Mapping) else "Splitting unavailable"
            axis.text(.03, .94, reason, transform=axis.transAxes, va="top", fontsize=7, color="#b45309", wrap=True)
        status_text = str(payload.get("status", "")).strip() if isinstance(payload, Mapping) else ""
        reason_text = str(payload.get("reason", "")).strip() if isinstance(payload, Mapping) else ""
        axis.set_title(f"Feature {metric_label} vs B", fontsize=9)
        axis.set_xlabel("Paired mean field B (T)")
        axis.set_ylabel(metric_label)
        if isinstance(payload, Mapping) and payload.get("status"):
            self.candidate_status.setText(f"{status_text} {reason_text}".strip())
        if reason_text or (not self._artists["feature_lines"] and status_text):
            detail = reason_text or status_text
            wrapped = "\n".join(detail[i:i + 42] for i in range(0, len(detail), 42))
            axis.text(.02, .95, f"{status_text}\n{wrapped}".strip(), transform=axis.transAxes, va="top", fontsize=7, color="#b45309", wrap=True)
        if self._artists["feature_lines"]:
            axis.legend(fontsize=7, frameon=False)
        self._feature_axis_metric = metric_key

    def _sync_candidate_overlay(self) -> None:
        """Synchronize candidate/window artists without changing limits or drawing."""
        if self._overlay_syncing:
            return
        self._overlay_syncing = True
        try:
            self._sync_candidate_overlay_impl()
        finally:
            self._overlay_syncing = False

    def _draw_feature_overlay(self) -> None:
        self._sync_candidate_overlay()
        if not self._rendering and not self._blit_update():
            self.canvas.draw_idle()

    def _sync_candidate_overlay_impl(self) -> None:
        # Candidate badges are static between selections. Bake them into the
        # background once so a long saved history does not slow every drag.
        self._blit_backgrounds = {}
        for artist in self._artists.get("feature_selection", ()):
            try:
                artist.remove()
            except (AttributeError, NotImplementedError, ValueError):
                pass
        self._artists["feature_selection"] = []
        selected = self.selected_candidate
        selected_id = str(selected.get("id", "")) if selected else ""
        for key in ("mcd_map", "spectra", "mcd_spectra"):
            axis = self.axes.get(key)
            if axis is None:
                continue
            low, high = axis.get_xlim()
            label_row_ends = [-float("inf")] * 3
            for candidate in self.visible_candidates:
                if key == "mcd_spectra" and candidate.get("domain") != "mcd":
                    continue
                center = self._candidate_center(candidate)
                if center is None or not (low <= center <= high):
                    continue
                identifier = str(candidate.get("id", candidate.get("label", "feature")))
                if key == "mcd_spectra":
                    pixel_x = axis.get_xaxis_transform().transform((center, 0))[0]
                    row = next((i for i, right in enumerate(label_row_ends) if pixel_x - right >= 42), 2)
                    label_row_ends[row] = pixel_x
                    historical = bool(candidate.get("history"))
                    color = "#7c3aed" if historical else "#64748b"
                    line = axis.axvline(center, color=color, ls="--" if historical else ":", lw=.8, alpha=.45)
                    label = axis.annotate(str(candidate.get("display_id", identifier)),
                        xy=(center, 0), xycoords=axis.get_xaxis_transform(),
                        xytext=(0, 4 + row * 14), textcoords="offset points", ha="center", va="bottom", fontsize=7,
                        color=color, bbox={"fc": "white", "ec": color, "alpha": .9, "boxstyle": "round,pad=.15"},
                        clip_on=True, zorder=10)
                    artists = (line, label)
                elif key == "mcd_map":
                    # Number every candidate; stagger nearby badges in pixel
                    # space so narrow Combo plots retain readable identifiers.
                    active = identifier == selected_id
                    pixel_x = axis.get_xaxis_transform().transform((center, 1))[0]
                    row = next((i for i, right in enumerate(label_row_ends)
                                if pixel_x - right >= 22), 2)
                    label_row_ends[row] = pixel_x
                    label = axis.annotate(str(candidate.get("display_id", identifier)),
                        xy=(center, 1), xycoords=axis.get_xaxis_transform(),
                        xytext=(0, -5 - row * 18), textcoords="offset points",
                        ha="center", va="top", fontsize=8, fontweight="bold",
                        color="white" if active else "#7c3aed" if candidate.get("history") else "#333333",
                        bbox={"boxstyle": "circle,pad=.2",
                              "fc": "#0f766e" if active else "white",
                              "ec": "#0f766e" if active else "#777777",
                              "lw": .7, "alpha": .95},
                        clip_on=True, zorder=10)
                    artists = (label,)
                elif identifier == selected_id:
                    outline = axis.axvline(center, color="white", lw=3.2, alpha=.95, zorder=8)
                    line = axis.axvline(center, color="#27313b", lw=1.5, ls="-", alpha=1.0, zorder=9)
                    triangle = axis.plot([center], [.96], transform=axis.get_xaxis_transform(), marker="^", ms=6,
                                         color="#27313b", markeredgecolor="white", markeredgewidth=.8,
                                         clip_on=True, zorder=10)[0]
                    kind = str(candidate.get("kind", "feature")).casefold()
                    locator_label = f"Window {center:.4f}" if kind == "window" else f"Ref {center:.4f}"
                    label = axis.text(center, .985, f"{candidate.get('display_id', identifier)}\n{locator_label}", transform=axis.get_xaxis_transform(),
                                      ha="center", va="top", fontsize=6, color="#27313b",
                                      bbox={"boxstyle": "round,pad=.12", "fc": "white", "ec": "#27313b", "lw": .5, "alpha": .9},
                                      clip_on=True, zorder=10)
                    artists = (outline, line, triangle, label)
                else:
                    line = axis.plot([center, center], [.96, 1.0], transform=axis.get_xaxis_transform(),
                                     color="#64748b", ls=(0, (2, 2)), lw=.8, alpha=.75,
                                     clip_on=True, zorder=7)[0]
                    # Keep the full inventory selectable without stacking labels.
                    artists = (line,)
                for artist in artists:
                    artist._mcd_candidate_id = identifier
                    self._artists["feature_selection"].append(artist)
            # The analysis window has a distinct color and label so it cannot
            # be mistaken for a candidate center.
            window_center = float(self.state.window_center_ev)
            if np.isfinite(window_center) and low <= window_center <= high and key != "mcd_spectra":
                window_line = axis.axvline(window_center, ymin=.985 if key == "mcd_map" else 0,
                    ymax=1, color="#0f766e", ls="--", lw=.9, alpha=.85, zorder=6)
                window_line._mcd_window_label = "Window"
                self._artists["feature_selection"].append(window_line)
                if key == "mcd_map":
                    continue
                window_text = axis.text(window_center, .80, f"Window {window_center:.4f} eV", transform=axis.get_xaxis_transform(),
                                        ha="center", va="top", fontsize=6, color="#0f766e", clip_on=True, zorder=9)
                window_text._mcd_window_label = "Window"
                self._artists["feature_selection"].append(window_text)
        if selected is not None:
            center = self._candidate_center(selected)
            status = self.analysis_payload.get("status") if isinstance(self.analysis_payload, Mapping) else None
            if str(selected.get("kind", "")).casefold() == "window":
                width = float(selected.get("width_mev", self.state.window_width_mev))
                self.candidate_status.setText(f"Window {float(center):.6f} eV · fixed {width:.4g} meV")
            else:
                self.candidate_status.setText(
                    str(status) if status else f"Selected {selected.get('id', 'feature')} · {float(center):.5f} eV"
                )
        # Selection changes only artists; numerical tracking remains owned by
        # the cancellable analysis worker.

    def _dynamic_artists_by_axis(self) -> dict[str, list[Any]]:
        artists = []
        for name in ("b_map_cursor", "window", "window_edges", "window_cursor", "window_status",
                     "b_status", "b_trace_cursor", "slope_readout", "slope_difference_readout"):
            artists.extend(self._artists.get(name, ()))
        artists.extend(item for item in self._artists.get("feature_selection", ())
                       if getattr(item, "_mcd_window_label", "") == "Window")
        artists.extend(item[0] for item in self._artists.get("spectrum_lines", ()))
        artists.extend(item[0] for item in self._artists.get("mcd_spectrum_lines", ()))
        artists.extend(line for _branch, line in self._artists.get("mcd_trace_lines", ()))
        for name in ("mcd_fit_extensions", "mcd_fit_lines", "slope_annotations"):
            artists.extend(self._artists.get(name, {}).values())
        if self._artists.get("spectra_legend") is not None:
            artists.append(self._artists["spectra_legend"])
        return {key: [item for item in artists if item.axes is axis] for key, axis in self.axes.items()}

    def _prepare_blit(self) -> None:
        """Cache static panel backgrounds for B/window cursor updates."""
        self._blit_backgrounds = {}
        self._blit_preparing = True
        dynamic = self._dynamic_artists_by_axis()
        for artists in dynamic.values():
            for artist in artists:
                try:
                    artist.set_animated(True)
                except AttributeError:
                    pass
        try:
            self.canvas.draw()
            for key, axis in self.axes.items():
                if axis is not None:
                    self._blit_backgrounds[key] = self.canvas.copy_from_bbox(axis.bbox)
            for key, artists in dynamic.items():
                axis = self.axes.get(key)
                if axis is None:
                    continue
                for artist in artists:
                    axis.draw_artist(artist)
                self.canvas.blit(axis.bbox)
        except (AttributeError, RuntimeError, ValueError):
            self._blit_backgrounds = {}
        finally:
            self._blit_preparing = False

    def _blit_update(self, *, axes_keys: Iterable[str] | None = None) -> bool:
        if not self._blit_backgrounds:
            return False
        try:
            dynamic = self._dynamic_artists_by_axis()
            for key, background in self._blit_backgrounds.items():
                if axes_keys is not None and key not in axes_keys:
                    continue
                axis = self.axes.get(key)
                if axis is None:
                    continue
                self.canvas.restore_region(background)
                for artist in dynamic.get(key, ()):
                    axis.draw_artist(artist)
                self.canvas.blit(axis.bbox)
            return True
        except (AttributeError, RuntimeError, ValueError):
            self._blit_backgrounds = {}
            return False

    def _on_canvas_draw(self, _event: Any) -> None:
        """Invalidate cached backgrounds after a full draw (resize/save)."""
        if not self._owns_current_figure():
            self._blit_backgrounds = {}
            self._blit_rebuild_pending = False
            return
        if self._blit_preparing:
            return
        if (abs(float(self.figure.bbox.height) - getattr(self, "_layout_height", 0.0)) > .5
                or abs(float(self.figure.bbox.width) - getattr(self, "_layout_width", 0.0)) > .5):
            self._update_subplot_spacing()
        dynamic = self._dynamic_artists_by_axis()
        all_animated = bool(dynamic) and all(
            getattr(artist, "get_animated", lambda: False)()
            for artists in dynamic.values() for artist in artists
        )
        backgrounds = {}
        if all_animated:
            try:
                backgrounds = {
                    key: self.canvas.copy_from_bbox(axis.bbox)
                    for key, axis in self.axes.items() if axis is not None
                }
            except (AttributeError, RuntimeError, ValueError):
                backgrounds = {}
        self._blit_backgrounds = backgrounds
        for artists in dynamic.values():
            for artist in artists:
                try:
                    # Animated artists were skipped by this full draw. Paint
                    # them into its renderer now, before Qt presents it.
                    if artist.get_animated() and artist.axes is not None and artist.get_visible():
                        artist.axes.draw_artist(artist)
                    artist.set_animated(False)
                except AttributeError:
                    pass
        if not self._blit_backgrounds and self.axes and not self._blit_rebuild_pending:
            self._blit_rebuild_pending = True
            QTimer.singleShot(0, self._rebuild_blit_after_full_draw)

    def _update_subplot_spacing(self, _event: Any = None) -> None:
        grid = getattr(self, "_plot_grid", None)
        if grid is None:
            return
        height = max(float(self.figure.bbox.height), 240.0)
        self._layout_height = height
        width = max(float(self.figure.bbox.width), 320.0)
        self._layout_width = width
        left_margin, right_margin, column_gap = 82.0, 18.0, 110.0
        panel_width = max((width - left_margin - right_margin - column_gap) / 2, 60.0)
        top_margin, bottom_margin, gap = 48.0, 48.0, 62.0
        panel_height = max((height - top_margin - bottom_margin - gap) / 2, 40.0)
        grid.update(left=left_margin/width, right=1-right_margin/width, top=1-top_margin/height,
                    bottom=bottom_margin/height, hspace=gap/panel_height, wspace=column_gap/panel_width)
        # This application owns Figure directly (no pyplot figure manager).
        # GridSpec.update alone therefore does not reposition existing axes.
        for axis in self.figure.axes:
            spec = axis.get_subplotspec()
            if spec is not None:
                axis.set_position(spec.get_position(self.figure))
        self._blit_backgrounds = {}

    def _rebuild_blit_after_full_draw(self) -> None:
        self._blit_rebuild_pending = False
        if self._owns_current_figure() and self._result is not None:
            self._prepare_blit()

    def _owns_current_figure(self) -> bool:
        """Return true only while every unified axis is live in Figure."""
        return bool(self.axes and all(axis in self.figure.axes for axis in self.axes.values()))

    def prepare_full_redraw(self) -> None:
        """Include interactive artists in the next toolbar/publication draw."""
        self._blit_backgrounds = {}
        self._blit_rebuild_pending = False
        for artists in self._dynamic_artists_by_axis().values():
            for artist in artists:
                try:
                    artist.set_animated(False)
                except AttributeError:
                    pass

    def restore_interactive_drawing(self) -> None:
        """Rebuild cached static backgrounds after a full publication draw."""
        self._prepare_blit()


class McdUnifiedControls(QWidget):
    """Compact feature/slope controls appended to the existing MCD sidebar."""

    def __init__(self, parent: QWidget | None = None, *, expander_factory=None) -> None:
        super().__init__(parent)
        from PySide6.QtWidgets import QFormLayout, QGroupBox, QLineEdit

        self.setObjectName("mcdUnifiedControls")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        feature = QGroupBox("Feature analysis", self)
        form = QFormLayout(feature)
        self.feature_method_combo = QComboBox(feature)
        self.feature_method_combo.addItems(["Raw spectrum", "Second derivative", "Local fit (advanced)"])
        self.feature_source_combo = QComboBox(feature)
        self.feature_source_combo.addItems(["Raw", "Corrected"])
        self.feature_search_low_spin = QDoubleSpinBox(feature)
        self.feature_search_high_spin = QDoubleSpinBox(feature)
        for spin in (self.feature_search_low_spin, self.feature_search_high_spin):
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(6)
        form.addRow("Method", self.feature_method_combo)
        form.addRow("Source", self.feature_source_combo)
        form.addRow("Search low (eV)", self.feature_search_low_spin)
        form.addRow("Search high (eV)", self.feature_search_high_spin)
        advanced = QGroupBox("Advanced reference / detection", feature)
        advanced_form = QFormLayout(advanced)
        self.reference_mode_combo = QComboBox(advanced)
        self.reference_mode_combo.addItems(["E(0) per channel / branch", "Manual reference"])
        self.manual_reference_spin = QDoubleSpinBox(advanced)
        self.manual_reference_spin.setRange(-1e6, 1e6)
        self.manual_reference_spin.setDecimals(6)
        self.manual_reference_spin.setEnabled(False)
        self.reference_mode_combo.currentIndexChanged.connect(
            lambda index: self.manual_reference_spin.setEnabled(index == 1)
        )
        self.prominence_spin = QDoubleSpinBox(advanced)
        self.prominence_spin.setRange(0.0, 1.0)
        self.prominence_spin.setDecimals(3)
        self.prominence_spin.setValue(0.03)
        advanced_form.addRow("Shift reference", self.reference_mode_combo)
        advanced_form.addRow("Manual E(0) (eV)", self.manual_reference_spin)
        self.manual_link_btn = QPushButton("Connections…", advanced)
        self.manual_link_btn.setToolTip("Manually connect an MCD candidate to one or more spectrum candidates")
        advanced_form.addRow("MCD ↔ spectrum", self.manual_link_btn)
        advanced_form.addRow("Prominence", self.prominence_spin)
        form.addRow(advanced)
        feature.setObjectName("mcdFeatureAnalysisContent")
        owner = self.window()
        factory = expander_factory or getattr(owner, "_make_expander", None)
        if factory is not None:
            self.feature_expander = factory("Feature analysis", feature, expanded=False)
        else:
            self.feature_expander = QWidget(self)
            fold_layout = QVBoxLayout(self.feature_expander)
            fold_layout.setContentsMargins(0, 0, 0, 0)
            head = QToolButton(self.feature_expander)
            head.setText("Feature analysis")
            head.setCheckable(True)
            head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            from PySide6.QtWidgets import QStyle
            def toggle_feature(on):
                feature.setVisible(on)
                head.setIcon(head.style().standardIcon(QStyle.SP_ArrowDown if on else QStyle.SP_ArrowRight))
            head.toggled.connect(toggle_feature)
            fold_layout.addWidget(head)
            fold_layout.addWidget(feature)
            toggle_feature(False)
        feature.setTitle("")
        layout.addWidget(self.feature_expander)

        slopes = QGroupBox("MCD slope ranges", self)
        slope_form = QFormLayout(slopes)
        self.slope_low_spin = QDoubleSpinBox(slopes)
        self.slope_low_end_spin = QDoubleSpinBox(slopes)
        self.slope_high_positive_spin = QDoubleSpinBox(slopes)
        self.slope_high_positive_end_spin = QDoubleSpinBox(slopes)
        self.slope_high_negative_spin = QDoubleSpinBox(slopes)
        self.slope_high_negative_end_spin = QDoubleSpinBox(slopes)
        # Compatibility alias for callers that only exposed one high-field
        # range before the unified page.
        self.slope_high_spin = self.slope_high_positive_spin
        for spin, value in ((self.slope_low_spin, -0.2), (self.slope_low_end_spin, 0.2), (self.slope_high_positive_spin, 1.5), (self.slope_high_positive_end_spin, 2.0), (self.slope_high_negative_spin, -2.0), (self.slope_high_negative_end_spin, -1.5)):
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(5)
            spin.setValue(value)
            spin.setSuffix(" T")
        low_row = QWidget(slopes)
        low_layout = QHBoxLayout(low_row)
        low_layout.setContentsMargins(0, 0, 0, 0)
        low_layout.addWidget(self.slope_low_spin)
        low_layout.addWidget(QLabel("to"))
        low_layout.addWidget(self.slope_low_end_spin)
        slope_form.addRow("Low-field range", low_row)
        pos_row = QWidget(slopes)
        pos_layout = QHBoxLayout(pos_row)
        pos_layout.setContentsMargins(0, 0, 0, 0)
        pos_layout.addWidget(self.slope_high_positive_spin)
        pos_layout.addWidget(QLabel("to"))
        pos_layout.addWidget(self.slope_high_positive_end_spin)
        neg_row = QWidget(slopes)
        neg_layout = QHBoxLayout(neg_row)
        neg_layout.setContentsMargins(0, 0, 0, 0)
        neg_layout.addWidget(self.slope_high_negative_spin)
        neg_layout.addWidget(QLabel("to"))
        neg_layout.addWidget(self.slope_high_negative_end_spin)
        self.slope_high_positive_spin.setSuffix("")
        self.slope_high_negative_spin.setSuffix("")
        slope_form.addRow("High + range", pos_row)
        slope_form.addRow("High − range", neg_row)
        self.slope_details_btn = QToolButton(slopes)
        self.slope_details_btn.setText("Details…")
        self.slope_details_btn.setToolTip("Full slope fits and high-minus-low differences appear after analysis.")
        slope_form.addRow("Fit readout", self.slope_details_btn)
        layout.addWidget(slopes)

        retained = QGroupBox("Retained results", self)
        retained_layout = QVBoxLayout(retained)
        self.retained_window_list = QListWidget(retained)
        self.retained_window_list.setMinimumHeight(48)
        self.retain_window_btn = QPushButton("Retain current window", retained)
        self.update_retained_btn = QPushButton("Update selected", retained)
        self.save_results_btn = QPushButton("Export retained results", retained)
        self.mcd_export_retained_btn = self.save_results_btn
        self.save_results_btn.setToolTip("Export only checked retained snapshots. Use the top Save button for the current center.")
        retained_layout.addWidget(self.retained_window_list)
        row = QHBoxLayout()
        row.addWidget(self.retain_window_btn)
        row.addWidget(self.update_retained_btn)
        retained_layout.addLayout(row)
        retained_layout.addWidget(self.save_results_btn)
        self.save_folder_label = QLabel("Save to: dataset folder / Processed Data / MCD (automatic)", retained)
        self.save_folder_label.setWordWrap(True)
        self.save_folder_label.setMinimumWidth(0)
        self.change_save_folder_btn = QPushButton("Change…", retained)
        retained_layout.addWidget(self.save_folder_label)
        retained_layout.addWidget(self.change_save_folder_btn)
        retained_layout.addWidget(QLabel("Retained feature tracks"))
        self.retained_feature_list = QListWidget(retained)
        self.retained_feature_list.setMinimumHeight(36)
        self.include_feature_chk = QPushButton("Include selected feature", retained)
        self.include_feature_chk.setCheckable(True)
        self.include_feature_chk.setChecked(True)
        self.include_feature_chk.setToolTip("Include the completed selected feature when saving the current center with the top Save button.")
        self.retain_feature_btn = QPushButton("Retain selected feature", retained)
        self.update_feature_btn = QPushButton("Update selected feature", retained)
        retained_layout.addWidget(self.retained_feature_list)
        feature_row = QHBoxLayout()
        feature_row.addWidget(self.include_feature_chk)
        feature_row.addWidget(self.retain_feature_btn)
        feature_row.addWidget(self.update_feature_btn)
        retained_layout.addLayout(feature_row)
        layout.addWidget(retained)


__all__ = ["McdUnifiedControls", "McdUnifiedView", "RetainedMcdWindow", "UnifiedMcdState"]
