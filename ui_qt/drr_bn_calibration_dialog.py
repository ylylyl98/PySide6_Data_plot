"""Compare a selected BN reference for one measurement group without blocking Qt.

The owning workflow supplies all I/O, calibration and persistence. This dialog
only requests application of a payload explicitly marked eligible by that
workflow. Viewing a saved result never applies it implicitly.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re
from threading import Event

import numpy as np
from matplotlib.figure import Figure
from core.drr_bn_store import joint_validation_passed, optimization_policy_current
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSplitter, QTabWidget, QTableView, QVBoxLayout, QWidget,
)

from ui_qt.common import QDoubleSpinBox
from ui_qt.fluent_ui.style import set_fluent_property
from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg


_DEFAULTS = {
    "train_region": [1.900, 1.935], "train_regions": [], "validation_region": [1.945, 1.970],
    "core_region": [1.900, 1.970], "taper_width": 0.010,
    "protected_regions": [[1.58, 1.86]], "trusted_regions": [],
    "gate_range": None, "gate_count": 12,
    "uncertain_regions": [[1.94, 2.02]], "auto_apply": False,
    "derivative_window": 21, "derivative_polyorder": 2,
}

_METHOD_NAMES = {
    "original": "Original BN",
    "local_shift": "Local fringe shift",
    "local_shift_gain": "Local shift + amplitude",
}
for _family in ("local_shift", "local_shift_gain"):
    _base_name = _METHOD_NAMES[_family]
    _METHOD_NAMES[_family] = f"{_base_name} · 100% correction"
    for _percent in (25, 50, 75):
        _METHOD_NAMES[f"{_family}_s{_percent}"] = f"{_base_name} · {_percent}% correction"
_STATUS_NAMES = {
    "accepted": "Validation passed", "fallback": "Original BN retained",
    "original": "Original BN retained", "exploratory": "Exploratory comparison",
    "invalid": "Cannot compare",
}
_REASON_TEXT = {
    "correction_support_not_explicitly_trusted": "The correction extends outside the trusted calibration range",
    "correction_overlaps_uncertain_region": "The correction overlaps an uncertain spectral feature",
    "insufficient_independent_repeats": "Too few independent repeats for final confirmation",
    "no_reliable_selection_benefit": "No candidate confirmed the D2E selection target",
    "selected_parameter_at_bound": "A fitted parameter reached its allowed limit",
    "final_holdout_benefit_not_confirmed": "The separate final holdout did not satisfy the DRR constraint",
    "temporal_parameters_unstable": "Calibration parameters vary too much between acquisition times",
    "temporal_shift_sign_reversal": "The fitted fringe shift changes direction between repeats",
    "background_repeat_mean_mismatch": "BN repeat averages do not match the selected background",
    "invalid_background_repeats": "The BN repeats cannot support a stability check",
    "background_sensitivity_unverified": "Independent BN stability has not been verified",
    "background_parameters_sensitive": "The fitted parameters depend too strongly on the BN frames used",
    "background_half_benefit_not_confirmed": "Separate BN halves did not satisfy the DRR constraint",
    "invalid_input_shape": "Measurement and background array shapes do not match",
    "nonfinite_input": "The input contains missing or nonfinite values",
    "nonpositive_intensity": "The input contains zero or negative intensities",
    "invalid_grid": "The energy or gate grid is invalid",
    "insufficient_template_bandwidth": "Too few spectral channels for the BN template",
    "correction_support_outside_grid": "The correction range extends beyond the measured spectrum",
    "invalid_repeat_order": "The acquisition order of the repeats is invalid",
    "no_gates_in_range": "No measured gate values lie in the requested range",
    "insufficient_unprotected_bandwidth": "Too few unprotected channels remain for fitting or checking",
    "nonfinite_candidate": "The candidate contains nonfinite values",
    "numerical_failure": "The numerical calculation could not be completed",
    "invalid_input_or_settings": "The input or calibration settings are invalid",
    "invalid_settings_region": "An energy or parameter interval is invalid",
    "invalid_settings_threshold": "A calibration threshold is invalid",
    "invalid_settings_integer": "A count or window setting is invalid",
    "invalid_settings_template": "The BN template window or polynomial order is invalid",
    "invalid_settings_derivative": "The SG window or polynomial order is invalid",
    "invalid_settings_improvement": "The required improvement threshold is invalid",
    "invalid_settings_gain_bounds": "The allowed amplitude range is invalid",
    "training_and_validation_overlap": "Fit and check energy ranges overlap",
    "scoring_band_outside_core": "Fit or check energy lies outside the correction range",
    "no_reliable_joint_selection_benefit": "No candidate combines D2E improvement with the DRR and spectral protection checks",
    "d2_selection_benefit_not_confirmed": "D2E improvement was not confirmed on selection repeats",
    "d2_final_holdout_benefit_not_confirmed": "D2E improvement was not confirmed on the final holdout",
    "d2_protected_region_changed": "D2E changed within a protected interval after SG processing",
    "d2_benefit_not_confirmed": "D2E improvement did not meet the validation threshold",
    "final_holdout_d2_benefit_not_confirmed": "The separate final holdout did not confirm D2E improvement",
    "final_holdout_joint_not_confirmed": "The separate final holdout did not satisfy the D2E target and DRR constraints",
    "training_full_axis_fidelity_not_confirmed": "The fitted repeats did not preserve full-spectrum DRR and D2E",
    "background_half_joint_not_confirmed": "Separate BN halves did not satisfy the D2E target and DRR constraints",
    "joint_validation_not_confirmed": "D2E improvement, DRR constraints, or spectral protection checks were not confirmed",
    "drr_gate_degradation": "DRR exceeded its allowed change at one or more measured gate values",
    "d2_gate_degradation": "D2E exceeded its allowed change at one or more measured gate values",
    "full_axis_fidelity_not_confirmed": "Protected or local spectral features were not preserved across the full spectrum",
    "drr_noninferiority_not_confirmed": "DRR changed beyond its allowed tolerance",
    "final_holdout_drr_noninferiority_not_confirmed": "DRR exceeded its allowed change on the separate final holdout",
    "drr_noninferiority_failed": "DRR changed beyond its allowed tolerance",
    "d2_training_not_verified": "The BN training fit did not verify the D2E objective",
    "no_resolvable_bn_texture": "The BN reference does not contain enough resolvable fringe texture for this fit",
    "no_resolvable_bn_d2_basis": "The BN fringe responses cannot identify all fitted parameters at the current SG settings",
    "insufficient_d2_training_bandwidth": "The fit energy range is too narrow for the current SG window; original BN retained",
    "influence_response_unidentifiable": "The BN response cannot reliably separate the fitted parameters",
    "influence_active_training_constraint": "The fit is too close to a DRR constraint for a reliable sensitivity check",
    "influence_nonsmooth_parameter_response": "The correction reaches a spectrum edge where weak-feature sensitivity cannot be verified",
    "training_feature_influence_screen_failed": "The weak-feature sensitivity screen exceeded its allowed change",
    "training_feature_refinement_screen_failed": "The finer weak-feature check exceeded its allowed change; original BN retained",
    "training_feature_influence_refit_failed": "Refitting weak test features did not preserve their shapes within tolerance",
    "training_feature_influence_not_confirmed": "Weak-feature preservation was not confirmed; original BN retained",
    "pre_final_selection_sensitivity_failed": "Candidate selection changes too much when weak test features are added; original BN retained",
}
_CODE_TEXT = {**_REASON_TEXT, **{key: value for key, value in _METHOD_NAMES.items() if key != "original"}}
_CODE_PATTERN = re.compile(r"\b(" + "|".join(re.escape(key) for key in sorted(_CODE_TEXT, key=len, reverse=True)) + r")\b")


def _readable_text(value):
    text = str(value)
    text = _CODE_PATTERN.sub(lambda match: _CODE_TEXT[match.group(0)], text)
    # Future engine rejection codes remain readable in status/summary views;
    # the unchanged machine keys remain available in Validation details.
    def readable_code(match):
        words = match.group(0).split("_")
        aliases = {"bn": "BN", "d2": "D2E", "drr": "DRR", "sg": "SG"}
        result = " ".join(aliases.get(word, word) for word in words)
        return result[0].upper() + result[1:]
    text = re.sub(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", readable_code, text)
    text = re.sub(r"(?<=: )original(?=[.;]|$)", "Original BN", text)
    for status, label in _STATUS_NAMES.items():
        if text.startswith(status + ":"):
            return label + text[len(status):]
    return text


def _final_confirmation_text(payload, report, *, derivative=False):
    confirmation = report.get("final_confirmation_d2" if derivative else "final_confirmation")
    product = "D2E" if derivative else "DRR"
    unavailable = f"{product}: No final holdout result available. Selection scores are not independent final confirmation."
    if not isinstance(confirmation, dict):
        return unavailable
    try:
        old, new, gain = (float(confirmation[key]) for key in
                          ("mean_original_rms", "mean_candidate_rms", "mean_improvement_fraction"))
        repeats = confirmation.get("repeat_indices", [])
        if (not isinstance(repeats, (list, tuple)) or not repeats
                or not np.isfinite([old, new, gain]).all() or old < 0 or new < 0):
            return unavailable
    except (KeyError, TypeError, ValueError):
        return unavailable
    method = payload.get("selected_candidate", report.get("selected_method", "Selected candidate"))
    name = _METHOD_NAMES.get(method, _readable_text(method))
    rms = f"{old:.4g} → {new:.4g} eV⁻²" if derivative else f"{100 * old:.4f}% → {100 * new:.4f}%"
    metric = "production residual RMS" if derivative else "residual RMS (diagnostic)"
    text = (f"{product} final holdout · {name}: {metric} {rms}; "
            f"gain {gain:+.1%} ({len(repeats)} repeat(s)).")
    checks = report.get('joint_validation', {}).get('final') or {}
    flag = checks.get('d2_benefit_passed' if derivative else 'drr_noninferiority_passed')
    independent = (len(repeats) >= 2
                   and report.get('split', {}).get('independent_final_confirmation_available') is not False)
    if not independent:
        flag = None
    if derivative:
        text += (" D2E target confirmed." if flag is True else
                 " D2E target not confirmed." if flag is False else " D2E target unverified.")
    else:
        text += (" DRR constraint passed." if flag is True else
                 " DRR constraint failed." if flag is False else " DRR constraint unverified.")
    if not independent:
        text += " Too few independent repeats for confirmation."
    else:
        text += " Application still requires all checks."
    return text


def _gain_text(value):
    try:
        numeric = float(value)
        return f"{numeric:+.1%}" if np.isfinite(numeric) else "—"
    except (TypeError, ValueError):
        return "—"


def _report_derivative_settings(payload):
    report = payload.get("report", {}).get("derivative_settings", {})
    settings = payload.get("settings", {})
    return (report.get("window_length", settings.get("derivative_window")),
            report.get("polyorder", settings.get("derivative_polyorder")))


def _range_text(ranges):
    return "; ".join(f"{float(lo):g}, {float(hi):g}" for lo, hi in ranges)


def _read_ranges(text, name):
    ranges = []
    for entry in text.split(";"):
        if not entry.strip():
            continue
        values = entry.replace("–", ",").replace("−", "-").split(",")
        if len(values) != 2:
            # Also accept the familiar positive-energy form 1.81-1.86.
            values = re.split(r"(?<=\d)\s*-\s*(?=\d)", entry.strip())
        try:
            lo, hi = (float(value) for value in values)
        except (ValueError, TypeError):
            raise ValueError(f"{name}: enter energy pairs such as 1.81, 1.86; 1.94, 2.02.") from None
        if not np.isfinite([lo, hi]).all() or lo <= 0 or hi <= lo:
            raise ValueError(f"{name}: each range needs finite positive energies with low < high.")
        ranges.append([lo, hi])
    return ranges


class _ComparisonSignals(QObject):
    progress = Signal(object)
    log = Signal(str)
    finished = Signal()


class _ComparisonWorker(QRunnable):

    def __init__(self, callback, settings):
        super().__init__()
        self.signals = _ComparisonSignals()
        self.callback = callback
        self.settings = deepcopy(settings)
        self.cancellation = Event()
        self.payload = None
        self.error = ""

    def run(self):
        try:
            self.payload = self.callback(
                self.settings, self.signals.progress.emit, self.signals.log.emit, self.cancellation)
            if not isinstance(self.payload, dict) and not self.cancellation.is_set():
                raise ValueError("The comparison did not return a result report.")
        except Exception as exc:
            if not self.cancellation.is_set():
                self.error = str(exc) or type(exc).__name__
        finally:
            self.signals.finished.emit()


class DrrBnCalibrationDialog(QDialog):
    """A per-group comparison view with cooperative cancellation.

    ``run_comparison(settings, progress, log, cancellation_event)`` runs in a
    background thread pool and returns a dictionary. It must check the event during
    I/O and calculation. The payload contains ``eligible``, ``summary``,
    ``report`` and, for previews, ``energy``, ``original_background``,
    ``candidate_background`` and optionally ``mean_raw``. Candidate table rows
    have ``name``, ``improvement_fraction``, ``eligible`` and ``reason``.

    Signals are requests: the owner must recheck freshness and persist/apply
    before calling ``set_applied``. ``initial_payload`` is display-only.
    """
    applyRequested = Signal(object)
    originalRequested = Signal()
    settingsChanged = Signal(dict)
    comparisonCompleted = Signal(object)

    def __init__(self, parent=None, *, context_label, settings=None,
                 run_comparison=None, initial_payload=None, current_mode="original"):
        super().__init__(parent)
        self.setWindowTitle("Compare BN background for this group")
        self.setObjectName("drrBnCalibrationDialog")
        self.resize(1080, 780)
        self._run_comparison = run_comparison
        self._worker = None
        self._payload = None
        self._settings_dirty = False
        self._close_pending = False
        self._application_busy = False
        self._initializing = True
        self._current_mode = current_mode
        self._base_settings = deepcopy(_DEFAULTS)
        self._base_settings.update(deepcopy(settings or {}))

        layout = QVBoxLayout(self)
        context = QLabel(context_label)
        context.setWordWrap(True)
        context.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(context)
        self.mode_label = QLabel()
        self.mode_label.setObjectName("bnCurrentMode")
        layout.addWidget(self.mode_label)
        self._update_mode_label()

        explanation = QLabel(
            "Compare the original BN with local fringe shift or shift + amplitude at "
            "25%, 50%, 75%, and 100% correction strength. "
            "D2E improvement is the target; DRR must remain within its allowed change. "
            "Selection and final holdout checks use the current SG settings. "
            "Protected or uncertain spectral features can still prevent use.")
        explanation.setObjectName('bnOptimizationPolicy')
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)
        settings_scroll = QScrollArea()
        settings_scroll.setObjectName("bnSettingsScroll")
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setMinimumWidth(275)
        self.settings_panel = QWidget()
        settings_layout = QVBoxLayout(self.settings_panel)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        settings_layout.addLayout(form)
        self.ranges = {}
        for key, title in (("train_region", "Fit energy"), ("validation_region", "Check energy"),
                           ("core_region", "Correction energy")):
            pair_widget = QWidget()
            pair_layout = QHBoxLayout(pair_widget)
            pair_layout.setContentsMargins(0, 0, 0, 0)
            spins = []
            for index, endpoint in enumerate(("low", "high")):
                spin = QDoubleSpinBox()
                spin.setRange(0.001, 100.0)
                spin.setDecimals(4)
                spin.setSingleStep(0.005)
                spin.setSuffix(" eV")
                spin.setObjectName(f"bn_{key}_{endpoint}")
                spin.setAccessibleName(f"{title} {endpoint}")
                spin.setValue(float(self._base_settings[key][index]))
                spin.valueChanged.connect(self._settings_edited)
                pair_layout.addWidget(spin)
                spins.append(spin)
            self.ranges[key] = spins
            form.addRow(title, pair_widget)
        self.train_regions = QLineEdit(_range_text(self._base_settings.get("train_regions") or []))
        self.train_regions.setObjectName("bnTrainRegions")
        self.train_regions.setAccessibleName("Fit energy ranges (optional)")
        self.train_regions.setPlaceholderText("Empty: use Fit energy above")
        self.train_regions.setToolTip(
            "Optional: low, high; low, high. Empty uses Fit energy above; entered ranges replace it. "
            "Use only background ranges confirmed by independent physical evidence.")
        self.train_regions.textChanged.connect(self._settings_edited)
        fit_ranges_label = QLabel("Fit ranges (optional, eV)")
        fit_ranges_label.setBuddy(self.train_regions)
        form.addRow(fit_ranges_label)
        form.addRow(self.train_regions)
        self.taper = QDoubleSpinBox()
        self.taper.setRange(0.0001, 1.0)
        self.taper.setDecimals(4)
        self.taper.setSingleStep(0.005)
        self.taper.setSuffix(" eV")
        self.taper.setObjectName("bnTaperWidth")
        self.taper.setValue(float(self._base_settings["taper_width"]))
        self.taper.valueChanged.connect(self._settings_edited)
        form.addRow("Transition width", self.taper)
        self.protected = QLineEdit(_range_text(self._base_settings["protected_regions"]))
        self.uncertain = QLineEdit(_range_text(self._base_settings["uncertain_regions"]))
        self.trusted = QLineEdit(_range_text(self._base_settings["trusted_regions"]))
        for widget, name in ((self.protected, "bnProtectedRanges"), (self.uncertain, "bnUncertainRanges"),
                             (self.trusted, "bnTrustedRanges")):
            widget.setObjectName(name)
            widget.setPlaceholderText("low, high; low, high")
            widget.textChanged.connect(self._settings_edited)
        form.addRow("Protected energy (eV)", self.protected)
        form.addRow("Uncertain energy (eV)", self.uncertain)
        form.addRow("Trusted calibration energy (eV)", self.trusted)
        note = QLabel(
            "Ranges: low, high; low, high. Optional fit ranges replace Fit energy above; "
            "leave empty to use that single range. Fit ranges need independent physical confirmation. "
            "The default uncertain interval includes the "
            "unresolved structure near 1.98 eV. Leave trusted calibration energy empty for preview only. "
            "Use a trusted range only with independent evidence that it is suitable for background calibration. "
            "Trusted ranges do not remove uncertain ranges; editing intervals does not itself validate a candidate.")
        note.setWordWrap(True)
        settings_layout.addWidget(note)
        self.protection_note = QLabel(
            "Protected intervals keep raw ΔR/R unchanged. Derivative smoothing and interpolation "
            "can spread differences beyond the correction support; wide SG windows need separate validation.")
        self.protection_note.setObjectName("bnProtectionScope")
        self.protection_note.setWordWrap(True)
        settings_layout.addWidget(self.protection_note)
        self.derivative_settings_label = QLabel()
        self.derivative_settings_label.setObjectName("bnDerivativeSettings")
        self.derivative_settings_label.setWordWrap(True)
        self.derivative_settings_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        settings_layout.addWidget(self.derivative_settings_label)
        self._update_derivative_label()
        self.auto_apply = QCheckBox("Apply automatically after validation")
        self.auto_apply.setObjectName("bnAutoApply")
        self.auto_apply.setChecked(bool(self._base_settings.get("auto_apply", False)))
        self.auto_apply.setToolTip("Otherwise retain the original BN. Saved reports are never applied just by opening this window.")
        settings_layout.insertWidget(1, self.auto_apply)
        settings_layout.addStretch(1)
        settings_scroll.setWidget(self.settings_panel)
        settings_scroll.setMinimumWidth(max(
            settings_scroll.minimumWidth(), self.settings_panel.minimumSizeHint().width()
            + settings_scroll.verticalScrollBar().sizeHint().width() + 2 * settings_scroll.frameWidth()))
        splitter.addWidget(settings_scroll)

        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 750])
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        self.summary = QLabel("No comparison yet. Original BN is retained until an eligible result is applied.")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        result_layout.addWidget(self.summary)
        self.final_confirmation = QLabel(
            "No final holdout result available. Selection scores are not independent final confirmation.")
        self.final_confirmation.setObjectName("bnFinalConfirmation")
        self.final_confirmation.setWordWrap(True)
        self.final_confirmation.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.final_confirmation_d2 = QLabel("D2E: No final holdout result available.")
        self.final_confirmation_d2.setObjectName("bnFinalConfirmationD2")
        self.final_confirmation_d2.setWordWrap(True)
        self.final_confirmation_d2.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        result_layout.addWidget(self.final_confirmation_d2)
        result_layout.addWidget(self.final_confirmation)
        self.joint_validation = QLabel("DRR + D2E: awaiting joint validation.")
        self.joint_validation.setObjectName("bnJointValidation")
        self.joint_validation.setWordWrap(True)
        result_layout.addWidget(self.joint_validation)
        self.table = QTableView()
        self.table.setObjectName("bnCandidateTable")
        self.table.setAccessibleName("BN candidate validation results")
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.model = QStandardItemModel(0, 5, self)
        self.model.setHorizontalHeaderLabels(["Candidate", "D2E selection gain", "DRR diagnostic gain", "Joint validation", "Reason"])
        self.model.horizontalHeaderItem(0).setToolTip(
            "Correction strength is the fraction of the fitted BN correction. "
            "It is separate from the amplitude fitted by the shift + amplitude method.")
        self.model.horizontalHeaderItem(1).setToolTip(
            "Primary target: production D2E residual RMS reduction on selection repeats. "
            "The BN projection used for fitting is a separate training metric. See the independent final holdout above.")
        self.model.horizontalHeaderItem(2).setToolTip(
            "Display DRR residual diagnostic on selection repeats. Zero or negative gain does not itself fail validation; "
            "the DRR tolerance and spectral protection checks decide.")
        self.table.setModel(self.model)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().sectionResized.connect(lambda *_: self.table.resizeRowsToContents())
        self.table.setWordWrap(True)
        result_layout.addWidget(self.table, 1)
        self.tabs.addTab(result_widget, "Results")
        self.figure = Figure(figsize=(7, 5), layout="constrained")
        self.canvas = ThemeAwareFigureCanvasQTAgg(self.figure)
        self.canvas.setAccessibleName("Original and candidate BN preview")
        self.tabs.addTab(self.canvas, "Curves")
        self.report_text = QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setObjectName("bnValidationReport")
        self.tabs.addTab(self.report_text, "Validation details")
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1000)
        self.tabs.addTab(self.log_text, "Log")

        self.status = QLabel("Ready to compare.")
        self.status.setObjectName("bnComparisonStatus")
        self.status.setWordWrap(True)
        self.status.setAccessibleName("BN comparison status")
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setAccessibleName("BN comparison progress")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.hide()
        layout.addWidget(self.progress)
        buttons = QHBoxLayout()
        self.compare_btn = QPushButton("Compare candidates")
        self.compare_btn.setObjectName("bnCompare")
        self.compare_btn.setDefault(True)
        set_fluent_property(self.compare_btn, "fluentAppearance", "primary")
        self.compare_btn.clicked.connect(self.start_comparison)
        self.cancel_btn = QPushButton("Cancel comparison")
        self.cancel_btn.clicked.connect(self.cancel_comparison)
        self.cancel_btn.setEnabled(False)
        self.use_btn = QPushButton("Use validated candidate")
        self.use_btn.setObjectName("bnUseValidated")
        self.use_btn.clicked.connect(self._request_apply)
        self.use_btn.setEnabled(False)
        self.original_btn = QPushButton("Use original BN")
        self.original_btn.setObjectName("bnUseOriginal")
        self.original_btn.clicked.connect(self._request_original)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        for button in (self.compare_btn, self.cancel_btn, self.use_btn, self.original_btn):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)
        self._initializing = False
        self.compare_btn.setEnabled(callable(self._run_comparison))
        if not callable(self._run_comparison):
            self.status.setText("Saved report preview. Comparison is unavailable in this context.")
        if initial_payload is not None:
            self.set_payload(initial_payload)

    def settings(self):
        result = deepcopy(self._base_settings)
        for key, spins in self.ranges.items():
            lo, hi = [spin.value() for spin in spins]
            if lo >= hi:
                raise ValueError(f"{key.replace('_', ' ').capitalize()}: low energy must be below high energy.")
            result[key] = [lo, hi]
        result["taper_width"] = self.taper.value()
        result["train_regions"] = _read_ranges(self.train_regions.text(), "Fit energy ranges")
        result["protected_regions"] = _read_ranges(self.protected.text(), "Protected energy")
        result["uncertain_regions"] = _read_ranges(self.uncertain.text(), "Uncertain energy")
        result["trusted_regions"] = _read_ranges(self.trusted.text(), "Trusted calibration energy")
        result["auto_apply"] = self.auto_apply.isChecked()
        return result

    def _update_derivative_label(self):
        self.derivative_settings_label.setText(
            f"Current D2E comparison: SG {self._base_settings['derivative_window']} points, "
            f"polynomial order {self._base_settings['derivative_polyorder']}. "
            "These settings follow the main DRR controls.")

    def set_derivative_settings(self, window, polyorder):
        """Keep saved curves viewable while withdrawing current joint confirmation."""
        self._base_settings.update(derivative_window=int(window), derivative_polyorder=int(polyorder))
        self._update_derivative_label()
        if self._payload is None or _report_derivative_settings(self._payload) != (int(window), int(polyorder)):
            self._settings_edited()
            self.joint_validation.setText("DRR + D2E: current SG settings have not been validated. Saved results are preview only.")
            self.mode_label.setText("Currently using: " + ("original BN" if self._current_mode == "original"
                                                        else "processed BN · current SG requires comparison"))
            for row in range(self.model.rowCount()):
                if self.model.item(row, 3).text() != "Baseline":
                    self.model.item(row, 3).setText("Preview only")

    @Slot()
    def _settings_edited(self, *_):
        if self._initializing:
            return
        self._settings_dirty = True
        self.use_btn.setEnabled(False)
        self.status.setText("Settings changed. Compare again before using a candidate.")
        for field in (self.train_regions, self.protected, self.uncertain, self.trusted):
            set_fluent_property(field, "fluentInvalid", False)
            field.setAccessibleDescription("")

    @Slot()
    def start_comparison(self):
        if self._worker is not None or self._application_busy or not callable(self._run_comparison):
            return
        try:
            settings = self.settings()
        except ValueError as exc:
            self.status.setText(str(exc))
            self.status.setAccessibleDescription(str(exc))
            for field, prefix in ((self.train_regions, "Fit energy ranges"),
                                  (self.protected, "Protected"), (self.uncertain, "Uncertain"),
                                  (self.trusted, "Trusted")):
                invalid = str(exc).startswith(prefix)
                set_fluent_property(field, "fluentInvalid", invalid)
                field.setAccessibleDescription(str(exc) if invalid else "")
            return
        self.settingsChanged.emit(deepcopy(settings))
        self._settings_dirty = False
        self._payload = None
        self.model.removeRows(0, self.model.rowCount())
        self.summary.setText("Comparing candidates and checking held-out data…")
        self.final_confirmation.setText("Final holdout result pending; selection scores alone do not establish improvement.")
        self.final_confirmation_d2.setText("D2E final holdout pending.")
        self.joint_validation.setText("DRR + D2E: comparing both products at the current SG settings…")
        self.report_text.clear()
        self.log_text.clear()
        self.figure.clear()
        self.canvas.draw_idle()
        self.status.setText("Preparing comparison…")
        self._set_busy(True)
        self._worker = _ComparisonWorker(self._run_comparison, settings)
        self._worker.signals.progress.connect(self._set_progress)
        self._worker.signals.log.connect(self._append_log)
        self._worker.signals.finished.connect(self._comparison_finished)
        self.destroyed.connect(self._worker.cancellation.set)
        QThreadPool.globalInstance().start(self._worker)

    def _set_busy(self, busy):
        busy = busy or self._application_busy
        self.settings_panel.setEnabled(not busy)
        self.compare_btn.setEnabled(not busy and callable(self._run_comparison))
        self.cancel_btn.setEnabled(busy)
        self.original_btn.setEnabled(not busy)
        self.use_btn.setEnabled(not busy and self._is_eligible())
        self.progress.setVisible(busy)
        self.progress.setRange(0, 0 if busy else 100)
        set_fluent_property(self.compare_btn, "fluentBusy", busy)

    @Slot(object)
    def _set_progress(self, value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if np.isfinite(numeric):
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, round(numeric))))

    @Slot(str)
    def _append_log(self, message):
        self.log_text.appendPlainText(message)
        if self._worker is not None and not self._worker.cancellation.is_set():
            self.status.setText(message)

    @Slot()
    def cancel_comparison(self):
        if self._worker is not None:
            self._worker.cancellation.set()
            self.cancel_btn.setEnabled(False)
            self.status.setText("Cancelling comparison…")

    @Slot()
    def _comparison_finished(self):
        worker = self._worker
        if worker is None:
            return
        self._worker = None
        cancelled = worker.cancellation.is_set()
        error, payload = worker.error, worker.payload
        self._set_busy(False)
        if self._close_pending:
            self._close_pending = False
            super().reject()
            return
        if cancelled:
            self.summary.setText("Comparison cancelled. No background was applied.")
            self.final_confirmation.setText("No final holdout result from this comparison.")
            self.final_confirmation_d2.setText("D2E: No final holdout result from this comparison.")
            self.joint_validation.setText("DRR + D2E: no joint confirmation from this comparison.")
            self.status.setText("Cancelled. Current background is unchanged.")
        elif error:
            self.summary.setText("Comparison failed. No background was applied.")
            self.final_confirmation.setText("No final holdout result from this comparison.")
            self.final_confirmation_d2.setText("D2E: No final holdout result from this comparison.")
            self.joint_validation.setText("DRR + D2E: no joint confirmation from this comparison.")
            self.status.setText(error)
            self.log_text.appendPlainText(error)
        else:
            self.set_payload(payload)
            self.comparisonCompleted.emit(payload)
            if self.auto_apply.isChecked() and self._is_eligible():
                self._request_apply()

    def _is_eligible(self):
        return bool(self._payload and self._payload.get("eligible") is True
                    and joint_validation_passed(self._payload.get("report", {}))
                    and _report_derivative_settings(self._payload) ==
                        (self._base_settings['derivative_window'], self._base_settings['derivative_polyorder'])
                    and not self._settings_dirty)

    def set_payload(self, payload):
        """Display a fresh or stored report; this method never applies it."""
        if self._worker is not None:
            raise RuntimeError("Cannot replace a report while comparison is running.")
        if not isinstance(payload, dict):
            raise ValueError("A BN comparison payload must be a dictionary.")
        self._payload = payload
        self._settings_dirty = False
        self._update_mode_label()
        report = payload.get("report", {})
        self.summary.setText(_readable_text(payload.get("summary") or report.get("summary")
                                           or "Comparison finished; see validation details."))
        self.final_confirmation.setText(_final_confirmation_text(payload, report))
        self.final_confirmation_d2.setText(_final_confirmation_text(payload, report, derivative=True))
        derivative = report.get("derivative_settings", {})
        saved_window, saved_poly = _report_derivative_settings(payload)
        saved_settings = (f"SG {saved_window}, order {saved_poly}"
                          + (f", used {derivative['used_window_length']} points" if 'used_window_length' in derivative else ""))
        if not optimization_policy_current(report):
            self.joint_validation.setText(
                "Saved comparison is preview only: it uses an older optimization policy. "
                "Compare again with D2E as the target and DRR as a constraint, including the current weak-feature checks.")
        elif self._is_eligible():
            self.joint_validation.setText(
                f"D2E target and DRR constraint: joint validation passed · {saved_settings}.")
        else:
            self.joint_validation.setText(
                f"D2E target and DRR constraint: preview only; all protection checks must pass. Report: {saved_settings}.")
        self.model.removeRows(0, self.model.rowCount())
        candidates = payload.get("candidates", report.get("candidates", []))
        if isinstance(candidates, dict):
            selected = payload.get('selected_candidate', report.get('selected_method'))
            candidates = [dict(detail, name=name, eligible=self._is_eligible() and name == selected)
                          for name, detail in candidates.items() if isinstance(detail, dict)]
        for row in candidates:
            method = str(row.get("name", "Candidate"))
            detail = report.get('candidates', {}).get(method, {}) if isinstance(report.get('candidates'), dict) else {}
            improvement_text = _gain_text(row.get("improvement_fraction", detail.get('selection', {}).get('mean_improvement_fraction')))
            d2_gain = row.get("improvement_fraction_d2", detail.get('selection_d2', {}).get('mean_improvement_fraction'))
            eligible = row.get("eligible") is True and self._is_eligible()
            original = str(row.get("name", "")).lower() in ("original", "original bn")
            values = [_METHOD_NAMES.get(method, _readable_text(method)).replace(" · ", "\n"), _gain_text(d2_gain), improvement_text,
                      "Baseline" if original else ("Passed" if eligible else "Preview only"),
                      _readable_text(row.get("reason", ""))]
            items = [QStandardItem(value) for value in values]
            for item in items:
                item.setToolTip(item.text())
            items[1].setToolTip(_gain_text(d2_gain) + " production D2E residual gain on selection repeats only; see D2E final holdout above.")
            items[2].setToolTip(improvement_text + " DRR diagnostic gain on selection repeats; the DRR constraint decides acceptance.")
            self.model.appendRow(items)
        self.table.resizeRowsToContents()
        self.report_text.setPlainText(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        self.use_btn.setEnabled(self._is_eligible())
        self.use_btn.setToolTip("Use the selected candidate after the workflow rechecks the inputs."
                                if self._is_eligible() else
                                "No validated candidate is available. Unvalidated candidates can only be previewed.")
        self.status.setText("Validated candidate available."
                            if self._is_eligible() else "Original BN recommended; no candidate passed all checks.")
        self._plot_payload(payload)

    def _plot_payload(self, payload):
        self.figure.clear()
        try:
            energy = np.asarray(payload["energy"], dtype=float)
            original = np.asarray(payload["original_background"], dtype=float)
            candidate = np.asarray(payload["candidate_background"], dtype=float)
            if energy.ndim != 1 or energy.shape != original.shape or energy.shape != candidate.shape:
                raise ValueError("Curve shapes differ.")
            raw = payload.get("mean_raw")
            if raw is not None:
                raw = np.asarray(raw, dtype=float)
                if raw.ndim == 2:
                    raw = np.nanmean(raw, axis=0)
                if raw.shape != energy.shape:
                    raw = None
        except (KeyError, TypeError, ValueError):
            axis = self.figure.subplots()
            text = axis.text(.5, .5, "No spectral preview in this saved report.", ha="center", va="center",
                             transform=axis.transAxes)
            text._use_theme_text = True
            axis.set_axis_off()
            self.canvas.draw_idle()
            return
        axes = self.figure.subplots(3 if raw is not None else 1, 1, squeeze=False).ravel()
        original_line, = axes[0].plot(energy, original, label="Original BN")
        candidate_line, = axes[0].plot(energy, candidate, linestyle="--", label="Candidate BN")
        axes[0].set_ylabel("BN intensity")
        axes[0].legend()
        if raw is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                drr = np.stack((raw / original - 1, raw / candidate - 1))
            axes[1].plot(energy, 100 * drr[0], color=original_line.get_color(), label="Original BN")
            axes[1].plot(energy, 100 * drr[1], color=candidate_line.get_color(), linestyle="--", label="Candidate BN")
            axes[1].set_ylabel("Mean ΔR/R (%)")
            axes[1].legend()
            try:
                # Only two representative mean spectra are processed here, never a full gate map.
                from core.loader import DataCube
                from core.processing import apply_sg_derivative_energy
                saved = _report_derivative_settings(payload)
                window = saved[0] if saved[0] is not None else self._base_settings['derivative_window']
                poly = saved[1] if saved[1] is not None else self._base_settings['derivative_polyorder']
                cube = DataCube(energy, np.arange(2), drr, 'Preview', 'Mean DRR', 'DRR')
                derivative, used = apply_sg_derivative_energy(cube, derivative=2, window_length=int(window), polyorder=int(poly))
                for index, line in enumerate((original_line, candidate_line)):
                    axes[2].plot(energy, derivative.Z[index], color=line.get_color(),
                                 linestyle="--" if index else "-", label=line.get_label())
                axes[2].set_title(f"Production D2E · SG {used} points / order {poly}")
                axes[2].legend()
            except (ValueError, TypeError, ArithmeticError) as exc:
                text = axes[2].text(.5, .5, f"D2E preview unavailable: {exc}", ha="center", va="center", transform=axes[2].transAxes)
                text._use_theme_text = True
            axes[2].set_ylabel("Mean D2E (eV⁻²)")
        axes[-1].set_xlabel("Energy (eV)")
        self.canvas.draw_idle()

    @Slot()
    def _request_apply(self):
        if self._worker is None and not self._application_busy and self._is_eligible():
            self.status.setText("Applying validated candidate for this group…")
            self.applyRequested.emit(self._payload)

    @Slot()
    def _request_original(self):
        if self._worker is None and not self._application_busy:
            self.originalRequested.emit()

    def set_application_busy(self, busy):
        """Prevent duplicate actions while the owner verifies and persists a choice."""
        self._application_busy = bool(busy)
        self._set_busy(self._worker is not None)
        self.cancel_btn.setEnabled(self._worker is not None)

    def set_application_state(self, mode, summary=None):
        self.set_applied(mode, message=summary)

    def set_applied(self, mode, *, message=None):
        """Acknowledge successful application by the owning workflow."""
        self.set_application_busy(False)
        self._current_mode = mode
        self._update_mode_label()
        self.status.setText(message or ("Original BN is now used for this group." if mode == "original"
                                       else "Validated candidate is now used for this group."))
        self._finish_pending_close()

    def set_error(self, message):
        """Display a persistence/application failure without claiming success."""
        self.set_application_busy(False)
        self.status.setText(str(message))
        self.log_text.appendPlainText(str(message))
        self._finish_pending_close()

    def _update_mode_label(self):
        mode = ("original BN" if self._current_mode == "original" else
                "validated BN candidate" if self._is_eligible() else
                "processed BN · compare again with current weak-feature checks"
                if self._payload and not optimization_policy_current(self._payload.get('report', {})) else
                "processed BN · current SG requires comparison")
        self.mode_label.setText("Currently using: " + mode)

    def reject(self):
        if self._worker is not None:
            self._close_pending = True
            self.cancel_comparison()
            return
        if self._application_busy:
            self._close_pending = True
            self.status.setText("Finishing the background change before closing…")
            return
        super().reject()

    def _finish_pending_close(self):
        if self._close_pending and not self._application_busy and self._worker is None:
            self._close_pending = False
            super().reject()

    def closeEvent(self, event):
        if self._worker is not None:
            self._close_pending = True
            self.cancel_comparison()
            event.ignore()
            return
        if self._application_busy:
            self._close_pending = True
            self.status.setText("Finishing the background change before closing…")
            event.ignore()
            return
        super().closeEvent(event)
