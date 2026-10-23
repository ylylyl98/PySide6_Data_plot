"""Configure and inspect the exact sample background used by DRR."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QLineEdit, QCheckBox, QDialogButtonBox, QInputDialog, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QWidget, QScrollArea, QFrame, QSizePolicy)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.widgets import SpanSelector

from core.drr_fitted_background import (BackgroundFitSettings, FITTED_MODE,
    read_reference_raw, validate_recipe, prepare_fitted_group, fit_prepared_group,
    settings_for_group, reference_matches_files)
from core.drr_background_quality import merge_regions, suggest_regions
from core.drr_background_review import check_group, compare_methods
from core.processing_run import resolve_shared_y_axis_request
from core.load_cancellation import LoadCancellation
from ui_qt.common import Worker, QComboBox, QDoubleSpinBox, QSpinBox
from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg
from ui_qt.theme import alias, theme_manager
from ui_qt.fluent_ui.style import themed_icon


def parse_regions(text):
    regions = []
    for part in text.split(';'):
        if not part.strip():
            continue
        pair = part.replace(',', ' ').split()
        if len(pair) != 2:
            raise ValueError('Use two energy limits per region: 1.56 1.64; 1.69 1.77')
        regions.append(tuple(map(float, pair)))
    return tuple(regions)


def periodic_texture_status(reference):
    """Describe what was applied, including validation failures and fallbacks."""
    if not reference.settings.periodic_texture:
        return ''
    if not reference.settings.shared_texture:
        return 'Periodic gaps: inactive while shared texture is off.'
    report = reference.diagnostics.get('periodic_texture', {})
    fraction = report.get('applied_fraction', report.get('prediction_fraction', report.get('corrected_fraction', 0))) or 0
    additional = report.get('additional_corrected_fraction')
    status = report.get('status', 'Not evaluated')
    if fraction <= 0:
        text = 'Periodic gaps: no validated prediction applied.'
    elif additional is not None:
        text = (f'Periodic gaps: {status}; additional corrected spectral points: {100*additional:.0f}%. '
                f'Prediction applied at {100*fraction:.1f}% of spectral points.')
    else:
        text = f'Periodic gaps: {status}; prediction applied at {100*fraction:.0f}% of spectral points.'
    windows = report.get('windows', ())
    applied_windows = [window for window in windows
                       if isinstance(window, dict) and window.get('status') == 'Applied']
    if windows:
        text += f' Validated windows: {len(applied_windows)}.'
        counts = []
        for window in applied_windows:
            components = window.get('components', ())
            count = window.get('component_count', len(components) if isinstance(components, (list, tuple)) else components)
            if isinstance(count, (int, float)) and count > 0:
                counts.append(count)
        if counts:
            text += f' Max. components per local fit: {max(counts):g}.'
    elif report.get('components'):
        components = report['components']
        count = len(components) if isinstance(components, (list, tuple)) else components
        text += f' Components: {count}.'
    reason = report.get('reason') or report.get('rejection_reason')
    if not reason and isinstance(report.get('validation'), dict):
        reason = report['validation'].get('reason') or report['validation'].get('rejection_reason')
    if reason:
        text += f' {reason}'
    elif fraction <= 0 and str(status).lower() not in {'applied', 'success', 'validated'}:
        text += f' {status}.'
    return text


def reference_texture_status(reference):
    """Selecting a reference is distinct from applying validated correction."""
    if not reference.settings.texture_reference_source:
        return ''
    if not reference.settings.shared_texture:
        return 'Fringe reference: inactive while shared texture is off.'
    report = reference.diagnostics.get('reference_texture', {})
    fraction = report.get('applied_fraction', 0) or 0
    if fraction > 0:
        text = f'Fringe reference: applied at {100*fraction:.1f}% of spectral points.'
        additional = report.get('additional_corrected_fraction')
        if additional is not None:
            text += f' Additional corrected spectral points: {100*additional:.1f}%.'
    else:
        text = 'Fringe reference: no validated correction applied.'
    windows = report.get('windows', ())
    if windows:
        count = sum(isinstance(window, dict) and window.get('status') == 'Applied' for window in windows)
        text += f' Validated windows: {count}.'
    reason = report.get('reason') or report.get('rejection_reason')
    if not reason and isinstance(report.get('validation'), dict):
        reason = report['validation'].get('reason') or report['validation'].get('rejection_reason')
    if reason:
        text += f' {reason}'
        if not str(reason).endswith(('.', '!', '?')):
            text += '.'
    elif fraction <= 0:
        status = report.get('status', 'Not evaluated')
        if str(status).lower() not in {'applied', 'success', 'validated'}:
            text += f' {status}.'
    return text


def _preview(folder, files, axis, settings, reference, generation, *, protected_regions=(), progress, log):
    files = (files,) if isinstance(files, str) else tuple(files)
    source = files[0]
    if reference is not None:
        if not reference_matches_files(reference, files):
            raise ValueError('The fitted background does not belong to this selected raw average. Refit the average.')
        raw = read_reference_raw(folder, reference, y_axis=axis)
    else:
        prepared = prepare_fitted_group(folder, files, y_axis=axis, progress=progress.emit, log=log.emit)
        raw = prepared.cube
        if settings is not None:
            _, reference = fit_prepared_group(prepared, settings, stage=log.emit)
    if reference is None:
        report = {**suggest_regions(raw, protected_regions), 'status': 'Needs fit',
                  'warnings': ['Review protection, then fit. Broad or weak features may be missed by screening.']}
        return dict(kind='raw', generation=generation, raw=raw, reference=None, report=report, source=source)
    return generation, raw, reference


class DrrBackgroundDialog(QDialog):
    applied = Signal(object)

    def __init__(self, folder, files, *, y_axis='auto', settings=None, references=(), pool,
                 initial_gate=0., settings_store=None, comparison_window=21, comparison_poly=2, parent=None,
                 draft_mode=False):
        super().__init__(parent)
        self.setWindowTitle('DRR · Fit / view background')
        self.draft_mode = bool(draft_mode)
        self.resize(1100, 900)
        self.folder, self.files, self.y_axis = str(folder), tuple(files), y_axis
        self.grouped = len(self.files) > 1
        self.fit_sources = self.files[:1]
        self.input_label = f'Average ({len(self.files)} files)' if self.grouped else Path(self.files[0]).name
        self.resolved_axis = resolve_shared_y_axis_request(files, y_axis)
        self.pool = pool
        supplied_references = tuple(references)
        self.references = self.compatible_references(supplied_references)
        self.legacy_reference_notice = ('Previous per-file backgrounds cannot be reused for the raw average. Fit the average again.'
                                        if supplied_references and not self.references else '')
        self.recipe_migration_notice = self.group_recipe_error = ''
        self.legacy_manual_regions = ()
        self.raw = self.reference = self.background = None
        self.worker = self.pending = None
        self.generation = 0
        self.closed = False
        self.awaiting_drr = False
        self.settings_store = settings_store
        self.comparison_window, self.comparison_poly = comparison_window, comparison_poly
        self.comparison = None
        self.reports = {}
        self.preview_references = {}
        self.preferred_gate = initial_gate
        self.selector = None
        self.view_defaults = None
        outer = QVBoxLayout(self)
        controls = QWidget()
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 10, 0)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setWidget(controls)
        self.controls_scroll.setMinimumHeight(120)
        self.controls_scroll.setMaximumHeight(400)
        self.controls_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        outer.addWidget(self.controls_scroll)
        self.info = QLabel(('Average the selected raw intensities first, then fit one background and calculate DRR. '
                           if self.grouped else '')
                          + 'Fit outside protected resonances. The estimated denominator is shared in shape across gates, with per-gate gain and tilt. Sample envelope; optional reference supplies fine texture only.')
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.file_combo = QComboBox()
        self.file_combo.setMinimumContentsLength(28)
        self.file_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.file_combo.addItem(self.input_label, self.files[0])
        self.file_combo.setItemData(0, '\n'.join(self.files), Qt.ToolTipRole)
        self.file_combo.setAccessibleName('Averaged measurements to inspect' if self.grouped else 'Measurement to inspect')
        layout.addWidget(self.file_combo)
        if not settings and supplied_references:
            first = supplied_references[0]
            settings = asdict(first.settings)
            overrides = {ref.source: ref.settings.protected_regions for ref in supplied_references
                         if ref.settings.protected_regions != first.settings.protected_regions}
            if overrides:
                settings['protected_regions_by_file'] = overrides
        settings = self.normalize_display_recipe(settings or {})
        self.base_regions_text = self.region_text(settings.get('protected_regions', ()))
        overrides = settings.get('protected_regions_by_file', {})
        self.override_texts = ({key: self.region_text(value) for key, value in overrides.items() if key in self.files}
                               if isinstance(overrides, dict) else {})
        self.group_notice = QLabel()
        self.group_notice.setWordWrap(True)
        self.group_notice.setAccessibleName('Average fit migration notice')
        layout.addWidget(self.group_notice)
        self.update_group_notice()
        row = QHBoxLayout()
        self.template_combo = QComboBox()
        self.template_combo.setAccessibleName('Reusable background recipe template')
        self.template_combo.setMinimumContentsLength(20)
        self.template_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.template_load = QPushButton('Load template')
        self.template_save = QPushButton('Save template…')
        row.addWidget(QLabel('Template'))
        row.addWidget(self.template_combo, 1)
        row.addWidget(self.template_load)
        row.addWidget(self.template_save)
        layout.addLayout(row)
        self.refresh_templates()
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.auto_protect = QCheckBox('Automatically protect peaks at each gate')
        self.auto_protect.setChecked(settings.get('automatic_protection', not bool(settings.get('protected_regions'))))
        self.auto_protect.setToolTip(('Estimate protection at each gate of the averaged raw spectrum. '
                                     if self.grouped else 'Estimate protection at each gate of the measurement. ')
                                    + 'The plot shows the current gate. Unchecking uses the union of all gate regions for manual protection.')
        form.addRow('Peak protection', self.auto_protect)
        self.texture_check = QCheckBox('Correct shared fine texture (learned outside peaks)')
        self.texture_check.setChecked(settings.get('shared_texture', True))
        self.texture_check.setToolTip('At each energy, validates the unprotected gates in two separate groups, then applies their shared response to all gates. Grey bands protect the selected gate during estimation. Texture coverage shows where correction is applied. Disable to compare with the smooth envelope. Protect weak stationary sample features manually.')
        form.addRow('Shared texture', self.texture_check)
        row = QHBoxLayout()
        self.reference_source_edit = QLineEdit(settings.get('texture_reference_source', ''))
        self.reference_source_edit.setReadOnly(True)
        self.reference_source_edit.setPlaceholderText('Optional reference CSV; select explicitly with Browse')
        self.reference_source_edit.setAccessibleName('Fringe reference CSV path')
        self.reference_source_edit.setAccessibleDescription('Fine-texture reference from the same sample position and optical path. The sample envelope is fitted independently. Use Browse to select a file or Clear to disable the reference.')
        self.reference_source_edit.setCursorPosition(0)
        self.reference_browse_button = QPushButton('Browse…')
        self.reference_browse_button.setAccessibleName('Browse for a fringe reference CSV')
        self.reference_browse_button.setToolTip('Select a fine-texture reference from the same sample position and optical path. The sample envelope stays independently fitted.')
        self.reference_clear_button = QPushButton('Clear')
        self.reference_clear_button.setAccessibleName('Clear fringe reference')
        self.reference_clear_button.setToolTip('Disable the external fringe reference.')
        row.addWidget(self.reference_source_edit, 1)
        row.addWidget(self.reference_browse_button)
        row.addWidget(self.reference_clear_button)
        reference_label = QLabel('Fringe reference')
        reference_label.setBuddy(self.reference_source_edit)
        form.addRow(reference_label, row)
        self.update_reference_tooltip()
        row = QHBoxLayout()
        self.periodic_check = QCheckBox('Predict fringes across protected gaps (validated only)')
        self.periodic_check.setChecked(settings.get('periodic_texture', False))
        self.periodic_check.setAccessibleName('Predict periodic fringes across protected gaps')
        self.periodic_check.setToolTip('Optional extension of shared texture. Learn periodic components outside protected peaks, then test prediction on held-out continuous bands. Only supported gaps receive a prediction; failed validation keeps the existing shared correction. Requires shared texture to be enabled.')
        self.periodic_check.setAccessibleDescription(self.periodic_check.toolTip())
        self.periodic_components_spin = QSpinBox()
        self.periodic_components_spin.setRange(1, 5)
        self.periodic_components_spin.setValue(settings.get('periodic_components', 3))
        self.periodic_components_spin.setKeyboardTracking(False)
        self.periodic_components_spin.setAccessibleName('Maximum periodic fringe components per local fit')
        self.periodic_components_spin.setToolTip('Maximum number of periodic components per local spectral fit (1–5). Each window is validated separately and may select fewer components or reject prediction.')
        self.periodic_components_spin.setAccessibleDescription(self.periodic_components_spin.toolTip())
        row.addWidget(self.periodic_check, 1)
        components_label = QLabel('Max. components')
        components_label.setBuddy(self.periodic_components_spin)
        row.addWidget(components_label)
        row.addWidget(self.periodic_components_spin)
        form.addRow('Periodic gaps', row)
        self.regions_edit = QLineEdit(self.base_regions_text)
        self.regions_edit.setPlaceholderText('Lower upper; lower upper (eV), or drag on the spectrum')
        self.regions_edit.setAccessibleName('Protected resonance intervals in electron volts')
        row = QHBoxLayout()
        row.addWidget(self.regions_edit, 1)
        self.file_only = QCheckBox('Only this file')
        self.file_only.setToolTip('Override protection for this measurement while keeping the group method unchanged.')
        self.file_only.setVisible(not self.grouped)
        row.addWidget(self.file_only)
        form.addRow('Protected regions (eV)', row)
        row = QHBoxLayout()
        self.method_combo = QComboBox()
        self.method_combo.setAccessibleName('Background fitting method for the group')
        self.method_combo.addItem('Robust spline', 'spline')
        self.method_combo.addItem('Low-order polynomial', 'polynomial')
        self.method_combo.setCurrentIndex(max(0, self.method_combo.findData(settings.get('method', 'spline'))))
        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(2, 4)
        self.degree_spin.setValue(settings.get('polynomial_degree', 3))
        self.degree_spin.setAccessibleName('Background polynomial degree')
        row.addWidget(self.method_combo, 1)
        row.addWidget(QLabel('Polynomial degree'))
        row.addWidget(self.degree_spin)
        self.compare_button = QPushButton('Compare methods')
        row.addWidget(self.compare_button)
        form.addRow('Method for this group', row)
        row = QHBoxLayout()
        self.intervals_spin = QSpinBox()
        self.intervals_spin.setRange(4, 40)
        self.intervals_spin.setValue(settings.get('spline_intervals', 10))
        self.intervals_spin.setAccessibleName('Background spline intervals')
        self.intervals_spin.setToolTip('More intervals allow a more flexible shared background. Protect resonances first.')
        self.smoothing_spin = QDoubleSpinBox()
        self.smoothing_spin.setDecimals(6)
        self.smoothing_spin.setRange(.000001, 10)
        self.smoothing_spin.setValue(settings.get('smoothing', .001))
        self.smoothing_spin.setKeyboardTracking(False)
        self.smoothing_spin.setAccessibleName('Background smoothing penalty')
        self.smoothing_spin.setToolTip('Larger values penalize curvature more strongly. Compare the background before applying.')
        row.addWidget(QLabel('Spline intervals'))
        row.addWidget(self.intervals_spin)
        row.addWidget(QLabel('Smoothing'))
        row.addWidget(self.smoothing_spin)
        self.fit_button = QPushButton('Fit preview')
        self.fit_button.setDefault(True)
        row.addWidget(self.fit_button)
        form.addRow(row)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.check_button = QPushButton('Check average' if self.grouped else 'Check measurement')
        self.suggest_button = QPushButton('Add suggested regions')
        self.suggest_button.setEnabled(False)
        self.suggest_button.setToolTip('Add suggestions to the fitted input, preserving all existing protected regions. Refit to apply.')
        self.checks_visible = QCheckBox('Show average check' if self.grouped else 'Show file checks')
        row.addWidget(self.check_button)
        row.addWidget(self.suggest_button)
        row.addWidget(self.checks_visible)
        row.addStretch(1)
        layout.addLayout(row)
        self.check_table = QTableWidget(0, 5)
        self.check_table.setHorizontalHeaderLabels(['Measurement', 'Status', 'Anchor RMS (%)', 'Worst gate', 'Review notes'])
        self.check_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.check_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.check_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.check_table.verticalHeader().hide()
        self.check_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.check_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.check_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.check_table.setMaximumHeight(150)
        self.check_table.setMinimumHeight(110)
        self.check_table.setAccessibleName('Averaged-spectrum background quality check' if self.grouped else 'Measurement background quality check')
        self.check_table.hide()
        layout.addWidget(self.check_table)
        layout = outer
        row = QHBoxLayout()
        self.gate_label = QLabel('Gate / frame')
        self.gate_spin = QDoubleSpinBox()
        self.gate_spin.setRange(-1e12, 1e12)
        self.gate_spin.setDecimals(6)
        self.gate_spin.setKeyboardTracking(False)
        self.gate_spin.setAccessibleName('Background preview gate')
        self.gate_spin.setEnabled(False)
        self.raw_check = QCheckBox('Overlay raw spectrum')
        self.raw_check.setChecked(True)
        self.drag_check = QCheckBox('Drag to protect a region')
        self.view_combo = QComboBox()
        self.view_combo.addItems(['Background', 'Second derivative comparison', 'Applied texture correction (%)', 'Texture coverage'])
        self.view_combo.setToolTip('Texture views include the final shared correction, validated reference correction, and periodic gap predictions. Coverage: 0% leaves the spectrum unchanged; 100% applies the full correction. Intermediate weights taper at boundaries. This is not a confidence score.')
        self.view_combo.setEnabled(False)
        for widget in (self.gate_label, self.gate_spin, self.raw_check, self.drag_check):
            row.addWidget(widget)
        row.addWidget(self.view_combo)
        row.addStretch(1)
        layout.addLayout(row)
        self.figure = Figure(constrained_layout=True)
        self.canvas = ThemeAwareFigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(180)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.status = QLabel('Loading original intensity…')
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.apply_button = buttons.addButton('Use in DRR', QDialogButtonBox.ApplyRole)
        self.apply_button.setEnabled(False)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.file_combo.currentIndexChanged.connect(self.load_source)
        self.file_only.toggled.connect(self.scope_changed)
        self.auto_protect.toggled.connect(self.automatic_changed)
        self.texture_check.toggled.connect(self.invalidate_fit)
        self.reference_source_edit.textChanged.connect(self.reference_source_changed)
        self.reference_browse_button.clicked.connect(self.browse_reference)
        self.reference_clear_button.clicked.connect(self.clear_reference)
        self.periodic_check.toggled.connect(self.invalidate_fit)
        self.periodic_components_spin.valueChanged.connect(self.invalidate_fit)
        self.template_save.clicked.connect(self.save_template)
        self.template_load.clicked.connect(self.load_template)
        self.method_combo.currentIndexChanged.connect(self.invalidate_fit)
        self.degree_spin.valueChanged.connect(self.invalidate_fit)
        self.compare_button.clicked.connect(self.compare)
        self.check_button.clicked.connect(self.check_files)
        self.suggest_button.clicked.connect(self.add_suggestions)
        self.checks_visible.toggled.connect(self.show_checks)
        self.check_table.cellDoubleClicked.connect(self.inspect_row)
        self.view_combo.currentIndexChanged.connect(self.draw)
        self.fit_button.clicked.connect(self.fit_preview)
        self.apply_button.clicked.connect(self.apply_fit)
        self.gate_spin.valueChanged.connect(self.draw)
        self.raw_check.toggled.connect(self.draw)
        self.drag_check.toggled.connect(self.draw)
        self.regions_edit.textChanged.connect(self.regions_changed)
        self.intervals_spin.valueChanged.connect(self.invalidate_fit)
        self.smoothing_spin.valueChanged.connect(self.invalidate_fit)
        self.finished.connect(self.stop)
        manager = theme_manager()
        if manager is not None:
            manager.themeChanged.connect(self.theme_changed)
        self.theme_changed(None)
        QTimer.singleShot(0, self.load_source)

    @Slot(object)
    def theme_changed(self, _theme):
        icons = {'Home': 'home.svg', 'Back': 'arrow-left.svg', 'Forward': 'arrow-right.svg',
                 'Pan': 'cursor-move.svg', 'Zoom': 'zoom.svg', 'Subplots': 'layout.svg',
                 'Customize': 'edit.svg', 'Save': 'save.svg'}
        for action in self.toolbar.actions():
            if action.text() in icons:
                action.setIcon(themed_icon(icons[action.text()]))
        self.draw()

    @staticmethod
    def region_text(regions):
        return '; '.join(f'{a:.9g} {b:.9g}' for a, b in regions)

    def compatible_references(self, references):
        references = tuple(references)
        if len(references) != 1 or not reference_matches_files(references[0], self.files):
            return {}
        return {self.files[0]: references[0]}

    def normalize_display_recipe(self, settings):
        self.group_recipe_error = self.recipe_migration_notice = ''
        self.legacy_manual_regions = ()
        if not self.grouped or not settings.get('protected_regions_by_file'):
            return settings
        try:
            normalized = settings_for_group(settings, self.files)
            if normalized.automatic_protection:
                self.recipe_migration_notice = 'Automatic peak protection is recalculated on the averaged raw spectrum; previous per-file overrides are not used.'
                try:
                    self.legacy_manual_regions = settings_for_group(
                        {**settings, 'automatic_protection': False}, self.files).protected_regions
                except (TypeError, ValueError):
                    pass
            else:
                self.recipe_migration_notice = 'Previous per-file protected regions were combined into one protection set for the averaged raw spectrum.'
            return asdict(normalized)
        except (TypeError, ValueError) as exc:
            self.group_recipe_error = 'Review previous per-file protection before fitting the average: ' + str(exc)
            self.recipe_migration_notice = self.group_recipe_error
            return settings

    def update_group_notice(self):
        self.group_notice.setText(' '.join(text for text in (self.legacy_reference_notice, self.recipe_migration_notice) if text))
        self.group_notice.setVisible(bool(self.group_notice.text()))

    def settings(self):
        if self.group_recipe_error:
            raise ValueError(self.group_recipe_error)
        return self.base_settings(() if self.auto_protect.isChecked() else parse_regions(self.regions_edit.text())).validate()

    def base_settings(self, regions):
        auto = self.auto_protect.isChecked()
        return BackgroundFitSettings(() if auto else regions, self.intervals_spin.value(), self.smoothing_spin.value(),
                                     self.method_combo.currentData(), self.degree_spin.value(), auto, self.texture_check.isChecked(),
                                     periodic_texture=self.periodic_check.isChecked(),
                                     periodic_components=self.periodic_components_spin.value(),
                                     texture_reference_source=self.reference_source_edit.text())

    def recipe(self):
        if self.group_recipe_error:
            raise ValueError(self.group_recipe_error)
        recipe = asdict(self.base_settings(() if self.auto_protect.isChecked() else parse_regions(self.base_regions_text)))
        if self.override_texts and not self.auto_protect.isChecked():
            recipe['protected_regions_by_file'] = {key: parse_regions(value) for key, value in self.override_texts.items()}
        return recipe

    def show_file_regions(self):
        source = self.file_combo.currentData()
        for widget in (self.file_only, self.regions_edit):
            widget.blockSignals(True)
        self.file_only.setChecked(source in self.override_texts)
        self.regions_edit.setText(self.override_texts.get(source, self.base_regions_text))
        if self.auto_protect.isChecked():
            reference = self.references.get(source) or self.preview_references.get(source)
            self.regions_edit.setText(self.region_text(reference.protected_regions_at_gate(
                int(np.argmin(abs(reference.gate-self.preferred_gate)))))
                                     if reference and reference.settings.automatic_protection else '')
        for widget in (self.file_only, self.regions_edit):
            widget.blockSignals(False)
        self.regions_edit.setCursorPosition(0)

    def regions_changed(self, text):
        if self.grouped:
            self.group_recipe_error = ''
            self.override_texts.clear()
            self.base_regions_text = text
            if self.recipe_migration_notice:
                self.recipe_migration_notice = 'One protection set is used for the averaged raw spectrum.'
                self.update_group_notice()
            self.invalidate_fit()
            return
        if self.file_only.isChecked():
            self.override_texts[self.file_combo.currentData()] = text
        else:
            self.base_regions_text = text
        self.invalidate_fit()

    def automatic_changed(self, automatic):
        if not automatic:
            # Seed manual editing with the protection on the fitted input.
            self.base_regions_text = (self.region_text(self.reference.settings.protected_regions)
                                      if self.reference is not None else
                                      self.region_text(self.legacy_manual_regions) if self.legacy_manual_regions else self.regions_edit.text())
            references = {**self.references, **self.preview_references}
            self.override_texts = {} if self.grouped else {source: self.region_text(ref.settings.protected_regions)
                for source, ref in references.items()
                if self.region_text(ref.settings.protected_regions) != self.base_regions_text}
        if self.grouped:
            self.group_recipe_error = ''
            self.override_texts.clear()
            if self.recipe_migration_notice:
                self.recipe_migration_notice = ('Automatic peak protection is recalculated on the averaged raw spectrum.'
                                                if automatic else 'One combined protection set is used for the averaged raw spectrum.')
                self.update_group_notice()
        self.invalidate_fit()
        self.show_file_regions()

    def scope_changed(self, checked):
        if self.grouped:
            return
        source = self.file_combo.currentData()
        if checked:
            self.override_texts[source] = self.regions_edit.text()
        else:
            self.override_texts.pop(source, None)
        self.show_file_regions()
        self.invalidate_fit()

    def restore_recipe(self, settings):
        settings = self.normalize_display_recipe(settings)
        self.base_regions_text = self.region_text(settings.get('protected_regions', ()))
        overrides = settings.get('protected_regions_by_file', {})
        self.override_texts = ({key: self.region_text(value) for key, value in overrides.items() if key in self.files}
                               if isinstance(overrides, dict) else {})
        widgets = (self.method_combo, self.degree_spin, self.intervals_spin, self.smoothing_spin,
                   self.auto_protect, self.texture_check, self.periodic_check, self.periodic_components_spin,
                   self.reference_source_edit)
        for widget in widgets:
            widget.blockSignals(True)
        self.method_combo.setCurrentIndex(max(0, self.method_combo.findData(settings.get('method', 'spline'))))
        self.degree_spin.setValue(settings.get('polynomial_degree', 3))
        self.intervals_spin.setValue(settings.get('spline_intervals', 10))
        self.smoothing_spin.setValue(settings.get('smoothing', .001))
        self.auto_protect.setChecked(settings.get('automatic_protection', False))
        self.texture_check.setChecked(settings.get('shared_texture', True))
        self.periodic_check.setChecked(settings.get('periodic_texture', False))
        self.periodic_components_spin.setValue(settings.get('periodic_components', 3))
        self.reference_source_edit.setText(settings.get('texture_reference_source', ''))
        self.reference_source_edit.setCursorPosition(0)
        self.update_reference_tooltip()
        for widget in widgets:
            widget.blockSignals(False)
        self.show_file_regions()
        self.update_method_controls()
        self.update_group_notice()

    def update_method_controls(self):
        idle = self.worker is None and self.pending is None
        self.reference_browse_button.setEnabled(idle and self.texture_check.isChecked())
        self.reference_clear_button.setEnabled(idle and self.texture_check.isChecked() and bool(self.reference_source_edit.text()))
        self.periodic_check.setEnabled(idle and self.texture_check.isChecked())
        self.periodic_components_spin.setEnabled(idle and self.texture_check.isChecked() and self.periodic_check.isChecked())
        spline = self.method_combo.currentData() == 'spline'
        self.intervals_spin.setEnabled(spline)
        self.smoothing_spin.setEnabled(spline)
        self.degree_spin.setEnabled(not spline)
        automatic = self.auto_protect.isChecked()
        self.regions_edit.setReadOnly(automatic)
        self.file_only.setEnabled(not automatic and not self.grouped)
        self.drag_check.setEnabled(not automatic)
        if automatic:
            self.drag_check.setChecked(False)

    def update_reference_tooltip(self):
        text = 'Fine texture only; use the same sample position and optical path. The sample envelope is fitted independently.'
        if self.reference_source_edit.text():
            text += '\n' + self.reference_source_edit.text()
        self.reference_source_edit.setToolTip(text)

    def reference_source_changed(self, *_):
        self.reference_source_edit.setCursorPosition(0)
        self.update_reference_tooltip()
        self.invalidate_fit()

    def browse_reference(self, *_):
        if not self.reference_browse_button.isEnabled():
            return
        selected = self.reference_source_edit.text()
        start = str(Path(selected).parent) if selected and Path(selected).parent.is_dir() else self.folder
        source, _ = QFileDialog.getOpenFileName(self, 'Select fringe reference CSV', start, 'CSV measurements (*.csv)')
        if source:
            self.reference_source_edit.setText(str(Path(source).resolve()))

    def clear_reference(self, *_):
        if self.reference_clear_button.isEnabled():
            self.reference_source_edit.clear()

    def refresh_templates(self):
        self.templates = {}
        if self.settings_store is not None:
            try:
                saved = json.loads(self.settings_store.value('drr/background-templates-v1', '{}'))
                if isinstance(saved, dict):
                    self.templates = {name: item for name, item in saved.items() if isinstance(item, dict)}
            except (ValueError, TypeError):
                pass
        self.template_combo.clear()
        self.template_combo.addItems(sorted(self.templates))
        self.template_load.setEnabled(bool(self.templates))
        self.template_save.setEnabled(self.settings_store is not None)

    def save_template(self, name=None):
        try:
            settings = self.settings()
        except (TypeError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        if self.settings_store is None:
            return
        if not isinstance(name, str):
            name, ok = QInputDialog.getText(self, 'Save background template',
                'Name this sample / measurement condition (stores recipe, not intensity):')
            if not ok:
                return
        name = name.strip()
        if not name:
            return
        self.templates[name] = asdict(settings)
        self.settings_store.setValue('drr/background-templates-v1', json.dumps(self.templates))
        self.refresh_templates()
        self.template_combo.setCurrentText(name)
        self.status.setText('Template saved. Review the averaged spectrum after loading it for a new measurement group.'
                            if self.grouped else 'Template saved. Review the spectrum after loading it for a new measurement.')

    def load_template(self, *_):
        settings = self.templates.get(self.template_combo.currentText())
        if not settings:
            return
        try:
            validated = BackgroundFitSettings.from_mapping(settings)
        except (TypeError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        self.restore_recipe(asdict(validated))
        self.invalidate_fit()
        self.status.setText('Template loaded. Check the averaged spectrum for new or shifted resonances before applying.'
                            if self.grouped else 'Template loaded. Check the measurement for new or shifted resonances before applying.')

    def check_files(self):
        try:
            recipe = self.recipe()
        except (TypeError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        self.request(recipe, None, job='checks')

    def compare(self):
        try:
            settings = self.settings()
        except (TypeError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        self.request(settings, None, job='compare')

    def update_checks(self):
        self.check_table.setRowCount(len(self.fit_sources))
        for row, source in enumerate(self.fit_sources):
            item = self.reports.get(source, {})
            report, ref = item.get('report', {}), item.get('reference')
            rms = f'{ref.anchor_rms_percent:.3g}' if ref is not None else '—'
            notes = ' '.join(report.get('warnings', ())) or ('No heuristic alerts; inspect protected peaks.' if report else '')
            if ref is not None:
                notes += ' ' + reference_texture_status(ref) + ' ' + periodic_texture_status(ref)
            if report.get('suggested_regions'):
                notes += ' Suggested protection (eV): ' + self.region_text(report['suggested_regions'])
            values = [self.input_label, report.get('status', 'Not checked'), rms,
                      f"{report['worst_gate']:.6g}" if 'worst_gate' in report else '—',
                      notes]
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setToolTip('\n'.join(self.files) if column == 0 else value)
                self.check_table.setItem(row, column, cell)
        report = self.reports.get(self.file_combo.currentData(), {}).get('report', {})
        self.suggest_button.setEnabled(bool(report.get('suggested_regions')) and self.worker is None and not self.auto_protect.isChecked())
        self.suggest_button.setToolTip(('Add suggestions to the averaged spectrum, preserving existing protection. Refit to apply.'
            if self.grouped else 'Add suggestions to this measurement, preserving existing protection. Refit to apply.')
            + ('\nSuggested protection (eV): ' + self.region_text(report['suggested_regions'])
               if report.get('suggested_regions') else ''))

    def show_checks(self, visible):
        self.check_table.setVisible(visible)
        if visible:
            QTimer.singleShot(0, lambda: self.controls_scroll.ensureWidgetVisible(self.check_table)
                              if not self.closed else None)

    def inspect_row(self, row, _column):
        self.file_combo.setCurrentIndex(row)
        report = self.reports.get(self.fit_sources[row], {}).get('report', {})
        if 'worst_gate' in report:
            self.preferred_gate = report['worst_gate']
            if self.raw is not None:
                self.gate_spin.setValue(self.preferred_gate)

    def add_suggestions(self):
        report = self.reports.get(self.file_combo.currentData(), {}).get('report', {})
        additions = report.get('suggested_regions', ())
        if not additions:
            return
        try:
            combined = merge_regions((*parse_regions(self.regions_edit.text()), *additions))
        except (TypeError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        if not self.grouped:
            self.file_only.setChecked(True)
        self.regions_edit.setText(self.region_text(combined))
        self.status.setText('Suggestions added to the averaged spectrum. Existing protected regions were preserved. Fit preview to review.'
                            if self.grouped else 'Suggestions added to this measurement. Existing protected regions were preserved. Fit preview to review.')

    def invalidate_fit(self, *_):
        self.awaiting_drr = False
        self.generation += 1
        self.pending = None
        if self.worker:
            self.worker.cancellation.cancel()
        self.reference = self.background = None
        self.comparison = None
        self.preview_references.clear()
        self.reports.clear()
        self.view_combo.setEnabled(False)
        self.view_combo.setCurrentIndex(0)
        self.update_method_controls()
        self.update_checks()
        self.apply_button.setText('Use in DRR')
        self.apply_button.setEnabled(False)
        self.status.setText('Settings changed. Fit preview before using this background in DRR.')
        self.draw()

    def load_source(self, *_):
        source = self.file_combo.currentData()
        self.show_file_regions()
        self.update_method_controls()
        self.raw = self.reference = self.background = None
        self.comparison = None
        self.view_combo.setEnabled(False)
        self.view_combo.setCurrentIndex(0)
        self.gate_spin.setEnabled(False)
        if self.selector is not None:
            self.selector.disconnect_events()
        self.figure.clear()
        self.ax = None
        self.view_defaults = None
        self.canvas.draw_idle()
        reference = self.references.get(source) or self.preview_references.get(source)
        try:
            actual = (replace(reference.settings, protected_regions=())
                      if reference and reference.settings.automatic_protection else reference.settings if reference else None)
            if reference is not None and actual != self.settings():
                reference = None
        except (ValueError, TypeError):
            reference = None
        self.request(None, reference)

    def fit_preview(self):
        self.awaiting_drr = False
        try:
            settings = self.settings()
        except (TypeError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        self.request(settings, None)

    def loaded_changed(self, loaded):
        """Inspect the completed DRR result after automatic loading or Apply."""
        if (self.closed or loaded.mode != 'DRR' or str(loaded.folder) != self.folder
                or tuple(loaded.selected_files) != self.files or loaded.y_axis_spec != self.y_axis):
            return
        was_loaded = any(self.reference is ref for ref in self.references.values())
        self.references = self.compatible_references(loaded.drr_fitted_references)
        if loaded.drr_fitted_references and not self.references:
            self.legacy_reference_notice = 'Previous per-file backgrounds cannot be reused for the raw average. Fit the average again.'
            self.update_group_notice()
            self.preview_references.clear()
            self.reports.clear()
            self.load_source()
            return
        elif self.references:
            self.legacy_reference_notice = ''
            self.update_group_notice()
        refresh = self.references and (self.awaiting_drr or was_loaded or
            (self.reference is None and self.auto_protect.isChecked()))
        self.awaiting_drr = False
        if refresh:
            self.restore_recipe(loaded.drr_fitted_settings)
            self.reports = {ref.source: dict(reference=ref, report=ref.diagnostics) for ref in self.references.values()}
            self.update_checks()
            self.load_source()
        else:
            self.update_apply_state()
            self.draw()

    def update_apply_state(self):
        applied = self.reference is not None and self.reference is self.references.get(self.file_combo.currentData())
        self.apply_button.setText('Use in settings' if self.draft_mode else ('Applied to DRR' if applied else 'Use in DRR'))
        self.apply_button.setEnabled(self.reference is not None and not applied)

    def request(self, settings, reference, *, job='preview'):
        if self.closed:
            return
        self.generation += 1
        self.pending = (settings, reference, self.generation, self.file_combo.currentData(), job)
        self.update_method_controls()
        self.apply_button.setEnabled(False)
        self.fit_button.setEnabled(False)
        self.check_button.setEnabled(False)
        self.compare_button.setEnabled(False)
        self.suggest_button.setEnabled(False)
        self.status.setText('Fitting sample background…' if settings else 'Loading spectrum and fitted background…')
        if job == 'checks':
            self.status.setText('Checking the averaged raw spectrum…' if self.grouped else 'Checking the measurement…')
        elif job == 'compare':
            self.status.setText('Comparing spline and polynomial with identical protection and SG settings…')
        if self.worker:
            self.worker.cancellation.cancel()
        else:
            self.start_pending()

    def start_pending(self):
        if self.closed or self.pending is None:
            return
        settings, reference, generation, source, job = self.pending
        self.pending = None
        if job == 'checks':
            self.worker = Worker(check_group, self.folder, self.files, self.resolved_axis, settings, generation,
                                 cancellation=LoadCancellation())
        elif job == 'compare':
            self.worker = Worker(compare_methods, self.folder, self.files, self.resolved_axis, settings, generation,
                self.comparison_window, self.comparison_poly, cancellation=LoadCancellation())
        else:
            try:
                regions = () if self.group_recipe_error else parse_regions(self.regions_edit.text())
            except ValueError:
                regions = ()
            self.worker = Worker(_preview, self.folder, self.files, self.resolved_axis, settings, reference, generation,
                                 protected_regions=regions, cancellation=LoadCancellation())
        self.worker.signals.result.connect(self.ready, Qt.QueuedConnection)
        self.worker.signals.error.connect(self.failed, Qt.QueuedConnection)
        self.worker.signals.finished.connect(self.finished_work, Qt.QueuedConnection)
        self.pool.start(self.worker)

    @Slot(object)
    def ready(self, payload):
        if isinstance(payload, dict):
            if self.closed or payload['generation'] != self.generation:
                return
            if payload['kind'] == 'checks':
                self.reports = {item['source']: item for item in payload['results']}
                self.preview_references = {item['source']: item['reference'] for item in payload['results'] if item['reference'] is not None}
                self.update_checks()
                self.checks_visible.setChecked(True)
                alerts = sum(item['report']['status'] != 'Checked' for item in payload['results'])
                self.status.setText(f'{self.input_label} checked; {alerts} need review. Double-click to inspect the worst gate. Suggestions never change protection automatically.')
                self.reference = self.background = None
                self.comparison = None
                self.view_combo.setEnabled(False)
                self.view_combo.setCurrentIndex(0)
                current = self.preview_references.get(self.file_combo.currentData())
                if current is not None:
                    self.request(None, current)
                else:
                    self.draw()
                return
            generation, raw, reference = payload['generation'], payload['raw'], payload['reference']
            if payload['kind'] == 'raw':
                self.comparison = None
                self.reports[payload['source']] = dict(report=payload['report'], reference=None)
                self.update_checks()
            else:
                self.comparison = payload
                self.view_combo.setEnabled(True)
        else:
            generation, raw, reference = payload
            self.comparison = None
            self.view_combo.setEnabled(False)
            self.view_combo.setCurrentIndex(0)
        if self.closed or generation != self.generation:
            return
        if reference is not None and not reference_matches_files(reference, self.files):
            self.reference = self.background = None
            self.apply_button.setEnabled(False)
            self.view_combo.setEnabled(False)
            self.status.setText('The fitted background does not belong to this selected raw average. Fit the average again.')
            return
        self.raw, self.reference = raw, reference
        self.background = reference.background() if reference is not None else None
        self.view_combo.setEnabled(reference is not None)
        self.view_combo.model().item(1).setEnabled(self.comparison is not None)
        self.view_combo.model().item(2).setEnabled(reference is not None and reference.log_texture is not None)
        self.view_combo.model().item(3).setEnabled(reference is not None and reference.texture_coverage is not None)
        if reference is not None:
            self.legacy_reference_notice = ''
            self.update_group_notice()
            self.preview_references[reference.source] = reference
            self.reports[reference.source] = dict(reference=reference, report=reference.diagnostics)
            self.update_checks()
            if self.auto_protect.isChecked():
                self.regions_edit.blockSignals(True)
                self.regions_edit.setText(self.region_text(reference.protected_regions_at_gate(
                    int(np.argmin(abs(reference.gate-self.preferred_gate))))))
                self.regions_edit.setCursorPosition(0)
                self.regions_edit.blockSignals(False)
        self.gate_label.setText(raw.gate_label)
        self.gate_spin.blockSignals(True)
        self.gate_spin.setRange(float(raw.gate.min()), float(raw.gate.max()))
        self.gate_spin.setValue(float(raw.gate[np.argmin(abs(raw.gate-self.preferred_gate))]))
        if len(raw.gate) > 1:
            spacing = np.diff(np.unique(raw.gate))
            if spacing.size:
                self.gate_spin.setSingleStep(float(np.min(spacing)))
        self.gate_spin.blockSignals(False)
        self.gate_spin.setEnabled(len(raw.gate) > 1)
        self.update_apply_state()
        self.status.setText((f"{reference.diagnostics.get('status', 'Ready')} · {'Cache reused' if reference.cache_hit else 'Fit calculated'} · "
            + ('Automatic peak protection · ' if reference.settings.automatic_protection else '')
            + f"Anchor coverage: {100*reference.anchor_masks().mean():.0f}% · "
            + f"Texture: {reference.diagnostics.get('shared_texture', {}).get('status', 'Disabled')} · "
            + f"Corrected spectral points: {100*np.mean(reference.texture_coverage > .5) if reference.texture_coverage is not None else 0:.0f}% · "
            + f'Anchor RMS: {reference.anchor_rms_percent:.3g}%. '
            + reference_texture_status(reference) + ' '
            + periodic_texture_status(reference) + ' '
            + ' '.join(reference.diagnostics.get('warnings', ()))
            + ' Residual is not uncertainty under peaks.')
            if reference else 'All gates screened. Review suggested regions or draw protection, then Fit preview. Broad/weak peaks may be missed.')
        if self.comparison is not None:
            metrics = self.comparison['metrics']
            delta = metrics['derivative_relative_rms_percent']
            background_delta = metrics['background_max_difference_percent']
            self.status.setText(f"Comparison only; group method stays {reference.settings.method}. "
                + (f"Maximum background difference in protected bands: {background_delta:.3g}%. "
                   if background_delta is not None else 'No protected bands were detected. ')
                + (f'Derivative RMS difference / primary RMS: {delta:.3g}%. ' if delta is not None else '')
                + f"SG {metrics['sg_window']} points / degree {metrics['sg_polyorder']}. Switch the plot view to inspect d²/dE².")
            scores = []
            for ref, score in zip((reference, self.comparison['alternate']), metrics['validation']):
                value = score['rms_percent']
                scores.append(f"{ref.settings.method}: {value:.3g}% ({score['blocks']} bands)" if value is not None
                              else f'{ref.settings.method}: unavailable (insufficient support)')
            self.status.setText(self.status.text() + ' Envelope held-out RMS — ' + '; '.join(scores)
                                + '. This does not validate background under peaks. '
                                + reference_texture_status(reference) + ' '
                                + periodic_texture_status(reference))
        self.draw()

    @Slot(str)
    def failed(self, message):
        if not self.closed and not self.worker.cancellation.cancelled:
            self.reference = self.background = None
            self.comparison = None
            self.view_combo.setEnabled(False)
            self.view_combo.setCurrentIndex(0)
            self.status.setText(message.split('\n\n')[0])
            self.apply_button.setEnabled(False)
            self.draw()

    @Slot()
    def finished_work(self):
        self.worker = None
        if not self.closed:
            if self.pending is not None:
                self.start_pending()
                return
            self.fit_button.setEnabled(True)
            self.check_button.setEnabled(True)
            self.compare_button.setEnabled(True)
            self.update_method_controls()
            self.update_checks()

    def draw(self, *_):
        if self.closed or self.raw is None:
            return
        if self.selector is not None:
            self.selector.disconnect_events()
        previous_axis = getattr(self, 'ax', None)
        # Preserve a user's zoom while stepping through gates or hiding raw data.
        limits = None
        if previous_axis is not None and self.view_defaults is not None:
            current_limits = previous_axis.get_xlim(), previous_axis.get_ylim()
            if not np.allclose(current_limits, self.view_defaults):
                limits = current_limits
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        index = int(np.argmin(abs(self.raw.gate-self.gate_spin.value())))
        self.preferred_gate = float(self.raw.gate[index])
        if self.auto_protect.isChecked() and self.reference is not None:
            self.regions_edit.blockSignals(True)
            self.regions_edit.setText(self.region_text(self.reference.protected_regions_at_gate(index)))
            self.regions_edit.setCursorPosition(0)
            self.regions_edit.blockSignals(False)
        self.gate_spin.blockSignals(True)
        self.gate_spin.setValue(self.preferred_gate)
        self.gate_spin.blockSignals(False)
        self.raw_line = self.background_line = None
        self.alternate_line = self.envelope_line = self.texture_line = None
        self.coverage_line = self.support_line = None
        derivative_view = self.comparison is not None and self.view_combo.currentIndex() == 1
        texture_view = self.reference is not None and self.view_combo.currentIndex() == 2
        coverage_view = self.reference is not None and self.view_combo.currentIndex() == 3
        if coverage_view:
            self.coverage_line, = self.ax.plot(self.reference.energy, 100*self.reference.texture_coverage,
                color=alias('source_processed_foreground'), label='Applied correction weight (not confidence)')
            self.support_line, = self.ax.plot(self.reference.energy, 100*self.reference.anchor_masks().mean(axis=0),
                linestyle='--', color=alias('source_new_foreground'), label='Unprotected gates available (%)')
            self.ax.set_ylim(-3, 105)
        elif texture_view:
            texture = self.reference.log_texture
            self.texture_line, = self.ax.plot(self.reference.energy,
                100*np.expm1(texture) if texture is not None else np.zeros(len(self.reference.energy)),
                color=alias('source_processed_foreground'), label='Applied texture (B / smooth envelope − 1)')
            self.ax.axhline(0, color=alias('text_secondary'), linewidth=.7)
        elif derivative_view:
            for cube, reference, style, color in zip(self.comparison['derivatives'],
                    (self.reference, self.comparison['alternate']), ('-', '--'),
                    ('source_new_foreground', 'source_processed_foreground')):
                self.ax.plot(cube.energy, cube.Z[index], linestyle=style, color=alias(color),
                             label=f'{reference.settings.method}: d²(I/B − 1)/dE²')
        elif self.raw_check.isChecked() or self.background is None:
            self.raw_line, = self.ax.plot(self.raw.energy, self.raw.Z[index], color=alias('source_new_foreground'),
                                         linewidth=1, label='Averaged raw spectrum' if self.grouped else 'Raw sample spectrum')
        if self.background is not None and not derivative_view and not texture_view and not coverage_view:
            is_loaded = self.reference is self.references.get(self.file_combo.currentData())
            self.background_line, = self.ax.plot(self.raw.energy, self.background[index], linestyle='--',
                linewidth=1.7, color=alias('source_processed_foreground'),
                label='Background used in loaded DRR' if is_loaded else 'Fitted background (preview)')
            if self.reference.log_texture is not None and np.any(self.reference.log_texture):
                self.envelope_line, = self.ax.plot(self.raw.energy, self.reference.background(include_texture=False)[index],
                    linestyle=':', linewidth=1.1, color=alias('text_secondary'), label='Smooth envelope (before texture correction)')
            if self.comparison is not None:
                other = self.comparison['alternate']
                self.alternate_line, = self.ax.plot(other.energy, self.comparison['alternate_background'][index], linestyle=':',
                    linewidth=1.7, color=alias('text_secondary'), label=f'{other.settings.method} background (comparison)')
        try:
            regions = parse_regions(self.regions_edit.text())
        except ValueError:
            regions = ()
        if coverage_view:
            from core.drr_background_structure import mask_regions
            regions = mask_regions(self.reference.energy, self.reference.texture_coverage == 0)
        for i,(lo,hi) in enumerate(regions):
            if np.isfinite([lo,hi]).all() and lo < hi:
                self.ax.axvspan(lo, hi, color=alias('text_secondary'), alpha=.12,
                               label=('No texture correction' if coverage_view else 'Protected from fitting') if i == 0 else None)
        self.ax.set(xlabel='Photon energy (eV)', ylabel=('Coverage (%)' if coverage_view else 'Applied correction (%)' if texture_view else
                    'd²(I/B − 1)/dE² (eV⁻²)' if derivative_view else 'Intensity (counts)'),
                    title='Applied texture coverage across gates' if coverage_view else f'{self.raw.gate_label} = {self.preferred_gate:.6g}')
        self.ax.legend(loc='best')
        # SpanSelector starts with a hidden rectangle at x=0. Explicit spectral
        # limits keep that artist out of the initial autoscaled energy range.
        self.ax.set_xlim(float(self.raw.energy[0]), float(self.raw.energy[-1]))
        self.view_defaults = self.ax.get_xlim(), self.ax.get_ylim()
        view_changed = self.view_combo.currentIndex() != getattr(self, '_last_plot_view', 0)
        if limits is not None and not view_changed:
            self.ax.set_xlim(limits[0])
            self.ax.set_ylim(limits[1])
        self.selector = SpanSelector(self.ax, self.add_region, 'horizontal', useblit=False,
            props=dict(facecolor=alias('source_new_foreground'), alpha=.2))
        self.selector.set_active(self.drag_check.isChecked() and not derivative_view and not texture_view and not coverage_view)
        self._last_plot_view = self.view_combo.currentIndex()
        self.toolbar.update()
        self.canvas.draw_idle()

    def add_region(self, lo, hi):
        if self.toolbar.mode or hi <= lo:
            return
        text = self.regions_edit.text().strip().rstrip(';')
        self.regions_edit.setText((text + '; ' if text else '') + f'{lo:.6g} {hi:.6g}')

    def apply_fit(self):
        if self.reference is not None and self.apply_button.isEnabled():
            try:
                recipe = self.recipe()
                validate_recipe(recipe, self.files)
            except (TypeError, ValueError) as exc:
                self.status.setText(str(exc))
                return
            self.applied.emit(recipe)

    def stop(self, *_):
        self.closed = True
        self.generation += 1
        self.pending = None
        if self.worker:
            self.worker.cancellation.cancel()

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)


def open_background_dialog(owner):
    files, folder = tuple(owner.drr_selected_files), str(owner.current_folder or '')
    if not folder or not files or any(Path(name).suffix.lower() != '.csv' for name in files):
        owner._show_error('Select original DRR CSV measurements to fit or inspect their backgrounds.')
        return
    axis = owner._selected_y_axis_spec('drr')
    loaded = owner.loaded
    references = ()
    if (loaded and loaded.mode == 'DRR' and loaded.folder == folder
            and tuple(loaded.selected_files) == files and loaded.y_axis_spec == axis):
        references = getattr(loaded, 'drr_fitted_references', ())
    previous = getattr(owner, '_drr_background_dialog', None)
    if previous is not None:
        reusable = (references if len(references) == 1 and reference_matches_files(references[0], files) else ())
        if (previous.files == files and previous.folder == folder and previous.y_axis == axis
                and tuple(map(id, previous.references.values())) == tuple(map(id, reusable))):
            previous.show()
            previous.raise_()
            previous.activateWindow()
            return
        previous.close()
    key = 'drr/fitted-background/' + hashlib.sha256(json.dumps([folder, files, axis]).encode()).hexdigest()
    settings = dict(getattr(owner, '_drr_fitted_settings', {}))
    try:
        saved = json.loads(owner.settings.value(key, '{}'))
        if saved:
            validate_recipe(saved, files)
            settings = saved
    except (TypeError, ValueError):
        pass
    if references:
        settings = dict(loaded.drr_fitted_settings) or asdict(references[0].settings)
    dialog = DrrBackgroundDialog(folder, files, y_axis=axis, settings=settings, references=references,
        pool=owner.thread_pool, initial_gate=owner.drr_spins['gate'].value(), settings_store=owner.settings,
        comparison_window=owner.drr_sg_window_spin.value(), comparison_poly=owner.drr_sg_poly_spin.value(), parent=owner)
    owner._drr_background_dialog = dialog

    def apply(settings):
        if tuple(owner.drr_selected_files) != files or str(owner.current_folder) != folder or owner._selected_y_axis_spec('drr') != axis:
            dialog.status.setText('DRR source selection changed. Reopen Fit / view background for the current selection.')
            return
        owner._drr_fitted_settings = settings
        owner.settings.setValue(key, json.dumps(settings))
        owner.drr_baseline_combo.blockSignals(True)
        owner.drr_baseline_combo.setCurrentText(FITTED_MODE)
        owner.drr_baseline_combo.blockSignals(False)
        owner.drr_controller._cancel_auto_external()
        owner._drr_assignments = ()
        owner._drr_assignments_automatic = False
        owner._drr_baseline_user_selected = True
        owner.drr_baseline_files_manual = []
        owner.drr_baseline_files_found = []
        blocked = owner.drr_pin_baseline_chk.blockSignals(True)
        owner.drr_pin_baseline_chk.setChecked(False)
        owner.drr_pin_baseline_chk.blockSignals(blocked)
        owner.drr_controller._update_drr_baseline_controls()
        owner.drr_controller._update_drr_selection_labels()
        dialog.awaiting_drr = True
        owner._start_load('DRR')
        dialog.status.setText('Applying fitted background to DRR. The completed result will appear here for inspection.')

    def release():
        if getattr(owner, '_drr_background_dialog', None) is dialog:
            owner._drr_background_dialog = None
        dialog.deleteLater()

    dialog.applied.connect(apply)
    dialog.finished.connect(release)
    dialog.show()
