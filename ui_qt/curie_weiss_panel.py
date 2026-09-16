"""Curie–Weiss view of the Organizer's selected temperature series."""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                               QLabel, QComboBox, QDoubleSpinBox, QPushButton,
                               QPlainTextEdit, QScrollArea, QFileDialog, QMessageBox, QSizePolicy,
                               QCheckBox, QTabWidget)


class CurieWeissPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.records = []
        self.groups = {}
        self.branches = ()
        self.series_id = None
        self.results = {}
        self.points = []
        self.field_fits = []
        self.field_figure = self.field_canvas = None
        self.figure = self.canvas = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.refit)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        self.layout = QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        hint = QLabel('s(T) = s0 + A / (T − θCW) · near-zero MCD slopes\n'
                      'Select a Temperature series and one energy group. Use conditions on the right to exclude points.')
        hint.setWordWrap(True)
        self.layout.addWidget(hint)
        grid = QGridLayout()
        self.group_combo = QComboBox()
        self.group_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.group_combo.setMinimumContentsLength(12)
        self.temperature_combo = QComboBox()
        self.temperature_combo.addItem('Setpoint / catalog T', 'setpoint')
        self.temperature_combo.addItem('Measured T only', 'measured')
        self.t_min, self.t_max = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (self.t_min, self.t_max):
            spin.setRange(0, 100000)
            spin.setDecimals(4)
            spin.setSuffix(' K')
            spin.setKeyboardTracking(False)
        self.t_max.setValue(1000)
        self.field_refit_chk = QCheckBox('Refit slopes from MCD–B')
        self.field_refit_chk.setToolTip('Use one B interval for every temperature and branch. Unchecked: use saved slopes.')
        self.b_min, self.b_max = QDoubleSpinBox(), QDoubleSpinBox()
        for spin, value in ((self.b_min, -.2), (self.b_max, .2)):
            spin.setRange(-1000, 1000)
            spin.setDecimals(4)
            spin.setSingleStep(.05)
            spin.setSuffix(' T')
            spin.setValue(value)
            spin.setKeyboardTracking(False)
            spin.setEnabled(False)
        self.background_combo = QComboBox()
        self.method_combo = QComboBox()
        self.method_combo.addItem('Inverse slope linear · Tang-style', 'inverse')
        self.method_combo.addItem('Slope nonlinear · previous method', 'slope')
        self.method_combo.setAccessibleName('CW fitting method')
        self.layout.addWidget(self.method_combo)
        for text, key in [('Zero background', 'zero'), ('Fixed background', 'fixed'),
                          ('Fit background', 'fit')]:
            self.background_combo.addItem(text, key)
        self.background_spin = QDoubleSpinBox()
        self.background_spin.setRange(-1e6, 1e6)
        self.background_spin.setDecimals(8)
        self.background_spin.setSingleStep(.001)
        self.background_spin.setKeyboardTracking(False)
        self.background_spin.setEnabled(False)
        for row, (left, control, right, other) in enumerate([
                ('Energy group', self.group_combo, 'Temperature', self.temperature_combo),
                ('T min', self.t_min, 'T max', self.t_max),
                ('Background', self.background_combo, 's0 (MCD/T)', self.background_spin),
                ('B min', self.b_min, 'B max', self.b_max)]):
            label, label2 = QLabel(left), QLabel(right)
            label.setBuddy(control)
            label2.setBuddy(other)
            grid.addWidget(label, row, 0)
            grid.addWidget(control, row, 1)
            grid.addWidget(label2, row, 2)
            grid.addWidget(other, row, 3)
            control.setAccessibleName(left)
            other.setAccessibleName(right)
        self.layout.addWidget(self.field_refit_chk)
        self.layout.addLayout(grid)
        actions = QHBoxLayout()
        self.fit_btn = QPushButton('Refit')
        self.export_btn = QPushButton('Save CW fit…')
        self.export_btn.setEnabled(False)
        actions.addWidget(self.fit_btn)
        actions.addWidget(self.export_btn)
        actions.addStretch(1)
        self.layout.addLayout(actions)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(115)
        self.summary.setMaximumHeight(125)
        self.summary.setAccessibleName('Curie–Weiss results and diagnostics')
        summary_container = QWidget()
        summary_container.setMinimumHeight(125)
        summary_container.setMaximumHeight(150)
        summary_layout = QVBoxLayout(summary_container)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.addWidget(self.summary)
        self.layout.addWidget(summary_container)
        self.empty_label = QLabel('Select a Temperature comparison series.')
        self.empty_label.setWordWrap(True)
        self.layout.addWidget(self.empty_label, 1)
        self.plot_tabs = QTabWidget()
        self.cw_plot_host = QWidget()
        self.cw_plot_layout = QVBoxLayout(self.cw_plot_host)
        self.cw_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.field_plot_host = QWidget()
        self.field_plot_layout = QVBoxLayout(self.field_plot_host)
        self.field_record_combo = QComboBox()
        self.field_record_combo.setAccessibleName('Temperature curve for field fit preview')
        self.field_record_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.field_record_combo.setMinimumContentsLength(20)
        self.field_summary = QLabel('Enable “Refit slopes from MCD–B” to inspect the linear fits.')
        self.field_summary.setWordWrap(True)
        self.field_plot_layout.addWidget(self.field_record_combo)
        self.field_plot_layout.addWidget(self.field_summary)
        self.plot_tabs.addTab(self.cw_plot_host, 'CW fit')
        self.plot_tabs.addTab(self.field_plot_host, 'MCD–B slopes')
        self.layout.addWidget(self.plot_tabs, 1)
        note = QLabel('Default: equal-weight linear fit of 1/(s−s0) vs T, θCW = −intercept/slope. '
                      'Error bars: propagated slope SE; fixed background treated as exact. '
                      'Reference: Tang et al., Nat. Nanotechnol. (2023), doi:10.1038/s41565-022-01309-8. '
                      'Regression weights are an app choice; not specified by the cited figure caption. '
                      'Requires a paramagnetic temperature range and approximately temperature-independent '
                      'MCD-to-magnetization conversion. θCW is not proof of magnetic order. '
                      'Confidence intervals exclude temperature, optical and other systematic errors.')
        note.setWordWrap(True)
        self.layout.addWidget(note)
        for combo in (self.group_combo, self.temperature_combo, self.background_combo, self.method_combo):
            combo.currentIndexChanged.connect(self._changed)
        for spin in (self.t_min, self.t_max, self.background_spin, self.b_min, self.b_max):
            spin.valueChanged.connect(self._changed)
        self.field_refit_chk.toggled.connect(self._changed)
        self.field_record_combo.currentIndexChanged.connect(self._draw_field)
        self.fit_btn.clicked.connect(self.refit)
        self.export_btn.clicked.connect(self._save)

    def _changed(self, *_):
        inverse = self.method_combo.currentData() == 'inverse'
        self.background_combo.model().item(2).setEnabled(not inverse)
        if inverse and self.background_combo.currentData() == 'fit':
            self.background_combo.setCurrentIndex(0)
        self.background_spin.setEnabled(self.background_combo.currentData() == 'fixed')
        self.b_min.setEnabled(self.field_refit_chk.isChecked())
        self.b_max.setEnabled(self.field_refit_chk.isChecked())
        self.results = {}
        self.export_btn.setEnabled(False)
        self.summary.setPlainText('Updating fit…')
        self._timer.start()

    def clear(self, message='Select a Temperature comparison series.'):
        self._timer.stop()
        self.records = []
        self.results = {}
        self.points = []
        self.field_fits = []
        self._update_field_preview()
        self.series_id = None
        self.export_btn.setEnabled(False)
        self.summary.setPlainText(message)
        self.empty_label.setText(message)
        self.empty_label.show()
        if self.canvas is not None:
            self.figure.clear()
            self.canvas.hide()

    def set_series(self, series_id, records, groups, branches):
        changed = series_id != self.series_id
        self.series_id = series_id
        self.records = list(records)
        self.groups = dict(groups)
        self.branches = tuple(branches)
        previous = self.group_combo.currentData() if not changed else None
        names = sorted({groups.get(r.record_id, '') for r in records} - {''})
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        for name in names:
            members = [r for r in records if groups.get(r.record_id) == name]
            energy = np.mean([r.center_ev for r in members])
            self.group_combo.addItem(f'{name} · {energy:.5f} eV', name)
        self.group_combo.setCurrentIndex(max(0, self.group_combo.findData(previous)))
        self.group_combo.blockSignals(False)
        if changed:
            self.t_min.blockSignals(True)
            self.t_max.blockSignals(True)
            self.t_min.setValue(0)
            self.t_max.setValue(1000)
            self.t_min.blockSignals(False)
            self.t_max.blockSignals(False)
        self._changed()

    def _temperature(self, record):
        if self.temperature_combo.currentData() == 'measured':
            return record.temperature_measured_k
        if 'assumed' in record.condition_sources.get('T', '').lower():
            return None
        return record.condition_value('T')

    def refit(self):
        self._timer.stop()
        self.results = {}
        self.points = []
        self.field_fits = []
        self.export_btn.setEnabled(False)
        if not self.records:
            self.summary.setPlainText('No included temperature points.')
            if self.figure is not None:
                self.figure.clear()
                self.canvas.hide()
            self.empty_label.setText('No included temperature points.')
            self.empty_label.show()
            self._update_field_preview()
            return
        from core.curie_weiss import fit_curie_weiss, fit_inverse_curie_weiss
        group = self.group_combo.currentData()
        members = [r for r in self.records if self.groups.get(r.record_id) == group]
        warnings = []
        use_field = self.field_refit_chk.isChecked()
        if use_field and self.b_min.value() >= self.b_max.value():
            self.summary.setPlainText('B min must be smaller than B max.')
            self._draw()
            return
        if use_field and not self.b_min.value() < 0 < self.b_max.value():
            warnings.append('B interval does not straddle zero; this slope may not represent zero-field susceptibility.')
        if self.t_min.value() > self.t_max.value():
            self.summary.setPlainText('T min must not exceed T max.')
            self._draw()
            return
        eligible = [r for r in members if self._temperature(r) is not None
                    and np.isfinite(self._temperature(r)) and self._temperature(r) > 0
                    and self.t_min.value() <= self._temperature(r) <= self.t_max.value()]
        if eligible and np.ptp([r.center_ev for r in eligible]) > 1e-6:
            warnings.append('Window centers vary: verify the same optical transition at every temperature.')
        if len({round(r.width_mev, 6) for r in eligible}) > 1:
            self.summary.setPlainText('Window widths differ. Select windows with the same width.')
            self._draw()
            return
        if not use_field and (len({r.fit_window_t for r in eligible}) > 1 or any(r.fit_window_t is None for r in eligible)):
            warnings.append('Verify saved low-field fit ranges are consistent (some may be unspecified).')
        missing = 0
        for record in members:
            temperature = self._temperature(record)
            if temperature is None or not np.isfinite(temperature) or temperature <= 0:
                missing += 1
                continue
            if not self.t_min.value() <= temperature <= self.t_max.value():
                continue
            field_lookup = {}
            if use_field:
                from core.mcd_slope_refit import refit_record_slopes
                try:
                    for fitted in refit_record_slopes(record, self.b_min.value(), self.b_max.value(), self.branches):
                        fitted.update(record_id=record.record_id, source=record.source_file,
                                      trace_path=str(record.trace_path), temperature_k=float(temperature),
                                      center_ev=record.center_ev, width_mev=record.width_mev)
                        self.field_fits.append(fitted)
                        field_lookup[fitted['branch']] = fitted
                except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as exc:
                    warnings.append(f'{temperature:g} K: cannot refit {record.source_file}: {exc}')
                    continue
            for branch in self.branches:
                field_fit = field_lookup.get(branch)
                slope = field_fit['slope'] if use_field and field_fit else (None if use_field else record.slope(branch, 'near_zero'))
                if field_fit and field_fit['status'] != 'ok':
                    warnings.append(f"{temperature:g} K / {branch}: {field_fit['status']} "
                                    f"({field_fit['n']} finite points); need at least 3 distinct B values.")
                if slope is None or not np.isfinite(slope):
                    missing += 1
                    continue
                if field_fit and (field_fit['jump_flag'] or field_fit['curvature_flag']):
                    warnings.append(f'{temperature:g} K / {branch}: inspect MCD–B residuals for curvature or jumps.')
                if field_fit and field_fit['n'] < 5:
                    warnings.append(f'{temperature:g} K / {branch}: only {field_fit["n"]} B points; slope may be noise-sensitive.')
                self.points.append(dict(record_id=record.record_id, source=record.source_file,
                    settings_path=str(record.settings_path), branch=branch, temperature_k=float(temperature),
                    temperature_source=('measured' if self.temperature_combo.currentData() == 'measured'
                                        else record.condition_sources.get('T', 'catalog')),
                    center_ev=record.center_ev, width_mev=record.width_mev,
                    saved_fit_window_t=record.fit_window_t, slope=float(slope),
                    slope_source='refit' if use_field else 'saved',
                    slope_se=field_fit['slope_se'] if field_fit else self._saved_slope_se(record, branch, slope),
                    field_point_count=field_fit['n'] if field_fit else None,
                    field_actual_min_t=field_fit['field_min_t'] if field_fit else None,
                    field_actual_max_t=field_fit['field_max_t'] if field_fit else None))
        if missing:
            warnings.append(f'{missing} record/branch entries omitted: missing/assumed temperature or invalid slope.')
        lines = ([f'MCD–B refit: [{self.b_min.value():g}, {self.b_max.value():g}] T; free intercept.']
                 if use_field else ['Using saved near-zero slopes.'])
        inverse_method = self.method_combo.currentData() == 'inverse'
        lines.append('Equal-weight linear fit: 1/(s−s0) = mT+b; θCW = −b/m.' if inverse_method
                     else 'Equal-weight nonlinear fit in slope space.')
        for branch in self.branches:
            block = sorted([p for p in self.points if p['branch'] == branch], key=lambda p:p['temperature_k'])
            try:
                fit_fn = fit_inverse_curie_weiss if inverse_method else fit_curie_weiss
                extra = {'slope_se': [p['slope_se'] for p in block]} if inverse_method else {}
                result = fit_fn([p['temperature_k'] for p in block], [p['slope'] for p in block],
                    background_mode=self.background_combo.currentData(), background=self.background_spin.value(), **extra)
            except ValueError as exc:
                lines.append(f'{branch}: {exc}')
                continue
            self.results[branch] = result
            ci = result['theta_ci95_k']
            interval = f'[{ci[0]:.3g}, {ci[1]:.3g}] K' if ci is not None else 'unavailable'
            lines.append(f"{branch}: θCW = {result['theta_k']:.4g} K; 95% CI {interval}\n"
                         f"  A={result['amplitude']:.5g}, s0={result['background']:.5g}; "
                         f"R²={result['r_squared']:.3f}; N={result['n']}; "
                         f"T={result['temperature_min_k']:.4g}–{result['temperature_max_k']:.4g} K\n"
                         f"  {result['interpretation']} (model-dependent)")
            warnings.extend(result['warnings'])
            for point, predicted, residual in zip(block, result['predicted'], result['residuals']):
                point.update(predicted_slope=predicted, residual=residual)
            if inverse_method:
                for i, point in enumerate(block):
                    point.update(inverse_slope=result['inverse_slope'][i],
                                 inverse_slope_se=result['inverse_slope_se'][i],
                                 predicted_inverse=result['predicted_inverse'][i],
                                 inverse_residual=result['inverse_residuals'][i])
        self.warnings = list(dict.fromkeys(warnings))
        lines += self.warnings
        self.summary.setPlainText('\n'.join(lines) or 'No included points in this energy group.')
        self._draw()
        self.export_btn.setEnabled(bool(self.results))

    @staticmethod
    def _saved_slope_se(record, branch, slope):
        """Use only the error of the exact saved window/branch/slope."""
        try:
            payload = json.loads(record.settings_path.read_text(encoding='utf-8-sig'))
            fits = payload.get('slopes', [])
            if isinstance(fits, dict):
                fits = fits.get('fits', [])
            if isinstance(fits, dict):
                fits = list(fits.values())
            for fit in fits:
                if (fit.get('branch') == branch and fit.get('region') == 'low'
                        and fit.get('status') == 'ok'
                        and np.isclose(float(fit['center_ev']), record.center_ev, rtol=0, atol=1e-7)
                        and np.isclose(float(fit['width_mev']), record.width_mev, rtol=0, atol=1e-6)
                        and np.isclose(float(fit['slope']), slope, rtol=1e-8, atol=1e-14)):
                    se = float(fit['slope_se'])
                    if np.isfinite(se) and se >= 0:
                        return se
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            pass
        return None

    def _draw(self):
        self._update_field_preview()
        if self.figure is None:
            from matplotlib.figure import Figure
            from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg
            self.figure = Figure(figsize=(7, 5), dpi=100, layout='constrained')
            self.canvas = ThemeAwareFigureCanvasQTAgg(self.figure)
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
            self.canvas.setMinimumHeight(330)
            self.cw_plot_layout.addWidget(self.canvas, 1)
        self.empty_label.hide()
        self.canvas.show()
        self.figure.clear()
        axis = self.figure.subplots()
        intercepts = []
        reciprocal_points = []
        measured_grids = []
        for index, branch in enumerate(self.branches):
            points = sorted([p for p in self.points if p['branch'] == branch], key=lambda p:p['temperature_k'])
            if not points:
                continue
            t = np.array([p['temperature_k'] for p in points])
            y = np.array([p['slope'] for p in points])
            color = f'C{index}'
            label = 'Increasing' if branch == 'B increasing' else 'Decreasing'
            fit = self.results.get(branch)
            errors = np.array([p['slope_se'] if p['slope_se'] is not None else np.nan for p in points])
            base = fit['background'] if fit else (self.background_spin.value() if self.background_combo.currentData() == 'fixed' else 0.)
            # Do not suggest a known reciprocal background when a free fit failed.
            if fit or self.background_combo.currentData() != 'fit':
                mask = np.abs(y - base) > max(np.max(np.abs(y)) * 1e-12, 1e-15)
                reciprocal = 1/(y[mask]-base)
                axis.errorbar(t[mask], reciprocal, yerr=errors[mask]/(y[mask]-base)**2,
                              fmt='o', color=color, label=label, markersize=5, capsize=3)
                reciprocal_points.append((t[mask], reciprocal, color))
            if fit:
                grid = np.linspace(t.min(), t.max(), 200)
                style = '-' if index == 0 else '--'
                if abs(fit['amplitude']) > 1e-15:
                    axis.plot(grid, (grid-fit['theta_k'])/fit['amplitude'], style, color=color)
                    measured_grids.append((grid, (grid-fit['theta_k'])/fit['amplitude'], style, color))
                    # Boundary/singular solutions have no identified intercept.
                    # Keep their numerical value in the legend, but do not
                    # turn an optimizer limit into a seemingly physical marker.
                    if fit['theta_ci95_k'] is not None and self.method_combo.currentData() != 'inverse':
                        theta = fit['theta_k']
                        extrapolated = np.array([theta, float(t.min())])
                        axis.plot(extrapolated, (extrapolated-theta)/fit['amplitude'],
                                     ':', color=color, linewidth=1.2)
                        marker, = axis.plot([theta], [0.], 'D', color=color, markersize=5)
                        marker.set_gid(f'theta-intercept-{branch}')
                        intercepts.append((theta, label, color))
        axis.set_title('Inverse slope · CW fit', fontsize=11)
        axis.set_xlabel('Temperature (K)')
        axis.set_ylabel('1 / (s − s0) (T/MCD)')
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(fontsize=9, loc='upper right')
        if self.method_combo.currentData() == 'inverse':
            axis.set_title('Inverse slope · linear CW fit', fontsize=11)
            labels = []
            for branch, fit in self.results.items():
                ci = fit['theta_ci95_k']
                short = 'Inc' if branch == 'B increasing' else 'Dec'
                se = fit.get('theta_se_k')
                estimate = (f"{fit['theta_k']:.3g} ± {se:.3g} K (approx. 1σ SE)"
                            if se is not None and np.isfinite(se)
                            else f"{fit['theta_k']:.3g} K; SE unavailable")
                labels.append(f"{short} θCW = {estimate}\n" +
                              (f"95% CI [{ci[0]:.3g}, {ci[1]:.3g}] K" if ci else '95% CI not reliably constrained'))
            annotation = axis.text(.03, .97, '\n'.join(labels), transform=axis.transAxes,
                                   va='top', fontsize=9)
            annotation._use_theme_text = True
            axis.margins(y=.4)
        if intercepts:
            axis.set_title('Reciprocal / extrapolated θCW', fontsize=10)
            axis.axhline(0., color='0.5', linewidth=.8)
            # Make room below zero for labels without allowing confidence
            # limits to squash the measured points into the edge of the plot.
            low, high = axis.get_ylim()
            span = high-low
            axis.set_ylim(min(low, -span*.25), max(high, span*.25))
            for i, (theta, label, color) in enumerate(sorted(intercepts, key=lambda item: item[0])):
                annotation = axis.annotate(
                    f"{'Inc' if label == 'Increasing' else 'Dec'} θCW = {theta:.3g} K",
                    xy=(theta, 0.), xycoords='data',
                    xytext=(.04 + .49*i, .07), textcoords='axes fraction',
                    ha='left', va='bottom', fontsize=8, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, linewidth=.8))
                annotation.set_gid(f'theta-label-{label}')
            # Measured-range zoom preserves readability while the parent
            # axes show the potentially distant temperature-axis intercepts.
            zoom = axis.inset_axes([.08, .57, .43, .32])
            zoom.set_label('CW measured-temperature zoom')
            zoom.set_title('Measured T range', fontsize=7)
            for t, reciprocal, color in reciprocal_points:
                zoom.plot(t, reciprocal, 'o', color=color, markersize=2.5)
            for grid, reciprocal, style, color in measured_grids:
                zoom.plot(grid, reciprocal, style, color=color, linewidth=.8)
            zoom.tick_params(labelsize=6)
            zoom.grid(alpha=.2)
        self.canvas.draw_idle()

    def _update_field_preview(self):
        previous = self.field_record_combo.currentData()
        self.field_record_combo.blockSignals(True)
        self.field_record_combo.clear()
        seen = set()
        for fit in sorted(self.field_fits, key=lambda f:(f['temperature_k'], f['record_id'])):
            if fit['record_id'] not in seen:
                self.field_record_combo.addItem(f"{fit['temperature_k']:g} K · {fit['center_ev']:.6f} eV · {fit['source']}", fit['record_id'])
                seen.add(fit['record_id'])
        self.field_record_combo.setCurrentIndex(max(0, self.field_record_combo.findData(previous)))
        self.field_record_combo.blockSignals(False)
        self._draw_field()

    def _draw_field(self, *_):
        fits = [f for f in self.field_fits if f['record_id'] == self.field_record_combo.currentData()]
        if not fits:
            self.field_summary.setText('No MCD–B refits to display. Enable refitting and include valid curves.')
            if self.field_canvas is not None:
                self.field_figure.clear()
                self.field_canvas.hide()
            return
        if self.field_figure is None:
            from matplotlib.figure import Figure
            from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg
            self.field_figure = Figure(figsize=(7, 5), dpi=100, layout='constrained')
            self.field_canvas = ThemeAwareFigureCanvasQTAgg(self.field_figure)
            self.field_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
            self.field_canvas.setMinimumHeight(330)
            self.field_plot_layout.addWidget(self.field_canvas, 1)
        self.field_canvas.show()
        self.field_figure.clear()
        axes = np.atleast_1d(self.field_figure.subplots(1, len(fits)))
        lines = []
        for index, (fit, axis) in enumerate(zip(fits, axes)):
            samples = fit['samples']
            x = np.array([p['B_T'] for p in samples])
            y = np.array([p['mcd'] for p in samples])
            mask = np.array([p['used'] for p in samples], dtype=bool)
            color = f'C{index}'
            low, high = fit['requested_b_min_t'], fit['requested_b_max_t']
            axis.plot(x, y, 'o', color=color, alpha=.25, ms=3, label='Measured')
            axis.plot(x[mask], y[mask], 'o', color=color, ms=4, label='In B window')
            axis.axvspan(low, high, color=color, alpha=.07)
            if fit['status'] == 'ok':
                grid = np.array([fit['field_min_t'], fit['field_max_t']])
                axis.plot(grid, fit['slope']*grid+fit['intercept'], '-', color=color, label='Linear fit')
                lines.append(f"{fit['branch']}: s={fit['slope']:.5g} ± {fit['slope_se']:.2g} MCD/T (SE); "
                             f"N={fit['n']}; R²={fit['r_squared']:.4f}; RMS={fit['residual_rms']:.2g}; "
                             f"B=[{fit['field_min_t']:.4g}, {fit['field_max_t']:.4g}] T")
            else:
                lines.append(f"{fit['branch']}: {fit['status']}; N={fit['n']}. No slope used in CW fit.")
            margin = (high-low)*.25
            axis.set_xlim(low-margin, high+margin)
            # Set y limits using visible points so distant high-field values
            # do not flatten the low-field diagnostic.
            visible = y[(x >= low-margin) & (x <= high+margin)]
            if visible.size:
                padding = max(float(np.ptp(visible))*.15, abs(float(np.mean(visible)))*.01, 1e-6)
                axis.set_ylim(float(visible.min())-padding, float(visible.max())+padding)
            axis.set_title(f"{fit['temperature_k']:g} K · {fit['branch']}", fontsize=9)
            axis.set_xlabel('B field (T)')
            axis.set_ylabel('Corrected signed-mean MCD')
            axis.grid(alpha=.25)
            axis.legend(fontsize=7)
        self.field_summary.setText('\n'.join(lines))
        self.field_canvas.draw_idle()

    def export_to(self, folder):
        # Flush pending control changes before taking the export snapshot.
        self.refit()
        if not self.results:
            raise ValueError('No valid fit to save.')
        target = Path(folder) / datetime.now().strftime('Curie_Weiss_%Y%m%d_%H%M%S_%f')
        target.mkdir(parents=True, exist_ok=False)
        payload = dict(schema_version=3, series_id=self.series_id,
            fit_method=self.method_combo.currentData(),
            reference='https://doi.org/10.1038/s41565-022-01309-8',
            energy_group=self.group_combo.currentData(), temperature_mode=self.temperature_combo.currentData(),
            temperature_range_k=[self.t_min.value(), self.t_max.value()],
            model='s(T) = s0 + A / (T - theta_CW)', results=self.results,
            warnings=self.warnings, points=self.points,
            slope_source='refit' if self.field_refit_chk.isChecked() else 'saved',
            field_range_t=[self.b_min.value(), self.b_max.value()] if self.field_refit_chk.isChecked() else None,
            field_fits=self.field_fits, field_preview_record_id=self.field_record_combo.currentData(),
            assumptions='Paramagnetic regime; temperature-independent optical conversion. Interaction tendency, not magnetic order.')
        (target/'fit.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        fields = list(dict.fromkeys(k for point in self.points for k in point))
        with (target/'points.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.points)
        (target/'fit.txt').write_text(self.summary.toPlainText() + '\n\n' + payload['assumptions'], encoding='utf-8')
        with self.canvas.publication_context():
            self.figure.savefig(target/'fit.png', dpi=180, bbox_inches='tight')
        if self.field_fits:
            rows = [dict(record_id=fit['record_id'], source=fit['source'], branch=fit['branch'],
                         temperature_k=fit['temperature_k'], **sample)
                    for fit in self.field_fits for sample in fit['samples']]
            if rows:
                with (target/'field_points.csv').open('w', newline='', encoding='utf-8-sig') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            with self.field_canvas.publication_context():
                self.field_figure.savefig(target/'field_fit_preview.png', dpi=180, bbox_inches='tight')
        return target

    def _save(self):
        folder = QFileDialog.getExistingDirectory(self, 'Save Curie–Weiss fit')
        if not folder:
            return
        try:
            target = self.export_to(folder)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Curie–Weiss export', str(exc))
            return
        QMessageBox.information(self, 'Curie–Weiss export', f'Saved fit, points and figure:\n{target}')
