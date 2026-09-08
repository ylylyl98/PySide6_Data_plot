"""Choose spectral peaks and independently adjustable power-law intervals."""
import numpy as np
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QTableWidget, QHeaderView, QDoubleSpinBox, QDialogButtonBox)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.widgets import SpanSelector
from core.plotting import plain_log_ticks
from core.power_law import fit_power_law, suggest_power_range, local_slopes
from core.power_law_analysis import peak_series, evaluate_ranges, draw_power_laws


class PowerLawDialog(QDialog):
    def __init__(self, owner, results, records, ranges=()):
        super().__init__(owner)
        self.setWindowTitle('Power-law fit — integrated peak intensity')
        self.resize(1050, 780)
        self.results, self.records = results, records
        self.series = peak_series(results, records)
        self.ranges, self.fits = [], []
        self.suggested = {(r['channel'], r['peak']) for r in ranges if r.get('suggested')}
        self._updating = True
        layout = QVBoxLayout(self)
        info = QLabel('Select one or more peaks. Enter power limits or drag on the plot. '
            'Dragging adjusts the highlighted row, or all selected peaks when ranges are shared. '
            'Only existing fitted integrated intensities are used.')
        info.setWordWrap(True)
        layout.addWidget(info)
        row = QHBoxLayout()
        self.shared = QCheckBox('Use a shared fitting range')
        self.shared.setChecked(not ranges or len({(r['lower'], r['upper']) for r in ranges}) <= 1)
        self.local = QCheckBox('Show local slope')
        self.suggest = QPushButton('Suggest ranges near alpha = 1')
        for control in (self.shared, self.local, self.suggest): row.addWidget(control)
        layout.addLayout(row)
        self.table = QTableWidget(len(self.series), 4)
        self.table.setHorizontalHeaderLabels(['Fit peak', 'Min power (uW)', 'Max power (uW)', 'Result (alpha +/- 1 SE)'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMaximumHeight(230)
        self.table.setFixedHeight(min(230, 30 + 48 * len(self.series)))
        self.table.verticalHeader().setDefaultSectionSize(48)
        layout.addWidget(self.table)
        self.controls = []
        saved = {(r['channel'], r['peak']): r for r in ranges}
        all_power = np.concatenate([s['power'] for s in self.series]) if self.series else np.array([])
        positive = all_power[np.isfinite(all_power) & (all_power > 0)]
        defaults = (float(min(positive)), float(max(positive))) if len(positive) else (1., 10.)
        for i, source in enumerate(self.series):
            state = saved.get((source['channel'], source['peak']))
            check = QCheckBox(source['name'])
            check.setChecked(state is not None if ranges else source['peak'] == 1)
            spins = []
            for value in ((state['lower'], state['upper']) if state else defaults):
                spin = QDoubleSpinBox()
                spin.setDecimals(10)
                spin.setRange(1e-10, 1e15)
                spin.setValue(value)
                spin.setKeyboardTracking(False)
                spin.valueChanged.connect(lambda _, index=i: self.range_changed(index))
                spins.append(spin)
            label = QLabel()
            label.setWordWrap(True)
            self.controls.append((check, *spins, label))
            for col, widget in enumerate(self.controls[-1]): self.table.setCellWidget(i, col, widget)
            check.toggled.connect(lambda _, index=i: self.range_changed(index))
        self.table.setCurrentCell(0, 0)
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        self.message = QLabel('Suggestions are exploratory: selecting for alpha near 1 is not independent evidence of linearity. '
            'Defaults: at least 6 points, 0.5 decade, and slopes within 0.15 of 1. '
            'Uncertainty is regression standard error; gain-calibration uncertainty is not included.')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.suggest.clicked.connect(self.suggest_ranges)
        self.local.toggled.connect(self.refresh)
        self.shared.toggled.connect(lambda _: self.range_changed(max(0, self.table.currentRow())))
        self.table.currentCellChanged.connect(self.refresh)
        self._updating = False
        self.refresh()

    def range_changed(self, index):
        if self._updating: return
        if self.shared.isChecked():
            self.suggested.clear()
        else:
            source = self.series[index]
            self.suggested.discard((source['channel'], source['peak']))
        self._updating = True
        if self.shared.isChecked():
            _, lo, hi, _ = self.controls[index]
            for check, lower, upper, _ in self.controls:
                if check.isChecked():
                    lower.setValue(lo.value()); upper.setValue(hi.value())
        self._updating = False
        self.refresh()

    def selected_ranges(self):
        return [dict(channel=s['channel'], peak=s['peak'], lower=lo.value(), upper=hi.value(),
                     suggested=(s['channel'], s['peak']) in self.suggested)
            for s, (check, lo, hi, _) in zip(self.series, self.controls) if check.isChecked()]

    def suggest_ranges(self):
        self._updating = True
        failures = []
        suggestions = []
        for i, (source, (check, lo, hi, _)) in enumerate(zip(self.series, self.controls)):
            if not check.isChecked(): continue
            try:
                fit = suggest_power_range(source['power'], source['intensity'], valid=source['valid'])
                lo.setValue(fit['lower']); hi.setValue(fit['upper'])
                suggestions.append(i)
                self.suggested.add((source['channel'], source['peak']))
            except ValueError as exc:
                failures.append(f"{source['name']}: {exc}")
        if sum(c[0].isChecked() for c in self.controls) > 1: self.shared.setChecked(False)
        self._updating = False
        self.refresh()
        for fit in self.fits:
            if any(self.series[i]['channel'] == fit['channel'] and self.series[i]['peak'] == fit['peak'] for i in suggestions):
                fit['suggested'] = True
        self.message.setText((' '.join(failures) + ' Previous limits retained for these peaks. ') if failures else
            'Suggested ranges are shown for review; adjust them before Apply if needed. '
            'Selection for alpha near 1 is exploratory, not independent evidence of linearity.')

    def span_selected(self, lo, hi):
        index = max(0, self.table.currentRow())
        self._updating = True
        self.controls[index][1].setValue(lo)
        self.controls[index][2].setValue(hi)
        self._updating = False
        self.range_changed(index)

    def refresh(self, *_):
        if self._updating: return
        self.ranges = self.selected_ranges()
        self.fits = []
        errors = False
        for source, (check, lo, hi, label) in zip(self.series, self.controls):
            label.setText('')
            if not check.isChecked(): continue
            try:
                fit = evaluate_ranges(self.results, self.records, [dict(channel=source['channel'], peak=source['peak'], lower=lo.value(), upper=hi.value(),
                    suggested=(source['channel'], source['peak']) in self.suggested)])[0]
                self.fits.append(fit)
                label.setText(f"{fit['alpha']:.3f} +/- {fit['alpha_error']:.3f}; n={fit['n']}; omitted (incl. outside range)={fit['excluded']}"
                    + ('; crosses source boundary' if fit['crosses_sources'] else ''))
            except ValueError as exc:
                label.setText(str(exc)); errors = True
        self.buttons.button(QDialogButtonBox.Apply).setEnabled(not errors)
        if hasattr(self, 'selector'): self.selector.disconnect_events()
        self.figure.clear()
        axes = self.figure.subplots(2 if self.local.isChecked() else 1, 1, squeeze=False).ravel()
        axis = axes[0]
        axis.set(xscale='log', yscale='log', xlabel='Power (uW)', ylabel='Integrated PL (a.u. eV)')
        plain_log_ticks(axis.xaxis); plain_log_ticks(axis.yaxis)
        for i, (source, (check, _, _, _)) in enumerate(zip(self.series, self.controls)):
            if not check.isChecked(): continue
            valid = source['valid'] & np.isfinite(source['power']) & np.isfinite(source['intensity']) & (source['power'] > 0) & (source['intensity'] > 0)
            axis.plot(source['power'][valid], source['intensity'][valid], 'o-', color=f'C{i % 10}', label=source['name'])
            for boundary in source['boundaries']: axis.axvline(boundary, color='gray', alpha=.25, ls=':')
            if len(axes) > 1:
                x, slopes = local_slopes(source['power'], source['intensity'], valid=source['valid'])
                axes[1].plot(x, slopes, '.-', color=f'C{i % 10}', label=source['name'])
        draw_power_laws(axis, self.fits)
        if len(axes) > 1:
            axes[1].set(xscale='log', xlabel='Power (uW)', ylabel='Local log slope (7 points)')
            plain_log_ticks(axes[1].xaxis)
            axes[1].axhline(1, color='gray', ls='--')
        self.selector = SpanSelector(axis, self.span_selected, 'horizontal', useblit=False,
            interactive=True, props=dict(alpha=.15, facecolor='gray'))
        if self.controls:
            _, lo, hi, _ = self.controls[max(0, self.table.currentRow())]
            if lo.value() < hi.value(): self.selector.extents = (lo.value(), hi.value())
        self.canvas.draw_idle()
