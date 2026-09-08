"""Review any number of explicitly selected power sweeps before Save and plot."""
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLabel, QComboBox,
    QDoubleSpinBox, QLineEdit, QCheckBox, QTableWidget, QHeaderView, QDialogButtonBox, QPushButton, QSizePolicy)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from core.power_combine import combine_many_power_sweeps
from core.power_workflow import corrected, estimate_background, estimate_factor, fingerprint, acquisition_info, suspicious_rows, suggested_name
from core.plotting import HeatmapParams, plot_heatmap
from ui_qt.power_combine_dialog import PowerCombineDialog


class PowerCombineReview(PowerCombineDialog):
    def __init__(self, parent, controller, keys, draft=None):
        QDialog.__init__(self, parent)
        self.controller, self.keys = controller, tuple(keys)
        self.saved_path = None
        self.open_requested = self.back_requested = False
        self.result = self.preview_result = None
        self.original_inputs = tuple(controller._power_load_group_result(k) for k in keys)
        self._updating = True
        self.setWindowTitle("Review power combination")
        self.resize(1100, 850)
        layout = QVBoxLayout(self)
        note = QLabel(f"{len(keys)} sweeps selected. Combine ranges of the same sample/channel. "
                      "Each additional sweep is corrected directly against the reference; measured powers are preserved.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.settings_form = form
        layout.addLayout(form)
        self.reference = QComboBox()
        self.comparison = QComboBox()
        for combo in (self.reference, self.comparison):
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        sources = controller._power_current_sources()
        for k in keys:
            for combo in (self.reference, self.comparison):
                combo.addItem(sources[k].file_name or sources[k].title, k)
        self.comparison.setCurrentIndex(1)
        form.addRow("Reference sweep", self.reference)
        self.automatic = QCheckBox("Choose reference and corrections automatically")
        self.automatic.setChecked(True)
        form.addRow(self.automatic)
        self.policy = QComboBox()
        for title, value in (("Keep reference / first selected", "first"), ("Keep last selected", "second"), ("Average all matching measurements", "average")):
            self.policy.addItem(title, value)
        form.addRow("Exact matching powers", self.policy)
        self.manual_background = QCheckBox("Enter backgrounds manually (otherwise use the signal-free energy band)")
        form.addRow(self.manual_background)
        lo = max(r.cube.energy[0] for r in self.original_inputs)
        hi = min(r.cube.energy[-1] for r in self.original_inputs)
        self.background_band = []
        for label, value in (("Background band min (eV)", hi - .2 * (hi - lo)), ("Background band max (eV)", hi)):
            control = self.spin(float(value), -10000, 10000, 6)
            self.background_band.append(control)
            form.addRow(label, control)
        self.table = QTableWidget(len(keys), 5)
        self.table.setHorizontalHeaderLabels(["Sweep / power range", "Correction", "Background", "Multiplier", "Exclude spectra (1-based)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMinimumHeight(145)
        self.controls = {}
        for row, (key, result) in enumerate(zip(keys, self.original_inputs)):
            name = Path(result.records[0].file_name).name
            short = name if len(name) <= 35 else name[:14] + '…' + name[-18:]
            label = QLabel(f"{short}\n{min(result.cube.gate):.6g}–{max(result.cube.gate):.6g} uW\n{len(result.records)} points")
            label.setWordWrap(True)
            label.setToolTip(name)
            mode = QComboBox()
            mode.addItems(["No scaling", "Known factor", "Estimate from overlap"])
            background, factor = self.spin(0., -1e12, 1e12), self.spin(1., 1e-9, 1e9, 9)
            excluded = QLineEdit()
            excluded.setPlaceholderText("e.g. 1, 8")
            self.controls[key] = (mode, background, factor, excluded)
            for col, widget in enumerate((label, mode, background, factor, excluded)):
                self.table.setCellWidget(row, col, widget)
            mode.currentIndexChanged.connect(self.refresh)
            background.valueChanged.connect(self.refresh)
            factor.valueChanged.connect(self.refresh)
            excluded.editingFinished.connect(self.refresh)
        self.table.resizeRowsToContents()
        layout.addWidget(self.table)
        self.advanced = QCheckBox("Advanced settings — backgrounds, exclusions and matching powers")
        layout.addWidget(self.advanced)
        preview_form = QFormLayout()
        layout.addLayout(preview_form)
        preview_form.addRow("Compare with reference", self.comparison)
        if len(keys) == 2:
            self.comparison.hide()
            preview_form.labelForField(self.comparison).hide()
        self.pair = QComboBox()
        preview_form.addRow("Nearby measured powers (within 2%)", self.pair)
        self.show_original = QCheckBox("Show original spectra")
        preview_form.addRow(self.show_original)
        self.figure = Figure(figsize=(9, 3), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.details = QLabel()
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.filename = QLineEdit(suggested_name(*self.original_inputs[:2]))
        if len(keys) > 2:
            self.filename.setText(self.filename.text().replace('__combined.csv', f'__combined_{len(keys)}sweeps.csv'))
        preview_form.addRow("Save under Processed as", self.filename)
        self.compatible = QCheckBox("I reviewed compatibility, correction quality, backgrounds and flagged measurements.")
        layout.addWidget(self.compatible)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Save).setText("Save and plot")
        self.back_button = QPushButton("Back to selection")
        self.buttons.addButton(self.back_button, QDialogButtonBox.ActionRole)
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)
        self.back_button.clicked.connect(self.go_back)
        self.compatible.toggled.connect(self._update_save)
        self.reference.currentIndexChanged.connect(self.reference_changed)
        self.policy.currentIndexChanged.connect(self.refresh)
        self.manual_background.toggled.connect(self.refresh)
        for control in self.background_band:
            control.valueChanged.connect(self.refresh)
        self.comparison.currentIndexChanged.connect(self.update_pairs)
        self.pair.currentIndexChanged.connect(self.draw)
        self.show_original.toggled.connect(self.draw)
        self.advanced.toggled.connect(self.show_advanced)
        self.automatic.toggled.connect(self.refresh)
        if draft:
            self.restore(draft)
        self._updating = False
        self.show_advanced(self.advanced.isChecked())
        self.refresh()

    def show_advanced(self, visible):
        for control in (self.reference, self.automatic, self.policy, self.manual_background, *self.background_band):
            control.setVisible(visible)
            label = self.settings_form.labelForField(control)
            if label is not None:
                label.setVisible(visible)
        self.table.setColumnHidden(2, not visible)
        self.table.setColumnHidden(4, not visible)
        self.table.setColumnHidden(1, not visible)
        self.details.setVisible(visible)
        self.table.resizeRowsToContents()

    @staticmethod
    def spin(value, lo, hi, decimals=6):
        control = QDoubleSpinBox()
        control.setRange(lo, hi)
        control.setDecimals(decimals)
        control.setValue(value)
        control.setKeyboardTracking(False)
        return control

    def snapshot(self):
        return dict(reference=self.reference.currentData(), policy=self.policy.currentIndex(),
            automatic=self.automatic.isChecked(),
            advanced=self.advanced.isChecked(),
            comparison=self.comparison.currentData(), show_original=self.show_original.isChecked(),
            manual=self.manual_background.isChecked(), band=[v.value() for v in self.background_band],
            filename=self.filename.text(), rows={k: [m.currentIndex(), b.value(), f.value(), e.text()]
                for k, (m, b, f, e) in self.controls.items()})

    def restore(self, state):
        self.automatic.setChecked(state.get('automatic', True))
        idx = self.reference.findData(state['reference'])
        if idx >= 0:
            self.reference.setCurrentIndex(idx)
        self.policy.setCurrentIndex(state['policy'])
        self.manual_background.setChecked(state['manual'])
        self.advanced.setChecked(state.get('advanced', False))
        self.show_original.setChecked(state.get('show_original', False))
        comparison = self.comparison.findData(state.get('comparison'))
        if comparison >= 0:
            self.comparison.setCurrentIndex(comparison)
        self.filename.setText(state['filename'])
        for c, v in zip(self.background_band, state['band']): c.setValue(v)
        for key, values in state['rows'].items():
            if key in self.controls:
                m, b, f, e = self.controls[key]
                m.setCurrentIndex(values[0]); b.setValue(values[1]); f.setValue(values[2]); e.setText(values[3])

    def go_back(self):
        self.back_requested = True
        self.reject()

    def _after_save(self):
        self._open_saved()

    def reference_changed(self):
        if self._updating: return
        self._updating = True
        for _, _, factor, _ in self.controls.values(): factor.setValue(1.)
        if len(self.keys) == 2:
            self.comparison.setCurrentIndex(1 - self.reference.currentIndex())
        self._updating = False
        self.refresh()

    def refresh(self, *_):
        if self._updating: return
        self._updating = True
        self.result = self.preview_result = None
        self.compatible.setChecked(False)
        self.inputs = ()
        try:
            if self.automatic.isChecked():
                self.choose_automatic_reference()
            ref = self.reference.currentIndex()
            self.reference.setEnabled(not self.automatic.isChecked())
            active = any(m.currentIndex() != 0 for k, (m, _, _, _) in self.controls.items() if k != self.keys[ref])
            prepared, rows, notes, acquisitions = [], [], [], []
            flagged_count = 0
            for i, (key, source) in enumerate(zip(self.keys, self.original_inputs)):
                mode, bg, factor, exclude = self.controls[key]
                mode.setEnabled(i != ref and not self.automatic.isChecked())
                factor.setEnabled(i != ref and mode.currentIndex() == 1)
                background_used = active
                bg.setEnabled(background_used and self.manual_background.isChecked())
                excluded = [int(v.strip()) - 1 for v in exclude.text().split(',') if v.strip()]
                if any(v < 0 or v >= len(source.records) for v in excluded): raise ValueError(f"Sweep {i + 1}: excluded spectrum number is outside the sweep.")
                kept = corrected(source, excluded=excluded)
                background = bg.value() if background_used else 0.
                if background_used and not self.manual_background.isChecked():
                    background = estimate_background(kept, *[c.value() for c in self.background_band])
                    bg.setValue(background)
                prepared.append(corrected(kept, background=background))
                rows.append(dict(key=key, background=background, excluded=excluded, factor=1., mode=mode.currentText()))
                flags = suspicious_rows(self.controller.current_folder, source)
                flagged_count += len([i for i in flags if i not in excluded])
                metadata = acquisition_info(self.controller.current_folder, source)
                acquisitions.append(metadata)
                notes.append(f"Sweep {i + 1}: " + '; '.join(f"{m['exposure_ms']} ms, gain {m['gain']}" for m in metadata)
                             + (f"; stage flags { [v + 1 for v in flags] }" if flags else ''))
            reject = False
            output = list(prepared)
            for i, key in enumerate(self.keys):
                mode, _, control, _ = self.controls[key]
                factor = 1.
                if i != ref and mode.currentIndex() == 1: factor = control.value()
                if i != ref and mode.currentIndex() == 2:
                    try:
                        factor, quality = estimate_factor(prepared[ref], prepared[i])
                    except ValueError as exc:
                        reject = True
                        rows[i]['error'] = f"Sweep {i + 1}: {exc}"
                        notes.append(rows[i]['error'])
                        control.setValue(1.)
                        continue
                    rows[i]['quality'] = quality
                    if not quality['accepted']:
                        reject = True
                        notes.append(f"Sweep {i + 1}: correction REJECTED; not applied.")
                        factor = 1.
                    notes.append(f"Sweep {i + 1}: {len(quality['pairs'])} calibration pairs, mismatch {quality['max_relative_residual']:.1%}, scatter {quality['relative_scatter']:.1%}" + ('; provisional' if quality['provisional'] else ''))
                control.setValue(factor)
                rows[i]['factor'] = factor
                output[i] = corrected(prepared[i], factor=factor)
            order = [ref] + [i for i in range(len(output)) if i != ref]
            self.preview_result = combine_many_power_sweeps([output[i] for i in order], duplicate_policy=self.policy.currentData())
            self.inputs = tuple(output)
            self.combine_settings = dict(reference=self.keys[ref], rows=rows, duplicate_policy=self.policy.currentData(),
                acquisition=acquisitions,
                backgrounds=[r['background'] for r in rows], factor_second=rows[1]['factor'],
                background_method='manual' if self.manual_background.isChecked() else 'energy band median',
                background_band_eV=[c.value() for c in self.background_band])
            self.request_signature = fingerprint(self.original_inputs, self.combine_settings)
            if not reject: self.result = self.preview_result
            notes.append(f"{len(self.preview_result.records)} combined points. When scaling is enabled, backgrounds are removed from every sweep. Multipliers include exposure differences.")
            status = f"{'Automatic reference' if self.automatic.isChecked() else 'Reference'}: sweep {ref + 1}. {len(self.keys)} sweeps → {len(self.preview_result.records)} combined points."
            if flagged_count:
                status += f" {flagged_count} measurements have stage discrepancies; review exclusions in Advanced settings."
            if active:
                status += f" Background band: {self.background_band[0].value():.4f}–{self.background_band[1].value():.4f} eV (adjust in Advanced)."
            for i, row in enumerate(rows):
                if 'error' in row:
                    status += ' ' + row['error'] + ' Correction not applied; saving is blocked.'
                if 'quality' in row:
                    q = row['quality']
                    status += f" Sweep {i + 1}: {'REJECTED' if not q['accepted'] else 'provisional' if q['provisional'] else 'estimated'}, mismatch {q['max_relative_residual']:.1%}."
            self.summary.setText(status)
            self.details.setText('\n'.join(notes))
        except (ValueError, OSError) as exc:
            self.summary.setText(str(exc))
            self.details.setText('')
        finally:
            self._updating = False
        self._update_save()
        self.update_pairs()

    def choose_automatic_reference(self):
        """Prefer a reference with valid direct calibration to every other sweep."""
        prepared = []
        for key, source in zip(self.keys, self.original_inputs):
            _, bg, _, exclude = self.controls[key]
            excluded = [int(v.strip()) - 1 for v in exclude.text().split(',') if v.strip()]
            if any(v < 0 or v >= len(source.records) for v in excluded):
                raise ValueError('Excluded spectrum number is outside the sweep.')
            kept = corrected(source, excluded=excluded)
            background = bg.value() if self.manual_background.isChecked() else estimate_background(
                kept, *[c.value() for c in self.background_band])
            prepared.append(corrected(kept, background=background))
        scores = []
        for i, source in enumerate(prepared):
            accepted = supported = 0
            for j, other in enumerate(prepared):
                if i == j:
                    continue
                try:
                    _, quality = estimate_factor(source, other)
                    supported += 1
                    accepted += int(quality['accepted'])
                except ValueError:
                    pass
            scores.append((accepted, supported, float(np.ptp(source.cube.gate)), -i))
        ref = max(range(len(scores)), key=scores.__getitem__)
        self.reference.setCurrentIndex(ref)
        if self.comparison.currentIndex() == ref:
            self.comparison.setCurrentIndex(next(i for i in range(len(self.keys)) if i != ref))
        for i, key in enumerate(self.keys):
            self.controls[key][0].setCurrentIndex(0 if i == ref else 2)

    def update_pairs(self, *_):
        if self._updating: return
        self.pair.blockSignals(True)
        self.pair.clear()
        ref, other = self.reference.currentIndex(), self.comparison.currentIndex()
        if self.inputs and ref != other:
            a, b = self.inputs[ref].cube, self.inputs[other].cube
            for i, p in enumerate(a.gate):
                for j in np.flatnonzero(abs(b.gate - p) <= .02 * np.maximum(np.maximum(abs(p), abs(b.gate)), 1e-30)):
                    self.pair.addItem(f"Reference {p:.6g} / sweep {other + 1} {b.gate[j]:.6g} uW", (i, int(j)))
        self.pair.blockSignals(False)
        self.draw()

    def draw(self, *_):
        if self._updating: return
        self.figure.clear()
        if self.inputs:
            trend, spectrum, heat = self.figure.subplots(1, 3)
            for i, source in enumerate(self.inputs):
                trend.plot(source.cube.gate, np.nanmean(source.cube.Z, axis=1), '.-', label=f"Sweep {i + 1}")
            trend.set(xlabel='Power (uW)', ylabel='Mean intensity'); trend.legend(fontsize=7)
            pair = self.pair.currentData()
            if pair:
                for index, row in zip((self.reference.currentIndex(), self.comparison.currentIndex()), pair):
                    c = self.inputs[index].cube
                    spectrum.plot(c.energy, c.Z[row], color=f'C{index % 10}', label=f'Sweep {index + 1}')
                    if self.show_original.isChecked():
                        raw = self.original_inputs[index].cube
                        j = int(np.argmin(abs(raw.gate - c.gate[row])))
                        spectrum.plot(raw.energy, raw.Z[j], '--', color=f'C{index % 10}', alpha=.5)
                spectrum.legend(fontsize=7)
            else: spectrum.text(.5, .5, 'No nearby power pair', ha='center', transform=spectrum.transAxes)
            spectrum.set(xlabel='Energy (eV)', ylabel='PL intensity')
            c = self.preview_result.cube
            title = 'Combined preview' if self.result is not None else 'Preview — correction incomplete'
            plot_heatmap(heat, c, HeatmapParams(title, 'Energy (eV)', 'Power (uW)', 'Intensity',
                float(np.nanmin(c.Z)), float(np.nanmax(c.Z)), (c.energy[0], c.energy[-1]), (min(c.gate), max(c.gate))))
        self.canvas.draw_idle()
