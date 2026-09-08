"""Review and save two sweep segments as an ordinary power table."""

from pathlib import Path
import json

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QLabel, QLineEdit, QMessageBox, QSizePolicy, QVBoxLayout, QDoubleSpinBox, QPushButton, QScrollArea, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from core.power_combine import combine_power_sweeps, save_combined_power_sweep
from core.power_workflow import (COMBINED_FOLDER, is_combined, corrected, estimate_factor,
    acquisition_info, suspicious_rows, fingerprint, suggested_name, estimate_background)


class PowerCombineDialog(QDialog):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.saved_path = None
        self.open_requested = False
        self.original_inputs = None
        self.preview_result = None
        self._background_sources = None
        self._manual_name = False
        self.result = None
        self.inputs = None
        self.setWindowTitle("Combine power sweeps")
        self.resize(1050, 850)
        layout = QVBoxLayout(self)
        note = QLabel("Combine segments of the same channel (KK with KK, or KKp with KKp). "
                      "Review optional intensity correction before saving. Nearby powers remain separate; only exact "
                      "matching powers use the policy below. Energy is aligned within the shared range.")
        note.setWordWrap(True)
        layout.addWidget(note)
        controls = QWidget()
        form = QFormLayout(controls)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMaximumHeight(330)
        layout.addWidget(scroll)
        self.first = QComboBox()
        self.second = QComboBox()
        for combo in (self.first, self.second):
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.sources = controller._power_current_sources()
        for key, source in self.sources.items():
            if is_combined(controller.current_folder, source.file_name):
                continue
            self.first.addItem(source.title, key)
            self.second.addItem(source.title, key)
        self.second.setCurrentIndex(1 if self.second.count() > 1 else 0)
        selected = self.first.findData(controller._power_selected_group_key())
        if selected >= 0:
            self.first.setCurrentIndex(selected)
            if self.second.currentIndex() == selected and self.second.count() > 1:
                self.second.setCurrentIndex((selected + 1) % self.second.count())
        form.addRow("First sweep", self.first)
        form.addRow("Second sweep", self.second)
        self.include_combined = QCheckBox("Include combined sweeps")
        form.addRow(self.include_combined)
        self.policy = QComboBox()
        for title, key in [("Keep first sweep", "first"), ("Keep second sweep", "second"),
                           ("Average matching measurements", "average")]:
            self.policy.addItem(title, key)
        form.addRow("Matching powers", self.policy)
        self.pair = QComboBox()
        form.addRow("Overlap spectra", self.pair)
        self.show_original = QCheckBox("Show original spectra alongside corrected spectra")
        form.addRow(self.show_original)
        self.correction = QComboBox()
        self.correction.addItems(["No correction", "Known correction factor", "Estimate from overlap"])
        form.addRow("Intensity correction", self.correction)
        self.manual_background = QCheckBox("Enter backgrounds manually")
        self.manual_background.setToolTip("Otherwise estimate a constant background in the energy band below. Choose a band with no PL peaks.")
        form.addRow(self.manual_background)
        self.background_band = []
        for label in ("Background band min (eV)", "Background band max (eV)"):
            control = QDoubleSpinBox()
            control.setDecimals(6)
            control.setRange(-10000, 10000)
            control.setKeyboardTracking(False)
            form.addRow(label, control)
            self.background_band.append(control)
        self.backgrounds = []
        self.exclusions = []
        for name in ("First", "Second"):
            background = QDoubleSpinBox()
            background.setRange(-1e12, 1e12)
            background.setDecimals(6)
            background.setKeyboardTracking(False)
            form.addRow(f"{name} background (counts)", background)
            self.backgrounds.append(background)
            exclusion = QLineEdit()
            exclusion.setPlaceholderText("e.g. 1, 8, 26 (spectrum numbers in power order)")
            exclusion.setToolTip("1-based positions in the loaded, power-sorted sweep. The summary lists suspicious spectrum numbers. Raw rows are retained in provenance.")
            form.addRow(f"Exclude {name.lower()} spectra", exclusion)
            self.exclusions.append(exclusion)
        self.factor = QDoubleSpinBox()
        self.factor.setRange(.000000001, 1e9)
        self.factor.setDecimals(9)
        self.factor.setValue(1.)
        self.factor.setKeyboardTracking(False)
        form.addRow("Multiply second sweep by", self.factor)
        self.filename = QLineEdit()
        self.filename.textEdited.connect(lambda *_: setattr(self, "_manual_name", True))
        self.filename.setToolTip("Existing files are preserved; a numbered suffix is added if this name is already taken.")
        output_form = QFormLayout()
        output_form.addRow("Save under Processed as", self.filename)
        layout.addLayout(output_form)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.figure = Figure(figsize=(8, 3), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        self.compatible = QCheckBox("I checked that the sample/channel and acquisition settings are compatible.")
        layout.addWidget(self.compatible)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)
        self.open_button = QPushButton("Open combined sweep")
        self.open_button.hide()
        self.open_button.clicked.connect(self._open_saved)
        layout.addWidget(self.open_button)
        self.compatible.toggled.connect(self._update_save)
        self.first.currentIndexChanged.connect(self._refresh)
        self.second.currentIndexChanged.connect(self._refresh)
        self.policy.currentIndexChanged.connect(self._refresh)
        self.pair.currentIndexChanged.connect(self._draw)
        self.show_original.toggled.connect(self._draw)
        self.include_combined.toggled.connect(self._repopulate)
        self.correction.currentIndexChanged.connect(self._refresh)
        self.manual_background.toggled.connect(self._refresh)
        for control in self.background_band:
            control.valueChanged.connect(self._refresh)
        self.factor.valueChanged.connect(self._refresh)
        for control in self.backgrounds:
            control.valueChanged.connect(self._refresh)
        for control in self.exclusions:
            control.editingFinished.connect(self._refresh)
        self._refresh()

    def _repopulate(self):
        for combo in (self.first, self.second):
            key = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for source_key, source in self.sources.items():
                combined = is_combined(self.controller.current_folder, source.file_name)
                if combined and not self.include_combined.isChecked():
                    continue
                combo.addItem(("[Combined] " if combined else "") + source.title, source_key)
            combo.setCurrentIndex(max(0, combo.findData(key)))
            combo.blockSignals(False)
        self._refresh()

    def _open_saved(self):
        self.open_requested = True
        self.accept()

    def _update_save(self):
        self.buttons.button(QDialogButtonBox.Save).setEnabled(
            self.result is not None and self.compatible.isChecked())

    def _refresh(self):
        self.first.setToolTip(self.first.currentText())
        self.second.setToolTip(self.second.currentText())
        self.result = None
        self.inputs = None
        self.original_inputs = None
        self.preview_result = None
        self.compatible.setChecked(False)
        self.pair.blockSignals(True)
        self.pair.clear()
        try:
            if not self.first.currentData() or not self.second.currentData():
                raise ValueError("Detect at least two power sweeps first.")
            a = self.controller._power_load_group_result(self.first.currentData())
            b = self.controller._power_load_group_result(self.second.currentData())
            self.original_inputs = (a, b)
            source_keys = (self.first.currentData(), self.second.currentData())
            if source_keys != self._background_sources:
                lo = max(a.cube.energy[0], b.cube.energy[0])
                hi = min(a.cube.energy[-1], b.cube.energy[-1])
                for control, value in zip(self.background_band, (hi - .2 * (hi - lo), hi)):
                    control.blockSignals(True)
                    control.setValue(float(value))
                    control.blockSignals(False)
                self._background_sources = source_keys
            mode = self.correction.currentText()
            self.factor.setEnabled(mode == "Known correction factor")
            excludes = []
            for control, result in zip(self.exclusions, (a, b)):
                values = [int(value.strip()) - 1 for value in control.text().split(',') if value.strip()]
                if any(i < 0 or i >= len(result.records) for i in values):
                    raise ValueError("Excluded spectrum numbers must be within the sweep.")
                excludes.append(values)
            automatic = mode != "No correction" and not self.manual_background.isChecked()
            backgrounds = [spin.value() if mode != "No correction" else 0. for spin in self.backgrounds]
            if automatic:
                backgrounds = [estimate_background(corrected(source, excluded=excluded),
                    self.background_band[0].value(), self.background_band[1].value())
                    for source, excluded in zip(self.original_inputs, excludes)]
                for control, value in zip(self.backgrounds, backgrounds):
                    control.blockSignals(True)
                    control.setValue(value)
                    control.blockSignals(False)
            for control in self.backgrounds:
                control.setEnabled(mode != "No correction" and not automatic)
            self.manual_background.setEnabled(mode != "No correction")
            for control in self.background_band:
                control.setEnabled(automatic)
            a = corrected(a, background=backgrounds[0], excluded=excludes[0])
            b = corrected(b, background=backgrounds[1], excluded=excludes[1])
            factor = self.factor.value() if mode == "Known correction factor" else 1.
            diagnostic = ""
            rejected = False
            if mode == "Estimate from overlap":
                factor, quality = estimate_factor(a, b)
                diagnostic = (f"{len(quality['pairs'])} calibration pairs; factor scatter {quality['relative_scatter']:.1%}; "
                              f"maximum spectral mismatch {quality['max_relative_residual']:.1%}. "
                              + ("PROVISIONAL: insufficient pairs or inconsistent shapes. " if quality['provisional'] else ""))
                self.factor.blockSignals(True)
                self.factor.setValue(factor)
                self.factor.blockSignals(False)
                rejected = not quality['accepted']
                if rejected:
                    diagnostic = (f"CORRECTION REJECTED: proposed multiplier {factor:.6g} fails the 10% scatter/mismatch limit. "
                                  + diagnostic + "Showing background-subtracted inputs WITHOUT scaling. Saving is disabled. ")
                    factor = 1.
            b = corrected(b, factor=factor)
            self.combine_settings = {"mode": mode, "backgrounds": backgrounds, "factor_second": factor,
                "excluded_spectra": excludes, "duplicate_policy": self.policy.currentData(), "diagnostic": diagnostic,
                "acquisition": [acquisition_info(self.controller.current_folder, source) for source in self.original_inputs]}
            self.combine_settings['background_method'] = 'energy band median' if automatic else 'manual'
            self.combine_settings['background_band_eV'] = [c.value() for c in self.background_band] if automatic else None
            self.request_signature = fingerprint(self.original_inputs, self.combine_settings)
            self.result = combine_power_sweeps(a, b, duplicate_policy=self.policy.currentData())
            self.preview_result = self.result
            if rejected:
                self.result = None
            self.inputs = (a, b)
            if not self._manual_name:
                self.filename.setText(suggested_name(*self.original_inputs))
            p1, p2 = a.cube.gate, b.cube.gate
            low, high = max(min(p1), min(p2)), min(max(p1), max(p2))
            for i, power in enumerate(p1):
                for j in np.flatnonzero(abs(p2 - power) <= .02 * np.maximum(np.maximum(abs(power), abs(p2)), 1e-30)):
                    self.pair.addItem(f"First: {power:.9g} uW / Second: {p2[j]:.9g} uW", (i, j))
            overlap = (f"Range intersection: {low:.6g}–{high:.6g} uW; {self.pair.count()} pairs within 2%." if low <= high
                       else "No power overlap: compatibility cannot be checked from overlapping spectra.")
            duplicates = len(np.intersect1d(p1, p2))
            e = self.preview_result.cube.energy
            info = []
            for name, source in zip(("First", "Second"), self.original_inputs):
                flags = suspicious_rows(self.controller.current_folder, source)
                metadata = acquisition_info(self.controller.current_folder, source)
                details = '; '.join(f"exposure {m['exposure_ms'] if m['exposure_ms'] is not None else 'unknown'} ms, "
                    f"frames {m['frames'] if m['frames'] is not None else 'unknown'}, gain {m['gain']}" for m in metadata)
                info.append(f"{name}: {len(source.records)} points, {min(source.cube.gate):.6g}–{max(source.cube.gate):.6g} uW; "
                    f"{details}. "
                    f"Stage discrepancy >2: spectrum numbers {[i + 1 for i in flags] or 'none detected'}.")
            self.summary.setText('\n'.join(info) + '\n' + diagnostic + f"Second intensity × {factor:.6g}. "
                                 f"{len(self.preview_result.records)} preview points; {duplicates} exact matching powers. "
                                 f"{overlap} Shared energy: {e[0]:.6g}–{e[-1]:.6g} eV. "
                                 "Solid spectra show corrected inputs; enable Show original spectra for dashed raw inputs. "
                                 "Check that the background band contains no PL peaks. Factor includes exposure differences; check saturation before accepting.")
        except Exception as exc:
            self.summary.setText(str(exc))
        finally:
            self.pair.blockSignals(False)
        self._update_save()
        self._draw()

    def _draw(self):
        self.figure.clear()
        if self.inputs is not None:
            trace, spectra, heatmap = self.figure.subplots(1, 3)
            pair = self.pair.currentData()
            for index, (label, result) in enumerate(zip(("First", "Second"), self.inputs)):
                cube = result.cube
                trace.plot(cube.gate, np.nanmean(cube.Z, axis=1), "o-", label=label)
                if pair is not None:
                    row = pair[index]
                    spectra.plot(cube.energy, cube.Z[row],
                                 color=f"C{index}", label=f"{label}: {cube.gate[row]:.6g} uW")
                    original = self.original_inputs[index].cube
                    raw_row = int(np.argmin(abs(original.gate - cube.gate[row])))
                    if self.show_original.isChecked():
                        spectra.plot(original.energy, original.Z[raw_row], '--', color=f"C{index}", alpha=.5, label=f"{label} original")
            trace.set(xlabel="Power (uW)", ylabel="Mean spectral intensity (a.u.)")
            trace.legend()
            spectra.set(xlabel="Energy (eV)", ylabel="Intensity (a.u.)")
            if pair is not None:
                spectra.legend(fontsize=7, loc="best")
            else:
                spectra.text(0.5, 0.5, "No overlap spectra available", ha="center",
                             transform=spectra.transAxes)
            from core.plotting import HeatmapParams, plot_heatmap
            cube = self.preview_result.cube
            params = HeatmapParams("Combined preview", "Energy (eV)", "Power (uW)", "Intensity",
                float(np.nanmin(cube.Z)), float(np.nanmax(cube.Z)),
                (float(cube.energy[0]), float(cube.energy[-1])),
                (float(min(cube.gate)), float(max(cube.gate))))
            plot_heatmap(heatmap, cube, params)
        self.canvas.draw_idle()

    def _save(self):
        if self.result is None or not self.compatible.isChecked():
            return
        name = self.filename.text().strip()
        if not name or Path(name).name != name or not name.lower().endswith(".csv"):
            QMessageBox.warning(self, "Invalid filename", "Enter a CSV filename without a folder path.")
            return
        folder = Path(self.controller.current_folder) / COMBINED_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        for manifest in folder.glob("*.combine.json"):
            try:
                previous = json.loads(manifest.read_text(encoding="utf-8"))
                existing = manifest.with_name(manifest.name.removesuffix(".combine.json") + ".csv")
                if previous.get("signature") == self.request_signature and existing.is_file():
                    prompt = QMessageBox(self)
                    prompt.setWindowTitle("Saved combination exists")
                    prompt.setText("These unchanged sweeps and settings already have a saved combination.")
                    open_existing = prompt.addButton("Open existing", QMessageBox.AcceptRole)
                    prompt.addButton("Create another", QMessageBox.ActionRole)
                    cancel = prompt.addButton(QMessageBox.Cancel)
                    prompt.exec()
                    if prompt.clickedButton() == cancel:
                        return
                    if prompt.clickedButton() == open_existing:
                        self.saved_path = existing
                        self._open_saved()
                        return
                    break
            except (ValueError, OSError):
                continue
        path = folder / name
        try:
            path = save_combined_power_sweep(path, self.result, unique=True)
            path.with_suffix('.combine.json').write_text(json.dumps({"signature": self.request_signature,
                "settings": self.combine_settings,
                "sources": [[r.file_name for r in source.records] for source in self.original_inputs]}, indent=2), encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Cannot save combined sweep", str(exc))
            return
        self.saved_path = path
        self._after_save()

    def _after_save(self):
        path = self.saved_path
        self.summary.setText(f"Combined sweep saved under Processed:\n{path.name}")
        self.buttons.button(QDialogButtonBox.Save).setEnabled(False)
        self.buttons.button(QDialogButtonBox.Cancel).setText("Close")
        for control in (self.first, self.second, self.policy, self.include_combined, self.correction,
                        self.factor, self.filename, self.compatible, self.manual_background,
                        *self.background_band, *self.backgrounds, *self.exclusions):
            control.setEnabled(False)
        self.open_button.show()
