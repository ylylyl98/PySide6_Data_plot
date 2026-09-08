"""Preview a full-spectrum manual refit for just one measured power row."""

import numpy as np
from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QHeaderView, QLabel, QPushButton, QTableWidget, QVBoxLayout,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from core.loader import DataCube
from core.power_multi_peaks import MultiPeakSettings, PeakSeed, center_bounds
from ui_qt.power_peak_controller import _FitWorker


class PowerRefitDialog(QDialog):
    def __init__(self, owner, cubes, results, settings, power):
        super().__init__(owner)
        self.owner, self.cubes, self.results = owner, cubes, results
        self.global_settings, self.power = settings, power
        self.worker = None
        self.close_after_fit = False
        self.accepted_fit = None
        self.trial_fit = None
        self.setWindowTitle("Manual refit — selected power only")
        self.resize(900, 760)
        layout = QVBoxLayout(self)
        info = QLabel("Adjust guesses and center bounds, then Refit to preview. Apply replaces only this channel/power. "
                      "Peak numbers remain the same as the sweep; change peak count in the main Peak Analysis table.")
        info.setWordWrap(True)
        layout.addWidget(info)
        form = QFormLayout()
        layout.addLayout(form)
        self.channel = QComboBox()
        self.channel.addItems(list(cubes))
        self.model = QComboBox()
        self.model.addItems(["Lorentzian", "Gaussian"])
        self.power_label = QLabel()
        form.addRow("Channel", self.channel)
        form.addRow("Actual selected power", self.power_label)
        form.addRow("Model", self.model)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Center (eV)", "FWHM (meV)", "Height guess", "Center min (eV)", "Center max (eV)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(190)
        layout.addWidget(self.table)
        self.figure = Figure(figsize=(8, 4), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("button_press_event", self._click)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.refit = QPushButton("Refit selected spectrum")
        layout.addWidget(self.refit)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        layout.addWidget(self.buttons)
        self.apply_button = self.buttons.button(QDialogButtonBox.Apply)
        self.apply_button.clicked.connect(self._apply)
        self.buttons.rejected.connect(self.reject)
        self.refit.clicked.connect(self._run)
        self.channel.currentTextChanged.connect(self._load)
        self.model.currentTextChanged.connect(self._changed)
        self._load()

    def _load(self, *_):
        label = self.channel.currentText()
        cube = self.cubes[label]
        self.index = int(np.argmin(abs(cube.gate - self.power)))
        self.original_fit = self.results[label][self.index]
        self.power_label.setText(f"{float(cube.gate[self.index]):.9g} uW")
        settings = self.original_fit.settings
        bounds = center_bounds(settings, cube.energy)
        self.model.blockSignals(True)
        self.model.setCurrentText(settings.model)
        self.model.blockSignals(False)
        self.table.setRowCount(len(settings.peaks))
        self.table.setVerticalHeaderLabels([f"Peak {i + 1}" for i in range(len(settings.peaks))])
        for row, (seed, (lo, hi)) in enumerate(zip(settings.peaks, bounds)):
            if self.original_fit.parameters:
                height, center, width = self.original_fit.parameters[2 + 3 * row:5 + 3 * row]
            else:
                center, width, height = seed.center_ev, seed.fwhm_ev, max(float(np.nanmax(cube.Z[self.index]) - np.nanmin(cube.Z[self.index])), 1.)
            for col, value in enumerate((center, width * 1000, max(height, 1e-9), lo, hi)):
                spin = QDoubleSpinBox()
                spin.setDecimals(7 if col in (0, 3, 4) else 5)
                spin.setRange(-10000 if col in (0, 3, 4) else .000001, 10000 if col in (0, 3, 4) else 1e15)
                spin.setKeyboardTracking(False)
                spin.setValue(float(value))
                spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
                spin.valueChanged.connect(self._changed)
                self.table.setCellWidget(row, col, spin)
        self._changed()

    def settings(self):
        peaks = []
        for row in range(self.table.rowCount()):
            center, width, height, lo, hi = [self.table.cellWidget(row, col).value() for col in range(5)]
            peaks.append(PeakSeed(center, width / 1000, height, lo, hi))
        return MultiPeakSettings(tuple(peaks), self.model.currentText())

    def _changed(self, *_):
        self.trial_fit = None
        self.apply_button.setEnabled(False)
        self.status.setText("Edit guesses/bounds, then Refit. Click the spectrum to move the selected peak's center guess.")
        self._draw(self.original_fit, "Current fit")

    def _draw(self, fit, title):
        self.figure.clear()
        self.spectrum_axis, residual = self.figure.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        cube = self.cubes[self.channel.currentText()]
        self.spectrum_axis.plot(cube.energy, cube.Z[self.index], color="black", lw=1, label="Measured")
        if fit.fitted is not None:
            self.spectrum_axis.plot(fit.x, fit.fitted, "--", label="Total fit")
            for index, curve in enumerate(fit.component_curves):
                self.spectrum_axis.plot(fit.x, curve + fit.baseline, ":", label=f"Peak {index + 1}")
            residual.plot(fit.x, fit.residual)
        residual.axhline(0, color="gray", lw=.7)
        self.spectrum_axis.set(title=title, ylabel="PL intensity (a.u.)")
        self.spectrum_axis.legend(fontsize=8)
        residual.set(xlabel="Energy (eV)", ylabel="Residual")
        self.canvas.draw_idle()

    def _click(self, event):
        if self.worker is None and event.button == 1 and event.inaxes is self.spectrum_axis and event.xdata is not None:
            row = self.table.currentRow()
            if row >= 0:
                self.table.cellWidget(row, 0).setValue(float(event.xdata))

    def _run(self):
        if self.worker is not None:
            return
        label = self.channel.currentText()
        cube = self.cubes[label]
        settings = self.settings()
        try:
            center_bounds(settings, cube.energy)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        single = DataCube(cube.energy.copy(), cube.gate[self.index:self.index + 1].copy(),
                           cube.Z[self.index:self.index + 1].copy(), cube.gate_label, cube.title, cube.cbar_label)
        self.worker = _FitWorker(settings, {label: single}, settings)
        self.worker.signals.done.connect(self._finished)
        for widget in (self.channel, self.model, self.table, self.refit, self.apply_button):
            widget.setEnabled(False)
        self.status.setText("Refitting only this spectrum…")
        self.owner.thread_pool.start(self.worker)

    @Slot(object, object, str)
    def _finished(self, settings, results, error):
        self.worker = None
        if self.close_after_fit:
            super().reject()
            return
        for widget in (self.channel, self.model, self.table, self.refit):
            widget.setEnabled(True)
        if error:
            self.status.setText(error)
            return
        fit = results[self.channel.currentText()][0]
        fit.manual = True
        self.trial_fit = fit
        self.apply_button.setEnabled(fit.fitted is not None)
        self.status.setText("; ".join(f"Peak {i + 1}: {p.status}" for i, p in enumerate(fit.components)))
        self._draw(fit, "Manual refit preview — Apply to replace this power only")

    def _apply(self):
        if self.trial_fit is not None:
            self.accepted_fit = (self.channel.currentText(), self.index, self.trial_fit)
            self.accept()

    def reject(self):
        if self.worker is not None:
            self.worker.cancelled.set()
            self.close_after_fit = True
            return
        super().reject()
