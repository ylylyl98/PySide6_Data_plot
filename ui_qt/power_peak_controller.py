"""Background peak fitting and linked power trend views."""

import hashlib
import threading

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QLabel, QWidget, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QHBoxLayout,
)

from core.loader import DataCube
from core.power_peaks import draw_peak_trends
from core.power_multi_peaks import MultiPeakSettings, PeakSeed, detect_peak_seeds, fit_power_multi_peaks


class _Signals(QObject):
    done = Signal(object, object, str)


class _FitWorker(QRunnable):
    def __init__(self, key, cubes, settings):
        super().__init__()
        self.key, self.cubes, self.settings = key, cubes, settings
        self.signals = _Signals()
        self.cancelled = threading.Event()

    def run(self):
        try:
            results = {label: fit_power_multi_peaks(cube, self.settings, cancelled=self.cancelled.is_set)
                       for label, cube in self.cubes.items()}
            self.signals.done.emit(self.key, results, "")
        except Exception as exc:
            self.signals.done.emit(self.key, {}, str(exc))


class PowerPeakController(QObject):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.cache = {}
        self.worker = None
        self.axes = None
        self.residual_axis = None
        self.current_results = {}
        self.manual_overrides = {}
        self.power_law_ranges = {}
        self.power_law_link = None

    def build_controls(self):
        box = QWidget()
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.enabled = QCheckBox("Intensity + linewidth vs power")
        self.model = QComboBox()
        self.model.addItems(["Lorentzian", "Gaussian"])
        self.peak_table = QTableWidget(0, 2)
        self.peak_table.setHorizontalHeaderLabels(["Center (eV)", "FWHM guess (meV)"])
        self.peak_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.peak_table.setMinimumWidth(0)
        self.peak_table.setMaximumHeight(145)
        self.peak_table.setToolTip("Peak 1 is lowest energy. Every power is fitted across the full spectrum with all listed peaks.")
        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        self.detect_btn = QPushButton("Detect peaks")
        self.add_btn = QPushButton("Add")
        self.remove_btn = QPushButton("Remove")
        for button in (self.detect_btn, self.add_btn, self.remove_btn):
            row.addWidget(button)
        self.manual_btn = QPushButton("Manual refit selected power…")
        self.metric = QComboBox()
        self.metric.addItems(["Integrated area", "Peak height"])
        self.status = QLabel("Whole-spectrum fit: detect or add peaks, then adjust center/width guesses. Click the spectrum to move the selected peak guess.")
        self.status.setWordWrap(True)
        form.addRow(self.enabled)
        form.addRow("Peak model", self.model)
        form.addRow(self.peak_table)
        form.addRow(buttons)
        form.addRow("PL intensity", self.metric)
        self.intensity_scale = QComboBox()
        self.linewidth_scale = QComboBox()
        for label, control in (("Intensity Y axis", self.intensity_scale), ("Linewidth Y axis", self.linewidth_scale)):
            control.addItems(["Log", "Linear"])
            control.currentTextChanged.connect(self.changed)
            form.addRow(label, control)
        form.addRow(self.manual_btn)
        self.power_law_btn = QPushButton("Fit power law…")
        self.power_law_btn.clicked.connect(self.power_law_dialog)
        form.addRow(self.power_law_btn)
        form.addRow(self.status)
        self.metric.setToolTip("Area integrates each fitted peak component over the full measured spectrum. Height is that component's amplitude.")
        self.peak_table.cellChanged.connect(self.changed)
        self.detect_btn.clicked.connect(self.detect)
        self.add_btn.clicked.connect(self.add_peak)
        self.remove_btn.clicked.connect(self.remove_peak)
        self.manual_btn.clicked.connect(self.manual_refit)
        self.enabled.toggled.connect(self.changed)
        self.model.currentTextChanged.connect(self.changed)
        self.metric.currentTextChanged.connect(self.changed)
        return box

    def is_enabled(self):
        return self.enabled.isChecked() and self.owner.power_controller._power_view() == "Intensity"

    def source_records(self, cubes):
        w = self.owner
        records = {}
        for label in cubes:
            if label == "Power":
                records[label] = getattr(w.loaded, "power_records", ())
            else:
                try:
                    records[label] = w.power_controller._power_load_group_result(w.power_controller._power_role_group_key(label)).records
                except (ValueError, OSError, KeyError):
                    records[label] = ()
        return records

    def changed(self, *_):
        if not self.enabled.isChecked() and self.worker is not None:
            self.worker.cancelled.set()
        self.owner._invalidate_export_move_sources()
        self.owner._schedule_plot_redraw("Power Dependent")

    def settings(self):
        peaks = tuple(PeakSeed(float(self.peak_table.item(index, 0).text()),
                              float(self.peak_table.item(index, 1).text()) / 1000.)
                      for index in range(self.peak_table.rowCount()))
        return MultiPeakSettings(peaks, self.model.currentText())

    def set_peaks(self, peaks):
        self.peak_table.blockSignals(True)
        try:
            self.peak_table.setRowCount(len(peaks))
            self.peak_table.setVerticalHeaderLabels([f"Peak {i + 1}" for i in range(len(peaks))])
            for row, seed in enumerate(peaks):
                self.peak_table.setItem(row, 0, QTableWidgetItem(f"{seed.center_ev:.7g}"))
                self.peak_table.setItem(row, 1, QTableWidgetItem(f"{seed.fwhm_ev * 1000:.7g}"))
        finally:
            self.peak_table.blockSignals(False)

    def detect(self):
        w = self.owner
        if not w.loaded or w.loaded.mode != "Power Dependent":
            self.status.setText("Load a power sweep first.")
            return
        try:
            self.set_peaks(detect_peak_seeds(w.loaded.cube))
            self.changed()
        except ValueError as exc:
            self.status.setText(str(exc))

    def add_peak(self):
        try:
            peaks = list(self.settings().peaks)
        except ValueError:
            self.status.setText("Enter numeric peak center and width guesses first.")
            return
        if len(peaks) >= 8:
            return
        w = self.owner
        if w.loaded and w.loaded.mode == "Power Dependent":
            energy = w.loaded.cube.energy
            gaps = np.r_[energy[0], [p.center_ev for p in peaks], energy[-1]]
            index = int(np.argmax(np.diff(gaps)))
            center = .5 * (gaps[index] + gaps[index + 1])
        else:
            center = peaks[-1].center_ev + .02 if peaks else 1.5
        peaks.append(PeakSeed(float(center)))
        self.set_peaks(sorted(peaks, key=lambda seed: seed.center_ev))
        self.changed()

    def remove_peak(self):
        if self.peak_table.rowCount() <= 1:
            return
        index = self.peak_table.currentRow()
        self.peak_table.blockSignals(True)
        self.peak_table.removeRow(index if index >= 0 else self.peak_table.rowCount() - 1)
        self.peak_table.setVerticalHeaderLabels([f"Peak {i + 1}" for i in range(self.peak_table.rowCount())])
        self.peak_table.blockSignals(False)
        self.changed()

    def initialize_center(self, cubes):
        if self.peak_table.rowCount():
            return
        cube = next(iter(cubes.values()))
        self.set_peaks(detect_peak_seeds(cube))

    def key(self, cubes):
        digest = hashlib.sha256()
        for label, cube in cubes.items():
            digest.update(label.encode())
            for array in (cube.energy, cube.gate, cube.Z):
                array = np.asarray(array, dtype=float)
                digest.update(str(array.shape).encode())
                digest.update(array.tobytes())
        return (self.settings(), digest.hexdigest())

    def request(self, cubes):
        self.initialize_center(cubes)
        key = self.key(cubes)
        if self.worker is not None and self.worker.key != key:
            self.worker.cancelled.set()
        if key in self.cache:
            results, error = self.cache[key]
            if error:
                self.status.setText(error)
            return results
        if self.worker is None:
            copies = {label: DataCube(c.energy.copy(), c.gate.copy(), c.Z.copy(),
                                      c.gate_label, c.title, c.cbar_label) for label, c in cubes.items()}
            self.worker = _FitWorker(key, copies, self.settings())
            self.worker.signals.done.connect(self.finished)
            self.owner.thread_pool.start(self.worker)
        self.status.setText("Fitting spectra… Plot controls remain available.")
        return None

    @Slot(object, object, str)
    def finished(self, key, results, error):
        self.worker = None
        if self.owner._is_closing:
            return
        if error != "cancelled":
            for (label, index), fit in self.manual_overrides.get(key, {}).items():
                if label in results:
                    rows = list(results[label])
                    rows[index] = fit
                    results[label] = tuple(rows)
            self.cache[key] = (results, error)
        while len(self.cache) > 4:
            del self.cache[next(iter(self.cache))]
        self.owner._schedule_plot_redraw("Power Dependent")

    def render(self, cubes):
        if not self.is_enabled() or self.axes is None:
            return
        results = self.request(cubes)
        self.current_results = results or {}
        w = self.owner
        limits = tuple(sorted((w.power_spins["ymin"].value(), w.power_spins["ymax"].value())))
        draw_peak_trends(self.axes, self.current_results, log_power=w.power_controller._power_axis_log(),
                         metric=self.metric.currentText(), power_limits=limits,
                         log_intensity=self.intensity_scale.currentText() == "Log",
                         log_linewidth=self.linewidth_scale.currentText() == "Log", records=self.source_records(cubes))
        if results:
            from core.power_law_analysis import draw_power_laws
            if self.metric.currentText() == 'Integrated area':
                try:
                    draw_power_laws(self.axes[0], self.power_law_fits(cubes, results))
                except ValueError:
                    self.axes[0].text(.02, .88, 'Power-law range needs review after refitting', transform=self.axes[0].transAxes, fontsize=7)
            self.power_law_link = self.axes[0].text(1, 1.02, 'Fit power law…', ha='right',
                transform=self.axes[0].transAxes, fontsize=8, color='tab:blue', clip_on=False)
        self.residual_axis.clear()
        self.residual_axis.set_ylabel("Residual")
        self.residual_axis.set_xlabel("Energy (eV)")
        self.residual_axis.axhline(0, color="gray", lw=0.7)
        if results is None:
            for axis in self.axes:
                axis.text(.5, .5, "Fitting…", transform=axis.transAxes, ha="center")
            return
        summaries = []
        selected_power = w.power_spins["gate"].value()
        for channel_index, (label, fits) in enumerate(results.items()):
            good = sum(sum(p.status == "ok" for p in fit.components) for fit in fits)
            total = sum(len(fit.components) for fit in fits)
            summaries.append(f"{label}: {good}/{total} peak fits valid")
            if not fits:
                continue
            fit = min(fits, key=lambda fit: abs(fit.power_uw - selected_power))
            if fit.fitted is not None:
                w._power_spectrum_ax.plot(fit.x, fit.fitted, "--", lw=1.4,
                                          label=f"{label} total fit")
                for index, curve in enumerate(fit.component_curves):
                    color = f"C{(channel_index * len(fit.components) + index) % 10}"
                    w._power_spectrum_ax.plot(fit.x, curve + fit.baseline, ":", color=color, lw=1.1,
                                              label=f"{label} Peak {index + 1}")
                w._power_spectrum_ax.plot(fit.x, fit.baseline, color="gray", lw=.8, label=f"{label} baseline")
                self.residual_axis.plot(fit.x, fit.residual, lw=1, label=label)
            statuses = ", ".join(f"P{i + 1}: {p.status}" for i, p in enumerate(fit.components))
            summaries.append(f"{label} @ {fit.power_uw:.6g} uW{' (manual)' if fit.manual else ''}: {statuses}")
            records = self.source_records(cubes).get(label, ())
            selected_index = int(np.argmin(abs(np.asarray(cubes[label].gate) - fit.power_uw)))
            if selected_index < len(records):
                from core.power_workflow import record_sources
                summaries.append("Source: " + ", ".join(record_sources(records[selected_index])))
        omitted = sum(np.count_nonzero(np.asarray(c.gate) <= 0) for c in cubes.values())
        if omitted and w.power_controller._power_axis_log():
            summaries.append(f"{omitted} nonpositive power points omitted from log views")
        if results:
            self.status.setText("; ".join(summaries))
            w._power_spectrum_ax.legend(fontsize=7, loc="best")
            self.residual_axis.set_xlim(w._power_spectrum_ax.get_xlim())

    def click(self, event):
        if self.is_enabled() and self.axes is not None and self.power_law_link is not None and self.power_law_link.contains(event)[0]:
            self.power_law_dialog()
            return True
        if not self.is_enabled() or self.axes is None or event.xdata is None:
            return False
        if self.owner.toolbar.mode:
            return False
        if event.inaxes is self.owner._power_spectrum_ax:
            row = self.peak_table.currentRow()
            if row >= 0:
                self.peak_table.item(row, 0).setText(f"{float(event.xdata):.7g}")
            else:
                self.status.setText("Select a peak row in the table before clicking its center on the spectrum.")
            return True
        if event.inaxes not in self.axes:
            return False
        candidates = [fit.power_uw for fits in self.current_results.values() for fit in fits
                      if any(p.status == "ok" for p in fit.components)
                      and (not self.owner.power_controller._power_axis_log() or fit.power_uw > 0)]
        if candidates:
            power = min(candidates, key=lambda p: abs(np.log(p) - np.log(event.xdata))
                        if self.owner.power_controller._power_axis_log() and event.xdata > 0 else abs(p - event.xdata))
            self.owner._power_selected_row_index = None
            self.owner.power_spins["gate"].setValue(power)
            self.owner.power_controller._update_power_compare_spectrum_and_lines(self.owner._power_active_cubes)
        return True

    def apply_manual_fit(self, key, label, index, fit):
        if key not in self.cache:
            raise ValueError("The source or fit settings changed. Plot the current data before applying this refit.")
        results, error = self.cache[key]
        updated = dict(results)
        rows = list(updated[label])
        rows[index] = fit
        updated[label] = tuple(rows)
        self.cache[key] = (updated, error)
        self.manual_overrides.setdefault(key, {})[(label, index)] = fit
        self.current_results = updated
        self.owner._invalidate_export_move_sources()
        self.owner._schedule_plot_redraw("Power Dependent")

    def manual_refit(self):
        from ui_qt.power_refit_dialog import PowerRefitDialog
        w = self.owner
        if not self.is_enabled() or not w._power_active_cubes:
            self.status.setText("Enable peak analysis and plot a power sweep first.")
            return
        cubes = w._power_active_cubes
        results = self.request(cubes)
        if not results:
            self.status.setText("Wait for the full-spectrum fits before manually refitting one power.")
            return
        key = self.key(cubes)
        dialog = PowerRefitDialog(w, cubes, results, self.settings(), w.power_spins["gate"].value())
        if dialog.exec() and dialog.accepted_fit is not None:
            if key != self.key(w._power_active_cubes):
                self.status.setText("The source changed while refitting; the refit was not applied.")
                return
            label, index, fit = dialog.accepted_fit
            self.apply_manual_fit(key, label, index, fit)

    def export_payload(self, cubes, records, background):
        results = self.request(cubes)
        if not results:
            raise ValueError("Peak analysis is not ready. Wait for the fit status before saving.")
        w = self.owner
        return {"results": results, "records": records, "settings": self.settings(),
                "power_laws": self.power_law_fits(cubes, results),
                "power_law_ranges": self.power_law_ranges.get(self.key(cubes), []),
                "background": float(background), "metric": self.metric.currentText(),
                "log_power": w.power_controller._power_axis_log(),
                "log_intensity": self.intensity_scale.currentText() == "Log",
                "log_linewidth": self.linewidth_scale.currentText() == "Log",
                "power_limits": tuple(sorted((w.power_spins["ymin"].value(), w.power_spins["ymax"].value())))}

    def power_law_fits(self, cubes, results):
        from core.power_law_analysis import evaluate_ranges
        return evaluate_ranges(results, self.source_records(cubes), self.power_law_ranges.get(self.key(cubes), []))

    def power_law_dialog(self):
        from ui_qt.power_law_dialog import PowerLawDialog
        cubes = getattr(self.owner, '_power_active_cubes', {})
        if not self.is_enabled() or not cubes:
            self.status.setText('Enable peak analysis and load a sweep first.')
            return
        results = self.request(cubes)
        if not results:
            self.status.setText('Wait for the spectrum fits before fitting a power law.')
            return
        key = self.key(cubes)
        dialog = PowerLawDialog(self.owner, results, self.source_records(cubes), self.power_law_ranges.get(key, []))
        if dialog.exec():
            self.power_law_ranges[key] = dialog.ranges
            self.metric.setCurrentText('Integrated area')
            self.changed()
        dialog.deleteLater()
