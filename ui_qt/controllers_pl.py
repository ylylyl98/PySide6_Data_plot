"""Controller for PL source selection and source-to-source navigation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, Signal
from PySide6.QtGui import QColor
from scipy.optimize import curve_fit
from PySide6.QtWidgets import (
    QListWidgetItem,
)

from core import data_io
from core.drr_sources import resolve_source_path
from core.loader import DataCube
from core.processing import nearest_gate_spectrum
from ui_qt.common import QComboBox
from ui_qt.theme import alias as theme_alias
from ui_qt.source_picker_dialog import SourcePickerDialog


def _multi_lorentz_model_worker(x: np.ndarray, *p: float) -> np.ndarray:
    x_arr = np.asarray(x, float)
    out = p[0] + p[1] * x_arr
    n = (len(p) - 2) // 3
    for i in range(n):
        amp = p[2 + 3 * i]
        cen = p[3 + 3 * i]
        gam = max(1e-12, p[4 + 3 * i])
        out = out + amp * (gam * gam) / ((x_arr - cen) * (x_arr - cen) + gam * gam)
    return out


class _PlFitSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class _PlFitWorker(QRunnable):
    def __init__(self, x: np.ndarray, y: np.ndarray, p0: list[float], lo: list[float], hi: list[float]) -> None:
        super().__init__()
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.p0, self.lo, self.hi = p0, lo, hi
        self.signals = _PlFitSignals()

    def run(self) -> None:
        try:
            popt, _ = curve_fit(
                _multi_lorentz_model_worker,
                self.x,
                self.y,
                p0=np.asarray(self.p0, float),
                bounds=(np.asarray(self.lo, float), np.asarray(self.hi, float)),
                maxfev=50000,
            )
            self.signals.result.emit(np.asarray(popt, float))
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class PlController:
    """Own PL source-selection behavior while sharing app state with the owner.

    The owner proxy is intentional during this incremental refactor: source
    selection is now isolated, while plotting and processing remain in the
    main window until their state boundary is extracted.
    """

    def __init__(self, owner) -> None:
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_owner"), name)

    def __setattr__(self, name, value) -> None:
        if name == "_owner":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_owner"), name, value)

    def _set_pl_gate_spin_value(self, gate_value: float) -> None:
        spin = self.pl_spins["gate"]
        old = spin.blockSignals(True)
        try:
            spin.setValue(float(gate_value))
        finally:
            spin.blockSignals(old)

    def _current_pl_spectrum(self, cube: DataCube) -> tuple[float, np.ndarray, np.ndarray]:
        gate_value = float(self.pl_spins["gate"].value())
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        return gate_used, x, np.asarray(y, float).ravel()

    def _on_pl_selection_changed(self) -> None:
        self._invalidate_pending_pl_fit("Fit discarded: PL source changed.")
        selected = self._selected(self.pl_files)
        file_name = selected[0] if selected else ""
        self._update_pl_selection_summary()
        self._repopulate_yaxis_combo("pl", xlsx=data_io.is_xlsx_map_file(file_name))

    def _pl_is_saved_dat(self, source: str) -> bool:
        normalized = source.replace("\\", "/").casefold()
        return normalized.startswith("processed data/pl/") and Path(source).suffix.lower() == ".dat"

    def _pl_source_is_processed(self, source: str) -> bool:
        return self._pl_is_saved_dat(source) or source in self.pl_processed_status

    def _update_pl_selection_summary(self) -> None:
        if not hasattr(self, "pl_selection_summary"):
            return
        selected = self._selected(self.pl_files)
        if not selected:
            self.pl_selection_summary.set_status(
                "No PL file selected.", tooltip="", app_role=None, badge_state=None
            )
            return
        source = selected[0]
        display_name = Path(source).name.replace("_", "_\u200b").replace("-", "-\u200b")
        processed_at = self.pl_processed_status.get(source, "")
        if self._pl_is_saved_dat(source):
            state = "◆ SAVED DAT — Processed result"
            badge_state = "saved"
        elif processed_at:
            state = f"✓ PROCESSED\nLast saved: {processed_at[:16].replace('T', ' ')}"
            badge_state = "processed"
        else:
            state = "● NEW — No saved analysis"
            badge_state = "new"
        self.pl_selection_summary.set_status(
            f"{state}\nSelected: {display_name}",
            tooltip=source,
            app_role="sourceBadge",
            badge_state=badge_state,
        )

    def _pl_source_modified(self, source: str) -> float:
        cached = self._pl_source_mtime_cache.get(source)
        if cached is not None:
            return cached
        try:
            modified = resolve_source_path(self.current_folder, source).stat().st_mtime
        except OSError:
            modified = 0.0
        self._pl_source_mtime_cache[source] = modified
        return modified

    def _pl_sources_newest_first(self) -> list[str]:
        return sorted(
            self.pl_available_files,
            key=lambda source: (-self._pl_source_modified(source), source.casefold()),
        )

    def _pl_saved_source_filter(self) -> str:
        value = str(getattr(self, "_pl_source_filter_preference", "all")).casefold()
        return value if value in {"all", "unprocessed", "processed"} else "all"

    def _pl_source_filter_counts(self) -> dict[str, int]:
        processed = sum(1 for source in self.pl_available_files if self._pl_source_is_processed(source))
        return {
            "all": len(self.pl_available_files),
            "unprocessed": len(self.pl_available_files) - processed,
            "processed": processed,
        }

    def _open_pl_source_dialog(self, selected: str) -> str | None:
        state_filter = QComboBox()
        self._style_combo_popup(state_filter)
        dlg = SourcePickerDialog(
            self._owner,
            title="Choose PL File",
            hint=(
            "Choose one PL source. Raw CSV/XLSX inputs and saved PL DAT results are listed together, newest first."
            ),
            selected=selected,
            filter_controls=(("Status", state_filter),),
            filter_interval=140,
        )
        file_list = dlg.source_list
        details = dlg.details_label
        ok_button = dlg.ok_button
        refresh_btn = dlg.refresh_button

        def _populate_filter_counts() -> None:
            current = str(state_filter.currentData() or self._pl_saved_source_filter())
            counts = self._pl_source_filter_counts()
            blocked = state_filter.blockSignals(True)
            state_filter.clear()
            state_filter.addItem(f"All ({counts['all']})", "all")
            state_filter.addItem(f"New ({counts['unprocessed']})", "unprocessed")
            state_filter.addItem(f"Processed ({counts['processed']})", "processed")
            index = state_filter.findData(current)
            state_filter.setCurrentIndex(index if index >= 0 else 0)
            state_filter.blockSignals(blocked)

        def _refresh_view() -> None:
            needle = filter_edit.text().strip().casefold()
            wanted = str(state_filter.currentData() or "all")
            def _populate(widget) -> None:
                for source in self._pl_sources_newest_first():
                    is_processed = self._pl_source_is_processed(source)
                    if wanted == "unprocessed" and is_processed:
                        continue
                    if wanted == "processed" and not is_processed:
                        continue
                    if needle and needle not in source.casefold():
                        continue
                    modified = self._pl_source_modified(source)
                    modified_text = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M") if modified else "date unavailable"
                    processed_at = self.pl_processed_status.get(source, "")
                    if self._pl_is_saved_dat(source):
                        text = f"◆ SAVED DAT — {Path(source).name}\nModified {modified_text} · Ready to view"
                        color = QColor(theme_alias("source_saved_foreground"))
                        bold = False
                    elif processed_at:
                        text = f"✓ PROCESSED — {Path(source).name}\nModified {modified_text} · Saved {processed_at[:16].replace('T', ' ')}"
                        color = QColor(theme_alias("source_processed_foreground"))
                        bold = False
                    else:
                        text = f"● NEW — {Path(source).name}\nModified {modified_text} · No saved analysis"
                        color = QColor(theme_alias("source_new_foreground"))
                        bold = True
                    item = QListWidgetItem(text)
                    item.setData(Qt.UserRole, source)
                    item.setToolTip(source)
                    item.setForeground(color)
                    font = item.font(); font.setBold(bold); item.setFont(font)
                    widget.addItem(item)
            dlg.repopulate(_populate, fallback_selection=selected)

        def _update_details() -> None:
            item = file_list.currentItem()
            ok_button.setEnabled(item is not None)
            if item is None:
                details.setText("No matching PL files.")
                return
            source = str(item.data(Qt.UserRole))
            if self._pl_is_saved_dat(source):
                state = "◆ SAVED DAT — Previously exported PL result."
            elif source in self.pl_processed_status:
                state = f"✓ PROCESSED — Last saved {self.pl_processed_status[source][:16].replace('T', ' ')}"
            else:
                state = "● NEW — No saved PL analysis was found."
            details.setText(f"{source}\n{state}")

        def _reload_catalog() -> None:
            self._refresh_file_lists(auto=True)
            _populate_filter_counts(); _refresh_view(); _update_details()

        def _on_filter_changed() -> None:
            value = str(state_filter.currentData() or "all")
            self._pl_source_filter_preference = value
            self.settings.setValue(self.SETTINGS_PL_SOURCE_FILTER, value)
            _refresh_view()

        filter_edit = dlg.filter_edit
        dlg.filter_requested.connect(_refresh_view)
        state_filter.currentIndexChanged.connect(lambda _index: _on_filter_changed())
        file_list.currentItemChanged.connect(lambda _current, _previous: _update_details())
        refresh_btn.clicked.connect(_reload_catalog)
        _populate_filter_counts(); _refresh_view(); _update_details()
        if dlg.exec() != SourcePickerDialog.Accepted:
            return None
        return dlg.selected_source()

    def _edit_pl_source(self) -> None:
        selected = self._selected(self.pl_files)
        previous = selected[0] if selected else ""
        chosen = self._open_pl_source_dialog(previous)
        if not chosen:
            return
        if chosen != previous:
            self._restore_list_selection(self.pl_files, [chosen])
        self._pl_auto_next_queue = []
        self._pl_auto_next_active = False
        self._status(f"Selected PL source: {Path(chosen).name}. Loading now...")
        self._start_load("PL")

    def _clear_pl_source(self) -> None:
        self._pl_auto_next_queue = []
        self._pl_auto_next_active = False
        self.pl_files.clearSelection()
        self._invalidate_export_move_sources()
        if self.loaded and self.loaded.mode == "PL":
            self.loaded = None
        if self.last_plotted_mode == "PL":
            self.last_plotted_mode = None
            self.figure.clear(); self.canvas.draw_idle()
        self._set_stage("No PL source")
        self._update_action_states()

    def _auto_load_next_unprocessed_pl(self, completed_source: str) -> bool:
        if not hasattr(self, "pl_auto_next_chk") or not self.pl_auto_next_chk.isChecked():
            return False
        self._pl_auto_next_queue = [
            source
            for source in self._pl_sources_newest_first()
            if not self._pl_is_saved_dat(source)
            and source not in self.pl_processed_status
            and source != completed_source
        ]
        if not self._pl_auto_next_queue:
            self._status("All PL measurements are processed.")
            return False
        return self._load_next_pl_from_queue(completed_source)

    def _load_next_pl_from_queue(self, completed_source: str = "") -> bool:
        if not self._pl_auto_next_queue:
            self._pl_auto_next_active = False
            self._status("No valid unprocessed PL measurements remain.")
            return False
        chosen = self._pl_auto_next_queue.pop(0)
        self._pl_auto_next_active = True
        self._restore_list_selection(self.pl_files, [chosen])
        prefix = f"Saved {Path(completed_source).name}; " if completed_source else ""
        self._status(f"{prefix}loading next new PL file: {Path(chosen).name}")
        self._start_load("PL")
        return True
    def _on_pl_find_peaks(self) -> None:
        self._invalidate_pending_pl_fit("Fit discarded: peak candidates changed.")
        self.results_dock.show()
        self._update_results_dock_page()
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
        self.results_dock.show()
        self._update_results_dock_page()
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
        self._pl_fit_generation = getattr(self, "_pl_fit_generation", 0) + 1
        generation = self._pl_fit_generation
        source = str(self.loaded.primary_file or "") if self.loaded else ""
        requested_gate = float(self.pl_spins["gate"].value())
        worker = _PlFitWorker(x_sel, y_sel, p0, lo, hi)
        workers = getattr(self, "_pl_fit_workers", None)
        if workers is None:
            workers = []
            self._pl_fit_workers = workers
        workers.append(worker)
        worker.signals.result.connect(
            lambda popt, g=generation, c=self._pl_last_plot_cube, s=source, gate=gate_used, requested=requested_gate, peaks=n_peaks:
            self._on_pl_fit_finished(g, c, s, gate, requested, peaks, x.copy(), popt)
        )
        worker.signals.error.connect(
            lambda message, g=generation: self._on_pl_fit_error(g, message)
        )
        worker.signals.finished.connect(lambda w=worker: self._finish_pl_fit_worker(w))
        self.pl_fit_status.setText("Fitting Lorentz peaks…")
        self.thread_pool.start(worker)

    def _finish_pl_fit_worker(self, worker: _PlFitWorker) -> None:
        workers = getattr(self, "_pl_fit_workers", [])
        try:
            workers.remove(worker)
        except ValueError:
            pass

    def _on_pl_fit_error(self, generation: int, message: str) -> None:
        if generation == getattr(self, "_pl_fit_generation", 0):
            self.pl_fit_status.setText(f"Fit failed: {message}")

    def _on_pl_fit_finished(
        self, generation: int, cube: DataCube, source: str, gate_used: float,
        requested_gate: float, n_peaks: int, x: np.ndarray, popt: np.ndarray,
    ) -> None:
        if generation != getattr(self, "_pl_fit_generation", 0):
            return
        if (
            self.last_plotted_mode != "PL" or self._pl_last_plot_cube is not cube
            or not self.loaded or str(self.loaded.primary_file or "") != source
        ):
            if str(self.pl_fit_status.text()).startswith("Fitting"):
                self.pl_fit_status.setText("Fit discarded: PL source changed.")
            return
        current_gate, _y = self._current_pl_spectrum(cube)
        if abs(float(current_gate) - float(gate_used)) > 1e-9:
            if str(self.pl_fit_status.text()).startswith("Fitting"):
                self.pl_fit_status.setText("Fit discarded: gate changed.")
            return
        y_fit = _multi_lorentz_model_worker(x, *popt)
        centers_fit = np.asarray([popt[3 + 3 * i] for i in range(n_peaks)], float)
        self._pl_fit_gate = float(gate_used)
        self._pl_fit_x = x
        self._pl_fit_y = np.asarray(y_fit, float)
        self._pl_fit_centers = np.asarray(np.sort(centers_fit), float)
        self.pl_fit_status.setText("Fit centers: " + ", ".join(f"{c:.4f}" for c in self._pl_fit_centers[:4]))
        self._update_pl_spectrum_with_analysis(cube)
    def _on_pl_clear_fit(self) -> None:
        self._invalidate_pending_pl_fit()
        self._pl_fit_gate = None
        self._pl_fit_x = None
        self._pl_fit_y = None
        self._pl_fit_centers = None
        self.pl_fit_status.setText("")
        if self.last_plotted_mode == "PL" and self._pl_last_plot_cube is not None:
            self._update_pl_spectrum_with_analysis(self._pl_last_plot_cube)
    def _on_pl_analysis_view_changed(self) -> None:
        if self.last_plotted_mode == "PL" and self._pl_last_plot_cube is not None:
            self._update_pl_spectrum_and_gate_line(self._pl_last_plot_cube)
    def _on_pl_gate_changed(self) -> None:
        self._invalidate_pending_pl_fit("Fit discarded: gate changed.")
        if self.last_plotted_mode == "PL" and self._pl_last_plot_cube is not None:
            self._update_pl_spectrum_and_gate_line(self._pl_last_plot_cube)
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
    def _on_pl_plot_param_changed(self) -> None:
        self._invalidate_pending_pl_fit("Fit discarded: plot range or display settings changed.")
        self._invalidate_export_move_sources()
        self._apply_dat_y_axis_selection()
        if self.loaded and self.loaded.mode == "PL":
            sender = self.sender()
            if sender in (
                self.pl_spins["xmin"], self.pl_spins["xmax"],
                self.pl_spins["ymin"], self.pl_spins["ymax"], self.pl_log_chk,
            ):
                self._refresh_automatic_ranges(
                    "PL",
                    refresh_split=True,
                    center_split=sender in (self.pl_spins["xmin"], self.pl_spins["xmax"]),
                )
            self._schedule_plot_redraw("PL")

    def _invalidate_pending_pl_fit(self, message: str = "") -> None:
        """Invalidate a running fit without disturbing a newer fit result."""
        self._pl_fit_generation = getattr(self, "_pl_fit_generation", 0) + 1
        if message and str(self.pl_fit_status.text()).startswith("Fitting"):
            self.pl_fit_status.setText(message)
    def _on_pl_dat_y_axis_changed(self, _text: str) -> None:
        is_dat = Path((self.loaded.primary_file if self.loaded else "") or "").suffix.lower() == ".dat"
        custom = self.pl_yaxis_combo.currentText() == "Custom"
        self.pl_dat_yaxis_label_edit.setVisible(is_dat and custom)
        self.pl_dat_yaxis_unit_edit.setVisible(is_dat)
        self._on_pl_plot_param_changed()
    def _auto_pl_vrange(self) -> None:
        if not self.loaded or self.loaded.mode != "PL" or self.loaded.cube is None:
            return
        cube = self.loaded.cube
        x = np.asarray(cube.energy, float).ravel()
        y = np.asarray(cube.gate, float).ravel()
        z = np.asarray(cube.Z, float)
        x0, x1 = sorted((float(self.pl_spins["xmin"].value()), float(self.pl_spins["xmax"].value())))
        y0, y1 = sorted((float(self.pl_spins["ymin"].value()), float(self.pl_spins["ymax"].value())))
        x_mask = (x >= x0) & (x <= x1)
        y_mask = (y >= y0) & (y <= y1)
        z_roi = z[np.ix_(y_mask, x_mask)] if np.any(y_mask) and np.any(x_mask) else z
        finite = z_roi[np.isfinite(z_roi)]
        if finite.size == 0:
            self._status("State: Auto vmin/vmax skipped (no finite values in selected x/y range).")
            return
        if self._mode_log("PL"):
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
        self.pl_spins["vmin"].setValue(vmin)
        self.pl_spins["vmax"].setValue(vmax)
        self._status(f"State: Auto vmin/vmax (ROI) = {vmin:.4g}, {vmax:.4g}")
        self._schedule_plot_redraw("PL")
    def _auto_pl_xrange(self) -> None:
        if not self.loaded or self.loaded.mode != "PL" or self.loaded.cube is None:
            return
        self.pl_spins["xmin"].setValue(float(np.nanmin(self.loaded.cube.energy)))
        self.pl_spins["xmax"].setValue(float(np.nanmax(self.loaded.cube.energy)))
        self._status("State: Auto xmin/xmax set from energy axis.")
        self._schedule_plot_redraw("PL")
    def _auto_pl_yrange(self) -> None:
        if not self.loaded or self.loaded.mode != "PL" or self.loaded.cube is None:
            return
        self.pl_spins["ymin"].setValue(float(np.nanmin(self.loaded.cube.gate)))
        self.pl_spins["ymax"].setValue(float(np.nanmax(self.loaded.cube.gate)))
        self._status("State: Auto ymin/ymax set from gate axis.")
        self._schedule_plot_redraw("PL")
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
    def _ensure_pl_gate_line(self, cube: DataCube, gate_value: float) -> None:
        if self._pl_heatmap_ax is None:
            return
        gate = np.asarray(cube.gate, float).ravel()
        gate_clamped = float(np.clip(gate_value, float(np.nanmin(gate)), float(np.nanmax(gate))))
        if self._pl_gate_line is None or getattr(self._pl_gate_line, "axes", None) is not self._pl_heatmap_ax:
            self._pl_gate_line = self._pl_heatmap_ax.axhline(
                y=gate_clamped,
                lw=1.2,
                alpha=0.9,
                color="#222",
                linestyle="--",
                zorder=20,
            )
        else:
            self._pl_gate_line.set_ydata([gate_clamped, gate_clamped])
            self._pl_gate_line.set_linestyle("--")
    def _update_pl_spectrum_and_gate_line(self, cube: DataCube) -> None:
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
        xlim = self._safe_spectrum_xlim(
            x,
            (float(self.pl_spins["xmin"].value()), float(self.pl_spins["xmax"].value())),
        )
        self._pl_spectrum_ax.set_xlim(xlim)
        self._auto_scale_spectrum_y(self._pl_spectrum_ax, x, y, xlim)
        self._set_pl_gate_spin_value(gate_used)
        self._ensure_pl_gate_line(cube, gate_used)
        self._draw_pl_analysis_overlays(gate_used, x, np.asarray(y, float))
        self._update_pl_analysis_text(gate_used, x, np.asarray(y, float))
        self.canvas.draw_idle()
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
