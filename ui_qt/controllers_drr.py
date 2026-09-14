"""Controller for the DRR plot-control workflow."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import re
from typing import List

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QSize, Qt, Signal, QItemSelectionModel
from PySide6.QtGui import QColor, QFont, QPainter, QPalette
from scipy.optimize import curve_fit
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStyle,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from core import data_io
from core.drr_baseline_candidates import (
    candidate_recommendation_map,
    parse_drr_acquisition_conditions,
)
from core.drr_sources import (
    DrrSource,
    assess_background_gate_files,
    compatible_drr_repeats,
    discover_drr_sources,
    extract_wavelength_center_nm,
    find_saved_drr_recipe,
    guess_drr_background,
    group_drr_sources,
    inspect_csv_spectral_grid,
    inspect_csv_wavelength_center,
    resolve_source_path,
    resolve_drr_background_assignments,
    wavelength_centers_match,
)
from core.loader import DataCube
from core.processing import apply_sg_derivative_energy, clamp_sg_window, nearest_gate_spectrum
from ui_qt.common import Worker, WrappedFilenameDelegate
from ui_qt.theme import alias as theme_alias
from ui_qt.source_picker_dialog import SourcePickerDialog
from ui_qt.axes_region_blitter import AxesRegionBlitter


# ``None`` is a valid explicit request for the baseline DR/R product, so the
# omitted argument needs its own marker.  This lets display and export request
# independent products while retaining the selected derivative for legacy
# callers.
_CURRENT_DRR_DERIVATIVE = object()


def _drr_condition_values(source) -> dict[str, str]:
    """Return display values from the original acquisition filename."""
    filename = str(getattr(source, "filename", source))
    conditions = parse_drr_acquisition_conditions(filename)
    stem = Path(filename).stem
    values: dict[str, str] = {}
    if conditions.rotation:
        values["rotation"] = " · ".join(
            part.split("=", 1)[-1] + "°" for part in conditions.rotation.split(";")
        )
    # TG/BG describe the encoded acquisition condition.  Keep the numeric
    # expression from the filename; it is not necessarily a calibrated gate.
    for key in ("tg", "bg"):
        match = re.search(
            rf"(?<![A-Za-z]){key}(?:\s*[:=]?\s*)([+\-]?\d+(?:[pP.]\d+)?)",
            stem,
            re.IGNORECASE,
        )
        if match:
            values[key] = match.group(1).replace("p", ".").replace("P", ".")
    wavelength = getattr(source, "wavelength_center_nm", None)
    if wavelength is None:
        wavelength = extract_wavelength_center_nm(filename)
    if wavelength is not None:
        values["wavelength"] = f"{float(wavelength):g} nm"
    return values


def format_drr_source_summary(source, peers=(), *, peer_condition_values=None) -> str:
    """Build a compact metadata-first label for one DRR picker row."""
    filename = str(getattr(source, "filename", source))
    conditions = parse_drr_acquisition_conditions(filename)
    stem = Path(filename).stem
    sample_match = re.search(r"(?<![A-Za-z0-9])(YZ\d+)(?![A-Za-z0-9])", stem, re.IGNORECASE)
    point_match = re.search(r"(?<![A-Za-z0-9])(p[A-Za-z0-9]+)(?![A-Za-z0-9])", stem, re.IGNORECASE)
    identity_parts = []
    if sample_match:
        identity_parts.append(sample_match.group(1))
    elif conditions.sample:
        identity_parts.append(conditions.sample)
    if point_match:
        identity_parts.append(point_match.group(1))
    elif conditions.position:
        identity_parts.append(conditions.position)
    identity = " · ".join(identity_parts) or Path(filename).name

    first = [identity]
    # classify_pl_source deliberately handles attached markers such as
    # ``1.67KREF``; do not duplicate a boundary-sensitive local regex here.
    measurement_type = data_io.classify_pl_source(filename)
    if measurement_type.casefold() not in {"unknown", "measurement"}:
        first.append(measurement_type.upper())
    if conditions.magnetic_field:
        first.append(conditions.magnetic_field.replace("T", " T"))
    if conditions.temperature:
        first.append(conditions.temperature.replace("K", " K"))
    wavelength = getattr(source, "wavelength_center_nm", None)
    if wavelength is None:
        wavelength = extract_wavelength_center_nm(filename)
    if wavelength is not None:
        first.append(f"{float(wavelength):g} nm")

    values = _drr_condition_values(source)
    peer_values = {key: set() for key in values}
    peer_condition_values = (
        tuple(peer_condition_values)
        if peer_condition_values is not None
        else tuple(_drr_condition_values(peer) for peer in (source, *tuple(peers or ())))
    )
    for peer_values_for_file in peer_condition_values:
        for key in peer_values:
            peer_values[key].add(peer_values_for_file.get(key, ""))
    # Keep every available difference parameter, placing varying conditions
    # first so a multi-file session is scannable at a glance.
    changed = [key for key in values if len(peer_values[key]) > 1]
    ordered = changed + [key for key in values if key not in changed]
    labels = {"rotation": "Rot", "tg": "TG", "bg": "BG", "wavelength": "λ"}
    differences = [f"{labels[key]} {values[key]}" for key in ordered]

    range_line = _compact_drr_gate_text(source)
    lines = [" · ".join(first)]
    if differences:
        lines.append(" · ".join(differences))
    if range_line:
        lines.append(range_line)
    return "\n".join(lines)


def _compact_drr_gate_text(source) -> str:
    labels = tuple(str(label) for label in getattr(source, "gate_labels", ()) or ())
    set_labels = {label.casefold() for label in labels if label.casefold().endswith("_set")}
    direction_map = {}
    for part in str(getattr(source, "gate_direction", "") or "").split(","):
        bits = part.strip().rsplit(" ", 1)
        if len(bits) == 2:
            direction_map[bits[0].casefold()] = bits[1]
    raw_direction = str(getattr(source, "gate_direction", "") or "").strip()
    if len(labels) == 1 and raw_direction and "," not in raw_direction:
        direction_map.setdefault(labels[0].casefold(), raw_direction)
    parts = []
    for label, (low, high) in zip(labels, getattr(source, "gate_ranges", ()) or ()):
        lowered = label.casefold()
        if lowered.endswith("_meas") and lowered[:-5] + "_set" in set_labels:
            continue
        display_label = label[:-4] if lowered.endswith("_set") else label
        direction = direction_map.get(lowered, "")
        parts.append(
            f"{display_label} {direction} {low:.2f}–{high:.2f} V"
            .replace("  ", " ").strip()
        )
    if parts:
        return "; ".join(parts)
    return raw_direction if raw_direction and "," not in raw_direction else ""


def _drr_source_aux(source) -> str:
    """Return the small fourth line shown below a source summary."""
    modified_time = getattr(source, "modified_time", 0)
    modified = (
        datetime.fromtimestamp(float(modified_time)).strftime("%Y-%m-%d %H:%M")
        if np.isfinite(modified_time) and modified_time > 0 else "time unknown"
    )
    status = "PROCESSED" if bool(getattr(source, "processed", False)) else "UNPROCESSED"
    frame_count = getattr(source, "frame_count", None)
    frames = f"{frame_count} frames" if frame_count is not None else "frames unknown"
    return f"{status} · {frames} · {modified}"


class DrrSourceSummaryDelegate(QStyledItemDelegate):
    """Paint DRR picker summaries in at most four controlled rows."""

    def sizeHint(self, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        lines = str(index.data(Qt.DisplayRole) or "").splitlines()[:4]
        base = super().sizeHint(opt, index)
        line_height = max(1, opt.fontMetrics.height())
        return QSize(base.width(), max(base.height(), len(lines) * line_height + 10))

    def paint(self, painter: QPainter, option, index) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text = str(index.data(Qt.DisplayRole) or "")
        opt.text = ""
        if opt.widget is not None:
            opt.widget.style().drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)
        rect = option.rect.adjusted(8, 4, -8, -4)
        lines = text.splitlines()[:4]
        has_auxiliary_line = bool(index.data(Qt.UserRole + 5))
        painter.save()
        try:
            painter.setClipRect(rect)
            y_offset = 0
            for row, line in enumerate(lines):
                font = QFont(option.font)
                if has_auxiliary_line and row == len(lines) - 1:
                    font.setPointSize(max(7, font.pointSize() - 2))
                painter.setFont(font)
                metrics = painter.fontMetrics()
                line_rect = rect.adjusted(0, y_offset, 0, 0)
                line_rect.setHeight(metrics.height())
                if not (option.state & QStyle.State_Enabled):
                    text_color = option.palette.color(QPalette.Disabled, QPalette.Text)
                elif option.state & QStyle.State_Selected:
                    text_color = option.palette.highlightedText().color()
                else:
                    text_color = opt.palette.text().color()
                painter.setPen(text_color)
                painter.drawText(
                    line_rect,
                    Qt.AlignLeft | Qt.AlignVCenter,
                    metrics.elidedText(line, Qt.ElideRight, max(0, line_rect.width())),
                )
                y_offset += metrics.height()
        finally:
            painter.restore()


class DrrSessionList(QListWidget):
    """Lay out wrapped session filenames at their final viewport width."""

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Rows may have been measured by setCurrentRow while the dialog was
        # still hidden. Finish layout before the first paint, not on a later
        # catalog completion (which made clipped rows suddenly grow).
        self.doItemsLayout()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self.doItemsLayout()


def _drr_scroll_anchor(widget):
    for row in range(widget.count()):
        item = widget.item(row)
        rect = widget.visualItemRect(item)
        if rect.bottom() >= 0:
            return item.data(Qt.UserRole), rect.top()
    return None, 0


def _sync_drr_rows(widget, rows, *, preserve_view=False, select_first=False):
    """Update identities in place and keep the visible file at its pixel offset."""
    current = widget.currentItem()
    current_key = current.data(Qt.UserRole) if current is not None else None
    selected = {item.data(Qt.UserRole) for item in widget.selectedItems()}
    anchor_key, anchor_offset = _drr_scroll_anchor(widget)
    scroll = widget.verticalScrollBar()
    old_scroll = scroll.value()
    wanted = {item.data(Qt.UserRole) for item in rows}
    blocked = widget.blockSignals(True)
    updates = widget.updatesEnabled()
    widget.setUpdatesEnabled(False)
    try:
        for row in range(widget.count() - 1, -1, -1):
            if widget.item(row).data(Qt.UserRole) not in wanted:
                widget.takeItem(row)
        existing = {widget.item(i).data(Qt.UserRole): widget.item(i)
                    for i in range(widget.count())}
        roles = (Qt.DisplayRole, Qt.ToolTipRole, Qt.ForegroundRole,
                 Qt.BackgroundRole, Qt.FontRole, *range(int(Qt.UserRole), int(Qt.UserRole) + 6))
        for row, incoming in enumerate(rows):
            key = incoming.data(Qt.UserRole)
            item = existing.get(key)
            if item is None:
                widget.insertItem(row, incoming)
                existing[key] = incoming
                continue
            old_row = widget.row(item)
            if old_row != row:
                widget.insertItem(row, widget.takeItem(old_row))
            for role in roles:
                if item.data(role) != incoming.data(role):
                    item.setData(role, incoming.data(role))
            if item.flags() != incoming.flags():
                item.setFlags(incoming.flags())
        target = existing.get(current_key) if preserve_view or select_first else None
        if target is None and select_first and rows and (not preserve_view or current_key is None):
            target = widget.item(0)
            selected = {target.data(Qt.UserRole)}
        if widget.currentItem() is not target:
            widget.setCurrentItem(target, QItemSelectionModel.NoUpdate)
        for row in range(widget.count()):
            item = widget.item(row)
            item.setSelected(item.data(Qt.UserRole) in selected if preserve_view or select_first else False)
        widget.doItemsLayout()
        anchor = existing.get(anchor_key) if preserve_view else None
        if anchor is not None:
            scroll.setValue(scroll.value() + widget.visualItemRect(anchor).top() - anchor_offset)
        else:
            scroll.setValue(old_scroll if preserve_view else 0)
    finally:
        widget.blockSignals(blocked)
        widget.setUpdatesEnabled(updates)


def _add_copy_filename_menu(widget: QListWidget) -> None:
    widget.setContextMenuPolicy(Qt.CustomContextMenu)

    def show_menu(position) -> None:
        item = widget.itemAt(position)
        if item is None:
            return
        source_data = item.data(Qt.UserRole + 4)
        if isinstance(source_data, (list, tuple)):
            source = "\n".join(str(value) for value in source_data)
        else:
            source = str(source_data or item.data(Qt.UserRole) or item.text())
        menu = QMenu(widget)
        action = menu.addAction("Copy full filename")
        action.triggered.connect(lambda: QApplication.clipboard().setText(source))
        menu.exec(widget.viewport().mapToGlobal(position))

    widget.customContextMenuRequested.connect(show_menu)


class _DrrFitSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


def _drr_multi_lorentz_model(x: np.ndarray, *p: float) -> np.ndarray:
    x_arr = np.asarray(x, float)
    out = p[0] + p[1] * x_arr
    for i in range((len(p) - 2) // 3):
        amp, cen, gam = p[2 + 3*i], p[3 + 3*i], max(1e-12, p[4 + 3*i])
        out = out + amp * (gam * gam) / ((x_arr - cen) * (x_arr - cen) + gam * gam)
    return out


class _DrrFitWorker(QRunnable):
    def __init__(self, x: np.ndarray, y: np.ndarray, p0: list[float], lo: list[float], hi: list[float]) -> None:
        super().__init__()
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.p0, self.lo, self.hi = p0, lo, hi
        self.signals = _DrrFitSignals()

    def run(self) -> None:
        try:
            popt, _ = curve_fit(_drr_multi_lorentz_model, self.x, self.y,
                                p0=np.asarray(self.p0, float),
                                bounds=(np.asarray(self.lo, float), np.asarray(self.hi, float)),
                                maxfev=50000)
            self.signals.result.emit(np.asarray(popt, float))
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class DrrController:
    """Own DRR control reactions while sharing the application context."""

    def __init__(self, owner) -> None:
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_owner"), name)

    def __setattr__(self, name, value) -> None:
        if name == "_owner":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_owner"), name, value)

    def _drr_missing_sources(self, sources) -> list[str]:
        missing: list[str] = []
        for source in dict.fromkeys(str(value) for value in (sources or ())):
            try:
                exists = resolve_source_path(self.current_folder, source).is_file()
            except OSError:
                exists = False
            if not exists:
                missing.append(source)
        return missing

    def _repopulate_drr_yaxis(self) -> None:
        if not hasattr(self, "drr_yaxis_combo"):
            return
        first = self.drr_selected_files[0] if self.drr_selected_files else ""
        self._repopulate_yaxis_combo("drr", xlsx=data_io.is_xlsx_map_file(first))

    def _reject_mixed_xlsx_selection(self, selected: List[str]) -> None:
        if not selected:
            return
        xlsx = [name for name in selected if data_io.is_xlsx_map_file(name)]
        if not xlsx:
            return
        if len(xlsx) != 1 or len(selected) != 1:
            raise ValueError(
                "Select exactly one XLSX map for direct DR/R display "
                "(XLSX maps cannot be mixed with CSV measurements)."
            )

    def _drr_source_center(self, source_name: str) -> float | None:
        for source in self.drr_available_sources:
            if source.source == source_name and source.wavelength_center_nm is not None:
                return float(source.wavelength_center_nm)
        named = extract_wavelength_center_nm(source_name)
        if named is not None:
            return float(named)
        path = resolve_source_path(self.current_folder, source_name)
        if path.suffix.lower() == ".csv" and path.is_file():
            return inspect_csv_wavelength_center(path)
        return None

    def _restore_saved_drr_recipe(self) -> bool:
        resolved = resolve_drr_background_assignments(
            self.current_folder, self.drr_available_sources, self.drr_selected_files,
        )
        if resolved.resolved:
            self._drr_assignments = tuple(resolved.assignments)
            self._drr_assignments_automatic = True
            baseline_files = list(dict.fromkeys(
                source for assignment in resolved.assignments
                for source in assignment.baseline_files
            ))
            self.drr_baseline_files_manual = baseline_files
            self.drr_baseline_files_found = list(baseline_files)
            if baseline_files:
                blocked = self.drr_baseline_combo.blockSignals(True)
                self.drr_baseline_combo.setCurrentText("External")
                self.drr_baseline_combo.blockSignals(blocked)
                combine_text = {
                    "first": "First frame from each file, then average",
                    "last": "Last frame from each file, then average",
                    "all": "Average all frames in each file, then average files",
                }.get(resolved.assignments[0].baseline_which)
                if combine_text:
                    blocked = self.drr_baseline_combine_combo.blockSignals(True)
                    self.drr_baseline_combine_combo.setCurrentText(combine_text)
                    self.drr_baseline_combine_combo.blockSignals(blocked)
            else:
                blocked = self.drr_baseline_combo.blockSignals(True)
                self.drr_baseline_combo.setCurrentText(resolved.assignments[0].baseline_mode)
                self.drr_baseline_combo.blockSignals(blocked)
            self.drr_baseline_combine_combo.setEnabled(False)
            self.drr_pin_baseline_chk.setEnabled(False)
            self._update_drr_selection_labels()
            self._drr_background_guess = None
            self._status("Restored automatic DRR background assignments for each measurement.")
            return True
        self._status(resolved.reason or "Saved DRR background assignment is unavailable.")
        return False

    def _apply_drr_background_gate_default(self) -> bool:
        if not self.drr_baseline_files_manual:
            return True
        assessment = assess_background_gate_files(
            self.current_folder,
            self.drr_baseline_files_manual,
        )
        if not assessment.all_constant:
            return True
        if len(assessment.profiles) > 1 and not assessment.same_constant_values:
            self._invalidate_drr_for_background_selection(
                "Selected backgrounds have different constant gate values. "
                "Choose files from one gate condition."
            )
            return False
        blocked = self.drr_baseline_combine_combo.blockSignals(True)
        self.drr_baseline_combine_combo.setCurrentText(
            "Average all frames in each file, then average files"
        )
        self.drr_baseline_combine_combo.blockSignals(blocked)
        self._status(
            "Constant-gate background detected: averaging all frames per file"
            + (" and then averaging files." if len(assessment.profiles) > 1 else ".")
        )
        return True

    def _guess_drr_background_for_selection(self) -> bool:
        resolved = resolve_drr_background_assignments(
            self.current_folder, self.drr_available_sources, self.drr_selected_files,
        )
        if resolved.resolved:
            self._drr_assignments = tuple(resolved.assignments)
            self._drr_assignments_automatic = True
            baseline_files = list(dict.fromkeys(
                source for assignment in resolved.assignments
                for source in assignment.baseline_files
            ))
            self.drr_baseline_files_manual = baseline_files
            self.drr_baseline_files_found = list(baseline_files)
            blocked = self.drr_baseline_combo.blockSignals(True)
            self.drr_baseline_combo.setCurrentText("External")
            self.drr_baseline_combo.blockSignals(blocked)
            self.drr_baseline_combine_combo.setEnabled(False)
            self.drr_pin_baseline_chk.setEnabled(False)
            self._update_drr_selection_labels()
            self._drr_background_guess = None
            self._status("Automatically restored a validated DRR background for each measurement.")
            return True
        self._status(resolved.reason or "DRR background could not be resolved automatically.")
        return False

    def _drr_selected_wavelength_center(self) -> float | None:
        catalog_centers = {
            source.source: source.wavelength_center_nm for source in self.drr_available_sources
        }
        centers = [
            center
            for name in self.drr_selected_files
            if (
                center := catalog_centers.get(name)
                if catalog_centers.get(name) is not None
                else extract_wavelength_center_nm(name)
            ) is not None
        ]
        if not centers:
            return None
        first = float(centers[0])
        return first if all(wavelength_centers_match(first, center) for center in centers[1:]) else None

    def _drr_derivative_value(self) -> int | None:
        text = self.drr_derivative_combo.currentText()
        return None if text == "None" else (1 if text == "dE" else 2)

    def _enforce_drr_sg_constraints(self, *, show_status: bool) -> int:
        poly = int(self.drr_sg_poly_spin.value())
        req_win = int(self.drr_sg_window_spin.value())
        n_energy = (
            int(np.asarray(self.loaded.cube.energy).size)
            if self.loaded and self.loaded.mode == "DRR" and self.loaded.cube is not None
            else 401
        )
        used_win = clamp_sg_window(req_win, n_energy=n_energy, polyorder=poly)
        if used_win != req_win:
            blocked = self.drr_sg_window_spin.blockSignals(True)
            self.drr_sg_window_spin.setValue(used_win)
            self.drr_sg_window_spin.blockSignals(blocked)
            if show_status:
                self._status(f"State: SG window clamped to {used_win} (odd, valid for order={poly}).")
        return used_win

    def _drr_cube_with_metadata(
        self, derivative=_CURRENT_DRR_DERIVATIVE,
    ) -> tuple[DataCube, int | None, int, int]:
        """Return the DRR cube plus the derivative parameters actually applied."""
        if not self.loaded or self.loaded.mode != "DRR" or self.loaded.cube is None:
            raise ValueError("No DRR data loaded.")
        deriv = (
            self._drr_derivative_value()
            if derivative is _CURRENT_DRR_DERIVATIVE
            else derivative
        )
        if deriv not in (None, 1, 2):
            raise ValueError("DRR derivative must be None, 1, or 2.")
        poly = int(self.drr_sg_poly_spin.value())
        req_win = self._enforce_drr_sg_constraints(show_status=True)
        cache_key = (id(self.loaded.cube), deriv, int(req_win), poly)
        cached = self._drr_derivative_cache.get(cache_key)
        if cached is not None:
            cube, used_win = cached
            return cube, deriv, used_win, poly
        cube, used_win = apply_sg_derivative_energy(
            self.loaded.cube, derivative=deriv, window_length=req_win, polyorder=poly,
        )
        if cube is not self.loaded.cube:
            # Derivative processing creates a new DataCube.  Carry the source
            # axis semantics through so DAT metadata and linecut labels remain
            # truthful for both paired products.
            cube.gate_unit = getattr(self.loaded.cube, "gate_unit", "")
            cube.y_axis_semantic = getattr(self.loaded.cube, "y_axis_semantic", "")
        if len(self._drr_derivative_cache) >= 12:
            self._drr_derivative_cache.pop(next(iter(self._drr_derivative_cache)))
        self._drr_derivative_cache[cache_key] = (cube, int(used_win))
        if deriv is not None and used_win != req_win:
            self._status(f"State: SG window adjusted to {used_win}.")
        return cube, deriv, used_win, poly

    def _drr_cube_for_display(self) -> DataCube:
        cube, _deriv, _used_win, _poly = self._drr_cube_with_metadata()
        return cube

    def _drr_baseline_key(self) -> str:
        text = self.drr_baseline_combo.currentText()
        if text == "Self (first frame)":
            return "self_first"
        if text == "External":
            return f"external_{self.drr_baseline_combine_combo.currentText()}"
        return "self_last"

    def _read_drr_params(self):
        s = self.drr_spins
        assignments = tuple(
            assignment.to_dict() if hasattr(assignment, "to_dict") else dict(assignment)
            for assignment in getattr(self, "_drr_assignments", ())
        )
        baseline_mode = self.drr_baseline_combo.currentText()
        if assignments and self._drr_assignments_automatic:
            baseline_mode = "Automatic"
        return {
            "baseline_mode": baseline_mode,
            "baseline_which": self.drr_baseline_combine_combo.currentText(),
            "baseline_files": tuple(self.drr_baseline_files_manual),
            "selected_files": tuple(self.drr_selected_files),
            "y_axis_spec": self._selected_y_axis_spec("drr"),
            "derivative": self.drr_derivative_combo.currentText(),
            "sg_window": int(self.drr_sg_window_spin.value()),
            "sg_poly": int(self.drr_sg_poly_spin.value()),
            "cmap": self._resolved_cmap(self.drr_cmap),
            "vmin": float(s["vmin"].value()),
            "vmax": float(s["vmax"].value()),
            "xmin": float(s["xmin"].value()),
            "xmax": float(s["xmax"].value()),
            "ymin": float(s["ymin"].value()),
            "ymax": float(s["ymax"].value()),
            "gate": self._drr_gate_value(),
            "log": bool(self.drr_log_chk.isChecked()),
            "clip": bool(self.drr_clip_chk.isChecked()),
            "center_zero": bool(self.drr_center_zero_chk.isChecked()),
            "drr_assignments": assignments,
        }

    def _is_drr_gate_only_change(self, new_key: tuple) -> bool:
        if self._last_plot_params_key is None or self._last_plot_cube is None:
            return False
        if len(new_key) != len(self._last_plot_params_key):
            return False
        gate_idx = 17
        return (
            new_key[:gate_idx] == self._last_plot_params_key[:gate_idx]
            and new_key[gate_idx + 1 :] == self._last_plot_params_key[gate_idx + 1 :]
            and new_key[gate_idx] != self._last_plot_params_key[gate_idx]
        )

    def _ensure_gate_line(self, cube: DataCube, gate_value: float) -> None:
        axes = getattr(self, "_drr_heatmap_axes", {}) or {}
        if not axes and self._drr_heatmap_ax is not None:
            axes = {"active": self._drr_heatmap_ax}
        if not axes:
            return
        gate = np.asarray(cube.gate, float).ravel()
        gate_clamped = float(np.clip(gate_value, float(np.nanmin(gate)), float(np.nanmax(gate))))
        lines = getattr(self, "_drr_gate_lines", {})
        for key, axis in axes.items():
            line = lines.get(key)
            if line is None or getattr(line, "axes", None) is not axis:
                line = axis.axhline(
                    y=gate_clamped, lw=1.2, alpha=0.9, color="#222",
                    linestyle="--", zorder=20,
                )
                lines[key] = line
            else:
                line.set_ydata([gate_clamped, gate_clamped])
                line.set_linestyle("--")
        self._drr_gate_lines = lines
        self._gate_line = lines.get("raw") or lines.get("active") or next(iter(lines.values()), None)

    def _on_drr_derivative_changed(self) -> None:
        update_label = getattr(self, "_update_drr_advanced_derivative_label", None)
        if update_label is not None:
            update_label()
        self._invalidate_pending_drr_fit("Fit discarded: derivative changed.")
        self._invalidate_export_move_sources()
        derivative_active = self._drr_derivative_value() is not None
        self.drr_sg_window_spin.setVisible(derivative_active)
        self.drr_sg_poly_spin.setVisible(derivative_active)
        self._enforce_drr_sg_constraints(show_status=True)
        if self.loaded and self.loaded.mode == "DRR" and not self._suspend_drr_autoplot:
            self._refresh_automatic_ranges("DRR", refresh_split=True)
            self._schedule_plot_redraw("DRR")

    def _invalidate_drr_analysis_product(self) -> None:
        """Discard fit/peak results when switching the displayed product."""
        self._invalidate_pending_drr_fit("Fit discarded: DRR product changed.")
        self._drr_peak_gate = None
        self._drr_peak_indices = None
        self._drr_fit_gate = None
        self._drr_fit_x = None
        self._drr_fit_y = None
        self._drr_fit_centers = None
        status = getattr(self, "drr_fit_status", None)
        if status is not None:
            status.setText("")

    def _on_drr_plot_param_changed(self, source=None) -> None:
        if source in {
            self.drr_spins.get("xmin"), self.drr_spins.get("xmax"),
            self.drr_spins.get("ymin"), self.drr_spins.get("ymax"),
        }:
            # Manual range edits are authoritative over the last Matplotlib
            # zoom snapshot.  Shared raw/d2 axes receive the same limits.
            self._drr_view_limits = (
                (float(self.drr_spins["xmin"].value()), float(self.drr_spins["xmax"].value())),
                (float(self.drr_spins["ymin"].value()), float(self.drr_spins["ymax"].value())),
            )
            self._drr_limits_from_controls = True
        if source is getattr(self, "drr_baseline_combo", None):
            self._drr_baseline_user_selected = True
            # A baseline-mode change is an explicit recipe change.  Clear
            # every previous per-measurement assignment, including explicit
            # mappings, so Self cannot accidentally retain external files.
            if self._drr_assignments:
                self._drr_assignments = ()
            self._drr_assignments_automatic = False
            self.drr_baseline_combine_combo.setEnabled(True)
            self.drr_pin_baseline_chk.setEnabled(True)
            if self.drr_baseline_combo.currentText().startswith("Self"):
                self.drr_baseline_files_manual = []
                self.drr_baseline_files_found = []
                blocked = self.drr_pin_baseline_chk.blockSignals(True)
                self.drr_pin_baseline_chk.setChecked(False)
                self.drr_pin_baseline_chk.blockSignals(blocked)
        self._invalidate_pending_drr_fit("Fit discarded: plot range or display settings changed.")
        self._invalidate_export_move_sources()
        external_baseline = self.drr_baseline_combo.currentText() == "External"
        self._update_drr_baseline_controls()
        if source is getattr(self, "drr_baseline_combo", None):
            self._update_drr_selection_labels()
        if external_baseline and not self.drr_baseline_files_manual:
            self._invalidate_drr_for_background_selection(
                "Select an external background before processing."
            )
            return
        if (
            source is getattr(self, "drr_baseline_combo", None)
            and not external_baseline
            and self.drr_selected_files
            and self.current_folder
            and not self._suspend_drr_autoplot
            and (not self.loaded or self.loaded.mode != "DRR")
        ):
            self._start_load("DRR")
            return
        if self.loaded and self.loaded.mode == "DRR" and not self._suspend_drr_autoplot:
            sender = source if source is not None else self.sender()
            if sender in (
                self.drr_spins["xmin"], self.drr_spins["xmax"],
                self.drr_spins["ymin"], self.drr_spins["ymax"],
                self.drr_log_chk, self.drr_center_zero_chk,
            ):
                self._pending_range_refresh["DRR"] = (
                    bool(self._pending_range_refresh.get("DRR", False))
                    or sender in (self.drr_spins["xmin"], self.drr_spins["xmax"])
                )
            gate_only = sender is self.drr_spins.get("gate")
            self._schedule_plot_redraw("DRR", delay_ms=0 if gate_only else 90)

    def _on_drr_baseline_mode_changed(self) -> None:
        if self._drr_assignments:
            # The visible frame selector is only a summary for automatic
            # assignments.  A user change must opt into one common recipe;
            # otherwise it would silently reuse stale per-file assignments.
            self._drr_assignments = ()
            self._drr_assignments_automatic = False
            self.drr_baseline_combine_combo.setEnabled(True)
            self.drr_pin_baseline_chk.setEnabled(True)
            self._status("Frame method changed; automatic per-measurement backgrounds were cleared.")
        self._status(f"State: Baseline mode set: {self.drr_baseline_combine_combo.currentText()}.")
        self._update_drr_selection_labels()
        if self.loaded and self.loaded.mode == "DRR" and not self._suspend_drr_autoplot:
            self._schedule_plot_redraw("DRR")

    def _auto_drr_vrange(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        if getattr(self._owner, "_drr_plot_view", "raw") == "second":
            self._owner._auto_drr_second_vrange()
            return
        cube = self._drr_cube_for_display()
        x = np.asarray(cube.energy, float).ravel()
        y = np.asarray(cube.gate, float).ravel()
        z = np.asarray(cube.Z, float)
        x0, x1 = sorted((float(self.drr_spins["xmin"].value()), float(self.drr_spins["xmax"].value())))
        y0, y1 = sorted((float(self.drr_spins["ymin"].value()), float(self.drr_spins["ymax"].value())))
        x_mask = (x >= x0) & (x <= x1)
        y_mask = (y >= y0) & (y <= y1)
        if np.any(y_mask) and np.any(x_mask):
            z_roi = z[np.ix_(y_mask, x_mask)]
        else:
            z_roi = z
        finite = z_roi[np.isfinite(z_roi)]
        if finite.size == 0:
            self._status("State: Auto vmin/vmax skipped (no finite values in selected x/y range).")
            return
        if self._mode_log("DRR"):
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
        spins = self.drr_spins
        spins["vmin"].setValue(vmin)
        spins["vmax"].setValue(vmax)
        self._status(f"State: Auto vmin/vmax (ROI) = {vmin:.4g}, {vmax:.4g}")
        self._schedule_plot_redraw("DRR")

    def _auto_drr_xrange(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        cube = self._drr_cube_for_display()
        self.drr_spins["xmin"].setValue(float(np.nanmin(cube.energy)))
        self.drr_spins["xmax"].setValue(float(np.nanmax(cube.energy)))
        self._status("State: Auto xmin/xmax set from energy axis.")
        self._schedule_plot_redraw("DRR")

    def _auto_drr_yrange(self) -> None:
        if not self.loaded or self.loaded.mode != "DRR":
            return
        cube = self._drr_cube_for_display()
        self.drr_spins["ymin"].setValue(float(np.nanmin(cube.gate)))
        self.drr_spins["ymax"].setValue(float(np.nanmax(cube.gate)))
        self._status("State: Auto ymin/ymax set from gate axis.")
        self._schedule_plot_redraw("DRR")
    def _update_drr_selection_labels(self) -> None:
        def _brief(names: List[str]) -> str:
            if not names:
                return "none"
            # Zero-width break opportunities preserve the complete first name
            # while allowing underscore-heavy measurement names to wrap.
            first = Path(names[0]).name.replace("_", "_\u200b").replace("-", "-\u200b")
            return first if len(names) == 1 else f"{first} (+{len(names) - 1} more)"

        mode_map = {
            "Last frame from each file, then average": "last",
            "First frame from each file, then average": "first",
            "Average all frames in each file, then average files": "all frames",
        }
        mode_short = mode_map.get(self.drr_baseline_combine_combo.currentText(), "last")
        missing_measurements = self._drr_missing_sources(self.drr_selected_files)
        measurement_label = _brief(self.drr_selected_files)
        if missing_measurements:
            measurement_label += " · Missing: " + ", ".join(Path(name).name for name in missing_measurements)
        self.drr_measurement_summary.setText(f"Measurement: {len(self.drr_selected_files)} files ({measurement_label})")
        self.drr_measurement_summary.setToolTip(
            "Selected measurement files:\n" + "\n".join(self.drr_selected_files)
            if self.drr_selected_files
            else "No measurement files selected."
        )
        if self.drr_baseline_combo.currentText().startswith("Self"):
            frame = "first" if "first" in self.drr_baseline_combo.currentText() else "last"
            self.drr_baseline_summary.setText(f"Using {frame} frame")
            self.drr_baseline_summary.setToolTip(
                f"Using the {frame} frame from each selected measurement as its own background."
            )
        elif self._drr_assignments_automatic and self._drr_assignments:
            mapping = "\n".join(
                f"{Path(item.measurement_file).name} ← "
                + (", ".join(item.baseline_files) if item.baseline_files else item.baseline_mode)
                + f" [{item.baseline_which}]"
                for item in self._drr_assignments
            )
            self.drr_baseline_summary.setText(
                f"Backgrounds: automatic per measurement ({len(self._drr_assignments)} mappings)"
            )
            self.drr_baseline_summary.setToolTip(
                "Automatic per-measurement background mapping:\n" + mapping
            )
        else:
            missing_baselines = self._drr_missing_sources(self.drr_baseline_files_manual)
            baseline_label = (
                "\nMissing: " + ", ".join(Path(name).name for name in missing_baselines)
                if missing_baselines else ""
            )
            self.drr_baseline_summary.setText(
                f"Baselines: {len(self.drr_baseline_files_manual)} files (mode: {mode_short}){baseline_label}"
            )
            self.drr_baseline_summary.setToolTip(
                "Selected baseline files:\n" + "\n".join(self.drr_baseline_files_manual)
                if self.drr_baseline_files_manual
                else "No external baseline files selected."
            )
        self._update_drr_baseline_controls()
        self._repopulate_drr_yaxis()

    def _update_drr_baseline_controls(self) -> None:
        """Keep the baseline area stable while enabling its active recipe."""
        external = self.drr_baseline_combo.currentText() == "External"
        automatic = bool(external and self._drr_assignments_automatic and self._drr_assignments)
        self.drr_external_baseline_row.setVisible(True)
        self.drr_baseline_combine_combo.setVisible(True)
        self.drr_pin_baseline_chk.setVisible(True)
        self.drr_edit_baselines_btn.setEnabled(external)
        self.drr_baseline_autofind_btn.setEnabled(external)
        self.drr_baseline_combine_combo.setEnabled(external and not automatic)
        self.drr_pin_baseline_chk.setEnabled(external and not automatic)
    def _edit_drr_measurements(self) -> None:
        previous = list(self.drr_selected_files)
        selected = self._open_drr_source_dialog(
            title="Choose DRR Measurement Group",
            selected=self.drr_selected_files,
            baseline_mode=False,
        )
        self._reject_mixed_xlsx_selection(selected)
        self.drr_selected_files = selected
        measurement_changed = selected != previous
        if measurement_changed and not self.drr_pin_baseline_chk.isChecked():
            self._drr_assignments = ()
            self._drr_assignments_automatic = False
            self._drr_baseline_user_selected = False
            self.drr_baseline_files_manual = []
            self.drr_baseline_files_found = []
            self._drr_background_guess = None
            restored = self._restore_saved_drr_recipe()
            if (
                not restored
                and self.drr_baseline_combo.currentText() == "External"
                and not self.drr_baseline_files_manual
            ):
                # A previous unpinned External choice belongs to the old
                # measurement.  Once that measurement changes its baseline
                # files are intentionally cleared; leaving External selected
                # would make the new selection fail before the load worker
                # starts.  Fall back to the safe per-file Self recipe so a
                # back-gate sweep remains directly plottable.
                blocked = self.drr_baseline_combo.blockSignals(True)
                self.drr_baseline_combo.setCurrentText("Self (last frame)")
                self.drr_baseline_combo.blockSignals(blocked)
        self._update_drr_selection_labels()
        if measurement_changed:
            self._clear_loaded_drr_view()
        if (
            measurement_changed
            and self.drr_baseline_combo.currentText() == "External"
            and not self.drr_baseline_files_manual
        ):
            self._invalidate_drr_for_background_selection(
                "Measurement changed. Select an external background."
            )
            return
        if self.drr_selected_files:
            self._start_load("DRR")
    def _clear_drr_measurements(self) -> None:
        self.drr_selected_files = []
        self._drr_assignments = ()
        self._drr_assignments_automatic = False
        self._drr_baseline_user_selected = False
        self.drr_baseline_combine_combo.setEnabled(True)
        self.drr_pin_baseline_chk.setEnabled(True)
        if not self.drr_pin_baseline_chk.isChecked():
            self.drr_baseline_files_manual = []
        self._update_drr_selection_labels()
        self._clear_loaded_drr_view()
        self._set_stage("No DRR measurement")
        self._update_action_states()
    def _edit_drr_baselines_dialog(self) -> None:
        self.drr_baseline_files_manual = self._open_drr_source_dialog(
            title="Choose Historical or External Baseline",
            selected=self.drr_baseline_files_manual,
            baseline_mode=True,
        )
        self._drr_background_guess = None
        self._drr_assignments = ()
        self._drr_assignments_automatic = False
        self.drr_baseline_combine_combo.setEnabled(True)
        self.drr_pin_baseline_chk.setEnabled(True)
        if self.drr_baseline_files_manual:
            blocked = self.drr_baseline_combo.blockSignals(True)
            self.drr_baseline_combo.setCurrentText("External")
            self.drr_baseline_combo.blockSignals(blocked)
        self._update_drr_selection_labels()
        if not self._apply_drr_background_gate_default():
            return
        if self.drr_selected_files:
            self._start_load("DRR")
    def _clear_drr_baselines(self) -> None:
        self.drr_baseline_files_manual = []
        self.drr_baseline_files_found = []
        self._drr_assignments = ()
        self._drr_assignments_automatic = False
        self.drr_baseline_combine_combo.setEnabled(True)
        self.drr_pin_baseline_chk.setEnabled(True)
        self._drr_background_guess = None
        self.drr_pin_baseline_chk.setChecked(False)
        self._update_drr_selection_labels()
        if self.drr_baseline_combo.currentText() == "External":
            self._invalidate_drr_for_background_selection(
                "External background cleared. Select a background before processing."
            )
    def _on_drr_pin_baseline_toggled(self, checked: bool) -> None:
        if checked and self._drr_assignments_automatic:
            blocked = self.drr_pin_baseline_chk.blockSignals(True)
            self.drr_pin_baseline_chk.setChecked(False)
            self.drr_pin_baseline_chk.blockSignals(blocked)
            self._status("Automatic per-measurement backgrounds cannot be pinned as one common background.")
            return
        if checked and not self.drr_baseline_files_manual:
            blocked = self.drr_pin_baseline_chk.blockSignals(True)
            self.drr_pin_baseline_chk.setChecked(False)
            self.drr_pin_baseline_chk.blockSignals(blocked)
            self._status("Select an external background before pinning it.")
            return
        if checked:
            self._drr_assignments_automatic = False
        self._status("External background pinned." if checked else "External background follows measurement selection.")
    def _open_drr_source_dialog(
        self,
        *,
        title: str,
        selected: List[str],
        baseline_mode: bool,
    ) -> List[str]:
        """Browse recent DRR groups without flattening the complete device history."""
        dlg = QDialog(self._owner)
        dlg.setWindowTitle(title)
        if not self.windowIcon().isNull():
            dlg.setWindowIcon(self.windowIcon())
        dlg.setMinimumSize(920, 560)
        dlg.resize(1120, 680)
        layout = QVBoxLayout(dlg)

        hint_text = (
            "Background history includes earlier measurement groups; select any compatible file or group."
            if baseline_mode
            else "The newest unprocessed measurement group is shown first. Search to reach older sessions."
        )

        filter_row = QHBoxLayout()
        filter_edit = QLineEdit()
        filter_edit.setPlaceholderText("Search group, date, or filename...")
        show_all = QCheckBox("Show all history")
        unprocessed_only = QCheckBox("Unprocessed only")
        if baseline_mode:
            unprocessed_only.setChecked(False)
        else:
            unprocessed_only.setChecked(
                self.settings.value("drr/source_unprocessed_only", True, type=bool)
            )

            def _remember_unprocessed_only(checked: bool) -> None:
                self.settings.setValue("drr/source_unprocessed_only", bool(checked))
                self.settings.sync()

            unprocessed_only.toggled.connect(_remember_unprocessed_only)
        unprocessed_only.setVisible(not baseline_mode)
        include_backgrounds = QCheckBox("Include background candidates")
        include_backgrounds.setVisible(not baseline_mode)
        include_backgrounds.setToolTip(
            "Show files classified as backgrounds so an unusual constant-gate measurement can be restored manually."
        )
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Scan Initial Data and its measurement folders for new files.")
        filter_row.addWidget(QLabel("Find"))
        filter_row.addWidget(filter_edit, 1)
        filter_row.addWidget(unprocessed_only)
        filter_row.addWidget(refresh_btn)
        filters_btn = QPushButton("Filters")
        filters_btn.setCheckable(True)
        filters_btn.setToolTip(hint_text)
        filter_row.addWidget(filters_btn)
        layout.addLayout(filter_row)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Data type"))
        type_combo = QComboBox()
        type_combo.setObjectName("drr_source_type_combo")
        type_combo.addItem("REF", "REF")
        type_combo.addItem("All data", "All")
        type_combo.setCurrentIndex(1 if getattr(self._owner, "_drr_include_all_sources", False) else 0)
        type_combo.setToolTip("Filter files by PL source marker. All data includes PL, REF, and Unknown.")
        type_row.addWidget(type_combo)
        type_container = QWidget()
        type_container.setLayout(type_row)

        status_row = QHBoxLayout()
        type_hint = QLabel()
        type_hint.setWordWrap(True)
        type_hint.setVisible(False)
        type_hint.setObjectName("drrSourceTypeHint")
        status_row.addWidget(type_hint, 1)
        empty_hint = QLabel()
        empty_hint.setWordWrap(True)
        empty_hint.setObjectName("drrSourceEmptyHint")
        status_row.addWidget(empty_hint, 1)
        layout.addLayout(status_row)

        filters_panel = QWidget()
        filters_layout = QHBoxLayout(filters_panel)
        filters_layout.setContentsMargins(0, 0, 0, 0)
        filters_layout.addWidget(include_backgrounds)
        filters_layout.addWidget(show_all)
        filters_layout.addWidget(type_container)
        filters_layout.addStretch(1)
        filters_panel.setVisible(False)
        layout.addWidget(filters_panel)

        def _update_filters_button() -> None:
            active = int(include_backgrounds.isChecked()) + int(show_all.isChecked())
            active += int(type_combo.currentData() == "All")
            filters_btn.setText(f"Filters ({active})" if active else "Filters")

        filters_btn.toggled.connect(filters_panel.setVisible)
        include_backgrounds.toggled.connect(lambda _checked: _update_filters_button())
        show_all.toggled.connect(lambda _checked: _update_filters_button())
        type_combo.currentIndexChanged.connect(lambda _index: _update_filters_button())
        _update_filters_button()

        panes = QSplitter(Qt.Horizontal)
        group_list = DrrSessionList()
        file_list = QListWidget()
        selected_list = QListWidget()
        group_list.setObjectName("drr_source_group_list")
        file_list.setObjectName("drr_source_file_list")
        selected_list.setObjectName("drr_source_chosen_list")
        for widget in (group_list, file_list, selected_list):
            SourcePickerDialog.configure_source_list(
                widget,
                selection_mode=QAbstractItemView.ExtendedSelection,
                spacing=3,
            )
            widget.setWordWrap(False)
            widget.setTextElideMode(Qt.ElideNone)
            widget.setItemDelegate(DrrSourceSummaryDelegate(widget))
            _add_copy_filename_menu(widget)

        # Session rows must expose every actual filename, including suffixes
        # omitted by the grouping title. Grow only these rows when necessary.
        group_list.setItemDelegate(WrappedFilenameDelegate(group_list))
        group_list.setWordWrap(True)
        group_list.setResizeMode(QListWidget.Adjust)

        chosen_label = QLabel("Chosen files (0)")

        def _panel(label: str | QLabel, widget: QListWidget) -> QWidget:
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.addWidget(QLabel(label) if isinstance(label, str) else label)
            panel_layout.addWidget(widget, 1)
            return panel

        panes.addWidget(_panel("Data history" if baseline_mode else "Measurement sessions", group_list))
        panes.addWidget(_panel("Files in selected session", file_list))
        panes.addWidget(_panel(chosen_label, selected_list))
        panes.setChildrenCollapsible(False)
        panes.setStretchFactor(0, 3)
        panes.setStretchFactor(1, 5)
        panes.setStretchFactor(2, 2)
        panes.setSizes([320, 520, 260])
        layout.addWidget(panes, 1)

        details_btn = QPushButton("Details")
        details_btn.setCheckable(True)
        details_btn.setChecked(False)
        details_btn.setToolTip("Show metadata for the selected group or file.")
        layout.addWidget(details_btn)
        group_detail = QLabel()
        group_detail.setWordWrap(True)
        group_detail.setObjectName("drrSourceGroupDetail")
        group_detail.setVisible(False)
        details_btn.toggled.connect(group_detail.setVisible)
        layout.addWidget(group_detail)

        action_row = QHBoxLayout()
        add_group_btn = QPushButton("Add Entire Group")
        add_files_btn = QPushButton("Add Selected Files")
        add_compatible_btn = QPushButton("Add Compatible Repeats")
        add_compatible_btn.setToolTip(
            "Add repeats matching the selected reference file's full gate and spectral grids."
        )
        add_compatible_btn.setVisible(not baseline_mode)
        remove_btn = QPushButton("Remove")
        clear_btn = QPushButton("Clear")
        browse_btn = QPushButton("Browse File Anywhere...")
        browse_btn.setVisible(baseline_mode)
        action_row.addWidget(add_group_btn)
        action_row.addWidget(add_files_btn)
        action_row.addWidget(add_compatible_btn)
        action_row.addWidget(browse_btn)
        action_row.addStretch(1)
        action_row.addWidget(remove_btn)
        action_row.addWidget(clear_btn)
        layout.addLayout(action_row)

        def _source_kind(source) -> str:
            source_path = source.source if hasattr(source, "source") else str(source)
            return data_io.classify_pl_source(source_path)

        def _source_label(source) -> str:
            source_path = source.source if hasattr(source, "source") else str(source)
            suffix = " · XLSX map" if data_io.is_xlsx_map_file(source_path) else ""
            return f"{_source_kind(source)}{suffix}"

        def _source_state_alias(source, *, missing: bool = False) -> str:
            """Resolve row color while retaining DRR's special source states."""
            if missing:
                return "danger_foreground"
            if not hasattr(source, "source"):
                return "text_tertiary"
            if bool(getattr(source, "is_background", False)) or getattr(source, "classification", "") in {
                "background", "likely_background"
            }:
                return "text_tertiary"
            return (
                "source_processed_foreground"
                if bool(getattr(source, "processed", False))
                else "source_new_foreground"
            )

        def _style_source_item(item, source, *, missing: bool = False) -> None:
            item.setForeground(QColor(theme_alias(_source_state_alias(source, missing=missing))))
            font = item.font()
            font.setBold(
                bool(missing)
                or (
                    hasattr(source, "source")
                    and not bool(getattr(source, "is_background", False))
                    and getattr(source, "classification", "") not in {"background", "likely_background"}
                    and not bool(getattr(source, "processed", False))
                )
            )
            item.setFont(font)

        baseline_recommendations = {}

        def _baseline_measurements():
            selected_paths = {str(path) for path in self.drr_selected_files}
            catalog_measurements = tuple(
                source for source in self.drr_available_sources
                if source.source in selected_paths and not source.is_background
            )
            if len(catalog_measurements) == len(selected_paths):
                return catalog_measurements
            # Selected files outside the accepted partition require a worker
            # catalog refresh. Never inspect headers/stat from this GUI path.
            queue_refresh = getattr(self, "_queue_drr_catalog_refresh", None)
            if callable(queue_refresh):
                queue_refresh(auto=True, old_source_files=set(selected_paths),
                              selected_sources=tuple(selected_paths))
            return ()

        def _catalog_groups():
            nonlocal baseline_recommendations
            catalog_sources = (
                [
                    source
                    for source in self.drr_available_sources
                    if Path(source.source).suffix.lower() == ".csv"
                ]
                if baseline_mode
                else self.drr_available_sources
            )
            if baseline_mode:
                all_csv = tuple(catalog_sources)
                baseline_recommendations = candidate_recommendation_map(
                    _baseline_measurements(), all_csv
                )
                catalog_sources = [
                    source for source in all_csv
                    if source.source in baseline_recommendations
                ]
            selected_type = str(type_combo.currentData() or "REF")
            if selected_type != "All":
                catalog_sources = [
                    source for source in catalog_sources if _source_kind(source) == selected_type
                ]
            result = group_drr_sources(catalog_sources)
            if baseline_mode:
                order = {source: index for index, source in enumerate(baseline_recommendations)}
                result = [
                    replace(
                        group,
                        files=tuple(sorted(
                            group.files,
                            key=lambda source: (
                                order.get(source.source, 10**9),
                                source.filename.casefold(),
                            ),
                        )),
                    )
                    for group in result
                ]
                result.sort(key=lambda group: (
                    min((order.get(source.source, 10**9) for source in group.files), default=10**9),
                    -group.modified_time,
                    group.title.casefold(),
                ))
            return result

        groups = _catalog_groups()
        groups_by_key = {group.key: group for group in groups}
        measurement_center = self._drr_selected_wavelength_center()

        def _group_search_text(group) -> str:
            return " ".join(
                [group.title, group.session_date, *(source.filename for source in group.files)]
            ).casefold()

        group_search_text = {group.key: _group_search_text(group) for group in groups}

        def _path_matches_selected_measurements(source: str) -> bool:
            if source in baseline_recommendations:
                return True
            measurements = _baseline_measurements()
            if not measurements:
                return False
            try:
                grid = np.asarray(
                    inspect_csv_spectral_grid(resolve_source_path(self.current_folder, source)),
                    dtype=float,
                )
            except (OSError, ValueError, TypeError):
                return False
            return all(
                np.asarray(measurement.spectral_grid, dtype=float).shape == grid.shape
                and np.allclose(
                    np.asarray(measurement.spectral_grid, dtype=float),
                    grid, rtol=1e-9, atol=1e-10,
                )
                for measurement in measurements
            )

        def _add_chosen(source: str, *, allow_existing: bool = False) -> bool:
            existing = {
                str(selected_list.item(index).data(Qt.UserRole) or selected_list.item(index).text())
                for index in range(selected_list.count())
            }
            if source in existing:
                return True
            incompatible = baseline_mode and not _path_matches_selected_measurements(source)
            if incompatible and not allow_existing:
                self._status(
                    f"Not added: {Path(source).name} does not match every selected measurement spectral grid."
                )
                return False
            catalog_source = next(
                (entry for entry in self.drr_available_sources if entry.source == source),
                source,
            )
            peers = tuple(
                entry for entry in self.drr_available_sources
                if entry.source != source
                and entry.group_key == getattr(catalog_source, "group_key", "")
            )
            peer_condition_values = tuple(
                _drr_condition_values(entry) for entry in (catalog_source, *peers)
            )
            item = QListWidgetItem(
                format_drr_source_summary(
                    catalog_source,
                    peers,
                    peer_condition_values=peer_condition_values,
                )
                + "\n"
                + _drr_source_aux(catalog_source)
            )
            item.setData(Qt.UserRole, source)
            item.setData(Qt.UserRole + 4, source)
            item.setData(Qt.UserRole + 5, True)
            item.setData(Qt.UserRole + 1, bool(incompatible))
            item.setData(Qt.UserRole + 2, bool(allow_existing))
            full_detail = f"{source}\n{format_drr_source_summary(catalog_source, peers)}"
            item.setData(Qt.UserRole + 3, full_detail)
            item.setToolTip(full_detail)
            selected_list.addItem(item)
            return True

        selected_list.setUpdatesEnabled(False)
        try:
            for source in selected:
                _add_chosen(source, allow_existing=True)
        finally:
            selected_list.setUpdatesEnabled(True)

        def _selected_group():
            item = group_list.currentItem()
            return groups_by_key.get(str(item.data(Qt.UserRole))) if item is not None else None

        def _compact_gate_text(source) -> str:
            return _compact_drr_gate_text(source) or "gate range unknown"

        def _show_selected_detail(_current=None, _previous=None) -> None:
            item = file_list.currentItem()
            if item is None:
                return
            group_detail.setText(str(item.data(Qt.UserRole + 3) or ""))

        def _populate_files(*, preserve_view=False) -> None:
            group = _selected_group()
            rows = []
            file_list.setUpdatesEnabled(False)
            try:
                if group is None:
                    _sync_drr_rows(file_list, [], preserve_view=preserve_view)
                    group_detail.clear()
                    return
                frame_text = (
                    f"{group.frame_count_range[0]}–{group.frame_count_range[1]}"
                    if group.frame_count_range else "unknown"
                )
                modes = (
                    f" · saved baseline modes: {', '.join(group.saved_baseline_modes)}"
                    if group.saved_baseline_modes else ""
                )
                source_by_path = {source.source: source for source in self.drr_available_sources}
                def _linked_label(path: str) -> str:
                    linked = source_by_path.get(path)
                    if linked is None:
                        return f"{path} (gate details unavailable)"
                    if linked.gate_ranges:
                        gate_text = ", ".join(
                            f"{label} {low:g}–{high:g}"
                            for label, (low, high) in zip(linked.gate_labels, linked.gate_ranges)
                        )
                    elif linked.gate_grid:
                        first_frame = linked.gate_grid[0]
                        gate_text = "first frame: " + ", ".join(
                            f"{label}={first_frame[index]:g}"
                            for index, label in enumerate(linked.gate_labels)
                            if index < len(first_frame)
                        )
                    else:
                        gate_text = "gate details unavailable"
                    return f"{path} [{gate_text}]"
                links = (
                    " · linked backgrounds: " + ", ".join(
                        _linked_label(path) for path in group.linked_backgrounds
                    )
                    if group.linked_backgrounds else ""
                )
                gate_ranges = (
                    " · first file gate ranges: " + ", ".join(
                        f"{label} {low:g}–{high:g}"
                        for label, (low, high) in zip(group.gate_labels, group.gate_ranges)
                    )
                    if group.gate_ranges else ""
                )
                group_detail.setText(
                    f"{group.title} · processed {group.processed_count}/{len(group.files)}"
                    f" · frames/file {frame_text} · gate direction {group.gate_direction or 'unknown'}"
                    f" · per-file acquisition grids {'known' if group.grid_complete else 'unknown'}"
                    f"{gate_ranges}{modes}{links}"
                )
                # Display newest files first without changing the group's
                # reference-file order used by batch processing.
                displayed_files = group.files if baseline_mode else sorted(
                    group.files, key=lambda source: (-source.modified_time, source.filename.casefold())
                )
                peer_condition_values = tuple(
                    _drr_condition_values(entry) for entry in group.files
                )
                for source in displayed_files:
                    status = "processed" if source.processed else "new"
                    frames = (
                        f"{source.frame_count} frames"
                        if source.frame_count is not None else "frames unknown"
                    )
                    spectral = (
                        f"spectral grid {len(source.spectral_grid)} pts"
                        f" ({source.spectral_grid[0]:g}–{source.spectral_grid[-1]:g})"
                        if len(source.spectral_grid) >= 2 else "spectral grid unknown"
                    )
                    ranges = (
                        " · " + ", ".join(
                            f"{label} {low:g}–{high:g}"
                            for label, (low, high) in zip(source.gate_labels, source.gate_ranges)
                        )
                        if source.gate_ranges else ""
                    )
                    modified = (
                        datetime.fromtimestamp(source.modified_time).strftime("%Y-%m-%d %H:%M:%S")
                        if np.isfinite(source.modified_time) and source.modified_time > 0 else "unknown"
                    )
                    detail = f" · Modified {modified} · {status} · {frames} · {spectral} · gate {source.gate_direction or 'unknown'}{ranges}"
                    if source.saved_baseline_modes:
                        detail += f"\nSaved baseline: {', '.join(source.saved_baseline_modes)}"
                    recommendation = baseline_recommendations.get(source.source)
                    if baseline_mode and recommendation is not None:
                        detail += f"\nRecommendation: {recommendation.reason}"
                    if source.classification != "measurement":
                        detail += f"\n{source.classification_reason}"
                    item = QListWidgetItem(
                        format_drr_source_summary(
                            source,
                            group.files,
                            peer_condition_values=peer_condition_values,
                        )
                        + "\n"
                        + _drr_source_aux(source)
                    )
                    item.setData(Qt.UserRole, source.source)
                    item.setData(Qt.UserRole + 4, source.source)
                    item.setData(Qt.UserRole + 5, True)
                    tooltip_links = (
                        "\nLinked backgrounds: " + ", ".join(
                            _linked_label(path) for path in source.linked_backgrounds
                        )
                        if source.linked_backgrounds else ""
                    )
                    full_detail = (
                        f"{source.source}\nType: {_source_label(source)}\n"
                        f"Modified {modified} · {status} · {frames} · {spectral} · "
                        f"gate {source.gate_direction or 'unknown'}{ranges}"
                        + (f"\nSaved baseline: {', '.join(source.saved_baseline_modes)}" if source.saved_baseline_modes else "")
                        + (f"\nRecommendation: {recommendation.reason}" if baseline_mode and recommendation is not None else "")
                        + (f"\n{source.classification_reason}" if source.classification != "measurement" else "")
                        + tooltip_links
                    )
                    item.setData(
                        Qt.UserRole + 3,
                        full_detail,
                    )
                    _style_source_item(item, source)
                    item.setToolTip(full_detail)
                    rows.append(item)
                _sync_drr_rows(file_list, rows, preserve_view=preserve_view)
                _show_selected_detail()
            finally:
                file_list.setUpdatesEnabled(True)
                file_list.viewport().update()

        def _refresh_groups(*, preserve_view=False) -> None:
            nonlocal group_search_text
            needle = filter_edit.text().strip().casefold()
            all_history = show_all.isChecked() or bool(needle)
            current_key = (
                str(group_list.currentItem().data(Qt.UserRole))
                if group_list.currentItem() is not None
                else None
            )
            visible = []
            for group in groups:
                if not baseline_mode and group.is_background and not include_backgrounds.isChecked():
                    continue
                if not baseline_mode and unprocessed_only.isChecked() and group.processed:
                    continue
                if needle and needle not in group_search_text.get(group.key, ""):
                    continue
                visible.append(group)
            if not all_history:
                recent = visible[:25]
                if preserve_view:
                    anchor_key, _ = _drr_scroll_anchor(group_list)
                    keep = {current_key, anchor_key}
                    keep.update(item.data(Qt.UserRole) for item in group_list.selectedItems())
                    recent.extend(group for group in visible[25:] if group.key in keep)
                visible = recent
            if not visible and str(type_combo.currentData() or "REF") == "REF":
                if refresh_in_progress:
                    empty_hint.setText("Loading DRR catalog…")
                elif baseline_mode and not _baseline_measurements():
                    empty_hint.setText(
                        "No baseline recommendations: selected measurements need complete cached spectral grids."
                    )
                else:
                    empty_hint.setText("No matching REF files; choose All data for PL or unclassified files.")
            else:
                empty_hint.clear()
            group_list.setUpdatesEnabled(False)
            signals_blocked = group_list.blockSignals(True)
            try:
                rows = []
                for group in visible:
                    kind = {
                        "background": "background",
                        "likely_background": "likely background",
                        "review": "constant gate · review",
                    }.get(group.classification, "measurement")
                    center_text = (
                        " · " + "/".join(f"{center:g}" for center in group.wavelength_centers_nm) + " nm"
                        if group.wavelength_centers_nm
                        else " · wavelength unknown"
                    )
                    modified = (
                        datetime.fromtimestamp(group.modified_time).strftime("%Y-%m-%d %H:%M:%S")
                        if np.isfinite(group.modified_time) and group.modified_time > 0 else "unknown"
                    )
                    summary = (
                        f"Modified {modified} · {len(group.files)} file"
                        f"{'s' if len(group.files) != 1 else ''} · {kind}{center_text}"
                    )
                    if not baseline_mode:
                        badge = (
                            f"{group.processed_count}/{len(group.files)} PROCESSED"
                            if group.processed_count == len(group.files)
                            else f"PARTIAL {group.processed_count}/{len(group.files)}"
                            if group.processed_count
                            else f"0/{len(group.files)}"
                        )
                        summary = f"{badge} · {summary}"
                    type_counts = {}
                    for source in group.files:
                        kind = _source_kind(source)
                        type_counts[kind] = type_counts.get(kind, 0) + 1
                    type_text = " · ".join(
                        f"{kind} {count}" for kind, count in sorted(type_counts.items())
                    )
                    processed_text = (
                        "PROCESSED" if group.processed_count == len(group.files)
                        else "PARTIAL" if group.processed_count
                        else "UNPROCESSED"
                    )
                    processed_text += f" {group.processed_count}/{len(group.files)}"
                    file_word = "file" if len(group.files) == 1 else "files"
                    filenames = "\n".join(source.filename for source in group.files)
                    item = QListWidgetItem(
                        f"{processed_text} · {len(group.files)} {file_word} · {modified}\n{filenames}"
                    )
                    item.setData(Qt.UserRole, group.key)
                    item.setData(Qt.UserRole + 4, tuple(source.source for source in group.files))
                    item.setData(Qt.UserRole + 5, False)
                    item.setToolTip(
                        f"{group.title}\n{summary}\n\n"
                        + "\n".join(
                            f"[{_source_label(source)}] {source.source} — {source.classification_reason}"
                            for source in group.files
                        )
                    )
                    if not baseline_mode:
                        item.setForeground(
                            QColor(
                                theme_alias(
                                    "source_processed_foreground" if group.processed else "source_new_foreground"
                                )
                            )
                        )
                        font = item.font(); font.setBold(not group.processed); item.setFont(font)
                    rows.append(item)
                _sync_drr_rows(group_list, rows, preserve_view=preserve_view, select_first=True)
            finally:
                group_list.blockSignals(signals_blocked)
                group_list.setUpdatesEnabled(True)
                group_list.viewport().update()
            _populate_files(preserve_view=preserve_view)

        def _change_type_filter(_index: int) -> None:
            nonlocal groups, groups_by_key, group_search_text
            groups = _catalog_groups()
            groups_by_key = {group.key: group for group in groups}
            group_search_text = {group.key: _group_search_text(group) for group in groups}
            _refresh_groups()
            include_all = type_combo.currentData() == "All"
            if include_all != bool(getattr(self._owner, "_drr_include_all_sources", False)):
                self._owner._drr_include_all_sources = include_all
                getattr(self._owner, "_catalog_displayed_modes", set()).discard("DRR")
                if self.current_folder:
                    self._owner._queue_drr_catalog_refresh(auto=True, old_source_files={s.source for s in self.drr_available_sources})

        refresh_in_progress = bool(
            getattr(self._owner, "_drr_refresh_running", False)
        )
        dialog_closed = False
        catalog_signal = getattr(self._owner, "drr_catalog_refresh_finished", None)
        preview_signal = getattr(self._owner, "drr_catalog_preview_ready", None)

        def _apply_catalog_preview(folder: str) -> None:
            nonlocal groups, groups_by_key, measurement_center, group_search_text
            if dialog_closed or str(folder).casefold() != str(self.current_folder).casefold():
                return
            updated_groups = _catalog_groups()
            # Cache validation commonly returns the same immutable sources.
            # Resetting both models then flickers and drops the user's current
            # file selection/scroll position while they are using the picker.
            if groups and updated_groups == groups and group_list.count():
                return
            groups = updated_groups
            groups_by_key = {group.key: group for group in groups}
            group_search_text = {group.key: _group_search_text(group) for group in groups}
            measurement_center = self._drr_selected_wavelength_center()
            _refresh_groups(preserve_view=True)
            _update_type_hint()

        def _apply_catalog_completion(folder: str, success: bool) -> None:
            nonlocal refresh_in_progress
            if dialog_closed:
                return
            if str(folder).casefold() != str(self.current_folder).casefold():
                return
            if not success:
                refresh_in_progress = False
                refresh_btn.setEnabled(True)
                refresh_btn.setText("Refresh")
                return
            refresh_in_progress = False
            _apply_catalog_preview(folder)
            refresh_btn.setEnabled(True)
            refresh_btn.setText("Refresh")
            self._status(f"DRR catalog refreshed: {len(self.drr_available_sources)} files.")

        def _reload_catalog() -> None:
            nonlocal refresh_in_progress
            if refresh_in_progress or not self.current_folder:
                return
            refresh_in_progress = True
            refresh_btn.setEnabled(False)
            refresh_btn.setText("Refreshing...")
            old_source_files = (
                set(getattr(self._owner, "available_files", ()))
                | {source.source for source in self.drr_available_sources}
            )
            # A picker refresh only needs the independent DRR catalog worker;
            # unrelated PL/Compare/Power/MCD scans should not be started.
            self._owner._queue_drr_catalog_refresh(
                auto=False, old_source_files=old_source_files
            )

        def _add_group() -> None:
            group = _selected_group()
            if group is None:
                return
            members = list(group.files)
            if not baseline_mode and len(members) > 1:
                reference = members[0].source
                compatible = set(compatible_drr_repeats(self.drr_available_sources, reference))
                member_paths = {source.source for source in members}
                if member_paths - compatible:
                    answer = QMessageBox.question(
                        dlg,
                        "Review acquisition grid",
                        "This group contains a different or unknown full gate/spectral grid. Add it anyway?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if answer != QMessageBox.StandardButton.Yes:
                        return
            for source in members:
                _add_chosen(source.source)
            _update_type_hint()

        def _add_files() -> None:
            for item in file_list.selectedItems():
                _add_chosen(str(item.data(Qt.UserRole)))
            _update_type_hint()

        def _add_compatible() -> None:
            item = file_list.currentItem()
            if item is None:
                self._status("Select a reference file before adding compatible repeats.")
                return
            reference = str(item.data(Qt.UserRole))
            group = _selected_group()
            if group is None:
                self._status("Select a group before adding compatible repeats.")
                return
            compatible = compatible_drr_repeats(self.drr_available_sources, reference)
            group_members = {source.source for source in group.files}
            compatible = tuple(source for source in compatible if source in group_members)
            if not compatible:
                self._status(
                    "No compatible repeats found: full gate or spectral grid is unknown or different."
                )
                return
            for source in compatible:
                _add_chosen(source)
            _update_type_hint()
            self._status(f"Added {len(compatible)} repeat(s) compatible with {Path(reference).name}.")

        def _remove() -> None:
            for item in selected_list.selectedItems():
                selected_list.takeItem(selected_list.row(item))
            _update_type_hint()

        def _clear_chosen() -> None:
            selected_list.clear()
            _update_type_hint()

        def _browse_external() -> None:
            paths, _selected_filter = QFileDialog.getOpenFileNames(
                dlg,
                "Choose External DRR Baseline",
                self.current_folder or self._browse_start_folder(),
                "DRR baseline files (*.csv)",
            )
            for path in paths:
                _add_chosen(str(Path(path).resolve()))
            _update_type_hint()

        def _update_type_hint() -> None:
            chosen_label.setText(f"Chosen files ({selected_list.count()})")
            for index in range(selected_list.count()):
                item = selected_list.item(index)
                source = str(item.data(Qt.UserRole) or "")
                missing = source in self._drr_missing_sources([source])
                incompatible = baseline_mode and not _path_matches_selected_measurements(source)
                source_obj = next(
                    (entry for entry in self.drr_available_sources if entry.source == source),
                    None,
                )
                base = Path(source).name
                status = (
                    "Missing" if missing else "Incompatible" if incompatible
                    else "PROCESSED" if source_obj is not None and source_obj.processed
                    else "UNPROCESSED" if source_obj is not None
                    else "STATUS UNKNOWN"
                )
                item.setText(f"{status}\n{base}")
                item.setData(Qt.UserRole + 5, False)
                item.setToolTip(
                    f"{source}\n"
                    f"{'Missing source; remove or replace it.' if missing else 'Spectral grid does not match every selected measurement.' if incompatible else 'Source available.'}"
                )
                _style_source_item(item, source_obj, missing=missing or incompatible)
            chosen_kinds = {
                _source_kind(path)
                for path in (
                    str(selected_list.item(index).data(Qt.UserRole))
                    for index in range(selected_list.count())
                )
            }
            if "PL" in chosen_kinds:
                type_hint.setText("Marked PL; confirm suitability for DRR.")
                type_hint.setVisible(True)
            else:
                type_hint.clear()
                type_hint.setVisible(False)

        def _accept_chosen() -> None:
            paths = [
                str(selected_list.item(index).data(Qt.UserRole) or selected_list.item(index).text())
                for index in range(selected_list.count())
            ]
            missing = self._drr_missing_sources(paths)
            if missing:
                _update_type_hint()
                group_detail.setText("Missing DRR source(s): " + ", ".join(missing))
                details_btn.setChecked(True)
                return
            incompatible = [
                path for path in paths
                if baseline_mode and not _path_matches_selected_measurements(path)
            ]
            if incompatible:
                _update_type_hint()
                group_detail.setText(
                    "Incompatible DRR baseline grid(s): " + ", ".join(Path(path).name for path in incompatible)
                )
                details_btn.setChecked(True)
                return
            dlg.accept()

        group_list.currentRowChanged.connect(lambda _row: _populate_files())
        file_list.currentItemChanged.connect(_show_selected_detail)
        group_list.itemDoubleClicked.connect(lambda _item: _add_group())
        file_list.itemDoubleClicked.connect(lambda _item: _add_files())
        SourcePickerDialog.connect_debounced_filter(
            filter_edit, _refresh_groups, dlg, interval=180
        )
        show_all.toggled.connect(lambda _checked: _refresh_groups())
        unprocessed_only.toggled.connect(lambda _checked: _refresh_groups())
        include_backgrounds.toggled.connect(lambda _checked: _refresh_groups())
        type_combo.currentIndexChanged.connect(_change_type_filter)
        add_group_btn.clicked.connect(_add_group)
        add_files_btn.clicked.connect(_add_files)
        add_compatible_btn.clicked.connect(_add_compatible)
        remove_btn.clicked.connect(_remove)
        clear_btn.clicked.connect(_clear_chosen)
        browse_btn.clicked.connect(_browse_external)
        refresh_btn.clicked.connect(_reload_catalog)
        _refresh_groups()
        _update_type_hint()
        if catalog_signal is not None:
            catalog_signal.connect(_apply_catalog_completion)
            if refresh_in_progress:
                refresh_btn.setEnabled(False)
                refresh_btn.setText("Refreshing...")

        if preview_signal is not None:
            preview_signal.connect(_apply_catalog_preview)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(_accept_chosen)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        try:
            if dlg.exec() != QDialog.Accepted:
                return selected
            return [
                str(selected_list.item(index).data(Qt.UserRole) or selected_list.item(index).text())
                for index in range(selected_list.count())
            ]
        finally:
            # ``finished`` is not guaranteed by test doubles or unusual
            # dialog exits; always detach before widgets become unreachable.
            dialog_closed = True
            if preview_signal is not None:
                try:
                    preview_signal.disconnect(_apply_catalog_preview)
                except (RuntimeError, TypeError):
                    pass
            if catalog_signal is not None:
                try:
                    catalog_signal.disconnect(_apply_catalog_completion)
                except (RuntimeError, TypeError):
                    pass
    def _set_drr_gate_spin_value(self, gate_value: float) -> None:
        spin = self.drr_spins["gate"]
        old = spin.blockSignals(True)
        try:
            spin.setValue(float(gate_value))
        finally:
            spin.blockSignals(old)
        self._owner._set_drr_gate_toolbar_value(float(gate_value))

    def _drr_gate_value(self) -> float:
        return self._owner._drr_gate_input_value()
    def _current_drr_spectrum(self, cube: DataCube) -> tuple[float, np.ndarray, np.ndarray]:
        gate_value = self._drr_gate_value()
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        return gate_used, x, np.asarray(y, float).ravel()
    def _draw_drr_analysis_overlays(self, cube: DataCube, gate_used: float, x: np.ndarray, y: np.ndarray) -> None:
        if self._drr_spectrum_ax is None or self._drr_heatmap_ax is None:
            return
        if self._drr_heatmap_peak_artist is not None:
            try:
                self._drr_heatmap_peak_artist.remove()
            except Exception:
                pass
            self._drr_heatmap_peak_artist = None
        if self._drr_heatmap_fit_artist is not None:
            try:
                self._drr_heatmap_fit_artist.remove()
            except Exception:
                pass
            self._drr_heatmap_fit_artist = None
        if self._drr_peak_gate is not None and abs(float(gate_used) - float(self._drr_peak_gate)) > 1e-9:
            self._drr_peak_gate = None
            self._drr_peak_indices = None
        if self._drr_fit_gate is not None and abs(float(gate_used) - float(self._drr_fit_gate)) > 1e-9:
            self._drr_fit_gate = None
            self._drr_fit_x = None
            self._drr_fit_y = None
            self._drr_fit_centers = None
            self.drr_fit_status.setText("")

        if (
            self.drr_peak_show_chk.isChecked()
            and self._drr_peak_indices is not None
            and self._drr_peak_gate is not None
            and abs(float(gate_used) - float(self._drr_peak_gate)) <= 1e-9
            and self._drr_peak_indices.size > 0
        ):
            pidx = np.asarray(self._drr_peak_indices, dtype=int)
            self._drr_spectrum_ax.scatter(x[pidx], y[pidx], s=26, marker="o", facecolor="#ffd84d", edgecolor="#222", zorder=30)
            self._drr_heatmap_peak_artist = self._drr_heatmap_ax.scatter(
                x[pidx],
                np.full(pidx.size, float(gate_used)),
                s=28,
                marker="o",
                facecolor="#ffd84d",
                edgecolor="#111",
                zorder=35,
            )

        if (
            self.drr_fit_show_chk.isChecked()
            and self._drr_fit_gate is not None
            and self._drr_fit_x is not None
            and self._drr_fit_y is not None
            and abs(float(gate_used) - float(self._drr_fit_gate)) <= 1e-9
        ):
            self._drr_spectrum_ax.plot(self._drr_fit_x, self._drr_fit_y, color="#f28e2b", linewidth=1.6, zorder=28)
            if self._drr_fit_centers is not None and self._drr_fit_centers.size:
                self._drr_heatmap_fit_artist = self._drr_heatmap_ax.scatter(
                    np.asarray(self._drr_fit_centers, float),
                    np.full(int(self._drr_fit_centers.size), float(gate_used)),
                    s=34,
                    marker="x",
                    color="#f28e2b",
                    linewidths=1.5,
                    zorder=36,
                )
    def _on_drr_find_peaks(self) -> None:
        self.results_dock.show()
        self._update_results_dock_page()
        if self.last_plotted_mode != "DRR" or self._last_plot_cube is None:
            return
        gate_used, x, y = self._current_drr_spectrum(self._last_plot_cube)
        peaks = self._compute_peak_indices(
            x,
            y,
            spins=self.drr_spins,
            prom_spin=self.drr_peak_prom_spin,
            dist_spin=self.drr_peak_dist_spin,
            max_spin=self.drr_peak_max_spin,
            mode=self.drr_peak_mode_combo.currentText(),
        )
        self._drr_peak_gate = float(gate_used)
        self._drr_peak_indices = peaks
        if peaks.size > 0:
            self._set_fit_n_from_found(self.drr_fit_n_spin, int(peaks.size))
        self.drr_fit_status.setText(f"Peaks: {int(peaks.size)}")
        self._update_drr_spectrum_and_gate_line(self._last_plot_cube)
    def _on_drr_fit_lorentz(self) -> None:
        self.results_dock.show()
        self._update_results_dock_page()
        if self.last_plotted_mode != "DRR" or self._last_plot_cube is None:
            return
        gate_used, x, y = self._current_drr_spectrum(self._last_plot_cube)
        mask = self._visible_x_mask(x, self.drr_spins) & np.isfinite(y)
        if np.count_nonzero(mask) < 8:
            self.drr_fit_status.setText("Fit failed: not enough points in x-range.")
            return
        x_sel = x[mask]
        y_sel = y[mask]
        n_peaks = int(self.drr_fit_n_spin.value())
        if self._drr_peak_indices is None or self._drr_peak_gate is None or abs(float(gate_used) - float(self._drr_peak_gate)) > 1e-9:
            peak_idx = self._compute_peak_indices(
                x,
                y,
                spins=self.drr_spins,
                prom_spin=self.drr_peak_prom_spin,
                dist_spin=self.drr_peak_dist_spin,
                max_spin=self.drr_peak_max_spin,
                mode=self.drr_peak_mode_combo.currentText(),
            )
        else:
            peak_idx = np.asarray(self._drr_peak_indices, dtype=int)
        peak_idx = peak_idx[(peak_idx >= 0) & (peak_idx < x.size)]
        if peak_idx.size < n_peaks:
            order = np.argsort(np.abs(y_sel - np.nanmedian(y_sel)))[::-1]
            idx_sel = np.where(mask)[0]
            extra = idx_sel[order[: max(1, n_peaks - peak_idx.size)]]
            peak_idx = np.unique(np.concatenate([peak_idx, extra]))
        if peak_idx.size == 0:
            self.drr_fit_status.setText("Fit failed: no peak candidates.")
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
        self._drr_fit_generation = getattr(self, "_drr_fit_generation", 0) + 1
        generation = self._drr_fit_generation
        cube = self._last_plot_cube
        source_key = tuple(self.drr_selected_files)
        worker = _DrrFitWorker(x_sel, y_sel, p0, lo, hi)
        workers = getattr(self, "_drr_fit_workers", None)
        if workers is None:
            workers = []; self._drr_fit_workers = workers
        workers.append(worker)
        worker.signals.result.connect(
            lambda popt, g=generation, c=cube, sk=source_key, gate=gate_used,
            requested=self._drr_gate_value(), peaks=n_peaks, xx=x.copy():
            self._on_drr_fit_finished(g, c, sk, gate, requested, peaks, xx, popt)
        )
        worker.signals.error.connect(lambda message, g=generation: self._on_drr_fit_error(g, message))
        worker.signals.finished.connect(lambda w=worker: self._finish_drr_fit_worker(w))
        self.drr_fit_status.setText("Fitting Lorentz peaks…")
        self.thread_pool.start(worker)

    def _finish_drr_fit_worker(self, worker) -> None:
        try:
            self._drr_fit_workers.remove(worker)
        except (AttributeError, ValueError):
            pass

    def _on_drr_fit_error(self, generation: int, message: str) -> None:
        if generation == getattr(self, "_drr_fit_generation", 0):
            self.drr_fit_status.setText(f"Fit failed: {message}")

    def _on_drr_fit_finished(self, generation: int, cube, source_key, gate_used: float,
                             requested_gate: float, n_peaks: int, x: np.ndarray, popt: np.ndarray) -> None:
        if generation != getattr(self, "_drr_fit_generation", 0):
            return
        if (self.last_plotted_mode != "DRR" or self._last_plot_cube is not cube
                or tuple(self.drr_selected_files) != tuple(source_key)):
            if str(self.drr_fit_status.text()).startswith("Fitting"):
                self.drr_fit_status.setText("Fit discarded: DRR source changed.")
            return
        current_gate, _ = self._current_drr_spectrum(cube)
        if abs(float(current_gate) - float(gate_used)) > 1e-9 or abs(float(self._drr_gate_value()) - float(requested_gate)) > 1e-9:
            if str(self.drr_fit_status.text()).startswith("Fitting"):
                self.drr_fit_status.setText("Fit discarded: gate changed.")
            return
        y_fit = _drr_multi_lorentz_model(x, *popt)
        centers_fit = np.asarray([popt[3 + 3 * i] for i in range(n_peaks)], float)
        self._drr_fit_gate = float(gate_used)
        self._drr_fit_x = x
        self._drr_fit_y = np.asarray(y_fit, float)
        self._drr_fit_centers = np.asarray(np.sort(centers_fit), float)
        self.drr_fit_status.setText("Fit centers: " + ", ".join(f"{c:.4f}" for c in self._drr_fit_centers[:4]))
        self._update_drr_spectrum_and_gate_line(cube)

    def _invalidate_pending_drr_fit(self, message: str = "") -> None:
        self._drr_fit_generation = getattr(self, "_drr_fit_generation", 0) + 1
        if message and hasattr(self, "drr_fit_status") and str(self.drr_fit_status.text()).startswith("Fitting"):
            self.drr_fit_status.setText(message)
    def _on_drr_clear_fit(self) -> None:
        self._invalidate_pending_drr_fit()
        self._drr_fit_gate = None
        self._drr_fit_x = None
        self._drr_fit_y = None
        self._drr_fit_centers = None
        self.drr_fit_status.setText("")
        if self.last_plotted_mode == "DRR" and self._last_plot_cube is not None:
            self._update_drr_spectrum_and_gate_line(self._last_plot_cube)
    def _on_drr_analysis_view_changed(self) -> None:
        if self.last_plotted_mode == "DRR" and self._last_plot_cube is not None:
            self._update_drr_spectrum_and_gate_line(self._last_plot_cube)
    def _update_drr_analysis_text(self, gate_used: float, x: np.ndarray, y: np.ndarray) -> None:
        lines: list[str] = []
        if (
            self._drr_peak_indices is not None
            and self._drr_peak_gate is not None
            and abs(float(gate_used) - float(self._drr_peak_gate)) <= 1e-9
            and self._drr_peak_indices.size > 0
        ):
            pidx = np.asarray(self._drr_peak_indices, dtype=int)
            pairs = [f"({float(x[i]):.5f}, {float(y[i]):.4g})" for i in pidx]
            lines.append("Found: " + "; ".join(pairs))
        else:
            lines.append("Found: none")
        if (
            self._drr_fit_centers is not None
            and self._drr_fit_centers.size > 0
            and self._drr_fit_gate is not None
            and abs(float(gate_used) - float(self._drr_fit_gate)) <= 1e-9
            and self._drr_fit_x is not None
            and self._drr_fit_y is not None
        ):
            yc = np.interp(np.asarray(self._drr_fit_centers, float), np.asarray(self._drr_fit_x, float), np.asarray(self._drr_fit_y, float))
            fit_pairs = [f"({float(cx):.5f}, {float(cy):.4g})" for cx, cy in zip(np.asarray(self._drr_fit_centers, float), np.asarray(yc, float))]
            lines.append("Fit: " + "; ".join(fit_pairs))
        else:
            lines.append("Fit: none")
        self.drr_analysis_text.setPlainText("\n".join(lines))
    def _remove_nearest_drr_peak(self, x_click: float) -> bool:
        if self._last_plot_cube is None or self._drr_peak_indices is None or self._drr_peak_indices.size == 0:
            return False
        gate_used, _y = nearest_gate_spectrum(self._last_plot_cube, self._drr_gate_value())
        if self._drr_peak_gate is None or abs(float(gate_used) - float(self._drr_peak_gate)) > 1e-9:
            return False
        x = np.asarray(self._last_plot_cube.energy, float).ravel()
        pidx = np.asarray(self._drr_peak_indices, dtype=int)
        j = int(np.argmin(np.abs(x[pidx] - float(x_click))))
        self._drr_peak_indices = np.delete(pidx, j)
        self.drr_fit_status.setText(f"Peaks: {int(self._drr_peak_indices.size)}")
        self._update_drr_spectrum_and_gate_line(self._last_plot_cube)
        return True
    def _remove_peak_from_drr_heatmap_click(self, x_click: float, y_click: float) -> bool:
        if (
            self._last_plot_cube is None
            or self._drr_peak_indices is None
            or self._drr_peak_indices.size == 0
            or self._drr_peak_gate is None
        ):
            return False
        gate_axis = np.asarray(self._last_plot_cube.gate, float).ravel()
        if gate_axis.size == 0:
            return False
        gmin, gmax = float(np.nanmin(gate_axis)), float(np.nanmax(gate_axis))
        grng = max(1e-12, gmax - gmin)
        dg = float(np.nanmedian(np.abs(np.diff(gate_axis)))) if gate_axis.size > 1 else grng * 0.02
        tol = max(0.02 * grng, 0.75 * max(dg, 1e-9))
        if abs(float(y_click) - float(self._drr_peak_gate)) > tol:
            return False
        return self._remove_nearest_drr_peak(float(x_click))
    def _update_drr_spectrum_and_gate_line(self, cube: DataCube) -> None:
        if self._drr_spectrum_ax is None:
            return
        if self._drr_heatmap_ax is not None:
            self._drr_view_limits = (
                tuple(float(value) for value in self._drr_heatmap_ax.get_xlim()),
                tuple(float(value) for value in self._drr_heatmap_ax.get_ylim()),
            )
        # In paired view, refresh every visible linecut from its cached
        # product.  The active product still owns peak/fit overlays below.
        spectrum_axes = getattr(self, "_drr_spectrum_axes", {}) or {}
        plot_cubes = getattr(self, "_drr_plot_cubes", {}) or {}
        if len(spectrum_axes) > 1 and plot_cubes:
            gate_value = self._drr_gate_value()
            xlim = (
                self._drr_view_limits[0]
                if getattr(self, "_drr_view_limits", None) is not None
                else (float(self.drr_spins["xmin"].value()), float(self.drr_spins["xmax"].value()))
            )
            for key, axis in spectrum_axes.items():
                product = plot_cubes.get(key)
                if product is None:
                    continue
                gate_used, y = nearest_gate_spectrum(product, gate_value)
                x = np.asarray(product.energy, float).ravel()
                line = getattr(self, "_drr_spectrum_lines", {}).get(key)
                if line is None or getattr(line, "axes", None) is not axis:
                    line, = axis.plot(x, np.asarray(y, float), linewidth=1.3)
                    self._drr_spectrum_lines = {**getattr(self, "_drr_spectrum_lines", {}), key: line}
                else:
                    line.set_data(x, np.asarray(y, float))
                axis.set_title(f"{product.cbar_label} @ {gate_used:.6g} V")
                axis.set_xlabel("Photon Energy (eV)")
                axis.set_ylabel(
                    "d²(DR/R)/dE²" if key == "second" else product.cbar_label
                )
                axis.grid(alpha=0.25)
                safe_xlim = self._safe_spectrum_xlim(x, xlim)
                axis.set_xlim(safe_xlim)
                self._auto_scale_spectrum_y(axis, x, y, safe_xlim)
            gate_used, _ = nearest_gate_spectrum(cube, gate_value)
            self._set_drr_gate_spin_value(gate_used)
            self._ensure_gate_line(cube, gate_used)
            self._draw_drr_analysis_overlays(cube, gate_used, np.asarray(cube.energy, float), np.asarray(nearest_gate_spectrum(cube, gate_used)[1], float))
            self._update_drr_analysis_text(gate_used, np.asarray(cube.energy, float), np.asarray(nearest_gate_spectrum(cube, gate_used)[1], float))
            self._draw_drr_regions()
            return
        gate_value = self._drr_gate_value()
        gate_used, y = nearest_gate_spectrum(cube, gate_value)
        x = np.asarray(cube.energy, float).ravel()
        line = getattr(self, "_drr_spectrum_line", None)
        if line is None or getattr(line, "axes", None) is not self._drr_spectrum_ax:
            line, = self._drr_spectrum_ax.plot(x, np.asarray(y, float), linewidth=1.3)
            self._drr_spectrum_line = line
        else:
            line.set_data(x, np.asarray(y, float))
        self._drr_spectrum_ax.set_title(f"Spectrum @ {gate_used:.6g} V")
        self._drr_spectrum_ax.set_xlabel("Photon Energy (eV)")
        self._drr_spectrum_ax.set_ylabel(cube.cbar_label)
        self._drr_spectrum_ax.grid(alpha=0.25)
        xlim = self._safe_spectrum_xlim(
            x,
            self._drr_view_limits[0]
            if getattr(self, "_drr_view_limits", None) is not None
            else (float(self.drr_spins["xmin"].value()), float(self.drr_spins["xmax"].value())),
        )
        self._drr_spectrum_ax.set_xlim(xlim)
        self._auto_scale_spectrum_y(self._drr_spectrum_ax, x, y, xlim)
        self._set_drr_gate_spin_value(gate_used)
        self._ensure_gate_line(cube, gate_used)
        self._draw_drr_analysis_overlays(cube, gate_used, x, np.asarray(y, float))
        self._update_drr_analysis_text(gate_used, x, np.asarray(y, float))
        self._draw_drr_regions()

    def _draw_drr_regions(self) -> None:
        entries = []
        for key, axis in (getattr(self, "_drr_spectrum_axes", {}) or {}).items():
            line = getattr(self, "_drr_spectrum_lines", {}).get(key)
            if line is not None:
                entries.append((f"spectrum:{key}", axis, (line,)))
        axis = getattr(self, "_drr_spectrum_ax", None)
        line = getattr(self, "_drr_spectrum_line", None)
        if axis is not None and line is not None and not entries:
            entries.append(("spectrum", axis, (line,)))
        gate_lines = getattr(self, "_drr_gate_lines", {}) or {}
        for key, gate in gate_lines.items():
            ax = getattr(gate, "axes", None)
            if ax is not None:
                entries.append((f"gate:{key}", ax, (gate,)))
        helpers = getattr(self, "_drr_region_blitters", {})
        for key, axis, artists in entries:
            helper = helpers.get(key)
            bbox = tuple(round(float(v), 3) for v in axis.bbox.bounds)
            if helper is None:
                helper = AxesRegionBlitter(self.canvas); helpers[key] = helper
                helper.configure(axis, artists); helper._layout_bbox = bbox
                helper.restore_interactive_drawing()
            elif (helper._layout_bbox != bbox or helper.axes is not axis
                  or helper.artists != tuple(artists)):
                helper.configure(axis, artists); helper._layout_bbox = bbox
                helper.restore_interactive_drawing()
            elif not helper.draw():
                helper.restore_interactive_drawing()
        self._drr_region_blitters = helpers
