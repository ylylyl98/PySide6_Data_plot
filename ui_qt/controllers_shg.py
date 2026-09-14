"""Controller for SHG processing view interactions."""

import numpy as np
from datetime import datetime
from PySide6.QtCore import QObject, QRunnable, QTimer, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QListWidgetItem

from core.shg import ShgSettings, process_shg_sweep
from core.shg_fit import ShgFitSettings, fit_shg_angular_result, fit_shg_twist_comparison
from ui_qt.theme import alias as theme_alias
from ui_qt.source_picker_dialog import SourcePickerDialog


class _ShgWorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class _ShgWorker(QRunnable):
    """Run the CPU-heavy SHG processing/fit away from the GUI thread."""

    def __init__(self, payload: tuple[object, ...]) -> None:
        super().__init__()
        self.payload = payload
        self.signals = _ShgWorkerSignals()

    def run(self) -> None:
        try:
            data, data_b, settings, fit_settings, background, background_b, compare = self.payload[:7]
            processed = self.payload[7] if len(self.payload) > 7 else None
            if (
                isinstance(processed, tuple)
                and len(processed) == 2
                and processed[0] is not None
                and (not compare or processed[1] is not None)
            ):
                result, result_b = processed
            else:
                result = process_shg_sweep(data, settings, background=background)
                result_b = None
                if compare:
                    result_b = process_shg_sweep(data_b, settings, background=background_b)
            fit = fit_b = twist = None
            if compare:
                if fit_settings.enabled:
                    try:
                        twist = fit_shg_twist_comparison(result, result_b, fit_settings)
                        fit, fit_b = twist.reference_fit, twist.sample_fit
                    except ValueError:
                        pass
            elif fit_settings.enabled:
                try:
                    fit = fit_shg_angular_result(result, fit_settings)
                except ValueError:
                    pass
            self.signals.result.emit((result, result_b, fit, fit_b, twist))
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class ShgController:
    def __init__(self, owner):
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_shg_reprocess_timer", None)
        object.__setattr__(self, "_shg_reprocess_generation", 0)
        object.__setattr__(self, "_shg_reprocess_key", None)
        object.__setattr__(self, "_shg_loaded_processing_key", None)
        object.__setattr__(self, "_shg_reprocess_workers", [])
        object.__setattr__(self, "_shg_reprocess_pending_payload", None)
        object.__setattr__(self, "_shg_reprocess_source_key", None)
        object.__setattr__(self, "_shg_reprocess_timer_source_key", None)
        object.__setattr__(self, "_shg_reprocess_processing_key", None)
        object.__setattr__(self, "_shg_processing_cache", None)
        object.__setattr__(self, "_shg_loaded_result_seed", None)
        object.__setattr__(self, "_shg_selected_source", "")

    def __getattr__(self, name):
        return object.__getattribute__(object.__getattribute__(self, "_owner"), name)

    def __setattr__(self, name, value):
        if name == "_owner" or name.startswith("_shg_reprocess") or name.startswith("_shg_processing") or name == "_shg_loaded_processing_key":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_owner"), name, value)

    def _shg_source_identity(self, loaded=None, *, compare=None):
        """Return the immutable identity of one loaded SHG state.

        File names alone are insufficient because two experiment folders may
        contain the same names.  Retain the loaded state and data object
        identities as well as the owner's load lifecycle generation so a late
        worker result can never be published into a replacement state.
        """
        if loaded is None:
            loaded = getattr(self, "loaded", None)
        if loaded is None or getattr(loaded, "mode", None) != "SHG Processing":
            return None
        if compare is None:
            compare = bool(getattr(loaded, "shg_compare", False))
        owner = object.__getattribute__(self, "_owner")
        lifecycle_generation = getattr(owner, "_load_lifecycle_generation", None)
        return (
            str(getattr(loaded, "folder", "")).replace("\\", "/").casefold(),
            lifecycle_generation,
            id(loaded),
            id(getattr(loaded, "shg_data", None)),
            id(getattr(loaded, "shg_data_b", None)),
            id(getattr(loaded, "shg_background", None)),
            id(getattr(loaded, "shg_background_b", None)),
            tuple(getattr(loaded, "selected_files", ()) or ()),
            tuple(getattr(loaded, "baseline_files", ()) or ()),
            bool(compare),
        )

    def _shg_pending_processing_identity(self):
        payload = self._shg_reprocess_pending_payload
        if payload is None or len(payload) < 8:
            return None
        return payload[7]

    @staticmethod
    def _shg_object_revision(value):
        if value is None:
            return None
        return (
            id(value),
            getattr(value, "revision", None),
            getattr(value, "_revision", None),
            id(getattr(value, "spectra", None)),
        )

    def _shg_processing_identity(self, loaded, settings, compare: bool):
        """Identity for numerical processing; fit settings are intentionally absent."""
        return (
            self._shg_source_identity(loaded, compare=compare),
            self._shg_object_revision(getattr(loaded, "shg_data", None)),
            self._shg_object_revision(getattr(loaded, "shg_data_b", None)) if compare else None,
            self._shg_object_revision(getattr(loaded, "shg_background", None)),
            self._shg_object_revision(getattr(loaded, "shg_background_b", None)) if compare else None,
            settings,
            bool(compare),
        )

    def _shg_processed_for_request(self, loaded, settings, compare: bool):
        """Return a valid current-source result pair for a fit-only request."""
        key = self._shg_processing_identity(loaded, settings, compare)
        cached = self._shg_processing_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        if cached is not None and cached[0][0] != key[0]:
            object.__setattr__(self, "_shg_processing_cache", None)
        # A load result is an already validated processing result and seeds the
        # controller cache the first time a refit is requested.
        result = getattr(loaded, "shg_result", None)
        result_b = getattr(loaded, "shg_result_b", None)
        seed = self._shg_loaded_result_seed
        if seed is not None and seed[0] == id(result):
            # The same loaded result cannot be reused after its data or
            # background identity/revision changed.
            return None
        if (
            result is not None
            and (not compare or result_b is not None)
            and getattr(result, "data", None) is getattr(loaded, "shg_data", None)
            and (not compare or getattr(result_b, "data", None) is getattr(loaded, "shg_data_b", None))
            and bool(getattr(loaded, "shg_compare", False)) == bool(compare)
            and getattr(loaded, "shg_settings", None) == settings
            and getattr(result, "settings", settings) == settings
            and (not compare or getattr(result_b, "settings", settings) == settings)
        ):
            pair = (result, result_b if compare else None)
            object.__setattr__(self, "_shg_processing_cache", (key, pair))
            object.__setattr__(self, "_shg_loaded_result_seed", (id(result), key))
            return pair
        return None

    def _shg_reprocess_pending_for_export(self) -> bool:
        """Whether the current loaded SHG state lacks the requested result.

        MainWindow uses this query immediately before export.  It covers the
        debounce timer, the active worker, and the single latest pending
        snapshot while ignoring work belonging to an older loaded state.
        """
        loaded = getattr(self, "loaded", None)
        current_identity = self._shg_source_identity(loaded)
        if current_identity is None:
            return False

        pending_identity = self._shg_pending_processing_identity()
        if pending_identity == current_identity:
            return True

        timer = self._shg_reprocess_timer
        if (
            timer is not None
            and timer.isActive()
            and self._shg_reprocess_timer_source_key == current_identity
        ):
            return True

        if not self._shg_reprocess_workers:
            return False
        if self._shg_reprocess_source_key != current_identity:
            return False
        # A result signal can arrive just before finished.  Once that result
        # has been published, the worker is harmless to export even though it
        # is still being removed from the thread-pool registry.
        return self._shg_loaded_processing_key != self._shg_reprocess_key

    def _shg_selected_file(self) -> str:
        if not hasattr(self, "shg_files"):
            return ""
        selected = self._selected(self.shg_files)
        if selected:
            item = self.shg_files.selectedItems()[0]
            source = str(item.data(Qt.UserRole) or selected[0])
            object.__setattr__(self, "_shg_selected_source", source)
            return source
        return str(self._shg_selected_source or "")

    def _shg_source_filter(self) -> str:
        value = str(getattr(self, "_shg_source_filter_preference", "all")).casefold()
        return value if value in {"all", "new", "processed", "unknown"} else "all"

    def _shg_source_processed(self, source: str) -> bool:
        return source in getattr(self, "shg_processed_status", {}) and source not in getattr(self, "shg_processing_ambiguous", set())

    def _shg_source_modified(self, source: str) -> float:
        return float(getattr(self, "_shg_source_mtime_cache", {}).get(source, 0.0))

    def _shg_source_filter_counts(self) -> dict[str, int]:
        sources = list(getattr(self, "available_files", ()))
        unknown = sum(1 for source in sources if source in getattr(self, "shg_processing_ambiguous", set()))
        processed = sum(1 for source in sources if self._shg_source_processed(source))
        return {"all": len(sources), "new": len(sources) - processed - unknown, "processed": processed, "unknown": unknown}

    def _shg_source_row(self, source: str) -> QListWidgetItem:
        processed_at = str(getattr(self, "shg_processed_status", {}).get(source, ""))
        unknown = source in getattr(self, "shg_processing_ambiguous", set())
        if processed_at and not unknown:
            state, color_name = "✓ PROCESSED", "source_processed_foreground"
        elif unknown:
            state, color_name = "? HISTORY UNKNOWN", "source_new_foreground"
        else:
            state, color_name = "● NEW", "source_new_foreground"
        modified = self._shg_source_modified(source)
        try:
            modified_text = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M") if modified else "unknown"
        except (OverflowError, OSError, ValueError):
            modified_text = "unknown"
        detail = f"Modified {modified_text}"
        if processed_at and not unknown:
            detail += f" · Last saved {processed_at[:16].replace('T', ' ')}"
        item = QListWidgetItem(f"{state} — {source}\n{detail}")
        item.setData(Qt.UserRole, source)
        item.setToolTip(source)
        item.setForeground(QColor(theme_alias(color_name)))
        return item

    def _shg_source_combo_label(self, source: str) -> tuple[str, str]:
        """Return a concise status label while keeping combo data as source."""
        if source in getattr(self, "shg_processing_ambiguous", set()):
            return "? UNKNOWN", "source_new_foreground"
        if self._shg_source_processed(source):
            return "✓ SAVED", "source_processed_foreground"
        return "● NEW", "source_new_foreground"

    def _shg_add_source_combo_item(self, combo, source: str) -> None:
        state, color_name = self._shg_source_combo_label(source)
        index = combo.count()
        combo.addItem(f"{state} · {source}", source)
        combo.setItemData(index, source, Qt.ToolTipRole)
        combo.setItemData(index, QColor(theme_alias(color_name)), Qt.ForegroundRole)

    def _shg_apply_source_filter(self) -> None:
        if not hasattr(self, "shg_files"):
            return
        selected = self._shg_selected_file()
        wanted = self._shg_source_filter()
        sources = sorted(
            getattr(self, "available_files", ()),
            key=lambda source: (-self._shg_source_modified(source), str(source).casefold()),
        )
        if wanted == "new":
            sources = [source for source in sources if not self._shg_source_processed(source) and source not in getattr(self, "shg_processing_ambiguous", set())]
        elif wanted == "processed":
            sources = [source for source in sources if self._shg_source_processed(source)]
        elif wanted == "unknown":
            sources = [source for source in sources if source in getattr(self, "shg_processing_ambiguous", set())]
        blocked = self.shg_files.blockSignals(True)
        try:
            staged_rows = [self._shg_source_row(source) for source in sources]
            def row_descriptor(item):
                color = item.data(Qt.ForegroundRole)
                color_key = str(color.name(QColor.HexArgb)) if isinstance(color, QColor) else str(color)
                def role_value(value):
                    if isinstance(value, QColor):
                        return ("QColor", value.name(QColor.HexArgb))
                    try:
                        hash(value)
                    except TypeError:
                        return repr(value)
                    return value
                roles = tuple(
                    (int(role), role_value(item.data(role)))
                    for role in (Qt.DisplayRole, Qt.UserRole, Qt.ToolTipRole, Qt.ForegroundRole)
                )
                return (roles, color_key, bool(item.font().bold()), int(item.flags().value))
            descriptors = tuple(row_descriptor(item) for item in staged_rows)
            def populate(widget):
                for item in staged_rows:
                    widget.addItem(QListWidgetItem(item))
            if not SourcePickerDialog.replace_rows_if_changed(
                self.shg_files, populate, content_key=("shg-rows", descriptors)
            ):
                return
            if selected:
                for index in range(self.shg_files.count()):
                    if str(self.shg_files.item(index).data(Qt.UserRole) or "") == selected:
                        self.shg_files.item(index).setSelected(True)
                        break
        finally:
            self.shg_files.blockSignals(blocked)

    def _on_shg_source_filter_changed(self, index: int = -1) -> None:
        combo = getattr(self, "shg_source_filter_combo", None)
        if combo is not None and index >= 0:
            object.__setattr__(self, "_shg_source_filter_preference", str(combo.itemData(index) or "all"))
        self._shg_apply_source_filter()

    def _shg_background_file(self) -> str:
        return str(self.shg_background_combo.currentData() or "") if hasattr(self, "shg_background_combo") else ""

    def _shg_compare_mode(self) -> bool:
        return hasattr(self, "shg_workflow_tabs") and self.shg_workflow_tabs.currentIndex() == 1

    @staticmethod
    def _combo_data_text(combo) -> str:
        return str(combo.currentData() or "")

    def _shg_compare_files(self) -> tuple[str, str]:
        return (
            self._combo_data_text(self.shg_compare_reference_combo),
            self._combo_data_text(self.shg_compare_sample_combo),
        )

    def _shg_compare_background_files(self) -> tuple[str, str]:
        return (
            self._combo_data_text(self.shg_compare_background_a_combo),
            self._combo_data_text(self.shg_compare_background_b_combo),
        )

    def _shg_refresh_sources(self) -> None:
        if not hasattr(self, "shg_files"):
            return
        old_source = self._shg_selected_file()
        old_background = self._shg_background_file()
        old_compare = self._shg_compare_files()
        old_compare_backgrounds = self._shg_compare_background_files()
        source_blocked = self.shg_files.blockSignals(True)
        background_blocked = self.shg_background_combo.blockSignals(True)
        compare_combos = (
            self.shg_compare_reference_combo,
            self.shg_compare_sample_combo,
            self.shg_compare_background_a_combo,
            self.shg_compare_background_b_combo,
        )
        compare_blocked = [combo.blockSignals(True) for combo in compare_combos]
        try:
            object.__setattr__(self, "_shg_selected_source", old_source if old_source in self.available_files else "")
            self.shg_background_combo.clear()
            self.shg_background_combo.addItem("— None —", "")
            for file_name in self.available_files:
                self._shg_add_source_combo_item(self.shg_background_combo, file_name)
            background_index = self.shg_background_combo.findData(old_background)
            self.shg_background_combo.setCurrentIndex(background_index if background_index >= 0 else 0)
            for combo in (self.shg_compare_reference_combo, self.shg_compare_sample_combo):
                combo.clear()
                combo.addItem("— Select —", "")
                for file_name in self.available_files:
                    self._shg_add_source_combo_item(combo, file_name)
            for combo in (self.shg_compare_background_a_combo, self.shg_compare_background_b_combo):
                combo.clear()
                combo.addItem("— None —", "")
                for file_name in self.available_files:
                    self._shg_add_source_combo_item(combo, file_name)
            for combo, previous in zip(
                (self.shg_compare_reference_combo, self.shg_compare_sample_combo),
                old_compare,
            ):
                index = combo.findData(previous)
                combo.setCurrentIndex(index if index >= 0 else 0)
            for combo, previous in zip(
                (self.shg_compare_background_a_combo, self.shg_compare_background_b_combo),
                old_compare_backgrounds,
            ):
                index = combo.findData(previous)
                combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self.shg_files.blockSignals(source_blocked)
            self.shg_background_combo.blockSignals(background_blocked)
            for combo, blocked in zip(compare_combos, compare_blocked):
                combo.blockSignals(blocked)
        self._shg_update_background_controls()
        if hasattr(self, "shg_source_filter_combo"):
            counts = self._shg_source_filter_counts()
            current = self._shg_source_filter()
            filter_blocked = self.shg_source_filter_combo.blockSignals(True)
            try:
                self.shg_source_filter_combo.clear()
                self.shg_source_filter_combo.addItem(f"All ({counts['all']})", "all")
                self.shg_source_filter_combo.addItem(f"New ({counts['new']})", "new")
                self.shg_source_filter_combo.addItem(f"Processed ({counts['processed']})", "processed")
                self.shg_source_filter_combo.addItem(f"History unknown ({counts['unknown']})", "unknown")
                idx = self.shg_source_filter_combo.findData(current)
                self.shg_source_filter_combo.setCurrentIndex(idx if idx >= 0 else 0)
            finally:
                self.shg_source_filter_combo.blockSignals(filter_blocked)
        self._shg_apply_source_filter()
        self._shg_update_summary()

    def _shg_fit_settings_from_ui(self) -> ShgFitSettings:
        angle_min = float(self.shg_fit_min_spin.value())
        angle_max = float(self.shg_fit_max_spin.value())
        if angle_min >= angle_max:
            raise ValueError("SHG fit minimum angle must be smaller than the maximum angle.")
        return ShgFitSettings(
            enabled=bool(self.shg_fit_enable_chk.isChecked()),
            angle_min_deg=angle_min,
            angle_max_deg=angle_max,
            use_uncertainty_weights=bool(self.shg_fit_weighted_chk.isChecked()),
            include_excluded_rows=bool(self.shg_fit_include_excluded_chk.isChecked()),
            phase_branch=int(self.shg_fit_branch_spin.value()),
        )

    def _shg_settings_from_ui(self) -> ShgSettings:
        method_map = {
            "Local linear": "local_linear",
            "Local quadratic": "local_quadratic",
            "External + local residual": "external",
            "None": "none",
        }
        wrap_map = {"None": None, "0-180°": 180.0, "0-360°": 360.0}
        integration_wavelength = float(self.shg_peak_center_spin.value())
        integration_half_range = float(self.shg_gate_half_range_spin.value())
        sideband_gap = float(self.shg_sideband_gap_spin.value())
        sideband_width = float(self.shg_sideband_width_spin.value())
        gate_min = integration_wavelength - integration_half_range
        gate_max = integration_wavelength + integration_half_range
        return ShgSettings(
            peak_center_nm=integration_wavelength,
            gate_min_nm=gate_min,
            gate_max_nm=gate_max,
            left_min_nm=gate_min - sideband_gap - sideband_width,
            left_max_nm=gate_min - sideband_gap,
            right_min_nm=gate_max + sideband_gap,
            right_max_nm=gate_max + sideband_gap + sideband_width,
            background_method=method_map.get(self.shg_background_method_combo.currentText(), "local_linear"),
            sigma_clip=float(self.shg_sigma_clip_spin.value()),
            remove_cosmic_rays=bool(self.shg_cosmic_enable_chk.isChecked()),
            cosmic_threshold_mad=float(self.shg_cosmic_threshold_spin.value()),
            cosmic_window_points=int(self.shg_cosmic_window_spin.value()),
            cosmic_max_width_points=int(self.shg_cosmic_max_width_spin.value()),
            angle_scale=float(self.shg_angle_scale_spin.value()),
            angle_offset_deg=float(self.shg_angle_offset_spin.value()),
            angle_wrap_deg=wrap_map.get(self.shg_angle_wrap_combo.currentText()),
            include_failed_rows=bool(self.shg_include_failed_chk.isChecked()),
        )

    def _shg_update_background_controls(self) -> None:
        if hasattr(self, "shg_background_combo"):
            external = self.shg_background_method_combo.currentText() == "External + local residual"
            self.shg_background_combo.setEnabled(external)
            self.shg_compare_background_a_combo.setEnabled(external)
            self.shg_compare_background_b_combo.setEnabled(external)

    def _shg_update_cosmic_controls(self) -> None:
        if not hasattr(self, "shg_cosmic_enable_chk"):
            return
        window = int(self.shg_cosmic_window_spin.value())
        if window % 2 == 0:
            window = min(self.shg_cosmic_window_spin.maximum(), window + 1)
            self.shg_cosmic_window_spin.setValue(window)
        self.shg_cosmic_max_width_spin.setMaximum(max(1, window - 1))
        enabled = bool(self.shg_cosmic_enable_chk.isChecked())
        for widget in (self.shg_cosmic_threshold_spin, self.shg_cosmic_window_spin, self.shg_cosmic_max_width_spin):
            widget.setEnabled(enabled)

    def _shg_update_summary(self) -> None:
        if not hasattr(self, "shg_summary"):
            return
        if self.loaded and self.loaded.mode == "SHG Processing" and self.loaded.shg_result is not None:
            result = self.loaded.shg_result
            data = result.data
            included = int(np.count_nonzero(result.included))
            failures = len(result.quality_flags) - included
            measured_column = data.detected_columns.get("measured_angle", "measured position")
            lines = [
                data.source_file,
                f"{data.spectra.shape[0]} acquisitions, {data.wavelength_nm.size} wavelengths",
                f"Wavelength: {float(np.nanmin(data.wavelength_nm)):.6g}-{float(np.nanmax(data.wavelength_nm)):.6g} nm",
                f"Integrated corrected area: {result.settings.peak_center_nm:g} ± {0.5 * (result.settings.gate_max_nm - result.settings.gate_min_nm):g} nm",
                f"Background sidebands: {result.settings.left_min_nm:g}-{result.settings.left_max_nm:g} and {result.settings.right_min_nm:g}-{result.settings.right_max_nm:g} nm",
                f"Cosmic rays: {int(np.sum(result.cosmic_pixels_removed))} pixel(s) removed from {int(np.count_nonzero(result.cosmic_pixels_removed))} acquisition(s)",
                f"Angle column: {measured_column}; included: {included}; excluded/warned: {failures}",
            ]
            self.shg_summary.setPlainText("\n".join(lines))
        elif self.available_files:
            selected = self._shg_selected_file()
            selected_line = f"Selected: {selected}\n" if selected else "Select one CSV file from the list.\n"
            self.shg_summary.setPlainText(selected_line + f"{len(self.available_files)} CSV file(s) available; press Load to validate and process.\nThe selected file will be validated as an SHG table during loading.")
        else:
            self.shg_summary.setPlainText("No CSV files are available in the selected folder.")
        if hasattr(self, "shg_compare_summary"):
            if self.loaded and self.loaded.mode == "SHG Processing" and self.loaded.shg_compare:
                reference_name = self.loaded.shg_data.source_file if self.loaded.shg_data is not None else ""
                sample_name = self.loaded.shg_data_b.source_file if self.loaded.shg_data_b is not None else ""
                self.shg_compare_summary.setPlainText(f"Reference A: {reference_name}\nSample B: {sample_name}\nBoth curves use the shared processing settings. Twist sign is Sample B minus Reference A.")
            else:
                reference_name, sample_name = self._shg_compare_files()
                self.shg_compare_summary.setPlainText(f"Reference A: {reference_name or 'select a file'}\nSample B: {sample_name or 'select a different file'}\nPress Load to process and fit both files. Twist sign is Sample B minus Reference A.")
        self._shg_update_fit_summary()

    def _shg_update_fit_summary(self) -> None:
        if not hasattr(self, "shg_fit_summary"):
            return
        if not self.loaded or self.loaded.mode != "SHG Processing":
            self.shg_fit_summary.setPlainText("Load SHG data to calculate the angular fit.")
            return
        if self.loaded.shg_twist is not None:
            twist = self.loaded.shg_twist
            self.shg_fit_summary.setPlainText(f"A xc = {twist.reference_fit.x_center_deg:.6g} ± {twist.reference_fit.x_center_uncertainty_deg:.3g}°\nB xc = {twist.sample_fit.x_center_deg:.6g} ± {twist.sample_fit.x_center_uncertainty_deg:.3g}°\nΔxc = {twist.delta_x_center_deg:.6g} ± {twist.delta_x_center_uncertainty_deg:.3g}°\nTwist = {twist.signed_twist_angle_deg:.6g} ± {twist.twist_uncertainty_deg:.3g}° (|twist| = {twist.absolute_twist_angle_deg:.6g}°)")
        elif self.loaded.shg_fit is not None:
            fit = self.loaded.shg_fit
            self.shg_fit_summary.setPlainText(f"xc = {fit.x_center_deg:.6g} ± {fit.x_center_uncertainty_deg:.3g}°\nI₀ = {fit.i0:.6g}; A = {fit.amplitude:.6g}\nR² = {fit.r_squared:.6g}; RMSE = {fit.rmse:.6g}; n = {fit.point_count}")
        else:
            self.shg_fit_summary.setPlainText("Angular fit is disabled or unavailable.")

    def _on_shg_source_changed(self) -> None:
        if hasattr(self, "shg_files") and not self.shg_files.selectedItems():
            # Filtering rebuilds the list with signals blocked and therefore
            # never reaches this path.  An actual user clear should release
            # the retained source identity so it cannot be loaded again.
            object.__setattr__(self, "_shg_selected_source", "")
        self._invalidate_export_move_sources()
        self._shg_update_summary()
        if self._shg_inputs_complete():
            self._start_load("SHG Processing")
        else:
            invalidate = getattr(self, "_invalidate_active_load", None)
            if callable(invalidate):
                invalidate("SHG Processing")

    def _on_shg_workflow_changed(self) -> None:
        compare = self._shg_compare_mode()
        self.shg_fit_branch_spin.setEnabled(compare)
        self._shg_update_summary()
        self._status("SHG Compare / Twist Angle selected." if compare else "SHG Single File selected.")
        if self._shg_inputs_complete():
            self._start_load("SHG Processing")
        else:
            invalidate = getattr(self, "_invalidate_active_load", None)
            if callable(invalidate):
                invalidate("SHG Processing")

    def _shg_inputs_complete(self) -> bool:
        if not getattr(self, "current_folder", ""):
            return False
        if self._shg_compare_mode():
            reference, sample = self._shg_compare_files()
            if not reference or not sample or reference == sample:
                return False
            if getattr(self, "shg_background_method_combo", None) is not None and self.shg_background_method_combo.currentText() == "External + local residual":
                background_a, background_b = self._shg_compare_background_files()
                return bool(background_a and background_b)
            return True
        if not self._shg_selected_file():
            return False
        if getattr(self, "shg_background_method_combo", None) is not None and self.shg_background_method_combo.currentText() == "External + local residual":
            return bool(self._shg_background_file())
        return True

    def _on_shg_param_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._shg_update_background_controls()
        if self.loaded and self.loaded.mode == "SHG Processing" and self.loaded.shg_compare == self._shg_compare_mode():
            self._request_shg_reprocess()

    def _on_shg_angle_cursor_changed(self) -> None:
        """Update the displayed nearest-angle spectrum without reprocessing."""
        if self.loaded and self.loaded.mode == "SHG Processing":
            self._schedule_plot_redraw("SHG Processing")

    def _on_shg_cosmic_param_changed(self) -> None:
        self._shg_update_cosmic_controls()
        self._on_shg_param_changed()

    def _on_shg_fit_param_changed(self) -> None:
        enabled = bool(self.shg_fit_enable_chk.isChecked())
        for widget in (self.shg_fit_min_spin, self.shg_fit_max_spin, self.shg_fit_weighted_chk, self.shg_fit_include_excluded_chk):
            widget.setEnabled(enabled)
        self.shg_fit_branch_spin.setEnabled(enabled and self._shg_compare_mode())
        self._on_shg_param_changed()

    def _on_shg_spectrum_view_changed(self) -> None:
        if self.loaded and self.loaded.mode == "SHG Processing" and self.loaded.shg_compare == self._shg_compare_mode():
            # View-only changes do not require reprocessing; coalesce the
            # Matplotlib redraw with other controls.
            self._schedule_plot_redraw("SHG Processing")

    def _request_shg_reprocess(self) -> None:
        timer = self._shg_reprocess_timer
        if timer is None:
            timer = QTimer(self._owner)
            timer.setSingleShot(True)
            timer.setInterval(100)
            timer.timeout.connect(self._start_shg_reprocess)
            object.__setattr__(self, "_shg_reprocess_timer", timer)
        object.__setattr__(
            self,
            "_shg_reprocess_timer_source_key",
            self._shg_source_identity(getattr(self, "loaded", None)),
        )
        timer.start()

    def _stop_shg_reprocessing(self) -> None:
        timer = self._shg_reprocess_timer
        if timer is not None:
            timer.stop()
        object.__setattr__(self, "_shg_reprocess_generation", self._shg_reprocess_generation + 1)
        object.__setattr__(self, "_shg_reprocess_pending_payload", None)
        object.__setattr__(self, "_shg_reprocess_source_key", None)
        object.__setattr__(self, "_shg_reprocess_timer_source_key", None)
        object.__setattr__(self, "_shg_reprocess_key", None)
        object.__setattr__(self, "_shg_reprocess_processing_key", None)
        object.__setattr__(self, "_shg_processing_cache", None)
        object.__setattr__(self, "_shg_loaded_result_seed", None)
        object.__setattr__(self, "_shg_loaded_processing_key", None)

    def _start_shg_reprocess(self, pending_payload=None) -> None:
        object.__setattr__(self, "_shg_reprocess_timer_source_key", None)
        loaded = self.loaded
        if not loaded or loaded.mode != "SHG Processing" or loaded.shg_data is None:
            return
        try:
            if pending_payload is None:
                settings = self._shg_settings_from_ui()
                fit_settings = self._shg_fit_settings_from_ui()
            else:
                settings, fit_settings = pending_payload[2], pending_payload[3]
        except ValueError as exc:
            self._status(str(exc))
            return
        compare = bool(loaded.shg_compare) if pending_payload is None else bool(pending_payload[6])
        if compare and loaded.shg_data_b is None:
            self._status("The SHG comparison sample file is not loaded.")
            return
        source_key = (
            self._shg_source_identity(loaded, compare=compare)
            if pending_payload is None else pending_payload[7]
        )
        if pending_payload is not None and source_key != (
            self._shg_source_identity(loaded)
        ):
            return
        processing_key = self._shg_processing_identity(loaded, settings, compare)
        processed = self._shg_processed_for_request(loaded, settings, compare)
        if pending_payload is not None and len(pending_payload) > 8:
            pending_processed = pending_payload[8]
            if (
                len(pending_payload) > 9
                and pending_payload[9] == processing_key
                and
                isinstance(pending_processed, tuple)
                and len(pending_processed) == 2
                and pending_processed[0] is not None
                and (not compare or pending_processed[1] is not None)
            ):
                processed = pending_processed
        if self._shg_reprocess_workers and self._shg_reprocess_processing_key not in (None, processing_key):
            # A newer numerical request is already running.  Queue this fit
            # with no stale processing result; the latest worker will publish
            # the only result eligible for this source.
            processed = None
        if self._shg_reprocess_workers and pending_payload is None:
            self._shg_reprocess_generation += 1
            object.__setattr__(self, "_shg_reprocess_pending_payload", (
                loaded.shg_data, loaded.shg_data_b, settings, fit_settings,
                loaded.shg_background, loaded.shg_background_b, compare, source_key, processed,
                processing_key,
            ))
            self._status("Updating SHG processing… latest settings queued.")
            return
        self._shg_reprocess_generation += 1
        generation = self._shg_reprocess_generation
        self._shg_reprocess_key = (generation, settings, fit_settings, compare)
        object.__setattr__(self, "_shg_reprocess_source_key", source_key)
        object.__setattr__(self, "_shg_reprocess_processing_key", processing_key)
        self._status("Updating SHG processing…")
        payload = pending_payload or (
            loaded.shg_data, loaded.shg_data_b, settings, fit_settings,
            loaded.shg_background, loaded.shg_background_b, compare, source_key,
        )
        # The controller keeps source identity in its request snapshot, but
        # the worker contract reserves slot 7 for the validated result pair.
        worker = _ShgWorker((*payload[:7], processed))
        worker.signals.result.connect(lambda result, g=generation: self._on_shg_reprocessed(g, result))
        worker.signals.error.connect(lambda message, g=generation: self._on_shg_reprocess_error(g, message))
        workers = self._shg_reprocess_workers
        workers.append(worker)
        worker.signals.finished.connect(lambda w=worker: self._finish_shg_worker(w))
        self.thread_pool.start(worker)

    def _finish_shg_worker(self, worker: _ShgWorker) -> None:
        workers = self._shg_reprocess_workers
        try:
            workers.remove(worker)
        except ValueError:
            pass
        if not workers and self._shg_reprocess_pending_payload is not None:
            pending = self._shg_reprocess_pending_payload
            object.__setattr__(self, "_shg_reprocess_pending_payload", None)
            QTimer.singleShot(0, self._owner, lambda payload=pending: self._start_shg_reprocess(payload))

    def _on_shg_reprocessed(self, generation: int, payload: tuple[object, ...]) -> None:
        if generation != self._shg_reprocess_generation:
            return
        loaded = self.loaded
        if not loaded or loaded.mode != "SHG Processing":
            return
        current_source_key = self._shg_source_identity(loaded)
        if self._shg_reprocess_source_key is not None and current_source_key != self._shg_reprocess_source_key:
            return
        loaded.shg_result, loaded.shg_result_b, loaded.shg_fit, loaded.shg_fit_b, loaded.shg_twist = payload
        _generation, settings, fit_settings, _compare = self._shg_reprocess_key
        processing_key = self._shg_reprocess_processing_key
        if processing_key is not None:
            object.__setattr__(self, "_shg_processing_cache", (processing_key, (loaded.shg_result, loaded.shg_result_b)))
        loaded.shg_settings = settings
        loaded.shg_fit_settings = fit_settings
        self._shg_loaded_processing_key = self._shg_reprocess_key
        self._shg_update_summary()
        self._schedule_plot_redraw("SHG Processing", delay_ms=0)

    def _on_shg_reprocess_error(self, generation: int, message: str) -> None:
        if generation != self._shg_reprocess_generation:
            return
        self._status(f"SHG processing unavailable: {message.splitlines()[0]}")

    def _on_shg_background_method_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._shg_update_background_controls()
        if not self.loaded or self.loaded.mode != "SHG Processing":
            return
        if self.shg_background_method_combo.currentText() == "External + local residual":
            if self._shg_compare_mode():
                if not all(self._shg_compare_background_files()):
                    self._status("Select external SHG background CSVs for both comparison files.")
                    return
            elif not self._shg_background_file():
                self._status("Select an SHG background CSV for external background mode.")
                return
            self._start_load("SHG Processing")
            return
        self._request_shg_reprocess()
