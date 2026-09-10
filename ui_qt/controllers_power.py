"""Controller for Power-dependent view interactions."""

from typing import Any, Dict, Sequence

import numpy as np

from core import data_io
from core.loader import DataCube
from core.processing import (
    background_correct_cube,
    estimate_constant_background,
    nearest_gate_spectrum,
    power_group_title,
    power_stage_paired_vp_cubes,
    power_valley_polarization_cube,
)


def _vp_short_title(kk_title: str, kkp_title: str) -> str:
    kk_words = kk_title.split()
    kkp_words = kkp_title.split()
    n_pre = 0
    for a, b in zip(kk_words, kkp_words):
        if a == b:
            n_pre += 1
        else:
            break
    kk_mid = kk_words[n_pre:]
    kkp_mid = kkp_words[n_pre:]
    n_suf = 0
    for a, b in zip(reversed(kk_mid), reversed(kkp_mid)):
        if a == b:
            n_suf += 1
        else:
            break
    prefix = " ".join(kk_words[:n_pre])
    suffix = " ".join(kk_words[len(kk_words) - n_suf:]) if n_suf else ""
    if prefix or suffix:
        parts = ([prefix] if prefix else []) + ["KK/KKp"] + ([suffix] if suffix else [])
        return "VP " + " ".join(parts)
    return f"VP {kk_title} / {kkp_title}"


class PowerController:
    def __init__(self, owner):
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name):
        return object.__getattribute__(object.__getattribute__(self, "_owner"), name)

    def __setattr__(self, name, value):
        if name == "_owner":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_owner"), name, value)

    def _power_candidate_files(self) -> list[str]:
        from core.power_workflow import discover_power_files
        return discover_power_files(self.current_folder, include_legacy=getattr(self, '_power_include_legacy', False))

    def _power_choose_dataset(self, role=None, _force_legacy=False):
        # The primary Power action selects one complete measurement context.
        # Role-specific pickers remain available as explicit manual overrides.
        if role is None and not _force_legacy:
            return self._power_choose_measurement_group()
        role = role if role in ('KK', 'KKp') else None
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (QComboBox, QListWidgetItem, QCheckBox, QListWidget,
            QSplitter, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QAbstractItemView)
        from ui_qt.source_picker_dialog import SourcePickerDialog
        from core.power_workflow import is_combined, processed_source_names
        status = QComboBox()
        status.addItems(["All", "New", "Processed"])
        status.setCurrentText(getattr(self, "_power_picker_status_filter", "All"))
        legacy = QCheckBox("Include filename-based series")
        legacy.setToolTip("Optional legacy grouping. Ordinary PL filenames containing a power value may also match.")
        legacy.setChecked(getattr(self, '_power_include_legacy', False))
        dialog = SourcePickerDialog(self._owner, title=f"Choose {role} Power Sweep" if role else "Choose Power Dataset",
            hint=(f"Choose one sweep for {role}. The other channel's assigned sweep is excluded."
                  if role else "Select available sweeps and click Add Selected, or double-click a file. One chosen sweep opens; several go to combination review. For KK/KKp, enable Compare KK / KKp on the Power page."),
            filter_controls=(("Status", status), (None, legacy)),
            selection_mode=QAbstractItemView.ExtendedSelection)
        selected_key = self._power_role_group_key(role) if role else self._power_selected_group_key()
        checked = dict.fromkeys([selected_key] if selected_key else [])
        other_key = self._power_role_group_key('KKp' if role == 'KK' else 'KK') if role else ''
        dialog.source_list.itemDoubleClicked.disconnect()
        chosen = QListWidget()
        chosen.setObjectName('power_chosen_files')
        SourcePickerDialog.configure_source_list(chosen, selection_mode=QAbstractItemView.ExtendedSelection)
        splitter = QSplitter(Qt.Horizontal)
        def panel(title, widget):
            box = QWidget()
            layout = QVBoxLayout(box)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(QLabel(title))
            layout.addWidget(widget)
            return box
        dialog.layout().removeWidget(dialog.source_list)
        splitter.addWidget(panel('Available sweeps', dialog.source_list))
        splitter.addWidget(panel('Chosen sweeps', chosen))
        splitter.setSizes([600, 400])
        dialog.layout().insertWidget(2, splitter, 1)
        actions = QHBoxLayout()
        add = QPushButton('Add Selected')
        remove = QPushButton('Remove')
        clear = QPushButton('Clear')
        add.setObjectName('power_add_selected')
        remove.setObjectName('power_remove_selected')
        clear.setObjectName('power_clear_selected')
        for button in (add, remove, clear): actions.addWidget(button)
        dialog.layout().insertLayout(3, actions)
        def populate():
            dialog.source_list.blockSignals(True)
            dialog.source_list.clear()
            processed_names = processed_source_names(self.current_folder)
            sources = {key: source for key, source in self._power_current_sources().items()
                       if not role or key != other_key}
            for key in list(checked):
                if key not in sources:
                    checked.pop(key)
            for key, source in sources.items():
                combined = is_combined(self.current_folder, source.file_name)
                names = [source.file_name] if source.file_name else [r.file_name for r in source.records]
                processed = combined or bool(names) and all(name.replace('\\', '/').casefold() in processed_names for name in names)
                state = "Processed" if processed else "New"
                label = f"{state} · {'Combined · ' if combined else ''}{source.file_name or source.title}"
                if status.currentText() not in ("All", state) or dialog.filter_edit.text().casefold() not in label.casefold():
                    continue
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, key)
                item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if key in checked else Qt.Unchecked)
                dialog.source_list.addItem(item)
            dialog.source_list.blockSignals(False)
            details()

        def details():
            old_selected = {i.data(Qt.UserRole) for i in chosen.selectedItems()}
            chosen.clear()
            for key in checked:
                item = QListWidgetItem(str(key).removeprefix('csv::'))
                item.setData(Qt.UserRole, key)
                chosen.addItem(item)
                item.setSelected(key in old_selected)
            count = len(checked)
            dialog.ok_button.setEnabled(count == 1 if role else count > 0)
            dialog.ok_button.setText(f"Use as {role}" if role else "Review combination…" if count > 1 else "Open sweep")
            dialog.details_label.setText(f"{count} sweep{'s' if count != 1 else ''} chosen. Status and search filters do not remove chosen sweeps.")
        def add_selected(*_):
            if role and dialog.source_list.selectedItems():
                checked.clear()
            for item in dialog.source_list.selectedItems():
                checked[item.data(Qt.UserRole)] = None
                if role:
                    break
            details()
        def remove_selected():
            for item in chosen.selectedItems(): checked.pop(item.data(Qt.UserRole), None)
            populate()
        def clear_selected():
            checked.clear()
            populate()
        add.clicked.connect(add_selected)
        dialog.source_list.itemDoubleClicked.connect(add_selected)
        remove.clicked.connect(remove_selected)
        clear.clicked.connect(clear_selected)
        def selection_changed(item):
            key = item.data(Qt.UserRole)
            if item.checkState() == Qt.Checked:
                if role:
                    checked.clear()
                checked[key] = None
            else: checked.pop(key, None)
            details()
        dialog.filter_requested.connect(populate)
        def toggle_legacy(enabled):
            self._power_include_legacy = enabled
            populate()
        legacy.toggled.connect(toggle_legacy)
        status.currentTextChanged.connect(populate)
        dialog.refresh_button.clicked.connect(populate)
        dialog.source_list.currentItemChanged.connect(lambda *_: details())
        dialog.source_list.itemChanged.connect(selection_changed)
        populate()
        if role:
            dialog.source_list.setSelectionMode(QAbstractItemView.SingleSelection)
        draft = None
        while dialog.exec() and checked:
            self._power_status_filter = status.currentText()
            self._power_picker_status_filter = status.currentText()
            self._power_picker_selection = tuple(checked)
            self._power_refresh_groups()
            if role:
                combo = self.power_kk_group_combo if role == 'KK' else self.power_kkp_group_combo
                blocked = combo.blockSignals(True)
                try:
                    combo.setCurrentIndex(combo.findData(next(iter(checked))))
                finally:
                    combo.blockSignals(blocked)
                # Route picker assignments through the same state transition as
                # the combo boxes: invalidate exports, clear selected rows, and
                # redraw an already loaded Power view when appropriate.
                self._on_power_source_assignment_changed()
                if self._power_has_distinct_role_groups():
                    self._start_load("Power Dependent")
                return
            if len(checked) == 1:
                key = next(iter(checked))
                self.power_group_combo.setCurrentIndex(self.power_group_combo.findData(key))
                self._start_load("Power Dependent")
                return
            from ui_qt.power_combine_review import PowerCombineReview
            try:
                review = PowerCombineReview(self._owner, self, tuple(checked), draft)
            except (ValueError, OSError) as exc:
                self._show_error(str(exc))
                continue
            review.exec()
            state = review.snapshot()
            if draft:
                state['rows'] = {**draft['rows'], **state['rows']}
            draft = state
            if review.saved_path:
                self._power_finish_combination(review)
                return
            if not review.back_requested:
                return
            details()

    def _power_choose_measurement_group(self):
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QDialog
        from ui_qt.power_group_dialog import PowerGroupDialog
        existing = getattr(self, '_power_group_dialog', None)
        if existing is not None:
            existing.raise_()
            return
        dlg = PowerGroupDialog(self, self._owner)
        self._power_group_dialog = dlg

        def finished(result):
            def apply_and_release():
                try:
                    if result == QDialog.DialogCode.Accepted and not getattr(self, '_is_closing', False):
                        self._apply_power_measurement_group(dlg)
                finally:
                    self._power_group_dialog = None
                    dlg.deleteLater()
            # Keep all picker widgets alive until their signal delivery ends.
            QTimer.singleShot(0, self._owner, apply_and_release)

        dlg.finished.connect(finished)
        dlg.open()

    def _apply_power_measurement_group(self, dlg):
        selection = dlg.selection()
        import logging
        logging.getLogger('dptk').info('Open Power group: folder=%s selection=%s', self.current_folder, selection)
        self._power_include_legacy = bool(selection["legacy"])
        self._power_picker_status_filter = selection["status"]
        self._power_measurement_group_key = selection["group"]
        self._power_refresh_groups()
        combos = [getattr(self, name, None) for name in ("power_group_combo", "power_kk_group_combo", "power_kkp_group_combo", "power_compare_chk", "power_pair_mode_combo")]
        combos = [c for c in combos if c is not None]
        blocks = [c.blockSignals(True) for c in combos]
        try:
            for name, key in (("power_kk_group_combo", selection["KK"]), ("power_kkp_group_combo", selection["KKp"])):
                combo = getattr(self, name, None)
                if combo is not None: combo.setCurrentIndex(max(0, combo.findData(key)))
            primary = selection["single"] if selection['action'] == 'Single intensity' else selection['KK']
            if primary and hasattr(self, "power_group_combo"):
                self.power_group_combo.setCurrentIndex(max(0, self.power_group_combo.findData(primary)))
            if hasattr(self, 'power_compare_chk'):
                self.power_compare_chk.setChecked(selection['action'] != 'Single intensity')
            if hasattr(self, 'power_pair_mode_combo'):
                self.power_pair_mode_combo.setCurrentText('Stage' if selection['pairing'] == 'Pair by Stage' else 'Power Interpolation')
        finally:
            for combo, blocked in zip(combos, blocks): combo.blockSignals(blocked)
        if hasattr(self, 'power_roles_widget'):
            self.power_roles_widget.setVisible(selection['action'] != 'Single intensity')
        validated = getattr(dlg, '_validation_results', {})
        if isinstance(validated, dict) and primary in validated:
            # The picker already loaded and checked this exact pair. Keep it
            # alive through the GUI handoff instead of starting another worker.
            self._power_result_cache.update(validated)
            self._power_prevalidated_load = (self.current_folder, primary, validated[primary])
        timer = getattr(self, '_plot_redraw_timers', {}).get('Power Dependent')
        if timer is not None:
            timer.stop()
        getattr(self, '_plot_redraw_pending', set()).discard('Power Dependent')
        self._power_selected_row_index = None
        # Do not redraw an old LoadedState using the newly assigned channels.
        self._invalidate_export_move_sources()
        if hasattr(self, "_power_set_view_mode"):
            self._power_set_view_mode("VP" if selection["action"] == "VP" else "Intensity")
        if hasattr(self, "_start_load"):
            self._start_load("Power Dependent")

    def _power_axis_log(self) -> bool:
        return hasattr(self, "power_axis_scale_combo") and self.power_axis_scale_combo.currentText() == "Log"

    def _power_view(self) -> str:
        if hasattr(self, "power_view_vp_btn") and self.power_view_vp_btn.isChecked():
            return "VP"
        return "Intensity"

    def _power_set_view_mode(self, mode: str) -> None:
        vp_mode = mode == "VP"
        if hasattr(self, "power_view_intensity_btn"):
            self.power_view_intensity_btn.setChecked(not vp_mode)
        if hasattr(self, "power_view_vp_btn"):
            self.power_view_vp_btn.setChecked(vp_mode)
        self._power_update_view_mode()

    def _power_update_view_mode(self) -> None:
        vp_mode = self._power_view() == "VP"
        if hasattr(self, "power_group_combo"):
            self.power_group_combo.setEnabled(not self.power_compare_chk.isChecked())
        if hasattr(self, "power_choose_btn"):
            self.power_choose_btn.setEnabled(True)
        if hasattr(self, "power_pair_mode_combo"):
            self.power_pair_mode_combo.setEnabled(vp_mode)
        if vp_mode and hasattr(self.power_peak_controller, "status"):
            self.power_peak_controller.status.setText("Peak analysis applies to PL intensity. Switch to Intensity to view the trends.")
        self._update_plot_view_bar_visibility()

    def _power_pairing_mode(self) -> str:
        if not hasattr(self, "power_pair_mode_combo"):
            return "stage"
        return "power" if self.power_pair_mode_combo.currentText() == "Power Interpolation" else "stage"

    def _power_selected_group_key(self) -> str:
        if hasattr(self, "power_compare_chk") and self.power_compare_chk.isChecked():
            return self._power_role_group_key("KK")
        if not hasattr(self, "power_group_combo"):
            return ""
        data = self.power_group_combo.currentData()
        return str(data or "")

    def _power_role_group_key(self, role: str) -> str:
        combo = self.power_kk_group_combo if role == "KK" else self.power_kkp_group_combo
        data = combo.currentData()
        return str(data or "")

    def _power_current_sources(self) -> Dict[str, data_io.PowerSeriesSource]:
        files = tuple(self._power_candidate_files())
        from pathlib import Path
        signature = []
        for name in files:
            try:
                stat = (Path(self.current_folder) / name).stat()
                signature.append((name, stat.st_size, stat.st_mtime_ns))
            except OSError:
                signature.append((name, None, None))
        signature = (self.current_folder, tuple(signature))
        if (
            self._power_sources_cache is not None
            and self._power_sources_cache_files == signature
        ):
            return self._power_sources_cache
        sources = data_io.get_power_series_sources(self.current_folder, list(files))
        self._power_result_cache.clear()
        self._power_sources_cache = sources
        self._power_sources_cache_files = signature
        return sources

    def _power_refresh_groups(self) -> None:
        if not hasattr(self, "power_group_combo"):
            return
        old_key = str(self.power_group_combo.currentData() or "")
        old_kk = self._power_role_group_key("KK") if hasattr(self, "power_kk_group_combo") else ""
        old_kkp = self._power_role_group_key("KKp") if hasattr(self, "power_kkp_group_combo") else ""
        sources = self._power_current_sources()
        old = self.power_group_combo.blockSignals(True)
        old_role_blocks: list[tuple[Any, bool]] = []
        if hasattr(self, "power_kk_group_combo"):
            old_role_blocks = [
                (self.power_kk_group_combo, self.power_kk_group_combo.blockSignals(True)),
                (self.power_kkp_group_combo, self.power_kkp_group_combo.blockSignals(True)),
            ]
        try:
            self.power_group_combo.clear()
            self.power_group_combo.addItem("— Select power sweep —", "")
            if hasattr(self, "power_kk_group_combo"):
                self.power_kk_group_combo.clear()
                self.power_kkp_group_combo.clear()
                self.power_kk_group_combo.addItem("— Not assigned —", "")
                self.power_kkp_group_combo.addItem("— Not assigned —", "")
            ordered_sources = sorted(
                sources.items(),
                key=lambda item: (0 if item[1].source_format == "table" else 1, item[1].title.lower()),
            )
            for key, source in ordered_sources:
                from core.power_workflow import is_combined
                if source.source_format == "table":
                    label = f"{source.file_name}  (Power_uW table)"
                else:
                    powers = [float(record.power_uW) for record in source.records]
                    if powers:
                        label = f"{source.title}  ({len(source.records)} files, {min(powers):.4g}-{max(powers):.4g} uW)"
                    else:
                        label = f"{source.title}  ({len(source.records)} files)"
                if is_combined(self.current_folder, source.file_name):
                    label = "[Processed · Combined] " + label
                self.power_group_combo.addItem(label, key)
                if hasattr(self, "power_kk_group_combo"):
                    self.power_kk_group_combo.addItem(label, key)
                    self.power_kkp_group_combo.addItem(label, key)
            if old_key:
                idx = self.power_group_combo.findData(old_key)
                if idx >= 0:
                    self.power_group_combo.setCurrentIndex(idx)
            if hasattr(self, "power_kk_group_combo"):
                kk_idx = self.power_kk_group_combo.findData(old_kk)
                if kk_idx < 0:
                    kk_idx = 0
                kkp_idx = self.power_kkp_group_combo.findData(old_kkp)
                if kkp_idx < 0:
                    kkp_idx = 0
                if kk_idx >= 0:
                    self.power_kk_group_combo.setCurrentIndex(kk_idx)
                if kkp_idx >= 0:
                    self.power_kkp_group_combo.setCurrentIndex(kkp_idx)
        finally:
            self.power_group_combo.blockSignals(old)
            for combo, blocked in old_role_blocks:
                combo.blockSignals(blocked)
        self._power_update_group_summary()
        self._power_update_vp_availability()

    def _power_update_group_summary(self) -> None:
        if not hasattr(self, "power_group_summary"):
            return
        sources = self._power_current_sources()
        key = self._power_selected_group_key()
        source = sources.get(key)
        if source is None:
            self.power_group_summary.setPlainText(
                "Select a detected power sweep."
                if sources
                else "No power sweeps found. Expected a Power_uW table or filenames containing values such as 37.96uW."
            )
            return
        if source.source_format == "table":
            try:
                result = self._power_load_group_result(key)
            except Exception as exc:
                self.power_group_summary.setPlainText(f"{source.file_name}\nInvalid power table: {exc}")
                return
            records = result.records
            cube = result.cube
            stages = [record.stage for record in records if getattr(record, "stage", None) is not None]
            lines = [
                f"{source.file_name}",
                f"{len(records)} power rows: {float(np.nanmin(cube.gate)):.6g}-{float(np.nanmax(cube.gate)):.6g} uW",
                f"{len(cube.energy)} spectral points: {float(np.nanmin(cube.energy)):.6g}-{float(np.nanmax(cube.energy)):.6g} eV",
                f"stage_pos: {'available' if stages else 'not available'}",
            ]
        else:
            records = source.records
            lines = [f"{source.title}", f"{len(records)} legacy files sorted by power:"]
        for record in records[:8]:
            stage = getattr(record, "stage", None)
            stage_text = "" if stage is None else f", Stage {stage:g}"
            row = getattr(record, "row_index", None)
            row_text = "" if row is None else f", row {row}"
            lines.append(f"{getattr(record, 'power_uW', 0.0):.6g} uW{stage_text}{row_text}")
        if source.file_name:
            import json
            from pathlib import Path
            try:
                metadata = json.loads((Path(self.current_folder) / source.file_name).with_suffix('.combine.json').read_text(encoding='utf-8'))
                settings = metadata['settings']
                if 'rows' in settings:
                    lines.insert(1, 'Combined: ' + '; '.join(f"sweep {i + 1} × {row['factor']:.6g}, background {row['background']:.6g}"
                        for i, row in enumerate(settings['rows'])))
                else:
                    lines.insert(1, f"Combined: second sweep intensity × {settings['factor_second']:.6g}; backgrounds {settings['backgrounds']}.")
                for index, names in enumerate(metadata['sources']):
                    lines.append(f"Input {index + 1}: " + ', '.join(dict.fromkeys(names)))
            except (OSError, ValueError, KeyError):
                pass
        if len(records) > 8:
            lines.append(f"+{len(records) - 8} more")
        from core.power_workflow import record_sources
        origins = list(dict.fromkeys(tuple(record_sources(record)) for record in records))
        if len(origins) > 1:
            for index, names in enumerate(origins):
                lines.append(f"Plot source {index + 1}: " + ', '.join(names))
        if hasattr(self, "power_kk_group_combo") and self._power_has_distinct_role_groups():
            lines.append(f"KK -> {power_group_title(self._power_role_group_key('KK')) or 'none'}")
            lines.append(f"KKp -> {power_group_title(self._power_role_group_key('KKp')) or 'none'}")
        self.power_group_summary.setPlainText("\n".join(lines))

    def _power_background_auto_enabled(self) -> bool:
        return bool(hasattr(self, "power_background_auto_chk") and self.power_background_auto_chk.isChecked())

    def _power_set_background_spin_silent(self, value: float) -> None:
        if not hasattr(self, "power_background_spin"):
            return
        old = self.power_background_spin.blockSignals(True)
        try:
            self.power_background_spin.setValue(float(value))
        finally:
            self.power_background_spin.blockSignals(old)

    def _power_background_value(self, cubes: Sequence[DataCube] | Dict[str, DataCube] | None = None) -> float:
        if not hasattr(self, "power_background_spin"):
            return 0.0
        if self._power_background_auto_enabled() and cubes:
            value = estimate_constant_background(cubes, percentile=1.0)
            self._power_set_background_spin_silent(value)
            return value
        return float(self.power_background_spin.value())

    def _power_load_group_result(self, group_key: str) -> data_io.PowerSeriesResult:
        cached = self._power_result_cache.get(group_key)
        if cached is not None:
            return cached
        result = data_io.load_power_series_cube(
            self.current_folder,
            self._power_candidate_files(),
            group_key=group_key,
            y_axis="auto",
        )
        self._power_result_cache[group_key] = result
        return result

    def _power_corrected_cube(self, cube: DataCube, background: float | None = None) -> DataCube:
        if background is None:
            background = self._power_background_value([cube])
        return background_correct_cube(cube, background, title=cube.title)

    def _power_role_payload(self) -> tuple[data_io.PowerSeriesResult, data_io.PowerSeriesResult, str, str]:
        kk_key = self._power_role_group_key("KK")
        kkp_key = self._power_role_group_key("KKp")
        if not kk_key or not kkp_key:
            raise ValueError("Assign KK and KKp power-sweep sources.")
        if kk_key == kkp_key:
            raise ValueError("KK and KKp must use different power-sweep sources.")
        kk_result = self._power_load_group_result(kk_key)
        kkp_result = self._power_load_group_result(kkp_key)
        return kk_result, kkp_result, kk_key, kkp_key

    def _power_has_distinct_role_groups(self) -> bool:
        if hasattr(self, "power_compare_chk") and not self.power_compare_chk.isChecked():
            return False
        kk_key = self._power_role_group_key("KK")
        kkp_key = self._power_role_group_key("KKp")
        return bool(kk_key and kkp_key and kk_key != kkp_key)

    def _power_update_vp_availability(self) -> None:
        has_distinct = self._power_has_distinct_role_groups()
        if hasattr(self, "power_view_vp_btn"):
            self.power_view_vp_btn.setEnabled(has_distinct)
        if not has_distinct and self._power_view() == "VP":
            self._power_set_view_mode("Intensity")
        if hasattr(self, "power_pair_mode_combo"):
            stage_available = False
            if has_distinct:
                try:
                    kk_result, kkp_result, _kk_key, _kkp_key = self._power_role_payload()
                    kk_stages = [float(record.stage) for record in kk_result.records if record.stage is not None]
                    kkp_stages = [float(record.stage) for record in kkp_result.records if record.stage is not None]
                    stage_available = (bool(set(kk_stages) & set(kkp_stages))
                                       and len(kk_stages) == len(set(kk_stages))
                                       and len(kkp_stages) == len(set(kkp_stages)))
                except Exception:
                    stage_available = False
            item = self.power_pair_mode_combo.model().item(0)
            if item is not None:
                item.setEnabled(stage_available)
            if not stage_available and self.power_pair_mode_combo.currentIndex() == 0:
                blocked = self.power_pair_mode_combo.blockSignals(True)
                try:
                    self.power_pair_mode_combo.setCurrentIndex(1)
                finally:
                    self.power_pair_mode_combo.blockSignals(blocked)
            self.power_pair_mode_combo.setToolTip(
                "Pair matching stage_pos rows, or interpolate by Power_uW."
                if stage_available
                else "Stage pairing needs shared, unique stage_pos values; Power Interpolation is selected."
            )

    def _power_vp_payload(self) -> tuple[DataCube, DataCube, DataCube, tuple[Any, ...], tuple[Any, ...], str, str, float, str, tuple[Any, ...]]:
        kk_result, kkp_result, kk_key, kkp_key = self._power_role_payload()
        background = self._power_background_value([kk_result.cube, kkp_result.cube])
        vp_title = _vp_short_title(power_group_title(kk_key), power_group_title(kkp_key))
        pairing_mode = self._power_pairing_mode()
        from core.power_workflow import validate_power_vp_pairing
        pairing_error = validate_power_vp_pairing(kk_result, kkp_result, mode=pairing_mode)
        if pairing_error:
            raise ValueError(pairing_error)
        if pairing_mode == "stage":
            kk_cube, kkp_cube, vp_cube, stage_pairs = power_stage_paired_vp_cubes(
                kk_result.cube,
                kk_result.records,
                kkp_result.cube,
                kkp_result.records,
                background=background,
                title=vp_title,
            )
        else:
            kk_cube, kkp_cube, vp_cube = power_valley_polarization_cube(
                kk_result.cube,
                kkp_result.cube,
                background=background,
                title=vp_title,
            )
            stage_pairs = ()
        return (
            kk_cube,
            kkp_cube,
            vp_cube,
            kk_result.records,
            kkp_result.records,
            kk_key,
            kkp_key,
            background,
            pairing_mode,
            tuple(stage_pairs),
        )

    def _set_power_gate_spin_value(self, gate_value: float) -> None:
        spin = self.power_spins["gate"]
        old = spin.blockSignals(True)
        try:
            spin.setValue(float(gate_value))
        finally:
            spin.blockSignals(old)

    def _display_power_cube(self, cube: DataCube) -> tuple[DataCube, np.ndarray, np.ndarray]:
        true_power = np.asarray(cube.gate, float).ravel()
        z = np.asarray(cube.Z, float)
        if self._power_axis_log():
            positive = np.isfinite(true_power) & (true_power > 0)
            if not np.any(positive):
                raise ValueError("Log power axis needs at least one positive power point. Use Linear for this sweep.")
            true_power, z = true_power[positive], z[positive]
        display_power = true_power.astype(float, copy=True)
        if display_power.size > 1:
            finite = display_power[np.isfinite(display_power)]
            span = float(np.nanmax(finite) - np.nanmin(finite)) if finite.size else 1.0
            scale = max(span, float(np.nanmax(np.abs(finite))) if finite.size else 1.0, 1.0)
            eps = scale * 1e-6
            for idx in range(1, display_power.size):
                if not np.isfinite(display_power[idx]) or not np.isfinite(display_power[idx - 1]):
                    continue
                if display_power[idx] <= display_power[idx - 1]:
                    display_power[idx] = display_power[idx - 1] + eps
        display_cube = DataCube(
            np.asarray(cube.energy, float).copy(),
            display_power,
            z.copy(),
            cube.gate_label,
            cube.title,
            cube.cbar_label,
        )
        return display_cube, true_power, display_power

    def _apply_power_tick_labels(self, ax, true_power: np.ndarray, display_power: np.ndarray) -> None:
        if true_power.size == 0 or true_power.size > 14:
            return
        if np.allclose(true_power, display_power, rtol=1e-10, atol=1e-12):
            return
        ax.set_yticks(display_power)
        ax.set_yticklabels([f"{float(value):.6g}" for value in true_power])

    def _update_power_compare_spectrum_and_lines(self, cubes: Dict[str, DataCube]) -> None:
        if self._power_spectrum_ax is None or not cubes:
            return
        power_value = float(self.power_spins["gate"].value())
        self._power_spectrum_ax.clear()
        first_cube = next(iter(cubes.values()))
        power_grid = np.asarray(first_cube.gate, float).ravel()
        if self._power_selected_row_index is not None and 0 <= self._power_selected_row_index < power_grid.size:
            idx = int(self._power_selected_row_index)
        else:
            idx = int(np.argmin(np.abs(power_grid - power_value)))
        power_used = float(power_grid[idx]) if len(cubes) == 1 else power_value
        x_ref = np.asarray(first_cube.energy, float).ravel()
        for label, cube in cubes.items():
            z = np.asarray(cube.Z, float)
            row = idx if cube is first_cube else int(np.argmin(np.abs(np.asarray(cube.gate) - power_used)))
            y = z[row, :]
            x = np.asarray(cube.energy, float).ravel()
            self._power_spectrum_ax.plot(x, np.asarray(y, float), linewidth=1.3,
                                        label=f"{label}: {float(cube.gate[row]):.6g} uW")
        if len(cubes) > 1:
            self._power_spectrum_ax.legend(loc="best", fontsize=9)
        self._power_spectrum_ax.set_title(f"Spectrum @ {power_used:.6g} uW")
        self._power_spectrum_ax.set_xlabel("Photon Energy (eV)")
        self._power_spectrum_ax.set_ylabel(first_cube.cbar_label)
        self._power_spectrum_ax.grid(alpha=0.25)
        xlim = self._safe_spectrum_xlim(
            x_ref,
            (float(self.power_spins["xmin"].value()), float(self.power_spins["xmax"].value())),
        )
        self._power_spectrum_ax.set_xlim(xlim)
        ys = []
        xs = []
        for cube in cubes.values():
            z = np.asarray(cube.Z, float)
            row = idx if cube is first_cube else int(np.argmin(np.abs(np.asarray(cube.gate) - power_used)))
            y = z[row, :]
            ys.append(np.asarray(y, float))
            xs.append(np.asarray(cube.energy, float))
        if ys:
            merged_y = np.concatenate([y[np.isfinite(y)] for y in ys if np.any(np.isfinite(y))])
            if merged_y.size:
                fake_x = np.linspace(xlim[0], xlim[1], merged_y.size)
                self._auto_scale_spectrum_y(self._power_spectrum_ax, fake_x, merged_y, xlim)
        self._set_power_gate_spin_value(power_used)
        for key, ax in self._power_heatmap_axes.items():
            cube = self._power_active_cubes.get(key)
            if cube is None:
                continue
            display_cube, true_power, display_power = self._display_power_cube(cube)
            idx = int(np.argmin(np.abs(true_power - power_used)))
            y_line = float(display_power[idx]) if idx < display_power.size else power_used
            line = self._power_gate_lines.get(key)
            if line is None or getattr(line, "axes", None) is not ax:
                self._power_gate_lines[key] = ax.axhline(
                    y=y_line,
                    lw=1.2,
                    alpha=0.9,
                    color="#222",
                    linestyle="--",
                    zorder=20,
                )
            else:
                line.set_ydata([y_line, y_line])
        self.power_peak_controller.render(cubes)
        if self.power_peak_controller.is_enabled():
            self._power_spectrum_ax.set_xlabel("")
        self.canvas.draw_idle()

    def _on_power_axis_scale_changed(self) -> None:
        self._invalidate_export_move_sources()
        if self.loaded and self.loaded.mode == "Power Dependent" and self._power_axis_log():
            powers = np.asarray(self.loaded.cube.gate)
            positive = powers[powers > 0]
            if positive.size and self.power_spins["ymin"].value() <= 0:
                self._power_linear_ymin_before_log = (self._power_selected_group_key(),
                    self.power_spins["ymin"].value(), float(positive.min()))
                self._set_spin_value_silent(self.power_spins["ymin"], float(positive.min()))
        elif not self._power_axis_log():
            previous = getattr(self, "_power_linear_ymin_before_log", None)
            if previous and previous[0] == self._power_selected_group_key() and self.power_spins["ymin"].value() == previous[2]:
                self._set_spin_value_silent(self.power_spins["ymin"], previous[1])
            self._power_linear_ymin_before_log = None
        self._power_update_group_summary()
        self._on_power_plot_param_changed()

    def _on_power_background_mode_changed(self, _checked: bool) -> None:
        self._invalidate_export_move_sources()
        self.power_background_spin.setEnabled(not self._power_background_auto_enabled())
        self._on_power_plot_param_changed(self.power_background_auto_chk)

    def _on_power_plot_param_changed(self, sender=None) -> None:
        self._invalidate_export_move_sources()
        if sender is None or isinstance(sender, int):
            self._power_update_group_summary()
        if self.loaded and self.loaded.mode == "Power Dependent":
            if sender is self.power_spins['gate']:
                self._power_selected_row_index = None
                if self._power_active_cubes and self.last_plotted_mode == 'Power Dependent':
                    self._update_power_compare_spectrum_and_lines(self._power_active_cubes)
                    return
            if sender in (
                self.power_spins["xmin"], self.power_spins["xmax"],
                self.power_spins["ymin"], self.power_spins["ymax"],
                self.power_log_chk, self.power_background_spin,
                self.power_background_auto_chk,
            ):
                self._power_pending_range_refresh = True
                self._power_pending_center_split = bool(getattr(self, '_power_pending_center_split', False)) or sender in (
                    self.power_spins['xmin'], self.power_spins['xmax'])
            self._schedule_plot_redraw("Power Dependent")

    def _on_power_source_assignment_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._power_selected_row_index = None
        self._power_update_group_summary()
        self._power_update_vp_availability()
        self._update_action_states()
        if self.loaded and self.loaded.mode == "Power Dependent":
            if not self._power_selected_group_key() or (
                self.power_compare_chk.isChecked() and not self._power_has_distinct_role_groups()
            ):
                return
            self._refresh_automatic_ranges("Power Dependent", refresh_split=True)
            self._schedule_plot_redraw("Power Dependent")

    def _on_power_comparison_changed(self, checked: bool) -> None:
        self.power_roles_widget.setVisible(checked)
        self._power_update_vp_availability()
        self._power_update_view_mode()
        self._update_action_states()
        self._on_power_source_assignment_changed()

    def _on_power_combine(self) -> None:
        self._power_choose_combination()

    def _power_choose_combination(self) -> None:
        """Open the dedicated same-channel combination workflow."""
        return self._power_choose_dataset(_force_legacy=True)

    def _power_finish_combination(self, dialog):
        if dialog.saved_path is not None:
            from pathlib import Path
            name = str(dialog.saved_path.relative_to(Path(self.current_folder)))
            if name not in self.available_files:
                self.available_files.append(name)
            self._power_sources_cache = None
            self._power_result_cache.clear()
            self._power_refresh_groups()
            if not dialog.open_requested:
                return
            self._power_status_filter = "Processed"
            self.power_compare_chk.setChecked(False)
            index = self.power_group_combo.findData(data_io.power_sweep_source_key(name))
            if index < 0:
                self._show_error(f"The combined sweep was saved but could not be discovered: {dialog.saved_path}")
                return
            self.power_group_combo.setCurrentIndex(index)
            self._start_load("Power Dependent")

    def _on_power_plot_view_button_clicked(self, mode: str) -> None:
        self._invalidate_export_move_sources()
        self._power_selected_row_index = None
        self._power_set_view_mode(mode)
        self._update_action_states()
        if self.loaded and self.loaded.mode == "Power Dependent" and mode == "Intensity":
            self._refresh_automatic_ranges("Power Dependent", refresh_split=True)
        self._on_power_plot_param_changed()

    def _auto_power_vrange(self) -> None:
        if not self.loaded or self.loaded.mode != "Power Dependent" or self.loaded.cube is None:
            return
        x0, x1 = sorted((float(self.power_spins["xmin"].value()), float(self.power_spins["xmax"].value())))
        y0, y1 = sorted((float(self.power_spins["ymin"].value()), float(self.power_spins["ymax"].value())))

        def _roi_finite(cube) -> np.ndarray:
            x = np.asarray(cube.energy, float).ravel()
            y = np.asarray(cube.gate, float).ravel()
            z = np.asarray(cube.Z, float)
            x_mask = (x >= x0) & (x <= x1)
            y_mask = (y >= y0) & (y <= y1)
            z_roi = z[np.ix_(y_mask, x_mask)] if np.any(y_mask) and np.any(x_mask) else z
            return z_roi[np.isfinite(z_roi)]

        if self._power_has_distinct_role_groups():
            try:
                kk_result, kkp_result, _kk_key, _kkp_key = self._power_role_payload()
                bg = self._power_background_value([kk_result.cube, kkp_result.cube])
                parts = [
                    _roi_finite(self._power_corrected_cube(kk_result.cube, background=bg)),
                    _roi_finite(self._power_corrected_cube(kkp_result.cube, background=bg)),
                ]
                vals_all = np.concatenate([p for p in parts if p.size > 0]) if any(p.size for p in parts) else np.array([])
            except Exception:
                vals_all = np.array([])
        else:
            vals_all = np.array([])
        if vals_all.size == 0:
            vals_all = _roi_finite(self.loaded.cube)
        if vals_all.size == 0:
            return
        vals = vals_all
        if self._mode_log("Power Dependent"):
            pos = vals_all[vals_all > 0]
            vals = pos if pos.size else vals_all
        vmin, vmax = map(float, np.nanpercentile(vals, [0.01, 99.99]))
        if self._mode_log("Power Dependent"):
            vmin = max(vmin, 1e-12)
            vmax = max(vmax, vmin * 1.01)
        self.power_spins["vmin"].setValue(vmin)
        self.power_spins["vmax"].setValue(vmax)
        self._schedule_plot_redraw("Power Dependent")

    def _auto_power_xrange(self) -> None:
        if not self.loaded or self.loaded.mode != "Power Dependent" or self.loaded.cube is None:
            return
        self.power_spins["xmin"].setValue(float(np.nanmin(self.loaded.cube.energy)))
        self.power_spins["xmax"].setValue(float(np.nanmax(self.loaded.cube.energy)))
        self._schedule_plot_redraw("Power Dependent")

    def _auto_power_yrange(self) -> None:
        if not self.loaded or self.loaded.mode != "Power Dependent" or self.loaded.cube is None:
            return
        powers = np.asarray(self.loaded.cube.gate, float)
        positive = powers[np.isfinite(powers) & (powers > 0)]
        vals = positive if self._power_axis_log() and positive.size else powers[np.isfinite(powers)]
        if vals.size == 0:
            return
        self.power_spins["ymin"].setValue(float(np.nanmin(vals)))
        self.power_spins["ymax"].setValue(float(np.nanmax(vals)))
        self._schedule_plot_redraw("Power Dependent")
