"""Group based Power source selection dialog.

The dialog is deliberately controller agnostic: it owns a draft catalog and
returns a selection only after the current files have been validated.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout,
    QLabel, QListWidget, QListWidgetItem, QSplitter, QVBoxLayout,
    QWidget, QPushButton, QHBoxLayout)

from ui_qt.source_picker_dialog import SourcePickerDialog


class PowerGroupDialog(SourcePickerDialog):
    """Pick a measurement group, roles, view action, and VP pairing mode."""

    ACTIONS = ("Single intensity", "Compare intensity", "VP")
    PAIRING = ("Pair by Stage", "Power Interpolation")

    def __init__(self, controller, parent=None):
        from core.power_workflow import group_power_measurement_sources, processed_source_names
        self.controller = controller
        self._group_fn = group_power_measurement_sources
        self._processed_fn = processed_source_names
        self._groups = []
        self._catalog_groups = []
        self._sources = {}
        self._files = []
        self._folder = controller.current_folder
        self._validation_results = {}
        self._drafts = {}
        self._saved_assignments = {}
        self._manual_edits = set()
        self._pairing_manual = False
        self._manifest_warning = ""
        self._show_all = False
        self._initial = {
            "group": getattr(controller, "_power_measurement_group_key", ""),
            "status": getattr(controller, "_power_picker_status_filter", "All"),
            "legacy": bool(getattr(controller, "_power_include_legacy", False)),
        }
        self._legacy_value = self._initial["legacy"]
        self._selected_group_key = self._initial["group"]
        self._initial_roles = {}
        for role, name in (("single", "power_group_combo"), ("KK", "power_kk_group_combo"), ("KKp", "power_kkp_group_combo")):
            combo = getattr(controller, name, None)
            self._initial_roles[role] = str(combo.currentData() or '') if combo is not None else ''
        status = QComboBox(); status.addItems(("All", "New", "Partly processed", "Processed"))
        status.setObjectName("power_group_status_filter")
        status.setCurrentText(self._initial['status'])
        legacy = QCheckBox("Include filename-based series"); legacy.setObjectName("power_group_legacy")
        legacy.setChecked(self._legacy_value)
        super().__init__(parent or getattr(controller, "_owner", None),
            title="Choose Power Measurement Group",
            hint="Choose a recent measurement to open its sweep or KK/KKp comparison.",
            selected=self._initial["group"], filter_placeholder="Search sample, session, filename...",
            filter_controls=(("Status", status), (None, legacy)), minimum_size=(920, 580), size=(1120, 700))
        self.status_combo, self.legacy_check = status, legacy
        self.legacy_check.setVisible(False)
        # Replace the single list with a left/right splitter while retaining
        # the SourcePickerDialog shell's filtering and buttons.
        old_list = self.source_list
        self.layout().removeWidget(old_list)
        old_list.hide()
        old_list.deleteLater()
        self.source_list = QListWidget(); self.source_list.setObjectName("power_group_list")
        self.configure_source_list(self.source_list)
        # Wrapped row heights must follow the viewport when the dialog or
        # details splitter changes width (including scrollbar appearance).
        self.source_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.assignment_box = QWidget(); form = QFormLayout(self.assignment_box)
        self.action_combo = QComboBox(); self.action_combo.setObjectName("power_group_action_selector"); self.action_combo.addItems(self.ACTIONS)
        self.single_combo = QComboBox(); self.single_combo.setObjectName("power_group_single_selector")
        self.kk_combo = QComboBox(); self.kk_combo.setObjectName("power_group_kk_selector")
        self.kkp_combo = QComboBox(); self.kkp_combo.setObjectName("power_group_kkp_selector")
        self.pair_combo = QComboBox(); self.pair_combo.setObjectName("power_group_pairing_selector"); self.pair_combo.addItems(self.PAIRING)
        compare = getattr(controller, 'power_compare_chk', None)
        view = controller._power_view() if hasattr(controller, '_power_view') else 'Intensity'
        self.action_combo.setCurrentText('VP' if view == 'VP' else 'Compare intensity' if compare is not None and compare.isChecked() else 'Single intensity')
        self._owner_action = self.action_combo.currentText()
        pair = getattr(controller, 'power_pair_mode_combo', None)
        self.pair_combo.setCurrentText('Power Interpolation' if pair is not None and pair.currentText() == 'Power Interpolation' else 'Pair by Stage')
        self._pairing_manual = pair is not None and bool(pair.currentText())
        form.addRow("Action", self.action_combo); form.addRow("Single intensity", self.single_combo)
        form.addRow("KK", self.kk_combo); form.addRow("KKp", self.kkp_combo); form.addRow("VP pairing", self.pair_combo)
        self.source_details = QLabel()
        self.source_details.setWordWrap(True)
        self.source_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right = QWidget(); self._details_panel = right; right_l = QVBoxLayout(right); right_l.addWidget(self.assignment_box)
        right_l.addWidget(self.source_details); right_l.addStretch()
        split = QSplitter(Qt.Horizontal); split.addWidget(self.source_list); split.addWidget(right); split.setSizes([560, 440])
        self.layout().insertWidget(2, split, 1)
        self.details_toggle = QPushButton("Details / Adjust pairing")
        self.details_toggle.setObjectName("power_group_details_toggle")
        self.details_toggle.setCheckable(True)
        self.swap_button = QPushButton('Swap KK ↔ KKp')
        self.swap_button.setObjectName('power_group_swap_channels')
        self.swap_button.setToolTip('Swap the two channel assignments. Opening this group saves the choice; Cancel discards it.')
        self.assignment_summary = QLabel()
        self.assignment_summary.setObjectName('power_group_assignment_summary')
        quick_actions = QHBoxLayout()
        quick_actions.addWidget(self.assignment_summary, 1)
        quick_actions.addWidget(self.swap_button)
        self.swap_button.clicked.connect(self._swap_channels)
        self.show_older_button = QPushButton("Show older measurements")
        self.show_older_button.setObjectName("power_group_show_older")
        self.show_older_button.setVisible(False)
        self.layout().insertWidget(3, self.details_toggle)
        self.layout().insertWidget(4, self.show_older_button)
        self.layout().insertLayout(3, quick_actions)
        self.details_toggle.toggled.connect(self._set_details_visible)
        self.show_older_button.clicked.connect(self._show_older)
        self._set_details_visible(False)
        self._wire()
        self._restore_initial = True
        self.refresh()

    @staticmethod
    def _default_action(group):
        if ("KK" in group.mapping and "KKp" in group.mapping and
                not group.duplicates and not group.issues):
            return "Compare intensity"
        if len(group.sources) == 1 and not group.duplicates and not group.issues:
            return "Single intensity"
        return "Compare intensity"

    @staticmethod
    def _group_modified(group):
        for name in ("modified", "modified_at", "date_modified", "latest_modified", "mtime"):
            value = getattr(group, name, None)
            if value:
                try:
                    return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M")
                except (TypeError, ValueError, OSError, OverflowError):
                    return str(value)
        return ""

    def _group_row_text(self, group):
        modified = self._group_modified(group)
        identity = getattr(group, "context", "") or getattr(group, "label", group.key)
        identity = str(identity).split(" · ")[0]
        channels = ", ".join(k for k in ("KK", "KKp") if k in group.mapping)
        if not channels:
            channels = "Single sweep" if len(group.sources) == 1 else "Needs pairing"
        power = f" · {group.power_min:.6g}–{group.power_max:.6g} uW" if group.power_min is not None else ""
        return f"{modified + ' · ' if modified else ''}{identity} · {channels}{power}"

    def _set_details_visible(self, visible):
        self._details_panel.setVisible(bool(visible))
        self.assignment_box.setVisible(bool(visible))
        self.source_details.setVisible(bool(visible))
        self.legacy_check.setVisible(bool(visible))
        self.details_toggle.setText("Hide details / Adjust pairing" if visible else "Details / Adjust pairing")

    def _show_older(self):
        self._show_all = True
        self._render_groups()

    def _update_action_button(self):
        item = self.source_list.currentItem()
        group = next((g for g in self._groups if g.key == str(item.data(Qt.UserRole))), None) if item else None
        action = self._drafts.get(group.key, {}).get("action") if group else None
        if action == "Single intensity": text = "Open sweep"
        elif group and (("KK" in group.mapping and "KKp" in group.mapping) or
                        (self._drafts.get(group.key, {}).get("KK") and self._drafts.get(group.key, {}).get("KKp"))): text = "Open comparison"
        elif group: text = "Resolve pairing"
        else: text = "Open"
        self.ok_button.setText(text)
        draft = self._drafts.get(group.key, {}) if group else {}
        kk, kkp = draft.get('KK', ''), draft.get('KKp', '')
        ready = bool(kk and kkp and kk != kkp and kk in self._sources and kkp in self._sources)
        self.swap_button.setEnabled(ready and action != 'Single intensity')
        from core.processing import parse_compare_rotation_angles
        def describe(key):
            if not key:
                return 'unassigned'
            name = str(key).removeprefix('csv::')
            angles = parse_compare_rotation_angles(name)
            if angles.rot2 is not None:
                return f'{angles.rot2:g}°'
            if angles.rot1 is not None:
                return f'input {angles.rot1:g}°'
            return Path(name).name
        self.assignment_summary.setText(f'KK: {describe(kk)}   |   KKp: {describe(kkp)}' if ready else '')
        self.assignment_summary.setWordWrap(True)
        self.assignment_summary.setToolTip(f'KK: {kk}\nKKp: {kkp}' if ready else '')

    def _swap_channels(self):
        if not self.swap_button.isEnabled():
            return
        kk, kkp = self.kk_combo.currentData(), self.kkp_combo.currentData()
        for combo, value in ((self.kk_combo, kkp), (self.kkp_combo, kk)):
            blocked = combo.blockSignals(True)
            try:
                combo.setCurrentIndex(combo.findData(value))
            finally:
                combo.blockSignals(blocked)
        self._manual_edits.add(self._selected_group_key)
        self._controls_changed()

    def _refs(self):
        out = {}
        for key, name in (("in_k", "cmp_in_k_angle_spin"), ("in_kp", "cmp_in_kp_angle_spin"), ("out_k", "cmp_out_k_angle_spin"), ("out_kp", "cmp_out_kp_angle_spin")):
            w = getattr(self.controller, name, None)
            if w is not None: out[key] = float(w.value())
        return out

    def _tolerance(self):
        w = getattr(self.controller, "cmp_angle_tolerance_spin", None)
        return float(w.value()) if w is not None else 45.0

    def _catalog(self):
        from core import data_io
        from core.power_workflow import discover_power_files
        files = discover_power_files(self.controller.current_folder, include_legacy=self._legacy_value)
        self._files = files
        return data_io.get_power_series_sources(self.controller.current_folder, files)

    def _wire(self):
        self.filter_requested.connect(self._render_groups); self.refresh_button.clicked.connect(self.refresh)
        self.status_combo.currentTextChanged.connect(self._render_groups); self.legacy_check.toggled.connect(self._legacy_changed)
        self.source_list.currentItemChanged.connect(self._group_changed)
        for c in (self.action_combo, self.single_combo, self.kk_combo, self.kkp_combo, self.pair_combo): c.currentIndexChanged.connect(self._controls_changed)
        try: self.button_box.accepted.disconnect(self.accept)
        except (TypeError, RuntimeError): pass
        self.button_box.accepted.connect(self._accept_checked)
        self.ok_button.setText("Load")

    def _legacy_changed(self, value):
        self._legacy_value = bool(value); self.refresh()

    def _group_changed(self, *_):
        item = self.source_list.currentItem()
        if item is None:
            message = "Selected group is missing or hidden by the current filters." if self._selected_group_key else "Select a measurement group."
            self.set_details(message)
            self.ok_button.setEnabled(False)
            self.swap_button.setEnabled(False)
            self.assignment_summary.clear()
            return
        self._selected_group_key = str(item.data(Qt.UserRole))
        group = next((g for g in self._groups if g.key == str(item.data(Qt.UserRole))), None)
        if group is None: return
        draft = self._drafts.setdefault(group.key, {})
        if not draft:
            saved = self._saved_assignments.get(group.key, {})
            draft.update({k: str(v) for k, v in saved.items() if k in ("single", "KK", "KKp")})
        draft.setdefault("action", self._owner_action if self._restore_initial else self._default_action(group))
        self._fill(self.single_combo, group, "single", group.sources[0] if len(group.sources) == 1 else "", draft.get("single"))
        self._fill(self.kk_combo, group, "KK", group.mapping.get("KK", ""), draft.get("KK"))
        self._fill(self.kkp_combo, group, "KKp", group.mapping.get("KKp", ""), draft.get("KKp"))
        for role, combo in (("single", self.single_combo), ("KK", self.kk_combo), ("KKp", self.kkp_combo)):
            value = draft[role] if role in draft else (combo.currentData() or "")
            draft[role] = str(value or "")
            combo.setCurrentIndex(max(0, combo.findData(value)))
        self._controls_changed()

    def _fill(self, combo, group, role, default, selected=None):
        combo.blockSignals(True); combo.clear(); combo.addItem("— Not assigned —", "")
        for key in group.sources:
            combo.addItem(str(key).removeprefix("csv::"), key)
            combo.setItemData(combo.count() - 1, str(key).removeprefix('csv::'), Qt.ToolTipRole)
        wanted = selected if selected is not None else default
        if wanted and combo.findData(wanted) < 0:
            combo.addItem(f"Missing: {Path(str(wanted).removeprefix('csv::')).name}", wanted)
        combo.setCurrentIndex(max(0, combo.findData(wanted))); combo.blockSignals(False)

    def _controls_changed(self, *_):
        item = self.source_list.currentItem()
        if item is None: return
        draft = self._drafts.setdefault(str(item.data(Qt.UserRole)), {})
        if self.sender() is self.action_combo:
            draft["action"] = self.action_combo.currentText()
        if self.sender() is self.pair_combo and self.details_toggle.isChecked():
            self._pairing_manual = True
        previous_roles = {role: draft.get(role, "") for role in ("single", "KK", "KKp")}
        for role, combo in (("single", self.single_combo), ("KK", self.kk_combo), ("KKp", self.kkp_combo)): draft[role] = str(combo.currentData() or "")
        if self.details_toggle.isChecked() and any(previous_roles[r] != draft[r] for r in previous_roles):
            self._manual_edits.add(str(item.data(Qt.UserRole)))
        action = draft.get("action", self.action_combo.currentText())
        self.action_combo.blockSignals(True)
        self.action_combo.setCurrentText(action)
        self.action_combo.blockSignals(False)
        compare = action != "Single intensity"
        self.single_combo.setEnabled(action == "Single intensity")
        self.kk_combo.setEnabled(compare); self.kkp_combo.setEnabled(compare); self.pair_combo.setEnabled(action == "VP")
        validation_error = self._validate(item, action)
        group = next(g for g in self._groups if g.key == str(item.data(Qt.UserRole)))
        lines = list(group.issues)
        if self._manifest_warning and self.details_toggle.isChecked():
            lines.append(self._manifest_warning)
        for role, keys in group.duplicates.items():
            lines.append(f'{role}: {len(keys)} candidates; choose one explicitly.')
        for key, result in self._validation_results.items():
            powers = result.cube.gate
            stages = sum(record.stage is not None for record in result.records)
            lines.append(f"{key.removeprefix('csv::')}\n{len(powers)} points; {min(powers):.6g}–{max(powers):.6g} uW; stages {stages}/{len(result.records)}")
        lines.append('Angle-based assignments use the references configured in Compare.')
        self.source_details.setText('\n\n'.join(lines))
        display = validation_error
        if not validation_error:
            group = next(g for g in self._groups if g.key == str(item.data(Qt.UserRole)))
            names = ", ".join(Path(str(k).removeprefix("csv::")).name for k in group.sources)
            range_text = (f"; {group.power_count} points, {group.power_min:.6g}–{group.power_max:.6g} uW"
                          if group.power_min is not None else "")
            display = f"Ready to load: {group.context or group.key}{range_text}"
        self.set_details(display)
        ambiguous = bool(group.duplicates or group.issues or (len(group.sources) > 1 and not group.mapping))
        self.ok_button.setEnabled(not validation_error or (ambiguous and not self.details_toggle.isChecked()))
        self._update_action_button()

    def _validate(self, item, action):
        self._validation_results = {}
        if self.controller.current_folder != self._folder:
            return 'The experiment folder changed. Reopen the Power group picker.'
        group = next(g for g in self._groups if g.key == str(item.data(Qt.UserRole)))
        d = self._drafts.get(group.key, {}); required = [d.get("single")] if action == "Single intensity" else [d.get("KK"), d.get("KKp")]
        if any(not x for x in required): return "Assign every required source before loading."
        if any(x not in self._sources for x in required): return "A selected source is missing after refresh. Choose another source."
        if action != "Single intensity" and d.get("KK") == d.get("KKp"): return "KK and KKp must use distinct sources."
        try:
            from core import data_io
            results = []
            for key in required:
                source = self._sources[key]
                files = [source.file_name] if source.file_name else [record.file_name for record in source.records]
                results.append(data_io.load_power_series_cube(self.controller.current_folder, files, group_key=key, y_axis='auto'))
            self._validation_results = dict(zip(required, results))
        except Exception as exc: return f"Cannot load selected source: {exc}"
        if action != "Single intensity" and not self._pairing_manual:
            from core.power_workflow import validate_power_vp_pairing
            for label, mode in (("Pair by Stage", "stage"), ("Power Interpolation", "power")):
                if validate_power_vp_pairing(results[0], results[1], mode=mode) is None:
                    self.pair_combo.blockSignals(True); self.pair_combo.setCurrentText(label); self.pair_combo.blockSignals(False)
                    break
        if action == "VP":
            from core.power_workflow import validate_power_vp_pairing
            err = validate_power_vp_pairing(results[0], results[1], mode="power" if self.pair_combo.currentText() == "Power Interpolation" else "stage")
            if err: return err
        return ""

    def refresh(self):
        try: sources = self._catalog(); self._sources = sources
        except Exception as exc:
            self._groups = []
            self.set_details(str(exc))
            self.ok_button.setEnabled(False)
            return
        try:
            from core.power_selection_store import load_power_selections
            self._saved_assignments = load_power_selections(self.controller.current_folder, sources)
        except (ImportError, OSError, ValueError, TypeError) as exc:
            self._saved_assignments = {}
            self._manifest_warning = f"Saved pairing preferences could not be read: {exc}"
        groups = self._group_fn(self.controller.current_folder, sources, angle_refs=self._refs(), angle_tolerance=self._tolerance(), processed_names=self._processed_fn(self.controller.current_folder))
        self._catalog_groups = groups
        if self._restore_initial:
            primary = self._initial_roles['single'] if self.action_combo.currentText() == 'Single intensity' else self._initial_roles['KK']
            active_group = next((g for g in groups if primary and primary in g.sources), None)
            if active_group is not None:
                self._selected_group_key = active_group.key
                saved = dict(self._saved_assignments.get(active_group.key, {}))
                saved.update({k: v for k, v in self._initial_roles.items() if v})
                self._drafts[active_group.key] = saved
                self._drafts[active_group.key]["action"] = self._owner_action
            elif self._selected_group_key:
                saved = dict(self._saved_assignments.get(self._selected_group_key, {}))
                saved.update({k: v for k, v in self._initial_roles.items() if v})
                self._drafts[self._selected_group_key] = saved
            self._restore_initial = False
        self._render_groups()

    def _render_groups(self, *_):
        """Search the cached catalog; only Refresh rescans physical files."""
        status, needle = self.status_combo.currentText(), self.filter_edit.text().casefold().strip()
        filtered_groups = [g for g in self._catalog_groups if (status == "All" or g.status == status) and (not needle or needle in (g.key + " " + g.label + " " + " ".join(g.sources)).casefold())]
        total = len(filtered_groups)
        self._groups = filtered_groups
        if not self._show_all:
            self._groups = self._groups[:20]
            wanted_key = self._selected_group_key
            if wanted_key:
                selected = next((g for g in filtered_groups if g.key == wanted_key), None)
                if selected is not None and selected not in self._groups:
                    self._groups.append(selected)
        self.show_older_button.setVisible(total > 20 and not self._show_all)
        wanted = self._selected_group_key
        self.source_list.blockSignals(True); self.source_list.clear()
        for g in self._groups:
            it = QListWidgetItem(self._group_row_text(g)); it.setData(Qt.UserRole, g.key); it.setToolTip("\n".join(g.sources)); self.source_list.addItem(it)
        self.source_list.blockSignals(False)
        for i in range(self.source_list.count()):
            if str(self.source_list.item(i).data(Qt.UserRole)) == str(wanted): self.source_list.setCurrentRow(i); break
        else:
            if self.source_list.count() and not wanted:
                self.source_list.setCurrentRow(0)
        self._group_changed()

    def _accept_checked(self):
        # Validate only the chosen sources. The loader checks modification
        # signatures, so changed/deleted files remain detectable without
        # rescanning every unrelated sweep on the UI thread.
        item = self.source_list.currentItem()
        if item is None: return
        group = next((g for g in self._groups if g.key == str(item.data(Qt.UserRole))), None)
        draft = self._drafts.get(group.key, {}) if group else {}
        resolved = bool(draft.get("KK") and draft.get("KKp") and draft.get("KK") != draft.get("KKp"))
        ambiguous = group and not resolved and (group.duplicates or group.issues or (len(group.sources) > 1 and not group.mapping))
        if ambiguous and not self.details_toggle.isChecked():
            self.details_toggle.setChecked(True)
            self.set_details("Choose the KK and KKp sweeps, then load the comparison.")
            self._update_action_button()
            return
        error = self._validate(item, self.action_combo.currentText())
        if error:
            self.set_details(error)
            self.ok_button.setEnabled(False)
            return
        # Persist only an explicit advanced assignment. The normal automatic
        # choice remains ephemeral and Cancel never writes a manifest.
        if str(item.data(Qt.UserRole)) in self._manual_edits:
            try:
                from core.power_selection_store import save_power_selection
                key = str(item.data(Qt.UserRole))
                d = self._drafts.get(key, {})
                save_power_selection(self.controller.current_folder, key,
                                     {role: d.get(role, "") for role in ("single", "KK", "KKp")},
                                     self._sources)
            except (ImportError, OSError, ValueError, TypeError) as exc:
                self.set_details(f"Could not save this pairing: {exc}")
                return
        self.accept()

    def selection(self):
        item = self.source_list.currentItem(); key = str(item.data(Qt.UserRole))
        d = self._drafts[key]; action = d.get("action", self.action_combo.currentText())
        return {"group": key, "action": action, "single": d.get("single", ""), "KK": d.get("KK", ""), "KKp": d.get("KKp", ""), "pairing": self.pair_combo.currentText(), "status": self.status_combo.currentText(), "legacy": self._legacy_value}
