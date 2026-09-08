"""Controller for Compare workflow actions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem, QDoubleSpinBox, QToolButton

from core import data_io
from core.export import compare_source_title, vp_compare_export_base, vp_compare_title
from core.plotting import COMPARE_PANEL_ORDER
from core.processing import (
    background_correct_cube,
    classify_angle_state,
    classify_compare_channel,
    coherent_compare_auto_assignment,
    estimate_constant_background,
    infer_compare_angle_references,
    parse_compare_gate_condition,
    parse_compare_rotation_angles,
    valley_polarization_cube,
    group_compare_sources,
)
from ui_qt.source_picker_dialog import SourcePickerDialog


class CompareController:
    """Own Compare user actions while sharing the application context."""

    def __init__(self, owner) -> None:
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name):
        owner = object.__getattribute__(self, "_owner")
        return object.__getattribute__(owner, name)

    def __setattr__(self, name, value) -> None:
        if name == "_owner":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_owner"), name, value)

    def _cmp_assign_candidate_files(self) -> list[str]:
        pl_sources = getattr(self, "pl_available_files", None)
        sources = pl_sources or getattr(self, "available_files", [])
        raw_sources = [
            str(source) for source in sources
            if Path(source).suffix.lower() in {".csv", ".xlsx"}
            and data_io.classify_pl_source(source) != "DAT"
        ]
        # Older callers populate only ``available_files``. Keep that direct
        # assignment path usable while the dedicated PL catalog is empty;
        # once the PL catalog has entries, the explicit type filter remains
        # authoritative for normal source discovery.
        if not pl_sources and getattr(self, "available_files", None):
            return raw_sources
        if self._cmp_source_filter() == "all":
            return raw_sources
        return [source for source in raw_sources if data_io.classify_pl_source(source) == "PL"]

    def _cmp_source_filter(self) -> str:
        combo = getattr(self, "cmp_source_filter_combo", None)
        value = str(combo.currentData() if combo is not None else "pl")
        return value if value in {"pl", "all"} else "pl"

    def _cmp_source_filter_label(self) -> str:
        return "All raw data" if self._cmp_source_filter() == "all" else "PL raw sources"

    def _cmp_group_tolerance_fraction(self) -> float:
        spin = getattr(self, "cmp_group_power_tolerance_spin", None)
        if spin is not None:
            return max(0.001, min(1.0, float(spin.value()) / 100.0))
        return max(0.001, min(1.0, float(getattr(self, "cmp_group_power_tolerance_percent", 5.0)) / 100.0))

    @staticmethod
    def _cmp_group_value(group, name: str, default=""):
        value = getattr(group, name, default)
        return default if value is None else value

    def _cmp_source_groups(self, candidates=None, *, power_tolerance_fraction=None):
        """Return coherent compare groups from the current raw source catalog."""
        candidates = list(self._cmp_assign_candidate_files() if candidates is None else candidates)
        helper = group_compare_sources
        if power_tolerance_fraction is None:
            power_tolerance_fraction = self._cmp_group_tolerance_fraction()
        return list(helper(
            candidates,
            in_k_angle=float(self.cmp_in_k_angle_spin.value()),
            out_k_angle=float(self.cmp_out_k_angle_spin.value()),
            in_kp_angle=float(self.cmp_in_kp_angle_spin.value()),
            out_kp_angle=float(self.cmp_out_kp_angle_spin.value()),
            tolerance=float(self.cmp_angle_tolerance_spin.value()),
            power_tolerance_fraction=float(power_tolerance_fraction),
        ))

    def _cmp_update_group_badge(self) -> None:
        badge = getattr(self, "cmp_group_selection_summary", None)
        if badge is None:
            return
        key = str(getattr(self, "cmp_selected_group_key", "") or "")
        label = str(getattr(self, "cmp_selected_group_label", "") or "")
        sources = getattr(self, "cmp_selected_group_sources", ()) or ()
        if not key:
            badge.set_status("No compare group selected.", tooltip="Choose a coherent compare group.")
            return
        tooltip = "\n".join(str(source) for source in sources)
        badge.set_status(label or key, tooltip=tooltip, badge_state="selected")

    def _cmp_set_mapping(self, mapping: dict[str, str], *, update_summary: bool = True) -> None:
        """Set hidden combo state as one logical assignment transaction."""
        for key, combo in self.cmp_channel_combos.items():
            old = combo.blockSignals(True)
            try:
                combo.setCurrentText(str(mapping.get(key, "") or ""))
            finally:
                combo.blockSignals(old)
        if update_summary:
            self._cmp_update_assignment_summary()
            if hasattr(self, "loaded"):
                self._on_cmp_plot_param_changed()

    def _cmp_group_mapping(self, group) -> dict[str, str]:
        mapping = self._cmp_group_value(group, "mapping", {})
        return {str(key): str(value) for key, value in dict(mapping or {}).items() if value}

    def _cmp_open_group_dialog(self) -> None:
        selected_key = str(getattr(self, "cmp_selected_group_key", "") or "")
        tolerance_spin = QDoubleSpinBox()
        tolerance_spin.setRange(0.1, 100.0)
        tolerance_spin.setDecimals(1)
        tolerance_spin.setSingleStep(0.5)
        tolerance_spin.setSuffix(" %")
        tolerance_spin.setValue(float(getattr(self, "cmp_group_power_tolerance_percent", 5.0)))
        dlg = SourcePickerDialog(
            self._owner,
            title="Choose Compare Group",
            hint=("Choose a coherent KK / KKp / KpK / KpKp source group. "
                  "Files are grouped by shared measurement context and nearby power."),
            selected=selected_key,
            filter_controls=(("Power tolerance", tolerance_spin),),
            filter_interval=100,
            minimum_size=(860, 520),
            size=(1040, 660),
        )

        def _groups_for_dialog():
            return self._cmp_source_groups(
                power_tolerance_fraction=float(tolerance_spin.value()) / 100.0
            )

        def _refresh_view() -> None:
            needle = dlg.filter_edit.text().strip().casefold()
            groups = _groups_for_dialog()

            def _populate(widget) -> None:
                for group in groups:
                    key = str(self._cmp_group_value(group, "key", ""))
                    label = str(self._cmp_group_value(group, "label", key))
                    sources = tuple(self._cmp_group_value(group, "sources", ()) or ())
                    haystack = " ".join((key, label, *map(str, sources))).casefold()
                    if needle and needle not in haystack:
                        continue
                    item = QListWidgetItem(
                        label + ("\n" + " · ".join(Path(source).name for source in sources) if sources else "")
                    )
                    item.setData(Qt.UserRole, key)
                    item.setToolTip("\n".join(map(str, sources)))
                    widget.addItem(item)
            dlg._cmp_groups = groups
            dlg.repopulate(_populate, fallback_selection=selected_key)
            if not groups:
                dlg.set_details("No coherent compare groups match the current source filter and angle rules.")

        def _update_details() -> None:
            item = dlg.source_list.currentItem()
            if item is None:
                dlg.set_details("Select a group to assign its channels.")
                return
            key = str(item.data(Qt.UserRole) or "")
            group = next((entry for entry in getattr(dlg, "_cmp_groups", ())
                          if str(self._cmp_group_value(entry, "key", "")) == key), None)
            sources = tuple(self._cmp_group_value(group, "sources", ()) or ()) if group else ()
            mapping = self._cmp_group_mapping(group) if group else {}
            duplicates = self._cmp_group_value(group, "duplicates", {}) if group else {}
            detail_lines = [
                f"{channel} → {Path(source).name}" for channel, source in mapping.items()
            ]
            detail_lines.extend(
                f"{channel} → choose manually ({len(names)} matches)"
                for channel, names in dict(duplicates or {}).items()
            )
            missing = [channel for channel in COMPARE_PANEL_ORDER if channel not in mapping and channel not in duplicates]
            if missing:
                detail_lines.append("Missing: " + ", ".join(missing))
            if sources:
                detail_lines.append("Sources: " + ", ".join(Path(source).name for source in sources))
            dlg.set_details("\n".join(detail_lines) or "No channel assignments in this group.")

        dlg.filter_requested.connect(_refresh_view)
        dlg.source_list.currentItemChanged.connect(lambda _current, _previous: _update_details())
        tolerance_spin.valueChanged.connect(lambda _value: _refresh_view())
        dlg.refresh_button.clicked.connect(_refresh_view)
        _refresh_view()
        _update_details()
        if dlg.exec() != SourcePickerDialog.Accepted:
            return
        chosen_key = str(dlg.selected_source() or "")
        group = next((entry for entry in getattr(dlg, "_cmp_groups", ())
                      if str(self._cmp_group_value(entry, "key", "")) == chosen_key), None)
        if group is None:
            return
        self.cmp_group_power_tolerance_percent = float(tolerance_spin.value())
        self.cmp_selected_group_key = chosen_key
        self.cmp_selected_group_label = str(self._cmp_group_value(group, "label", chosen_key))
        self.cmp_selected_group_sources = tuple(self._cmp_group_value(group, "sources", ()) or ())
        self.cmp_kk_swap_active = False
        self._cmp_set_mapping(self._cmp_group_mapping(group))
        # Re-run assignment after selecting a group so angle-bearing files can
        # be inferred from this group when the current references miss them.
        self._cmp_auto_assign_channels(preserve_existing=False)

    def _cmp_clear_group(self) -> None:
        self.cmp_selected_group_key = ""
        self.cmp_selected_group_label = ""
        self.cmp_selected_group_sources = ()
        self.cmp_group_selection_cleared = True
        self.cmp_kk_swap_active = False
        self._cmp_set_mapping({})

    def _cmp_swap_kk_channels(self) -> None:
        current = self._cmp_current_mapping()
        swapped = dict(current)
        swapped["KK"], swapped["KKp"] = current.get("KKp", ""), current.get("KK", "")
        self.cmp_kk_swap_active = not bool(getattr(self, "cmp_kk_swap_active", False))
        self._cmp_set_mapping(swapped)

    def _cmp_set_channel_combo_items(self) -> None:
        all_sources = {
            str(source) for source in (getattr(self, "pl_available_files", None) or getattr(self, "available_files", []))
            if Path(source).suffix.lower() in {".csv", ".xlsx"}
            and data_io.classify_pl_source(source) != "DAT"
        }
        candidates = self._cmp_assign_candidate_files()
        for combo in self.cmp_channel_combos.values():
            current = combo.currentText()
            old = combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItem("")
                for source in candidates:
                    combo.addItem(source)
                    kind = data_io.classify_pl_source(source)
                    if kind == "Unknown":
                        combo.setItemData(
                            combo.count() - 1,
                            "Unknown raw data; choose All raw data for automatic detection.",
                            Qt.ToolTipRole,
                        )
                if current and current not in candidates and current in all_sources:
                    combo.addItem(current)
                    index = combo.findText(current)
                    combo.setItemData(index, f"Currently assigned; outside {self._cmp_source_filter_label()}.", Qt.ToolTipRole)
                if current:
                    combo.setCurrentText(current)
            finally:
                combo.blockSignals(old)

    def _on_cmp_source_filter_changed(self, _value: str = "") -> None:
        self._cmp_set_channel_combo_items()
        self._cmp_update_assignment_summary()

    def _cmp_assignment_advisories(self, mapping: dict[str, str]) -> list[str]:
        values = [combo.currentText().strip() for combo in self.cmp_channel_combos.values()]
        duplicates = sorted({name for name, count in Counter(v for v in values if v).items() if count > 1})
        advisories: list[str] = []
        if duplicates:
            advisories.append("Advisory: duplicate assignment(s): " + ", ".join(duplicates))
        if self._cmp_is_vp_view():
            missing = [key for key in ("KK", "KKp") if key not in mapping]
            if missing:
                advisories.append("Advisory: VP missing required channel(s): " + ", ".join(missing))
        else:
            visible = self._cmp_visible_channels(mapping)
            if len(visible) < 2:
                advisories.append("Advisory: select at least two visible channels for Intensity Compare")
        groups = {parse_compare_gate_condition(name) for name in mapping.values()}
        groups.discard(None)
        groups.discard("")
        if len(groups) > 1:
            advisories.append("Advisory: assigned files use inconsistent gate/condition groups")
        return advisories

    def _cmp_view_mode(self) -> str:
        if hasattr(self, "cmp_view_vp_btn") and self.cmp_view_vp_btn.isChecked():
            return "Valley Polarization"
        return "Intensity Compare"

    def _cmp_set_view_mode(self, mode: str) -> None:
        vp_mode = mode == "Valley Polarization"
        if hasattr(self, "cmp_view_intensity_btn"):
            self.cmp_view_intensity_btn.setChecked(not vp_mode)
        if hasattr(self, "cmp_view_vp_btn"):
            self.cmp_view_vp_btn.setChecked(vp_mode)

    def _cmp_is_vp_view(self) -> bool:
        return self._cmp_view_mode() == "Valley Polarization"

    def _cmp_background_auto_enabled(self) -> bool:
        return bool(
            hasattr(self, "cmp_vp_auto_background_chk")
            and self.cmp_vp_auto_background_chk.isChecked()
        )

    def _cmp_update_background_mode(self) -> None:
        if hasattr(self, "cmp_vp_background_spin"):
            self.cmp_vp_background_spin.setEnabled(not self._cmp_background_auto_enabled())

    @staticmethod
    def _cmp_background_source_cubes(cubes):
        kk_pair = {key: cubes[key] for key in ("KK", "KKp") if key in cubes}
        return kk_pair if kk_pair else dict(cubes)

    def _cmp_set_background_spin_silent(self, value: float) -> None:
        if not hasattr(self, "cmp_vp_background_spin"):
            return
        old = self.cmp_vp_background_spin.blockSignals(True)
        try:
            self.cmp_vp_background_spin.setValue(float(value))
        finally:
            self.cmp_vp_background_spin.blockSignals(old)

    def _cmp_background_value(self, cubes=None, *, update_spin: bool = True) -> float:
        if not hasattr(self, "cmp_vp_background_spin"):
            return 0.0
        if self._cmp_background_auto_enabled() and cubes:
            value = estimate_constant_background(
                self._cmp_background_source_cubes(cubes), percentile=1.0
            )
            if update_spin:
                self._cmp_set_background_spin_silent(value)
            return value
        return float(self.cmp_vp_background_spin.value())

    def _cmp_scale_tag(self) -> str:
        return "log" if bool(self.cmp_log_chk.isChecked()) else "linear"

    def _cmp_source_mapping(self) -> dict[str, str]:
        if self.loaded and self.loaded.mode == "Compare" and self.loaded.compare_sources:
            return dict(self.loaded.compare_sources)
        return self._cmp_current_mapping()

    def _cmp_corrected_cubes(self, cubes, source_files=None, background=None):
        if background is None:
            background = self._cmp_background_value(cubes)
        source_files = source_files or self._cmp_source_mapping()
        corrected = {}
        for key, cube in cubes.items():
            title = compare_source_title(source_files.get(key, cube.title))
            corrected[key] = background_correct_cube(cube, background, title=title)
        return corrected

    def _cmp_vp_cube(self, cubes, source_files=None, background=None):
        if "KK" not in cubes or "KKp" not in cubes:
            raise ValueError("VP needs assigned KK and KKp channels.")
        if background is None:
            background = self._cmp_background_value(cubes)
        source_files = source_files or self._cmp_source_mapping()
        return valley_polarization_cube(
            cubes["KK"], cubes["KKp"], background=background,
            title=vp_compare_title(source_files, background, self._cmp_scale_tag()),
        )

    def _cmp_update_title_previews(self) -> None:
        if not hasattr(self, "cmp_vp_filename_preview"):
            return
        mapping = self._cmp_current_mapping()
        loaded_cubes = self.loaded.compare_cubes if self.loaded and self.loaded.mode == "Compare" else None
        background = self._cmp_background_value(loaded_cubes, update_spin=loaded_cubes is not None)
        kk_title = compare_source_title(mapping["KK"]) if "KK" in mapping else "Assign KK"
        kkp_title = compare_source_title(mapping["KKp"]) if "KKp" in mapping else "Assign KKp"
        self.cmp_kk_title_preview.setText(kk_title)
        self.cmp_kkp_title_preview.setText(kkp_title)
        if "KK" in mapping and "KKp" in mapping:
            base = vp_compare_export_base(mapping, background, self._cmp_scale_tag())
            title = vp_compare_title(mapping, background, self._cmp_scale_tag())
            self.cmp_vp_filename_preview.setText(f"{base}.png / .dat")
            self.cmp_vp_title_preview.setText(title)
        else:
            self.cmp_vp_filename_preview.setText("Assign KK and KKp")
            self.cmp_vp_title_preview.setText("Assign KK and KKp")

    def _cmp_current_mapping(self) -> dict[str, str]:
        mapping = {}
        used = set()
        for key in COMPARE_PANEL_ORDER:
            name = self.cmp_channel_combos[key].currentText().strip()
            if not name or name in used:
                continue
            mapping[key] = name
            used.add(name)
        return mapping

    def _cmp_visible_channels(self, mapping=None) -> list[str]:
        mapping = mapping or self._cmp_current_mapping()
        preset = self.cmp_display_preset_combo.currentText()
        if preset == "KK + KKp":
            order = ["KK", "KKp"]
        elif preset == "KpK + KpKp":
            order = ["KpK", "KpKp"]
        elif preset == "All four":
            order = list(COMPARE_PANEL_ORDER)
        else:
            order = [key for key in COMPARE_PANEL_ORDER if self.cmp_show_checks[key].isChecked()]
        return [key for key in order if key in mapping]

    def _cmp_apply_display_preset(self) -> None:
        preset = self.cmp_display_preset_combo.currentText()
        enabled = preset == "Custom" and not self._cmp_is_vp_view()
        desired = {
            "KK + KKp": {"KK", "KKp"},
            "KpK + KpKp": {"KpK", "KpKp"},
            "All four": set(COMPARE_PANEL_ORDER),
            "Custom": None,
        }[preset]
        for key, chk in self.cmp_show_checks.items():
            chk.setEnabled(enabled)
            if desired is not None:
                old = chk.blockSignals(True)
                try:
                    chk.setChecked(key in desired)
                finally:
                    chk.blockSignals(old)

    def _cmp_update_assignment_summary(self) -> None:
        mapping = self._cmp_current_mapping()
        visible = self._cmp_visible_channels(mapping)
        lines = self._cmp_assignment_advisories(mapping)
        for key in COMPARE_PANEL_ORDER:
            lines.append(f"{key} -> {mapping.get(key, 'missing')}")
        if self._cmp_is_vp_view():
            lines.append("Visible -> VP from KK, KKp" if "KK" in mapping and "KKp" in mapping else "Visible -> VP needs KK and KKp")
        elif visible:
            lines.append("Visible -> " + ", ".join(visible))
        else:
            lines.append("Visible -> none")
        lines.append(f"Sources -> {self._cmp_source_filter_label()}")
        self.cmp_assignment_summary.setPlainText("\n".join(lines))
        self._cmp_update_group_badge()
        self._cmp_update_title_previews()

    def _cmp_update_view_mode(self) -> None:
        vp_mode = self._cmp_is_vp_view()
        self.cmp_display_preset_combo.setEnabled(not vp_mode)
        for chk in self.cmp_show_checks.values():
            chk.setEnabled((not vp_mode) and self.cmp_display_preset_combo.currentText() == "Custom")
        self._update_plot_view_bar_visibility()

    def _cmp_selection_from_ui(self) -> data_io.CompareSelection:
        mapping = self._cmp_current_mapping()
        if self._cmp_is_vp_view():
            missing = [key for key in ("KK", "KKp") if key not in mapping]
            if missing:
                raise ValueError("VP needs assigned KK and KKp channels.")
            return data_io.CompareSelection.from_mapping(mapping, visible_order=("KK", "KKp"))
        visible = self._cmp_visible_channels(mapping)
        if len(visible) < 1:
            raise ValueError("Assign at least one compare channel.")
        if len(visible) < 2:
            raise ValueError("Select at least two visible compare channels.")
        return data_io.CompareSelection.from_mapping(mapping, visible_order=visible)

    def _cmp_infer_angle_references(self) -> None:
        candidates = self._cmp_assign_candidate_files()
        selected_sources = {
            str(source) for source in (getattr(self, "cmp_selected_group_sources", ()) or ())
        }
        if selected_sources:
            scoped = [source for source in candidates if source in selected_sources]
            if scoped:
                candidates = scoped
        inference = infer_compare_angle_references(
            candidates,
            in_k_anchor=float(self.cmp_in_k_angle_spin.value()),
            out_k_anchor=float(self.cmp_out_k_angle_spin.value()),
            cluster_tolerance=float(self.cmp_angle_tolerance_spin.value()),
        )
        for spin, value in (
            (self.cmp_in_k_angle_spin, inference.in_k),
            (self.cmp_in_kp_angle_spin, inference.in_kp),
            (self.cmp_out_k_angle_spin, inference.out_k),
            (self.cmp_out_kp_angle_spin, inference.out_kp),
        ):
            if value is None:
                continue
            blocked = spin.blockSignals(True)
            try:
                spin.setValue(float(value))
            finally:
                spin.blockSignals(blocked)

        def _clusters_text(values: tuple[float, ...]) -> str:
            return ", ".join(f"{value:.3g}" for value in values) if values else "none"

        self._append_log(
            "Angle inference: "
            f"Rot1 clusters=[{_clusters_text(inference.rot1_clusters)}], "
            f"Rot2 clusters=[{_clusters_text(inference.rot2_clusters)}]"
        )
        suggestions: list[str] = []
        if inference.in_k is not None and inference.in_kp is not None:
            suggestions.append(f"In K={inference.in_k:.3g}, In Kp={inference.in_kp:.3g}")
        if inference.out_k is not None and inference.out_kp is not None:
            suggestions.append(f"Out K={inference.out_k:.3g}, Out Kp={inference.out_kp:.3g}")
        if suggestions:
            self._append_log("  suggested: " + "; ".join(suggestions))
        else:
            self._append_log(
                "  no references changed: inference requires exactly two clusters for an arm"
            )

    def _cmp_auto_assign_channels(
        self, *, preserve_existing: bool = True, allow_inference: bool = True
    ) -> None:
        candidates = self._cmp_assign_candidate_files()
        in_k = float(self.cmp_in_k_angle_spin.value())
        in_kp = float(self.cmp_in_kp_angle_spin.value())
        out_k = float(self.cmp_out_k_angle_spin.value())
        out_kp = float(self.cmp_out_kp_angle_spin.value())
        tolerance = float(self.cmp_angle_tolerance_spin.value())
        selected_group_key = str(getattr(self, "cmp_selected_group_key", "") or "")
        had_selected_group = bool(selected_group_key)
        selected_group = None
        if selected_group_key:
            selected_group = next(
                (
                    group for group in self._cmp_source_groups(candidates)
                    if str(self._cmp_group_value(group, "key", "")) == selected_group_key
                ),
                None,
            )
        elif (not preserve_existing) or not bool(getattr(self, "cmp_group_selection_cleared", False)):
            groups = self._cmp_source_groups(candidates)
            if groups:
                selected_group = max(
                    enumerate(groups),
                    key=lambda pair: (
                        int("KK" in self._cmp_group_mapping(pair[1]) and "KKp" in self._cmp_group_mapping(pair[1])),
                        len(self._cmp_group_mapping(pair[1])),
                        -pair[0],
                    ),
                )[1]
                selected_group_key = str(self._cmp_group_value(selected_group, "key", ""))

        # Keep inference scoped to the selected group and use it only as a
        # fallback.  A manual mapping with at least two channels remains
        # authoritative, including a manual KK/KKp swap on refresh.
        current_mapping = self._cmp_current_mapping()
        inferred_selected_mapping: dict[str, str] | None = None
        manual_mapping_is_usable = (
            had_selected_group
            and preserve_existing
            and len(current_mapping) >= 2
            and all(value in candidates for value in current_mapping.values())
        )
        if allow_inference and selected_group is not None and not manual_mapping_is_usable:
            current_group_mapping = self._cmp_group_mapping(selected_group)
            if len(current_group_mapping) < 2:
                selected_sources = tuple(
                    self._cmp_group_value(selected_group, "sources", ()) or ()
                )
                inference = infer_compare_angle_references(
                    selected_sources,
                    in_k_anchor=in_k,
                    out_k_anchor=out_k,
                    cluster_tolerance=tolerance,
                )
                inferred_values = {
                    "in_k_angle": inference.in_k,
                    "in_kp_angle": inference.in_kp,
                    "out_k_angle": inference.out_k,
                    "out_kp_angle": inference.out_kp,
                }
                refs = {
                    "in_k_angle": in_k,
                    "in_kp_angle": in_kp,
                    "out_k_angle": out_k,
                    "out_kp_angle": out_kp,
                }
                refs.update({key: float(value) for key, value in inferred_values.items() if value is not None})
                inferred_groups = group_compare_sources(
                    selected_sources,
                    in_k_angle=refs["in_k_angle"],
                    in_kp_angle=refs["in_kp_angle"],
                    out_k_angle=refs["out_k_angle"],
                    out_kp_angle=refs["out_kp_angle"],
                    tolerance=tolerance,
                    power_tolerance_fraction=self._cmp_group_tolerance_fraction(),
                )
                inferred_found: dict[str, str] = {}
                if inferred_groups:
                    inferred_group = max(
                        inferred_groups,
                        key=lambda group: len(self._cmp_group_mapping(group)),
                    )
                    inferred_found = self._cmp_group_mapping(inferred_group)
                if len(inferred_found) > len(current_group_mapping):
                    inferred_selected_mapping = inferred_found
                    for spin, key in (
                        (self.cmp_in_k_angle_spin, "in_k_angle"),
                        (self.cmp_in_kp_angle_spin, "in_kp_angle"),
                        (self.cmp_out_k_angle_spin, "out_k_angle"),
                        (self.cmp_out_kp_angle_spin, "out_kp_angle"),
                    ):
                        value = inferred_values[key]
                        if value is None:
                            continue
                        blocked = spin.blockSignals(True)
                        try:
                            spin.setValue(float(value))
                        finally:
                            spin.blockSignals(blocked)
                    in_k = float(self.cmp_in_k_angle_spin.value())
                    in_kp = float(self.cmp_in_kp_angle_spin.value())
                    out_k = float(self.cmp_out_k_angle_spin.value())
                    out_kp = float(self.cmp_out_kp_angle_spin.value())
                    # Group keys are context/power based, but refresh the
                    # record so its mapping reflects the committed references.
                    selected_group = next(
                        (
                            group for group in self._cmp_source_groups(candidates)
                            if str(self._cmp_group_value(group, "key", "")) == selected_group_key
                        ),
                        selected_group,
                    )
        found, duplicates, gate_group, gate_groups = coherent_compare_auto_assignment(
            candidates,
            in_k_angle=in_k,
            in_kp_angle=in_kp,
            out_k_angle=out_k,
            out_kp_angle=out_kp,
            tolerance=tolerance,
        )
        for key in duplicates:
            found.pop(key, None)
        if selected_group_key:
            if selected_group is None:
                # A folder refresh can temporarily omit a selected group. Keep
                # the user's assignment intact instead of jumping to another
                # measurement group.
                found = current_mapping if all(value in candidates for value in current_mapping.values()) else {}
            elif (
                had_selected_group
                and preserve_existing
                and current_mapping
                and inferred_selected_mapping is None
                and all(value in candidates for value in current_mapping.values())
            ):
                found = current_mapping
            else:
                found = (
                    dict(inferred_selected_mapping)
                    if inferred_selected_mapping is not None
                    else self._cmp_group_mapping(selected_group)
                )
                if bool(getattr(self, "cmp_kk_swap_active", False)):
                    found["KK"], found["KKp"] = found.get("KKp", ""), found.get("KK", "")
        if selected_group is not None:
            # The selected identity follows the same context after angle edits;
            # a refresh keeps an existing manual swap if its files remain.
            self.cmp_selected_group_key = selected_group_key
            self.cmp_selected_group_sources = tuple(
                self._cmp_group_value(selected_group, "sources", ()) or ()
            )
            self.cmp_selected_group_label = str(
                self._cmp_group_value(selected_group, "label", selected_group_key)
            )
            self.cmp_group_selection_cleared = False
        if bool(getattr(self, "cmp_group_selection_cleared", False)) and preserve_existing and not had_selected_group:
            found = {}
        self._cmp_set_mapping(found, update_summary=False)
        self._cmp_update_assignment_summary()
        classified_counts: dict[str, int] = {}
        group_keys: dict[str, set[str]] = {}
        rejected: list[str] = []
        unlabeled_count = 0
        for fname in candidates:
            ch = classify_compare_channel(
                fname,
                in_k_angle=in_k,
                in_kp_angle=in_kp,
                out_k_angle=out_k,
                out_kp_angle=out_kp,
                tolerance=tolerance,
            )
            if ch:
                classified_counts[ch] = classified_counts.get(ch, 0) + 1
                gk = parse_compare_gate_condition(fname) or "__ungrouped__"
                group_keys.setdefault(gk, set()).add(ch)
                continue
            angles = parse_compare_rotation_angles(fname)
            if angles.rot1 is None and angles.rot2 is None:
                unlabeled_count += 1
            reasons: list[str] = []
            for arm, angle, k_angle, kp_angle in (
                ("Rot1", angles.rot1, in_k, in_kp),
                ("Rot2", angles.rot2, out_k, out_kp),
            ):
                if angle is None:
                    continue
                match = classify_angle_state(
                    angle, k_angle=k_angle, kp_angle=kp_angle, tolerance=tolerance
                )
                if match.state is None:
                    reasons.append(
                        f"{arm}={angle:g}: {match.reason} "
                        f"(dK={match.distance_k:.3g}, dKp={match.distance_kp:.3g})"
                    )
            if reasons:
                rejected.append(f"{Path(fname).name}: " + "; ".join(reasons))
        assigned = [k for k in ("KK", "KKp", "KpK", "KpKp") if k in found]
        missing = [k for k in ("KK", "KKp", "KpK", "KpKp") if k not in found]
        self._append_log(
            f"Auto-assign (InK={in_k:.1f}, InKp={in_kp:.1f}, "
            f"OutK={out_k:.1f}, OutKp={out_kp:.1f}, tol={tolerance:.1f} deg): "
            + f"classified {classified_counts} across {len(group_keys)} group(s)"
        )
        for detail in rejected[:8]:
            self._append_log(f"  unassigned angle: {detail}")
        if unlabeled_count:
            self._append_log(
                f"  skipped {unlabeled_count} file(s) without recognizable angles or an "
                "unambiguous channel label; available for manual assignment"
            )
        if len(rejected) > 8:
            self._append_log(f"  +{len(rejected) - 8} more unassigned angle match(es)")
        for gk, keys in sorted(group_keys.items()):
            marker = " <-- selected" if gk == (gate_group or "__ungrouped__") else ""
            self._append_log(f"  group [{gk}]: keys={sorted(keys)}{marker}")
        if assigned:
            self._append_log(f"  assigned: {', '.join(assigned)}")
            for fname in dict.fromkeys(found[key] for key in assigned):
                angles = parse_compare_rotation_angles(fname)
                if (angles.rot1 is None) != (angles.rot2 is None):
                    detected = (
                        f"Rot1={angles.rot1:g} deg"
                        if angles.rot1 is not None
                        else f"Rot2={angles.rot2:g} deg"
                    )
                    fixed = "output" if angles.rot1 is not None else "input"
                    self._append_log(
                        f"  partial rotation: {detected}; missing fixed {fixed} arm treated as K"
                    )
        if missing:
            reason_parts: list[str] = []
            for mk in missing:
                if mk not in classified_counts:
                    reason_parts.append(f"{mk}=no file classified as {mk}")
                else:
                    in_selected = mk in group_keys.get(gate_group or "__ungrouped__", set())
                    reason_parts.append(
                        f"{mk}={'only in other gate group(s)' if not in_selected else 'duplicate (already assigned)'}"
                    )
            self._append_log(f"  MISSING: {'; '.join(reason_parts)}")
        if gate_group and len(set(gate_groups)) > 1:
            self._append_log(
                "Compare auto-detect found multiple gate groups: "
                + ", ".join(sorted(set(gate_groups))) + f". Using {gate_group}."
            )
        if duplicates:
            dup_text = "; ".join(f"{k}: {', '.join(v)}" for k, v in duplicates.items())
            self._append_log(f"Compare auto-detect found duplicate matches -> {dup_text}")
        self._on_cmp_plot_param_changed()

    def _set_cmp_gate_spin_value(self, gate_value: float) -> None:
        spin = self.cmp_spins["gate"]
        old = spin.blockSignals(True)
        try:
            spin.setValue(float(gate_value))
        finally:
            spin.blockSignals(old)

    def _ensure_cmp_gate_lines(self, cubes, gate_value: float) -> None:
        active_keys = set(cubes.keys())
        for key in list(self._cmp_gate_lines.keys()):
            if key in active_keys and key in self._cmp_heatmap_axes:
                continue
            line = self._cmp_gate_lines.pop(key, None)
            if line is not None:
                try:
                    line.remove()
                except Exception:
                    pass
        for key, cube in cubes.items():
            ax = self._cmp_heatmap_axes.get(key)
            if ax is None:
                continue
            gate = np.asarray(cube.gate, float).ravel()
            gate_clamped = float(
                np.clip(gate_value, float(np.nanmin(gate)), float(np.nanmax(gate)))
            )
            line = self._cmp_gate_lines.get(key)
            if line is None or getattr(line, "axes", None) is not ax:
                self._cmp_gate_lines[key] = ax.axhline(
                    y=gate_clamped,
                    lw=1.2,
                    alpha=0.95,
                    color="#222",
                    linestyle="--",
                    zorder=50,
                )
            else:
                line.set_ydata([gate_clamped, gate_clamped])
                line.set_linestyle("--")

    def _on_cmp_auto_assign_requested(self) -> None:
        self._invalidate_export_move_sources()
        self.cmp_kk_swap_active = False
        self._cmp_auto_assign_channels(preserve_existing=False, allow_inference=True)
        self._on_cmp_plot_param_changed()

    def _on_cmp_angle_reference_changed(self) -> None:
        """Reclassify with edited references without re-inferring them."""
        self._invalidate_export_move_sources()
        self.cmp_kk_swap_active = False
        self._cmp_auto_assign_channels(preserve_existing=False, allow_inference=False)
        self._on_cmp_plot_param_changed()

    def _on_cmp_infer_angles_requested(self) -> None:
        self._invalidate_export_move_sources()
        self.cmp_kk_swap_active = False
        self._cmp_infer_angle_references()
        self._cmp_auto_assign_channels(preserve_existing=False, allow_inference=False)
        self._on_cmp_plot_param_changed()

    def _on_cmp_display_preset_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._cmp_apply_display_preset()
        self._cmp_update_assignment_summary()
        if self.loaded and self.loaded.mode == "Compare":
            self._refresh_automatic_ranges("Compare", refresh_split=True)
        self._on_cmp_plot_param_changed()

    def _on_cmp_plot_view_button_clicked(self, mode: str) -> None:
        self._cmp_set_view_mode(mode)
        self._on_cmp_view_changed()

    def _on_cmp_view_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._cmp_update_view_mode()
        if self._cmp_is_vp_view() and hasattr(self, "cmp_vp_expander"):
            head = self.cmp_vp_expander.findChild(QToolButton)
            if head is not None:
                head.setChecked(True)
        self._cmp_update_assignment_summary()
        self._update_action_states()
        if self.loaded and self.loaded.mode == "Compare" and not self._cmp_is_vp_view():
            self._refresh_automatic_ranges("Compare", refresh_split=True)
        self._on_cmp_plot_param_changed()

    def _on_cmp_background_mode_changed(self, _checked: bool) -> None:
        self._invalidate_export_move_sources()
        self._cmp_update_background_mode()
        if self.loaded and self.loaded.mode == "Compare" and self.loaded.compare_cubes:
            self._cmp_background_value(self.loaded.compare_cubes)
        self._on_cmp_plot_param_changed(self.cmp_vp_auto_background_chk)

    def _on_cmp_plot_param_changed(self, source=None) -> None:
        self._invalidate_export_move_sources()
        self._cmp_update_assignment_summary()
        if self.loaded and self.loaded.mode == "Compare":
            sender = source
            if sender in (
                self.cmp_spins["xmin"], self.cmp_spins["xmax"],
                self.cmp_spins["ymin"], self.cmp_spins["ymax"],
                self.cmp_log_chk, self.cmp_vp_background_spin,
                self.cmp_vp_auto_background_chk,
            ) or sender in tuple(self.cmp_channel_combos.values()) or sender in tuple(self.cmp_show_checks.values()):
                self._refresh_automatic_ranges(
                    "Compare",
                    refresh_split=True,
                    center_split=sender in (self.cmp_spins["xmin"], self.cmp_spins["xmax"]),
                )
            self._schedule_plot_redraw("Compare")

    def _cmp_vp_color_limits(self) -> tuple[float, float]:
        """Return the manually selected VP color limits, bounded to [-1, 1]."""
        spins = getattr(self, "cmp_vp_spins", None)
        if not spins:
            return -1.0, 1.0
        vmin = max(-1.0, min(1.0, float(spins["vmin"].value())))
        vmax = max(-1.0, min(1.0, float(spins["vmax"].value())))
        if vmax <= vmin:
            raise ValueError("Compare VP color scale requires vmin < vmax.")
        return vmin, vmax

    def _auto_cmp_vp_range(self) -> None:
        """Set unlocked VP limits from finite extrema in the current VP ROI."""
        if not self.loaded or self.loaded.mode != "Compare" or not self.loaded.compare_cubes:
            return
        source_files = self._cmp_source_mapping()
        raw_cubes = {
            key: self.loaded.compare_cubes[key]
            for key in COMPARE_PANEL_ORDER
            if key in self.loaded.compare_cubes
        }
        if "KK" not in raw_cubes or "KKp" not in raw_cubes:
            return
        background = self._cmp_background_value(raw_cubes)
        vp_cube = self._cmp_vp_cube(raw_cubes, source_files, background=background)
        x0, x1 = sorted((float(self.cmp_spins["xmin"].value()), float(self.cmp_spins["xmax"].value())))
        y0, y1 = sorted((float(self.cmp_spins["ymin"].value()), float(self.cmp_spins["ymax"].value())))
        x = np.asarray(vp_cube.energy, float).ravel()
        y = np.asarray(vp_cube.gate, float).ravel()
        z = np.asarray(vp_cube.Z, float)
        x_mask = (x >= x0) & (x <= x1)
        y_mask = (y >= y0) & (y <= y1)
        region = z[np.ix_(y_mask, x_mask)] if np.any(x_mask) and np.any(y_mask) else z
        finite = region[np.isfinite(region)]
        if finite.size:
            vmin = max(-1.0, min(1.0, float(np.min(finite))))
            vmax = max(-1.0, min(1.0, float(np.max(finite))))
        else:
            vmin, vmax = -1.0, 1.0

        # A constant VP field still needs a valid Normalize range.  Pad in
        # the physical interval, using the full interval at an all-NaN ROI.
        if vmax <= vmin:
            center = max(-1.0, min(1.0, vmin))
            pad = min(0.05, max(0.005, abs(center) * 0.01))
            vmin = max(-1.0, center - pad)
            vmax = min(1.0, center + pad)
            if vmax <= vmin:
                vmin, vmax = -1.0, 1.0

        spins = getattr(self, "cmp_vp_spins", None)
        checks = getattr(self, "cmp_vp_fix_checks", None)
        if not spins or not checks:
            return
        if checks["vmin"].isChecked():
            vmin = float(spins["vmin"].value())
        if checks["vmax"].isChecked():
            vmax = float(spins["vmax"].value())
        if vmax <= vmin:
            self._status("Auto VP color scale conflicts with a fixed limit; adjust or unfix it first.")
            return
        updated: list[str] = []
        for key, value in (("vmin", vmin), ("vmax", vmax)):
            if checks[key].isChecked():
                continue
            old = spins[key].blockSignals(True)
            try:
                spins[key].setValue(float(value))
            finally:
                spins[key].blockSignals(old)
            updated.append(key)
        if updated:
            self._status(f"Auto VP color scale = {vmin:.4g}, {vmax:.4g}.")
        self._schedule_plot_redraw("Compare")

    def _auto_cmp_vrange(self) -> None:
        if self._cmp_is_vp_view():
            self._auto_cmp_vp_range()
            return
        if not self.loaded or self.loaded.mode != "Compare" or not self.loaded.compare_cubes:
            return
        x0, x1 = sorted((float(self.cmp_spins["xmin"].value()), float(self.cmp_spins["xmax"].value())))
        y0, y1 = sorted((float(self.cmp_spins["ymin"].value()), float(self.cmp_spins["ymax"].value())))
        vals: list[np.ndarray] = []
        background = self._cmp_background_value(self.loaded.compare_cubes)
        for cube in self._cmp_corrected_cubes(self.loaded.compare_cubes, background=background).values():
            x = np.asarray(cube.energy, float).ravel()
            y = np.asarray(cube.gate, float).ravel()
            z = np.asarray(cube.Z, float)
            x_mask = (x >= x0) & (x <= x1)
            y_mask = (y >= y0) & (y <= y1)
            z_roi = z[np.ix_(y_mask, x_mask)] if np.any(y_mask) and np.any(x_mask) else z
            finite = z_roi[np.isfinite(z_roi)]
            if finite.size:
                vals.append(finite)
        if not vals:
            return
        finite = np.concatenate(vals)
        if self._mode_log("Compare"):
            pos = finite[finite > 0]
            if pos.size:
                vmin, vmax = np.nanpercentile(pos, [0.01, 99.99])
                vmin = float(max(vmin, 1e-12))
                vmax = float(max(vmax, vmin * 1.01))
            else:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            vmin, vmax = np.nanpercentile(finite, [0.01, 99.99])
            vmin, vmax = float(vmin), float(vmax)
        self.cmp_spins["vmin"].setValue(vmin)
        self.cmp_spins["vmax"].setValue(vmax)
        self._schedule_plot_redraw("Compare")

    def _auto_cmp_xrange(self) -> None:
        if not self.loaded or self.loaded.mode != "Compare" or not self.loaded.compare_cubes:
            return
        mins = [float(np.nanmin(c.energy)) for c in self.loaded.compare_cubes.values()]
        maxs = [float(np.nanmax(c.energy)) for c in self.loaded.compare_cubes.values()]
        self.cmp_spins["xmin"].setValue(min(mins))
        self.cmp_spins["xmax"].setValue(max(maxs))
        self._schedule_plot_redraw("Compare")

    def _auto_cmp_yrange(self) -> None:
        if not self.loaded or self.loaded.mode != "Compare" or not self.loaded.compare_cubes:
            return
        mins = [float(np.nanmin(c.gate)) for c in self.loaded.compare_cubes.values()]
        maxs = [float(np.nanmax(c.gate)) for c in self.loaded.compare_cubes.values()]
        self.cmp_spins["ymin"].setValue(min(mins))
        self.cmp_spins["ymax"].setValue(max(maxs))
        self._schedule_plot_redraw("Compare")
