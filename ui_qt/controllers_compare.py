"""Controller for Compare workflow actions."""

from __future__ import annotations

import numpy as np
from pathlib import Path

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
)


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
        return list(self.available_files)

    def _cmp_set_channel_combo_items(self) -> None:
        files = [""] + list(self.available_files)
        for combo in self.cmp_channel_combos.values():
            current = combo.currentText()
            old = combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItems(files)
                if current in files:
                    combo.setCurrentText(current)
            finally:
                combo.blockSignals(old)

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
        lines = []
        for key in COMPARE_PANEL_ORDER:
            lines.append(f"{key} -> {mapping.get(key, 'missing')}")
        if self._cmp_is_vp_view():
            lines.append("Visible -> VP from KK, KKp" if "KK" in mapping and "KKp" in mapping else "Visible -> VP needs KK and KKp")
        elif visible:
            lines.append("Visible -> " + ", ".join(visible))
        else:
            lines.append("Visible -> none")
        self.cmp_assignment_summary.setPlainText("\n".join(lines))
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
        inference = infer_compare_angle_references(
            self._cmp_assign_candidate_files(),
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

    def _cmp_auto_assign_channels(self) -> None:
        candidates = self._cmp_assign_candidate_files()
        in_k = float(self.cmp_in_k_angle_spin.value())
        in_kp = float(self.cmp_in_kp_angle_spin.value())
        out_k = float(self.cmp_out_k_angle_spin.value())
        out_kp = float(self.cmp_out_kp_angle_spin.value())
        tolerance = float(self.cmp_angle_tolerance_spin.value())
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
        for key, combo in self.cmp_channel_combos.items():
            old = combo.blockSignals(True)
            try:
                combo.setCurrentText(found.get(key, ""))
            finally:
                combo.blockSignals(old)
        self._cmp_update_assignment_summary()
        classified_counts: dict[str, int] = {}
        group_keys: dict[str, set[str]] = {}
        rejected: list[str] = []
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
        self._cmp_auto_assign_channels()
        self._on_cmp_plot_param_changed()

    def _on_cmp_infer_angles_requested(self) -> None:
        self._invalidate_export_move_sources()
        self._cmp_infer_angle_references()
        self._cmp_auto_assign_channels()
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

    def _auto_cmp_vrange(self) -> None:
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
