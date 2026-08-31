"""Controller for Compare workflow actions."""

from __future__ import annotations

from ui_qt.main_window import *


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
        self._on_cmp_plot_param_changed()

    def _on_cmp_plot_param_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._cmp_update_assignment_summary()
        if self.loaded and self.loaded.mode == "Compare":
            sender = self.sender()
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
