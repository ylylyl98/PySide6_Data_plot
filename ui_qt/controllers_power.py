"""Controller for Power-dependent view interactions."""

from ui_qt.main_window import *


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

    def _on_power_axis_scale_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._power_update_group_summary()
        self._on_power_plot_param_changed()

    def _on_power_background_mode_changed(self, _checked: bool) -> None:
        self._invalidate_export_move_sources()
        self.power_background_spin.setEnabled(not self._power_background_auto_enabled())
        self._on_power_plot_param_changed()

    def _on_power_plot_param_changed(self, sender=None) -> None:
        self._invalidate_export_move_sources()
        self._power_update_group_summary()
        if self.loaded and self.loaded.mode == "Power Dependent":
            if sender in (
                self.power_spins["xmin"], self.power_spins["xmax"],
                self.power_spins["ymin"], self.power_spins["ymax"],
                self.power_log_chk, self.power_background_spin,
                self.power_background_auto_chk,
            ):
                self._refresh_automatic_ranges(
                    "Power Dependent",
                    refresh_split=True,
                    center_split=sender in (self.power_spins["xmin"], self.power_spins["xmax"]),
                )
            self._schedule_plot_redraw("Power Dependent")

    def _on_power_source_assignment_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._power_update_group_summary()
        self._power_update_vp_availability()
        if self.loaded and self.loaded.mode == "Power Dependent":
            self._refresh_automatic_ranges("Power Dependent", refresh_split=True)
            self._schedule_plot_redraw("Power Dependent")

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
