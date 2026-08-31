"""Controller for MCD processing and interactive display behavior."""

from __future__ import annotations

from ui_qt.main_window import *


class McdController:
    """Own MCD reactions while sharing the application state through owner."""

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

    def _on_mcd_params_changed(self) -> None:
        self._invalidate_export_move_sources()
        self._update_mcd_background_preview()
        if self.loaded and self.loaded.mode == "MCD":
            self.mcd_apply_correction_btn.setText("Pending update...")
            self._status("MCD processing settings changed; recalculation is scheduled automatically.")
            self._mcd_auto_apply_timer.start()

    def _apply_mcd_now(self) -> None:
        self._mcd_auto_apply_timer.stop()
        self._apply_pending_mcd_settings()

    def _apply_pending_mcd_settings(self) -> None:
        """Apply the newest MCD settings once after a burst of UI changes."""
        if not self.loaded or self.loaded.mode != "MCD":
            return
        if self._load_in_progress:
            self._mcd_reapply_pending = True
            self.mcd_apply_correction_btn.setText("Pending update...")
            return
        if not self._selected(self.mcd_files):
            self.mcd_apply_correction_btn.setText("Recalculate now")
            return
        try:
            self._mcd_settings_from_ui()
        except Exception as exc:
            self.mcd_apply_correction_btn.setText("Fix settings")
            self._status(f"MCD automatic recalculation paused: {exc}")
            return
        self._mcd_reapply_pending = False
        self.mcd_apply_correction_btn.setText("Applying...")
        self._status("Applying updated MCD processing settings...")
        self._start_load("MCD")

    def _on_mcd_correction_mode_changed(self, text: str) -> None:
        self.mcd_spectral_order_combo.setEnabled("spectral baseline" in text.lower())
        self._on_mcd_params_changed()

    def _on_mcd_background_ranges_changed(self) -> None:
        self._mcd_background_suggestion = None
        self._on_mcd_params_changed()

    def _on_mcd_source_changed(self) -> None:
        self._mcd_background_suggestion = None
        self._mcd_center_refresh_timer.stop()
        self._update_mcd_selection_summary()
        self._mcd_detect_available_angles()
        if not self._selected(self.mcd_files):
            return
        self._mcd_auto_apply_timer.stop()
        if self._load_in_progress:
            self._mcd_reapply_pending = True
            self.mcd_apply_correction_btn.setText("Pending update...")
            return
        self._status("Loading the selected MCD source...")
        self._start_load("MCD")

    def _on_mcd_angle_assignment_changed(self, automatic: bool) -> None:
        self.mcd_sigma_plus_combo.setEnabled(not automatic)
        self.mcd_sigma_minus_combo.setEnabled(not automatic)
        if automatic:
            self._mcd_detect_available_angles()
        self._on_mcd_params_changed()

    def _on_mcd_reference_mode_changed(self, _text: str) -> None:
        self.mcd_zero_spin.setEnabled(self.mcd_reference_mode_combo.currentIndex() == 1)
        self._on_mcd_params_changed()

    def _on_mcd_plot_changed(self, source=None) -> None:
        self._invalidate_export_move_sources()
        sender = source if source is not None else self._owner.sender()
        if self._mcd_center_candidates:
            if sender in (
                self.mcd_window_width_spin, self.mcd_window_metric_combo,
                self.mcd_spins["xmin"], self.mcd_spins["xmax"],
            ):
                self._clear_mcd_center_candidates(restore_manual=False)
            elif sender is self.mcd_window_center_spin and not self._mcd_candidate_applying:
                self._mcd_candidate_active_index = None
                self._update_mcd_candidate_bar()
        if (
            sender is self.mcd_window_center_spin
            and not self._mcd_candidate_applying
            and self.loaded is not None
            and self.loaded.mode == "MCD"
            and self.loaded.mcd_result is not None
        ):
            self.loaded.mcd_result.summary["window_center_selection"] = {"method": "manual"}
        if self.loaded and self.loaded.mode == "MCD":
            if sender in (self.mcd_map_combo, self.mcd_center_zero_chk):
                self._refresh_automatic_ranges("MCD", refresh_split=False)
            center_only_controls = {
                self.mcd_window_center_spin,
                self.mcd_window_width_spin,
                self.mcd_fit_b_window_spin,
            }
            if sender in center_only_controls:
                # Spin boxes can emit many values while their arrows are held.
                # Coalesce those events and render only the latest requested value.
                self._mcd_center_refresh_timer.start()
                return
            self._mcd_center_refresh_timer.stop()
            trace_only_controls = {
                self.mcd_window_metric_combo,
                self.mcd_show_raw_chk,
                self.mcd_show_signed_mean_chk,
                self.mcd_show_absolute_mean_chk,
                self.mcd_show_unsigned_absolute_mean_chk,
                self.mcd_show_integral_chk,
                self.mcd_fit_zero_chk,
            }
            if sender in trace_only_controls and self._refresh_mcd_trace_panel():
                return
            self._plot_mode("MCD")

    def _apply_pending_mcd_center_refresh(self) -> None:
        if not self.loaded or self.loaded.mode != "MCD":
            return
        if not self._refresh_mcd_center_trace():
            self._plot_mode("MCD")

    def _mcd_window_metric(self) -> str:
        return {
            "Field-signed absolute mean": "field_signed_absolute_mean",
            "Signed mean": "mean",
            "Signed integral": "integral",
            "Unsigned absolute mean (diagnostic)": "absolute_mean",
        }.get(self.mcd_window_metric_combo.currentText(), "mean")

    def _clear_mcd_center_candidates(self, *, restore_manual: bool) -> None:
        manual_center = self._mcd_manual_center_before_suggestions
        self._mcd_center_candidates = ()
        self._mcd_candidate_active_index = None
        self._mcd_manual_center_before_suggestions = None
        self._mcd_candidate_search_range = None
        self._update_mcd_candidate_bar()
        removed_candidate_artists = set(self._mcd_candidate_artists.values())
        for artist in removed_candidate_artists:
            try:
                artist.remove()
            except (KeyError, ValueError):
                pass
        self._mcd_candidate_artists = {}
        if self._mcd_blit_enabled:
            self._mcd_heat_dynamic_artists = [
                artist for artist in self._mcd_heat_dynamic_artists
                if artist not in removed_candidate_artists
            ]
            self._mcd_overlay_artists["heat"] = self._mcd_heat_dynamic_artists
            self._blit_mcd_regions("heat")
        if self.loaded is not None and self.loaded.mode == "MCD" and self.loaded.mcd_result is not None:
            self.loaded.mcd_result.summary["window_center_selection"] = {"method": "manual"}
        if restore_manual and manual_center is not None:
            self.mcd_window_center_spin.setValue(float(manual_center))

    def _find_mcd_center_candidates(
        self,
        _checked: bool = False,
        *,
        replot: bool = True,
        announce: bool = True,
    ) -> None:
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            if announce:
                self._status("Load an MCD measurement before finding centers.")
            return
        result = self.loaded.mcd_result
        search_range = (
            float(self.mcd_spins["xmin"].value()),
            float(self.mcd_spins["xmax"].value()),
        )
        if search_range[1] <= search_range[0]:
            search_range = (float(np.nanmin(result.energy_ev)), float(np.nanmax(result.energy_ev)))
        self._mcd_manual_center_before_suggestions = float(self.mcd_window_center_spin.value())
        self._mcd_candidate_search_range = tuple(sorted(search_range))
        candidates = suggest_mcd_window_centers(
            result,
            float(self.mcd_window_width_spin.value()),
            metric=self._mcd_window_metric(),
            energy_range=search_range,
            max_candidates=5,
        )
        self._mcd_center_candidates = candidates
        self._mcd_candidate_active_index = None
        self._update_mcd_candidate_bar()
        if replot:
            self._plot_mode("MCD")
        if not candidates:
            if announce:
                self._status("No reliable MCD center candidates were found in the visible energy range.")
            return
        if announce:
            self._status(
                f"Found {len(candidates)} suggested MCD center"
                f"{'s' if len(candidates) != 1 else ''}. Click a numbered marker or energy chip to preview."
            )

    def _prepare_mcd_center_for_loaded_energy(self) -> None:
        """Keep a valid manual center before automatic suggestions snapshot it."""
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            return
        energy = np.asarray(self.loaded.mcd_result.energy_ev, float)
        finite = energy[np.isfinite(energy)]
        if finite.size == 0:
            return
        current = float(self.mcd_window_center_spin.value())
        if float(np.nanmin(finite)) <= current <= float(np.nanmax(finite)):
            return
        self._set_spin_value_silent(
            self.mcd_window_center_spin,
            float(np.nanmedian(finite)),
        )
        self._mcd_center_refresh_timer.start()

    def _use_mcd_center_candidate(self, index: int) -> None:
        if not 0 <= int(index) < len(self._mcd_center_candidates):
            return
        index = int(index)
        candidate = self._mcd_center_candidates[index]
        self._mcd_candidate_active_index = index
        self._update_mcd_candidate_bar()
        self._update_mcd_candidate_artist_styles()
        if self.loaded is not None and self.loaded.mcd_result is not None:
            self.loaded.mcd_result.summary["window_center_selection"] = {
                "method": "suggested",
                "candidate_energy_order": index + 1,
                "candidate_rank": candidate.score_rank,
                "candidate_count": len(self._mcd_center_candidates),
                "center_ev": candidate.center_ev,
                "score": candidate.score,
                "snr": candidate.snr,
                "branch_agreement": candidate.branch_agreement,
                "search_range_ev": self._mcd_candidate_search_range,
                "width_mev": float(self.mcd_window_width_spin.value()),
                "metric": self._mcd_window_metric(),
            }
        self._mcd_candidate_applying = True
        try:
            previous = float(self.mcd_window_center_spin.value())
            self.mcd_window_center_spin.setValue(candidate.center_ev)
            if np.isclose(previous, float(self.mcd_window_center_spin.value()), atol=5e-7, rtol=0.0):
                self.canvas.draw_idle()
        finally:
            self._mcd_candidate_applying = False
        self._status(
            f"Candidate {index + 1}/{len(self._mcd_center_candidates)}: "
            f"E = {format_mcd_energy(candidate.center_ev)} eV; SNR {candidate.snr:.3g}; "
            f"branch agreement {100.0 * candidate.branch_agreement:.0f}%."
        )

    def _step_mcd_center_candidate(self, direction: int) -> None:
        if not self._mcd_center_candidates:
            return
        current = self._mcd_candidate_active_index
        if current is None:
            target = 0 if direction >= 0 else len(self._mcd_center_candidates) - 1
        else:
            target = (current + int(direction)) % len(self._mcd_center_candidates)
        self._use_mcd_center_candidate(target)

    def _return_to_manual_mcd_center(self) -> None:
        if not self._mcd_center_candidates:
            return
        manual = self._mcd_manual_center_before_suggestions
        self._clear_mcd_center_candidates(restore_manual=True)
        if manual is not None:
            self._status(f"Returned to manual MCD center {format_mcd_energy(manual)} eV.")

    def _mcd_candidate_marker_at(self, xdata: float, ydata: float) -> int | None:
        if not self._mcd_center_candidates or self._mcd_heatmap_ax is None:
            return None
        ymin, ymax = (float(value) for value in self._mcd_heatmap_ax.get_ylim())
        span = ymax - ymin
        if span == 0 or (float(ydata) - ymin) / span < 0.86:
            return None
        xmin, xmax = (float(value) for value in self._mcd_heatmap_ax.get_xlim())
        tolerance = max(float(self.mcd_window_width_spin.value()) * 5e-4, 0.012 * abs(xmax - xmin))
        distances = np.asarray(
            [abs(float(xdata) - candidate.center_ev) for candidate in self._mcd_center_candidates],
            float,
        )
        index = int(np.argmin(distances))
        return index if distances[index] <= tolerance else None

    @staticmethod
    def _clamp_mcd_window_center(center_ev: float, energy_ev: np.ndarray, width_mev: float) -> float:
        finite = np.asarray(energy_ev, float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return float(center_ev)
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
        half = max(0.0, float(width_mev)) * 5e-4
        if 2.0 * half >= high - low:
            return 0.5 * (low + high)
        return float(np.clip(float(center_ev), low + half, high - half))

    def _add_mcd_window_overlay(
        self,
        axis: Any,
        center_ev: float,
        half_width_ev: float,
        *,
        draggable: bool = False,
    ) -> None:
        """Draw a low-obstruction fixed-width MCD energy band."""
        left = float(center_ev - half_width_ev)
        right = float(center_ev + half_width_ev)
        patch = axis.axvspan(left, right, color="#2f80c9", alpha=0.12, linewidth=0, zorder=20)
        edge_effect = [path_effects.Stroke(linewidth=2.4, foreground="#242424", alpha=0.55), path_effects.Normal()]
        left_line = axis.axvline(left, color="white", lw=0.9, alpha=0.95, zorder=22)
        right_line = axis.axvline(right, color="white", lw=0.9, alpha=0.95, zorder=22)
        left_line.set_path_effects(edge_effect)
        right_line.set_path_effects(edge_effect)
        marker = None
        if draggable:
            (marker,) = axis.plot(
                [center_ev], [0.995], marker="v", ms=5.0, color="white",
                markeredgecolor="#242424", markeredgewidth=0.7,
                transform=axis.get_xaxis_transform(), clip_on=False, zorder=23,
            )
        self._mcd_window_artists.append(
            {"patch": patch, "left": left_line, "right": right_line, "marker": marker}
        )

    def _move_mcd_window_artists(self, center_ev: float, *, draw: bool = True) -> None:
        half = float(self.mcd_window_width_spin.value()) * 5e-4
        left = float(center_ev - half)
        right = float(center_ev + half)
        for artists in self._mcd_window_artists:
            patch = artists["patch"]
            patch.set_x(left)
            patch.set_width(right - left)
            artists["left"].set_xdata([left, left])
            artists["right"].set_xdata([right, right])
            marker = artists.get("marker")
            if marker is not None:
                marker.set_xdata([center_ev])
        if draw:
            if not self._blit_mcd_regions(
                "pair_overlay", "linecut_overlay", "heat"
            ):
                self.canvas.draw_idle()

    def _mcd_window_contains_energy(self, energy_ev: float) -> bool:
        center = float(self.mcd_window_center_spin.value())
        half = float(self.mcd_window_width_spin.value()) * 5e-4
        return center - half <= float(energy_ev) <= center + half

    def _select_mcd_pair_at_b(self, requested_b: float, *, clicked: bool = False) -> None:
        """Select the nearest real pair for a B value requested on the map."""
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            return
        result = self.loaded.mcd_result
        if result.pair_b.size == 0:
            return
        distance = np.abs(result.pair_b - float(requested_b))
        pair_index = int(np.nanargmin(distance))
        # If two sweep branches have the same nearest B, retaining the active
        # branch makes repeated clicking at that field deterministic.
        nearest = float(distance[pair_index])
        candidates = np.flatnonzero(np.isclose(distance, nearest, atol=1e-10, rtol=0.0))
        current_index = self.mcd_pair_b_combo.currentData()
        if current_index is not None and int(current_index) in candidates:
            pair_index = int(current_index)
        selected_b = float(result.pair_b[pair_index])
        if current_index is not None and int(current_index) == pair_index:
            if clicked:
                self._status(
                    f"The nearest paired measurement, {selected_b:.4g} T, is already displayed."
                )
            return
        combo_blocked = self.mcd_pair_b_combo.blockSignals(True)
        self.mcd_pair_b_combo.setCurrentIndex(pair_index)
        self.mcd_pair_b_combo.blockSignals(combo_blocked)
        if clicked:
            self._status(f"Clicked {float(requested_b):.4g} T; showing nearest paired measurement {selected_b:.4g} T.")
        if not self._refresh_mcd_pair_panels():
            self._on_mcd_plot_changed()

    def _on_mcd_pair_selection_changed(self, _index: int) -> None:
        """Redraw all MCD pair displays after a manual pair selection."""
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            return
        if not self._refresh_mcd_pair_panels():
            self._on_mcd_plot_changed()

    def _auto_mcd_vrange(self) -> None:
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            return
        cube = self.loaded.mcd_result.cube(self.mcd_map_combo.currentText())
        finite = np.asarray(cube.Z, float)[np.isfinite(cube.Z)]
        if not finite.size:
            return
        bound = float(np.nanpercentile(np.abs(finite), 99.5))
        if self.mcd_center_zero_chk.isChecked():
            self.mcd_spins["vmin"].setValue(-bound); self.mcd_spins["vmax"].setValue(bound)
        else:
            lo, hi = np.nanpercentile(finite, [0.5, 99.5])
            self.mcd_spins["vmin"].setValue(float(lo)); self.mcd_spins["vmax"].setValue(float(hi))
        self._plot_mode("MCD")
