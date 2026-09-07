"""Controller for MCD processing and interactive display behavior."""

from __future__ import annotations

import matplotlib.patheffects as path_effects
import numpy as np
from datetime import datetime
from collections import OrderedDict
from pathlib import Path
from time import time as wall_time
from typing import Any, Sequence
from dataclasses import replace

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)
from core.drr_sources import resolve_source_path
from core.mcd import (
    McdBackgroundSuggestion, McdResult, McdSettings, background_fit_regions,
    detect_angles, discover_mcd_processing_status, low_field_mcd_branch_fits,
    mcd_annotation_layout, pair_window_trace_by_branch,
    suggest_mcd_background_ranges,
)
from ui_qt.common import QComboBox, Worker
from ui_qt.theme import alias as theme_alias
from ui_qt.matplotlib_theme import ThemeAwareFigureCanvasQTAgg
from ui_qt.source_picker_dialog import SourcePickerDialog

from core.mcd import format_mcd_energy, suggest_mcd_window_centers


def _detect_mcd_angles_worker(path: str, signature: tuple[int, int], *, progress, log) -> tuple[str, tuple[int, int], tuple[float, ...]]:
    """Read MCD angle values away from the GUI thread."""
    return path, signature, tuple(float(value) for value in detect_angles(path))

class McdController:
    """Own MCD reactions while sharing the application state through owner."""

    _LOCAL_LIFECYCLE_STATE = frozenset({
        "_mcd_reapply_pending",
        "_mcd_auto_apply_timer",
        "_mcd_center_refresh_timer",
        "_mcd_source_stability_timer",
        "_mcd_source_stability_source",
        "_mcd_source_stability_folder",
        "_mcd_source_stability_generation",
        "_mcd_background_suggestion",
        "_mcd_result_cache",
        "_mcd_result_cache_limit",
        "_mcd_source_observations",
        "_mcd_angle_cache",
        "_mcd_angle_generation",
        "_mcd_angle_workers",
    })

    def __init__(self, owner) -> None:
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_mcd_reapply_pending", False)
        auto_apply_timer = QTimer(owner)
        auto_apply_timer.setSingleShot(True)
        auto_apply_timer.setInterval(650)
        auto_apply_timer.timeout.connect(self._apply_pending_mcd_settings)
        object.__setattr__(self, "_mcd_auto_apply_timer", auto_apply_timer)
        center_refresh_timer = QTimer(owner)
        center_refresh_timer.setSingleShot(True)
        center_refresh_timer.setInterval(40)
        center_refresh_timer.timeout.connect(self._apply_pending_mcd_center_refresh)
        object.__setattr__(self, "_mcd_center_refresh_timer", center_refresh_timer)
        source_stability_timer = QTimer(owner)
        source_stability_timer.setSingleShot(True)
        source_stability_timer.setInterval(350)
        source_stability_timer.timeout.connect(self._on_mcd_source_stability_timeout)
        object.__setattr__(self, "_mcd_source_stability_timer", source_stability_timer)
        object.__setattr__(self, "_mcd_source_stability_source", "")
        object.__setattr__(self, "_mcd_source_stability_folder", "")
        object.__setattr__(self, "_mcd_source_stability_generation", 0)
        object.__setattr__(self, "_mcd_background_suggestion", None)
        object.__setattr__(self, "_mcd_result_cache", OrderedDict())
        object.__setattr__(self, "_mcd_result_cache_limit", 3)
        object.__setattr__(self, "_mcd_source_observations", {})
        object.__setattr__(self, "_mcd_angle_cache", {})
        object.__setattr__(self, "_mcd_angle_generation", 0)
        object.__setattr__(self, "_mcd_angle_workers", [])

    def __getattr__(self, name):
        owner = object.__getattribute__(self, "_owner")
        return object.__getattribute__(owner, name)

    def __setattr__(self, name, value) -> None:
        if name == "_owner" or name in McdController._LOCAL_LIFECYCLE_STATE:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_owner"), name, value)

    def _shutdown_mcd_lifecycle(self) -> None:
        """Stop MCD callbacks and invalidate queued results before window close."""
        self._mcd_auto_apply_timer.stop()
        self._mcd_center_refresh_timer.stop()
        self._mcd_source_stability_timer.stop()
        self._mcd_angle_generation += 1
        self._mcd_source_stability_generation += 1
        self._mcd_source_stability_source = ""
        self._mcd_source_stability_folder = ""

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

    def _on_mcd_plot_changed(self, _signal_value=None, *, source=None) -> None:
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

    def _update_mcd_selection_summary(self) -> None:
        if not hasattr(self, "mcd_selection_summary"):
            return
        selected = self._selected(self.mcd_files)
        if not selected:
            self.mcd_selection_summary.set_status(
                "No MCD CSV selected.", tooltip="", app_role=None, badge_state=None
            )
            return
        source = selected[0]
        display_name = Path(source).name.replace("_", "_\u200b").replace("-", "-\u200b")
        processed_at = self.mcd_processed_status.get(source, "")
        if processed_at:
            state = f"✓ PROCESSED\nLast saved: {processed_at[:16].replace('T', ' ')}"
            badge_state = "processed"
        else:
            state = "● NEW — No saved analysis"
            badge_state = "new"
        self.mcd_selection_summary.set_status(
            f"{state}\nSelected: {display_name}",
            tooltip=source,
            app_role="sourceBadge",
            badge_state=badge_state,
        )

    def _mcd_source_modified(self, source: str) -> float:
        try:
            return resolve_source_path(self.current_folder, source).stat().st_mtime
        except OSError:
            return 0.0

    def _mcd_sources_newest_first(self) -> list[str]:
        return sorted(
            self.mcd_available_files,
            key=lambda source: (-self._mcd_source_modified(source), source.casefold()),
        )

    def _mcd_saved_source_filter(self) -> str:
        value = str(getattr(self, "_mcd_source_filter_preference", "all")).casefold()
        return value if value in {"all", "unprocessed", "processed"} else "all"

    def _mcd_source_filter_counts(self) -> dict[str, int]:
        processed = sum(
            1 for source in self.mcd_available_files if source in self.mcd_processed_status
        )
        return {
            "all": len(self.mcd_available_files),
            "unprocessed": len(self.mcd_available_files) - processed,
            "processed": processed,
        }

    def _open_mcd_source_dialog(self, selected: str) -> str | None:
        """Choose exactly one raw MCD CSV with DRR-style processing filters."""
        state_filter = QComboBox()
        self._style_combo_popup(state_filter)
        dlg = SourcePickerDialog(
            self._owner,
            title="Choose MCD CSV",
            hint=(
                "Choose one raw B-sweep CSV from the experiment's mcd folder. "
                "Processed means a saved MCD analysis already exists; selecting it again allows reprocessing."
            ),
            selected=selected,
            filter_controls=(("Status", state_filter),),
            # Preserve MCD's synchronous pre-extraction filtering behavior.
            filter_interval=0,
        )
        file_list = dlg.source_list
        details = dlg.details_label
        ok_button = dlg.ok_button
        refresh_btn = dlg.refresh_button

        def _populate_filter_counts() -> None:
            current = str(state_filter.currentData() or self._mcd_saved_source_filter())
            counts = self._mcd_source_filter_counts()
            blocked = state_filter.blockSignals(True)
            state_filter.clear()
            state_filter.addItem(f"All ({counts['all']})", "all")
            state_filter.addItem(f"Unprocessed ({counts['unprocessed']})", "unprocessed")
            state_filter.addItem(f"Processed ({counts['processed']})", "processed")
            index = state_filter.findData(current)
            state_filter.setCurrentIndex(index if index >= 0 else 0)
            state_filter.blockSignals(blocked)

        _populate_filter_counts()

        def _refresh_view() -> None:
            needle = filter_edit.text().strip().casefold()
            wanted = str(state_filter.currentData() or "all")
            candidates = self._mcd_sources_newest_first()
            def _populate(widget) -> None:
                for source in candidates:
                    processed_at = self.mcd_processed_status.get(source, "")
                    if wanted == "unprocessed" and processed_at:
                        continue
                    if wanted == "processed" and not processed_at:
                        continue
                    if needle and needle not in source.casefold():
                        continue
                    modified = self._mcd_source_modified(source)
                    modified_text = (
                        datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
                        if modified
                        else "date unavailable"
                    )
                    if processed_at:
                        text = (
                            f"✓ PROCESSED — {Path(source).name}\n"
                            f"Modified {modified_text} · Saved {processed_at[:16].replace('T', ' ')}"
                        )
                        color = QColor(theme_alias("source_processed_foreground"))
                    else:
                        text = (
                            f"● NEW — {Path(source).name}\n"
                            f"Modified {modified_text} · No saved analysis"
                        )
                        color = QColor(theme_alias("source_new_foreground"))
                    item = QListWidgetItem(text)
                    item.setData(Qt.UserRole, source)
                    item.setToolTip(source)
                    item.setForeground(color)
                    font = item.font()
                    font.setBold(not bool(processed_at))
                    item.setFont(font)
                    widget.addItem(item)
            dlg.repopulate(_populate, fallback_selection=selected)

        def _update_details() -> None:
            item = file_list.currentItem()
            ok_button.setEnabled(item is not None)
            if item is None:
                details.setText("No matching MCD CSV files.")
                return
            source = str(item.data(Qt.UserRole))
            processed_at = self.mcd_processed_status.get(source, "")
            state = (
                f"✓ PROCESSED — Last saved {processed_at[:16].replace('T', ' ')}"
                if processed_at
                else "● NEW — No saved MCD analysis was found."
            )
            details.setText(f"{source}\n{state}")

        def _reload_catalog() -> None:
            self._refresh_file_lists(auto=True)
            _populate_filter_counts()
            _refresh_view()
            _update_details()

        def _on_filter_changed() -> None:
            value = str(state_filter.currentData() or "all")
            self._mcd_source_filter_preference = value
            self.settings.setValue(self.SETTINGS_MCD_SOURCE_FILTER, value)
            _refresh_view()

        filter_edit = dlg.filter_edit
        dlg.filter_requested.connect(_refresh_view)
        state_filter.currentIndexChanged.connect(lambda _index: _on_filter_changed())
        file_list.currentItemChanged.connect(lambda _current, _previous: _update_details())
        refresh_btn.clicked.connect(_reload_catalog)
        _refresh_view()
        _update_details()

        if dlg.exec() != SourcePickerDialog.Accepted:
            return None
        return dlg.selected_source()

    def _edit_mcd_source(self) -> None:
        selected = self._selected(self.mcd_files)
        previous = selected[0] if selected else ""
        chosen = self._open_mcd_source_dialog(previous)
        if not chosen:
            return
        if chosen != previous:
            blocked = self.mcd_files.blockSignals(True)
            self._restore_list_selection(self.mcd_files, [chosen])
            self.mcd_files.blockSignals(blocked)
            self._update_mcd_selection_summary()
            self._mcd_detect_available_angles()
        self._mcd_auto_apply_timer.stop()
        self._mcd_reapply_pending = False
        self._status(f"Selected MCD source: {Path(chosen).name}. Loading now...")
        self._start_load("MCD")

    def _clear_mcd_source(self) -> None:
        self.mcd_files.clearSelection()
        self._invalidate_export_move_sources()
        if self.loaded and self.loaded.mode == "MCD":
            self.loaded = None
        if self.last_plotted_mode == "MCD":
            self.last_plotted_mode = None
            self._disable_mcd_blitting()
            self.figure.clear()
            self.canvas.draw_idle()
        self._set_stage("No MCD source")
        self._update_action_states()

    def _mcd_refresh_sources(self) -> None:
        if not hasattr(self, "mcd_files"):
            return
        pending_status = getattr(self, "_mcd_status_from_refresh", None)
        if pending_status is not None:
            self.mcd_processed_status = dict(pending_status)
            self._mcd_status_from_refresh = None
        else:
            self.mcd_processed_status = discover_mcd_processing_status(
                self.current_folder, self.mcd_available_files
            )
        selected = self._selected(self.mcd_files)
        dark_pos = str(self.mcd_dark_pos_combo.currentData() or "")
        dark_neg = str(self.mcd_dark_neg_combo.currentData() or "")
        widgets = (self.mcd_files, self.mcd_dark_pos_combo, self.mcd_dark_neg_combo)
        blocked = [widget.blockSignals(True) for widget in widgets]
        try:
            self.mcd_files.clear()
            for file_name in self.mcd_available_files:
                item = QListWidgetItem(file_name)
                item.setToolTip(file_name)
                self.mcd_files.addItem(item)
            retained = [
                file_name for file_name in selected
                if file_name in self.mcd_available_files
            ]
            self._restore_list_selection(self.mcd_files, retained)
            if (
                not retained
                and len(self.mcd_available_files) == 1
                and Path(self.mcd_available_files[0]).parts
                and Path(self.mcd_available_files[0]).parts[0].casefold() == "mcd"
            ):
                self.mcd_files.item(0).setSelected(True)
            for combo, old in ((self.mcd_dark_pos_combo, dark_pos), (self.mcd_dark_neg_combo, dark_neg)):
                combo.clear(); combo.addItem("-- No dark / offset file --", "")
                for file_name in self.mcd_available_files:
                    combo.addItem(file_name, file_name)
                combo.setCurrentIndex(max(0, combo.findData(old)))
        finally:
            for widget, state in zip(widgets, blocked):
                widget.blockSignals(state)
        self._update_mcd_selection_summary()
        self._mcd_detect_available_angles()

    def _mcd_detect_available_angles(self) -> None:
        self._mcd_angle_generation += 1
        generation = self._mcd_angle_generation
        if not self.current_folder:
            return
        selected = self._selected(self.mcd_files)
        source = selected[0] if selected else ""
        if not source:
            self.mcd_source_summary.setText("Select a B-sweep CSV.")
            self.mcd_source_summary.setToolTip("")
            for combo in (self.mcd_sigma_plus_combo, self.mcd_sigma_minus_combo):
                blocked = combo.blockSignals(True)
                combo.clear()
                combo.addItem("-- Select source CSV first --", None)
                combo.blockSignals(blocked)
            return
        try:
            source_path = resolve_source_path(self.current_folder, source)
            stat = source_path.stat()
            cache_key = str(source_path.resolve()).casefold()
            signature = (int(stat.st_size), int(stat.st_mtime_ns))
            cached = self._mcd_angle_cache.get(cache_key)
            if cached is not None and cached[:2] == signature:
                angles = cached[2]
                self._apply_mcd_detected_angles(angles)
                return
        except Exception as exc:
            message = f"Could not read MCD angles: {str(exc).splitlines()[0]}"
            self.mcd_source_summary.setText(message)
            self.mcd_source_summary.setToolTip(message)
            return
        self.mcd_source_summary.setText("Detecting MCD angles…")
        self.mcd_source_summary.setToolTip(str(source_path))
        worker = Worker(_detect_mcd_angles_worker, str(source_path), signature)
        self._mcd_angle_workers.append(worker)
        worker.signals.result.connect(
            lambda result, generation=generation, folder=self.current_folder, source_key=cache_key:
            self._on_mcd_angles_result(result, generation, folder, source_key)
        )
        worker.signals.error.connect(
            lambda message, generation=generation: self._on_mcd_angles_error(message, generation)
        )
        worker.signals.finished.connect(lambda worker=worker: self._finish_mcd_angle_worker(worker))
        self.thread_pool.start(worker)
        return

    def _on_mcd_angles_result(self, result, generation: int, folder: str, source_key: str) -> None:
        if generation != self._mcd_angle_generation or str(folder).casefold() != str(self.current_folder).casefold():
            return
        path, signature, angles = result
        current = self._selected(self.mcd_files)
        try:
            if str(resolve_source_path(self.current_folder, current[0]).resolve()).casefold() != source_key:
                return
        except (IndexError, OSError):
            return
        self._mcd_angle_cache[source_key] = (int(signature[0]), int(signature[1]), tuple(angles))
        self._apply_mcd_detected_angles(tuple(angles))

    def _on_mcd_angles_error(self, message: str, generation: int) -> None:
        if generation == self._mcd_angle_generation:
            first = str(message).splitlines()[0]
            self.mcd_source_summary.setText(f"Could not read MCD angles: {first}")

    def _finish_mcd_angle_worker(self, worker: Worker) -> None:
        try:
            self._mcd_angle_workers.remove(worker)
        except ValueError:
            pass

    def _apply_mcd_detected_angles(self, angles: tuple[float, ...]) -> None:
        if len(angles) < 2:
            message = "MCD CSV needs at least two distinct angle values."
            self.mcd_source_summary.setText(message)
            self.mcd_source_summary.setToolTip(message)
            return
        plus_old = self.mcd_sigma_plus_combo.currentData()
        minus_old = self.mcd_sigma_minus_combo.currentData()
        combos = (self.mcd_sigma_plus_combo, self.mcd_sigma_minus_combo)
        blocked = [combo.blockSignals(True) for combo in combos]
        try:
            for combo in combos:
                combo.clear()
                for angle in angles:
                    combo.addItem(f"{angle:g} deg", float(angle))
            if self.mcd_auto_angles_chk.isChecked():
                self.mcd_sigma_plus_combo.setCurrentIndex(len(angles) - 1)
                self.mcd_sigma_minus_combo.setCurrentIndex(0)
            else:
                plus_index = self.mcd_sigma_plus_combo.findData(plus_old)
                minus_index = self.mcd_sigma_minus_combo.findData(minus_old)
                self.mcd_sigma_plus_combo.setCurrentIndex(plus_index if plus_index >= 0 else len(angles) - 1)
                self.mcd_sigma_minus_combo.setCurrentIndex(minus_index if minus_index >= 0 else 0)
        finally:
            for combo, state in zip(combos, blocked):
                combo.blockSignals(state)
        message = "Detected angles: " + ", ".join(f"{angle:g} deg" for angle in angles)
        self.mcd_source_summary.setText(message)
        self.mcd_source_summary.setToolTip(message)

    def _mcd_settings_from_ui(self) -> McdSettings:
        gain = {
            "Per wavelength": "per_wavelength",
            "Smoothed per wavelength": "smoothed",
            "Scalar (diagnostic only)": "scalar",
        }.get(self.mcd_gain_combo.currentText(), "per_wavelength")
        sigma_plus = self.mcd_sigma_plus_combo.currentData()
        sigma_minus = self.mcd_sigma_minus_combo.currentData()
        if sigma_plus is None or sigma_minus is None:
            raise ValueError("Select an MCD source CSV so its sigma+ and sigma- angles can be detected.")
        background_ranges: list[tuple[float, float]] = []
        range_text = self.mcd_background_ranges_edit.text().replace(";", ",").replace("–", "-").strip()
        for item in (part.strip() for part in range_text.split(",") if part.strip()):
            values = item.split("-")
            if len(values) != 2:
                raise ValueError("Fit background energies must be comma-separated ranges such as 1.50-1.58, 1.73-1.79.")
            try:
                start, stop = (float(value.strip()) for value in values)
            except ValueError as exc:
                raise ValueError("Fit background energies must be numeric eV ranges.") from exc
            if start >= stop:
                raise ValueError("Each fit background energy range must increase from left to right.")
            background_ranges.append((start, stop))
        correction_mode = {
            "Global reference gain (current)": "global",
            "Global gain + per-pair scale": "pair_scale",
            "Global gain + per-pair scale/offset": "pair_affine",
            "Global gain + per-pair spectral baseline": "pair_spectral",
        }.get(self.mcd_correction_mode_combo.currentText(), "global")
        suggestion = self._mcd_background_suggestion
        suggestion_active = bool(
            suggestion is not None
            and range_text == self._format_mcd_background_ranges(suggestion.ranges)
        )
        selection_mode = "suggested" if suggestion_active else ("manual" if background_ranges else "auto")
        return McdSettings(
            pos_angle=float(sigma_plus),
            neg_angle=float(sigma_minus),
            max_sequence_gap=int(self.mcd_gap_spin.value()), max_delta_b=float(self.mcd_delta_b_spin.value()),
            pair_b_alignment=("interpolate" if self.mcd_pair_alignment_combo.currentIndex() == 1 else "direct"),
            zero_window_t=float(self.mcd_zero_spin.value()),
            reference_mode=("nearest" if self.mcd_reference_mode_combo.currentIndex() == 0 else "window"),
            bin_decimals=int(self.mcd_bin_spin.value()),
            gain_mode=gain, correction_mode=correction_mode,
            spectral_order=(2 if self.mcd_spectral_order_combo.currentIndex() == 1 else 1),
            background_ranges_ev=tuple(background_ranges),
            background_selection=selection_mode,
            suggestion_protected_ranges_ev=(suggestion.protected_ranges if suggestion_active else ()),
            manual_protected_ranges_ev=(suggestion.manual_protected_ranges if suggestion_active else ()),
            suggestion_linear_validation_rms=(suggestion.linear_validation_rms if suggestion_active else None),
            suggestion_quadratic_validation_rms=(suggestion.quadratic_validation_rms if suggestion_active else None),
            suggestion_algorithm=("manual_unprotected_bands_v3" if suggestion_active else None),
            dark_pos_file=str(self.mcd_dark_pos_combo.currentData() or "") or None,
            dark_neg_file=str(self.mcd_dark_neg_combo.currentData() or "") or None,
        )

    @staticmethod

    def _format_mcd_background_ranges(ranges: Sequence[tuple[float, float]]) -> str:
        return ", ".join(f"{float(start):.4f}-{float(stop):.4f}" for start, stop in ranges)

    def _update_mcd_background_preview(self) -> None:
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            self.mcd_background_preview.setText("Auto outer 15% ranges are shown after loading an MCD sweep.")
            return
        text = self.mcd_background_ranges_edit.text().strip()
        if text:
            self.mcd_background_preview.setText("Manual ranges are active. Blue shading shows the ranges used.")
            return
        ranges = background_fit_regions(self.loaded.mcd_result.energy_ev, ())
        self.mcd_background_preview.setText(f"Using: {self._format_mcd_background_ranges(ranges)} eV")

    def _suggest_mcd_background_ranges(self) -> None:
        if not self.loaded or self.loaded.mode != "MCD" or self.loaded.mcd_result is None:
            self._show_error("Load an MCD sweep before requesting a full-sweep background suggestion.")
            return
        result = self.loaded.mcd_result
        energy = np.asarray(result.energy_ev, float)
        center = float(self.mcd_window_center_spin.value())
        if not np.isfinite(center) or center <= float(np.nanmin(energy)) or center >= float(np.nanmax(energy)):
            center = float(np.nanmedian(energy))
        # Protect the active MCD(B) window plus a guard.  Feature protection
        # is deliberately manual: the review plot opens ready for the user to
        # drag exactly across each resonance they want excluded.
        guard_ev = max(0.010, float(self.mcd_window_width_spin.value()) * 1e-3)
        protected = ((center - guard_ev, center + guard_ev),)
        prior_manual: tuple[tuple[float, float], ...] = ()
        if self._mcd_background_suggestion is not None:
            prior_manual = tuple(self._mcd_background_suggestion.manual_protected_ranges)
        elif self.loaded.mcd_settings is not None:
            prior_manual = tuple(self.loaded.mcd_settings.manual_protected_ranges_ev)
        try:
            suggestion = suggest_mcd_background_ranges(
                result,
                protected_ranges_ev=protected + prior_manual,
                auto_detect_features=False,
                use_all_unprotected_bands=True,
            )
            suggestion = replace(
                suggestion,
                requested_protected_ranges=protected,
                manual_protected_ranges=prior_manual,
            )
        except Exception as exc:
            self._show_error(f"Could not suggest MCD background ranges: {exc}")
            return
        self._show_mcd_background_suggestion_dialog(result, suggestion)

    def _show_mcd_background_suggestion_dialog_legacy(self, suggestion: McdBackgroundSuggestion) -> None:
        dlg = QDialog(self._owner)
        dlg.setWindowTitle("Review MCD background suggestion")
        if not self.windowIcon().isNull():
            dlg.setWindowIcon(self.windowIcon())
        dlg.setMinimumSize(760, 560)
        dlg.resize(930, 680)
        layout = QVBoxLayout(dlg)
        figure = Figure(figsize=(8.2, 4.8), dpi=100)
        canvas = ThemeAwareFigureCanvasQTAgg(figure)
        spectrum_ax, score_ax = figure.subplots(2, 1, sharex=True, height_ratios=[2.2, 1.0])
        energy = np.asarray(suggestion.energy_ev, float)
        spectrum_ax.plot(energy, suggestion.median_reflectance, color="#303030", lw=1.1, label="full-sweep median reflectance")
        for index, (start, stop) in enumerate(suggestion.ranges):
            spectrum_ax.axvspan(start, stop, color="#5790b7", alpha=0.24, label="suggested fit band" if index == 0 else "_nolegend_")
        for index, (start, stop) in enumerate(suggestion.detected_feature_ranges):
            spectrum_ax.axvspan(start, stop, color="#e28743", alpha=0.20, label="auto-detected feature" if index == 0 else "_nolegend_")
        for index, (start, stop) in enumerate(suggestion.requested_protected_ranges):
            spectrum_ax.axvspan(start, stop, color="#c94c00", alpha=0.22, label="active protected window" if index == 0 else "_nolegend_")
        spectrum_ax.set_ylabel("Reflection (a.u.)")
        spectrum_ax.grid(alpha=0.22)
        spectrum_ax.legend(fontsize=8, loc="best")
        score_ax.plot(energy, suggestion.suitability, color="#6a3d9a", lw=0.9)
        for start, stop in suggestion.ranges:
            score_ax.axvspan(start, stop, color="#5790b7", alpha=0.24)
        for start, stop in suggestion.detected_feature_ranges:
            score_ax.axvspan(start, stop, color="#e28743", alpha=0.20)
        for start, stop in suggestion.requested_protected_ranges:
            score_ax.axvspan(start, stop, color="#c94c00", alpha=0.22)
        score_ax.set_xlabel("Energy (eV)")
        score_ax.set_ylabel("feature / noise score")
        score_ax.grid(alpha=0.22)
        figure.tight_layout(pad=1.0)
        layout.addWidget(canvas, 1)
        linear = suggestion.linear_validation_rms
        quadratic = suggestion.quadratic_validation_rms
        summary = QPlainTextEdit()
        summary.setReadOnly(True)
        summary.setMaximumHeight(132)
        summary.setPlainText(
            "Suggested manual background ranges (eV):\n"
            f"{self._format_mcd_background_ranges(suggestion.ranges)}\n\n"
            f"Active protected window: {self._format_mcd_background_ranges(suggestion.requested_protected_ranges)} eV\n"
            f"Auto-detected feature windows: {self._format_mcd_background_ranges(suggestion.detected_feature_ranges) or 'none'} eV\n"
            f"All excluded windows: {self._format_mcd_background_ranges(suggestion.protected_ranges)} eV\n"
            f"Coverage: {100.0 * suggestion.coverage_fraction:.1f}% of valid spectral points; "
            f"span: {100.0 * suggestion.span_fraction:.1f}% of the energy axis\n"
            f"Held-out RMS — linear: {linear:.4g}; quadratic: {quadratic:.4g}\n"
            f"Held-out diagnostic preference: {'Quadratic' if suggestion.suggested_order == 2 else 'Linear'} (does not change your selected order)\n\n"
            + "\n".join(f"- {note}" for note in suggestion.notes)
        )
        layout.addWidget(summary)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        accept = buttons.addButton("Use suggestion", QDialogButtonBox.AcceptRole)
        accept.setToolTip("Copy the suggested ranges and model into the MCD controls. It will not reprocess until Apply is clicked.")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec() != QDialog.Accepted:
            return
        self._mcd_background_suggestion = suggestion
        self.mcd_background_ranges_edit.setText(self._format_mcd_background_ranges(suggestion.ranges))
        self.mcd_correction_mode_combo.setCurrentText("Global gain + per-pair spectral baseline")
        self._update_mcd_background_preview()
        self._on_mcd_params_changed()
        self._mcd_auto_apply_timer.start(0)
        self._status("Selected protection regions saved; MCD recalculation is starting automatically.")

    def _show_mcd_background_suggestion_dialog(self, result: McdResult, suggestion: McdBackgroundSuggestion) -> None:
        """Let the user draw exact feature-protection windows before fitting."""
        dlg = QDialog(self._owner)
        dlg.setWindowTitle("Select MCD feature-protection regions")
        if not self.windowIcon().isNull():
            dlg.setWindowIcon(self.windowIcon())
        dlg.setMinimumSize(800, 660)
        dlg.resize(980, 820)
        layout = QVBoxLayout(dlg)
        figure = Figure(figsize=(8.2, 4.7), dpi=100)
        canvas = ThemeAwareFigureCanvasQTAgg(figure)
        spectrum_ax, score_ax = figure.subplots(2, 1, sharex=True, height_ratios=[2.2, 1.0])
        layout.addWidget(canvas, 1)
        energy = np.asarray(suggestion.energy_ev, float)
        energy_min, energy_max = float(np.nanmin(energy)), float(np.nanmax(energy))

        feature_box = QGroupBox("Feature-protection windows")
        feature_layout = QVBoxLayout(feature_box)
        note = QLabel(
            "Drag horizontally on the reflection plot to add a protection window. "
            "Only the active MCD(B) window and the regions you select are excluded from the blue background-fit bands."
        )
        note.setWordWrap(True)
        feature_layout.addWidget(note)
        table = QTableWidget(0, 8)
        table.setHorizontalHeaderLabels(["Use", "Type", "Center (eV)", "Start (eV)", "Stop (eV)", "Width (meV)", "SNR", "Status"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setMaximumHeight(180)
        feature_layout.addWidget(table)
        controls = QHBoxLayout()
        remove = QPushButton("Remove selected")
        select_range = QPushButton("Draw protection region")
        select_range.setToolTip("Drag horizontally on the upper reflection plot to protect exactly that energy region.")
        controls.addWidget(remove)
        controls.addStretch(1)
        controls.addWidget(select_range)
        feature_layout.addLayout(controls)
        layout.addWidget(feature_box)

        summary = QPlainTextEdit()
        summary.setReadOnly(True)
        summary.setMaximumHeight(125)
        layout.addWidget(summary)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        accept = buttons.addButton("Use selected regions", QDialogButtonBox.AcceptRole)
        accept.setToolTip("Copy fit ranges recalculated from your selected protection regions. Processing starts only after Apply.")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        row_sources: list[str] = []
        row_features: list[object | None] = []
        current: dict[str, McdBackgroundSuggestion] = {"value": suggestion}
        is_refreshing = {"value": False}
        selection_state: dict[str, object] = {"active": True, "selector": None}

        def on_manual_span(left: float, right: float) -> None:
            if abs(float(right) - float(left)) < 5e-4:
                return
            selection_state["active"] = True
            add_row("Manual", min(left, right), max(left, right), source="manual")
            select_range.setText("Drawing: drag another region")
            refresh()

        def configure_span_selector() -> None:
            old_selector = selection_state.get("selector")
            if old_selector is not None and hasattr(old_selector, "disconnect_events"):
                old_selector.disconnect_events()
            selector = SpanSelector(
                spectrum_ax, on_manual_span, "horizontal", useblit=False,
                props={"facecolor": "#8e44ad", "alpha": 0.25}, interactive=False,
            )
            selector.set_active(bool(selection_state["active"]))
            selection_state["selector"] = selector

        def selected_windows() -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...], tuple[tuple[float, float], ...], tuple[str, ...]]:
            active: list[tuple[float, float]] = []
            automatic: list[tuple[float, float]] = []
            manual: list[tuple[float, float]] = []
            automatic_kinds: list[str] = []
            for row in range(table.rowCount()):
                checkbox = table.cellWidget(row, 0)
                left = table.cellWidget(row, 3)
                right = table.cellWidget(row, 4)
                if not isinstance(checkbox, QCheckBox) or not checkbox.isChecked():
                    continue
                if not isinstance(left, QDoubleSpinBox) or not isinstance(right, QDoubleSpinBox):
                    continue
                window = tuple(sorted((float(left.value()), float(right.value()))))
                source = row_sources[row]
                if source == "active":
                    active.append(window)
                elif source == "auto":
                    automatic.append(window)
                    label = table.item(row, 1).text() if table.item(row, 1) else "feature"
                    automatic_kinds.append(label.removeprefix("Auto "))
                else:
                    manual.append(window)
            return tuple(active), tuple(automatic), tuple(manual), tuple(automatic_kinds)

        def merged_enabled_windows() -> tuple[tuple[float, float], ...]:
            active, automatic, manual, _kinds = selected_windows()
            windows = sorted(tuple(active) + tuple(automatic) + tuple(manual))
            merged: list[tuple[float, float]] = []
            for left, right in windows:
                if not merged or left > merged[-1][1] + 1e-12:
                    merged.append((left, right))
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            return tuple(merged)

        def unchecked_auto_windows() -> tuple[tuple[float, float], ...]:
            windows: list[tuple[float, float]] = []
            for row, source in enumerate(row_sources):
                if source != "auto":
                    continue
                checkbox = table.cellWidget(row, 0)
                left = table.cellWidget(row, 3)
                right = table.cellWidget(row, 4)
                if isinstance(checkbox, QCheckBox) and not checkbox.isChecked() and isinstance(left, QDoubleSpinBox) and isinstance(right, QDoubleSpinBox):
                    windows.append(tuple(sorted((float(left.value()), float(right.value())))))
            return tuple(windows)

        def redraw() -> None:
            displayed = current["value"]
            active, automatic, manual, _kinds = selected_windows()
            unchecked_auto = unchecked_auto_windows()
            spectrum_ax.clear()
            score_ax.clear()
            spectrum_ax.plot(energy, displayed.median_reflectance, color="#303030", lw=1.1, label="full-sweep median reflection")
            if np.any(np.isfinite(displayed.feature_baseline)):
                spectrum_ax.plot(energy, np.exp(displayed.feature_baseline), color="#8c8c8c", lw=0.9, ls="--", label="broad reflection baseline")
            for index, (left, right) in enumerate(displayed.ranges):
                spectrum_ax.axvspan(left, right, color="#5790b7", alpha=0.24, label="suggested fit band" if index == 0 else "_nolegend_")
            for index, (left, right) in enumerate(active):
                spectrum_ax.axvspan(left, right, color="#c94c00", alpha=0.22, label="active MCD(B) window" if index == 0 else "_nolegend_")
            for index, (left, right) in enumerate(automatic):
                spectrum_ax.axvspan(left, right, color="#e28743", alpha=0.22, label="enabled auto feature" if index == 0 else "_nolegend_")
            for index, (left, right) in enumerate(manual):
                spectrum_ax.axvspan(left, right, color="#8e44ad", alpha=0.20, label="manual protection" if index == 0 else "_nolegend_")
            for index, (left, right) in enumerate(unchecked_auto):
                spectrum_ax.axvspan(left, right, facecolor="none", edgecolor="#e28743", lw=0.9, ls="--", alpha=0.8, label="review-only candidate" if index == 0 else "_nolegend_")
            for feature in suggestion.detected_features:
                marker = "^" if feature.kind == "peak" else "v"
                color = "#e28743" if feature.recommended else "#b27b4d"
                y = float(np.interp(feature.center_ev, energy, displayed.median_reflectance))
                spectrum_ax.plot(feature.center_ev, y, marker=marker, color=color, ms=5, zorder=4)
                if feature.recommended:
                    spectrum_ax.annotate(
                        f"{feature.center_ev:.4f} eV\nSNR {feature.snr:.1f}",
                        (feature.center_ev, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=color,
                    )
            spectrum_ax.set_ylabel("Reflection (a.u.)")
            spectrum_ax.grid(alpha=0.22)
            spectrum_ax.legend(fontsize=8, loc="best")
            score_ax.plot(energy, displayed.suitability, color="#6a3d9a", lw=0.9, label="background unsuitability")
            if np.any(np.isfinite(displayed.feature_detection_score)):
                score_ax.plot(energy, displayed.feature_detection_score, color="#e28743", lw=0.8, alpha=0.85, label="feature SNR score")
            for left, right in displayed.ranges:
                score_ax.axvspan(left, right, color="#5790b7", alpha=0.24)
            for left, right in active:
                score_ax.axvspan(left, right, color="#c94c00", alpha=0.18)
            for left, right in automatic:
                score_ax.axvspan(left, right, color="#e28743", alpha=0.18)
            for left, right in manual:
                score_ax.axvspan(left, right, color="#8e44ad", alpha=0.16)
            score_ax.set_xlabel("Energy (eV)")
            score_ax.set_ylabel("background / feature score")
            score_ax.grid(alpha=0.22)
            score_ax.legend(fontsize=8, loc="best")
            figure.tight_layout(pad=1.0)
            candidate_lines = [
                f"{feature.center_ev:.4f} eV  {feature.kind}  width {1000.0 * feature.width_ev:.1f} meV  "
                f"prominence {100.0 * np.expm1(feature.prominence_log):.1f}%  SNR {feature.snr:.1f}  "
                f"{'recommended' if feature.recommended else 'review only'}"
                for feature in suggestion.detected_features
            ]
            summary.setPlainText(
                "Suggested manual background ranges (eV):\n"
                f"{self._format_mcd_background_ranges(displayed.ranges)}\n\n"
                f"Enabled protection windows: {self._format_mcd_background_ranges(displayed.protected_ranges) or 'none'} eV\n"
                f"Coverage: {100.0 * displayed.coverage_fraction:.1f}% of valid spectral points; "
                f"span: {100.0 * displayed.span_fraction:.1f}% of the energy axis\n"
                f"Held-out RMS - linear: {displayed.linear_validation_rms:.4g}; quadratic: {displayed.quadratic_validation_rms:.4g}\n"
                f"Held-out diagnostic preference: {'Quadratic' if displayed.suggested_order == 2 else 'Linear'} (does not change your selected order)\n\n"
                + "\n".join(f"- {item}" for item in displayed.notes)
                + ("\n\nDetected reflection candidates:\n" + "\n".join(candidate_lines) if candidate_lines else "")
            )
            configure_span_selector()
            canvas.draw_idle()

        def refresh() -> None:
            if is_refreshing["value"]:
                return
            is_refreshing["value"] = True
            try:
                active, automatic, manual, automatic_kinds = selected_windows()
                selected = merged_enabled_windows()
                recalculated = suggest_mcd_background_ranges(
                    result,
                    protected_ranges_ev=selected,
                    auto_detect_features=False,
                    use_all_unprotected_bands=True,
                )
                current["value"] = replace(
                    recalculated,
                    requested_protected_ranges=active,
                    manual_protected_ranges=manual,
                    detected_feature_ranges=automatic,
                    detected_feature_kinds=automatic_kinds,
                    feature_baseline=suggestion.feature_baseline,
                    feature_residual=suggestion.feature_residual,
                    feature_detection_score=suggestion.feature_detection_score,
                    detected_features=suggestion.detected_features,
                )
                redraw()
            except Exception as exc:
                summary.setPlainText(f"No usable fit bands remain for these protection windows:\n{exc}")
            finally:
                is_refreshing["value"] = False

        def connect_row(row: int) -> None:
            checkbox = table.cellWidget(row, 0)
            left = table.cellWidget(row, 3)
            right = table.cellWidget(row, 4)
            if isinstance(checkbox, QCheckBox):
                checkbox.toggled.connect(refresh)
            if isinstance(left, QDoubleSpinBox):
                left.valueChanged.connect(refresh)
            if isinstance(right, QDoubleSpinBox):
                right.valueChanged.connect(refresh)
            def update_width() -> None:
                left_control = table.cellWidget(row, 3)
                right_control = table.cellWidget(row, 4)
                width_item = table.item(row, 5)
                if isinstance(left_control, QDoubleSpinBox) and isinstance(right_control, QDoubleSpinBox) and width_item is not None:
                    width_item.setText(f"{1000.0 * abs(right_control.value() - left_control.value()):.2f}")
            if isinstance(left, QDoubleSpinBox):
                left.valueChanged.connect(update_width)
            if isinstance(right, QDoubleSpinBox):
                right.valueChanged.connect(update_width)

        def add_row(kind: str, left: float, right: float, *, source: str, enabled: bool = True, feature: object | None = None) -> None:
            row = table.rowCount()
            table.insertRow(row)
            checkbox = QCheckBox()
            checkbox.setChecked(enabled)
            table.setCellWidget(row, 0, checkbox)
            label = QTableWidgetItem(kind)
            label.setFlags(label.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 1, label)
            center = float(getattr(feature, "center_ev", 0.5 * (left + right)))
            center_item = QTableWidgetItem(f"{center:.6f}")
            center_item.setFlags(center_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 2, center_item)
            for column, value in ((3, left), (4, right)):
                spin = QDoubleSpinBox()
                spin.setRange(energy_min, energy_max)
                spin.setDecimals(6)
                spin.setSingleStep(0.001)
                spin.setValue(float(value))
                table.setCellWidget(row, column, spin)
            width_item = QTableWidgetItem(f"{1000.0 * (right - left):.2f}")
            width_item.setFlags(width_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 5, width_item)
            snr = getattr(feature, "snr", None)
            snr_item = QTableWidgetItem(f"{float(snr):.1f}" if snr is not None else "-")
            snr_item.setFlags(snr_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 6, snr_item)
            status = "recommended" if bool(getattr(feature, "recommended", False)) else ("manual" if source == "manual" else "active" if source == "active" else "review only")
            status_item = QTableWidgetItem(status)
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 7, status_item)
            row_sources.append(source)
            row_features.append(feature)
            connect_row(row)

        for left, right in suggestion.requested_protected_ranges:
            add_row("Active MCD(B)", left, right, source="active")
        for left, right in suggestion.manual_protected_ranges:
            add_row("Manual", left, right, source="manual")
        def begin_manual_selection() -> None:
            selection_state["active"] = True
            selector = selection_state.get("selector")
            if selector is not None:
                selector.set_active(True)
            select_range.setText("Drag on reflection plot…")
            select_range.setToolTip("Drag across the reflection plot to protect exactly that energy region.")

        def remove_selected() -> None:
            row = table.currentRow()
            if row < 0:
                return
            table.removeRow(row)
            row_sources.pop(row)
            row_features.pop(row)
            refresh()

        select_range.clicked.connect(begin_manual_selection)
        remove.clicked.connect(remove_selected)
        redraw()
        if dlg.exec() != QDialog.Accepted:
            return
        accepted = current["value"]
        self._mcd_background_suggestion = accepted
        self.mcd_background_ranges_edit.setText(self._format_mcd_background_ranges(accepted.ranges))
        self.mcd_correction_mode_combo.setCurrentText("Global gain + per-pair spectral baseline")
        self._update_mcd_background_preview()
        self._on_mcd_params_changed()
        self._mcd_auto_apply_timer.start(0)
        self._status("Selected protection regions saved; MCD recalculation is starting automatically.")

    def _mcd_source_needs_stability_wait(self, source: str) -> bool:
        path = resolve_source_path(self.current_folder, source)
        try:
            stat = path.stat()
        except OSError:
            return False
        signature = (int(stat.st_size), int(stat.st_mtime_ns))
        key = str(path.resolve()).casefold()
        previous = self._mcd_source_observations.get(key)
        self._mcd_source_observations[key] = signature
        if wall_time() - float(stat.st_mtime) >= 1.25 or previous == signature:
            return False
        self._status("MCD source is still being written; waiting for it to become stable...")
        self._mcd_source_stability_generation += 1
        self._mcd_source_stability_source = source
        self._mcd_source_stability_folder = str(self.current_folder)
        self._mcd_source_stability_timer.start()
        return True

    def _on_mcd_source_stability_timeout(self, generation: int | None = None) -> None:
        """Retry a recently-written source only if its selection and folder remain current."""
        current_generation = self._mcd_source_stability_generation
        if generation is not None and generation != current_generation:
            return
        source = self._mcd_source_stability_source
        folder = self._mcd_source_stability_folder
        self._mcd_source_stability_source = ""
        self._mcd_source_stability_folder = ""
        if not source or getattr(self, "_is_closing", False) or self._load_in_progress:
            return
        if str(folder).casefold() != str(self.current_folder).casefold():
            return
        if self._selected(self.mcd_files) == [source]:
            self._start_load("MCD")

    @staticmethod
    def _mcd_result_cache_key(path: Path, settings: McdSettings) -> tuple[Any, ...]:
        stat = path.stat()
        reference_fingerprints: list[tuple[str, int, int]] = []
        for reference in (settings.dark_pos_file, settings.dark_neg_file):
            if not reference:
                continue
            reference_path = Path(reference)
            reference_stat = reference_path.stat()
            reference_fingerprints.append((
                str(reference_path.resolve()).casefold(),
                int(reference_stat.st_size),
                int(reference_stat.st_mtime_ns),
            ))
        return (
            str(path.resolve()).casefold(),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            settings,
            tuple(reference_fingerprints),
        )

    def _cached_mcd_result(
        self, path: Path, settings: McdSettings,
    ) -> tuple[McdResult | None, tuple[Any, ...]]:
        key = self._mcd_result_cache_key(path, settings)
        result = self._mcd_result_cache.get(key)
        if result is not None:
            self._mcd_result_cache.move_to_end(key)
        return result, key

    def _store_cached_mcd_result(self, key: tuple[Any, ...], result: McdResult) -> None:
        self._mcd_result_cache[key] = result
        self._mcd_result_cache.move_to_end(key)
        while len(self._mcd_result_cache) > self._mcd_result_cache_limit:
            self._mcd_result_cache.popitem(last=False)

    @staticmethod
    def _mcd_pair_correction_label(result: McdResult, pair_index: int) -> str:
        correction_mode = str(result.summary.get("correction_mode", "global"))
        residual = float(result.pair_background_rms[pair_index])
        residual_before = float(result.pair_background_rms_before[pair_index])
        residual_text = f"{residual:.3g}" if np.isfinite(residual) else "--"
        residual_before_text = f"{residual_before:.3g}" if np.isfinite(residual_before) else "--"
        if correction_mode == "pair_spectral":
            fit_name = (
                "quadratic" if int(result.summary.get("spectral_order", 1)) == 2
                else "linear"
            )
            label = (
                f"spectral {fit_name}: gain@Ec={result.pair_scale[pair_index]:.4g}; "
                f"slope={result.pair_spectral_slope[pair_index]:+.3g}/eV\n"
                f"RMS {residual_before_text}->{residual_text}; gain range "
                f"{result.pair_correction_min[pair_index]:.3g}-"
                f"{result.pair_correction_max[pair_index]:.3g}"
            )
        else:
            label = (
                f"correction: scale={result.pair_scale[pair_index]:.4g}; "
                f"offset={result.pair_offset[pair_index]:.3g}; RMS={residual_text}"
            )
        if correction_mode in {"pair_scale", "pair_affine", "pair_spectral"}:
            label += " (blue regions used)"
        return label

    def _disable_mcd_blitting(self) -> None:
        self._mcd_center_refresh_timer.stop()
        for axes in self._mcd_blit_axes.values():
            for axis in axes:
                try:
                    axis.set_animated(False)
                except Exception:
                    pass
        for artist in (
            item for group in self._mcd_overlay_artists.values() for item in group
        ):
            try:
                artist.set_animated(False)
            except Exception:
                pass
        self._mcd_blit_enabled = False
        self._mcd_blit_backgrounds = {}
        self._mcd_blit_bboxes = {}
        self._mcd_blit_axes = {}
        self._mcd_heat_dynamic_artists = []
        self._mcd_overlay_artists = {}

    def _on_theme_changed(self, _theme: Any = None) -> None:
        """Invalidate MCD blit backgrounds after display presentation changes."""
        was_enabled = bool(self._mcd_blit_enabled)
        self._disable_mcd_blitting()
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return
        try:
            canvas.draw()
        except Exception:
            return
        if was_enabled:
            self._configure_mcd_blitting()
            canvas.draw_idle()

    def _configure_mcd_blitting(self) -> None:
        if any(axis is None for axis in (
            self._mcd_pair_ax, self._mcd_spectrum_ax,
            self._mcd_trace_ax, self._mcd_integral_ax,
        )):
            self._disable_mcd_blitting()
            return
        self._mcd_blit_axes = {
            "pair": (self._mcd_pair_ax,),
            "linecut": (self._mcd_spectrum_ax,),
            "trace": (self._mcd_trace_ax, self._mcd_integral_ax),
        }
        for axes in self._mcd_blit_axes.values():
            for axis in axes:
                axis.set_animated(True)
        overlay_artists: dict[str, list[Any]] = {
            "pair_overlay": [], "linecut_overlay": [], "heat": [],
        }
        for artists in self._mcd_window_artists:
            owner_axis = getattr(artists.get("patch"), "axes", None)
            group = (
                "heat" if owner_axis is self._mcd_heatmap_ax
                else "pair_overlay" if owner_axis is self._mcd_pair_ax
                else "linecut_overlay" if owner_axis is self._mcd_spectrum_ax
                else ""
            )
            if group:
                overlay_artists[group].extend(
                    artist for artist in artists.values() if artist is not None
                )
        if self._mcd_pair_cursor is not None:
            overlay_artists["heat"].append(self._mcd_pair_cursor)
        overlay_artists["heat"].extend(self._mcd_candidate_artists.values())
        self._mcd_overlay_artists = overlay_artists
        self._mcd_heat_dynamic_artists = overlay_artists["heat"]
        for artist in (
            item for group in overlay_artists.values() for item in group
        ):
            artist.set_animated(True)
        self._mcd_blit_backgrounds = {}
        self._mcd_blit_bboxes = {}
        self._mcd_blit_enabled = True

    def _mcd_group_bbox(self, axes: tuple[Any, ...], renderer: Any) -> Bbox:
        bounds = [axis.get_tightbbox(renderer) for axis in axes]
        union = Bbox.union([bound for bound in bounds if bound is not None])
        pad = 8.0
        figure_bbox = self.figure.bbox
        return Bbox.from_extents(
            max(float(figure_bbox.x0), float(union.x0) - pad),
            max(float(figure_bbox.y0), float(union.y0) - pad),
            min(float(figure_bbox.x1), float(union.x1) + pad),
            min(float(figure_bbox.y1), float(union.y1) + pad),
        )

    def _draw_mcd_blit_group(self, name: str) -> None:
        if name in {"heat", "pair_overlay", "linecut_overlay"}:
            axis = (
                self._mcd_heatmap_ax if name == "heat"
                else self._mcd_pair_ax if name == "pair_overlay"
                else self._mcd_spectrum_ax
            )
            if axis is None:
                return
            for artist in self._mcd_overlay_artists.get(name, ()):
                axis.draw_artist(artist)
            return
        for axis in self._mcd_blit_axes.get(name, ()):
            self.figure.draw_artist(axis)
        if name in {"pair", "linecut"} and name in self._mcd_blit_bboxes:
            overlay_name = f"{name}_overlay"
            bbox = self._mcd_blit_bboxes[name]
            self._mcd_blit_bboxes[overlay_name] = bbox
            self._mcd_blit_backgrounds[overlay_name] = self.canvas.copy_from_bbox(bbox)
            self._draw_mcd_blit_group(overlay_name)

    def _on_canvas_draw(self, event: Any) -> None:
        if (
            not self._mcd_blit_enabled or self._mcd_blit_in_draw
            or self.loaded is None or self.loaded.mode != "MCD"
        ):
            return
        self._mcd_blit_in_draw = True
        try:
            groups = dict(self._mcd_blit_axes)
            if self._mcd_heatmap_ax is not None:
                groups["heat"] = (self._mcd_heatmap_ax,)
            for name, axes in groups.items():
                bbox = (
                    self._mcd_heatmap_ax.bbox
                    if name == "heat" else self._mcd_group_bbox(axes, event.renderer)
                )
                self._mcd_blit_bboxes[name] = bbox
                self._mcd_blit_backgrounds[name] = self.canvas.copy_from_bbox(bbox)
            for name in groups:
                self._draw_mcd_blit_group(name)
                self.canvas.blit(self._mcd_blit_bboxes[name])
        finally:
            self._mcd_blit_in_draw = False

    def _blit_mcd_regions(self, *names: str) -> bool:
        if not self._mcd_blit_enabled:
            return False
        selected = tuple(dict.fromkeys(names))
        if any(
            name not in self._mcd_blit_backgrounds or name not in self._mcd_blit_bboxes
            for name in selected
        ):
            return False
        for name in selected:
            self.canvas.restore_region(self._mcd_blit_backgrounds[name])
            self._draw_mcd_blit_group(name)
            self.canvas.blit(self._mcd_blit_bboxes[name])
        return True

    def _prepare_mcd_toolbar_save(self) -> None:
        if not self._mcd_blit_enabled:
            self._mcd_toolbar_save_animated_axes = ()
            return
        axes = tuple(
            axis for group in self._mcd_blit_axes.values() for axis in group
        )
        self._mcd_toolbar_save_animated_axes = axes
        self._mcd_blit_enabled = False
        for axis in axes:
            axis.set_animated(False)
        for artist in (
            item for group in self._mcd_overlay_artists.values() for item in group
        ):
            artist.set_animated(False)
        self.canvas.draw()

    def _restore_mcd_toolbar_save(self) -> None:
        if not self._mcd_toolbar_save_animated_axes:
            return
        for axis in self._mcd_toolbar_save_animated_axes:
            axis.set_animated(True)
        for artist in (
            item for group in self._mcd_overlay_artists.values() for item in group
        ):
            artist.set_animated(True)
        self._mcd_toolbar_save_animated_axes = ()
        self._mcd_blit_enabled = True
        self._mcd_blit_backgrounds = {}
        self._mcd_blit_bboxes = {}
        self.canvas.draw_idle()

    def _refresh_mcd_pair_panels(self) -> bool:
        """Update the selected pair without reconstructing unrelated MCD axes."""
        if (
            self.loaded is None or self.loaded.mode != "MCD"
            or self.loaded.mcd_result is None or self._mcd_pair_ax is None
            or self._mcd_spectrum_ax is None or self._mcd_pair_cursor is None
            or len(self._mcd_pair_spectrum_lines) != 4
            or len(self._mcd_linecut_lines) != 2
        ):
            return False
        result = self.loaded.mcd_result
        pair_index = int(self.mcd_pair_b_combo.currentData() or 0)
        pair_index = int(np.clip(pair_index, 0, result.pair_b.size - 1))
        pair_b = float(result.pair_b[pair_index])
        order = np.argsort(1239.841984 / result.wavelength_nm)
        pair_rows = (
            result.pair_raw_pos, result.pair_raw_neg,
            result.pair_corrected_pos, result.pair_corrected_neg,
        )
        for line, rows in zip(self._mcd_pair_spectrum_lines, pair_rows):
            line.set_ydata(np.asarray(rows[pair_index, order], float))
        for line, rows in zip(
            self._mcd_linecut_lines,
            (result.pair_mcd_raw, result.pair_mcd_corrected),
        ):
            line.set_ydata(np.asarray(rows[pair_index, order], float))
        self._mcd_pair_ax.set_title(f"Paired spectra: B = {pair_b:.5g} T")
        self._mcd_spectrum_ax.set_title(f"MCD linecut: B = {pair_b:.5g} T")
        self._mcd_pair_cursor.set_ydata([pair_b, pair_b])
        legend = self._mcd_pair_ax.get_legend()
        if legend is not None and legend.get_texts():
            legend.get_texts()[-1].set_text(
                self._mcd_pair_correction_label(result, pair_index)
            )
        delta_b = float(result.pair_delta_b[pair_index])
        alignment = "; aligned" if (
            result.pair_interpolated_pos[pair_index]
            or result.pair_interpolated_neg[pair_index]
        ) else ""
        if self._mcd_linecut_diagnostic_text is not None:
            self._mcd_linecut_diagnostic_text.set_text(
                f"dB = {delta_b:+.4g} T{alignment}"
            )
        for axis in (self._mcd_pair_ax, self._mcd_spectrum_ax):
            xlim = axis.get_xlim()
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)
            axis.set_xlim(xlim)
        if not self._blit_mcd_regions("pair", "linecut", "heat"):
            self.canvas.draw_idle()
        self._status(f"Showing MCD pair at {pair_b:.5g} T.")
        return True

    @staticmethod
    def _mcd_preview_slope_lines(
        branch_fits: dict[str, tuple[float, float]],
    ) -> list[str]:
        lines: list[str] = []
        for branch, label in (
            ("B increasing", "Increasing slope"),
            ("B decreasing", "Decreasing slope"),
        ):
            slope = branch_fits.get(branch, (float("nan"), 0.0))[0]
            if np.isfinite(slope):
                lines.append(f"{label}: {slope:.6g} T⁻¹")
        return lines

    @staticmethod
    def _mcd_preview_annotation_fontsize(axis: Any, lines: list[str]) -> float:
        """Fit preview text to its small axes; exports intentionally stay 16 pt."""
        if not lines:
            return 9.0
        width_points = (
            float(axis.get_position().width)
            * float(axis.figure.get_figwidth())
            * 72.0
        )
        longest = max(len(line) for line in lines)
        estimated = 0.88 * width_points / max(0.58 * longest, 1.0)
        return float(np.clip(estimated, 7.0, 9.0))

    def _add_mcd_preview_slope_box(
        self,
        axis: Any,
        branch_fits: dict[str, tuple[float, float]],
        corner: str,
    ) -> Any:
        """Add a compact, axes-contained slope box to the live preview."""
        lines = self._mcd_preview_slope_lines(branch_fits)
        if not lines:
            return None
        anchor = {
            "upper left": (0.025, 0.975, "left", "top"),
            "upper right": (0.975, 0.975, "right", "top"),
            "lower left": (0.025, 0.025, "left", "bottom"),
            "lower right": (0.975, 0.025, "right", "bottom"),
        }[corner]
        return axis.text(
            anchor[0], anchor[1], "\n".join(lines),
            transform=axis.transAxes, ha=anchor[2], va=anchor[3],
            fontsize=self._mcd_preview_annotation_fontsize(axis, lines),
            fontweight="semibold", color="#303030", linespacing=1.08,
            zorder=31, clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.22", "facecolor": "white",
                "edgecolor": "#9a9a9a", "linewidth": 0.65, "alpha": 0.88,
            },
        )

    def _refresh_mcd_trace_panel(self) -> bool:
        """Refresh only center-dependent MCD artists, preserving the heatmap."""
        if (
            self.loaded is None or self.loaded.mode != "MCD"
            or self.loaded.mcd_result is None or self._mcd_trace_ax is None
        ):
            return False
        result = self.loaded.mcd_result
        trace_ax = self._mcd_trace_ax
        old_integral = self._mcd_integral_ax
        if old_integral is not None and old_integral is not trace_ax:
            old_integral.clear()
        trace_ax.clear()
        self._mcd_trace_lines = {}
        self._mcd_fit_lines = {}
        self._mcd_slope_text = None
        e0 = float(self.mcd_window_center_spin.value())
        width = float(self.mcd_window_width_spin.value())
        self._move_mcd_window_artists(e0, draw=False)

        trace_specs = (
            ("mean", "Signed mean", "#1666b0", self.mcd_show_signed_mean_chk.isChecked()),
            ("field_signed_absolute_mean", "Field-signed |MCD|", "#c94c00", self.mcd_show_absolute_mean_chk.isChecked()),
            ("absolute_mean", "Unsigned |MCD|", "#777777", self.mcd_show_unsigned_absolute_mean_chk.isChecked()),
            ("integral", "Signed integral", "#6a3d9a", self.mcd_show_integral_chk.isChecked()),
        )
        show_raw = self.mcd_show_raw_chk.isChecked()
        requested_metrics = [name for name, _label, _color, visible in trace_specs if visible]
        if self.mcd_fit_zero_chk.isChecked() and "mean" not in requested_metrics:
            requested_metrics.append("mean")
        traces = pair_window_trace_by_branch(
            result, e0, width, metrics=requested_metrics, include_raw=show_raw
        )
        branch_fits = (
            low_field_mcd_branch_fits(traces, float(self.mcd_fit_b_window_spin.value()))
            if self.mcd_fit_zero_chk.isChecked() else {}
        )
        integral_ax = (
            old_integral
            if old_integral is not None and old_integral is not trace_ax
            else trace_ax.twinx()
        )
        self._mcd_integral_ax = integral_ax
        trace_ax.set_animated(self._mcd_blit_enabled)
        integral_ax.set_animated(self._mcd_blit_enabled)
        if self._mcd_blit_enabled:
            self._mcd_blit_axes["trace"] = (trace_ax, integral_ax)
        primary_values: list[np.ndarray] = []
        integral_values: list[np.ndarray] = []
        for metric_name, _label, color, visible in trace_specs:
            if not visible:
                continue
            target = integral_ax if metric_name == "integral" else trace_ax
            for branch, style, fill in (
                ("B increasing", "-", color), ("B decreasing", "--", "white")
            ):
                for source, alpha in (("corrected", 1.0), ("raw", 0.72)):
                    if source == "raw" and not show_raw:
                        continue
                    b_values, values = traces[branch][f"{source}_{metric_name}"]
                    finite = np.asarray(values, float)
                    (integral_values if metric_name == "integral" else primary_values).append(
                        finite[np.isfinite(finite)]
                    )
                    line, = target.plot(
                        b_values, values, f"o{style}", ms=3.1, lw=1.25,
                        color=color, alpha=alpha, markerfacecolor=fill,
                        markeredgecolor=color, markeredgewidth=0.9, label="_nolegend_",
                    )
                    self._mcd_trace_lines[(metric_name, branch, source)] = line
        if branch_fits:
            fields = np.concatenate([
                np.asarray(traces[branch]["corrected_mean"][0], float)
                for branch in ("B increasing", "B decreasing")
            ])
            finite_fields = fields[np.isfinite(fields)]
            if finite_fields.size >= 2:
                fit_x = np.asarray([np.min(finite_fields), np.max(finite_fields)], float)
                styles = {
                    "B increasing": ("#d55e00", "-"),
                    "B decreasing": ("#7a3db8", "--"),
                }
                for branch, (slope, intercept) in branch_fits.items():
                    color, style = styles[branch]
                    line, = trace_ax.plot(
                        fit_x, slope * fit_x + intercept, style, color=color,
                        lw=2.2, zorder=26, label="_nolegend_",
                    )
                    line.set_path_effects([
                        path_effects.Stroke(linewidth=3.5, foreground="white", alpha=0.95),
                        path_effects.Normal(),
                    ])
                    self._mcd_fit_lines[branch] = line

        def apply_data_ylim(target: Any, arrays: list[np.ndarray]) -> None:
            finite_arrays = [array for array in arrays if array.size]
            if not finite_arrays:
                return
            values = np.concatenate(finite_arrays)
            low, high = float(np.min(values)), float(np.max(values))
            span = high - low
            pad = 0.05 * (span if span > 0 else max(abs(low), abs(high), 1.0))
            target.set_ylim(low - pad, high + pad)

        apply_data_ylim(trace_ax, primary_values)
        if self.mcd_show_integral_chk.isChecked():
            apply_data_ylim(integral_ax, integral_values)
            integral_ax.set_ylabel("Integrated MCD (eV)", labelpad=2)
        else:
            integral_ax.set_yticks([])
            integral_ax.spines["right"].set_visible(False)
        trace_ax.axhline(0, color="#555", lw=0.7)
        trace_ax.set_title(f"MCD(B): E = {format_mcd_energy(e0)} eV", pad=3)
        trace_ax.set_xlabel("B field (T)")
        trace_ax.set_ylabel("MCD (mean / absolute mean)", labelpad=10)
        trace_ax.grid(alpha=0.25)

        visible_metrics = [spec for spec in trace_specs if spec[3]]
        layout = mcd_annotation_layout(
            trace_ax, integral_ax, show_conditions=False,
            show_slopes=bool(branch_fits), show_metric_legend=len(visible_metrics) > 1,
        )
        if branch_fits:
            self._mcd_slope_text = self._add_mcd_preview_slope_box(
                trace_ax, branch_fits, layout["slopes"]
            )
        branch_legend = trace_ax.legend(
            [
                Line2D([0], [0], color="#333", marker="o", markerfacecolor="#333", lw=1.15),
                Line2D([0], [0], color="#333", marker="o", markerfacecolor="white", lw=1.15, ls="--"),
            ],
            ["B increasing", "B decreasing"], title="Branch", fontsize=5.8,
            title_fontsize=6.0, frameon=True, framealpha=0.88,
            loc=layout["branch_legend"],
        )
        trace_ax.add_artist(branch_legend)
        if len(visible_metrics) > 1:
            trace_ax.legend(
                [Line2D([0], [0], color=color, lw=1.5) for _name, _label, color, _visible in visible_metrics],
                [f"{label} (right axis)" if name == "integral" else label
                 for name, label, _color, _visible in visible_metrics],
                title="Metric", fontsize=5.8, title_fontsize=6.0,
                frameon=True, framealpha=0.88, loc=layout["metric_legend"],
            )
        if not self._blit_mcd_regions(
            "trace", "pair_overlay", "linecut_overlay", "heat"
        ):
            self.canvas.draw_idle()
        self._status("Updated MCD integration preview.")
        return True

    def _refresh_mcd_center_trace(self) -> bool:
        """Update center-dependent data without reconstructing the trace axes."""
        if (
            self.loaded is None or self.loaded.mode != "MCD"
            or self.loaded.mcd_result is None or self._mcd_trace_ax is None
            or not self._mcd_trace_lines
        ):
            return False
        result = self.loaded.mcd_result
        e0 = float(self.mcd_window_center_spin.value())
        width = float(self.mcd_window_width_spin.value())
        requested_metrics = sorted({key[0] for key in self._mcd_trace_lines})
        show_raw = any(key[2] == "raw" for key in self._mcd_trace_lines)
        if self.mcd_fit_zero_chk.isChecked() and "mean" not in requested_metrics:
            requested_metrics.append("mean")
        traces = pair_window_trace_by_branch(
            result, e0, width, metrics=requested_metrics, include_raw=show_raw
        )

        primary_values: list[np.ndarray] = []
        integral_values: list[np.ndarray] = []
        for (metric, branch, source), line in self._mcd_trace_lines.items():
            b_values, values = traces[branch][f"{source}_{metric}"]
            line.set_data(b_values, values)
            finite = np.asarray(values, float)
            finite = finite[np.isfinite(finite)]
            (integral_values if metric == "integral" else primary_values).append(finite)

        branch_fits = (
            low_field_mcd_branch_fits(
                traces, float(self.mcd_fit_b_window_spin.value())
            )
            if self.mcd_fit_zero_chk.isChecked() else {}
        )
        if bool(branch_fits) != bool(self._mcd_fit_lines):
            return False
        if branch_fits:
            fields = np.concatenate([
                np.asarray(traces[branch]["corrected_mean"][0], float)
                for branch in ("B increasing", "B decreasing")
            ])
            finite_fields = fields[np.isfinite(fields)]
            if finite_fields.size < 2:
                return False
            fit_x = np.asarray([np.min(finite_fields), np.max(finite_fields)], float)
            for branch, (slope, intercept) in branch_fits.items():
                line = self._mcd_fit_lines.get(branch)
                if line is None:
                    return False
                line.set_data(fit_x, slope * fit_x + intercept)

        def apply_data_ylim(axis: Any, arrays: list[np.ndarray]) -> None:
            finite_arrays = [array for array in arrays if array.size]
            if not finite_arrays:
                return
            values = np.concatenate(finite_arrays)
            low, high = float(np.min(values)), float(np.max(values))
            span = high - low
            pad = 0.05 * (span if span > 0 else max(abs(low), abs(high), 1.0))
            axis.set_ylim(low - pad, high + pad)

        apply_data_ylim(self._mcd_trace_ax, primary_values)
        if self.mcd_show_integral_chk.isChecked() and self._mcd_integral_ax is not None:
            apply_data_ylim(self._mcd_integral_ax, integral_values)
        self._mcd_trace_ax.set_title(
            f"MCD(B): E = {format_mcd_energy(e0)} eV", pad=3
        )
        if branch_fits:
            if self._mcd_slope_text is None:
                return False
            lines = self._mcd_preview_slope_lines(branch_fits)
            self._mcd_slope_text.set_text("\n".join(lines))
            self._mcd_slope_text.set_fontsize(
                self._mcd_preview_annotation_fontsize(self._mcd_trace_ax, lines)
            )
        self._move_mcd_window_artists(e0, draw=False)
        if not self._blit_mcd_regions(
            "trace", "pair_overlay", "linecut_overlay", "heat"
        ):
            self.canvas.draw_idle()
        self._status("Updated MCD integration preview.")
        return True

    def _update_mcd_candidate_bar(self) -> None:
        candidates = self._mcd_center_candidates
        for index, button in enumerate(self.mcd_candidate_buttons):
            if index >= len(candidates):
                button.setVisible(False)
                button.setChecked(False)
                continue
            candidate = candidates[index]
            button.setText(f"{index + 1}  {format_mcd_energy(candidate.center_ev)}")
            button.setToolTip(
                f"Candidate {index + 1}: {format_mcd_energy(candidate.center_ev)} eV\n"
                f"Signal rank {candidate.score_rank}; SNR {candidate.snr:.3g}; "
                f"branch agreement {100.0 * candidate.branch_agreement:.0f}%\n"
                "Click to preview this fixed-width center."
            )
            button.setChecked(index == self._mcd_candidate_active_index)
            button.setVisible(True)
        enabled = bool(candidates)
        self.mcd_candidate_label.setText(
            f"Suggested ({len(candidates)}):" if enabled else "Suggested:"
        )
        self.mcd_previous_candidate_btn.setEnabled(enabled)
        self.mcd_next_candidate_btn.setEnabled(enabled)
        self.mcd_clear_candidates_btn.setEnabled(enabled)

    def _update_mcd_candidate_artist_styles(self) -> None:
        for index, artist in self._mcd_candidate_artists.items():
            patch = artist.get_bbox_patch()
            if patch is not None:
                patch.set_facecolor(
                    "#0078d4" if index == self._mcd_candidate_active_index else "#f0a202"
                )
        if self._mcd_blit_enabled:
            self._blit_mcd_regions("heat")








    def _on_canvas_release(self, event: Any) -> None:
        if not self._mcd_window_dragging:
            return
        self._mcd_window_dragging = False
        self.canvas.unsetCursor()
        center = self._mcd_window_drag_center
        moved = self._mcd_window_drag_moved
        self._mcd_window_drag_center = None
        self._mcd_window_drag_moved = False
        if moved and center is not None:
            self.mcd_window_center_spin.setValue(float(center))
            self._status(
                f"MCD window centered at {format_mcd_energy(center)} eV; "
                f"width remains {self.mcd_window_width_spin.value():g} meV."
            )

    def _on_mcd_canvas_motion(self, event: Any) -> None:
        """Handle MCD-only motion behavior from the shared canvas dispatcher."""
        if self._mcd_window_dragging:
            if event.xdata is None or self.loaded is None or self.loaded.mcd_result is None:
                return
            requested = float(event.xdata) - self._mcd_window_drag_offset
            center = self._clamp_mcd_window_center(
                requested,
                self.loaded.mcd_result.energy_ev,
                float(self.mcd_window_width_spin.value()),
            )
            self._mcd_window_drag_center = center
            self._mcd_window_drag_moved = True
            self._move_mcd_window_artists(center)
            self.canvas.setCursor(Qt.SizeHorCursor)
            self.status_bar_view.set_cursor_readback(
                f"Move MCD window: E = {format_mcd_energy(center)} eV "
                f"(fixed width {self.mcd_window_width_spin.value():g} meV)"
            )
            return
        if event.inaxes is self._mcd_heatmap_ax and event.xdata is not None:
            if self._mcd_window_contains_energy(float(event.xdata)):
                self.canvas.setCursor(Qt.SizeHorCursor)
                self.status_bar_view.set_cursor_readback(
                    "Drag the highlighted MCD band left or right; its width stays fixed."
                )
            else:
                self.canvas.unsetCursor()
        else:
            self.canvas.unsetCursor()

    def _on_mcd_canvas_click(self, event: Any) -> None:
        """Handle MCD-only click behavior from the shared canvas dispatcher."""
        if event.button != 1:
            return
        if event.inaxes is not self._mcd_heatmap_ax or event.ydata is None:
            return
        if event.xdata is not None:
            candidate_index = self._mcd_candidate_marker_at(
                float(event.xdata), float(event.ydata)
            )
            if candidate_index is not None:
                self._use_mcd_center_candidate(candidate_index)
                return
        key = str(getattr(event, "key", "") or "").casefold()
        control_pressed = (
            "control" in key or "ctrl" in key
            or bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        )
        if control_pressed:
            self._select_mcd_pair_at_b(float(event.ydata), clicked=True)
            return
        if event.xdata is not None and self._mcd_window_contains_energy(float(event.xdata)):
            self._mcd_window_dragging = True
            self._mcd_window_drag_moved = False
            self._mcd_window_drag_offset = (
                float(event.xdata) - float(self.mcd_window_center_spin.value())
            )
            self._mcd_window_drag_center = float(self.mcd_window_center_spin.value())
            self.canvas.setCursor(Qt.SizeHorCursor)
            return
        self._status("Hold Ctrl and click the MCD colormap to select a B-field pair.")

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
