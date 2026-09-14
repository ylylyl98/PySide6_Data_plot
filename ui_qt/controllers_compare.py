"""Controller for Compare workflow actions."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QListWidgetItem, QComboBox, QDoubleSpinBox, QToolButton

from core import data_io
from core.compare_history import compare_history_for_selection
from core.export import compare_source_title, vp_compare_export_base, vp_compare_title
from core.plotting import COMPARE_PANEL_ORDER
from core.source_identity import match_source_identity
from core.processing import (
    background_correct_cube,
    classify_angle_state,
    classify_compare_channel,
    coherent_compare_auto_assignment,
    estimate_constant_background,
    infer_compare_angle_references,
    nearest_gate_spectrum,
    parse_compare_gate_condition,
    parse_compare_rotation_angles,
    valley_polarization_cube,
    group_compare_sources,
)
from ui_qt.source_picker_dialog import SourcePickerDialog
from ui_qt.theme import alias as theme_alias


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

    def _cmp_rot1_is_output(self) -> bool:
        combo = getattr(self, "cmp_rotation_mapping_combo", None)
        return combo is not None and combo.currentData() == "rot1_output"

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
        groups = list(helper(
            candidates,
            rot1_is_output=self._cmp_rot1_is_output(),
            in_k_angle=float(self.cmp_in_k_angle_spin.value()),
            out_k_angle=float(self.cmp_out_k_angle_spin.value()),
            in_kp_angle=float(self.cmp_in_kp_angle_spin.value()),
            out_kp_angle=float(self.cmp_out_kp_angle_spin.value()),
            tolerance=float(self.cmp_angle_tolerance_spin.value()),
            power_tolerance_fraction=float(power_tolerance_fraction),
        ))
        # PL presents newest sources first.  Compare groups use the newest
        # member's source mtime as their recency while retaining helper order
        # as the stable tie-breaker.
        return [
            group for _index, group in sorted(
                enumerate(groups),
                key=lambda pair: (-self._cmp_group_modified(pair[1]), pair[0]),
            )
        ]

    def _cmp_source_modified(self, source: str) -> float:
        root = str(getattr(self, "current_folder", "") or "")
        cache_folder = str(getattr(self, "_cmp_source_mtime_cache_folder", "") or "")
        cache = getattr(self, "_cmp_source_mtime_cache", None)
        if not isinstance(cache, dict) or cache_folder != root:
            cache = {}
            self._cmp_source_mtime_cache = cache
            self._cmp_source_mtime_cache_folder = root
        cache_key = str(source)
        # Missing metadata is intentionally unknown until the owner catalog
        # worker publishes it.  Compare picker code must never stat files on
        # the GUI thread.
        try:
            return float(cache.get(cache_key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _cmp_group_modified(self, group) -> float:
        sources = tuple(self._cmp_group_value(group, "sources", ()) or ())
        return max((self._cmp_source_modified(str(source)) for source in sources), default=0.0)

    def _cmp_group_history_mapping(self, group) -> dict[str, str]:
        mapping = self._cmp_group_mapping(group)
        if self._cmp_is_vp_view():
            return {key: mapping[key] for key in ("KK", "KKp") if key in mapping}
        return {
            key: mapping[key]
            for key in self._cmp_visible_channels(mapping)
            if key in mapping
        }

    def _cmp_group_history_matches(self, group) -> list[dict]:
        folder = str(getattr(self, "current_folder", "") or "")
        mapping = self._cmp_group_history_mapping(group)
        if not folder or not mapping:
            return []
        return compare_history_for_selection(
            getattr(self, "_cmp_history_records", ()),
            folder,
            mapping,
            view="VP" if self._cmp_is_vp_view() else "Intensity",
            sources=tuple(dict.fromkeys([
                *map(str, getattr(self, "pl_available_files", ()) or ()),
                *mapping.values(),
            ])),
        )

    def _cmp_group_history_uncertain(self, group) -> bool:
        """Detect legacy basename ambiguity without guessing a source identity."""
        folder = str(getattr(self, "current_folder", "") or "")
        mapping = self._cmp_group_history_mapping(group)
        if not folder or not mapping:
            return False
        candidates = tuple(dict.fromkeys([
            *map(str, getattr(self, "pl_available_files", ()) or ()),
            *mapping.values(),
            *map(str, self._cmp_group_value(group, "sources", ()) or ()),
        ]))
        wanted_view = "VP" if self._cmp_is_vp_view() else "Intensity"
        for record in getattr(self, "_cmp_history_records", ()):
            if not isinstance(record, dict):
                continue
            operation = str(record.get("operation", record.get("workflow", "")))
            record_view = "VP" if operation.casefold() == "compare/vp" else "Intensity"
            if not operation.casefold().startswith("compare/") or record_view != wanted_view:
                continue
            processing = record.get("processing", {})
            combo = processing.get("compare_source_mapping") if isinstance(processing, dict) else None
            if isinstance(combo, dict):
                descriptors = {
                    str(key): str(value)
                    for key, value in combo.items()
                    if value and str(key) in mapping
                }
            else:
                channel = operation.split("/", 1)[1] if "/" in operation else ""
                descriptors = {channel: ""} if channel in mapping else {}
                raw_sources = record.get("sources", [])
                if isinstance(raw_sources, list):
                    for descriptor in raw_sources:
                        if not isinstance(descriptor, dict):
                            continue
                        role = str(descriptor.get("role", ""))
                        if wanted_view == "VP":
                            role = role.removeprefix("source_")
                        elif role == "source":
                            role = channel
                        if role in mapping:
                            descriptors[role] = descriptor
            for channel, recorded in descriptors.items():
                expected = mapping.get(channel, "")
                if not expected:
                    continue
                if isinstance(recorded, dict):
                    matched, uncertain = match_source_identity(
                        folder, candidates,
                        relative_path=str(recorded.get("source_relative_path", "")),
                        path=str(recorded.get("source_path", recorded.get("path", ""))),
                        legacy_name=str(recorded.get("name", recorded.get("filename", ""))),
                    )
                else:
                    matched, uncertain = match_source_identity(
                        folder, candidates, legacy_name=str(recorded),
                    )
                if matched is None and expected in uncertain:
                    return True
        return False

    def _cmp_group_status(self, group) -> str:
        """Classify a group using the existing Compare history semantics."""
        if not str(getattr(self, "current_folder", "") or ""):
            return "unknown"
        mapping = self._cmp_group_history_mapping(group)
        if self._cmp_is_vp_view():
            if set(mapping) != {"KK", "KKp"}:
                return "unknown"
        elif len(mapping) < 2:
            return "unknown"
        matches = self._cmp_group_history_matches(group)
        if any(item.get("history_scope") == "combination" for item in matches):
            return "processed"
        if any(item.get("history_scope") == "individual_panel" for item in matches):
            return "mixed"
        if self._cmp_group_history_uncertain(group):
            return "unknown"
        return "new"

    @staticmethod
    def _cmp_group_status_text(status: str) -> str:
        return {
            "processed": "✓ PROCESSED",
            "mixed": "◐ MIXED",
            "unknown": "? HISTORY UNKNOWN",
            "new": "● NEW",
        }.get(status, "● NEW")

    def _cmp_group_modified_text(self, group) -> str:
        modified = self._cmp_group_modified(group)
        return datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M") if modified else "date unavailable"

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
        records = getattr(self, "_cmp_history_records", ())
        folder = str(getattr(self, "current_folder", "") or "")
        if folder:
            view = "VP" if self._cmp_is_vp_view() else "Intensity"
            # History freshness follows the channels currently shown.  Keep
            # hidden combo assignments in the controller so they survive a
            # preset change, but do not let a hidden channel invalidate the
            # visible saved combination.
            history_mapping = self._cmp_current_mapping()
            if view == "VP":
                history_mapping = {
                    key: history_mapping[key]
                    for key in ("KK", "KKp")
                    if key in history_mapping
                }
            else:
                visible = self._cmp_visible_channels(history_mapping)
                history_mapping = {
                    key: history_mapping[key]
                    for key in visible
                    if key in history_mapping
                }
            matches = compare_history_for_selection(
                records, folder, history_mapping, view=view,
                sources=tuple(getattr(self, "pl_available_files", ()) or ()),
            )
            if matches:
                newest = max(matches, key=lambda item: str(item.get("created_utc", "")))
                stamp = str(newest.get("created_utc", ""))[:16].replace("T", " ") or "date unavailable"
                scope = str(newest.get("history_scope", "combination"))
                text = f"Saved Compare {view} history: {stamp}"
                if scope == "individual_panel":
                    text += "\nIndividual panel history; combined selection not verified"
                if scope == "individual_panel":
                    details = "; ".join(
                        f"{item.get('channel', 'panel')} {str(item.get('created_utc', ''))[:16].replace('T', ' ')}"
                        for item in matches
                    )
                    text += f"\nMatched: {details}"
                badge.set_status(text, tooltip=tooltip + "\nHistorical export; current settings not verified", badge_state="selected")
                return
            badge.set_status("No saved Compare history found", tooltip=tooltip, badge_state="new")
            return
        badge.set_status(label or key, tooltip=tooltip, badge_state="selected")

    def _cmp_refresh_history_cache(self, *, force: bool = False) -> None:
        folder = str(getattr(self, "current_folder", "") or "")
        ready = bool(getattr(self, "_cmp_history_cache_ready", False))
        pending = bool(getattr(self, "_cmp_history_cache_pending", False))
        cached_folder = str(getattr(self, "_cmp_history_cache_folder", "") or "")
        owner_cache_folder = str(getattr(self._owner, "_cmp_history_cache_folder", "") or "")
        owner_generation = getattr(self._owner, "_file_refresh_generation", None)
        published_generation = getattr(self._owner, "_cmp_history_published_generation", None)
        published_folder = str(getattr(self._owner, "_cmp_history_published_folder", "") or "")
        requested_generation = getattr(self, "_cmp_history_cache_generation", None)
        if not folder:
            self._cmp_history_records = []
            self._cmp_history_cache_folder = ""
            self._cmp_history_cache_ready = False
            self._cmp_history_cache_pending = False
            return

        if cached_folder != folder:
            self._cmp_history_records = []
            self._cmp_history_cache_folder = folder
            ready = False
            pending = False

        owner_records = list(getattr(self._owner, "_cmp_history_records", ()) or ())
        owner_ready_modes = getattr(self._owner, "_catalog_ready_modes", set())
        if (
            not force and pending and folder == owner_cache_folder
            and "Compare" in owner_ready_modes
            and published_folder.casefold() == folder.casefold()
            and requested_generation is not None and published_generation is not None
            # A force request queued behind an in-flight scan may publish at
            # a later generation.  Once all coalesced work is drained, any
            # accepted publication at or after the request satisfies it.
            and int(published_generation) >= int(requested_generation)
            and not bool(getattr(self._owner, "_file_refresh_pending", False))
            and not bool(getattr(self._owner, "_catalog_pending_requests", {}).get("Compare", False))
        ):
            self._cmp_history_records = owner_records
            self._cmp_history_cache_ready = True
            self._cmp_history_cache_pending = False
            return
        # A forced request always reaches the owner worker, even with warm
        # records.  Keep those records visible while the replacement is in
        # flight, but never treat them as the fresh snapshot.
        if not force and (ready or pending):
            return
        if not force and folder == owner_cache_folder and owner_records:
            self._cmp_history_records = owner_records
            self._cmp_history_cache_ready = True
            self._cmp_history_cache_pending = False
            self._cmp_history_cache_generation = getattr(self._owner, "_file_refresh_generation", None)
            return

        queue = getattr(self._owner, "_refresh_file_lists", None)
        self._cmp_history_cache_pending = True
        self._cmp_history_cache_ready = ready and cached_folder == folder
        self._cmp_history_cache_generation = getattr(self._owner, "_file_refresh_generation", None)
        # MainWindow marks the mode only after the worker result is accepted.
        # Clear the previous publication token so a same-folder cold/forced
        # request cannot certify itself from the request generation.
        if isinstance(owner_ready_modes, set):
            owner_ready_modes.discard("Compare")
        if callable(queue):
            queue(auto=not force, mode="Compare")
            self._cmp_history_cache_generation = getattr(self._owner, "_file_refresh_generation", None)

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
        status_filter = QComboBox()
        status_filter.setObjectName("compare_group_status_filter")
        style_combo = getattr(self, "_style_combo_popup", None)
        if callable(style_combo):
            style_combo(status_filter)
        dlg = SourcePickerDialog(
            self._owner,
            title="Choose Compare Group",
            hint=("Choose a coherent KK / KKp / KpK / KpKp source group. "
                  "Files are grouped by shared measurement context and nearby power."),
            selected=selected_key,
            filter_controls=(("Power tolerance", tolerance_spin), ("Status", status_filter)),
            filter_interval=100,
            minimum_size=(860, 520),
            size=(1040, 660),
            auto_select_single=False,
        )
        refresh_pending = False
        dialog_alive = True
        refresh_folder = str(getattr(self, "current_folder", "") or "")

        def _groups_for_dialog():
            return self._cmp_source_groups(
                power_tolerance_fraction=float(tolerance_spin.value()) / 100.0
            )

        def _populate_status_filter(groups) -> None:
            current = str(status_filter.currentData() or "all")
            counts = Counter(self._cmp_group_status(group) for group in groups)
            blocked = status_filter.blockSignals(True)
            try:
                status_filter.clear()
                status_filter.addItem(f"All ({len(groups)})", "all")
                status_filter.addItem(f"New ({counts.get('new', 0)})", "new")
                status_filter.addItem(f"Processed ({counts.get('processed', 0)})", "processed")
                status_filter.addItem(f"Mixed ({counts.get('mixed', 0)})", "mixed")
                status_filter.addItem(f"History unknown ({counts.get('unknown', 0)})", "unknown")
                index = status_filter.findData(current)
                status_filter.setCurrentIndex(index if index >= 0 else 0)
            finally:
                status_filter.blockSignals(blocked)

        def _refresh_view() -> None:
            needle = dlg.filter_edit.text().strip().casefold()
            # History is populated by the catalog/export refresh lifecycle;
            # search, status, and tolerance changes must remain cache-only.
            self._cmp_refresh_history_cache()
            groups = _groups_for_dialog()
            _populate_status_filter(groups)
            wanted_status = str(status_filter.currentData() or "all")
            row_descriptors = []
            for group in groups:
                key = str(self._cmp_group_value(group, "key", ""))
                label = str(self._cmp_group_value(group, "label", key))
                sources = tuple(self._cmp_group_value(group, "sources", ()) or ())
                haystack = " ".join((key, label, *map(str, sources))).casefold()
                if needle and needle not in haystack:
                    continue
                status = self._cmp_group_status(group)
                if wanted_status != "all" and status != wanted_status:
                    continue
                status_text = self._cmp_group_status_text(status)
                modified_text = self._cmp_group_modified_text(group)
                foreground = str(theme_alias(
                    "source_processed_foreground"
                    if status == "processed" else "source_new_foreground"
                ))
                text = status_text + " — " + label + (
                    "\nModified " + modified_text + " · " + " · ".join(Path(source).name for source in sources)
                    if sources else "\nModified " + modified_text
                )
                row_descriptors.append((
                    key, text, tuple(map(str, sources)), status, foreground,
                    bool(status != "processed"), int((Qt.ItemIsEnabled | Qt.ItemIsSelectable).value),
                ))

            def _populate(widget) -> None:
                for key, text, sources, status, foreground, bold, _flags in row_descriptors:
                    item = QListWidgetItem(
                        text
                    )
                    item.setData(Qt.UserRole, key)
                    item.setToolTip("\n".join(map(str, sources)))
                    item.setForeground(QColor(foreground))
                    font = item.font()
                    font.setBold(bold)
                    item.setFont(font)
                    widget.addItem(item)
            dlg._cmp_groups = groups
            key = ("compare-rows", tuple(row_descriptors))
            dlg.repopulate(_populate, fallback_selection=selected_key, content_key=key)
            if not groups:
                dlg.set_details("No coherent compare groups match the current source filter and angle rules.")

        def _request_refresh() -> None:
            nonlocal refresh_pending
            refresh = getattr(self, "_refresh_file_lists", None)
            if not callable(refresh):
                self._cmp_refresh_history_cache(force=True)
                _refresh_view()
                return
            refresh_pending = True
            dlg.refresh_button.setEnabled(False)
            refresh(auto=False, mode="Compare")

        def _catalog_refresh_finished(folder: str, success: bool) -> None:
            nonlocal refresh_pending
            history_pending = bool(getattr(self, "_cmp_history_cache_pending", False))
            if not dialog_alive or (not refresh_pending and not history_pending):
                return
            if str(folder).casefold() != refresh_folder.casefold():
                return
            # MainWindow emits completion for the scan that just finished
            # before starting a queued follow-up scan. Keep this dialog's
            # request pending until that final catalog is applied.
            if success and bool(getattr(self, "_file_refresh_pending", False)):
                return
            if success and history_pending:
                owner_folder = str(getattr(self._owner, "_cmp_history_cache_folder", "") or "")
                owner_generation = getattr(self._owner, "_file_refresh_generation", None)
                requested_generation = getattr(self, "_cmp_history_cache_generation", None)
                if owner_folder.casefold() != refresh_folder.casefold():
                    return
                if (requested_generation is not None and owner_generation is not None
                        and int(owner_generation) < int(requested_generation)):
                    return
                self._cmp_history_records = list(getattr(self._owner, "_cmp_history_records", ()) or ())
                self._cmp_history_cache_folder = refresh_folder
                self._cmp_history_cache_ready = True
                self._cmp_history_cache_pending = False
            was_refresh_pending = refresh_pending
            if was_refresh_pending:
                refresh_pending = False
                dlg.refresh_button.setEnabled(True)
            if success:
                _refresh_view()
                _update_details()
            elif was_refresh_pending:
                self._cmp_history_cache_pending = False
                self._cmp_history_cache_ready = False
                dlg.set_details("Refresh failed; existing Compare selection was retained.")

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
            status = self._cmp_group_status(group) if group else "unknown"
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
            if group:
                detail_lines.append(
                    f"Status: {self._cmp_group_status_text(status)} · Modified {self._cmp_group_modified_text(group)}"
                )
                if status == "processed":
                    detail_lines.append("Matched saved Compare history for the active view.")
                elif status == "mixed":
                    detail_lines.append("Individual panel history exists; combined selection is not verified.")
                elif status == "unknown":
                    detail_lines.append("History cannot verify this group with the available channel identities.")
                else:
                    detail_lines.append("No saved Compare history found for this group.")
            dlg.set_details("\n".join(detail_lines) or "No channel assignments in this group.")

        def _dialog_finished(_result: int) -> None:
            nonlocal dialog_alive
            dialog_alive = False

        dlg.filter_requested.connect(_refresh_view)
        dlg.source_list.currentItemChanged.connect(lambda _current, _previous: _update_details())
        tolerance_spin.valueChanged.connect(lambda _value: _refresh_view())
        status_filter.currentIndexChanged.connect(lambda _index: _refresh_view())
        dlg.refresh_button.clicked.connect(_request_refresh)
        catalog_signal = getattr(self, "file_catalog_refresh_finished", None)
        if catalog_signal is not None:
            catalog_signal.connect(_catalog_refresh_finished)
            dlg.finished.connect(_dialog_finished)
            def _disconnect_catalog(_result: int) -> None:
                try:
                    catalog_signal.disconnect(_catalog_refresh_finished)
                except (RuntimeError, TypeError):
                    pass
            dlg.finished.connect(_disconnect_catalog)
        _refresh_view()
        _update_details()
        if dlg.exec() != SourcePickerDialog.Accepted:
            return
        if str(getattr(self, "current_folder", "") or "").casefold() != refresh_folder.casefold():
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
        self._cmp_maybe_auto_load()

    def _cmp_clear_group(self) -> None:
        invalidate = getattr(self, "_invalidate_active_load", None)
        if callable(invalidate):
            invalidate("Compare")
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
                if current and current not in candidates:
                    combo.addItem(current)
                    index = combo.findText(current)
                    root = str(getattr(self, "current_folder", "") or "")
                    path = Path(current) if Path(current).is_absolute() else Path(root) / current
                    detail = (
                        f"Missing source: {current}."
                        if root and not path.is_file()
                        else f"Currently assigned; outside {self._cmp_source_filter_label()}."
                    )
                    combo.setItemData(index, detail, Qt.ToolTipRole)
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

    @staticmethod
    def _cmp_background_cube_token(cube):
        """Return mutable-data identity used to validate an auto-background cache."""
        z_data = getattr(cube, "Z", None)
        revision = getattr(cube, "revision", None)
        if revision is None:
            revision = getattr(cube, "_revision", getattr(cube, "source_revision", None))
        return id(cube), id(z_data), revision

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
            owner = object.__getattribute__(self, "_owner")
            source_cubes = self._cmp_background_source_cubes(cubes)
            cache_key = tuple(
                (key, self._cmp_background_cube_token(cube))
                for key, cube in sorted(source_cubes.items())
            )
            cached = getattr(owner, "_cmp_background_cache", None)
            cache_sources = ()
            if isinstance(cached, tuple) and len(cached) == 3:
                cache_sources = cached[1]
            same_sources = (
                isinstance(cache_sources, tuple)
                and len(cache_sources) == len(source_cubes)
                and all(
                    key in source_cubes and source is source_cubes[key]
                    for key, source in cache_sources
                )
            )
            if (isinstance(cached, tuple) and len(cached) == 3
                    and cached[0] == cache_key and same_sources):
                value = float(cached[2])
            else:
                value = estimate_constant_background(source_cubes, percentile=1.0)
                # Keep strong references so an old id cannot become a false hit
                # after a load replaces the source cube.
                owner._cmp_background_cache = (
                    cache_key, tuple(sorted(source_cubes.items())), float(value)
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

    def _cmp_maybe_auto_load(self) -> None:
        """Start one load after a complete explicit channel assignment."""
        start_load = getattr(self, "_start_load", None)
        if not callable(start_load) or not getattr(self, "current_folder", ""):
            return
        try:
            self._cmp_selection_from_ui()
        except (ValueError, OSError):
            return
        start_load("Compare")

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
            active_mapping = {key: mapping[key] for key in ("KK", "KKp")}
            missing_sources = self._cmp_missing_sources(active_mapping)
            if missing_sources:
                raise ValueError("Compare source missing: " + ", ".join(missing_sources))
            return data_io.CompareSelection.from_mapping(mapping, visible_order=("KK", "KKp"))
        visible = self._cmp_visible_channels(mapping)
        if len(visible) < 1:
            raise ValueError("Assign at least one compare channel.")
        if len(visible) < 2:
            raise ValueError("Select at least two visible compare channels.")
        missing_sources = self._cmp_missing_sources({key: mapping[key] for key in visible})
        if missing_sources:
            raise ValueError("Compare source missing: " + ", ".join(missing_sources))
        return data_io.CompareSelection.from_mapping(mapping, visible_order=visible)

    def _cmp_missing_sources(self, mapping: dict[str, str]) -> list[str]:
        root = str(getattr(self, "current_folder", "") or "")
        if not root:
            return []
        missing: list[str] = []
        for source in mapping.values():
            path = Path(source) if Path(source).is_absolute() else Path(root) / source
            if not path.is_file() and source not in missing:
                missing.append(source)
        return missing

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
            rot1_is_output=self._cmp_rot1_is_output(),
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
        )
        if allow_inference and selected_group is not None and not manual_mapping_is_usable:
            current_group_mapping = self._cmp_group_mapping(selected_group)
            if len(current_group_mapping) < 2:
                selected_sources = tuple(
                    self._cmp_group_value(selected_group, "sources", ()) or ()
                )
                inference = infer_compare_angle_references(
                    selected_sources,
                    rot1_is_output=self._cmp_rot1_is_output(),
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
                    rot1_is_output=self._cmp_rot1_is_output(),
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
            rot1_is_output=self._cmp_rot1_is_output(),
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
                found = current_mapping
            elif (
                had_selected_group
                and preserve_existing
                and current_mapping
                and inferred_selected_mapping is None
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
                rot1_is_output=self._cmp_rot1_is_output(),
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
            angles = parse_compare_rotation_angles(fname, rot1_is_output=self._cmp_rot1_is_output())
            if angles.rot1 is None and angles.rot2 is None:
                unlabeled_count += 1
            reasons: list[str] = []
            for arm, angle, k_angle, kp_angle in (
                ("Rot1", angles.rot1, out_k if self._cmp_rot1_is_output() else in_k, out_kp if self._cmp_rot1_is_output() else in_kp),
                ("Rot2", angles.rot2, in_k if self._cmp_rot1_is_output() else out_k, in_kp if self._cmp_rot1_is_output() else out_kp),
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
                angles = parse_compare_rotation_angles(fname, rot1_is_output=self._cmp_rot1_is_output())
                if (angles.rot1 is None) != (angles.rot2 is None):
                    detected = (
                        f"Rot1={angles.rot1:g} deg"
                        if angles.rot1 is not None
                        else f"Rot2={angles.rot2:g} deg"
                    )
                    fixed = "input" if ((angles.rot1 is not None) == self._cmp_rot1_is_output()) else "output"
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

    def _update_cmp_gate_only(self) -> bool:
        """Update Compare linecut and gate markers while retaining heatmap axes."""
        owner = object.__getattribute__(self, "_owner")
        if getattr(owner, "last_plotted_mode", None) != "Compare":
            return False
        if getattr(owner, "_load_in_progress", False) or "Compare" in getattr(owner, "_plot_redraw_pending", set()):
            return False
        cubes = getattr(owner, "_cmp_active_cubes", None) or {}
        heatmap_axes = getattr(owner, "_cmp_heatmap_axes", None) or {}
        linecut_ax = getattr(owner, "_cmp_linecut_ax", None)
        if not cubes or not heatmap_axes or linecut_ax is None:
            return False
        try:
            current_key = owner._current_plot_params_key("Compare")
            previous_key = getattr(owner, "_last_plot_params_key", None)
        except (AttributeError, ValueError, TypeError):
            return False
        if not isinstance(current_key, tuple) or not isinstance(previous_key, tuple):
            return False
        if len(current_key) < 19 or len(previous_key) < 19:
            return False
        if current_key[:15] + current_key[16:] != previous_key[:15] + previous_key[16:]:
            return False
        shown_identity = getattr(owner, "_shown_draw_identity", None)
        identity_for_loaded = getattr(owner, "_shown_source_identity_for_loaded", None)
        if shown_identity is not None and callable(identity_for_loaded):
            if shown_identity != identity_for_loaded(getattr(owner, "loaded", None)):
                return False
        rendered_marker = getattr(owner, "_cmp_rendered_loaded_marker", None)
        if rendered_marker is not None:
            loaded = getattr(owner, "loaded", None)
            current_marker = (
                id(loaded),
                id(getattr(loaded, "compare_cubes", None)),
                str(getattr(loaded, "folder", "")),
                tuple(getattr(loaded, "selected_files", ()) or ()),
            )
            if current_marker != rendered_marker:
                return False

        gate_value = float(self.cmp_spins["gate"].value())
        if self._cmp_is_vp_view():
            if "VP" not in cubes or len(linecut_ax.lines) < 1:
                return False
            gate_used, y = nearest_gate_spectrum(cubes["VP"], gate_value)
            linecut_ax.lines[0].set_data(np.asarray(cubes["VP"].energy, float).ravel(), np.asarray(y, float))
            linecut_ax.set_title(f"VP Linecut @ {gate_used:.6g} V")
            linecut_ax.set_ylim(-1.05, 1.05)
        else:
            keys = [key for key in COMPARE_PANEL_ORDER if key in cubes]
            if not keys or len(linecut_ax.lines) < len(keys):
                return False
            used = []
            for line, key in zip(linecut_ax.lines, keys):
                gate, y = nearest_gate_spectrum(cubes[key], gate_value)
                line.set_data(np.asarray(cubes[key].energy, float).ravel(), np.asarray(y, float))
                used.append(float(gate))
            gate_used = float(np.median(used))
            linecut_ax.set_title(f"Compare Spectra @ {gate_used:.6g} V")
            finite_lines = [
                np.asarray(line.get_ydata(), float)
                for line in linecut_ax.lines[:len(keys)]
                if len(line.get_ydata())
            ]
            finite_values = np.concatenate([
                values[np.isfinite(values)] for values in finite_lines
                if np.any(np.isfinite(values))
            ]) if any(np.any(np.isfinite(values)) for values in finite_lines) else np.array([])
            if finite_values.size:
                ymin, ymax = float(np.min(finite_values)), float(np.max(finite_values))
                pad = max(1e-12, (ymax - ymin) * 0.08) if ymax != ymin else max(1e-12, abs(ymin) * 0.05, 1.0)
                linecut_ax.set_ylim(ymin - pad, ymax + pad)
        self._set_cmp_gate_spin_value(gate_used)
        self._ensure_cmp_gate_lines(cubes, gate_used)
        if self._configure_cmp_blitting():
            canvas = getattr(owner, "canvas", None)
            bbox = getattr(owner, "_cmp_blit_bbox", None)
            if canvas is not None and bbox is not None:
                canvas.restore_region(owner._cmp_blit_background)
                owner.figure.draw_artist(owner._cmp_linecut_ax)
                for line in (getattr(owner, "_cmp_gate_lines", {}) or {}).values():
                    axis = getattr(line, "axes", None)
                    if axis is not None:
                        axis.draw_artist(line)
                canvas.blit(bbox)
                owner._last_plot_params_key = current_key
                return True
        xlim = linecut_ax.get_xlim()
        if self._cmp_is_vp_view():
            # VP linecuts deliberately keep the fixed normalized range.
            linecut_ax.set_ylim(-1.05, 1.05)
        else:
            linecut_ax.relim()
            linecut_ax.autoscale_view(scalex=False, scaley=True)
        linecut_ax.set_xlim(xlim)
        canvas = getattr(owner, "canvas", None)
        if canvas is not None:
            canvas.draw_idle()
        owner._last_plot_params_key = current_key
        return True

    def _disable_cmp_blitting(self) -> None:
        owner = object.__getattribute__(self, "_owner")
        linecut = getattr(owner, "_cmp_linecut_ax", None)
        if linecut is not None:
            linecut.set_animated(False)
        for line in (getattr(owner, "_cmp_gate_lines", {}) or {}).values():
            line.set_animated(False)
        owner._cmp_blit_enabled = False
        owner._cmp_blit_bbox = None
        owner._cmp_blit_background = None

    def _configure_cmp_blitting(self) -> bool:
        owner = object.__getattribute__(self, "_owner")
        canvas = getattr(owner, "canvas", None)
        linecut = getattr(owner, "_cmp_linecut_ax", None)
        heat_axes = tuple((getattr(owner, "_cmp_heatmap_axes", {}) or {}).values())
        required = ("draw", "copy_from_bbox", "restore_region", "blit")
        if (canvas is None or linecut is None or not heat_axes
                or not bool(getattr(canvas, "supports_blit", False))
                or any(not callable(getattr(canvas, name, None)) for name in required)):
            return False
        if getattr(owner, "_cmp_blit_enabled", False):
            bbox = getattr(owner, "_cmp_blit_bbox", None)
            if bbox is not None and bbox.bounds == owner.figure.bbox.bounds:
                return True
            self._disable_cmp_blitting()
        self._disable_cmp_blitting()
        try:
            # Exclude only dynamic linecut/gate artists while capturing a
            # clean full-figure background. Heatmaps remain normal artists.
            linecut.set_animated(True)
            for line in (getattr(owner, "_cmp_gate_lines", {}) or {}).values():
                line.set_animated(True)
            canvas.draw()
            bbox = owner.figure.bbox.frozen()
            owner._cmp_blit_bbox = bbox
            owner._cmp_blit_background = canvas.copy_from_bbox(bbox)
            owner._cmp_blit_enabled = True
            self._draw_cmp_blit_frame()
            return True
        except Exception:
            self._disable_cmp_blitting()
            return False

    def _draw_cmp_blit_frame(self) -> None:
        owner = object.__getattribute__(self, "_owner")
        canvas = getattr(owner, "canvas", None)
        bbox = getattr(owner, "_cmp_blit_bbox", None)
        background = getattr(owner, "_cmp_blit_background", None)
        linecut = getattr(owner, "_cmp_linecut_ax", None)
        if canvas is None or bbox is None or background is None or linecut is None:
            return
        canvas.restore_region(background)
        owner.figure.draw_artist(linecut)
        for line in (getattr(owner, "_cmp_gate_lines", {}) or {}).values():
            axis = getattr(line, "axes", None)
            if axis is not None:
                axis.draw_artist(line)
        canvas.blit(bbox)

    def _on_compare_canvas_draw(self, event=None) -> None:
        owner = object.__getattribute__(self, "_owner")
        if not getattr(owner, "_cmp_blit_enabled", False):
            return
        canvas = getattr(owner, "canvas", None)
        linecut = getattr(owner, "_cmp_linecut_ax", None)
        if canvas is None or linecut is None:
            self._disable_cmp_blitting()
            return
        bbox = owner.figure.bbox.frozen()
        owner._cmp_blit_bbox = bbox
        owner._cmp_blit_background = canvas.copy_from_bbox(bbox)
        self._draw_cmp_blit_frame()

    def _prepare_cmp_toolbar_save(self) -> None:
        owner = object.__getattribute__(self, "_owner")
        if not getattr(owner, "_cmp_blit_enabled", False):
            return
        self._disable_cmp_blitting()
        canvas = getattr(owner, "canvas", None)
        if canvas is not None:
            canvas.draw()

    def _restore_cmp_toolbar_save(self) -> None:
        owner = object.__getattribute__(self, "_owner")
        if getattr(owner, "last_plotted_mode", None) != "Compare":
            return
        self._configure_cmp_blitting()

    def _on_cmp_auto_assign_requested(self) -> None:
        self._invalidate_export_move_sources()
        self.cmp_kk_swap_active = False
        self._cmp_auto_assign_channels(preserve_existing=False, allow_inference=True)
        self._on_cmp_plot_param_changed()

    def _on_cmp_rotation_mapping_changed(self) -> None:
        settings = getattr(self, "settings", None)
        if settings is not None:
            settings.setValue("compare/rotation_mapping", self.cmp_rotation_mapping_combo.currentData())
        self._on_cmp_angle_reference_changed()

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
        if source in tuple(getattr(self, "cmp_channel_combos", {}).values()):
            complete = True
            try:
                self._cmp_selection_from_ui()
            except (ValueError, OSError):
                complete = False
                invalidate = getattr(self, "_invalidate_active_load", None)
                if callable(invalidate):
                    invalidate("Compare")
            if self.loaded and self.loaded.mode == "Compare":
                if complete:
                    self._start_load("Compare")
            elif complete:
                self._cmp_maybe_auto_load()
            return
        if self.loaded and self.loaded.mode == "Compare":
            sender = source
            if sender in (
                self.cmp_spins["xmin"], self.cmp_spins["xmax"],
                self.cmp_spins["ymin"], self.cmp_spins["ymax"],
                self.cmp_log_chk, self.cmp_vp_background_spin,
                self.cmp_vp_auto_background_chk,
            ) or sender in tuple(self.cmp_channel_combos.values()) or sender in tuple(self.cmp_show_checks.values()):
                self._pending_range_refresh["Compare"] = (
                    bool(self._pending_range_refresh.get("Compare", False))
                    or sender in (self.cmp_spins["xmin"], self.cmp_spins["xmax"])
                )
            if sender is self.cmp_spins["gate"] and self._update_cmp_gate_only():
                return
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
