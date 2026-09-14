"""Thread-safe, non-Qt export and history helpers for MCD Peak Shift."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import tempfile
from typing import Any, Mapping

import numpy as np

from core.mcd_peak_shift import BOUNDARY_UNRELIABLE, valley_quantities
from core.mcd_valley_split import compute_valley_splitting


HISTORY_DIR = Path("Processed Data") / "MCD Peak Shift"
HISTORY_INDEX_NAME = ".peak_shift_history.json"


def source_descriptor(
    folder: str | Path | None,
    source_file: str | Path | None,
    *,
    include_hash: bool = False,
) -> dict[str, Any]:
    """Describe one source strongly enough to distinguish same-name files."""
    raw = str(source_file or "")
    path = Path(raw)
    if not path.is_absolute() and folder:
        path = Path(folder) / path
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    descriptor: dict[str, Any] = {
        "name": raw,
        "filename": resolved.name,
        "path": str(resolved),
    }
    return descriptor


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp = Path(raw_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def record_peak_shift_history(
    csv_path: str | Path,
    *,
    experiment_folder: str | Path | None,
    source: Mapping[str, Any],
    settings: Mapping[str, Any] | None = None,
) -> Path:
    """Write a CSV sidecar and, when possible, a small experiment index."""
    csv_file = Path(csv_path).resolve()
    created = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": 1,
        "workflow": "MCD Peak Shift",
        "dataset_type": "mcd_peak_shift",
        "created_utc": created,
        "source_descriptor": dict(source),
        "processing": dict(settings or {}),
        "output": csv_file.name,
    }
    sidecar = csv_file.with_suffix(".metadata.json")
    _atomic_json(sidecar, payload)
    if experiment_folder:
        root = Path(experiment_folder).expanduser().resolve() / HISTORY_DIR
        index = root / HISTORY_INDEX_NAME
        entries: list[dict[str, Any]] = []
        if index.is_file():
            try:
                value = json.loads(index.read_text(encoding="utf-8"))
                if isinstance(value, dict) and isinstance(value.get("exports"), list):
                    entries = [item for item in value["exports"] if isinstance(item, dict)]
            except (OSError, UnicodeError, json.JSONDecodeError):
                entries = []
        record = {
            "created_utc": created,
            "csv_path": str(csv_file),
            "sidecar_path": str(sidecar),
            "source_descriptor": dict(source),
            "processing": dict(settings or {}),
        }
        entries = [item for item in entries if str(item.get("csv_path", "")) != str(csv_file)]
        entries.append(record)
        entries.sort(key=lambda item: str(item.get("created_utc", "")), reverse=True)
        _atomic_json(index, {"schema_version": 1, "workflow": "MCD Peak Shift", "exports": entries})
    return sidecar


def _history_entries(experiment_folder: str | Path | None) -> list[dict[str, Any]] | None:
    if not experiment_folder:
        return []
    index = Path(experiment_folder).expanduser().resolve() / HISTORY_DIR / HISTORY_INDEX_NAME
    if not index.is_file():
        return []
    try:
        value = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("workflow") != "MCD Peak Shift"
        or not isinstance(value.get("exports"), list)
    ):
        return None
    return [item for item in value["exports"] if isinstance(item, dict)]


def peak_shift_history_status(
    experiment_folder: str | Path | None,
    source: Mapping[str, Any],
) -> tuple[str, str | None]:
    """Return ``processed``, ``unknown`` or ``new`` for one exact source."""
    expected_path = str(source.get("path", "")).casefold()
    expected_name = str(source.get("filename", source.get("name", ""))).casefold()
    unknown = False
    newest: str | None = None
    entries = _history_entries(experiment_folder)
    if entries is None:
        return "unknown", None
    for entry in entries:
        descriptor = entry.get("source_descriptor")
        if not isinstance(descriptor, dict):
            # Legacy history without a descriptor is never promoted to processed.
            legacy = str(entry.get("source_file", entry.get("source", "")))
            if Path(legacy).name.casefold() == expected_name:
                unknown = True
            continue
        path_match = str(descriptor.get("path", "")).casefold() == expected_path
        descriptor_path = str(descriptor.get("path", "")).strip()
        descriptor_name = str(
            descriptor.get("filename", descriptor.get("name", ""))
        )
        exact = bool(expected_path) and path_match
        if exact:
            stamp = str(entry.get("created_utc", ""))
            if newest is None or stamp > newest:
                newest = stamp
        elif not descriptor_path and Path(descriptor_name).name.casefold() == expected_name:
            # Structured records without a path are still legacy/insufficient
            # identity.  A complete record for another path is a different
            # source and therefore remains New, even when its basename matches.
            unknown = True
    if newest is not None:
        return "processed", newest
    return ("unknown", None) if unknown else ("new", None)


def export_peak_shift_csv(snapshot: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """Write the existing Peak Shift CSV from an immutable GUI snapshot."""
    result = snapshot["result"]
    method_results = snapshot.get("method_results", {})
    tracker_method = str(snapshot.get("tracker_method", ""))
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "peak_id", "feature_kind", "B_T", "branch", "E_peak_eV", "delta_E_eV", "status",
            "reference_method", "reference_field_T", "selected_K_peak_id",
            "selected_Kp_peak_id", "E_K_eV", "E_Kp_eV", "delta_E_K_eV",
            "delta_E_Kp_eV", "delta_E_Kp_minus_K_eV", "average_E_eV",
            "odd_average_E_eV", "even_average_E_eV", "odd_splitting_eV",
            "even_splitting_eV", "channel", "fit_source", "fit_window_low_eV",
            "fit_window_high_eV", "background_model", "locator_energy_eV",
            "model_name", "valley_status",
        ])
        if tracker_method == "Local mixed fit":
            selected = snapshot.get("selected")
            if selected is None or len(selected) != 4:
                raise ValueError("Local fit export needs a selected resonance.")
            channel, peak_id, branch, feature_kind = selected
            local_analysis = method_results.get("Local mixed fit", {})
            selected_analysis = local_analysis.get(str(channel))
            local_track = next(
                (track for track in selected_analysis.tracks
                 if track.peak_id == int(peak_id)
                 and track.branch == str(branch)
                 and track.feature_kind == str(feature_kind)),
                None,
            ) if selected_analysis is not None else None
            target_energy = local_track.reference_energy_ev if local_track is not None else None
            if target_energy is None:
                raise ValueError("Local fit export needs a fitted E0 reference.")
            branch_names = [
                str(value)
                for value in dict.fromkeys(
                    np.asarray(
                        snapshot.get("source_branch_labels", getattr(result, "branches", ())),
                        str,
                    ).tolist()
                )
            ]
            splits = {
                direction: compute_valley_splitting(
                    method_results, method="Local mixed fit", selected_channel=str(channel),
                    branch=direction, target_energy_ev=float(target_energy), tolerance_ev=0.005,
                    feature_kind=str(feature_kind), allow_energy_fallback=False,
                )
                for direction in branch_names
            }
            fixed_k = next((item.k_channel for item in splits.values() if item.k_channel in {"pos", "neg"}), None)
            if fixed_k in {"pos", "neg"}:
                splits = {
                    direction: compute_valley_splitting(
                        method_results, method="Local mixed fit", selected_channel=str(channel),
                        branch=direction, target_energy_ev=float(target_energy), tolerance_ev=0.005,
                        fixed_k_channel=fixed_k, feature_kind=str(feature_kind), allow_energy_fallback=False,
                    )
                    for direction in branch_names
                }
            first_analysis = next(iter(local_analysis.values()), None)
            fit_window = getattr(first_analysis, "fit_window_ev", None)
            window_low = fit_window[0] if fit_window else None
            window_high = fit_window[1] if fit_window else None
            background_model = str(snapshot.get("background_model", "linear"))
            model_name = getattr(first_analysis, "model_name", None)
            for local_channel, analysis in local_analysis.items():
                for track in analysis.tracks:
                    for point in track.points:
                        split = splits.get(str(point.branch))
                        split_point = next(
                            (item for item in split.points if abs(float(item.field_t) - float(point.field_t)) <= 1e-12),
                            None,
                        ) if split is not None else None
                        writer.writerow([
                            track.peak_id, track.feature_kind, point.field_t, point.branch,
                            point.energy_ev, point.delta_energy_ev, point.status,
                            track.reference_method, track.reference_field_t,
                            snapshot.get("selected_k"), snapshot.get("selected_kp"),
                            None, None, None, None,
                            None if split_point is None else split_point.splitting_ev,
                            None, None, None, None, None,
                            local_channel, snapshot.get("spectrum_source", ""), window_low,
                            window_high, background_model, track.locator_energy_ev,
                            track.model_name or model_name,
                            None if split is None else split.status,
                        ])
        else:
            visible_result = replace(result, tracks=tuple(track for track in result.tracks if track.quality != BOUNDARY_UNRELIABLE))
            selected_ids = (snapshot.get("selected_k"), snapshot.get("selected_kp"))
            valleys = {
                (round(float(value["B_T"]), 9), str(value["branch"])): value
                for value in valley_quantities(visible_result, selected_ids)
            } if all(value is not None for value in selected_ids) else {}
            for track in result.tracks:
                if track.quality == BOUNDARY_UNRELIABLE:
                    continue
                for point in track.points:
                    valley = valleys.get((round(point.field_t, 9), point.branch), {}) if track.feature_kind == "peak" else {}
                    writer.writerow([
                        track.peak_id, track.feature_kind, point.field_t, point.branch, point.energy_ev,
                        point.delta_energy_ev, point.status, track.reference_method,
                        track.reference_field_t, snapshot.get("selected_k"),
                        snapshot.get("selected_kp"), valley.get("E_K"), valley.get("E_Kp"),
                        valley.get("delta_E_K"), valley.get("delta_E_Kp"),
                        valley.get("splitting_E_Kp_minus_E_K"), valley.get("average_E"),
                        valley.get("odd_average_E"), valley.get("even_average_E"),
                        valley.get("odd_splitting"), valley.get("even_splitting"),
                        "", "", "", "", "", "", "", "",
                    ])
    return {"csv_path": str(out_path)}


def mcd_peak_shift_export_worker(
    snapshot: Mapping[str, Any], path: str | Path, *, progress=None, log=None
) -> dict[str, Any]:
    """Worker entry point; it touches only the supplied snapshot and files."""
    persisted_source = dict(snapshot.get("source_descriptor", {}))
    result = export_peak_shift_csv(snapshot, path)
    sidecar = record_peak_shift_history(
        path,
        experiment_folder=snapshot.get("experiment_folder"),
        source=persisted_source,
        settings=snapshot.get("history_settings", {}),
    )
    result["sidecar_path"] = str(sidecar)
    if log is not None:
        log.emit(f"Exported MCD Peak Shift CSV: {Path(path).name}")
    if progress is not None:
        progress.emit(100)
    return result
