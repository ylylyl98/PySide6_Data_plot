"""Stable source packages and transactional replacement for native Peak Shift."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from threading import RLock

import numpy as np

from core.load_cancellation import check_load_cancelled
from core.mcd_peak_export import (
    HISTORY_DIR, HISTORY_INDEX_NAME, _atomic_json, export_peak_shift_csv, peak_shift_history_lock,
)

_SAVE_LOCK = RLock()


def _digest(value, length=8):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:length]


def _safe(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-") or "PeakShift"


def _source_identity(descriptor):
    path = descriptor.get("path")
    if not path:
        raise ValueError("Peak Shift saving requires the original source path.")
    return str(Path(path).expanduser().resolve()).replace("\\", "/").casefold()


def _root(output):
    root = Path(output).expanduser().resolve()
    if root.name.casefold() == "mcd peak shift":
        return root
    if root.name.casefold() == "processed data":
        return root / "MCD Peak Shift"
    return root / HISTORY_DIR


def _source_package(root, descriptor):
    stem = Path(descriptor.get("filename") or descriptor["path"]).stem
    stem = re.sub(r"(?:_MCD|_PeakShift)$", "", stem, flags=re.IGNORECASE)
    prefix = _safe(stem)
    if len(prefix) > 120:
        prefix = prefix[:110].rstrip("._-") + "_" + hashlib.sha256(prefix.encode()).hexdigest()[:8]
    budget = max(24, 128 - len(str(root)))
    if len(prefix) > budget:
        prefix = prefix[:max(15, budget-9)].rstrip("._-") + "_" + hashlib.sha256(prefix.encode()).hexdigest()[:8]
    identity = _source_identity(descriptor)
    for suffix in ("", "_"+_digest(identity, 10)):
        package = root / (prefix+suffix+"_PeakShift")
        marker = package / ".source.json"
        if not package.exists() or not any(package.iterdir()):
            return package, identity
        try:
            if json.loads(marker.read_text(encoding="utf-8")).get("source_identity") == identity:
                return package, identity
        except (OSError, ValueError, AttributeError):
            pass
    raise ValueError("The Peak Shift source folder contains results with an unverified source identity.")


def _peak_locations(snapshot, channel, peak_id, kind="peak", branch=None):
    analyses = snapshot.get("method_results", {}).get(snapshot.get("tracker_method"), {})
    analysis = analyses.get(channel) or (snapshot.get("result") if channel == "pos" else None)
    tracks = [track for track in getattr(analysis, "tracks", ())
              if track.peak_id == int(peak_id) and track.feature_kind == kind
              and (branch is None or track.branch == branch)]
    locations = []
    for track in tracks:
        energy = track.locator_energy_ev
        if energy is None:
            energy = track.reference_energy_ev
        if energy is None:
            measured = [point.energy_ev for point in track.points
                        if point.status == "tracked" and point.energy_ev is not None and np.isfinite(point.energy_ev)]
            energy = float(np.median(measured)) if measured else None
        if energy is None or not np.isfinite(energy):
            raise ValueError("The selected peak has no measured location for a stable save identity.")
        locations.append(dict(branch=str(track.branch), energy_ev=round(float(energy), 6)))
    if not locations:
        raise ValueError("The selected peak is missing from the completed analysis.")
    return sorted(locations, key=lambda row: (row["branch"], row["energy_ev"]))


def analysis_identity(snapshot):
    settings = snapshot.get("history_settings", {})
    selected = snapshot.get("selected")
    feature = None
    if selected is not None:
        if len(selected) != 4:
            raise ValueError("Peak Shift saving requires a channel, peak, branch and feature kind.")
        channel, peak, branch, kind = selected
        feature = dict(channel=str(channel), kind=str(kind),
            locations=_peak_locations(snapshot, str(channel), peak, str(kind), str(branch)))
    mode = str(settings.get("resultMode", "Valley splitting"))
    return dict(feature=feature, method=str(snapshot.get("tracker_method", "Raw spectrum")),
        spectrum_source=str(snapshot.get("spectrum_source", "Raw R")), result_type=mode,
        display=str(settings.get("peakDisplay", "Delta E")) if mode == "Single peak shift" else None,
        selected_k=_peak_locations(snapshot, "pos", snapshot["selected_k"]) if snapshot.get("selected_k") is not None else None,
        selected_kp=_peak_locations(snapshot, "pos", snapshot["selected_kp"]) if snapshot.get("selected_kp") is not None else None)


def _analysis_label(identity):
    feature = identity["feature"]
    location = feature["locations"][0] if feature else None
    parts = ["AllPeaks"] if feature is None else [feature["channel"],
        ("P" if feature["kind"] == "peak" else "D")+f"_E{location['energy_ev']:.6f}eV",
        {"B increasing": "inc", "B decreasing": "dec"}.get(location["branch"], location["branch"])]
    parts += [{"Raw spectrum": "Raw", "Second derivative": "Derivative", "Local mixed fit": "LocalFit"}.get(identity["method"], identity["method"]),
              "CorrectedR" if identity["spectrum_source"] == "MCD-corrected R" else "RawR",
              "ValleySplitting" if identity["result_type"] == "Valley splitting" else "PeakShift"]
    if identity["display"]:
        parts.append("AbsoluteE" if identity["display"] == "Absolute E" else "DeltaE")
    for key, prefix in (("selected_k", "K"), ("selected_kp", "Kp")):
        if identity[key] is not None:
            parts.append(prefix+f"_E{identity[key][0]['energy_ev']:.6f}eV")
    return "_".join(_safe(part) for part in parts)[:76].rstrip("._-")+"_"+_digest(identity)


@contextmanager
def _output_lock(root):
    with _SAVE_LOCK:
        root.mkdir(parents=True, exist_ok=True)
        lock = root / ".peak-shift-export.lock"
        try:
            with lock.open("x"):
                pass
        except FileExistsError:
            raise RuntimeError("Another MCD Peak Shift save is using this output folder.") from None
        try:
            yield
        finally:
            lock.unlink(missing_ok=True)


def _history_document(index, destination, metadata):
    entries = []
    if index.is_file():
        try:
            previous = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("Peak Shift history could not be read; the previous results were preserved.") from error
        if not isinstance(previous, dict) or previous.get("workflow") != "MCD Peak Shift" or not isinstance(previous.get("exports"), list):
            raise ValueError("Peak Shift history is invalid; the previous results were preserved.")
        entries = [item for item in previous["exports"] if isinstance(item, dict)]
    entries = [item for item in entries if str(item.get("csv_path", "")).casefold() != str(destination).casefold()]
    entries.append(dict(created_utc=metadata["created_utc"], csv_path=str(destination),
        sidecar_path=str(destination.with_suffix(".metadata.json")), source_descriptor=metadata["source_descriptor"],
        processing=metadata["processing"], analysis_identity=metadata["analysis_identity"]))
    entries.sort(key=lambda item: str(item.get("created_utc", "")), reverse=True)
    return dict(schema_version=1, workflow="MCD Peak Shift", exports=entries)


def export_peak_shift_analysis(snapshot, output_root):
    """Save one analysis identity, preserving the previous pair on failure."""
    descriptor = dict(snapshot["source_descriptor"])
    identity = analysis_identity(snapshot)
    label = _analysis_label(identity)
    root = _root(output_root)
    with _output_lock(root), peak_shift_history_lock(snapshot.get("experiment_folder")):
        package, source_id = _source_package(root, descriptor)
        package.mkdir(parents=True, exist_ok=True)
        destination = package / (label+".csv")
        sidecar = destination.with_suffix(".metadata.json")
        updated = destination.exists()
        if destination.exists() or sidecar.exists():
            try:
                old = json.loads(sidecar.read_text(encoding="utf-8"))
                verified = (_source_identity(old["source_descriptor"]) == source_id
                            and old["analysis_identity"] == identity)
            except (OSError, ValueError, KeyError, TypeError):
                verified = False
            if not verified:
                raise ValueError("Existing Peak Shift files have an unverified analysis identity; they were preserved.")
        metadata = dict(schema_version=1, workflow="MCD Peak Shift", dataset_type="mcd_peak_shift",
            created_utc=datetime.now(timezone.utc).isoformat(), source_descriptor=descriptor,
            processing=dict(snapshot.get("history_settings", {})), output=destination.name,
            analysis_identity=identity, analysis_id=label,
            selection=dict(feature=list(snapshot["selected"]) if snapshot.get("selected") else None,
                           selected_k=snapshot.get("selected_k"), selected_kp=snapshot.get("selected_kp")))
        folder = snapshot.get("experiment_folder")
        index = Path(folder).expanduser().resolve()/HISTORY_DIR/HISTORY_INDEX_NAME if folder else None
        history = _history_document(index, destination, metadata) if index else None
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=package))
        published, backups = [], {}
        recovery_failed = False
        try:
            export_peak_shift_csv(snapshot, staging/destination.name)
            _atomic_json(staging/sidecar.name, metadata)
            marker = package/".source.json"
            targets = [destination, sidecar]
            if not marker.exists():
                _atomic_json(staging/marker.name, dict(source_identity=source_id, source_descriptor=descriptor))
                targets.append(marker)
            for number, target in enumerate(targets):
                if target.exists():
                    backup = staging/str(number)
                    shutil.copy2(target, backup)
                    backups[target] = backup
            check_load_cancelled()
            for target in targets:
                (staging/target.name).replace(target)
                published.append(target)
            if index:
                _atomic_json(index, history)
        except Exception as error:
            for target in reversed(published):
                try:
                    if target in backups:
                        backups[target].replace(target)
                    else:
                        target.unlink(missing_ok=True)
                except OSError:
                    recovery_failed = True
            if recovery_failed:
                raise RuntimeError(f"Peak Shift save failed; recovery copies are retained at {staging}") from error
            raise
        finally:
            if not recovery_failed and staging.resolve().parent == package.resolve():
                shutil.rmtree(staging)
    return dict(csv_path=str(destination), sidecar_path=str(sidecar), package_dir=str(package),
                analysis_id=label, updated=updated)
