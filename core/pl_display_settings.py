"""Portable, exact-file PL display history, independent of local manual locks."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
from threading import RLock
from uuid import uuid4

VERSION = 1
INDEX_NAME = ".display_settings.json"
_LOCK = RLock()
_AXES = ("xmin", "xmax", "ymin", "ymax")
_COLORS = ("vmin", "vmax", "leftMin", "leftMax", "middleMin", "middleMax", "rightMin", "rightMax")
_CONTEXT = ("y_axis", "dat_choice", "dat_label", "dat_unit")


def _file_digest(path):
    from core.load_cancellation import check_load_cancelled
    digest = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            check_load_cancelled()
            digest.update(chunk)
    return digest.hexdigest()


def _relative(value):
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ":" in text or ".." in path.parts:
        raise ValueError("PL history requires an experiment-relative source path.")
    return path.as_posix()


def source_key(folder, source):
    root = Path(folder).resolve()
    path = Path(source)
    if path.is_absolute():
        path = path.resolve().relative_to(root)
    return _relative(path.as_posix()).casefold()


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("PL display bounds must be finite numbers.")
    return float(value)


def _boolean(value):
    if type(value) is not bool:
        raise ValueError("PL display flags must be Boolean.")
    return value


def _flags(value, names):
    if not isinstance(value, dict) or any(key not in names for key in value):
        raise ValueError("Invalid PL display Fix fields.")
    return {key: _boolean(value.get(key, False)) for key in names}


def _context(value):
    if not isinstance(value, dict) or any(key not in _CONTEXT for key in value):
        raise ValueError("Invalid PL axis context.")
    defaults = dict(y_axis="auto", dat_choice="original", dat_label="", dat_unit="")
    if any(not isinstance(item, str) for item in value.values()):
        raise ValueError("Invalid PL axis context values.")
    return {**defaults, **value}


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("Invalid PL source or axis digest.")
    return value


def validate_snapshot(snapshot):
    """Return a normalized independent copy, or None without partial acceptance."""
    try:
        if not isinstance(snapshot, dict) or type(snapshot.get("version")) is not int or snapshot["version"] != VERSION:
            return None
        values = snapshot["values"]
        if not isinstance(values, dict):
            return None
        numeric = {key: _number(values[key]) for key in (*_AXES, "gate")}
        if numeric["xmin"] >= numeric["xmax"] or numeric["ymin"] >= numeric["ymax"]:
            return None
        if values["scale"] not in ("linear", "log") or not isinstance(values["cmap"], str):
            return None
        from matplotlib import colormaps
        if values["cmap"] not in colormaps:
            return None
        normalized = dict(version=VERSION, values={**numeric, "scale": values["scale"], "cmap": values["cmap"],
            "sideBySide": _boolean(values["sideBySide"]), "clip": _boolean(values.get("clip", False))},
            axis_fixed=_flags(snapshot.get("axis_fixed", {}), _AXES), axis_context=_context(snapshot.get("axis_context", {})))
        geometry = snapshot["geometry"]
        if not isinstance(geometry, dict) or type(geometry["count"]) is not int or geometry["count"] not in (1, 2, 3):
            return None
        x0, x1 = _number(geometry["x0"]), _number(geometry["x1"])
        if geometry["count"] == 3 and x0 >= x1:
            return None
        normalized["geometry"] = dict(count=geometry["count"], x0=x0, x1=x1,
            showBoundary=_boolean(geometry["showBoundary"]), fixed=_flags(geometry.get("fixed", {}), ("x0", "x1")))
        normalized["banks"] = {}
        for scale in ("linear", "log"):
            bank = snapshot["banks"][scale]
            if not isinstance(bank, dict) or _boolean(bank["log"]) != (scale == "log"):
                return None
            result = {field: _number(bank[field]) for field in _COLORS}
            for low, high in zip(_COLORS[::2], _COLORS[1::2]):
                if result[low] >= result[high] or scale == "log" and result[low] <= 0:
                    return None
            result.update(log=scale == "log", fixed=_flags(bank.get("fixed", {}), _COLORS), roi=[])
            normalized["banks"][scale] = result
        if "coordinates" in snapshot:
            normalized["coordinates"] = {key: _digest(snapshot["coordinates"][key]) for key in ("x", "y")}
        if "axis_signature" in snapshot:
            signature = snapshot["axis_signature"]
            if not isinstance(signature, dict) or set(signature) != {"label", "unit", "semantic"} or any(not isinstance(value, str) for value in signature.values()):
                return None
            normalized["axis_signature"] = dict(signature)
        if "source" in snapshot:
            source = snapshot["source"]
            normalized["source"] = dict(relative_path=_relative(source["relative_path"]), sha256=_digest(source["sha256"]))
            dependencies = source.get("dependencies", [])
            if not isinstance(dependencies, list):
                return None
            normalized["source"]["dependencies"] = [dict(relative_path=_relative(item["relative_path"]),
                sha256=_digest(item["sha256"]) if item.get("sha256") is not None else None) for item in dependencies]
        return normalized
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def source_context(folder, source, cube, request=None):
    """Capture loaded revisions and coordinate digests; never hash intensity maps."""
    import numpy as np
    root = Path(folder).resolve()
    path = Path(source)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    relative = _relative(path.relative_to(root).as_posix())
    signatures = getattr(cube, "_quick_pl_signatures", ())
    by_path = {str(Path(item[0]).resolve()).casefold(): item for item in signatures}
    signature = by_path.get(str(path).casefold())
    digest = signature[-1] if signature and len(signature) > 2 else _file_digest(path)
    dependencies = []
    for item in signatures:
        dependency = Path(item[0]).resolve()
        if dependency == path:
            continue
        dependencies.append(dict(relative_path=_relative(dependency.relative_to(root).as_posix()),
            sha256=item[-1] if len(item) > 2 else None))
    axis = dict(y_axis=getattr(request, "y_axis", "auto"), dat_choice=getattr(request, "dat_choice", "original"),
        dat_label=getattr(request, "dat_label", ""), dat_unit=getattr(request, "dat_unit", ""))
    return dict(source=dict(relative_path=relative, sha256=digest, dependencies=dependencies),
        coordinates={key: sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()
            for key, value in (("x", cube.energy), ("y", cube.gate))}, axis_context=axis,
        axis_signature=dict(label=str(getattr(cube, "gate_label", "")), unit=str(getattr(cube, "gate_unit", "")),
            semantic=str(getattr(cube, "y_axis_semantic", ""))))


def _read(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, ValueError):
        return {}


def _write(path, payload):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_snapshot(folder, source, snapshot, context=None, *, publication=None):
    """Publish a successful export's display snapshot, atomically and exactly."""
    candidate = {**deepcopy(snapshot), **deepcopy(context or {})}
    valid = validate_snapshot(candidate)
    if valid is None or "source" not in valid or "coordinates" not in valid:
        raise ValueError("Invalid PL display snapshot or missing source context.")
    key = source_key(folder, source)
    if source_key(folder, valid["source"]["relative_path"]) != key:
        raise ValueError("PL display snapshot source does not match the selected measurement.")
    path = Path(folder) / "Processed Data" / "PL" / INDEX_NAME
    with _LOCK:
        index = _read(path)
        if index and (type(index.get("version")) is not int or index["version"] != VERSION):
            raise ValueError("Unsupported PL display history version.")
        entries = index.get("files", {})
        if not isinstance(entries, dict):
            entries = {}
        entries[key] = {**valid, **({"publication": deepcopy(publication)} if publication else {})}
        path.parent.mkdir(parents=True, exist_ok=True)
        _write(path, dict(version=VERSION, files=entries))
    return path


def _snapshot_digest(snapshot):
    return sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def publish_snapshot(folder, source, snapshot, data_path):
    """Publish history only when its exact export metadata completes successfully.

    The staged index entry is gated by its metadata even across process failure.
    Ordinary completion errors restore the previous index; an interrupted stage
    can still recover its previous completed metadata through legacy lookup.
    """
    root = Path(folder).resolve()
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = root / data_path
    metadata = data_path.with_suffix(".metadata.json")
    valid = validate_snapshot(snapshot)
    if valid is None:
        raise ValueError("Invalid PL display publication snapshot.")
    publication = dict(metadata_path=_relative(metadata.resolve().relative_to(root).as_posix()),
        snapshot_sha256=_snapshot_digest(valid))
    index_path = root / "Processed Data" / "PL" / INDEX_NAME
    with _LOCK:
        existed, previous = index_path.exists(), _read(index_path)
        staged = False
        try:
            save_snapshot(root, source, snapshot, publication=publication)
            staged = True
            complete_metadata_snapshot(data_path)
            if not _publication_complete(root, {**valid, "publication": publication}):
                raise ValueError("PL display export metadata did not complete successfully.")
        except Exception:
            if staged:
                try:
                    if existed:
                        _write(index_path, previous)
                    else:
                        index_path.unlink(missing_ok=True)
                except OSError:
                    # A staged record stays unreadable while its metadata is
                    # pending; completed previous metadata remains recoverable.
                    pass
            raise
    return index_path


def _publication_complete(folder, raw):
    if "publication" not in raw:
        return True
    try:
        publication = raw["publication"]
        if not isinstance(publication, dict):
            return False
        metadata_path = Path(folder) / _relative(publication["metadata_path"])
        payload = _read(metadata_path)
        if payload.get("pl_display_save_pending"):
            return False
        expected = validate_snapshot(raw)
        saved = validate_snapshot(payload.get("pl_display"))
        return (expected is not None and saved == expected
            and _digest(publication["snapshot_sha256"]) == _snapshot_digest(expected))
    except (KeyError, ValueError, TypeError):
        return False


def complete_metadata_snapshot(data_path):
    data_path = Path(data_path)
    path = data_path.with_suffix(".metadata.json")
    with _LOCK:
        payload = _read(path)
        if payload.pop("pl_display_save_pending", False):
            payload["pl_display_output"] = dict(filename=data_path.name, sha256=_file_digest(data_path))
            _write(path, payload)


def _own_dat_snapshot(folder, source, cube, request):
    """Bind an exported DAT to its own sidecar, never to a matching basename."""
    path = Path(source)
    if not path.is_absolute():
        path = Path(folder) / path
    if path.suffix.casefold() != ".dat":
        return None
    payload = _read(path.with_suffix(".metadata.json"))
    if str(payload.get("workflow", payload.get("operation", ""))).casefold() != "pl" or payload.get("pl_display_save_pending"):
        return None
    output = payload.get("pl_display_output")
    manifest = payload.get("output_manifest")
    if (not isinstance(output, dict) or output.get("filename") != path.name or not isinstance(manifest, list)
            or not any(isinstance(item, dict) and item.get("role") == "data" and item.get("filename") == path.name for item in manifest)):
        return None
    snapshot = validate_snapshot(payload.get("pl_display"))
    if snapshot is None:
        return None
    try:
        if _digest(output.get("sha256")) != _file_digest(path):
            return None
        if cube is not None:
            # The verified exported matrix has its own serialized coordinates
            # and DAT semantics. Original acquisition derivation is provenance,
            # not the meaning of the already-exported Y coordinate column.
            snapshot.update(source_context(folder, path, cube, request))
        else:
            snapshot["source"] = dict(relative_path=_relative(path.resolve().relative_to(Path(folder).resolve()).as_posix()),
                sha256=output["sha256"], dependencies=[])
        return snapshot
    except (OSError, ValueError):
        return None


def _legacy_snapshot(payload):
    """Convert legacy PL export plots; absent Fix flags remain automatic."""
    if isinstance(payload.get("pl_display"), dict):
        return validate_snapshot(payload["pl_display"])
    try:
        plots = payload["plot"]
        if not isinstance(plots, dict):
            return None
        scale = payload.get("active_scale", "linear")
        if scale not in ("linear", "log"):
            scale = "linear"
        active = plots[scale]
        if not isinstance(active, dict):
            return None
        xmin, xmax = active["xlim"]
        ymin, ymax = active["ylim"]
        split = active.get("split_scale") or {}
        if not isinstance(split, dict):
            return None
        count = 3 if split.get("split_x2") is not None else 2 if split else 1
        width = xmax - xmin
        geometry = dict(count=count, x0=split.get("split_x", xmin + width / 3),
            x1=split.get("split_x2") if split.get("split_x2") is not None else xmin + width * 2 / 3,
            showBoundary=split.get("show_boundary", True), fixed={})
        banks = {}
        for name in ("linear", "log"):
            plot = plots[name]
            if not isinstance(plot, dict):
                return None
            region = plot.get("split_scale") or {}
            if not isinstance(region, dict):
                return None
            bank = dict(log=name == "log", vmin=plot["vmin"], vmax=plot["vmax"], fixed={})
            for side in ("left", "middle", "right"):
                for bound in ("min", "max"):
                    value = region.get(side + "_v" + bound)
                    bank[side + bound.capitalize()] = value if value is not None else plot["v" + bound]
            banks[name] = bank
        state = dict(version=VERSION, values=dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            gate=min(max(0., ymin), ymax), scale=scale, sideBySide=False, cmap=active.get("cmap", "turbo"),
            clip=active.get("clip_outliers", False)), geometry=geometry, axis_fixed={}, banks=banks,
            axis_context=payload.get("native_pl_axis", {}))
        return validate_snapshot(state)
    except (KeyError, TypeError, ValueError):
        return None


def _legacy_candidates(folder, source):
    from core.pl_sources import list_pl_source_files
    from core.source_identity import match_source_identity
    root = Path(folder)
    key = source_key(folder, source)
    names = list_pl_source_files(str(root))
    target = next((name for name in names if source_key(folder, name) == key), None)
    if target is None:
        return []
    candidates = []
    for path in sorted((root / "Processed Data" / "PL").rglob("*.metadata.json")):
        payload = _read(path)
        if str(payload.get("workflow", payload.get("operation", ""))).casefold() != "pl" or payload.get("pl_display_save_pending"):
            continue
        snapshot = _legacy_snapshot(payload)
        if snapshot is None:
            continue
        if snapshot.get("source"):
            matched = source_key(folder, snapshot["source"]["relative_path"]) == key
        else:
            matched = False
            descriptors = payload.get("sources", payload.get("inputs", []))
            for item in descriptors if isinstance(descriptors, list) else []:
                if not isinstance(item, dict) or item.get("role", "measurement") != "measurement":
                    continue
                match, _ = match_source_identity(root, names, relative_path=item.get("source_relative_path", ""),
                    path=item.get("source_path", item.get("path", "")),
                    legacy_name=item.get("name", item.get("source_file", item.get("filename", ""))))
                if match == target:
                    matched = True
        if matched:
            try:
                candidates.append((str(payload.get("created_utc", "")), path.stat().st_mtime_ns, path.name, snapshot))
            except OSError:
                continue
    return candidates


def find_snapshot(folder, source, cube=None, request=None):
    """Find exact-file history; a copied experiment does not need old absolute paths."""
    key = source_key(folder, source)
    index = _read(Path(folder) / "Processed Data" / "PL" / INDEX_NAME)
    snapshot = None
    if type(index.get("version")) is int and index["version"] == VERSION and isinstance(index.get("files"), dict):
        raw = index["files"].get(key)
        snapshot = validate_snapshot(raw) if isinstance(raw, dict) and _publication_complete(folder, raw) else None
        if snapshot and ("source" not in snapshot or source_key(folder, snapshot["source"]["relative_path"]) != key):
            snapshot = None
    if snapshot is None:
        snapshot = _own_dat_snapshot(folder, source, cube, request)
    if snapshot is None:
        candidates = _legacy_candidates(folder, source)
        if candidates:
            snapshot = max(candidates, key=lambda item: item[:3])[3]
    if snapshot is not None and cube is not None:
        current = source_context(folder, source, cube, request)
        coordinates = snapshot.get("coordinates", {})
        axis = (snapshot.get("axis_context", {}) == current["axis_context"]
            and snapshot.get("axis_signature", current["axis_signature"]) == current["axis_signature"])
        snapshot["compatibility"] = dict(x=coordinates.get("x", current["coordinates"]["x"]) == current["coordinates"]["x"],
            y=axis and coordinates.get("y", current["coordinates"]["y"]) == current["coordinates"]["y"],
            axis=axis, revision=snapshot.get("source") == current["source"])
    return snapshot
