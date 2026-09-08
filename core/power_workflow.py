"""Discovery, provenance and explicit intensity correction for combined sweeps."""
import hashlib
import json
import re
import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from core.loader import DataCube
from core.processing import PowerSweepPoint

COMBINED_FOLDER = Path("Processed Data") / "Power Dependence" / "Combined Sweeps"

ARCHIVE_FOLDERS = {'old', 'archive', 'archives', 'archived', 'initial data after processing',
                   '.git', '.venv', '__pycache__'}


def _active_files(folder, suffix):
    root = Path(folder)
    if not folder or not root.is_dir():
        return
    for directory, children, files in os.walk(root, followlinks=False):
        children[:] = [name for name in children if name.casefold() not in ARCHIVE_FOLDERS
                       and not (Path(directory) / name).is_symlink()
                       and not (Path(directory) / name).is_junction()]
        for name in files:
            if name.casefold().endswith(suffix):
                yield Path(directory) / name


@lru_cache(maxsize=2048)
def _sweep_header(path, size, modified):
    from core.data_io import _power_table_columns, processing_impl
    from core.processing import parse_power_series_file
    try:
        sep = processing_impl._guess_sep_from_first_line(Path(path))
        columns = pd.read_csv(path, sep=sep, nrows=0).columns
        power, _, spectra = _power_table_columns(columns)
        if power is not None:
            return 'table' if len(spectra) >= 2 else None
        return 'legacy' if parse_power_series_file(Path(path).name) is not None else None
    except (OSError, ValueError):
        return False


def discover_power_files(folder, *, include_legacy=False):
    """Header-only recursive discovery, excluding archives and analysis tables."""
    root = Path(folder)
    found = []
    legacy = []
    for path in _active_files(folder, '.csv'):
        try:
            stat = path.stat()
            relative = path.relative_to(root)
            kind = _sweep_header(str(path), stat.st_size, stat.st_mtime_ns)
            if not kind:
                continue
            if 'processed data' in {p.casefold() for p in relative.parts[:-1]} and not is_combined(folder, str(relative)):
                continue
            if kind == 'table':
                found.append(str(relative))
            elif include_legacy:
                legacy.append(str(relative))
        except OSError:
            continue
    from core.processing import group_power_series_files
    for records in group_power_series_files(legacy).values():
        if len({r.power_uW for r in records}) >= 2:
            found.extend(r.file_name for r in records)
    return sorted(set(found), key=str.casefold)


def record_sources(record):
    def walk(value, fallback):
        try:
            sources = json.loads(value).get("sources", [])
        except (ValueError, TypeError):
            sources = []
        if not sources:
            return {fallback}
        return set().union(*(walk(s.get("prior_provenance", ""), s.get("file", fallback))
                             if isinstance(s, dict) else {str(s)} for s in sources))
    return sorted(walk(getattr(record, "source_provenance", ""), record.file_name))


def combined_files(folder):
    return [name for name in discover_power_files(folder) if is_combined(folder, name)]


def processed_source_names(folder):
    """Original inputs with saved Power analysis, matching the PL status convention."""
    names = set()
    root = Path(folder).resolve()
    for path in _active_files(folder, '.metadata.json'):
        if 'Power Dependence' not in path.parts:
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            for source in obj.get("sources", []):
                if isinstance(source, dict) and source.get("name"):
                    raw = Path(source.get('source_path', source.get('path', '')))
                    if raw.is_absolute():
                        try:
                            names.add(raw.resolve().relative_to(root).as_posix().casefold())
                        except ValueError:
                            pass
                    else:
                        base = path
                        while base != base.parent and base.name != 'Processed Data':
                            base = base.parent
                        if base.name == 'Processed Data':
                            names.add((base.parent / source['name']).resolve().relative_to(root).as_posix().casefold())
        except (ValueError, OSError):
            continue
    return names


@lru_cache(maxsize=256)
def _combined_flag(path, size, modified):
    try:
        data = pd.read_csv(path, usecols=["source_provenance"], nrows=8)
        return any(isinstance(v, str) and '"sources"' in v for v in data.source_provenance)
    except (ValueError, OSError):
        return False


def is_combined(folder, filename):
    if not filename:
        return False
    p = Path(folder) / filename
    try:
        stat = p.stat()
        return _combined_flag(str(p), stat.st_size, stat.st_mtime_ns)
    except OSError:
        return False


def source_rows(result):
    def leaves(provenance, fallback):
        try:
            obj = json.loads(provenance)
        except (ValueError, TypeError):
            return {fallback}
        sources = obj.get("sources", [])
        if not sources:
            return {fallback}
        return set().union(*(leaves(s.get("prior_provenance", ""),
            (str(s.get("file", "")).replace('\\', '/').casefold(), s.get("row")))
            if isinstance(s, dict) else {(str(s).casefold(), None)} for s in sources))
    return set().union(*(leaves(getattr(r, "source_provenance", ""),
        (r.file_name.replace('\\', '/').casefold(), getattr(r, "row_index", None))) for r in result.records))


def acquisition_info(folder, result):
    names = {(Path(folder) / r.file_name).resolve() for r in result.records}
    found = []
    metadata_files = {p for directory in {p.parent for p in names} | {Path(folder)}
                      for p in directory.glob("*.experiment.metadata.json")}
    for path in sorted(metadata_files):
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            if any((path.parent / f['path']).resolve() in names for f in obj.get("files", []) if f.get('path')):
                settings = obj.get("settings", {}).get("requested", {})
                gains = {}
                def walk(value, prefix=""):
                    if isinstance(value, dict):
                        for key, val in value.items():
                            if "gain" in key.lower():
                                gains[prefix + key] = val
                            walk(val, prefix + key + ".")
                walk(obj)
                found.append({"exposure_ms": settings.get("exp_ms"), "frames": settings.get("frames"),
                              "gain": gains or "not recorded"})
        except (OSError, ValueError):
            continue
    return found or [{"gain": "not recorded", "exposure_ms": None, "frames": None}]


def suspicious_rows(folder, result):
    """Report telemetry discrepancies; never automatically remove data."""
    tables = {}
    flagged = []
    for index, record in enumerate(result.records):
        name = record.file_name
        if name not in tables:
            try:
                tables[name] = pd.read_csv(Path(folder) / name, usecols=["stage_pos", "stage_pos_actual"])
            except (OSError, ValueError):
                tables[name] = None
        table = tables[name]
        row = getattr(record, "row_index", None)
        if table is not None and row is not None and 0 <= row - 2 < len(table):
            wanted, actual = table.iloc[row - 2]
            if np.isfinite(wanted) and np.isfinite(actual) and abs(wanted - actual) > 2:
                flagged.append(index)
    return flagged


def corrected(result, *, background=0., factor=1., excluded=()):
    if not np.isfinite(factor) or factor <= 0 or not np.isfinite(background):
        raise ValueError("Use a positive correction factor and finite background.")
    keep = [i for i in range(len(result.records)) if i not in set(excluded)]
    if not keep:
        raise ValueError("Keep at least one measurement in each sweep.")
    cube = result.cube
    records = tuple(PowerSweepPoint(result.records[i].file_name, float(cube.gate[i]), result.records[i].stage,
        getattr(result.records[i], "row_index", None), source_provenance=json.dumps({
        "sources": [{"file": result.records[i].file_name, "row": getattr(result.records[i], "row_index", None),
                     "prior_provenance": getattr(result.records[i], "source_provenance", "")}],
        "intensity_correction": {"background": background, "factor": factor}})) for i in keep)
    return replace(result, cube=DataCube(cube.energy.copy(), cube.gate[keep].copy(),
        (cube.Z[keep] - background) * factor, cube.gate_label, cube.title, cube.cbar_label), records=records)


def estimate_background(result, energy_min, energy_max):
    """Robust constant baseline from an explicitly selected signal-free band."""
    mask = (result.cube.energy >= energy_min) & (result.cube.energy <= energy_max)
    values = result.cube.Z[:, mask]
    values = values[np.isfinite(values)]
    if energy_min >= energy_max or mask.sum() < 5 or len(values) < 5:
        raise ValueError("Background band must contain at least five spectral samples in each sweep.")
    return float(np.median(values))


def estimate_factor(first, second, *, _reverse=True):
    """Map B onto A. Interpolate in measured power only between nearby B samples."""
    a, b = first.cube, second.cube
    order = np.argsort(b.gate)
    power, unique = np.unique(b.gate[order], return_index=True)
    z = b.Z[order][unique]
    mask = (a.energy >= b.energy[0]) & (a.energy <= b.energy[-1])
    pairs = []
    samples = []
    for p, y in zip(a.gate, a.Z):
        if p < power[0] or p > power[-1]:
            continue
        j = int(np.searchsorted(power, p))
        if j < len(power) and power[j] == p:
            target = z[j]
        elif j > 0 and j < len(power) and power[j] / max(power[j - 1], 1e-30) <= 1.25:
            t = (p - power[j - 1]) / (power[j] - power[j - 1])
            target = (1 - t) * z[j - 1] + t * z[j]
        else:
            continue
        x = np.interp(a.energy[mask], b.energy, target)
        y = y[mask]
        good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if np.any(good):
            good &= (x > .1 * np.nanmax(x[good])) & (y > .1 * np.nanmax(y[good]))
        if good.sum() >= 8:
            x, y = x[good], y[good]
            factor = float(np.dot(x, y) / np.dot(x, x))
            residual = float(np.linalg.norm(y - factor * x) / np.linalg.norm(y))
            pairs.append((float(p), factor, residual))
            samples.append((x, y))
    if not pairs:
        if _reverse:
            inverse, quality = estimate_factor(second, first, _reverse=False)
            quality['pairs'] = [(p, 1 / factor, residual) for (p, factor, _), residual in
                                zip(quality['pairs'], quality.pop('reverse_residuals'))]
            quality['max_relative_residual'] = max(r for _, _, r in quality['pairs'])
            quality['accepted'] = quality['relative_scatter'] <= .1 and quality['max_relative_residual'] <= .1
            quality['provisional'] = quality['provisional'] or not quality['accepted']
            quality['interpolation'] = 'first sweep evaluated at second sweep powers'
            return 1 / inverse, quality
        raise ValueError("No supported overlap for calibration. Use matching powers or closely bracketing samples; no extrapolation is used.")
    factor = float(np.median([p[1] for p in pairs]))
    pairs = [(p, f, float(np.linalg.norm(y - factor * x) / np.linalg.norm(y)))
             for (p, f, _), (x, y) in zip(pairs, samples)]
    scatter = float(np.std([p[1] for p in pairs]) / factor)
    residual = float(max(p[2] for p in pairs))
    return factor, {"pairs": pairs, "relative_scatter": scatter, "max_relative_residual": residual,
                    "reverse_residuals": [float(np.linalg.norm(x - y / factor) / np.linalg.norm(x)) for x, y in samples],
                    "accepted": scatter <= .1 and residual <= .1,
                    "provisional": len(pairs) < 3 or scatter > .1 or residual > .1
                        or max(p[0] for p in pairs) < 1.1 * min(p[0] for p in pairs)}


def fingerprint(results, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for result in results:
        for array in (result.cube.energy, result.cube.gate, result.cube.Z):
            digest.update(np.ascontiguousarray(array, dtype=float).tobytes())
        digest.update(repr(sorted(source_rows(result), key=str)).encode())
        digest.update(repr(result.records).encode())
    return digest.hexdigest()


def suggested_name(first, second):
    names = [Path(r.records[0].file_name).stem for r in (first, second)]
    name = names[0] if names[1].startswith(names[0]) else names[0] + "__" + names[1]
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip(' .')
    if len(name) > 130:
        name = name[:115] + '_' + hashlib.sha256(name.encode()).hexdigest()[:8]
    return name + "__combined.csv"
