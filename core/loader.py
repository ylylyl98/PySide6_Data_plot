from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import OrderedDict
from contextvars import ContextVar
from functools import lru_cache, wraps
import json
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from core import processing_run as P
from core.drr_axis_coordinates import CoordinateAccumulator
from core.streaming_average import StreamingNanMean
from core.load_cancellation import check_load_cancelled
if False:  # pragma: no cover - type-only imports are deferred to avoid a cycle.
    from core.drr_sources import DrrMeasurementAssignment


XLSX_Y_LABEL_DOPING = "Doping (V)"
XLSX_Y_LABEL_EFIELD = "Efield (V)"
XLSX_Y_LABEL_OPTIONS = (XLSX_Y_LABEL_DOPING, XLSX_Y_LABEL_EFIELD)
DAT_Y_AXIS_OPTIONS = ("Y", "Doping", "Electric field", "Gate voltage", "Custom")


def is_xlsx_map_file(file_name: str) -> bool:
    """Return True when a file name selects the precomputed XLSX map format."""
    return Path(str(file_name)).suffix.lower() == ".xlsx"


def resolve_xlsx_y_label(y_axis: str) -> str:
    """Map a y-axis request to a label-only XLSX axis label."""
    request = str(y_axis or "").strip().lower()
    if request in {"efield", "efield (v)"}:
        return XLSX_Y_LABEL_EFIELD
    return XLSX_Y_LABEL_DOPING


@dataclass
class DataCube:
    energy: np.ndarray
    gate: np.ndarray
    Z: np.ndarray
    gate_label: str
    title: str
    cbar_label: str
    gate_unit: str = ""
    y_axis_semantic: str = ""
    gate_coordinates: dict[str, np.ndarray] = field(default_factory=dict)
    drr_background_boundaries: tuple[float, ...] = ()


def _dat_sidecar_candidates(path: Path) -> tuple[Path, ...]:
    return (
        path.with_suffix(".metadata.json"),
        Path(f"{path}.plotmeta.json"),
    )


def _load_dat_sidecar(path: Path) -> dict:
    for sidecar in _dat_sidecar_candidates(path):
        if not sidecar.is_file():
            continue
        try:
            value = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _dat_background_boundaries(processing: dict, energy: np.ndarray) -> tuple[float, ...]:
    """Restore explicit seams without silently discarding invalid metadata."""
    if "background_boundaries_ev" not in processing:
        return ()
    values = processing["background_boundaries_ev"]
    if (not isinstance(values, list)
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values)):
        raise ValueError("DAT background boundaries must be a list of numeric energies.")
    boundaries = np.asarray(values, dtype=float)
    if not np.isfinite(boundaries).all() or np.any(np.diff(boundaries) <= 0):
        raise ValueError("DAT background boundaries must be finite and strictly increasing.")
    if boundaries.size:
        if not np.isfinite(energy).all() or np.unique(energy).size != energy.size:
            raise ValueError("DAT background boundaries require finite, distinct energy coordinates.")
        if np.any(boundaries <= np.min(energy)) or np.any(boundaries >= np.max(energy)):
            raise ValueError("DAT background boundaries must lie strictly inside the energy range.")
    return tuple(float(value) for value in boundaries)


def resolve_dat_y_axis(choice: str, *, custom_label: str = "", custom_unit: str = "") -> tuple[str, str, str]:
    """Map a user-facing DAT Y-axis choice to label, unit, and semantic id."""
    selected = str(choice or "Y").strip()
    if selected == "Doping":
        return "Doping", str(custom_unit).strip(), "doping"
    if selected == "Electric field":
        return "Electric field", str(custom_unit).strip(), "electric_field"
    if selected == "Gate voltage":
        return "Gate voltage", str(custom_unit).strip(), "gate_voltage"
    if selected == "Custom":
        label = str(custom_label).strip() or "Y"
        return label, str(custom_unit).strip(), "custom"
    return "Y", "", "y"


def load_dat(path: str | Path) -> DataCube:
    """Load an Origin-friendly exported DAT matrix into the normal DataCube model."""
    dat_path = Path(path)
    if not dat_path.is_file():
        raise FileNotFoundError(f"DAT file not found: {dat_path}")
    try:
        lines = dat_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read DAT file {dat_path}: {exc}") from exc
    content = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not content:
        raise ValueError(f"DAT file {dat_path.name!r} contains no numeric table.")
    header = content[0].split("\t")
    if len(header) < 2:
        raise ValueError(f"DAT file {dat_path.name!r} must contain an X column and at least one Y column.")
    try:
        gate = np.asarray([float(value.strip()) for value in header[1:]], dtype=float)
    except ValueError as exc:
        raise ValueError(f"DAT file {dat_path.name!r} has non-numeric Y/gate headers.") from exc
    rows: list[list[float]] = []
    for row_number, line in enumerate(content[1:], start=2):
        fields = line.split("\t")
        if len(fields) != len(header):
            raise ValueError(
                f"DAT file {dat_path.name!r} row {row_number} has {len(fields)} columns; expected {len(header)}."
            )
        try:
            rows.append([float(value.strip()) for value in fields])
        except ValueError as exc:
            raise ValueError(f"DAT file {dat_path.name!r} has non-numeric data on row {row_number}.") from exc
    if not rows:
        raise ValueError(f"DAT file {dat_path.name!r} contains no data rows.")
    table = np.asarray(rows, dtype=float)
    energy = table[:, 0]
    z = table[:, 1:].T.copy()
    metadata = _load_dat_sidecar(dat_path)
    plot = metadata.get("plot", {}) if isinstance(metadata.get("plot", {}), dict) else {}
    if isinstance(plot.get("linear"), dict):
        plot = plot["linear"]
    processing = metadata.get("processing", {}) if isinstance(metadata.get("processing", {}), dict) else {}
    mode = str(metadata.get("operation", processing.get("mode", "")))
    sidecar_label = metadata.get("y_axis_label", processing.get("y_axis_label", plot.get("ylabel", "Y")))
    sidecar_unit = metadata.get("y_axis_unit", processing.get("y_axis_unit", plot.get("y_unit", "")))
    sidecar_semantic = metadata.get("y_axis_semantic", processing.get("y_axis_semantic", ""))
    if not sidecar_semantic:
        sidecar_semantic = {
            "doping": "doping",
            "electric field": "electric_field",
            "gate voltage": "gate_voltage",
        }.get(str(sidecar_label).strip().lower(), "" if str(sidecar_label).strip() in {"", "Y"} else "custom")
    cube = DataCube(
        energy=energy,
        gate=gate,
        Z=z,
        gate_label=str(sidecar_label or "Y"),
        title=str(plot.get("title", dat_path.stem)),
        cbar_label=str(plot.get("cbar_label", mode or "Imported DAT")),
        gate_unit=str(sidecar_unit or ""),
        y_axis_semantic=str(sidecar_semantic or ""),
        drr_background_boundaries=_dat_background_boundaries(processing, energy),
    )
    cube.plot_metadata = plot
    cube.import_metadata = metadata
    return cube


def _validate_cube_arrays(energy: np.ndarray, gate: np.ndarray, Z: np.ndarray, *, context: str) -> None:
    e = np.asarray(energy).ravel()
    g = np.asarray(gate).ravel()
    z = np.asarray(Z)

    if e.size == 0 or g.size == 0 or z.size == 0:
        raise ValueError(f"{context}: empty energy/gate/Z data.")
    if z.ndim != 2:
        raise ValueError(f"{context}: Z must be 2D, got shape {z.shape}.")
    if z.shape not in {(g.size, e.size), (e.size, g.size)}:
        raise ValueError(
            f"{context}: Z shape {z.shape} does not match gate ({g.size}) x energy ({e.size})."
        )


def _source_path(user_folder: str, file_name: str) -> Path:
    requested = Path(file_name)
    return requested.resolve() if requested.is_absolute() else (Path(user_folder) / requested).resolve()


def _csv_signature(user_folder: str, file_name: str) -> Tuple[int, int]:
    """Return cache key pieces from a CSV mtime and size."""
    folder = Path(user_folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {user_folder}")

    csv_path = _source_path(user_folder, file_name)
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    stt = csv_path.stat()
    return int(stt.st_mtime_ns), int(stt.st_size)


@lru_cache(maxsize=256)
def _peek_y_axis_options_cached(
    user_folder: str, file_name: str, csv_sig: Tuple[int, int]
) -> Tuple[tuple[str, ...], str]:
    del csv_sig

    # Preferred: use implementation that owns _load_canonical
    if hasattr(P, "peek_y_axis_options"):
        opts, default = P.peek_y_axis_options(user_folder, file_name)
    else:
        d = P._load_canonical(user_folder, file_name, y_axis="auto")  # type: ignore[attr-defined]
        opts = d.get("available_axes", ["Vbg", "Vtg"])
        default = d.get("default_axis", opts[0] if opts else "Vtg")

    if default not in opts and opts:
        default = opts[0]
    return tuple(str(o) for o in opts), str(default)


def peek_y_axis_options(user_folder: str, file_name: str) -> Tuple[list[str], str]:
    csv_sig = _csv_signature(user_folder, file_name)
    opts, default = _peek_y_axis_options_cached(user_folder, file_name, csv_sig)
    return list(opts), default


@lru_cache(maxsize=512)
def _load_pl_cached(
    user_folder: str, file_name: str, log_scale: bool, y_axis: str, csv_sig: Tuple[int, int]
) -> dict:
    del csv_sig

    return P.process_pl(
        user_folder=user_folder,
        file=file_name,
        y_axis=y_axis,
        plot_interactive=False,
        save_png=False,
        save_dat_file=False,
        move_original=False,
        pl_scales=("log" if log_scale else "linear",),
        open_both_interactive=False,
    )


def load_pl(user_folder: str, file_name: str, *, log_scale: bool = False, y_axis: str = "auto") -> DataCube:
    if is_xlsx_map_file(file_name):
        return load_xlsx_map(user_folder, file_name, y_label=resolve_xlsx_y_label(y_axis))

    csv_sig = _csv_signature(user_folder, file_name)
    effective_y_axis = P.resolve_shared_y_axis_request([file_name], y_axis)
    res = _load_pl_cached(user_folder, file_name, bool(log_scale), effective_y_axis, csv_sig)

    _validate_cube_arrays(res["energy"], res["gate_axis"], res["Z"], context="PL load")
    return DataCube(
        energy=np.asarray(res["energy"], dtype=float).copy(),
        gate=np.asarray(res["gate_axis"], dtype=float).copy(),
        Z=np.asarray(res["Z"], dtype=float).copy(),
        gate_label=res.get("gate_label", "Gate (V)"),
        title=res.get("title", file_name),
        cbar_label="PL (a.u.)",
    )


def _xlsx_signature(user_folder: str, file_name: str) -> Tuple[int, int]:
    """Return cache key pieces from an XLSX mtime and size."""
    folder = Path(user_folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {user_folder}")

    xlsx_path = _source_path(user_folder, file_name)
    if not xlsx_path.exists() or not xlsx_path.is_file():
        raise FileNotFoundError(f"XLSX not found: {xlsx_path}")

    stat = xlsx_path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


def _xlsx_cell_value(cell, *, context: str) -> float:
    coordinate = getattr(cell, "coordinate", "?")
    if cell.data_type == "f":
        raise ValueError(f"{context}: formulas are not supported (cell {coordinate}).")
    value = cell.value
    if value is None or isinstance(value, bool):
        raise ValueError(f"{context}: empty cell at {coordinate}.")
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{context}: non-numeric value {value!r} at {coordinate}.")
    if not np.isfinite(result):
        raise ValueError(f"{context}: non-finite value at {coordinate}.")
    return result


@lru_cache(maxsize=32)
def _load_xlsx_map_cached(
    user_folder: str, file_name: str, signature: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    del signature

    from openpyxl import load_workbook

    path = _source_path(user_folder, file_name)
    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        sheet_names = workbook.sheetnames
        if "dR_R" in sheet_names:
            sheet = workbook["dR_R"]
        elif len(sheet_names) == 1:
            sheet = workbook[sheet_names[0]]
        else:
            raise ValueError(
                f"XLSX map {file_name!r} has {len(sheet_names)} sheets; "
                "expected a single 'dR_R' sheet."
            )
        rows = list(sheet.iter_rows(values_only=False))
    finally:
        workbook.close()

    if len(rows) < 2:
        raise ValueError(
            f"XLSX map {file_name!r} must contain a header row and at least one data row."
        )

    header = rows[0]
    if len(header) < 2:
        raise ValueError(
            f"XLSX map {file_name!r} header must include a corner cell and at least one y-value column."
        )

    if header[0].data_type == "f":
        raise ValueError(
            f"XLSX map {file_name!r}: formulas are not supported in the header corner "
            f"({header[0].coordinate})."
        )

    gate = np.asarray(
        [_xlsx_cell_value(cell, context=f"XLSX map {file_name!r} header") for cell in header[1:]],
        dtype=float,
    )
    if np.unique(gate).size != gate.size:
        raise ValueError(f"XLSX map {file_name!r} contains duplicate y-axis (doping) values.")

    energy_rows: list[float] = []
    z_rows: list[list[float]] = []
    expected = len(header)
    for row_index, row in enumerate(rows[1:], start=2):
        if len(row) != expected:
            raise ValueError(
                f"XLSX map {file_name!r}: row {row_index} has {len(row)} columns, expected {expected}."
            )
        energy_rows.append(
            _xlsx_cell_value(row[0], context=f"XLSX map {file_name!r} row {row_index} energy")
        )
        z_rows.append(
            [
                _xlsx_cell_value(cell, context=f"XLSX map {file_name!r} row {row_index}")
                for cell in row[1:]
            ]
        )

    energy = np.asarray(energy_rows, dtype=float)
    z = np.asarray(z_rows, dtype=float)

    order = np.argsort(energy, kind="stable")
    energy = energy[order]
    z = z[order, :]
    z = z.T

    return energy.copy(), gate.copy(), z.copy(), Path(file_name).name


def load_xlsx_map(
    user_folder: str, file_name: str, *, y_label: str = XLSX_Y_LABEL_DOPING
) -> DataCube:
    """Load a precomputed dR/R XLSX map into the standard DataCube contract."""
    raw_name = str(file_name)
    if not is_xlsx_map_file(raw_name):
        raise ValueError(f"XLSX map must reference a .xlsx file, got {raw_name!r}")
    signature = _xlsx_signature(user_folder, file_name)
    energy, gate, z, title = _load_xlsx_map_cached(user_folder, raw_name, signature)
    _validate_cube_arrays(energy, gate, z, context="XLSX map load")
    return DataCube(
        energy=np.asarray(energy, dtype=float).copy(),
        gate=np.asarray(gate, dtype=float).copy(),
        Z=np.asarray(z, dtype=float).copy(),
        gate_label=resolve_xlsx_y_label(y_label),
        title=title,
        cbar_label="dR/R",
    )


_DRR_BASELINE_REUSE: ContextVar[OrderedDict | None] = ContextVar('drr_baseline_reuse', default=None)
_DRR_BASELINE_BUDGET = 16 * 1024 * 1024


def _baseline_signatures(user_folder, files):
    from core.drr_composite_background import composite_metadata_signature
    signatures = []
    try:
        for name in files:
            path = _source_path(user_folder, name)
            signature = composite_metadata_signature(path)
            if (not _reliable_background_signature(signature[:3])
                    or (signature[2] and not _reliable_background_signature(signature[3]))):
                return None
            signatures.append((path, signature))
    except OSError:
        return None  # Let the normal loader report missing/unreadable sources.
    return tuple(signatures)


def _reliable_background_signature(value):
    if value is None:
        return False
    return all(_reliable_background_signature(item) for item in value) if isinstance(value, (tuple, list)) else True


def _background_source_boundaries(user_folder, file_name):
    """Read frozen composite seams; never resolve the original component files."""
    from core.drr_composite_background import composite_background_boundaries
    check_load_cancelled()
    path = _source_path(user_folder, file_name)
    cache = _DRR_BASELINE_REUSE.get()
    signature = _baseline_signatures(user_folder, (file_name,)) if cache is not None else None
    key = ('composite_boundaries', path, signature)
    if cache is not None and signature is not None and key in cache:
        cache.move_to_end(key)
        return cache[key][0]
    boundaries = tuple(composite_background_boundaries(path))
    check_load_cancelled()
    if cache is not None and signature is not None and signature == _baseline_signatures(user_folder, (file_name,)):
        _remember_baseline(cache, key, boundaries, max(64, len(boundaries) * 24))
    return boundaries


def has_composite_background(user_folder, files):
    from core.drr_composite_background import is_composite_background
    return any(is_composite_background(_source_path(user_folder, name)) for name in files)


def effective_background_boundaries(user_folder, assignment, energy):
    """Combine recipe switches with source seams only where that source is used."""
    if assignment.baseline_mode != "External":
        return ()
    from core.drr_background_regions import background_boundaries
    result = set(background_boundaries(assignment, energy))
    minimum, maximum = float(np.min(energy)), float(np.max(energy))
    for source in assignment.baseline_source_files:
        for boundary in _background_source_boundaries(user_folder, source):
            if not minimum < boundary < maximum:
                continue
            # At a recipe endpoint the explicit switch is already accounted
            # for above. An inherited seam matters only if this source still
            # contributes immediately on both sides (including equal recipes).
            left = next((r.baseline_files for r in assignment.regions
                         if r.xmin_ev < boundary <= r.xmax_ev), assignment.baseline_files)
            right = next((r.baseline_files for r in assignment.regions
                          if r.xmin_ev <= boundary < r.xmax_ev), assignment.baseline_files)
            if source in left and source in right:
                result.add(float(boundary))
    return tuple(sorted(result))


def _remember_baseline(cache, key, value, size):
    if cache is None or size > _DRR_BASELINE_BUDGET:
        return
    while cache and (len(cache) >= 64 or size + sum(entry[1] for entry in cache.values())
                     > _DRR_BASELINE_BUDGET):
        cache.popitem(last=False)
    cache[key] = (value, size)


def _reuse_drr_baselines(function):
    @wraps(function)
    def load(*args, **kwargs):
        token = _DRR_BASELINE_REUSE.set(OrderedDict())
        try:
            return function(*args, **kwargs)
        finally:
            _DRR_BASELINE_REUSE.reset(token)
    return load


def build_external_baseline(user_folder: str, files: Sequence[str], *, which: str = "last") -> dict:
    """
    Returns dict with keys: energy, I0

    which:
      - "first" : use first frame in each file
      - "last"  : use last frame in each file
      - "all"   : average ALL frames within each file, then average across files
    """
    if not files:
        raise ValueError("build_external_baseline: 'files' is empty.")

    w = (which or "last").strip().lower()
    alias = {
        "first": "first",
        "1st": "first",
        "start": "first",
        "last": "last",
        "end": "last",
        "all": "all",
        "avg": "all",
        "mean": "all",
        "all_frames": "all",
        "all frames": "all",
        "frames": "all",
    }
    w = alias.get(w, w)
    if w not in ("first", "last", "all"):
        raise ValueError(f"Unknown which='{which}'. Use 'first', 'last', or 'all'.")

    check_load_cancelled()
    for name in files:
        _background_source_boundaries(user_folder, name)
    cache = _DRR_BASELINE_REUSE.get()
    signature = _baseline_signatures(user_folder, files) if cache is not None else None
    key = ('aggregate', signature, w)
    if cache is not None and signature is not None and key in cache:
        cache.move_to_end(key)
        return {name: array.copy() for name, array in cache[key][0].items()}
    energy, I0 = P.build_external_baseline_avg(
        user_folder=user_folder,
        files_zero=list(files),
        which=w,
        save_npz=None,
    )
    result = {"energy": np.asarray(energy, dtype=float).copy(), "I0": np.asarray(I0, dtype=float).copy()}
    check_load_cancelled()
    if cache is not None and signature is not None and signature == _baseline_signatures(user_folder, files):
        _remember_baseline(cache, key, {name: array.copy() for name, array in result.items()},
                           sum(array.nbytes for array in result.values()))
    return result


def load_drr_avg(
    user_folder: str,
    files: Sequence[str],
    *,
    bg_mode: str,
    y_axis: str = "auto",
    external_vector: Optional[np.ndarray] = None,
    external_energy: Optional[np.ndarray] = None,
    derivative: Optional[int] = None,
    dE_window_pts: int = 20,
    dE_polyorder: int = 2,
    dE_oversample: float = 1.0,
    dE_interp_kind: str = "cubic",
    dE_origin_like: bool = False,
    dE_pad_flat_edges: bool = True,
) -> DataCube:
    effective_y_axis = P.resolve_shared_y_axis_request(files, y_axis)
    res = P.process_ref_avg(
        user_folder=user_folder,
        files=list(files),
        bg_mode=bg_mode,
        y_axis=effective_y_axis,
        external_vector=external_vector,
        external_energy=external_energy,
        use_global_background=False,
        plot_interactive=False,
        save_png=False,
        save_dat_file=False,
        move_original=False,
        derivative=derivative,
        dE_window_pts=dE_window_pts,
        dE_polyorder=dE_polyorder,
        dE_oversample=dE_oversample,
        dE_interp_kind=dE_interp_kind,
        dE_origin_like=dE_origin_like,
        dE_pad_flat_edges=dE_pad_flat_edges,
        center_zero=True,
    )

    _validate_cube_arrays(res["energy"], res["gate_axis"], res["Z_out"], context="DRR load")

    cbar = "DR/R" if derivative is None else ("d(DR/R)/dE" if derivative == 1 else "d2(DR/R)/dE2")
    return DataCube(
        energy=np.asarray(res["energy"], dtype=float).copy(),
        gate=np.asarray(res["gate_axis"], dtype=float).copy(),
        Z=np.asarray(res["Z_out"], dtype=float).copy(),
        gate_label=res.get("gate_label", "Gate (V)"),
        title=res.get("title", "DR/R"),
        cbar_label=cbar,
        gate_coordinates={key: np.array(values, dtype=float, copy=True)
                          for key, values in res.get('gate_coordinates', {}).items()},
    )


def _validated_interp_axis(axis: np.ndarray, *, context: str) -> np.ndarray:
    values = np.asarray(axis, dtype=float).ravel()
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{context} axis must contain at least two finite points.")
    if np.unique(values).size != values.size:
        raise ValueError(f"{context} axis contains duplicate points.")
    return values


def _interp_rows_no_extrapolation(
    values: np.ndarray, source_axis: np.ndarray, target_axis: np.ndarray, *, context: str
) -> np.ndarray:
    original_source = _validated_interp_axis(source_axis, context=context)
    target = np.asarray(target_axis, dtype=float).ravel()
    if target.size < 1 or not np.all(np.isfinite(target)):
        raise ValueError(f"{context} target axis must contain finite points.")
    order = np.argsort(original_source, kind="stable")
    source = original_source[order]
    rows = np.asarray(values, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != source.size:
        raise ValueError(f"{context} values do not match its axis.")
    # An entirely disjoint member would otherwise become an all-NaN row and
    # disappear from the later nanmean.  Refuse it at the interpolation
    # boundary so the caller cannot silently average away a measurement.
    if not np.any((target >= source[0]) & (target <= source[-1])):
        raise ValueError(f"{context} axes have no finite overlap.")
    rows = rows[:, order]
    # Compare against the original ordering before sorting.  This preserves
    # exact descending grids and their NaN positions without interpolation.
    if target.shape == original_source.shape and np.array_equal(
        target, original_source, equal_nan=True
    ):
        return np.asarray(values, dtype=float).copy()
    if target.shape == source.shape and np.array_equal(target, source, equal_nan=True):
        return rows.copy()
    out = np.full((rows.shape[0], target.size), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        finite = np.isfinite(row)
        if np.count_nonzero(finite) < 2:
            continue
        x = source[finite][np.argsort(source[finite], kind="stable")]
        y = row[finite][np.argsort(source[finite], kind="stable")]
        valid = (target >= x[0]) & (target <= x[-1])
        out[row_index, valid] = np.interp(target[valid], x, y)
    return out


def _wavelength_center_nm(energy: np.ndarray) -> float:
    values = np.asarray(energy, dtype=float).ravel()
    finite = np.isfinite(values) & (values > 0.0)
    if not np.any(finite):
        return float("nan")
    wavelength = 1240.0 / values[finite]
    return 0.5 * (float(np.min(wavelength)) + float(np.max(wavelength)))


def _native_external_baseline(
    user_folder: str, assignment: DrrMeasurementAssignment, measurement_energy: np.ndarray, *, y_axis: str
) -> np.ndarray:
    check_load_cancelled()
    cache = _DRR_BASELINE_REUSE.get()
    signature = _baseline_signatures(user_folder, assignment.baseline_files) if cache is not None else None
    axis = np.asarray(measurement_energy, dtype=float).ravel()
    key = ('native', signature, assignment.baseline_which, y_axis, axis.tobytes())
    if cache is not None and signature is not None and key in cache:
        cache.move_to_end(key)
        return cache[key][0].copy()
    result = _native_external_baseline_uncached(user_folder, assignment, measurement_energy, y_axis=y_axis)
    check_load_cancelled()
    if cache is not None and signature is not None and signature == _baseline_signatures(user_folder, assignment.baseline_files):
        _remember_baseline(cache, key, result.copy(), result.nbytes + axis.nbytes)
    return result


def _native_external_baseline_uncached(
    user_folder: str, assignment: DrrMeasurementAssignment, measurement_energy: np.ndarray, *, y_axis: str
) -> np.ndarray:
    # Preserve the established recipe order: each source contributes one
    # frame vector, sources are aligned to the first baseline grid and then
    # averaged.  Only the resulting baseline is aligned to the measurement.
    for baseline_file in assignment.baseline_files:
        check_load_cancelled()
        data = P._load_canonical(user_folder, baseline_file, y_axis=y_axis)
        energy = _validated_interp_axis(data["energy"], context=f"Baseline {baseline_file}")
        z = np.asarray(data["Z"], dtype=float)
        if z.ndim != 2 or z.shape[1] != energy.size:
            raise ValueError(f"Baseline {baseline_file} data does not match its energy axis.")
    baseline = build_external_baseline(
        user_folder, assignment.baseline_files, which=assignment.baseline_which
    )
    if has_composite_background(user_folder, assignment.baseline_files):
        target = np.asarray(measurement_energy, float)
        source = np.asarray(baseline['energy'], float)
        order, target_order = np.argsort(source, kind='stable'), np.argsort(target, kind='stable')
        if (source.shape != target.shape or not np.allclose(source[order], target[target_order],
                                                          rtol=1e-9, atol=1e-10)):
            raise ValueError('Composite background requires the exact measurement energy grid.')
        # A frozen reference may deliberately contain missing samples. On its
        # validated grid preserve them; interpolation/coverage thresholds would
        # change the saved background when it is selected again.
        aligned = np.empty_like(np.asarray(baseline['I0'], float))
        aligned[target_order] = np.asarray(baseline['I0'], float)[order]
        return aligned
    measurement_center = _wavelength_center_nm(measurement_energy)
    baseline_center = _wavelength_center_nm(np.asarray(baseline["energy"], dtype=float))
    if (
        np.isfinite(measurement_center)
        and np.isfinite(baseline_center)
        and abs(measurement_center - baseline_center) > 1.0
    ):
        raise ValueError(
            "External baseline wavelength center must match the measurement: "
            f"measurement≈{measurement_center:.3g} nm, "
            f"background≈{baseline_center:.3g} nm."
        )
    aligned = _interp_rows_no_extrapolation(
        np.asarray(baseline["I0"], dtype=float)[None, :],
        np.asarray(baseline["energy"], dtype=float), measurement_energy,
        context="External baseline",
    )[0]
    minimum_overlap = max(2, int(np.ceil(0.5 * np.asarray(measurement_energy).size)))
    if np.count_nonzero(np.isfinite(aligned)) < minimum_overlap:
        raise ValueError("External baseline coverage is insufficient for this measurement.")
    return aligned


def build_effective_background(
    user_folder: str, assignment: DrrMeasurementAssignment, measurement_energy: np.ndarray,
    *, y_axis: str = "auto",
) -> np.ndarray:
    """Build the actual external denominator, with optional energy overrides.

    Every source must match the original acquisition grid. Region composition
    happens before normalization, and never averages unrelated region recipes.
    """
    if assignment.baseline_mode != "External":
        raise ValueError("An effective external background requires External DRR mode.")
    if not assignment.regions:
        return _native_external_baseline(user_folder, assignment, measurement_energy, y_axis=y_axis)
    from core.data_io import _validate_drr_external_spectral_grids
    from core.drr_background_regions import region_masks
    masks = region_masks(measurement_energy, assignment.regions)
    _validate_drr_external_spectral_grids(
        user_folder, (assignment.measurement_file,), assignment.baseline_source_files)
    vectors = {}
    target_energy = np.asarray(measurement_energy, dtype=float)
    target_order = np.argsort(target_energy, kind="stable")

    def reference(files, which):
        check_load_cancelled()
        key = (files, which)
        if key not in vectors:
            baseline = build_external_baseline(user_folder, files, which=which)
            source_energy = _validated_interp_axis(baseline["energy"], context="Regional background")
            values = np.asarray(baseline["I0"], dtype=float)
            source_order = np.argsort(source_energy, kind="stable")
            if (source_energy.shape != target_energy.shape or values.shape != source_energy.shape
                    or not np.allclose(source_energy[source_order], target_energy[target_order],
                                       rtol=1e-9, atol=1e-10)):
                raise ValueError("Regional background must use the measurement's exact energy grid.")
            # Original acquisition grids were checked above. Reorder samples
            # only: NaNs outside a region need no coverage and NaNs inside it
            # must remain missing rather than being interpolated away.
            aligned = np.empty_like(values)
            aligned[target_order] = values[source_order]
            vectors[key] = aligned
        return vectors[key]

    background = reference(assignment.baseline_files, assignment.baseline_which).copy()
    for region, mask in zip(assignment.regions, masks):
        background[mask] = reference(region.baseline_files, region.baseline_which)[mask]
    return background


@_reuse_drr_baselines
def load_drr_resolved_avg(
    user_folder: str,
    files: Sequence[str],
    *,
    assignments: Sequence[DrrMeasurementAssignment],
    y_axis: str = "auto",
    derivative: Optional[int] = None,
    dE_window_pts: int = 20,
    dE_polyorder: int = 2,
    dE_oversample: float = 1.0,
    dE_interp_kind: str = "cubic",
    dE_origin_like: bool = False,
    dE_pad_flat_edges: bool = True,
) -> DataCube:
    """Load DRR from validated per-measurement assignments.

    A common effective baseline delegates to the historic loader.  Different
    assignments are corrected on their native grids and aligned only after
    correction, with NaN outside coverage.
    """
    selected = tuple(str(item) for item in files)
    from core.drr_bn_store import validate_bn_background_usage
    validate_bn_background_usage(user_folder, selected, tuple(dict.fromkeys(
        source for item in assignments for source in item.baseline_source_files)), y_axis=y_axis)
    by_measurement = {item.measurement_file: item for item in assignments}
    if (
        not selected
        or len(assignments) != len(selected)
        or set(by_measurement) != set(selected)
        or len(by_measurement) != len(selected)
    ):
        raise ValueError("DRR assignments must contain exactly one entry per measurement file.")
    ordered = tuple(by_measurement[item] for item in selected)
    has_regions = any(item.regions for item in ordered)
    has_composite = any(has_composite_background(user_folder, item.baseline_source_files)
                        for item in ordered if item.baseline_mode == 'External')
    if has_regions:
        from core.data_io import _validate_drr_external_spectral_grids
        _validate_drr_external_spectral_grids(user_folder, selected, tuple(dict.fromkeys(
            source for item in ordered for source in item.baseline_source_files)))
    common = not (has_regions or has_composite) and all(item.baseline_mode == ordered[0].baseline_mode for item in ordered)
    external_descriptors_same = all(
        item.baseline_files == ordered[0].baseline_files
        and item.baseline_which == ordered[0].baseline_which
        for item in ordered
    ) if common else False
    if common and ordered[0].baseline_mode == "External" and external_descriptors_same:
        effective: list[tuple[np.ndarray, np.ndarray]] = []
        baseline = build_external_baseline(
            user_folder, ordered[0].baseline_files, which=ordered[0].baseline_which
        )
        effective.append((np.asarray(baseline["energy"], float), np.asarray(baseline["I0"], float)))
    elif common and ordered[0].baseline_mode == "External":
        effective = []
        for item in ordered:
            baseline = build_external_baseline(
                user_folder, item.baseline_files, which=item.baseline_which
            )
            effective.append((np.asarray(baseline["energy"], float), np.asarray(baseline["I0"], float)))
        common = all(
            energy.shape == effective[0][0].shape
            and np.array_equal(energy, effective[0][0], equal_nan=True)
            and np.array_equal(vector, effective[0][1], equal_nan=True)
            for energy, vector in effective[1:]
        )
    if common:
        first = ordered[0]
        if first.baseline_mode == "External":
            cube = load_drr_avg(
                user_folder, selected, bg_mode="external", y_axis=y_axis,
                external_vector=np.asarray(effective[0][1], float),
                external_energy=np.asarray(effective[0][0], float), derivative=derivative,
                dE_window_pts=dE_window_pts, dE_polyorder=dE_polyorder,
                dE_oversample=dE_oversample, dE_interp_kind=dE_interp_kind,
                dE_origin_like=dE_origin_like, dE_pad_flat_edges=dE_pad_flat_edges,
            )
        else:
            cube = load_drr_avg(
                user_folder, selected,
                bg_mode="self_first" if first.baseline_mode == "Self (first frame)" else "self_last",
                y_axis=y_axis, derivative=derivative,
                dE_window_pts=dE_window_pts, dE_polyorder=dE_polyorder,
                dE_oversample=dE_oversample, dE_interp_kind=dE_interp_kind,
                dE_origin_like=dE_origin_like, dE_pad_flat_edges=dE_pad_flat_edges,
            )
        cube.drr_numerical_path = "common"
        cube.drr_assignments = tuple(item.to_dict() for item in ordered)
        return cube

    effective_y_axis = P.resolve_shared_y_axis_request(selected, y_axis)
    boundaries = set()
    for index, (name, assignment) in enumerate(zip(selected, ordered)):
        check_load_cancelled()
        data = P._load_canonical(user_folder, name, y_axis=effective_y_axis)
        energy = _validated_interp_axis(data["energy"], context=f"Measurement {name}")
        gate = np.asarray(data["gate_axis"], dtype=float).ravel()
        if gate.size < 1 or not np.all(np.isfinite(gate)) or np.unique(gate).size != gate.size:
            raise ValueError(f"Measurement {name} gate axis is invalid.")
        if index == 0:
            first_energy = energy.copy()
            first_gate = _validated_interp_axis(gate, context=f"Measurement {name} gate") if gate.size >= 2 else gate.copy()
            gate_label = str(data.get('gate_label', 'Gate'))
            title = str(data.get('title_name', selected[0]))
            corrected_average = StreamingNanMean((first_gate.size, first_energy.size))
            coordinates = CoordinateAccumulator(first_gate)
        if str(data.get("gate_label", "Gate")) != gate_label:
            raise ValueError("DRR assignments use incompatible gate semantics.")
        z = np.asarray(data["Z"], dtype=float)
        if z.shape != (gate.size, energy.size):
            raise ValueError(f"Measurement {name} data does not match its grid.")
        coordinates.add(gate, data.get('gate_coordinates', {}))
        if assignment.baseline_mode == "External":
            baseline = build_effective_background(
                user_folder, assignment, energy, y_axis=effective_y_axis
            )
            boundaries.update(effective_background_boundaries(user_folder, assignment, energy))
            native = P._drr_from_Z(z, "external", baseline)
        else:
            native = P._drr_from_Z(
                z,
                "first" if assignment.baseline_mode == "Self (first frame)" else "last",
                None,
            )
        energy_changed = energy.shape != first_energy.shape or not np.array_equal(energy, first_energy)
        if energy_changed or (name != selected[0] and not (has_regions or has_composite)):
            native = _interp_rows_no_extrapolation(native, energy, first_energy, context=f"Measurement {name}")
        if gate.shape != first_gate.shape or not np.array_equal(gate, first_gate):
            native = _interp_rows_no_extrapolation(native.T, gate, first_gate, context=f"Measurement {name} gate").T
        check_load_cancelled()
        corrected_average.add(native)
        del data, z, native
    check_load_cancelled()
    result = corrected_average.mean()
    del corrected_average
    if derivative in (1, 2) and not (has_regions or boundaries):
        result = P.sg_derivative_origin_rows(
            result, first_energy, deriv=derivative,
            window_pts=dE_window_pts, polyorder=dE_polyorder,
            oversample=dE_oversample, interp_kind=dE_interp_kind,
            origin_like=dE_origin_like, pad_flat_edges=dE_pad_flat_edges,
        )
    check_load_cancelled()
    cube = DataCube(
        energy=first_energy.copy(), gate=first_gate.copy(), Z=result,
        gate_label=gate_label, title=title,
        cbar_label="DR/R" if derivative is None else ("d(DR/R)/dE" if derivative == 1 else "d2(DR/R)/dE2"),
        gate_coordinates=coordinates.result(),
        drr_background_boundaries=tuple(sorted(point for point in boundaries
            if float(np.min(first_energy)) < point < float(np.max(first_energy)))),
    )
    if derivative in (1, 2) and (has_regions or boundaries):
        from core.processing import apply_sg_derivative_energy
        cube, _ = apply_sg_derivative_energy(cube, derivative=derivative,
                                           window_length=dE_window_pts, polyorder=dE_polyorder)
    cube.drr_numerical_path = "heterogeneous"
    cube.drr_assignments = tuple(item.to_dict() for item in ordered)
    return cube
