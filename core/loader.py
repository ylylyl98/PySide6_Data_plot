from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from core import processing_run as P
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

    energy, I0 = P.build_external_baseline_avg(
        user_folder=user_folder,
        files_zero=list(files),
        which=w,
        save_npz=None,
    )
    return {"energy": np.asarray(energy, dtype=float).copy(), "I0": np.asarray(I0, dtype=float).copy()}


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
    # Preserve the established recipe order: each source contributes one
    # frame vector, sources are aligned to the first baseline grid and then
    # averaged.  Only the resulting baseline is aligned to the measurement.
    for baseline_file in assignment.baseline_files:
        data = P._load_canonical(user_folder, baseline_file, y_axis=y_axis)
        energy = _validated_interp_axis(data["energy"], context=f"Baseline {baseline_file}")
        z = np.asarray(data["Z"], dtype=float)
        if z.ndim != 2 or z.shape[1] != energy.size:
            raise ValueError(f"Baseline {baseline_file} data does not match its energy axis.")
    baseline = build_external_baseline(
        user_folder, assignment.baseline_files, which=assignment.baseline_which
    )
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
    by_measurement = {item.measurement_file: item for item in assignments}
    if (
        not selected
        or len(assignments) != len(selected)
        or set(by_measurement) != set(selected)
        or len(by_measurement) != len(selected)
    ):
        raise ValueError("DRR assignments must contain exactly one entry per measurement file.")
    ordered = tuple(by_measurement[item] for item in selected)
    common = all(item.baseline_mode == ordered[0].baseline_mode for item in ordered)
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
    canonical: list[dict[str, Any]] = [
        P._load_canonical(user_folder, item, y_axis=effective_y_axis) for item in selected
    ]
    first_energy = _validated_interp_axis(canonical[0]["energy"], context=f"Measurement {selected[0]}")
    first_gate = _validated_interp_axis(canonical[0]["gate_axis"], context=f"Measurement {selected[0]} gate") if np.asarray(canonical[0]["gate_axis"]).size >= 2 else np.asarray(canonical[0]["gate_axis"], float).ravel()
    if first_gate.size < 1 or not np.all(np.isfinite(first_gate)) or np.unique(first_gate).size != first_gate.size:
        raise ValueError(f"Measurement {selected[0]} gate axis is invalid.")
    gate_label = str(canonical[0].get("gate_label", "Gate"))
    corrected: list[np.ndarray] = []
    for name, assignment, data in zip(selected, ordered, canonical):
        energy = _validated_interp_axis(data["energy"], context=f"Measurement {name}")
        gate = np.asarray(data["gate_axis"], dtype=float).ravel()
        if gate.size < 1 or not np.all(np.isfinite(gate)) or np.unique(gate).size != gate.size:
            raise ValueError(f"Measurement {name} gate axis is invalid.")
        if str(data.get("gate_label", "Gate")) != gate_label:
            raise ValueError("DRR assignments use incompatible gate semantics.")
        z = np.asarray(data["Z"], dtype=float)
        if z.shape != (gate.size, energy.size):
            raise ValueError(f"Measurement {name} data does not match its grid.")
        if assignment.baseline_mode == "External":
            baseline = _native_external_baseline(
                user_folder, assignment, energy, y_axis=effective_y_axis
            )
            native = P._drr_from_Z(z, "external", baseline)
        else:
            native = P._drr_from_Z(
                z,
                "first" if assignment.baseline_mode == "Self (first frame)" else "last",
                None,
            )
        if name != selected[0] or energy.shape != first_energy.shape or not np.array_equal(energy, first_energy):
            native = _interp_rows_no_extrapolation(native, energy, first_energy, context=f"Measurement {name}")
        if gate.shape != first_gate.shape or not np.array_equal(gate, first_gate):
            native = _interp_rows_no_extrapolation(native.T, gate, first_gate, context=f"Measurement {name} gate").T
        corrected.append(native)
    result = np.nanmean(np.stack(corrected, axis=0), axis=0)
    if derivative in (1, 2):
        result = P.sg_derivative_origin_rows(
            result, first_energy, deriv=derivative,
            window_pts=dE_window_pts, polyorder=dE_polyorder,
            oversample=dE_oversample, interp_kind=dE_interp_kind,
            origin_like=dE_origin_like, pad_flat_edges=dE_pad_flat_edges,
        )
    cube = DataCube(
        energy=first_energy.copy(), gate=first_gate.copy(), Z=result,
        gate_label=gate_label, title=str(canonical[0].get("title_name", selected[0])),
        cbar_label="DR/R" if derivative is None else ("d(DR/R)/dE" if derivative == 1 else "d2(DR/R)/dE2"),
    )
    cube.drr_numerical_path = "heterogeneous"
    cube.drr_assignments = tuple(item.to_dict() for item in ordered)
    return cube
