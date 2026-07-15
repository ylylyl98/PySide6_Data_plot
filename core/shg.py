from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd


_NUMERIC_HEADER_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)(?:\.\d+)?\s*(?:nm)?\s*$", re.IGNORECASE)


def _norm_column(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    normalized = {column: _norm_column(column) for column in columns}
    for candidate in candidates:
        wanted = _norm_column(candidate)
        for column, value in normalized.items():
            if value == wanted:
                return column
    return None


def _numeric_header(value: object) -> float | None:
    match = _NUMERIC_HEADER_RE.match(str(value))
    if match is None:
        return None
    try:
        wavelength = float(match.group(1))
    except ValueError:
        return None
    return wavelength if 200.0 <= wavelength <= 2000.0 else None


def _guess_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            counts = {",": line.count(","), "\t": line.count("\t"), ";": line.count(";")}
            return max(counts, key=counts.get)
    return ","


@dataclass(frozen=True)
class ShgSweepData:
    source_file: str
    wavelength_nm: np.ndarray
    spectra: np.ndarray
    sweep_axis: tuple[str, ...]
    target_angle_deg: np.ndarray
    measured_angle_deg: np.ndarray
    move_error_deg: np.ndarray
    move_ok: np.ndarray
    acquisition_ok: np.ndarray
    source_rows: np.ndarray
    detected_columns: dict[str, str]
    angle_unit: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShgSettings:
    peak_center_nm: float = 515.0
    gate_min_nm: float = 513.0
    gate_max_nm: float = 517.0
    left_min_nm: float = 508.0
    left_max_nm: float = 512.0
    right_min_nm: float = 518.0
    right_max_nm: float = 522.0
    background_method: str = "local_linear"
    sigma_clip: float = 3.0
    angle_scale: float = 1.0
    angle_offset_deg: float = 0.0
    angle_wrap_deg: float | None = None
    include_failed_rows: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ShgProcessResult:
    data: ShgSweepData
    settings: ShgSettings
    measured_angle_deg: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    integrated_area: np.ndarray
    area_uncertainty: np.ndarray
    peak_height: np.ndarray
    peak_wavelength_nm: np.ndarray
    baseline_slope: np.ndarray
    baseline_at_center: np.ndarray
    baseline_rms: np.ndarray
    gate_points: np.ndarray
    included: np.ndarray
    quality_flags: tuple[str, ...]
    background_file: str | None = None


def inspect_shg_csv(folder: str, file_name: str) -> bool:
    path = Path(folder) / Path(file_name).name
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path, sep=_guess_separator(path), nrows=0)
    except Exception:
        return False
    columns = [str(column).strip() for column in frame.columns]
    measured = _find_column(
        columns,
        ["measured position", "measured_position", "measuredposition", "measured angle", "measured_angle"],
    )
    wavelengths = [_numeric_header(column) for column in columns]
    return measured is not None and sum(value is not None for value in wavelengths) >= 3


def _signature(folder: str, file_name: str) -> tuple[int, int]:
    path = Path(folder) / Path(file_name).name
    if not path.is_file():
        raise FileNotFoundError(f"SHG CSV not found in folder root: {path}")
    stat = path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


def _infer_move_error(
    frame: pd.DataFrame,
    columns: list[str],
    target_col: str | None,
    measured_col: str,
) -> str | None:
    explicit = _find_column(columns, ["move_error", "position_error", "angle_error", "error_value"])
    if explicit is not None:
        return explicit
    if target_col is None:
        return None
    target = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
    measured = pd.to_numeric(frame[measured_col], errors="coerce").to_numpy(dtype=float)
    expected = np.abs(target - measured)
    measured_index = columns.index(measured_col)
    for column in columns[measured_index + 1 : measured_index + 4]:
        if _numeric_header(column) is not None:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values) & np.isfinite(expected)
        if np.count_nonzero(finite) and np.allclose(values[finite], expected[finite], rtol=2e-3, atol=2e-4):
            return column
    return None


@lru_cache(maxsize=64)
def _load_shg_sweep_cached(
    folder: str,
    file_name: str,
    signature: tuple[int, int],
) -> ShgSweepData:
    del signature
    path = Path(folder) / Path(file_name).name
    frame = pd.read_csv(path, sep=_guess_separator(path))
    frame.columns = [str(column).strip() for column in frame.columns]
    columns = list(frame.columns)

    measured_col = _find_column(
        columns,
        ["measured position", "measured_position", "measuredposition", "measured angle", "measured_angle"],
    )
    if measured_col is None:
        raise ValueError(f"SHG CSV {file_name!r} is missing a 'measured position' column.")
    target_col = _find_column(columns, ["target_value", "target value", "target_val", "target angle"])
    sweep_col = _find_column(columns, ["sweep_axis", "sweep axis", "sweep_ax"])
    move_ok_col = _find_column(columns, ["move_ok", "move ok"])
    acquisition_ok_col = _find_column(columns, ["acquisition_ok", "acquisition ok", "acquisition"])
    move_error_col = _infer_move_error(frame, columns, target_col, measured_col)
    unit_col = _find_column(columns, ["value_unit", "value unit", "angle_unit", "position_unit", "unit"])
    if unit_col is None:
        measured_index = columns.index(measured_col)
        for column in columns[measured_index + 1 : measured_index + 5]:
            values = {str(value).strip().lower() for value in frame[column].dropna().tolist()}
            if values and values <= {"deg", "degree", "degrees", "°"}:
                unit_col = column
                break

    groups: dict[float, list[str]] = {}
    for column in columns:
        wavelength = _numeric_header(column)
        if wavelength is not None:
            groups.setdefault(round(float(wavelength), 9), []).append(column)
    if len(groups) < 3:
        raise ValueError(f"SHG CSV {file_name!r} needs at least three numeric wavelength columns.")

    wavelength_values: list[float] = []
    spectrum_columns: list[str] = []
    for wavelength, candidates in groups.items():
        chosen = max(
            candidates,
            key=lambda column: int(
                np.isfinite(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)).sum()
            ),
        )
        wavelength_values.append(float(wavelength))
        spectrum_columns.append(chosen)
    wavelength = np.asarray(wavelength_values, dtype=float)
    spectra = frame[spectrum_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite_columns = np.any(np.isfinite(spectra), axis=0)
    wavelength = wavelength[finite_columns]
    spectra = spectra[:, finite_columns]
    if wavelength.size < 3:
        raise ValueError(f"SHG CSV {file_name!r} has fewer than three wavelength columns with numeric data.")
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    spectra = spectra[:, order]

    measured = pd.to_numeric(frame[measured_col], errors="coerce").to_numpy(dtype=float)
    target = (
        pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
        if target_col is not None
        else measured.copy()
    )
    move_error = (
        pd.to_numeric(frame[move_error_col], errors="coerce").to_numpy(dtype=float)
        if move_error_col is not None
        else np.abs(target - measured)
    )

    def _ok_values(column: str | None) -> np.ndarray:
        if column is None:
            return np.ones(frame.shape[0], dtype=bool)
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        return np.isfinite(values) & (values != 0)

    sweep_axis = (
        tuple(str(value).strip() for value in frame[sweep_col].fillna(""))
        if sweep_col is not None
        else tuple("" for _ in range(frame.shape[0]))
    )
    detected = {
        "measured_angle": measured_col,
        "target_angle": target_col or "",
        "move_error": move_error_col or "derived_abs_target_minus_measured",
        "sweep_axis": sweep_col or "",
        "move_ok": move_ok_col or "",
        "acquisition_ok": acquisition_ok_col or "",
        "angle_unit": unit_col or "",
    }
    return ShgSweepData(
        source_file=Path(file_name).name,
        wavelength_nm=wavelength,
        spectra=spectra,
        sweep_axis=sweep_axis,
        target_angle_deg=target,
        measured_angle_deg=measured,
        move_error_deg=move_error,
        move_ok=_ok_values(move_ok_col),
        acquisition_ok=_ok_values(acquisition_ok_col),
        source_rows=np.arange(frame.shape[0], dtype=int) + 2,
        detected_columns=detected,
        angle_unit=(
            tuple(str(value).strip() for value in frame[unit_col].fillna(""))
            if unit_col is not None
            else tuple("" for _ in range(frame.shape[0]))
        ),
    )


def load_shg_sweep_csv(folder: str, file_name: str) -> ShgSweepData:
    data = _load_shg_sweep_cached(folder, Path(file_name).name, _signature(folder, file_name))
    return ShgSweepData(
        source_file=data.source_file,
        wavelength_nm=np.asarray(data.wavelength_nm, float).copy(),
        spectra=np.asarray(data.spectra, float).copy(),
        sweep_axis=tuple(data.sweep_axis),
        target_angle_deg=np.asarray(data.target_angle_deg, float).copy(),
        measured_angle_deg=np.asarray(data.measured_angle_deg, float).copy(),
        move_error_deg=np.asarray(data.move_error_deg, float).copy(),
        move_ok=np.asarray(data.move_ok, bool).copy(),
        acquisition_ok=np.asarray(data.acquisition_ok, bool).copy(),
        source_rows=np.asarray(data.source_rows, int).copy(),
        detected_columns=dict(data.detected_columns),
        angle_unit=tuple(data.angle_unit),
    )


def _robust_fit(design: np.ndarray, values: np.ndarray, sigma_clip: float) -> tuple[np.ndarray, np.ndarray]:
    finite = np.all(np.isfinite(design), axis=1) & np.isfinite(values)
    if np.count_nonzero(finite) < design.shape[1] + 1:
        raise ValueError("insufficient finite sideband points")
    keep = finite.copy()
    coefficients = np.zeros(design.shape[1], dtype=float)
    for _ in range(5):
        coefficients, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        residual = values - design @ coefficients
        centered = residual[keep] - np.nanmedian(residual[keep])
        sigma = 1.4826 * float(np.nanmedian(np.abs(centered)))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.nanstd(residual[keep]))
        if not np.isfinite(sigma) or sigma <= 0:
            break
        new_keep = finite & (np.abs(residual - np.nanmedian(residual[keep])) <= max(1.0, sigma_clip) * sigma)
        if np.count_nonzero(new_keep) < design.shape[1] + 1 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    coefficients, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
    return coefficients, keep


def _integration_window(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    xf = x[finite]
    yf = y[finite]
    if xf.size < 2 or lo < xf[0] or hi > xf[-1]:
        return np.asarray([], float), np.asarray([], float)
    inside = (xf > lo) & (xf < hi)
    xout = np.concatenate(([lo], xf[inside], [hi]))
    yout = np.concatenate(([np.interp(lo, xf, yf)], yf[inside], [np.interp(hi, xf, yf)]))
    return xout, yout


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    if x.size < 2:
        return np.asarray([], float)
    weights = np.zeros(x.size, dtype=float)
    gaps = np.diff(x)
    weights[:-1] += 0.5 * gaps
    weights[1:] += 0.5 * gaps
    return weights


def _validate_settings(settings: ShgSettings, wavelength: np.ndarray) -> None:
    if not (
        settings.left_min_nm < settings.left_max_nm < settings.gate_min_nm
        < settings.gate_max_nm < settings.right_min_nm < settings.right_max_nm
    ):
        raise ValueError(
            "SHG regions must be ordered: left sideband < peak gate < right sideband, without overlap."
        )
    if settings.left_min_nm < float(np.nanmin(wavelength)) or settings.right_max_nm > float(np.nanmax(wavelength)):
        raise ValueError(
            f"SHG sidebands {settings.left_min_nm:g}-{settings.right_max_nm:g} nm exceed the "
            f"available wavelength range {float(np.nanmin(wavelength)):.6g}-{float(np.nanmax(wavelength)):.6g} nm."
        )
    if settings.background_method not in {"local_linear", "local_quadratic", "external", "none"}:
        raise ValueError(f"Unknown SHG background method: {settings.background_method}")


def process_shg_sweep(
    data: ShgSweepData,
    settings: ShgSettings,
    *,
    background: ShgSweepData | None = None,
) -> ShgProcessResult:
    wavelength = np.asarray(data.wavelength_nm, float).ravel()
    spectra = np.asarray(data.spectra, float)
    if spectra.shape != (len(data.source_rows), wavelength.size):
        raise ValueError("SHG spectra shape does not match acquisition rows and wavelength axis.")
    _validate_settings(settings, wavelength)
    if settings.background_method == "external" and background is None:
        raise ValueError("External SHG background mode requires a background CSV.")

    angle = settings.angle_scale * np.asarray(data.measured_angle_deg, float) + settings.angle_offset_deg
    if settings.angle_wrap_deg is not None:
        if settings.angle_wrap_deg <= 0:
            raise ValueError("SHG angle wrap must be positive.")
        angle = np.mod(angle, settings.angle_wrap_deg)

    sideband = (
        ((wavelength >= settings.left_min_nm) & (wavelength <= settings.left_max_nm))
        | ((wavelength >= settings.right_min_nm) & (wavelength <= settings.right_max_nm))
    )
    gate_mask = (wavelength >= settings.gate_min_nm) & (wavelength <= settings.gate_max_nm)
    if np.count_nonzero(sideband) < 4:
        raise ValueError("SHG background fit needs at least four wavelength points across the two sidebands.")
    if np.count_nonzero(gate_mask) < 2:
        raise ValueError("SHG peak gate needs at least two wavelength points.")

    background_reference: np.ndarray | None = None
    if background is not None:
        reference = np.nanmean(np.asarray(background.spectra, float), axis=0)
        finite = np.isfinite(background.wavelength_nm) & np.isfinite(reference)
        if np.count_nonzero(finite) < 2:
            raise ValueError("External SHG background has insufficient finite spectral data.")
        background_reference = np.interp(
            wavelength,
            np.asarray(background.wavelength_nm, float)[finite],
            reference[finite],
            left=np.nan,
            right=np.nan,
        )

    count = spectra.shape[0]
    baseline = np.full_like(spectra, np.nan, dtype=float)
    corrected = np.full_like(spectra, np.nan, dtype=float)
    area = np.full(count, np.nan, dtype=float)
    uncertainty = np.full(count, np.nan, dtype=float)
    height = np.full(count, np.nan, dtype=float)
    peak_wavelength = np.full(count, np.nan, dtype=float)
    slope = np.full(count, np.nan, dtype=float)
    baseline_center = np.full(count, np.nan, dtype=float)
    rms = np.full(count, np.nan, dtype=float)
    gate_points = np.zeros(count, dtype=int)
    included = np.zeros(count, dtype=bool)
    quality: list[str] = []
    x_centered = wavelength - settings.peak_center_nm

    for row_index in range(count):
        row = spectra[row_index]
        flags: list[str] = []
        numerical_failure = False
        if not np.isfinite(angle[row_index]):
            flags.append("INVALID_ANGLE")
            numerical_failure = True
        if not bool(data.move_ok[row_index]):
            flags.append("MOVE_FAILED")
        if not bool(data.acquisition_ok[row_index]):
            flags.append("ACQUISITION_FAILED")
        try:
            if settings.background_method == "none":
                row_baseline = np.zeros(wavelength.size, dtype=float)
                keep = sideband & np.isfinite(row)
            elif settings.background_method == "local_linear":
                design = np.column_stack((np.ones(wavelength.size), x_centered))
                coefficients, keep_local = _robust_fit(
                    design[sideband], row[sideband], settings.sigma_clip
                )
                row_baseline = design @ coefficients
                side_indices = np.flatnonzero(sideband)
                keep = np.zeros(wavelength.size, dtype=bool)
                keep[side_indices[keep_local]] = True
            elif settings.background_method == "local_quadratic":
                design = np.column_stack((np.ones(wavelength.size), x_centered, x_centered**2))
                coefficients, keep_local = _robust_fit(
                    design[sideband], row[sideband], settings.sigma_clip
                )
                row_baseline = design @ coefficients
                side_indices = np.flatnonzero(sideband)
                keep = np.zeros(wavelength.size, dtype=bool)
                keep[side_indices[keep_local]] = True
            else:
                assert background_reference is not None
                design = np.column_stack((background_reference, np.ones(wavelength.size), x_centered))
                coefficients, keep_local = _robust_fit(
                    design[sideband], row[sideband], settings.sigma_clip
                )
                row_baseline = design @ coefficients
                side_indices = np.flatnonzero(sideband)
                keep = np.zeros(wavelength.size, dtype=bool)
                keep[side_indices[keep_local]] = True
            baseline[row_index] = row_baseline
            corrected[row_index] = row - row_baseline
            residual = corrected[row_index, keep]
            if residual.size:
                rms[row_index] = float(np.sqrt(np.nanmean(residual**2)))
            local_coefficients = np.polyfit(
                x_centered[gate_mask], row_baseline[gate_mask], deg=1
            )
            slope[row_index] = float(local_coefficients[0])
            baseline_center[row_index] = float(local_coefficients[1])
            x_gate, y_gate = _integration_window(
                wavelength,
                corrected[row_index],
                settings.gate_min_nm,
                settings.gate_max_nm,
            )
            gate_points[row_index] = int(x_gate.size)
            if x_gate.size < 2:
                raise ValueError("insufficient finite peak-gate data")
            area[row_index] = float(np.trapezoid(y_gate, x_gate))
            weights = _trapezoid_weights(x_gate)
            uncertainty[row_index] = float(rms[row_index] * np.sqrt(np.sum(weights**2)))
            maximum = int(np.nanargmax(y_gate))
            height[row_index] = float(y_gate[maximum])
            peak_wavelength[row_index] = float(x_gate[maximum])
            if area[row_index] < 0:
                flags.append("NEGATIVE_AREA")
            gate_width = settings.gate_max_nm - settings.gate_min_nm
            if abs(peak_wavelength[row_index] - settings.peak_center_nm) > 0.4 * gate_width:
                flags.append("PEAK_OFF_CENTER")
        except Exception:
            flags.append("PROCESSING_FAILED")
            numerical_failure = True

        acquisition_failure = (not bool(data.move_ok[row_index])) or (not bool(data.acquisition_ok[row_index]))
        included[row_index] = not numerical_failure and (settings.include_failed_rows or not acquisition_failure)
        quality.append("OK" if not flags else ";".join(dict.fromkeys(flags)))

    return ShgProcessResult(
        data=data,
        settings=settings,
        measured_angle_deg=angle,
        baseline=baseline,
        corrected=corrected,
        integrated_area=area,
        area_uncertainty=uncertainty,
        peak_height=height,
        peak_wavelength_nm=peak_wavelength,
        baseline_slope=slope,
        baseline_at_center=baseline_center,
        baseline_rms=rms,
        gate_points=gate_points,
        included=included,
        quality_flags=tuple(quality),
        background_file=(
            background.source_file
            if background is not None and settings.background_method == "external"
            else None
        ),
    )
