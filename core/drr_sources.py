from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Iterable, Sequence

import numpy as np

from core.processing import split_group_and_sort_key


SUPPORTED_DRR_SUFFIXES = {".csv", ".xlsx"}
_BACKGROUND_TOKEN = re.compile(
    r"(?:^|[_\-\s])(?:[np]?back(?:ground)?\d*|bg\d*|dark|i0|reference|ref)"
    r"(?=$|[_\-\s(\[])",
    re.IGNORECASE,
)
_WAVELENGTH_CENTER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?P<value>\d+(?:[pP.]\d+)?)\s*(?:nmc|nm[_\-\s]?center)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_CENTER_WAVELENGTH_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:center|cent)[_\-\s]?(?P<value>\d+(?:[pP.]\d+)?)\s*nm(?![A-Za-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DrrSource:
    source: str
    filename: str
    group_key: str
    session_date: str
    modified_time: float
    is_background: bool
    processed: bool = False
    classification: str = "measurement"
    classification_reason: str = ""
    gate_varies: bool | None = None
    frame_count: int | None = None
    size_bytes: int = 0
    wavelength_center_nm: float | None = None
    wavelength_center_source: str = ""
    metadata_role: str = ""
    linked_backgrounds: tuple[str, ...] = ()
    saved_baseline_modes: tuple[str, ...] = ()
    gate_grid: tuple[tuple[float, ...], ...] = ()
    spectral_grid: tuple[float, ...] = ()
    gate_direction: str = ""
    gate_ranges: tuple[tuple[float, float], ...] = ()
    gate_labels: tuple[str, ...] = ()
    grid_complete: bool = False


@dataclass(frozen=True)
class DrrSourceGroup:
    key: str
    title: str
    session_date: str
    files: tuple[DrrSource, ...]
    modified_time: float
    is_background: bool
    processed: bool
    classification: str = "measurement"
    wavelength_centers_nm: tuple[float, ...] = ()
    processed_count: int = 0
    frame_count_range: tuple[int, int] | None = None
    linked_backgrounds: tuple[str, ...] = ()
    saved_baseline_modes: tuple[str, ...] = ()
    gate_direction: str = ""
    gate_ranges: tuple[tuple[float, float], ...] = ()
    gate_labels: tuple[str, ...] = ()
    grid_complete: bool = False


class DrrSourceCache:
    """Incremental cache for the per-file work used by catalog discovery.

    File classification still runs on every discovery because the relative
    size/frame thresholds can change when files are added or removed.  The
    expensive file reads (gate profile and spectral-axis inspection) are
    reused while a file's size and modification timestamp are unchanged.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[int, int, dict[str, object]]] = {}

    def get(self, identity: str, *, modified_ns: int, size_bytes: int) -> dict[str, object] | None:
        entry = self._entries.get(identity)
        if entry is None or entry[:2] != (modified_ns, size_bytes):
            return None
        return dict(entry[2])

    def put(
        self,
        identity: str,
        *,
        modified_ns: int,
        size_bytes: int,
        metadata: dict[str, object],
    ) -> None:
        self._entries[identity] = (modified_ns, size_bytes, dict(metadata))

    def retain(self, identities: set[str]) -> None:
        for identity in tuple(self._entries):
            if identity not in identities:
                self._entries.pop(identity, None)

    def clear(self) -> None:
        self._entries.clear()

    def clone(self) -> "DrrSourceCache":
        """Return an independent snapshot suitable for a background scan."""
        clone = DrrSourceCache()
        clone._entries = {
            identity: (modified_ns, size_bytes, dict(metadata))
            for identity, (modified_ns, size_bytes, metadata) in self._entries.items()
        }
        return clone


@dataclass(frozen=True)
class DrrSavedRecipe:
    measurement_files: tuple[str, ...]
    baseline_files: tuple[str, ...]
    baseline_selection: str
    baseline_which: str
    metadata_path: str
    saved_time: float


@dataclass(frozen=True)
class DrrGateProfile:
    varies: bool | None
    frame_count: int | None
    constant_values: tuple[float, ...] = ()
    gate_grid: tuple[tuple[float, ...], ...] = ()
    spectral_grid: tuple[float, ...] = ()
    gate_direction: str = ""
    gate_ranges: tuple[tuple[float, float], ...] = ()
    gate_labels: tuple[str, ...] = ()
    grid_complete: bool = False


@dataclass(frozen=True)
class DrrBackgroundGateAssessment:
    all_constant: bool
    same_constant_values: bool
    profiles: tuple[DrrGateProfile, ...]


@dataclass(frozen=True)
class DrrBackgroundGuess:
    baseline_files: tuple[str, ...]
    baseline_which: str
    reason: str
    time_gap_seconds: float
    intensity_difference_percent: float | None = None
    shape_correlation: float | None = None
    points_within_tolerance_percent: float | None = None
    confidence: str = "unknown"
    candidate_group_count: int = 0


def resolve_source_path(root: str | Path, source: str | Path) -> Path:
    path = Path(source)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def portable_source_name(root: str | Path, source: str | Path) -> str:
    path = Path(source).resolve()
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def is_background_name(name: str) -> bool:
    return bool(_BACKGROUND_TOKEN.search(Path(name).stem))


def extract_wavelength_center_nm(name: str) -> float | None:
    stem = Path(name).stem
    match = _WAVELENGTH_CENTER_TOKEN.search(stem) or _CENTER_WAVELENGTH_TOKEN.search(stem)
    if match is None:
        return None
    try:
        return float(match.group("value").replace("p", ".").replace("P", "."))
    except ValueError:
        return None


def inspect_csv_wavelength_center(path: str | Path) -> float | None:
    """Estimate the spectrometer center from the first-row spectral axis."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            first = next((line for line in handle if line.strip()), "")
    except OSError:
        return None
    if not first:
        return None
    delimiter = max((",", "\t", ";"), key=first.count)
    row = next(csv.reader([first], delimiter=delimiter), [])
    header_is_text = any(_as_float(token) is None for token in row)
    spectral_tokens = row if header_is_text else row[4:]
    values = [value for token in spectral_tokens if (value := _as_float(token)) is not None]
    wavelength_values = [value for value in values if 200.0 <= value <= 2500.0]
    if len(wavelength_values) >= 2:
        return 0.5 * (min(wavelength_values) + max(wavelength_values))
    energy_values = [value for value in values if 0.2 <= value <= 10.0]
    if len(energy_values) >= 2:
        wavelengths = [1240.0 / value for value in energy_values]
        return 0.5 * (min(wavelengths) + max(wavelengths))
    return None


def wavelength_centers_match(first: float, second: float, *, tolerance_nm: float = 1.0) -> bool:
    return abs(float(first) - float(second)) <= float(tolerance_nm)


def validate_named_wavelength_centers(
    measurement_files: Sequence[str], baseline_files: Sequence[str]
) -> float | None:
    measurement_centers = [
        center for name in measurement_files
        if (center := extract_wavelength_center_nm(name)) is not None
    ]
    baseline_centers = [
        center for name in baseline_files
        if (center := extract_wavelength_center_nm(name)) is not None
    ]
    expected = measurement_centers[0] if measurement_centers else None
    if expected is not None and any(
        not wavelength_centers_match(expected, center) for center in measurement_centers[1:]
    ):
        raise ValueError("Selected DRR measurement files use different wavelength centers.")
    if baseline_centers and any(
        not wavelength_centers_match(baseline_centers[0], center) for center in baseline_centers[1:]
    ):
        raise ValueError("Selected DRR background files use different wavelength centers.")
    if expected is not None and baseline_centers and not wavelength_centers_match(expected, baseline_centers[0]):
        raise ValueError(
            f"Background wavelength center must match the measurement: "
            f"measurement={expected:g} nm, background={baseline_centers[0]:g} nm."
        )
    return expected


def _metadata_source_name(root: Path, item: dict[str, object], *, require_exists: bool = False) -> str | None:
    """Resolve a metadata source using only the paths recorded in that item.

    Export metadata contains both an absolute path and a portable ``name``.
    A moved experiment can therefore recover the portable path when it exists,
    while an unrelated file with the same basename is never substituted.
    """
    candidates: list[str] = []
    for key in ("source_path", "path", "processing_input_path", "name"):
        value = item.get(key)
        # A basename-only ``name`` is insufficient evidence after a move.
        # Keep it only when it carries a relative directory or is the only
        # recorded path (legacy metadata).
        if key == "name" and isinstance(value, str) and not Path(value).parent.parts:
            if any(candidates):
                continue
        if isinstance(value, str) and value.strip() and value not in candidates:
            candidates.append(value)
    if not candidates:
        return None
    fallback: str | None = None
    for raw in candidates:
        resolved = resolve_source_path(root, raw)
        if fallback is None:
            fallback = portable_source_name(root, resolved)
        try:
            if resolved.is_file():
                return portable_source_name(root, resolved)
        except OSError:
            continue
    return None if require_exists else fallback


def _read_drr_metadata(
    root: Path, *, require_drr_operation: bool = False
) -> tuple[dict[str, set[str]], list[DrrSavedRecipe]]:
    """Read valid DR/R source roles and associations without trusting filenames."""
    roles: dict[str, set[str]] = {}
    records: list[DrrSavedRecipe] = []
    metadata_root = root / "Processed Data" / "DRR"
    if not metadata_root.is_dir():
        return roles, records
    try:
        metadata_files = metadata_root.rglob("*.metadata.json")
    except OSError:
        return roles, records
    for metadata_path in metadata_files:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        operation = payload.get("operation", payload.get("workflow"))
        if (
            operation != "DR/R"
            if require_drr_operation
            else operation not in (None, "DR/R")
        ):
            continue
        inputs = payload.get("sources", payload.get("inputs", []))
        if not isinstance(inputs, list):
            continue
        measurements: list[str] = []
        baselines: list[str] = []
        for item in inputs:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").casefold()
            if role not in {"measurement", "background"}:
                continue
            source = _metadata_source_name(root, item)
            if source is None:
                continue
            identity = str(resolve_source_path(root, source)).casefold()
            roles.setdefault(identity, set()).add(role)
            if role == "measurement":
                measurements.append(source)
            else:
                baselines.append(source)
        if not measurements:
            continue
        processing = payload.get("processing", {})
        if not isinstance(processing, dict):
            processing = {}
        try:
            saved_time = float(metadata_path.stat().st_mtime)
        except OSError:
            saved_time = 0.0
        records.append(
            DrrSavedRecipe(
                measurement_files=tuple(measurements),
                baseline_files=tuple(baselines),
                baseline_selection=str(
                    processing.get("baseline_selection")
                    or ("External" if baselines else "Self (last frame)")
                ),
                baseline_which=str(processing.get("baseline_which") or "last"),
                metadata_path=str(metadata_path),
                saved_time=saved_time,
            )
        )
    return roles, records


def assess_background_gate_files(
    root: str | Path,
    sources: Sequence[str],
    *,
    tolerance: float = 1e-6,
) -> DrrBackgroundGateAssessment:
    profiles: list[DrrGateProfile] = []
    for source in sources:
        path = resolve_source_path(root, source)
        if path.suffix.lower() != ".csv":
            return DrrBackgroundGateAssessment(False, False, tuple(profiles))
        profiles.append(inspect_csv_gate_profile(path, max_rows=None))
    all_constant = bool(profiles) and all(
        profile.varies is False and bool(profile.constant_values)
        for profile in profiles
    )
    if not all_constant:
        return DrrBackgroundGateAssessment(False, False, tuple(profiles))
    reference = profiles[0].constant_values
    same = all(
        len(profile.constant_values) == len(reference)
        and all(
            abs(float(value) - float(expected))
            <= tolerance * max(1.0, abs(float(value)), abs(float(expected)))
            for value, expected in zip(profile.constant_values, reference)
        )
        for profile in profiles[1:]
    )
    return DrrBackgroundGateAssessment(True, same, tuple(profiles))


def find_saved_drr_recipe(
    root: str | Path,
    measurement_files: Sequence[str],
) -> DrrSavedRecipe | None:
    """Return the newest exact saved recipe for a DRR measurement selection."""
    experiment_root = Path(root).resolve()
    metadata_root = experiment_root / "Processed Data" / "DRR"
    if not measurement_files or not metadata_root.is_dir():
        return None

    wanted = {
        str(resolve_source_path(experiment_root, source)).casefold()
        for source in measurement_files
    }
    _roles, records = _read_drr_metadata(experiment_root, require_drr_operation=True)
    matches: list[DrrSavedRecipe] = []
    for record in records:
        found = {
            str(resolve_source_path(experiment_root, source)).casefold()
            for source in record.measurement_files
        }
        if found != wanted:
            continue
        matches.append(DrrSavedRecipe(
            measurement_files=record.measurement_files,
            baseline_files=record.baseline_files,
            baseline_selection=record.baseline_selection,
            baseline_which=record.baseline_which,
            metadata_path=record.metadata_path,
            saved_time=record.saved_time,
        ))
    return max(matches, key=lambda recipe: recipe.saved_time, default=None)


def guess_drr_background(
    root: str | Path,
    catalog: Sequence[DrrSource],
    measurement_files: Sequence[str],
) -> DrrBackgroundGuess | None:
    """Choose the unmodified background group that best overlaps the raw data.

    Automatic selection is deliberately conservative: measurement and
    background spectra must use the same sampled photon-energy grid.  No
    rescaling, offset correction, fitting, or interpolation is performed.
    Time proximity is used only after the raw-intensity comparison.
    """
    selected = set(measurement_files)
    measurements = [source for source in catalog if source.source in selected]
    if not measurements or {source.source for source in measurements} != selected:
        return None
    centers = [
        float(source.wavelength_center_nm)
        for source in measurements
        if source.wavelength_center_nm is not None
    ]
    if len(centers) != len(measurements) or any(
        not wavelength_centers_match(centers[0], center)
        for center in centers[1:]
    ):
        return None
    measurement_time = min(source.modified_time for source in measurements)
    measurement_sessions = {source.session_date for source in measurements}
    measurement_folders = {str(Path(source.source).parent).casefold() for source in measurements}
    try:
        measurement_energy, measurement_spectra = _measurement_spectral_sample(
            root, measurement_files
        )
    except (OSError, ValueError):
        return None

    compatible = []
    for source in catalog:
        if not source.is_background or Path(source.source).suffix.lower() != ".csv":
            continue
        if (
            source.wavelength_center_nm is not None
            and not wavelength_centers_match(centers[0], source.wavelength_center_nm)
        ):
            continue
        compatible.append(source)

    scored: list[tuple[tuple[float, float, int, int, float], DrrSourceGroup, tuple[str, ...], str, str, tuple[float, float, float]]] = []
    for group in group_drr_sources(compatible):
        files = [source.source for source in group.files]
        gate = assess_background_gate_files(root, files)
        if gate.all_constant and gate.same_constant_values:
            chosen = tuple(files)
            which = "all"
            gate_reason = "constant matching gates; averaging all frames and files"
        elif gate.all_constant:
            latest = max(group.files, key=lambda source: source.modified_time)
            chosen = (latest.source,)
            which = "all"
            gate_reason = "different constant gates; using only the closest file"
        else:
            chosen = tuple(files)
            which = "last"
            gate_reason = "gate varies; using the last frame from each file"
        try:
            background_energy, background = _background_group_spectrum(
                root, chosen, which
            )
        except (OSError, ValueError):
            continue
        if not _same_spectral_grid(measurement_energy, background_energy):
            continue
        metrics = _raw_spectrum_overlap_metrics(measurement_spectra, background)
        if metrics is None:
            continue
        difference_percent, correlation, within_percent = metrics
        time_gap = abs(measurement_time - group.modified_time)
        same_session_penalty = 0 if group.session_date in measurement_sessions else 1
        same_folder_penalty = 0 if any(
            str(Path(source.source).parent).casefold() in measurement_folders
            for source in group.files
        ) else 1
        # Intensity overlap is primary. Session/folder and time only break close calls.
        rank = (
            difference_percent,
            -correlation,
            same_session_penalty,
            same_folder_penalty,
            time_gap,
        )
        scored.append((rank, group, chosen, which, gate_reason, metrics))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    _rank, best_group, chosen, which, gate_reason, metrics = scored[0]
    difference_percent, correlation, within_percent = metrics
    gap = abs(measurement_time - best_group.modified_time)
    confidence = (
        "high" if difference_percent <= 5.0 and correlation >= 0.98
        else "medium" if difference_percent <= 15.0 and correlation >= 0.90
        else "low"
    )
    return DrrBackgroundGuess(
        baseline_files=chosen,
        baseline_which=which,
        reason=(
            f"best raw-spectrum overlap among {len(scored)} exact-grid background "
            f"group{'s' if len(scored) != 1 else ''} at {centers[0]:g} nm; "
            f"median intensity difference {difference_percent:.2f}%, "
            f"shape correlation {correlation:.4f}; {gate_reason}"
        ),
        time_gap_seconds=gap,
        intensity_difference_percent=difference_percent,
        shape_correlation=correlation,
        points_within_tolerance_percent=within_percent,
        confidence=confidence,
        candidate_group_count=len(scored),
    )


def _same_spectral_grid(first: np.ndarray, second: np.ndarray) -> bool:
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    return (
        first.shape == second.shape
        and first.size >= 2
        and np.all(np.isfinite(first))
        and np.all(np.isfinite(second))
        and np.allclose(first, second, rtol=1e-7, atol=1e-10)
    )


def _read_csv_spectral_sample(
    path: str | Path, *, max_frames: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Read an original spectral grid and a deterministic sample of raw frames."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        first_line = next((line for line in handle if line.strip()), "")
        if not first_line:
            raise ValueError(f"Empty DRR file: {source.name}")
        delimiter = max((",", "\t", ";"), key=first_line.count)
        first_row = next(csv.reader([first_line], delimiter=delimiter), [])
        header_is_text = any(_as_float(token) is None for token in first_row)
        if header_is_text:
            spectral_indices = [
                index for index, token in enumerate(first_row)
                if _as_float(token) is not None
            ]
            spectral_axis = np.asarray(
                [float(_as_float(first_row[index])) for index in spectral_indices], float
            )
            rows = csv.reader(handle, delimiter=delimiter)
        else:
            numeric_header = np.asarray(
                [np.nan if _as_float(token) is None else float(_as_float(token)) for token in first_row],
                float,
            )
            starts = np.flatnonzero(np.isfinite(numeric_header) & (numeric_header > 50.0))
            start = int(starts[0]) if starts.size else 4
            spectral_indices = list(range(start, len(first_row)))
            spectral_axis = numeric_header[start:]
            rows = csv.reader(handle, delimiter=delimiter)
        if spectral_axis.size < 2:
            raise ValueError(f"No usable spectral grid in {source.name}")

        sampled: list[np.ndarray] = []
        seen = 0
        rng = random.Random(0)
        for row in rows:
            values = [
                _as_float(row[index]) if index < len(row) else None
                for index in spectral_indices
            ]
            frame = np.asarray(
                [np.nan if value is None else float(value) for value in values], float
            )
            if frame.size != spectral_axis.size or np.count_nonzero(np.isfinite(frame)) < 2:
                continue
            seen += 1
            if max_frames is None or len(sampled) < max_frames:
                sampled.append(frame)
            else:
                replacement = rng.randrange(seen)
                if replacement < max_frames:
                    sampled[replacement] = frame
        if not sampled:
            raise ValueError(f"No usable spectra in {source.name}")

    energy = 1240.0 / spectral_axis if float(np.nanmedian(spectral_axis)) > 20.0 else spectral_axis.copy()
    order = np.argsort(energy)
    return np.asarray(energy[order], float), np.asarray(sampled, float)[:, order]


def _measurement_spectral_sample(
    root: str | Path, sources: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    energy0: np.ndarray | None = None
    samples: list[np.ndarray] = []
    per_file_limit = max(8, 64 // max(1, len(sources)))
    for source in sources:
        energy, spectra = _read_csv_spectral_sample(
            resolve_source_path(root, source), max_frames=per_file_limit
        )
        if energy0 is None:
            energy0 = energy
        elif not _same_spectral_grid(energy0, energy):
            raise ValueError("Selected measurements do not use one exact spectral grid.")
        samples.append(spectra)
    if energy0 is None or not samples:
        raise ValueError("No measurement spectra available.")
    return energy0, np.concatenate(samples, axis=0)


def _background_group_spectrum(
    root: str | Path, sources: Sequence[str], which: str
) -> tuple[np.ndarray, np.ndarray]:
    energy0: np.ndarray | None = None
    per_file: list[np.ndarray] = []
    for source in sources:
        energy, spectra = _read_csv_spectral_sample(
            resolve_source_path(root, source), max_frames=None
        )
        if energy0 is None:
            energy0 = energy
        elif not _same_spectral_grid(energy0, energy):
            raise ValueError("Background group does not use one exact spectral grid.")
        if which == "first":
            per_file.append(spectra[0])
        elif which == "last":
            per_file.append(spectra[-1])
        else:
            per_file.append(np.nanmean(spectra, axis=0))
    if energy0 is None or not per_file:
        raise ValueError("No background spectra available.")
    return energy0, np.nanmean(np.stack(per_file, axis=0), axis=0)


def _raw_spectrum_overlap_metrics(
    measurement_spectra: np.ndarray, background: np.ndarray
) -> tuple[float, float, float] | None:
    differences: list[float] = []
    correlations: list[float] = []
    within: list[float] = []
    background = np.asarray(background, float).ravel()
    for spectrum in np.asarray(measurement_spectra, float):
        valid = np.isfinite(spectrum) & np.isfinite(background)
        if np.count_nonzero(valid) < 3:
            continue
        measured = np.asarray(spectrum[valid], float)
        baseline = background[valid]
        intensity_scale = float(np.nanmedian(np.abs(measured)))
        if not np.isfinite(intensity_scale) or intensity_scale <= 1e-12:
            continue
        residual = np.abs(measured - baseline)
        differences.append(100.0 * float(np.nanmedian(residual)) / intensity_scale)
        tolerance = np.maximum(0.05 * np.abs(measured), 0.005 * intensity_scale)
        within.append(100.0 * float(np.mean(residual <= tolerance)))
        if np.nanstd(measured) <= 1e-12 or np.nanstd(baseline) <= 1e-12:
            correlations.append(1.0 if np.allclose(measured, baseline) else 0.0)
        else:
            correlation = float(np.corrcoef(measured, baseline)[0, 1])
            correlations.append(correlation if np.isfinite(correlation) else -1.0)
    if not differences:
        return None
    return (
        float(np.nanmedian(differences)),
        float(np.nanmedian(correlations)),
        float(np.nanmedian(within)),
    )


def measurement_group_key(name: str) -> str:
    key, _sort = split_group_and_sort_key(Path(name).name)
    key = re.sub(
        r"(?:[_\-](?:run|scan|repeat|part)\d+)$",
        "",
        key,
        flags=re.IGNORECASE,
    )
    return key.strip(" _-") or Path(name).stem


def _as_float(value: str) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def inspect_csv_gate_profile(
    path: str | Path,
    *,
    max_rows: int | None = 32,
) -> DrrGateProfile:
    """Return gate variation, frame count, gate grid, and spectral axis.

    Callers that need a cheap preview can retain the default row limit.  The
    catalog passes ``max_rows=None`` so frame counts and acquisition grids are
    complete; the result is cached by file size and modification timestamp.
    """
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            lines = []
            for line in handle:
                if line.strip():
                    lines.append(line)
                if max_rows is not None and len(lines) >= max_rows + 1:
                    break
    except OSError:
        return DrrGateProfile(None, None)
    if len(lines) < 2:
        return DrrGateProfile(None, max(0, len(lines) - 1))

    first = lines[0]
    delimiter = max((",", "\t", ";"), key=first.count)
    rows = list(csv.reader(lines, delimiter=delimiter))
    if len(rows) < 2:
        return DrrGateProfile(None, max(0, len(rows) - 1))

    header = rows[0]
    header_is_text = any(_as_float(value) is None for value in header)
    if header_is_text:
        normalized = [re.sub(r"[^a-z0-9]+", "", str(value).casefold()) for value in header]
        gate_indices = [
            index
            for index, name in enumerate(normalized)
            if name.startswith(("vbg", "vtg", "vbias"))
            or name in {"gate", "gatev", "field", "efield", "doping"}
        ]
        data_rows = rows[1:]
        spectral_indices = [
            index for index, value in enumerate(header) if _as_float(value) is not None
        ]
        spectral_grid = tuple(float(_as_float(header[index])) for index in spectral_indices)
        gate_labels = tuple(str(header[index]).strip() for index in gate_indices)
    else:
        gate_indices = [index for index in (0, 1) if index < len(header)]
        data_rows = rows[1:]
        spectral_indices = [
            index for index, value in enumerate(header)
            if _as_float(value) is not None and float(_as_float(value)) > 50.0
        ]
        spectral_grid = tuple(float(_as_float(header[index])) for index in spectral_indices)
        gate_labels = tuple(f"gate {index + 1}" for index in range(len(gate_indices)))

    if not gate_indices:
        return DrrGateProfile(None, len(data_rows))
    varied = False
    found_gate_values = False
    constant_values: list[float] = []
    gate_grid: list[tuple[float, ...]] = []
    grid_complete = True
    for row in data_rows:
        values = tuple(
            float(_as_float(row[index]))
            for index in gate_indices
            if index < len(row) and _as_float(row[index]) is not None
        )
        if len(values) != len(gate_indices):
            grid_complete = False
            continue
        if spectral_indices and any(
            index >= len(row) or _as_float(row[index]) is None
            for index in spectral_indices
        ):
            grid_complete = False
            continue
        if values:
            gate_grid.append(values)
    for index in gate_indices:
        values = [value for row in data_rows if index < len(row) for value in [_as_float(row[index])] if value is not None]
        if not values:
            continue
        found_gate_values = True
        constant_values.append(statistics.fmean(values))
        if max(values) - min(values) > 1e-9 * max(1.0, abs(min(values)), abs(max(values))):
            varied = True
    direction = ""
    if gate_grid and len(gate_grid[0]) >= 1 and len(gate_grid) > 1:
        directions: list[str] = []
        for column in range(len(gate_grid[0])):
            values = [row[column] for row in gate_grid]
            if all(second > first for first, second in zip(values, values[1:])):
                directions.append("increasing")
            elif all(second < first for first, second in zip(values, values[1:])):
                directions.append("decreasing")
            elif all(second == first for first, second in zip(values, values[1:])):
                directions.append("constant")
            else:
                directions.append("mixed")
        direction = ", ".join(
            f"{gate_labels[index] if index < len(gate_labels) else f'gate {index + 1}'} {value}"
            for index, value in enumerate(directions)
        )
    return DrrGateProfile(
        varied if found_gate_values else None,
        len(data_rows),
        tuple(constant_values) if found_gate_values and not varied else (),
        tuple(gate_grid),
        spectral_grid,
        direction,
        tuple(
            (min(row[index] for row in gate_grid), max(row[index] for row in gate_grid))
            for index in range(len(gate_indices))
        ) if gate_grid else (),
        gate_labels,
        grid_complete and bool(spectral_grid),
    )


def inspect_csv_gate(
    path: str | Path,
    *,
    max_rows: int | None = 32,
    include_profile: bool = False,
) -> tuple[bool | None, int | None] | DrrGateProfile:
    profile = inspect_csv_gate_profile(path, max_rows=max_rows)
    return profile if include_profile else (profile.varies, profile.frame_count)


def _processed_measurement_paths(root: Path) -> set[str]:
    roles, _records = _read_drr_metadata(root)
    return {
        identity for identity, values in roles.items() if "measurement" in values
    }


def discover_drr_sources(
    root: str | Path,
    *,
    cache: DrrSourceCache | None = None,
) -> list[DrrSource]:
    """Discover DRR files, reusing unchanged per-file inspections when possible."""
    experiment_root = Path(root).resolve()
    if not experiment_root.is_dir():
        if cache is not None:
            cache.clear()
        return []
    metadata_roles, metadata_records = _read_drr_metadata(experiment_root)
    processed_paths = {
        identity for identity, values in metadata_roles.items() if "measurement" in values
    }
    metadata_links: dict[str, list[DrrSavedRecipe]] = {}
    for record in metadata_records:
        for measurement in record.measurement_files:
            identity = str(resolve_source_path(experiment_root, measurement)).casefold()
            metadata_links.setdefault(identity, []).append(record)
    candidates: list[Path] = []
    try:
        candidates.extend(path for path in experiment_root.iterdir() if path.is_file())
    except OSError:
        return []
    initial_root = experiment_root / "Initial Data"
    if initial_root.is_dir():
        try:
            candidates.extend(path for path in initial_root.rglob("*") if path.is_file())
        except OSError:
            pass

    inspected: list[dict] = []
    seen: set[str] = set()
    for path in candidates:
        if path.suffix.lower() not in SUPPORTED_DRR_SUFFIXES:
            continue
        identity = str(path.resolve()).casefold()
        if identity in seen:
            continue
        seen.add(identity)
        try:
            stat = path.stat()
            modified = float(stat.st_mtime)
            size_bytes = int(stat.st_size)
            modified_ns = int(getattr(stat, "st_mtime_ns", round(modified * 1e9)))
        except OSError:
            modified = 0.0
            size_bytes = 0
            modified_ns = 0
        cached = (
            cache.get(identity, modified_ns=modified_ns, size_bytes=size_bytes)
            if cache is not None
            else None
        )
        if cached is None:
            profile = (
                inspect_csv_gate(path, max_rows=None, include_profile=True)
                if path.suffix.lower() == ".csv"
                else DrrGateProfile(None, None)
            )
            gate_varies, frame_count = profile.varies, profile.frame_count
            named_center = extract_wavelength_center_nm(path.name)
            spectral_center = (
                inspect_csv_wavelength_center(path)
                if path.suffix.lower() == ".csv" and named_center is None
                else None
            )
            cached = {
                "gate_varies": gate_varies,
                "frame_count": frame_count,
                "gate_grid": profile.gate_grid,
                "spectral_grid": profile.spectral_grid,
                "gate_direction": profile.gate_direction,
                "gate_ranges": profile.gate_ranges,
                "gate_labels": profile.gate_labels,
                "grid_complete": profile.grid_complete,
                "wavelength_center_nm": named_center if named_center is not None else spectral_center,
                "wavelength_center_source": (
                    "filename" if named_center is not None
                    else "spectral axis" if spectral_center is not None
                    else ""
                ),
            }
            if cache is not None:
                cache.put(
                    identity,
                    modified_ns=modified_ns,
                    size_bytes=size_bytes,
                    metadata=cached,
                )
        inspected.append(
            {
                "path": path,
                "identity": identity,
                "modified": modified,
                "size_bytes": size_bytes,
                **cached,
            }
        )

    if cache is not None:
        cache.retain(seen)

    csv_sizes = [item["size_bytes"] for item in inspected if item["path"].suffix.lower() == ".csv" and item["size_bytes"] > 0]
    frame_counts = [item["frame_count"] for item in inspected if item["frame_count"] is not None and item["frame_count"] > 0]
    median_size = float(statistics.median(csv_sizes)) if csv_sizes else 0.0
    median_frames = float(statistics.median(frame_counts)) if frame_counts else 0.0

    sources: list[DrrSource] = []
    for item in inspected:
        path = item["path"]
        source_roles = tuple(sorted(metadata_roles.get(item["identity"], ())))
        metadata_role = (
            "measurement" if "measurement" in source_roles
            else "background" if "background" in source_roles
            else ""
        )
        named_background = is_background_name(path.name)
        small_size = median_size > 0 and item["size_bytes"] <= 0.35 * median_size
        small_frames = (
            item["frame_count"] is not None
            and item["frame_count"] <= max(3, int(0.25 * median_frames))
        )
        if metadata_role == "measurement":
            classification = "measurement"
            reason = "saved metadata role: measurement"
        elif metadata_role == "background":
            classification = "background"
            reason = "saved metadata role: background"
        elif named_background:
            classification = "background"
            reason = "background keyword in filename"
        elif item["gate_varies"] is False and (small_size or small_frames):
            classification = "likely_background"
            reasons = ["gate does not vary"]
            if small_size:
                reasons.append("file is unusually small")
            if small_frames:
                reasons.append("few frames")
            reason = "; ".join(reasons)
        elif item["gate_varies"] is False:
            classification = "review"
            reason = "gate does not vary, but file size/frame count is not unusually small"
        else:
            classification = "measurement"
            reason = "gate varies" if item["gate_varies"] is True else "no background indicators found"
        is_background = classification in {"background", "likely_background"}
        links = metadata_links.get(item["identity"], [])
        linked_backgrounds = tuple(dict.fromkeys(
            baseline for record in links for baseline in record.baseline_files
        ))
        saved_modes = tuple(dict.fromkeys(record.baseline_selection for record in links))
        sources.append(
            DrrSource(
                source=portable_source_name(experiment_root, path),
                filename=path.name,
                group_key=measurement_group_key(path.name),
                session_date=datetime.fromtimestamp(item["modified"]).strftime("%Y-%m-%d") if item["modified"] else "Unknown date",
                modified_time=item["modified"],
                is_background=is_background,
                processed=item["identity"] in processed_paths,
                classification=classification,
                classification_reason=reason,
                gate_varies=item["gate_varies"],
                frame_count=item["frame_count"],
                size_bytes=item["size_bytes"],
                wavelength_center_nm=item["wavelength_center_nm"],
                wavelength_center_source=item["wavelength_center_source"],
                metadata_role=metadata_role,
                linked_backgrounds=linked_backgrounds,
                saved_baseline_modes=saved_modes,
                gate_grid=item.get("gate_grid", ()),
                spectral_grid=item.get("spectral_grid", ()),
                gate_direction=item.get("gate_direction", ""),
                gate_ranges=item.get("gate_ranges", ()),
                gate_labels=item.get("gate_labels", ()),
                grid_complete=bool(item.get("grid_complete", False)),
            )
        )
    return sorted(sources, key=lambda item: (-item.modified_time, item.filename.casefold()))


def group_drr_sources(sources: Sequence[DrrSource]) -> list[DrrSourceGroup]:
    grouped: dict[tuple[str, str, bool], list[DrrSource]] = {}
    for source in sources:
        grouped.setdefault((source.session_date, source.group_key, source.is_background), []).append(source)
    result: list[DrrSourceGroup] = []
    for (session_date, key, is_background), files in grouped.items():
        ordered = tuple(sorted(files, key=lambda item: (item.filename.casefold(), item.modified_time)))
        latest = max((item.modified_time for item in ordered), default=0.0)
        frame_counts = [item.frame_count for item in ordered if item.frame_count is not None]
        linked_backgrounds = tuple(dict.fromkeys(
            path for item in ordered for path in item.linked_backgrounds
        ))
        saved_modes = tuple(dict.fromkeys(
            mode for item in ordered for mode in item.saved_baseline_modes
        ))
        directions = {item.gate_direction for item in ordered if item.gate_direction}
        labels = next((item.gate_labels for item in ordered if item.gate_labels), ())
        ranges = next((item.gate_ranges for item in ordered if item.gate_ranges), ())
        complete = bool(ordered) and all(item.grid_complete for item in ordered)
        result.append(
            DrrSourceGroup(
                key=f"{session_date}|{key}|{'background' if is_background else 'measurement'}",
                title=key,
                session_date=session_date,
                files=ordered,
                modified_time=latest,
                is_background=is_background,
                processed=all(item.processed for item in ordered),
                classification=(
                    "background"
                    if any(item.classification == "background" for item in ordered)
                    else "likely_background"
                    if any(item.classification == "likely_background" for item in ordered)
                    else "review"
                    if any(item.classification == "review" for item in ordered)
                    else "measurement"
                ),
                wavelength_centers_nm=tuple(sorted({
                    float(item.wavelength_center_nm)
                    for item in ordered
                    if item.wavelength_center_nm is not None
                })),
                processed_count=sum(1 for item in ordered if item.processed),
                frame_count_range=(min(frame_counts), max(frame_counts)) if frame_counts else None,
                linked_backgrounds=linked_backgrounds,
                saved_baseline_modes=saved_modes,
                gate_direction=next(iter(directions)) if len(directions) == 1 else "mixed" if directions else "",
                gate_ranges=ranges,
                gate_labels=labels,
                grid_complete=complete,
            )
        )
    return sorted(result, key=lambda item: (-item.modified_time, item.title.casefold()))


def compatible_drr_repeats(
    catalog: Sequence[DrrSource], reference_file: str
) -> tuple[str, ...]:
    """Return files compatible with a selected reference's full acquisition grid.

    Compatibility requires a known, ordered gate grid, an identical spectral
    axis, and the existing filename-derived condition group.  Unknown grids
    produce no automatic additions, leaving manual selection available.
    """
    reference = next((item for item in catalog if item.source == reference_file), None)
    if reference is None or reference.is_background:
        return ()
    if not reference.grid_complete or not reference.gate_grid or len(reference.spectral_grid) < 2:
        return ()
    compatible: list[DrrSource] = []
    for source in catalog:
        if source.is_background or source.group_key != reference.group_key:
            continue
        if not source.grid_complete or not source.gate_grid or len(source.spectral_grid) < 2:
            continue
        if len(source.gate_grid) != len(reference.gate_grid):
            continue
        reference_gate_array = np.asarray(reference.gate_grid, float)
        source_gate_array = np.asarray(source.gate_grid, float)
        if reference_gate_array.shape != source_gate_array.shape:
            continue
        if tuple(label.casefold() for label in source.gate_labels) != tuple(
            label.casefold() for label in reference.gate_labels
        ):
            continue
        if len(source.spectral_grid) != len(reference.spectral_grid):
            continue
        if not np.allclose(
            source_gate_array,
            reference_gate_array,
            rtol=1e-9,
            atol=1e-10,
        ):
            continue
        if not np.allclose(
            np.asarray(source.spectral_grid, float),
            np.asarray(reference.spectral_grid, float),
            rtol=1e-9,
            atol=1e-10,
        ):
            continue
        compatible.append(source)
    return tuple(item.source for item in sorted(compatible, key=lambda item: (item.filename.casefold(), item.modified_time)))

def newest_measurement_group(
    groups: Iterable[DrrSourceGroup], *, prefer_unprocessed: bool = True
) -> DrrSourceGroup | None:
    measurements = [group for group in groups if not group.is_background]
    if prefer_unprocessed:
        unprocessed = [group for group in measurements if not group.processed]
        if unprocessed:
            measurements = unprocessed
    return max(measurements, key=lambda item: item.modified_time, default=None)
