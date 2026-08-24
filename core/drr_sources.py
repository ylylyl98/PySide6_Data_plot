from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import json
import math
from pathlib import Path
import re
import statistics
from typing import Iterable, Sequence

from core.processing import split_group_and_sort_key


SUPPORTED_DRR_SUFFIXES = {".csv", ".xlsx"}
_BACKGROUND_TOKEN = re.compile(r"(?:^|[_\-\s])(back(?:ground)?|bg|dark|i0|reference|ref)(?:$|[_\-\s])", re.IGNORECASE)
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
    matches: list[DrrSavedRecipe] = []
    try:
        metadata_files = metadata_root.rglob("*.metadata.json")
    except OSError:
        return None
    for metadata_path in metadata_files:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if payload.get("operation") != "DR/R":
            continue
        inputs = payload.get("sources", payload.get("inputs", []))
        if not isinstance(inputs, list):
            continue
        measurements: list[str] = []
        baselines: list[str] = []
        for item in inputs:
            if not isinstance(item, dict):
                continue
            raw = item.get("source_path") or item.get("path") or item.get("name")
            if not raw:
                continue
            portable = portable_source_name(
                experiment_root,
                resolve_source_path(experiment_root, str(raw)),
            )
            if item.get("role") == "measurement":
                measurements.append(portable)
            elif item.get("role") == "background":
                baselines.append(portable)
        found = {
            str(resolve_source_path(experiment_root, source)).casefold()
            for source in measurements
        }
        if found != wanted:
            continue
        processing = payload.get("processing", {})
        if not isinstance(processing, dict):
            processing = {}
        try:
            saved_time = float(metadata_path.stat().st_mtime)
        except OSError:
            saved_time = 0.0
        matches.append(
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
    return max(matches, key=lambda recipe: recipe.saved_time, default=None)


def guess_drr_background(
    root: str | Path,
    catalog: Sequence[DrrSource],
    measurement_files: Sequence[str],
) -> DrrBackgroundGuess | None:
    """Guess the closest earlier wavelength-compatible background group."""
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
    measurement_start = min(source.modified_time for source in measurements)
    compatible = [
        source
        for source in catalog
        if source.is_background
        and source.modified_time <= measurement_start
        and source.wavelength_center_nm is not None
        and wavelength_centers_match(centers[0], source.wavelength_center_nm)
    ]
    groups = group_drr_sources(compatible)
    if not groups:
        return None
    closest = max(groups, key=lambda group: group.modified_time)
    files = [source.source for source in closest.files]
    gate = assess_background_gate_files(root, files)
    if gate.all_constant and gate.same_constant_values:
        chosen = tuple(files)
        which = "all"
        gate_reason = "constant matching gates; averaging all frames and files"
    elif gate.all_constant:
        latest = max(closest.files, key=lambda source: source.modified_time)
        chosen = (latest.source,)
        which = "all"
        gate_reason = "different constant gates; using only the closest file"
    else:
        chosen = tuple(files)
        which = "last"
        gate_reason = "gate varies; using the last frame from each file"
    gap = max(0.0, measurement_start - closest.modified_time)
    return DrrBackgroundGuess(
        baseline_files=chosen,
        baseline_which=which,
        reason=(
            f"closest earlier background at {centers[0]:g} nm; {gate_reason}"
        ),
        time_gap_seconds=gap,
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
    """Return gate variation, sampled frame count, and constant gate values.

    This intentionally samples only the start of each file so catalog refreshes
    remain quick even when Initial Data contains a long measurement history.
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
    else:
        gate_indices = [index for index in (0, 1) if index < len(header)]
        data_rows = rows[1:]

    if not gate_indices:
        return DrrGateProfile(None, len(data_rows))
    varied = False
    found_gate_values = False
    constant_values: list[float] = []
    for index in gate_indices:
        values = [value for row in data_rows if index < len(row) for value in [_as_float(row[index])] if value is not None]
        if not values:
            continue
        found_gate_values = True
        constant_values.append(statistics.fmean(values))
        if max(values) - min(values) > 1e-9 * max(1.0, abs(min(values)), abs(max(values))):
            varied = True
    return DrrGateProfile(
        varied if found_gate_values else None,
        len(data_rows),
        tuple(constant_values) if found_gate_values and not varied else (),
    )


def inspect_csv_gate(path: str | Path, *, max_rows: int = 32) -> tuple[bool | None, int | None]:
    profile = inspect_csv_gate_profile(path, max_rows=max_rows)
    return profile.varies, profile.frame_count


def _processed_measurement_paths(root: Path) -> set[str]:
    processed: set[str] = set()
    metadata_root = root / "Processed Data" / "DRR"
    if not metadata_root.is_dir():
        return processed
    try:
        metadata_files = metadata_root.rglob("*.metadata.json")
    except OSError:
        return processed
    for metadata_file in metadata_files:
        try:
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        for item in payload.get("sources", payload.get("inputs", [])):
            if not isinstance(item, dict) or item.get("role") != "measurement":
                continue
            raw = item.get("source_path") or item.get("path") or item.get("name")
            if not raw:
                continue
            path = resolve_source_path(root, str(raw))
            processed.add(str(path).casefold())
    return processed


def discover_drr_sources(root: str | Path) -> list[DrrSource]:
    experiment_root = Path(root).resolve()
    if not experiment_root.is_dir():
        return []
    processed_paths = _processed_measurement_paths(experiment_root)
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
        except OSError:
            modified = 0.0
            size_bytes = 0
        gate_varies, frame_count = (
            inspect_csv_gate(path) if path.suffix.lower() == ".csv" else (None, None)
        )
        named_center = extract_wavelength_center_nm(path.name)
        spectral_center = (
            inspect_csv_wavelength_center(path)
            if path.suffix.lower() == ".csv" and named_center is None
            else None
        )
        inspected.append(
            {
                "path": path,
                "identity": identity,
                "modified": modified,
                "size_bytes": size_bytes,
                "gate_varies": gate_varies,
                "frame_count": frame_count,
                "wavelength_center_nm": named_center if named_center is not None else spectral_center,
                "wavelength_center_source": "filename" if named_center is not None else "spectral axis" if spectral_center is not None else "",
            }
        )

    csv_sizes = [item["size_bytes"] for item in inspected if item["path"].suffix.lower() == ".csv" and item["size_bytes"] > 0]
    frame_counts = [item["frame_count"] for item in inspected if item["frame_count"] is not None and item["frame_count"] > 0]
    median_size = float(statistics.median(csv_sizes)) if csv_sizes else 0.0
    median_frames = float(statistics.median(frame_counts)) if frame_counts else 0.0

    sources: list[DrrSource] = []
    for item in inspected:
        path = item["path"]
        named_background = is_background_name(path.name)
        small_size = median_size > 0 and item["size_bytes"] <= 0.35 * median_size
        small_frames = (
            item["frame_count"] is not None
            and item["frame_count"] <= max(3, int(0.25 * median_frames))
        )
        if named_background:
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
            )
        )
    return sorted(result, key=lambda item: (-item.modified_time, item.title.casefold()))


def newest_measurement_group(
    groups: Iterable[DrrSourceGroup], *, prefer_unprocessed: bool = True
) -> DrrSourceGroup | None:
    measurements = [group for group in groups if not group.is_background]
    if prefer_unprocessed:
        unprocessed = [group for group in measurements if not group.processed]
        if unprocessed:
            measurements = unprocessed
    return max(measurements, key=lambda item: item.modified_time, default=None)
