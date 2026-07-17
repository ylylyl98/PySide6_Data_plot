from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.signal import savgol_filter

from core.loader import DataCube


REP_SUB_SUFFIX_RE = re.compile(r"_rep(?P<rep>\d+)_(?P<sub>\d+)$", re.IGNORECASE)
REP_ONLY_SUFFIX_RE = re.compile(r"(?:\$_|_)rep(?P<rep>\d{1,3})$", re.IGNORECASE)
RUN_SUFFIX_RE = re.compile(r"(?:\$_|_)(?P<run>\d{3,})$")
POWER_TOKEN_RE = re.compile(r"(?P<power>[+-]?\d+(?:[pP\.]\d+)?)\s*uW", re.IGNORECASE)
STAGE_TOKEN_RE = re.compile(r"(?:^|[_\s-]+)Stage[+-]?\d+(?:[pP\.]\d+)?", re.IGNORECASE)
COMPARE_GATE_CONDITION_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<condition>(?:(?P<tg>\d+(?:[pP\.]\d+)?)\s*)?TG\s*(?P<sign>[+-])\s*"
    r"(?:(?P<bg>\d+(?:[pP\.]\d+)?)\s*)?BG\s*=\s*0(?:\.0+)?)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_GATE_SIGN_TRANSLATION = str.maketrans(
    {
        "\N{MINUS SIGN}": "-",
        "\N{HYPHEN}": "-",
        "\N{NON-BREAKING HYPHEN}": "-",
        "\N{FIGURE DASH}": "-",
        "\N{EN DASH}": "-",
        "\N{EM DASH}": "-",
        "\N{SMALL HYPHEN-MINUS}": "-",
        "\N{FULLWIDTH HYPHEN-MINUS}": "-",
        "\N{FULLWIDTH PLUS SIGN}": "+",
    }
)


def _normalize_gate_signs(text: str) -> str:
    return str(text or "").translate(_GATE_SIGN_TRANSLATION)


@dataclass(frozen=True)
class Limits:
    vmin: float
    vmax: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass(frozen=True)
class PowerSeriesFile:
    file_name: str
    power_uW: float
    stage: float | None
    group_key: str
    title: str


@dataclass(frozen=True)
class PowerSweepPoint:
    """One spectrum row from a single-file power sweep table."""

    file_name: str
    power_uW: float
    stage: float | None
    row_index: int
    stage_column: str | None = None


@dataclass(frozen=True)
class PowerStagePair:
    stage: float
    power_kk: float
    power_kkp: float
    power_axis: float
    kk_file: str
    kkp_file: str
    kk_row: int | None = None
    kkp_row: int | None = None


@dataclass(frozen=True)
class RotationAngles:
    """Independently detected compare-arm rotation angles."""

    rot1: float | None = None
    rot2: float | None = None


@dataclass(frozen=True)
class AngleStateMatch:
    """Nearest explicit K/Kp reference result for one rotation arm."""

    angle: float
    state: str | None
    distance_k: float
    distance_kp: float | None
    reason: str


@dataclass(frozen=True)
class CompareAngleInference:
    """Editable reference suggestions inferred from filename angle clusters."""

    rot1_clusters: tuple[float, ...]
    rot2_clusters: tuple[float, ...]
    in_k: float | None = None
    in_kp: float | None = None
    out_k: float | None = None
    out_kp: float | None = None


def _parse_power_number(text: str) -> float:
    return float(str(text).replace("p", ".").replace("P", "."))


def _compact_power_group_stem(file_name: str) -> str:
    stem = Path(file_name).stem.replace("$", "")
    stem = POWER_TOKEN_RE.sub("", stem)
    stem = STAGE_TOKEN_RE.sub("", stem)
    stem = re.sub(r'[<>:"/\\|?*]+', "_", stem)
    stem = re.sub(r"_+", "_", stem)
    stem = re.sub(r"[\s-]+", "_", stem)
    return stem.strip("_")


def power_group_title(group_key: str) -> str:
    raw = str(group_key)
    if raw.startswith("csv::"):
        raw = Path(raw[5:]).stem
    title = raw.replace("_", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title


def parse_power_series_file(file_name: str) -> PowerSeriesFile | None:
    stem = Path(file_name).stem
    power_match = POWER_TOKEN_RE.search(stem)
    if not power_match:
        return None
    stage_match = re.search(r"Stage(?P<stage>[+-]?\d+(?:[pP\.]\d+)?)", stem, flags=re.IGNORECASE)
    group_key = _compact_power_group_stem(file_name)
    if not group_key:
        return None
    return PowerSeriesFile(
        file_name=file_name,
        power_uW=_parse_power_number(power_match.group("power")),
        stage=_parse_power_number(stage_match.group("stage")) if stage_match else None,
        group_key=group_key,
        title=power_group_title(group_key),
    )


def group_power_series_files(files: Sequence[str]) -> Dict[str, List[PowerSeriesFile]]:
    groups: Dict[str, List[PowerSeriesFile]] = {}
    for file_name in files:
        parsed = parse_power_series_file(file_name)
        if parsed is None:
            continue
        groups.setdefault(parsed.group_key, []).append(parsed)

    out: Dict[str, List[PowerSeriesFile]] = {}
    for key in sorted(groups):
        out[key] = sorted(
            groups[key],
            key=lambda item: (
                float(item.power_uW),
                float(item.stage) if item.stage is not None else 0.0,
                item.file_name,
            ),
        )
    return out


def split_group_and_sort_key(filename: str) -> tuple[str, tuple[int, int, int]]:
    stem = Path(filename).stem

    m = REP_SUB_SUFFIX_RE.search(stem)
    if m:
        condition_key = re.sub(r"_rep\d+_\d+$", "", stem, flags=re.IGNORECASE)
        return condition_key, (0, int(m.group("rep")), int(m.group("sub")))

    m = REP_ONLY_SUFFIX_RE.search(stem)
    if m:
        prefix = stem[: m.start()]
        return prefix, (1, int(m.group("rep")), 0)

    m = RUN_SUFFIX_RE.search(stem)
    if m:
        prefix = stem[: m.start()]
        return prefix, (1, int(m.group("run")), 0)

    return stem, (2, 0, 0)


def group_measurement_files(files: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[tuple[tuple[int, int, int], str]]] = {}
    for file_name in files:
        gk, sort_key = split_group_and_sort_key(file_name)
        groups.setdefault(gk, []).append((sort_key, file_name))

    out: Dict[str, List[str]] = {}
    for key in sorted(groups):
        out[key] = [name for _, name in sorted(groups[key], key=lambda item: item[0])]
    return out


def compute_auto_limits(cube: DataCube, *, log_scale: bool = False, low: float = 0.01, high: float = 99.99) -> Limits:
    z = np.asarray(cube.Z, float)
    e = np.asarray(cube.energy, float).ravel()
    g = np.asarray(cube.gate, float).ravel()
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        raise ValueError("Data contains no finite values.")

    if log_scale:
        pos = z[np.isfinite(z) & (z > 0)]
        if pos.size:
            vmin, vmax = np.nanpercentile(pos, [low, high])
            vmin = float(max(vmin, 1e-12))
            vmax = float(max(vmax, vmin * 1.01))
        else:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    else:
        vmin, vmax = np.nanpercentile(finite, [low, high])
        vmin, vmax = float(vmin), float(vmax)

    return Limits(
        vmin=vmin,
        vmax=vmax,
        xmin=float(np.nanmin(e)),
        xmax=float(np.nanmax(e)),
        ymin=float(np.nanmin(g)),
        ymax=float(np.nanmax(g)),
    )


def nearest_gate_spectrum(cube: DataCube, gate_value: float) -> tuple[float, np.ndarray]:
    gate = np.asarray(cube.gate, float).ravel()
    idx = int(np.argmin(np.abs(gate - float(gate_value))))
    return float(gate[idx]), np.asarray(cube.Z, float)[idx, :]


def _parse_angle_token(text: str) -> float:
    return float(str(text).replace("p", ".").replace("P", "."))


def parse_compare_rotation_angles(file_name: str) -> RotationAngles:
    text = str(Path(file_name).stem)
    number = r"([\-+]?\d+(?:[pP\.]\d+)?)"
    unit = r"\s*(?:deg|degree)(?=$|[^A-Za-z0-9])"
    named_prefix = r"(?:^|[_\s-])"
    in_match = re.search(rf"{named_prefix}in[^0-9\-+]*{number}{unit}", text, flags=re.IGNORECASE)
    out_match = re.search(rf"{named_prefix}out[^0-9\-+]*{number}{unit}", text, flags=re.IGNORECASE)
    rot1 = re.search(rf"rot\s*1[^0-9\-+]*{number}{unit}", text, flags=re.IGNORECASE)
    rot2 = re.search(rf"rot\s*2[^0-9\-+]*{number}{unit}", text, flags=re.IGNORECASE)
    return RotationAngles(
        rot1=(
            _parse_angle_token(in_match.group(1))
            if in_match
            else (_parse_angle_token(rot1.group(1)) if rot1 else None)
        ),
        rot2=(
            _parse_angle_token(out_match.group(1))
            if out_match
            else (_parse_angle_token(rot2.group(1)) if rot2 else None)
        ),
    )


def parse_compare_in_out_angles(file_name: str) -> tuple[float | None, float | None]:
    """Backward-compatible tuple view of independently parsed Rot1/Rot2 angles."""
    angles = parse_compare_rotation_angles(file_name)
    return angles.rot1, angles.rot2


def circular_angle_distance(a: float, b: float, *, period: float = 360.0) -> float:
    period = float(period)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("Angle period must be finite and positive.")
    return abs(((float(a) - float(b) + 0.5 * period) % period) - 0.5 * period)


def classify_angle_state(
    angle: float,
    *,
    k_angle: float,
    kp_angle: float | None,
    tolerance: float,
    ambiguity_margin: float = 1.0,
    period: float = 360.0,
) -> AngleStateMatch:
    """Classify one angle by nearest explicit reference, with safe rejection."""
    angle = float(angle)
    tolerance = float(tolerance)
    ambiguity_margin = float(ambiguity_margin)
    if not np.isfinite(angle) or not np.isfinite(k_angle):
        raise ValueError("Angle references must be finite.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Angle tolerance must be finite and positive.")
    if not np.isfinite(ambiguity_margin) or ambiguity_margin < 0.0:
        raise ValueError("Angle ambiguity margin must be finite and non-negative.")

    distance_k = circular_angle_distance(angle, float(k_angle), period=period)
    if kp_angle is None:
        # Compatibility for non-UI callers that still provide only a K anchor.
        state = "K" if distance_k < tolerance else "Kp"
        return AngleStateMatch(angle, state, distance_k, None, "legacy-single-reference")

    if not np.isfinite(kp_angle):
        raise ValueError("Kp angle reference must be finite.")
    distance_kp = circular_angle_distance(angle, float(kp_angle), period=period)
    nearest = min(distance_k, distance_kp)
    if nearest > tolerance:
        return AngleStateMatch(angle, None, distance_k, distance_kp, "outside-tolerance")
    if abs(distance_k - distance_kp) <= ambiguity_margin:
        return AngleStateMatch(angle, None, distance_k, distance_kp, "ambiguous")
    state = "K" if distance_k < distance_kp else "Kp"
    return AngleStateMatch(angle, state, distance_k, distance_kp, "nearest-reference")


def _circular_cluster_center(values: Sequence[float], *, period: float) -> float:
    radians = np.asarray(values, float) * (2.0 * np.pi / float(period))
    mean_angle = np.arctan2(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians))))
    return float((mean_angle * float(period) / (2.0 * np.pi)) % float(period))


def _cluster_compare_angles(
    values: Sequence[float],
    *,
    tolerance: float,
    period: float,
) -> tuple[float, ...]:
    finite = sorted({float(value) for value in values if np.isfinite(value)})
    if not finite:
        return ()
    parents = list(range(len(finite)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(finite)):
        for right in range(left + 1, len(finite)):
            if circular_angle_distance(finite[left], finite[right], period=period) <= float(tolerance):
                union(left, right)

    groups: dict[int, list[float]] = {}
    for index, value in enumerate(finite):
        groups.setdefault(find(index), []).append(value)
    centers = [_circular_cluster_center(group, period=period) for group in groups.values()]
    return tuple(sorted(centers))


def infer_compare_angle_references(
    file_names: Sequence[str],
    *,
    in_k_anchor: float,
    out_k_anchor: float,
    cluster_tolerance: float = 15.0,
    period: float = 360.0,
) -> CompareAngleInference:
    """Suggest K/Kp anchors only when exactly two clusters exist for an arm."""
    parsed = [parse_compare_rotation_angles(file_name) for file_name in file_names]
    rot1_clusters = _cluster_compare_angles(
        [item.rot1 for item in parsed if item.rot1 is not None],
        tolerance=cluster_tolerance,
        period=period,
    )
    rot2_clusters = _cluster_compare_angles(
        [item.rot2 for item in parsed if item.rot2 is not None],
        tolerance=cluster_tolerance,
        period=period,
    )

    def assign(clusters: tuple[float, ...], anchor: float) -> tuple[float | None, float | None]:
        if len(clusters) != 2:
            return None, None
        k_index = min(
            range(2),
            key=lambda index: circular_angle_distance(clusters[index], anchor, period=period),
        )
        return clusters[k_index], clusters[1 - k_index]

    in_k, in_kp = assign(rot1_clusters, float(in_k_anchor))
    out_k, out_kp = assign(rot2_clusters, float(out_k_anchor))
    return CompareAngleInference(
        rot1_clusters=rot1_clusters,
        rot2_clusters=rot2_clusters,
        in_k=in_k,
        in_kp=in_kp,
        out_k=out_k,
        out_kp=out_kp,
    )



def parse_compare_gate_condition(file_name: str) -> str:
    stem = _normalize_gate_signs(Path(file_name).stem)
    match = COMPARE_GATE_CONDITION_RE.search(stem)
    if not match:
        return ""
    token = re.sub(r"\s+", "", match.group("condition"))
    token = re.sub("tg", "TG", token, flags=re.IGNORECASE)
    token = re.sub("bg", "BG", token, flags=re.IGNORECASE)
    return token


def classify_compare_channel(
    file_name: str,
    *,
    in_k_angle: float,
    out_k_angle: float,
    tolerance: float = 45.0,
    fixed_missing_arm: str | None = "K",
    in_kp_angle: float | None = None,
    out_kp_angle: float | None = None,
    ambiguity_margin: float = 1.0,
    angle_period: float = 360.0,
) -> str | None:
    in_angle, out_angle = parse_compare_in_out_angles(file_name)
    if in_angle is None and out_angle is None:
        return None
    missing_state = str(fixed_missing_arm or "").strip().casefold()
    if (in_angle is None or out_angle is None) and missing_state not in {"k", "kp", "kprime", "k'"}:
        return None
    missing_is_k = missing_state == "k"
    in_match = (
        None
        if in_angle is None
        else classify_angle_state(
            in_angle,
            k_angle=in_k_angle,
            kp_angle=in_kp_angle,
            tolerance=tolerance,
            ambiguity_margin=ambiguity_margin,
            period=angle_period,
        )
    )
    out_match = (
        None
        if out_angle is None
        else classify_angle_state(
            out_angle,
            k_angle=out_k_angle,
            kp_angle=out_kp_angle,
            tolerance=tolerance,
            ambiguity_margin=ambiguity_margin,
            period=angle_period,
        )
    )
    if (in_match is not None and in_match.state is None) or (out_match is not None and out_match.state is None):
        return None
    in_is_k = missing_is_k if in_match is None else in_match.state == "K"
    out_is_k = missing_is_k if out_match is None else out_match.state == "K"
    if in_is_k and out_is_k:
        return "KK"
    if in_is_k and (not out_is_k):
        return "KKp"
    if (not in_is_k) and out_is_k:
        return "KpK"
    return "KpKp"


def _compare_file_power(file_name: str) -> float:
    match = POWER_TOKEN_RE.search(str(Path(file_name).stem))
    if not match:
        return float("-inf")
    return _parse_power_number(match.group("power"))


def _compare_context_key(file_name: str, *, ignore_gate_condition: bool = False) -> str:
    stem = _normalize_gate_signs(Path(file_name).stem).replace("$", "")
    stem = POWER_TOKEN_RE.sub("", stem)
    stem = re.sub(
        r"(?:^|[_\s-]+)Rot\s*[12][^_\s-]*(?:deg|degree)",
        "_",
        stem,
        flags=re.IGNORECASE,
    )
    if ignore_gate_condition:
        stem = COMPARE_GATE_CONDITION_RE.sub("_", stem)
    stem = re.sub(r'[<>:"/\\|?*]+', "_", stem)
    stem = re.sub(r"_+", "_", stem)
    stem = re.sub(r"[\s-]+", "_", stem)
    return stem.strip("_").lower()


def _select_compare_records(
    records: Sequence[tuple[int, str, str]],
    *,
    ignore_gate_condition: bool = False,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    contexts: dict[str, list[tuple[int, str, str]]] = {}
    for record in records:
        contexts.setdefault(
            _compare_context_key(record[2], ignore_gate_condition=ignore_gate_condition),
            [],
        ).append(record)

    def context_score(item: tuple[str, list[tuple[int, str, str]]]) -> tuple[int, int, int, float, int]:
        _context, context_records = item
        keys = {key for _idx, key, _file_name in context_records}
        has_kk_pair = int("KK" in keys and "KKp" in keys)
        finite_powers = [
            _compare_file_power(file_name)
            for _idx, _key, file_name in context_records
            if np.isfinite(_compare_file_power(file_name))
        ]
        mean_power = float(np.mean(finite_powers)) if finite_powers else float("-inf")
        first_idx = min(idx for idx, _key, _file_name in context_records)
        return (has_kk_pair, len(keys), len(context_records), mean_power, -first_idx)

    _selected_context, selected_records = max(contexts.items(), key=context_score)
    found: dict[str, str] = {}
    for _idx, key, file_name in sorted(
        selected_records,
        key=lambda item: (-_compare_file_power(item[2]), item[0]),
    ):
        if key not in found:
            found[key] = file_name

    duplicates: dict[str, list[str]] = {}
    by_key: dict[str, list[str]] = {}
    for _idx, key, file_name in records:
        by_key.setdefault(key, []).append(file_name)
    for key, names in by_key.items():
        if len(names) <= 1 or key not in found:
            continue
        ordered = [found[key]] + [name for name in names if name != found[key]]
        duplicates[key] = ordered
    return found, duplicates


def coherent_compare_auto_assignment(
    file_names: Sequence[str],
    *,
    in_k_angle: float,
    out_k_angle: float,
    in_kp_angle: float | None = None,
    out_kp_angle: float | None = None,
    tolerance: float = 45.0,
    ambiguity_margin: float = 1.0,
    angle_period: float = 360.0,
) -> tuple[dict[str, str], dict[str, list[str]], str, list[str]]:
    groups: dict[str, list[tuple[int, str, str]]] = {}
    for idx, file_name in enumerate(file_names):
        key = classify_compare_channel(
            file_name,
            in_k_angle=in_k_angle,
            out_k_angle=out_k_angle,
            in_kp_angle=in_kp_angle,
            out_kp_angle=out_kp_angle,
            tolerance=tolerance,
            ambiguity_margin=ambiguity_margin,
            angle_period=angle_period,
        )
        if key is None:
            continue
        gate_key = parse_compare_gate_condition(file_name) or "__ungrouped__"
        groups.setdefault(gate_key, []).append((idx, key, file_name))

    if not groups:
        return {}, {}, "", []

    def _score(item: tuple[str, list[tuple[int, str, str]]]) -> tuple[int, int, int, int]:
        _gate_key, records = item
        keys = {key for _idx, key, _file_name in records}
        has_kk_pair = int("KK" in keys and "KKp" in keys)
        first_idx = min(idx for idx, _key, _file_name in records)
        return (has_kk_pair, len(keys), len(records), -first_idx)

    selected_gate, records = max(groups.items(), key=_score)
    selected_keys = {key for _idx, key, _file_name in records}
    if "KK" in selected_keys and "KKp" in selected_keys:
        found, duplicates = _select_compare_records(records)
    else:
        # A VP pair can be recorded with complementary gate-condition signs.
        # Only fall back to a sign-insensitive family when no exact gate group
        # already contains a coherent KK/KKp pair.
        families: dict[tuple[float, float], list[tuple[int, str, str]]] = {}
        for gate_records in groups.values():
            for record in gate_records:
                match = COMPARE_GATE_CONDITION_RE.search(
                    _normalize_gate_signs(Path(record[2]).stem)
                )
                if not match:
                    continue
                tg_coeff = _parse_angle_token(match.group("tg") or "1")
                bg_coeff = _parse_angle_token(match.group("bg") or "1")
                families.setdefault((tg_coeff, bg_coeff), []).append(record)

        compatible = [item for item in families.items() if {r[1] for r in item[1]} >= {"KK", "KKp"}]
        if compatible:
            _family, records = max(compatible, key=lambda item: _score((str(item[0]), item[1])))
            found, duplicates = _select_compare_records(records, ignore_gate_condition=True)
            selected_gate = parse_compare_gate_condition(found.get("KK", "")) or selected_gate
        else:
            found, duplicates = _select_compare_records(records)

    gate_groups = [gate for gate in groups if gate != "__ungrouped__"]
    gate_label = "" if selected_gate == "__ungrouped__" else selected_gate
    return found, duplicates, gate_label, gate_groups


def vp_map(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.asarray(a, float) + np.asarray(b, float)
    out = (np.asarray(a, float) - np.asarray(b, float)) / np.where(denom > eps, denom, np.nan)
    return np.clip(out, -1.0, 1.0)


def background_correct_cube(cube: DataCube, background: float, *, title: str | None = None) -> DataCube:
    z = np.asarray(cube.Z, float) - float(background)
    return DataCube(
        energy=np.asarray(cube.energy, float).copy(),
        gate=np.asarray(cube.gate, float).copy(),
        Z=z,
        gate_label=cube.gate_label,
        title=title if title is not None else cube.title,
        cbar_label="PL corr. (a.u.)",
    )


def _interp_rows_to_energy(cube: DataCube, energy_ref: np.ndarray) -> np.ndarray:
    e = np.asarray(cube.energy, float).ravel()
    z = np.asarray(cube.Z, float)
    if z.shape != (np.asarray(cube.gate).size, e.size):
        if z.shape == (e.size, np.asarray(cube.gate).size):
            z = z.T
        else:
            raise ValueError(f"Z shape {z.shape} does not match power/energy axes.")
    energy_ref = np.asarray(energy_ref, float).ravel()
    if e.shape == energy_ref.shape and np.allclose(e, energy_ref, rtol=1e-7, atol=1e-10):
        return z.astype(float, copy=True)
    out = np.empty((z.shape[0], energy_ref.size), dtype=float)
    for idx, row in enumerate(z):
        finite = np.isfinite(e) & np.isfinite(row)
        if np.count_nonzero(finite) < 2:
            out[idx, :] = np.nan
        else:
            order = np.argsort(e[finite])
            out[idx, :] = np.interp(energy_ref, e[finite][order], row[finite][order], left=np.nan, right=np.nan)
    return out


def _interp_power_rows(power_src: np.ndarray, z_src: np.ndarray, power_ref: np.ndarray) -> np.ndarray:
    p = np.asarray(power_src, float).ravel()
    z = np.asarray(z_src, float)
    pref = np.asarray(power_ref, float).ravel()
    if z.shape[0] != p.size:
        raise ValueError(f"Z rows ({z.shape[0]}) do not match power axis ({p.size}).")
    order = np.argsort(p)
    p = p[order]
    z = z[order, :]
    unique_p, unique_idx = np.unique(p, return_index=True)
    p = unique_p
    z = z[unique_idx, :]
    out = np.empty((pref.size, z.shape[1]), dtype=float)
    for col in range(z.shape[1]):
        y = z[:, col]
        finite = np.isfinite(p) & np.isfinite(y)
        if np.count_nonzero(finite) < 2:
            out[:, col] = np.nan
        else:
            out[:, col] = np.interp(pref, p[finite], y[finite], left=np.nan, right=np.nan)
    return out


def align_power_series_cubes(
    kk_cube: DataCube,
    kkp_cube: DataCube,
    *,
    title_kk: str = "KK",
    title_kkp: str = "KKp",
) -> tuple[DataCube, DataCube]:
    energy_ref = np.asarray(kk_cube.energy, float).ravel()
    kk_z_e = _interp_rows_to_energy(kk_cube, energy_ref)
    kkp_z_e = _interp_rows_to_energy(kkp_cube, energy_ref)
    kk_power = np.asarray(kk_cube.gate, float).ravel()
    kkp_power = np.asarray(kkp_cube.gate, float).ravel()
    p0 = max(float(np.nanmin(kk_power)), float(np.nanmin(kkp_power)))
    p1 = min(float(np.nanmax(kk_power)), float(np.nanmax(kkp_power)))
    if not np.isfinite(p0) or not np.isfinite(p1) or p0 > p1:
        raise ValueError("Power VP: KK and KKp power axes do not overlap.")
    power_ref = kk_power[np.isfinite(kk_power) & (kk_power >= p0) & (kk_power <= p1)]
    if power_ref.size < 2:
        power_ref = np.unique(np.concatenate([kk_power, kkp_power]))
        power_ref = power_ref[np.isfinite(power_ref) & (power_ref >= p0) & (power_ref <= p1)]
    if power_ref.size < 2:
        raise ValueError("Power VP needs at least two overlapping power values.")
    power_ref = np.asarray(sorted(np.unique(power_ref)), float)
    kk_aligned = _interp_power_rows(kk_power, kk_z_e, power_ref)
    kkp_aligned = _interp_power_rows(kkp_power, kkp_z_e, power_ref)
    return (
        DataCube(energy_ref.copy(), power_ref.copy(), kk_aligned, "Power (uW)", title_kk, kk_cube.cbar_label),
        DataCube(energy_ref.copy(), power_ref.copy(), kkp_aligned, "Power (uW)", title_kkp, kkp_cube.cbar_label),
    )


def power_valley_polarization_cube(
    kk_cube: DataCube,
    kkp_cube: DataCube,
    *,
    background: float,
    title: str = "Power VP",
    denom_eps: float = 1e-12,
) -> tuple[DataCube, DataCube, DataCube]:
    kk_corr = background_correct_cube(kk_cube, background, title="KK")
    kkp_corr = background_correct_cube(kkp_cube, background, title="KKp")
    kk_aligned, kkp_aligned = align_power_series_cubes(kk_corr, kkp_corr, title_kk="KK", title_kkp="KKp")
    vp = vp_map(kk_aligned.Z, kkp_aligned.Z, eps=denom_eps)
    vp_cube = DataCube(
        np.asarray(kk_aligned.energy, float).copy(),
        np.asarray(kk_aligned.gate, float).copy(),
        vp,
        "Power (uW)",
        title,
        "VP",
    )
    return kk_aligned, kkp_aligned, vp_cube


def _record_stage_map(
    records: Sequence[PowerSeriesFile | PowerSweepPoint],
) -> dict[float, tuple[int, PowerSeriesFile | PowerSweepPoint]]:
    out: dict[float, tuple[int, PowerSeriesFile | PowerSweepPoint]] = {}
    for idx, record in enumerate(records):
        if record.stage is None:
            continue
        key = float(record.stage)
        if key not in out:
            out[key] = (idx, record)
    return out


def power_stage_paired_vp_cubes(
    kk_cube: DataCube,
    kk_records: Sequence[PowerSeriesFile | PowerSweepPoint],
    kkp_cube: DataCube,
    kkp_records: Sequence[PowerSeriesFile | PowerSweepPoint],
    *,
    background: float,
    title: str = "Power VP",
    denom_eps: float = 1e-12,
) -> tuple[DataCube, DataCube, DataCube, tuple[PowerStagePair, ...]]:
    kk_map = _record_stage_map(kk_records)
    kkp_map = _record_stage_map(kkp_records)
    stages = sorted(set(kk_map) & set(kkp_map))
    if not stages:
        raise ValueError("Power VP stage pairing found no shared Stage values between KK and KKp groups.")

    energy_ref = np.asarray(kk_cube.energy, float).ravel()
    kk_z_e = _interp_rows_to_energy(kk_cube, energy_ref)
    kkp_z_e = _interp_rows_to_energy(kkp_cube, energy_ref)
    kk_corr = kk_z_e - float(background)
    kkp_corr = kkp_z_e - float(background)

    kk_rows: list[np.ndarray] = []
    kkp_rows: list[np.ndarray] = []
    pairs: list[PowerStagePair] = []
    for stage in stages:
        kk_idx, kk_record = kk_map[stage]
        kkp_idx, kkp_record = kkp_map[stage]
        if kk_idx >= kk_corr.shape[0] or kkp_idx >= kkp_corr.shape[0]:
            continue
        power_axis = 0.5 * (float(kk_record.power_uW) + float(kkp_record.power_uW))
        pairs.append(
            PowerStagePair(
                stage=float(stage),
                power_kk=float(kk_record.power_uW),
                power_kkp=float(kkp_record.power_uW),
                power_axis=float(power_axis),
                kk_file=kk_record.file_name,
                kkp_file=kkp_record.file_name,
                kk_row=getattr(kk_record, "row_index", None),
                kkp_row=getattr(kkp_record, "row_index", None),
            )
        )
        kk_rows.append(np.asarray(kk_corr[kk_idx, :], float))
        kkp_rows.append(np.asarray(kkp_corr[kkp_idx, :], float))

    if not pairs:
        raise ValueError("Power VP stage pairing produced no usable paired spectra.")

    order = sorted(range(len(pairs)), key=lambda idx: (pairs[idx].power_axis, pairs[idx].stage))
    pairs = [pairs[idx] for idx in order]
    kk_arr = np.vstack([kk_rows[idx] for idx in order])
    kkp_arr = np.vstack([kkp_rows[idx] for idx in order])
    power_axis = np.asarray([pair.power_axis for pair in pairs], float)
    vp = vp_map(kk_arr, kkp_arr, eps=denom_eps)
    kk_out = DataCube(energy_ref.copy(), power_axis.copy(), kk_arr, "Power (uW)", "KK", "PL corr. (a.u.)")
    kkp_out = DataCube(energy_ref.copy(), power_axis.copy(), kkp_arr, "Power (uW)", "KKp", "PL corr. (a.u.)")
    vp_out = DataCube(energy_ref.copy(), power_axis.copy(), vp, "Power (uW)", title, "VP")
    return kk_out, kkp_out, vp_out, tuple(pairs)


def estimate_constant_background(
    cubes: Sequence[DataCube] | Dict[str, DataCube],
    *,
    percentile: float = 1.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> float:
    """Estimate one scalar background from finite values across cubes."""
    iterable = cubes.values() if isinstance(cubes, dict) else cubes
    vals: list[np.ndarray] = []
    for cube in iterable:
        z = np.asarray(cube.Z, float)
        e = np.asarray(cube.energy, float).ravel()
        g = np.asarray(cube.gate, float).ravel()
        if z.shape != (g.size, e.size):
            roi = z
        else:
            x_mask = np.ones(e.shape, dtype=bool)
            y_mask = np.ones(g.shape, dtype=bool)
            if xlim is not None:
                x0, x1 = sorted((float(xlim[0]), float(xlim[1])))
                x_mask = (e >= x0) & (e <= x1)
            if ylim is not None:
                y0, y1 = sorted((float(ylim[0]), float(ylim[1])))
                y_mask = (g >= y0) & (g <= y1)
            roi = z[np.ix_(y_mask, x_mask)] if np.any(y_mask) and np.any(x_mask) else z
        finite = roi[np.isfinite(roi)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return 0.0
    p = float(np.clip(percentile, 0.0, 100.0))
    return float(np.nanpercentile(np.concatenate(vals), p))


def _require_compatible_grid(a: DataCube, b: DataCube, *, context: str) -> None:
    ea = np.asarray(a.energy, float).ravel()
    eb = np.asarray(b.energy, float).ravel()
    ga = np.asarray(a.gate, float).ravel()
    gb = np.asarray(b.gate, float).ravel()
    za = np.asarray(a.Z, float)
    zb = np.asarray(b.Z, float)
    if ea.shape != eb.shape or ga.shape != gb.shape or za.shape != zb.shape:
        raise ValueError(f"{context}: KK and KKp grids do not have matching shapes.")
    if not np.allclose(ea, eb, rtol=1e-7, atol=1e-10):
        raise ValueError(f"{context}: KK and KKp energy axes do not match.")
    if not np.allclose(ga, gb, rtol=1e-7, atol=1e-10):
        raise ValueError(f"{context}: KK and KKp gate axes do not match.")


def valley_polarization_cube(
    kk_cube: DataCube,
    kkp_cube: DataCube,
    *,
    background: float,
    denom_eps: float = 1e-12,
    title: str = "Valley Polarization",
) -> DataCube:
    kk_corr = background_correct_cube(kk_cube, background, title="KK")
    kkp_corr = background_correct_cube(kkp_cube, background, title="KKp")
    _require_compatible_grid(kk_corr, kkp_corr, context="VP")
    vp = vp_map(kk_corr.Z, kkp_corr.Z, eps=denom_eps)
    return DataCube(
        energy=np.asarray(kk_corr.energy, float).copy(),
        gate=np.asarray(kk_corr.gate, float).copy(),
        Z=vp,
        gate_label=kk_corr.gate_label,
        title=title,
        cbar_label="VP",
    )


def _clamp_sg_window(requested: int, *, n_energy: int, polyorder: int) -> int:
    win = int(requested)
    if win % 2 == 0:
        win += 1
    min_valid = max(3, polyorder + 2)
    if min_valid % 2 == 0:
        min_valid += 1
    max_valid = max(min_valid, int(n_energy) if int(n_energy) % 2 == 1 else int(n_energy) - 1)
    win = max(min_valid, min(win, max_valid))
    if win % 2 == 0:
        win = max(min_valid, win - 1)
    return int(win)


def clamp_sg_window(requested: int, *, n_energy: int, polyorder: int) -> int:
    return _clamp_sg_window(requested, n_energy=n_energy, polyorder=polyorder)


def apply_sg_derivative_energy(
    cube: DataCube,
    *,
    derivative: int | None,
    window_length: int,
    polyorder: int = 2,
) -> tuple[DataCube, int]:
    if derivative not in (None, 1, 2):
        raise ValueError("derivative must be None, 1, or 2.")
    if derivative is None:
        return cube, int(window_length)

    z = np.asarray(cube.Z, float)
    e = np.asarray(cube.energy, float).ravel()
    if z.ndim != 2 or e.size < 5:
        raise ValueError("Derivative needs 2D data with at least 5 energy points.")

    used_win = _clamp_sg_window(int(window_length), n_energy=e.size, polyorder=int(polyorder))
    delta = float(np.nanmedian(np.diff(e)))
    if not np.isfinite(delta) or delta == 0:
        delta = float(np.nanmean(np.diff(e)))
    if not np.isfinite(delta) or delta == 0:
        delta = 1.0

    z_out = savgol_filter(
        z,
        window_length=int(used_win),
        polyorder=int(polyorder),
        deriv=int(derivative),
        delta=delta,
        axis=1,
        mode="interp",
    )
    cbar = "d(DR/R)/dE" if int(derivative) == 1 else "d2(DR/R)/dE2"
    title = cube.title if cube.title else "DR/R"
    title = f"{title} ({cbar})"
    return (
        DataCube(
            energy=np.asarray(cube.energy, float).copy(),
            gate=np.asarray(cube.gate, float).copy(),
            Z=np.asarray(z_out, float).copy(),
            gate_label=cube.gate_label,
            title=title,
            cbar_label=cbar,
        ),
        int(used_win),
    )
