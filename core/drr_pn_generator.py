"""PN generation, distinct from replaying the validated, prepared V64 recipe.

The raw-PN method is a new reviewable draft: multiscale source screening followed
by slope-closed curvature bridges. It is not scientifically equivalent to V64.
The separate V64 renderer requires explicit prepared sources, parent and region.
Neither entry point reads research artifacts, BN, sample spectra or angle labels.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
from math import comb
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, PPoly

from core.drr_auto_composite import AutoCompositeSettings, _screen, _select_block
from core.drr_composite_background import (
    CompositeBackground, CompositeSource, load_composite_source,
)
from core.drr_sources import resolve_source_path
from core.load_cancellation import LoadCancelled, check_load_cancelled, load_scope


ALGORITHM_VERSION = "pn_regional_curvature_v1"
PREPARED_V64_VERSION = "prepared_v64_regional_increment_v1"


@dataclass(frozen=True)
class PnGeneratorSettings:
    feature_threshold: float = .0075
    minimum_segment_ev: float = .015
    transition_halfwidth_ev: float = .008
    widths_ev: tuple[float, ...] = (.035, .070, .140)

    def __post_init__(self):
        # Reuse the screening contract, not the old suggestion/gap generator.
        screen = AutoCompositeSettings(
            feature_threshold=self.feature_threshold,
            minimum_segment_ev=self.minimum_segment_ev, widths_ev=self.widths_ev)
        object.__setattr__(self, "widths_ev", screen.widths_ev)
        value = self.transition_halfwidth_ev
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError("transition_halfwidth_ev must be finite and positive.")


def default_settings() -> dict:
    result = asdict(PnGeneratorSettings())
    result["widths_ev"] = list(result["widths_ev"])
    return result


DEFAULT_SETTINGS = default_settings()


def dependency_paths() -> tuple[Path, ...]:
    """Code inputs to include in a saved-version fingerprint."""
    folder = Path(__file__).resolve().parent
    return tuple(folder / name for name in (
        "drr_pn_generator.py", "drr_auto_composite.py", "drr_background_structure.py",
        "drr_composite_background.py", "loader.py", "processing_run.py",
        "drr_sources.py", "provenance.py", "load_cancellation.py",
        "raw_spectrum_cache.py", "drr_axis_coordinates.py", "streaming_average.py"))


@dataclass(frozen=True)
class PnGenerationResult:
    document: CompositeBackground
    report: dict
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class V64RegionRecipe:
    """Native indices, explicitly supplied by the prepared model's recipe."""
    source_index: int
    outer_left: int
    core_left: int
    core_right: int
    outer_right: int


def _runs(mask):
    change = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)))


def _axis(energy):
    e = np.array(energy, dtype=float, copy=True)
    if (e.ndim != 1 or len(e) < 7 or not np.isfinite(e).all()
            or not np.all(e > 0) or not np.all(np.diff(e) > 0)):
        raise ValueError("At least seven finite, positive, increasing native energy points are required.")
    return e


def _regions(e, mask):
    edges = np.r_[e[0], (e[:-1] + e[1:]) / 2, e[-1]]
    return [[float(edges[a]), float(edges[b])] for a, b in _runs(mask)]


def _shift(coefficients, t0, width, size):
    """Ascending polynomial in t -> ascending local-native-x coefficients."""
    result = np.zeros(size)
    for j, value in enumerate(coefficients):
        for k in range(j + 1):
            result[k] += value * comb(j, k) * t0 ** (j-k) / width ** k
    return result


def _integral(coefficients, width):
    powers = np.arange(1, len(coefficients) + 1)
    return float(np.sum(coefficients * width ** powers / powers))


def _squared_integral(a, b, width):
    return _integral(np.polynomial.polynomial.polymul(a, b), width)


def _jumps(poly):
    errors = []
    dx = np.diff(poly.x)
    for order in range(3):
        derivative = poly.derivative(order)
        right = np.polynomial.polynomial.polyval(dx, derivative.c[::-1], tensor=False)
        errors.append(float(np.max(abs(right[:-1] - derivative.c[-1, 1:]))))
    return errors


def _transition_weights(e, selected, halfwidth):
    """Nonoverlapping collars; a too-coarse grid is rejected, not hard-spliced."""
    cross = np.flatnonzero(selected[1:] != selected[:-1]) + 1
    centers = (e[cross-1] + e[cross]) / 2
    weights = (selected == 0).astype(float)
    for number, (index, center) in enumerate(zip(cross, centers)):
        check_load_cancelled()
        distances = [center-e[0], e[-1]-center]
        if number:
            distances.append(center-centers[number-1])
        if number+1 < len(centers):
            distances.append(centers[number+1]-center)
        width = min(halfwidth, .45 * min(distances))
        mask = (e > center-width) & (e < center+width)
        if not mask.any():
            raise ValueError("Native grid is too coarse for the requested source transition; increase its width or use another grid.")
        t = (e[mask] - center + width) / (2*width)
        phi = t*t*t*(10 + t*(-15 + 6*t))
        weights[mask] = phi if selected[index] == 0 else 1-phi
    mixed = (weights > 0) & (weights < 1)
    supports = [(a-1, b) for a, b in _runs(mixed)]
    if any(a < 0 or b >= len(e) for a, b in supports):
        raise ValueError("A source transition has no native pure-source endpoints.")
    if any(weights[i] in (0., 1.) and weights[i+1] in (0., 1.) and weights[i] != weights[i+1]
           for i in range(len(e)-1)):
        raise ValueError("A source transition lacks an interior native point.")
    return weights, supports


def _curvature_bridges(e, logs, weights, supports):
    """One-moment bridges, retaining each pure segment up to a log constant."""
    source = CubicSpline(e, logs.T, axis=0)
    weight_poly = PchipInterpolator(e, weights)
    second = source.derivative(2)
    h = np.zeros((5, len(e)-1))
    for i in range(len(e)-1):
        product = np.polynomial.polynomial.polymul(
            weight_poly.c[::-1, i], second.c[::-1, i, 0]-second.c[::-1, i, 1])
        h[:len(product), i] = product
        h[:2, i] += second.c[::-1, i, 1]
    coeff = np.zeros((7, len(e)-1))  # descending powers
    offset = 0.
    cursor = 0
    selected = int(weights[0] == 0)
    bridges = []
    for left, right in supports:
        check_load_cancelled()
        if weights[left] not in (0., 1.) or weights[right] not in (0., 1.):
            raise ValueError("Bridge endpoints must select one source.")
        selected = int(weights[left] == 0)
        coeff[-4:, cursor:left] = source.c[:, cursor:left, selected]
        coeff[-1, cursor:left] += offset
        H = float(e[right]-e[left])
        next_source = int(weights[right] == 0)
        slope = float(source(e[left], 1)[selected])
        value = float(logs[selected, left] + offset)
        base_area = sum(_integral(h[:, i], e[i+1]-e[i]) for i in range(left, right))
        mismatch = float(source(e[right], 1)[next_source] - slope - base_area)
        correction = np.array([0., 6*mismatch/H, -6*mismatch/H])
        for i in range(left, right):
            dx = float(e[i+1]-e[i])
            curvature = h[:, i] + _shift(correction, (e[i]-e[left])/H, H, 5)
            cell = np.zeros(7)
            cell[0:2] = value, slope
            orders = np.arange(2, 7)
            cell[2:] = curvature / (orders*(orders-1))
            coeff[:, i] = cell[::-1]
            value = float(np.polynomial.polynomial.polyval(dx, cell))
            slope += _integral(curvature, dx)
        offset = value - float(logs[next_source, right])
        bridges.append(dict(indices=[int(left), int(right)], energy_bounds=e[[left, right]].tolist(),
                            source_from="PN"[selected], source_to="PN"[next_source],
                            slope_closure=mismatch, peak_added_log_curvature=1.5*abs(mismatch)/H,
                            following_log_offset=offset))
        cursor, selected = right, next_source
    coeff[-4:, cursor:] = source.c[:, cursor:, selected]
    coeff[-1, cursor:] += offset
    preliminary = PPoly(coeff, e)
    reference = weights*logs[0] + (1-weights)*logs[1]
    cell = np.gradient(e)
    anchor = float(cell @ (reference-preliminary(e)) / cell.sum())
    coeff[-1] += anchor
    poly = PPoly(coeff, e)
    return poly, source, bridges, anchor, reference


def generate_pn_arrays(energy, p, n, *, settings=None) -> tuple[dict, dict]:
    """Pure input-array draft generator; no angle, BN or sample input."""
    e = _axis(energy)
    settings = _settings(settings)
    span = e[-1]-e[0]
    if settings.minimum_segment_ev > span or 2*settings.transition_halfwidth_ev >= span:
        raise ValueError("Segment/transition settings must fit within the actual energy span.")
    raw = np.array([p, n], dtype=float, copy=True)
    if raw.shape != (2, len(e)) or not np.all(np.isfinite(raw) & (raw > 0)):
        raise ValueError("P and N must be finite, positive vectors on the same native energy grid.")
    columns = tuple(format(x, ".17g") for x in e)
    sources = tuple(CompositeSource("PN"[i], "PN"[i], (), "all", e, columns, raw[i]) for i in range(2))
    screen_settings = AutoCompositeSettings(feature_threshold=settings.feature_threshold,
        minimum_segment_ev=settings.minimum_segment_ev, widths_ev=settings.widths_ev)
    screen = _screen(e, sources, screen_settings, ())
    selected = _select_block(e, raw, screen["cost"], screen["valid"], settings.minimum_segment_ev, 0)
    weights, supports = _transition_weights(e, selected, settings.transition_halfwidth_ev)
    logs = np.log(raw)
    poly, source, bridges, anchor, reference = _curvature_bridges(e, logs, weights, supports)
    L = poly(e)
    background = np.exp(L)
    if not np.all(np.isfinite(background) & (background > 0)):
        raise ValueError("Generated background overflowed; the source-level mismatch is not suitable for this draft method.")
    uncertain = (~screen["supported"].all(0)) | screen["suspect"].all(0)
    pure_records = []
    edges = np.r_[e[0], (e[:-1]+e[1:])/2, e[-1]]
    for index in (0, 1):
        mask = weights == (1 if index == 0 else 0)
        for a, b in _runs(mask):
            shift = L[a:b]-logs[index, a:b]
            pure_records.append(dict(source_key="PN"[index], xmin_ev=float(edges[a]), xmax_ev=float(edges[b]),
                indices=[int(a), int(b-1)], log_scale=float(shift.mean()),
                intensity_scale=float(np.exp(shift.mean())),
                shape_residual_max=float(np.max(abs(shift-shift.mean())))))
    pure_records.sort(key=lambda row: row["xmin_ev"])
    arrays = dict(energy=e, background=background, background_log=L, raw_sources=raw,
        source_weights=np.array([weights, 1-weights]), hard_source_identity=selected,
        screen_score=screen["score"].copy(), screen_supported=screen["supported"].copy(),
        source_suspect_mask=screen["suspect"].copy(), uncertain_mask=uncertain,
        transition_endpoint_indices=np.asarray(supports, int).reshape(-1, 2),
        polynomial_knots=e.copy(), background_log_polynomial_coefficients=poly.c.copy(),
        source_log_polynomial_coefficients=source.c.copy(),
        reference_log=reference, global_log_anchor=np.array(anchor))
    report = dict(schema_version=1, report_kind="pn_generated_draft", algorithm=ALGORITHM_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        method=ALGORITHM_VERSION, status="draft_requires_preview", scientifically_validated=False,
        v64_equivalent=False, BN_used=False, sample_used=False, new_source_fit=False,
        fitted_global_log_anchor=True,
        settings={**asdict(settings), "widths_ev": list(settings.widths_ev)},
        original_default_source="P",
        input_source_keys=["P", "N"], points=len(e), resampled=False,
        source_segments=pure_records, transitions=bridges,
        unsupported_regions=_regions(e, ~screen["supported"].all(0)),
        unresolved_regions=_regions(e, uncertain), global_log_anchor=anchor,
        C2_join_error=_jumps(poly),
        source_weights_semantics="Curvature construction weights; not a pointwise intensity mixture.",
        limitations=[
            "This is a new raw-PN draft method, not the validated prepared V64 research chain.",
            "Screen scores are structure heuristics, not proof of featureless data.",
            "Shared PN resonances cannot be uniquely distinguished from their shared background.",
            "Pure segments retain their native log-source shape up to a constant level; absolute regional levels may change.",
            "C2 continuity does not guarantee absence of finite-window derivative artifacts.",
            "Review complete dRR and d2 maps in the main view before accepting this draft for analysis."])
    check_load_cancelled()
    return arrays, report


def _settings(value):
    if value is None:
        return PnGeneratorSettings()
    if isinstance(value, PnGeneratorSettings):
        return value
    if isinstance(value, Mapping):
        return PnGeneratorSettings(**dict(value))
    raise TypeError("settings must be PnGeneratorSettings or a settings mapping.")


class _CallbackCancellation:
    def __init__(self, callback):
        self.callback = callback

    def check(self):
        if hasattr(self.callback, "check"):
            self.callback.check()
        elif callable(self.callback):
            if self.callback():
                raise LoadCancelled()
        elif getattr(self.callback, "cancelled", False):
            raise LoadCancelled()


def generate_pn_background(folder, pair, p_which="all", n_which="all",
                           settings=None, measurements=None, cancelled=None) -> PnGenerationResult:
    """Load an explicitly selected PN pair and return an unsaved draft document.

    Measurements are intentionally not read by this algorithm. The caller owns
    versioned saving and user application; this function changes no defaults.
    """
    settings = _settings(settings)
    if p_which not in ("first", "last", "all") or n_which not in ("first", "last", "all"):
        raise ValueError("Frame choice must be first, last or all.")
    context = load_scope(_CallbackCancellation(cancelled)) if cancelled is not None else nullcontext()
    with context:
        check_load_cancelled()
        p_files = pair["p_files"] if isinstance(pair, Mapping) else pair.p_files
        n_files = pair["n_files"] if isinstance(pair, Mapping) else pair.n_files
        if any(isinstance(files, (str, Path)) for files in (p_files, n_files)):
            raise ValueError("P and N must each be a sequence of file paths.")
        p_files, n_files = tuple(p_files), tuple(n_files)
        if (not p_files or not n_files or len(set((*p_files, *n_files))) != len(p_files)+len(n_files)):
            raise ValueError("Select nonempty, disjoint P and N file groups.")
        resolved = tuple(resolve_source_path(folder, name) for name in (*p_files, *n_files))
        if len(set(resolved)) != len(resolved):
            raise ValueError("P and N require distinct files, including resolved path aliases.")
        sources = (
            load_composite_source(folder, p_files, p_which, key="P", label="P source"),
            load_composite_source(folder, n_files, n_which, key="N", label="N source"))
        p_source, n_source = sources
        if (not np.array_equal(p_source.energy, n_source.energy)
                or not np.array_equal(np.asarray(p_source.spectral_columns, float),
                                      np.asarray(n_source.spectral_columns, float))):
            raise ValueError("P and N require the same ordered native spectral grid; no interpolation is performed.")
        if any(source.boundaries_ev for source in sources):
            raise ValueError("Choose original continuous PN source spectra, not previously stitched backgrounds.")
        arrays, report = generate_pn_arrays(p_source.energy, p_source.intensity, n_source.intensity, settings=settings)
        report.update(p_frame_policy=p_which, n_frame_policy=n_which,
            original_source_hashes=[list(item) for source in sources for item in source.source_hashes],
            measurements_used=False)
        encoded = json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False)
        digest = hashlib.sha256(encoded.encode() + arrays["background"].tobytes()).hexdigest()
        generated = CompositeSource(
            key=f"PN-generated-{digest[:20]}", label="PN regional curvature — draft",
            files=tuple(path for source in sources for path in source.files), which="all",
            energy=p_source.energy, spectral_columns=p_source.spectral_columns,
            intensity=arrays["background"], boundaries_ev=(),
            source_hashes=tuple(item for source in sources for item in source.source_hashes),
            generation_json=encoded)
        check_load_cancelled()
        return PnGenerationResult(CompositeBackground((*sources, generated), generated.key), report, arrays)


def _hermite(left, right, width):
    c = np.zeros(6)
    c[:3] = left[0], width*left[1], width*width*left[2]/2
    d0 = right[0]-sum(c[:3])
    d1 = width*right[1]-c[1]-2*c[2]
    d2 = width*width*right[2]-2*c[2]
    c[3:] = 10*d0-4*d1+d2/2, -15*d0+7*d1-d2, 6*d0-3*d1+d2/2
    return c


def render_v64_region(energy, *, aligned_source_log, parent_background_log,
                      parent_polynomial_coefficients, recipe: V64RegionRecipe) -> tuple[dict, dict]:
    """Actual V64 increment mathematics, with an explicit prepared-region recipe.

    This renderer does not derive corrected sources, q, classification or parent
    from raw PN. Those dependencies must be supplied and fingerprinted by its
    caller. It never silently substitutes the raw-PN draft generator.
    """
    e = _axis(energy)
    A = np.asarray(aligned_source_log, float)
    parent_log = np.asarray(parent_background_log, float)
    coefficients = np.asarray(parent_polynomial_coefficients, float)
    if (A.shape != (2, len(e)) or parent_log.shape != e.shape
            or coefficients.ndim != 2 or coefficients.shape[1] != len(e)-1
            or not all(np.isfinite(a).all() for a in (A, parent_log, coefficients))):
        raise ValueError("Finite prepared sources, parent log and native PPoly coefficients are required.")
    if not isinstance(recipe, V64RegionRecipe):
        raise TypeError("An explicit V64RegionRecipe is required.")
    indices = (recipe.outer_left, recipe.core_left, recipe.core_right, recipe.outer_right)
    if (isinstance(recipe.source_index, bool)
            or not isinstance(recipe.source_index, (int, np.integer)) or recipe.source_index not in (0, 1)
            or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in indices)
            or not (0 <= indices[0] < indices[1] < indices[2] < indices[3] < len(e))):
        raise ValueError("Expected outer_left < core_left < core_right < outer_right native indices.")
    size = max(6, coefficients.shape[0])
    pc = np.pad(coefficients, ((size-coefficients.shape[0], 0), (0, 0)))
    parent = PPoly(pc, e)
    if not np.allclose(parent(e), parent_log, rtol=0, atol=1e-10):
        raise ValueError("Prepared parent polynomial does not reconstruct its log samples.")
    source = CubicSpline(e, A[recipe.source_index])
    outer_left, left, right, outer_right = indices
    zero = np.zeros_like(pc[::-1])
    mode = np.zeros_like(zero)
    linear = quadratic = 0.
    for side, (a, b) in enumerate(((outer_left, left), (right, outer_right))):
        check_load_cancelled()
        width = float(e[b]-e[a])
        at_left = np.zeros(3) if side == 0 else [source(e[a], nu=k)-parent(e[a], nu=k) for k in range(3)]
        at_right = [source(e[b], nu=k)-parent(e[b], nu=k) for k in range(3)] if side == 0 else np.zeros(3)
        base = _hermite(at_left, at_right, width)
        level = [0, 0, 0, 10, -15, 6] if side == 0 else [1, 0, 0, -10, 15, -6]
        for i in range(a, b):
            t0 = (e[i]-e[a])/width
            zero[:, i] = _shift(base, t0, width, size)
            mode[:, i] = _shift(level, t0, width, size)
            factors = np.arange(2, size)*np.arange(1, size-1)
            z = zero[2:, i]*factors
            m = mode[2:, i]*factors
            linear += _squared_integral(z, m, e[i+1]-e[i])
            quadratic += _squared_integral(m, m, e[i+1]-e[i])
    if not np.isfinite(quadratic) or quadratic <= 0:
        raise ValueError("The regional level objective is not strictly convex.")
    level = -linear/quadratic
    result = pc.copy()
    result[:, outer_left:outer_right] += (zero + level*mode)[::-1, outer_left:outer_right]
    result[:, left:right] = 0
    result[-4:, left:right] = source.c[:, left:right]
    result[-1, left:right] += level
    poly = PPoly(result, e)
    L = parent_log.copy()
    L[outer_left+1:outer_right] = poly(e[outer_left+1:outer_right])
    L[left:right+1] = A[recipe.source_index, left:right+1]+level
    B = np.exp(L)
    if not np.all(np.isfinite(B) & (B > 0)):
        raise ValueError("Prepared regional renderer produced nonfinite intensity.")
    return dict(energy=e, background=B, background_log=L,
                background_log_polynomial_coefficients=result, polynomial_knots=e.copy(),
                delta_background_log=L-parent_log, regional_log_constant=np.array(level)), dict(
        algorithm=PREPARED_V64_VERSION, explicit_prepared_recipe=asdict(recipe),
        raw_PN_preparation_performed=False, source_index=recipe.source_index,
        regional_log_constant=level, C2_join_error=_jumps(poly),
        global_adjustment=False, scientific_acceptance_not_inferred=True)
