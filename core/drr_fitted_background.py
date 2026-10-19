"""Direct sample-background estimation with inspectable, compact references."""
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from scipy.interpolate import BSpline

from core.load_cancellation import check_load_cancelled
from core.loader import DataCube, _interp_rows_no_extrapolation
from core.drr_axis_coordinates import CoordinateAccumulator
from core.streaming_average import StreamingNanMean

FITTED_MODE = 'Fitted (sample)'


@dataclass(frozen=True)
class BackgroundFitSettings:
    protected_regions: tuple[tuple[float, float], ...] = ()
    spline_intervals: int = 10
    smoothing: float = .001
    method: str = 'spline'
    polynomial_degree: int = 3
    automatic_protection: bool = False
    shared_texture: bool = True

    def validate(self):
        if not isinstance(self.automatic_protection, bool):
            raise ValueError('Automatic peak protection must be enabled or disabled.')
        if not isinstance(self.shared_texture, bool):
            raise ValueError('Shared texture correction must be enabled or disabled.')
        if not self.protected_regions and not self.automatic_protection:
            raise ValueError('Select protected resonance regions in Fit / view background first.')
        if not isinstance(self.spline_intervals, int) or not 4 <= self.spline_intervals <= 40:
            raise ValueError('Spline intervals must be an integer between 4 and 40.')
        if not np.isfinite(self.smoothing) or not .000001 <= self.smoothing <= 10:
            raise ValueError('Smoothing must be between 0.000001 and 10.')
        if self.method not in {'spline', 'polynomial'}:
            raise ValueError('Choose spline or polynomial background fitting.')
        if not isinstance(self.polynomial_degree, int) or not 2 <= self.polynomial_degree <= 4:
            raise ValueError('Polynomial degree must be an integer between 2 and 4.')
        for lo, hi in self.protected_regions:
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError('Each protected region needs two finite, increasing energy limits.')
        return self

    @classmethod
    def from_mapping(cls, values):
        if isinstance(values, cls):
            return values.validate()
        if not isinstance(values, dict):
            raise ValueError('Background settings must be a mapping.')
        return cls(tuple(tuple(map(float, pair)) for pair in values.get('protected_regions', ())),
                   values.get('spline_intervals', 10), float(values.get('smoothing', .001)),
                   values.get('method', 'spline'), values.get('polynomial_degree', 3),
                   values.get('automatic_protection', False), values.get('shared_texture', True)).validate()


def settings_for_source(settings, source):
    if isinstance(settings, BackgroundFitSettings):
        return settings.validate()
    if not isinstance(settings, dict):
        raise ValueError('Background recipe must be a mapping.')
    overrides = settings.get('protected_regions_by_file', {})
    if not isinstance(overrides, dict):
        raise ValueError('Per-file protected regions must be a mapping.')
    return BackgroundFitSettings.from_mapping({**settings,
        'protected_regions': overrides.get(source, settings.get('protected_regions', ()))})


def validate_recipe(settings, files):
    for source in files:
        settings_for_source(settings, source)


@dataclass
class FittedReference:
    source: str
    signature: tuple[int, int]
    energy: np.ndarray
    gate: np.ndarray
    log_shape: np.ndarray
    gain_tilt: np.ndarray
    fit_mask: np.ndarray
    gate_label: str
    settings: BackgroundFitSettings
    anchor_rms_percent: float
    diagnostics: dict = field(default_factory=dict)
    cache_hit: bool = False
    gate_fit_mask: np.ndarray | None = None
    log_texture: np.ndarray | None = None
    texture_coverage: np.ndarray | None = None

    def background(self, *, include_texture=True):
        x = 2 * (self.energy - self.energy[0]) / np.ptp(self.energy) - 1
        shape = self.log_shape + (self.log_texture if include_texture and self.log_texture is not None else 0)
        return np.exp(shape[None, :] + self.gain_tilt[:, :1] + self.gain_tilt[:, 1:] * x)

    def anchor_masks(self):
        return (self.gate_fit_mask if self.gate_fit_mask is not None
                else np.broadcast_to(self.fit_mask, (len(self.gate), len(self.energy))))

    def protected_regions_at_gate(self, index):
        from core.drr_background_structure import mask_regions
        return mask_regions(self.energy, ~self.anchor_masks()[index])

    def metadata(self):
        return dict(source=self.source, signature=list(self.signature), settings=asdict(self.settings),
                    gate_label=self.gate_label, anchor_rms_percent=self.anchor_rms_percent,
                    method=f'Shared log-intensity {self.settings.method}; symmetric Huber fit; per-frame gain and tilt; '
                        + ('gate-local peak protection; ' if self.gate_fit_mask is not None else 'manual peak protection; ')
                        + ('repeatable fine-texture correction enabled' if self.settings.shared_texture else 'smooth envelope only'),
                    diagnostics=self.diagnostics,
                    normalization='I / fitted_background - 1', bn_reference_used=False,
                    caveat='Anchor residual is not uncertainty under protected peaks. No additional dark subtraction.')


def source_signature(path):
    stat = Path(path).stat()
    return stat.st_mtime_ns, stat.st_size


def load_intensity(folder, source, *, y_axis='auto'):
    """Use the bounded raw cache and retain measured coordinates, as DRR does."""
    from core.processing_run import _load_canonical
    data = _load_canonical(folder, source, y_axis=y_axis)
    gate = np.asarray(data['gate_axis'], float)
    if not np.isfinite(gate).all() or not gate.size or np.any(np.diff(gate) <= 0):
        raise ValueError('Fitted DRR requires a distinct increasing gate axis; select the varying scan coordinate.')
    return DataCube(np.asarray(data['energy'], float), gate, np.asarray(data['Z'], float),
                    data.get('gate_label', 'Gate'), data.get('title_name', source), 'Intensity (counts)',
                    gate_coordinates=data.get('gate_coordinates', {}))


def _robust_fit(a, y, penalty=None):
    weights = np.ones(len(y))
    penalty = np.zeros((a.shape[1], a.shape[1])) if penalty is None else penalty
    coefficients = np.zeros(a.shape[1])
    for _ in range(30):
        check_load_cancelled()
        new = np.linalg.solve(a.T @ (weights[:, None] * a) + penalty + np.eye(a.shape[1]) * 1e-10,
                              a.T @ (weights * y))
        residual = y - a @ new
        scale = max(1.4826 * np.median(abs(residual - np.median(residual))), 1e-5)
        weights = np.minimum(1., 1.5 * scale / np.maximum(abs(residual), 1e-12))
        if np.max(abs(new - coefficients)) < 1e-9:
            return new
        coefficients = new
    return coefficients


def _masked_row_median(values, mask):
    """Exact finite-sample medians without nanmedian's per-row Python loop.

    Callers provide at least one finite anchor per row and a bounded gate
    batch. Masked samples sort after every anchor; odd/even counts select the
    same middle values as independently filtering each row.
    """
    ordered = np.sort(np.where(mask, values, np.inf), axis=1)
    count = mask.sum(axis=1)
    rows = np.arange(len(values))
    return ((ordered[rows, (count-1)//2] + ordered[rows, count//2]) * .5)[:, None]


def _robust_gain_tilt(x, values, valid=None):
    """Independent two-parameter Huber fits in bounded batches of gates.

    Keep the scalar solver's ridge, scale floor, 30-iteration limit and
    per-gate stopping rule. Converged gates leave the active batch immediately.
    """
    x = np.asarray(x, float)
    values = np.asarray(values, float)
    result = np.empty((len(values), 2))
    batch_size = max(1, min(64, 65536 // len(x)))
    xx = x*x
    for start in range(0, len(values), batch_size):
        check_load_cancelled()
        y = np.ascontiguousarray(values[start:start+batch_size])
        active = np.arange(len(y))
        mask = None if valid is None else np.asarray(valid[start:start+batch_size], bool)
        weights = np.ones_like(y) if mask is None else mask.astype(float)
        coefficients = np.zeros((len(y), 2))
        for _ in range(30):
            check_load_cancelled()
            normal = np.empty((len(y), 2, 2))
            normal[:, 0, 0] = weights.sum(axis=1) + 1e-10
            normal[:, 0, 1] = normal[:, 1, 0] = np.einsum('ij,j->i', weights, x)
            normal[:, 1, 1] = np.einsum('ij,j->i', weights, xx) + 1e-10
            weighted_y = weights*y
            rhs = np.column_stack((weighted_y.sum(axis=1), np.einsum('ij,j->i', weighted_y, x)))
            new = np.linalg.solve(normal, rhs[..., None])[..., 0]
            result[start+active] = new
            remaining = np.max(abs(new-coefficients), axis=1) >= 1e-9
            if not remaining.any():
                break
            # No later iteration may change a gate which already converged.
            active, y, coefficients = active[remaining], y[remaining], new[remaining]
            if mask is not None:
                mask = mask[remaining]
            residual = y - coefficients[:, :1] - coefficients[:, 1:]*x
            if mask is None:
                median = np.median(residual, axis=1, keepdims=True)
                deviation = np.median(abs(residual-median), axis=1, keepdims=True)
            else:
                median = _masked_row_median(residual, mask)
                deviation = _masked_row_median(abs(residual-median), mask)
            scale = np.maximum(1.4826*deviation, 1e-5)
            weights = np.minimum(1., 1.5*scale / np.maximum(abs(residual), 1e-12))
            if mask is not None:
                weights *= mask
    return result


def estimate_background(cube, settings, *, source='', signature=(0, 0), assess_quality=True, anchor_masks=None):
    settings = BackgroundFitSettings.from_mapping(settings)
    energy, gate, raw = (np.asarray(value, float) for value in (cube.energy, cube.gate, cube.Z))
    if (energy.ndim != 1 or gate.ndim != 1 or raw.shape != (len(gate), len(energy))
            or len(energy) < 24 or not len(gate)):
        raise ValueError('Background fitting needs a nonempty spectrum with at least 24 energy samples.')
    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(gate)):
        raise ValueError('Energy and gate coordinates must be finite.')
    if not np.all(np.isfinite(raw)) or np.any(raw <= 0):
        raise ValueError('Log-background fitting requires finite positive raw intensities; inspect missing/dark-corrected data first.')
    order = np.argsort(energy)
    energy, raw = energy[order], raw[:, order]
    if np.any(np.diff(energy) <= 0):
        raise ValueError('Background fitting requires distinct energy samples.')
    from core.drr_background_structure import gate_anchor_masks, mask_regions, pooled_profile, shared_texture
    gate_mask = gate_anchor_masks(energy, raw) if settings.automatic_protection and anchor_masks is None else None
    if anchor_masks is not None:
        supplied = np.asarray(anchor_masks)
        if supplied.shape != raw.shape or supplied.dtype != bool:
            raise ValueError('Supplied anchor masks must match the intensity matrix.')
        gate_mask = supplied[:, order].copy()
    mask = np.ones(len(energy), dtype=bool)
    for lo, hi in (() if settings.automatic_protection else settings.protected_regions):
        if lo <= energy[0] or hi >= energy[-1]:
            raise ValueError('Protected regions must lie inside the measured range, with background samples on both sides.')
        if not np.any((energy >= lo) & (energy <= hi)):
            raise ValueError('A protected region contains no measured energy samples.')
        mask &= ~((energy >= lo) & (energy <= hi))
    if gate_mask is not None:
        settings = replace(settings, protected_regions=mask_regions(energy, ~gate_mask.all(axis=0)))
        mask = gate_mask.sum(axis=0) >= max(1, int(np.ceil(.2*len(gate))))
    x = 2 * (energy - energy[0]) / np.ptp(energy) - 1
    if settings.method == 'spline':
        knots = np.r_[np.repeat(-1., 4), np.linspace(-1, 1, settings.spline_intervals + 1)[1:-1], np.repeat(1., 4)]
        design = BSpline.design_matrix(x, knots, 3).toarray()
        difference = np.diff(np.eye(design.shape[1]), n=2, axis=0)
        penalty = settings.smoothing * difference.T @ difference
    else:
        design = np.polynomial.legendre.legvander(x, settings.polynomial_degree)
        penalty = None
    if mask.sum() < max(24, 3 * design.shape[1]) or np.linalg.matrix_rank(design[mask]) < design.shape[1]:
        raise ValueError('Too little background support. Narrow the protected regions or reduce spline intervals.')
    log_raw = np.log(raw)
    if gate_mask is None:
        shape = design @ _robust_fit(design[mask], np.median(log_raw, axis=0)[mask], penalty)
        coefficients = _robust_gain_tilt(x[mask], log_raw[:, mask] - shape[mask])
    else:
        if np.any(gate_mask.sum(axis=1) < 24):
            raise ValueError('Too little background support at some gates. Review protection in manual mode.')
        profile, _ = pooled_profile(log_raw, gate_mask, minimum=1)
        shape = design @ _robust_fit(design[mask], profile[mask], penalty)
        # Alternate normalization and shared shape so a moving mask does not
        # turn changes in gate intensity into spurious envelope structure.
        for _ in range(3):
            coefficients = _robust_gain_tilt(x, log_raw-shape, gate_mask)
            normalized = log_raw-coefficients[:, :1]-coefficients[:, 1:]*x
            profile, _ = pooled_profile(normalized, gate_mask, minimum=1)
            shape = design @ _robust_fit(design[mask], profile[mask], penalty)
        coefficients = _robust_gain_tilt(x, log_raw-shape, gate_mask)
    reference = FittedReference(source, signature, energy.copy(), gate.copy(), shape, coefficients,
                                mask, cube.gate_label, settings, 0., gate_fit_mask=gate_mask)
    if settings.shared_texture:
        residual = log_raw-shape-coefficients[:, :1]-coefficients[:, 1:]*x
        reference.log_texture, reference.texture_coverage, texture_report = shared_texture(
            energy, residual, reference.anchor_masks())
    else:
        texture_report = dict(status='Disabled', corrected_fraction=0.)
    background = reference.background()
    if not np.isfinite(background).all() or np.any(background <= 0):
        raise ValueError('Background fit is numerically unstable; reduce its complexity or revise protected regions.')
    contrast = raw / background - 1
    reference.anchor_rms_percent = float(100 * np.sqrt(np.mean(contrast[reference.anchor_masks()] ** 2)))
    reference.diagnostics = dict(shared_texture=texture_report)
    if assess_quality:
        from core.drr_background_quality import inspect_background
        reference.diagnostics.update(inspect_background(replace(cube, energy=energy, Z=raw), reference))
    check_load_cancelled()
    return replace(cube, energy=energy.copy(), gate=gate.copy(), Z=contrast,
                   title=f'{cube.title} (fitted reference)', cbar_label='I/B_fit - 1'), reference


def load_fitted_group(folder, files, settings, *, y_axis='auto', progress=None, log=None):
    from core.processing_run import resolve_shared_y_axis_request
    from core.drr_sources import resolve_source_path
    if not files or any(Path(name).suffix.lower() != '.csv' for name in files):
        raise ValueError('Fitted backgrounds require original CSV intensity measurements.')
    axis = resolve_shared_y_axis_request(files, y_axis)
    references = []
    first = accumulator = None
    if progress:
        progress(0)
    for index, name in enumerate(files):
        check_load_cancelled()
        started = perf_counter()
        if log:
            log(f'Fitting background {index + 1}/{len(files)}: {Path(name).name}')
        path = resolve_source_path(folder, name)
        signature = source_signature(path)
        raw = load_intensity(folder, name, y_axis=axis)
        from core.drr_background_cache import fit_cached
        cube, reference = fit_cached(raw, settings_for_source(settings, name),
            source=name, signature=signature, source_path=path, y_axis=axis)
        if signature != source_signature(path):
            raise ValueError(f'{name}: source changed during fitting; reload when acquisition is complete.')
        if first is None:
            first = cube
            accumulator = StreamingNanMean(cube.Z.shape)
            coordinates = CoordinateAccumulator(cube.gate)
        elif cube.gate_label != first.gate_label:
            raise ValueError('Fitted measurements have different gate-axis meanings.')
        values = cube.Z
        for source_axis, target_axis, transpose in ((cube.energy, first.energy, False), (cube.gate, first.gate, True)):
            if not np.array_equal(source_axis, target_axis):
                if transpose:
                    values = values.T
                values = _interp_rows_no_extrapolation(values, source_axis, target_axis, context='Fitted DRR alignment')
                if transpose:
                    values = values.T
        accumulator.add(values)
        coordinates.add(cube.gate, cube.gate_coordinates)
        references.append(reference)
        if log:
            reused = 'cache reused' if reference.cache_hit else 'calculated'
            log(f'Fitted background {index + 1}/{len(files)}: {reused}, {perf_counter()-started:.2f} s')
        if progress:
            progress(int(90 * (index + 1) / len(files)))
    result = replace(first, Z=accumulator.mean(), gate_coordinates=coordinates.result())
    result.drr_numerical_path = 'heterogeneous'
    return result, tuple(references)


def read_reference_raw(folder, reference, *, y_axis='auto'):
    from core.drr_sources import resolve_source_path
    path = resolve_source_path(folder, reference.source)
    if source_signature(path) != reference.signature:
        raise ValueError('The source changed after fitting. Reload DRR before inspecting its fitted background.')
    cube = load_intensity(folder, reference.source, y_axis=y_axis)
    order = np.argsort(cube.energy)
    if (source_signature(path) != reference.signature or not np.array_equal(cube.energy[order], reference.energy)
            or not np.array_equal(cube.gate, reference.gate)):
        raise ValueError('Source coordinates changed after fitting; reload DRR.')
    return replace(cube, energy=cube.energy[order].copy(), Z=cube.Z[:, order].copy())


def export_references(folder, references):
    """Save the actual per-frame denominators, without re-reading or re-fitting."""
    directory = Path(folder) / 'Processed Data' / 'DRR' / 'Fitted Backgrounds'
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for reference in references:
        check_load_cancelled()
        payload = {**reference.metadata(), 'energy_eV': reference.energy.tolist(), 'gate': reference.gate.tolist(),
                   'log_shape': reference.log_shape.tolist(), 'gain_tilt': reference.gain_tilt.tolist(),
                   'log_texture': reference.log_texture.tolist() if reference.log_texture is not None else None,
                   'texture_coverage': reference.texture_coverage.tolist() if reference.texture_coverage is not None else None,
                   'protected_regions_by_gate_eV': [reference.protected_regions_at_gate(i) for i in range(len(reference.gate))]}
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]
        path = directory / f'{Path(reference.source).stem}_fitted_background_{digest}.dat'
        temporary = path.with_name(f'.{uuid4().hex}.tmp')
        metadata = path.with_suffix('.metadata.json')
        try:
            header = 'Photon energy(eV)\t' + '\t'.join(f'{value:.12g}' for value in reference.gate)
            np.savetxt(temporary, np.column_stack([reference.energy, reference.background().T]), delimiter='\t',
                       header=header, comments='', fmt='%.12g')
            temporary.replace(path)
            temporary.write_text(json.dumps(dict(operation='Fitted background', processing=payload,
                plot=dict(title='Fitted background', ylabel=reference.gate_label, cbar_label='Background (counts)')),
                indent=2, ensure_ascii=False), encoding='utf-8')
            temporary.replace(metadata)
        finally:
            temporary.unlink(missing_ok=True)
        outputs.append(str(path.relative_to(Path(folder))))
    return outputs
