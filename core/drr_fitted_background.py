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


from core.drr_fitted_settings import (
    BackgroundFitSettings, settings_for_source, settings_for_group, validate_recipe,
)


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
    input_sources: tuple[dict, ...] = ()

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
                    input_sources=list(self.input_sources),
                    processing_order=('raw intensity average -> fit one background -> I_mean / B_fit - 1'
                                      if self.input_sources else 'raw intensity -> fit background -> I / B_fit - 1'),
                    average_method=('equal file weights on common measured energy/gate support'
                                    if self.input_sources else None),
                    gate_label=self.gate_label, anchor_rms_percent=self.anchor_rms_percent,
                    method=f'Shared log-intensity {self.settings.method}; symmetric Huber fit; per-frame gain and tilt; '
                        + ('gate-local peak protection; ' if self.gate_fit_mask is not None else 'manual peak protection; ')
                        + ('repeatable fine-texture correction enabled' if self.settings.shared_texture else 'smooth envelope only')
                        + ('; external fringe reference selected' if self.settings.texture_reference_source else '')
                        + ('; validated periodic gap prediction requested' if self.settings.periodic_texture else ''),
                    diagnostics=self.diagnostics,
                    normalization='I / fitted_background - 1', bn_reference_used=False,
                    fringe_reference_used=self.diagnostics.get('reference_texture', {}).get('applied_fraction', 0.) > 0.,
                    fringe_reference_role='fringe_only' if self.settings.texture_reference_source else None,
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


def estimate_background(cube, settings, *, source='', signature=(0, 0), assess_quality=True, anchor_masks=None,
                        stage=None, texture_reference=None, texture_reference_meta=None):
    settings = BackgroundFitSettings.from_mapping(settings)
    reference_enabled = settings.shared_texture and bool(settings.texture_reference_source)
    if reference_enabled and texture_reference is None:
        raise ValueError('The selected fringe reference was not loaded; supply its raw frames before fitting.')
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
    if stage:
        stage('Identifying protected peaks')
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
    if stage:
        stage('Fitting background envelope')
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
    periodic_report = dict(status='Disabled', corrected_fraction=0., additional_corrected_fraction=0.)
    reference_report = dict(status='Disabled', corrected_fraction=0., additional_corrected_fraction=0.,
                            applied_fraction=0., role='fringe_only',
                            source_metadata=dict(texture_reference_meta or {}))
    if settings.texture_reference_source and not settings.shared_texture:
        reference_report['reason'] = 'Enable shared texture correction to use the selected fringe reference.'
    if settings.shared_texture:
        if stage:
            stage('Estimating shared texture')
        residual = log_raw-shape-coefficients[:, :1]-coefficients[:, 1:]*x
        reference.log_texture, reference.texture_coverage, texture_report = shared_texture(
            energy, residual, reference.anchor_masks())
        if reference_enabled:
            from core.drr_reference_texture import reference_texture
            if stage:
                stage('Validating the selected fringe reference')
            if not isinstance(texture_reference, DataCube):
                raise ValueError('The selected fringe reference must contain raw spectral frames.')
            reference_energy = np.asarray(texture_reference.energy, float)
            reference_raw = np.asarray(texture_reference.Z, float)
            if (reference_energy.ndim != 1 or reference_raw.ndim != 2
                    or reference_raw.shape[1] != len(reference_energy) or not len(reference_raw)
                    or not np.all(np.isfinite(reference_raw)) or np.any(reference_raw <= 0)):
                raise ValueError('Fringe reference frames must contain finite positive raw intensities on one energy axis.')
            reference_order = np.argsort(reference_energy)
            reference_energy = reference_energy[reference_order]
            reference_raw = reference_raw[:, reference_order]
            prediction, coverage, reference_report = reference_texture(
                energy, log_raw, reference.anchor_masks(), reference_energy, np.log(reference_raw),
                reference.texture_coverage)
            if (prediction.shape != energy.shape or coverage.shape != energy.shape
                    or not np.isfinite(prediction).all() or not np.isfinite(coverage).all()
                    or np.any((coverage < 0) | (coverage > 1))):
                raise ValueError('Fringe reference returned invalid correction arrays.')
            previous_coverage = reference.texture_coverage
            contribution = (1-previous_coverage)*coverage
            reference.log_texture = reference.log_texture+(1-previous_coverage)*prediction
            reference.texture_coverage = previous_coverage+contribution
            reference_report = {**reference_report, 'role': 'fringe_only',
                'source_metadata': dict(texture_reference_meta or {}),
                'applied_fraction': float(np.mean(contribution > 0)),
                'additional_corrected_fraction': float(np.mean(reference.texture_coverage > .5)
                                                       - np.mean(previous_coverage > .5))}
        if settings.periodic_texture:
            from core.drr_periodic_texture import periodic_texture
            if stage:
                stage('Validating periodic predictions across protected bands')
            prediction, coverage, periodic_report = periodic_texture(
                energy, residual, reference.anchor_masks(), maximum_components=settings.periodic_components)
            if (prediction.shape != energy.shape or coverage.shape != energy.shape
                    or not np.isfinite(prediction).all() or not np.isfinite(coverage).all()
                    or np.any((coverage < 0) | (coverage > 1))):
                raise ValueError('Periodic prediction returned invalid correction arrays.')
            previous_coverage = reference.texture_coverage
            # Predictions supplement only missing/tapered shared texture. A fully
            # supported measured template remains bit-for-bit unchanged.
            contribution = (1-previous_coverage)*coverage
            reference.log_texture = reference.log_texture + (1-previous_coverage)*prediction
            reference.texture_coverage = previous_coverage + contribution
            periodic_report = {**periodic_report,
                'applied_fraction': float(np.mean(contribution > 0)),
                'additional_corrected_fraction': float(np.mean(reference.texture_coverage > .5)
                                                       - np.mean(previous_coverage > .5))}
    else:
        texture_report = dict(status='Disabled', corrected_fraction=0.)
        if settings.periodic_texture:
            periodic_report['reason'] = 'Enable shared texture correction to use periodic gap prediction.'
    background = reference.background()
    if not np.isfinite(background).all() or np.any(background <= 0):
        raise ValueError('Background fit is numerically unstable; reduce its complexity or revise protected regions.')
    contrast = raw / background - 1
    reference.anchor_rms_percent = float(100 * np.sqrt(np.mean(contrast[reference.anchor_masks()] ** 2)))
    reference.diagnostics = dict(shared_texture=texture_report, periodic_texture=periodic_report,
                                 reference_texture=reference_report)
    if assess_quality:
        if stage:
            stage('Checking background quality')
        from core.drr_background_quality import inspect_background
        reference.diagnostics.update(inspect_background(replace(cube, energy=energy, Z=raw), reference))
    check_load_cancelled()
    return replace(cube, energy=energy.copy(), gate=gate.copy(), Z=contrast,
                   title=f'{cube.title} (fitted reference)', cbar_label='I/B_fit - 1'), reference


@dataclass(frozen=True)
class PreparedFittedGroup:
    cube: DataCube
    files: tuple[str, ...]
    signatures: tuple[tuple[int, int], ...]
    paths: tuple[str, ...]
    y_axis: str

    @property
    def source(self):
        return self.files[0]

    @property
    def input_sources(self):
        return tuple(dict(source=name, path=path, signature=list(signature))
                     for name, path, signature in zip(self.files, self.paths, self.signatures))


def verify_prepared_group(prepared):
    for name, path, signature in zip(prepared.files, prepared.paths, prepared.signatures):
        check_load_cancelled()
        if source_signature(path) != signature:
            raise ValueError(f'{name}: source changed during averaging/fitting; reload when acquisition is complete.')


def prepare_fitted_group(folder, files, *, y_axis='auto', progress=None, log=None):
    """Align and average raw intensities with equal file weights before any fit.

    Two bounded-memory passes discover common measured support and accumulate
    intensities. Missing members, invalid samples and changed files fail the
    whole group rather than silently changing the average's membership.
    """
    from core.processing_run import resolve_shared_y_axis_request
    from core.drr_sources import resolve_source_path
    files = tuple(files)
    if not files or any(Path(name).suffix.lower() != '.csv' for name in files):
        raise ValueError('Fitted backgrounds require original CSV intensity measurements.')
    if len(set(files)) != len(files):
        raise ValueError('Select each intensity measurement only once for the average.')
    axis = resolve_shared_y_axis_request(files, y_axis)
    paths = tuple(str(resolve_source_path(folder, name).resolve()) for name in files)
    signatures = tuple(source_signature(path) for path in paths)
    first = None
    energy = gate = None
    if progress:
        progress(0)

    def read(index):
        check_load_cancelled()
        if signatures[index] != source_signature(paths[index]):
            raise ValueError(f'{files[index]}: source changed during averaging; reload DRR.')
        cube = load_intensity(folder, files[index], y_axis=axis)
        check_load_cancelled()
        if signatures[index] != source_signature(paths[index]):
            raise ValueError(f'{files[index]}: source changed during averaging; reload DRR.')
        if (cube.Z.shape != (len(cube.gate), len(cube.energy)) or len(cube.energy) < 24
                or not len(cube.gate) or not np.isfinite(cube.energy).all()
                or not np.isfinite(cube.gate).all() or not np.isfinite(cube.Z).all()
                or np.any(cube.Z <= 0)):
            raise ValueError(f'{files[index]}: averaging for log-background fitting requires finite positive raw intensities and coordinates.')
        order = np.argsort(cube.energy)
        if np.any(np.diff(np.asarray(cube.energy)[order]) <= 0) or np.any(np.diff(cube.gate) <= 0):
            raise ValueError(f'{files[index]}: background averaging requires distinct increasing coordinates.')
        return replace(cube, energy=np.asarray(cube.energy)[order], Z=np.asarray(cube.Z)[:, order])

    for index, name in enumerate(files):
        if log:
            log(f'DRR load: Reading measurement {index + 1}/{len(files)}: {Path(name).name}')
        cube = read(index)
        if first is None:
            first = cube
            energy, gate = cube.energy.copy(), cube.gate.copy()
        elif cube.gate_label != first.gate_label:
            raise ValueError('Fitted measurements have different gate-axis meanings.')
        energy = energy[(energy >= cube.energy[0]) & (energy <= cube.energy[-1])]
        gate = gate[(gate >= cube.gate[0]) & (gate <= cube.gate[-1])]
        if len(energy) < 24 or not len(gate):
            raise ValueError('Fitted measurements need common measured overlap with at least 24 energy samples and one gate.')
        if progress:
            progress(int(20 * (index + 1) / len(files)))
    accumulator = StreamingNanMean((len(gate), len(energy)))
    coordinates = CoordinateAccumulator(gate)
    for index, name in enumerate(files):
        if log:
            log(f'Averaging raw intensity {index + 1}/{len(files)}: {Path(name).name}')
        check_load_cancelled()
        cube = first if index == 0 else read(index)
        values = cube.Z
        for source_axis, target_axis, transpose in ((cube.energy, energy, False), (cube.gate, gate, True)):
            if not np.array_equal(source_axis, target_axis):
                if transpose:
                    values = values.T
                values = _interp_rows_no_extrapolation(values, source_axis, target_axis, context='Raw intensity average')
                if transpose:
                    values = values.T
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('All selected files must supply finite positive raw intensity on the common grid.')
        accumulator.add(values)
        coordinates.add(cube.gate, cube.gate_coordinates)
        if progress:
            progress(20 + int(20 * (index + 1) / len(files)))
    averaged = replace(first, energy=energy, gate=gate, Z=accumulator.mean(),
                       title=first.title if len(files) == 1 else f'Average ({len(files)} files)',
                       gate_coordinates=coordinates.result())
    prepared = PreparedFittedGroup(averaged, files, signatures, paths, axis)
    verify_prepared_group(prepared)
    return prepared


def fit_prepared_group(prepared, settings, *, stage=None):
    from core.drr_background_cache import fit_cached
    verify_prepared_group(prepared)
    result, reference = fit_cached(prepared.cube, settings_for_group(settings, prepared.files),
        source=prepared.source, signature=prepared.signatures[0], source_path=prepared.paths[0],
        y_axis=prepared.y_axis, stage=stage, input_sources=prepared.input_sources)
    verify_prepared_group(prepared)
    result.drr_numerical_path = 'fitted_raw_average'
    return result, reference


def load_fitted_group(folder, files, settings, *, y_axis='auto', progress=None, log=None):
    started = perf_counter()
    prepared = prepare_fitted_group(folder, files, y_axis=y_axis, progress=progress, log=log)
    def report_stage(message):
        check_load_cancelled()
        if log:
            log(f'DRR load: {message} · Average ({len(prepared.files)} files)')
    if log:
        log(f'Fitting one background to average of {len(prepared.files)} files')
    result, reference = fit_prepared_group(prepared, settings, stage=report_stage)
    if log:
        reused = 'cache reused' if reference.cache_hit else 'calculated'
        log(f'Fitted average background: {reused}, {perf_counter()-started:.2f} s')
    if progress:
        progress(90)
    return result, (reference,)


def reference_matches_files(reference, files):
    files = tuple(files)
    if reference.input_sources:
        return tuple(member['source'] for member in reference.input_sources) == files
    return len(files) == 1 and reference.source == files[0]


def read_reference_raw(folder, reference, *, y_axis='auto'):
    from core.drr_sources import resolve_source_path
    if reference.settings.shared_texture and reference.settings.texture_reference_source:
        from core.drr_texture_reference_io import verify_texture_reference
        verify_texture_reference(reference.diagnostics.get('reference_texture', {}).get('source_metadata', {}))
    members = reference.input_sources or (dict(source=reference.source, signature=reference.signature),)
    for member in members:
        path = resolve_source_path(folder, member['source'])
        if source_signature(path) != tuple(member['signature']):
            raise ValueError('The source changed after fitting. Reload DRR before inspecting its fitted background.')
        if member.get('path') and str(path.resolve()) != member['path']:
            raise ValueError('The source location changed after fitting. Reload DRR before inspecting its fitted background.')
    prepared = prepare_fitted_group(folder, tuple(member['source'] for member in members), y_axis=y_axis)
    cube = prepared.cube
    if (prepared.signatures != tuple(tuple(member['signature']) for member in members)
            or not np.array_equal(cube.energy, reference.energy) or not np.array_equal(cube.gate, reference.gate)):
        raise ValueError('Source coordinates changed after fitting; reload DRR.')
    return cube


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
        stem = Path(reference.source).stem
        if len(reference.input_sources) > 1:
            stem = f'average_{len(reference.input_sources)}_files_{stem}'
        path = directory / f'{stem}_fitted_background_{digest}.dat'
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
