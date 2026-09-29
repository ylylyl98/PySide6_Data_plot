"""Read-only inspection of the sources behind a loaded DRR result."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from core import processing_run as P
from core.drr_sources import DrrMeasurementAssignment
from core.loader import DataCube, load_pl, build_external_baseline, _interp_rows_no_extrapolation, _native_external_baseline


@dataclass(frozen=True)
class RawSnapshot:
    folder: str
    measurements: tuple[str, ...]
    assignments: tuple[DrrMeasurementAssignment, ...]
    y_axis: str
    numerical_path: str = 'common'

    @property
    def backgrounds(self):
        return tuple(dict.fromkeys(source for item in self.assignments for source in
            (item.baseline_files if item.baseline_mode == 'External' else (item.measurement_file,))))


@dataclass(frozen=True)
class RawPreview:
    cube: DataCube
    description: str
    spectrum_only: bool = False


def snapshot_from_loaded(loaded):
    if loaded is None or loaded.mode != 'DRR':
        raise ValueError('Load DRR data first.')
    files = tuple(loaded.selected_files or ([loaded.primary_file] if loaded.primary_file else []))
    if loaded.drr_mode_label == 'DR/R Map' or any(Path(name).suffix.lower() != '.csv' for name in files):
        raise ValueError('This is a precomputed DRR map; original intensity sources are not available in this selection.')
    if not files:
        raise ValueError('No measurement sources are available.')
    assignments = tuple(loaded.drr_assignments)
    if not assignments and loaded.drr_baseline_text in {'Self (first frame)', 'Self (last frame)', 'External'}:
        assignments = tuple(DrrMeasurementAssignment(name, loaded.drr_baseline_text,
            tuple(loaded.baseline_files) if loaded.drr_baseline_text == 'External' else (),
            loaded.drr_baseline_which) for name in files)
    by_file = {item.measurement_file: item for item in assignments}
    if assignments and set(by_file) != set(files):
        raise ValueError('Loaded measurement/background mappings are incomplete.')
    recipes = {(item.baseline_mode, item.baseline_files, item.baseline_which) for item in assignments}
    numerical_path = getattr(getattr(loaded, 'cube', None), 'drr_numerical_path',
        'common' if len(recipes) <= 1 else 'heterogeneous')
    return RawSnapshot(str(loaded.folder), files, tuple(by_file[name] for name in files) if assignments else (), str(loaded.y_axis_spec), numerical_path)


def _raw_cube(snapshot, source, *, background=False):
    axis = 'auto' if background else P.resolve_shared_y_axis_request(snapshot.measurements, snapshot.y_axis)
    cube = load_pl(snapshot.folder, source, log_scale=False, y_axis=axis)
    cube.cbar_label = 'Raw intensity (a.u.)'
    # Constant/duplicate gate values are separate acquired frames, not one row.
    gate = np.asarray(cube.gate, float)
    if len(np.unique(gate)) != gate.size or not np.all(np.isfinite(gate)) or (gate.size > 1 and not (np.all(np.diff(gate) > 0) or np.all(np.diff(gate) < 0))):
        cube = replace(cube, gate=np.arange(gate.size, dtype=float), gate_label='Frame index')
    return cube


def _align(values, source, target, context):
    if np.array_equal(source, target):
        return np.asarray(values, float).copy()
    if len(source) == 1:
        output = np.full((values.shape[0], len(target)), np.nan)
        output[:, np.isclose(target, source[0], rtol=0, atol=1e-12)] = values[:, :1]
        if not np.any(np.isfinite(output)):
            raise ValueError(f'{context}: no overlapping samples.')
        return output
    return _interp_rows_no_extrapolation(values, source, target, context=context)


def _finite_mean(arrays):
    stack = np.asarray(arrays, float)
    finite = np.isfinite(stack)
    count = finite.sum(axis=0)
    return np.divide(np.where(finite, stack, 0).sum(axis=0), count,
        out=np.full(count.shape, np.nan, dtype=float), where=count > 0)


def _common_rows(values, source, target, *, filter_missing=False):
    """Match process_ref_avg's tolerance and interpolation order for references."""
    source, target = np.asarray(source), np.asarray(target)
    if source.shape == target.shape and np.allclose(source, target, rtol=1e-6, atol=1e-9, equal_nan=True):
        return np.asarray(values, float).copy()
    if filter_missing:
        return _align(values, source, target, 'Reference alignment')
    if source.size < 2:
        raise ValueError('Cannot interpolate a reference axis with fewer than two samples.')
    order = np.argsort(source, kind='stable')
    return np.vstack([np.interp(target, source[order], row[order], left=np.nan, right=np.nan) for row in values])


def average_raw_cubes(cubes):
    first = cubes[0]
    aligned = []
    for cube in cubes:
        if cube.gate_label != first.gate_label:
            raise ValueError('Raw files have different Y-axis meanings; inspect them individually.')
        values = _align(cube.Z, cube.energy, first.energy, 'Energy alignment')
        values = _align(values.T, cube.gate, first.gate, 'Gate alignment').T
        aligned.append(values)
    return replace(first, Z=_finite_mean(aligned), title=f'Average raw intensity ({len(cubes)} files)')


def load_preview(snapshot, role, scope, source=None):
    if role not in {'measurement', 'background'} or scope not in {'single', 'average', 'reference'}:
        raise ValueError('Unknown raw-data view.')
    sources = snapshot.measurements if role == 'measurement' else snapshot.backgrounds
    if scope == 'reference':
        if role != 'background' or not snapshot.assignments:
            raise ValueError('No resolved background recipe is available.')
        first = _raw_cube(snapshot, snapshot.measurements[0])
        vectors = []
        recipes = []
        for item in snapshot.assignments:
            if item.baseline_mode == 'External':
                if snapshot.numerical_path == 'heterogeneous':
                    measurement = _raw_cube(snapshot, item.measurement_file)
                    energy = measurement.energy
                    axis = P.resolve_shared_y_axis_request(snapshot.measurements, snapshot.y_axis)
                    vector = _native_external_baseline(snapshot.folder, item, energy, y_axis=axis)
                else:
                    baseline = build_external_baseline(snapshot.folder, item.baseline_files, which=item.baseline_which)
                    energy, vector = baseline['energy'], baseline['I0']
                recipe = f'External: {item.baseline_which} frame(s), {len(item.baseline_files)} file(s)'
            else:
                cube = _raw_cube(snapshot, item.measurement_file)
                index = 0 if item.baseline_mode == 'Self (first frame)' else -1
                if snapshot.numerical_path == 'common':
                    values = _common_rows(cube.Z, cube.energy, first.energy)
                    values = _common_rows(values.T, cube.gate, first.gate).T
                    cube = replace(cube, energy=first.energy, gate=first.gate, Z=values)
                energy, vector = cube.energy, cube.Z[index]
                recipe = item.baseline_mode
            rows = np.asarray(vector)[None, :]
            aligned = (_common_rows(rows, energy, first.energy, filter_missing=True) if snapshot.numerical_path == 'common'
                       else _align(rows, energy, first.energy, 'Reference alignment'))
            vectors.append(aligned[0])
            recipes.append(recipe)
        cube = DataCube(first.energy.copy(), np.array([0.]), _finite_mean(vectors)[None, :],
            'Reference', 'Mean background reference contributions', 'Raw intensity (a.u.)')
        description = '; '.join(dict.fromkeys(recipes)) + '. Equal weight per measurement; canonical frame order. '
        description += 'This summarizes the per-file references; it does not imply a single common background. Individual DRR ratios are averaged separately.'
        return RawPreview(cube, description, True)
    if not sources:
        raise ValueError('No background source files are associated with this result.')
    if scope == 'single':
        if source not in sources:
            raise ValueError('Select a source from the loaded DRR result.')
        cube = _raw_cube(snapshot, source, background=role == 'background')
        return RawPreview(cube, f'{source}\nOriginal intensity; no background subtraction, ratio or derivative.')
    cubes = [_raw_cube(snapshot, name, background=role == 'background') for name in sources]
    return RawPreview(average_raw_cubes(cubes),
        f'{len(sources)} raw files, equal weight. Aligned to the first file; no extrapolation; finite contributors only. '
        'This is an intensity average, not an average DRR ratio.' +
        (' Use Reference used to inspect the frame-selection recipe.' if role == 'background' else ''))
