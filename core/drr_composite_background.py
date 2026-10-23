"""Frozen, editable spectral background references with portable provenance."""
from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import io
import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

import numpy as np

from core.load_cancellation import check_load_cancelled
from core.provenance import _digest_signature, sha256_file


COMPOSITE_VERSION = 1
COMPOSITE_MARKER = "DRRComposite"
COMPOSITE_KIND = "drr_composite_background"
_MARKER_CACHE = OrderedDict()
_MARKER_LOCK = RLock()
_GENERATION_REGION_FIELDS = ('source_segments', 'estimated_regions', 'unresolved_regions', 'unsupported_regions',
                             'texture_missing_regions', 'texture_source_segments')


def composite_metadata_path(path) -> Path:
    return Path(path).with_suffix('.metadata.json')


def _header(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        line = next((line for line in stream if line.strip()), '')
    delimiter = max((',', '\t', ';'), key=line.count)
    return next(csv.reader([line], delimiter=delimiter), []), delimiter


def is_composite_background(path) -> bool:
    """Read only the header; a broken/missing sidecar does not hide the marker."""
    path = Path(path).resolve()
    before = _digest_signature(path)
    with _MARKER_LOCK:
        cached = _MARKER_CACHE.get(path)
        if before is not None and cached is not None and cached[0] == before:
            _MARKER_CACHE.move_to_end(path)
            return cached[1]
    try:
        header, _ = _header(path)
        marked = COMPOSITE_MARKER in [item.strip() for item in header]
    except (OSError, UnicodeError, csv.Error):
        return False
    if before is not None and before == _digest_signature(path):
        with _MARKER_LOCK:
            _MARKER_CACHE[path] = before, marked
            _MARKER_CACHE.move_to_end(path)
            while len(_MARKER_CACHE) > 4096:
                _MARKER_CACHE.popitem(last=False)
    return marked


def composite_metadata_signature(path):
    """Cheap change-sensitive dependencies, including a missing sidecar.

    None signatures must not be treated as proof that a cached value is fresh.
    Ordinary CSVs need no whole-file digest or metadata read.
    """
    path = Path(path).resolve()
    marked = is_composite_background(path)
    return (str(path), _digest_signature(path), marked,
            _digest_signature(composite_metadata_path(path)) if marked else None)


def _columns(path):
    header, _ = _header(path)
    values = []
    text_header = False
    for token in header:
        try:
            values.append(float(token.strip()))
        except ValueError:
            values.append(None)
            text_header = True
    columns = tuple(token.strip() for token, value in zip(header, values)
                    if value is not None and (text_header or value > 50.))
    _column_energy(columns)
    return columns


def _column_energy(columns):
    if not isinstance(columns, (tuple, list)) or len(columns) < 2:
        raise ValueError('A composite background needs at least two original spectral columns.')
    try:
        values = np.array([float(value) for value in columns], dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Composite spectral columns must be numeric.') from exc
    if not np.isfinite(values).all() or np.any(values <= 0) or np.unique(values).size != values.size:
        raise ValueError('Composite spectral columns must be finite, positive and distinct.')
    # Match the existing header CSV parser, including its duplicate precision.
    rounded = np.round(values, 9)
    if np.unique(rounded).size != values.size:
        raise ValueError('Spectral columns are not distinct at the CSV parser precision.')
    energy = 1240. / rounded if np.median(rounded) > 20 else rounded
    return energy


def _axis_match(first, second):
    return first.shape == second.shape and np.allclose(first, second, rtol=1e-9, atol=1e-10)


def _boundaries(values, energy):
    if not isinstance(values, (tuple, list)):
        raise ValueError('Composite background boundaries must be a list.')
    if any(isinstance(value, bool) or not isinstance(value, (int, float, np.number)) for value in values):
        raise ValueError('Composite background boundaries must be numeric.')
    seams = np.asarray(values, dtype=float)
    if (not np.isfinite(seams).all() or np.any(np.diff(seams) <= 0)
            or np.any(seams <= energy[0]) or np.any(seams >= energy[-1])):
        raise ValueError('Composite background boundaries must be ordered, distinct and inside the spectrum.')
    return tuple(float(value) for value in seams)


def _generation_interval(item):
    if isinstance(item, dict):
        return item['xmin_ev'], item['xmax_ev']
    if isinstance(item, list) and len(item) == 2:
        return item[0], item[1]
    raise ValueError('Generation regions must contain two limits or xmin_ev/xmax_ev objects.')


def _reject_json_constant(value):
    raise ValueError(f'Generation diagnostics must use finite numbers or null, not {value}.')


def _generation_json(value, energy):
    """Validate once and store immutable, deterministic JSON instead of nested dicts."""
    if not isinstance(value, str):
        raise ValueError('Source generation provenance must be a JSON string.')
    if not value:
        return ''
    try:
        report = json.loads(value, parse_constant=_reject_json_constant)
        if not isinstance(report, dict):
            raise ValueError('Source generation provenance must be a JSON object.')
        # allow_nan=False also catches numbers such as 1e999 that json.loads
        # accepts as infinity without calling parse_constant.
        canonical = json.dumps(report, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        if ('schema_version' in report
                and (type(report['schema_version']) is not int or report['schema_version'] != 1)):
            raise ValueError('Unsupported source generation schema version.')
        for field in ('report_kind', 'algorithm', 'kind', 'method', 'original_default_source'):
            if field in report and (not isinstance(report[field], str) or not report[field].strip()):
                raise ValueError(f'Generation {field} must be a nonempty string.')
        if 'settings' in report and not isinstance(report['settings'], dict):
            raise ValueError('Generation settings must be an object.')
        if 'input_source_keys' in report:
            keys = report['input_source_keys']
            if (not isinstance(keys, list) or any(not isinstance(key, str) or not key.strip() for key in keys)
                    or len(set(keys)) != len(keys)):
                raise ValueError('Generation input_source_keys must contain distinct source keys.')
        for field in _GENERATION_REGION_FIELDS:
            if field not in report:
                continue
            if not isinstance(report[field], list):
                raise ValueError(f'Generation {field} must be a list of regions.')
            previous_end = None
            previous_start = None
            source_ends = {}
            for item in report[field]:
                lo, hi = _generation_interval(item)
                texture_source = field == 'texture_source_segments'
                if texture_source and (not isinstance(item, dict) or not isinstance(item.get('source_key'), str)
                                       or not item['source_key'].strip()):
                    raise ValueError('Generation texture source segments require nonempty source keys.')
                # A blend can use several measured carriers over one interval.
                # Its contributions may overlap across sources, never within
                # a source; ordinary assignment/diagnostic spans remain disjoint.
                prior_end = source_ends.get(item['source_key']) if texture_source else previous_end
                if (any(type(limit) not in (int, float) for limit in (lo, hi))
                        or not np.isfinite([lo, hi]).all() or lo >= hi
                        or lo < energy[0] or hi > energy[-1]
                        or (prior_end is not None and lo < prior_end)
                        or (previous_start is not None and lo < previous_start)):
                    raise ValueError(f'Generation {field} must have ordered, nonoverlapping finite regions within the spectrum.')
                if (isinstance(item, dict) and 'source_key' in item
                        and (not isinstance(item['source_key'], str) or not item['source_key'].strip())):
                    raise ValueError('Generation source segments require nonempty source keys.')
                previous_end = hi
                previous_start = lo
                if texture_source:
                    source_ends[item['source_key']] = hi
        return canonical
    except (KeyError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError(f'Invalid source generation provenance: {exc}') from exc


@dataclass(frozen=True)
class CompositeSource:
    key: str
    label: str
    files: tuple[str, ...]
    which: str
    energy: np.ndarray
    spectral_columns: tuple[str, ...]
    intensity: np.ndarray
    boundaries_ev: tuple[float, ...] = ()
    source_hashes: tuple[tuple[str, str], ...] = ()
    generation_json: str = ''

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key.strip() or not isinstance(self.label, str):
            raise ValueError('Composite sources require a key and a label.')
        if not isinstance(self.files, (tuple, list)) or any(not isinstance(p, str) or not p for p in self.files):
            raise ValueError('Composite source files must be a list of paths.')
        if self.which not in ('first', 'last', 'all'):
            raise ValueError('Composite source frame method must be first, last or all.')
        energy, intensity = np.array(self.energy, float, copy=True), np.array(self.intensity, float, copy=True)
        columns = tuple(self.spectral_columns)
        expected = np.sort(_column_energy(columns))
        if (energy.ndim != 1 or intensity.shape != energy.shape or energy.size < 2
                or not np.isfinite(energy).all() or np.any(np.diff(energy) <= 0)
                or not _axis_match(energy, expected) or np.isinf(intensity).any()):
            raise ValueError('Composite source arrays must match the original spectral grid; only missing intensities may be NaN.')
        hashes = tuple(tuple(item) for item in self.source_hashes)
        if any(len(item) != 2 or not all(isinstance(value, str) for value in item)
               or len(item[1]) != 64 or any(c not in '0123456789abcdef' for c in item[1]) for item in hashes):
            raise ValueError('Composite source hashes must contain source paths and SHA-256 digests.')
        energy.setflags(write=False)
        intensity.setflags(write=False)
        object.__setattr__(self, 'energy', energy)
        object.__setattr__(self, 'intensity', intensity)
        object.__setattr__(self, 'spectral_columns', columns)
        object.__setattr__(self, 'files', tuple(self.files))
        object.__setattr__(self, 'source_hashes', hashes)
        object.__setattr__(self, 'boundaries_ev', _boundaries(self.boundaries_ev, energy))
        object.__setattr__(self, 'generation_json', _generation_json(self.generation_json, energy))


def source_generation(source) -> dict:
    """Return an independent report copy; editing it cannot mutate the snapshot."""
    return json.loads(source.generation_json) if source.generation_json else {}


@dataclass(frozen=True)
class CompositeRegion:
    xmin_ev: float
    xmax_ev: float
    source_key: str

    def __post_init__(self):
        if (isinstance(self.xmin_ev, bool) or isinstance(self.xmax_ev, bool)
                or not isinstance(self.xmin_ev, (int, float, np.number))
                or not isinstance(self.xmax_ev, (int, float, np.number))
                or not np.isfinite([self.xmin_ev, self.xmax_ev]).all() or self.xmin_ev >= self.xmax_ev
                or not isinstance(self.source_key, str) or not self.source_key):
            raise ValueError('Each composite region needs increasing finite limits and a source key.')
        object.__setattr__(self, 'xmin_ev', float(self.xmin_ev))
        object.__setattr__(self, 'xmax_ev', float(self.xmax_ev))


def _mask(energy, lo, hi):
    return (energy >= lo) & ((energy < hi) | ((hi == energy[-1]) & (energy == hi)))


@dataclass(frozen=True)
class CompositeBackground:
    sources: tuple[CompositeSource, ...]
    default_source: str
    regions: tuple[CompositeRegion, ...] = ()

    def __post_init__(self):
        if not isinstance(self.sources, (tuple, list)) or not self.sources or any(not isinstance(s, CompositeSource) for s in self.sources):
            raise ValueError('Choose at least one composite background source.')
        sources = tuple(self.sources)
        keys = {source.key for source in sources}
        if len(keys) != len(sources) or self.default_source not in keys:
            raise ValueError('Composite source keys must be unique and include the default source.')
        first = sources[0]
        grid = np.asarray(first.spectral_columns, float)
        for source in sources[1:]:
            if not _axis_match(grid, np.asarray(source.spectral_columns, float)) or not _axis_match(first.energy, source.energy):
                raise ValueError('Composite sources must use one exact original spectral grid, including column order.')
        if not isinstance(self.regions, (tuple, list)) or any(not isinstance(r, CompositeRegion) for r in self.regions):
            raise ValueError('Composite regions must be a list of CompositeRegion values.')
        regions = sorted(self.regions, key=lambda r: r.xmin_ev)
        normalized = []
        for index, region in enumerate(regions):
            if (region.source_key not in keys or region.xmin_ev < first.energy[0] or region.xmax_ev > first.energy[-1]
                    or not _mask(first.energy, region.xmin_ev, region.xmax_ev).any()):
                raise ValueError('Composite regions need a known source and measured points within the spectrum.')
            if index and regions[index - 1].xmax_ev > region.xmin_ev:
                raise ValueError('Composite regions must not overlap; use assign_region to replace an interval.')
            if region.source_key == self.default_source:
                continue
            if normalized and normalized[-1].source_key == region.source_key and normalized[-1].xmax_ev == region.xmin_ev:
                normalized[-1] = replace(normalized[-1], xmax_ev=region.xmax_ev)
            else:
                normalized.append(region)
        object.__setattr__(self, 'sources', sources)
        object.__setattr__(self, 'regions', tuple(normalized))


def assign_region(document, lo, hi, source_key):
    """Return an edited snapshot, splitting overlapping older assignments."""
    region = CompositeRegion(lo, hi, source_key)
    # Validate the requested selection before dropping empty leftover slivers.
    CompositeBackground(document.sources, document.default_source, (region,))
    energy = document.sources[0].energy
    regions = []
    for old in document.regions:
        if old.xmax_ev <= lo or old.xmin_ev >= hi:
            regions.append(old)
        else:
            if old.xmin_ev < lo and _mask(energy, old.xmin_ev, lo).any():
                regions.append(replace(old, xmax_ev=float(lo)))
            if old.xmax_ev > hi and _mask(energy, hi, old.xmax_ev).any():
                regions.append(replace(old, xmin_ev=float(hi)))
    if source_key != document.default_source:
        regions.append(region)
    return replace(document, regions=tuple(regions))


def effective_generation_regions(document, field='estimated_regions') -> tuple[tuple[float, float], ...]:
    """Clip generation diagnostics to where each source is actually selected."""
    if field not in _GENERATION_REGION_FIELDS:
        raise ValueError(f'Unknown generation region field: {field}')
    energy = document.sources[0].energy
    reports = {source.key: source_generation(source).get(field, []) for source in document.sources}
    edges = sorted({float(energy[0]), float(energy[-1]),
                    *(edge for region in document.regions for edge in (region.xmin_ev, region.xmax_ev))})
    intervals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        key = next((region.source_key for region in document.regions
                    if region.xmin_ev <= lo and region.xmax_ev >= hi), document.default_source)
        for item in reports[key]:
            start, stop = _generation_interval(item)
            start, stop = max(float(start), lo), min(float(stop), hi)
            if start < stop:
                intervals.append((start, stop))
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def compose_background(document):
    check_load_cancelled()
    sources = {source.key: source for source in document.sources}
    energy = document.sources[0].energy.copy()
    intensity = sources[document.default_source].intensity.copy()
    edges = {float(energy[0]), float(energy[-1])}
    for region in document.regions:
        check_load_cancelled()
        intensity[_mask(energy, region.xmin_ev, region.xmax_ev)] = sources[region.source_key].intensity[
            _mask(energy, region.xmin_ev, region.xmax_ev)]
        edges.update((region.xmin_ev, region.xmax_ev))
    edges = sorted(edges)
    boundaries = set()
    previous_key = None
    for lo, hi in zip(edges[:-1], edges[1:]):
        key = next((r.source_key for r in document.regions if r.xmin_ev <= lo and r.xmax_ev >= hi),
                   document.default_source)
        if previous_key is not None and previous_key != key:
            boundaries.add(lo)
        boundaries.update(b for b in sources[key].boundaries_ev if lo < b < hi)
        previous_key = key
    return energy, intensity, tuple(sorted(boundaries))


def load_composite_source(folder, files, which='last', *, key=None, label=None):
    """Freeze a file group's frame reduction, without a measurement selection."""
    from core.drr_sources import resolve_source_path
    from core.loader import build_external_baseline
    if isinstance(files, (str, Path)) or not files:
        raise ValueError('Choose one or more background CSV files.')
    paths = tuple(resolve_source_path(folder, str(name)) for name in files)
    if any(path.suffix.lower() != '.csv' for path in paths):
        raise ValueError('Composite background sources must be CSV files.')
    signatures = tuple(composite_metadata_signature(path) for path in paths)
    columns = _columns(paths[0])
    for path in paths[1:]:
        check_load_cancelled()
        if not _axis_match(np.asarray(columns, float), np.asarray(_columns(path), float)):
            raise ValueError('Composite sources must use one exact original spectral grid, including column order.')
    boundaries = set()
    for path in paths:
        check_load_cancelled()
        boundaries.update(composite_background_boundaries(path))
    baseline = build_external_baseline(str(folder), [str(path) for path in paths], which=which)
    energy = np.sort(_column_energy(columns))
    if not _axis_match(np.asarray(baseline['energy'], float), energy):
        raise ValueError('Loaded background does not match its original spectral columns.')
    hashes = tuple((str(path), sha256_file(path)) for path in paths)
    if any(value is None for _, value in hashes) or signatures != tuple(composite_metadata_signature(path) for path in paths):
        raise ValueError('A background source changed while it was being read; load it again.')
    check_load_cancelled()
    return CompositeSource(key or uuid4().hex, label or ', '.join(path.name for path in paths),
        tuple(str(path) for path in paths), str(which), energy, columns, baseline['I0'],
        tuple(sorted(boundaries)), hashes)


def _portable_path(path, directory):
    try:
        return Path(os.path.relpath(Path(path).resolve(), directory)).as_posix()
    except ValueError:  # Different Windows drives cannot form relative paths.
        return str(Path(path).resolve())


def _source_payload(source, directory):
    payload = dict(key=source.key, label=source.label, files=[_portable_path(p, directory) for p in source.files],
        which=source.which, energy_ev=source.energy.tolist(), spectral_columns=list(source.spectral_columns),
        intensity=[None if np.isnan(value) else float(value) for value in source.intensity],
        boundaries_ev=list(source.boundaries_ev),
        source_hashes=[[_portable_path(path, directory), digest] for path, digest in source.source_hashes])
    if source.generation_json:
        payload['generation'] = source_generation(source)
    payload['snapshot_sha256'] = _snapshot_digest(payload)
    return payload


def _snapshot_digest(payload):
    content = {key: value for key, value in payload.items() if key != 'snapshot_sha256'}
    encoded = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _csv_bytes(document, intensity):
    columns = document.sources[0].spectral_columns
    order = np.argsort(_column_energy(columns), kind='stable')
    original_order = np.empty_like(intensity)
    original_order[order] = intensity
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(('Vbg', 'Vtg', 'Vbias', COMPOSITE_MARKER, *columns))
    writer.writerow(('0', '0', '0', str(COMPOSITE_VERSION), *(format(float(v), '.17g') for v in original_order)))
    return output.getvalue().encode('utf-8')


def _publish_pair(path, csv_bytes, metadata_bytes, *, overwrite):
    metadata = composite_metadata_path(path)
    destinations = (path, metadata)
    if not overwrite and any(p.exists() for p in destinations):
        raise FileExistsError('A composite background with this name already exists; choose a new name or explicitly overwrite it.')
    if path.exists() and not is_composite_background(path):
        raise ValueError('An existing ordinary CSV cannot be overwritten by a composite background.')
    if metadata.exists() and not path.exists():
        raise ValueError('An existing metadata file without its composite CSV cannot be overwritten; choose another name.')
    before = tuple(_digest_signature(p) for p in destinations)
    if path.exists():
        # Overwrite only a verified pair, never a damaged or unrelated file.
        read_composite_background(path)
    token = uuid4().hex
    temporary = tuple(p.with_name(f'.{p.name}.{token}.tmp') for p in destinations)
    backups = {}
    installed = []
    try:
        for target, data in zip(temporary, (csv_bytes, metadata_bytes)):
            with target.open('xb') as stream:
                stream.write(data)
        check_load_cancelled()
        if before != tuple(_digest_signature(p) for p in destinations):
            raise ValueError('The destination changed during saving; choose another name or retry.')
        try:
            for target in destinations:
                if target.exists():
                    backup = target.with_name(f'.{target.name}.{token}.backup')
                    os.replace(target, backup)
                    backups[target] = backup
            # No cancellation between these promotions: either both artifacts
            # are published or the original pair is restored below.
            for source, target in zip(temporary, destinations):
                os.replace(source, target)
                installed.append(target)
        except BaseException:
            for target in installed:
                target.unlink(missing_ok=True)
            for target, backup in backups.items():
                os.replace(backup, target)
            raise
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    finally:
        for target in temporary:
            target.unlink(missing_ok=True)


def save_composite_background(path, document, *, overwrite=False) -> Path:
    """Publish a frozen CSV and editing snapshots, never replacing a source."""
    check_load_cancelled()
    path = Path(path).resolve()
    if path.suffix.lower() != '.csv':
        raise ValueError('Save the composite background as a .csv file.')
    if any(path == Path(source_file).resolve() for source in document.sources for source_file in source.files):
        raise ValueError('A composite background cannot overwrite any of its source CSV files.')
    energy, intensity, boundaries = compose_background(document)
    data = _csv_bytes(document, intensity)
    formula = ('Frozen background intensity selected by photon-energy region; generated source '
               'processing is recorded in each source generation report'
               if any(source.generation_json for source in document.sources) else
               'Frozen raw intensity selected by photon-energy region; no normalization or smoothing')
    payload = dict(schema_version=COMPOSITE_VERSION, kind=COMPOSITE_KIND,
        operation='Composite background', dataset_type='reference_1d',
        csv_sha256=hashlib.sha256(data).hexdigest(), default_source=document.default_source,
        regions=[dict(xmin_ev=r.xmin_ev, xmax_ev=r.xmax_ev, source_key=r.source_key) for r in document.regions],
        sources=[_source_payload(source, path.parent) for source in document.sources],
        processing=dict(background_boundaries_ev=list(boundaries),
            derivative_boundary_policy='independent_segments_half_window_mask',
            formula=formula,
            synthetic_gate_columns=True))
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    check_load_cancelled()
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish_pair(path, data, encoded, overwrite=overwrite)
    return path


def read_composite_background(path) -> CompositeBackground:
    """Validate the frozen result using snapshots; original sources are optional."""
    check_load_cancelled()
    path = Path(path).resolve()
    if not is_composite_background(path):
        raise ValueError('This CSV is not a marked composite background.')
    metadata = composite_metadata_path(path)
    try:
        payload = json.loads(metadata.read_text(encoding='utf-8-sig'))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f'Composite background metadata is missing or unreadable: {metadata.name}') from exc
    if (not isinstance(payload, dict) or type(payload.get('schema_version')) is not int
            or payload['schema_version'] != COMPOSITE_VERSION or payload.get('kind') != COMPOSITE_KIND):
        raise ValueError('Unsupported composite background metadata version or kind.')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != payload.get('csv_sha256'):
        raise ValueError('Composite background CSV hash does not match its metadata; restore the pair or save a new composite.')
    try:
        sources = []
        for item in payload['sources']:
            check_load_cancelled()
            if _snapshot_digest(item) != item.get('snapshot_sha256'):
                raise ValueError('A source snapshot hash does not match its saved metadata.')
            files = tuple(str((path.parent / name).resolve()) for name in item['files'])
            hashes = tuple((str((path.parent / name).resolve()), digest) for name, digest in item['source_hashes'])
            sources.append(CompositeSource(item['key'], item['label'], files, item['which'],
                item['energy_ev'], item['spectral_columns'],
                [np.nan if value is None else value for value in item['intensity']],
                item.get('boundaries_ev', ()), hashes,
                json.dumps(item['generation'], allow_nan=False) if 'generation' in item else ''))
        document = CompositeBackground(tuple(sources), payload['default_source'],
            tuple(CompositeRegion(item['xmin_ev'], item['xmax_ev'], item['source_key']) for item in payload['regions']))
        _energy, intensity, boundaries = compose_background(document)
        if tuple(payload['processing']['background_boundaries_ev']) != boundaries:
            raise ValueError('Composite boundary metadata disagrees with its source snapshots.')
        # Canonical writer bytes validate marker, column order, the one-frame
        # shape and all intensities against the edit recipe, including NaNs.
        if _csv_bytes(document, intensity) != data:
            raise ValueError('Composite CSV does not match its saved source snapshots and regions.')
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid composite background metadata: {exc}') from exc
    check_load_cancelled()
    return document


def composite_background_boundaries(path) -> tuple[float, ...]:
    if not is_composite_background(path):
        return ()
    return compose_background(read_composite_background(path))[2]
