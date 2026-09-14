"""Discovery, provenance and explicit intensity correction for combined sweeps."""
import hashlib
import json
import re
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from core.loader import DataCube
from core.processing import PowerSweepPoint


@dataclass(frozen=True)
class PowerMeasurementGroup:
    """A whole-sweep measurement context and its optional KK/KKp roles.

    ``sources`` contains source keys (``csv::...`` or legacy group keys).
    Grouping is deliberately independent of power values: a KK and KKp table
    with different grids still belongs to the same measurement context.
    """
    key: str
    label: str
    sources: tuple[str, ...]
    mapping: dict[str, str]
    duplicates: dict[str, tuple[str, ...]]
    status: str = "New"
    context: str = ""
    issues: tuple[str, ...] = ()
    power_min: float | None = None
    power_max: float | None = None
    power_count: int = 0
    modified: float = 0.0


def validate_power_vp_pairing(first, second, *, mode='stage'):
    """Validate the data overlap required by a power VP calculation.

    Returns a short human-readable explanation on failure and ``None`` when
    pairing is supported.  This check is intentionally stricter than the
    plotting routines, which may skip unavailable rows for display.
    """
    a, b = first.cube, second.cube
    ea, eb = np.asarray(a.energy, float).ravel(), np.asarray(b.energy, float).ravel()
    for cube in (a, b):
        energy = np.asarray(cube.energy, float).ravel()
        powers = np.asarray(cube.gate, float).ravel()
        if energy.size < 2 or not np.all(np.isfinite(energy)) or np.any(np.diff(energy) <= 0):
            return 'KK and KKp need increasing, finite spectral axes with at least two points.'
        if not powers.size or not np.all(np.isfinite(powers)):
            return 'KK and KKp need finite power values.'
        if np.asarray(cube.Z).shape != (powers.size, energy.size):
            return 'Sweep spectra do not match their power and spectral axes.'
    ea_f, eb_f = ea, eb
    lo, hi = max(float(ea_f.min()), float(eb_f.min())), min(float(ea_f.max()), float(eb_f.max()))
    if hi < lo:
        return 'KK and KKp have no overlapping spectral range.'
    if np.count_nonzero((ea_f >= lo) & (ea_f <= hi)) < 2:
        return 'KK and KKp need at least two overlapping spectral points.'
    if mode == 'stage':
        stages_a = [float(r.stage) for r in first.records if r.stage is not None and np.isfinite(float(r.stage))]
        stages_b = [float(r.stage) for r in second.records if r.stage is not None and np.isfinite(float(r.stage))]
        if len(stages_a) != len(set(stages_a)) or len(stages_b) != len(set(stages_b)):
            return 'Stage pairing requires unique stage_pos values in both sweeps.'
        if not set(stages_a) & set(stages_b):
            return 'KK and KKp have no shared stage_pos values.'
    elif mode == 'power':
        pa, pb = np.asarray(a.gate, float), np.asarray(b.gate, float)
        pa = pa[np.isfinite(pa)]; pb = pb[np.isfinite(pb)]
        if pa.size == 0 or pb.size == 0 or max(pa.min(), pb.min()) > min(pa.max(), pb.max()):
            return 'KK and KKp have no overlapping power range.'
        common = np.unique(np.concatenate((pa, pb)))
        common = common[(common >= max(pa.min(), pb.min())) & (common <= min(pa.max(), pb.max()))]
        if common.size < 2:
            return 'Power interpolation needs at least two overlapping power values.'
    else:
        return 'Choose Stage or Power Interpolation for VP pairing.'
    from core.processing import power_stage_paired_vp_cubes, power_valley_polarization_cube
    try:
        if mode == 'stage':
            paired = power_stage_paired_vp_cubes(a, first.records, b, second.records, background=0.)
        else:
            paired = power_valley_polarization_cube(a, b, background=0.)
    except (ValueError, IndexError) as exc:
        return str(exc)
    finite = np.isfinite(paired[0].Z) & np.isfinite(paired[1].Z)
    if not np.any(np.count_nonzero(finite, axis=1) >= 2):
        return 'No paired spectra contain two usable spectral points.'
    return None

COMBINED_FOLDER = Path("Processed Data") / "Power Dependence" / "Combined Sweeps"

ARCHIVE_FOLDERS = {'old', 'archive', 'archives', 'archived', 'initial data after processing',
                   '.git', '.venv', '__pycache__'}


def _active_files(folder, suffix):
    root = Path(folder)
    if not folder or not root.is_dir():
        return
    for directory, children, files in os.walk(root, followlinks=False):
        children[:] = [name for name in children if name.casefold() not in ARCHIVE_FOLDERS
                       and not (Path(directory) / name).is_symlink()
                       and not (Path(directory) / name).is_junction()]
        for name in files:
            if name.casefold().endswith(suffix):
                yield Path(directory) / name


@lru_cache(maxsize=2048)
def _sweep_header(path, size, modified):
    from core.data_io import _power_table_columns, processing_impl
    from core.processing import parse_power_series_file
    try:
        sep = processing_impl._guess_sep_from_first_line(Path(path))
        columns = pd.read_csv(path, sep=sep, nrows=0).columns
        power, _, spectra = _power_table_columns(columns)
        if power is not None:
            return 'table' if len(spectra) >= 2 else None
        return 'legacy' if parse_power_series_file(Path(path).name) is not None else None
    except (OSError, ValueError):
        return False


def discover_power_files(folder, *, include_legacy=False):
    """Header-only recursive discovery, excluding archives and analysis tables."""
    root = Path(folder)
    found = []
    if any(part.casefold() in ARCHIVE_FOLDERS | {'initial data'} for part in root.parts):
        return found
    legacy = []
    for path in _active_files(folder, '.csv'):
        try:
            stat = path.stat()
            relative = path.relative_to(root)
            if 'initial data' in {part.casefold() for part in relative.parts[:-1]}:
                continue
            kind = _sweep_header(str(path), stat.st_size, stat.st_mtime_ns)
            if not kind:
                continue
            if 'processed data' in {p.casefold() for p in relative.parts[:-1]} and not is_combined(folder, str(relative)):
                continue
            if kind == 'table':
                found.append(str(relative))
            elif include_legacy:
                legacy.append(str(relative))
        except OSError:
            continue
    from core.processing import group_power_series_files
    for records in group_power_series_files(legacy).values():
        if len({r.power_uW for r in records}) >= 2:
            found.extend(r.file_name for r in records)
    return sorted(set(found), key=str.casefold)


def record_sources(record):
    def walk(value, fallback):
        try:
            sources = json.loads(value).get("sources", [])
        except (ValueError, TypeError):
            sources = []
        if not sources:
            return {fallback}
        return set().union(*(walk(s.get("prior_provenance", ""), s.get("file", fallback))
                             if isinstance(s, dict) else {str(s)} for s in sources))
    return sorted(walk(getattr(record, "source_provenance", ""), record.file_name))


def combined_files(folder):
    return [name for name in discover_power_files(folder) if is_combined(folder, name)]


def processed_source_names(folder):
    """Original inputs with saved Power analysis, matching the PL status convention."""
    names = set()
    root = Path(folder).resolve()
    for path in _active_files(folder, '.metadata.json'):
        if 'power dependence' not in {part.casefold() for part in path.parts}:
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            for source in obj.get("sources", []):
                if isinstance(source, dict) and source.get("name"):
                    # ``name`` is the portable path captured at export time.
                    # Fall back to it when the recorded absolute path belongs
                    # to the experiment folder before it was moved.
                    raw = Path(source.get('source_path', source.get('path', '')))
                    if raw.is_absolute():
                        try:
                            names.add(raw.resolve().relative_to(root).as_posix().casefold())
                            continue
                        except ValueError:
                            pass
                    base = path
                    while base != base.parent and base.name.casefold() != 'processed data':
                        base = base.parent
                    if base.name.casefold() == 'processed data':
                        candidate = (base.parent / source['name']).resolve()
                        if candidate.is_file():
                            names.add(candidate.relative_to(root).as_posix().casefold())
        except (ValueError, OSError):
            continue
    return names


@lru_cache(maxsize=256)
def _combined_flag(path, size, modified):
    try:
        data = pd.read_csv(path, usecols=["source_provenance"], nrows=8)
        return any(isinstance(v, str) and '"sources"' in v for v in data.source_provenance)
    except (ValueError, OSError):
        return False


def is_combined(folder, filename):
    if not filename:
        return False
    p = Path(folder) / filename
    try:
        stat = p.stat()
        return _combined_flag(str(p), stat.st_size, stat.st_mtime_ns)
    except OSError:
        return False


def _power_context_and_channel(name: str, *, angle_refs=None, angle_tolerance=45.0, full_sweep=False):
    """Return a stable context stem and a safe channel suggestion.

    Explicit KK/KKp tokens win.  Angle matching is used only when both an
    angle and the corresponding explicit reference are present; an unknown or
    tied angle remains unresolved for the picker to assign manually.
    """
    from core.processing import (
        COMPARE_CHANNEL_TOKEN_RE, COMPARE_ANGLE_TOKEN_RE, POWER_TOKEN_RE, STAGE_TOKEN_RE,
        classify_compare_channel,
    )
    text = str(name).replace('\\', '/')
    stem = Path(text).stem
    labels = list(COMPARE_CHANNEL_TOKEN_RE.finditer(stem))
    channels = {m.group('channel').casefold() for m in labels}
    channel = None
    if len(channels) == 1:
        value = next(iter(channels))
        channel = {'kk': 'KK', 'kkp': 'KKp'}.get(value)
    elif len(channels) > 1:
        channel = None
    refs = angle_refs or {}
    # Any explicit channel token is authoritative, including KpK/KpKp or a
    # conflicting pair. Those channels are intentionally not Power KK roles.
    explicit_channel = bool(labels)
    if channel is None and not explicit_channel and refs:
        try:
            channel = classify_compare_channel(name, in_k_angle=refs['in_k'], out_k_angle=refs['out_k'],
                in_kp_angle=refs.get('in_kp'), out_kp_angle=refs.get('out_kp'), tolerance=angle_tolerance)
            if channel not in {'KK', 'KKp'}: channel = None
        except (KeyError, ValueError):
            channel = None
    # Remove only identity tokens.  Parent/session path remains part of the
    # context, keeping otherwise similar samples in separate groups.
    context = stem if full_sweep else POWER_TOKEN_RE.sub('', stem)
    if not full_sweep:
        context = STAGE_TOKEN_RE.sub('', context)
    context = COMPARE_CHANNEL_TOKEN_RE.sub('', context)
    context = COMPARE_ANGLE_TOKEN_RE.sub('', context)
    context = re.sub(r'[_\-\s]+', '_', context).strip('_').casefold() or 'measurement'
    parent = str(Path(text).parent).replace('\\', '/').casefold()
    if parent == '.':
        parent = ''
    return f'{parent}/{context}'.strip('/'), channel


def _power_metadata_index(folder):
    """Read each active acquisition sidecar once, indexed by resolved source."""
    root = Path(folder).resolve()
    index = {}
    for metadata in _active_files(folder, '.experiment.metadata.json'):
        try:
            obj = json.loads(metadata.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        for item in obj.get('files', ()):
            if not isinstance(item, dict) or not item.get('path'):
                continue
            candidate = (metadata.parent / str(item['path']).replace('\\', '/')).resolve()
            try:
                relative = candidate.relative_to(root).as_posix()
                scope = metadata.parent.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            values = {str(k).casefold(): v for k, v in obj.items()}
            values.update({str(k).casefold(): v for k, v in item.items()})
            identity_keys = ('session_id', 'measurement_id', 'sample', 'position', 'temperature', 'gate')
            parts = [f'{key}={values[key]}' for key in identity_keys if values.get(key) not in (None, '')]
            channel = values.get('channel', values.get('compare_channel', values.get('polarization')))
            if not parts and channel is None:
                continue
            fallback, filename_channel = _power_context_and_channel(relative)
            # A session or channel alone does not replace sample/condition
            # information carried in the measurement filename.
            if values.get('measurement_id') in (None, ''):
                parts.append(f'context={fallback}')
            context = '/'.join([scope, *parts]).casefold()
            channel = str(channel).casefold() if channel is not None else (filename_channel or '').casefold()
            index.setdefault(str(candidate).casefold(), set()).add((context, channel))
    return index


def _power_metadata_identity(folder, name, catalog=None):
    catalog = _power_metadata_index(folder) if catalog is None else catalog
    target = (Path(folder) / str(name).replace('\\', '/')).resolve()
    identities = catalog.get(str(target).casefold(), ())
    if not identities:
        return None
    if len(identities) == 1:
        return next(iter(identities))
    # Conflicting sidecars must not assign whichever metadata was read first.
    context = _power_context_and_channel(name)[0]
    return f'{context}/conflicting metadata', None


def group_power_measurement_sources(folder, sources, *, angle_refs=None,
                                    angle_tolerance=45.0, processed_names=(), individual=False):
    """Group complete Power sources by session/sample context.

    ``sources`` is the mapping returned by :func:`get_power_series_sources`.
    Unlike Compare grouping, power values never create buckets.  A source is
    one complete sweep, so unequal KK/KKp grids remain together.
    """
    buckets = {}
    metadata_catalog = _power_metadata_index(folder)
    from core.power_manifest import manifest_assignments
    manifest_catalog = manifest_assignments(folder, _active_files(folder, '.json'), angle_refs, angle_tolerance)
    processed = {str(v).replace('\\', '/').casefold() for v in processed_names}
    for key, source in dict(sources or {}).items():
        names = [source.file_name] if source.file_name else [r.file_name for r in source.records]
        names = [str(v) for v in names if v]
        if not names:
            continue
        # Combined outputs can be renamed to ``combined.csv``. Recover the
        # original identity from row provenance when it is still embedded in
        # the table, so the output stays with its source measurement group.
        identity_names = list(names)
        provenance_mixed = False
        if source.file_name and is_combined(folder, source.file_name):
            try:
                # Read the complete provenance column. A renamed combined
                # output may contain a mixed lineage after the first rows.
                frame = pd.read_csv(Path(folder) / source.file_name, usecols=['source_provenance'])
                lineage = []
                def walk(value):
                    try: obj = json.loads(value)
                    except (ValueError, TypeError): return
                    for record in obj.get('sources', []):
                        if isinstance(record, dict):
                            prior = record.get('prior_provenance', '')
                            if prior:
                                walk(prior)
                            elif record.get('file'):
                                lineage.append(str(record['file']))
                for value in frame['source_provenance']:
                    if isinstance(value, str): walk(value)
                if lineage:
                    names.extend(lineage)
                    identity_names = lineage
            except (OSError, ValueError, KeyError):
                pass
        contexts = []
        for value in dict.fromkeys(identity_names):
            manifest_identity = manifest_catalog.get(str(value).replace('\\', '/').casefold())
            if manifest_identity:
                contexts.append(manifest_identity)
                continue
            metadata_identity = _power_metadata_identity(folder, value, metadata_catalog)
            if metadata_identity:
                metadata_context, metadata_channel = metadata_identity
                if source.source_format == 'table' and not source.records:
                    from core.processing import COMPARE_CHANNEL_TOKEN_RE, parse_compare_rotation_angles
                    angles = parse_compare_rotation_angles(value)
                    if (angles.rot1 is not None or angles.rot2 is not None or
                            COMPARE_CHANNEL_TOKEN_RE.search(Path(value).stem)):
                        # Acquisition IDs cannot erase contradictory settings
                        # from a descriptive whole-sweep filename.
                        filename_context, _ = _power_context_and_channel(value, full_sweep=True)
                        metadata_context += '/filename=' + filename_context
                normalized = str(metadata_channel or '').casefold()
                contexts.append((metadata_context, {'kk': 'KK', 'kkp': 'KKp'}.get(normalized)))
            else:
                contexts.append(_power_context_and_channel(value, angle_refs=angle_refs,
                                                           angle_tolerance=angle_tolerance,
                                                           full_sweep=(source.source_format == 'table' and not source.records)))
        counts = {}
        for ctx, _channel in contexts: counts[ctx] = counts.get(ctx, 0) + 1
        context = max(counts, key=lambda ctx: (counts[ctx], ctx))
        provenance_mixed = len(counts) > 1
        # A mixed-lineage output is kept visible as one source, using its own
        # filename context as the stable bucket label. It must never inherit a
        # role from whichever lineage happened to have more rows.
        if provenance_mixed and source.file_name:
            context = _power_context_and_channel(source.file_name, angle_refs=angle_refs,
                                                 angle_tolerance=angle_tolerance)[0]
        # Legacy series members should naturally share one context. If a
        # malformed series spans contexts, keep it intact and disclose it.
        channel_candidates = [channel for ctx, channel in contexts if ctx == context and channel]
        channel = channel_candidates[0] if len(set(channel_candidates)) == 1 else None
        if individual:
            context = str(key).removeprefix('csv::')
        entry = buckets.setdefault(context, {'sources': [], 'channels': {}, 'issues': set(), 'source_names': {}})
        entry['sources'].append(str(key))
        entry['source_names'][str(key)] = tuple(dict.fromkeys(names))
        if context.endswith('/conflicting metadata'):
            entry['issues'].add('Conflicting acquisition metadata; assign the channel explicitly.')
        if channel and not provenance_mixed:
            entry['channels'].setdefault(channel, []).append(str(key))
        if provenance_mixed:
            entry['issues'].add('source members span multiple contexts')
            # Do not add a role candidate for this source.
    groups = []
    built = []
    for context, entry in sorted(buckets.items()):
        source_keys = tuple(dict.fromkeys(entry['sources']))
        mapping = {role: values[0] for role, values in entry['channels'].items() if len(values) == 1}
        duplicates = {role: tuple(values) for role, values in entry['channels'].items() if len(values) > 1}
        # Match Compare's group-local reference inference for legacy analyzer
        # pairs. Never infer across runs, repeated candidates, or provenance.
        if len(source_keys) == 2 and len(mapping) < 2 and not entry['issues'] and angle_refs:
            from core.processing import infer_compare_angle_references
            pair_names = [sources[key].file_name or '' for key in source_keys]
            if all(re.search(r'(?<![A-Za-z0-9])deg(?:ree)?[+\-]?\d', name, re.I)
                   and not _power_metadata_identity(folder, name, metadata_catalog)
                   and entry['source_names'].get(key) == (name,)
                   for key, name in zip(source_keys, pair_names)):
                inferred = infer_compare_angle_references(pair_names,
                    in_k_anchor=angle_refs.get('in_k', 0.), out_k_anchor=angle_refs.get('out_k', 0.))
                if inferred.out_k is not None and inferred.out_kp is not None:
                    refs = dict(angle_refs, out_k=inferred.out_k, out_kp=inferred.out_kp)
                    roles = [_power_context_and_channel(name, angle_refs=refs,
                             angle_tolerance=angle_tolerance, full_sweep=True)[1] for name in pair_names]
                    if set(roles) == {'KK', 'KKp'}:
                        mapping = dict(zip(roles, source_keys))
                        duplicates = {}
        source_names = []
        powers = []
        for key in source_keys:
            source = sources[key]
            source_names.extend(entry.get('source_names', {}).get(key,
                ([source.file_name] if source.file_name else [r.file_name for r in source.records])))
            powers.extend(float(r.power_uW) for r in source.records if np.isfinite(float(r.power_uW)))
            # Table sources intentionally have no PowerSeriesFile records;
            # derive their range/count from the Power_uW column for the picker.
            if not source.records and source.file_name:
                try:
                    from core.data_io import _power_header_column, processing_impl
                    csv_path = Path(folder) / source.file_name
                    sep = processing_impl._guess_sep_from_first_line(csv_path)
                    stat = csv_path.stat()
                    power_col = _power_header_column(str(csv_path.resolve()), stat.st_mtime_ns, stat.st_size)
                    if power_col is None:
                        raise ValueError('missing power column')
                    frame = pd.read_csv(csv_path, sep=sep, usecols=[power_col])
                    values = frame[power_col].to_numpy(float)
                    powers.extend(float(value) for value in values if np.isfinite(value))
                except (OSError, ValueError, KeyError):
                    pass
        source_completion = []
        for key in source_keys:
            source = sources[key]
            members = entry.get('source_names', {}).get(key,
                ([source.file_name] if source.file_name else [r.file_name for r in source.records]))
            direct_combined = bool(source.file_name and is_combined(folder, source.file_name))
            source_completion.append(direct_combined or all(str(member).replace('\\', '/').casefold() in processed for member in members))
        processed_count = sum(source_completion)
        status = ('Processed' if source_completion and processed_count == len(source_completion)
                  else 'Partly processed' if processed_count else 'New')
        readiness = 'KK/KKp ready' if 'KK' in mapping and 'KKp' in mapping else ('Needs assignment' if duplicates or entry['issues'] or len(source_keys) > 1 else 'Single sweep')
        label = f"{context.replace('_', ' ')} · {readiness} · {status}"
        if powers:
            label += f" · {min(powers):.6g}–{max(powers):.6g} uW"
        finite_powers = [value for value in powers if np.isfinite(value)]
        mtimes = []
        for name in source_names:
            try: mtimes.append((Path(folder) / name).stat().st_mtime)
            except OSError: pass
        built.append(PowerMeasurementGroup(
            key=context, label=label, sources=source_keys, mapping=mapping,
            duplicates=duplicates, status=status, context=context,
            issues=tuple(sorted(entry['issues'])),
            power_min=min(finite_powers) if finite_powers else None,
            power_max=max(finite_powers) if finite_powers else None,
            power_count=len(finite_powers),
            modified=max(mtimes, default=0.0),
        ))
    return sorted(built, key=lambda group: (-group.modified, group.key))


def source_rows(result):
    def leaves(provenance, fallback):
        try:
            obj = json.loads(provenance)
        except (ValueError, TypeError):
            return {fallback}
        sources = obj.get("sources", [])
        if not sources:
            return {fallback}
        return set().union(*(leaves(s.get("prior_provenance", ""),
            (str(s.get("file", "")).replace('\\', '/').casefold(), s.get("row")))
            if isinstance(s, dict) else {(str(s).casefold(), None)} for s in sources))
    return set().union(*(leaves(getattr(r, "source_provenance", ""),
        (r.file_name.replace('\\', '/').casefold(), getattr(r, "row_index", None))) for r in result.records))


def acquisition_info(folder, result):
    names = {(Path(folder) / r.file_name).resolve() for r in result.records}
    found = []
    metadata_files = {p for directory in {p.parent for p in names} | {Path(folder)}
                      for p in directory.glob("*.experiment.metadata.json")}
    for path in sorted(metadata_files):
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            if any((path.parent / f['path']).resolve() in names for f in obj.get("files", []) if f.get('path')):
                settings = obj.get("settings", {}).get("requested", {})
                gains = {}
                def walk(value, prefix=""):
                    if isinstance(value, dict):
                        for key, val in value.items():
                            if "gain" in key.lower():
                                gains[prefix + key] = val
                            walk(val, prefix + key + ".")
                walk(obj)
                found.append({"exposure_ms": settings.get("exp_ms"), "frames": settings.get("frames"),
                              "gain": gains or "not recorded"})
        except (OSError, ValueError):
            continue
    return found or [{"gain": "not recorded", "exposure_ms": None, "frames": None}]


def suspicious_rows(folder, result):
    """Report telemetry discrepancies; never automatically remove data."""
    tables = {}
    flagged = []
    for index, record in enumerate(result.records):
        name = record.file_name
        if name not in tables:
            try:
                tables[name] = pd.read_csv(Path(folder) / name, usecols=["stage_pos", "stage_pos_actual"])
            except (OSError, ValueError):
                tables[name] = None
        table = tables[name]
        row = getattr(record, "row_index", None)
        if table is not None and row is not None and 0 <= row - 2 < len(table):
            wanted, actual = table.iloc[row - 2]
            if np.isfinite(wanted) and np.isfinite(actual) and abs(wanted - actual) > 2:
                flagged.append(index)
    return flagged


def corrected(result, *, background=0., factor=1., excluded=()):
    if not np.isfinite(factor) or factor <= 0 or not np.isfinite(background):
        raise ValueError("Use a positive correction factor and finite background.")
    keep = [i for i in range(len(result.records)) if i not in set(excluded)]
    if not keep:
        raise ValueError("Keep at least one measurement in each sweep.")
    cube = result.cube
    records = tuple(PowerSweepPoint(result.records[i].file_name, float(cube.gate[i]), result.records[i].stage,
        getattr(result.records[i], "row_index", None), source_provenance=json.dumps({
        "sources": [{"file": result.records[i].file_name, "row": getattr(result.records[i], "row_index", None),
                     "prior_provenance": getattr(result.records[i], "source_provenance", "")}],
        "intensity_correction": {"background": background, "factor": factor}})) for i in keep)
    return replace(result, cube=DataCube(cube.energy.copy(), cube.gate[keep].copy(),
        (cube.Z[keep] - background) * factor, cube.gate_label, cube.title, cube.cbar_label), records=records)


def estimate_background(result, energy_min, energy_max):
    """Robust constant baseline from an explicitly selected signal-free band."""
    mask = (result.cube.energy >= energy_min) & (result.cube.energy <= energy_max)
    values = result.cube.Z[:, mask]
    values = values[np.isfinite(values)]
    if energy_min >= energy_max or mask.sum() < 5 or len(values) < 5:
        raise ValueError("Background band must contain at least five spectral samples in each sweep.")
    return float(np.median(values))


def estimate_factor(first, second, *, _reverse=True):
    """Map B onto A. Interpolate in measured power only between nearby B samples."""
    a, b = first.cube, second.cube
    order = np.argsort(b.gate)
    power, unique = np.unique(b.gate[order], return_index=True)
    z = b.Z[order][unique]
    mask = (a.energy >= b.energy[0]) & (a.energy <= b.energy[-1])
    pairs = []
    samples = []
    for p, y in zip(a.gate, a.Z):
        if p < power[0] or p > power[-1]:
            continue
        j = int(np.searchsorted(power, p))
        if j < len(power) and power[j] == p:
            target = z[j]
        elif j > 0 and j < len(power) and power[j] / max(power[j - 1], 1e-30) <= 1.25:
            t = (p - power[j - 1]) / (power[j] - power[j - 1])
            target = (1 - t) * z[j - 1] + t * z[j]
        else:
            continue
        x = np.interp(a.energy[mask], b.energy, target)
        y = y[mask]
        good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if np.any(good):
            good &= (x > .1 * np.nanmax(x[good])) & (y > .1 * np.nanmax(y[good]))
        if good.sum() >= 8:
            x, y = x[good], y[good]
            factor = float(np.dot(x, y) / np.dot(x, x))
            residual = float(np.linalg.norm(y - factor * x) / np.linalg.norm(y))
            pairs.append((float(p), factor, residual))
            samples.append((x, y))
    if not pairs:
        if _reverse:
            inverse, quality = estimate_factor(second, first, _reverse=False)
            quality['pairs'] = [(p, 1 / factor, residual) for (p, factor, _), residual in
                                zip(quality['pairs'], quality.pop('reverse_residuals'))]
            quality['max_relative_residual'] = max(r for _, _, r in quality['pairs'])
            quality['accepted'] = quality['relative_scatter'] <= .1 and quality['max_relative_residual'] <= .1
            quality['provisional'] = quality['provisional'] or not quality['accepted']
            quality['interpolation'] = 'first sweep evaluated at second sweep powers'
            return 1 / inverse, quality
        raise ValueError("No supported overlap for calibration. Use matching powers or closely bracketing samples; no extrapolation is used.")
    factor = float(np.median([p[1] for p in pairs]))
    pairs = [(p, f, float(np.linalg.norm(y - factor * x) / np.linalg.norm(y)))
             for (p, f, _), (x, y) in zip(pairs, samples)]
    scatter = float(np.std([p[1] for p in pairs]) / factor)
    residual = float(max(p[2] for p in pairs))
    return factor, {"pairs": pairs, "relative_scatter": scatter, "max_relative_residual": residual,
                    "reverse_residuals": [float(np.linalg.norm(x - y / factor) / np.linalg.norm(x)) for x, y in samples],
                    "accepted": scatter <= .1 and residual <= .1,
                    "provisional": len(pairs) < 3 or scatter > .1 or residual > .1
                        or max(p[0] for p in pairs) < 1.1 * min(p[0] for p in pairs)}


def fingerprint(results, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for result in results:
        for array in (result.cube.energy, result.cube.gate, result.cube.Z):
            digest.update(np.ascontiguousarray(array, dtype=float).tobytes())
        digest.update(repr(sorted(source_rows(result), key=str)).encode())
        digest.update(repr(result.records).encode())
    return digest.hexdigest()


def suggested_name(first, second):
    names = [Path(r.records[0].file_name).stem for r in (first, second)]
    name = names[0] if names[1].startswith(names[0]) else names[0] + "__" + names[1]
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip(' .')
    if len(name) > 130:
        name = name[:115] + '_' + hashlib.sha256(name.encode()).hexdigest()[:8]
    return name + "__combined.csv"
