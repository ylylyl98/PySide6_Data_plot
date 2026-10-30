"""Generate and verify per-segment P/N drafts without implicit application."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path

from core.drr_pn_batch_plan import PnBatchPlan, pn_batch_plan_is_current
from core.drr_pn_pairing import DrrPnPair
from core.drr_pn_store import DrrPnStore
from core.drr_sources import DrrMeasurementAssignment, portable_source_name
from core.load_cancellation import check_load_cancelled
from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft, generate_pn, prepare_application


_SEAL = object()


@dataclass(frozen=True)
class PnBatchResult:
    plan: PnBatchPlan
    versions: tuple
    documents: tuple
    reports: tuple
    assignments: tuple
    signature: str = ''
    _verified: object = field(default=None, repr=False, compare=False)


def _check(cancelled):
    check_load_cancelled()
    if cancelled is not None:
        from core.load_cancellation import LoadCancelled
        if hasattr(cancelled, 'check'):
            cancelled.check()
        elif cancelled.is_set() if hasattr(cancelled, 'is_set') else cancelled():
            raise LoadCancelled()


def _identity(result):
    payload = {'plan': result.plan.signature,
        'versions': [(v.version_id, v.fingerprint) for v in result.versions],
        'assignments': [a.to_dict() for a in result.assignments]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require_pn_batch_current(context, result, *, cancelled=None):
    """Check the exact group, signed mapping, raw inputs and saved versions."""
    _check(cancelled)
    if (not isinstance(context, BackgroundContext) or not isinstance(result, PnBatchResult)
            or result._verified is not _SEAL or result.signature != _identity(result)
            or context.folder != result.plan.root or context.measurements != result.plan.selected
            or len(result.versions) != len(result.plan.segments)
            or not pn_batch_plan_is_current(result.plan)):
        raise ValueError('P/N batch plan or measurement inputs changed; review and regenerate.')
    store = DrrPnStore(context.folder)
    mapping = {}
    for segment, expected in zip(result.plan.segments, result.versions):
        _check(cancelled)
        current = store.get_version(expected.version_id)
        if (current is None or not current.fresh or current.fingerprint != expected.fingerprint
                or current.algorithm_version != expected.algorithm_version
                or current.pair != DrrPnPair((segment.p_file,), (segment.n_file,))
                or tuple(current.settings.get('sample_analysis', {}).get('training_files', ())) != segment.training_files):
            raise ValueError('P/N batch generation inputs or saved version changed; regenerate.')
        name = portable_source_name(context.folder, current.csv_path)
        for measurement in segment.measurement_files:
            mapping[measurement] = name
    for assignment in result.assignments:
        if (assignment.measurement_file not in mapping or assignment.baseline_mode != 'External'
                or assignment.baseline_files != (mapping[assignment.measurement_file],)
                or assignment.baseline_which != 'all' or assignment.regions):
            raise ValueError('P/N batch mapping does not match the generated versions.')
    if tuple(a.measurement_file for a in result.assignments) != context.measurements:
        raise ValueError('P/N batch mapping is incomplete or in a different order.')
    _check(cancelled)
    return result



def batch_dependency_paths(context, result):
    """Frozen source and implementation paths included in trial-load signatures."""
    require_pn_batch_current(context, result)
    records = DrrPnStore(context.folder)._read()['versions']
    by_id = {item['version_id']: item for item in records}
    paths = [Path(path) for path, signature in result.plan.snapshot if signature is not None]
    for version in result.versions:
        record = by_id[version.version_id]
        provenance = record['provenance']
        for item in provenance.get('dependencies', ()):
            path = Path(item['path'])
            paths.append(path if path.is_absolute() else Path(context.folder) / path)
    return tuple(dict.fromkeys(path.resolve() for path in paths))

_PROOF_SEAL = object()


@dataclass(frozen=True)
class PreparedPnBatchProof:
    context: BackgroundContext
    result_signature: str
    signatures: tuple
    log_inventory: tuple
    _verified: object = field(default=None, repr=False, compare=False)


def _batch_log_inventory(result):
    from core.drr_pn_batch_plan import _log_inventory
    root = Path(result.plan.root)
    paths = [Path(root / name) for name in (*result.plan.selected, *result.plan.p_candidates, *result.plan.n_candidates)]
    return _log_inventory(root, paths)


def _batch_proof_snapshot(context, result):
    from core.provenance import _digest_signature
    paths = list(batch_dependency_paths(context, result))
    paths += [Path(path) for path, _ in result.plan.snapshot]
    paths += [DrrPnStore(context.folder).manifest_path]
    for version in result.versions:
        paths.extend((version.csv_path, version.metadata_path))
    return tuple((str(path), _digest_signature(path)) for path in dict.fromkeys(paths))


def prepare_pn_batch_proof(context, result):
    """Worker-only: seal strong file stats around full content verification."""
    require_pn_batch_current(context, result)
    before = _batch_proof_snapshot(context, result)
    inventory = _batch_log_inventory(result)
    require_pn_batch_current(context, result)
    after = _batch_proof_snapshot(context, result)
    if before != after or inventory != _batch_log_inventory(result):
        raise ValueError('P/N batch inputs changed during verification.')
    return PreparedPnBatchProof(context, result.signature, after, inventory, _PROOF_SEAL)


def pn_batch_proof_is_current(context, result, proof):
    """GUI-safe: check already-known file stats and capture-log inventory."""
    from core.provenance import _digest_signature
    try:
        return bool(isinstance(proof, PreparedPnBatchProof) and proof._verified is _PROOF_SEAL
            and proof.context == context and isinstance(result, PnBatchResult) and result._verified is _SEAL
            and result.signature == proof.result_signature and result.signature == _identity(result)
            and all(_digest_signature(Path(path)) == stamp for path, stamp in proof.signatures)
            and _batch_log_inventory(result) == proof.log_inventory)
    except (OSError, ValueError, TypeError):
        return False


def generate_pn_batch(context, plan, *, feature_protection=(), force=False, cancelled=None, progress=None):
    """Save per-segment drafts and return a frozen exact mapping; never apply."""
    _check(cancelled)
    if (not isinstance(context, BackgroundContext) or not isinstance(plan, PnBatchPlan)
            or context.folder != plan.root or context.measurements != plan.selected
            or not pn_batch_plan_is_current(plan)):
        raise ValueError('P/N batch plan is incomplete, stale or belongs to another group.')
    versions, documents, reports, mapping = [], [], [], {}
    for index, segment in enumerate(plan.segments):
        _check(cancelled)
        if not pn_batch_plan_is_current(plan):
            raise ValueError('P/N batch inputs changed during generation.')
        subcontext = replace(context, measurements=segment.measurement_files)
        pair = DrrPnPair((segment.p_file,), (segment.n_file,))
        version, document, report = generate_pn(subcontext, pair, mode='reconstruction',
            training_files=segment.training_files, feature_protection=feature_protection,
            force=force, cancelled=cancelled, progress=progress)
        versions.append(version)
        documents.append(document)
        reports.append(report)
        name = portable_source_name(context.folder, version.csv_path)
        for measurement in segment.measurement_files:
            mapping[measurement] = name
        if progress is not None:
            progress(f"segment {index + 1}/{len(plan.segments)}", "complete")
    assignments = tuple(DrrMeasurementAssignment(name, 'External', (mapping[name],), 'all',
        selection_reason='Explicit reviewed P/N batch segment; generated version retained by exact assignment.')
        for name in context.measurements)
    result = PnBatchResult(plan, tuple(versions), tuple(documents), tuple(reports), assignments, _verified=_SEAL)
    result = replace(result, signature=_identity(result))
    return require_pn_batch_current(context, result, cancelled=cancelled)


def _batch_provenance(result):
    plan = result.plan
    rows = []
    for row in plan.rows:
        selected = next(option for option in row.options
                        if (option.p_file, option.n_file) == (row.p_file, row.n_file))
        rows.append(dict(measurement_file=row.measurement_file, p_file=row.p_file, n_file=row.n_file,
            measurement_p_gap=selected.measurement_p_gap, measurement_n_gap=selected.measurement_n_gap,
            pair_gap=selected.pair_gap, timing_basis=row.timing_basis, manual=row.manual,
            warnings=list(row.warnings)))
    segments = [dict(measurement_files=list(segment.measurement_files), p_file=segment.p_file,
        n_file=segment.n_file, training_files=list(segment.training_files), method=segment.method,
        version_id=version.version_id) for segment, version in zip(plan.segments, result.versions)]
    return dict(schema_version=1, p_candidates=list(plan.p_candidates), n_candidates=list(plan.n_candidates),
                rows=rows, segments=segments)


def prepare_pn_batch_application(context, result, *, warnings_acknowledged=False):
    require_pn_batch_current(context, result)
    if result.plan.warnings and not warnings_acknowledged:
        raise ValueError('P/N batch warnings require explicit acknowledgement.')
    review = {'kind': 'pn_batch', 'plan_signature': result.plan.signature,
              'warnings': list(result.plan.warnings), 'warnings_acknowledged': bool(warnings_acknowledged),
              'version_ids': [item.version_id for item in result.versions],
              'pn_batch': _batch_provenance(result)}
    draft = BackgroundDraft(kind='pn', which='all', assignment_method='matched',
        assignment_method_explicit=True, saved_assignments=result.assignments,
        relaxed_conditions=result.plan.relaxed_conditions, saved_matching_review=review, batch_result=result)
    return prepare_application(context, draft)
