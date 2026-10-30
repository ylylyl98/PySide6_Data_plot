"""Widgets-free, transactional DRR background operations.

Preparing, previewing and publishing an immutable background do not select it.
Only ``commit_application`` records application, after the owner checks context.
Scientific calculations and their certificates stay in the existing core APIs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from threading import RLock

from core.drr_sources import DrrBackgroundResolution, DrrMeasurementAssignment, portable_source_name, resolve_source_path
from core.load_cancellation import check_load_cancelled
from core.energy_ranges import parse_energy_ranges as parse_regions


@dataclass(frozen=True)
class BackgroundContext:
    folder: str
    measurements: tuple[str, ...]
    y_axis: str = "auto"
    derivative_window: int = 21
    derivative_polyorder: int = 2

    def __post_init__(self):
        object.__setattr__(self, "folder", str(Path(self.folder).resolve()))
        if isinstance(self.measurements, str):
            raise ValueError("Measurements must be a sequence of original CSV files.")
        files = tuple(portable_source_name(self.folder, resolve_source_path(self.folder, name)) for name in self.measurements)
        if not files or len(set(files)) != len(files) or any(Path(name).suffix.lower() != ".csv" for name in files):
            raise ValueError("Advanced backgrounds require distinct original intensity CSV measurements.")
        object.__setattr__(self, "measurements", files)
        if self.derivative_window < 3 or self.derivative_window % 2 != 1 or not 0 <= self.derivative_polyorder < self.derivative_window:
            raise ValueError("SG window must be odd and larger than its polynomial order.")


@dataclass(frozen=True)
class BackgroundDraft:
    kind: str = "self"
    files: tuple[str, ...] = ()
    which: str = "last"
    regions: tuple = ()
    version: object = None
    allow_stale: bool = False
    bn_plan: object = None
    fitted_result: dict | None = None
    fixed: bool = False
    auto_bn_compare: bool = False
    calibration: dict | None = None
    bn_warnings_acknowledged: bool = False
    bn_candidates: tuple[str, ...] | None = None
    pn_plan: object = None
    pn_candidates: tuple[str, ...] | None = None
    pn_warnings_acknowledged: bool = False
    saved_assignments: tuple = ()
    assignment_method: str = "common"
    assignment_method_explicit: bool = False
    relaxed_conditions: tuple[str, ...] = ()
    saved_matching_review: dict | None = None
    batch_result: object = None


@dataclass(frozen=True)
class BackgroundRecipe:
    mode: str
    files: tuple[str, ...]
    which: str
    regions: tuple = ()
    assignments: tuple = ()
    fitted_settings: dict | None = None
    fixed: bool = False
    auto_bn_compare: bool = False
    draft: BackgroundDraft | None = None
    context: BackgroundContext | None = None


def matching_pool_draft(draft):
    """Carry only an explicit matching strategy, never a previous group's proof."""
    if (draft is None or draft.kind not in {"bn", "p", "n", "pn"} or draft.fixed
            or draft.version is not None or draft.calibration is not None):
        return None
    plan = draft.bn_plan or draft.pn_plan
    if draft.assignment_method != "matched" and plan is None:
        return None
    pool = draft.bn_candidates if draft.kind == "bn" else draft.pn_candidates
    if pool is None and plan is not None:
        pool = tuple(plan.candidate_files)
    if pool is None and draft.kind != "bn":
        pool = tuple(dict.fromkeys(name for item in draft.saved_assignments
                                  for name in item.baseline_source_files))
    candidates = {"bn_candidates" if draft.kind == "bn" else "pn_candidates": pool}
    return BackgroundDraft(kind=draft.kind, which=draft.which,
        auto_bn_compare=draft.auto_bn_compare, assignment_method="matched", assignment_method_explicit=True,
        relaxed_conditions=draft.relaxed_conditions, **candidates)


@dataclass(frozen=True)
class BackgroundLoad:
    cube: object
    references: tuple
    raw_snapshot: object
    input_signatures: tuple


_PREPARED = object()
# Only small gate-validation summaries are retained. Keys include the freshly
# read content hashes, so replacing bytes (even with restored timestamps) never
# reuses an old result. Successful validation publishes at most 32 groups.
_GATE_ASSESSMENTS = OrderedDict()
_GATE_ASSESSMENTS_LOCK = RLock()


@dataclass(frozen=True)
class PreparedBackgroundApplication:
    context: BackgroundContext
    draft: BackgroundDraft
    recipe: BackgroundRecipe
    verification: object = field(repr=False)
    signatures: tuple = field(repr=False)
    _verified: object = field(default=None, repr=False, compare=False)


def _signatures(context, draft, *, digest=False):
    """Include sidecars; native composites bind their frozen source snapshots."""
    from core.drr_composite_background import composite_metadata_path
    names = list(context.measurements) + list(draft.files)
    if not draft.saved_assignments:
        # A restored exact-group mapping depends on the references actually
        # used, rather than unused candidates retained as historical context.
        names.extend(draft.bn_candidates or ())
        names.extend(draft.pn_candidates or ())
    if draft.calibration:
        names.extend(item["path"] for item in draft.calibration["payload"]["source_identity"]["baseline_files"])
    plan = draft.bn_plan or draft.pn_plan
    assignments = plan.resolution.assignments if plan is not None else draft.saved_assignments
    for assignment in assignments:
        names.extend(assignment.baseline_source_files)
    for region in draft.regions:
        names.extend(region.baseline_files)
    if draft.fitted_result and draft.fitted_result.get("texture_reference_source"):
        names.append(draft.fitted_result["texture_reference_source"])
    paths = []
    for name in dict.fromkeys(names):
        path = resolve_source_path(context.folder, name)
        paths.append(path)
        metadata = composite_metadata_path(path)
        if metadata.is_file():
            paths.append(metadata)
        if ".drr_bn_calibration" in path.parts and path.with_suffix(".npz").is_file():
            paths.append(path.with_suffix(".npz"))
    if draft.batch_result is not None:
        from ui_qt.quick.drr_pn_batch_workflow import batch_dependency_paths
        paths.extend(batch_dependency_paths(context, draft.batch_result))
    values = []
    for path in dict.fromkeys(paths):
        check_load_cancelled()
        stat = path.stat()
        fingerprint = None
        if digest:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    check_load_cancelled()
                    hasher.update(block)
            after = path.stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("A background input changed while its content was being verified.")
            fingerprint = hasher.hexdigest()
        values.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns, fingerprint))
    return tuple(values)


def _unchanged(signatures):
    try:
        return all((Path(path).stat().st_size, Path(path).stat().st_mtime_ns) == (size, stamp)
                   for path, size, stamp, _ in signatures)
    except OSError:
        return False


def _recipe(context, draft):
    mode = ("Self (first frame)" if draft.which == "first" else "Self (last frame)") if draft.kind == "self" else (
        "Fitted (sample)" if draft.kind == "fitted" else "External")
    assignments = tuple(draft.saved_assignments)
    plan = draft.bn_plan or draft.pn_plan
    if plan is not None:
        assignments = tuple(plan.resolution.assignments)
    files = tuple(draft.files)
    if assignments:
        files = tuple(dict.fromkeys(name for item in assignments for name in item.baseline_source_files))
    common = not assignments or len({(item.baseline_mode, item.baseline_files, item.baseline_which, item.regions) for item in assignments}) == 1
    fixed = (draft.fixed and draft.assignment_method != "matched" and mode == "External" and common and draft.calibration is None
        and all(item.baseline_mode == "External" for item in assignments))
    if not assignments and mode != "Fitted (sample)":
        assignments = tuple(DrrMeasurementAssignment(name, mode, files if mode == "External" else (), draft.which,
                             regions=draft.regions if mode == "External" else ()) for name in context.measurements)
    return BackgroundRecipe(mode, files, draft.which, draft.regions, assignments, draft.fitted_result,
                            fixed, draft.auto_bn_compare, draft, context)


def prepare_application(context, draft):
    """Expensive validation; run in a cancellable worker, never the GUI thread."""
    from core.data_io import _validate_drr_external_spectral_grids
    from core.drr_composite_background import is_composite_background, read_composite_background
    from core.drr_sources import assess_background_gate_files
    if not isinstance(context, BackgroundContext) or not isinstance(draft, BackgroundDraft):
        raise TypeError("A native background context and draft are required.")
    if draft.kind not in {"self", "bn", "p", "n", "pn", "custom", "fitted"} or draft.which not in {"first", "last", "all"}:
        raise ValueError("Unknown background source or frame method.")
    if draft.assignment_method not in {"common", "matched"}:
        raise ValueError("Choose a background assignment method.")
    from core.drr_match_conditions import normalize_relaxed_conditions
    relaxed = normalize_relaxed_conditions(draft.relaxed_conditions)
    draft = replace(draft, relaxed_conditions=relaxed)
    plan = draft.bn_plan or draft.pn_plan
    if plan is not None and relaxed != normalize_relaxed_conditions(getattr(plan, "relaxed_conditions", ())):
        raise ValueError("Matching conditions changed; match again and review the new assignments.")
    if plan is None and not draft.saved_assignments:
        # Common references and generated/calibrated backgrounds do not use the
        # selectable matching policy. Keep it out of their applied provenance.
        draft = replace(draft, relaxed_conditions=())
    if (draft.assignment_method == "matched" and not
            (draft.bn_plan or draft.pn_plan or draft.saved_assignments)):
        raise ValueError("Wait for background matching and review every measurement before applying.")
    if draft.kind == "self" and draft.which == "all":
        raise ValueError("Self background requires the first or last frame.")
    if draft.batch_result is not None:
        from ui_qt.quick.drr_pn_batch_workflow import require_pn_batch_current
        require_pn_batch_current(context, draft.batch_result)
        if draft.saved_assignments != draft.batch_result.assignments:
            raise ValueError('P/N batch assignments changed after review.')
    before = _signatures(context, draft, digest=True)
    verification = None
    gate_key, gate_assessment = None, None
    if sum(bool(value) for value in (draft.bn_plan, draft.pn_plan, draft.saved_assignments, draft.version, draft.calibration)) > 1:
        raise ValueError("Choose one background mapping or common reference before applying.")
    if draft.saved_assignments:
        if draft.kind not in {"bn", "p", "n", "pn", "custom"}:
            raise ValueError("Saved assignments need their recorded background source mode.")
        if tuple(item.measurement_file for item in draft.saved_assignments) != context.measurements:
            raise ValueError("Saved background assignments belong to another measurement selection.")
        for item in draft.saved_assignments:
            if item.baseline_mode == "External":
                _validate_drr_external_spectral_grids(context.folder, (item.measurement_file,), item.baseline_source_files)
            for name in item.baseline_source_files:
                path = resolve_source_path(context.folder, name)
                if is_composite_background(path):
                    read_composite_background(path)
    elif draft.bn_plan is not None:
        from core.drr_bn_matching import BnMatchCache
        plan = draft.bn_plan
        if draft.kind != "bn" or tuple(plan.selected) != context.measurements or Path(plan.root).resolve() != Path(context.folder):
            raise ValueError("BN mapping belongs to a different measurement group.")
        verification = BnMatchCache().prepare_confirm(plan, warnings_acknowledged=draft.bn_warnings_acknowledged)
    elif draft.pn_plan is not None:
        from core.drr_pn_matching import prepare_pn_confirmation
        plan = draft.pn_plan
        pool = {resolve_source_path(context.folder, name) for name in draft.pn_candidates or ()}
        expected = {resolve_source_path(context.folder, name) for name in plan.candidate_files}
        if (draft.kind not in {"p", "n", "pn"} or plan.kind != draft.kind or plan.which != draft.which
                or plan.automatic_candidates or not pool or pool != expected or tuple(plan.selected) != context.measurements
                or Path(plan.root).resolve() != Path(context.folder)):
            raise ValueError("Match the explicitly selected P/N background pool again for this group.")
        verification = prepare_pn_confirmation(plan, warnings_acknowledged=draft.pn_warnings_acknowledged)
    elif draft.version is not None:
        if draft.kind not in {"p", "n", "pn"}:
            raise ValueError("A saved P/N version needs a P/N background source mode.")
        from core.drr_pn_store import DrrPnStore
        verification = DrrPnStore(context.folder).prepare_use(draft.version.version_id,
            measurements=context.measurements, allow_stale=draft.allow_stale)
        draft = replace(draft, files=(portable_source_name(context.folder, verification.version.csv_path),), which="all")
    elif draft.calibration is not None:
        if draft.kind != "bn":
            raise ValueError("A BN calibration requires the measured BN source mode.")
        from core.drr_bn_store import DrrBnStore
        payload = draft.calibration["payload"]
        source = payload["source_identity"]
        if (Path(source["root"]).resolve() != Path(context.folder)
                or tuple(item["path"] for item in source["measurements"]) != context.measurements
                and set(item["path"] for item in source["measurements"]) != set(context.measurements)
                or str(source["y_axis"]) != str(context.y_axis)):
            raise ValueError("BN comparison belongs to another measurement group or axis.")
        if draft.calibration["original"]:
            draft = replace(draft, files=tuple(item["path"] for item in source["baseline_files"]),
                            which=source["baseline_which"], fixed=False)
            _validate_drr_external_spectral_grids(context.folder, context.measurements, draft.files)
        else:
            verification = DrrBnStore(context.folder).get(payload["version_id"])
            if (verification is None or not verification.fresh or not verification.eligible
                    or (verification.settings.get("derivative_window"), verification.settings.get("derivative_polyorder"))
                    != (context.derivative_window, context.derivative_polyorder)):
                raise ValueError("BN candidate lacks current joint DR/R + D2E validation at these SG settings.")
            draft = replace(draft, files=(portable_source_name(context.folder, verification.csv_path),), which="last", fixed=False)
    elif draft.kind == "fitted":
        from core.drr_fitted_background import validate_recipe
        validate_recipe(draft.fitted_result, context.measurements)
    elif draft.kind != "self":
        if not draft.files:
            raise ValueError("Choose background files before applying.")
        sources = tuple(dict.fromkeys((*draft.files, *(name for region in draft.regions for name in region.baseline_files))))
        _validate_drr_external_spectral_grids(context.folder, context.measurements, sources)
        for name in sources:
            if is_composite_background(resolve_source_path(context.folder, name)):
                read_composite_background(resolve_source_path(context.folder, name))
        digests = {path: digest for path, _, _, digest in before}
        paths = tuple(str(resolve_source_path(context.folder, name)) for name in draft.files)
        gate_key = tuple((path, digests[path]) for path in paths)
        with _GATE_ASSESSMENTS_LOCK:
            gate_assessment = _GATE_ASSESSMENTS.get(gate_key)
            if gate_assessment is not None:
                _GATE_ASSESSMENTS.move_to_end(gate_key)
        if gate_assessment is None:
            assessment = assess_background_gate_files(context.folder, draft.files)
            gate_assessment = assessment.all_constant, assessment.same_constant_values
            # An unknown profile can reflect a transient read failure. Keep
            # the existing validation behavior, but never memoize uncertainty.
            if (len(assessment.profiles) != len(draft.files)
                    or any(profile.varies is None or profile.frame_count is None
                           for profile in assessment.profiles)):
                gate_key = None
        if gate_assessment[0] and len(draft.files) > 1 and not gate_assessment[1]:
            raise ValueError("Different constant background gates require an explicit P + N composite.")
    check_load_cancelled()
    if not _unchanged(before):
        raise ValueError("Background inputs changed during validation. Review the draft again.")
    after = _signatures(context, draft, digest=True)
    # Detect changed bytes even if a producer preserved a file timestamp.
    hashes = {path: digest for path, _, _, digest in before}
    if any(path in hashes and hashes[path] != digest for path, _, _, digest in after):
        raise ValueError("Background inputs changed during validation.")
    if gate_key is not None:
        with _GATE_ASSESSMENTS_LOCK:
            _GATE_ASSESSMENTS[gate_key] = gate_assessment
            _GATE_ASSESSMENTS.move_to_end(gate_key)
            while len(_GATE_ASSESSMENTS) > 32:
                _GATE_ASSESSMENTS.popitem(last=False)
    if draft.batch_result is not None:
        from ui_qt.quick.drr_pn_batch_workflow import prepare_pn_batch_proof
        verification = prepare_pn_batch_proof(context, draft.batch_result)
    return PreparedBackgroundApplication(context, draft, _recipe(context, draft), verification, after, _PREPARED)


def commit_application(prepared, *, current_context, settings):
    """Commit after successful trial loading/processing and the GUI context guard.

    The owner must keep the old visible products until this commit succeeds.
    Preparation and ``load_with_recipe`` never publish an applied association.
    """
    if not isinstance(prepared, PreparedBackgroundApplication) or prepared._verified is not _PREPARED:
        raise ValueError("Prepare and verify the background before application.")
    if prepared.context != current_context or not _unchanged(prepared.signatures):
        raise ValueError("Measurement context or background inputs changed; the previous background was retained.")
    draft, verification = prepared.draft, prepared.verification
    if draft.batch_result is not None:
        from ui_qt.quick.drr_pn_batch_workflow import pn_batch_proof_is_current
        if not pn_batch_proof_is_current(current_context, draft.batch_result, verification):
            raise ValueError("P/N batch inputs changed after validation; the previous background was retained.")
    if draft.bn_plan is not None:
        from core.drr_bn_matching import BnMatchCache
        try:
            cache = BnMatchCache.from_dict(json.loads(settings.value("drr/confirmed_bn_matches_v1", "{}")))
        except (ValueError, TypeError, KeyError):
            cache = BnMatchCache()
        cache.commit_confirm(verification)
        settings.setValue("drr/confirmed_bn_matches_v1", json.dumps(cache.to_dict()))
    elif draft.pn_plan is not None:
        from core.drr_pn_matching import confirmation_is_current
        if not confirmation_is_current(verification):
            raise ValueError("P/N matching inputs changed before application.")
    elif draft.version is not None:
        from core.drr_pn_store import DrrPnStore
        DrrPnStore(current_context.folder).commit_prepared_use(verification)
    elif draft.calibration is not None:
        from core.drr_bn_store import DrrBnStore
        store = DrrBnStore(current_context.folder)
        if draft.calibration["original"]:
            store.deactivate(current_context.measurements, draft.files, draft.which, restore_original=True)
        else:
            store.activate_verified(verification, measurements=current_context.measurements)
    settings.setValue("drr/bn_auto_compare", draft.auto_bn_compare)
    return prepared.recipe


def background_input_signatures(context, recipe):
    """Worker-side content identity for loading and subsequent export checks."""
    draft = recipe.draft or BackgroundDraft(kind="fitted" if recipe.mode == "Fitted (sample)" else
        "self" if recipe.mode.startswith("Self") else "custom", files=recipe.files, which=recipe.which,
        regions=recipe.regions, fitted_result=recipe.fitted_settings, saved_assignments=recipe.assignments if recipe.mode == "External" else ())
    return _signatures(context, draft, digest=True)


def prepare_fixed_application(old_recipe, new_context):
    """Rebind only an explicitly fixed common external recipe to a new group.

    Per-group matching proofs never transfer. A P/N version is independently
    prepared for the new measurement group; a BN calibration can never transfer.
    Experiment changes need an explicit new source choice, avoiding basename
    rebinding to unrelated files in another directory.
    """
    if (not isinstance(old_recipe, BackgroundRecipe) or not old_recipe.fixed or old_recipe.mode != "External"
            or old_recipe.draft is None or old_recipe.draft.calibration is not None):
        raise ValueError("Only an explicitly fixed common measured/custom background can follow a new group.")
    if old_recipe.context is None or old_recipe.context.folder != new_context.folder:
        raise ValueError("Choose the fixed reference explicitly after changing experiment folders.")
    recipes = {(item.baseline_files, item.baseline_which, item.regions) for item in old_recipe.assignments}
    if len(recipes) != 1:
        raise ValueError("Heterogeneous per-measurement assignments cannot be fixed across groups.")
    files, which, regions = next(iter(recipes))
    draft = replace(old_recipe.draft, files=files, which=which, regions=regions, fixed=True,
        bn_plan=None, pn_plan=None, saved_assignments=(), bn_candidates=None, batch_result=None,
        assignment_method="common",
        relaxed_conditions=(),
        pn_candidates=files if old_recipe.draft.kind in {"p", "n", "pn"} else None,
        bn_warnings_acknowledged=False, pn_warnings_acknowledged=False)
    return prepare_application(new_context, draft)


def load_with_recipe(context, recipe, *, prepared=None, progress=None, log=None, before_signatures=None):
    """Load exact legacy scientific semantics without recording application.

    Heterogeneous references are corrected per member before averaging. Native
    composite boundaries retain segmented derivatives. Fitted references average
    raw intensity first and fit one denominator. Returned references are needed
    for exporting the actual fitted background and its complete metadata.
    A caller that has already fingerprinted this recipe can pass that snapshot
    to include its own validation in the same before/after content check.
    """
    from core.data_io import load_drr_resolved_cube
    from core.drr_raw_preview import RawSnapshot
    if prepared is not None and (not isinstance(prepared, PreparedBackgroundApplication)
            or prepared._verified is not _PREPARED or prepared.context != context or prepared.recipe != recipe
            or not _unchanged(prepared.signatures)):
        raise ValueError("This prepared background no longer belongs to the current selection.")
    before = (background_input_signatures(context, recipe)
              if before_signatures is None else before_signatures)
    references = ()
    calibration = recipe.draft.calibration if recipe.draft else None
    if calibration and not calibration["original"]:
        from core.drr_bn_store import DrrBnStore
        version = DrrBnStore(context.folder).get(calibration["payload"]["version_id"])
        if (version is None or not version.fresh or not version.eligible or version.validation_status != "accepted"
                or (version.settings.get("derivative_window"), version.settings.get("derivative_polyorder"))
                != (context.derivative_window, context.derivative_polyorder)):
            raise ValueError("The active BN candidate needs a fresh joint DR/R + D2E certificate at the current SG settings.")
    if recipe.mode == "Fitted (sample)":
        from core.drr_fitted_background import load_fitted_group
        cube, references = load_fitted_group(context.folder, context.measurements, recipe.fitted_settings,
            y_axis=context.y_axis, progress=progress, log=log)
    elif calibration and not calibration["original"] and prepared is not None:
        # The normal loading facade requires an activated BN history choice.
        # This trial load uses only the fully verified candidate snapshot, then
        # the owner commits activation after all loading/processing succeeds.
        from core.drr_bn_store import DrrBnStore
        from core.loader import load_drr_avg
        version = DrrBnStore(context.folder).get(prepared.verification.version_id)
        if (version is None or not version.fresh or not version.eligible
                or version.validation_status != "accepted"
                or version.source_identity.get("y_axis") != str(context.y_axis)
                or set(item["path"] for item in version.source_identity["measurements"]) != set(context.measurements)
                or (version.settings.get("derivative_window"), version.settings.get("derivative_polyorder"))
                    != (context.derivative_window, context.derivative_polyorder)):
            raise ValueError("BN trial loading requires the current exact-group joint DR/R + D2E certificate.")
        payload = DrrBnStore(context.folder).load_result(version)
        cube = load_drr_avg(context.folder, context.measurements, bg_mode="external", y_axis=context.y_axis,
            external_vector=payload["candidate_background"], external_energy=payload["energy"])
        cube.drr_numerical_path = "common"
        cube.drr_assignments = tuple(item.to_dict() for item in recipe.assignments)
    else:
        cube = load_drr_resolved_cube(context.folder, context.measurements, recipe.assignments, y_axis=context.y_axis)
    check_load_cancelled()
    after = background_input_signatures(context, recipe)
    if before != after:
        raise ValueError("A measurement or background changed during loading; the applied result was retained.")
    snapshot = RawSnapshot(context.folder, context.measurements, recipe.assignments, context.y_axis,
        getattr(cube, "drr_numerical_path", "fitted_raw_average" if references else "common"))
    return BackgroundLoad(cube, tuple(references), snapshot, after)


def generate_pn(context, pair, *, mode="reconstruction", training_files=(), p_which="all", n_which="all",
                feature_protection=(), force=False, cancelled=None, progress=None):
    """Explicitly publish a version, without associating it with measurements."""
    from core.drr_pn_store import DrrPnStore
    from core.drr_composite_background import is_composite_background
    if any(Path(name).suffix.lower() != ".csv" or is_composite_background(resolve_source_path(context.folder, name))
           for name in (*pair.p_files, *pair.n_files)):
        raise ValueError("P/N generation requires original intensity CSV sources, not saved/composite backgrounds.")
    store = DrrPnStore(context.folder)
    if mode == "reconstruction":
        from core.drr_pn_reconstruction.app import prepare_reconstruction_request, generate_reconstruction_background
        if p_which != "all" or n_which != "all":
            raise ValueError("P/N reconstruction uses all acquired frames in each original source.")
        request = prepare_reconstruction_request(context.folder, pair, context.measurements,
            training_files=training_files,
            method="conditioned_single" if len(context.measurements) == 1 else "conditioned_bounded",
            feature_protection=feature_protection)
        function = lambda: generate_reconstruction_background(context.folder, pair, request=request,
            cancelled=cancelled, progress=progress, force_source_refit=force)
        key = request.key
    elif mode == "assisted":
        from core.drr_pn_assisted import prepare_assisted_request, generate_assisted_background
        request = prepare_assisted_request(context.folder, pair, p_which, n_which, measurements=context.measurements)
        key = request.key
        function = lambda: generate_assisted_background(context.folder, pair, p_which, n_which, request=request, cancelled=cancelled)
    elif mode == "pn-only":
        from core.drr_pn_generator import ALGORITHM_VERSION, default_settings, dependency_paths, generate_pn_background
        key = store.prepare(pair, p_which=p_which, n_which=n_which, settings=default_settings(),
                            algorithm_version=ALGORITHM_VERSION, dependencies=dependency_paths())
        function = lambda: generate_pn_background(context.folder, pair, p_which, n_which, cancelled=cancelled)
    else:
        raise ValueError("Select an explicit P/N generation method.")
    reused = None if force else store.find_reusable(key)
    if reused is not None:
        from core.drr_composite_background import read_composite_background
        return reused, read_composite_background(reused.csv_path), {"status": "Reused identical immutable version"}
    result = function()
    check_load_cancelled()
    version = store.save(result.document, key, force_new=force)
    return version, result.document, result.report


def compare_bn(context, files, which, *, settings=None, cancellation=None, progress=None, log=None):
    from core.drr_bn_service import compare_bn_group
    from core.drr_bn_store import DrrBnStore
    config = {**dict(settings or {}), "derivative_window": context.derivative_window,
              "derivative_polyorder": context.derivative_polyorder}
    payload = compare_bn_group(context.folder, context.measurements, files, which, config,
        y_axis=context.y_axis, cancellation=cancellation, progress=progress, log=log)
    check_load_cancelled()
    version = DrrBnStore(context.folder).save_result(payload)
    # Representative mean spectra are an inspector preview, not the unreduced
    # per-repeat/per-gate evidence that determines the core certificate.
    import numpy as np
    from core.loader import DataCube
    from core.processing import apply_sg_derivative_energy
    raw = np.asarray(payload["mean_raw"], float).mean(axis=0)
    ratios = np.stack((raw/payload["original_background"]-1, raw/payload["candidate_background"]-1))
    derivative, used = apply_sg_derivative_energy(DataCube(payload["energy"], np.arange(2), ratios,
        "Preview", "Representative mean DRR", "DR/R"), derivative=2,
        window_length=context.derivative_window, polyorder=context.derivative_polyorder)
    return {**payload, "version_id": version.version_id, "csv_path": str(version.csv_path),
            "preview": dict(drr=ratios, d2=derivative.Z, energy=derivative.energy, window=used)}


def import_pn(context, pair, path, *, p_which="all", n_which="all"):
    """Import native CSV + sidecar, retaining strict training/code provenance."""
    from core.drr_pn_store import DrrPnStore
    document = read_composite_background(resolve_source_path(context.folder, path))
    source = next(item for item in document.sources if item.key == document.default_source)
    report = json.loads(source.generation_json) if source.generation_json else {}
    method = report.get("algorithm_version") or report.get("algorithm") or "imported_composite"
    settings = report.get("settings", {})
    dependencies = ()
    from core.drr_pn_assisted import ASSISTED_ALGORITHM_VERSION
    if method == ASSISTED_ALGORITHM_VERSION:
        from core import drr_pn_assisted
        dependencies = _assisted_import_dependencies(context.folder, resolve_source_path(context.folder, path), settings, drr_pn_assisted)
    elif str(method).startswith("pn_raw_") or report.get("source_models_refitted_from_raw"):
        from core.drr_pn_reconstruction import app
        dependencies = _reconstruction_import_dependencies(context.folder, resolve_source_path(context.folder, path), settings, app,
            pair=pair, p_which=p_which, n_which=n_which)
    return DrrPnStore(context.folder).import_saved(path, pair, algorithm_version=method, settings=settings,
        dependencies=dependencies, p_which=p_which, n_which=n_which, provenance={"origin": "explicit_user_import"})


# These pure provenance checks mirror the existing P/N workflow.
from core.drr_composite_background import composite_metadata_path, read_composite_background
from core.drr_pn_store import _file_digest


def _assisted_import_dependencies(folder, path, settings, api):
    """Do not relabel unknown or changed training inputs as a fresh import."""
    training = settings.get("sample_analysis", {}).get("training_files", ())
    metadata = json.loads(composite_metadata_path(path).read_text(encoding="utf-8"))
    records = metadata.get("pn_generation", {}).get("dependencies", ())
    if not training or not records:
        raise ValueError("Sample-assisted import needs its original training-file fingerprints. "
                         "Open an existing historical snapshot or regenerate with the current samples.")
    original = {resolve_source_path(folder, item["path"]): item["sha256"] for item in records}
    training_paths = tuple(resolve_source_path(folder, name) for name in training)
    for target in training_paths:
        if target not in original or not target.is_file() or _file_digest(target) != original[target]:
            raise ValueError(f"Cannot import this assisted version as fresh: training input {target.name} "
                             "is missing, changed, or has no original fingerprint. Existing historical versions remain available.")
    code_paths = tuple(Path(name).resolve() for name in api.dependency_paths())
    code_records = [item for item in records if resolve_source_path(folder, item["path"]) not in training_paths]
    used = set()
    for target in code_paths:
        matches = [(index, item) for index, item in enumerate(code_records)
                   if str(item["path"]).replace("\\", "/").rsplit("/", 1)[-1].casefold() == target.name.casefold()]
        if (len(matches) != 1 or not target.is_file()
                or _file_digest(target) != matches[0][1]["sha256"]):
            raise ValueError(f"Cannot import this assisted version as fresh: original implementation {target.name} "
                             "is missing, ambiguous, or differs from the current implementation. "
                             "Open its existing historical snapshot or generate a new draft.")
        used.add(matches[0][0])
    if len(used) != len(code_records):
        raise ValueError("Assisted import has generation dependencies that cannot be mapped to this implementation.")
    return (*code_paths, *training_paths)


def _reconstruction_import_dependencies(folder, path, settings, api, *, pair, p_which, n_which):
    """Validate store exports or the public native writer's self-contained proof."""
    from core.drr_pn_reconstruction.pipeline import (
        ALGORITHM_VERSION as original_version, GENERALIZED_ALGORITHM_VERSION,
        LOCAL_ALGORITHM_VERSION, CONDITIONED_ALGORITHM_VERSION,
        CONDITIONED_BOUNDED_ALGORITHM_VERSION, CONDITIONED_SINGLE_ALGORITHM_VERSION,
        PROTOCOL, GENERALIZED_PROTOCOL, LOCAL_PROTOCOL, CONDITIONED_PROTOCOL,
        CONDITIONED_BOUNDED_PROTOCOL, CONDITIONED_SINGLE_PROTOCOL,
    )
    document = read_composite_background(path)
    source = next(s for s in document.sources if s.key == document.default_source)
    report = json.loads(source.generation_json) if source.generation_json else {}
    method = report.get("algorithm_version")
    protocols = {original_version: PROTOCOL, GENERALIZED_ALGORITHM_VERSION: GENERALIZED_PROTOCOL,
                 LOCAL_ALGORITHM_VERSION: LOCAL_PROTOCOL,
                 CONDITIONED_ALGORITHM_VERSION: CONDITIONED_PROTOCOL,
                 CONDITIONED_BOUNDED_ALGORITHM_VERSION: CONDITIONED_BOUNDED_PROTOCOL,
                 CONDITIONED_SINGLE_ALGORITHM_VERSION: CONDITIONED_SINGLE_PROTOCOL}
    if method not in protocols:
        raise ValueError(f"Unsupported reconstruction algorithm: {method!r}. "
                         "It cannot be imported as a fresh reconstruction with this implementation.")
    if (p_which, n_which) != ("all", "all"):
        raise ValueError("Reconstruction import needs All frames for both sources.")
    expected = dict(protocols[method])
    single = method == CONDITIONED_SINGLE_ALGORITHM_VERSION
    training_count = 1 if single else 2
    analysis = settings.get("sample_analysis")
    if analysis is not None:
        if len(analysis.get("training_files", [])) != training_count:
            raise ValueError(f"Reconstruction settings need the original {training_count} training-file roles.")
        expected["sample_analysis"] = {
            "training_files": analysis.get("training_files", []),
            "subset": "all gates and full native energy", "y_axis": "auto",
            "split_policy": ("single-measurement gate-group evidence; independent sample repeatability unavailable"
                if single else "explicit-two-training-repeats; all other selected files check only")}
    if settings.get("feature_protection"):
        from core.drr_pn_reconstruction.feature_protection import normalize_feature_protection
        if method not in (CONDITIONED_BOUNDED_ALGORITHM_VERSION, CONDITIONED_SINGLE_ALGORITHM_VERSION):
            raise ValueError("Feature protection requires the conditioned bounded reconstruction method.")
        expected["feature_protection"] = normalize_feature_protection(settings["feature_protection"])
    if settings != expected or report.get("settings") != settings:
        raise ValueError("Reconstruction settings do not match the declared implementation protocol.")
    records = report.get("input_records", [])
    if not records:
        # Store-native exports can carry their verified dependency records instead.
        metadata = json.loads(composite_metadata_path(path).read_text(encoding="utf8"))
        original = metadata.get("pn_generation", {})
        if (original.get("algorithm_version") != method or original.get("settings") != settings
                or original.get("frame_choice") != {"P": "all", "N": "all"}):
            raise ValueError("Reconstruction export has no consistent original generation record.")
        for role, selected in (("P", pair.p_files), ("N", pair.n_files)):
            sources = original.get("pair", {}).get(role, [])
            if (len(sources) != 1 or len(selected) != 1
                    or resolve_source_path(folder, sources[0]["path"]) != resolve_source_path(folder, selected[0])
                    or _file_digest(resolve_source_path(folder, selected[0])) != sources[0].get("sha256")):
                raise ValueError("Reconstruction export does not match the selected original P/N roles.")
        return _assisted_import_dependencies(folder, path, settings, api)
    if [row.get("role") for row in records] != ["P", "N", *(["training"] * training_count)]:
        raise ValueError(f"Reconstruction export needs the original P/N and {training_count} training-file roles.")
    paths = tuple(resolve_source_path(folder, row["path"]) for row in records)
    selected = tuple(resolve_source_path(folder, name) for name in (*pair.p_files, *pair.n_files))
    if len(set(paths)) != 2 + training_count or paths[:2] != selected:
        raise ValueError("Reconstruction export does not match the selected original P/N roles.")
    if analysis is not None and tuple(resolve_source_path(folder, name)
                                     for name in analysis.get("training_files", [])) != paths[2:]:
        raise ValueError("Reconstruction training settings disagree with its original input fingerprints.")
    for target, record in zip(paths, records):
        if not target.is_file() or _file_digest(target) != record.get("sha256"):
            raise ValueError(f"Original reconstruction input {target.name} is missing or changed.")
    code = report.get("implementation_sha256", {})
    current = tuple(Path(name).resolve() for name in api.dependency_paths())
    if not code or not current or len(code) != len(current):
        raise ValueError("Reconstruction export has incomplete implementation fingerprints.")
    used = set()
    for target in current:
        matches = [(name, digest) for name, digest in code.items()
                   if str(name).replace("\\", "/").rsplit("/", 1)[-1].casefold() == target.name.casefold()]
        if len(matches) != 1 or not target.is_file() or _file_digest(target) != matches[0][1]:
            raise ValueError(f"Original reconstruction implementation {target.name} is missing, ambiguous or changed.")
        used.add(matches[0][0])
    if len(used) != len(code):
        raise ValueError("Reconstruction export contains unmatched implementation fingerprints.")
    return (*current, *paths[2:])
