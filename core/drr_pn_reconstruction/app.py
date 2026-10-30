"""Frozen app requests for raw P/N reconstruction, with explicit sample roles.

Only the explicitly selected training files enter generation. Other measurements are
listed as checks but never hashed into the generator key or read by fitting.
Saving and applying remain the existing dialog/store's responsibility.
"""
from dataclasses import dataclass, replace
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

from core.drr_composite_background import CompositeBackground, is_composite_background
from core.drr_pn_generator import PnGenerationResult, _CallbackCancellation
from core.drr_pn_pairing import DrrPnPair
from core.drr_pn_store import DrrPnStore, PnGenerationKey
from core.drr_sources import portable_source_name, resolve_source_path
from core.load_cancellation import check_load_cancelled, load_scope
from .io import dependency_paths, generate_from_files
from .feature_protection import normalize_feature_protection
from .pipeline import (
    GENERALIZED_ALGORITHM_VERSION, GENERALIZED_PROTOCOL,
    CONDITIONED_BOUNDED_ALGORITHM_VERSION, CONDITIONED_BOUNDED_PROTOCOL,
    CONDITIONED_SINGLE_ALGORITHM_VERSION, CONDITIONED_SINGLE_PROTOCOL,
)


_METHOD_SPECS = {
    "generalized": (GENERALIZED_ALGORITHM_VERSION, GENERALIZED_PROTOCOL,
                    "PN adaptive reconstruction — draft"),
    "conditioned_bounded": (CONDITIONED_BOUNDED_ALGORITHM_VERSION, CONDITIONED_BOUNDED_PROTOCOL,
                            "PN conditioned + bounded texture — draft"),
    "conditioned_single": (CONDITIONED_SINGLE_ALGORITHM_VERSION, CONDITIONED_SINGLE_PROTOCOL,
                           "PN single measurement reconstruction — draft"),
}


def _method_spec(method):
    if not isinstance(method, str) or method not in _METHOD_SPECS:
        raise ValueError("Unknown P/N app reconstruction method.")
    return _METHOD_SPECS[method]


def _method_for_version(version):
    for method, (algorithm, _protocol, _label) in _METHOD_SPECS.items():
        if version == algorithm:
            return method
    raise ValueError("Unknown P/N reconstruction request algorithm version.")


@dataclass(frozen=True)
class ReconstructionRequest:
    key: PnGenerationKey
    training_files: tuple[str, ...]
    evaluation_files: tuple[str, ...]

    @property
    def settings(self):
        return self.key.provenance["settings"]

    @property
    def method(self):
        """Derive routing from the frozen key; there is no second method field."""
        return _method_for_version(self.key.provenance.get("algorithm_version"))


def _settings(training_files, method="generalized", feature_protection=()):
    _algorithm, protocol, _label = _method_spec(method)
    protected = normalize_feature_protection(feature_protection)
    if protected and method not in ("conditioned_bounded", "conditioned_single"):
        raise ValueError("Feature protection requires conditioned_bounded or conditioned_single reconstruction.")
    settings = {
        **protocol,
        "sample_analysis": {"training_files": list(training_files),
            "subset": "all gates and full native energy", "y_axis": "auto",
            "split_policy": ("single-measurement gate-group evidence; independent sample repeatability unavailable"
                if method == "conditioned_single" else
                "explicit-two-training-repeats; all other selected files check only")},
    }
    if protected:
        settings["feature_protection"] = protected
    return settings


def prepare_reconstruction_request(folder, pair, measurements, *, training_files,
                                   method="generalized", feature_protection=()):
    """Freeze training roles, declared method, source bytes and implementation.

    ``generalized`` retains the existing API default. ``conditioned_bounded``
    is an explicit opt-in for the conditioned parent with optional bounded raw
    common fine correction. ``conditioned_single`` uses one complete measurement
    with conditional gate evidence, without independent sample repeat checks.
    No request authorizes automatic application.
    """
    algorithm, _protocol, _label = _method_spec(method)
    single = method == "conditioned_single"
    pair = DrrPnPair(**pair) if isinstance(pair, dict) else pair
    if len(pair.p_files) != 1 or len(pair.n_files) != 1:
        raise ValueError("Raw reconstruction needs one original file per P/N role and all frames.")
    if isinstance(measurements, (str, Path)) or isinstance(training_files, (str, Path)):
        raise ValueError("Select measurement files and two distinct training repeats.")
    paths = tuple(resolve_source_path(folder, name) for name in measurements)
    train = tuple(resolve_source_path(folder, name) for name in training_files)
    sources = tuple(resolve_source_path(folder, name) for name in (*pair.p_files, *pair.n_files))
    if single and (len(train) != 1 or len(paths) != 1):
        raise ValueError("Single measurement reconstruction requires exactly one selected measurement and one training file.")
    if not single and (len(train) != 2 or len(set(train)) != 2):
        raise ValueError("Select exactly two distinct training measurement repeats.")
    if len(set(paths)) != len(paths) or not set(train).issubset(paths):
        raise ValueError("Training files must be distinct members of the selected measurement group.")
    if len(set(sources)) != 2 or set(sources).intersection(paths):
        raise ValueError("P, N and sample measurements must be distinct original files.")
    for path in (*sources, *paths):
        if path.suffix.lower() != ".csv" or is_composite_background(path):
            raise ValueError("Reconstruction requires original intensity CSVs, not saved backgrounds.")
    training = tuple(portable_source_name(folder, path) for path in train)
    checks = tuple(portable_source_name(folder, path) for path in paths if path not in train)
    settings = _settings(training, method, feature_protection)
    key = DrrPnStore(folder).prepare(pair, p_which="all", n_which="all", settings=settings,
        algorithm_version=algorithm, dependencies=(*dependency_paths(), *train))
    return ReconstructionRequest(key, training, checks)


def _verify_request(folder, pair, request):
    store = DrrPnStore(folder)
    provenance = request.key.provenance
    method = request.method
    if (Path(request.key.root).resolve() != store.root
            or request.key.pair_id != store.pair_id(pair)
            or provenance.get("frame_choice") != {"P": "all", "N": "all"}
            or provenance.get("settings") != _settings(request.training_files, method,
                provenance.get("settings", {}).get("feature_protection", ()))):
        raise ValueError("The reconstruction request no longer matches the selected files or method.")
    # A self-consistent store key is not sufficient if a caller omitted a
    # training/code dependency or attached a different source pair to its id.
    try:
        source_paths = {role: tuple(resolve_source_path(folder, row["path"])
                                   for row in provenance["pair"][role]) for role in ("P", "N")}
        expected_sources = {role: tuple(resolve_source_path(folder, name) for name in names)
                            for role, names in (("P", pair.p_files), ("N", pair.n_files))}
        training = tuple(resolve_source_path(folder, name) for name in request.training_files)
        dependencies = tuple(resolve_source_path(folder, row["path"]) for row in provenance["dependencies"])
        expected_dependencies = tuple(Path(path).resolve() for path in (*dependency_paths(), *training))
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid reconstruction source or dependency identity.") from exc
    if (source_paths != expected_sources or any(len(paths) != 1 for paths in source_paths.values())
            or len(training) != (1 if method == "conditioned_single" else 2)
            or len(set(training)) != len(training)
            or (method == "conditioned_single" and request.evaluation_files)
            or set(training).intersection((*source_paths["P"], *source_paths["N"]))
            or dependencies != expected_dependencies):
        raise ValueError("The reconstruction source or dependency identity does not match the frozen method and training files.")
    store._validate_key(request.key)


def generate_reconstruction_background(folder, pair, *, request, cancelled=None, progress=None,
                                       force_source_refit=False):
    """Return a new unsaved draft, never substitute an old experimental method."""
    scope = load_scope(_CallbackCancellation(cancelled)) if cancelled is not None else nullcontext()
    with scope:
        check_load_cancelled()
        result = _generate_reconstruction_background(folder, pair, request=request,
            cancelled=cancelled, progress=progress, force_source_refit=force_source_refit)
        check_load_cancelled()
        return result


def _generate_reconstruction_background(folder, pair, *, request, cancelled, progress,
                                        force_source_refit=False):
    pair = DrrPnPair(**pair) if isinstance(pair, dict) else pair
    _verify_request(folder, pair, request)
    method = request.method
    algorithm, protocol, label = _method_spec(method)
    protected = request.settings.get("feature_protection", ())
    if protected and method != "conditioned_single":
        label = "PN conditioned + explicit feature protection — draft"
    options = {"feature_protection": protected} if protected else {}
    result = generate_from_files(folder, pair.p_files[0], pair.n_files[0], request.training_files,
        method=method, cancelled=cancelled, progress=progress,
        force_source_refit=force_source_refit, **options)
    _verify_request(folder, pair, request)
    expected_protocol = dict(protocol)
    if protected:
        expected_protocol["feature_protection"] = protected
    if (result.report.get("algorithm_version") != algorithm
            or result.report.get("settings") != expected_protocol):
        raise ValueError("Generated reconstruction method/protocol does not match the frozen request.")
    report = {**result.report, "settings": request.settings,
        "reconstruction_method": method,
        "training_files": list(request.training_files),
        "generation_fingerprint": request.key.fingerprint,
        "p_frame_policy": "all", "n_frame_policy": "all",
        "automatic_application": False,
        "scientifically_validated": False,
        "scientific_acceptance": "Review dR/R and d2E for the intended features, their baseline neighborhoods and repeatability in this measurement group. Outside-range residuals are not an automatic rejection of local use; numerical generation does not establish scientific validation."}
    if method == "conditioned_single":
        report["scientific_acceptance"] = (
            "Review dR/R, d2E and feature neighborhoods. Gate groups and conditional P/N halves do not establish "
            "independent sample repeatability or cross-measurement drift. Fixed sample resonances may imitate "
            "background texture. Numerical generation does not establish scientific validation.")
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)
    key = hashlib.sha256(encoded.encode("utf8") + result.arrays["background"].tobytes()).hexdigest()
    source = replace(result.document.sources[-1], key=f"PN-reconstructed-{key[:20]}",
        label=label, generation_json=encoded)
    document = CompositeBackground((*result.document.sources[:-1], source), source.key)
    return PnGenerationResult(document, report, result.arrays)
