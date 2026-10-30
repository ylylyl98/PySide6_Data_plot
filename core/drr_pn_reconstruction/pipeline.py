"""Complete raw-array reconstruction, with historical data used only by tests."""
from __future__ import annotations

import hashlib
from time import perf_counter
import numpy as np

from core.load_cancellation import check_load_cancelled

ALGORITHM_VERSION = "pn_raw_v66_reconstruction_v1"
GENERALIZED_ALGORITHM_VERSION = "pn_raw_adaptive_regions_v1"
LOCAL_ALGORITHM_VERSION = "pn_raw_local_evidence_regions_v1"
CONDITIONED_ALGORITHM_VERSION = "pn_raw_conditioned_regions_v1"
CONDITIONED_BOUNDED_ALGORITHM_VERSION = "pn_raw_conditioned_bounded_texture_v2"
CONDITIONED_SINGLE_ALGORITHM_VERSION = "pn_raw_conditioned_single_gate_v1"
PROTOCOL = {
    "source_preparation": "adopted_v12_v20_v21_v24_v29_v31_v32",
    "classification": "two_explicit_training_repeats_full_gate_axis",
    "rendering": "v54_v57_v63_v64_v65_v66",
    "region_scope": "Historical three-pure-P-core recipe; third core receives the local shape repair.",
    "half_check": "conditional application of this run's full source fits and classifications",
}
GENERALIZED_PROTOCOL = {
    **PROTOCOL,
    "rendering": "source_symmetric_resolved_cores_and_local_curvature_bridges",
    "region_scope": "Select each current P or N core with a complete SG21 exclusive-source witness; no fixed count, ordinal, angle or energy region.",
}
LOCAL_PROTOCOL = {
    **GENERALIZED_PROTOCOL,
    "rendering": "local_exclusive_evidence_parent_preserving_shape_increment",
    "region_scope": "Current exclusive-source runs with complete SG21 support, two local bridges, no extrapolated pure-source tail.",
}
CONDITIONED_PROTOCOL = {
    **GENERALIZED_PROTOCOL,
    "rendering": "bounded_unopposed_regional_rule_with_compact_conflict_repairs",
    "region_scope": "Preserve existing regional construction for bounded unopposed source cores; restrict boundary/conflicted cores to supported compact repairs without changing protected core/bridge supports.",
}
CONDITIONED_BOUNDED_PROTOCOL = {
    **CONDITIONED_PROTOCOL,
    "source_preparation": "accepted_residual_v12_v20_v21_v24_v29_v31_v32",
    "source_alignment_policy": "accepted_only",
    "rendering": "conditioned_parent_with_optional_bounded_raw_common_fine",
    "texture_analysis": "raw_intensity_mean_log_Q4_full_fit_and_six_conditional_folds",
    "texture_support": "frozen_raw_common_contiguous_support_v1",
    "texture_eligibility": "current_v45_witness_eligible",
    "texture_selection": "original_proposal_rule_then_recomputed_bounded_full_and_eligible_CV",
    "half_check": "current conditioned half parents and their own raw common fine; full fitted transform and support held fixed",
    "automatic_application": False,
    "scientifically_validated": False,
}

CONDITIONED_SINGLE_PROTOCOL = {**CONDITIONED_PROTOCOL,
    "source_preparation": "accepted_residual_v12_v20_v21_v24_v29_v31_v32",
    "source_alignment_policy": "accepted_only", "automatic_application": False,
    "classification": "single_measurement_gate_group_stability_PN_half_bound_v1",
    "bounded_texture": "not_evaluated_requires_independent_sample_repeats", "independent_sample_repeats": 1,
    "scientifically_validated": False}


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def reconstruct_background(energy, pn_mean, pn_frames, training_samples, *,
                           cancelled=None, progress=None, method="v66",
                           source_cache=None, source_cache_identity=None,
                           force_source_refit=False, feature_protection=()):
    """Recompute source fits, classifications, references and regional construction.

    This entry point intentionally has no prepared-source, saved-background or
    BN arguments. Two explicit training repeats inform the existing methods;
    ``conditioned_single`` uses one measurement and conditional gate evidence. The
    ``method='v66'`` retains the historical topology for regression;
    ``method='generalized'`` selects current source-symmetric regions and reports
    deferred repairs; ``method='local_evidence'`` tests compact repairs over
    current exclusive evidence; ``method='conditioned'`` applies those compact
    repairs only to boundary/conflicted source cores. ``conditioned_bounded``
    adds an optional raw-common fine correction with frozen full-fit parameters,
    contiguous support and conditional fold checks. Rejection retains the exact
    conditioned parent. Each method reconstructs from the current inputs,
    without an old background template. Conditional support is not scientific
    acceptance, and shared P/N structure need not be purely optical.

    Returns ``(arrays, report)``. Conditional half backgrounds are diagnostics,
    not independently refitted backgrounds or independent scientific validation.
    """
    started = perf_counter()
    from .source_pipeline import prepare_sources
    from .regions_pipeline import prepare_regions
    from .feature_protection import normalize_feature_protection
    protected = normalize_feature_protection(feature_protection)
    if protected and method not in {"conditioned_bounded", "conditioned_single"}:
        raise ValueError("Feature protection requires conditioned_bounded reconstruction.")
    if method == "v66":
        from .render_pipeline import render_v66 as render
        version, protocol = ALGORITHM_VERSION, PROTOCOL
    elif method == "generalized":
        from .render_generalized import render_generalized as render
        version, protocol = GENERALIZED_ALGORITHM_VERSION, GENERALIZED_PROTOCOL
    elif method == "local_evidence":
        from .render_local_evidence import render_local_evidence as render
        version, protocol = LOCAL_ALGORITHM_VERSION, LOCAL_PROTOCOL
    elif method == "conditioned":
        from .render_conditioned import render_conditioned as render
        version, protocol = CONDITIONED_ALGORITHM_VERSION, CONDITIONED_PROTOCOL
    elif method in {"conditioned_bounded", "conditioned_single"}:
        from .render_conditioned import render_conditioned as render
        version, protocol = ((CONDITIONED_SINGLE_ALGORITHM_VERSION, CONDITIONED_SINGLE_PROTOCOL) if method == "conditioned_single"
                             else (CONDITIONED_BOUNDED_ALGORITHM_VERSION, CONDITIONED_BOUNDED_PROTOCOL))
    else:
        raise ValueError("Unknown P/N reconstruction method.")

    e = np.asarray(energy, float)
    samples = np.asarray(training_samples, float)
    if (samples.ndim != 3 or samples.shape[0] != (1 if method == "conditioned_single" else 2) or samples.shape[1] < 3
            or e.ndim != 1 or samples.shape[2] != len(e)
            or not np.all(np.isfinite(samples) & (samples > 0))):
        raise ValueError("The single-measurement reconstruction protocol needs one positive raw training measurement on the complete gate/energy grid."
                         if method == "conditioned_single" else
                         "The reconstruction protocol needs two positive raw training repeats on the complete gate/energy grid.")

    if method == "conditioned_single":
        frames = np.asarray(pn_frames)
        if frames.ndim != 3 or frames.shape[0] != 2 or frames.shape[1] < 4 or frames.shape[1] % 2:
            raise ValueError("Single-measurement reconstruction requires an even number of at least four P/N frames.")

    def check():
        check_load_cancelled()
        if cancelled is not None and cancelled():
            raise InterruptedError("P/N reconstruction cancelled")

    def update(stage, event):
        check()
        if progress is not None:
            progress(stage, event)
        check()

    check()
    timings = {}
    source_started = perf_counter()
    update("source_preparation", "start")
    alignment_policy = "accepted_only" if method in {"conditioned_bounded", "conditioned_single"} else "legacy_all_fitted"
    def prepare():
        return prepare_sources(e, pn_mean, pn_frames, cancelled=cancelled, progress=update,
                               alignment_policy=alignment_policy)
    source_reused, source_key = False, ""
    if source_cache is None:
        prepared = prepare()
    else:
        if source_cache_identity is None:
            raise ValueError("Source-fit reuse requires current raw-input and implementation identity.")
        prepared, source_reused, source_key = source_cache.prepare(
            e, pn_mean, pn_frames,
            identity={"input_identity": source_cache_identity, "source_alignment_policy": alignment_policy},
            compute=prepare, check=check, force=force_source_refit)
        if source_reused:
            update("source_preparation_cache", "hit")
    timings["source_preparation"] = perf_counter() - source_started
    update("source_preparation", "complete")
    region_started = perf_counter()
    update("regions", "start")
    regions = prepare_regions(e, pn_mean, pn_frames, samples, source_result=prepared,
                              cancelled=cancelled, progress=update, single_measurement=method == "conditioned_single")
    update("regions", "complete")
    timings["regions"] = perf_counter() - region_started
    rendered = {}
    for label, baseline in (("full", regions["full"]), *regions["halves"].items()):
        render_started = perf_counter()
        update(f"render_{label}", "start")
        rendered[label] = render(e, baseline=baseline, cancelled=cancelled, progress=update)
        update(f"render_{label}", "complete")
        timings[f"render_{label}"] = perf_counter() - render_started
    bounded_result = None
    if method == "conditioned_bounded":
        from .render_bounded import render_bounded
        bounded_started = perf_counter()
        bounded_result = render_bounded(e, pn_frames, samples,
            eligible_mask=regions["stages"]["v45"]["arrays"]["v45_witness_eligible"],
            conditioned_results=rendered, cancelled=cancelled, progress=update)
        timings["bounded_texture"] = perf_counter() - bounded_started
        arrays = bounded_result["arrays"]
    else:
        arrays = {k: np.array(v, copy=True) for k, v in rendered["full"]["arrays"].items()
                  if isinstance(v, np.ndarray)}
    arrays["energy"] = prepared["energy"].copy()
    arrays["source_feature_log"] = prepared["source_feature_log"].copy()
    arrays["corrected_source_log"] = prepared["corrected_source_log"].copy()
    arrays["aligned_source_log"] = prepared["aligned_source_log"].copy()
    arrays["selector_deferred_core_mask"] = regions["full"]["deferred_exclusive_mask"].copy()
    if bounded_result is None:
        for label, result in rendered.items():
            if label != "full":
                arrays[f"conditional_{label}_background"] = result["arrays"]["background"].copy()
    if not np.all(np.isfinite(arrays["background"]) & (arrays["background"] > 0)):
        raise ValueError("Reconstruction did not produce a positive finite background on the entire native grid.")
    report = {
        "algorithm_version": version, "settings": dict(protocol),
        "BN_used": False, "historical_background_used": False,
        "historical_fits_used": False, "source_models_refitted_from_raw": not source_reused,
        "sample_used": True, "training_shape": list(samples.shape),
        "scientifically_validated": False,
        "generation_scope": "Raw source reconstruction with the declared regional rule; numerical success is not scientific acceptance for a new measurement.",
        "raw_input_array_sha256": {name: _array_hash(value) for name, value in (
            ("energy", e), ("pn_mean", pn_mean), ("pn_frames", pn_frames), ("training_samples", samples))},
        "source_preparation": {**prepared["diagnostics"],
                               "raw_mean_refitted": not source_reused,
                               "cache_reused": source_reused},
        "source_stages": {name: {"family": item["family"], "diagnostics": item["diagnostics"]}
                          for name, item in prepared["stages"].items()},
        "region_diagnostics": regions.get("diagnostics", {}),
        "render_diagnostics": {label: item["diagnostics"] for label, item in rendered.items()},
        "half_check_limitation": "Same fitted source model and source decisions; these are conditional repeat checks.",
    }
    if bounded_result is not None:
        report.update(bounded_result["diagnostics"])
    if method == "conditioned_single":
        report.update(independent_sample_repeats=1,
            uncertainty_basis="gate_group_dispersion_and_conditional_PN_half_difference",
            cross_measurement_drift="not evaluable from one measurement",
            bounded_texture="not evaluated; independent sample repeats unavailable")
    if protected:
        from .feature_protection import apply_feature_protection
        protection_started = perf_counter()
        update("feature_protection", "start")
        arrays["feature_protection_parent_background"] = arrays["background"].copy()
        background, extra, detail = apply_feature_protection(
            e, arrays["background"], pn_mean, protected, cancelled=cancelled)
        arrays.update({f"feature_protection_{key}": np.array(value, copy=True)
                       for key, value in extra.items()})
        arrays["background"] = background
        arrays["background_log"] = np.log(background)
        halves = {}
        frames = np.asarray(pn_frames)
        split = frames.shape[1] // 2
        for label, raw_half in (("first_half", frames[:, :split]),
                                ("last_half", frames[:, split:])):
            key = f"conditional_{label}_background"
            arrays[f"feature_protection_parent_{key}"] = arrays[key].copy()
            half, _extra, half_detail = apply_feature_protection(
                e, arrays[key], raw_half.mean(axis=1), protected,
                frozen_parameters=detail["frozen_parameters"], cancelled=cancelled)
            arrays[key] = half
            arrays[f"{key}_log"] = np.log(half)
            halves[label] = half_detail
        arrays["construction_type"] = np.array(
            "conditioned_single_with_explicit_raw_source_protection" if method == "conditioned_single"
            else "conditioned_bounded_with_explicit_raw_source_protection")
        report["settings"]["feature_protection"] = protected
        report["feature_protection"] = {"full": detail, "conditional_halves": halves}
        report["generation_scope"] += (
            " Explicit protected ranges preserve the chosen raw source's log curvature with smooth curvature transitions and a full-fit global affine log gauge;"
            " intensity and DRR amplitudes can change across the complete spectrum."
            " source selection is user-declared, not an automatic featureless-source certificate.")
        report.setdefault("final_array_contract", {}).update({
            "background": "Final native background after explicit raw-source protection; use this sampled array, not inherited parent polynomials.",
            "feature_protection_prefix": "Protection geometry, weights and increments; parent arrays describe the pre-protection background.",
            "final_polynomial_available": False,
        })
        if method == "conditioned_bounded":
            report["final_array_contract"]["bounded_prefix"] = (
                "Fine-correction diagnostics before explicit feature protection; not the final applied intensity increment.")
        report["half_check_limitation"] += (
            " Protection halves use the same full-fit geometry and global affine integration constants with their own raw P/N means and parent;"
            " they are conditional checks, not independent validation.")
        for item in report["render_diagnostics"].values():
            item["explicit_feature_protection_applied"] = True
            item["final_background_stage"] = "feature_protection"
        update("feature_protection", "complete")
        timings["feature_protection"] = perf_counter() - protection_started
    timings["reconstruction_total"] = perf_counter() - started
    report["execution"] = {
        "source_preparation": {"reused": source_reused, "cache_key": source_key,
                               "scope": "process" if source_cache is not None else "disabled",
                               "seconds": timings["source_preparation"]},
        "timings_seconds": timings,
    }
    check()
    return arrays, _plain(report)
