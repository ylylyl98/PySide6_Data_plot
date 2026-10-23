"""Rebuild sample-supported region selection from this run's source fit.

No saved model, mask, background, angle label, or BN is read here. Historical
stage names document the adopted recipe. Two training repeats are required;
the historical 101-gate case is not an arbitrary-data quality guarantee.
"""
from __future__ import annotations

import numpy as np

from .regions_classifier import classify
from .regions_selection import source_selection, resolve_regions, remove_unknown_excursions
from .source_runtime import cancellation_scope, check_cancelled


def prepare_regions(energy, pn_mean, pn_frames, train_samples, *, source_result,
                    cancelled=None, progress=None):
    """Return ``full``, ``halves`` baselines and traceable selection ``stages``.

    ``train_samples`` is exactly two training repeats, shape (2, gates, energy).
    At least three gate rows are required; the IO layer validates coordinates.
    Full F/q from ``source_result`` is applied to first/last PN halves. Source
    classification and weights are estimated once from the full training data;
    half baselines recompute C/A and their own pointwise reference, not state.
    ``progress`` receives (stage_name, "start"/"complete").
    """
    e = np.asarray(energy, float)
    pn = np.asarray(pn_mean, float)
    frames = np.asarray(pn_frames, float)
    n = len(e)
    if e.ndim != 1 or n < 21 or not np.all(np.isfinite(e)) or np.any(np.diff(e) <= 0):
        raise ValueError("Complete increasing native energy with at least 21 points required")
    if pn.shape != (2, n) or not np.all(np.isfinite(pn) & (pn > 0)):
        raise ValueError("Positive P/N means with shape (2, energy) required")
    if frames.ndim != 3 or frames.shape[0] != 2 or frames.shape[2] != n or frames.shape[1] < 4:
        raise ValueError("P/N frames must have shape (2, at least 4 frames, energy)")
    if not np.all(np.isfinite(frames) & (frames > 0)):
        raise ValueError("P/N frames must be positive and finite")
    for key, value in (("energy", e), ("pn_mean", pn), ("pn_frames", frames)):
        np.testing.assert_array_equal(source_result[key], value)
    c = np.asarray(source_result["corrected_source_log"])
    q = np.asarray(source_result["alignment_q_log"])
    a = np.asarray(source_result["aligned_source_log"])
    half_c = np.asarray(source_result["half_corrected_source_log"])
    half_a = np.asarray(source_result["half_aligned_source_log"])
    half_pn = np.asarray(source_result["half_pn_mean"])
    signs = np.array([-.5, .5])[:, None]
    np.testing.assert_array_equal(a, c + signs*q)
    if half_c.shape != (2, 2, n) or half_a.shape != half_c.shape:
        raise ValueError("Two conditional P/N source halves required")
    np.testing.assert_array_equal(half_a, half_c + signs[None]*q)
    source = source_result["stages"]["v32"]["arrays"]
    risk = np.asarray(source_result["stages"]["v29"]["arrays"]["v29_curvature_risk"])
    if risk.shape != (2, n) or not np.all(np.isfinite(risk) & (risk >= 0)):
        raise ValueError("Current fitted source risk must be finite and nonnegative")
    stages = {}

    def start(name):
        check_cancelled()
        if progress is not None:
            progress(name, "start")

    def save(name, arrays, diagnostics):
        check_cancelled()
        stages[name] = {"arrays": arrays, "diagnostics": diagnostics}
        if progress is not None:
            progress(name, "complete")
        return arrays

    with cancellation_scope(cancelled):
        start("v35")
        selector = source_selection(e, risk)
        selector["energy"] = e.copy()
        save("v35", selector, {"control_only": True,
             "limitation": "Model curvature risk is not measured source cleanliness."})
        start("v36")
        w = selector["source_weights"]
        residual = source["v32_effective_residual_log"]
        # Preserve the original control arithmetic, including its rounding.
        control_b = np.exp(c.mean(0) + (w[0]-.5)*residual)
        control = dict(energy=e.copy(), background=control_b, source_weights=w.copy())
        save("v36", control, {"family": "piecewise_old_sources",
             "V34_source_update_used": False, "source_stage": "current v32"})
        start("v45")
        classifier, classifier_diag = classify(e, pn, train_samples, baseline=source,
            control=control, half_aligned_sources=np.concatenate((a[None], half_a), axis=0))
        save("v45", classifier, classifier_diag)
        start("v50")
        original = dict(energy=e.copy(), original_corrected_source_log=c.copy(),
            original_aligned_source_log=a.copy(), alignment_q_log=q.copy(),
            state=classifier["v45_state"].copy(),
            source_usable_masks=classifier["source_usable_masks"].copy(),
            raw_geometric_mean=np.sqrt(pn[0]*pn[1]))
        full, region_diag = resolve_regions(e, baseline=original, control=control,
                                            classifier=classifier)
        full["baseline_reference_background"] = full["background"].copy()
        full["reference_source_weights"] = full["source_weights"].copy()
        save("v50", full, region_diag)
        start("v57_selection")
        effective, decisions = remove_unknown_excursions(e, full["source_weights"], full["state"])
        save("v57_selection", effective, decisions)
        halves = {}
        for i, label in enumerate(("first_half", "last_half")):
            check_cancelled()
            half = {key: np.array(value, copy=True) for key, value in full.items()}
            half["original_corrected_source_log"] = half_c[i].copy()
            half["original_aligned_source_log"] = half_c[i] + signs*q
            half["source_difference_log"] = half["original_aligned_source_log"][0] - half["original_aligned_source_log"][1]
            half["background"] = np.exp(np.sum(full["source_weights"]*half["original_aligned_source_log"], axis=0))
            half["baseline_reference_background"] = half["background"].copy()
            half["baseline_background"] = half["background"].copy()
            half["raw_geometric_mean"] = np.sqrt(half_pn[i, 0]*half_pn[i, 1])
            half["repair_log_delta"] = np.log(half["background"]/half["raw_geometric_mean"])
            halves[label] = half
    return {"energy": e.copy(), "full": full, "halves": halves, "stages": stages,
        "diagnostics": {"scientifically_validated": False, "BN_used": False,
            "sample_training_used": True, "training_shape": list(np.shape(train_samples)),
            "old_saved_model_or_background_used": False,
            "classification": classifier_diag,
            "control_switches_eV": selector["v35_switch_energy"].tolist(),
            "control_switch_halfwidths_eV": selector["v35_switch_halfwidth"].tolist(),
            "resolved_regions": region_diag,
            "excursion_selection": decisions,
            "half_semantics": "Current full F/q/state/weights held fixed; each half has its own C/A and intensity reference.",
            "applicability": "Two complete training repeats with at least three gate rows are required; rendering also verifies its prescribed regional geometry."}}
