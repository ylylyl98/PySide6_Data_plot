"""Reconstruct the adopted PN source-preparation chain from measured arrays.

This module receives no stored features, classifier masks, angle labels, BN or
sample intensities. The version-named private kernels preserve the adopted
experimental mathematics; their historical names are lineage, not data inputs.
An arbitrary acquisition is not thereby scientifically validated.
"""
from __future__ import annotations

import numpy as np

from . import source_v12, source_v20, source_v21, source_v24
from . import source_v29, source_v31, source_v32
from .source_runtime import cancellation_scope, check_cancelled

ALGORITHM_VERSION = "pn_raw_source_reconstruction_v2"
LEGACY_ALGORITHM_VERSION = "pn_raw_source_reconstruction_v1"
LEGACY_STAGE_FAMILIES = {
    "v12": "native_residual_weighted",
    "v20": "residual_pass_fixed_weights",
    "v21": "source_hypothesis_fixed_weights",
    "v24": "narrow_pass_post_cost_weights",
    "v29": "all_applied_curvature_weights",
    "v31": "common_residual_resonance",
    "v32": "full_source_residual_cleanup",
}
STAGE_FAMILIES = {**LEGACY_STAGE_FAMILIES, "v12": "accepted_residual_weighted"}


def _validate(energy, pn_mean, pn_frames):
    # Own copies retain the supplied layout/order used by the native fits.
    e = np.array(energy, dtype=float, copy=True, order="K")
    pn = np.array(pn_mean, dtype=float, copy=True, order="K")
    frames = np.array(pn_frames, dtype=float, copy=True, order="K")
    if e.ndim != 1 or len(e) < 20 or not np.all(np.isfinite(e)) or not np.all(np.diff(e) > 0):
        raise ValueError("Expected at least 20 finite increasing native energies")
    if pn.shape != (2, len(e)) or not np.all(np.isfinite(pn) & (pn > 0)):
        raise ValueError("P/N mean must have positive finite shape (2, energy)")
    if (frames.ndim != 3 or frames.shape[0] != 2 or frames.shape[2] != len(e)
            or frames.shape[1] < 4 or not np.all(np.isfinite(frames) & (frames > 0))):
        raise ValueError("P/N frames must have positive shape (2, frames>=4, energy)")
    if not np.allclose(frames.mean(axis=1), pn, rtol=1e-13, atol=0):
        raise ValueError("P/N mean must equal the intensity mean of the supplied frames")
    return e, pn, frames


def prepare_sources(energy, pn_mean, pn_frames, *, cancelled=None, progress=None,
                    alignment_policy="accepted_only"):
    """Fit sources from raw arrays and return adopted stages and final C/F/q/A.

    ``pn_frames`` is (P/N, repeats, energy). First/last halves are conditional
    reapplications of this run's full-data F/q, as in the original procedure;
    V31 itself uses both halves for its conditional fit stability check.
    ``cancelled`` is an optional zero-argument boolean callback. ``progress``
    receives the stage name and ``"start"``/``"complete"``. Neither callback is
    required, and this function performs no file reads or writes.
    """
    if alignment_policy not in ("accepted_only", "legacy_all_fitted"):
        raise ValueError("Unknown P/N source alignment policy")
    families = STAGE_FAMILIES if alignment_policy == "accepted_only" else LEGACY_STAGE_FAMILIES
    e, pn, frames = _validate(energy, pn_mean, pn_frames)
    stages = {}

    def run(name, function):
        check_cancelled()
        if progress is not None:
            progress(name, "start")
        result = function()
        arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
        if not np.all(np.isfinite(arrays["background"]) & (arrays["background"] > 0)):
            raise ValueError(f"Nonpositive or nonfinite source-stage background: {name}")
        stages[name] = {"arrays": arrays, "diagnostics": result["diagnostics"],
                        "family": families[name]}
        if progress is not None:
            progress(name, "complete")
        return stages[name]

    with cancellation_scope(cancelled):
        v12 = run("v12", lambda: source_v12.generate(e, pn, family=families["v12"]))
        v20 = run("v20", lambda: source_v20.generate(e, pn, baseline=v12["arrays"],
            baseline_diagnostics=v12["diagnostics"], family=families["v20"]))
        v21 = run("v21", lambda: source_v21.generate(e, pn, baseline_v12=v12["arrays"],
            baseline_diagnostics=v12["diagnostics"], v20_result=v20["arrays"],
            v20_diagnostics=v20["diagnostics"], family=families["v21"]))
        v24 = run("v24", lambda: source_v24.generate(e, pn, baseline=v21["arrays"],
            baseline_diagnostics=v21["diagnostics"], family=families["v24"]))
        v29 = run("v29", lambda: source_v29.generate(e, pn, baseline=v24["arrays"],
            baseline_diagnostics=v24["diagnostics"], family=families["v29"]))
        v31 = run("v31", lambda: source_v31.generate(e, pn, pn_frames=frames,
            baseline=v29["arrays"], baseline_diagnostics=v29["diagnostics"],
            family=families["v31"]))
        v32 = run("v32", lambda: source_v32.generate(e, pn, baseline=v29["arrays"],
            baseline_diagnostics=v29["diagnostics"], v31_diagnostics=v31["diagnostics"],
            family=families["v32"]))
        check_cancelled()

    final = v32["arrays"]
    c = final["v32_effective_corrected_source_log"].copy()
    q = final["v32_alignment_q_log"].copy()
    # Historical V39 half replay recovered F this way. Use the same arithmetic
    # rather than substituting the algebraically equal sum of feature arrays.
    f = np.log(pn) - c
    split = frames.shape[1] // 2
    half_pn = np.array([frames[:, :split].mean(1), frames[:, split:].mean(1)])
    half_c = np.log(half_pn) - f[None, :, :]
    signs = np.array([-.5, .5])[:, None]
    aligned = c + signs*q
    half_a = half_c + signs[None, :, :]*q
    np.testing.assert_allclose(c[0]-c[1], final["v32_effective_residual_log"]+q,
                               rtol=0, atol=1e-14)
    return {
        "energy": e, "pn_mean": pn, "pn_frames": frames, "stages": stages,
        "source_feature_log": f, "source_feature_sum_log": final["v32_effective_source_feature_log"].copy(),
        "corrected_source_log": c, "alignment_q_log": q, "aligned_source_log": aligned,
        "differential_nuisance_log": final["differential_nuisance_log"].copy(),
        "residual_log": final["v32_effective_residual_log"].copy(),
        "source_weights": final["source_weights"].copy(),
        "half_pn_mean": half_pn, "half_corrected_source_log": half_c,
        "half_aligned_source_log": half_a,
        "diagnostics": {
            "algorithm_version": ALGORITHM_VERSION if alignment_policy == "accepted_only" else LEGACY_ALGORITHM_VERSION,
            "stage_families": dict(families), "source_alignment_policy": alignment_policy,
            "raw_mean_refitted": True, "stored_model_used_as_input": False,
            "BN_used": False, "sample_used": False, "scientifically_validated": False,
            "frame_count": int(frames.shape[1]), "half_frame_counts": [split, frames.shape[1]-split],
            "half_semantics": "mean intensity before log; this run's full F/q fixed; conditional replay, not independent source refits",
            "source_decomposition_limit": "PN-only feature and envelope attribution remains conditional; no arbitrary-data acceptance claim",
        },
    }
