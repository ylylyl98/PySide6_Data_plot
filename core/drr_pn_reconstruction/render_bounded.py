"""Assemble a bounded correction from this call's conditioned parents and raw data.

This internal renderer does not load a background, fit or evaluation reference.
The public raw-array pipeline creates its parents once from the current inputs.
Numerical support permits an optional correction; it does not validate the
background scientifically or identify shared P/N structure as purely optical.
"""
from __future__ import annotations

import numpy as np

from core.drr_pn_texture_compatibility import (
    analyze_raw_common_texture, apply_common_fine_delta,
)
from core.drr_pn_texture_support import bound_raw_common_texture
from core.load_cancellation import check_load_cancelled

from .source_runtime import cancellation_scope, check_cancelled


RECIPE_VERSION = "pn_conditioned_bounded_raw_common_fine_v1"
HALF_LABELS = ("first_half", "last_half")


def render_bounded(energy, pn_frames, training_samples, *, eligible_mask,
                   conditioned_results, cancelled=None, progress=None):
    """Use one full fit for full/first/last corrections; never refit either half.

    All inherited conditioned arrays are explicitly ``conditioned_parent_*``.
    In particular their PPoly coefficients, derivatives, weights and masks do
    not describe the final corrected background. Only native final intensity
    and its pointwise log are provided; no final PPoly is asserted. Raw/common
    diagnostics are ``raw_common_texture_*`` and bounded results ``bounded_*``.
    Conditional fold checks use the current v45 mask, not a blind test set.
    """
    e = np.asarray(energy, dtype=float)
    if set(conditioned_results) != {"full", *HALF_LABELS}:
        raise ValueError("Current full, first_half and last_half conditioned renders required")

    def cancellation_requested():
        check_load_cancelled()
        return cancelled is not None and cancelled()

    def update(stage, event):
        check_cancelled()
        if progress is not None:
            progress(stage, event)
        check_cancelled()

    with cancellation_scope(cancellation_requested):
        update("raw_common_texture", "start")
        raw_arrays, raw_report = analyze_raw_common_texture(
            e, pn_frames, training_samples, eligible_mask,
            cancelled=cancellation_requested)
        update("raw_common_texture", "complete")
        update("bounded_texture", "start")
        bounded_arrays, bounded_report = bound_raw_common_texture(
            e, raw_arrays, raw_report, cancelled=cancellation_requested)
        update("bounded_texture", "complete")

        parent = conditioned_results["full"]["arrays"]
        arrays = {f"conditioned_parent_{key}": np.array(value, copy=True)
                  for key, value in parent.items() if isinstance(value, np.ndarray)}
        arrays.update({f"raw_common_texture_{key}": np.array(value, copy=True)
                       for key, value in raw_arrays.items()})
        arrays.update({f"bounded_{key}": np.array(value, copy=True)
                       for key, value in bounded_arrays.items()})
        update("apply_bounded_texture", "start")
        arrays["background"] = apply_common_fine_delta(
            parent["background"], bounded_arrays["applied_delta_log"])
        arrays["background_log"] = np.log(arrays["background"])
        for index, label in enumerate(HALF_LABELS):
            check_cancelled()
            half_parent = conditioned_results[label]["arrays"]["background"]
            arrays[f"conditioned_parent_conditional_{label}_background"] = np.array(half_parent, copy=True)
            half = apply_common_fine_delta(
                half_parent, bounded_arrays["half_applied_delta_log"][index])
            arrays[f"conditional_{label}_background"] = half
            arrays[f"conditional_{label}_background_log"] = np.log(half)
        arrays["energy"] = e.copy()
        arrays["method_name"] = np.array(RECIPE_VERSION)
        arrays["construction_type"] = np.array("conditioned_parent_times_exp_bounded_raw_common_fine")
        arrays["diagnostic_metadata_revision"] = np.array(1)
        update("apply_bounded_texture", "complete")

    array_contract = {
        "background": "Current conditioned parent multiplied by exp(bounded_applied_delta_log); exact parent values on zero delta.",
        "background_log": "Pointwise natural log of final native background intensity.",
        "conditioned_parent_prefix": "Every inherited full conditioned array describes only the parent, including all polynomials, derivatives, source weights and repair masks.",
        "final_polynomial_available": False,
        "raw_common_texture_prefix": "Current raw-data full/half/fold analysis before support weighting.",
        "bounded_prefix": "Frozen transforms, deterministic support, proposed and selected applied deltas.",
        "half_order": list(HALF_LABELS),
    }
    diagnostics = {
        "automatic_application": False,
        "scientifically_validated": False,
        "generation_scope": "Current raw-array conditioned reconstruction with optional bounded raw-common fine correction. Conditional numerical support is not local or global scientific acceptance.",
        "bounded_selection": bounded_report["selection"],
        "region_applicability": bounded_report["region_applicability"],
        "conditioned_parent_diagnostics": {
            label: result["diagnostics"] for label, result in conditioned_results.items()},
        "raw_common_texture_diagnostics": raw_report,
        "bounded_texture_diagnostics": bounded_report,
        "final_array_contract": array_contract,
        "render_diagnostics": {
            label: {"recipe_version": RECIPE_VERSION, "scientifically_validated": False,
                    "automatic_application": False,
                    "parent_diagnostics_path": f"conditioned_parent_diagnostics.{label}",
                    "correction_selected": bool(bounded_report["selection"]["proposal_eligible"]),
                    "delta_array": "bounded_applied_delta_log" if label == "full" else "bounded_half_applied_delta_log",
                    "half_index": None if label == "full" else HALF_LABELS.index(label),
                    "final_polynomial_available": False}
            for label in ("full", *HALF_LABELS)},
        "half_check_limitation": "Same current full source model, source decisions, texture shift/gain and support weight; each half uses its own raw common fine and conditioned parent. These are conditional checks, not independent validation.",
    }
    return {"arrays": arrays, "diagnostics": diagnostics}
