"""Adopted V66 bridge recipe, reconstructed exclusively from current arrays.

Each call recomputes the cubic reference, local closure, regional level, and
both wider bridges. This deliberately retains the recipe's geometry guards;
an incompatible selection raises rather than returning a different method.
"""
from __future__ import annotations

import numpy as np

from . import render_cubic, render_excursion, render_shape, render_right, render_left
from .regions_selection import remove_unknown_excursions
from .source_runtime import cancellation_scope, check_cancelled

RECIPE_VERSION = "pn_raw_v66_regional_recipe_v1"


def render_v66(energy, *, baseline, cancelled=None, progress=None):
    """Render one full/conditional-half baseline returned by prepare_regions.

    ``baseline.source_weights`` are this run's original V50 weights. The
    explicit intensity reference stays independent of subsequent selection.
    No full-data coefficients are reused when this function is called on a half.
    """
    e = np.asarray(energy)
    np.testing.assert_array_equal(e, baseline["energy"])
    reference = np.asarray(baseline["baseline_reference_background"])
    np.testing.assert_array_equal(baseline["background"], reference)
    np.testing.assert_array_equal(baseline["reference_source_weights"], baseline["source_weights"])
    stages = {}

    def run(name, function):
        check_cancelled()
        if progress is not None:
            progress(name, "start")
        arrays, diagnostics = function()
        check_cancelled()
        stages[name] = {"arrays": arrays, "diagnostics": diagnostics}
        if progress is not None:
            progress(name, "complete")
        return arrays

    with cancellation_scope(cancelled):
        parent = run("v54", lambda: render_cubic.generate(e, baseline=baseline))
        selection = run("v57_selection", lambda: remove_unknown_excursions(
            e, baseline["source_weights"], baseline["state"]))
        selected = dict(baseline)
        selected.update(source_weights=selection["source_weights"],
                        parent_source_weights=baseline["source_weights"],
                        reference_source_weights=baseline["source_weights"],
                        baseline_reference_background=reference)
        local = run("v63", lambda: render_excursion.generate(e, baseline=selected, parent=parent))
        regional = run("v64", lambda: render_shape.generate(e, baseline=local))
        right = run("v65", lambda: render_right.generate(e, baseline=regional,
                                            region=regional["bridge_endpoint_indices"][1]))
        # Recover the original left bridge from the current pure-P region, as
        # in the adopted V66 runner. No angle-specific index is encoded here.
        w = right["source_weights"][0]
        core_right = int(right["bridge_endpoint_indices"][0])
        if w[core_right] != 1:
            raise ValueError("Right bridge does not begin at the selected P core")
        core_left = core_right
        while core_left > 0 and w[core_left-1] == 1:
            core_left -= 1
        previous_n = np.flatnonzero(w[:core_left] == 0)
        if not len(previous_n):
            raise ValueError("Adopted left bridge needs a preceding pure N region")
        original_left = int(previous_n[-1])
        if not np.all((w[original_left+1:core_left] > 0) & (w[original_left+1:core_left] < 1)):
            raise ValueError("Adopted left bridge requires its contiguous mixed support")
        final = run("v66", lambda: render_left.generate(e, baseline=right,
                                                    region=np.array([original_left, core_left])))
    return {"arrays": final, "stages": stages,
        "diagnostics": {"algorithm_version": RECIPE_VERSION,
            "scientifically_validated": False, "BN_used": False,
            "stored_coefficients_or_background_used": False,
            "half_coefficients_recomputed_by_same_API": True,
            "stage_diagnostics": {name: item["diagnostics"] for name, item in stages.items()},
            "scope": "The adopted V66 regional topology, not a universal clean-source model.",
            "source_weights_semantics": "Curvature selection; final background is not a pointwise source blend.",
            "applicability": "Three selected pure-P regions, the prescribed unknown excursion and supported neighboring source bridges are required; incompatible geometry raises."}}
