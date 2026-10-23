"""Bound an already-frozen raw-common proposal to conditional continuous support.

No parameters are refitted. Support describes only this optional correction,
never scientific validity of the parent background or absence of sample peaks.
The retained cubic splines have global coefficients, so the local guards below
are not a proof of compact causal isolation; complete products still need review.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from .drr_pn_reconstruction.regions_helpers import local_quadratic, runs
from .drr_pn_reconstruction.source_runtime import cancellation_scope, check_cancelled


BOUNDED_TEXTURE_PROTOCOL = {
    "method": "frozen_raw_common_contiguous_support_v1",
    "refit": False,
    "Q4": {"full_width_ev": 0.004, "minimum_native_points": 7},
    "registration_support": "Q4 stencils at i-1, i, i+1 must lie within the same original eligible run; the complete allowed shift range is guarded.",
    "SG": {"window": 21, "polyorder": 3, "uniform_grid_points": "native energy count",
           "uniform_grid_bounds": "complete native energy endpoints"},
    "endpoint_guard_width": "20*(energy[-1]-energy[0])/(n_energy-1), the entire SG21 endpoint fit span",
    "taper_width": "same SG21 full span as endpoint guard, per side in measured energy",
    "taper": "s(t)=t^3*(10-15*t+6*t^2), t=clip(min(left_distance,right_distance)/width,0,1)",
    "minimum_contiguous_span": "two taper widths after Q4/shift and endpoint guards",
    "holes": "never filled; original contiguous eligible runs only",
    "CV": {"energy_blocks": 6, "guard_ev": 0.035, "minimum_train_points": 150,
           "minimum_test_points": 25, "minimum_covered_test_points": 25,
           "domains": ["full", "eligible"], "nonworsening_absolute_tolerance": 1e-12,
           "prediction": "raw_common + support_weight*(frozen_fold_prediction-raw_common)",
           "conditional_not_blind": True},
    "selection": "Original eligible-domain six-fold rule is necessary; additionally every bounded full/eligible fold must be supported and nonworsening. Rejection applies exact zero delta everywhere.",
    "source_of_delta": "Only stored raw-common full/half proposals, multiplied by the same deterministic support weight.",
    "limitations": [
        "Support refers only to optional raw-common fine correction, not reliability of the parent background.",
        "Retained CubicSpline registration and derivative regridding have global coefficients; local stencils do not establish mathematically compact dependence.",
        "Guarded corrections can still affect derivatives through spline tails and transitions; inspect complete unchanged dR/R and SG11/21 products.",
        "P/N-common and gate-common structure need not be purely optical; these are retrospective conditional checks, not scientific acceptance.",
    ],
}


def _validate(energy, arrays, report):
    e = np.asarray(energy, dtype=float)
    if (e.ndim != 1 or len(e) < 21 or not np.isfinite(e).all()
            or not np.all(np.diff(e) > 0)):
        raise ValueError("Finite ascending native energy with at least 21 points required")
    if not np.array_equal(e, arrays["energy"]):
        raise ValueError("Frozen analysis energy mismatch")
    n = len(e)
    shapes = {
        "raw_common_fine": (n,), "target_train_fine": (n,), "parameters": (2,),
        "proposed_delta_log": (n,), "half_raw_common_fine": (2, n),
        "half_proposed_delta_log": (2, n), "fold_parameters": (6, 2),
        "raw_source_fine": (2, n), "half_raw_source_fine": (2, 2, n),
        "gate_group_target_fine": (3, n), "repeat_target_fine": (2, n),
    }
    values = {}
    for name, shape in shapes.items():
        value = np.asarray(arrays[name], dtype=float)
        if value.shape != shape or (name != "fold_parameters" and not np.isfinite(value).all()):
            raise ValueError(f"Invalid frozen {name}")
        values[name] = value
    eligible = np.asarray(arrays["training_mask"])
    if eligible.dtype.kind != "b" or eligible.shape != (n,):
        raise ValueError("Original boolean training_mask required")
    values["training_mask"] = eligible
    p = values["parameters"]
    if not (-1 <= p[0] <= 1 and .7 <= p[1] <= 1.3):
        raise ValueError("Frozen full parameters exceed original bounds")
    if len(report["folds"]) != 6:
        raise ValueError("Six original fold reports required")
    for i, fold in enumerate(report["folds"]):
        if fold["block"] != i:
            raise ValueError("Original fold order mismatch")
        fp = values["fold_parameters"][i]
        if fold["parameters"] is not None:
            if (not np.array_equal(fp, fold["parameters"]) or not np.isfinite(fp).all()
                    or not (-1 <= fp[0] <= 1 and .7 <= fp[1] <= 1.3)):
                raise ValueError("Frozen fold parameter mismatch")
        elif np.isfinite(fp).any():
            raise ValueError("Unsupported fold parameters must remain unavailable")
    if report["optimizer"]["parameters"] is not None:
        if not np.array_equal(p, report["optimizer"]["parameters"]):
            raise ValueError("Frozen optimizer parameter mismatch")
    return e, values


def _errors(before, after, target, mask):
    count = int(mask.sum())
    if not count:
        return dict(count=0, before=None, after=None, relative_RMS_change=None)
    old = float(np.sqrt(np.mean((before[mask]-target[mask])**2)))
    new = float(np.sqrt(np.mean((after[mask]-target[mask])**2)))
    return dict(count=count, before=old, after=new,
                relative_RMS_change=new/old-1 if old else None)


def _support(e, eligible):
    n = len(e)
    width = 20*(e[-1]-e[0])/(n-1)
    left = np.searchsorted(e, e-.002)
    right = np.searchsorted(e, e+.002, side="right")
    narrow = right-left < 7
    left[narrow] = np.clip(np.arange(n)[narrow]-3, 0, n-7)
    right[narrow] = left[narrow]+7
    index = np.arange(n)
    stencil_left = np.minimum.reduce([left[np.clip(index+j, 0, n-1)] for j in (-1, 0, 1)])
    stencil_right = np.maximum.reduce([right[np.clip(index+j, 0, n-1)] for j in (-1, 0, 1)])
    no_extrapolation = (index >= 1) & (index < n-1)
    endpoint_guard = (e < e[0]+width) | (e > e[-1]-width)
    weight = np.zeros(n)
    status = np.full(n, "unsupported", dtype="U40")
    status[eligible] = "boundary_guard"
    run_reports = []
    for run_index, (start, stop) in enumerate(runs(eligible)):
        check_cancelled()
        local = ((index >= start) & (index < stop) & (stencil_left >= start)
                 & (stencil_right <= stop) & no_extrapolation & ~endpoint_guard)
        candidates = np.flatnonzero(local)
        item = dict(run=run_index, original_indices=[int(start), int(stop-1)],
                    original_energy_bounds=[float(e[start]), float(e[stop-1])],
                    original_count=int(stop-start), taper_width_ev=float(width))
        if candidates.size:
            first, last = int(candidates[0]), int(candidates[-1])
            span = float(e[last]-e[first])
            item.update(guarded_indices=[first, last],
                        guarded_energy_bounds=[float(e[first]), float(e[last])],
                        guarded_span_ev=span)
        else:
            span = 0.
            item.update(guarded_indices=None, guarded_energy_bounds=None, guarded_span_ev=0.)
        if not candidates.size or span < 2*width:
            status[start:stop] = "insufficient_contiguous_support"
            item.update(status="insufficient_contiguous_support", covered_count=0,
                        reason="After operator guards this run cannot contain two full SG21 taper widths.")
        else:
            t = np.clip(np.minimum(e[candidates]-e[first], e[last]-e[candidates])/width, 0, 1)
            weight[candidates] = t**3*(10+t*(-15+6*t))
            status[candidates[weight[candidates] > 0]] = "supported_candidate"
            item.update(status="supported_candidate", covered_count=int(np.count_nonzero(weight[candidates])),
                        left_taper_bounds=[float(e[first]), float(e[first]+width)],
                        right_taper_bounds=[float(e[last]-width), float(e[last])],
                        plateau_energy_bounds=[float(e[first]+width), float(e[last]-width)],
                        reason="Same-run Q4/shift support with two SG21-width C2 transitions.")
        run_reports.append(item)
    status[endpoint_guard] = "boundary_guard"
    return weight, status, run_reports, {
        "uniform_grid_step_ev": float(width/20), "SG21_complete_window_span_ev": float(width),
        "endpoint_guard_bounds": [[float(e[0]), float(e[0]+width)], [float(e[-1]-width), float(e[-1])]],
        "endpoint_guard_indices": np.flatnonzero(endpoint_guard).tolist(),
        "Q4_shift_footprint_left": stencil_left, "Q4_shift_footprint_stop": stencil_right,
    }


def bound_raw_common_texture(energy, analysis_arrays, analysis_report, *, cancelled=None):
    """Bound frozen full/half/fold transforms; return arrays plus JSON-safe report.

    ``analysis_report`` is the report returned by analyze_raw_common_texture,
    not the enclosing experiment JSON. All original full/fold parameters remain
    unchanged. No data is read or refitted. Regions never bridge eligibility
    holes. A rejected selection returns exact zero applied deltas, retaining
    weighted proposals for diagnosis. Cancellation raises InterruptedError.
    """
    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be callable or None")
    with cancellation_scope(cancelled):
        e, values = _validate(energy, analysis_arrays, analysis_report)
        return _bound(e, values, analysis_report)


def _bound(e, a, original_report):
    common, target, eligible = a["raw_common_fine"], a["target_train_fine"], a["training_mask"]
    x = np.arange(len(e), dtype=float)
    spline = CubicSpline(x, common, extrapolate=True)
    def prediction(p):
        check_cancelled()
        return p[1]*spline(x+p[0])
    np.testing.assert_allclose(a["proposed_delta_log"], prediction(a["parameters"])-common, rtol=0, atol=1e-14)
    halves = a["half_raw_common_fine"]
    half_prediction = a["parameters"][1]*CubicSpline(x, halves, axis=-1)(x+a["parameters"][0])
    np.testing.assert_allclose(a["half_proposed_delta_log"], half_prediction-halves, rtol=0, atol=1e-14)
    weight, statuses, run_reports, guard = _support(e, eligible)
    proposed = weight*a["proposed_delta_log"]
    half_proposed = weight[None, :]*a["half_proposed_delta_log"]
    covered = weight > 0
    block_edges = np.linspace(e[0], e[-1], 7)
    original_ok = bool(original_report["optimizer"]["success"] and eligible.sum() >= 150)
    reasons = [] if original_ok else ["original_full_optimizer_or_training_support_failed"]
    folds = []
    for i, (left, right) in enumerate(zip(block_edges[:-1], block_edges[1:])):
        check_cancelled()
        test = (e >= left) & ((e <= right) if i == 5 else (e < right))
        train = eligible & ((e < left-.035) | (e > right+.035))
        old = original_report["folds"][i]
        valid_parameters = bool(np.isfinite(a["fold_parameters"][i]).all())
        frozen = prediction(a["fold_parameters"][i]) if valid_parameters else common.copy()
        bounded = common+weight*(frozen-common)
        original_metrics = _errors(common, frozen, target, test & eligible)
        old_supported = bool(train.sum() >= 150 and (test & eligible).sum() >= 25
                             and old["optimizer_success"] and valid_parameters)
        old_pass = bool(old_supported and original_metrics["after"] <= original_metrics["before"]+1e-12)
        original_ok = original_ok and old_pass
        if not old_pass:
            reasons.append(f"original_fold_{i}_failed_or_unsupported")
        domains = {}
        for name, mask in (("full", test), ("eligible", test & eligible)):
            count, cover_count = int(mask.sum()), int((mask & covered).sum())
            metrics = _errors(common, bounded, target, mask)
            supported = bool(train.sum() >= 150 and count >= 25 and cover_count >= 25
                             and old["optimizer_success"] and valid_parameters)
            nonworsening = bool(supported and metrics["after"] <= metrics["before"]+1e-12)
            domains[name] = dict(**metrics, supported=supported, covered_count=cover_count,
                coverage_fraction=cover_count/count if count else 0.,
                status="evaluated" if supported else "insufficient_support_or_coverage",
                nonworsening=nonworsening,
                has_benefit=bool(supported and metrics["after"] < metrics["before"]))
            if not nonworsening:
                reasons.append(f"bounded_fold_{i}_{name}_failed_or_unsupported")
        folds.append(dict(block=i, energy_bounds=[float(left), float(right)],
            parameters=a["fold_parameters"][i].tolist() if valid_parameters else None,
            optimizer_success=bool(old["optimizer_success"]), train_count=int(train.sum()),
            original_eligible_metrics=original_metrics, original_rule_passed=old_pass,
            supported=all(v["supported"] for v in domains.values()), domains=domains))
    all_bounded_pass = all(d["nonworsening"] for f in folds for d in f["domains"].values())
    selected = bool(original_ok and all_bounded_pass and np.isfinite(proposed).all()
                    and np.isfinite(half_proposed).all())
    applied = proposed.copy() if selected else np.zeros_like(proposed)
    half_applied = half_proposed.copy() if selected else np.zeros_like(half_proposed)
    if not selected:
        statuses[statuses == "supported_candidate"] = "proposal_rejected"
    bin_edges = np.r_[e[0], (e[:-1]+e[1:])/2, e[-1]]
    boundaries = np.r_[0, np.flatnonzero(statuses[1:] != statuses[:-1])+1, len(e)]
    explanations = {
        "unsupported": "Outside the original eligibility mask; no optional correction is applied.",
        "boundary_guard": "Protected by Q4/allowed-shift support, SG21 endpoint guard, or zero taper boundary.",
        "insufficient_contiguous_support": "The original eligible run is too short after operator guards for two complete taper widths.",
        "supported_candidate": "The bounded retrospective conditional proposal passes the fixed numerical selection rule; scientific acceptance remains false.",
        "proposal_rejected": "A required original or bounded fold rule failed; proposed corrections are retained for diagnosis and applied delta is zero.",
    }
    regions = [dict(energy_bounds=[float(bin_edges[start]), float(bin_edges[stop])],
                    native_indices=[int(start), int(stop-1)], source="P/N common fine",
                    status=str(statuses[start]), reason=explanations[str(statuses[start])],
                    correction_applied=bool(np.any(applied[start:stop] != 0)))
               for start, stop in zip(boundaries[:-1], boundaries[1:])]
    q4 = local_quadratic(e, proposed, .004)
    medium = q4-local_quadratic(e, proposed, .035)

    def frozen_bounded(source):
        check_cancelled()
        shifted = a["parameters"][1]*CubicSpline(
            x, source, axis=-1, extrapolate=True,
        )(x+a["parameters"][0])
        return source+weight*(shifted-source)

    def check_domains(before, after, reference):
        result = {domain:_errors(before, after, reference, mask) for domain,mask in (
            ("full", np.ones(len(e), bool)), ("eligible", eligible), ("covered", covered))}
        result["blocks"] = []
        for i, (left, right) in enumerate(zip(block_edges[:-1], block_edges[1:])):
            test = (e >= left) & ((e <= right) if i == 5 else (e < right))
            result["blocks"].append(dict(block=i, energy_bounds=[float(left), float(right)],
                **{domain:_errors(before, after, reference, mask) for domain,mask in (
                    ("full", test), ("eligible", test & eligible), ("covered", test & covered))}))
        return result

    bounded_sources = frozen_bounded(a["raw_source_fine"])
    bounded_half_sources = frozen_bounded(a["half_raw_source_fine"])
    same_checks = {
        "parameters": a["parameters"].tolist(), "refitted": False,
        "uses_proposed_not_applied": True,
        "meaning": "Diagnostic bounded proposals with the same full parameters and support weight; global rejection still applies exact identity to the actual background.",
        "common": check_domains(common, common+proposed, target),
        "sources": {label:check_domains(a["raw_source_fine"][i], bounded_sources[i], target)
                    for i,label in enumerate(("P", "N"))},
        "pn_halves": [{"half": h,
            "common": check_domains(halves[h], halves[h]+half_proposed[h], target),
            "sources": {label:check_domains(a["half_raw_source_fine"][h,i], bounded_half_sources[h,i], target)
                        for i,label in enumerate(("P", "N"))}} for h in range(2)],
        "gate_groups": [{"group": i, "domains": check_domains(common, common+proposed, reference)}
                        for i,reference in enumerate(a["gate_group_target_fine"])],
        "training_repeats": [{"repeat": i, "domains": check_domains(common, common+proposed, reference)}
                             for i,reference in enumerate(a["repeat_target_fine"])],
    }
    def rms(value):
        return float(np.sqrt(np.mean(value**2)))
    delta_rms = rms(proposed)
    coverage = float(covered.mean())
    report = {
        "method": BOUNDED_TEXTURE_PROTOCOL["method"], "scientifically_validated": False,
        "automatic_application": False, "refitted": False,
        "parameters": a["parameters"].tolist(), "folds": folds,
        "same_transform_checks": same_checks,
        "selection": {"proposal_eligible": selected, "original_proposal_eligible": bool(original_ok),
            "bounded_all_domains_nonworsening": bool(all_bounded_pass), "reasons": reasons,
            "at_least_one_fold_strictly_improves": bool(selected and any(
                d["has_benefit"] for f in folds for d in f["domains"].values())),
            "scientific_acceptance": False},
        "coverage": {"fraction": coverage, "covered_count": int(covered.sum()),
            "eligible_count": int(eligible.sum()),
            "eligible_fraction": float((covered & eligible).sum()/eligible.sum()) if eligible.any() else 0.,
            "applied_count": int(np.count_nonzero(applied))},
        "operator_guard": {key:value for key,value in guard.items() if not key.startswith("Q4_shift")},
        "contiguous_runs": run_reports,
        "training_errors": {domain:_errors(common, common+proposed, target, mask)
            for domain,mask in (("full", np.ones(len(e), bool)), ("eligible", eligible))},
        "leakage": {"proposed_delta_RMS": delta_rms, "Q4_delta_RMS": rms(q4),
            "medium_delta_RMS": rms(medium), "Q4_delta_max_abs": float(abs(q4).max()),
            "medium_delta_max_abs": float(abs(medium).max()),
            "Q4_delta_RMS_over_delta_RMS": rms(q4)/delta_rms if delta_rms else None,
            "medium_delta_RMS_over_delta_RMS": rms(medium)/delta_rms if delta_rms else None,
            "applied_delta_RMS": rms(applied)},
        "region_applicability": {"schema_version": 1, "scientifically_validated": False,
            "automatic_application": False, "scope": "optional_raw_common_fine_correction",
            "regions": regions, "coverage_fraction": coverage,
            "notes": list(BOUNDED_TEXTURE_PROTOCOL["limitations"])},
        "limitations": list(BOUNDED_TEXTURE_PROTOCOL["limitations"]),
    }
    arrays = {
        "energy": e.copy(), "support_weight": weight, "training_mask": eligible.copy(),
        "proposed_delta_log": proposed, "applied_delta_log": applied,
        "half_proposed_delta_log": half_proposed, "half_applied_delta_log": half_applied,
        "parameters": a["parameters"].copy(), "fold_parameters": a["fold_parameters"].copy(),
        "block_edges": block_edges, "Q4_delta_leakage": q4, "medium_delta_leakage": medium,
        "Q4_shift_footprint_left": guard["Q4_shift_footprint_left"],
        "Q4_shift_footprint_stop": guard["Q4_shift_footprint_stop"],
    }
    check_cancelled()
    return arrays, report
