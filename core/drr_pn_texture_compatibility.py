"""Experimental raw P/N common-texture diagnostics; no automatic application.

The sample-derived eligibility mask makes energy-block CV conditional, not blind
validation. A structure shared by P/N and by most gates can still be a sample
feature; it is not proof of a purely optical background. Numerical support or
smaller residuals must never be interpreted as scientific acceptance.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from .drr_pn_reconstruction.regions_helpers import local_quadratic
from .drr_pn_reconstruction.source_runtime import (
    cancellation_scope,
    check_cancelled,
    least_squares,
)


_MIN_TRAIN = 150
_MIN_TEST = 25
_GUARD_EV = 0.035


def _validate(energy, pn_frames, training_samples, eligible_mask):
    e = np.asarray(energy, dtype=float)
    pn = np.asarray(pn_frames, dtype=float)
    samples = np.asarray(training_samples, dtype=float)
    eligible = np.asarray(eligible_mask)
    if (e.ndim != 1 or e.size < 21 or not np.all(np.isfinite(e))
            or not np.all(np.diff(e) > 0)):
        raise ValueError("Finite ascending native energy with at least 21 points required")
    if (pn.ndim != 3 or pn.shape[0] != 2 or pn.shape[1] < 4
            or pn.shape[1] % 2 or pn.shape[2] != e.size):
        raise ValueError("pn_frames must have shape (2, even frames >= 4, energy)")
    if (samples.ndim != 3 or samples.shape[0] != 2
            or samples.shape[1] < 3 or samples.shape[2] != e.size):
        raise ValueError("training_samples must have shape (2, gates >= 3, energy)")
    if not np.all(np.isfinite(pn) & (pn > 0)):
        raise ValueError("Positive finite P/N intensities required")
    if not np.all(np.isfinite(samples) & (samples > 0)):
        raise ValueError("Positive finite training intensities required")
    if eligible.dtype.kind != "b" or eligible.shape != e.shape:
        raise ValueError("An explicit boolean eligible_mask with energy shape is required")
    return e, pn, samples, eligible


def _rms(values):
    if not values.size:
        return None
    value = float(np.sqrt(np.mean(np.square(values))))
    return value if np.isfinite(value) else None


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    value = numerator / denominator
    return float(value) if np.isfinite(value) else None


def _errors(before, after, target, mask):
    count = int(mask.sum())
    old = _rms((before - target)[mask])
    new = _rms((after - target)[mask])
    ratio = _ratio(new, old)
    return {
        "count": count, "before": old, "after": new,
        "relative_RMS_change": None if ratio is None else ratio - 1.0,
    }


def _domain_errors(before, after, target, eligible):
    return {
        "full": _errors(before, after, target, np.ones(eligible.shape, dtype=bool)),
        "eligible": _errors(before, after, target, eligible),
    }


def _optimizer_report(solution, count):
    if solution is None:
        return {
            "status": "insufficient_support", "success": False,
            "training_count": count, "parameters": None, "nfev": 0,
            "message": "Fewer than 150 eligible training points; identity fallback only.",
        }
    return {
        "status": "converged" if solution.success else "not_converged",
        "success": bool(solution.success), "training_count": count,
        "parameters": solution.x.tolist(), "nfev": int(solution.nfev),
        "solver_status": int(solution.status), "message": str(solution.message),
        "cost": float(solution.cost), "optimality": float(solution.optimality),
        "at_parameter_bound": bool(np.any(np.isclose(
            solution.x, [-1.0, 0.7], atol=1e-7, rtol=0)) or np.any(np.isclose(
                solution.x, [1.0, 1.3], atol=1e-7, rtol=0))),
    }


def analyze_raw_common_texture(
    energy, pn_frames, training_samples, eligible_mask, *, cancelled=None,
):
    """Return diagnostic arrays and a JSON-safe report for one frozen transform.

    P/N frames are intensity-averaged before taking logs. The fitted source is
    their mean fine band, ``(I-Q4)log(I)``; Q4 retains the existing native full
    4 meV/minimum-seven-point local quadratic definition. The target averages
    the two training intensities before log/filtering and takes a gate median.

    Only shift [-1, 1] native pixel and gain [0.7, 1.3] are fitted. Six equal
    energy blocks exclude a 35 meV guard from eligible training points; tests
    report both full and eligible domains, conditional on the provided mask.
    Each source/half/gate-group/repeat check reuses frozen full-fit parameters.
    PN halves are the first and second contiguous halves in supplied order;
    gate groups are three contiguous, approximately equal index groups.

    Fewer than 150 eligible points yields an explicitly unavailable candidate
    with identity parameters and zero delta. Unsupported fold parameters are
    NaN in the returned numeric array and null in the JSON report. No files,
    input arrays, source metadata or backgrounds are changed. ``cancelled`` is
    a zero-argument predicate; true raises InterruptedError, including in the
    local quadratic kernels and each least-squares evaluation.
    """
    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be a callable or None")
    with cancellation_scope(cancelled):
        e, pn, samples, eligible = _validate(
            energy, pn_frames, training_samples, eligible_mask,
        )
        return _analyze(e, pn, samples, eligible)


def _analyze(e, pn, samples, eligible):
    n = e.size
    gates = samples.shape[1]
    half_count = pn.shape[1] // 2
    # All intensity averages precede log; do not average log frames.
    raw_logs = np.log(pn.mean(axis=1))
    half_logs = np.log(np.stack((
        pn[:, :half_count].mean(axis=1), pn[:, half_count:].mean(axis=1),
    )))
    mean_sample_logs = np.log(samples.mean(axis=0))
    repeat_logs = np.log(samples)
    logs = np.concatenate((
        raw_logs, half_logs.reshape(4, n), mean_sample_logs,
        repeat_logs.reshape(2 * gates, n),
    ))
    check_cancelled()
    q4 = local_quadratic(e, logs, 0.004)
    fine = logs - q4
    source_fine = fine[:2]
    half_source_fine = fine[2:6].reshape(2, 2, n)
    sample_fine = fine[6:6 + gates]
    repeat_fine = fine[6 + gates:].reshape(2, gates, n)
    common = source_fine.mean(axis=0)
    half_common = half_source_fine.mean(axis=1)
    target = np.median(sample_fine, axis=0)
    groups = np.array_split(np.arange(gates), 3)
    group_targets = np.array([np.median(sample_fine[g], axis=0) for g in groups])
    repeat_targets = np.median(repeat_fine, axis=1)
    pixel = np.arange(n, dtype=float)
    spline = CubicSpline(pixel, common, extrapolate=True)

    def prediction(parameters):
        return parameters[1] * spline(pixel + parameters[0])

    def fit(mask):
        return least_squares(
            lambda p: prediction(p)[mask] - target[mask], [0.0, 1.0],
            jac=lambda p: np.column_stack((
                p[1] * spline(pixel + p[0], 1), spline(pixel + p[0]),
            ))[mask],
            bounds=([-1.0, 0.7], [1.0, 1.3]), loss="soft_l1", f_scale=0.0004,
            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300,
        )

    training_count = int(eligible.sum())
    solution = fit(eligible) if training_count >= _MIN_TRAIN else None
    parameters = np.array([0.0, 1.0]) if solution is None else solution.x.copy()

    def frozen_transform(values):
        check_cancelled()
        if np.array_equal(parameters, [0.0, 1.0]):
            return values.copy()
        return parameters[1] * CubicSpline(
            pixel, values, axis=-1, extrapolate=True,
        )(pixel + parameters[0])

    registered = frozen_transform(common)
    delta = registered - common
    registered_halves = frozen_transform(half_common)
    half_delta = registered_halves - half_common
    optimizer = _optimizer_report(solution, training_count)
    block_edges = np.linspace(e[0], e[-1], 7)
    fold_parameters = np.full((6, 2), np.nan)
    folds = []
    for index, (left, right) in enumerate(zip(block_edges[:-1], block_edges[1:])):
        check_cancelled()
        test = (e >= left) & ((e <= right) if index == 5 else (e < right))
        train = eligible & ((e < left - _GUARD_EV) | (e > right + _GUARD_EV))
        masks = {"full": test, "eligible": test & eligible}
        counts = {domain: int(mask.sum()) for domain, mask in masks.items()}
        train_count = int(train.sum())
        can_fit = train_count >= _MIN_TRAIN and max(counts.values()) >= _MIN_TEST
        fold_solution = fit(train) if can_fit else None
        fitted = None if fold_solution is None else prediction(fold_solution.x)
        if fold_solution is not None:
            fold_parameters[index] = fold_solution.x
        domains = {}
        for domain, mask in masks.items():
            supported = train_count >= _MIN_TRAIN and counts[domain] >= _MIN_TEST
            metrics = (_errors(common, fitted, target, mask)
                       if supported and fitted is not None else {
                           "count": counts[domain], "before": None, "after": None,
                           "relative_RMS_change": None,
                       })
            domains[domain] = {
                "supported": supported,
                "status": "evaluated" if supported else "insufficient_support",
                **metrics,
            }
        folds.append({
            "block": index, "energy_bounds": [float(left), float(right)],
            "train_count": train_count, "test_count": counts["eligible"],
            "supported": all(d["supported"] for d in domains.values()),
            "status": "evaluated" if can_fit else "insufficient_support",
            "optimizer_success": bool(fold_solution is not None and fold_solution.success),
            "parameters": None if fold_solution is None else fold_solution.x.tolist(),
            "optimizer": _optimizer_report(fold_solution, train_count),
            "domains": domains,
        })

    registered_sources = frozen_transform(source_fine)
    registered_half_sources = frozen_transform(half_source_fine)

    def same_domains(before, after, reference):
        result = _domain_errors(before, after, reference, eligible)
        result["blocks"] = []
        for index, (left, right) in enumerate(zip(block_edges[:-1], block_edges[1:])):
            mask = (e >= left) & ((e <= right) if index == 5 else (e < right))
            result["blocks"].append({
                "block": index, "energy_bounds": [float(left), float(right)],
                "full": _errors(before, after, reference, mask),
                "eligible": _errors(before, after, reference, mask & eligible),
            })
        return result

    checks = {
        "parameters": parameters.tolist(), "refitted": False,
        "source_check_operation": "Apply the same shift/gain operator to each raw source fine band.",
        "sample_check_operation": "Compare the same registered common source to each unmodified target.",
        "sources": {
            label: same_domains(source_fine[i], registered_sources[i], target)
            for i, label in enumerate(("P", "N"))
        },
        "pn_halves": [{
            "half": h, "frame_indices": [h * half_count, (h + 1) * half_count],
            "common": same_domains(half_common[h], registered_halves[h], target),
            "sources": {
                label: same_domains(
                    half_source_fine[h, i], registered_half_sources[h, i], target,
                ) for i, label in enumerate(("P", "N"))
            },
        } for h in range(2)],
        "gate_groups": [{
            "group": i, "gate_indices": group.tolist(),
            "domains": same_domains(common, registered, group_targets[i]),
        } for i, group in enumerate(groups)],
        "training_repeats": [{
            "repeat": i,
            "domains": same_domains(common, registered, repeat_targets[i]),
        } for i in range(2)],
    }
    q4_delta = local_quadratic(e, delta, 0.004)
    medium_delta = q4_delta - local_quadratic(e, delta, 0.035)
    delta_rms = _rms(delta)
    finite_candidate = bool(np.all(np.isfinite(delta)) and np.all(np.isfinite(half_delta)))
    report = {
        "method": "raw_pn_common_fine_registration_v1",
        "scientifically_validated": False, "automatic_application": False,
        "candidate_available": bool(optimizer["success"] and finite_candidate),
        "protocol": {
            "source": "mean(P fine, N fine); intensity mean before log, no fitted-background shape",
            "fine": "(I-Q4)log(I), native full 4 meV local quadratic, minimum 7 points",
            "target": "gate median of fine(log(mean(two training intensities)))",
            "prediction": "gain*CubicSpline(native_pixel,raw_common_fine)(pixel+shift)",
            "bounds": {"shift_pixels": [-1.0, 1.0], "gain": [0.7, 1.3]},
            "initial_parameters": [0.0, 1.0],
            "solver": {"loss": "soft_l1", "f_scale": 0.0004,
                       "xtol": 1e-12, "ftol": 1e-12, "gtol": 1e-12, "max_nfev": 300},
            "cv": {"energy_blocks": 6, "guard_ev": _GUARD_EV,
                   "minimum_train_points": _MIN_TRAIN, "minimum_test_points": _MIN_TEST,
                   "conditional_on_eligibility_mask": True, "blind_validation": False},
        },
        "optimizer": optimizer,
        "training_errors": _domain_errors(common, registered, target, eligible),
        "folds": folds, "same_transform_checks": checks,
        "support": {
            "training_count_sufficient": training_count >= _MIN_TRAIN,
            "all_fold_domain_counts_sufficient": all(f["supported"] for f in folds),
            "all_optimizers_successful": bool(optimizer["success"] and all(
                f["optimizer_success"] for f in folds)),
            "finite_candidate": finite_candidate,
            "meaning": "Point-count and optimization support only; no scientific acceptance decision.",
        },
        "leakage": {
            "delta_log_RMS": delta_rms,
            "delta_log_range": [float(delta.min()), float(delta.max())],
            "Q4_delta_RMS": _rms(q4_delta),
            "Q4_delta_max_abs": float(np.max(np.abs(q4_delta))),
            "Q4_delta_RMS_over_delta_RMS": _ratio(_rms(q4_delta), delta_rms),
            "medium_delta_RMS": _rms(medium_delta),
            "medium_delta_max_abs": float(np.max(np.abs(medium_delta))),
            "medium_delta_RMS_over_delta_RMS": _ratio(_rms(medium_delta), delta_rms),
        },
        "limitations": [
            "Eligibility may use the full training energy range; CV is conditional, not blind.",
            "P/N-common or gate-common structure does not prove purely optical origin.",
            "Q4 is not an idempotent projection; changes can leak into Q4 and the medium band.",
            "Source/half/group/repeat checks reuse fitted training information and are not independent validation.",
            "Smaller texture errors do not establish full dR/R, derivative or feature preservation.",
            "No source PPoly metadata is propagated or updated by this experimental module.",
        ],
    }
    arrays = {
        "energy": e.copy(), "raw_source_fine": source_fine.copy(),
        "raw_common_fine": common.copy(), "target_train_fine": target.copy(),
        "training_mask": eligible.copy(), "proposed_delta_log": delta,
        "registered_common_fine": registered, "parameters": parameters,
        "block_edges": block_edges, "fold_parameters": fold_parameters,
        "half_raw_source_fine": half_source_fine.copy(),
        "half_raw_common_fine": half_common.copy(),
        "half_proposed_delta_log": half_delta,
        "gate_group_target_fine": group_targets, "repeat_target_fine": repeat_targets,
        "Q4_delta_leakage": q4_delta, "medium_delta_leakage": medium_delta,
    }
    check_cancelled()
    return arrays, report


def apply_common_fine_delta(parent_background, delta_log):
    """Explicitly construct a positive experimental background, without metadata.

    The caller must assess support and full products before opting to use this
    candidate. This function does not validate scientific suitability. A zero
    delta returns an exact copy, avoiding an exp(log(parent)) round trip. Inputs
    must be equal-shape, nonempty 1-D arrays; source PPoly metadata is neither
    accepted nor propagated, because it no longer represents this new array.
    """
    parent = np.asarray(parent_background)
    delta = np.asarray(delta_log, dtype=float)
    if parent.ndim != 1 or not parent.size or delta.shape != parent.shape:
        raise ValueError("parent_background and delta_log must be nonempty 1-D arrays of equal shape")
    if not np.all(np.isfinite(parent) & (parent > 0)):
        raise ValueError("Positive finite parent_background required")
    if not np.all(np.isfinite(delta)):
        raise ValueError("Finite delta_log required")
    if not np.any(delta):
        return parent.copy()
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.asarray(parent, dtype=float) * np.exp(delta)
    if not np.all(np.isfinite(result) & (result > 0)):
        raise ValueError("The proposed delta produces a nonpositive or nonfinite background")
    return result
