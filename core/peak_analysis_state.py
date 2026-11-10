"""Freshness and availability of reviewed Peak Analysis results."""
from copy import deepcopy

FIT_FIELDS = ("fit_status", "fit_center", "center_error_ev", "fwhm_mev", "fwhm_error_mev",
              "height", "height_error", "area", "area_error", "fit_area_full", "fit_area_full_error",
              "uncertainty_status", "rms", "fit_quality_status", "fit_local_residual_sigma",
              "fit_height_to_local_noise")


def invalidate_fit(dataset):
    """Discard computed fits while preserving candidates and manual constraints.

    Copy the result before changing it: captured worker requests can share the
    previously committed result with the GUI.
    """
    if dataset.get("result") is not None:
        result = deepcopy(dataset["result"])
        result["fits"] = []
        if 'method' in result:
            result['method'] = result['method'].split(' + bounded multi-peak ', 1)[0]
        for point in result.get("points", []):
            for field in FIT_FIELDS:
                point.pop(field, None)
        dataset["result"] = result
    dataset["overlay_position"] = "detected"
    return dataset


def analysis_state(dataset):
    result = dataset.get("result") or {}
    points = result.get("points", [])
    enabled = {b["id"] for b in dataset.get("branches", []) if b.get("enabled", True)}
    scope = dataset.get("analysis_scope", "map")
    reviewed = [p for p in points if p.get("status") == "accepted" and p.get("branch_id") in enabled
                and (scope != "spectrum" or p.get("row_index") == dataset.get("row"))]
    detection_dirty = bool(dataset.get("detection_dirty"))
    filters_pending = bool(dataset.get("filters_pending"))
    dirty = detection_dirty or filters_pending
    if detection_dirty:
        status = "Detection settings changed. Find peaks again."
    elif filters_pending:
        status = "Apply filters to update the reviewed peaks."
    elif not result:
        status = "Find peaks to begin."
    elif not reviewed:
        status = "Select an accepted peak to fit."
    elif result.get("fits"):
        status = "Fit results available."
    else:
        status = "Review peaks, then fit."
    return dict(total_count=len(points), review_count=sum(p.get("fit_status") != "ok" for p in points), accepted_count=len(reviewed),
                fitted_count=sum(p.get("fit_status") == "ok" for p in reviewed),
                fit_count=len(result.get("fits", [])),
                detection_dirty=detection_dirty, filters_pending=filters_pending, dirty=dirty,
                can_fit=dataset.get("kind") == "PL" and bool(reviewed) and not dirty,
                can_export=bool(result) and not dirty, status=status)
