"""Read saved regional provenance without inferring scientific acceptance.

This presentation adapter never loads measurements or reconstructs a background.
Region rows describe saved construction/evidence records, which can overlap;
missing records are not interpreted as supported or featureless spectra.
"""
from dataclasses import dataclass
import math
from numbers import Integral
from collections.abc import Mapping

import numpy as np


@dataclass(frozen=True)
class PnRegionRow:
    xmin_ev: float
    xmax_ev: float
    source: str
    operation: str
    support: str
    reason: str
    record_path: str


@dataclass(frozen=True)
class PnRegionReport:
    rows: tuple[PnRegionRow, ...]
    scientific_review: str
    notes: tuple[str, ...]


def describe_pn_regions(energy, report) -> PnRegionReport:
    """Normalize known P/N report formats, retaining uncertainty and gaps.

    Indices in renderer plans are inclusive native-point indices. Repair plans
    describe requests/operations, not a complete map of the final intensity's
    source. Optional fine-correction coverage is separate from those records.
    A saved ``scientifically_validated`` flag alone cannot validate the current
    measurement, whose identity and complete products are not inputs here.
    """
    e = np.asarray(energy, dtype=float)
    if e.ndim != 1 or e.size < 2 or not np.isfinite(e).all() or np.any(np.diff(e) <= 0):
        raise ValueError("A complete increasing native energy axis is required")
    report = report if isinstance(report, Mapping) else {}
    has_protection = isinstance(report.get("feature_protection"), Mapping)
    rows, construction_bounds, notes = [], [], []
    malformed = 0

    def bounds(record, *fields):
        nonlocal malformed
        for field in fields:
            value = record.get(field) if isinstance(record, Mapping) else record
            if value is None:
                continue
            try:
                if isinstance(value, (str, bytes)) or len(value) != 2:
                    raise ValueError
                if "indices" in field or field == "affected_bridge_support":
                    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in value):
                        raise ValueError
                    first, last = value
                    if not 0 <= first <= last < len(e):
                        raise ValueError
                    lo, hi = float(e[first]), float(e[last])
                else:
                    if any(isinstance(v, bool) for v in value):
                        raise ValueError
                    lo, hi = map(float, value)
                if not (math.isfinite(lo) and math.isfinite(hi) and e[0] <= lo <= hi <= e[-1]):
                    raise ValueError
                return lo, hi
            except (TypeError, ValueError, OverflowError):
                malformed += 1
                return None
        malformed += 1
        return None

    def records(container, key):
        nonlocal malformed
        value = container.get(key, [])
        if not isinstance(value, (tuple, list)):
            malformed += 1
            return []
        return value

    def reason_text(value):
        if isinstance(value, (list, tuple)):
            return "; ".join(str(v).replace("_", " ") for v in value)
        return str(value).replace("_", " ") if value is not None else ""

    def source_text(value):
        return value if isinstance(value, str) and value.strip() else "Not recorded"

    def add(limits, source, operation, support, reason, path, *, construction=False):
        if limits is None:
            return
        rows.append(PnRegionRow(*limits, str(source), operation, support, reason, path))
        if construction:
            construction_bounds.append(limits)

    for i, row in enumerate(records(report, "source_segments")):
        if not isinstance(row, Mapping):
            malformed += 1
            continue
        limits = bounds([row.get("xmin_ev"), row.get("xmax_ev")], "energy_bounds")
        add(limits, source_text(row.get("source_key")), "Saved source segment",
            "Conditional construction", "Source choice does not establish a featureless spectrum.",
            f"source_segments[{i}]", construction=True)
    for i, row in enumerate(records(report, "transitions")):
        if not isinstance(row, Mapping):
            malformed += 1
            continue
        add(bounds(row, "energy_bounds", "indices"),
            f"{source_text(row.get('source_from'))} → {source_text(row.get('source_to'))}",
            "Transition / bridge", "Conditional construction",
            "Bridge continuity does not establish complete derivative quality.",
            f"transitions[{i}]", construction=True)
    for key, support, reason in (
        ("unsupported_regions", "Unsupported", "Saved source-support check is insufficient."),
        ("unresolved_regions", "Unresolved", "Saved source ambiguity remains unresolved."),
        ("sample_protected_regions", "Protected / conditional", "Sample evidence protected this region; it does not certify the background."),
    ):
        for i, row in enumerate(records(report, key)):
            add(bounds(row, "energy_bounds"), "Not specified", "Saved evidence flag", support, reason, f"{key}[{i}]")

    selector = report.get("region_diagnostics", {})
    if not isinstance(selector, Mapping):
        selector = {}
        malformed += 1
    resolved = selector.get("resolved_regions", {})
    if not isinstance(resolved, Mapping):
        resolved = {}
        malformed += 1
    deferred = records(resolved, "deferred_runs")
    for i, row in enumerate(deferred):
        if not isinstance(row, Mapping):
            malformed += 1
            continue
        retained = row.get("new_switch_authorized") is False
        explanation = reason_text(row.get("reasons")) or "Deferral reason not recorded."
        if row.get("nominal_SG21_supported") is True:
            explanation += "; nominal SG21 window supported, but the source switch was deferred."
        add(bounds(row, "core_bounds_eV"), f"Requested {source_text(row.get('source'))}",
            "Source selection deferred",
            "Control retained at selector only" if retained else "Selector retention not recorded",
            explanation, f"region_diagnostics.resolved_regions.deferred_runs[{i}]")
    if deferred:
        notes.append("Selector deferrals describe source-weight decisions before rendering; they do not describe final construction coverage or scientific acceptance.")

    render = report.get("render_diagnostics", {})
    full = render.get("full", {}) if isinstance(render, Mapping) else {}
    full = full if isinstance(full, Mapping) else {}
    parent_scope = ("conditioned_parent_diagnostics" in report
                    or full.get("parent_diagnostics_path") == "conditioned_parent_diagnostics.full")
    diagnostics_path = "render_diagnostics.full"
    if parent_scope:
        # The bounded renderer moves the inherited construction here. Its
        # polynomials/derivatives describe the parent, never the final candidate.
        diagnostics_path = "conditioned_parent_diagnostics.full"
        parent = report.get("conditioned_parent_diagnostics")
        full = parent.get("full") if isinstance(parent, Mapping) else None
        if not isinstance(full, Mapping):
            full = {}
            malformed += 1
            notes.append("Parent construction metadata is missing or unreadable; no source support is inferred.")
        notes.append("Parent construction records precede optional fine correction; parent polynomials and derivatives do not describe the final candidate.")
    for key, operation in (("repair_plan", "Regional repair"), ("regional_plan", "Regional repair"),
                           ("local_plan", "Compact repair")):
        if parent_scope:
            operation = "Parent " + operation.lower()
        for i, row in enumerate(records(full, key)):
            if not isinstance(row, Mapping):
                malformed += 1
                continue
            applied = row.get("apply") is True
            route = row.get("route")
            source = row.get("source")
            if not isinstance(source, str) or not source:
                source_index = row.get("source_index")
                source = ("PN"[source_index] if isinstance(source_index, Integral)
                          and not isinstance(source_index, bool) and source_index in (0, 1) else "Not recorded")
            status = ("Applied / conditional" if applied else "Deferred / unresolved"
                      if row.get("apply") is False else "Application not recorded")
            if applied and parent_scope:
                status = "Applied to parent / conditional"
            if route == "compact_only":
                status = "Routed to compact repair"
            explanation = reason_text(row.get("reasons")) or "Conditional source evidence; complete products still require review."
            limits = bounds(row, "core_energy_bounds", "core_indices")
            add(limits, source if applied else f"Requested {source}", operation, status, explanation,
                f"{diagnostics_path}.{key}[{i}]", construction=True)
            # Show the support separately: it is not a pure measured-source span.
            support_fields = ("support_indices",) if key == "local_plan" else ("affected_bridge_support",)
            if any(field in row for field in support_fields):
                support_limits = bounds(row, *support_fields)
                if support_limits != limits:
                    add(support_limits, f"{source} / parent", f"{operation} support", status, explanation,
                        f"{diagnostics_path}.{key}[{i}].{support_fields[0]}")
            conflicts = row.get("opposite_exclusive_indices", [])
            if not isinstance(conflicts, (tuple, list)):
                malformed += 1
                conflicts = []
            for j, index in enumerate(conflicts):
                add(bounds({"indices": [index, index]}, "indices"), "Opposite-source evidence",
                    "Parent source conflict" if parent_scope else "Source conflict", "Unresolved",
                    "Requested source and exclusive source evidence disagree.",
                    f"{diagnostics_path}.{key}[{i}].opposite_exclusive_indices[{j}]")
    for i, row in enumerate(records(full, "unmet_exclusive_intervals")):
        if not isinstance(row, Mapping):
            malformed += 1
            continue
        add(bounds(row, "energy_bounds", "indices"), f"Required {source_text(row.get('required_source'))}",
            "Parent unmet source evidence" if parent_scope else "Unmet source evidence", "Unresolved",
            reason_text(row.get("reason")), f"{diagnostics_path}.unmet_exclusive_intervals[{i}]")

    applicability = report.get("region_applicability")
    if (isinstance(applicability, Mapping) and type(applicability.get("schema_version")) is int
            and applicability["schema_version"] == 1):
        labels = {
            "supported_candidate": "Conditional correction candidate",
            "boundary_guard": "Boundary guard / no correction",
            "unsupported": "Unsupported / no correction",
            "insufficient_contiguous_support": "Insufficient continuous support",
            "proposal_rejected": "Correction proposal rejected",
        }
        for i, row in enumerate(records(applicability, "regions")):
            if not isinstance(row, Mapping):
                malformed += 1
                continue
            correction = ("Included in candidate" if row.get("correction_applied") is True else
                          "Parent retained; this does not establish its suitability"
                          if row.get("correction_applied") is False else "Correction application not recorded")
            if has_protection:
                correction = ("Included before feature protection" if row.get("correction_applied") is True else
                              "Parent retained before feature protection; this does not establish its suitability"
                              if row.get("correction_applied") is False else "Pre-protection correction application not recorded")
            reason = reason_text(row.get("reason"))
            status = row.get("status")
            add(bounds(row, "energy_bounds"), source_text(row.get("source")),
                "Pre-protection fine correction" if has_protection else "Optional fine correction",
                labels.get(status, "Correction support not recorded") if isinstance(status, str) else "Correction support not recorded",
                f"{reason}. {correction}." if reason else f"{correction}.", f"region_applicability.regions[{i}]")
        notes.append("Optional fine-correction coverage describes the correction only, not the validity of the base background.")
        coverage = applicability.get("coverage_fraction")
        if isinstance(coverage, (int, float)) and not isinstance(coverage, bool) and math.isfinite(coverage) and 0 <= coverage <= 1:
            notes.append(f"Recorded optional correction coverage: {coverage:.1%}; this is not a scientific pass rate.")
    elif applicability is not None:
        malformed += 1

    protection = report.get("feature_protection")
    if protection is not None:
        protected_full = protection.get("full") if isinstance(protection, Mapping) else None
        if not isinstance(protected_full, Mapping):
            malformed += 1
        else:
            for i, row in enumerate(records(protected_full, "regions")):
                if not isinstance(row, Mapping):
                    malformed += 1
                    continue
                path = f"feature_protection.full.regions[{i}]"
                source = source_text(row.get("source"))
                requested = bounds(row, "requested_bounds_eV")
                plateau = bounds(row, "plateau_bounds_eV")
                support = bounds(row, "support_bounds_eV")
                add(requested, f"Raw {source}", "Explicit feature protection",
                    "User selected / requires review",
                    "Preserves the selected raw source's local log line shape; source selection does not establish scientific validity.",
                    path)
                add(plateau, f"Raw {source}", "Feature protection and derivative guard",
                    "Conditional construction",
                    "Includes the nominal SG21 guard around the requested feature range.",
                    path + ".plateau_bounds_eV")
                if support is not None and plateau is not None and support[0] <= plateau[0] <= plateau[1] <= support[1]:
                    for side, limits in (("left", (support[0], plateau[0])),
                                         ("right", (plateau[1], support[1]))):
                        add(limits, f"Raw {source} / parent", "Feature protection transition",
                            "Requires complete-product review",
                            "Smooth construction alone does not establish the absence of artificial features.",
                            path + f".{side}_transition")
                elif support is not None and plateau is not None:
                    malformed += 1
            notes.append("Explicit feature protection is applied after the parent and optional fine correction. Earlier source and correction rows describe the pre-protection construction; they do not override the saved protection settings.")
            notes.append("Protection records are construction choices, not evidence that a stationary feature is noise or that the result has passed scientific review.")

    # Explicitly expose areas lacking construction records; evidence flags and
    # optional correction rows must not fill gaps in base-source provenance.
    gap_operation = "Parent construction not recorded" if parent_scope else "Base construction not recorded"
    cursor = float(e[0])
    for lo, hi in sorted(construction_bounds):
        if lo > cursor:
            add((cursor, lo), "Not recorded", gap_operation, "Not recorded",
                "Saved metadata does not describe source/support here; no suitability is inferred.", "metadata_gap")
        cursor = max(cursor, hi)
    if cursor < e[-1] or not construction_bounds:
        add((cursor, float(e[-1])), "Not recorded", gap_operation, "Not recorded",
            "Saved metadata does not describe source/support here; no suitability is inferred.", "metadata_gap")
    if malformed:
        notes.append(f"{malformed} regional record(s) could not be read; no support is inferred from them.")
    notes.append("Regional support is conditional. Review complete ΔR/R and d²E for the current measurement.")
    review = ("Not scientifically validated for this measurement" if report.get("scientifically_validated") is False
              else "Scientific review of this measurement is not recorded")
    if report.get("scientifically_validated") is True:
        notes.append("A saved validation flag does not establish applicability to the current measurement.")
    return PnRegionReport(tuple(sorted(rows, key=lambda r: (r.xmin_ev, r.xmax_ev, r.operation))), review, tuple(notes))
