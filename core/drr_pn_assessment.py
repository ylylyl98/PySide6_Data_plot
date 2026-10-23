"""Independent, non-optimizing full-map diagnostics for a generated P/N draft.

These descriptive measurements cannot identify all real resonances or certify
scientific acceptance. In particular, a smaller stationary d2E component is
not a quality objective: real gate-stationary features contribute to it too.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import numpy as np

from core import processing_run as P
from core import processing
from core.drr_composite_background import compose_background, read_composite_background
from core.drr_pn_generator import _CallbackCancellation, generate_pn_arrays
from core.drr_pn_store import DrrPnStore, _file_digest
from core.drr_sources import portable_source_name, resolve_source_path
from core.load_cancellation import check_load_cancelled, load_scope


def _background_identity(energy, intensity):
    return hashlib.sha256(np.asarray(energy, dtype="<f8").tobytes()
                          + np.asarray(intensity, dtype="<f8").tobytes()).hexdigest()


def _summary(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite full-map diagnostic values.")
    return {"min": float(values.min()), "max": float(values.max()),
            "percentiles_1_5_50_95_99": np.percentile(values, [1, 5, 50, 95, 99]).tolist(),
            "rms": float(np.sqrt(np.mean(values**2)))}


def assess_assisted_result(folder, result, request, *, cancelled=None, windows=(11, 21), polyorder=3):
    """Assess once, after generation; never change or rank backgrounds.

Use only the withheld selection when it exists, otherwise clearly label the
training-data diagnostic. Each repeat is retained until raw intensity averaging
before division. Derivatives run on the entire native energy range.
"""
    from core.drr_pn_assisted import load_sample_stack

    scope = load_scope(_CallbackCancellation(cancelled)) if cancelled is not None else nullcontext()
    with scope:
        check_load_cancelled()
        if (not isinstance(polyorder, int) or isinstance(polyorder, bool) or polyorder < 2
                or not windows or any(not isinstance(w, int) or isinstance(w, bool)
                    or w <= polyorder + 1 or w % 2 != 1 for w in windows)):
            raise ValueError("Assessment requires odd SG windows and integer polynomial order >= 2.")
        files = request.evaluation_files or request.training_files
        paths = [resolve_source_path(folder, name) for name in files]
        records = [{"path": portable_source_name(folder, path), "sha256": _file_digest(path)} for path in paths]
        source = result.document.sources[0]
        samples, gate, label = load_sample_stack(folder, files, source.energy, source.spectral_columns)
        if records != [{"path": portable_source_name(folder, path), "sha256": _file_digest(path)} for path in paths]:
            raise ValueError("Assessment inputs changed while reading; no assessment was saved.")
        e = np.asarray(source.energy, float)
        native_e, native_b, _ = compose_background(result.document)
        if not np.array_equal(native_e, e) or not np.array_equal(native_b, result.arrays["background"]):
            raise ValueError("Assessment arrays do not match the generated document.")
        raw = np.asarray(result.arrays["raw_sources"], float)
        settings = {k: v for k, v in request.settings.items() if k != "sample_analysis"}
        pn_only, _ = generate_pn_arrays(e, raw[0], raw[1], settings=settings)
        backgrounds = {"sample_assisted": result.arrays["background"],
                       "pn_only": pn_only["background"], "P": raw[0], "N": raw[1]}
        average = samples.mean(axis=0)
        diagnostics = {}
        for method, background in backgrounds.items():
            check_load_cancelled()
            ratios = samples / background - 1
            drr = average / background - 1
            metrics = {"drr": _summary(drr), "d2E": {}}
            if len(samples) > 1:
                metrics["drr_repeat_spread_rms"] = float(np.sqrt(np.mean(np.var(ratios, axis=0, ddof=1))))
            for window in windows:
                if len(e) < window:
                    metrics["d2E"][str(window)] = {"status": "unavailable: native grid shorter than requested SG window"}
                    continue
                derivative = processing._sg_derivative_energy_rows(drr, e, window, polyorder, 2)
                metrics["d2E"][str(window)] = {
                    **_summary(derivative),
                    "stationary_component_rms_not_a_quality_score": float(np.sqrt(np.mean(np.median(derivative, axis=0)**2))),
                }
            diagnostics[method] = metrics
        report = {
            "schema_version": 1, "report_kind": "pn_sample_assisted_full_map_diagnostics",
            "generation_fingerprint": request.key.fingerprint,
            "assessed_background_sha256": _background_identity(native_e, native_b),
            "assessment_inputs": records,
            "role": "withheld_from_generation" if request.evaluation_files else "training_data_only",
            "independent_holdout": bool(request.evaluation_files),
            "holdout_scope": "Withheld from this generation call; not a claim of previously unseen experimental data.",
            "settings": {"sg_windows": list(windows), "polyorder": polyorder, "deriv": 2,
                         "oversample": 1., "interp_kind": "cubic", "origin_like": False,
                         "raw_average_before_ratio": True, "full_native_energy": True},
            "assessment_code_sha256": _file_digest(Path(__file__)),
            "derivative_code_sha256": _file_digest(Path(P.__file__)),
            "main_processing_code_sha256": _file_digest(Path(processing.__file__)),
            "sample_shape": list(samples.shape), "gate_label": label,
            "gate_range": [float(gate.min()), float(gate.max())],
            "energy_range_eV": [float(e[0]), float(e[-1])], "metrics": diagnostics,
            "acceptance": {"complete_drr": "requires review", "complete_d2E": "requires review", "passed": False},
            "interpretation": "Descriptive full-range diagnostics only; no automatic acceptance or method ranking. "
                "Review baseline, peak positions, line shapes, weak features and branches in both main maps "
                "with fixed processing and display scales. Lower d2E magnitude can mean erased real features. "
                "BN is an optional effect comparator and is never a generation input.",
        }
        check_load_cancelled()
        return report


def save_assessment(folder, version, assessment):
    """Content-addressed side report; never mutate the generation manifest."""
    check_load_cancelled()
    if assessment.get("generation_fingerprint") != version.fingerprint:
        raise ValueError("Assessment belongs to a different generated background.")
    if "assessed_background_sha256" in assessment:
        energy, background, _ = compose_background(read_composite_background(version.csv_path))
        if assessment["assessed_background_sha256"] != _background_identity(energy, background):
            raise ValueError("Assessment background values differ from the saved version.")
    elif assessment.get("status") != "error":
        raise ValueError("Assessment is missing the evaluated background identity.")
    payload = {**assessment, "background_version_id": version.version_id,
               "background_csv_sha256": _file_digest(version.csv_path)}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    directory = DrrPnStore(folder).directory / "Evaluations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{version.version_id}_{digest[:20]}.json"
    if not path.exists():
        check_load_cancelled()
        DrrPnStore._atomic_json(path, payload)
    return path
