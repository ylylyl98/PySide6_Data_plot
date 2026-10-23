"""Read original P/N frames and explicitly selected training measurements.

No saved preparation, background, fit, or evaluation reference is accepted by
this boundary. Generation returns a native composite but neither saves nor
applies it. Callers choose the destination through the existing store/export.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from core.drr_composite_background import (
    CompositeBackground, CompositeSource, is_composite_background, load_composite_source,
)
from core.drr_pn_assisted import load_sample_stack
from core.drr_pn_generator import PnGenerationResult
from core.drr_pn_store import _file_digest
from core.drr_sources import resolve_source_path
from core.load_cancellation import check_load_cancelled


@dataclass(frozen=True)
class RawPnInputs:
    energy: np.ndarray
    pn_mean: np.ndarray
    pn_frames: np.ndarray
    training_samples: np.ndarray
    gate: np.ndarray
    gate_label: str
    sources: tuple[CompositeSource, CompositeSource]
    input_records: tuple[dict, ...]


def _readonly(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def load_raw_inputs(folder, p_file, n_file, training_files):
    """Load one original CSV per source, all frames, with explicit sample roles.

    Frame order and the native spectral grid are retained. The initial
    reconstruction protocol uses equal, even P/N frame counts for its first/
    second-half checks. Evaluation files must be passed separately to evaluators.
    """
    if isinstance(training_files, (str, Path)) or len(training_files) != 2:
        raise ValueError("Select exactly two distinct raw training measurement files for this protocol.")
    paths = tuple(resolve_source_path(folder, str(p)) for p in (p_file, n_file, *training_files))
    if len(set(paths)) != len(paths):
        raise ValueError("P, N and training measurements must be distinct original files.")
    if any(p.suffix.lower() != ".csv" for p in paths):
        raise ValueError("Raw reconstruction accepts original intensity CSV files only.")
    if any(is_composite_background(p) for p in paths):
        raise ValueError("Raw reconstruction cannot take a saved composite as an original input.")
    records = tuple({"role": role, "path": str(path), "sha256": _file_digest(path)}
                    for role, path in zip(("P", "N", *("training" for _ in training_files)), paths))
    sources = tuple(load_composite_source(folder, (str(path),), "all", key=role, label=f"{role} source")
                    for path, role in zip(paths[:2], ("P", "N")))
    p, n = sources
    if (p.spectral_columns != n.spectral_columns or not np.array_equal(p.energy, n.energy)):
        raise ValueError("P and N must have the same ordered native spectral columns.")
    frames = []
    numeric = np.asarray(p.spectral_columns, float)
    axis = 1240.0 / numeric if np.median(numeric) > 20 else numeric
    order = np.argsort(axis)
    for path, source in zip(paths[:2], sources):
        check_load_cancelled()
        frame = pd.read_csv(path)
        values = frame[list(source.spectral_columns)].to_numpy(float)[:, order]
        if (not np.isfinite(values).all() or np.any(values <= 0)
                or values.shape[0] < 4 or values.shape[0] % 2):
            raise ValueError("P/N require positive finite spectra and an even count of at least four frames.")
        if not np.allclose(values.mean(axis=0), source.intensity, rtol=1e-12, atol=0):
            raise ValueError("The original frame mean disagrees with the native baseline loader.")
        frames.append(values)
    if frames[0].shape != frames[1].shape:
        raise ValueError("This reconstruction protocol requires equal P/N frame counts.")
    samples, gate, label = load_sample_stack(folder, tuple(str(p) for p in paths[2:]),
                                            p.energy, p.spectral_columns)
    for record in records:
        if _file_digest(Path(record["path"])) != record["sha256"]:
            raise ValueError("A raw input changed during loading; retry with a stable selection.")
    return RawPnInputs(_readonly(p.energy), _readonly([p.intensity, n.intensity]),
                       _readonly(frames), _readonly(samples), _readonly(gate), label, sources, records)


def dependency_paths():
    """Implementation identity, separate from any historical regression oracle."""
    from core.drr_pn_assisted import dependency_paths as raw_dependencies
    return tuple(sorted(set((*Path(__file__).parent.rglob("*.py"), *raw_dependencies(),
                             *(Path(__file__).parents[1] / name for name in (
                                 "drr_pn_store.py", "drr_pn_texture_compatibility.py",
                                 "drr_pn_texture_support.py"))))))


def generate_from_files(folder, p_file, n_file, training_files, *, cancelled=None, progress=None,
                        method="v66", force_source_refit=False, feature_protection=()):
    """Return an unsaved composite from raw files and the declared protocol.

    The method selects historical V66 regression (``v66``), adaptive pure-source
    regions (``generalized``), experimental compact repairs (``local_evidence``),
    bounded-source construction with conditional compact repairs (``conditioned``),
    or that construction with optional supported raw-common fine correction
    (``conditioned_bounded``).
    Scientific acceptance for other measurements is not inferred from a
    successful numerical reconstruction. No saved baseline is overwritten.
    """
    from .pipeline import reconstruct_background
    from .source_cache import SOURCE_PREPARATION_CACHE
    from .feature_protection import normalize_feature_protection
    from core.drr_pn_generator import _CallbackCancellation
    from core.load_cancellation import load_scope
    from contextlib import nullcontext

    scope = load_scope(_CallbackCancellation(cancelled)) if cancelled is not None else nullcontext()
    with scope:
        protected = normalize_feature_protection(feature_protection)
        if protected and method != "conditioned_bounded":
            raise ValueError("Feature protection requires conditioned_bounded reconstruction.")
        started = perf_counter()
        code = {str(path): _file_digest(path) for path in dependency_paths()}
        load_started = perf_counter()
        raw = load_raw_inputs(folder, p_file, n_file, training_files)
        load_seconds = perf_counter() - load_started
        source_identity = {"implementation": code,
                           "raw_sources": [{"role": row["role"], "sha256": row["sha256"]}
                                           for row in raw.input_records if row["role"] in ("P", "N")],
                           "spectral_columns": raw.sources[0].spectral_columns}
        arrays, report = reconstruct_background(raw.energy, raw.pn_mean, raw.pn_frames,
            raw.training_samples, cancelled=cancelled, progress=progress, method=method,
            source_cache=SOURCE_PREPARATION_CACHE, source_cache_identity=source_identity,
            force_source_refit=force_source_refit,
            **({"feature_protection": protected} if protected else {}))
        for record in raw.input_records:
            if _file_digest(Path(record["path"])) != record["sha256"]:
                raise ValueError("A reconstruction input changed during fitting; discard this result.")
        if code != {str(path): _file_digest(path) for path in dependency_paths()}:
            raise ValueError("Reconstruction implementation changed during fitting; discard this result.")
        report = {**report, "input_source_keys": ["P", "N"],
                  "input_records": list(raw.input_records), "implementation_sha256": code,
                  "training_gate_label": raw.gate_label,
                  "training_shape": list(raw.training_samples.shape),
                  "scientifically_validated": False,
                  "scientific_acceptance": "Review dR/R and d2E for the intended features, their baseline neighborhoods and repeatability in this measurement group. Outside-range residuals are not an automatic rejection of local use."}
        execution = dict(report.get("execution", {}))
        execution["timings_seconds"] = {**execution.get("timings_seconds", {}),
                                       "raw_input_load": load_seconds,
                                       "io_total": perf_counter() - started}
        report["execution"] = execution
        encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)
        key = hashlib.sha256(encoded.encode("utf8") + np.asarray(arrays["background"]).tobytes()).hexdigest()
        generated = CompositeSource(f"PN-reconstructed-{key[:20]}", "PN reconstructed source model",
            tuple(str(path) for source in raw.sources for path in source.files), "all",
            raw.energy, raw.sources[0].spectral_columns, arrays["background"],
            source_hashes=tuple(item for source in raw.sources for item in source.source_hashes),
            generation_json=encoded)
        check_load_cancelled()
        return PnGenerationResult(CompositeBackground((*raw.sources, generated), generated.key), report, arrays)
