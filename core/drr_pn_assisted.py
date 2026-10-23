"""File/provenance boundary for sample-assisted P/N draft generation.

Samples may inform source choice, never supply the denominator's intensity.
Assessment is separate: its files and derivative settings do not define the
generation fingerprint and cannot be fed back into this generation request.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from core import processing_run as P
from core.drr_baseline_candidates import parse_drr_acquisition_conditions
from core.drr_composite_background import CompositeBackground, CompositeSource, load_composite_source
from core.drr_pn_generator import (
    PnGenerationResult, _CallbackCancellation, _settings, dependency_paths as pn_dependencies,
)
from core.drr_pn_pairing import DrrPnPair
from core.drr_pn_store import DrrPnStore, PnGenerationKey
from core.drr_sources import inspect_csv_spectral_grid, portable_source_name, resolve_source_path
from core.load_cancellation import check_load_cancelled, load_scope


ASSISTED_ALGORITHM_VERSION = "pn_sample_assisted_curvature_v1"
SPLIT_POLICY = "ordered-alternating-v1: at least 4 files, otherwise all training"


@dataclass(frozen=True)
class AssistedRequest:
    key: PnGenerationKey
    training_files: tuple[str, ...]
    evaluation_files: tuple[str, ...]

    @property
    def settings(self):
        return self.key.provenance["settings"]


def split_measurements(measurements):
    """Freeze roles from explicit selection order before inspecting intensities."""
    if isinstance(measurements, (str, Path)):
        raise ValueError("Select a sequence of raw measurement files.")
    files = tuple(str(name) for name in measurements)
    if not files or any(not name.strip() for name in files):
        raise ValueError("Sample-assisted generation needs raw measurement files.")
    if len(files) >= 4:
        return files[::2], files[1::2]
    return files, ()


def dependency_paths():
    folder = Path(__file__).resolve().parent
    return (*pn_dependencies(), folder / "drr_pn_assisted.py",
            folder / "drr_pn_sample_analysis.py", folder / "drr_baseline_candidates.py")


def prepare_assisted_request(folder, pair, p_which="all", n_which="all", *,
                             settings=None, measurements=()):
    """Hash generation inputs only. No spectra or assessment results are read."""
    pair = DrrPnPair(**pair) if isinstance(pair, dict) else pair
    training, evaluation = split_measurements(measurements)
    all_paths = tuple(resolve_source_path(folder, name) for name in (*training, *evaluation))
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("Measurement repeats must be distinct files, including path aliases.")
    sources = {resolve_source_path(folder, name) for name in (*pair.p_files, *pair.n_files)}
    if sources.intersection(all_paths):
        raise ValueError("P/N sources cannot also be selected as sample measurements.")
    if any(path.suffix.lower() != ".csv" for path in all_paths):
        raise ValueError("Sample-assisted generation requires original intensity CSV files.")
    training = tuple(portable_source_name(folder, resolve_source_path(folder, name)) for name in training)
    evaluation = tuple(portable_source_name(folder, resolve_source_path(folder, name)) for name in evaluation)
    config = asdict(_settings(settings))
    config["widths_ev"] = list(config["widths_ev"])
    config["sample_analysis"] = {
        "training_files": list(training), "subset": "all gates and full native energy",
        "y_axis": "auto", "split_policy": SPLIT_POLICY,
    }
    key = DrrPnStore(folder).prepare(pair, p_which=p_which, n_which=n_which,
        settings=config, algorithm_version=ASSISTED_ALGORITHM_VERSION,
        dependencies=(*dependency_paths(), *(resolve_source_path(folder, name) for name in training)))
    return AssistedRequest(key, training, evaluation)


def matching_sample_context(version, measurements):
    """A lightweight preference hint, never proof of content freshness or fit."""
    if version.algorithm_version != ASSISTED_ALGORITHM_VERSION:
        return False
    try:
        training, _ = split_measurements(measurements)
        normalize = lambda name: str(name).replace("\\", "/").casefold()
        saved = version.settings["sample_analysis"]["training_files"]
        return tuple(map(normalize, training)) == tuple(map(normalize, saved))
    except (ValueError, KeyError, TypeError):
        return False


def load_sample_stack(folder, files, energy, spectral_columns, *, y_axis="auto"):
    """Retain repeats and gates; reject mixed grids instead of interpolating.

Known acquisition conflicts between samples block treating them as repeats.
Unknown filename conditions remain unknown; equality is not physical proof.
"""
    if not files:
        raise ValueError("No sample measurements were selected.")
    conditions = [asdict(parse_drr_acquisition_conditions(name)) for name in files]
    for key in conditions[0]:
        known = {item[key] for item in conditions if item[key] is not None}
        if len(known) > 1:
            raise ValueError(f"Sample repeats have different {key}; select one measurement group.")
    axis = P.resolve_shared_y_axis_request(files, y_axis)
    gate = label = coordinates = None
    stack = []
    expected = np.asarray(spectral_columns, float)
    for name in files:
        check_load_cancelled()
        path = resolve_source_path(folder, name)
        actual = np.asarray(inspect_csv_spectral_grid(path), float)
        if not np.array_equal(actual, expected):
            raise ValueError(f"{path.name}: sample/P/N require the same ordered native spectral grid.")
        data = P._load_canonical(str(folder), str(path), y_axis=axis)
        e, g, values = (np.asarray(data[key], float) for key in ("energy", "gate_axis", "Z"))
        if not np.array_equal(e, energy):
            raise ValueError(f"{path.name}: sample/P/N energy grids differ; no resampling is performed.")
        if (g.ndim != 1 or not g.size or not np.isfinite(g).all()
                or (len(g) > 1 and np.any(np.diff(g) <= 0))):
            raise ValueError(f"{path.name}: distinct ordered gate coordinates are required.")
        if (values.shape != (len(g), len(e)) or not np.isfinite(values).all() or np.any(values <= 0)):
            raise ValueError(f"{path.name}: raw sample intensities must be finite and positive.")
        current_coordinates = {k: np.asarray(v, float) for k, v in data.get("gate_coordinates", {}).items()}
        if gate is None:
            gate, label, coordinates = g.copy(), data["gate_label"], current_coordinates
        elif (not np.array_equal(g, gate) or data["gate_label"] != label
              or set(current_coordinates) != set(coordinates)
              or any(not np.array_equal(v, coordinates[k], equal_nan=True)
                     for k, v in current_coordinates.items())):
            raise ValueError("Sample repeats require matching gate coordinates and scan meaning; no resampling is performed.")
        stack.append(values.copy())
    return np.stack(stack), gate, label


def _verify_request(folder, pair, p_which, n_which, request):
    store = DrrPnStore(folder)
    provenance = request.key.provenance
    if (Path(request.key.root).resolve() != store.root or request.key.pair_id != store.pair_id(pair)
            or provenance["algorithm_version"] != ASSISTED_ALGORITHM_VERSION
            or provenance["frame_choice"] != {"P": p_which, "N": n_which}
            or tuple(provenance["settings"]["sample_analysis"]["training_files"]) != request.training_files):
        raise ValueError("The prepared sample-assisted request no longer matches this selection.")
    store._validate_key(request.key)


def generate_assisted_background(folder, pair, p_which="all", n_which="all", *, request, cancelled=None):
    """Return an unsaved, unaccepted background; never apply it to the app."""
    from core.drr_pn_sample_analysis import generate_sample_assisted_arrays

    pair = DrrPnPair(**pair) if isinstance(pair, dict) else pair
    scope = load_scope(_CallbackCancellation(cancelled)) if cancelled is not None else nullcontext()
    with scope:
        check_load_cancelled()
        _verify_request(folder, pair, p_which, n_which, request)
        sources = (load_composite_source(folder, pair.p_files, p_which, key="P", label="P source"),
                   load_composite_source(folder, pair.n_files, n_which, key="N", label="N source"))
        p_source, n_source = sources
        if (not np.array_equal(p_source.energy, n_source.energy)
                or not np.array_equal(np.asarray(p_source.spectral_columns, float),
                                      np.asarray(n_source.spectral_columns, float))):
            raise ValueError("P and N require the same ordered native spectral grid.")
        if any(source.boundaries_ev for source in sources):
            raise ValueError("Choose original continuous P/N spectra, not stitched backgrounds.")
        samples, gate, gate_label = load_sample_stack(folder, request.training_files,
            p_source.energy, p_source.spectral_columns)
        base_settings = {k: v for k, v in request.settings.items() if k != "sample_analysis"}
        arrays, report = generate_sample_assisted_arrays(p_source.energy, p_source.intensity,
            n_source.intensity, samples, gate, settings=base_settings)
        report.update(algorithm_version=ASSISTED_ALGORITHM_VERSION, settings=request.settings,
            training_files=list(request.training_files), training_shape=list(samples.shape),
            training_gate_label=gate_label, measurements_used=True, sample_used=True,
            p_frame_policy=p_which, n_frame_policy=n_which,
            original_source_hashes=[list(item) for source in sources for item in source.source_hashes],
            scientific_acceptance="not assessed; complete dR/R and d2E both require review",
            v66_equivalent=False)
        _verify_request(folder, pair, p_which, n_which, request)
        encoded = json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False)
        digest = hashlib.sha256(encoded.encode() + arrays["background"].tobytes()).hexdigest()
        generated = CompositeSource(key=f"PN-assisted-{digest[:20]}", label="PN sample-assisted — draft",
            files=tuple(path for source in sources for path in source.files), which="all",
            energy=p_source.energy, spectral_columns=p_source.spectral_columns,
            intensity=arrays["background"], boundaries_ev=(),
            source_hashes=tuple(item for source in sources for item in source.source_hashes),
            generation_json=encoded)
        check_load_cancelled()
        return PnGenerationResult(CompositeBackground((*sources, generated), generated.key), report, arrays)


# Assessment implementation and parameters are deliberately outside generation
# dependencies. Changing these cannot change a background's generation key.
from core.drr_pn_assessment import assess_assisted_result, save_assessment  # noqa: E402
