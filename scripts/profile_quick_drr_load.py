"""Read-only DRR worker benchmark with isolated caches and inclusive phase timing.

Each mode runs in a fresh Python process. "Cold" means empty application caches,
not flushed operating-system disk caches. cProfile overhead is reported separately
by a final profiled repeat; the cold/warm totals do not include cProfile.
No background association is committed and a Python audit hook rejects writes
inside the selected experiment, including settings/history/cache writes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import cProfile
from dataclasses import replace
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import tempfile
import time
import traceback


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, action="append", default=[])
    parser.add_argument("--background", type=Path, action="append", default=[])
    parser.add_argument("--modes", nargs="+", choices=("self", "savedrestore", "fixedBN"),
                        default=("self", "savedrestore", "fixedBN"))
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("build/diagnostics/drr-load/worker.json"))
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--worker", choices=("self", "savedrestore", "fixedBN"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    args.experiment = args.experiment.resolve()
    args.output = args.output.resolve()
    if not args.experiment.is_dir():
        parser.error("--experiment must be an existing directory")
    try:
        args.output.relative_to(args.experiment)
    except ValueError:
        pass
    else:
        parser.error("--output must be outside the read-only experiment")
    return args


def _inputs(args):
    def unique(pattern):
        matches = sorted(args.experiment.rglob(pattern))
        if len(matches) != 1:
            raise ValueError(f"Expected one {pattern!r}; found {len(matches)}. Pass explicit paths.")
        return matches[0].resolve()

    measurements = args.measurement or [unique("*pwn1_9T*0p05sx20*RotIn49deg_TG*BG=-10_PIXIS.csv")]
    backgrounds = args.background or [unique(pattern) for pattern in (
        "*pwn1_9T*0p05sx20*RotIn94deg_BNbackground_PIXIS.csv",
        "*pwn1_9T*0p05sx20*RotIn49deg_BNbackground_PIXIS.csv",
        "*pwn1_0T*0p05sx20*RotIn49deg_BNbackground_PIXIS.csv")]
    def resolve(path):
        path = path if path.is_absolute() else args.experiment / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    return tuple(map(resolve, measurements)), tuple(map(resolve, backgrounds))


def _protect_experiment(root):
    protected = os.path.normcase(str(root.resolve()))
    attempted = []

    def inside(value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        path = os.path.normcase(os.path.abspath(os.fsdecode(value)))
        try:
            return os.path.commonpath((path, protected)) == protected
        except ValueError:
            return False

    def audit(event, values):
        candidates = ()
        if event == "open":
            path, mode, flags = values
            if (isinstance(mode, str) and any(flag in mode for flag in "wax+")) or (
                    isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                candidates = (path,)
        elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime", "os.truncate"}:
            candidates = values[:1]
        elif event in {"os.rename", "os.link", "os.symlink"}:
            candidates = values[:2]
        if any(inside(path) for path in candidates):
            attempted.append({"event": event, "paths": list(map(str, candidates))})
            raise PermissionError(f"Read-only benchmark rejected {event} inside {protected}")
    sys.addaudithook(audit)
    return attempted


class _Phases:
    def __init__(self):
        self.values = {}
        self.originals = []

    @contextmanager
    def phase(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            value = self.values.setdefault(name, {"seconds": 0., "calls": 0})
            value["seconds"] += time.perf_counter() - started
            value["calls"] += 1

    def wrap(self, module, name, label):
        original = getattr(module, name)
        @wraps(original)
        def measured(*args, **kwargs):
            with self.phase(label):
                return original(*args, **kwargs)
        setattr(module, name, measured)
        self.originals.append((module, name, original))


def _cube_identity(cube):
    import numpy as np
    if cube is None:
        return None
    arrays = {}
    for name in ("energy", "gate", "Z"):
        array = np.ascontiguousarray(getattr(cube, name), dtype=np.float64)
        arrays[name] = {"shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                        "nan": int(np.isnan(array).sum()), "finite": int(np.isfinite(array).sum())}
    return {"arrays": arrays, "gate_label": cube.gate_label, "cbar_label": cube.cbar_label,
            "numerical_path": getattr(cube, "drr_numerical_path", "")}


def _worker(args):
    measurements, backgrounds = _inputs(args)
    attempted_writes = _protect_experiment(args.experiment)
    # Import Qt before scientific packages so Windows DLL directories are set.
    imported = time.perf_counter()
    from PySide6 import QtCore  # noqa: F401
    from core import processing_run, loader, processing, drr_sources, drr_display_settings
    from core.load_timing import collect_load_timings
    from ui_qt.quick import drr_workflow, drr_background_workflow, drr_background_restore
    import_seconds = time.perf_counter() - imported
    phases = _Phases()
    for module, name, label in (
        (drr_workflow, "_input_signatures", "input_hash_verify"),
        (drr_workflow, "process_drr", "process_products"),
        (drr_background_workflow, "_signatures", "background_hash_verify"),
        (drr_background_workflow, "prepare_application", "prepare_background"),
        (drr_background_workflow, "load_with_recipe", "load_recipe"),
        (drr_background_restore, "restore_saved_background", "restore_background"),
        (drr_background_restore, "_read_drr_metadata", "source_history"),
        (drr_sources, "_read_drr_metadata", "source_history"),
        (drr_display_settings, "find_candidates", "display_history"),
        (drr_display_settings, "_legacy_catalog", "display_legacy_catalog"),
        (loader, "load_drr_avg", "load_common"),
        (loader, "load_drr_resolved_avg", "load_resolved"),
        (loader, "build_external_baseline", "baseline"),
        (processing_run, "_load_canonical", "canonical_read"),
        (processing_run, "_parse_raw_spectrum", "csv_parse"),
        (processing_run, "process_ref_avg", "ratio_average"),
        (processing, "apply_sg_derivative_energy", "derivative"),
    ):
        phases.wrap(module, name, label)
    request = drr_workflow.DrrRequest(files=measurements, experiment_folder=str(args.experiment))
    records = []
    profile_summary = None
    def run(label, profile=None):
        phases.values = {}
        started = time.perf_counter()
        prepared = None
        captured = request
        if profile is not None:
            profile.enable()
        error = None
        products = None
        try:
            with collect_load_timings() as collected:
                if args.worker == "savedrestore":
                    captured = drr_background_restore.restore_saved_background(request)
                elif args.worker == "fixedBN":
                    context = drr_workflow.background_context(request)
                    prepared = drr_background_workflow.prepare_application(context,
                        drr_background_workflow.BackgroundDraft(kind="bn", files=backgrounds, which="all", fixed=True))
                    captured = replace(request, background_mode="external", background_frame="all",
                                       backgrounds=backgrounds, background_recipe=prepared.recipe)
                with phases.phase("load_total"):
                    products = drr_workflow.load_drr(captured, prepared_background=prepared)
        except Exception:
            collected = locals().get("collected", {})
            error = traceback.format_exc()
        finally:
            if profile is not None:
                profile.disable()
        elapsed = time.perf_counter() - started
        record = {"sample": label, "elapsed_seconds": elapsed, "phase_times_inclusive": phases.values,
                  "core_load_timings_inclusive": collected, "error": error,
                  "background_mode": captured.background_mode, "background_frame": captured.background_frame,
                  "backgrounds": list(captured.backgrounds), "recipe_present": captured.background_recipe is not None}
        # Output checksums are outside the measured load, and cover every cell.
        if products is not None:
            record.update(source=_cube_identity(products.source), raw=_cube_identity(products.raw),
                          second=_cube_identity(products.second), second_error=products.second_error,
                          input_signatures=products.input_signatures)
        concise = ", ".join(f"{name}={value['seconds']:.3f}s/{value['calls']}" for name, value in phases.values.items()
                            if name in {"csv_parse", "canonical_read", "restore_background", "source_history", "display_history",
                                        "input_hash_verify", "background_hash_verify", "prepare_background", "derivative"})
        print(f"{args.worker} {label}: {elapsed:.3f}s; {concise}" + ("; FAILED" if error else ""), flush=True)
        return record
    for index in range(args.samples):
        records.append(run("cold" if index == 0 else f"warm{index}"))
    if not args.no_profile:
        profiler = cProfile.Profile()
        profile_summary = run("profiled_repeat", profiler)
        profile_path = args.output.with_suffix(".prof")
        profiler.dump_stats(str(profile_path))
        with args.output.with_suffix(".profile.txt").open("w", encoding="utf-8") as stream:
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(70)
    identities = [{key: record.get(key) for key in ("source", "raw", "second")} for record in records if not record["error"]]
    result = {"mode": args.worker, "scope": "worker request preparation, exact history restoration, loading and derivatives; excludes GUI/plot",
              "fixedBN_scope": "Each fixedBN sample includes draft preparation and trial loading; no association is committed.",
              "cold_definition": "fresh application caches; OS disk cache not flushed", "import_seconds": import_seconds,
              "measurement_files": list(measurements), "fixed_background_files": list(backgrounds), "samples": records,
              "profiled_repeat": profile_summary, "cold_warm_identical": bool(identities) and all(item == identities[0] for item in identities),
              "attempted_experiment_writes": attempted_writes}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return int(bool(attempted_writes) or any(item["error"] for item in records))


def main():
    args = _arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.worker:
        return _worker(args)
    measurements, backgrounds = _inputs(args)
    results = []
    with tempfile.TemporaryDirectory(prefix="dptk-drr-load-profile-") as temporary:
        for mode in args.modes:
            output = args.output.with_name(f"{args.output.stem}-{mode}.json")
            environment = os.environ.copy()
            isolated = Path(temporary) / mode
            isolated.mkdir()
            for name in ("LOCALAPPDATA", "APPDATA", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "MPLCONFIGDIR", "DPTK_FITTED_CACHE_DIR"):
                path = isolated / name.lower()
                path.mkdir()
                environment[name] = str(path)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            command = [sys.executable, str(Path(__file__).resolve()), "--experiment", str(args.experiment),
                       "--worker", mode, "--samples", str(args.samples), "--output", str(output)]
            for name in measurements:
                command.extend(("--measurement", name))
            for name in backgrounds:
                command.extend(("--background", name))
            if args.no_profile:
                command.append("--no-profile")
            completed = subprocess.run(command, env=environment, check=False)
            results.append({"mode": mode, "returncode": completed.returncode, "result": str(output)})
    args.output.write_text(json.dumps({"runs": results}, indent=2), encoding="utf-8")
    return int(any(item["returncode"] for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
