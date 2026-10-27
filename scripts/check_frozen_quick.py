"""Exercise a built DPTK executable using isolated native Quick smoke cases.

Run ``python scripts/check_frozen_quick.py dist/DPTK/DPTK.exe --report ...``.
Use ``--source`` (and optionally a Python interpreter as the positional target)
to exercise the same public run_qt.py dispatch before a build. Fixtures and all
settings/session/cache directories are temporary; reports and captures remain
beside the requested report. This script never builds or installs anything.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


WORKFLOWS = ("pl", "drr", "cmp", "power", "mcd", "mcd_peak_shift", "shg", "slides", "tools")
FAILURE_MARKERS = ("Traceback (most recent call last)", "ModuleNotFoundError", "ImportError:", "DLL load failed",
    "QQmlApplicationEngine failed", "failed to load component", "is not a type", "is not installed",
    "Cannot load library", "ReferenceError", "TypeError:", "Binding loop", "Cannot assign", "Unable to assign")


@dataclass(frozen=True)
class SmokeCase:
    name: str
    arguments: tuple[str, ...]
    kind: str = "main"
    workflow: str = "pl"
    expect_plot: bool = False
    expect_background: bool = False


def isolated_environment(root):
    root = Path(root)
    paths = dict(LOCALAPPDATA=root / "local", APPDATA=root / "roaming", MPLCONFIGDIR=root / "matplotlib",
        XDG_CONFIG_HOME=root / "config", XDG_CACHE_HOME=root / "cache", XDG_DATA_HOME=root / "data",
        TEMP=root / "temp", TMP=root / "temp")
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return dict(os.environ, **{name: str(path) for name, path in paths.items()}, QT_QPA_PLATFORM="offscreen",
        QT_QUICK_BACKEND="software", QT_QUICK_CONTROLS_STYLE="Basic", MPLBACKEND="agg", PYTHONIOENCODING="utf-8")


@contextmanager
def fixture_environment(environment):
    """Keep fixture-generation imports from touching the caller's caches."""
    keys = ("LOCALAPPDATA", "APPDATA", "MPLCONFIGDIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "MPLBACKEND")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update({key: environment[key] for key in keys})
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def create_fixtures(root):
    """Create real DAT/CSV/NPZ fixtures using the supported scientific schemas."""
    import numpy as np
    from core.loader import DataCube
    from core.peak_workspace import create_dataset, save_workspace
    from core.drr_analysis_workspace import create_dataset as create_drr_dataset, analyze_dataset as analyze_drr
    from core.drr_workspace_session import save_session
    from ui_qt.quick.peak_analysis_workflow import analyze_dataset
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    energy, gate = np.linspace(1.60, 1.80, 151), np.linspace(-2., 2., 9)
    peak = 1. / (1. + (2. * (energy[None, :] - 1.65 - gate[:, None] * .001) / .008) ** 2)
    second = .6 / (1. + (2. * (energy[None, :] - 1.75 + gate[:, None] * .001) / .012) ** 2)
    pl = DataCube(energy, gate, .1 + peak + second, "Gate (V)", "Synthetic PL", "Intensity", "V", "gate_voltage")
    drr = DataCube(energy, gate, .025 * (peak - second), "Gate", "Synthetic DRR", "ΔR/R", "V", "gate_voltage")
    output = {}
    for key, cube in (("pl", pl), ("drr", drr)):
        path = root / (key + ".dat")
        header = "Energy (eV)\t" + "\t".join(format(float(value), ".12g") for value in gate)
        np.savetxt(path, np.column_stack((energy, cube.Z.T)), delimiter="\t", header=header, comments="", fmt="%.12g")
        path.with_suffix(".metadata.json").write_text(json.dumps(dict(operation="PL" if key == "pl" else "DRR",
            y_axis_label=cube.gate_label, y_axis_unit=cube.gate_unit, y_axis_semantic=cube.y_axis_semantic,
            plot=dict(title=cube.title, cbar_label=cube.cbar_label))), encoding="utf-8")
        output[key] = path
    compare = root / "compare"
    compare.mkdir()
    channels = []
    for index, channel in enumerate(("KK", "KKp")):
        path = compare / ("synthetic_PL_" + channel + ".csv")
        intensity = 100. + index * 12. + (gate[:, None] + 2.) * 7. + 80. * peak
        table = np.column_stack((gate, gate * .3 + 1., np.zeros_like(gate), intensity))
        header = "Vbg,Vtg,Vbias," + ",".join(format(1240. / value, ".12g") for value in energy)
        np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.12g")
        channels.append(path)
    output["compare"] = tuple(channels)
    output["drr_csv"] = root / "synthetic_REF_730nmc.csv"
    output["drr_csv"].write_bytes(channels[0].read_bytes())
    dataset = create_dataset(pl, "PL", "PL", str(output["pl"]), "Synthetic fitted PL")
    dataset["settings"].update(min_snr=0., neighbor_support=0)
    dataset = analyze_dataset(analyze_dataset(dataset, "detect"), "fit")
    dataset["show_residual"] = True
    output["peak_snapshot"] = root / "peak-snapshot.npz"
    save_workspace(output["peak_snapshot"], [dataset], dataset["key"])
    analysis = create_drr_dataset(drr, "Synthetic DRR Analysis", dict(measurement_files=[str(output["drr"])],
        background_files=[], background_mode="Imported map", background_which="all", y_axis="auto"))
    analysis.result = analyze_drr(analysis)
    output["drr_snapshot"] = root / "drr-snapshot.npz"
    save_session(output["drr_snapshot"], [analysis], active_key=analysis.key)
    for key in ("drr_analysis_folder", "organizer_folder"):
        output[key] = root / key
        output[key].mkdir()
    output["shape"] = dict(rows=len(gate), columns=len(energy), peak_fits=len(dataset["result"]["fits"]))
    return output


def build_cases(fixtures):
    cases = [SmokeCase("default-check", ()), SmokeCase("demo-check", ("--demo",), expect_plot=True)]
    cases.extend(SmokeCase("empty-" + workflow, ("--workflow", workflow), workflow=workflow) for workflow in WORKFLOWS)
    cases.extend((SmokeCase("synthetic-pl-dat", ("--workflow", "pl", "--open", str(fixtures["pl"])), expect_plot=True),
        SmokeCase("synthetic-drr-dat", ("--workflow", "drr", "--open", str(fixtures["drr"])), workflow="drr", expect_plot=True),
        SmokeCase("synthetic-drr-csv-background", ("--workflow", "drr", "--open", str(fixtures["drr_csv"])),
            workflow="drr", expect_background=True),
        SmokeCase("synthetic-compare-csv", ("--workflow", "cmp", "--open", str(fixtures["compare"][0]),
            "--open", str(fixtures["compare"][1])), workflow="cmp", expect_plot=True),
        SmokeCase("standalone-peak", ("--peak-analysis-quick", "--snapshot", str(fixtures["peak_snapshot"])), kind="peak", expect_plot=True),
        SmokeCase("standalone-drr", ("--drr-analysis-quick", str(fixtures["drr_analysis_folder"]),
            "--snapshot", str(fixtures["drr_snapshot"])), kind="drr", expect_plot=True),
        SmokeCase("standalone-mcd-organizer", ("--mcd-organizer-quick", str(fixtures["organizer_folder"])), kind="organizer")))
    return cases


def png_evidence(path):
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(24)
        stream.seek(-12, 2)
        ending = stream.read(12)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR" or ending[4:8] != b"IEND":
        raise ValueError("The window capture is not a complete PNG")
    width, height = struct.unpack(">II", header[16:24])
    if not width or not height:
        raise ValueError("The captured frame has no pixels")
    return dict(path=str(path), width=width, height=height, bytes=path.stat().st_size)


def _text(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value or "")


def _messages(value):
    if not value:
        return []
    return [_text(item) for item in value] if isinstance(value, (list, tuple)) else [_text(value)]


def inspect_report(report, case):
    if not isinstance(report, dict):
        return ["The application report must be a JSON object"]
    errors = []
    if case.kind == "main":
        if report.get("status") != "ok":
            errors.append("The native shell report did not declare status=ok")
        if report.get("widgets_imported") is not False:
            errors.append("The native shell report did not confirm widgets_imported=false")
        if not report.get("settings_isolated"):
            errors.append("The --check shell did not confirm isolated settings")
        if report.get("workflow") != case.workflow:
            errors.append("The rendered workflow differs from the requested workflow")
        frames = report.get("visible_frames", 0)
        if not isinstance(frames, (int, float)) or frames <= 0:
            errors.append("The native shell did not present a Quick frame")
        errors.extend(_messages(report.get("errors") or report.get("Errors")))
        timings = report.get("timing_ms") or {}
        plot_frame = timings.get("first_plot_frame_ms", 0) if isinstance(timings, dict) else 0
        if case.expect_plot and (not isinstance(plot_frame, (int, float)) or plot_frame <= 0):
            errors.append("The scientific Agg plot did not reach a presented Quick frame")
        if case.expect_background and (report.get("background_required") is not True
                or report.get("background_dialog_open") is not True or report.get("can_export") is not False
                or not report.get("background_visible_frames", 0) or plot_frame):
            errors.append("The DRR CSV must open background settings without a provisional plot or export")
    elif case.kind == "drr":
        if report.get("widgetsImported") is not False:
            errors.append("The DRR report did not confirm widgetsImported=false")
        if report.get("application") != "QGuiApplication" or report.get("qmlRoots", 0) < 1:
            errors.append("The DRR report did not confirm a QGuiApplication and loaded QML")
        errors.extend(_messages(report.get("warnings")))
        if report.get("error"):
            errors.append(_text(report["error"]))
        if report.get("datasets", 0) < 1 or report.get("renderedFrames", 0) < 1:
            errors.append("The synthetic DRR workspace did not produce an Agg plot")
    return errors


def run_case(prefix, case, temporary, artifacts, timeout):
    case_root, case_artifacts = Path(temporary) / case.name, Path(artifacts) / case.name
    case_root.mkdir(parents=True)
    case_artifacts.mkdir(parents=True, exist_ok=True)
    environment = isolated_environment(case_root / "environment")
    report_path, capture_path = case_artifacts / "application-report.json", case_artifacts / "window.png"
    # A failed rerun must never reuse an earlier successful application's
    # report or capture as current evidence. Only these generated files clear.
    report_path.unlink(missing_ok=True)
    capture_path.unlink(missing_ok=True)
    arguments = list(case.arguments) + ["--check"]
    session = case_root / "workspace.npz"
    if case.kind in {"peak", "drr"}:
        arguments.extend(("--session", str(session)))
    if case.kind in {"peak", "organizer"}:
        arguments.extend(("--settings", str(case_root / "isolated.ini")))
    elif case.kind == "drr":
        arguments.extend(("--settings-file", str(case_root / "isolated.ini")))
    if case.kind in {"main", "drr"}:
        arguments.extend(("--report", str(report_path)))
    if case.kind != "drr":
        arguments.extend(("--capture", str(capture_path)))
    command = [*prefix, *arguments]
    evidence = dict(name=case.name, kind=case.kind, arguments=arguments, status="error", errors=[],
        settings_scope="Explicit temporary INI/session or the entrypoint's isolated --check profile")
    started = perf_counter()
    try:
        process = subprocess.run(command, cwd=case_root, env=environment, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        stdout, stderr = process.stdout, process.stderr
        evidence["returncode"] = process.returncode
        if process.returncode != 0:
            evidence["errors"].append("The process exited with code " + str(process.returncode))
    except subprocess.TimeoutExpired as error:
        stdout, stderr = _text(error.stdout), _text(error.stderr)
        evidence["errors"].append(f"The process exceeded the {timeout:g}-second timeout")
    except OSError as error:
        stdout, stderr = "", str(error)
        evidence["errors"].append("The target process could not start: " + str(error))
    evidence["elapsed_seconds"] = round(perf_counter() - started, 3)
    diagnostic_logs = []
    for log in sorted((case_root / "environment" / "local" / "DPTK" / "logs").glob("*.log")):
        text = log.read_text(encoding="utf-8", errors="replace")
        saved = case_artifacts / log.name
        saved.write_text(text, encoding="utf-8")
        diagnostic_logs.append(text)
    (case_artifacts / "stdout.txt").write_text(_text(stdout), encoding="utf-8")
    (case_artifacts / "stderr.txt").write_text(_text(stderr), encoding="utf-8")
    logs = "\n".join((_text(stdout), _text(stderr), *diagnostic_logs))
    evidence["failure_markers"] = [marker for marker in FAILURE_MARKERS if marker.casefold() in logs.casefold()]
    if evidence["failure_markers"]:
        evidence["errors"].append("Import/QML/runtime failures were found in the captured diagnostic logs")
    if case.kind in {"main", "drr"}:
        try:
            application_report = json.loads(report_path.read_text(encoding="utf-8"))
            evidence["application_report"] = application_report
            evidence["errors"].extend(inspect_report(application_report, case))
        except (OSError, ValueError, TypeError) as error:
            evidence["errors"].append("The application report could not be read: " + str(error))
    if case.kind != "drr":
        try:
            evidence["capture"] = png_evidence(capture_path)
        except (OSError, ValueError) as error:
            evidence["errors"].append("No valid captured window frame: " + str(error))
    if case.kind == "peak":
        try:
            import numpy as np
            with np.load(session, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                datasets = metadata["datasets"]
                shapes = [list(archive[f"{index}_Z"].shape) for index in range(len(datasets))]
            if not datasets:
                raise ValueError("The fitted synthetic snapshot was not restored")
            evidence["restored_workspace"] = dict(datasets=len(datasets), shapes=shapes)
        except (OSError, ValueError, KeyError) as error:
            evidence["errors"].append("The Peak workspace was not saved after restoring its synthetic snapshot: " + str(error))
    evidence["scope"] = ({"main": "QML page, isolated settings, no Widgets import, captured window; real data cases additionally require a presented scientific plot.",
        "drr": "Synthetic DRR workspace loaded, QGuiApplication/QML and at least one Agg scientific render; no Widgets import, warnings or reported errors.",
        "peak": "Fitted PL snapshot restored and saved, completed --check and captured window; no missing-import/runtime markers. Peak has no explicit Widgets/frame telemetry report.",
        "organizer": "Empty Organizer QML window completed --check and captured a frame; no missing-import/runtime markers. This case does not exercise MCD scientific analysis or expose Widgets telemetry."})[case.kind]
    evidence["artifacts"] = str(case_artifacts)
    evidence["status"] = "ok" if not evidence["errors"] else "error"
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, nargs="?", help="Built EXE; with --source, an optional Python interpreter")
    parser.add_argument("--source", action="store_true", help="Validate the source public dispatcher instead of a frozen EXE")
    parser.add_argument("--report", type=Path, default=Path("build/frozen-quick-report.json"))
    parser.add_argument("--timeout", type=float, default=90., help="Per-process deadline, default90seconds")
    parser.add_argument("--case", action="append", dest="selected_cases", help="Run only this named case; repeat to select several")
    args = parser.parse_args(argv)
    if not args.source and args.executable is None:
        parser.error("Provide a built executable, or use --source")
    if args.timeout <= 0:
        parser.error("The timeout must be positive")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    target = (args.executable or Path(sys.executable)).resolve()
    if not target.is_file():
        parser.error("The target executable does not exist: " + str(target))
    report_path = args.report.resolve()
    artifacts = report_path.parent / (report_path.stem + "-artifacts")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    prefix = [str(target), str(root / "run_qt.py")] if args.source else [str(target)]
    started = perf_counter()
    results = []
    with TemporaryDirectory(prefix="dptk-frozen-quick-check-") as directory:
        temporary = Path(directory)
        try:
            with fixture_environment(isolated_environment(temporary / "fixture-environment")):
                fixtures = create_fixtures(temporary / "fixtures")
            cases = build_cases(fixtures)
            selected = set(args.selected_cases or ())
            unknown = selected - {case.name for case in cases}
            if unknown:
                parser.error("Unknown case: " + ", ".join(sorted(unknown)))
            for case in cases:
                if selected and case.name not in selected:
                    continue
                result = run_case(prefix, case, temporary, artifacts, args.timeout)
                results.append(result)
                print(f"{result['status'].upper()}: {case.name} ({result['elapsed_seconds']:.2f}s)", flush=True)
        except Exception as error:
            results.append(dict(name="fixture-generation", status="error", errors=[str(error)]))
    failed = sum(result["status"] != "ok" for result in results)
    report = dict(status="ok" if not failed and results else "error", mode="source" if args.source else "frozen",
        target=str(target), elapsed_seconds=round(perf_counter() - started, 3), passed=len(results) - failed, failed=failed,
        case_count=len(results), cases=results, artifacts=str(artifacts),
        isolation="All measurement fixtures, INI profiles, sessions, LOCALAPPDATA/APPDATA and scientific caches were temporary; no user data/settings or installers were used.",
        limits=["Offscreen software-renderer smoke; this does not replace native GPU/high-DPI visual QA.",
            "No automated download, install, real instrument connection or external Office launch is exercised.",
            "Peak/Organizer lack explicit import/frame reports; their stated evidence is limited to completed check, captures/logs, and Peak restored-workspace inspection."])
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{report['status'].upper()}: {report['passed']}/{report['case_count']} cases; report: {report_path}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
