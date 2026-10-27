"""Plan, run and verify complete, isolated unittest release shards.

The plan is created once with runtime dependencies installed. Each shard runs
its modules in fresh processes (expensive BN modules run one test per process).
Verification needs only the standard library and fails closed on missing tests,
timeouts, crashes, stale sources, or missing normal-completion reports.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
ESTIMATED_PROCESS_STARTUP_SECONDS = 2.0
TIMING_HINTS_PATH = "scripts/release_test_timings.json"
ACCEPTABLE = {"success", "skip", "expected_failure"}
SPLIT_MODULES = {
    "test_drr_bn_calibration", "test_drr_bn_joint",
    "test_drr_bn_joint_independent", "test_drr_bn_shrinkage",
    "test_drr_bn_shrinkage_independent", "test_drr_bn_d2_priority_independent",
    "test_drr_bn_derivative_protection", "test_drr_bn_d2_objective",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_snapshot(repo, tests_dir):
    directory = (repo / tests_dir).resolve()
    require(directory.is_relative_to(repo), "tests-dir must be inside repo")
    require(directory.is_dir(), f"Missing tests directory: {directory}")
    return {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*.py"))}


def flatten(suite):
    require(type(suite) is unittest.TestSuite,
            "Custom TestSuite behavior cannot be preserved by isolated release tests")
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def prepare_runtime():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # The Windows QtGui DLL directory must be registered before NumPy/SciPy.
    # Minimal fixture repositories may intentionally have no Qt dependency.
    if sys.platform == "win32" and importlib.util.find_spec("PySide6") is not None:
        importlib.import_module("PySide6.QtGui")


class ReleaseTestLoader(unittest.TestLoader):
    def loadTestsFromModule(self, module, *, pattern=None):
        require(getattr(module, "load_tests", None) is None,
                f"Unsupported load_tests hook in {module.__name__}: "
                "isolated release tests require standard unittest discovery")
        return super().loadTestsFromModule(module, pattern=pattern)


def require_reconstructible(test):
    cls = type(test)
    require(isinstance(test, unittest.TestCase)
            and cls.__init__ is unittest.TestCase.__init__
            and cls.id is unittest.TestCase.id and cls.run is unittest.TestCase.run,
            f"Custom TestCase construction/id/run is unsupported: {cls}")
    require(test.id() == f"{cls.__module__}.{cls.__qualname__}.{test._testMethodName}",
            "Test ID cannot reconstruct its discovered class")
    fresh = cls(test._testMethodName)
    require(test.__dict__ == fresh.__dict__,
            f"Custom discovered instance state cannot be reconstructed: {test.id()}")


def discover_tests(repo, tests_dir):
    prepare_runtime()
    sys.path.insert(0, str(repo))
    loader = ReleaseTestLoader()
    suite = loader.discover(str(repo / tests_dir), pattern="test*.py", top_level_dir=str(repo))
    require(not loader.errors, "Test discovery failed:\n" + "\n".join(loader.errors))
    unique, counts = {}, Counter()
    for test in flatten(suite):
        require_reconstructible(test)
        test_id = test.id()
        require(isinstance(test_id, str) and test_id, "Discovery returned an invalid test ID")
        if test_id in unique:
            require(type(unique[test_id]) is type(test)
                    and unique[test_id]._testMethodName == test._testMethodName,
                    f"Ambiguous duplicate test ID: {test_id}")
        else:
            unique[test_id] = test
        counts[test_id] += 1
    require(unique, "Discovery returned no tests")
    return unique, {name: count for name, count in sorted(counts.items()) if count > 1}


def weight(test_id, module):
    short_module = module.rsplit(".", 1)[-1]
    if short_module in SPLIT_MODULES:
        return 60.0, "estimated_expensive_bn_test"
    return 1.0, "estimated_test_count"


def timing_source_digest(path):
    # Scheduling hints must survive Git's LF/CRLF checkout conversion. The
    # execution plan's independent, byte-exact source validation is unchanged.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def read_timing_hints(repo):
    path = repo / TIMING_HINTS_PATH
    if not path.exists():
        return {}, None

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate timing hint key: {key}")
            result[key] = value
        return result

    def hex_digest(value, length=64):
        return (isinstance(value, str) and len(value) == length
                and all(char in "0123456789abcdef" for char in value))

    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=unique_object)
    require(isinstance(data, dict) and set(data) == {"schema_version", "source", "units"}
            and type(data["schema_version"]) is int and data["schema_version"] == 1,
            "Invalid timing hints schema")
    source = data["source"]
    require(isinstance(source, dict)
            and type(source.get("workflow_run_id")) is int and source["workflow_run_id"] > 0
            and hex_digest(source.get("commit"), 40)
            and hex_digest(source.get("plan_id"))
            and isinstance(source.get("recorded_at_utc"), str)
            and source["recorded_at_utc"].endswith("Z")
            and source.get("measurement") == "unit_wall_seconds_including_startup",
            "Invalid timing hints provenance")
    datetime.strptime(source["recorded_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
    hints = data["units"]
    require(isinstance(hints, dict), "Invalid timing hints units")
    for key, hint in hints.items():
        require(isinstance(key, str) and "." in key
                and all(part.isidentifier() for part in key.split(".")),
                f"Invalid stable timing key: {key}")
        require(isinstance(hint, dict)
                and set(hint) == {"duration_seconds", "source_sha256_lf", "test_ids_sha256"},
                f"Invalid timing hint: {key}")
        seconds = hint["duration_seconds"]
        require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 86400,
                f"Invalid timing duration: {key}")
        require(hex_digest(hint["source_sha256_lf"]) and hex_digest(hint["test_ids_sha256"]),
                f"Invalid timing fingerprint: {key}")
    return hints, {"path": TIMING_HINTS_PATH, "sha256": hashlib.sha256(raw).hexdigest(),
                   "source": source}


def make_plan(repo, tests_dir="tests", shards=4):
    repo = Path(repo).resolve()
    require(shards > 0, "shards must be positive")
    tests_dir = (repo / tests_dir).resolve().relative_to(repo).as_posix()
    before = source_snapshot(repo, tests_dir)
    timing_hints, timing_provenance = read_timing_hints(repo)
    discovered, duplicates = discover_tests(repo, tests_dir)
    tests, groups = [], defaultdict(list)
    for test_id, case in sorted(discovered.items()):
        module = type(case).__module__
        seconds, basis = weight(test_id, module)
        tests.append({"id": test_id, "module": module,
                      "weight_seconds": seconds, "weight_basis": basis})
        key = test_id if module.rsplit(".", 1)[-1] in SPLIT_MODULES else module
        groups[key].append(test_id)
    weights = {test["id"]: test["weight_seconds"] for test in tests}
    modules = {test["id"]: test["module"] for test in tests}
    units, timing_sources = [], {}
    for index, (key, ids) in enumerate(sorted(groups.items())):
        seconds = ESTIMATED_PROCESS_STARTUP_SECONDS + sum(weights[name] for name in ids)
        basis = "estimated_tests_plus_startup"
        hint = timing_hints.get(key)
        if hint is not None:
            module_path = modules[ids[0]].replace(".", "/") + ".py"
            if module_path not in timing_sources:
                timing_sources[module_path] = (timing_source_digest(repo / module_path)
                                               if module_path in before else None)
            if (hint["source_sha256_lf"] == timing_sources[module_path]
                    and hint["test_ids_sha256"] == digest(sorted(ids))):
                # The measured child wall time already includes process startup.
                seconds, basis = hint["duration_seconds"], "measured_unit_wall"
            else:
                # Changed tests no longer have a current measurement, but a
                # known expensive unit must not collapse to a count estimate.
                # This only estimates placement; execution validates the new
                # plan's byte-exact sources and complete test IDs independently.
                seconds = max(seconds, hint["duration_seconds"])
                basis = "estimated_previous_unit_wall"
        units.append({"id": f"unit-{index:04d}", "test_ids": ids, "timing_key": key,
                      "weight_seconds": seconds, "weight_basis": basis})
    assignments = [{"id": index, "unit_ids": [], "test_ids": [], "weight_seconds": 0.0}
                   for index in range(shards)]
    for unit in sorted(units, key=lambda item: (-item["weight_seconds"], item["id"])):
        shard = min(assignments, key=lambda item: (item["weight_seconds"], item["id"]))
        shard["unit_ids"].append(unit["id"])
        shard["test_ids"].extend(unit["test_ids"])
        shard["weight_seconds"] += unit["weight_seconds"]
    after = source_snapshot(repo, tests_dir)
    require(before == after, "Test sources changed during discovery")
    plan = {"schema_version": SCHEMA_VERSION, "tests_dir": tests_dir,
            "sources": before, "source_fingerprint": digest(before),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "process_startup_weight_seconds": ESTIMATED_PROCESS_STARTUP_SECONDS,
            "process_startup_weight_basis": "estimated_per_process",
            "timing_hints": timing_provenance,
            "discovery_duplicates": duplicates, "tests": tests, "units": units,
            "shards": assignments, "shard_count": shards}
    plan["plan_id"] = digest(plan)
    validate_plan(plan, repo)
    return plan


def exact_ids(actual, expected, label):
    require(isinstance(actual, list) and all(isinstance(name, str) for name in actual),
            f"Invalid {label} IDs")
    require(len(actual) == len(set(actual)), f"Duplicate {label} IDs")
    require(set(actual) == set(expected), f"Missing or unexpected {label} IDs")


def validate_plan(plan, repo):
    require(plan.get("schema_version") == SCHEMA_VERSION, "Unsupported plan schema")
    require(plan.get("plan_id") == digest({k: v for k, v in plan.items() if k != "plan_id"}),
            "Plan digest mismatch")
    require(plan["source_fingerprint"] == digest(plan["sources"]), "Source digest mismatch")
    require(plan["sources"] == source_snapshot(repo, plan["tests_dir"]),
            "Test sources differ from the plan; regenerate the plan")
    require(plan["runner_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "Release runner differs from the plan")
    test_ids = [test["id"] for test in plan["tests"]]
    require(test_ids, "Plan contains no tests")
    exact_ids(test_ids, test_ids, "plan test")
    units = {unit["id"]: unit for unit in plan["units"]}
    require(len(units) == len(plan["units"]), "Duplicate execution unit")
    exact_ids([name for unit in units.values() for name in unit["test_ids"]], test_ids, "unit test")
    require(plan["shard_count"] > 0, "Invalid shard count")
    require([s["id"] for s in plan["shards"]] == list(range(plan["shard_count"])),
            "Missing or duplicate shard assignments")
    exact_ids([name for shard in plan["shards"] for name in shard["unit_ids"]],
              list(units), "assigned unit")
    for shard in plan["shards"]:
        exact_ids(shard["test_ids"],
                  [name for uid in shard["unit_ids"] for name in units[uid]["test_ids"]],
                  "assigned test")


class RecordingResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity, events):
        super().__init__(stream, descriptions, verbosity)
        self.events = events
        self.records = []
        self.active = {}
        self.fixture_skips = []

    def emit(self, event):
        self.events.write(json.dumps(event, sort_keys=True) + "\n")
        self.events.flush()  # Keep crash evidence without fsync per test.

    def startTest(self, test):
        super().startTest(test)
        self.active[test.id()] = {"id": test.id(), "status": "success",
                                  "started": time.monotonic()}
        self.emit({"event": "test_start", "id": test.id()})

    def mark(self, test, status, detail=None):
        record = self.active.get(test.id())
        if record is not None and record["status"] in ACCEPTABLE:
            record["status"] = status
            if detail:
                record["detail"] = str(detail)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.mark(test, "failure", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self.mark(test, "error", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        parent = getattr(test, "test_case", test)
        if parent.id() in self.active:
            self.mark(parent, "skip", reason)
        else:
            self.fixture_skips.append({"id": test.id(), "reason": str(reason)})

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self.mark(test, "expected_failure")

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.mark(test, "unexpected_success")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            status = "failure" if issubclass(err[0], test.failureException) else "error"
            self.mark(test, status, self._exc_info_to_string(err, subtest))

    def stopTest(self, test):
        record = self.active.pop(test.id())
        record["duration_seconds"] = time.monotonic() - record.pop("started")
        self.records.append(record)
        self.emit({"event": "test_end", **record})
        super().stopTest(test)


def run_child(plan_path, unit_id, repo, report_path, events_path):
    plan = read_json(plan_path)
    validate_plan(plan, repo)
    unit = next(unit for unit in plan["units"] if unit["id"] == unit_id)
    prepare_runtime()
    sys.path.insert(0, str(repo))
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(unit["test_ids"])
    require(not loader.errors, "Test loading failed:\n" + "\n".join(loader.errors))
    exact_ids([test.id() for test in flatten(suite)], unit["test_ids"], "loaded test")
    with Path(events_path).open("w", encoding="utf-8") as events:
        runner = unittest.TextTestRunner(
            verbosity=2, resultclass=lambda *args: RecordingResult(*args, events=events))
        result = runner.run(suite)
        # unittest does not call start/stopTest for class/module fixture skips.
        # Expand only explicit setUp skips to the affected planned IDs.
        seen = {record["id"] for record in result.records}
        for skip in result.fixture_skips:
            fixture_id = skip["id"]
            if fixture_id.startswith(("setUpClass (", "setUpModule (")) and fixture_id.endswith(")"):
                prefix = fixture_id.split(" (", 1)[1][:-1] + "."
                for test_id in unit["test_ids"]:
                    if test_id.startswith(prefix) and test_id not in seen:
                        record = {"id": test_id, "status": "skip", "duration_seconds": 0.0,
                                  "detail": skip["reason"], "fixture_skip": fixture_id}
                        result.records.append(record)
                        result.emit({"event": "test_end", **record})
                        seen.add(test_id)
        report = {"schema_version": SCHEMA_VERSION, "plan_id": plan["plan_id"],
                  "unit_id": unit_id, "normal_completion": True,
                  "tests_run": result.testsRun, "successful": result.wasSuccessful(),
                  "records": result.records,
                  "fixture_skips": result.fixture_skips,
                  "failures": len(result.failures), "errors": len(result.errors),
                  "unexpected_successes": len(result.unexpectedSuccesses)}
        write_json(report_path, report)
    return 0 if result.wasSuccessful() else 1


def check_report(report, plan, unit):
    require(report.get("schema_version") == SCHEMA_VERSION, "Unsupported result schema")
    require(report.get("plan_id") == plan["plan_id"], "Result belongs to a different plan")
    require(report.get("unit_id") == unit["id"], "Wrong unit report")
    require(report.get("normal_completion") is True, "Child did not finish normally")
    records = report["records"]
    exact_ids([record["id"] for record in records], unit["test_ids"], "executed test")
    require(report["tests_run"] == sum("fixture_skip" not in record for record in records),
            "unittest execution count differs from recorded tests")
    skips = {skip["id"]: skip["reason"] for skip in report.get("fixture_skips", [])}
    for record in records:
        require(record.get("status") in ACCEPTABLE, f"Test did not pass: {record['id']}")
        duration = record.get("duration_seconds")
        require(isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= 0,
                f"Invalid test duration: {record['id']}")
        if "fixture_skip" in record:
            fixture = record["fixture_skip"]
            require(record["status"] == "skip" and fixture in skips
                    and fixture.startswith(("setUpClass (", "setUpModule ("))
                    and fixture.endswith(")")
                    and record["id"].startswith(fixture.split(" (", 1)[1][:-1] + "."),
                    "Invalid fixture skip evidence")
    require(report.get("successful") is True and report.get("failures") == 0
            and report.get("errors") == 0 and report.get("unexpected_successes") == 0,
            "unittest did not report success")


def terminate_process_tree(process):
    """Kill only the process tree/session started for this execution unit."""
    if os.name == "nt":
        taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
        subprocess.run([str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=15, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=15)


def run_shard(plan_path, shard_id, repo, output_dir, timeout=900):
    plan = read_json(plan_path)
    validate_plan(plan, repo)
    require(0 <= shard_id < plan["shard_count"], "Invalid shard index")
    require(timeout > 0 and math.isfinite(timeout), "timeout must be positive and finite")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    require(not any(output_dir.iterdir()), "Output directory must be empty to avoid stale results")
    shard = plan["shards"][shard_id]
    units = {unit["id"]: unit for unit in plan["units"]}
    summary = {"schema_version": SCHEMA_VERSION, "plan_id": plan["plan_id"],
               "shard": shard_id, "test_ids": shard["test_ids"], "units": [],
               "complete": False, "successful": False}
    started = time.monotonic()
    for unit_id in shard["unit_ids"]:
        unit = units[unit_id]
        report_path = output_dir / f"{unit_id}.result.json"
        log_path = output_dir / f"{unit_id}.log"
        events_path = output_dir / f"{unit_id}.events.jsonl"
        command = [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()), "_child",
                   "--plan", str(Path(plan_path).resolve()), "--repo", str(repo),
                   "--unit", unit_id, "--report", str(report_path), "--events", str(events_path)]
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONUNBUFFERED": "1"}
        item = {"unit_id": unit_id, "test_ids": unit["test_ids"],
                "report": report_path.name, "log": log_path.name, "events": events_path.name,
                "returncode": None, "timed_out": False, "validation_errors": []}
        unit_started = time.monotonic()
        print(f"Shard {shard_id}: {unit_id} ({len(unit['test_ids'])} tests)", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            options = ({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt"
                       else {"start_new_session": True})
            with subprocess.Popen(command, cwd=repo, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, **options) as process:
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    item["timed_out"] = True
                    item["validation_errors"].append(f"Exceeded {timeout:g} seconds")
                    try:
                        terminate_process_tree(process)
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        item["validation_errors"].append(f"Process cleanup failed: {exc}")
                        process.kill()
                item["returncode"] = process.wait()
        item["duration_seconds"] = time.monotonic() - unit_started
        try:
            require(item["returncode"] == 0, f"Child exit code: {item['returncode']}")
            check_report(read_json(report_path), plan, unit)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            item["validation_errors"].append(str(exc))
        summary["units"].append(item)
        write_json(output_dir / "summary.json", summary)
        if item["validation_errors"]:
            print(f"FAILED {unit_id}: {'; '.join(item['validation_errors'])}", flush=True)
    summary["complete"] = True
    summary["successful"] = all(not unit["validation_errors"] for unit in summary["units"])
    summary["duration_seconds"] = time.monotonic() - started
    write_json(output_dir / "summary.json", summary)
    return 0 if summary["successful"] else 1


def local_artifact(directory, name):
    require(isinstance(name, str), "Invalid artifact path")
    path = (directory / name).resolve()
    require(path.is_relative_to(directory.resolve()), "Result artifact escapes its directory")
    require(path.is_file(), f"Missing result artifact: {name}")
    return path


def verify_results(plan_path, results_dir, repo):
    plan = read_json(plan_path)
    validate_plan(plan, repo)
    paths = sorted(Path(results_dir).rglob("summary.json"))
    require(len(paths) == plan["shard_count"], "Missing or extra shard summaries")
    units = {unit["id"]: unit for unit in plan["units"]}
    seen_shards, all_ids = set(), []
    for path in paths:
        summary = read_json(path)
        require(summary.get("schema_version") == SCHEMA_VERSION, "Unsupported shard schema")
        require(summary.get("plan_id") == plan["plan_id"], "Shard belongs to a different plan")
        shard_id = summary["shard"]
        require(type(shard_id) is int and 0 <= shard_id < plan["shard_count"]
                and shard_id not in seen_shards, "Duplicate or invalid shard")
        seen_shards.add(shard_id)
        shard = plan["shards"][shard_id]
        exact_ids(summary["test_ids"], shard["test_ids"], "shard test")
        exact_ids([item["unit_id"] for item in summary["units"]], shard["unit_ids"], "shard unit")
        require(summary.get("complete") is True and summary.get("successful") is True,
                "Shard did not finish successfully")
        for item in summary["units"]:
            unit = units[item["unit_id"]]
            exact_ids(item["test_ids"], unit["test_ids"], "reported unit test")
            require(item.get("returncode") == 0 and item.get("timed_out") is False
                    and item.get("validation_errors") == [], "Unit failed or did not exit normally")
            report = read_json(local_artifact(path.parent, item["report"]))
            local_artifact(path.parent, item["log"])
            events_path = local_artifact(path.parent, item["events"])
            check_report(report, plan, unit)
            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            ended = [{k: v for k, v in event.items() if k != "event"}
                     for event in events if event.get("event") == "test_end"]
            require(ended == report["records"], "Test event log differs from completion report")
            all_ids.extend(record["id"] for record in report["records"])
    exact_ids(all_ids, [test["id"] for test in plan["tests"]], "release test")
    print(f"Verified {len(all_ids)} unique tests across {len(seen_shards)} shards.", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--tests-dir", default="tests")
    plan_parser.add_argument("--shards", type=int, default=4)
    plan_parser.add_argument("--output", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--shard", type=int, required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--timeout", type=float, default=900)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--results-dir", required=True)
    child_parser = commands.add_parser("_child", help=argparse.SUPPRESS)
    child_parser.add_argument("--unit", required=True)
    child_parser.add_argument("--report", required=True)
    child_parser.add_argument("--events", required=True)
    for command in (plan_parser, run_parser, verify_parser, child_parser):
        command.add_argument("--repo", type=Path, default=ROOT)
        if command is not plan_parser:
            command.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    args.repo = args.repo.resolve()
    try:
        if args.command == "plan":
            plan = make_plan(args.repo, args.tests_dir, args.shards)
            write_json(args.output, plan)
            print(f"Planned {len(plan['tests'])} unique tests in {args.shards} shards; "
                  f"{len(plan['discovery_duplicates'])} imported duplicate IDs normalized.")
            measured = sum(unit["weight_basis"] == "measured_unit_wall" for unit in plan["units"])
            print(f"Scheduling weights: {measured} measured units; "
                  f"{len(plan['units']) - measured} fallback estimates.")
            for shard in plan["shards"]:
                print(f"Shard {shard['id']}: {len(shard['test_ids'])} tests, "
                      f"scheduling weight {shard['weight_seconds']:.1f} seconds (includes estimates)")
            return 0
        if args.command == "run":
            return run_shard(args.plan, args.shard, args.repo, args.output_dir, args.timeout)
        if args.command == "verify":
            return verify_results(args.plan, args.results_dir, args.repo)
        return run_child(args.plan, args.unit, args.repo, args.report, args.events)
    except (ValueError, KeyError, TypeError, OSError, ImportError, StopIteration) as exc:
        print(f"Release test validation failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
