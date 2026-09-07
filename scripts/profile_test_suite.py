"""Bounded, unittest-compatible test profiler with crash-safe JSONL evidence.

The public command is a small supervisor.  It starts one child process, which
performs ordinary unittest discovery/execution and writes a flushed event before
and after every test.  Keeping the test runner in a child means a native Qt
crash cannot erase the supervisor's exit record or leave the calling shell
waiting forever.

Examples (use the repository virtual environment):

    .venv\\Scripts\\python.exe scripts\\profile_test_suite.py --budget 240
    .venv\\Scripts\\python.exe scripts\\profile_test_suite.py \
        --module tests.test_common --max-tests 20
    .venv\\Scripts\\python.exe scripts\\profile_test_suite.py \
        --id tests.test_common.CommonSymbolsTests.test_state_dataclasses_construct

The generated JSONL and reports belong under ``build/test-profile`` (ignored by
the repository).  No application code is imported by the supervisor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import unittest
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
DEFAULT_BUDGET_SECONDS = 240.0
DEFAULT_PATTERN = "test*.py"
ROOT = Path(__file__).resolve().parents[1]
SEGMENT_TERMINATIONS = frozenset({
    "COMPLETED",
    "BUDGET_EXPIRED",
    "TEST_FAILURE",
    "NATIVE_CRASH",
    "INTERRUPTED",
})
ACCEPTABLE_STATUSES = frozenset({"success", "skip", "expected_failure"})
BAD_STATUSES = frozenset({"failure", "error", "unexpected_success"})
STATUS_ALIASES = {
    "expectedFailure": "expected_failure",
    "unexpectedSuccess": "unexpected_success",
    "expected-failure": "expected_failure",
    "unexpected-success": "unexpected_success",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe_text(value: object, limit: int = 8000) -> str:
    text = str(value)
    return text if len(text) <= limit else text[-limit:]


def _new_segment_id() -> str:
    return f"segment-{uuid.uuid4().hex}"


def _legacy_segment_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.expanduser().resolve()).encode("utf-8")).hexdigest()[:16]
    return f"legacy-segment-{digest}"


def _event_identity(event: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(event.get("segment_id") or "legacy-segment-unknown"),
        str(event.get("attempt_id") or "legacy-attempt-unknown"),
        str(event.get("full_id") or ""),
    )


def _normalise_events(events: Sequence[dict[str, Any]], source: Path) -> list[dict[str, Any]]:
    """Annotate old JSONL records with deterministic segment/attempt identity."""

    segment_id = next((str(event["segment_id"]) for event in events if event.get("segment_id")), None)
    segment_id = segment_id or _legacy_segment_id(source)
    next_attempt: defaultdict[str, int] = defaultdict(int)
    pending: defaultdict[str, list[str]] = defaultdict(list)
    last_started: dict[str, str] = {}
    next_operation = 0
    normalised: list[dict[str, Any]] = []
    for original in events:
        event = dict(original)
        event.setdefault("segment_id", segment_id)
        kind = event.get("event")
        if kind == "SUPERVISOR_START":
            event.setdefault("segment_started_at_utc", event.get("timestamp_utc"))
        elif kind == "SUPERVISOR_EXIT":
            event.setdefault("segment_ended_at_utc", event.get("timestamp_utc"))
            if not event.get("termination"):
                classification = str(event.get("classification") or "")
                event["termination"] = {
                    "success": "COMPLETED",
                    "timeout": "BUDGET_EXPIRED",
                    "interrupted": "INTERRUPTED",
                    "native_crash": "NATIVE_CRASH",
                    "test_failure": "TEST_FAILURE",
                }.get(classification, "INTERRUPTED")
        elif kind == "RUN_SUMMARY":
            event.setdefault("termination", "COMPLETED" if event.get("was_successful") else "TEST_FAILURE")
        full_id = str(event.get("full_id")) if event.get("full_id") is not None else None
        if kind == "TEST_START" and full_id is not None:
            if not event.get("attempt_id"):
                next_attempt[full_id] += 1
                event["attempt_id"] = f"legacy-attempt-{next_attempt[full_id]}"
            attempt_id = str(event["attempt_id"])
            pending[full_id].append(attempt_id)
            last_started[full_id] = attempt_id
        elif kind == "TEST_END" and full_id is not None:
            status_value = event.get("status")
            if isinstance(status_value, str) and status_value in STATUS_ALIASES:
                event["status"] = STATUS_ALIASES[status_value]
            if not event.get("attempt_id"):
                if pending[full_id]:
                    event["attempt_id"] = pending[full_id].pop(0)
                else:
                    next_attempt[full_id] += 1
                    event["attempt_id"] = f"legacy-attempt-{next_attempt[full_id]}"
            else:
                attempt_id = str(event["attempt_id"])
                if attempt_id in pending[full_id]:
                    pending[full_id].remove(attempt_id)
            last_started[full_id] = str(event["attempt_id"])
        elif kind in {"INSTRUMENTATION_START", "INSTRUMENTATION_END"}:
            if full_id is not None and not event.get("attempt_id"):
                event["attempt_id"] = last_started.get(full_id)
            if not event.get("operation_id"):
                next_operation += 1
                event["operation_id"] = next_operation
        if kind == "TEST_END":
            nested = []
            for record in event.get("instrumentation") or []:
                item = dict(record)
                item.setdefault("segment_id", event["segment_id"])
                item.setdefault("attempt_id", event.get("attempt_id"))
                item.setdefault("full_id", event.get("full_id"))
                nested.append(item)
            event["instrumentation"] = nested
        normalised.append(event)
    return normalised


class JsonlWriter:
    """Append one complete, flushed JSON object per line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", encoding="utf-8", newline="\n")

    def emit(self, event: str, **fields: Any) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "event": event,
            "timestamp_utc": _utc_now(),
            "monotonic_ns": time.perf_counter_ns(),
            **fields,
        }
        self._stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self._stream.flush()
        if event in {"SUPERVISOR_START", "SUPERVISOR_EXIT", "CHILD_START", "TEST_START", "TEST_END"}:
            os.fsync(self._stream.fileno())

    def close(self) -> None:
        self._stream.close()


def _exception_payload(exc_info: tuple[type[BaseException], BaseException, Any]) -> dict[str, str]:
    exc_type, exc, tb = exc_info
    return {
        "exception_type": getattr(exc_type, "__name__", str(exc_type)),
        "exception_message": _safe_text(exc),
        "traceback": _safe_text("".join(traceback.format_exception(exc_type, exc, tb))),
    }


def _test_id(test: unittest.case.TestCase | object) -> str:
    method = getattr(test, "id", None)
    if callable(method):
        try:
            return str(method())
        except Exception:
            pass
    return str(test)


def _test_parts(full_id: str, test: object | None = None) -> tuple[str, str, str]:
    module = getattr(test.__class__, "__module__", "") if test is not None else ""
    class_name = getattr(test.__class__, "__qualname__", "") if test is not None else ""
    name = getattr(test, "_testMethodName", "") if test is not None else ""
    bits = full_id.split(".")
    if not module and len(bits) >= 2:
        module = ".".join(bits[:-2])
    if not class_name and len(bits) >= 2:
        class_name = bits[-2]
    if not name and bits:
        name = bits[-1]
    return module or "unknown", class_name or "unknown", name or full_id


def _flatten(suite: unittest.TestSuite | unittest.case.TestCase) -> list[unittest.case.TestCase | object]:
    if isinstance(suite, unittest.TestSuite):
        tests: list[unittest.case.TestCase | object] = []
        for child in suite:
            tests.extend(_flatten(child))
        return tests
    return [suite]


def _load_tests(repo: Path, ids: Sequence[str], modules: Sequence[str], pattern: str) -> tuple[list[object], list[str], float]:
    loader = unittest.TestLoader()
    discovery_start = time.perf_counter()
    errors: list[str] = []
    if ids or modules:
        suites = []
        # When both are supplied, modules form an explicit prefix and IDs are
        # appended.  This supports bounded order/context reproductions while
        # leaving the ordinary IDs-only and modules-only behavior unchanged.
        for name in (*modules, *ids):
            try:
                suites.append(loader.loadTestsFromName(name))
            except Exception:
                errors.append(_safe_text(traceback.format_exc()))
        tests = [test for suite in suites for test in _flatten(suite)]
    else:
        suite = loader.discover(
            start_dir=str(repo / "tests"),
            pattern=pattern,
            top_level_dir=str(repo),
        )
        tests = _flatten(suite)
        errors.extend(_safe_text(error) for error in getattr(loader, "errors", []))
    return tests, errors, time.perf_counter() - discovery_start


def category_for(module: str, class_name: str, name: str) -> str:
    """Return one deterministic, non-overlapping accounting category."""

    haystack = f"{module}.{class_name}.{name}".lower()
    if "preview_ui" in haystack or "gallery" in haystack:
        return "preview/gallery"
    if "subprocess" in haystack:
        return "subprocess"
    ui_modules = (
        "phase7_layout_audit",
        "phase7_accessibility",
        "dense_form_layout",
        "dock_host",
        "menu_toolbar",
        "presentation_widget",
        "source_picker_dialog",
        "status_bar",
        "ui_filename_delegate",
        "ui_split_scale_controls",
        "workspace_shell",
        "workflow_navigation",
    )
    if any(token in haystack for token in ui_modules) or "main_window" in haystack or "layout" in haystack:
        return "real MainWindow/UI integration"
    controller_modules = (
        "compare_vp",
        "drr_sources",
        "mcd",
        "mcd_extract",
        "mcd_organizer",
        "mcd_peak_shift",
        "pl_source_workflow",
        "power_series",
        "shg",
    )
    if any(token in haystack for token in controller_modules) or "controller" in haystack:
        return "controller"
    io_modules = (
        "dat_loader",
        "export_",
        "file_ops",
        "loader",
        "provenance",
        "release_version",
        "windows_powerpoint_packaging",
        "xlsx_map_input",
    )
    if any(token in haystack for token in io_modules) or any(token in haystack for token in ("file", "export", "load")):
        return "filesystem/I/O"
    return "pure unit/core"


class RuntimeInstrumentation:
    """Test-only wrappers for real subprocess.run and MainWindow.__init__ calls."""

    def __init__(self, writer: JsonlWriter, segment_id: str = "legacy-segment-runtime") -> None:
        self.writer = writer
        self.segment_id = segment_id
        self._test_id: str | None = None
        self._attempt_id: str | None = None
        self._records: list[dict[str, Any]] = []
        self._restore: list[tuple[object, str, object]] = []
        self._next_operation_id = 0

    def install(self) -> None:
        import subprocess as subprocess_module

        original_run = subprocess_module.run

        def wrapped_run(*args: Any, **kwargs: Any) -> Any:
            operation = self._begin("subprocess.run")
            returncode: int | None = None
            try:
                result = original_run(*args, **kwargs)
                returncode = getattr(result, "returncode", None)
                return result
            finally:
                if operation is not None:
                    self._record("subprocess.run", operation, returncode=returncode)

        subprocess_module.run = wrapped_run  # type: ignore[assignment]
        self._restore.append((subprocess_module, "run", original_run))

        main_window_module = sys.modules.get("ui_qt.main_window")
        main_window = getattr(main_window_module, "MainWindow", None)
        original_init = getattr(main_window, "__init__", None)
        if main_window is not None and callable(original_init):
            def wrapped_init(instance: Any, *args: Any, **kwargs: Any) -> Any:
                operation = self._begin("MainWindow.__init__")
                try:
                    return original_init(instance, *args, **kwargs)
                finally:
                    if operation is not None:
                        self._record(
                            "MainWindow.__init__",
                            operation,
                            instance_identity=id(instance),
                        )

            main_window.__init__ = wrapped_init  # type: ignore[assignment]
            self._restore.append((main_window, "__init__", original_init))

    def set_test(self, full_id: str | None, attempt_id: str | None = None) -> None:
        self._test_id = full_id
        self._attempt_id = attempt_id
        if full_id is None:
            os.environ.pop("PROFILE_TEST_CURRENT_ID", None)
            os.environ.pop("PROFILE_TEST_CURRENT_ATTEMPT_ID", None)
            os.environ.pop("PROFILE_TEST_CURRENT_SEGMENT_ID", None)
        else:
            os.environ["PROFILE_TEST_CURRENT_ID"] = full_id
            os.environ["PROFILE_TEST_CURRENT_ATTEMPT_ID"] = str(attempt_id)
            os.environ["PROFILE_TEST_CURRENT_SEGMENT_ID"] = self.segment_id

    def _begin(self, kind: str) -> dict[str, Any] | None:
        if self._test_id is None:
            return None
        self._next_operation_id += 1
        operation = {
            "operation_id": self._next_operation_id,
            "segment_id": self.segment_id,
            "attempt_id": self._attempt_id,
            "kind": kind,
            "full_id": self._test_id,
            "started_at": time.perf_counter(),
        }
        self.writer.emit(
            "INSTRUMENTATION_START",
            operation_id=operation["operation_id"],
            segment_id=self.segment_id,
            attempt_id=operation["attempt_id"],
            kind=kind,
            full_id=self._test_id,
        )
        return operation

    def _record(self, kind: str, operation: dict[str, Any], **fields: Any) -> None:
        duration = time.perf_counter() - float(operation["started_at"])
        record = {
            "operation_id": operation["operation_id"],
            "segment_id": self.segment_id,
            "attempt_id": operation.get("attempt_id"),
            "kind": kind,
            "full_id": operation["full_id"],
            "duration_s": duration,
            **fields,
        }
        self._records.append(record)
        self.writer.emit(
            "INSTRUMENTATION_END",
            operation_id=operation["operation_id"],
            segment_id=self.segment_id,
            attempt_id=operation.get("attempt_id"),
            kind=kind,
            full_id=operation["full_id"],
            duration_s=duration,
            **fields,
        )

    def take(self, full_id: str, attempt_id: str | None = None) -> list[dict[str, Any]]:
        selected = [
            record for record in self._records
            if record["full_id"] == full_id and record.get("attempt_id") == attempt_id
        ]
        self._records = [
            record for record in self._records
            if not (record["full_id"] == full_id and record.get("attempt_id") == attempt_id)
        ]
        return selected

    def restore(self) -> None:
        for target, attr, original in reversed(self._restore):
            setattr(target, attr, original)
        self._restore.clear()


class ProfilingResult(unittest.TestResult):
    def __init__(self, writer: JsonlWriter, instrumentation: RuntimeInstrumentation, segment_id: str | None = None) -> None:
        super().__init__()
        self.writer = writer
        self.instrumentation = instrumentation
        self.segment_id = segment_id or instrumentation.segment_id
        self._starts: dict[int, tuple[str, str, float, str]] = {}
        self._parts: dict[str, tuple[str, str, str]] = {}
        self._status: dict[int, str] = {}
        self._end_events: list[dict[str, Any]] = []
        self._next_attempt_id = 0

    def startTest(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        super().startTest(test)
        full_id = _test_id(test)
        module, class_name, name = _test_parts(full_id, test)
        self._parts[full_id] = (module, class_name, name)
        self._next_attempt_id += 1
        attempt_id = f"attempt-{self._next_attempt_id}"
        started_at = _utc_now()
        self.writer.emit(
            "TEST_START",
            segment_id=self.segment_id,
            attempt_id=attempt_id,
            full_id=full_id,
            module=module,
            class_name=class_name,
            name=name,
            started_at_utc=started_at,
        )
        # Do not charge the durable START write to the test itself.
        self._starts[id(test)] = (full_id, attempt_id, time.perf_counter(), started_at)
        self.instrumentation.set_test(full_id, attempt_id)

    def stopTest(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        full_id = _test_id(test)
        attempt = self._starts.pop(id(test), None)
        duration = time.perf_counter() - attempt[2] if attempt is not None else None
        attempt_id = attempt[1] if attempt is not None else f"attempt-unknown-{id(test)}"
        started_at = attempt[3] if attempt is not None else None
        module, class_name, name = self._parts.get(full_id, _test_parts(full_id, test))
        status = self._status.pop(id(test), "success")
        instrumentation = self.instrumentation.take(full_id, attempt_id)
        category = category_for(module, class_name, name)
        measured_kinds = {record.get("kind") for record in instrumentation}
        if "MainWindow.__init__" in measured_kinds:
            category = "real MainWindow/UI integration"
        elif "subprocess.run" in measured_kinds and category == "pure unit/core":
            category = "subprocess"
        event = {
            "segment_id": self.segment_id,
            "attempt_id": attempt_id,
            "full_id": full_id,
            "module": module,
            "class_name": class_name,
            "name": name,
            "status": status,
            "duration_s": duration,
            "started_at_utc": started_at,
            "ended_at_utc": _utc_now(),
            "category": category,
            "instrumentation": instrumentation,
        }
        self._end_events.append(event)
        self.writer.emit("TEST_END", **event)
        if module == "tests.test_preview_ui" and os.environ.get("PROFILE_PHASE_LOG"):
            try:
                from tests.test_preview_ui import _qt_state_snapshot
                from PySide6.QtWidgets import QApplication

                _qt_state_snapshot("post_test_stop", QApplication.instance())
            except Exception as exc:
                self.writer.emit(
                    "QT_STATE_ERROR",
                    full_id=full_id,
                    exception_type=type(exc).__name__,
                    exception_message=_safe_text(exc),
                )
        self.instrumentation.set_test(None)
        super().stopTest(test)

    def _set_status(self, test: object, status: str) -> None:
        self._status[id(test)] = status

    def _attempt_fields(self, test: object) -> dict[str, Any]:
        attempt = self._starts.get(id(test))
        return {
            "segment_id": self.segment_id,
            "attempt_id": attempt[1] if attempt is not None else None,
            "full_id": _test_id(test),
        }

    def addSuccess(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        self._set_status(test, "success")
        super().addSuccess(test)

    def addFailure(self, test: unittest.case.TestCase, err: Any) -> None:  # noqa: N802
        self._set_status(test, "failure")
        self.writer.emit("TEST_FAILURE", **self._attempt_fields(test), **_exception_payload(err))
        super().addFailure(test, err)

    def addError(self, test: unittest.case.TestCase, err: Any) -> None:  # noqa: N802
        self._set_status(test, "error")
        self.writer.emit("TEST_ERROR", **self._attempt_fields(test), **_exception_payload(err))
        super().addError(test, err)

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:  # noqa: N802
        self._set_status(test, "skip")
        self.writer.emit("TEST_SKIP", **self._attempt_fields(test), reason=_safe_text(reason))
        super().addSkip(test, reason)

    def addExpectedFailure(self, test: unittest.case.TestCase, err: Any) -> None:  # noqa: N802
        self._set_status(test, "expected_failure")
        self.writer.emit("TEST_EXPECTED_FAILURE", **self._attempt_fields(test), **_exception_payload(err))
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        self._set_status(test, "unexpected_success")
        self.writer.emit("TEST_UNEXPECTED_SUCCESS", **self._attempt_fields(test))
        super().addUnexpectedSuccess(test)

    def addSubTest(self, test: unittest.case.TestCase, subtest: unittest.case.TestCase, err: Any) -> None:  # noqa: N802
        parent_id = _test_id(test)
        if err is None:
            status = "success"
            payload: dict[str, Any] = {}
        else:
            failure_exception = getattr(test, "failureException", AssertionError)
            try:
                is_failure = issubclass(err[0], failure_exception)
            except TypeError:
                is_failure = issubclass(err[0], AssertionError)
            status = "failure" if is_failure else "error"
            self._status[id(test)] = status
            payload = _exception_payload(err)
        self.writer.emit(
            "SUBTEST",
            **self._attempt_fields(test),
            subtest_id=_test_id(subtest),
            status=status,
            **payload,
        )
        super().addSubTest(test, subtest, err)


def _child_main(args: argparse.Namespace) -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    writer = JsonlWriter(Path(args.log))
    segment_id = str(getattr(args, "segment_id", None) or _legacy_segment_id(Path(args.log)))
    child_started = time.perf_counter()
    resume_state: dict[str, Any] | None = None
    if args.resume is not None:
        try:
            resume_state = _load_resume_state(Path(args.resume))
        except ValueError as exc:
            writer.emit("RESUME_ERROR", reason=_safe_text(exc), source_log=str(args.resume))
            writer.close()
            return 1
    writer.emit(
        "CHILD_START",
        segment_id=segment_id,
        pid=os.getpid(),
        python=sys.executable,
        platform=platform.platform(),
        ids=list(args.ids),
        modules=list(args.modules),
        pattern=args.pattern,
        max_tests=args.max_tests,
        resume_log=str(Path(args.resume).resolve()) if args.resume is not None else None,
    )
    tests, discovery_errors, discovery_duration = _load_tests(ROOT, args.ids, args.modules, args.pattern)
    if args.max_tests is not None:
        tests = tests[: args.max_tests]
    discovered_ids = [_test_id(test) for test in tests]
    writer.emit(
        "DISCOVERY_END",
        segment_id=segment_id,
        duration_s=discovery_duration,
        tests_discovered=len(discovered_ids),
        test_ids=discovered_ids,
        errors=discovery_errors,
    )
    if resume_state is not None:
        try:
            _validate_resume_source(resume_state, discovered_ids)
        except ValueError as exc:
            writer.emit("RESUME_ERROR", reason=_safe_text(exc), source_log=resume_state["path"])
            writer.emit(
                "RUN_SUMMARY",
                pid=os.getpid(),
                tests_run=0,
                status_counts={},
                completed_test_count=0,
                test_duration_s=0.0,
                runner_duration_s=0.0,
                discovery_duration_s=discovery_duration,
                discovery_errors=discovery_errors,
                child_runtime_s=time.perf_counter() - child_started,
                was_successful=False,
                resume_error=True,
            )
            writer.close()
            return 1
        before_count = len(tests)
        tests = _filter_completed_tests(tests, resume_state["completed_ids"])
        writer.emit(
            "RESUME_FILTER",
            source_log=resume_state["path"],
            ancestor_logs=resume_state["ancestor_logs"],
            completed_id_count=len(resume_state["completed_ids"]),
            skipped_test_count=before_count - len(tests),
            selected_test_count=len(tests),
            selected_test_ids=[_test_id(test) for test in tests],
        )
    instrumentation = RuntimeInstrumentation(writer, segment_id)
    instrumentation.install()
    result = ProfilingResult(writer, instrumentation, segment_id)
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=0, resultclass=lambda *_: result)
    run_started = time.perf_counter()
    try:
        suite = unittest.TestSuite(tests)
        runner.run(suite)
    finally:
        instrumentation.restore()
    run_duration = time.perf_counter() - run_started
    status_counts = Counter(event["status"] for event in result._end_events)
    writer.emit(
        "RUN_SUMMARY",
        pid=os.getpid(),
        tests_run=result.testsRun,
        status_counts=dict(status_counts),
        completed_test_count=len(result._end_events),
        test_duration_s=sum(float(event["duration_s"] or 0.0) for event in result._end_events),
        runner_duration_s=run_duration,
        discovery_duration_s=discovery_duration,
        discovery_errors=discovery_errors,
        child_runtime_s=time.perf_counter() - child_started,
        was_successful=result.wasSuccessful(),
        segment_id=segment_id,
        termination="COMPLETED" if result.wasSuccessful() and not discovery_errors else "TEST_FAILURE",
    )
    writer.close()
    return 0 if result.wasSuccessful() and not discovery_errors else 1


def _read_events(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    malformed: list[str] = []
    if not path.exists():
        return events, [f"missing log: {path}"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(event)
            else:
                malformed.append(f"line {line_number}: JSON value is not an object")
        except json.JSONDecodeError as exc:
            malformed.append(f"line {line_number}: {exc}")
    return events, malformed


def _filter_completed_tests(tests: Sequence[object], completed_ids: set[str]) -> list[object]:
    """Keep discovery order while removing IDs with a durable TEST_END."""

    return [test for test in tests if _test_id(test) not in completed_ids]


def _resolve_resume_reference(reference: object, source: Path) -> Path:
    path = Path(str(reference)).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def _load_resume_state(path: Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load completed IDs and all linked ancestor evidence from a JSONL log."""

    source = path.expanduser().resolve()
    if source in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, source))
        raise ValueError(f"resume log ancestry cycle: {chain}")
    if not source.exists() or not source.is_file():
        raise ValueError(f"resume log does not exist: {source}")
    events, malformed = _read_events(source)
    events = _normalise_events(events, source)
    if malformed:
        raise ValueError(f"invalid resume log {source}: {'; '.join(malformed[:3])}")
    discovery_events = [event for event in events if event.get("event") == "DISCOVERY_END"]
    if not discovery_events:
        raise ValueError(f"resume log has no DISCOVERY_END: {source}")

    supervisor_start = next(
        (event for event in events if event.get("event") == "SUPERVISOR_START"),
        {},
    )
    references: list[Path] = []
    for reference in supervisor_start.get("resume_ancestor_logs") or []:
        references.append(_resolve_resume_reference(reference, source))
    if supervisor_start.get("resume_log"):
        references.append(_resolve_resume_reference(supervisor_start["resume_log"], source))
    references = list(dict.fromkeys(references))

    ancestor_logs: list[str] = []
    seen: set[Path] = set()
    for reference in references:
        linked = _load_resume_state(reference, (*_stack, source))
        for linked_path in linked["ancestor_logs"]:
            linked_resolved = Path(linked_path).resolve()
            if linked_resolved in seen:
                continue
            seen.add(linked_resolved)
            ancestor_logs.append(str(linked_resolved))

    if source not in seen:
        ancestor_logs.append(str(source))

    all_events: list[dict[str, Any]] = []
    all_phase_events: list[dict[str, Any]] = []
    for ancestor_path in ancestor_logs:
        ancestor_file = Path(ancestor_path)
        ancestor_events, ancestor_malformed = _read_events(ancestor_file)
        ancestor_events = _normalise_events(ancestor_events, ancestor_file)
        if ancestor_malformed:
            raise ValueError(f"invalid resume log {ancestor_file}: {'; '.join(ancestor_malformed[:3])}")
        all_events.extend(ancestor_events)
        phase_path = ancestor_file.with_name("phase-events.jsonl")
        if phase_path.exists():
            phase_events, phase_malformed = _read_events(phase_path)
            if phase_malformed:
                raise ValueError(f"invalid resume phase log {phase_path}: {'; '.join(phase_malformed[:3])}")
            all_phase_events.extend(phase_events)

    completed_ids = {
        str(event.get("full_id"))
        for event in all_events
        if event.get("event") == "TEST_END" and event.get("full_id") is not None
    }
    starts = Counter(
        _event_identity(event)
        for event in all_events
        if event.get("event") == "TEST_START" and event.get("full_id") is not None
    )
    ends = Counter(
        _event_identity(event)
        for event in all_events
        if event.get("event") == "TEST_END" and event.get("full_id") is not None
    )
    unfinished_attempts = []
    for identity, count in starts.items():
        remaining = count - ends[identity]
        if remaining > 0:
            unfinished_attempts.extend({
                "segment_id": identity[0],
                "attempt_id": identity[1],
                "full_id": identity[2],
            } for _ in range(remaining))
    unfinished_ids = list(dict.fromkeys(item["full_id"] for item in unfinished_attempts))
    acceptable_ids = {
        str(event.get("full_id"))
        for event in all_events
        if event.get("event") == "TEST_END"
        and event.get("full_id") is not None
        and str(event.get("status")) in ACCEPTABLE_STATUSES
    }
    source_ids = discovery_events[-1].get("test_ids") or []
    if not isinstance(source_ids, list):
        raise ValueError(f"resume log DISCOVERY_END has invalid test_ids: {source}")
    return {
        "path": str(source),
        "completed_ids": completed_ids,
        "unfinished_ids": unfinished_ids,
        "unfinished_attempts": unfinished_attempts,
        "resolved_unfinished_ids": sorted(set(unfinished_ids) & acceptable_ids),
        "source_discovered_ids": [str(test_id) for test_id in source_ids],
        "ancestor_logs": ancestor_logs,
        "events": all_events,
        "phase_events": all_phase_events,
    }


def _validate_resume_source(state: dict[str, Any], discovered_ids: Sequence[str]) -> None:
    source_ids = set(state.get("source_discovered_ids") or [])
    current_ids = set(discovered_ids)
    missing = sorted(source_ids - current_ids)
    if missing:
        preview = ", ".join(missing[:3])
        if len(missing) > 3:
            preview += ", ..."
        raise ValueError(f"resume source mismatch: {len(missing)} source test IDs are not currently discoverable ({preview})")


def _classify_exit(exit_code: int | None, termination_reason: str | None, summary_seen: bool) -> str:
    if termination_reason == "timeout":
        return "timeout"
    if termination_reason == "interrupted":
        return "interrupted"
    if exit_code == 0 and summary_seen:
        return "success"
    if exit_code is not None and (exit_code < 0 or exit_code >= 0xC0000000):
        return "native_crash"
    if not summary_seen and exit_code not in (0, None):
        return "interrupted"
    return "test_failure"


def _termination_for(
    classification: str,
    termination_reason: str | None,
    summary: dict[str, Any] | None = None,
) -> str:
    if termination_reason == "timeout":
        return "BUDGET_EXPIRED"
    if termination_reason == "interrupted":
        return "INTERRUPTED"
    if classification == "native_crash":
        return "NATIVE_CRASH"
    if classification == "success" and (summary is None or summary.get("was_successful", True)):
        return "COMPLETED"
    return "TEST_FAILURE"


def _terminate_child(proc: subprocess.Popen[Any]) -> str:
    if proc.poll() is not None:
        return "already_exited"
    pid = proc.pid
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    return "terminated_child_tree"


def _write_report(
    output_dir: Path,
    log_path: Path,
    supervisor_exit: dict[str, Any],
    *,
    resume_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    current_events, malformed = _read_events(log_path)
    current_events = _normalise_events(current_events, log_path)
    events = list(resume_state["events"]) + current_events if resume_state is not None else current_events
    if resume_state is not None:
        malformed.extend(resume_state.get("malformed_lines") or [])
    phase_log_path = log_path.with_name("phase-events.jsonl")
    current_phase_events, phase_malformed = _read_events(phase_log_path)
    phase_events = list(resume_state["phase_events"]) + current_phase_events if resume_state is not None else current_phase_events
    current_ends = [event for event in current_events if event.get("event") == "TEST_END"]
    starts: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    ends: list[dict[str, Any]] = []
    instrumentation: list[dict[str, Any]] = []
    discovery: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    instrumentation_starts: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    instrumentation_ends: set[tuple[str, str, str, int]] = set()
    for event in events:
        kind = event.get("event")
        if kind == "TEST_START":
            starts[_event_identity(event)].append(event)
        elif kind == "TEST_END":
            ends.append(event)
        elif kind == "DISCOVERY_END":
            discovery = event
        elif kind == "RUN_SUMMARY":
            summary = event
        elif kind == "INSTRUMENTATION_START":
            try:
                identity = _event_identity(event) + (int(event["operation_id"]),)
                instrumentation_starts[identity] = event
            except (KeyError, TypeError, ValueError):
                malformed.append("instrumentation START has no valid operation_id")
        elif kind == "INSTRUMENTATION_END":
            try:
                instrumentation_ends.add(_event_identity(event) + (int(event["operation_id"]),))
            except (KeyError, TypeError, ValueError):
                malformed.append("instrumentation END has no valid operation_id")
        if kind == "TEST_END":
            instrumentation.extend(event.get("instrumentation") or [])

    unfinished: list[dict[str, Any]] = []
    ended_attempts = Counter(_event_identity(event) for event in ends)
    for identity, start_events in starts.items():
        remaining_ends = ended_attempts[identity]
        for start in start_events:
            if remaining_ends:
                remaining_ends -= 1
            else:
                unfinished.append({
                    "segment_id": identity[0],
                    "attempt_id": identity[1],
                    "full_id": identity[2],
                    "module": start.get("module"),
                    "class_name": start.get("class_name"),
                    "name": start.get("name"),
                    "started_at": start.get("started_at_utc") or start.get("timestamp_utc"),
                })
        ended_attempts[identity] = remaining_ends

    module_totals: defaultdict[str, float] = defaultdict(float)
    module_counts: Counter[str] = Counter()
    category_totals: defaultdict[str, float] = defaultdict(float)
    status_counts: Counter[str] = Counter()
    for event in ends:
        duration = float(event.get("duration_s") or 0.0)
        module_totals[str(event.get("module") or "unknown")] += duration
        module_counts[str(event.get("module") or "unknown")] += 1
        category_totals[str(event.get("category") or "unknown/unattributed")] += duration
        status_counts[str(event.get("status") or "unknown")] += 1
    top_tests = sorted(ends, key=lambda event: float(event.get("duration_s") or 0.0), reverse=True)[:20]
    top_modules = sorted(module_totals.items(), key=lambda item: item[1], reverse=True)[:15]
    contribution_totals: defaultdict[str, float] = defaultdict(float)
    for event in instrumentation:
        contribution_totals[str(event.get("kind"))] += float(event.get("duration_s") or 0.0)
    phase_totals: defaultdict[str, float] = defaultdict(float)
    phase_counts: Counter[str] = Counter()
    for event in phase_events:
        if event.get("event") == "PHASE_END":
            phase = str(event.get("phase") or "unknown")
            phase_totals[phase] += float(event.get("duration_s") or 0.0)
            phase_counts[phase] += 1
    unfinished_instrumentation = [
        {
            "segment_id": identity[0],
            "attempt_id": identity[1],
            "full_id": identity[2],
            "operation_id": identity[3],
            "kind": event.get("kind"),
            "started_at": event.get("timestamp_utc"),
        }
        for identity, event in instrumentation_starts.items()
        if identity not in instrumentation_ends
    ]
    completed_test_duration = sum(float(event.get("duration_s") or 0.0) for event in ends)
    current_completed_test_duration = sum(float(event.get("duration_s") or 0.0) for event in current_ends)
    runner_duration = float(summary.get("runner_duration_s") or 0.0)
    discovery_duration = float(discovery.get("duration_s") or 0.0)
    child_runtime = float(summary.get("child_runtime_s") or 0.0)
    segment_summaries = [
        event for event in events
        if event.get("event") == "RUN_SUMMARY"
    ]
    segment_exits = [
        event for event in events
        if event.get("event") == "SUPERVISOR_EXIT"
    ]
    merged_discovery_duration = sum(
        float(event.get("duration_s") or 0.0)
        for event in events
        if event.get("event") == "DISCOVERY_END"
    )
    merged_runner_duration = sum(float(event.get("runner_duration_s") or 0.0) for event in segment_summaries)
    merged_child_runtime = sum(float(event.get("child_runtime_s") or 0.0) for event in segment_summaries)
    merged_supervisor_wall = sum(float(event.get("supervisor_duration_s") or 0.0) for event in segment_exits)
    merged_status_counts: Counter[str] = Counter()
    for segment in segment_summaries:
        merged_status_counts.update(segment.get("status_counts") or {})

    def outcome_counts(counts: Counter[str]) -> dict[str, int]:
        return {
            "pass": sum(counts.get(status, 0) for status in ("success", "expected_failure")),
            "skip": counts.get("skip", 0),
            "fail": sum(counts.get(status, 0) for status in ("failure", "error", "unexpected_success")),
        }

    tests_over_5s = [
        event for event in sorted(ends, key=lambda event: float(event.get("duration_s") or 0.0), reverse=True)
        if float(event.get("duration_s") or 0.0) > 5.0
    ]
    tests_over_2s = [
        event for event in sorted(ends, key=lambda event: float(event.get("duration_s") or 0.0), reverse=True)
        if float(event.get("duration_s") or 0.0) > 2.0
    ]
    setup_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in instrumentation:
        setup_groups[str(record.get("kind") or "unknown")].append(record)
    repeated_costly_setup = []
    for kind, records in setup_groups.items():
        durations = [float(record.get("duration_s") or 0.0) for record in records]
        if len(records) > 1 and max(durations, default=0.0) > 2.0:
            repeated_costly_setup.append({
                "kind": kind,
                "count": len(records),
                "unique_tests": len({str(record.get("full_id")) for record in records}),
                "duration_s": sum(durations),
                "max_duration_s": max(durations),
                "average_duration_s": sum(durations) / len(durations),
            })
    repeated_costly_setup.sort(key=lambda record: record["duration_s"], reverse=True)

    segment_paths = (
        list(resume_state["ancestor_logs"]) + [str(log_path.resolve())]
        if resume_state is not None
        else [str(log_path.resolve())]
    )
    wall_by_segment = []
    for segment_path in segment_paths:
        segment_file = Path(segment_path)
        segment_events, _ = _read_events(segment_file)
        segment_events = _normalise_events(segment_events, segment_file)
        segment_summary = next(
            (event for event in reversed(segment_events) if event.get("event") == "RUN_SUMMARY"),
            {},
        )
        segment_exit = next(
            (event for event in reversed(segment_events) if event.get("event") == "SUPERVISOR_EXIT"),
            {},
        )
        segment_ends = [event for event in segment_events if event.get("event") == "TEST_END"]
        segment_status = Counter(str(event.get("status") or "unknown") for event in segment_ends)
        wall_by_segment.append({
            "log_path": segment_path,
            "segment_id": next((event.get("segment_id") for event in segment_events if event.get("event") == "SUPERVISOR_START"), _legacy_segment_id(segment_file)),
            "segment_started_at_utc": next((event.get("segment_started_at_utc") for event in segment_events if event.get("event") == "SUPERVISOR_START"), None),
            "segment_ended_at_utc": segment_exit.get("segment_ended_at_utc"),
            "classification": segment_exit.get("classification"),
            "termination": segment_exit.get("termination") or segment_summary.get("termination"),
            "completed_test_count": len(segment_ends),
            "status_counts": dict(segment_status),
            "outcome_counts": outcome_counts(segment_status),
            "measured_test_duration_s": sum(float(event.get("duration_s") or 0.0) for event in segment_ends),
            "discovery_s": float(segment_summary.get("discovery_duration_s") or 0.0),
            "runner_s": float(segment_summary.get("runner_duration_s") or 0.0),
            "child_runtime_s": float(segment_summary.get("child_runtime_s") or 0.0),
            "supervisor_wall_s": float(segment_exit.get("supervisor_duration_s") or 0.0),
        })
    discovered_ids = [str(test_id) for test_id in (discovery.get("test_ids") or [])]
    completed_ids = {str(event.get("full_id")) for event in ends if event.get("full_id") is not None}
    acceptable_ends = [
        event for event in ends
        if str(event.get("status")) in ACCEPTABLE_STATUSES and event.get("full_id") is not None
    ]
    acceptable_ids = {str(event["full_id"]) for event in acceptable_ends}
    unresolved_intended_ids = sorted(set(discovered_ids) - acceptable_ids)
    unfinished_ids = sorted({str(item["full_id"]) for item in unfinished})
    resolved_unfinished_ids = sorted(set(unfinished_ids) & acceptable_ids)
    bad_outcomes = [
        event for event in ends
        if str(event.get("status")) in BAD_STATUSES
    ]
    segment_terminations = [
        str(event.get("termination"))
        for event in events
        if event.get("event") == "SUPERVISOR_EXIT" and event.get("termination")
    ]
    native_crash_unresolved_ids = sorted({
        item["full_id"]
        for item in unfinished
        if any(
            event.get("event") == "SUPERVISOR_EXIT"
            and event.get("segment_id") == item["segment_id"]
            and event.get("termination") == "NATIVE_CRASH"
            for event in events
        )
        and item["full_id"] in unresolved_intended_ids
    })
    attempts = []
    for identity in sorted(set(starts) | {_event_identity(event) for event in ends}):
        attempt_ends = [event for event in ends if _event_identity(event) == identity]
        start_events = starts.get(identity, [])
        attempt = {
            "segment_id": identity[0],
            "attempt_id": identity[1],
            "full_id": identity[2],
            "started_at_utc": start_events[0].get("started_at_utc") if start_events else None,
            "ended_at_utc": attempt_ends[-1].get("ended_at_utc") if attempt_ends else None,
            "status": attempt_ends[-1].get("status") if attempt_ends else None,
            "duration_s": attempt_ends[-1].get("duration_s") if attempt_ends else None,
            "resolved": bool(attempt_ends),
        }
        attempts.append(attempt)
    merged_passed = (
        bool(discovered_ids)
        and not unresolved_intended_ids
        and not bad_outcomes
        and not native_crash_unresolved_ids
        and "TEST_FAILURE" not in segment_terminations
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "log_path": str(log_path),
        "platform": platform.platform(),
        "python": sys.executable,
        "exit": supervisor_exit,
        "discovery": {
            "duration_s": discovery.get("duration_s"),
            "tests_discovered": discovery.get("tests_discovered", 0),
            "test_ids": discovery.get("test_ids", []),
            "errors": discovery.get("errors", []),
        },
        "summary": summary,
        "completed_test_count": len(ends),
        "segment_completed_test_count": len(current_ends),
        "status_counts": dict(status_counts),
        "outcome_counts": outcome_counts(status_counts),
        "coverage": {
            "discovered_count": len(discovered_ids),
            "discovered_unique_count": len(set(discovered_ids)),
            "completed_unique_count": len(completed_ids),
            "unmeasured_discovered_ids": sorted(set(discovered_ids) - completed_ids),
            "acceptable_unique_count": len(acceptable_ids),
            "unresolved_intended_ids": unresolved_intended_ids,
            "resolved_unfinished_ids": resolved_unfinished_ids,
            "native_crash_unresolved_ids": native_crash_unresolved_ids,
        },
        "runtime_accounting": {
            "discovery_s": discovery_duration,
            "completed_test_duration_s": completed_test_duration,
            "runner_overhead_outside_tests_s": max(0.0, runner_duration - completed_test_duration),
            "child_startup_shutdown_unaccounted_s": max(0.0, child_runtime - discovery_duration - runner_duration) if child_runtime else None,
            "supervisor_wall_s": float(supervisor_exit.get("supervisor_duration_s") or 0.0),
            "note": "Discovery/import, runner overhead, and supervisor wall time are reported separately; category totals contain completed test durations only.",
        },
        "top20_completed_tests": top_tests,
        "top15_modules_by_cumulative_test_time": [
            {"module": module, "completed_count": module_counts[module], "duration_s": duration}
            for module, duration in top_modules
        ],
        "tests_over_5s": tests_over_5s,
        "tests_over_2s": tests_over_2s,
        "repeated_costly_setup": repeated_costly_setup,
        "wall_by_segment": wall_by_segment,
        "attempts": attempts,
        "category_totals_non_overlapping": [
            {"category": category, "duration_s": duration, "completed_count": sum(1 for event in ends if event.get("category") == category)}
            for category, duration in sorted(category_totals.items(), key=lambda item: item[1], reverse=True)
        ],
        "category_rules": {
            "accounting": "Each completed test receives exactly one category; explicit module/name rules are applied first, then measured MainWindow construction overrides to UI integration and measured subprocess.run overrides an otherwise pure-unit test to subprocess. Category totals therefore do not double count test durations.",
            "order": ["preview/gallery", "subprocess", "real MainWindow/UI integration", "controller", "filesystem/I/O", "pure unit/core"],
            "unknown": "Tests that match no explicit module/name rule are pure unit/core; unattributed is reserved for missing event metadata.",
        },
        "measured_contributions": {
            "totals_s": dict(contribution_totals),
            "records": instrumentation,
            "limitations": [
                "MainWindow.__init__ is wrapped only when ui_qt.main_window was imported during discovery.",
                "subprocess.run timing includes the called process and is independent of test category totals; it must not be added to category totals.",
                "Calls replaced by a test mock are intentionally not reported as real subprocess costs.",
            ],
        },
        "phase_timing": {
            "log_path": str(phase_log_path),
            "records": phase_events,
            "totals_s": dict(phase_totals),
            "completed_counts": dict(phase_counts),
            "malformed_log_lines": phase_malformed,
        },
        "unfinished_start_ids": sorted(set(unfinished_ids) - acceptable_ids),
        "historical_unfinished_start_ids": unfinished_ids,
        "unfinished_attempts": unfinished,
        "resolved_unfinished_ids": resolved_unfinished_ids,
        "unresolved_intended_ids": unresolved_intended_ids,
        "unfinished_instrumentation": unfinished_instrumentation,
        "malformed_log_lines": malformed,
        "scope_note": "Completed durations describe this bounded sample only; prefix samples are not whole-suite rankings.",
    }
    report["merged_summary"] = {
        "segment_count": len(wall_by_segment),
        "completed_test_count": len(ends),
        "status_counts": dict(status_counts),
        "outcome_counts": outcome_counts(status_counts),
        "measured_test_duration_s": completed_test_duration,
        "runtime_accounting": {
            "discovery_s": merged_discovery_duration,
            "runner_s": merged_runner_duration,
            "runner_overhead_outside_tests_s": max(0.0, merged_runner_duration - completed_test_duration),
            "child_runtime_s": merged_child_runtime,
            "supervisor_wall_s": merged_supervisor_wall,
            "segmented_note": "Sums separate supervisor segments; it is not an uninterrupted single-process wall time.",
        },
        "passed": merged_passed,
        "segment_terminations": segment_terminations,
        "bad_outcome_count": len(bad_outcomes),
    }
    if resume_state is not None:
        report["resume"] = {
            "source_log": resume_state["path"],
            "ancestor_logs": resume_state["ancestor_logs"],
            "previous_completed_test_count": len(ends) - len(current_ends),
            "current_completed_test_count": len(current_ends),
            "current_measured_test_duration_s": current_completed_test_duration,
            "unfinished_source_ids": resume_state.get("unfinished_ids", []),
        }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"Profile: {log_path}",
        f"Exit: {supervisor_exit.get('classification')} (code={supervisor_exit.get('exit_code')})",
        f"Completed tests: {len(ends)}; unfinished STARTs: {len(unfinished)}",
        f"Discovery: {discovery.get('duration_s', 0):.3f}s; test duration sum: {sum(float(event.get('duration_s') or 0) for event in ends):.3f}s",
        "",
        "Top completed tests:",
    ]
    lines.extend(f"{index:2d}. {event.get('duration_s', 0):8.3f}s {event.get('full_id')} [{event.get('status')}]" for index, event in enumerate(top_tests, 1))
    lines.extend(["", "Top modules:"])
    lines.extend(f"{index:2d}. {duration:8.3f}s {module}" for index, (module, duration) in enumerate(top_modules, 1))
    lines.extend(["", "Category totals (non-overlapping):"])
    lines.extend(f"- {entry['duration_s']:.3f}s {entry['category']} ({entry['completed_count']} tests)" for entry in report["category_totals_non_overlapping"])
    lines.extend([
        "",
        f"Merged segments: {report['merged_summary']['segment_count']}; outcomes: {report['outcome_counts']}; measured test duration: {completed_test_duration:.3f}s",
        "Tests over 5s:",
    ])
    lines.extend(f"- {event.get('duration_s', 0):.3f}s {event.get('full_id')}" for event in tests_over_5s)
    lines.extend(["", "Tests over 2s:"])
    lines.extend(f"- {event.get('duration_s', 0):.3f}s {event.get('full_id')}" for event in tests_over_2s)
    if unfinished:
        lines.extend(["", "Unfinished START IDs:"])
        lines.extend(f"- {entry['full_id']} ({entry['segment_id']}/{entry['attempt_id']})" for entry in unfinished)
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_supervised(
    *,
    repo: Path = ROOT,
    output_dir: Path,
    ids: Sequence[str] = (),
    modules: Sequence[str] = (),
    pattern: str = DEFAULT_PATTERN,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
    max_tests: int | None = None,
    env: dict[str, str] | None = None,
    resume_log: Path | None = None,
) -> dict[str, Any]:
    """Run one verified child and return its measured report."""

    evidence_names = (
        "events.jsonl",
        "phase-events.jsonl",
        "child.stdout.log",
        "child.stderr.log",
        "report.json",
        "report.txt",
    )
    resume_state = _load_resume_state(resume_log) if resume_log is not None else None
    segment_id = _new_segment_id()
    if output_dir.exists():
        conflicts = [name for name in evidence_names if (output_dir / name).exists()]
        if conflicts:
            raise FileExistsError(
                f"Refusing to reuse profiling evidence directory {output_dir}; "
                f"existing evidence: {', '.join(conflicts)}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "events.jsonl"
    phase_log_path = output_dir / "phase-events.jsonl"
    phase_log_path.touch()
    stdout_path = output_dir / "child.stdout.log"
    stderr_path = output_dir / "child.stderr.log"
    append = JsonlWriter(log_path)
    supervisor_started = time.perf_counter()
    append.emit(
        "SUPERVISOR_START",
        segment_id=segment_id,
        segment_started_at_utc=_utc_now(),
        supervisor_pid=os.getpid(),
        budget_seconds=budget_seconds,
        ids=list(ids),
        modules=list(modules),
        pattern=pattern,
        resume_log=resume_state["path"] if resume_state is not None else None,
        resume_ancestor_logs=resume_state["ancestor_logs"] if resume_state is not None else [],
        resume_completed_id_count=len(resume_state["completed_ids"]) if resume_state is not None else 0,
    )
    append.close()
    child_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--log",
        str(log_path),
        "--pattern",
        pattern,
        "--segment-id",
        segment_id,
    ]
    for name in ids:
        child_args.extend(["--id", name])
    for name in modules:
        child_args.extend(["--module", name])
    if max_tests is not None:
        child_args.extend(["--max-tests", str(max_tests)])
    if resume_state is not None:
        child_args.extend(["--resume", resume_state["path"]])
    child_env = dict(os.environ if env is None else env)
    child_env["PROFILE_TEST_CHILD"] = "1"
    child_env["PROFILE_PHASE_LOG"] = str(output_dir / "phase-events.jsonl")
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    child_stdout = stdout_path.open("w", encoding="utf-8")
    child_stderr = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        child_args,
        cwd=repo,
        env=child_env,
        stdout=child_stdout,
        stderr=child_stderr,
    )
    termination_reason: str | None = None
    try:
        deadline = time.monotonic() + budget_seconds
        while proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                termination_reason = "timeout"
                _terminate_child(proc)
                break
            time.sleep(min(0.2, remaining))
    except KeyboardInterrupt:
        termination_reason = "interrupted"
        _terminate_child(proc)
    finally:
        child_stdout.close()
        child_stderr.close()
    exit_code = proc.returncode
    events, _ = _read_events(log_path)
    summary_seen = any(event.get("event") == "RUN_SUMMARY" for event in events)
    classification = _classify_exit(exit_code, termination_reason, summary_seen)
    summary = next((event for event in reversed(events) if event.get("event") == "RUN_SUMMARY"), {})
    termination = _termination_for(classification, termination_reason, summary)
    supervisor_exit = {
        "segment_id": segment_id,
        "segment_ended_at_utc": _utc_now(),
        "supervisor_pid": os.getpid(),
        "child_pid": proc.pid,
        "exit_code": exit_code,
        "termination_reason": termination_reason,
        "timed_out": termination_reason == "timeout",
        "interrupted": termination_reason == "interrupted",
        "native_crash": classification == "native_crash",
        "classification": classification,
        "termination": termination,
        "summary_seen": summary_seen,
        "supervisor_duration_s": time.perf_counter() - supervisor_started,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    append = JsonlWriter(log_path)
    append.emit("SUPERVISOR_EXIT", **supervisor_exit)
    append.close()
    return _write_report(output_dir, log_path, supervisor_exit, resume_state=resume_state)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--log", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--segment-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--budget", "--timeout", dest="budget", type=float, default=DEFAULT_BUDGET_SECONDS, help="Maximum wall seconds for the child (default: 240).")
    parser.add_argument("--max-tests", type=int, default=None, help="Profile no more than this many discovered tests.")
    parser.add_argument("--max-duration", type=float, default=None, help="Alias for the bounded wall budget, useful in scripts.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--id", dest="ids", action="append", default=[], help="Explicit unittest test ID; repeatable.")
    parser.add_argument("--module", dest="modules", action="append", default=[], help="Explicit unittest module/name; repeatable.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from an existing profiler events.jsonl log.")
    return parser


def _default_output_dir(*, continuation: bool) -> Path:
    prefix = "resume" if continuation else "run"
    stamp = datetime.now().strftime(f"{prefix}-%Y%m%d-%H%M%S-%f")
    candidate = ROOT / "build" / "test-profile" / stamp
    suffix = 1
    while candidate.exists():
        candidate = ROOT / "build" / "test-profile" / f"{stamp}-{suffix}"
        suffix += 1
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.child:
        if args.log is None:
            parser.error("--child requires --log")
        return _child_main(args)
    budget = args.max_duration if args.max_duration is not None else args.budget
    if budget <= 0:
        parser.error("budget must be positive")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = _default_output_dir(continuation=args.resume is not None)
    try:
        report = run_supervised(
            repo=ROOT,
            output_dir=output_dir,
            ids=args.ids,
            modules=args.modules,
            pattern=args.pattern,
            budget_seconds=budget,
            max_tests=args.max_tests,
            resume_log=args.resume,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({
        "output_dir": str(output_dir),
        "classification": report["exit"]["classification"],
        "completed_test_count": report["completed_test_count"],
        "unfinished_start_count": len(report["unfinished_start_ids"]),
        "report": str(output_dir / "report.json"),
    }, ensure_ascii=False))
    return 0 if report["merged_summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
