"""Cheap synthetic coverage for the bounded profiler's resume contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import profile_test_suite as profiler


def _write_log(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps({"schema_version": 1, **event}) + "\n" for event in events),
        encoding="utf-8",
    )


def _render_report(root: Path, log: Path, exit_event: dict[str, object], **kwargs: object) -> dict[str, object]:
    output = root / "report"
    output.mkdir()
    return profiler._write_report(output, log, exit_event, **kwargs)


class _SyntheticTest:
    def __init__(self, full_id: str) -> None:
        self.full_id = full_id

    def id(self) -> str:
        return self.full_id


class ProfileResumeTests(unittest.TestCase):
    def test_resume_filters_completed_ids_in_discovery_order(self) -> None:
        tests = [_SyntheticTest("test.a"), _SyntheticTest("test.b"), _SyntheticTest("test.c")]

        remaining = profiler._filter_completed_tests(tests, {"test.b"})

        self.assertEqual([test.id() for test in remaining], ["test.a", "test.c"])

    def test_resume_chaining_accumulates_all_ancestor_end_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root.jsonl"
            continuation = Path(temp_dir) / "continuation.jsonl"
            discovered = ["test.a", "test.b", "test.c"]
            _write_log(
                root,
                [
                    {"event": "DISCOVERY_END", "test_ids": discovered},
                    {"event": "TEST_END", "full_id": "test.a", "status": "success", "duration_s": 1.0},
                    {"event": "TEST_END", "full_id": "test.b", "status": "skip", "duration_s": 0.0},
                ],
            )
            _write_log(
                continuation,
                [
                    {
                        "event": "SUPERVISOR_START",
                        "resume_log": str(root),
                        "resume_ancestor_logs": [str(root)],
                    },
                    {"event": "DISCOVERY_END", "test_ids": discovered},
                    {"event": "TEST_START", "full_id": "test.c"},
                    {"event": "TEST_END", "full_id": "test.c", "status": "success", "duration_s": 2.0},
                ],
            )

            state = profiler._load_resume_state(continuation)

            self.assertEqual(state["completed_ids"], {"test.a", "test.b", "test.c"})
            self.assertEqual(state["ancestor_logs"], [str(root), str(continuation)])

    def test_resume_does_not_treat_unfinished_start_as_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "crashed.jsonl"
            _write_log(
                path,
                [
                    {"event": "DISCOVERY_END", "test_ids": ["test.a", "test.b"]},
                    {"event": "TEST_START", "full_id": "test.a"},
                    {"event": "TEST_END", "full_id": "test.a", "status": "success", "duration_s": 1.0},
                    {"event": "TEST_START", "full_id": "test.b"},
                ],
            )

            state = profiler._load_resume_state(path)

            self.assertEqual(state["completed_ids"], {"test.a"})
            self.assertEqual(state["unfinished_ids"], ["test.b"])

    def test_resume_rejects_malformed_source_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.jsonl"
            path.write_text("not-json\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                profiler._load_resume_state(path)

    def test_resume_rejects_source_ids_missing_from_current_discovery(self) -> None:
        state = {"source_discovered_ids": ["test.a", "test.removed"]}

        with self.assertRaisesRegex(ValueError, "source mismatch"):
            profiler._validate_resume_source(state, ["test.a", "test.new"])

    def test_budget_interrupted_attempt_is_resolved_by_successful_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "budget.jsonl"
            second = root / "continuation.jsonl"
            _write_log(
                first,
                [
                    {"event": "SUPERVISOR_START", "segment_id": "seg-budget"},
                    {"event": "DISCOVERY_END", "test_ids": ["test.a"]},
                    {"event": "TEST_START", "segment_id": "seg-budget", "attempt_id": "a1", "full_id": "test.a"},
                    {"event": "SUPERVISOR_EXIT", "segment_id": "seg-budget", "termination": "BUDGET_EXPIRED", "classification": "timeout"},
                ],
            )
            _write_log(
                second,
                [
                    {"event": "SUPERVISOR_START", "segment_id": "seg-success", "resume_log": str(first)},
                    {"event": "DISCOVERY_END", "test_ids": ["test.a"]},
                    {"event": "TEST_START", "segment_id": "seg-success", "attempt_id": "a1", "full_id": "test.a"},
                    {"event": "TEST_END", "segment_id": "seg-success", "attempt_id": "a1", "full_id": "test.a", "status": "success", "duration_s": 1.0},
                    {"event": "SUPERVISOR_EXIT", "segment_id": "seg-success", "termination": "COMPLETED", "classification": "success"},
                ],
            )

            state = profiler._load_resume_state(second)
            report = _render_report(root, second, {"classification": "success", "termination": "COMPLETED"}, resume_state=state)

            self.assertEqual(state["completed_ids"], {"test.a"})
            self.assertEqual(state["unfinished_ids"], ["test.a"])
            self.assertEqual(report["coverage"]["unresolved_intended_ids"], [])
            self.assertEqual(report["coverage"]["resolved_unfinished_ids"], ["test.a"])
            self.assertEqual(report["unfinished_start_ids"], [])
            self.assertTrue(report["merged_summary"]["passed"])

    def test_failure_in_earlier_segment_stays_sticky_after_later_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "failure.jsonl"
            second = root / "success.jsonl"
            _write_log(first, [
                {"event": "DISCOVERY_END", "test_ids": ["test.a"]},
                {"event": "TEST_START", "segment_id": "seg-1", "attempt_id": "a1", "full_id": "test.a"},
                {"event": "TEST_END", "segment_id": "seg-1", "attempt_id": "a1", "full_id": "test.a", "status": "failure", "duration_s": 1.0},
            ])
            _write_log(second, [
                {"event": "SUPERVISOR_START", "segment_id": "seg-2", "resume_log": str(first)},
                {"event": "DISCOVERY_END", "test_ids": ["test.a"]},
                {"event": "TEST_START", "segment_id": "seg-2", "attempt_id": "a1", "full_id": "test.a"},
                {"event": "TEST_END", "segment_id": "seg-2", "attempt_id": "a1", "full_id": "test.a", "status": "success", "duration_s": 1.0},
            ])

            state = profiler._load_resume_state(second)
            report = _render_report(root, second, {"classification": "success", "termination": "COMPLETED"}, resume_state=state)

            self.assertFalse(report["merged_summary"]["passed"])
            self.assertEqual(report["outcome_counts"]["fail"], 1)

    def test_expected_failure_is_an_acceptable_pass_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log = root / "expected.jsonl"
            _write_log(log, [
                {"event": "DISCOVERY_END", "test_ids": ["test.a"]},
                {"event": "TEST_START", "segment_id": "seg", "attempt_id": "a1", "full_id": "test.a"},
                {"event": "TEST_END", "segment_id": "seg", "attempt_id": "a1", "full_id": "test.a", "status": "expected_failure", "duration_s": 1.0},
            ])

            report = _render_report(root, log, {"classification": "success", "termination": "COMPLETED"})

            self.assertEqual(report["outcome_counts"], {"pass": 1, "skip": 0, "fail": 0})
            self.assertTrue(report["merged_summary"]["passed"])

    def test_unexpected_success_is_an_unacceptable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log = root / "unexpected.jsonl"
            _write_log(log, [
                {"event": "DISCOVERY_END", "test_ids": ["test.a"]},
                {"event": "TEST_START", "segment_id": "seg", "attempt_id": "a1", "full_id": "test.a"},
                {"event": "TEST_END", "segment_id": "seg", "attempt_id": "a1", "full_id": "test.a", "status": "unexpected_success", "duration_s": 1.0},
            ])

            report = _render_report(root, log, {"classification": "success", "termination": "COMPLETED"})

            self.assertEqual(report["outcome_counts"], {"pass": 0, "skip": 0, "fail": 1})
            self.assertFalse(report["merged_summary"]["passed"])

    def test_interrupted_start_and_resumed_attempt_do_not_cross_bind(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "interrupted.jsonl"
            second = root / "resumed.jsonl"
            _write_log(first, [
                {"event": "DISCOVERY_END", "test_ids": ["test.a"]},
                {"event": "TEST_START", "segment_id": "seg-old", "attempt_id": "old", "full_id": "test.a"},
            ])
            _write_log(second, [
                {"event": "SUPERVISOR_START", "segment_id": "seg-new", "resume_log": str(first)},
                {"event": "DISCOVERY_END", "test_ids": ["test.a"]},
                {"event": "TEST_START", "segment_id": "seg-new", "attempt_id": "new", "full_id": "test.a"},
                {"event": "TEST_END", "segment_id": "seg-new", "attempt_id": "new", "full_id": "test.a", "status": "success", "duration_s": 1.0},
            ])

            state = profiler._load_resume_state(second)
            report = _render_report(root, second, {"classification": "success", "termination": "COMPLETED"}, resume_state=state)

            self.assertEqual(len(report["unfinished_attempts"]), 1)
            self.assertEqual(report["unfinished_attempts"][0]["attempt_id"], "old")
            self.assertEqual(report["coverage"]["resolved_unfinished_ids"], ["test.a"])
            self.assertEqual(report["coverage"]["unresolved_intended_ids"], [])

    def test_same_test_id_in_different_segments_has_distinct_attempt_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "one.jsonl"
            second = root / "two.jsonl"
            for path, segment, attempt in ((first, "seg-1", "a1"), (second, "seg-2", "a1")):
                _write_log(path, [
                    {"event": "DISCOVERY_END", "test_ids": ["test.a"]},
                    {"event": "TEST_START", "segment_id": segment, "attempt_id": attempt, "full_id": "test.a"},
                    {"event": "TEST_END", "segment_id": segment, "attempt_id": attempt, "full_id": "test.a", "status": "success", "duration_s": 1.0},
                ])

            state = profiler._load_resume_state(second)
            report = _render_report(root, second, {"classification": "success", "termination": "COMPLETED"}, resume_state={**state, "ancestor_logs": [str(first), str(second)], "events": profiler._read_events(first)[0] + state["events"]})
            identities = {(event["segment_id"], event["attempt_id"], event["full_id"]) for event in report["attempts"]}

            self.assertEqual(identities, {("seg-1", "a1", "test.a"), ("seg-2", "a1", "test.a")})

    def test_instrumentation_operation_identity_is_scoped_to_segment_and_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "one.jsonl"
            second = root / "two.jsonl"
            for path, segment, attempt in ((first, "seg-1", "a1"), (second, "seg-2", "a1")):
                _write_log(path, [
                    {"event": "DISCOVERY_END", "test_ids": ["test.a"]},
                    {"event": "TEST_START", "segment_id": segment, "attempt_id": attempt, "full_id": "test.a"},
                    {"event": "INSTRUMENTATION_START", "segment_id": segment, "attempt_id": attempt, "full_id": "test.a", "operation_id": 1, "kind": "MainWindow.__init__"},
                    {"event": "INSTRUMENTATION_END", "segment_id": segment, "attempt_id": attempt, "full_id": "test.a", "operation_id": 1, "kind": "MainWindow.__init__", "duration_s": 0.1},
                    {"event": "TEST_END", "segment_id": segment, "attempt_id": attempt, "full_id": "test.a", "status": "success", "duration_s": 1.0, "instrumentation": [{"segment_id": segment, "attempt_id": attempt, "operation_id": 1, "full_id": "test.a", "kind": "MainWindow.__init__", "duration_s": 0.1}]},
                ])

            state = profiler._load_resume_state(second)
            report = _render_report(root, second, {"classification": "success", "termination": "COMPLETED"}, resume_state={**state, "ancestor_logs": [str(first), str(second)], "events": profiler._read_events(first)[0] + state["events"]})
            identities = {(record["segment_id"], record["attempt_id"], record["operation_id"]) for record in report["measured_contributions"]["records"]}

            self.assertEqual(identities, {("seg-1", "a1", 1), ("seg-2", "a1", 1)})
            self.assertEqual(report["unfinished_instrumentation"], [])


if __name__ == "__main__":
    unittest.main()
