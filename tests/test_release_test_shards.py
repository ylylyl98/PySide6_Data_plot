"""Release gate regression tests use small temporary suites, never the full suite."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

from scripts import run_release_tests as release


RUNNER = Path(release.__file__).resolve()
PASSING = """
import unittest
class Passing(unittest.TestCase):
    def test_ok(self):
        self.assertTrue(True)
"""


class ReleaseTestShardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="release-shards-")
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        (self.repo / "suite").mkdir()
        (self.repo / "suite" / "__init__.py").write_text("", encoding="utf-8")
        self.plan = self.repo / "plan.json"
        self.results = self.repo / "results"

    def source(self, filename, source):
        path = self.repo / "suite" / filename
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        return path

    def cli(self, command, *arguments, expected=0):
        result = subprocess.run(
            [sys.executable, str(RUNNER), command, "--repo", str(self.repo), *map(str, arguments)],
            cwd=self.repo, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=45,
            encoding="utf-8", errors="replace", check=False,
        )
        if expected == 0:
            self.assertEqual(result.returncode, 0, result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def make_plan(self, shards=1, expected=0):
        return self.cli("plan", "--tests-dir", "suite", "--shards", shards,
                        "--output", self.plan, expected=expected)

    def run_shard(self, shard=0, timeout=20, expected=0):
        return self.cli("run", "--plan", self.plan, "--shard", shard,
                        "--output-dir", self.results / f"artifact-{shard}",
                        "--timeout", timeout, expected=expected)

    def verify(self, expected=0):
        return self.cli("verify", "--plan", self.plan,
                        "--results-dir", self.results, expected=expected)

    def successful_run(self):
        self.source("test_ok.py", PASSING)
        self.make_plan()
        self.run_shard()
        self.verify()

    def timing_hints(self, seconds_by_key):
        plan = release.read_json(self.plan)
        modules = {test["id"]: test["module"] for test in plan["tests"]}
        units = {}
        for unit in plan["units"]:
            key = unit["timing_key"]
            if key not in seconds_by_key:
                continue
            source = self.repo / (modules[unit["test_ids"][0]].replace(".", "/") + ".py")
            units[key] = {
                "duration_seconds": seconds_by_key[key],
                "source_sha256_lf": hashlib.sha256(source.read_text(encoding="utf-8").encode()).hexdigest(),
                "test_ids_sha256": release.digest(sorted(unit["test_ids"])),
            }
        data = {"schema_version": 1,
                "source": {"workflow_run_id": 1, "commit": "a" * 40, "plan_id": "b" * 64,
                           "recorded_at_utc": "2026-10-24T01:36:13Z",
                           "measurement": "unit_wall_seconds_including_startup"},
                "units": units}
        path = self.repo / release.TIMING_HINTS_PATH
        release.write_json(path, data)
        return path, data

    def test_measured_weights_balance_modules_and_survive_id_and_checkout_changes(self):
        for name in ("a", "b", "c", "d"):
            self.source(f"test_{name}.py", PASSING)
        self.make_plan(shards=2)
        original = release.read_json(self.plan)
        self.timing_hints({"suite.test_a": 2, "suite.test_b": 100,
                           "suite.test_c": 20, "suite.test_d": 80})
        self.make_plan(shards=2)
        measured = release.read_json(self.plan)
        self.assertCountEqual([s["weight_seconds"] for s in measured["shards"]], [102, 100])
        self.assertTrue(all(u["weight_basis"] == "measured_unit_wall" for u in measured["units"]))
        self.assertCountEqual([t["id"] for t in measured["tests"]], [t["id"] for t in original["tests"]])
        for shard in range(2):
            self.run_shard(shard)
        self.verify()

        # Ordinal unit IDs shift; an LF -> CRLF checkout must not lose the hint.
        old_id = next(u["id"] for u in measured["units"] if u["timing_key"] == "suite.test_b")
        self.source("test_0new.py", PASSING)
        path = self.repo / "suite" / "test_b.py"
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.make_plan(shards=2)
        changed = release.read_json(self.plan)
        by_key = {u["timing_key"]: u for u in changed["units"]}
        self.assertNotEqual(by_key["suite.test_b"]["id"], old_id)
        self.assertEqual(by_key["suite.test_b"]["weight_seconds"], 100)
        self.assertEqual(by_key["suite.test_0new"]["weight_basis"], "estimated_tests_plus_startup")
        self.assertEqual(len(changed["tests"]), 5)

    def test_changed_source_and_inherited_membership_fall_back_without_hiding_failures(self):
        base = PASSING.replace("Passing", "Base")
        self.source("fixture_base.py", base)
        self.source("test_child.py", "from suite.fixture_base import Base\nclass Derived(Base): pass\n")
        self.source("test_control.py", PASSING)
        self.make_plan()
        self.timing_hints({"suite.fixture_base": 10, "suite.test_child": 20, "suite.test_control": 30})
        self.source("fixture_base.py", base + "\n    def test_added(self):\n        self.fail('new regression')\n")
        self.make_plan()
        plan = release.read_json(self.plan)
        by_key = {u["timing_key"]: u for u in plan["units"]}
        for key in ("suite.fixture_base", "suite.test_child"):
            self.assertEqual(by_key[key]["weight_basis"], "estimated_tests_plus_startup")
            self.assertEqual(by_key[key]["weight_seconds"], 4)
        self.assertEqual(by_key["suite.test_control"]["weight_seconds"], 30)
        self.assertEqual(len(plan["tests"]), 5)
        self.run_shard(expected=1)
        self.verify(expected=1)

    def test_split_test_timings_preserve_per_case_process_isolation(self):
        self.source("test_drr_bn_joint.py", """
            import os
            import unittest
            class Split(unittest.TestCase):
                def test_a_set_environment(self):
                    os.environ['RELEASE_SPLIT_TIMING_STATE'] = 'changed'
                def test_b_environment_is_isolated(self):
                    self.assertNotIn('RELEASE_SPLIT_TIMING_STATE', os.environ)
        """)
        self.make_plan()
        first = release.read_json(self.plan)
        ids = sorted(t["id"] for t in first["tests"])
        self.timing_hints({ids[0]: 30, ids[1]: 5})
        self.make_plan()
        plan = release.read_json(self.plan)
        self.assertEqual(len(plan["units"]), 2)
        self.assertCountEqual([u["timing_key"] for u in plan["units"]], ids)
        self.assertCountEqual([u["weight_seconds"] for u in plan["units"]], [30, 5])
        self.run_shard()
        self.verify()

    def test_invalid_timing_hints_fail_closed_instead_of_silently_changing_scheduling(self):
        self.source("test_ok.py", PASSING)
        self.make_plan()
        path, valid = self.timing_hints({"suite.test_ok": 1})
        for field, invalid in (("duration_seconds", 0), ("duration_seconds", -1),
                               ("duration_seconds", True), ("duration_seconds", float("nan")),
                               ("duration_seconds", float("inf")), ("duration_seconds", "10"),
                               ("source_sha256_lf", "wrong"), ("test_ids_sha256", "wrong")):
            with self.subTest(field=field, value=invalid):
                data = json.loads(json.dumps(valid))
                data["units"]["suite.test_ok"][field] = invalid
                release.write_json(path, data)
                with self.assertRaises(ValueError):
                    release.read_timing_hints(self.repo)
        for text in ('{"schema_version": 1, "schema_version": 1}', '[1]',
                     json.dumps({**valid, "schema_version": 2}), json.dumps({**valid, "units": []})):
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    release.read_timing_hints(self.repo)
        self.plan.unlink()
        self.assertIn("Invalid timing hints", self.make_plan(expected=1).stdout)
        self.assertFalse(self.plan.exists())

    def test_plan_discovers_new_tests_and_assigns_every_unique_id_once(self):
        self.source("test_first.py", PASSING)
        self.make_plan(shards=4)
        first = release.read_json(self.plan)
        self.source("test_new.py", PASSING.replace("test_ok", "test_added"))
        self.make_plan(shards=4)
        current = release.read_json(self.plan)
        ids = [test["id"] for test in current["tests"]]
        assigned = [name for shard in current["shards"] for name in shard["test_ids"]]
        self.assertEqual(len(ids), 2)
        self.assertCountEqual(ids, assigned)
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertNotEqual(current["plan_id"], first["plan_id"])
        self.assertTrue(all(test["weight_basis"] == "estimated_test_count" for test in current["tests"]))

    def test_imported_test_classes_are_normalized_and_documented(self):
        self.source("test_source.py", PASSING)
        self.source("test_import.py", "from suite.test_source import Passing\n")
        self.make_plan()
        plan = release.read_json(self.plan)
        self.assertEqual(len(plan["tests"]), 1)
        self.assertEqual(plan["discovery_duplicates"], {"suite.test_source.Passing.test_ok": 2})
        self.run_shard()
        self.verify()

    def test_parameterized_duplicate_ids_are_rejected(self):
        self.source("test_custom.py", """
            import unittest
            class Custom(unittest.TestCase):
                def test_value(self):
                    pass
            def load_tests(loader, tests, pattern):
                a, b = Custom('test_value'), Custom('test_value')
                a.value, b.value = 1, 2
                return unittest.TestSuite([a, b])
        """)
        result = self.make_plan(expected=1)
        self.assertIn("Unsupported load_tests", result.stdout)

    def test_unique_parameterized_instance_and_global_state_hooks_are_rejected(self):
        for body in ("test.must_fail = True", "Custom.must_fail = True"):
            with self.subTest(body=body):
                self.source("test_custom.py", f"""
                    import unittest
                    class Custom(unittest.TestCase):
                        def test_value(self):
                            self.assertFalse(getattr(self, 'must_fail', False))
                    def load_tests(loader, tests, pattern):
                        test = Custom('test_value')
                        {body}
                        return unittest.TestSuite([test])
                """)
                self.assertIn("Unsupported load_tests", self.make_plan(expected=1).stdout)

    def test_custom_testcase_construction_is_rejected(self):
        self.source("test_custom.py", """
            import unittest
            class Custom(unittest.TestCase):
                def __init__(self, methodName='runTest'):
                    super().__init__(methodName)
                    self.special_state = True
                def test_value(self):
                    pass
        """)
        self.assertIn("Custom TestCase construction", self.make_plan(expected=1).stdout)

    def test_modules_execute_in_independent_processes(self):
        self.source("test_a.py", """
            import os
            import unittest
            class State(unittest.TestCase):
                def test_set_environment(self):
                    os.environ['RELEASE_SHARD_FIXTURE_STATE'] = 'changed'
        """)
        self.source("test_b.py", """
            import os
            import unittest
            class State(unittest.TestCase):
                def test_environment_is_isolated(self):
                    self.assertNotIn('RELEASE_SHARD_FIXTURE_STATE', os.environ)
        """)
        self.make_plan()
        self.run_shard()
        self.verify()

    def test_discovery_import_errors_and_empty_suites_fail(self):
        self.assertIn("no tests", self.make_plan(expected=1).stdout)
        self.source("test_broken.py", "raise ImportError('fixture dependency missing')\n")
        self.assertIn("discovery failed", self.make_plan(expected=1).stdout)

    def test_four_shards_accept_success_skip_expected_failure_and_fixture_skip(self):
        self.source("test_outcomes.py", """
            import unittest
            class Outcomes(unittest.TestCase):
                def test_ok(self):
                    with self.subTest(value=1):
                        self.assertEqual(1, 1)
                @unittest.skip('existing skip')
                def test_skipped(self):
                    pass
                @unittest.expectedFailure
                def test_expected_failure(self):
                    self.fail('known failure')
            class FixtureSkip(unittest.TestCase):
                @classmethod
                def setUpClass(cls):
                    raise unittest.SkipTest('fixture unavailable')
                def test_class_skipped(self):
                    pass
        """)
        self.source("test_module_skip.py", """
            import unittest
            def setUpModule():
                raise unittest.SkipTest('module unavailable')
            class ModuleSkip(unittest.TestCase):
                def test_module_skipped(self):
                    pass
        """)
        self.source("test_other.py", PASSING)
        self.make_plan(shards=4)
        for shard in range(4):
            self.run_shard(shard)
        result = self.verify()
        self.assertIn("Verified 6 unique tests across 4 shards", result.stdout)

    def test_assertion_failure_blocks_run_and_verification(self):
        self.source("test_bad.py", PASSING.replace("self.assertTrue(True)", "self.fail('broken')"))
        self.make_plan()
        self.run_shard(expected=1)
        self.verify(expected=1)

    def test_subtest_failure_and_unexpected_success_block_run(self):
        self.source("test_bad.py", """
            import unittest
            class Bad(unittest.TestCase):
                def test_subtest(self):
                    with self.subTest(value=1):
                        self.fail('subtest failed')
                @unittest.expectedFailure
                def test_unexpected_success(self):
                    pass
        """)
        self.make_plan()
        self.run_shard(expected=1)
        reports = list(self.results.rglob("*.result.json"))
        statuses = {record["status"] for record in release.read_json(reports[0])["records"]}
        self.assertEqual(statuses, {"failure", "unexpected_success"})
        self.verify(expected=1)

    def test_class_setup_error_cannot_pass_with_zero_tests_run(self):
        self.source("test_bad.py", """
            import unittest
            class Bad(unittest.TestCase):
                @classmethod
                def setUpClass(cls):
                    raise RuntimeError('setup failed')
                def test_not_run(self):
                    pass
        """)
        self.make_plan()
        self.run_shard(expected=1)
        self.verify(expected=1)

    def test_timeout_blocks_release_and_retains_supervisor_evidence(self):
        self.source("test_sleep.py", """
            import time
            import unittest
            class Sleep(unittest.TestCase):
                def test_hang(self):
                    time.sleep(30)
        """)
        self.make_plan()
        self.run_shard(timeout=2, expected=1)
        summary = release.read_json(self.results / "artifact-0" / "summary.json")
        self.assertTrue(summary["units"][0]["timed_out"])
        self.assertFalse(summary["successful"])
        self.verify(expected=1)

    def test_timeout_also_stops_spawned_descendants(self):
        heartbeat = self.repo / "heartbeat.txt"
        child_code = (
            "import pathlib,time; p=pathlib.Path(" + repr(str(heartbeat)) + "); "
            "exec(\"while True:\\n p.write_text(str(time.time()))\\n time.sleep(0.05)\")"
        )
        self.source("test_spawn.py", f"""
            import subprocess
            import sys
            import time
            import unittest
            class Spawn(unittest.TestCase):
                def test_spawn(self):
                    subprocess.Popen([sys.executable, '-c', {child_code!r}])
                    time.sleep(30)
        """)
        self.make_plan()
        self.run_shard(timeout=8, expected=1)
        self.assertTrue(heartbeat.exists(), "Descendant never started")
        before = heartbeat.read_text()
        time.sleep(0.3)
        self.assertEqual(heartbeat.read_text(), before, "Timed-out descendant is still running")
        self.verify(expected=1)

    def test_zero_exit_without_normal_completion_report_is_rejected(self):
        self.source("test_exit.py", """
            import os
            import unittest
            class Exit(unittest.TestCase):
                def test_exit(self):
                    os._exit(0)
        """)
        self.make_plan()
        self.run_shard(expected=1)
        summary = release.read_json(self.results / "artifact-0" / "summary.json")
        self.assertEqual(summary["units"][0]["returncode"], 0)
        self.assertFalse(summary["successful"])
        self.verify(expected=1)

    def test_abnormal_process_exit_is_rejected(self):
        self.source("test_crash.py", """
            import os
            import unittest
            class Crash(unittest.TestCase):
                def test_crash(self):
                    os._exit(23)
        """)
        self.make_plan()
        self.run_shard(expected=1)
        self.verify(expected=1)

    def test_changed_sources_and_modified_plan_are_rejected(self):
        path = self.source("test_ok.py", PASSING)
        self.make_plan()
        path.write_text(PASSING + "\n# changed\n", encoding="utf-8")
        self.assertIn("sources differ", self.run_shard(expected=1).stdout)
        self.make_plan()
        plan = release.read_json(self.plan)
        plan["shards"][0]["test_ids"] = []
        release.write_json(self.plan, plan)
        self.assertIn("digest mismatch", self.run_shard(expected=1).stdout)
        plan["plan_id"] = release.digest({k: v for k, v in plan.items() if k != "plan_id"})
        release.write_json(self.plan, plan)
        self.assertIn("Missing or unexpected assigned test", self.run_shard(expected=1).stdout)

    def test_missing_duplicate_failed_or_forged_results_block_verification(self):
        self.successful_run()
        directory = self.results / "artifact-0"
        summary_path = directory / "summary.json"
        summary = release.read_json(summary_path)
        summary_path.rename(directory / "hidden-summary.json")
        self.assertIn("Missing or extra", self.verify(expected=1).stdout)
        (directory / "hidden-summary.json").rename(summary_path)
        duplicate = self.results / "duplicate"
        shutil.copytree(directory, duplicate)
        self.assertIn("Missing or extra", self.verify(expected=1).stdout)
        shutil.rmtree(duplicate)
        for change in ({"successful": False}, {"plan_id": "wrong"},
                       {"test_ids": summary["test_ids"] * 2}):
            with self.subTest(change=change):
                release.write_json(summary_path, {**summary, **change})
                self.verify(expected=1)
        release.write_json(summary_path, summary)
        report_path = directory / summary["units"][0]["report"]
        original = release.read_json(report_path)
        for change in ({"normal_completion": False}, {"tests_run": 0}, {"records": []},
                       {"records": original["records"] * 2}):
            with self.subTest(change=change):
                release.write_json(report_path, {**original, **change})
                self.verify(expected=1)
        report = json.loads(json.dumps(original))
        report["records"][0]["status"] = "failure"
        release.write_json(report_path, report)
        self.verify(expected=1)
        release.write_json(report_path, original)
        (directory / summary["units"][0]["events"]).write_text("", encoding="utf-8")
        self.assertIn("event log differs", self.verify(expected=1).stdout)


if __name__ == "__main__":
    unittest.main()
