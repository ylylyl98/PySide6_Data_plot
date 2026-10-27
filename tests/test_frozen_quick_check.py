"""Fast checker-contract tests; fake subprocesses never start Qt or build an EXE."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from scripts.check_frozen_quick import SmokeCase, WORKFLOWS, build_cases, inspect_report, png_evidence, run_case

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6VDAAAAAASUVORK5CYII="


class FrozenQuickCheckerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="frozen-checker-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    @staticmethod
    def shell_report(**updates):
        return dict(status="ok", errors=[], widgets_imported=False, settings_isolated=True,
            workflow="pl", visible_frames=1, timing_ms={"first_plot_frame_ms": 2.}, **updates)

    def fake_target(self, *, log_error=False, write_outputs=True):
        return [sys.executable, "-c", f'''
import base64,json,os,sys
from pathlib import Path
arguments=sys.argv[1:]
if {write_outputs!r}:
    Path(arguments[arguments.index('--report')+1]).write_text({json.dumps(json.dumps(self.shell_report()))})
    Path(arguments[arguments.index('--capture')+1]).write_bytes(base64.b64decode({PNG!r}))
if {log_error!r}:
    logs=Path(os.environ['LOCALAPPDATA'])/'DPTK'/'logs'
    logs.mkdir(parents=True)
    (logs/'app-test.log').write_text("Qt: TypeError: late shutdown callback failed")
''']

    def test_matrix_covers_nine_workflows_scientific_inputs_and_three_standalones(self):
        fixtures = {key: self.root / key for key in ("pl", "drr", "drr_csv", "peak_snapshot", "drr_snapshot", "drr_analysis_folder", "organizer_folder")}
        fixtures["compare"] = (self.root / "KK.csv", self.root / "KKp.csv")
        cases = build_cases(fixtures)
        self.assertEqual(len(cases), 18)
        self.assertEqual({case.workflow for case in cases if case.name.startswith("empty-")}, set(WORKFLOWS))
        self.assertEqual({case.kind for case in cases}, {"main", "peak", "drr", "organizer"})

    def test_report_requires_current_qml_frame_no_widgets_and_actual_plot_when_requested(self):
        self.assertEqual(inspect_report(self.shell_report(), SmokeCase("shell", ())), [])
        report = self.shell_report()
        report["widgets_imported"] = True
        self.assertIn("widgets_imported=false", " ".join(inspect_report(report, SmokeCase("shell", ()))))
        report = self.shell_report()
        report["timing_ms"] = {}
        self.assertIn("scientific Agg", " ".join(inspect_report(report, SmokeCase("plot", (), expect_plot=True))))
        report = self.shell_report()
        report["Errors"] = "single QML error"
        self.assertIn("single QML error", inspect_report(report, SmokeCase("shell", ())))
        self.assertTrue(inspect_report([], SmokeCase("shell", ())))
        pending = dict(self.shell_report(), workflow="drr", timing_ms={}, background_required=True,
            background_dialog_open=True, can_export=False, background_visible_frames=1)
        case = SmokeCase("background", (), workflow="drr", expect_background=True)
        self.assertEqual(inspect_report(pending, case), [])
        for key, value in (("background_required", False), ("background_dialog_open", False),
                           ("can_export", True), ("background_visible_frames", 0),
                           ("timing_ms", {"first_plot_frame_ms": 1.})):
            with self.subTest(key=key):
                self.assertTrue(inspect_report({**pending, key: value}, case))

    def test_drr_report_includes_postload_datasets_render_and_warning_checks(self):
        report = dict(application="QGuiApplication", qmlRoots=1, widgetsImported=False, warnings=[], error="", datasets=1, renderedFrames=2)
        case = SmokeCase("drr", (), kind="drr", expect_plot=True)
        self.assertEqual(inspect_report(report, case), [])
        report["warnings"] = ["Late QML warning"]
        self.assertIn("Late QML warning", inspect_report(report, case))

    def test_zero_exit_and_clean_report_do_not_hide_windowed_exe_log_errors(self):
        result = run_case(self.fake_target(log_error=True), SmokeCase("late-error", ()), self.root, self.root / "artifacts", 10)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["status"], "error")
        self.assertIn("TypeError:", result["failure_markers"])
        self.assertTrue(result["capture"]["width"])

    def test_failed_rerun_cannot_reuse_a_previous_report_or_capture(self):
        artifacts = self.root / "artifacts"
        case = SmokeCase("rerun", ())
        first = run_case(self.fake_target(), case, self.root / "first", artifacts, 10)
        self.assertEqual(first["status"], "ok")
        second = run_case(self.fake_target(write_outputs=False), case, self.root / "second", artifacts, 10)
        self.assertEqual(second["status"], "error")
        self.assertNotIn("application_report", second)
        self.assertNotIn("capture", second)

    def test_timeout_and_incomplete_png_fail(self):
        result = run_case([sys.executable, "-c", "import time; time.sleep(5)"], SmokeCase("timeout", ()), self.root, self.root / "artifacts", .05)
        self.assertEqual(result["status"], "error")
        self.assertTrue(any("timeout" in message for message in result["errors"]))
        path = self.root / "broken.png"
        path.write_bytes(base64.b64decode(PNG)[:-12])
        with self.assertRaises(ValueError):
            png_evidence(path)


if __name__ == "__main__":
    unittest.main()
