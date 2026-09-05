from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


SCALES = ("1", "1.25", "1.5", "2")
EXPECTED_WORKFLOWS = (
    "PL",
    "DRR",
    "Compare",
    "Power",
    "MCD",
    "MCD Peak Shift",
    "SHG",
    "Tools",
)


class Phase7LayoutAuditTests(unittest.TestCase):
    """Parent process for the isolated, real-window layout audit probes."""

    def test_real_main_window_layout_audit_all_scales(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        expected_python = repo / ".venv" / "Scripts" / "python.exe"
        self.assertEqual(Path(sys.executable).resolve(), expected_python.resolve())
        probe = repo / "tests" / "phase7_layout_audit.py"
        aggregate: list[dict] = []
        child_errors: list[dict] = []
        schema_errors: list[dict] = []
        finding_schema_errors: list[dict] = []
        runs: list[dict] = []
        for scale in SCALES:
            env = os.environ.copy()
            env["QT_QPA_PLATFORM"] = "offscreen"
            env["QT_SCALE_FACTOR"] = scale
            proc = subprocess.run(
                [sys.executable, str(probe), "--scale", scale],
                cwd=repo,
                env=env,
                text=True,
                capture_output=True,
                timeout=180,
            )
            runs.append({"scale": scale, "exit_code": proc.returncode, "command": [sys.executable, str(probe), "--scale", scale]})
            try:
                payload = json.loads(proc.stdout)
            except json.JSONDecodeError:
                payload = {
                    "schema_version": None,
                    "scale": scale,
                    "audited_workflows": [],
                    "findings": [],
                    "counts": {},
                    "screenshot_paths": [],
                    "fatal_error": "child emitted non-JSON stdout",
                    "stdout": proc.stdout[-2000:],
                    "stderr": proc.stderr[-2000:],
                }
            if proc.returncode != 0 or payload.get("fatal_error"):
                child_errors.append(
                    {
                        "scale": scale,
                        "exit_code": proc.returncode,
                        "fatal_error": payload.get("fatal_error"),
                        "stderr": proc.stderr[-2000:],
                    }
                )
            problems: list[str] = []
            if payload.get("schema_version") != 1:
                problems.append("schema_version")
            if str(payload.get("scale")) != scale:
                problems.append("scale")
            if payload.get("audited_workflows") != list(EXPECTED_WORKFLOWS):
                problems.append("audited_workflows")
            if not isinstance(payload.get("findings"), list):
                problems.append("findings")
            if not isinstance(payload.get("intended_scrolling"), list):
                problems.append("intended_scrolling")
            if not isinstance(payload.get("counts"), dict):
                problems.append("counts")
            if not isinstance(payload.get("screenshot_paths"), list):
                problems.append("screenshot_paths")
            if not isinstance(payload.get("expanded_headers"), dict):
                problems.append("expanded_headers")
            coverage = payload.get("coverage")
            if not isinstance(coverage, dict):
                problems.append("coverage")
            else:
                missing_workflows = [workflow for workflow in EXPECTED_WORKFLOWS if workflow not in coverage]
                if missing_workflows:
                    problems.append(f"coverage_missing_workflows:{','.join(missing_workflows)}")
                for workflow in EXPECTED_WORKFLOWS:
                    entry = coverage.get(workflow)
                    if not isinstance(entry, dict):
                        continue
                    required_coverage_keys = {
                        "layouts_audited",
                        "visible_interactive_controls",
                        "spinboxes_audited",
                        "spinbox_coverage",
                        "spinbox_zero_workflow",
                        "spinbox_values_measured",
                        "spinbox_candidates",
                        "text_controls_audited",
                        "text_control_coverage",
                    }
                    missing_keys = sorted(required_coverage_keys.difference(entry))
                    if missing_keys:
                        problems.append(f"coverage_keys:{workflow}:{','.join(missing_keys)}")
                        continue
                    if entry["layouts_audited"] <= 0 or entry["visible_interactive_controls"] <= 0:
                        problems.append(f"coverage_empty:{workflow}")
                    if entry["spinbox_coverage"] == "none":
                        if not entry["spinbox_zero_workflow"] or entry["spinboxes_audited"] != 0 or entry["spinbox_values_measured"] != 0:
                            problems.append(f"coverage_zero_spinbox_invalid:{workflow}")
                    elif entry["spinbox_coverage"] == "audited":
                        if entry["spinboxes_audited"] <= 0 or entry["spinbox_values_measured"] <= 0:
                            problems.append(f"coverage_spinbox_empty:{workflow}")
                        candidates = entry.get("spinbox_candidates")
                        if not isinstance(candidates, list):
                            problems.append(f"coverage_candidates_type:{workflow}")
                        else:
                            by_spinbox: dict[str, set[str]] = {}
                            bounds: dict[str, tuple[float, float]] = {}
                            for candidate in candidates:
                                if not isinstance(candidate, dict):
                                    problems.append(f"coverage_candidate_entry:{workflow}")
                                    continue
                                path = candidate.get("widget_path")
                                source = candidate.get("source")
                                if not isinstance(path, str) or not isinstance(source, str):
                                    problems.append(f"coverage_candidate_identity:{workflow}")
                                    continue
                                by_spinbox.setdefault(path, set()).add(source)
                                try:
                                    bounds[path] = (float(candidate["spin_minimum"]), float(candidate["spin_maximum"]))
                                except (KeyError, TypeError, ValueError):
                                    problems.append(f"coverage_candidate_bounds:{workflow}")
                            if len(by_spinbox) != entry["spinboxes_audited"]:
                                problems.append(f"coverage_spinbox_identity_count:{workflow}")
                            for path, sources in by_spinbox.items():
                                if not {"current", "minimum", "maximum"}.issubset(sources):
                                    problems.append(f"coverage_candidate_required_sources:{workflow}:{path}")
                                lower, upper = bounds.get(path, (0.0, 0.0))
                                if lower <= 0 <= upper and "zero" not in sources:
                                    problems.append(f"coverage_candidate_zero_missing:{workflow}:{path}")
                                if lower <= -12 <= upper and "negative_twelve" not in sources:
                                    problems.append(f"coverage_candidate_negative_twelve_missing:{workflow}:{path}")
                    else:
                        problems.append(f"coverage_spinbox_status:{workflow}")
                    if entry["text_control_coverage"] == "audited" and entry["text_controls_audited"] <= 0:
                        problems.append(f"coverage_text_empty:{workflow}")
                    elif entry["text_control_coverage"] not in {"audited", "none"}:
                        problems.append(f"coverage_text_status:{workflow}")
            effective_dpr = payload.get("effective_device_pixel_ratio")
            try:
                if effective_dpr is None or abs(float(effective_dpr) - float(scale)) > 0.2:
                    problems.append("effective_device_pixel_ratio")
            except (TypeError, ValueError):
                problems.append("effective_device_pixel_ratio")
            if not isinstance(payload.get("font"), dict):
                problems.append("font")
            if problems:
                schema_errors.append({"scale": scale, "problems": problems, "payload": payload})
            required_finding_keys = {
                "workflow",
                "scale",
                "check",
                "reason",
                "widget_class",
                "objectName",
                "widget_path",
                "parent_class",
                "parent_objectName",
                "layout_class",
                "layout_identifier",
                "widget_rect",
                "content_rect",
                "screenshot_path",
                "coordinate_spaces",
            }
            for finding in payload.get("findings", []):
                missing = sorted(required_finding_keys.difference(finding))
                if missing:
                    finding_schema_errors.append({"scale": scale, "missing": missing, "finding": finding})
                if finding.get("screenshot_path"):
                    screenshot = Path(finding["screenshot_path"])
                    if not screenshot.is_absolute() or not screenshot.exists():
                        finding_schema_errors.append({"scale": scale, "problem": "invalid screenshot_path", "finding": finding})
            aggregate.append(payload)

        findings = [finding for payload in aggregate for finding in payload.get("findings", [])]
        # Batch B requires every non-exempt actionable finding to be resolved.
        # Intentional exemptions are asserted separately and are not emitted as
        # findings by the audit probe.
        if child_errors or schema_errors or finding_schema_errors or findings:
            grouped: dict[str, list[dict]] = {}
            for finding in findings:
                grouped.setdefault(
                    f"{finding.get('workflow')}@scale={finding.get('scale')}", []
                ).append(finding)
            counts_by_workflow = {
                key: dict(Counter(item.get("check", "unknown") for item in items))
                for key, items in grouped.items()
            }
            report = {
                "runs": runs,
                "child_errors": child_errors,
                "schema_errors": schema_errors,
                "finding_schema_errors": finding_schema_errors,
                "grouped_findings": grouped,
                "counts_by_workflow": counts_by_workflow,
                "coverage_by_scale": {
                    payload.get("scale"): payload.get("coverage", {}) for payload in aggregate
                },
                "scale_payloads": aggregate,
            }
            coverage_summary = {
                scale: (
                    {
                        workflow: {
                            key: value
                            for key, value in entry.items()
                            if key != "spinbox_candidates"
                        }
                        for workflow, entry in coverage.items()
                        if isinstance(entry, dict)
                    }
                    if isinstance(coverage, dict)
                    else {}
                )
                for scale, coverage in report["coverage_by_scale"].items()
            }
            report_path = Path(tempfile.gettempdir()) / "PySide6_Data_Plot_Phase7_Layout_Audit" / "aggregate_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            self.fail(
                "Phase 7 layout audit findings/fatal errors (all actionable findings must be resolved):\n"
                + f"aggregate_report={report_path.resolve()}\n"
                + json.dumps(
                    {
                        "runs": runs,
                        "child_error_count": len(child_errors),
                        "schema_error_count": len(schema_errors),
                        "finding_schema_error_count": len(finding_schema_errors),
                        "finding_count": len(findings),
                        "counts_by_workflow": counts_by_workflow,
                        "coverage_summary_by_scale": coverage_summary,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )

    def test_intentional_exemptions_are_structured_and_narrow(self) -> None:
        """Sentinel bounds and the nine compact MCD selectors are accounted separately."""
        repo = Path(__file__).resolve().parents[1]
        expected_python = repo / ".venv" / "Scripts" / "python.exe"
        self.assertEqual(Path(sys.executable).resolve(), expected_python.resolve())
        probe = repo / "tests" / "phase7_layout_audit.py"
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["QT_SCALE_FACTOR"] = "1"
        proc = subprocess.run(
            [sys.executable, str(probe), "--scale", "1"],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        exemptions = payload["exemptions"]
        self.assertEqual(exemptions["total"], 83)
        self.assertEqual(exemptions["by_reason"]["sentinel_bound"], 74)
        self.assertEqual(exemptions["by_reason"]["mcd_compact_selector"], 9)
        self.assertEqual(payload["counts"].get("SPINBOXES", 0), 0)
        self.assertTrue(all("exemption_reason" not in finding for finding in payload["findings"]))


if __name__ == "__main__":
    unittest.main()
