"""Run the six deliberately expensive Tier2A visual checks and only those checks."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = (
    "tests.test_phase7_layout_audit.Phase7LayoutAuditTests.test_real_main_window_layout_audit_all_scales",
    "tests.test_phase7_layout_audit.Phase7LayoutAuditTests.test_intentional_exemptions_are_structured_and_narrow",
    "tests.test_preview_ui.PreviewUiTests.test_screenshot_builds_readable_real_window",
    "tests.test_preview_ui.PreviewUiTests.test_font_regression_rejects_offscreen_tofu_font",
    "tests.test_preview_ui.PreviewUiTests.test_settings_sentinel_survives_isolated_preview_subprocess",
    "tests.test_ui_split_scale_controls.SplitScaleControlTests.test_axis_range_rows_keep_fix_and_auto_controls_contained",
)


def _flatten(suite: object) -> list[object]:
    import unittest

    if isinstance(suite, unittest.TestSuite):
        result: list[object] = []
        for child in suite:
            result.extend(_flatten(child))
        return result
    return [suite]


def _discover_exact_ids() -> list[str]:
    import unittest

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    loader = unittest.TestLoader()
    tests: list[object] = []
    for full_id in EXPECTED_IDS:
        tests.extend(_flatten(loader.loadTestsFromName(full_id)))
    return [str(test.id()) for test in tests if callable(getattr(test, "id", None))]


def verify_tier2a_report(report: dict[str, object], completed_events: list[dict[str, object]]) -> None:
    """Enforce exact membership and success, including against accidental skips."""

    discovered = report.get("discovery", {}).get("test_ids", [])  # type: ignore[union-attr]
    if discovered != list(EXPECTED_IDS):
        raise ValueError(f"Tier2A runner executed an unexpected ID set: {json.dumps(discovered)}")
    completed_ids = [event.get("full_id") for event in completed_events]
    if completed_ids != list(EXPECTED_IDS):
        raise ValueError(f"Tier2A runner did not complete exactly the six requested tests: {json.dumps(completed_ids)}")
    non_success = [
        {"full_id": event.get("full_id"), "status": event.get("status")}
        for event in completed_events
        if event.get("status") != "success"
    ]
    if non_success:
        raise ValueError(f"Tier2A requires six successful tests; non-success statuses: {json.dumps(non_success)}")


def main() -> int:
    discovered = _discover_exact_ids()
    if discovered != list(EXPECTED_IDS):
        raise SystemExit(
            "Tier2A ID verification failed; expected exactly six IDs in declared order, "
            f"got {json.dumps(discovered)}"
        )

    # These gates are intentionally opt-in in the individual test modules.
    os.environ["RUN_UI_VISUAL_TESTS"] = "1"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    output_dir = ROOT / "build" / "test-profile" / datetime.now().strftime("tier2a-%Y%m%d-%H%M%S")
    from scripts.profile_test_suite import run_supervised

    report = run_supervised(
        repo=ROOT,
        output_dir=output_dir,
        ids=EXPECTED_IDS,
        budget_seconds=300.0,
    )
    # Preserve event order from the durable log for exact membership/status checks.
    events_path = output_dir / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed_events = [event for event in events if event.get("event") == "TEST_END"]
    try:
        verify_tier2a_report(report, completed_events)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    completed_in_order = [event["full_id"] for event in completed_events]
    print(json.dumps({
        "tier": "Tier2A",
        "ids": list(EXPECTED_IDS),
        "completed": completed_in_order,
        "classification": report["exit"]["classification"],
        "report": str(output_dir / "report.json"),
    }, ensure_ascii=False))
    return 0 if report["exit"]["classification"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
