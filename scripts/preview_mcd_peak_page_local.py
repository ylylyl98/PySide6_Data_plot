"""Render the real Qt MCD Peak Shift page after local-fit completion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.mcd import process_mcd
from scripts.preview_ui import build_preview_window
from ui_qt.common import LoadedState


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path, required=True)
    parser.add_argument("--timeout-ms", type=int, default=420000)
    args = parser.parse_args()
    app = QApplication(sys.argv)
    result = process_mcd(str(args.input.resolve()))
    window = build_preview_window(workflow="MCD Peak Shift", theme="light", scale="1", width=1450, height=950, sidebar_width=440, demo_data=False)[0]
    window.loaded = LoadedState(mode="MCD", folder=str(args.input.parent), primary_file=args.input.name, selected_files=[args.input.name], cube=result.cube("Combo"), mcd_result=result)
    window._update_mcd_peak_shift_source(result)
    window.mcd_peak_source_combo.setCurrentText("Raw R")
    window._request_mcd_local_fit(seed_energy_ev=1.6436, locator_energy_ev=1.6436, feature_kind="peak")
    window.show()
    deadline = max(1000, int(args.timeout_ms))
    elapsed = 0

    def poll() -> None:
        nonlocal elapsed
        app.processEvents()
        analysis = getattr(window, "mcd_peak_result", None)
        ready = analysis is not None and getattr(analysis, "model_name", None) == "Local mixed fit (linear background)" and bool(getattr(analysis, "tracks", ()))
        if ready:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.screenshot.resolve()))
            args.screenshot.with_suffix(".json").write_text(json.dumps({"workflow": "MCD Peak Shift", "model": analysis.model_name, "seed_ev": 1.6436, "tracks": len(analysis.tracks), "status": window.mcd_peak_status.text()}, indent=2), encoding="utf-8")
            app.quit()
            return
        elapsed += 500
        if elapsed >= deadline:
            cancel_event = getattr(window, "_mcd_peak_fit_cancel_event", None)
            if cancel_event is not None:
                cancel_event.set()
            window.mcd_peak_status.setText("Preview timed out waiting for local fit")
            app.quit()
            return
        QTimer.singleShot(500, poll)

    QTimer.singleShot(500, poll)
    app.exec()
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
