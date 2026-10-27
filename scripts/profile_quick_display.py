"""Isolated synthetic DRR comparison; run each backend in a fresh process.

This measures shell/page construction and already-loaded gate-to-Matplotlib-blit
latency. It excludes instrument/file I/O, source matching, analysis and export;
it is not a full application benchmark or evidence of a general speedup.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("quick", "widgets"), required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--rows", type=int, default=157)
    parser.add_argument("--cols", type=int, default=1340)
    parser.add_argument("--platform", choices=("offscreen", "native"), default="offscreen")
    parser.add_argument("--theme", choices=("light", "dark"), default="light")
    parser.add_argument("--width", type=int, default=1320)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if not 2 <= args.samples <= args.rows or args.cols < 2:
        parser.error("require 2 <= samples <= rows and cols >= 2")
    os.environ["QT_QPA_PLATFORM"] = "offscreen" if args.platform == "offscreen" else (
        "windows" if sys.platform == "win32" else "cocoa" if sys.platform == "darwin" else "xcb")
    os.environ["QT_QUICK_BACKEND"] = "software"
    # Register Qt's DLL directories before NumPy/SciPy on Windows.
    from PySide6 import QtGui  # noqa: F401
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.profile_gate_response import _cube, _drain, _measure_control, _set_loaded
    from scripts.preview_ui import build_preview_window
    imports_done = time.perf_counter()
    # Keep this script's --platform option out of QApplication's own parser.
    app = QApplication([sys.argv[0]])
    window, app, settings_dir = build_preview_window(
        workflow="PL", theme=args.theme, scale="1", width=args.width, height=args.height,
        sidebar_width=320, demo_data=False, drr_display_backend=args.backend)
    constructed = time.perf_counter()
    try:
        window.workflow_tabs.setCurrentIndex(1)
        window.tabs.setCurrentIndex(1)
        window.show()
        _drain(app)
        host = getattr(window, "drr_display_host", None)
        if args.backend == "quick" and (host is None or not host.ensure_loaded()):
            raise RuntimeError(f"Quick backend did not load: {getattr(host, 'load_error', '')}")
        activated = time.perf_counter()
        cube = _cube(args.rows, args.cols, title=f"Synthetic DRR {args.rows}×{args.cols}")
        cube.cbar_label = "Synthetic ΔR/R (a.u.)"
        plot_started = time.perf_counter()
        _set_loaded(window, "DRR", cube)
        window._apply_auto_limits_for_loaded()
        window._drr_view_limits = None
        window._drr_limits_from_controls = True
        window._plot_mode("DRR")
        window.canvas.draw()
        _drain(app)
        plotted = time.perf_counter()
        heatmap = window._drr_heatmap_axes["raw"]
        view = (*heatmap.get_xlim(), *heatmap.get_ylim())
        domain = (float(cube.energy.min()), float(cube.energy.max()), float(cube.gate.min()), float(cube.gate.max()))
        if not heatmap.collections or any(abs(actual - expected) > 1e-6 for actual, expected in zip(view, domain)):
            raise RuntimeError(f"Synthetic heatmap is not displaying its full domain: {view}")
        gate = window.drr_gate_toolbar_spin
        warm_value = float(cube.gate[-1]) if gate.value() != float(cube.gate[-1]) else float(cube.gate[0])
        warm = _measure_control(window, app, gate, [warm_value],
                                label="warmup", endpoint="blit")
        if warm["latency"]["count"] != 1:
            raise RuntimeError("The DRR gate warmup did not reach canvas.blit")
        indices = [round(i * (args.rows - 1) / (args.samples - 1)) for i in range(args.samples)]
        measured = _measure_control(window, app, gate, [float(cube.gate[i]) for i in indices],
                                    label="synthetic DRR gate", endpoint="blit")
        if measured["latency"]["count"] != args.samples:
            raise RuntimeError(f"Only {measured['latency']['count']}/{args.samples} inputs reached canvas.blit")
        report = {
            "scope": "synthetic already-loaded DRR; no source I/O, analysis or export",
            "backend": args.backend, "qt_platform": app.platformName(),
            "qt_quick_backend": os.environ["QT_QUICK_BACKEND"], "theme": args.theme,
            "rows": args.rows, "cols": args.cols,
            "heatmap_view_limits": view,
            "settings_isolated": Path(window.settings.fileName()).resolve().is_relative_to(Path(settings_dir).resolve()),
            "timing_ms": {"imports": (imports_done - started) * 1000,
                          "window_construct": (constructed - imports_done) * 1000,
                          "drr_page_activate": (activated - constructed) * 1000,
                          "initial_plot_setup": (plotted - plot_started) * 1000,
                          "startup_to_page": (activated - started) * 1000},
            "gate": {"endpoint": measured["endpoint"], "control": "drr_gate_toolbar_spin.setValue",
                     "input_count": args.samples,
                     "completion_events": measured["completion_events"], **measured["latency"]},
        }
        if args.screenshot is not None:
            window._status("Synthetic DRR baseline; source loading and analysis are excluded")
            _drain(app)
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(args.screenshot), "PNG"):
                raise RuntimeError(f"Could not save {args.screenshot}")
            report["screenshot"] = str(args.screenshot.resolve())
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
    finally:
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
        window._preview_settings_temp.cleanup()
        window._preview_mpl_cache_temp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
