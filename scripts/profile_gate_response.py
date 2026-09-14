"""Diagnostic gate/cursor response audit for the production Qt window.

This script uses a real ``MainWindow`` and its Qt Matplotlib canvas with
deterministic synthetic arrays. It only measures already-loaded synthetic
data; source matching, file I/O, and worker loading are deliberately excluded.
The endpoint is normally the canvas ``draw_event``; MCD uses the real
``canvas.blit`` completion because its normal center-cursor path is blitted.
The results include Qt event-loop delivery and offscreen rendering and are not
a proxy for instrument-data or interactive-window performance.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import MethodType

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Import QtCore first: this avoids the Windows DLL import ordering issue seen
# by the project's existing Qt harnesses.
from PySide6.QtCore import QCoreApplication, QEvent

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.loader import DataCube
from core.mcd import McdResult, McdSettings
from ui_qt.common import LoadedState
from scripts.preview_ui import build_preview_window


def _cube(rows: int, cols: int, *, title: str = "Synthetic") -> DataCube:
    energy = np.linspace(1.70, 2.70, cols)
    gate = np.linspace(-1.0, 1.0, rows)
    x = energy[None, :]
    y = gate[:, None]
    z = 1.0 + 0.15 * np.sin(18.0 * x) + 0.08 * y + np.exp(-((x - 2.20) / 0.035) ** 2)
    return DataCube(energy, gate, z, "Gate (V)", title, "PL corr. (a.u.)")


def _percentile(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "p50_ms": float("nan"), "p95_ms": float("nan"), "max_ms": float("nan")}
    a = np.asarray(values, float)
    if a.size == 1:
        return {"count": 1, "observation_ms": float(a[0])}
    return {"count": int(a.size), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(np.max(a))}


def _pump(app, predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        app.processEvents(QEventLoopFlags)
        if predicate():
            return True
    return bool(predicate())


# QEventLoop.AllEvents is deliberately held as a module value to avoid a Qt
# enum lookup in the hot measurement loop.
from PySide6.QtCore import QEventLoop
QEventLoopFlags = QEventLoop.ProcessEventsFlag.AllEvents


def _install_draw_probe(window):
    draws: list[float] = []
    cid = window.canvas.mpl_connect("draw_event", lambda _event: draws.append(time.perf_counter()))
    return draws, cid


def _drain(app) -> None:
    for _ in range(4):
        app.processEvents(QEventLoopFlags)


def _measure_control(window, app, control, values: list[float], *, label: str,
                     rapid: bool = False, timer_ms: int | None = None,
                     endpoint: str = "draw",
                     reset=None) -> dict[str, object]:
    """Measure input-to-next-real-draw and separately record redraw dispatch."""
    if reset is not None:
        reset()
    _drain(app)
    draw_times, cid = _install_draw_probe(window)
    completion_times = draw_times
    original_blit = None
    if endpoint == "blit":
        blit_times: list[float] = []
        original_blit = window.canvas.blit
        def timed_blit(*args, **kwargs):
            result = original_blit(*args, **kwargs)
            blit_times.append(time.perf_counter())
            return result
        window.canvas.blit = timed_blit
        completion_times = blit_times
    dispatch: list[float] = []
    original_plot = getattr(window, "_plot_mode", None)

    if original_plot is not None:
        def timed_plot(self, mode, *args, **kwargs):
            dispatch.append(time.perf_counter())
            return original_plot(mode, *args, **kwargs)
        window._plot_mode = MethodType(timed_plot, window)

    samples: list[float] = []
    debounce_samples: list[float] = []
    try:
        burst_t0 = time.perf_counter() if rapid else None
        burst_before_draw = len(completion_times)
        for value in values:
            if timer_ms is not None and hasattr(window, "_mcd_center_refresh_timer"):
                pass
            before_draw = len(completion_times)
            t0 = time.perf_counter()
            control.setValue(float(value))
            if rapid:
                continue
            ok = _pump(app, lambda: len(completion_times) > before_draw)
            if ok:
                samples.append((completion_times[-1] - t0) * 1000.0)
                if dispatch:
                    debounce_samples.append(max(0.0, (dispatch[-1] - t0) * 1000.0))
        if rapid:
            t0 = burst_t0 if burst_t0 is not None else time.perf_counter()
            before_draw = burst_before_draw
            ok = _pump(app, lambda: len(completion_times) > before_draw)
            if ok:
                # For a burst there is no per-input draw endpoint.  Record the
                # burst completion latency once and retain its explicit count.
                samples.append((completion_times[-1] - t0) * 1000.0)
                if dispatch:
                    debounce_samples.append(max(0.0, (dispatch[-1] - t0) * 1000.0))
    finally:
        if original_plot is not None:
            window._plot_mode = original_plot
        if original_blit is not None:
            window.canvas.blit = original_blit
        window.canvas.mpl_disconnect(cid)
    return {
        "label": label,
        "input_count": len(values),
        "rapid": rapid,
        "latency": _percentile(samples),
        "dispatch_delay": _percentile(debounce_samples),
        "endpoint": endpoint,
        "completion_events": len(completion_times),
    }


def _set_loaded(window, mode: str, cube: DataCube) -> None:
    index = {"PL": 0, "DRR": 1, "Compare": 2, "Power": 3, "MCD": 4}[mode]
    window.tabs.setCurrentIndex(index)
    window.workflow_tabs.setCurrentIndex(index)
    if mode == "Compare":
        c2 = DataCube(cube.energy.copy(), cube.gate.copy(), cube.Z * 1.03, cube.gate_label, "B", cube.cbar_label)
        loaded = LoadedState(mode="Compare", folder="", compare_cubes={"KK": cube, "KKp": c2},
                             compare_sources={"KK": "synthetic-a.csv", "KKp": "synthetic-b.csv"})
    else:
        loaded = LoadedState(mode="Power Dependent" if mode == "Power" else mode, folder="",
                             cube=cube, primary_file="synthetic.csv", power_group_key="synthetic")
    window.loaded = loaded
    # This diagnostic setup intentionally bypasses source matching/reload.  It
    # leaves the real plot dispatch and controller code intact while ensuring
    # no filesystem worker can run from synthetic inputs.
    window._ensure_loaded_matches_ui_params = lambda _mode: False
    window.last_plotted_mode = None
    window._plot_mode(loaded.mode)


def _mcd_result(rows: int, cols: int) -> McdResult:
    energy = np.linspace(1.70, 2.70, cols)
    fields = np.linspace(-1.0, 1.0, rows)
    xx = energy[None, :]
    bb = fields[:, None]
    pos = 1.0 + 0.08 * np.sin(18.0 * xx) + 0.15 * bb + np.exp(-((xx - 2.2) / 0.035) ** 2)
    neg = 1.0 + 0.08 * np.sin(18.0 * xx) - 0.12 * bb + np.exp(-((xx - 2.2) / 0.035) ** 2)
    mcd = (pos - neg) / np.maximum(pos + neg, 1e-12)
    return McdResult(
        source_file="synthetic-mcd.csv", wavelength_nm=1239.841984 / energy,
        energy_ev=energy, pos_angle=45.0, neg_angle=-45.0, pair_b=fields,
        pair_b_pos=fields.copy(), pair_b_neg=fields.copy(), pair_delta_b=np.zeros(rows),
        pair_sequence_gap=np.zeros(rows, dtype=int), pair_interpolated_pos=np.zeros(rows, dtype=bool),
        pair_interpolated_neg=np.zeros(rows, dtype=bool), pair_labels=np.asarray([str(v) for v in fields]),
        pair_raw_pos=pos, pair_raw_neg=neg, pair_corrected_pos=pos, pair_corrected_neg=neg,
        pair_mcd_raw=mcd, pair_mcd_corrected=mcd, pair_scale=np.ones(rows), pair_offset=np.zeros(rows),
        pair_spectral_slope=np.zeros(rows), pair_spectral_curvature=np.zeros(rows),
        pair_correction_min=np.ones(rows), pair_correction_max=np.ones(rows),
        pair_background_rms_before=np.zeros(rows), pair_background_rms=np.zeros(rows),
        reference_b=0.0, gain=np.ones(cols), reference_pos=np.ones(cols), reference_neg=np.ones(cols),
        dark_pos=np.zeros(cols), dark_neg=np.zeros(cols),
        maps={"Combo": DataCube(energy, fields, mcd, "B (T)", "Synthetic MCD", "MCD")},
        summary={}, acquisition_conditions={},
    )


def _run_mcd(window, app) -> dict[str, object]:
    window.tabs.setCurrentIndex(4)
    window.workflow_tabs.setCurrentIndex(4)
    result = _mcd_result(64, 128)
    window.loaded = LoadedState(mode="MCD", folder="", primary_file="synthetic-mcd.csv",
                                mcd_result=result, mcd_settings=McdSettings())
    window._ensure_loaded_matches_ui_params = lambda _mode: False
    window.last_plotted_mode = None
    window._plot_mode("MCD")
    _drain(app)
    energy = np.linspace(1.85, 2.55, 8).tolist()
    single = _measure_control(window, app, window.mcd_window_center_spin, energy,
                              label="MCD:64x128 center cursor", endpoint="blit")
    rapid = _measure_control(window, app, window.mcd_window_center_spin, energy,
                             label="MCD:64x128 center cursor:rapid", rapid=True, endpoint="blit")
    return {"control": "mcd_window_center_spin (energy cursor)", "status": "measured",
            "results": [{**single, "rows": 64, "cols": 128}, {**rapid, "rows": 64, "cols": 128}],
            "note": "normal MCD blit path measured; completion endpoint is canvas.blit"}


def _run_page(window, app, page: str, sizes: tuple[tuple[int, int], ...]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for rows, cols in sizes:
        mode = page
        _set_loaded(window, mode, _cube(rows, cols, title=f"Synthetic {page} {rows}x{cols}"))
        _drain(app)
        spins = window.pl_spins if page == "PL" else window.drr_spins if page == "DRR" else window.cmp_spins if page == "Compare" else window.power_spins
        values = np.linspace(-0.85, 0.85, 8).tolist()
        # The initial plot has snapped the spin to a real gate.  Stay inside
        # its range and use unique values so valueChanged always fires.
        values = [float(v) for v in np.linspace(float(spins["gate"].minimum()), float(spins["gate"].maximum()), 8)]
        single = _measure_control(window, app, spins["gate"], values, label=f"{page}:{rows}x{cols}")
        rapid = _measure_control(window, app, spins["gate"], values, label=f"{page}:{rows}x{cols}:rapid", rapid=True)
        out.extend([{**single, "rows": rows, "cols": cols}, {**rapid, "rows": rows, "cols": cols}])
    return out


def main() -> int:
    window, app, settings_dir = build_preview_window(
        workflow="PL", theme="light", scale="1", width=1200, height=800,
        sidebar_width=380, demo_data=False,
    )
    results: dict[str, object] = {
        "environment": {"qt_platform": os.environ.get("QT_QPA_PLATFORM"), "backend": str(window.canvas.__class__.__name__),
                        "offscreen": True, "settings_dir": str(settings_dir)},
        "pages": {},
    }
    try:
        for page in ("PL", "DRR", "Compare", "Power"):
            results["pages"][page] = _run_page(window, app, page, ((64, 128), (256, 512)))
        results["pages"]["MCD"] = _run_mcd(window, app)
        # The cursor control inventory is intentionally explicit.  MCD Peak
        # Shift and SHG require file-backed analysis/worker results; their
        # static dispatch evidence is recorded in the report rather than
        # fabricating a latency number from an incomplete result.
        results["pages"].update({
            "MCD Peak Shift": {"control": "mcd_peak_field_combo / canvas B cursor + feature cursor", "status": "static_only",
                                "reason": "selection is coupled to analyze/local-fit result and draw path; no valid result fixture"},
            "SHG": {"control": "peak center/integration gate + angle cursor", "status": "static_only",
                    "reason": "editingFinished requests worker reprocess from source-backed ShgSweepData; no source file read permitted"},
            "Slides": {"control": "N/A", "status": "N/A"},
            "Tools": {"control": "N/A", "status": "N/A"},
        })
    finally:
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
