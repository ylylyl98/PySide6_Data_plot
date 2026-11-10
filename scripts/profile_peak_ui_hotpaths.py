"""Repeatable synthetic Peak Analysis GUI bookkeeping and spectrum-switch timings."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QUICK_BACKEND", "software")
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QGuiApplication
    import numpy as np
    from core.loader import DataCube
    from core.peak_workspace import create_dataset
    from core.peak_plotting import overlay_series
    from ui_qt.quick.peak_analysis_bridge import PeakAnalysisBridge
    app = QGuiApplication([])
    def median_ms(operation, count=5):
        values = []
        for _ in range(count):
            start = perf_counter(); operation(); values.append((perf_counter() - start) * 1000)
        return statistics.median(values)
    with TemporaryDirectory(prefix="peak-ui-profile-") as temporary:
        x, y = np.linspace(1.6, 1.8, 501), np.arange(80.)
        cube = DataCube(x, y, np.tile(1 / (1 + ((x - 1.7) / .004) ** 2), (80, 1)), "Gate", "Profile", "Intensity")
        dataset = create_dataset(cube, "PL", "PL", "profile", "Profile")
        dataset["branches"] = [dict(id=str(i), name=f"P{i}", color=i, enabled=True, seed_row=0, seed_energy=1.65) for i in range(400)]
        dataset["result"] = dict(dataset_key=dataset["key"], branches=dataset["branches"], fits=[], points=[
            dict(branch_id=str(i), row_index=row, y=float(y[row]), energy=1.6 + i * .0004,
                 status="accepted", polarity="peak", segment=0) for row in range(80) for i in range(400)])
        bridge = PeakAnalysisBridge(settings=QSettings(str(Path(temporary) / "settings.ini"), QSettings.IniFormat),
            session_path=Path(temporary) / "session.npz")
        bridge.datasets[dataset["key"]] = dataset; bridge.active_key = dataset["key"]
        bridge._update_models()  # warm imports
        times = dict(overlay_series_ms=median_ms(lambda: overlay_series(dataset)),
                     hidden_models_ms=median_ms(bridge._update_models),
                     snapshot_ms=median_ms(lambda: bridge.snapshot, 20))
        plain = create_dataset(cube, "PL", "PL", "single", "Single spectrum")
        plain["analysis_scope"] = "spectrum"
        bridge.datasets = {plain["key"]: plain}; bridge.active_key = plain["key"]
        bridge._select_dataset()
        def idle():
            deadline = perf_counter() + 10
            while True:
                app.processEvents()
                if not bridge._row_timer.isActive() and not bridge._plot_timer.isActive() and bridge._canvas and not bridge._canvas.backend._timer.isActive():
                    return
                if perf_counter() > deadline: raise RuntimeError("Plot did not settle")
        idle()
        draws, blits = bridge._canvas.draw_count, bridge._canvas.blit_count
        samples = []
        for row in range(1, 11):
            start = perf_counter(); bridge.setParameter("row", row); idle()
            samples.append((perf_counter() - start) * 1000)
        times["spectrum_switch_ms"] = statistics.median(samples)
        report = dict(timings=times, map_points=32000, map_branches=400, spectrum_switches=10,
                      spectrum_draws=bridge._canvas.draw_count-draws, spectrum_blits=bridge._canvas.blit_count-blits,
                      scope="Synthetic warm operations; offscreen software renderer. Median timings, no measured user data.")
        bridge.datasets.clear(); bridge.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
