"""Measure Power analysis, persistence and interaction with isolated synthetic data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--points", type=int, default=1001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QUICK_BACKEND", "software")
    os.environ.setdefault("MPLBACKEND", "agg")
    with TemporaryDirectory(prefix="power-performance-") as temporary:
        os.environ["MPLCONFIGDIR"] = temporary
        import numpy as np
        from PySide6.QtCore import QSettings, QTimer
        from PySide6.QtGui import QGuiApplication
        from matplotlib.backend_bases import MouseEvent
        from core.loader import DataCube
        from core.processing import PowerSweepPoint
        from core.power_multi_peaks import multi_peak_model
        from core.power_analysis_store import save_analysis
        from ui_qt.quick.power_bridge import PowerBridge
        from ui_qt.quick.power_workflow import fingerprints

        app = QGuiApplication([])
        root = Path(temporary)
        energy = np.linspace(1.34, 1.58, args.points)
        power = np.geomspace(1., 100., args.rows)
        z = np.asarray([multi_peak_model(energy, [100., 10., 100*p, 1.40, .014,
            50*p**.8, 1.46, .012, 25*p**1.1, 1.52, .010],
            model="Lorentzian", origin=energy.mean()) for p in power])
        cube = DataCube(energy, power, z, "Power (uW)", "Synthetic", "PL (a.u.)")
        path = root / "sweep.csv"
        np.savetxt(path, np.column_stack((power, z)), delimiter=",",
            header="power_uW," + ",".join(map(str, energy)), comments="")
        signatures = fingerprints(str(root), [path.name])
        settings = QSettings(str(root / "settings.ini"), QSettings.IniFormat)

        def create():
            bridge = PowerBridge(settings=settings)
            bridge._folder, bridge._stage = str(root), "plot"
            bridge._values.update(mode="single", background=0., autoBackground=False,
                xmin=float(energy[0]), xmax=float(energy[-1]), ymin=float(power[0]), ymax=float(power[-1]),
                vmin=float(z.min()), vmax=float(z.max()), gate=float(power[0]), clip=False)
            records = tuple(PowerSweepPoint(path.name, float(p), float(i), i+2) for i, p in enumerate(power))
            bridge._products = dict(cubes={"Power": cube}, records={"Power": records}, background=0.)
            bridge._active_selection = dict(single=path.name, mode="single", pairing="power")
            bridge._signatures = signatures
            bridge._analysis_data_changed()
            return bridge

        def settle(bridge):
            end = perf_counter() + 120
            while perf_counter() < end:
                app.processEvents()
                save_timer = getattr(bridge, "_analysis_save_timer", None)
                future = bridge._analysis_save_future
                canvas = bridge._canvas
                if (not bridge._pending and not bridge._plot_timer.isActive()
                        and not (save_timer and save_timer.isActive())
                        and (future is None or future.done())
                        and not (canvas and canvas.backend._timer.isActive())):
                    app.processEvents()
                    return
                sleep(.001)
            raise TimeoutError("Power analysis did not settle")

        samples, heartbeat = [], []
        last = [perf_counter()]
        timer = QTimer()
        timer.setInterval(10)
        def tick():
            now = perf_counter()
            heartbeat.append((now-last[0])*1000)
            last[0] = now
        timer.timeout.connect(tick)
        timer.start()
        with patch("core.power_analysis_store.save_analysis", wraps=save_analysis) as saves:
            def measure(bridge, name, operation):
                settle(bridge)
                canvas = bridge._canvas
                draws, blits = (canvas.draw_count, canvas.blit_count) if canvas else (0, 0)
                writes = saves.call_count
                heartbeat.clear()
                start = last[0] = perf_counter()
                operation()
                slot = (perf_counter()-start)*1000
                settle(bridge)
                row = dict(operation=name, slot_ms=round(slot, 3), settled_ms=round((perf_counter()-start)*1000, 3),
                    max_gui_gap_ms=round(max(heartbeat, default=slot), 3),
                    full_draws=bridge._canvas.draw_count-draws, blits=bridge._canvas.blit_count-blits,
                    cache_writes=saves.call_count-writes)
                samples.append(row)
                print(json.dumps(row), flush=True)

            bridge = create()
            try:
                measure(bridge, "first analysis", lambda: bridge.invoke("analysis.fit"))
                measure(bridge, "reuse memory cache", lambda: bridge.invoke("analysis.fit"))
                def preferences():
                    for _ in range(5):
                        bridge.setParameter("analysis.advanced", not bridge._analysis_values["advanced"])
                measure(bridge, "five display edits", preferences)
                measure(bridge, "spectrum click mode", lambda: bridge.setParameter("analysis.pickMode", "add"))
                bridge.setParameter("analysis.pickMode", "move")
                settle(bridge)
                original = bridge._peak_rows[0]["center"]
                def drag():
                    axis, canvas = bridge._analysis_spectrum, bridge._canvas
                    def event(kind, value):
                        x, y = axis.transData.transform((value, sum(axis.get_ylim()) / 2))
                        MouseEvent(kind, canvas, x, y, button=1)._process()
                    event("button_press_event", original)
                    for value in np.linspace(original, original+.001, 30):
                        event("motion_notify_event", value)
                        app.processEvents()
                    event("button_release_event", original+.001)
                measure(bridge, "30 drag updates", drag)
                bridge.setParameter("peak.0.center", original)
                bridge.invoke("analysis.fit")
                settle(bridge)
            finally:
                bridge.close()
            restored = create()
            try:
                measure(restored, "restore disk cache", lambda: restored.invoke("analysis.fit"))
            finally:
                restored.close()
        timer.stop()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(rows=args.rows, points=args.points, peaks=3,
            evidence="isolated synthetic data; GUI settling includes pending persistence", measurements=samples), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
