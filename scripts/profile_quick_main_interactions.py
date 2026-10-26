"""Measure native PL/DRR GUI operations using isolated, synthetic data.

Times separate synchronous slots, settling, and a 10 ms event-loop heartbeat.
The render-lock probe simulates another worker saving an independent figure.
No measurement files or user settings are read or written.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-full-redraw", action="store_true",
                        help="Emulate the former full rebuild and blocking render paths on the same source/data.")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    with TemporaryDirectory(prefix="main-interaction-audit-") as temporary:
        os.environ.update(QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                          MPLBACKEND="agg", MPLCONFIGDIR=temporary,
                          XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary)
        from PySide6.QtCore import QSettings, QTimer, QUrl
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtQml import QQmlApplicationEngine
        from ui_qt.quick.application_bridge import QuickApplicationBridge
        from ui_qt.quick.plot_surface import register_plot_surface
        import numpy as np
        from core.loader import DataCube
        from core.typography import matplotlib_font_context
        from ui_qt.quick.drr_workflow import DrrProducts, DrrRequest

        app = QGuiApplication([])
        from ui_qt.typography import install_application_font
        install_application_font(app)
        if args.reference_full_redraw:
            from contextlib import nullcontext
            from ui_qt.quick import raster_heatmaps
            from ui_qt.quick.plot_surface import _CanvasBackend
            from ui_qt.axes_region_blitter import AxesRegionBlitter
            QuickApplicationBridge._recolor_plot = lambda self: False
            QuickApplicationBridge._rebuild_plot = QuickApplicationBridge._rebuild_plot_available
            raster_heatmaps.screen_heatmaps = lambda canvas: nullcontext()
            AxesRegionBlitter.draw = AxesRegionBlitter._draw_available
            def blocking_draw(backend):
                canvas = backend._canvas_ref()
                if canvas is not None and not canvas.closed:
                    canvas.draw()
            _CanvasBackend._draw = blocking_draw
        bridge = QuickApplicationBridge(settings=QSettings(str(Path(temporary)/"settings.ini"), QSettings.IniFormat))
        register_plot_surface()
        engine = QQmlApplicationEngine()
        warnings = []
        engine.warnings.connect(lambda items: warnings.extend(item.toString() for item in items))
        engine.setInitialProperties(dict(appBridge=bridge, width=1320, height=900))
        engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/"ui_qt/quick/qml/Application.qml")))
        if not engine.rootObjects():
            raise RuntimeError(warnings)
        window = engine.rootObjects()[0]
        beats = []
        heartbeat = QTimer()
        heartbeat.setInterval(10)
        heartbeat.timeout.connect(lambda: beats.append(time.perf_counter()))
        heartbeat.start()

        def pump(seconds):
            deadline = time.perf_counter()+seconds
            while time.perf_counter()<deadline:
                app.processEvents()
                time.sleep(.001)

        def settled():
            return not (bridge._plot_timer.isActive() or bridge._publish_timer.isActive()
                        or bridge._pl_range_timer.isActive()
                        or bridge._drr_display is not None and bridge._drr_display._pending
                        or bridge._canvas is not None and bridge._canvas.backend._timer.isActive())

        def wait():
            deadline=time.perf_counter()+30
            while time.perf_counter()<deadline:
                app.processEvents()
                if settled():
                    pump(.03)
                    if settled():
                        return
                time.sleep(.001)
            raise RuntimeError("Plot did not settle")

        rows=[]
        def measure(name, action, pause=0):
            wait();pump(.03)
            beats.clear()
            start=time.perf_counter();beats.append(start)
            action()
            slot=time.perf_counter()-start
            if pause:
                pump(pause)
            wait()
            end=time.perf_counter();beats.append(end)
            gaps=[(b-a)*1000 for a,b in zip(beats,beats[1:])]
            rows.append(dict(operation=name,slot_ms=slot*1000,total_ms=(end-start)*1000,
                             max_gui_gap_ms=max(gaps),heartbeats=len(beats)-2))
            print(json.dumps(rows[-1]),flush=True)

        energy=np.linspace(1.65,1.95,1340);gate=np.linspace(-15,15,541)
        z=1+np.cos(energy[None,:]*30)*.1+np.exp(-((energy[None,:]-1.8-gate[:,None]*.001)/.012)**2)
        title="YZ383_pwn1_9T_1.67KREF_730nmc_0p05sx20_RotIn49deg_TG+1BG=0_PIXIS"
        cube=DataCube(energy,gate,z,"Gate (V)",title,"PL (a.u.)")
        bridge.set_cube(cube);wait()
        for workflow in ("pl","drr"):
            if workflow=="drr":
                bridge.selectWorkflow("drr")
                raw=DataCube(energy,gate,z-1,"Gate (V)",title,"ΔR/R")
                second=DataCube(energy,gate,np.gradient(np.gradient(z,energy,axis=1),energy,axis=1),"Gate (V)",title,"d²(ΔR/R)/dE²")
                bridge._drr_products=DrrProducts(DrrRequest(()),None,raw,second,21,21)
                bridge.set_cube(raw);wait()
            for i in range(3):
                measure(f"{workflow}.gate.{i}",lambda i=i:bridge.setParameter("gate",float(gate[200+i*30])))
            for i in range(3):
                if workflow=="pl":
                    action=lambda i=i:bridge.setParameter("vmax",float(2.3+i*.1))
                else:
                    action=lambda i=i:bridge._drr_display.setNumber("raw.vmax",float(.8+i*.1))
                measure(f"{workflow}.color.{i}",action)
            measure(f"{workflow}.divider",lambda:bridge.setParameter("heatmapFraction",.65))
        measure("tabs.pl",lambda:bridge.selectWorkflow("pl"))
        measure("tabs.drr",lambda:bridge.selectWorkflow("drr"))

        for operation in ("full", "gate"):
            held=threading.Event()
            def worker():
                with matplotlib_font_context():
                    held.set();time.sleep(.6)
            thread=threading.Thread(target=worker)
            wait();thread.start();held.wait(5)
            # Do not settle before initiating the measured operation: that
            # would consume the deliberately held render lock first.
            beats.clear();start=time.perf_counter();beats.append(start)
            if operation=="full":
                bridge._canvas.draw_idle()
            else:
                bridge.setParameter("gate",float(gate[333]))
            slot=time.perf_counter()-start
            pump(.7);wait();thread.join()
            end=time.perf_counter();beats.append(end)
            rows.append(dict(operation=f"export_contention.{operation}",slot_ms=slot*1000,
                             total_ms=(end-start)*1000,
                             max_gui_gap_ms=max((b-a)*1000 for a,b in zip(beats,beats[1:])),heartbeats=len(beats)-2))
            print(json.dumps(rows[-1]),flush=True)

        report=dict(shape=list(z.shape),window=[1320,900],rows=rows,qml_warnings=warnings,
                    reference_full_redraw=args.reference_full_redraw,
                    font=app.font().family(),
                    measurement="10 ms GUI heartbeat, identical synthetic cube; optional emulation of former main plotting paths")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
        window.grabWindow().save(str(args.output.with_suffix(".png")))
        heartbeat.stop();window.close();bridge.close();engine.deleteLater();pump(.05)


if __name__=="__main__":
    main()
