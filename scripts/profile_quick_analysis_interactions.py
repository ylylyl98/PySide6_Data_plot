"""Measure independent native analysis pages with synthetic data and settings.

Records synchronous slot time separately from settling/background time and the
largest 10 ms GUI heartbeat gap. The report deliberately includes ordinary plot
painting: completion of a worker alone does not mean the page is responsive.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, sleep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after"), default="after")
    parser.add_argument("--output", type=Path, default=Path("build/diagnostics/interaction-audit/analysis.json"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QUICK_BACKEND", "software")
    os.environ.setdefault("MPLBACKEND", "agg")
    with TemporaryDirectory(prefix="dptk-analysis-interactions-") as temporary:
        os.environ["MPLCONFIGDIR"] = temporary
        import numpy as np
        from PySide6.QtCore import QSettings, QTimer
        from PySide6.QtGui import QGuiApplication
        from core.loader import DataCube
        from core.peak_workspace import create_dataset as peak_dataset
        from core.drr_analysis_workspace import create_dataset as drr_dataset
        from ui_qt.quick.peak_analysis_bridge import PeakAnalysisBridge
        from ui_qt.quick.drr_analysis_bridge import DrrAnalysisBridge
        from ui_qt.quick.organizer_bridge import OrganizerBridge
        from tests.test_mcd_extract import McdExtractTests

        app = QGuiApplication([])
        root = Path(temporary)
        results = []
        heartbeats = []
        last = [perf_counter()]
        timer = QTimer()
        timer.setInterval(10)
        def tick():
            now = perf_counter()
            heartbeats.append((now - last[0]) * 1000)
            last[0] = now
        timer.timeout.connect(tick)
        timer.start()

        def pending(bridge):
            timers = ("_plot_timer", "_filter_timer", "_p2p_timer", "_preview_timer", "_cw_timer")
            return (any(getattr(bridge, name, None) and getattr(bridge, name).isActive() for name in timers)
                or bool(getattr(bridge, "_pending", None)) or bool(getattr(bridge, "_busy", False))
                or bool(getattr(bridge, "_analysis_busy", False))
                or bool(bridge._canvas and bridge._canvas.backend._timer.isActive()))

        def settle(bridge):
            end = perf_counter() + 120
            quiet = None
            while perf_counter() < end:
                app.processEvents()
                if not pending(bridge):
                    quiet = quiet or perf_counter()
                    if perf_counter() - quiet > .04:
                        return
                else:
                    quiet = None
                sleep(.001)
            raise TimeoutError(type(bridge).__name__ + " did not become idle")

        def measure(page, operation, bridge, call, *, repeats=1):
            samples = []
            for _ in range(repeats):
                settle(bridge)
                start = perf_counter()
                last[0] = start
                heartbeats.clear()
                draws = bridge._canvas.draw_count if bridge._canvas else 0
                blits = bridge._canvas.blit_count if bridge._canvas else 0
                call()
                slot = (perf_counter() - start) * 1000
                settle(bridge)
                samples.append(dict(slot_ms=slot, settled_ms=(perf_counter()-start)*1000,
                    max_gui_heartbeat_gap_ms=max(heartbeats, default=slot),
                    full_draws=(bridge._canvas.draw_count if bridge._canvas else 0)-draws,
                    blits=(bridge._canvas.blit_count if bridge._canvas else 0)-blits))
            row = dict(page=page, operation=operation, evidence="measured native bridge action + settled Agg canvas", repeats=repeats,
                **{key:round(statistics.median(sample[key] for sample in samples), 3) for key in samples[0]})
            results.append(row)
            print(json.dumps(row), flush=True)

        def settings(name):
            return QSettings(str(root/(name+".ini")), QSettings.IniFormat)

        x, y = np.linspace(1.6, 1.8, 501), np.linspace(-3, 3, 160)
        z = np.asarray([.1 + 1/(1+(2*(x-(1.65+gate*.001))/.008)**2)
            + .6/(1+(2*(x-(1.75-gate*.001))/.012)**2) for gate in y])
        cube = DataCube(x, y, z, "Gate (V)", "Synthetic", "Intensity")
        started = perf_counter()
        peak = PeakAnalysisBridge(settings=settings("peak"), session_path=root/"peak.npz")
        results.append(dict(page="Peak Analysis", operation="empty bridge construction", slot_ms=round((perf_counter()-started)*1000, 3)))
        first = peak_dataset(cube,"PL","PL","synthetic-a","Synthetic A")
        second = peak_dataset(cube,"PL","PL","synthetic-b","Synthetic B")
        measure("Peak Analysis", "add two datasets and first plot", peak, lambda:peak.addDatasets([first, second]))
        measure("Peak Analysis", "find all peaks", peak, lambda:peak.invoke("detect"))
        index = [0]
        def peak_row():
            index[0] += 7
            peak.setParameter("row",index[0] % len(y))
        measure("Peak Analysis", "row slider", peak, peak_row, repeats=7)
        measure("Peak Analysis", "cached filter edit", peak, lambda:peak.setParameter("min_snr",1.))
        measure("Peak Analysis", "fit selected branches", peak, lambda:peak.invoke("fit"))
        measure("Peak Analysis", "figure export", peak, lambda:peak.saveFigure(str(root/"peak.png")))
        measure("Peak Analysis", "select dataset", peak, lambda:peak.setParameter("dataset",first["key"]))

        started = perf_counter()
        drr = DrrAnalysisBridge(settings=settings("drr"), session_path=root/"drr.npz", start_folder=root)
        results.append(dict(page="DRR Analysis", operation="empty bridge construction", slot_ms=round((perf_counter()-started)*1000, 3)))
        dz = z-z[:, :1]-.4*np.exp(-((x-1.705)/.008)**2)
        dcube = DataCube(x,y,dz,"Gate (V)","Synthetic DRR","DR/R")
        dset = drr_dataset(dcube,"Synthetic DRR",{})
        drr._values["display"] = 1
        measure("DRR Analysis", "add dataset and first plot", drr, lambda:drr.addDatasets([dset]))
        drr._values["automatic"] = False
        measure("DRR Analysis", "analyze selected", drr, lambda:drr.invoke("analyze"))
        def gate_row():
            index[0] += 7
            drr.setParameter("gate",float(y[index[0] % len(y)]))
        measure("DRR Analysis", "gate slider", drr, gate_row, repeats=7)
        measure("DRR Analysis", "result filter edit", drr, lambda:drr.setParameter("filter.min_prominence",.05))
        measure("DRR Analysis", "P2P page and calculation", drr, lambda:drr.setParameter("tab",1))
        measure("DRR Analysis", "P2P inspection row", drr, lambda:drr.setParameter("p2p.row",5), repeats=3)
        drr.setParameter("p2p.style.dpi",100)
        settle(drr)
        measure("DRR Analysis", "inspection PNG export", drr, lambda:drr.exportTo("inspection",str(root/"drr-inspection.png")))

        writer = McdExtractTests()
        experiment = root/"organizer-data"
        for field in (0., 1.):
            for index, temperature in enumerate(np.linspace(3.,60.,24)):
                writer._write_result(experiment,f"field{field:g}T{index}",energy=1.66,doping=6.3,efield=field,
                    temperature=float(temperature),increasing_slope=2/(temperature+4),decreasing_slope=2.2/(temperature+4))
        started = perf_counter()
        organizer = OrganizerBridge(settings=settings("organizer"))
        results.append(dict(page="MCD Organizer", operation="empty bridge construction", slot_ms=round((perf_counter()-started)*1000, 3)))
        organizer.setParameter("compareVariable","Temperature")
        measure("MCD Organizer", "load synthetic catalog and first series", organizer, lambda:organizer.setExperimentFolder(str(experiment)))
        condition = [0]
        def focus():
            condition[0] = (condition[0]+1) % len(organizer._conditions)
            organizer.selectCondition(organizer._conditions[condition[0]]["id"],False)
        measure("MCD Organizer", "focus condition in MCD view", organizer, focus, repeats=7)
        measure("MCD Organizer", "select another series", organizer, lambda:organizer.selectSeries(organizer._series[-1].series_id))
        measure("MCD Organizer", "CW page and fit", organizer, lambda:organizer.setParameter("view","cw"))
        measure("MCD Organizer", "CW field parameter edit", organizer, lambda:organizer.setParameter("fieldHalfwidth",.15))
        measure("MCD Organizer", "method comparison page", organizer, lambda:organizer.setParameter("view","methods"))
        measure("MCD Organizer", "focus condition in diagnostics", organizer, focus, repeats=3)
        measure("MCD Organizer", "CW export", organizer, lambda:organizer.invoke("export-cw"))
        organizer.close()
        drr.close()
        peak.close()
        timer.stop()
        report = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
        report.update(scope="Synthetic data only, isolated temporary INI/workspaces/exports; native bridge+Agg offscreen software renderer. Imports are warmed across pages. settled_ms includes 40ms quiet interval; slot_ms excludes queued rendering/worker completion. Heartbeat interval 10ms; no fixed pass/fail thresholds.",
            dimensions=dict(peak_drr_rows=len(y),peak_drr_columns=len(x),organizer_conditions_per_series=24,organizer_series=2))
        report["inventory"] = {
            "Peak Analysis": {
                "measured": ["empty bridge construction (excludes process imports/QML)","add datasets and first plot","dataset selection","row slider","full-map detection","cached filter edit","selected-branch fitting","figure export"],
                "regression_tested_not_timed": ["populated QML window and controls","single-instance snapshot handoff/restore","Lorentzian/Gaussian residual parity","CSV/settings/PNG package export","workspace persistence","cancel/restart"],
                "inspected_not_measured": ["large workspaces containing many datasets","real experiment disk/network source loading"]},
            "DRR Analysis": {
                "measured": ["empty bridge construction (excludes process imports/QML)","add dataset and first plot","selected analysis","gate slider","cached result filter edit","P2P page/calculation","P2P row inspection","inspection PNG export"],
                "regression_tested_not_timed": ["populated QML window and controls","multiple dataset/workspace state","CSV/XLSX/PNG P2P exports","peak/summary export","raw measurement loading","cancellation and stale results"],
                "inspected_not_measured": ["large comparison catalogs","real experiment disk/network source loading"]},
            "MCD Organizer": {
                "measured": ["empty bridge construction (excludes process imports/QML)","synthetic catalog and first series","condition focus in MCD view","series selection","CW page/fit","CW field parameter edit","method comparison page","condition focus in diagnostics","CW export"],
                "regression_tested_not_timed": ["standalone QML at wide/compact sizes","group assignment/persistence","field-range scan","theta comparison","complete series package export","matched-method diagnostics"],
                "inspected_not_measured": ["thousands of saved conditions","real experiment disk/network source loading"]}}
        report["limitations"] = [
            "First figure construction, switching plot geometry/datasets and diagnostic layouts still require GUI work; improvements target repeated interactions without changing scientific grids/results.",
            "Warmed synthetic, offscreen software-renderer measurements are machine-specific, not physical desktop input-to-photon or cold process startup timings.",
            "Earlier before capture may overlap other validation processes; frame counts, immutable snapshots and worker-thread regression assertions provide causal evidence independent of wall-clock variation."]
        report[args.phase] = results
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
