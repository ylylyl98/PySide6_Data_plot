"""Isolated synthetic measurements for retained Peak Analysis plots and caches."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/quick-peak-profile.json"))
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--fit", action="store_true", help="Include fitted components and residuals in the row benchmark")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QUICK_BACKEND", "software")
    os.environ.setdefault("MPLBACKEND", "agg")
    with TemporaryDirectory(prefix="quick-peak-profile-") as temporary:
        os.environ["MPLCONFIGDIR"] = temporary
        from PySide6.QtCore import QSettings
        from PySide6.QtGui import QGuiApplication
        import numpy as np
        from core.loader import DataCube
        from core.peak_workspace import create_dataset
        from ui_qt.quick.peak_analysis_bridge import PeakAnalysisBridge
        from ui_qt.quick.peak_analysis_workflow import analyze_dataset
        app = QGuiApplication([])
        settings = QSettings(str(Path(temporary) / "isolated.ini"), QSettings.Format.IniFormat)
        bridge = PeakAnalysisBridge(settings=settings, session_path=Path(temporary) / "workspace.npz")
        x, y = np.linspace(1.60, 1.80, 501), np.linspace(-2, 2, args.rows)
        z = np.array([.1 + 1 / (1 + (2 * (x - (1.65 + gate * .001)) / .008) ** 2)
            + .6 / (1 + (2 * (x - (1.75 - gate * .001)) / .012) ** 2) for gate in y])
        dataset = create_dataset(DataCube(x, y, z, "Gate (V)", "Profile", "Intensity"), "PL", "PL", "profile", "Profile")
        # Warm imports independently of the recorded operations.
        analyze_dataset(dataset, "detect")
        times = {}
        start = perf_counter()
        dataset = analyze_dataset(dataset, "detect")
        times["detect_and_quality_ms"] = (perf_counter() - start) * 1000
        start = perf_counter()
        dataset = analyze_dataset(dataset, "filter")
        times["cached_filter_ms"] = (perf_counter() - start) * 1000
        if args.fit:
            dataset = analyze_dataset(dataset, "fit")
            dataset["show_residual"] = True
        bridge.addDatasets([dataset])
        for _ in range(6):
            app.processEvents()
        if bridge._canvas.backend._timer.isActive():
            bridge._canvas.draw()
        def measure(force_draw):
            samples = []
            for index in range(20):
                start = perf_counter()
                if force_draw:
                    bridge._background_image = None
                bridge.setParameter("row", index % args.rows)
                app.processEvents()
                samples.append((perf_counter() - start) * 1000)
            return statistics.median(samples)
        # Reproduce the previous native row path on the same canvas, data,
        # viewport and process. It rebuilds only spectrum artists and blits the
        # retained heatmap; this is the baseline improved by the artist pool.
        retained_selection = bridge._select_plot_row
        def rebuild_selection():
            from core.peak_plotting import draw_spectrum
            bridge._cursor_clear()
            limits = bridge._spectrum.get_xlim()
            draw_spectrum(dataset, dataset["row"], bridge._spectrum, bridge._residual)
            bridge._spectrum.set_xlim(limits)
            bridge._gate_line.set_ydata([float(dataset["cube"].gate[dataset["row"]])] * 2)
            bridge._style_figure()
            bridge._canvas.restore_region(bridge._background_image)
            bridge._draw_dynamic()
            bridge._canvas.blit(bridge._canvas.figure.bbox)
        bridge._select_plot_row = rebuild_selection
        draws, blits = bridge._canvas.draw_count, bridge._canvas.blit_count
        times["row_rebuild_blit_median_ms"] = measure(False)
        baseline_counts = dict(draws=bridge._canvas.draw_count - draws, blits=bridge._canvas.blit_count - blits)
        bridge._select_plot_row = retained_selection
        bridge._render()
        for _ in range(6):
            app.processEvents()
        times["row_pool_full_draw_median_ms"] = measure(True)
        draws, blits = bridge._canvas.draw_count, bridge._canvas.blit_count
        times["row_pool_blit_median_ms"] = measure(False)
        retained_counts = dict(draws=bridge._canvas.draw_count - draws, blits=bridge._canvas.blit_count - blits)
        times["row_full_draw_median_ms"] = times["row_pool_full_draw_median_ms"]
        times["row_retained_blit_median_ms"] = times["row_pool_blit_median_ms"]
        times["cursor_canvas_draws"] = 0
        draw_count = bridge._canvas.draw_count
        from types import SimpleNamespace
        for _ in range(100):
            bridge._cursor_motion(SimpleNamespace(inaxes=bridge._heat, xdata=1.65, ydata=0.))
        bridge._cursor_flush()
        times["cursor_canvas_draws"] = bridge._canvas.draw_count - draw_count
        report = dict(rows=args.rows, columns=len(x), points=len(dataset["result"]["points"]), fitted=bool(args.fit), timings=times,
            comparison=dict(row_pool_speedup=times["row_rebuild_blit_median_ms"] / times["row_pool_blit_median_ms"],
                baseline_frame_counts=baseline_counts, pool_frame_counts=retained_counts),
            scope="Synthetic warmed imports, offscreen software renderer. Rebuild/blit and pool/blit use the same Agg figure, measured data and viewport; each records20row changes. No user data/settings.")
        bridge.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
