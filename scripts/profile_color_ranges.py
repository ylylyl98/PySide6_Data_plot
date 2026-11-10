"""Synthetic color-slider benchmark; uses isolated settings and no user data.

Run with the project Python. Results include Agg drawing/publication, not the
physical monitor's presentation latency. No fixed FPS assertion is appropriate
across machines; final image/state consistency and bounded work are checked.
"""
from pathlib import Path
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QGuiApplication
    from core.loader import DataCube
    from ui_qt.quick.application_bridge import QuickApplicationBridge

    app = QGuiApplication([])
    with tempfile.TemporaryDirectory(prefix="color-profile-") as folder:
        bridge = QuickApplicationBridge(settings=QSettings(str(Path(folder)/"settings.ini"), QSettings.IniFormat))
        def idle():
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                app.processEvents()
                if (bridge._canvas is not None and not bridge._plot_timer.isActive()
                        and not bridge._publish_timer.isActive()
                        and not bridge._canvas.backend._timer.isActive()):
                    return
                time.sleep(.001)
            raise RuntimeError("Plot did not settle")
        try:
            x, y = np.linspace(1.,3.,2001), np.linspace(-2.,2.,501)
            z = 1 + np.exp(-((x[None,:]-2-y[:,None]*.1)/.08)**2)
            bridge.set_cube(DataCube(x,y,z,"Gate (V)","Color slider benchmark","PL"))
            idle()
            bridge._canvas.set_logical_size(1200,750)
            bridge.plDisplayBridge.setChoice('regionCount',2)
            idle()
            controller = bridge.colorRangeController
            groups = controller.snapshot['groups']
            assert len(groups) == 3 and bridge._values['spectrumVisible']
            lines = tuple(line for _, line in bridge._spectra)
            data = [np.array(line.get_ydata(),copy=True) for line in lines]
            before = bridge._canvas.draw_count
            draws, timings, preview_frames = before, [], 0
            for step in range(120):
                ident = groups[(step//20)%3]['id']
                phase = (step%20)/20
                controller.moveRange(ident,.1+phase*.15,.9-phase*.15,False)
                app.processEvents()
                if bridge._canvas.draw_count != draws:
                    timings.append(bridge._canvas.last_render_ms)
                    preview_frames += int(getattr(bridge._canvas, 'color_range_preview', False))
                    draws = bridge._canvas.draw_count
                time.sleep(.008)
            released = time.perf_counter()
            controller.moveRange(ident,.22,.81,True)
            idle()
            final_ms = (time.perf_counter()-released)*1000
            assert not controller._pending
            updated = controller.snapshot['groups']
            meshes = bridge._maps[0][0]._dptk_quick_render[0].images
            for group, mesh in zip(updated,meshes):
                np.testing.assert_allclose(mesh.get_clim(),(group['low'],group['high']))
            for old, (_, line), values in zip(lines,bridge._spectra,data):
                assert old is line
                np.testing.assert_array_equal(line.get_ydata(),values)
            frames = bridge._canvas.draw_count-before
            assert frames < 120, frames
            print(json.dumps(dict(cells=z.size,regions=3,spectrum=True,previewRequests=120,
                renderedFrames=frames,previewFrames=preview_frames,medianRenderMs=round(float(np.median(timings)),2),
                maxRenderMs=round(max(timings),2),adaptiveIntervalMs=round(controller._interval*1000,2),
                releaseToFinalMs=round(final_ms,2),finalRangesMatch=True),indent=2))
        finally:
            bridge.close()


if __name__ == '__main__':
    main()
