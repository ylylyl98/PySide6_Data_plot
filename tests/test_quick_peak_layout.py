"""Compact Peak Analysis controls remain reachable when filters expand."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class PeakLayoutTests(unittest.TestCase):
    def test_compact_custom_controls(self):
        code = r'''
import os, sys, time
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PySide6.QtCore import QObject, QPointF, QSettings, QUrl
from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
from core.loader import DataCube
from ui_qt.quick.peak_analysis_bridge import PeakAnalysisBridge
from ui_qt.quick.peak_analysis_workflow import prepare_sources
from ui_qt.quick.plot_surface import register_plot_surface
QQuickStyle.setStyle("Basic")
app = QGuiApplication([])
if sys.platform == "win32":
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    app.setFont(QFont("Segoe UI", 9))
register_plot_surface()
with TemporaryDirectory() as temporary:
    bridge = PeakAnalysisBridge(settings=QSettings(str(Path(temporary)/"settings.ini"), QSettings.Format.IniFormat), session_path=Path(temporary)/"workspace.npz")
    x = np.linspace(1.4, 1.5, 101)
    cube = DataCube(x, np.arange(3.), np.tile(np.exp(-((x-1.45)/.004)**2), (3,1)), gate_label="Gate", gate_unit="V", title="PL", cbar_label="Intensity")
    bridge.addDatasets(prepare_sources([dict(cube=cube, kind="PL", channel="PL", source="sample", name="sample", row=1)]))
    engine = QQmlApplicationEngine()
    engine.setInitialProperties({"peakBridge": bridge})
    engine.load(QUrl.fromLocalFile(str(Path("ui_qt/quick/qml/PeakAnalysis.qml").resolve())))
    assert engine.rootObjects()
    window = engine.rootObjects()[0]
    window.setWidth(760); window.setHeight(600)
    bridge.setParameter("filterPreset", "Custom")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        app.processEvents()
        button = window.findChild(QObject, "app_parameter_find-fit")
        surface = window.findChild(QObject, "peak_plot_surface")
        if (button and button.property("visible") and button.property("height") > 0
                and surface.property("presentedGeneration") > 0
                and not bridge._plot_timer.isActive() and not surface._size_timer.isActive()
                and not bridge._canvas.backend._timer.isActive()
                and surface.property("presentedGeneration") == bridge._canvas.frame_generation):
            break
        time.sleep(.01)
    assert button and button.property("visible") and button.property("enabled")
    panel = window.findChild(QObject, "peak_analysis_controls")
    position = button.mapToItem(panel, QPointF(0, 0))
    assert 0 <= position.y() and position.y() + button.property("height") <= panel.property("height")
    label = bridge._spectrum.xaxis.label.get_window_extent(bridge._canvas.get_renderer())
    assert label.y0 >= 0, f"Clipped: label={label.bounds}, fig={bridge._canvas.figure.bbox.bounds}, inches={bridge._canvas.figure.get_figheight()}, dpi={bridge._canvas.figure.dpi}, margins={bridge._heat.get_subplotspec().get_gridspec().bottom}, font={bridge._spectrum.xaxis.label.get_fontsize()}"
    assert window.findChild(QObject, "app_parameter_detect").property("visible")
    if os.environ.get("DPTK_PEAK_REVIEW_DIR"):
        Path(os.environ["DPTK_PEAK_REVIEW_DIR"]).mkdir(parents=True, exist_ok=True)
        window.grabWindow().save(str(Path(os.environ["DPTK_PEAK_REVIEW_DIR"])/"pl-compact-custom.png"))
    bridge.dataset['settings']['model'] = 'Gaussian'
    bridge.invoke('find-fit')
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        app.processEvents()
        if not bridge.snapshot['busy'] and bridge.dataset.get('result'):
            break
        time.sleep(.01)
    assert bridge.dataset.get('result')
    bridge.setParameter('analysisPlot', 'trends')
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        app.processEvents()
        if len(bridge._trend_axes) == 2 and not bridge._plot_timer.isActive() and not bridge._canvas.backend._timer.isActive():
            break
        time.sleep(.01)
    assert len(bridge._trend_axes) == 2
    if os.environ.get('DPTK_PEAK_REVIEW_DIR'):
        window.grabWindow().save(str(Path(os.environ['DPTK_PEAK_REVIEW_DIR'])/'pl-fit-trends.png'))
    bridge.setParameter('analysisPlot', 'spectra')
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        app.processEvents()
        if bridge._heat is not None and not bridge._plot_timer.isActive():
            break
        time.sleep(.01)
    assert bridge._heat is not None
    window.setVisible(False)
    bridge.close()
'''
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software", MPLBACKEND="agg")
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                                env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
