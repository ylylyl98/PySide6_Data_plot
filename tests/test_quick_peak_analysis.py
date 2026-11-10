"""Native standalone Peak Analysis parity, lifecycle and retained-plot checks."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickPeakAnalysisTests(unittest.TestCase):
    def test_widgets_free_peak_analysis(self):
        with TemporaryDirectory(prefix="quick-peak-process-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software", MPLBACKEND="agg",
                MPLCONFIGDIR=temporary, XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_PEAK_ANALYSIS_TESTS=", result.stdout)

    def test_single_instance_handoff_and_fresh_restore(self):
        import json
        import time
        code = '''
import importlib.abc, json, sys
from pathlib import Path
class NoWidgets(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"PySide6.QtWidgets", "PySide6.QtQuickWidgets", "ui_qt.main_window", "ui_qt.common"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, NoWidgets())
from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication
import run_peak_analysis_quick
original = QGuiApplication.exec
def run(app):
    timer = QTimer()
    def observe():
        for window in app.allWindows():
            if window.objectName() != "peak_analysis_window": continue
            bridge = window.property("peakBridge")
            if bridge.datasets and not bridge.snapshot["busy"]:
                from core.pl_peak_analysis import detect
                for dataset in bridge.datasets.values():
                    if not dataset["branches"]:
                        dataset["branches"] = detect(dataset, 0, dataset["settings"])
                        if dataset["branches"]: dataset["branches"][0]["name"] = "Saved exciton"
                target = Path(sys.argv[2])
                temporary = target.with_suffix(".tmp")
                temporary.write_text(json.dumps({"count": len(bridge.datasets)}))
                temporary.replace(target)
                if Path(sys.argv[1]).exists(): bridge.requestClose()
    timer.timeout.connect(observe)
    timer.start(50)
    return original()
QGuiApplication.exec = run
raise SystemExit(run_peak_analysis_quick.main(sys.argv[3:]))
'''
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(prefix="quick-peak-ipc-") as temporary:
            directory = Path(temporary)
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software", MPLBACKEND="agg",
                MPLCONFIGDIR=temporary, XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary, PYTHONIOENCODING="utf-8")
            from core.peak_workspace import create_dataset, save_workspace, load_workspace
            from tests.test_peak_workspace import sample
            snapshots = [directory / "pl.npz", directory / "drr.npz"]
            for path, kind in zip(snapshots, ("PL", "DRR")):
                dataset = create_dataset(sample(kind), kind, "PL" if kind == "PL" else "second", kind, kind)
                dataset["settings"]["detection_method"] = "sg"
                save_workspace(path, [dataset], dataset["key"])
            stop, ready = directory / "stop", directory / "ready.json"
            session, settings = directory / "workspace.npz", directory / "isolated.ini"
            command = [sys.executable, "-c", code, str(stop), str(ready), "--session", str(session), "--settings", str(settings)]
            process = subprocess.Popen([*command, "--snapshot", str(snapshots[0])], cwd=root, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            def wait_count(expected):
                deadline = time.monotonic() + 35
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail(process.communicate()[1].decode(errors="replace"))
                    try:
                        if json.loads(ready.read_text())["count"] == expected:
                            return
                    except (OSError, ValueError):
                        pass
                    time.sleep(.05)
                self.fail("Native Peak Analysis did not accept a forwarded snapshot")
            try:
                wait_count(1)
                forwarded = subprocess.run([sys.executable, str(root / "run_peak_analysis_quick.py"), "--session", str(session),
                    "--settings", str(settings), "--snapshot", str(snapshots[1])], cwd=root, env=environment,
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(forwarded.returncode, 0, forwarded.stderr)
                wait_count(2)
                stop.touch()
                _, errors = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, errors.decode(errors="replace"))
                datasets, active = load_workspace(session)
                self.assertEqual(len(datasets), 2)
                self.assertTrue(all(dataset["branches"][0]["name"] == "Saved exciton" for dataset in datasets))
                derivative = next(dataset for dataset in datasets if dataset["kind"] == "DRR")
                self.assertEqual(derivative["settings"]["detection_method"], "local")
                derivative["settings"]["detection_method"] = "sg"
                save_workspace(session, datasets, active)
                restored = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=35)
                self.assertEqual(restored.returncode, 0, restored.stderr)
                self.assertEqual(next(dataset for dataset in load_workspace(session)[0] if dataset["kind"] == "DRR")["settings"]["detection_method"], "sg")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()


def _run_isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import importlib.abc
    import json
    import time
    import traceback
    from types import SimpleNamespace
    from threading import Event
    from unittest.mock import patch

    class NoWidgets(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {"PySide6.QtWidgets", "PySide6.QtQuickWidgets", "ui_qt.main_window", "ui_qt.common",
                "ui_qt.peak_analysis_window", "ui_qt.peak_analysis_controls", "ui_qt.peak_analysis_bridge",
                "matplotlib.backends.backend_qt", "matplotlib.backends.backend_qtagg"}:
                raise AssertionError("Native Peak Analysis imported Widgets: " + fullname)
    sys.meta_path.insert(0, NoWidgets())
    from PySide6.QtCore import QCoreApplication, QEvent, QMetaObject, QObject, QSettings, QUrl, Qt, qInstallMessageHandler
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from ui_qt.quick.peak_analysis_bridge import PeakAnalysisBridge, QuickPeakAnalysisLauncher
    assert not any(name in sys.modules for name in ("numpy", "scipy", "matplotlib")), "Peak shell imported scientific stack eagerly"
    from ui_qt.quick.peak_analysis_launcher import launch_arguments, server_name
    from ui_qt.quick.peak_analysis_workflow import prepare_sources, analyze_dataset, add_seed, dataset_snapshot
    from ui_qt.quick.plot_surface import register_plot_surface
    from core.loader import DataCube
    from core.peak_workspace import create_dataset, save_workspace, load_workspace
    import numpy as np
    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    if sys.platform == "win32":
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
        app.setFont(QFont("Segoe UI", 9))
    python_errors, qml_errors = [], []
    def exception(kind, value, trace):
        python_errors.append("".join(traceback.format_exception(kind, value, trace)))
        traceback.print_exception(kind, value, trace)
    sys.excepthook = exception
    def qt_message(kind, context, message):
        if any(word in message for word in ("ReferenceError", "TypeError", "Binding loop", "Cannot assign", "Unable to assign", "Error:")):
            qml_errors.append(str(message))
            print(message, file=sys.stderr)
    old_handler = qInstallMessageHandler(qt_message)
    register_plot_surface()

    def wait(predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                app.processEvents()
                return
            time.sleep(.005)
        raise AssertionError("Timed out waiting for native Peak Analysis")

    def find(root, name):
        if root.objectName() == name:
            return root
        children = root.childItems() if hasattr(root, "childItems") else root.children()
        for child in children:
            match = find(child, name)
            if match is not None:
                return match
        return None

    def cube(model="Lorentzian", multiplier=1.):
        from core.power_multi_peaks import multi_peak_model
        energy, gate = np.linspace(1.36, 1.53, 401), np.arange(5.)
        values = np.array([multi_peak_model(energy, [1., .2, 8., 1.415 + row * .0006, .006,
            4., 1.474 - row * .0004, .009], model=model, origin=1.44) for row in gate]) * multiplier
        return DataCube(energy, gate, values, gate_label="Gate", gate_unit="V", title="Synthetic map", cbar_label="Intensity")

    class Cases(unittest.TestCase):
        def setUp(self):
            initial_python, initial_qml = len(python_errors), len(qml_errors)
            self.addCleanup(lambda: self.assertEqual(python_errors[initial_python:], [], "Uncaught callback"))
            self.addCleanup(lambda: self.assertEqual(qml_errors[initial_qml:], [], "QML errors"))
            self.temporary = TemporaryDirectory(prefix="quick-peak-case-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name)
            self.settings = QSettings(str(self.root / "isolated.ini"), QSettings.Format.IniFormat)
            self.settings.setValue("unrelated/preserved", "keep")
            self.bridge = PeakAnalysisBridge(settings=self.settings, session_path=self.root / "workspace.npz")
            self.addCleanup(self.bridge.close)

        def install(self, model="Lorentzian", kind="PL", channel="PL", source="synthetic.csv"):
            dataset = create_dataset(cube(model), kind, channel, source, source)
            dataset["settings"].update(min_snr=0., neighbor_support=0, min_width_mev=0., prominence=.03, detection_method="local")
            self.bridge.addDatasets([dataset])
            return dataset

        def idle(self):
            wait(lambda: not self.bridge.snapshot["busy"] and not self.bridge._filter_timer.isActive() and not self.bridge._plot_timer.isActive()
                and (self.bridge._canvas is None or not self.bridge._canvas.backend._timer.isActive()))
            self.assertFalse(self.bridge.snapshot["error"], self.bridge.snapshot["error"])

        def detect(self):
            self.bridge.invoke("detect")
            self.idle()
            self.assertEqual(self.bridge.snapshot["resultCount"], 10)

        def test_pl_manual_editor_preserves_edits_and_fits_fixed_center(self):
            dataset = self.install()
            self.detect()
            self.bridge.setParameter('controlsPage', 'branches')
            row = dataset['row']
            point = next(p for p in dataset['result']['points'] if p['row_index'] == row)
            identity = point['branch_id']
            self.bridge.setParameter('manual.target', identity)
            center = point['energy'] + .00012
            self.bridge.setParameter('manual.energy', center)
            self.bridge.setParameter('manual.fixed', True)
            self.bridge.invoke('manual-move'); self.idle()
            self.assertEqual(dataset['manual_peak_edits'][0]['energy'], center)
            self.bridge.setParameter('model', 'Asymmetric Lorentzian'); self.idle()
            self.bridge.invoke('fit'); self.idle()
            moved = next(p for p in dataset['result']['points'] if p['row_index'] == row and p['branch_id'] == identity)
            self.assertEqual(moved['fit_center'], center)
            self.assertTrue(moved['fit_center_fixed'])
            self.assertTrue(dataset['result']['fits'])
            self.bridge.invoke('manual-exclude'); self.idle()
            self.assertFalse(dataset['result']['fits'])
            self.bridge.invoke('detect'); self.idle()
            self.assertFalse(any(p['row_index'] == row and abs(p['energy']-center) < .0003 for p in dataset['result']['points']))
            self.bridge.setParameter('manual.energy', 1.452123)
            self.bridge.setParameter('manual.fixed', False)
            self.bridge.invoke('manual-add'); self.idle()
            added = [p for p in dataset['result']['points'] if p.get('manual')]
            self.assertEqual(len(added), 1)
            self.assertEqual(added[0]['energy'], 1.452123)
            self.assertEqual(added[0]['row_index'], row)
            saved = self.root / 'manual.npz'
            self.bridge.saveWorkspace(str(saved)); self.idle()
            self.assertEqual(load_workspace(saved)[0][0]['manual_peak_edits'], dataset['manual_peak_edits'])
            engine = QQmlApplicationEngine()
            engine.setInitialProperties({'peakBridge': self.bridge})
            engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / 'ui_qt/quick/qml/PeakAnalysis.qml')))
            window = engine.rootObjects()[0]
            self.addCleanup(lambda: (window.setVisible(False), engine.deleteLater(), app.processEvents()))
            window.setWidth(1200); window.setHeight(1000); self.idle()
            self.assertIsNotNone(find(window.contentItem(), 'app_parameter_manual.energy'))
            self.assertTrue(find(window.contentItem(), 'app_parameter_manual-pick').property('visible'))
            self.assertTrue(find(window.contentItem(), 'peak_workflow_primary').property('visible'))
            if os.environ.get('DPTK_PEAK_REVIEW_DIR'):
                path = Path(os.environ['DPTK_PEAK_REVIEW_DIR']); path.mkdir(parents=True, exist_ok=True)
                window.grabWindow().save(str(path / 'pl-manual-editor.png'))
            self.bridge.invoke('manual-restore'); self.idle()
            self.assertFalse(dataset.get('manual_peak_edits'))

        def test_pl_four_step_workflow_and_stale_result_guards(self):
            dataset = self.install()
            engine = QQmlApplicationEngine()
            engine.setInitialProperties({'peakBridge': self.bridge})
            engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / 'ui_qt/quick/qml/PeakAnalysis.qml')))
            window = engine.rootObjects()[0]
            self.addCleanup(lambda: (window.setVisible(False), engine.deleteLater(), app.processEvents()))
            def capture(page):
                self.bridge.setParameter('controlsPage', page); self.idle()
                self.assertTrue(find(window.contentItem(), 'peak_step_'+page).property('checked'))
                if os.environ.get('DPTK_PEAK_REVIEW_DIR'):
                    path = Path(os.environ['DPTK_PEAK_REVIEW_DIR']); path.mkdir(parents=True, exist_ok=True)
                    window.grabWindow().save(str(path / f'journey-{page}.png'))
            capture('find')
            self.assertFalse(self.bridge.snapshot['canFit'])
            self.bridge.invoke('workflow-find'); self.idle()
            self.assertEqual(self.bridge.snapshot['controlsPage'], 'branches')
            capture('branches')
            self.bridge.setParameter('min_snr', .1); self.idle()
            self.assertTrue(self.bridge.snapshot['filtersPending'])
            self.assertFalse(self.bridge.snapshot['canFit'])
            self.assertFalse(self.bridge.snapshot['canExport'])
            self.bridge.invoke('filter'); self.idle()
            self.assertTrue(self.bridge.snapshot['canFit'])
            capture('fit')
            before = [(p['row_index'], p['energy']) for p in dataset['result']['points']]
            with patch('core.peak_candidates.filter_heatmap', side_effect=AssertionError('Fit must use checked peaks')):
                self.bridge.invoke('workflow-fit'); self.idle()
            self.assertEqual(self.bridge.snapshot['controlsPage'], 'results')
            self.assertEqual(before, [(p['row_index'], p['energy']) for p in dataset['result']['points']])
            capture('results')
            self.assertTrue(self.bridge.snapshot['canExport'])
            pool = dataset['candidates']; auto = dataset.get('auto_filters')
            self.bridge.setParameter('model', 'Gaussian'); self.idle()
            self.assertIs(pool, dataset['candidates'])
            self.assertEqual(auto, dataset.get('auto_filters'))
            self.assertFalse(dataset['result']['fits'])
            self.assertFalse(self.bridge.snapshot['filtersPending'])
            self.bridge.setParameter('detection_method', 'sg'); self.idle()
            self.assertTrue(dataset['detection_dirty'])
            self.assertFalse(self.bridge.snapshot['canExport'])
            self.assertFalse(self.bridge.snapshot['canFit'])
            self.bridge.invoke('workflow-find'); self.idle()
            self.assertFalse(dataset['detection_dirty'])
            for width, height, theme in ((760, 600, 'light'), (1240, 900, 'dark')):
                window.setWidth(width); window.setHeight(height)
                self.bridge._appearance.setMode(theme); self.idle()
                for page in ('find', 'branches', 'fit', 'results'):
                    self.bridge.setParameter('controlsPage', page); self.idle()
                    primary = find(window.contentItem(), 'peak_manual_fit' if page == 'fit' else 'peak_workflow_primary')
                    self.assertTrue(primary.property('visible'))
                    self.assertGreater(primary.width(), 100)
                    if os.environ.get('DPTK_PEAK_REVIEW_DIR'):
                        window.grabWindow().save(str(Path(os.environ['DPTK_PEAK_REVIEW_DIR']) / f'journey-{page}-{width}-{theme}.png'))

        def test_sources_fingerprint_versions_and_vp_exclusion(self):
            original = cube()
            sources = [dict(cube=original, kind="PL", channel=channel, source=channel, name=channel, row=3,
                view={"xlim": [1.4, 1.5], "ylim": [1, 3], "display": {"cmap": "viridis"}}) for channel in ("PL", "Compare", "Power", "VP")]
            prepared = prepare_sources(sources)
            self.assertEqual(len(prepared), 3)
            self.assertTrue(all(item["kind"] == "PL" and item["row"] == 3 for item in prepared))
            self.assertFalse(prepared[0]["cube"].Z.flags.writeable)
            first = prepared[0]["key"]
            original.Z[0, 0] += 1.
            second = prepare_sources(sources)[0]
            self.assertNotEqual(first, second["key"])
            self.bridge.addDatasets([prepared[0], second])
            self.assertEqual(len(self.bridge.datasets), 2)
            self.assertEqual(prepared[0]["view"]["display"]["cmap"], "viridis")

        def test_new_pl_handoff_defaults_to_full_map(self):
            source = dict(cube=cube(), kind="PL", channel="PL", source="full-pl.csv", name="PL", row=2,
                          view={"xlim": [1.40, 1.45], "ylim": [1, 3]})
            dataset = prepare_sources([source])[0]
            self.assertEqual(dataset["analysis_scope"], "map")
            self.assertEqual(dataset["scope"], 0)
            self.assertEqual(dataset["settings"]["x_min"], dataset["cube"].energy.min())
            self.assertEqual(dataset["settings"]["y_min"], dataset["cube"].gate.min())
            self.bridge.addDatasets([dataset])
            self.bridge.invoke("find-fit"); self.idle()
            result = self.bridge.dataset["result"]
            self.assertEqual(self.bridge.dataset["candidates"]["detected_rows"], list(range(5)))
            self.assertEqual({p["row_index"] for p in result["points"]}, set(range(5)))
            self.bridge.setParameter("row", 4); self.idle()
            self.assertIs(self.bridge.dataset["result"], result)
            self.assertEqual(self.bridge.dataset["row"], 4)

        def test_drr_group_raw_second_mixed_without_derivative(self):
            raw, second = cube(), cube(multiplier=-1.)
            sources = [dict(cube=item, group_cube=raw, kind="DRR", channel=channel, source="drr.csv", name=channel,
                source_name="DRR file", product_label="Mixed derivative" if channel == "second" else "Î”R/R",
                view={"xlim": [1.40, 1.45], "ylim": [1, 2]})
                for item, channel in ((raw, "raw"), (second, "second"))]
            datasets = prepare_sources(sources)
            self.assertEqual(datasets[0]["group_id"], datasets[1]["group_id"])
            self.assertEqual(datasets[1]["settings"]["detection_method"], "local")
            with patch("core.peak_candidates.savgol_filter", side_effect=AssertionError("Extra derivative or smoothing")):
                result = analyze_dataset(datasets[1], "detect")
            self.assertTrue(result["result"]["points"])
            self.assertEqual(datasets[1]["analysis_scope"], "map")
            self.assertEqual(datasets[1]["scope"], 0)
            self.assertEqual(result["candidates"]["detected_rows"], list(range(5)))
            self.assertEqual({point["row_index"] for point in result["result"]["points"]}, set(range(5)))
            self.assertTrue(any(point["energy"] > 1.45 for point in result["result"]["points"]))
            self.assertIn("dip", {point["polarity"] for point in result["result"]["points"]})
            self.bridge.addDatasets(datasets)
            self.assertEqual(len(self.bridge.snapshot["datasetChoices"]), 1)
            self.assertEqual(self.bridge.snapshot["products"][1]["label"], "Mixed derivative")
            self.bridge.setParameter("product", "raw")
            self.assertEqual(self.bridge.dataset["channel"], "raw")

        def test_figure_save_requires_applied_filters(self):
            from unittest.mock import patch
            source = dict(cube=cube(multiplier=-1), kind="DRR", channel="second", source="drr.csv", name="DRR")
            self.bridge.addDatasets(prepare_sources([source])); self.idle()
            dataset = self.bridge.dataset
            with patch.object(self.bridge, "_start") as start, patch.object(self.bridge, "_message") as message:
                self.bridge.saveFigure(str(self.root / "empty.png"))
                start.assert_called_once()
                start.reset_mock()
                dataset["filters_pending"] = True
                self.bridge.saveFigure(str(self.root / "pending.png"))
                start.assert_not_called()
                self.assertIn("Apply filters", message.call_args.args[0])
                self.bridge.invoke("save-figure")
                self.assertIn("Apply filters", message.call_args.args[0])
                dataset["filters_pending"] = False
                dataset["detection_dirty"] = True
                self.bridge.saveFigure(str(self.root / "dirty.png"), automatic=True)
                start.assert_not_called()
                self.assertIn("Find peaks", message.call_args.args[0])
                self.bridge.invoke("save-figure-auto")
                self.assertIn("Find peaks", message.call_args.args[0])

        def test_drr_full_dataset_controls_and_inspection(self):
            source = dict(cube=cube(multiplier=-1), kind="DRR", channel="second", source="drr.csv",
                name="DRR · second · " + "long_measurement_name_" * 6, product_label="2nd derivative")
            self.bridge.addDatasets(prepare_sources([source]))
            self.bridge.invoke("detect"); self.idle()
            result = self.bridge.dataset["result"]
            self.assertTrue(self.bridge.snapshot["autoFilters"])
            self.assertIn("Automatic starting filters", self.bridge.dataset["auto_filter_summary"])
            for key in ("min_snr", "min_width_mev", "neighbor_support", "neighbor_tolerance_mev"):
                self.assertEqual(self.bridge.dataset["settings"][key], result["settings"][key])
            self.bridge.setParameter("row", 3); self.idle()
            self.assertIs(self.bridge.dataset["result"], result)
            self.assertEqual(self.bridge.dataset["candidates"]["detected_rows"], list(range(5)))
            engine = QQmlApplicationEngine()
            engine.setInitialProperties({"peakBridge": self.bridge})
            engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / "ui_qt/quick/qml/PeakAnalysis.qml")))
            window = engine.rootObjects()[0]
            self.addCleanup(lambda: (window.setVisible(False), engine.deleteLater(), app.processEvents()))
            marker_size = find(window.contentItem(), "peak_marker_size")
            self.assertIsNotNone(marker_size)
            self.assertTrue(marker_size.property("visible"))
            self.assertEqual(self.bridge.snapshot["markerSize"], 4)
            self.bridge.setParameter("markerSize", 9); self.idle()
            self.assertEqual(self.bridge.snapshot["markerSize"], 9)
            self.bridge.setParameter("markerSize", 4); self.idle()
            for width, height, theme in ((1240, 900, "light"), (760, 600, "light"), (760, 600, "dark")):
                window.setWidth(width); window.setHeight(height)
                self.bridge._appearance.setMode(theme)
                self.idle()
                surface = window.findChild(QObject, "peak_plot_surface")
                wait(lambda: surface.property("presentedGeneration") > 0 and not surface._size_timer.isActive()
                     and not self.bridge._canvas.backend._timer.isActive())
                scope = find(window.contentItem(), "app_parameter_analysisScope")
                self.assertTrue(scope.property("visible"))
                button = find(window.contentItem(), "app_parameter_detect")
                self.assertEqual(button.property("text"), "Find candidates")
                apply = find(window.contentItem(), "app_parameter_filter")
                self.assertTrue(apply.property("visible"))
                self.assertTrue(apply.property("enabled"))
                groups = {group["key"]: group for group in self.bridge.snapshot["parameterGroups"]}
                self.assertEqual([group["key"] for group in self.bridge.snapshot["parameterGroups"]][:3],
                                 ["find-candidates", "detection-settings", "filters"])
                find_fields = {field["key"]: field for field in groups["find-candidates"]["fields"]}
                fields = {field["key"]: field for field in groups["filters"]["fields"]}
                self.assertTrue({"analysisScope", "detect", "candidate-explanation"}.issubset(find_fields))
                self.assertTrue({"polarity", "scope", "filterPreset", "showRejected"}.issubset(fields))
                self.assertEqual(fields["polarity"]["label"], "Keep")
                self.assertEqual(fields["scope"]["label"], "Keep range")
                self.assertTrue(all(not fields[key].get("visible", True) for key in ("x_min", "x_max", "y_min", "y_max")))
                for key in ("filter-settings", "detection-settings", "details"):
                    self.assertTrue(groups[key].get("advanced") or groups[key].get("collapsible"))
                    self.assertFalse(find(window.contentItem(), "app_parameter_group_body_" + key).property("visible"))
                from PySide6.QtCore import QPointF
                panel = window.findChild(QObject, "peak_analysis_controls")
                position = button.mapToItem(panel, QPointF(0, 0))
                self.assertGreaterEqual(position.y(), 0)
                self.assertLessEqual(position.y() + button.height(), panel.height())
                self.assertGreater(self.bridge._heat.bbox.height, self.bridge._spectrum.bbox.height)
                if os.environ.get("DPTK_PEAK_REVIEW_DIR"):
                    path = Path(os.environ["DPTK_PEAK_REVIEW_DIR"])
                    path.mkdir(parents=True, exist_ok=True)
                    window.grabWindow().save(str(path / f"drr-map-{width}-{theme}.png"))
            slider = find(window.contentItem(), "peak_map_height")
            slider.setProperty("value", 40)
            QMetaObject.invokeMethod(slider, "moved")
            self.idle()
            self.assertAlmostEqual(self.bridge._heat.bbox.height / self.bridge._spectrum.bbox.height, 40 / 60, places=2)
            pool = self.bridge.dataset["candidates"]
            self.bridge.setParameter("scope", 2); self.idle()
            groups = {group["key"]: group for group in self.bridge.snapshot["parameterGroups"]}
            fields = {field["key"]: field for field in groups["filters"]["fields"]}
            self.assertTrue(all(fields[key].get("visible", True) for key in ("x_min", "x_max", "y_min", "y_max")))
            heading = find(window.contentItem(), "app_parameter_group_filter-settings")
            QMetaObject.invokeMethod(heading, "clicked")
            wait(lambda: find(window.contentItem(), "app_parameter_group_body_filter-settings").property("visible"))
            self.assertTrue(find(window.contentItem(), "app_parameter_min_snr").property("visible"))
            if os.environ.get("DPTK_PEAK_REVIEW_DIR"):
                window.grabWindow().save(str(Path(os.environ["DPTK_PEAK_REVIEW_DIR"]) / "drr-expanded-custom.png"))
            self.bridge.setParameter("x_max", 1.44)
            self.bridge.setParameter("y_min", 1.)
            self.bridge.setParameter("y_max", 2.)
            self.idle()
            self.assertIs(self.bridge.dataset["result"], result, "Filter drafts must wait for Apply filters")
            self.assertTrue(self.bridge.snapshot["filtersPending"])
            self.assertFalse(self.bridge.snapshot["autoFilters"])
            self.assertFalse(self.bridge.snapshot["canExport"])
            with patch("core.peak_candidates.detect_heatmap", side_effect=AssertionError("Apply filters redetected")):
                QMetaObject.invokeMethod(find(window.contentItem(), "app_parameter_filter"), "clicked")
                self.idle()
            self.assertIs(self.bridge.dataset["candidates"], pool)
            self.assertFalse(self.bridge.snapshot["filtersPending"])
            self.assertTrue(self.bridge.dataset["result"]["points"])
            self.assertTrue(all(p["energy"] <= 1.44 and 1 <= p["y"] <= 2 for p in self.bridge.dataset["result"]["points"]))
            self.bridge.setParameter("polarity", "dips")
            self.bridge.invoke("reset-filters")
            self.assertEqual(self.bridge.dataset["settings"]["x_max"], 1.44)
            self.assertEqual(self.bridge.dataset["settings"]["y_min"], 1.)
            self.assertEqual(self.bridge.dataset["settings"]["polarity"], "dips")
            self.bridge.setParameter("polarity", "both")
            self.bridge.setParameter("scope", 0)
            self.bridge.invoke("filter"); self.idle()
            self.assertEqual(len(self.bridge.dataset["result"]["points"]), len(pool["points"]))
            with patch("core.peak_candidates.detect_heatmap", side_effect=AssertionError("Polarity filter redetected")):
                for polarity in ("peaks", "dips", "both"):
                    previous = self.bridge.dataset["result"]
                    self.bridge.setParameter("polarity", polarity); self.idle()
                    self.assertIs(self.bridge.dataset["result"], previous)
                    self.assertTrue(self.bridge.snapshot["filtersPending"])
                    self.bridge.invoke("filter"); self.idle()
                    kept = self.bridge.dataset["result"]["points"]
                    self.assertTrue(kept)
                    self.assertEqual({point["polarity"] for point in kept},
                        {"peak", "dip"} if polarity == "both" else {polarity[:-1]})
                    self.assertIs(self.bridge.dataset["candidates"], pool)
                    self.assertFalse(self.bridge.snapshot["filtersPending"])
                self.assertEqual(len(kept), len(pool["points"]))
                self.bridge.invoke("auto-filter"); self.idle()
                self.assertTrue(self.bridge.snapshot["autoFilters"])
                self.assertIs(self.bridge.dataset["candidates"], pool)
                self.assertFalse(self.bridge.snapshot["filtersPending"])
            from ui_qt.quick.figure_canvas import QuickNavigationToolbar
            self.assertIsInstance(self.bridge._navigation, QuickNavigationToolbar)
            saved = self.root / "layout.npz"
            self.bridge.saveWorkspace(str(saved)); self.idle()
            self.assertEqual(load_workspace(saved)[0][0]["map_height_percent"], 40)
            self.bridge.setParameter("mapHeightPercent", "auto"); self.idle()
            self.assertGreater(self.bridge._heat.bbox.height, self.bridge._spectrum.bbox.height)
            self.bridge.setParameter("analysisScope", "spectrum"); self.idle()
            self.assertTrue(find(window.contentItem(), "app_parameter_analysisScope").property("visible"))
            self.bridge.setParameter("scope", 2); self.idle()
            groups = {group["key"]: group for group in self.bridge.snapshot["parameterGroups"]}
            fields = {field["key"]: field for field in groups["filters"]["fields"]}
            self.assertTrue(fields["x_min"].get("visible", True))
            self.assertFalse(fields["y_min"].get("visible", True))
            self.assertFalse(fields["y_max"].get("visible", True))
            self.bridge.invoke("detect"); self.idle()
            self.assertEqual(self.bridge.dataset["candidates"]["detected_rows"], [3])

        def test_drr_preview_explains_applied_filters_and_restores(self):
            source = dict(cube=cube(multiplier=-1), kind='DRR', channel='second', source='preview.csv', name='Filter preview')
            self.bridge.addDatasets(prepare_sources([source]))
            self.bridge.invoke('detect'); self.idle()
            self.bridge.setParameter('min_width_mev', 100.)
            self.bridge.invoke('filter'); self.idle()
            self.bridge.setParameter('showRejected', True); self.idle()
            dataset = self.bridge.dataset
            preview = dataset['result']['filter_preview']
            self.assertTrue(preview['reasons'])
            self.assertTrue(self.bridge.snapshot['showRejected'])
            marker = next(p for p in dataset['candidates']['points'] if preview['reasons'].get(p['candidate_id']) == 'min_width')
            heat = self.bridge._heat
            before_draws = self.bridge._canvas.draw_count
            self.bridge._cursor_motion(SimpleNamespace(inaxes=heat, xdata=marker['energy'], ydata=marker['y']))
            self.bridge._cursor_flush()
            self.assertIn('Excluded', self.bridge.cursor['text'])
            self.assertIn('minimum width', self.bridge.cursor['tooltip'])
            self.assertEqual(self.bridge._canvas.draw_count, before_draws)
            self.bridge.setParameter('min_width_mev', 0.); self.idle()
            self.assertIn('last applied', self.bridge._filter_preview_summary(dataset))
            self.assertIs(dataset['result']['filter_preview'], preview)
            engine = QQmlApplicationEngine()
            engine.setInitialProperties({'peakBridge': self.bridge})
            engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / 'ui_qt/quick/qml/PeakAnalysis.qml')))
            window = engine.rootObjects()[0]
            self.addCleanup(lambda: (window.setVisible(False), engine.deleteLater(), app.processEvents()))
            for theme in ('light', 'dark'):
                window.setWidth(1080); window.setHeight(800)
                self.bridge._appearance.setMode(theme); self.idle()
                self.assertTrue(find(window.contentItem(), 'app_parameter_showRejected').property('visible'))
                if os.environ.get('DPTK_PEAK_REVIEW_DIR'):
                    path = Path(os.environ['DPTK_PEAK_REVIEW_DIR']); path.mkdir(parents=True, exist_ok=True)
                    window.grabWindow().save(str(path / f'drr-filter-preview-{theme}.png'))
            saved = self.root / 'preview.npz'
            self.bridge.saveWorkspace(str(saved)); self.idle()
            restored = load_workspace(saved)[0][0]
            self.assertTrue(restored['show_rejected'])
            self.assertEqual(restored['result']['filter_preview'], preview)

        def test_scatter_filter_is_manual_reversible_and_not_a_detection_setting(self):
            self.bridge.addDatasets(prepare_sources([dict(cube=cube(multiplier=-1), kind='DRR',
                channel='second', source='scatter.csv', name='Scatter controls')]))
            self.bridge.invoke('detect'); self.idle()
            dataset = self.bridge.dataset
            result, pool = dataset['result'], dataset['candidates']
            self.bridge.setParameter('remove_scatter', True); self.idle()
            self.assertIs(dataset['result'], result)
            self.assertTrue(dataset['filters_pending'])
            self.assertFalse(dataset['auto_filters'])
            with patch('core.peak_candidates.detect_heatmap', side_effect=AssertionError('Scatter filter redetected')):
                self.bridge.invoke('filter'); self.idle()
            self.assertIs(dataset['candidates'], pool)
            self.assertTrue(dataset['result']['settings']['remove_scatter'])
            saved = self.root / 'scatter.npz'
            self.bridge.saveWorkspace(str(saved)); self.idle()
            self.assertTrue(load_workspace(saved)[0][0]['settings']['remove_scatter'])
            self.bridge.invoke('preset:Sensitive'); self.idle()
            self.assertFalse(dataset['settings']['remove_scatter'])
            self.bridge.setParameter('remove_scatter', True)
            self.bridge.invoke('reset-filters'); self.idle()
            self.assertFalse(dataset['settings']['remove_scatter'])
            self.bridge.invoke('filter'); self.idle()
            self.assertEqual(len(dataset['result']['points']), len(pool['points']))

        def test_width_adaptive_filter_is_pending_reversible_and_saved(self):
            self.bridge.addDatasets(prepare_sources([dict(cube=cube(multiplier=-1), kind='DRR',
                channel='second', source='adaptive.csv', name='Adaptive controls')]))
            self.bridge.invoke('detect'); self.idle()
            dataset = self.bridge.dataset
            result, pool = dataset['result'], dataset['candidates']
            self.assertTrue(dataset['settings']['neighbor_adaptive'])
            field = next(f for g in self.bridge._parameter_groups() for f in g['fields']
                         if f['key'] == 'neighbor_adaptive')
            self.assertTrue(field['visible'])
            self.bridge.setParameter('neighbor_adaptive', False); self.idle()
            self.assertIs(dataset['result'], result)
            self.assertTrue(dataset['filters_pending'])
            self.assertFalse(dataset['auto_filters'])
            with patch('core.peak_candidates.detect_heatmap', side_effect=AssertionError('Redetected')):
                self.bridge.invoke('filter'); self.idle()
                off_ids = {p['candidate_id'] for p in dataset['result']['points']}
                self.bridge.setParameter('neighbor_adaptive', True)
                self.bridge.invoke('filter'); self.idle()
                self.assertIs(dataset['candidates'], pool)
                self.assertTrue(dataset['result']['settings']['neighbor_adaptive'])
                saved = self.root / 'adaptive.npz'
                self.bridge.saveWorkspace(str(saved)); self.idle()
                self.assertTrue(load_workspace(saved)[0][0]['settings']['neighbor_adaptive'])
                self.bridge.setParameter('neighbor_adaptive', False)
                self.bridge.invoke('filter'); self.idle()
                self.assertEqual(off_ids, {p['candidate_id'] for p in dataset['result']['points']})
            self.bridge.setParameter('neighbor_adaptive', True)
            self.bridge.invoke('preset:Sensitive'); self.idle()
            self.assertFalse(dataset['settings']['neighbor_adaptive'])
            self.bridge.setParameter('neighbor_adaptive', True)
            self.bridge.invoke('reset-filters'); self.idle()
            self.assertFalse(dataset['settings']['neighbor_adaptive'])

        def test_detection_cache_filters_preserve_branch_identity(self):
            dataset = self.install()
            self.detect()
            pool = dataset["candidates"]
            original = dataset["branches"][0]
            identity, color = original["id"], original["color"]
            self.bridge.setParameter("branch." + identity + ".name", "Exciton")
            with patch("core.peak_candidates.detect_heatmap", side_effect=AssertionError("Filter redetected the map")):
                self.bridge.setParameter("min_snr", .1)
                self.idle()
            self.assertIs(dataset["candidates"], pool)

            branch = next(item for item in dataset["branches"] if item["id"] == identity)
            self.assertEqual((branch["name"], branch["color"]), ("Exciton", color))
            self.bridge.setParameter("detection_method", "sg")
            self.assertIn("Detection changed", self.bridge.snapshot["cacheStatus"])
            self.assertIs(dataset["candidates"], pool)

        def test_hidden_models_are_lazy_and_spectrum_refresh_is_coalesced(self):
            dataset = self.install()
            self.bridge.setParameter("analysisScope", "spectrum")
            self.bridge.invoke("detect"); self.idle()
            self.assertEqual(self.bridge.resultsModel.rowCount(), 0)
            self.assertEqual(self.bridge.branchesModel.rowCount(), 0)
            self.bridge.invoke("results")
            self.assertEqual(self.bridge.resultsModel.rowCount(), len(dataset["result"]["points"]))
            self.bridge.setParameter("controlsPage", "branches")
            self.assertEqual(self.bridge.branchesModel.rowCount(), len(dataset["branches"]))
            self.bridge.invoke("close-results")
            self.bridge.setParameter("controlsPage", "find")
            self.assertEqual(self.bridge.resultsModel.rowCount(), 0)
            self.assertEqual(self.bridge.branchesModel.rowCount(), 0)
            with patch.object(self.bridge._spectrum_pool, "update", wraps=self.bridge._spectrum_pool.update) as update:
                self.bridge.setParameter("row", 1)
                self.idle()
                self.assertEqual(update.call_count, 1, "A row change must not repaint the spectrum twice")
        def test_filter_and_overlay_updates_retain_measured_map_and_navigation(self):
            self.install();self.detect()
            heat=self.bridge._heat;mesh=heat.collections[0];canvas=self.bridge._canvas
            self.bridge._navigation.push_current()
            history=len(self.bridge._navigation._nav_stack)
            draws,blits=canvas.draw_count,canvas.blit_count
            self.bridge.setParameter("min_snr",.1);self.idle()
            self.bridge.invoke('filter'); self.idle()
            self.assertIs(self.bridge._heat,heat)
            self.assertIs(heat.collections[0],mesh)
            self.assertEqual(canvas.draw_count,draws)
            self.assertGreater(canvas.blit_count,blits)
            self.assertEqual(len(self.bridge._navigation._nav_stack),history)
            self.bridge.setParameter("showOverlay",False);self.idle()
            self.assertEqual(self.bridge._overlay_artists,[])
            self.assertEqual(canvas.draw_count,draws)
            self.bridge.setParameter("showOverlay",True);self.idle()
            self.assertTrue(self.bridge._overlay_artists)
            self.assertTrue(all(artist.get_animated() for artist in self.bridge._overlay_artists))
            self.bridge.setParameter("showResidual",True);self.idle()
            self.assertIs(self.bridge._heat,heat)
            self.assertIsNone(self.bridge._residual)
            self.bridge.invoke('fit');self.idle()
            self.assertIsNot(self.bridge._heat,heat)
            self.assertIsNotNone(self.bridge._residual)
            self.assertGreater(canvas.draw_count,draws)

        def test_pl_lorentz_gauss_fit_residual_and_export_restore(self):
            for model in ("Lorentzian", "Gaussian"):
                with self.subTest(model=model):
                    dataset = self.install(model, source=model)
                    dataset["settings"]["model"] = model
                    self.detect()
                    self.bridge.invoke("fit")
                    self.idle()
                    self.assertEqual(len(dataset["result"]["fits"]), 5)
                    self.assertTrue(all(point["fit_status"] == "ok" for point in dataset["result"]["points"]))
                    self.assertLess(abs(dataset["result"]["points"][0]["fit_center"] - 1.415), .0001)
            self.bridge.setParameter("showResidual", True)
            self.idle()
            self.assertIsNotNone(self.bridge._residual)
            self.bridge.setParameter("analysisPlot", "trends")
            self.idle()
            self.assertEqual(len(self.bridge._trend_axes), 2)
            self.assertIsNone(self.bridge._heat)
            self.assertTrue(any(axis.containers for axis in self.bridge._trend_axes))
            self.bridge.setParameter("analysisPlot", "spectra")
            self.idle()
            self.assertIsNotNone(self.bridge._heat)
            self.bridge.dataset.setdefault('view', {})['display'] = dict(cmap='viridis', vmin=0., vmax=2.)
            self.bridge.exportTo(str(self.root))
            self.idle()
            exported = [Path(path) for path in self.bridge.snapshot['resultPaths']]
            self.assertEqual({path.name for path in exported}, {"Gaussian_analysis_settings.json", "Gaussian_peak_overlay.png", "Gaussian_peak_results.xlsx"})
            self.assertTrue(all(path.parent.samefile(self.root) for path in exported))
            from openpyxl import load_workbook
            workbook = load_workbook(self.root / "Gaussian_peak_results.xlsx", read_only=True)
            try:
                self.assertEqual(set(workbook.sheetnames), {'Detected_peaks', 'Fitted_peaks', 'FWHM', 'Height', 'Area'})
                self.assertTrue(any('eV' in str(header) for header in next(workbook['Fitted_peaks'].values)))
                self.assertTrue(any('meV' in str(header) for header in next(workbook['FWHM'].values)))
            finally:
                workbook.close()
            self.assertFalse((self.root / "Gaussian_peak_details.csv").exists())
            self.assertIn('fit_center', (self.root / 'Gaussian_analysis_settings.json').read_text(encoding='utf-8'))
            self.bridge.saveWorkspace(str(self.root / "copy.npz"))
            self.idle()
            restored, active = load_workspace(self.root / "copy.npz")
            self.assertEqual(len(restored), 2)
            self.assertEqual(active, self.bridge.active_key)
            self.assertTrue(all(len(item["result"]["fits"]) == 5 for item in restored))
            self.assertEqual(self.settings.value("unrelated/preserved"), "keep")

        def test_manual_seed_current_spectrum_and_fit(self):
            dataset = self.install()
            self.bridge._start_seed(1.415)
            self.idle()
            self.assertEqual(len(dataset["branches"]), 1)
            self.assertTrue(dataset['manual_peak_edits'])
            self.assertEqual(len(dataset["result"]["points"]), 1)
            self.assertEqual(dataset['result']['points'][0]['energy'], 1.415)
            self.bridge.invoke("fit")
            self.idle()
            self.assertTrue(dataset["result"]["fits"])
            self.assertEqual({f['row_index'] for f in dataset['result']['fits']}, {dataset['row']})

        def test_cursor_marker_and_row_selection_retain_heatmap(self):
            dataset = self.install()
            self.detect()
            heat, mesh = self.bridge._heat, self.bridge._heat.collections[0]
            count = self.bridge._canvas.draw_count
            for _ in range(30):
                self.bridge._cursor_motion(SimpleNamespace(inaxes=heat, xdata=1.415, ydata=2.))
            self.assertTrue(self.bridge._cursor_timer.isActive())
            self.bridge._cursor_flush()
            self.assertIn("Z (nearest)", self.bridge.cursor["text"])
            self.assertIn("before color clipping", self.bridge.cursor["tooltip"])
            self.assertEqual(self.bridge._canvas.draw_count, count)
            self.bridge.setParameter("markerSize", 16)
            self.idle()
            count = self.bridge._canvas.draw_count
            self.bridge.setParameter("resultRow", 6)
            self.idle()
            self.assertIs(self.bridge._heat, heat)
            self.assertIs(self.bridge._heat.collections[0], mesh)
            self.assertEqual(self.bridge._canvas.draw_count, count)
            self.assertGreater(self.bridge._canvas.blit_count, 0)

        def test_fitted_row_updates_reuse_artists_without_heatmap_draw_or_restyle(self):
            self.install()
            self.detect()
            self.bridge.invoke("fit")
            self.idle()
            self.bridge.setParameter("showResidual", True)
            self.idle()
            heat, spectrum, residual = self.bridge._heat, self.bridge._spectrum, self.bridge._residual
            pool = self.bridge._spectrum_pool
            artists = (pool.measured, pool.smoothed, pool.total, *pool.components, pool.residual, pool.markers)
            draw_count, blit_count = self.bridge._canvas.draw_count, self.bridge._canvas.blit_count
            with patch("core.peak_plotting.draw_spectrum", side_effect=AssertionError("Row cleared/rebuilt spectrum")), \
                 patch.object(self.bridge, "_style_figure", side_effect=AssertionError("Row restyled full figure")), \
                 patch.object(heat, "draw", side_effect=AssertionError("Row redrew heatmap")):
                for row in (1, 4, 2, 0):
                    self.bridge.setParameter("row", row)
                    self.idle()
            self.assertEqual(self.bridge._canvas.draw_count, draw_count)
            self.assertEqual(self.bridge._canvas.blit_count, blit_count + 4)
            self.assertEqual((self.bridge._heat, self.bridge._spectrum, self.bridge._residual), (heat, spectrum, residual))
            self.assertEqual(artists, (pool.measured, pool.smoothed, pool.total, *pool.components, pool.residual, pool.markers))

        def test_queued_handoff_cancel_and_auto_save(self):
            dataset = self.install()
            replacement = create_dataset(cube(multiplier=2.), "PL", "PL", "second.csv", "second.csv")
            snapshot = self.root / "handoff.npz"
            save_workspace(snapshot, [replacement], replacement["key"])
            entered, release = Event(), Event()
            def slow(cancel, progress):
                entered.set()
                release.wait(5)
                return analyze_dataset(dataset, "detect", cancelled=cancel, progress=progress)
            self.bridge._start("detect", slow, dataset_key=dataset["key"])
            self.assertTrue(entered.wait(5))
            self.bridge.loadSnapshot(snapshot, True, "new-reply")
            self.bridge.invoke("cancel")
            release.set()
            wait(lambda: len(self.bridge.datasets) == 2 and not self.bridge.snapshot["busy"])
            self.assertEqual(self.bridge.dataset["reply_server"], "new-reply")
            self.assertIsNone(dataset["result"])
            self.bridge.invoke("restart")
            wait(lambda: self.bridge.snapshot["closeReady"])
            saved, _ = load_workspace(self.root / "workspace.npz")
            self.assertEqual(len(saved), 2)
            self.assertTrue(self.bridge.snapshot["restartRequested"])

        def test_restart_during_running_analysis_cancels_then_saves(self):
            dataset = self.install()
            entered, release = Event(), Event()
            def slow(cancel, progress):
                entered.set()
                release.wait(5)
                return analyze_dataset(dataset, "detect", cancelled=cancel, progress=progress)
            self.bridge._start("detect", slow, dataset_key=dataset["key"])
            self.assertTrue(entered.wait(5))
            self.bridge.invoke("restart")
            release.set()
            wait(lambda: self.bridge.snapshot["closeReady"])
            saved, _ = load_workspace(self.root / "workspace.npz")
            self.assertEqual(len(saved), 1)
            self.assertIsNone(saved[0]["result"])
            self.assertTrue(self.bridge.snapshot["restartRequested"])

        def test_launcher_snapshot_validation_stale_replies_and_launch_args(self):
            from core.peak_plotting import overlay_payload
            from matplotlib.figure import Figure
            from ui_qt.quick.figure_canvas import QuickFigureCanvas
            launcher = QuickPeakAnalysisLauncher(settings=self.settings, session_path=self.root / "child" / "workspace.npz")
            self.addCleanup(launcher.close)
            original = cube()
            canvas = QuickFigureCanvas(Figure())
            self.addCleanup(canvas.close)
            source = dict(cube=original, kind="PL", channel="PL", source="source.csv", name="source", axes=[canvas.figure.subplots()])
            with patch("ui_qt.quick.peak_analysis_launcher.QProcess.startDetached", return_value=(True, 42)) as launch:
                launcher.openSources([source])
                wait(lambda: not launcher.snapshot["busy"])
                self.assertEqual(launch.call_count, 1)
                arguments = launch.call_args.args[1]
                self.assertIn("--settings", arguments)
                self.assertIn("--snapshot", arguments)
                self.assertIn("run_peak_analysis_quick.py", arguments[0])
            dataset = analyze_dataset(prepare_sources([source])[0], "detect")
            received = []
            launcher.overlayReady.connect(received.append)
            launcher.accept_overlay(overlay_payload(dataset))
            wait(lambda: not launcher.snapshot["busy"])
            self.assertEqual(len(received), 1)
            self.assertTrue(launcher._artists)
            original.Z[0, 0] += 2.
            launcher.accept_overlay(overlay_payload(dataset))
            wait(lambda: not launcher.snapshot["busy"])
            self.assertEqual(len(received), 1)
            self.assertFalse(launcher._artists)
            self.assertFalse(launcher.snapshot["hasOverlay"])
            launcher.receive({"overlay": str(self.root / "outside.json")})
            self.assertFalse(launcher.snapshot["busy"])
            self.assertNotEqual(server_name(launcher.session_path), server_name(launcher.session_path, legacy=True))

        def test_simple_spectrum_custom_filters_and_scope(self):
            datasets = prepare_sources([dict(cube=cube(), kind="PL", channel="PL", source="simple.csv", name="PL sample",
                row=2, analysis_scope="spectrum", view={"xlim": [1.40, 1.45], "ylim": [0, 4]})])
            self.bridge.addDatasets(datasets)
            dataset = self.bridge.dataset
            self.assertEqual(dataset["analysis_scope"], "spectrum")
            self.assertEqual(dataset["settings"]["x_min"], 1.4)
            self.assertEqual(dataset["scope"], 2)
            engine = QQmlApplicationEngine()
            engine.setInitialProperties({"peakBridge": self.bridge})
            engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / "ui_qt/quick/qml/PeakAnalysis.qml")))
            self.assertTrue(engine.rootObjects())
            window = engine.rootObjects()[0]
            self.addCleanup(lambda: (window.setVisible(False), engine.deleteLater(), app.processEvents()))
            self.idle()
            self.bridge.setParameter('controlsPage', 'branches'); self.idle()
            wait(lambda: find(window.contentItem(), "peak_workflow_primary") is not None)
            QMetaObject.invokeMethod(find(window.contentItem(), 'app_parameter_group_review-filters'), 'clicked', Qt.ConnectionType.DirectConnection)
            snr = lambda: find(window.contentItem(), "app_parameter_min_snr")
            self.assertFalse(snr().property("visible"))
            self.bridge.setParameter("filterPreset", "Custom")
            wait(lambda: snr().property("visible"))
            self.bridge.setParameter("min_snr", 1.)
            self.bridge.invoke("find-fit")
            self.idle()
            self.assertEqual({point["row_index"] for point in dataset["result"]["points"]}, {2})
            self.assertTrue(dataset["result"]["fits"])
            self.assertEqual(dataset["result"]["settings"]["neighbor_support"], 0)
            self.assertTrue(find(window.contentItem(), "app_parameter_group_body_review-filters").property("visible"))
            surface = window.findChild(QObject, "peak_plot_surface")
            wait(lambda: surface.property("presentedGeneration") > 0)
            control = window.findChild(QObject, "peak_analysis_controls")
            self.assertGreater(control.property("x"), surface.property("x"))
            review = os.environ.get("DPTK_PEAK_REVIEW_DIR")
            if review:
                Path(review).mkdir(parents=True, exist_ok=True)
                self.assertTrue(window.grabWindow().save(str(Path(review) / "pl-custom.png")))
            self.bridge.setParameter("filterPreset", "Balanced")
            self.idle()
            wait(lambda: not snr().property("visible"))
            self.assertFalse(dataset["result"]["fits"])
            self.bridge.invoke("find-fit"); self.idle()
            saved = self.root / "simple-copy.npz"
            self.bridge.saveWorkspace(str(saved)); self.idle()
            self.bridge.setParameter("filterPreset", "Strict"); self.idle()
            self.bridge.loadSnapshot(saved, fresh=False); self.idle()
            dataset = self.bridge.dataset
            self.assertTrue(dataset["result"]["fits"])
            self.assertEqual(dataset["analysis_scope"], "spectrum")
            self.assertEqual(dataset["settings"]["prominence"], .05)
            if review:
                self.assertTrue(window.grabWindow().save(str(Path(review) / "pl-balanced.png")))
                self.bridge._appearance.setMode("dark")
                self.idle()
                self.assertTrue(window.grabWindow().save(str(Path(review) / "pl-dark.png")))
            self.bridge.setParameter("row", 3); self.idle()
            self.assertFalse(dataset.get("result"))
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.bridge.saveWorkspace(str(self.root / "changed-row.npz")); self.idle()
            self.assertIsNone(load_workspace(self.root / "changed-row.npz")[0][0]["result"])
            self.bridge.invoke("find-fit"); self.idle()
            self.assertEqual(dataset["candidates"]["detected_rows"], [3])
            self.bridge.setParameter("analysisScope", "map"); self.idle()
            self.assertFalse(dataset.get("candidates"))
            self.bridge.invoke("detect"); self.idle()
            self.assertEqual(dataset["candidates"]["detected_rows"], list(range(5)))

        def test_single_spectrum_drr_product_and_handoff_clear_stale_results(self):
            sources = [dict(cube=cube(multiplier=sign), group_cube=cube(), kind="DRR", channel=channel,
                source="products.csv", name=channel, row=0, analysis_scope="spectrum") for sign, channel in ((1, "raw"), (-1, "second"))]
            datasets = prepare_sources(sources)
            self.bridge.addDatasets(datasets)
            self.bridge.invoke("detect"); self.idle()
            self.assertTrue(self.bridge.dataset["result"])
            self.bridge.setParameter("product", "raw")
            self.bridge.setParameter("row", 2)
            self.bridge.setParameter("product", "second"); self.idle()
            self.assertEqual(self.bridge.dataset["row"], 2)
            self.assertIsNone(self.bridge.dataset["result"])
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.bridge.invoke("detect"); self.idle()
            sources[1]["row"] = 3
            self.bridge.addDatasets(prepare_sources([sources[1]])); self.idle()
            self.assertEqual(self.bridge.dataset["row"], 3)
            self.assertIsNone(self.bridge.dataset["result"])
            self.bridge.saveWorkspace(str(self.root / "product-change.npz")); self.idle()
            restored, _ = load_workspace(self.root / "product-change.npz")
            self.assertTrue(all(item["analysis_scope"] == "spectrum" for item in restored))
            self.bridge.invoke("detect"); self.idle()
            review = os.environ.get("DPTK_PEAK_REVIEW_DIR")
            if review:
                engine = QQmlApplicationEngine()
                engine.setInitialProperties({"peakBridge": self.bridge})
                engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / "ui_qt/quick/qml/PeakAnalysis.qml")))
                window = engine.rootObjects()[0]
                self.addCleanup(lambda: (window.setVisible(False), engine.deleteLater(), app.processEvents()))
                surface = window.findChild(QObject, "peak_plot_surface")
                wait(lambda: surface.property("presentedGeneration") > 0)
                self.assertTrue(window.grabWindow().save(str(Path(review) / "drr-both.png")))
                window.setWidth(760); window.setHeight(600)
                app.processEvents()
                self.assertTrue(window.grabWindow().save(str(Path(review) / "drr-compact.png")))

        def test_empty_qml_workflow_status_before_dataset_arrives(self):
            engine = QQmlApplicationEngine()
            engine.setInitialProperties({"peakBridge": self.bridge})
            engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / "ui_qt/quick/qml/PeakAnalysis.qml")))
            self.assertTrue(engine.rootObjects())
            window = engine.rootObjects()[0]
            self.addCleanup(lambda: (window.setVisible(False), engine.deleteLater(), app.processEvents()))
            status = find(window, "peak_workflow_status")
            self.assertIsNotNone(status)
            self.assertEqual(status.property("text"), "")
            self.install()
            self.idle()
            self.assertEqual(status.property("text"), self.bridge.snapshot["workflow"]["status"])

        def test_populated_qml_native_surface_controls_and_results(self):
            self.install()
            self.detect()
            engine = QQmlApplicationEngine()
            engine.setInitialProperties({"peakBridge": self.bridge})
            engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / "ui_qt/quick/qml/PeakAnalysis.qml")))
            self.assertTrue(engine.rootObjects())
            window = engine.rootObjects()[0]
            self.addCleanup(lambda: (window.setVisible(False), engine.deleteLater(), app.processEvents()))
            surface = window.findChild(QObject, "peak_plot_surface")
            wait(lambda: surface.property("presentedGeneration") > 0)
            self.bridge.invoke("results")
            self.bridge.setParameter("controlsPage", "fit")
            wait(lambda: find(window.contentItem(), "app_parameter_model") is not None)
            self.assertTrue(window.findChild(QObject, "peak_analysis_results").property("visible"))
            self.assertFalse(window.grabWindow().isNull())
            self.bridge.setParameter("controlsPage", "find")
            wait(lambda: find(window.contentItem(), "app_parameter_detection_method") is not None)
            self.assertTrue(find(window.contentItem(), "peak_workflow_primary").property("enabled"))
            from tests.quick_selection_checks import assert_text_selection_palette
            for mode in ("light", "dark", "high-contrast"):
                self.bridge._appearance.setMode(mode); self.idle(); app.processEvents()
                assert_text_selection_palette(self, window, self.bridge.colors)
            self.bridge._appearance.setMode("light"); self.idle()
            for name in ("peak_file_dialog", "peak_export_dialog"):
                dialog = window.findChild(QObject, name)
                self.assertIsNotNone(dialog)
                self.assertTrue(QMetaObject.invokeMethod(dialog, "open", Qt.ConnectionType.DirectConnection))
                self.assertTrue(QMetaObject.invokeMethod(dialog, "reject", Qt.ConnectionType.DirectConnection))
                app.processEvents()

        def test_legacy_owner_blocks_shared_workspace(self):
            from ui_qt.peak_analysis_ipc import MessageServer
            server = MessageServer(server_name(self.bridge.session_path, legacy=True))
            self.addCleanup(lambda owner=server: owner.server.close())
            self.assertTrue(server.listening)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "run_peak_analysis_quick.py"),
                "--session", str(self.bridge.session_path), "--settings", str(self.root / "child.ini")],
                capture_output=True, text=True, env=dict(os.environ), timeout=20)
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("legacy Peak Analysis", result.stderr)
            self.assertFalse(self.bridge.session_path.exists())

        def test_source_adapter_drr_uses_published_derivative_provenance(self):
            from matplotlib.figure import Figure
            from ui_qt.quick.drr_workflow import DrrRequest, process_drr
            from ui_qt.quick.peak_source_adapters import capture_peak_sources
            original = cube()
            request = DrrRequest(("original.csv",), raw_derivative=1, second_derivative=("mixed", 5), window=11, polyorder=2)
            products = process_drr(original, request)
            figure = Figure()
            axes = figure.subplots(1, 2)
            owner = SimpleNamespace(_workflow="drr", _busy=False, _closed=False, _plot_timer=None, _cube=products.raw,
                _drr_products=products, _source=Path("original.csv"), _sources=("original.csv",),
                _maps=[(axis, None) for axis in axes], _plot_cubes=[products.raw, products.second],
                _values={"gate": 3.}, _drr_processing={"sgWindow": 31, "secondDerivative": "second"})
            with patch("core.processing.apply_sg_derivative_energy", side_effect=AssertionError("Adapter recalculated derivatives")):
                descriptors = capture_peak_sources(owner)
            self.assertEqual([item["product_label"] for item in descriptors], ["1st derivative", "Mixed derivative"])
            self.assertEqual(descriptors[0]["source_processing"]["window_samples"], 11)
            self.assertEqual(descriptors[1]["source_processing"]["scan_window_samples"], 5)
            self.assertTrue(all(item["row"] == 3 for item in descriptors))
            self.assertIs(descriptors[0]["cube"], products.raw)
            self.assertIs(descriptors[1]["cube"], products.second)
            self.assertIs(descriptors[0]["group_cube"], products.source)
            owner._plot_timer = SimpleNamespace(isActive=lambda: True)
            self.assertEqual(capture_peak_sources(owner), [])

        def test_source_adapter_power_keeps_physical_coordinates_and_log_transform(self):
            from dataclasses import replace
            from matplotlib.figure import Figure
            from ui_qt.quick.peak_source_adapters import capture_peak_sources
            measured = replace(cube(), gate=np.array([0., 1., 10., 100., 1000.]))
            axis = Figure().subplots()
            axis.set_yscale("log")
            axis.set_ylim(1., 1000.)
            child = SimpleNamespace(_closed=False, _busy=False, _plot_timer=None, _stage="plot", _running=set(),
                _values={"mode": "single", "gate": 1000.}, _products={"mode": "single", "cubes": {"KK": measured}},
                _maps={"KK": axis}, _display_coordinates={"KK": (measured.gate[1:], measured.gate[1:], np.arange(1, 5))},
                _folder="power", _active_selection={"single": "sample.csv"}, _analysis_selected_rows={"KK": 2}, _selected_row_index=4)
            owner = SimpleNamespace(_workflow="power", _closed=False, _busy=False, _workflow_bridges={"power": child})
            descriptor = capture_peak_sources(owner)[0]
            self.assertIs(descriptor["cube"], measured)
            self.assertEqual(descriptor["row"], 2)
            self.assertTrue(descriptor["view"]["y_axis_log"])
            mapped = descriptor["transform_y"]([0., 1., 10., 1000.])
            self.assertTrue(np.isnan(mapped[0]))
            np.testing.assert_array_equal(mapped[1:], [1., 10., 1000.])
            # Duplicate input stays untouched and the existing workspace
            # validator gives a readable separate/aggregate-repeats error.
            measured.gate[2] = 1.
            captured = capture_peak_sources(owner)
            self.assertIs(captured[0]["cube"], measured)
            with self.assertRaisesRegex(ValueError, "Repeated scan coordinates"):
                prepare_sources(captured)
            child._values["mode"] = "vp"
            self.assertEqual(capture_peak_sources(owner), [])

        def test_source_adapter_pl_and_compare_preserve_linear_intensity(self):
            from matplotlib.figure import Figure
            from ui_qt.quick.peak_source_adapters import capture_peak_sources
            scientific = cube()
            axes = Figure().subplots(1, 2)
            owner = SimpleNamespace(_workflow="pl", _busy=False, _closed=False, _plot_timer=None,
                _cube=scientific, _plot_cubes=[scientific, scientific], _maps=[(axis, None) for axis in axes],
                _source=Path("pl.csv"), _sources=("pl.csv",), _values={"gate": 2., "scale": "log"})
            axes[1].set_xlim(1.40, 1.50)
            descriptor = capture_peak_sources(owner)[0]
            self.assertIs(descriptor["cube"].Z, scientific.Z)
            self.assertEqual(descriptor["view"]["xlim"], [1.40, 1.50])
            self.assertEqual(descriptor["axes"], list(axes))
            child = SimpleNamespace(_busy=False, _closed=False, _plot_timer=None, _error="", _values={"gate": 1., "view": "intensity"},
                _derived={"KK": scientific, "KKp": scientific}, _cubes={"KK": scientific, "KKp": scientific},
                _maps=dict(zip(("KK", "KKp"), axes)), _folder="compare", _loaded_mapping=(("KK", "a.csv"), ("KKp", "b.csv")),
                _assignments={"KK": "a.csv", "KKp": "b.csv"}, _active_keys=lambda: ["KK", "KKp"])
            owner._workflow = "cmp"
            owner._workflow_bridges = {"cmp": child}
            descriptors = capture_peak_sources(owner)
            self.assertEqual(len(descriptors), 2)
            self.assertTrue(all(item["kind"] == "PL" and item["cube"] is scientific for item in descriptors))
            child._values["view"] = "vp"
            self.assertEqual(capture_peak_sources(owner), [])

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    qInstallMessageHandler(old_handler)
    print("QUICK_PEAK_ANALYSIS_TESTS=" + str(result.testsRun))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
