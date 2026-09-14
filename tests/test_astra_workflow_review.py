"""Independent regressions for the reviewed workflow cache contracts."""
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
import unittest
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from core.shg import ShgSettings
from core.shg_fit import ShgFitSettings
from ui_qt.common import LoadedState
from ui_qt.controllers_shg import ShgController, _ShgWorker


class AstraShgReviewTests(unittest.TestCase):
    def fixture(self):
        settings = ShgSettings(background_method="external")
        data = SimpleNamespace(source_file="sample.csv", spectra=object(), revision=1)
        background = SimpleNamespace(source_file="background.csv", spectra=object(), revision=1)
        result = SimpleNamespace(data=data, settings=settings)
        loaded = LoadedState(mode="SHG Processing", folder="synthetic", shg_data=data,
                             shg_background=background, shg_result=result, shg_settings=settings,
                             shg_compare=False)
        workers = []
        owner = SimpleNamespace(loaded=loaded, _load_lifecycle_generation=1,
                                _status=lambda message: None,
                                thread_pool=SimpleNamespace(start=workers.append))
        controller = ShgController(owner)
        object.__setattr__(controller, "_shg_settings_from_ui", lambda: settings)
        object.__setattr__(controller, "_shg_fit_settings_from_ui", lambda: ShgFitSettings())
        return controller, loaded, settings, workers

    def test_controller_dispatch_passes_valid_processed_pair(self):
        controller, loaded, _, workers = self.fixture()
        controller._start_shg_reprocess()
        self.assertEqual(len(workers), 1)
        self.assertIs(workers[0].payload[7][0], loaded.shg_result)

    def test_background_replacement_does_not_reseed_old_result(self):
        controller, loaded, settings, _ = self.fixture()
        self.assertIsNotNone(controller._shg_processed_for_request(loaded, settings, False))
        loaded.shg_background = SimpleNamespace(source_file="new.csv", spectra=object(), revision=2)
        self.assertIsNone(controller._shg_processed_for_request(loaded, settings, False))

    def test_data_revision_change_does_not_reseed_old_result(self):
        controller, loaded, settings, _ = self.fixture()
        self.assertIsNotNone(controller._shg_processed_for_request(loaded, settings, False))
        loaded.shg_data.revision += 1
        self.assertIsNone(controller._shg_processed_for_request(loaded, settings, False))

    def test_pending_cached_pair_is_revalidated_after_revision_change(self):
        controller, loaded, _, workers = self.fixture()
        controller._start_shg_reprocess()
        controller._start_shg_reprocess()
        pending = controller._shg_reprocess_pending_payload
        self.assertIsNotNone(pending)
        loaded.shg_data.revision += 1
        # Simulate completion of the first worker without publishing obsolete data.
        controller._shg_reprocess_workers.clear()
        controller._start_shg_reprocess(pending_payload=pending)
        self.assertEqual(len(workers), 2)
        self.assertIsNone(workers[-1].payload[7])

    def assert_result_equal(self, left, right):
        if isinstance(left, np.ndarray):
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True)
        elif is_dataclass(left):
            for field in fields(left):
                self.assert_result_equal(getattr(left, field.name), getattr(right, field.name))
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_result_equal(a, b)
        elif isinstance(left, float):
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True)
        else:
            self.assertEqual(left, right)

    def test_real_single_and_compare_reuse_matches_full_processing(self):
        from tests.test_shg_fit import _angular_sweep
        data_a, result_a = _angular_sweep(8)
        data_b, result_b = _angular_sweep(14)
        original = data_a.spectra.copy()
        fit = ShgFitSettings(angle_min_deg=5, angle_max_deg=175,
                             use_uncertainty_weights=False, phase_branch=1)
        def run(payload):
            worker = _ShgWorker(payload)
            output, errors = [], []
            worker.signals.result.connect(output.append)
            worker.signals.error.connect(errors.append)
            worker.run()
            self.assertEqual(errors, [])
            self.assertEqual(len(output), 1)
            return output[0]
        for compare in (False, True):
            with self.subTest(compare=compare):
                base = (data_a, data_b if compare else None, result_a.settings, fit, None, None, compare)
                full = run((*base, None))
                reused = run((*base, (result_a, result_b if compare else None)))
                self.assert_result_equal(full, reused)
                np.testing.assert_array_equal(data_a.spectra, original)


class AstraPeakShiftReviewTests(unittest.TestCase):
    def owner(self, *, active=False, local=False):
        import threading
        from unittest.mock import Mock
        source = object()
        key = (id(source), "numerical settings")
        cache = {"Raw spectrum": {"pos": object()}, "Second derivative": {"pos": object()}}
        owner = SimpleNamespace(
            loaded=SimpleNamespace(mode="MCD", mcd_result=source),
            _mcd_peak_computation_key=lambda: key,
            _mcd_peak_analysis_key=key if local else ("old numerical settings",),
            _mcd_peak_analysis_cache={key: cache},
            _mcd_peak_analysis_cancel_event=threading.Event() if active else None,
            _mcd_peak_analysis_generation=3,
            _mcd_peak_analysis_worker=object() if active else None,
            mcd_peak_tracker_method_combo=SimpleNamespace(currentText=lambda: "Raw spectrum"),
            _mcd_peak_selected_result_method="Local mixed fit" if local else "Raw spectrum",
            mcd_peak_result=object(),
            mcd_peak_method_results={"Local mixed fit": {"pos": object()}} if local else {},
            mcd_peak_analyze_btn=Mock(),
            _apply_mcd_peak_shift_results=Mock(),
        )
        return owner, source, cache

    def test_return_from_local_tracker_applies_cached_legacy_result(self):
        from ui_qt.feature_pages import FeatureTabsMixin
        owner, source, cache = self.owner(local=True)
        FeatureTabsMixin._request_mcd_peak_shift_analysis(owner)
        owner._apply_mcd_peak_shift_results.assert_called_once_with(cache, source)

    def test_cached_result_reenables_analyze_after_canceling_active_analysis(self):
        from ui_qt.feature_pages import FeatureTabsMixin
        owner, source, cache = self.owner(active=True)
        cancel = owner._mcd_peak_analysis_cancel_event
        FeatureTabsMixin._request_mcd_peak_shift_analysis(owner)
        self.assertTrue(cancel.is_set())
        owner._apply_mcd_peak_shift_results.assert_called_once_with(cache, source)
        owner.mcd_peak_analyze_btn.setEnabled.assert_called_with(True)


    def test_three_requests_keep_only_latest_pending_while_worker_is_owned(self):
        import threading
        from unittest.mock import Mock
        from ui_qt.feature_pages import FeatureTabsMixin
        owner, source, _ = self.owner(active=True)
        owner._mcd_peak_analysis_cache = {}
        owner.mcd_peak_result = None
        owner._mcd_peak_analysis_workers = {owner._mcd_peak_analysis_worker}
        owner._mcd_peak_analysis_pending_request = None
        requested = ["second"]
        owner._mcd_peak_computation_key = lambda: (id(source), requested[0])
        owner.mcd_peak_source_combo = SimpleNamespace(currentText=lambda: "Raw R")
        for field in ("prom", "dist", "smooth", "jump", "max", "deriv_window"):
            setattr(owner, f"mcd_peak_{field}_spin", SimpleNamespace(value=lambda: 1))
        owner._launch_mcd_peak_shift_analysis = Mock()
        # The first worker remains owned/running even after its cancellation flag is set.
        FeatureTabsMixin._request_mcd_peak_shift_analysis(owner)
        requested[0] = "third"
        FeatureTabsMixin._request_mcd_peak_shift_analysis(owner)
        owner._launch_mcd_peak_shift_analysis.assert_not_called()
        self.assertEqual(owner._mcd_peak_analysis_pending_request[1], (id(source), "third"))

    def test_obsolete_legacy_finished_does_not_enable_active_local_fit(self):
        from ui_qt.feature_pages import FeatureTabsMixin
        owner, _, _ = self.owner(active=True)
        worker = owner._mcd_peak_analysis_worker
        owner._mcd_peak_analysis_workers = {worker}
        owner._mcd_peak_analysis_pending_request = None
        owner._mcd_peak_analysis_generation = 4
        owner.mcd_peak_tracker_method_combo.currentText = lambda: "Local mixed fit"
        FeatureTabsMixin._mcd_peak_shift_analysis_finished(owner, worker, 3)
        owner.mcd_peak_analyze_btn.setEnabled.assert_not_called()


class AstraPeakShiftEventTests(unittest.TestCase):
    def window_context(self):
        from contextlib import ExitStack
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import patch
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
        import ui_qt.main_window as mw
        from tests.test_mcd_peak_shift import _result
        stack = ExitStack()
        folder = stack.enter_context(TemporaryDirectory())
        stack.enter_context(patch.object(mw, "QSettings", lambda *a, **k:
            QSettings(str(Path(folder) / "settings.ini"), QSettings.IniFormat)))
        stack.enter_context(patch.object(mw.MainWindow, "_restore_last_folder"))
        stack.enter_context(patch.object(mw.MainWindow, "_schedule_automatic_update_check"))
        app = QApplication.instance() or QApplication([])
        window = mw.MainWindow()
        source = _result([-1., 0., 1.])
        window.loaded = mw.LoadedState(mode="MCD", folder="", mcd_result=source)
        window._update_mcd_peak_shift_source(source)
        stack.callback(window.close)
        return stack, app, window

    def wait_until(self, app, predicate):
        import time
        deadline = time.monotonic() + 5.
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.001)
        self.assertTrue(predicate(), "Qt worker condition did not complete before deadline")

    def test_real_worker_runs_only_first_and_latest_requests(self):
        import threading
        from unittest.mock import patch
        stack, app, window = self.window_context()
        started, release = threading.Event(), threading.Event()
        seen = []
        def controlled_worker(source, **options):
            seen.append(options["prominence"])
            if len(seen) == 1:
                started.set()
                release.wait(4.)
            return None
        try:
            with stack, patch("ui_qt.feature_pages._mcd_peak_shift_worker", controlled_worker):
                window.mcd_peak_prom_spin.setValue(.03)
                window._request_mcd_peak_shift_analysis()
                self.wait_until(app, started.is_set)
                window.mcd_peak_prom_spin.setValue(.04)
                window._request_mcd_peak_shift_analysis()
                window.mcd_peak_prom_spin.setValue(.05)
                window._request_mcd_peak_shift_analysis()
                self.assertEqual(seen, [.03])
                release.set()
                self.wait_until(app, lambda: window._mcd_peak_analysis_worker is None)
                self.assertEqual(seen, [.03, .05])
        finally:
            release.set()
            stack.close()

    def test_close_cancels_worker_before_waiting_for_completion(self):
        import threading
        from unittest.mock import patch
        stack, app, window = self.window_context()
        started = threading.Event()
        observed = []
        def controlled_worker(source, **options):
            started.set()
            observed.append(options["cancel_event"].wait(4.))
            return None
        with stack, patch("ui_qt.feature_pages._mcd_peak_shift_worker", controlled_worker):
            window._request_mcd_peak_shift_analysis()
            self.wait_until(app, started.is_set)
            window.close()
            self.assertEqual(observed, [True])


class AstraCompareReviewTests(unittest.TestCase):
    def fixture(self, vp=False):
        from unittest.mock import Mock
        from matplotlib.figure import Figure
        from ui_qt.controllers_compare import CompareController
        from ui_qt.main_window import MainWindow
        figure = Figure()
        heat, line = figure.subplots(2, 1)
        cube = SimpleNamespace(energy=np.array([1., 2., 3.]), gate=np.array([0., 1.]),
                               Z=np.array([[1., 2., 3.], [100., 200., 300.]]))
        if vp:
            cube.Z = np.array([[.1, .2, .3], [-.4, -.2, .5]])
        previous = list(range(19)); previous[15] = 0.
        current = list(previous); current[15] = 1.
        loaded = SimpleNamespace(mode="Compare", folder="synthetic", selected_files=["a.csv"],
                                 compare_cubes={"KK": cube})
        channel = "VP" if vp else "KK"
        owner = SimpleNamespace(last_plotted_mode="Compare", _load_in_progress=False,
            _plot_redraw_pending=set(), _cmp_active_cubes={channel: cube},
            _cmp_heatmap_axes={channel: heat}, _cmp_linecut_ax=line,
            _last_plot_params_key=tuple(previous), _current_plot_params_key=lambda mode: tuple(current),
            _shown_draw_identity=("a.csv",), _shown_source_identity_for_loaded=lambda state: ("a.csv",),
            loaded=loaded, canvas=SimpleNamespace(draw_idle=Mock()),
            cmp_spins={"gate": SimpleNamespace(value=lambda: 1.)},
            _safe_spectrum_xlim=lambda energy, limits: limits)
        owner._cmp_rendered_loaded_marker = (id(loaded), id(loaded.compare_cubes), loaded.folder,
                                              tuple(loaded.selected_files))
        controller = CompareController(owner)
        object.__setattr__(controller, "_cmp_is_vp_view", lambda: vp)
        object.__setattr__(controller, "_set_cmp_gate_spin_value", Mock())
        object.__setattr__(controller, "_ensure_cmp_gate_lines", Mock())
        if vp:
            line.plot(cube.energy, cube.Z[0], label="VP")
            line.axhline(0.)
            line.set_ylim(-1.05, 1.05)
            line.set_xlim(1., 3.)
        else:
            MainWindow._plot_compare_linecut(owner, line, {"KK": cube}, gate_value=0., xlim=(1., 3.))
        return owner, controller, line

    def test_intensity_gate_update_rescales_like_full_linecut(self):
        owner, controller, line = self.fixture()
        initial_line = line.lines[0]
        self.assertFalse(line.get_autoscaley_on())
        self.assertTrue(controller._update_cmp_gate_only())
        self.assertIs(line.lines[0], initial_line)
        np.testing.assert_allclose(line.lines[0].get_ydata(), [100., 200., 300.])
        np.testing.assert_allclose(line.get_ylim(), [84., 316.])
        self.assertEqual(line.get_xlim(), (1., 3.))

    def test_vp_gate_update_preserves_fixed_limits_and_zero_line(self):
        owner, controller, line = self.fixture(vp=True)
        spectrum, zero = tuple(line.lines)
        self.assertTrue(controller._update_cmp_gate_only())
        self.assertIs(line.lines[0], spectrum)
        self.assertIs(line.lines[1], zero)
        np.testing.assert_allclose(spectrum.get_ydata(), [-.4, -.2, .5])
        np.testing.assert_allclose(zero.get_ydata(), [0., 0.])
        self.assertEqual(line.get_ylim(), (-1.05, 1.05))
        self.assertEqual(line.get_xlim(), (1., 3.))

    def test_same_filename_replacement_source_rejects_old_render(self):
        owner, controller, line = self.fixture()
        owner.loaded = SimpleNamespace(mode="Compare", folder="synthetic", selected_files=["a.csv"],
                                       compare_cubes=dict(owner.loaded.compare_cubes))
        self.assertFalse(controller._update_cmp_gate_only())
        np.testing.assert_allclose(line.lines[0].get_ydata(), [1., 2., 3.])

    def test_pending_non_gate_redraw_rejects_fast_path(self):
        owner, controller, line = self.fixture()
        owner._plot_redraw_pending.add("Compare")
        self.assertFalse(controller._update_cmp_gate_only())
        np.testing.assert_allclose(line.lines[0].get_ydata(), [1., 2., 3.])


    def pixel_fixture(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        owner, controller, linecut = self.fixture()
        owner.figure = linecut.figure
        owner.canvas = FigureCanvasAgg(owner.figure)
        owner._cmp_blit_supported = True
        owner.canvas.mpl_connect("draw_event", controller._on_compare_canvas_draw)
        heat = owner._cmp_heatmap_axes["KK"]
        heat.imshow(np.array([[0., 1., 0.], [1., 0., 1.]]), extent=(1., 3., 0., 2.),
                    aspect="auto", origin="lower", cmap="viridis")
        marker = heat.axhline(.25, color="red", linewidth=3.)
        owner._cmp_gate_lines = {"KK": marker}
        object.__setattr__(controller, "_ensure_cmp_gate_lines", lambda cubes, gate: marker.set_ydata([gate, gate]))
        owner.canvas.draw()
        return owner, controller, heat

    def test_warm_blit_pixels_match_clean_full_render_without_old_marker_trail(self):
        owner, controller, heat = self.pixel_fixture()
        self.assertTrue(controller._configure_cmp_blitting())
        self.assertTrue(controller._update_cmp_gate_only())
        blitted = np.asarray(owner.canvas.buffer_rgba()).copy()
        controller._disable_cmp_blitting()
        owner.canvas.draw()
        reference = np.asarray(owner.canvas.buffer_rgba()).copy()
        # Exclude 3px spines: dynamic overlays draw after static spine antialiasing.
        # Interior remains exact, detecting old marker trails and blank heatmaps.
        x0, y0, x1, y1 = map(int, heat.bbox.extents)
        height = reference.shape[0]
        np.testing.assert_array_equal(blitted[height-y1+3:height-y0-3, x0+3:x1-3],
                                      reference[height-y1+3:height-y0-3, x0+3:x1-3])

    def test_ordinary_draw_keeps_heatmap_visible_after_enabling_blit(self):
        owner, controller, heat = self.pixel_fixture()
        self.assertTrue(controller._configure_cmp_blitting())
        owner.canvas.draw()
        redrawn = np.asarray(owner.canvas.buffer_rgba()).copy()
        controller._disable_cmp_blitting()
        owner.canvas.draw()
        reference = np.asarray(owner.canvas.buffer_rgba()).copy()
        x0, y0, x1, y1 = map(int, heat.bbox.extents)
        height = reference.shape[0]
        np.testing.assert_array_equal(redrawn[height-y1+3:height-y0-3, x0+3:x1-3],
                                      reference[height-y1+3:height-y0-3, x0+3:x1-3])


    def test_toolbar_png_contains_dynamic_artists_and_restores_visible_frame(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import patch
        from PIL import Image
        from PySide6.QtWidgets import QApplication, QMainWindow
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from ui_qt.main_window import _PlotToolbar
        from ui_qt.controllers_compare import CompareController
        app = QApplication.instance() or QApplication([])
        fake, _, linecut = self.fixture(vp=True)
        window = QMainWindow()
        for name, value in vars(fake).items():
            setattr(window, name, value)
        window.figure = linecut.figure
        window.canvas = FigureCanvasQTAgg(window.figure)
        window.setCentralWidget(window.canvas)
        heat = window._cmp_heatmap_axes["VP"]
        heat.imshow(np.array([[0., 1.], [1., 0.]]), extent=(1., 3., 0., 2.), aspect="auto")
        window._cmp_gate_lines = {"VP": heat.axhline(.4, color="red", linewidth=3.)}
        window._cmp_blit_supported = True
        controller = CompareController(window)
        window.compare_controller = controller
        window.canvas.mpl_connect("draw_event", controller._on_compare_canvas_draw)
        toolbar = _PlotToolbar(window.canvas, window)
        try:
            with TemporaryDirectory() as tmp:
                expected = Path(tmp) / "expected.png"
                saved = Path(tmp) / "saved.png"
                window.canvas.draw()
                window.figure.savefig(expected)
                self.assertTrue(controller._configure_cmp_blitting())
                with patch.object(NavigationToolbar2QT, "save_figure", side_effect=lambda *a: window.figure.savefig(saved)):
                    toolbar.save_figure()
                np.testing.assert_array_equal(np.asarray(Image.open(saved)), np.asarray(Image.open(expected)))
                # Restore must render the dynamic linecut immediately, without another gate event.
                restored = np.asarray(window.canvas.buffer_rgba()).copy()
                controller._disable_cmp_blitting()
                window.canvas.draw()
                reference = np.asarray(window.canvas.buffer_rgba()).copy()
                x0, y0, x1, y1 = map(int, linecut.bbox.extents)
                height = reference.shape[0]
                np.testing.assert_array_equal(restored[height-y1+3:height-y0-3, x0+3:x1-3],
                                              reference[height-y1+3:height-y0-3, x0+3:x1-3])
                self.assertTrue(controller._configure_cmp_blitting())
                with patch.object(NavigationToolbar2QT, "save_figure", side_effect=OSError("synthetic save error")):
                    with self.assertRaises(OSError):
                        toolbar.save_figure()
                self.assertTrue(window._cmp_blit_enabled)
        finally:
            window.close()
            window.deleteLater()
            app.processEvents()


class AstraPowerReviewTests(unittest.TestCase):
    def fixture(self):
        from unittest.mock import Mock
        from ui_qt.controllers_power import PowerController
        selection = {"legacy": False, "status": "All", "group": "new-group",
                     "single": "csv::new.csv", "action": "Single intensity",
                     "KK": "", "KKp": "", "pairing": "Pair by Stage"}
        validated = object()
        dialog = SimpleNamespace(selection=lambda: dict(selection),
                                 _validation_results={selection["single"]: validated})
        owner = SimpleNamespace(current_folder="synthetic", _power_catalog_folder="synthetic",
            _power_catalog_include_legacy=False, _power_include_legacy=False,
            _power_sources_cache={"csv::old.csv": object()}, _power_result_cache={},
            _refresh_file_lists=Mock(), _start_load=Mock(), _invalidate_export_move_sources=Mock())
        controller = PowerController(owner)
        object.__setattr__(controller, "_power_refresh_groups", Mock())
        object.__setattr__(controller, "_power_set_view_mode", Mock())
        return owner, controller, dialog, selection, validated

    def test_newly_selected_group_missing_from_valid_catalog_waits_for_refresh(self):
        owner, controller, dialog, selection, _ = self.fixture()
        controller._apply_power_measurement_group(dialog)
        owner._start_load.assert_not_called()
        owner._refresh_file_lists.assert_called_once()
        self.assertIsNotNone(getattr(owner, "_power_pending_measurement_selection", None))

    def test_deferred_group_keeps_prevalidated_data_through_catalog_acceptance(self):
        owner, controller, dialog, selection, validated = self.fixture()
        owner._power_sources_cache = None
        controller._apply_power_measurement_group(dialog)
        owner._power_sources_cache = {selection["single"]: object()}
        controller._apply_power_measurement_group_from_selection(selection)
        self.assertIs(owner._power_result_cache.get(selection["single"]), validated)

    def test_new_cached_selection_supersedes_older_pending_selection(self):
        owner, controller, dialog, selection, _ = self.fixture()
        controller._apply_power_measurement_group(dialog)
        newer = dict(selection, single="csv::old.csv", group="old-group")
        owner._power_pending_open_name = "earlier-combined.csv"
        controller._apply_power_measurement_group(SimpleNamespace(
            selection=lambda: newer, _validation_results={}))
        owner._start_load.assert_called_once()
        self.assertIsNone(getattr(owner, "_power_pending_measurement_selection", None))
        self.assertFalse(getattr(owner, "_power_pending_open_name", None))

    def test_new_uncatalogued_selection_supersedes_older_combined_open(self):
        owner, controller, dialog, _, _ = self.fixture()
        owner._power_pending_open_name = "earlier-combined.csv"
        owner._power_pending_open_generation = 2
        controller._apply_power_measurement_group(dialog)
        self.assertFalse(getattr(owner, "_power_pending_open_name", None))
        self.assertIsNotNone(owner._power_pending_measurement_selection)

    def test_new_combined_open_supersedes_older_pending_selection(self):
        from pathlib import Path
        from unittest.mock import Mock
        owner, controller, dialog, _, _ = self.fixture()
        controller._apply_power_measurement_group(dialog)
        owner.available_files = []
        owner.power_compare_chk = SimpleNamespace(setChecked=Mock())
        controller._power_finish_combination(SimpleNamespace(
            saved_path=Path("synthetic") / "new-combined.csv", open_requested=True))
        self.assertIsNone(getattr(owner, "_power_pending_measurement_selection", None))
        self.assertEqual(owner._power_pending_open_name, "new-combined.csv")


if __name__ == "__main__":
    unittest.main()
