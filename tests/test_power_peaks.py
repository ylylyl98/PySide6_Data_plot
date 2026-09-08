import csv
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.loader import DataCube
from core.power_peaks import PeakSettings, fit_power_peaks, peak_model, export_peak_analysis
from core.processing import PowerSweepPoint
from tests import test_power_combine as workflow_helpers


def synthetic(model="Lorentzian", powers=(0., 1., 10., 100.)):
    energy = np.linspace(1.55, 1.65, 201)
    widths = np.linspace(.004, .010, len(powers))
    z = np.array([peak_model(energy, 4, 10, 20 * (i + 1), 1.6, width,
                             model=model, origin=1.6) for i, width in enumerate(widths)])
    return DataCube(energy, np.array(powers), z, "Power (uW)", "Synthetic", "PL"), widths


class PeakFitTests(unittest.TestCase):
    def test_known_widths_heights_and_measured_areas(self):
        for model in ("Lorentzian", "Gaussian"):
            cube, widths = synthetic(model)
            fits = fit_power_peaks(cube, PeakSettings(1.6, .04, model))
            self.assertTrue(all(f.status == "ok" for f in fits), [f.status for f in fits])
            np.testing.assert_allclose([f.fwhm_mev for f in fits], widths * 1000, rtol=1e-5)
            np.testing.assert_allclose([f.height for f in fits], [20, 40, 60, 80], rtol=1e-5)
            mask = (cube.energy >= 1.56) & (cube.energy <= 1.64)
            expected = np.trapezoid(cube.Z[:, mask] - 4 - 10 * (cube.energy[mask] - 1.6), cube.energy[mask], axis=1)
            np.testing.assert_allclose([f.area for f in fits], expected, rtol=1e-5)
            self.assertTrue(all(np.isfinite(f.fwhm_error_mev) for f in fits))

    def test_failed_spectra_have_status_and_no_trend_values(self):
        cube, _ = synthetic()
        cube.Z[0] = 3
        cube.Z[1] = np.nan
        fits = fit_power_peaks(cube, PeakSettings(1.6, .04))
        self.assertEqual(fits[0].status, "flat spectrum")
        self.assertIn("insufficient", fits[1].status)
        self.assertTrue(np.isnan(fits[0].fwhm_mev))
        self.assertTrue(np.isnan(fits[1].area))
        self.assertEqual(fits[2].status, "ok")

    def test_noise_uncertainty_and_outside_window(self):
        cube, widths = synthetic()
        cube.Z += np.random.default_rng(8).normal(0, .2, cube.Z.shape)
        fits = fit_power_peaks(cube, PeakSettings(1.6, .04))
        np.testing.assert_allclose([f.fwhm_mev for f in fits], widths * 1000, rtol=.05)
        self.assertTrue(all(f.fwhm_error_mev > 0 for f in fits))
        outside = fit_power_peaks(cube, PeakSettings(2., .02))
        self.assertTrue(all(f.status != "ok" for f in outside))

    def test_export_rows_settings_and_provenance(self):
        cube, _ = synthetic()
        cube.Z[0] = 0
        records = tuple(PowerSweepPoint("combined.csv", p, None, i+2,
                         source_provenance='{"sources":["low.csv","high.csv"]}') for i, p in enumerate(cube.gate))
        settings = PeakSettings(1.6, .04)
        payload = {"results": {"Power": fit_power_peaks(cube, settings)}, "settings": settings,
                   "records": {"Power": records}, "metric": "Integrated area", "background": 0.,
                   "log_power": True, "power_limits": (1., 100.)}
        with tempfile.TemporaryDirectory() as folder:
            path = export_peak_analysis(folder, payload)
            with path.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["status"], "flat spectrum")
            self.assertEqual(rows[1]["source_provenance"], records[1].source_provenance)
            self.assertEqual(json.loads(path.with_suffix(".json").read_text())["power_axis"], "log")
            self.assertTrue(path.with_suffix(".png").is_file())


class PeakWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        workflow_helpers.PowerWorkflowTests.setUp(self)

    def tearDown(self):
        workflow_helpers.PowerWorkflowTests.tearDown(self)

    def prepare_sweep(self):
        from ui_qt.main_window import LoadedState
        w = self.window
        cube, _ = synthetic()
        records = tuple(PowerSweepPoint("combined.csv", p, None, i+2) for i, p in enumerate(cube.gate))
        w.power_group_combo.blockSignals(True)
        w.power_group_combo.addItem("Combined", "csv::combined.csv")
        w.power_group_combo.setCurrentIndex(w.power_group_combo.findData("csv::combined.csv"))
        w.loaded = LoadedState(mode="Power Dependent", folder="", primary_file="combined.csv",
            selected_files=["combined.csv"], cube=cube, power_records=records,
            power_groups={}, power_group_key="csv::combined.csv", y_axis_spec="auto")
        w.power_background_auto_chk.setChecked(False)
        w.power_background_spin.setValue(0)
        w._apply_auto_limits_for_loaded()
        c = w.power_peak_controller
        from core.power_multi_peaks import PeakSeed
        c.set_peaks((PeakSeed(1.6, .008),))
        c.enabled.setChecked(True)
        return cube, records

    def wait_fits(self):
        c = self.window.power_peak_controller
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.app.processEvents()
            if c.current_results:
                return
            time.sleep(.01)
        self.fail("Peak fitting did not finish")

    def test_linked_scales_cache_metric_and_click(self):
        w = self.window
        cube, records = self.prepare_sweep()
        with patch.object(w, "_show_error") as error:
            w._plot_mode("Power Dependent")
            self.wait_fits()
            c = w.power_peak_controller
            key = c.key({"Power": cube})
            self.assertIn(key, c.cache)
            with patch("ui_qt.power_peak_controller.fit_power_multi_peaks", side_effect=AssertionError("Display change refitted")):
                self.assertEqual(c.axes[0].get_yscale(), "log")
                self.assertEqual(c.axes[1].get_yscale(), "log")
                c.intensity_scale.setCurrentText("Linear")
                w._plot_mode("Power Dependent")
                self.assertEqual(c.axes[0].get_yscale(), "linear")
                self.assertEqual(c.axes[1].get_yscale(), "log")
                c.linewidth_scale.setCurrentText("Linear")
                w._plot_mode("Power Dependent")
                self.assertEqual(c.axes[1].get_yscale(), "linear")
                w.power_axis_scale_combo.setCurrentText("Log")
                w._plot_mode("Power Dependent")
                self.assertEqual(w._power_heatmap_ax.get_yscale(), "log")
                self.assertTrue(all(ax.get_xscale() == "log" for ax in c.axes))
                self.assertIn("nonpositive", c.status.text())
                c.metric.setCurrentText("Peak height")
                w._plot_mode("Power Dependent")
                np.testing.assert_allclose(c.axes[0].lines[0].get_ydata(), [40, 60, 80], rtol=1e-5)
                c.click(SimpleNamespace(inaxes=c.axes[1], xdata=10.))
                self.assertAlmostEqual(w.power_spins["gate"].value(), 10.)
                self.assertGreater(len(c.residual_axis.lines), 1)
                w.power_axis_scale_combo.setCurrentText("Linear")
                w._plot_mode("Power Dependent")
                self.assertTrue(all(ax.get_xscale() == "linear" for ax in c.axes))
                self.assertEqual(len(c.axes[0].lines[0].get_xdata()), 4)
                self.assertEqual(c.axes[0].get_xlim()[0], 0.)
                self.assertIsNone(c.worker)
            error.assert_not_called()

    def test_manual_refit_preview_replaces_only_selected_power(self):
        from ui_qt.power_refit_dialog import PowerRefitDialog
        w = self.window
        cube, _ = self.prepare_sweep()
        w._plot_mode("Power Dependent")
        self.wait_fits()
        controller = w.power_peak_controller
        original = controller.current_results["Power"]
        dialog = PowerRefitDialog(w, {"Power": cube}, controller.current_results,
                                  controller.settings(), 10.)
        self.assertEqual(dialog.index, 2)
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog._run()
        deadline = time.monotonic() + 10
        while dialog.worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertIsNone(dialog.worker)
        self.assertTrue(dialog.apply_button.isEnabled(), dialog.status.text())
        dialog._apply()
        label, index, fit = dialog.accepted_fit
        self.assertTrue(fit.manual)
        controller.apply_manual_fit(controller.key({"Power": cube}), label, index, fit)
        updated = controller.current_results["Power"]
        self.assertIs(updated[2], fit)
        for i in (0, 1, 3):
            self.assertIs(updated[i], original[i])
        dialog.deleteLater()

    def test_log_toggle_draws_without_math_fonts_through_ui_events(self):
        # Check the rendered pixels, not just get_yscale(): Qt's deferred draw
        # used to fail in MathText while leaving the old linear image visible.
        from matplotlib.mathtext import MathTextParser
        w = self.window
        self.prepare_sweep()
        w._plot_mode("Power Dependent")
        self.wait_fits()
        w.canvas.draw()
        linear = np.asarray(w.canvas.buffer_rgba()).copy()
        with patch.object(MathTextParser, "parse", side_effect=AssertionError("Math fonts unavailable")):
            w.power_axis_scale_combo.setCurrentText("Log")
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                self.app.processEvents()
                if w._power_heatmap_ax.get_yscale() == "log":
                    break
                time.sleep(.01)
            self.assertEqual(w._power_heatmap_ax.get_yscale(), "log")
            w.canvas.draw()
            self.assertFalse(np.array_equal(linear, np.asarray(w.canvas.buffer_rgba())))
            self.assertTrue(all(ax.get_xscale() == "log" for ax in w.power_peak_controller.axes))
            w.power_log_chk.setChecked(True)
            w._run_scheduled_plot_redraw("Power Dependent")
            w.canvas.draw()

    def test_combine_saves_then_explicitly_loads(self):
        from ui_qt.power_combine_dialog import PowerCombineDialog
        w = self.window
        with tempfile.TemporaryDirectory() as folder:
            w.current_folder = folder
            path = Path(folder) / "combined.csv"
            path.write_text("Power_uW,1.5,1.6\n1,2,3\n2,4,5\n")
            def accept(dialog):
                dialog.saved_path = path
                dialog.open_requested = True
                return 1
            with patch.object(PowerCombineDialog, "exec", accept), patch.object(w, "_start_load") as load:
                w.power_controller._on_power_combine()
            load.assert_called_once_with("Power Dependent")
            self.assertEqual(w.power_group_combo.currentData(), "csv::combined.csv")

    def test_save_without_open_keeps_current_selection(self):
        from ui_qt.power_combine_dialog import PowerCombineDialog
        w = self.window
        self.prepare_sweep()
        previous = w.power_group_combo.currentData()
        with tempfile.TemporaryDirectory() as folder:
            w.current_folder = folder
            path = Path(folder) / "saved.csv"
            path.write_text("Power_uW,1.5,1.6\n1,2,3\n2,4,5\n")
            def close(dialog):
                dialog.saved_path = path
                dialog.open_requested = False
                return 0
            with patch.object(PowerCombineDialog, "exec", close), patch.object(w, "_start_load") as load:
                w.power_controller._on_power_combine()
            load.assert_not_called()
            # The selected source must remain discoverable, as it is in the real workflow.
            self.assertNotEqual(w.power_group_combo.currentData(), "csv::saved.csv")

    def test_plot_setup_updates_loaded_combined_sweep(self):
        w = self.window
        self.prepare_sweep()
        w.power_peak_controller.enabled.setChecked(False)
        with patch.object(w, "_show_error") as error:
            w._plot_mode("Power Dependent")
            w.power_background_spin.setValue(2.)
            w.power_cmap.setCurrentText("viridis")
            w.power_fix_checks["xmin"].setChecked(True)
            w.power_fix_checks["xmax"].setChecked(True)
            w.power_spins["xmin"].setValue(1.57)
            w.power_spins["xmax"].setValue(1.63)
            w._run_scheduled_plot_redraw("Power Dependent")
            np.testing.assert_allclose(w._power_active_cubes["Power"].Z, w.loaded.cube.Z - 2)
            np.testing.assert_allclose(w._power_heatmap_ax.get_xlim(), [1.57, 1.63])
            self.assertEqual(w._power_heatmap_ax.collections[0].cmap.name, "viridis")
            error.assert_not_called()

    def test_fresh_combined_save_loads_and_plots_without_prior_loaded_state(self):
        from ui_qt.power_combine_dialog import PowerCombineDialog
        w = self.window
        self.assertIsNone(w.loaded)
        with tempfile.TemporaryDirectory() as folder:
            w.current_folder = folder
            from core.power_workflow import COMBINED_FOLDER
            path = Path(folder) / COMBINED_FOLDER / "combined.csv"
            path.parent.mkdir(parents=True)
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['Power_uW', '1.5', '1.6', 'source_provenance'])
                for power in (1, 2):
                    writer.writerow([power, 2 * power, 3 * power, json.dumps({'sources': [{'file': 'raw.csv', 'row': power + 1}]})])
            def accept(dialog):
                dialog.saved_path = path
                dialog.open_requested = True
                return 1
            with patch.object(PowerCombineDialog, "exec", accept), patch.object(w, "_show_error") as error:
                w.power_controller._on_power_combine()
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and (w._load_in_progress or w.loaded is None):
                    self.app.processEvents()
                    time.sleep(.01)
                self.assertIsNotNone(w.loaded)
                self.assertEqual(w.last_plotted_mode, "Power Dependent")
                from core.data_io import power_sweep_source_key
                expected = power_sweep_source_key(str(path.relative_to(folder)))
                self.assertEqual(w.loaded.power_group_key, expected)
                self.assertEqual(w.power_group_combo.currentData(), expected)
                error.assert_not_called()

    def test_save_includes_peak_analysis_with_heatmap(self):
        w = self.window
        self.prepare_sweep()
        with tempfile.TemporaryDirectory() as folder, patch.object(w, "_show_error") as error:
            w.current_folder = folder
            w.loaded.folder = folder
            (Path(folder) / "combined.csv").write_text("source fixture\n")
            w._plot_mode("Power Dependent")
            self.wait_fits()
            w.power_axis_scale_combo.setCurrentText("Log")
            w._plot_mode("Power Dependent")
            w._start_export("Power Dependent")
            deadline = time.monotonic() + 10
            while w._export_in_progress and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.01)
            self.assertFalse(w._export_in_progress)
            error.assert_not_called()
            files = list(Path(folder).rglob("Power_peak_analysis.csv"))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].with_suffix(".png").is_file())
            self.assertEqual(len(list(files[0].parent.glob("*.dat"))), 1)
            self.assertEqual(json.loads(files[0].with_suffix(".json").read_text())["power_axis"], "log")
            error.assert_not_called()

    def test_comparison_fits_each_channel_independently(self):
        from core.data_io import PowerSeriesResult, PowerSeriesSource
        w = self.window
        a, records = self.prepare_sweep()
        b, widths = synthetic(powers=(0., 2., 20., 200.))
        b.Z *= 2
        records_b = tuple(PowerSweepPoint("other.csv", p, None, i+2) for i, p in enumerate(b.gate))
        first, second = "csv::combined.csv", "csv::other.csv"
        sources = {first: PowerSeriesResult(a, first, records, {}),
                   second: PowerSeriesResult(b, second, records_b, {})}
        for combo, key in [(w.power_kk_group_combo, first), (w.power_kkp_group_combo, second)]:
            combo.blockSignals(True)
            combo.addItem(key, key)
            combo.setCurrentIndex(combo.findData(key))
        c = w.power_controller
        with patch.object(type(c), "_power_current_sources", return_value={
            key: PowerSeriesSource(key, key, "table") for key in sources
        }), patch.object(type(c), "_power_load_group_result", side_effect=sources.__getitem__), \
                patch.object(w, "_show_error") as error:
            w.power_compare_chk.setChecked(True)
            w._plot_mode("Power Dependent")
            self.wait_fits()
            fits = w.power_peak_controller.current_results
            self.assertEqual(set(fits), {"KK", "KKp"})
            np.testing.assert_allclose([f.components[0].fwhm_mev for f in fits["KKp"]], widths * 1000, rtol=1e-5)
            np.testing.assert_allclose([f.components[0].height for f in fits["KKp"]], [40, 80, 120, 160], rtol=1e-5)
            w.power_peak_controller.click(SimpleNamespace(inaxes=w.power_peak_controller.axes[0], xdata=20.))
            self.assertEqual(w.power_spins["gate"].value(), 20.)
            error.assert_not_called()
