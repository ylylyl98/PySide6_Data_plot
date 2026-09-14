"""DRR redraw, analysis ownership and bounded fit queue regressions."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PySide6.QtWidgets import QApplication

from core.loader import DataCube
from ui_qt.controllers_drr import DrrController, _DrrFitWorker


class DrrInteractionOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasAgg(self.figure)
        heat, spectrum = self.figure.subplots(2)
        self.cube = DataCube(np.linspace(1, 2, 21), np.array([0., 1.]),
                             np.ones((2, 21)), "Gate", "test", "DR/R")
        self.owner = SimpleNamespace(
            canvas=self.canvas, figure=self.figure,
            _drr_spectrum_ax=spectrum, _drr_heatmap_ax=heat,
            _drr_spectrum_axes={"raw": spectrum}, _drr_heatmap_axes={"raw": heat},
            _drr_heatmap_peak_artist=None, _drr_heatmap_fit_artist=None,
            _drr_peak_gate=0., _drr_peak_indices=np.array([4]),
            _drr_fit_gate=0., _drr_fit_x=self.cube.energy,
            _drr_fit_y=self.cube.Z[0], _drr_fit_centers=np.array([1.2]),
            drr_peak_show_chk=SimpleNamespace(isChecked=lambda: True),
            drr_fit_show_chk=SimpleNamespace(isChecked=lambda: True),
            drr_fit_status=Mock(), drr_selected_files=["test.csv"],
            loaded=SimpleNamespace(cube=self.cube),
            _last_plot_cube=self.cube, last_plotted_mode="DRR",
            _drr_fit_generation=1, thread_pool=SimpleNamespace(start=Mock()),
        )
        self.controller = DrrController(self.owner)

    def test_overlay_artists_are_reused_and_hidden(self):
        c, o = self.controller, self.owner
        for _ in range(4):
            c._draw_drr_analysis_overlays(self.cube, 0., self.cube.energy, self.cube.Z[0])
        self.assertEqual(len(o._drr_spectrum_ax.lines), 1)
        self.assertEqual(len(o._drr_spectrum_ax.collections), 1)
        self.assertEqual(len(o._drr_heatmap_ax.collections), 2)
        before = tuple(o._drr_spectrum_ax.lines) + tuple(o._drr_spectrum_ax.collections)
        o.drr_peak_show_chk.isChecked = lambda: False
        o.drr_fit_show_chk.isChecked = lambda: False
        c._draw_drr_analysis_overlays(self.cube, 0., self.cube.energy, self.cube.Z[0])
        self.assertEqual(before, tuple(o._drr_spectrum_ax.lines) + tuple(o._drr_spectrum_ax.collections))
        self.assertFalse(any(a.get_visible() for a in before))

    def test_regions_share_one_full_draw_and_discard_removed_axes(self):
        o, c = self.owner, self.controller
        line, = o._drr_spectrum_ax.plot([1, 2], [1, 2])
        gate = o._drr_heatmap_ax.axhline(0.)
        o._drr_spectrum_lines = {"raw": line}
        o._drr_gate_lines = {"raw": gate}
        with patch.object(self.canvas, "draw", wraps=self.canvas.draw) as draw:
            c._draw_drr_regions()
            self.assertEqual(draw.call_count, 1)
            draw.reset_mock()
            o._drr_spectrum_ax.set_ylim(-2, 4)
            c._draw_drr_regions()
            self.assertEqual(draw.call_count, 1)
            draw.reset_mock()
            c._draw_drr_regions()
            self.assertEqual(draw.call_count, 0)
        o._drr_gate_lines = {}
        c._draw_drr_regions()
        self.assertEqual(len(o._drr_region_blitters), 1)

    def test_fit_result_publishes_current_three_value_spectrum(self):
        c = self.controller
        self.owner._drr_gate_input_value = lambda: 0.
        object.__setattr__(c, "_update_drr_spectrum_and_gate_line", Mock())
        c._on_drr_fit_finished(1, self.cube, ("test.csv",), 0., 0., 1,
                               self.cube.energy, np.array([0, 0, 1, 1.5, .1]))
        c._update_drr_spectrum_and_gate_line.assert_called_once_with(self.cube)

    def test_dual_view_initializes_four_regions_with_one_draw(self):
        self.figure.clear()
        axes = self.figure.subplots(2, 2)
        o = self.owner
        o._drr_spectrum_axes = {"raw": axes[1, 0], "second": axes[1, 1]}
        o._drr_spectrum_lines = {
            key: axis.plot([1, 2], [1, 2])[0] for key, axis in o._drr_spectrum_axes.items()}
        o._drr_gate_lines = {"raw": axes[0, 0].axhline(0), "second": axes[0, 1].axhline(0)}
        with patch.object(self.canvas, "draw", wraps=self.canvas.draw) as draw:
            self.controller._draw_drr_regions()
            self.assertEqual(draw.call_count, 1)
        self.assertEqual(len(o._drr_region_blitters), 4)

    def test_fit_worker_still_solves_uncancelled_data(self):
        from ui_qt.controllers_drr import _drr_multi_lorentz_model
        x = np.linspace(1, 2, 101)
        parameters = np.array([.1, .02, 1., 1.5, .08])
        y = _drr_multi_lorentz_model(x, *parameters)
        job = _DrrFitWorker(x, y, [.1, .01, .9, 1.48, .1],
                            [-1, -1, 0, 1, .001], [1, 1, 3, 2, .5])
        results, errors = [], []
        job.signals.result.connect(results.append)
        job.signals.error.connect(errors.append)
        job.run()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 1)
        np.testing.assert_allclose(_drr_multi_lorentz_model(x, *results[0]), y, atol=1e-6)

    def test_fit_queue_retains_only_latest_pending_request(self):
        c, o = self.controller, self.owner
        jobs = [SimpleNamespace(cancel=Mock()) for _ in range(3)]
        for job in jobs:
            c._queue_drr_fit_worker(job)
        self.assertEqual(o.thread_pool.start.call_count, 1)
        jobs[1].cancel.assert_called_once()
        c._finish_drr_fit_worker(jobs[0])
        self.assertEqual(o.thread_pool.start.call_args.args, (jobs[2],))
        c._invalidate_pending_drr_fit()
        jobs[2].cancel.assert_called()

    def test_cancelled_worker_skips_solver_and_emits_finished(self):
        job = _DrrFitWorker(np.arange(10.), np.arange(10.), [0, 0, 1, 5, 1],
                            [-10] * 5, [10] * 5)
        finished = Mock()
        job.signals.finished.connect(finished)
        job.cancel()
        with patch("ui_qt.controllers_drr.curve_fit") as fit:
            job.run()
        fit.assert_not_called()
        finished.assert_called_once()

    def test_running_worker_checks_cancellation_inside_solver(self):
        job = _DrrFitWorker(np.arange(10.), np.arange(10.), [0, 0, 1, 5, 1],
                            [-10] * 5, [10] * 5)
        result, error = Mock(), Mock()
        job.signals.result.connect(result)
        job.signals.error.connect(error)

        def solver(model, x, y, **kwargs):
            model(x, *kwargs["p0"])
            job.cancel()
            model(x, *kwargs["p0"])
            self.fail("cancelled solver should stop at its next evaluation")

        with patch("ui_qt.controllers_drr.curve_fit", side_effect=solver):
            job.run()
        result.assert_not_called()
        error.assert_not_called()

    def test_overlay_blit_matches_full_render_after_data_and_visibility_updates(self):
        o, c = self.owner, self.controller
        line, = o._drr_spectrum_ax.plot(self.cube.energy, self.cube.Z[0])
        o._drr_spectrum_lines = {"raw": line}
        o._drr_gate_lines = {"raw": o._drr_heatmap_ax.axhline(0.)}
        c._draw_drr_analysis_overlays(self.cube, 0., self.cube.energy, self.cube.Z[0])
        c._draw_drr_regions()
        o._drr_fit_y = self.cube.Z[0] * .8
        o.drr_peak_show_chk.isChecked = lambda: False
        c._draw_drr_analysis_overlays(self.cube, 0., self.cube.energy, self.cube.Z[0])
        c._draw_drr_regions()
        actual = np.asarray(self.canvas.buffer_rgba()).copy()
        for helper in o._drr_region_blitters.values():
            helper.prepare_full_redraw()
        self.canvas.draw()
        expected = np.asarray(self.canvas.buffer_rgba()).copy()
        # Axis boundary antialiasing can differ between full draw and blit.
        self.assertLess(float(np.mean(np.abs(actual.astype(float) - expected))), .5)

    def test_display_preview_reuses_same_data_and_invalidates_replacement(self):
        c = self.controller
        large = DataCube(np.linspace(1, 2, 1000), np.arange(1000.),
                         np.ones((1000, 1000)), "Gate", "large", "DR/R")
        self.owner.loaded.cube = large
        a = c._drr_display_preview(large)
        self.assertIs(a, c._drr_display_preview(large))
        self.assertLessEqual(a.Z.size, 250_000)
        self.assertEqual(large.Z.shape, (1000, 1000))
        large.Z = np.full((1000, 1000), 2.)
        b = c._drr_display_preview(large)
        self.assertIsNot(a, b)
        np.testing.assert_array_equal(b.Z, 2.)


if __name__ == "__main__":
    unittest.main()
