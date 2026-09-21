import numpy as np
from tests.test_drr_dual_view_regressions import DrrDualViewRegressionTests


class DrrPlotReuseTests(DrrDualViewRegressionTests):
    def test_three_region_reuse_updates_middle_and_colorbar_endpoints(self):
        w = self.window
        w.drr_region_count_combo.setCurrentIndex(2)
        self._wait_derivative()
        w._plot_mode('DRR')
        render = w._drr_heatmap_renders['raw']
        meshes = render.images
        self.assertEqual(len(meshes), 3)
        self._set_silently(w.drr_split_spins['middle_vmax'], 123)
        w._plot_mode('DRR')
        self.assertEqual(w._drr_heatmap_renders['raw'].images, meshes)
        w.canvas.draw()
        self.assertEqual(meshes[1].norm.vmax, 123)
        self.assertEqual(meshes[1].colorbar.get_ticks()[-1], 123)
        updated = np.asarray(w.canvas.buffer_rgba()).copy()
        w._drr_reuse_state = None
        w._plot_mode('DRR')
        w.canvas.draw()
        np.testing.assert_array_equal(updated, np.asarray(w.canvas.buffer_rgba()))
        self._set_silently(w.drr_split_spins['middle_vmin'], -1)
        blocked = w.drr_center_zero_chk.blockSignals(True)
        w.drr_center_zero_chk.setChecked(True)
        w.drr_center_zero_chk.blockSignals(blocked)
        w._plot_mode('DRR')
        w.canvas.draw()
        middle = w._drr_heatmap_renders['raw'].images[1]
        self.assertEqual(len(middle.colorbar.get_ticks()), 2)
        self.errors.assert_not_called()

    def test_clear_releases_reuse_snapshot(self):
        self.assertIsNotNone(self.window._drr_reuse_state)
        self.window._clear_loaded_drr_view()
        self.assertIsNone(self.window._drr_reuse_state)
        self.assertEqual(self.window._drr_heatmap_renders, {})

    def test_both_split_update_preserves_meshes_and_matches_rebuild(self):
        w = self.window
        w.drr_view_side_btn.setChecked(True)
        self._wait_derivative()
        w.drr_split_scale_chk.setChecked(True)
        self._wait_derivative()
        axes = tuple(w.figure.axes)
        meshes = tuple(w._drr_heatmap_axes['second'].collections)
        self._set_silently(w.drr_sg_window_spin, 11)
        w._plot_mode('DRR')
        self._wait_derivative()
        self.assertEqual(tuple(w.figure.axes), axes)
        self.assertEqual(tuple(w._drr_heatmap_axes['second'].collections), meshes)
        w.canvas.draw()
        updated = np.asarray(w.canvas.buffer_rgba()).copy()
        w._drr_reuse_state = None
        w._plot_mode('DRR')
        w.canvas.draw()
        np.testing.assert_array_equal(updated, np.asarray(w.canvas.buffer_rgba()))
        self.errors.assert_not_called()

    def test_color_and_gate_update_do_not_restore_stale_pixels(self):
        w = self.window
        axes = tuple(w.figure.axes)
        self._set_silently(w.drr_spins['vmax'], 12)
        w._plot_mode('DRR')
        self.assertEqual(tuple(w.figure.axes), axes)
        w.canvas.draw()
        self._set_silently(w.drr_spins['gate'], 1)
        w._plot_mode('DRR')
        w.canvas.draw()
        updated = np.asarray(w.canvas.buffer_rgba()).copy()
        w._drr_reuse_state = None
        w._last_plot_params_key = None
        w._plot_mode('DRR')
        w.canvas.draw()
        np.testing.assert_array_equal(updated, np.asarray(w.canvas.buffer_rgba()))

    def test_new_cube_rebuilds_even_when_grid_matches(self):
        from dataclasses import replace
        w = self.window
        axes = tuple(w.figure.axes)
        w.loaded.cube = replace(self.cube, Z=self.cube.Z*2)
        w._plot_mode('DRR')
        self.assertNotEqual(tuple(w.figure.axes), axes)

    def test_second_sg_update_reuses_axes_mesh_colorbar_and_line(self):
        w = self.window
        w._on_drr_plot_view_changed('second')
        self._wait_derivative()
        axes = tuple(w.figure.axes)
        mesh = w._drr_heatmap_axes['second'].collections[0]
        colorbar = mesh.colorbar
        line = w._drr_spectrum_lines['second']
        self._set_silently(w.drr_sg_window_spin, 11)
        w._plot_mode('DRR')
        self._wait_derivative()
        self.assertEqual(tuple(w.figure.axes), axes)
        self.assertIs(w._drr_heatmap_axes['second'].collections[0], mesh)
        self.assertIs(mesh.colorbar, colorbar)
        self.assertIs(w._drr_spectrum_lines['second'], line)
        np.testing.assert_allclose(mesh.get_array(), w._drr_plot_cubes['second'].Z)
        self.errors.assert_not_called()

    def test_reused_render_matches_full_rebuild_pixels(self):
        w = self.window
        w._on_drr_plot_view_changed('second')
        self._wait_derivative()
        self._set_silently(w.drr_sg_window_spin, 11)
        w._plot_mode('DRR')
        self._wait_derivative()
        w.canvas.draw()
        updated = np.asarray(w.canvas.buffer_rgba()).copy()
        w._drr_reuse_state = None
        w._plot_mode('DRR')
        w.canvas.draw()
        np.testing.assert_array_equal(updated, np.asarray(w.canvas.buffer_rgba()))

    def test_layout_switch_rebuilds_and_keeps_zoom(self):
        w = self.window
        axes = tuple(w.figure.axes)
        w._on_drr_plot_view_changed('second')
        self._wait_derivative()
        self.assertNotEqual(tuple(w.figure.axes), axes)
        self._assert_visible_limits((-2,2),(-1,1))
