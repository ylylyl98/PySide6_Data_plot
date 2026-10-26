import unittest
import numpy as np
from tests.test_drr_dual_view_regressions import DrrDualViewFixture


class DrrPlotReuseTests(DrrDualViewFixture, unittest.TestCase):
    def test_split_color_partial_draw_matches_rebuild_for_both_products(self):
        from unittest.mock import patch
        w = self.window
        # Leave the shared Arial suptitle outside the color redraw rectangle.
        w.resize(1700, 1100)
        w.show()
        w.drr_view_side_btn.setChecked(True)
        self._wait_derivative()
        for count in (2, 3):
            w.resize(2200 if count == 3 else 1700, 1100)
            self.app.processEvents()
            self._set_silently(w.drr_split_spins['x0'], -.6)
            self._set_silently(w.drr_split_spins['x1'], .6)
            w.drr_region_count_combo.setCurrentIndex(count - 1)
            self._wait_derivative()
            w.canvas.draw()
            self.app.processEvents()
            title = w.figure._suptitle.get_window_extent(w.canvas.get_renderer())
            frame = w._drr_region_blitters['gate_view']._color_frame
            self.assertTrue(all(not title.overlaps(box.padded(3)) for box in frame.bounds.values()))
            for product, controls in (('raw', w.drr_split_spins), ('second', w.drr_second_split_spins)):
                with self.subTest(regions=count, product=product):
                    self._set_silently(controls['left_vmax'], 5. + count)
                    self._set_silently(controls['right_vmin'], -2. - count)
                    if count == 3:
                        self._set_silently(controls['middle_vmax'], 10. + count)
                    with patch.object(w.canvas, 'blit', wraps=w.canvas.blit) as blit:
                        w._plot_mode('DRR')
                        blit.assert_called_once()
                    actual = np.asarray(w.canvas.buffer_rgba()).copy()
                    w._drr_reuse_state = None
                    w._last_plot_params_key = None
                    w._plot_mode('DRR')
                    w.canvas.draw()
                    np.testing.assert_array_equal(actual, np.asarray(w.canvas.buffer_rgba()))
                    self.app.processEvents()
        self.errors.assert_not_called()

    def test_coalesced_color_and_gate_edit_refreshes_linecut(self):
        from core.processing import nearest_gate_spectrum
        w = self.window
        w.canvas.draw()
        self.app.processEvents()
        self._set_silently(w.drr_spins['vmax'], 25.)
        self._set_silently(w.drr_spins['gate'], 1.)
        w._plot_mode('DRR')
        self.app.processEvents()
        _, expected = nearest_gate_spectrum(self.cube, 1.)
        np.testing.assert_array_equal(w._drr_spectrum_line.get_ydata(), expected)
        self.errors.assert_not_called()

    def test_overlapping_colorbar_labels_fall_back_to_complete_draw(self):
        from unittest.mock import patch
        w = self.window
        w.resize(1400, 1000)
        w.show()
        w.drr_view_side_btn.setChecked(True)
        self._set_silently(w.drr_split_spins['x0'], -.6)
        self._set_silently(w.drr_split_spins['x1'], .6)
        w.drr_region_count_combo.setCurrentIndex(2)
        self._wait_derivative()
        # Overlap header decorations, preserving disjoint measured-data
        # surfaces. Moving a whole map was specific to the old right-side
        # bars and now also overlaps gate overlays in the wider map layout.
        raw_map = w._drr_heatmap_axes['raw']
        other_map = w._drr_heatmap_axes['second']
        raw_host = w._drr_heatmap_renders['raw'].colorbar_host
        other_host = w._drr_heatmap_renders['second'].colorbar_host
        renderer = w.canvas.get_renderer()
        raw_header = raw_host.get_tightbbox(renderer)
        quantity = other_host._dptk_header_quantity
        quantity.set_x((raw_header.x1 - 8 - other_host.bbox.x0) / other_host.bbox.width)
        w.canvas.draw()
        self.app.processEvents()
        renderer = w.canvas.get_renderer()
        self.assertFalse(raw_map.bbox.overlaps(other_map.bbox))
        self.assertTrue(raw_host.get_tightbbox(renderer).overlaps(quantity.get_window_extent(renderer)))
        self._set_silently(w.drr_split_spins['left_vmax'], 1.e6)
        with patch.object(w.canvas, 'draw', wraps=w.canvas.draw) as draw:
            w._plot_mode('DRR')
            self.app.processEvents()
            draw.assert_called_once()
        actual = np.asarray(w.canvas.buffer_rgba()).copy()
        w._drr_region_blitters['gate_view'].prepare_full_redraw()
        w.canvas.draw()
        np.testing.assert_array_equal(actual, np.asarray(w.canvas.buffer_rgba()))
        self.errors.assert_not_called()

    def test_color_edit_repaints_only_changed_map_then_gate_keeps_new_colors(self):
        from unittest.mock import patch
        w = self.window
        # This case exercises a nonoverlapping color update; overlap fallback
        # is covered separately, including figure titles and colorbar labels.
        w.resize(1700, 1100)
        w.show()
        w.drr_view_side_btn.setChecked(True)
        self._wait_derivative()
        w.canvas.draw()
        self.app.processEvents()
        raw = w._drr_heatmap_axes['raw']
        other = w._drr_heatmap_axes['second']
        spectrum = w._drr_spectrum_axes['raw']
        title = w.figure._suptitle.get_window_extent(w.canvas.get_renderer())
        frame = w._drr_region_blitters['gate_view']._color_frame
        self.assertTrue(all(not title.overlaps(box.padded(3)) for box in frame.bounds.values()))
        for limit in (100., 1.e5, 20.):
            self._set_silently(w.drr_spins['vmax'], limit)
            with patch.object(other, 'draw', wraps=other.draw) as unchanged, \
                    patch.object(spectrum, 'draw', wraps=spectrum.draw) as linecut, \
                    patch.object(w.canvas, 'blit', wraps=w.canvas.blit) as blit:
                w._plot_mode('DRR')
                blit.assert_called_once()
                unchanged.assert_not_called()
                linecut.assert_not_called()
            actual = np.asarray(w.canvas.buffer_rgba()).copy()
            helper = w._drr_region_blitters['gate_view']
            helper.prepare_full_redraw()
            w.canvas.draw()
            np.testing.assert_array_equal(actual, np.asarray(w.canvas.buffer_rgba()))
            helper.restore_interactive_drawing()
            self.app.processEvents()
        self._set_silently(w.drr_spins['gate'], 1)
        w._plot_mode('DRR')
        actual = np.asarray(w.canvas.buffer_rgba()).copy()
        helper.prepare_full_redraw()
        w.canvas.draw()
        np.testing.assert_array_equal(actual, np.asarray(w.canvas.buffer_rgba()))
        self.errors.assert_not_called()

    def test_region_changes_keep_measured_right_axis(self):
        from dataclasses import replace
        from unittest.mock import patch
        from core.plotting import RightAxis

        w = self.window
        make_params = w._make_drr_params
        mapping = RightAxis('Measured field', (-1., 0., 1.), (2., 4., 7.))
        def params(*args, **kwargs):
            return replace(make_params(*args, **kwargs), right_axis=mapping)
        with patch.object(w, '_make_drr_params', side_effect=params):
            w._drr_reuse_state = None
            w._plot_mode('DRR')
            axis = w._drr_heatmap_axes['raw']
            right_axes = tuple(axis.child_axes)
            self.assertEqual(len(right_axes), 1)
            for count in (2, 3, 1):
                self._set_silently(w.drr_split_spins['x0'], -.6)
                self._set_silently(w.drr_split_spins['x1'], .6)
                w.drr_region_count_combo.setCurrentIndex(count - 1)
                self._wait_derivative()
                w._plot_mode('DRR')
                self.assertIs(w._drr_heatmap_axes['raw'], axis)
                self.assertEqual(tuple(axis.child_axes), right_axes)
            w.canvas.draw()
            actual = np.asarray(w.canvas.buffer_rgba()).copy()
            w._drr_reuse_state = None
            w._plot_mode('DRR')
            w.canvas.draw()
            np.testing.assert_array_equal(actual, np.asarray(w.canvas.buffer_rgba()))
        self.errors.assert_not_called()

    def test_region_count_changes_keep_maps_linecuts_and_match_full_rebuild(self):
        w = self.window
        for paired in (False, True):
            w.drr_view_side_btn.setChecked(paired)
            self._wait_derivative()
            for count in (2, 3, 1, 3, 2, 1):
                with self.subTest(paired=paired, regions=count):
                    maps = dict(w._drr_heatmap_axes)
                    lines = dict(w._drr_spectrum_lines)
                    axes = tuple(w.figure.axes)
                    self._set_silently(w.drr_split_spins['x0'], -.6)
                    self._set_silently(w.drr_split_spins['x1'], .6)
                    w.drr_region_count_combo.setCurrentIndex(count - 1)
                    self._wait_derivative()
                    w._plot_mode('DRR')
                    self.assertEqual(w._drr_heatmap_axes, maps)
                    self.assertEqual(w._drr_spectrum_lines, lines)
                    self.assertEqual(tuple(w.figure.axes), axes)
                    for render in w._drr_heatmap_renders.values():
                        self.assertEqual(len(render.images), count)
                        self.assertEqual(len(render.primary.axes.collections), count)
                        self.assertEqual(len(render.colorbar_host.child_axes), count)
                    w.canvas.draw()
                    actual = np.asarray(w.canvas.buffer_rgba()).copy()
                    w._drr_reuse_state = None
                    w._last_plot_params_key = None
                    w._plot_mode('DRR')
                    w.canvas.draw()
                    np.testing.assert_array_equal(actual, np.asarray(w.canvas.buffer_rgba()))
        self.errors.assert_not_called()

    def test_moving_split_reuses_axes_and_colorbars_and_matches_rebuild(self):
        w = self.window
        for count in (2, 3):
            with self.subTest(regions=count):
                self._set_silently(w.drr_split_spins['x0'], -.6)
                self._set_silently(w.drr_split_spins['x1'], .6)
                w.drr_region_count_combo.setCurrentIndex(count - 1)
                self._wait_derivative()
                w._plot_mode('DRR')
                axes = tuple(w.figure.axes)
                colorbars = tuple(m.colorbar for m in w._drr_heatmap_renders['raw'].images)
                for boundary in (-.2, -.8, -.4):
                    self._set_silently(w.drr_split_spins['x0'], boundary)
                    w._plot_mode('DRR')
                    self.assertEqual(tuple(w.figure.axes), axes)
                    meshes = w._drr_heatmap_renders['raw'].images
                    self.assertEqual(tuple(m.colorbar for m in meshes), colorbars)
                    self.assertEqual(len(w._drr_heatmap_axes['raw'].collections), count)
                    w.canvas.draw()
                    actual = np.asarray(w.canvas.buffer_rgba()).copy()
                    w._drr_reuse_state = None
                    w._last_plot_params_key = None
                    w._plot_mode('DRR')
                    w.canvas.draw()
                    np.testing.assert_array_equal(actual, np.asarray(w.canvas.buffer_rgba()))
                    axes = tuple(w.figure.axes)
                    colorbars = tuple(m.colorbar for m in w._drr_heatmap_renders['raw'].images)
        self.errors.assert_not_called()

    def test_split_range_edit_reuses_meshes_and_matches_rebuild(self):
        w = self.window
        for count in (2, 3):
            with self.subTest(regions=count):
                self._set_silently(w.drr_split_spins['x0'], -.6)
                self._set_silently(w.drr_split_spins['x1'], .6)
                w.drr_region_count_combo.setCurrentIndex(count - 1)
                self._wait_derivative()
                w._plot_mode('DRR')
                axes = tuple(w.figure.axes)
                meshes = w._drr_heatmap_renders['raw'].images
                for key, value in (('xmin', -1.8), ('xmax', 1.8), ('ymin', -.7), ('ymax', .7)):
                    self._set_silently(w.drr_spins[key], value)
                w._drr_limits_from_controls = True
                w._drr_view_limits = None
                w._plot_mode('DRR')
                self.assertEqual(tuple(w.figure.axes), axes)
                self.assertEqual(w._drr_heatmap_renders['raw'].images, meshes)
                self._assert_visible_limits((-1.8, 1.8), (-.7, .7))
                w.canvas.draw()
                actual = np.asarray(w.canvas.buffer_rgba()).copy()
                w._drr_reuse_state = None
                w._plot_mode('DRR')
                w.canvas.draw()
                np.testing.assert_array_equal(actual, np.asarray(w.canvas.buffer_rgba()))
        self.errors.assert_not_called()

    def test_range_edit_does_not_rebuild_unchanged_colorbar(self):
        from unittest.mock import patch
        w = self.window
        mesh = w._drr_heatmap_renders['raw'].primary
        self._set_silently(w.drr_spins['ymin'], -.5)
        with patch.object(mesh, 'changed', wraps=mesh.changed) as changed:
            w._plot_mode('DRR')
        changed.assert_not_called()
        self.assertIs(w._drr_heatmap_renders['raw'].primary, mesh)
        self.errors.assert_not_called()

    def test_unsplit_range_edit_reuses_axes_mesh_and_matches_rebuild(self):
        w = self.window
        w.drr_region_count_combo.setCurrentIndex(0)
        w.drr_split_scale_chk.setChecked(False)
        self._wait_derivative()
        w._plot_mode('DRR')
        axes = tuple(w.figure.axes)
        mesh = w._drr_heatmap_renders['raw'].primary
        colorbar = mesh.colorbar
        w.drr_spins['xmin'].setValue(-1.5)
        w.drr_spins['xmax'].setValue(1.5)
        w.drr_spins['ymin'].setValue(-.5)
        w.drr_spins['ymax'].setValue(.5)
        w._plot_redraw_timers['DRR'].stop()
        w._run_scheduled_plot_redraw('DRR')
        self.assertEqual(tuple(w.figure.axes), axes)
        self.assertIs(w._drr_heatmap_renders['raw'].primary, mesh)
        self.assertIs(mesh.colorbar, colorbar)
        self._assert_visible_limits((-1.5, 1.5), (-.5, .5))
        w.canvas.draw()
        updated = np.asarray(w.canvas.buffer_rgba()).copy()
        w._drr_reuse_state = None
        w._last_plot_params_key = None
        w._plot_mode('DRR')
        w.canvas.draw()
        np.testing.assert_array_equal(updated, np.asarray(w.canvas.buffer_rgba()))
        self.errors.assert_not_called()

    def test_full_resolution_survives_color_update(self):
        from core.loader import DataCube
        w = self.window
        cube = DataCube(np.linspace(-2, 2, 1340), np.linspace(-1, 1, 301),
                        np.ones((301, 1340)), 'Gate', 'large', 'DR/R')
        cube.Z[150, 671] = 12345
        w.loaded.cube = cube
        w.drr_split_scale_chk.setChecked(False)
        w.drr_clip_chk.setChecked(False)
        w._plot_mode('DRR')
        mesh = w._drr_heatmap_renders['raw'].primary
        self.assertEqual(mesh.get_array().shape, cube.Z.shape)
        self.assertEqual(mesh.get_array()[150, 671], 12345)
        self._set_silently(w.drr_spins['vmax'], 10)
        w._plot_mode('DRR')
        self.assertIs(w._drr_heatmap_renders['raw'].primary, mesh)
        np.testing.assert_array_equal(mesh.get_array(), cube.Z)

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
