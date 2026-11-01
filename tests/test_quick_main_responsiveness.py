"""Regression coverage for retained color edits and export/GUI contention."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class NativeMainResponsivenessTests(unittest.TestCase):
    def test_isolated_native_operations(self):
        with TemporaryDirectory(prefix="quick-main-responsiveness-") as folder:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="agg",
                       QT_QUICK_BACKEND="software", MPLCONFIGDIR=folder,
                       XDG_CONFIG_HOME=folder, XDG_CACHE_HOME=folder)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                                    cwd=Path(__file__).resolve().parents[1], env=env,
                                    capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout)


def isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import threading
    import time
    from dataclasses import replace
    from unittest.mock import patch
    import numpy as np
    from PySide6.QtCore import QSettings, QTimer
    from PySide6.QtGui import QGuiApplication
    from core.loader import DataCube
    from core.typography import matplotlib_font_context
    from ui_qt.quick.application_bridge import QuickApplicationBridge
    from ui_qt.quick.drr_workflow import DrrProducts, DrrRequest

    app = QGuiApplication([])

    def wait(predicate, timeout=10):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            app.processEvents()
            if predicate():
                return
            time.sleep(.001)
        raise AssertionError("Native plot did not settle")

    class Cases(unittest.TestCase):
        def setUp(self):
            temporary = TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            self.bridge = QuickApplicationBridge(settings=QSettings(str(Path(temporary.name)/"settings.ini"), QSettings.IniFormat))
            self.addCleanup(self.bridge.close)
            self.energy = np.linspace(1.7,1.9,91)
            self.gate = np.linspace(-2,2,21)
            self.z = 1 + np.exp(-((self.energy[None,:]-1.8-self.gate[:,None]*.01)/.02)**2)
            self.cube = DataCube(self.energy,self.gate,self.z,"Gate (V)","Synthetic responsiveness test","PL")
            self.bridge.set_cube(self.cube)
            self.idle()

        def idle(self):
            b=self.bridge
            wait(lambda:not b._plot_timer.isActive() and not b._publish_timer.isActive()
                 and b._canvas is not None and not b._canvas.backend._timer.isActive())

        def drr(self):
            b=self.bridge;b.selectWorkflow("drr")
            raw=DataCube(self.energy,self.gate,self.z-1.3,"Gate (V)","DRR","ΔR/R")
            b._drr_products=DrrProducts(DrrRequest(()),None,raw,raw,21,21)
            b.set_cube(raw);self.idle()
            return raw

        def test_pl_colors_keep_axes_and_restore_clipped_source(self):
            b=self.bridge
            axes=tuple(b._canvas.figure.axes)
            line=b._spectra[0][1]
            b.setParameter("pl.clip",True);self.idle()
            b.setParameter("vmax",1.4);self.idle()
            mesh=b._maps[0][0]._dptk_quick_render[0].primary
            np.testing.assert_allclose(mesh.get_array(),np.minimum(self.z,1.4))
            b.setParameter("vmax",2.5);self.idle()
            self.assertEqual(tuple(b._canvas.figure.axes),axes)
            self.assertIs(b._spectra[0][1],line)
            np.testing.assert_allclose(mesh.get_array(),self.z)
            b.setParameter("cmap","magma");self.idle()
            self.assertEqual(mesh.get_cmap().name,"magma")
            self.assertEqual(tuple(b._canvas.figure.axes),axes)

        def test_drr_color_norm_and_clipping_match_full_redraw(self):
            raw=self.drr();b=self.bridge
            axes=tuple(b._canvas.figure.axes)
            b._drr_display.setFlag("clip",True)
            b._drr_display.setNumber("raw.vmax","0.4");self.idle()
            mesh=b._maps[0][0]._dptk_quick_render[0].primary
            np.testing.assert_allclose(mesh.get_array(),np.minimum(raw.Z,.4))
            self.assertEqual(tuple(b._canvas.figure.axes),axes)
            b._drr_display.setNumber("raw.vmax","0.9");self.idle()
            expected=mesh.to_rgba(mesh.get_array()).copy()
            b._schedule_plot();self.idle()
            replacement=b._maps[0][0]._dptk_quick_render[0].primary
            np.testing.assert_allclose(replacement.to_rgba(replacement.get_array()),expected)

        def test_pending_new_data_cannot_be_suppressed_by_color_edit(self):
            b=self.bridge;old=b._maps[0][0]
            other=DataCube(self.energy,self.gate,self.z*2,"Gate (V)","New data","PL")
            b.set_cube(other);b.setParameter("vmax",5.);self.idle()
            self.assertIsNot(b._maps[0][0],old)
            np.testing.assert_allclose(b._maps[0][0]._dptk_quick_render[0].primary.get_array(),other.Z)

        def test_new_drr_data_reuses_artists_and_matches_full_rebuild_pixels(self):
            from core.processing import nearest_gate_spectrum
            previous = self.drr()
            b = self.bridge
            axes = tuple(b._canvas.figure.axes)
            render = b._maps[0][0]._dptk_quick_render[0]
            mesh, bar = render.primary, render.primary.colorbar
            b._spectra[0][0].set_ylim(-99., -98.)
            b._maps[0][0].set_xlim(1.75, 1.85)
            raw = replace(previous, Z=previous.Z * 3 + .2, title="Another measurement")
            b._drr_products = replace(b._drr_products, source=raw, raw=raw)
            b.set_cube(raw)
            self.idle()
            self.assertEqual(tuple(b._canvas.figure.axes), axes)
            self.assertIs(b._maps[0][0]._dptk_quick_render[0].primary, mesh)
            self.assertIs(mesh.colorbar, bar)
            np.testing.assert_array_equal(mesh.get_array(), raw.Z)
            self.assertEqual(render.colorbar_host._dptk_header_title, raw.title)
            self.assertTrue(b._spectra[0][0].get_autoscaley_on())
            self.assertEqual(b._maps[0][0].get_xlim(), b._drr_display.heatmap_params(raw).xlim)
            gate, spectrum = nearest_gate_spectrum(raw, b._values["gate"])
            np.testing.assert_array_equal(b._spectra[0][1].get_ydata(), spectrum)
            np.testing.assert_array_equal(b._maps[0][1].get_ydata(), [gate, gate])
            retained = np.asarray(b._canvas.buffer_rgba()).copy()
            with patch("ui_qt.quick.drr_plot_update.reuse_drr_plot", return_value=False):
                b._schedule_plot()
                self.idle()
            np.testing.assert_array_equal(np.asarray(b._canvas.buffer_rgba()), retained)

        def test_drr_pair_reuse_keeps_gate_colors_and_manual_overlays_current(self):
            from core.processing import nearest_gate_spectrum
            previous = self.drr()
            b = self.bridge
            b.setParameter("drrView", "pair")
            self.idle()
            axes = tuple(b._canvas.figure.axes)
            raw = replace(previous, Z=previous.Z * 2, title="Next source")
            second = replace(b._drr_products.second, Z=b._drr_products.second.Z - .1)
            b._drr_products = replace(b._drr_products, source=raw, raw=raw, second=second)
            b.set_cube(raw)
            self.idle()
            self.assertEqual(tuple(b._canvas.figure.axes), axes)
            for cube, (axis, _), artists in zip((raw, second), b._maps, b._manual_artists):
                self.assertIs(axis._dptk_quick_render[1], cube)
                self.assertIs(artists[0], cube)
                np.testing.assert_array_equal(axis._dptk_quick_render[0].primary.get_array(), cube.Z)
            b.setParameter("gate", float(self.gate[15]))
            self.idle()
            for cube, (_, line) in zip((raw, second), b._spectra):
                _, values = nearest_gate_spectrum(cube, self.gate[15])
                np.testing.assert_array_equal(line.get_ydata(), values)
            b._drr_display.setNumber("raw.vmax", "2.5")
            self.idle()
            self.assertEqual(tuple(b._canvas.figure.axes), axes)
            self.assertEqual(b._maps[0][0]._dptk_quick_render[0].primary.get_clim()[1], 2.5)

        def test_incompatible_second_grid_rebuilds_both_products_before_mutation(self):
            previous = self.drr()
            b = self.bridge
            b.setParameter("drrView", "pair")
            self.idle()
            old_axis = b._maps[0][0]
            old_mesh = old_axis._dptk_quick_render[0].primary
            raw = replace(previous, Z=previous.Z + .2)
            second = replace(b._drr_products.second, energy=self.energy + .001)
            b._drr_products = replace(b._drr_products, source=raw, raw=raw, second=second)
            with patch.object(old_mesh, "set_array", wraps=old_mesh.set_array) as update:
                b.set_cube(raw)
                self.idle()
                self.assertEqual(update.call_count, 0)
            self.assertIsNot(b._maps[0][0], old_axis)
            self.assertIs(b._plot_cubes[1], second)

        def test_split_drr_pair_with_missing_values_matches_full_rebuild(self):
            previous = self.drr()
            b = self.bridge
            b.setParameter("drrView", "pair")
            b._drr_display.setChoice("regionCount", 1)
            self.idle()
            axes = tuple(b._canvas.figure.axes)
            values = previous.Z.copy()
            values[4, 20] = np.nan
            values[8, 60] = np.inf
            raw = replace(previous, Z=values)
            second = replace(previous, Z=-values)
            b._drr_products = replace(b._drr_products, source=raw, raw=raw, second=second)
            b.set_cube(raw)
            self.idle()
            self.assertEqual(tuple(b._canvas.figure.axes), axes)
            self.assertTrue(all(len(axis._dptk_quick_render[0].images) == 2 for axis, _ in b._maps))
            retained = np.asarray(b._canvas.buffer_rgba()).copy()
            with patch("ui_qt.quick.drr_plot_update.reuse_drr_plot", return_value=False):
                b._schedule_plot()
                self.idle()
            np.testing.assert_array_equal(np.asarray(b._canvas.buffer_rgba()), retained)

        def test_derivative_reuse_keeps_only_unchanged_product_manual_spectrum_y(self):
            previous = self.drr()
            b = self.bridge
            b.setParameter("drrView", "pair")
            self.idle()
            axes = tuple(b._canvas.figure.axes)
            for axis, _ in b._spectra:
                axis.set_ylim(-99., -98.)
            second = replace(previous, Z=previous.Z * 4)
            b._drr_products = replace(b._drr_products, second=second)
            b.set_cube(previous)
            self.idle()
            self.assertEqual(tuple(b._canvas.figure.axes), axes)
            self.assertEqual(b._spectra[0][0].get_ylim(), (-99., -98.))
            self.assertFalse(b._spectra[0][0].get_autoscaley_on())
            self.assertTrue(b._spectra[1][0].get_autoscaley_on())
            self.assertNotEqual(b._spectra[1][0].get_ylim(), (-99., -98.))

        def test_gate_autoscale_repaints_spectrum_without_redrawing_heatmap(self):
            for workflow in ("pl", "drr"):
                with self.subTest(workflow=workflow):
                    b=self.bridge
                    if workflow=="drr":self.drr()
                    values=self.z*np.linspace(1.,100.,len(self.gate))[:,None]
                    cube=DataCube(self.energy,self.gate,values,"Gate (V)","Changing spectrum amplitude","Intensity")
                    b.set_cube(cube);self.idle()
                    draws=b._canvas.draw_count
                    blits=b._canvas.blit_count
                    b.setParameter("gate",float(self.gate[-1]));self.idle()
                    self.assertEqual(b._canvas.draw_count,draws,"Gate/Y-tick changes redrew the whole heatmap")
                    self.assertEqual(b._canvas.blit_count,blits+1,"One gate edit painted the frame twice")
                    np.testing.assert_array_equal(b._spectra[0][1].get_ydata(),values[-1])
                    partial=np.asarray(b._canvas.buffer_rgba()).copy()
                    b._canvas.draw()
                    np.testing.assert_array_equal(np.asarray(b._canvas.buffer_rgba()),partial)

        def test_color_partial_frame_matches_full_redraw_pixels(self):
            for workflow in ("pl", "drr"):
                with self.subTest(workflow=workflow):
                    b=self.bridge
                    if workflow=="drr":self.drr()
                    if workflow=="pl":b.setParameter("vmax",1.5)
                    else:b._drr_display.setNumber("raw.vmax","0.6")
                    self.idle()
                    partial=np.asarray(b._canvas.buffer_rgba()).copy()
                    b._canvas.draw()
                    np.testing.assert_array_equal(np.asarray(b._canvas.buffer_rgba()),partial)

        def test_export_lock_defers_full_frame_and_gate_without_blocking_events(self):
            b=self.bridge;held=threading.Event();release=threading.Event();expired=[]
            def worker():
                with matplotlib_font_context():
                    held.set()
                    if not release.wait(3):expired.append(True)
            thread=threading.Thread(target=worker)
            self.addCleanup(thread.join)
            self.addCleanup(release.set)
            thread.start();self.assertTrue(held.wait(3))
            generation=b._canvas.frame_generation
            b.setParameter("gate",float(self.gate[15]))
            b._canvas.draw_idle()
            ticks=[];timer=QTimer();timer.setInterval(5);timer.timeout.connect(lambda:ticks.append(True));timer.start()
            self.addCleanup(timer.stop)
            wait(lambda:len(ticks)>=5,timeout=1)
            self.assertFalse(expired,"GUI waited for worker instead of processing its timers")
            self.assertEqual(b._canvas.frame_generation,generation)
            release.set();thread.join();self.idle()
            self.assertGreater(b._canvas.frame_generation,generation)
            np.testing.assert_allclose(b._spectra[0][1].get_ydata(),self.z[15])

        def test_structural_plot_change_deferred_under_export_lock(self):
            b=self.bridge;held=threading.Event();release=threading.Event()
            def worker():
                with matplotlib_font_context():held.set();release.wait(3)
            thread=threading.Thread(target=worker);thread.start()
            self.addCleanup(thread.join);self.addCleanup(release.set)
            self.assertTrue(held.wait(3))
            old=tuple(b._canvas.figure.axes)
            b.setParameter("sideBySide",True)
            signalled=[];QTimer.singleShot(25,lambda:signalled.append(True))
            wait(lambda:bool(signalled),timeout=1)
            self.assertFalse(release.is_set())
            self.assertEqual(tuple(b._canvas.figure.axes),old)
            release.set();thread.join();self.idle()
            self.assertEqual(len(b._maps),2)

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    print(f"QUICK_MAIN_RESPONSIVENESS_TESTS={result.testsRun}")
    return 0 if result.wasSuccessful() else 1


if __name__=="__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(isolated())
    unittest.main()
