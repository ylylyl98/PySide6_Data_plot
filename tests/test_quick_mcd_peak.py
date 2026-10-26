"""Scientific and live-QML MCD Peak Shift parity, using isolated settings."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickMcdPeakTests(unittest.TestCase):
    def test_native_peak_workflow(self):
        with TemporaryDirectory() as temporary:
            env=dict(os.environ,QT_QPA_PLATFORM="offscreen",QT_QUICK_BACKEND="software",QT_SCALE_FACTOR="1",MPLBACKEND="agg",
                MPLCONFIGDIR=temporary,LOCALAPPDATA=temporary,XDG_CONFIG_HOME=temporary,XDG_CACHE_HOME=temporary,QML_DISK_CACHE_PATH=temporary)
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--isolated"],cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


def _isolated():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import csv
    from dataclasses import asdict,replace
    import json
    from threading import Event
    import time
    from unittest.mock import patch
    import numpy as np
    from PySide6.QtCore import QObject,QSettings,QUrl
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlComponent,QQmlEngine
    from PySide6.QtQuick import QQuickWindow
    from core.mcd import McdSettings
    from core.mcd_peak_shift import analyze_peak_shift
    from core.load_cancellation import LoadCancellation,LoadCancelled,load_scope
    from ui_qt.quick.mcd_bridge import McdBridge
    from ui_qt.quick.mcd_peak_bridge import McdPeakBridge
    from ui_qt.quick import mcd_peak_workflow as workflow
    from ui_qt.quick.plot_surface import register_plot_surface
    app=QGuiApplication([]);register_plot_surface()
    colors=dict(surface="#f3f5f8",control="#ffffff",text="#20242b",muted="#606978",border="#c6cbd3",accent="#0067c0",hover="#edf1f6",pressed="#e0e7ef",error="#b42318",disabled="#939aa5",selectedText="#ffffff",inset="#e9edf2",divider="#dce1e8",borderHover="#aeb7c4",borderDisabled="#dce1e8",controlDisabled="#edf0f4",focus="#20242b",accentText="#0067c0",selected="#e6f0fa",selectedBorder="#b4d6fa",errorSurface="#fdf1f0")
    def settle(milliseconds=15):
        deadline=time.monotonic()+milliseconds/1000
        while time.monotonic()<deadline:app.processEvents();time.sleep(.001)
        app.processEvents()
    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary=TemporaryDirectory();self.addCleanup(self.temporary.cleanup);self.root=Path(self.temporary.name)
            self.source=self.root/"mcd"/"sample.csv";self.source.parent.mkdir();self.energy=np.linspace(1.6,1.8,151)
            self.write_source();self.settings=QSettings(str(self.root/"settings.ini"),QSettings.Format.IniFormat)
            self.bridge=McdPeakBridge(settings=self.settings);self.addCleanup(self.bridge.close)
        def write_source(self,fields=None,mixed=False):
            fields=np.r_[np.linspace(-1,1,9),np.linspace(.75,-1,8)] if fields is None else fields
            with self.source.open("w",newline="") as stream:
                writer=csv.writer(stream);writer.writerow(["B_T","angle_deg",*list(1239.841984/self.energy)])
                for field in fields:
                    for angle,sign in ((90,-1),(0,1)):
                        if mixed:
                            from core.mcd_local_fit import mixed_model
                            values=mixed_model((self.energy-1.68)*1000,np.array([10000.,0.,1800.,150.,sign*field,4.]))
                        else:values=10000+1800*np.exp(-((self.energy-1.68-sign*.001*field)/.008)**2)+350*np.exp(-((self.energy-1.75)/.012)**2)
                        writer.writerow([field,angle,*values])
        def wait(self):
            deadline=time.monotonic()+35
            while time.monotonic()<deadline:
                settle();state=self.bridge.snapshot
                if not state["busy"] and not state["analysisBusy"] and not state["catalogBusy"] and not self.bridge._load_timer.isActive():break
            self.assertFalse(self.bridge.snapshot["busy"]);self.assertFalse(self.bridge.snapshot["analysisBusy"])
            self.assertFalse(self.bridge.snapshot["error"],self.bridge.snapshot["error"]);settle(35)
        def load(self):self.bridge.setExperimentFolder(str(self.root));self.bridge.openFiles("sources",[str(self.source)]);self.wait()
        def test_catalog_calls_exact_core_for_both_methods_and_measured_channels(self):
            self.load();result=self.bridge._products.result;values=self.bridge._values
            self.assertEqual(set(self.bridge._methods),{"Raw spectrum","Second derivative"})
            self.assertTrue(self.bridge.snapshot["peakCandidates"])
            for method,channels in self.bridge._methods.items():
                for channel,actual in channels.items():
                    expected=analyze_peak_shift(workflow.channel_view(result,channel),source=workflow.channel_source(values,channel),prominence_fraction=.03,min_distance_points=5,smoothing_points=7,max_jump_ev=.04,max_peaks=6,tracking_method=method,derivative_window_points=35)
                    self.assertEqual(actual.tracks,expected.tracks);np.testing.assert_array_equal(actual.fields_t,expected.fields_t)
        def test_full_native_control_contract_and_exact_source_history(self):
            self.load();state=self.bridge.snapshot;keys={field["key"] for group in state["parameterGroups"] for field in group["fields"]}
            for key in ("automaticAngles","darkPlus","pair_b_alignment","correction_mode","peakSource","trackerMethod","peakBranch","selectedField","selectedPeak","peakProminence","peakDistance","peakSmoothing","peakJump","peakLimit","derivativeWindow","mapMode","showPeakTracks","showDerivative","localBackground","manualCenter","selectedK","selectedKp","peak-table","valley-table","export"):self.assertIn(key,keys)
            self.assertEqual(state["values"]["trackerMethod"],"Raw spectrum");self.assertEqual(state["peakHistory"],"new")
            self.assertTrue(state["canExport"]);self.assertTrue(state["curves"])
        def test_result_curves_match_conservative_core_and_keep_one_positive_field_mapping(self):
            from core.mcd_valley_split import compute_valley_splitting
            self.load();curves=self.bridge.snapshot["curves"];selected=self.bridge._peak_selected
            track=workflow.track_for(self.bridge._methods,"Raw spectrum",selected)
            mappings={row.get("k_channel") for row in curves if row["status"]=="ok"};self.assertEqual(len(mappings),1)
            for row in curves:
                expected=compute_valley_splitting(self.bridge._methods,method=row["method"],selected_channel=selected[0],branch=row["branch"],target_energy_ev=track.reference_energy_ev,feature_kind=selected[3],allow_energy_fallback=False,fixed_k_channel=next(iter(mappings)))
                self.assertEqual(row["x"],[point.field_t for point in expected.points]);self.assertEqual(row["y"],[None if point.splitting_ev is None else 1000*point.splitting_ev for point in expected.points])
            self.bridge.setParameter("resultMode","Single peak shift");self.bridge.setParameter("peakDisplay","Absolute E")
            self.assertTrue(self.bridge.snapshot["curves"]);self.assertTrue(all(value is None or 1.6<value<1.8 for row in self.bridge.snapshot["curves"] for value in row["y"]))
        def test_interpolated_rows_use_pair_field_and_raw_rows_keep_actual_field(self):
            self.load();result=self.bridge._products.result;count=len(result.pair_b)
            altered=replace(result,pair_b_pos=result.pair_b+.01,pair_interpolated_pos=np.arange(count)%2==0)
            view=workflow.channel_view(altered,"pos")
            np.testing.assert_array_equal(view.pair_b[::2],result.pair_b[::2]);np.testing.assert_array_equal(view.pair_b[1::2],result.pair_b[1::2]+.01)
            np.testing.assert_array_equal(altered.pair_b,result.pair_b)
        def test_absolute_energies_remain_available_without_inventing_zero_field_reference(self):
            self.load();methods={}
            for method,channels in self.bridge._methods.items():
                methods[method]={channel:replace(analysis,tracks=tuple(replace(track,reference_energy_ev=None,reference_field_t=None,reference_method="unavailable",points=tuple(replace(point,delta_energy_ev=None) for point in track.points)) for track in analysis.tracks)) for channel,analysis in channels.items()}
            values=dict(self.bridge._values,resultMode="Single peak shift",peakDisplay="Delta E")
            curves,note=workflow.result_curves(self.bridge._products.result,methods,values,self.bridge._peak_selected)
            self.assertFalse(curves);self.assertIn("E0",note)
            values["peakDisplay"]="Absolute E";curves,note=workflow.result_curves(self.bridge._products.result,methods,values,self.bridge._peak_selected)
            self.assertTrue(curves);self.assertTrue(any(value is not None for row in curves for value in row["y"]))
        def test_field_updates_reuse_axes_lines_and_do_not_recompute_science(self):
            self.load();self.bridge._render();settle(50);axes=dict(self.bridge._axes);line=self.bridge._raw_lines["pos"];methods=self.bridge._methods
            draws=self.bridge._canvas.draw_count;blits=self.bridge._canvas.blit_count
            indices=self.bridge._field_indices();self.bridge.setParameter("selectedField",int(indices[-1]));settle(50)
            self.assertIs(self.bridge._methods,methods);self.assertEqual(self.bridge._axes,axes);self.assertIs(self.bridge._raw_lines["pos"],line)
            self.assertEqual(self.bridge._canvas.draw_count,draws);self.assertGreater(self.bridge._canvas.blit_count,blits)
            expected=workflow.source_spectra(self.bridge._products.result,"raw pos")[indices[-1]];np.testing.assert_array_equal(line.get_ydata(),expected)
        def test_manual_reseek_merges_distinct_features_without_growing_duplicates(self):
            self.load();selected=self.bridge._peak_selected;track=workflow.track_for(self.bridge._methods,"Raw spectrum",selected)
            count=sum(len(analysis.tracks) for channels in self.bridge._methods.values() for analysis in channels.values())
            self.bridge.setParameter("manualCenter",track.reference_energy_ev);self.bridge.invoke("manual-reseek");self.wait()
            once=sum(len(analysis.tracks) for channels in self.bridge._methods.values() for analysis in channels.values())
            self.assertGreaterEqual(once,count);self.bridge.invoke("manual-reseek");self.wait()
            self.assertEqual(sum(len(analysis.tracks) for channels in self.bridge._methods.values() for analysis in channels.values()),once)
        def test_display_changes_reuse_completed_numerical_tables(self):
            from unittest.mock import patch
            self.load();tables=self.bridge._peak_tables
            with patch.object(workflow,"table_rows",side_effect=AssertionError("display edit rebuilt numerical tables")):
                self.bridge.setParameter("showPeakMaps",True);settle(50)
                self.bridge.setParameter("peakDisplay","Absolute E");settle(50)
            self.assertIs(self.bridge._peak_tables,tables)
        def test_actual_local_fit_keeps_locator_model_and_gap_protocol(self):
            self.write_source(fields=(-.5,.5),mixed=True);self.load();self.bridge.setParameter("trackerMethod","Local mixed fit");self.wait()
            self.assertEqual(set(self.bridge._methods),{"Local mixed fit"})
            for channel,analysis in self.bridge._methods["Local mixed fit"].items():
                self.assertEqual(analysis.tracking_method,"Local mixed fit");self.assertIsNotNone(analysis.locator_energy_ev)
                self.assertEqual(len(analysis.local_fits),2)
                self.assertTrue(all(fit.n_points>=12 for fit in analysis.local_fits));self.assertTrue(analysis.tracks)
        def test_source_sharing_adopts_one_product_without_signal_echo(self):
            self.load();other=McdBridge(settings=self.settings);self.addCleanup(other.close)
            self.bridge.productsChanged.connect(other.adoptProducts);other.productsChanged.connect(self.bridge.adoptProducts)
            events=[];self.bridge.productsChanged.connect(events.append)
            other.adoptProducts(self.bridge._products);settle(50)
            self.assertIs(other._products,self.bridge._products);self.assertFalse(events)
            self.assertEqual(other._source,str(self.source))
        def test_display_and_cached_method_switch_never_restart_scientific_worker(self):
            self.load();cached=self.bridge._methods
            with patch("ui_qt.quick.mcd_peak_workflow.analyze_catalog",side_effect=AssertionError("unexpected scientific recompute")):
                self.bridge.setParameter("trackerMethod","Second derivative");self.wait()
                self.bridge.setParameter("showDerivative",False);self.bridge.setParameter("mapMode","Second derivative")
                self.bridge.setParameter("trackerMethod","Raw spectrum");self.wait()
            self.assertIs(self.bridge._methods,cached)
        def test_superseded_catalog_cannot_replace_new_source_generation(self):
            self.load();old=self.bridge._methods;entered=Event();release=Event()
            def slow(*args,**kwargs):entered.set();release.wait(5);return old
            with patch("ui_qt.quick.mcd_peak_workflow.analyze_catalog",slow):
                self.bridge._peak_cache.clear();self.bridge.invoke("analyze-peaks")
                deadline=time.monotonic()+5
                while not entered.is_set() and time.monotonic()<deadline:settle()
                self.assertTrue(entered.is_set());self.bridge.invoke("cancel");release.set();settle(80)
                self.assertFalse(self.bridge.snapshot["analysisBusy"])
                self.assertFalse(self.bridge._methods)
            self.bridge.invoke("analyze-peaks");self.wait();self.assertTrue(self.bridge._methods)
        def test_async_export_is_frozen_and_exact_source_history_is_recorded(self):
            self.load();self.bridge.invoke("export");self.wait();paths=self.bridge.snapshot["resultPaths"]
            self.assertTrue(any(path.endswith(".csv") for path in paths));metadata=json.loads(Path(next(path for path in paths if path.endswith(".json"))).read_text())
            self.assertEqual(metadata["source_descriptor"]["path"],str(self.source));self.assertEqual(metadata["workflow"],"MCD Peak Shift")
            self.assertTrue(self.bridge.snapshot["peakHistory"].startswith("processed"))
        def test_export_refuses_changed_source_before_writing_files(self):
            self.load();self.source.write_text(self.source.read_text()+"\n")
            self.bridge.invoke("export");deadline=time.monotonic()+10
            while self.bridge.snapshot["busy"] and time.monotonic()<deadline:settle()
            self.assertIn("changed",self.bridge.snapshot["error"]);self.assertFalse(self.bridge.snapshot["resultPaths"])
        def test_native_wide_compact_maps_splitter_and_tables_without_widgets(self):
            self.load();engine=QQmlEngine();warnings=[];engine.warnings.connect(lambda values:warnings.extend(value.toString() for value in values))
            qml=Path(__file__).resolve().parents[1]/"ui_qt"/"quick"/"qml"
            component=QQmlComponent(engine,QUrl.fromLocalFile(str(qml/"McdPeakWorkspace.qml")));item=component.createWithInitialProperties(dict(bridge=self.bridge,colors=colors))
            self.assertIsNotNone(item,"\n".join(error.toString() for error in component.errors()))
            parameter_component=QQmlComponent(engine,QUrl.fromLocalFile(str(qml/"McdPeakParameters.qml")));pane=parameter_component.createWithInitialProperties(dict(bridge=self.bridge,colors=colors))
            self.assertIsNotNone(pane,"\n".join(error.toString() for error in parameter_component.errors()))
            window=QQuickWindow();window.resize(1440,780);pane.setParentItem(window.contentItem());pane.setWidth(240);pane.setHeight(780);item.setParentItem(window.contentItem());item.setX(240);item.setWidth(1200);item.setHeight(780);window.show()
            surface=item.findChild(QObject,"mcd_peak_plot_surface");self.assertIsNotNone(surface)
            def plot_ready(axes):
                canvas=self.bridge._canvas
                size=(max(1,round(surface.width()*window.devicePixelRatio())),max(1,round(surface.height()*window.devicePixelRatio())))
                return canvas is not None and not surface._size_pending and not surface._size_timer.isActive() and canvas.get_width_height(physical=True)==size and not self.bridge._plot_timer.isActive() and not canvas.backend._timer.isActive() and set(self.bridge._axes)==set(axes)
            def wait_plot(axes):
                # Quick polish applies the viewport size before the queued Agg render.
                deadline=time.monotonic()+10
                while not plot_ready(axes) and time.monotonic()<deadline:settle()
                self.assertTrue(plot_ready(axes),f"Plot did not settle: expected {axes}, actual {tuple(self.bridge._axes)}")
            wait_plot({"spectra","derivative","result"})
            self.assertEqual(set(self.bridge._axes),{"spectra","derivative","result"});self.assertFalse(window.grabWindow().isNull())
            capture=os.environ.get("DPTK_MCD_PEAK_CAPTURE")
            if capture:window.grabWindow().save(capture+"-wide.png")
            pane.setWidth(220);pane.setHeight(600);item.setX(220);item.setWidth(500);item.setHeight(600);window.resize(720,600);wait_plot({"spectra","derivative"})
            self.assertTrue(self.bridge.snapshot["compactInspection"])
            self.bridge.setParameter("peakPanel","maps");wait_plot({"map","map-neg","spectra"});self.assertEqual(set(self.bridge._axes),{"map","map-neg","spectra"})
            if capture:window.grabWindow().save(capture+"-compact-maps.png")
            self.bridge.setParameter("dividerY",self.bridge.snapshot["dividerY"]-.06);wait_plot({"map","map-neg","spectra"});self.assertLess(self.bridge._values["heatmapFraction"],.7)
            self.bridge.invoke("toggle-spectrum");wait_plot({"map","map-neg"});self.assertNotIn("spectra",self.bridge._axes)
            self.bridge.invoke("toggle-spectrum");self.bridge.setParameter("peakPanel","result");wait_plot({"result"});self.assertEqual(set(self.bridge._axes),{"result"})
            if capture:window.grabWindow().save(capture+"-compact-result.png")
            self.bridge.invoke("peak-table");settle(60);self.assertTrue(self.bridge.snapshot["peakRows"]);self.bridge.invoke("close-sheet")
            self.assertFalse(warnings,"\n".join(warnings));self.assertNotIn("PySide6.QtWidgets",sys.modules);self.assertNotIn("matplotlib.backends.backend_qtagg",sys.modules)
            window.close();item.setParentItem(None);pane.setParentItem(None);item.deleteLater();pane.deleteLater();settle()
    suite=unittest.TestSuite([Cases("test_native_wide_compact_maps_splitter_and_tables_without_widgets")]) if "--qml-only" in sys.argv else unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__=="__main__":
    if "--isolated" in sys.argv:raise SystemExit(_isolated())
    unittest.main()
