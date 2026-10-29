"""Native main PL/DRR parity, source catalog and scientific worker tests."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class NativePlTests(unittest.TestCase):
    def test_native_pl_and_catalog(self):
        with TemporaryDirectory(prefix="quick-pl-process-") as folder:
            env=dict(os.environ,QT_QPA_PLATFORM="offscreen",QT_QUICK_BACKEND="software",MPLBACKEND="agg",MPLCONFIGDIR=folder,
                XDG_CONFIG_HOME=folder,XDG_CACHE_HOME=folder,LOCALAPPDATA=folder,APPDATA=folder,PYTHONIOENCODING="utf-8")
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--isolated"],cwd=Path(__file__).resolve().parents[1],
                env=env,capture_output=True,text=True,timeout=240)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn("QUICK_PL_NATIVE_TESTS=",result.stdout)
        print(result.stdout.strip())


def isolated():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import importlib.abc,json,time,traceback,threading
    from copy import deepcopy
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import patch
    class NoWidgets(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname in {"PySide6.QtWidgets","PySide6.QtQuickWidgets","ui_qt.common","ui_qt.main_window","matplotlib.backends.backend_qtagg","matplotlib.backends.backend_qt"}:
                raise AssertionError("Native PL imported Widgets: "+fullname)
    sys.meta_path.insert(0,NoWidgets())
    from PySide6.QtCore import QEvent,QMetaObject,QObject,QSettings,QUrl,qInstallMessageHandler
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from ui_qt.quick.application_bridge import QuickApplicationBridge
    if "core.loader" in sys.modules:raise AssertionError("PL extension made startup import the scientific loader")
    QQuickStyle.setStyle("Basic");app=QGuiApplication([])
    errors,qml=[] ,[]
    def uncaught(kind,value,tb):errors.append("".join(traceback.format_exception(kind,value,tb)));traceback.print_exception(kind,value,tb)
    sys.excepthook=uncaught
    def message(kind,context,text):
        if any(part in text for part in ("ReferenceError","TypeError","Cannot assign","Unable to assign","Binding loop","Error:")):qml.append(text);print(text,file=sys.stderr)
    previous=qInstallMessageHandler(message)
    import numpy as np
    from core.loader import DataCube
    from ui_qt.quick import pl_workflow
    def wait(predicate,timeout=40):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            app.processEvents()
            if predicate():app.processEvents();return
            time.sleep(.005)
        raise AssertionError("Timed out waiting for native PL/catalog state")
    class Cases(unittest.TestCase):
        def setUp(self):
            e,q=len(errors),len(qml)
            self.addCleanup(lambda:self.assertEqual(errors[e:],[]))
            self.addCleanup(lambda:self.assertEqual(qml[q:],[]))
            self.tmp=TemporaryDirectory(prefix="quick-pl-case-");self.addCleanup(self.tmp.cleanup)
            self.root=Path(self.tmp.name).resolve()
            self.bridge=QuickApplicationBridge(settings=QSettings(str(self.root/"settings.ini"),QSettings.IniFormat))
            self.bridge.setParameter("pl.autoNext",False)
            self.addCleanup(self.bridge.close)
        def idle(self):
            wait(lambda:not self.bridge._pending and not self.bridge._busy and not self.bridge._plot_timer.isActive() and not self.bridge._manual_task.busy and not self.bridge._pl_range_timer.isActive())
            if self.bridge._canvas:wait(lambda:not self.bridge._canvas.backend._timer.isActive())
        def write_dat(self,name="sample.dat",offset=0.):
            x=np.linspace(1.7,1.9,201)+offset;y=np.array([-1.,0.,1.])
            rows=[]
            for gate in y:
                center=1.77+offset+gate*.002
                rows.append(1+3*.004**2/((x-center)**2+.004**2)+2*.006**2/((x-(1.84+offset))**2+.006**2))
            z=np.asarray(rows);path=self.root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text("X\t"+"\t".join(map(str,y))+"\n"+"\n".join("\t".join(map(str,[energy,*z[:,index]])) for index,energy in enumerate(x))+"\n")
            return path,DataCube(x,y,z,"Y",name,"PL")
        def csv(self,name="sample_PL_TG.csv",gain=1.,tg_offset=0.):
            wave=np.linspace(700,750,101);bg=np.arange(5.);tg=bg*2+tg_offset
            spectra=1+gain*(1+bg[:,None]*.1)*np.exp(-((wave[None,:]-725)/3)**2)
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
            np.savetxt(path,np.column_stack((bg,tg,spectra)),delimiter=",",header="Vbg,Vtg,"+",".join(map(str,wave)),comments="")
            return path
        def load(self,path=None):
            path=path or self.write_dat()[0]
            self.bridge.openFile(QUrl.fromLocalFile(str(path)));self.idle()
            self.assertIsNotNone(self.bridge._cube,self.bridge.snapshot["status"])
            self.assertTrue(self.bridge._maps)
            return path
        def test_raw_axis_dat_label_and_loaded_hash_guard(self):
            path=self.csv()
            self.load(path)
            np.testing.assert_allclose(self.bridge._cube.gate,np.arange(5.)*2)
            self.bridge.setParameter("pl.yAxis","BG");self.idle()
            np.testing.assert_allclose(self.bridge._cube.gate,np.arange(5.))
            self.bridge.setParameter("pl.yAxis","Advanced…")
            self.bridge.setParameter("pl.a",2.)
            self.bridge.setParameter("pl.b",3.)
            self.bridge.setParameter("pl.c",.5);self.idle()
            np.testing.assert_allclose(self.bridge._cube.gate,np.arange(5.)*7+.5)
            self.assertTrue(self.bridge._pl_source_hashes())
            before=self.bridge._cube.Z.copy()
            path.write_text(path.read_text().replace("Vbg","Vbg ",1))
            operation=self.bridge._pl_export_request()
            with self.assertRaisesRegex(ValueError,"changed"):operation()
            np.testing.assert_array_equal(self.bridge._cube.Z,before)
            dat=self.write_dat()[0];self.load(dat)
            self.bridge.setParameter("pl.datChoice","Custom")
            self.bridge.setParameter("pl.datLabel","Carrier density")
            self.bridge.setParameter("pl.datUnit","cm^-2");self.idle()
            self.assertEqual((self.bridge._cube.gate_label,self.bridge._cube.gate_unit),("Carrier density","cm^-2"))
        def test_independent_banks_split_roi_and_fixed_axes(self):
            first=self.load()
            self.bridge.setParameter("vmin",.9)
            self.bridge.setParameter("vmax",5.)
            self.bridge.setParameter("pl.bank.split",True)
            self.bridge.setParameter("pl.bank.splitX",1.8)
            self.bridge.setParameter("pl.bank.leftMin",.8)
            self.bridge.setParameter("pl.bank.leftMax",5.5);self.idle()
            self.bridge.setParameter("pl.fixed.leftMax",True)
            linear=deepcopy(self.bridge._pl_banks["linear"])
            self.bridge.setParameter("scale","log");self.idle()
            self.bridge.setParameter("vmin",1.)
            self.bridge.setParameter("vmax",4.)
            self.bridge.setParameter("pl.bank.split",True);self.idle()
            self.assertEqual(self.bridge._pl_banks["linear"],linear)
            self.bridge.setParameter("scale","linear");self.idle()
            self.assertEqual(self.bridge._params("linear").split_scale.left_vmax,5.5)
            self.bridge.setParameter("xmin",1.71)
            self.bridge.setParameter("xmax",1.79);self.idle()
            self.assertEqual(self.bridge._params("linear").vmax,5.5)
            self.bridge.setParameter("pl.fixed.xmin",True)
            self.bridge.setParameter("pl.fixed.xmax",True)
            second=self.write_dat("second.dat",offset=.01)[0]
            self.load(second)
            self.assertEqual((self.bridge._values["xmin"],self.bridge._values["xmax"]),(1.71,1.79))
            self.bridge.setParameter("sideBySide",True);self.idle()
            self.assertFalse(self.bridge.snapshot["railVisible"])
            self.bridge.setParameter("sideBySide",False);self.idle()
            self.assertTrue(self.bridge.snapshot["railVisible"])
            original=self.bridge._values["xmin"]
            self.bridge.setParameter("xmin",99.)
            self.assertEqual(self.bridge._values["xmin"],original)
            cmap=self.bridge._values["cmap"]
            self.bridge.setParameter("cmap","does-not-exist")
            self.assertEqual(self.bridge._values["cmap"],cmap)
        def test_manual_peaks_lorentz_removal_and_retained_overlays(self):
            self.load();self.bridge.invoke("spectrum-find");self.idle()
            state=self.bridge._manual_states["pl"]
            self.assertEqual(len(state["indices"]),2)
            self.bridge.invoke("spectrum-fit");self.idle()
            self.assertIsNotNone(state["fit"],state["status"])
            np.testing.assert_allclose(state["fit"]["centers"],[1.77,1.84],atol=1e-5)
            artists=tuple(id(artist) for item in self.bridge._manual_artists for artist in item[3:])
            spectrum=self.bridge._spectra[0][0]
            self.assertTrue(self.bridge._pl_plot_click(SimpleNamespace(button=1,inaxes=spectrum,xdata=1.77,ydata=3.)))
            self.assertEqual(len(state["indices"]),1)
            self.assertEqual(artists,tuple(id(artist) for item in self.bridge._manual_artists for artist in item[3:]))
            self.bridge.setParameter("gate",1.);self.idle()
            self.assertEqual(len(self.bridge._manual_artists[0][3].get_offsets()),0)
            self.bridge.setParameter("spectrum.mode","dips")
            self.bridge.invoke("spectrum-find");self.idle()
            self.assertGreaterEqual(len(state["indices"]),1)
        def test_reload_same_source_refreshes_only_automatic_limits(self):
            path=self.csv(gain=1.)
            self.load(path)
            original=self.bridge._pl_banks["linear"]["vmax"]
            self.bridge.setParameter("vmin",.5)
            self.bridge.setParameter("pl.fixed.vmin",True);self.idle()
            self.csv(gain=10.)
            self.load(path)
            bank=self.bridge._pl_banks["linear"]
            self.assertEqual(bank["vmin"],.5)
            self.assertGreater(bank["vmax"],original*2)
            self.assertTrue(bank["fixed"]["vmin"])
            self.assertFalse(bank["fixed"].get("vmax",False))
        def assert_y_view(self,expected_bounds,expected_gate):
            self.assertEqual((self.bridge._values["ymin"],self.bridge._values["ymax"]),expected_bounds)
            self.assertEqual(self.bridge._values["gate"],expected_gate)
            for scale in ("linear","log"):
                self.assertEqual(self.bridge._params(scale).ylim,expected_bounds)
            row=int(np.argmin(abs(self.bridge._cube.gate-expected_gate)))
            for (axis,gate_line),(_,spectrum) in zip(self.bridge._maps,self.bridge._spectra):
                np.testing.assert_allclose(axis.get_ylim(),expected_bounds)
                np.testing.assert_allclose(gate_line.get_ydata(),[expected_gate,expected_gate])
                np.testing.assert_allclose(spectrum.get_ydata(),self.bridge._cube.Z[row])
        def test_changed_y_coordinates_refresh_view_and_spectrum(self):
            self.load(self.csv(tg_offset=100.))
            self.assert_y_view((100.,108.),100.)
            self.bridge.setParameter("gate",104.)
            xmin,xmax=self.bridge._values["xmin"]+.01,self.bridge._values["xmax"]-.01
            self.bridge.setParameter("xmin",xmin);self.bridge.setParameter("xmax",xmax)
            self.bridge.setParameter("vmin",.5);self.bridge.setParameter("vmax",5.)
            self.bridge.setParameter("pl.fixed.vmin",True);self.bridge.setParameter("pl.fixed.vmax",True)
            self.bridge.setParameter("scale","log")
            self.bridge.setParameter("vmin",.25);self.bridge.setParameter("vmax",8.)
            self.bridge.setParameter("pl.fixed.vmin",True);self.bridge.setParameter("pl.fixed.vmax",True)
            self.bridge.setParameter("pl.bank.split",True)
            self.bridge.setParameter("sideBySide",True)
            self.bridge.setParameter("cmap","plasma");self.idle()
            self.bridge.setParameter("pl.yAxis","BG");self.idle()
            np.testing.assert_allclose(self.bridge._cube.gate,np.arange(5.))
            self.assert_y_view((0.,4.),0.)
            self.assertEqual((self.bridge._values["xmin"],self.bridge._values["xmax"]),(xmin,xmax))
            self.assertEqual((self.bridge._values["scale"],self.bridge._values["sideBySide"],self.bridge._values["cmap"]),("log",True,"plasma"))
            self.assertEqual((self.bridge._pl_banks["linear"]["vmin"],self.bridge._pl_banks["linear"]["vmax"]),(.5,5.))
            self.assertEqual((self.bridge._pl_banks["log"]["vmin"],self.bridge._pl_banks["log"]["vmax"]),(.25,8.))
            self.assertTrue(self.bridge._pl_banks["log"]["split"])
            self.assertFalse(self.bridge._pl_axis_fixed["ymin"])
            self.assertFalse(self.bridge._pl_axis_fixed["ymax"])
        def test_advanced_coefficient_changes_refresh_y_view(self):
            self.load(self.csv(tg_offset=100.))
            self.bridge.setParameter("pl.yAxis","Advanced…");self.idle()
            self.assertEqual(self.bridge._pl_values["yAxis"],"Advanced…")
            self.assert_y_view((100.,104.),100.)
            self.bridge.setParameter("ymin",100.5);self.bridge.setParameter("ymax",103.5);self.idle()
            for key,value,bounds,gate in (("a",2.,(200.,212.),200.),("b",3.,(200.,228.),200.),("c",-250.,(-50.,-22.),-22.)):
                with self.subTest(coefficient=key):
                    self.bridge.setParameter("pl."+key,value);self.idle()
                    self.assert_y_view(bounds,gate)
        def test_changed_y_derivation_does_not_reuse_incompatible_fixed_bounds(self):
            path=self.csv(tg_offset=100.)
            for fixed in (("ymin",),("ymax",),("ymin","ymax")):
                with self.subTest(fixed=fixed):
                    self.bridge.setParameter("pl.yAxis","TG");self.idle();self.load(path)
                    for field in ("ymin","ymax"):self.bridge.setParameter("pl.fixed."+field,False)
                    self.bridge.setParameter("ymin",-10.);self.bridge.setParameter("ymax",120.)
                    for field in fixed:self.bridge.setParameter("pl.fixed."+field,True)
                    self.bridge.setParameter("pl.yAxis","BG");self.idle()
                    self.assert_y_view((0.,4.),0.)
                    self.assertFalse(any(self.bridge._pl_axis_fixed[field] for field in ("ymin","ymax")))
        def test_unchanged_coordinates_restore_each_file_view(self):
            first=self.csv("first_PL_TG.csv",tg_offset=100.)
            second=self.csv("second_PL_TG.csv",tg_offset=100.)
            self.load(first)
            self.bridge.setParameter("ymin",102.);self.bridge.setParameter("ymax",106.)
            self.bridge.setParameter("gate",104.);self.idle()
            self.load(second);self.assert_y_view((100.,108.),100.)
            self.load(first);self.assert_y_view((102.,106.),104.)
            self.csv("first_PL_TG.csv",gain=10.,tg_offset=100.)
            self.load(first);self.assert_y_view((102.,106.),104.)
        def test_stale_manual_fit_is_discarded(self):
            self.load();self.bridge.invoke("spectrum-find")
            entered,release=threading.Event(),threading.Event();original=pl_workflow.fit_spectrum
            def slow(*args,**kwargs):entered.set();release.wait(10);return original(*args,**kwargs)
            with patch.object(pl_workflow,"fit_spectrum",side_effect=slow):
                self.bridge.invoke("spectrum-fit");wait(entered.is_set)
                self.bridge.setParameter("gate",1.)
                release.set();self.idle()
            self.assertIsNone(self.bridge._manual_states["pl"]["fit"])
        def test_close_pending_manual_fit_joins_before_qobject_destruction(self):
            from shiboken6 import delete
            self.load();self.bridge.invoke("spectrum-find")
            entered,exited=threading.Event(),threading.Event();published=[]
            task=self.bridge._manual_task
            task.completed.connect(lambda *values:published.append(values))
            def pending(*args,cancellation=None,**kwargs):
                entered.set()
                try:
                    while not cancellation.cancelled:time.sleep(.002)
                    cancellation.check()
                finally:exited.set()
            with patch.object(pl_workflow,"fit_spectrum",side_effect=pending):
                self.bridge.invoke("spectrum-fit");wait(entered.is_set)
                self.bridge.close()
                self.assertTrue(exited.is_set())
                self.assertTrue(task.future.done())
                self.assertIsNone(task.future.exception())
                self.assertEqual(published,[])
                delete(task);app.processEvents()
            self.assertEqual(published,[])
            print("MANUAL_CLOSE_LIFECYCLE="+json.dumps(dict(joined=True,published=0,deletedQObjectSafe=True)))
        def test_close_pending_catalog_releases_cache_handles_before_return(self):
            from shiboken6 import delete
            from core.source_catalog_cache import SourceCatalogCache
            from ui_qt.quick.source_catalog import SourceCatalogBridge
            self.csv()
            catalog=SourceCatalogBridge(settings=QSettings(str(self.root/"catalog-settings.ini"),QSettings.IniFormat))
            entered,exited=threading.Event(),threading.Event();published=[];paths=[]
            catalog.operationCompleted.connect(lambda *values:published.append(values))
            original=SourceCatalogCache._write
            def pending(cache,*args,**kwargs):
                original(cache,*args,**kwargs)
                paths.append(cache.path)
                with cache.path.open("r",encoding="utf-8"):
                    entered.set()
                    while not catalog._cancel.cancelled:time.sleep(.002)
                exited.set()
            with patch.object(SourceCatalogCache,"_write",pending):
                catalog.open("pl",str(self.root),[],False);wait(entered.is_set)
                catalog.close()
                self.assertTrue(exited.is_set())
                self.assertTrue(catalog._future.done())
                self.assertIsNone(catalog._future.exception())
                self.assertEqual(published,[])
                delete(catalog);app.processEvents()
            paths[0].unlink()
            self.assertFalse(paths[0].exists())
            self.assertEqual(published,[])
            print("CATALOG_CLOSE_LIFECYCLE="+json.dumps(dict(joined=True,published=0,cacheHandleReleased=True,deletedQObjectSafe=True)))
        def test_main_drr_manual_primary_and_derivative_analysis(self):
            path=self.write_dat("reflectance.dat")[0]
            self.bridge.selectWorkflow("drr")
            self.load(path)
            self.bridge.invoke("spectrum-find");self.bridge.invoke("spectrum-fit");self.idle()
            state=self.bridge._manual_states["drr"]
            self.assertIsNotNone(state["fit"],state["status"])
            np.testing.assert_allclose(state["fit"]["centers"],[1.77,1.84],atol=1e-5)
            self.bridge.setParameter("drrView","pair");self.idle()
            self.bridge.setParameter("spectrum.product","second")
            self.bridge.invoke("spectrum-find");self.idle()
            self.assertIs(state["cube"],self.bridge._drr_products.second)
            self.assertGreater(len(state["indices"]),0)
            self.assertTrue(any(len(artists[3].get_offsets()) for artists in self.bridge._manual_artists))
        def test_pan_callbacks_settle_and_keep_automatic_scales_responsive(self):
            self.load()
            self.bridge.setParameter("sideBySide",True);self.idle()
            canvas=self.bridge._canvas;start=canvas.draw_count
            axis=self.bridge._maps[0][0]
            axis.set_xlim(1.72,1.88)
            self.idle()
            self.assertEqual((self.bridge._values["xmin"],self.bridge._values["xmax"]),(1.72,1.88))
            stable=canvas.draw_count
            deadline=time.monotonic()+.5
            while time.monotonic()<deadline:app.processEvents();time.sleep(.005)
            self.assertEqual(canvas.draw_count,stable)
            self.assertLess(stable-start,4)
        def test_pl_catalog_filters_history_saved_and_selection(self):
            one=self.csv("first_PL_TG.csv");two=self.csv("second_PL_TG.csv");ref=self.csv("reference_REF_TG.csv")
            self.write_dat("Processed Data/PL/saved.dat")
            folder=self.root/"Processed Data/PL"
            (folder/"status.metadata.json").write_text(json.dumps(dict(operation="PL",created_utc="2026-10-24T12:00:00Z",sources=[dict(role="measurement",source_relative_path=one.name)])))
            catalog=self.bridge._source_catalog
            catalog.open("pl",str(self.root),[],False);wait(lambda:not catalog.snapshot["busy"])
            self.assertEqual(catalog.filesModel.rowCount(),2)
            catalog.setParameter("statusFilter","unprocessed")
            self.assertEqual([row["id"] for row in catalog.filesModel.rows],[str(two)])
            catalog.setParameter("type","REF");catalog.setParameter("statusFilter","all")
            self.assertEqual(catalog.filesModel.rows[0]["id"],str(ref))
            catalog.setParameter("savedOnly",True)
            self.assertEqual(catalog.filesModel.rowCount(),1)
            catalog.selectFile(str(two),True)
            chosen=list(catalog.snapshot["selected"])
            catalog.refreshFolder(str(self.root));wait(lambda:not catalog.snapshot["busy"])
            self.assertEqual(catalog.snapshot["selected"],chosen)
            generation=catalog._generation
            catalog.refreshFolder(str(self.root/"unrelated"))
            self.assertEqual(catalog._generation,generation)
        def test_drr_catalog_group_compatible_selection_and_history(self):
            one=self.csv("YZ365_p5n2_1T_1.67KREF_720nmc_Rot49deg_TG=18_rep01.csv")
            two=self.csv("YZ365_p5n2_1T_1.67KREF_720nmc_Rot49deg_TG=18_rep02.csv")
            catalog=self.bridge._source_catalog
            catalog.open("drr",str(self.root),[],False);catalog.setParameter("onlyNew",False)
            wait(lambda:not catalog.snapshot["busy"])
            self.assertGreaterEqual(catalog.filesModel.rowCount(),2)
            self.assertTrue(catalog.groupsModel.rows)
            catalog.selectFile(str(one),True);catalog.invoke("compatible")
            self.assertIn(str(two),catalog.snapshot["selected"])
            accepted=[];catalog.selectionAccepted.connect(lambda workflow,files:accepted.append((workflow,files)))
            catalog.invoke("accept");self.assertEqual(accepted[0][0],"drr")
            self.assertEqual(set(accepted[0][1]),{str(one),str(two)})
        def test_cached_catalog_force_rebuild_and_acquisition_watcher(self):
            from ui_qt.quick.source_catalog import scan_catalog
            from core import pl_sources
            self.csv("one_PL_TG.csv")
            cache=self.root/"isolated-cache"/"details.json"
            original=pl_sources.list_pl_source_files
            with patch.object(pl_sources,"list_pl_source_files",wraps=original) as scan:
                scan_catalog(self.root,"pl",cache_path=cache)
                scan_catalog(self.root,"pl",cache_path=cache)
                self.assertEqual(scan.call_count,1)
                scan_catalog(self.root,"pl",cache_path=cache,force=True)
                self.assertEqual(scan.call_count,2)
            catalog=self.bridge._source_catalog
            catalog.open("pl",str(self.root),[],False);wait(lambda:not catalog.snapshot["busy"])
            added=self.csv("two_PL_TG.csv")
            wait(lambda:any(row["id"]==str(added) for row in catalog.filesModel.rows))
        def test_separate_exports_and_auto_next_first_save_only(self):
            one=self.csv("first_PL_TG.csv");two=self.csv("second_PL_TG.csv")
            self.load(one)
            self.bridge.setParameter("pl.autoNext",True)
            operation=self.bridge._pl_export_request();paths=operation()
            self.assertEqual(set(paths),{"png_linear","png_log","dat"})
            self.assertFalse(self.bridge._pl_export_context["wasProcessed"])
            self.bridge._pl_export_completed(paths)
            wait(lambda:self.bridge._source==two);self.idle()
            self.load(one)
            again=self.bridge._pl_export_request();again()
            self.assertTrue(self.bridge._pl_export_context["wasProcessed"])
        def test_pure_kernels_preserve_gaps_and_fit_known_parameters(self):
            x=np.linspace(1,1.1,501)
            y=pl_workflow.multi_lorentz(x,1.,.2,2.,1.03,.002,-1.,1.075,.003)
            settings=dict(mode="peaks",prominence=.05,distance=5,maximum=6,fitCount=2)
            peaks=pl_workflow.find_spectrum_peaks(x,y,(1,1.1),settings)
            dips=pl_workflow.find_spectrum_peaks(x,y,(1,1.1),{**settings,"mode":"dips"})
            fit=pl_workflow.fit_spectrum(x,y,(1,1.1),settings,tuple(np.r_[peaks,dips]))
            np.testing.assert_allclose(fit["centers"],[1.03,1.075],atol=2e-5)
            broken=y.copy();broken[149:152]=np.nan
            self.assertNotIn(150,pl_workflow.find_spectrum_peaks(x,broken,(1,1.1),settings))
        def test_native_qml_source_picker_and_manual_controls(self):
            from ui_qt.quick.plot_surface import register_plot_surface
            register_plot_surface();self.load()
            engine=QQmlApplicationEngine();engine.setInitialProperties({"appBridge":self.bridge})
            engine.load(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/"ui_qt/quick/qml/Application.qml")))
            self.assertTrue(engine.rootObjects());window=engine.rootObjects()[0]
            self.bridge.invoke("select-files")
            wait(lambda:window.findChild(QObject,"quick_source_catalog") is not None)
            picker=window.findChild(QObject,"quick_source_catalog")
            wait(lambda:picker.property("visible"))
            self.assertTrue(QMetaObject.invokeMethod(picker,"reject"));app.processEvents()
            self.bridge.invoke("spectrum-find");self.bridge.invoke("spectrum-fit");self.idle()
            self.assertIn("manualSpectrum",[group["key"] for group in self.bridge.snapshot["parameterGroups"]])
            window.close();engine.deleteLater();app.sendPostedEvents(None,QEvent.DeferredDelete);app.processEvents()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    qInstallMessageHandler(previous)
    print("QUICK_PL_NATIVE_TESTS="+json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),widgetsImported="PySide6.QtWidgets" in sys.modules)))
    return 0 if result.wasSuccessful() else 1


if __name__=="__main__":
    raise SystemExit(isolated()) if "--isolated" in sys.argv else unittest.main()
