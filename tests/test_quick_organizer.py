"""Full native MCD Organizer, isolated from user settings and real data."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickOrganizerTests(unittest.TestCase):
    def test_native_organizer(self):
        with TemporaryDirectory() as temporary:
            env=dict(os.environ,QT_QPA_PLATFORM="offscreen",QT_QUICK_BACKEND="software",QT_SCALE_FACTOR="1",MPLBACKEND="agg",MPLCONFIGDIR=temporary,LOCALAPPDATA=temporary,XDG_CONFIG_HOME=temporary,XDG_CACHE_HOME=temporary,QML_DISK_CACHE_PATH=temporary)
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--isolated"],cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True,timeout=240)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


def _isolated():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from dataclasses import replace
    import json
    import time
    from threading import Event
    from unittest.mock import patch
    import numpy as np
    from PySide6.QtCore import QSettings,QUrl
    from PySide6.QtGui import QFont,QFontDatabase,QGuiApplication
    from PySide6.QtQml import QQmlComponent,QQmlEngine
    from core.mcd_extract import BRANCHES,discover_processed_mcd,organize_mcd_series
    from core.curie_weiss import fit_inverse_curie_weiss
    from ui_qt.quick.organizer_bridge import OrganizerBridge
    from ui_qt.quick.organizer_state import OrganizerSelections,merge_changes
    from ui_qt.quick import organizer_science as science
    from ui_qt.quick.plot_surface import register_plot_surface
    from tests.test_mcd_extract import McdExtractTests
    app=QGuiApplication([]);register_plot_surface()
    if sys.platform=="win32":QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf");app.setFont(QFont("Segoe UI",9))
    def settle(milliseconds=15):
        deadline=time.monotonic()+milliseconds/1000
        while time.monotonic()<deadline:app.processEvents();time.sleep(.001)
        app.processEvents()
    class Cases(unittest.TestCase):
        def setUp(self):
            self.temporary=TemporaryDirectory();self.addCleanup(self.temporary.cleanup);self.root=Path(self.temporary.name);writer=McdExtractTests()
            for field in (0.,1.):
                for index,t in enumerate((3.,4.,6.,9.,15.,25.)):
                    slope=2/(t+4);writer._write_result(self.root,f"f{int(field)}t{index}",energy=1.66,doping=6.3,efield=field,temperature=t,increasing_slope=slope,decreasing_slope=slope*1.1)
            self.settings=QSettings(str(self.root/"quick.ini"),QSettings.Format.IniFormat);self.bridge=OrganizerBridge(settings=self.settings);self.addCleanup(self.bridge.close)
            self.bridge.setParameter("compareVariable","Temperature");self.bridge.setExperimentFolder(str(self.root));self.wait()
        def wait(self):
            deadline=time.monotonic()+50
            while time.monotonic()<deadline:
                settle()
                state=self.bridge.snapshot
                pending=any(timer.isActive() for timer in (self.bridge._preview_timer,self.bridge._cw_timer))
                if not state["busy"] and not state["analysisBusy"] and not pending:break
            self.assertFalse(self.bridge.snapshot["busy"]);self.assertFalse(self.bridge.snapshot["analysisBusy"])
            self.assertFalse(self.bridge.snapshot["error"],self.bridge.snapshot["error"]);settle(35)
        def cw(self):self.bridge.setParameter("view","cw");self.wait();self.assertTrue(self.bridge._cw["results"])
        def test_catalog_and_series_match_core_with_every_control(self):
            expected=organize_mcd_series(discover_processed_mcd(self.root),"Temperature",include_singletons=False)
            self.assertEqual([series.series_id for series in self.bridge._series],[series.series_id for series in expected]);self.assertEqual(len(self.bridge._series),2)
            keys={field["key"] for group in self.bridge.snapshot["parameterGroups"] for field in group["fields"]}
            for key in ("compareVariable","newestOnly","energyFilter","groupTolerance","fixedGroupCount","groupCount","assign-selected","edit-groups","reset-groups","default-palette","low_minus_negative","refitSlopes","asymmetricRange","fieldHalfwidth","cwMethod","backgroundMode","temperatureMode","scan-ranges","visibleWeighted","thetaAllGroups","theta-bulk-apply","thetaCompareMethods","thetaReliableY","export-cw","export-methods","export-scan","export-theta"):self.assertIn(key,keys)
        def test_selection_group_and_palette_persist_and_restore(self):
            row=self.bridge.snapshot["conditions"][0];self.bridge.selectCondition(row["id"],False);self.bridge.setParameter("assignGroup","Group 7");self.bridge.invoke("assign-selected")
            self.bridge.includeCondition(row["id"],False);self.bridge.setParameter("palette","plasma");self.bridge.invoke("default-palette");self.wait();self.bridge._save()
            restored=OrganizerSelections(self.root);restored.bind(self.bridge._series);series=self.bridge._current();groups=restored.groups(series,self.bridge._series)
            self.assertEqual(groups[row["id"]],"Group 7");self.assertNotIn(row["id"],restored.included[series.series_id]);self.assertEqual(restored.palette,"plasma")
            self.bridge.invoke("restore-all");self.wait();self.assertTrue(all(item["included"] for item in self.bridge.snapshot["conditions"]))
        def test_two_windows_merge_independent_exclusions_and_manual_groups(self):
            self.bridge._save();first=OrganizerSelections(self.root);second=OrganizerSelections(self.root)
            for state in (first,second):state.bind(self.bridge._series);state.groups(self.bridge._current(),self.bridge._series)
            series=self.bridge._current();a,b=series.records[:2]
            first.assign(series,self.bridge._series,[a.record_id],"Group 7");first.included[series.series_id].discard(a.record_id);first.save(self.bridge._series)
            second.assign(series,self.bridge._series,[b.record_id],"Group 8");second.included[series.series_id].discard(b.record_id);second.save(self.bridge._series)
            restored=OrganizerSelections(self.root);restored.bind(self.bridge._series);groups=restored.groups(series,self.bridge._series)
            self.assertEqual((groups[a.record_id],groups[b.record_id]),("Group 7","Group 8"));self.assertFalse({a.record_id,b.record_id}&restored.included[series.series_id])
            self.assertEqual(merge_changes(["a","b","c"],["a","c"],["b","c"]),["c"])
        def test_fixed_group_count_does_not_merge_out_of_range_assignments(self):
            row=self.bridge.snapshot["conditions"][0];self.bridge.assignRecordGroup(row["id"],"Group 7");self.bridge.setParameter("groupCount",2);self.bridge.setParameter("fixedGroupCount",True)
            self.assertTrue(self.bridge._groups[row["id"]].startswith("Unassigned "));self.bridge.invoke("reset-groups");self.assertEqual(set(self.bridge._groups.values()),{"Group 1"})
            self.bridge.assignRecordGroup(row["id"],"");self.assertTrue(self.bridge._groups[row["id"]].startswith("Unassigned "))
        def test_condition_focus_reuses_curves_and_diagnostics_do_not_redraw(self):
            canvas=self.bridge._canvas;axes=tuple(canvas.figure.axes);curves=tuple(self.bridge._plot_hits)
            identifier=self.bridge._conditions[1]["id"];draws,blits=canvas.draw_count,canvas.blit_count
            self.bridge.selectCondition(identifier,False);self.wait()
            self.assertEqual(tuple(canvas.figure.axes),axes);self.assertEqual(tuple(self.bridge._plot_hits),curves)
            self.assertEqual(canvas.draw_count,draws);self.assertGreater(canvas.blit_count,blits)
            for line,record in self.bridge._plot_hits.items():self.assertEqual(line.get_linewidth(),2.1 if record.record_id==identifier else 1.)
            self.assertTrue(next(row for row in self.bridge._conditions if row["id"]==identifier)["focused"])
            for axis in axes:
                if axis.get_legend():
                    source=[line for line in curves if line.axes is axis]
                    self.assertEqual([line.get_linewidth() for line in axis.get_legend().get_lines()],[line.get_linewidth() for line in source])
            self.cw();self.bridge.setParameter("view","methods");self.wait();draws,blits=canvas.draw_count,canvas.blit_count
            self.bridge.selectCondition(self.bridge._conditions[2]["id"],False);self.wait()
            self.assertEqual(canvas.draw_count,draws);self.assertEqual(canvas.blit_count,blits)
        def test_latest_series_preview_paints_once_and_discards_old_completion(self):
            from ui_qt.quick import organizer_workflow
            original=organizer_workflow.load_traces;entered=Event();release=Event();calls=[]
            first=self.bridge._selected;second=next(series.series_id for series in self.bridge._series if series.series_id!=first)
            canvas=self.bridge._canvas;draws=canvas.draw_count
            def delayed(*args):
                calls.append(args[0])
                if len(calls)==1:entered.set();release.wait(10)
                return original(*args)
            with patch.object(organizer_workflow,"load_traces",side_effect=delayed):
                self.bridge.selectSeries(second)
                self.assertTrue(self.bridge.snapshot["previewPending"]);self.assertEqual(self.bridge._plot_hits,{})
                self.assertFalse(self.bridge._plot_timer.isActive())
                deadline=time.monotonic()+5
                while not entered.is_set() and time.monotonic()<deadline:settle()
                self.assertTrue(entered.is_set());self.assertEqual(canvas.draw_count,draws)
                self.bridge.selectSeries(first);self.wait()
                arrays=self.bridge._arrays
                self.assertEqual(set(arrays),{record.record_id for record in self.bridge._members()})
                self.assertEqual(canvas.draw_count,draws+1)
                self.assertFalse(self.bridge.snapshot["previewPending"])
                release.set();settle(100)
                self.assertIs(self.bridge._arrays,arrays);self.assertEqual(canvas.draw_count,draws+1)
        def test_failed_and_cancelled_preview_clear_stale_series_plot(self):
            from ui_qt.quick import organizer_workflow
            first=self.bridge._selected;second=next(series.series_id for series in self.bridge._series if series.series_id!=first)
            with patch.object(organizer_workflow,"load_traces",side_effect=ValueError("synthetic missing traces")):
                self.bridge.selectSeries(second);deadline=time.monotonic()+5
                while (self.bridge.snapshot["previewPending"] or self.bridge._plot_timer.isActive()) and time.monotonic()<deadline:settle()
                settle(80)
            self.assertFalse(self.bridge.snapshot["previewPending"]);self.assertIn("synthetic missing traces",self.bridge.snapshot["error"])
            self.assertEqual(self.bridge._arrays,{});self.assertEqual(self.bridge._plot_hits,{})
            self.bridge.selectSeries(first);self.wait();self.assertTrue(self.bridge._arrays)
            entered=Event();release=Event();original=organizer_workflow.load_traces
            def delayed(*args):entered.set();release.wait(10);return original(*args)
            with patch.object(organizer_workflow,"load_traces",side_effect=delayed):
                self.bridge.selectSeries(second);deadline=time.monotonic()+5
                while not entered.is_set() and time.monotonic()<deadline:settle()
                self.assertTrue(entered.is_set());self.bridge.invoke("cancel");release.set();settle(100)
            self.assertFalse(self.bridge.snapshot["previewPending"])
            self.assertEqual(self.bridge._arrays,{});self.assertEqual(self.bridge._plot_hits,{})
            self.assertIn("cancelled",self.bridge.snapshot["status"])
        def test_queued_preview_uses_latest_branch_and_member_selection(self):
            second=next(series.series_id for series in self.bridge._series if series.series_id!=self.bridge._selected)
            draws=self.bridge._canvas.draw_count
            self.bridge.selectSeries(second)
            identifier=self.bridge._conditions[0]["id"]
            self.bridge.setParameter("showIncreasing",False)
            self.bridge.includeCondition(identifier,False)
            self.assertTrue(self.bridge.snapshot["previewPending"])
            self.wait()
            self.assertEqual(self.bridge._canvas.draw_count,draws+1)
            self.assertNotIn(identifier,self.bridge._arrays)
            self.assertEqual(set(self.bridge._arrays),{record.record_id for record in self.bridge._members()})
            self.assertTrue(all(line.get_marker()=="s" for line in self.bridge._plot_hits))
        def test_failed_selection_save_preserves_previous_file_and_reports_error(self):
            self.bridge._save();before=self.bridge._store.path.read_bytes();record=self.bridge._current().records[0]
            self.bridge._store.assign(self.bridge._current(),self.bridge._series,[record.record_id],"Group 8")
            with patch("ui_qt.quick.organizer_state.QSaveFile") as saved:
                saved.return_value.open.return_value=False;saved.return_value.errorString.return_value="Permission denied"
                self.assertFalse(self.bridge._save())
            self.assertEqual(self.bridge._store.path.read_bytes(),before);self.assertIn("save failed",self.bridge.snapshot["saveError"])
            self.assertTrue(self.bridge._save())
        def test_series_switch_ignores_previous_cw_worker_completion(self):
            self.cw();old=self.bridge._cw;entered=Event();release=Event();actual=science.fit_cw;calls=[]
            def slow(*args,**kwargs):
                calls.append(True)
                if len(calls)==1:entered.set();release.wait(5);return old
                return actual(*args,**kwargs)
            with patch("ui_qt.quick.organizer_science.fit_cw",slow):
                self.bridge.invoke("fit-cw");deadline=time.monotonic()+5
                while not entered.is_set() and time.monotonic()<deadline:settle()
                self.assertTrue(entered.is_set());target=next(series for series in self.bridge._series if series.series_id!=self.bridge._selected)
                self.bridge.selectSeries(target.series_id);release.set();self.wait()
                self.assertEqual(self.bridge._cw["series_id"],target.series_id)
                self.assertIsNot(self.bridge._cw,old)
        def test_saved_cw_fit_matches_core_and_missing_errors_stay_unavailable(self):
            self.cw();payload=self.bridge._cw
            for branch,result in payload["results"].items():
                points=sorted([point for point in payload["points"] if point["branch"]==branch],key=lambda point:point["temperature_k"])
                expected=fit_inverse_curie_weiss([point["temperature_k"] for point in points],[point["slope"] for point in points],slope_se=[point["slope_se"] for point in points])
                self.assertEqual(result,expected);self.assertAlmostEqual(result["theta_k"],-4.,places=7)
            weighted=[row for row in payload["method_comparison"] if row["method"]=="weighted"]
            self.assertTrue(weighted);self.assertTrue(all(row["status"]=="unavailable" for row in weighted))
        def test_failed_primary_cw_still_exports_complete_method_failure_diagnostics(self):
            self.bridge._all_records=[replace(record,increasing_slope_per_t=1.,decreasing_slope_per_t=1.) for record in self.bridge._all_records];self.bridge._regroup();self.cw_if_failed()
            self.assertFalse(self.bridge._cw["results"]);self.assertEqual(len(self.bridge._cw["method_comparison"]),6)
            self.bridge.invoke("export-methods");self.wait();metadata=json.loads(Path(next(path for path in self.bridge.snapshot["resultPaths"] if path.endswith(".json"))).read_text())
            self.assertEqual(len(metadata["rows"]),6);self.assertTrue(all(row["theta_k"] is None for row in metadata["rows"]))
        def cw_if_failed(self):self.bridge.setParameter("view","cw");self.wait()
        def test_field_refit_and_scan_preserve_requested_asymmetric_range(self):
            self.bridge.setParameter("refitSlopes",True);self.bridge.setParameter("asymmetricRange",True);self.bridge.setParameter("fieldMin",-.2);self.bridge.setParameter("fieldMax",.3);self.wait()
            self.assertTrue(self.bridge._cw["field_fits"]);self.assertTrue(all(fit["requested_b_min_t"]==-.2 and fit["requested_b_max_t"]==.3 for fit in self.bridge._cw["field_fits"]))
            self.bridge.invoke("scan-ranges");self.wait();self.assertTrue(self.bridge._scan["rows"])
            self.assertEqual((self.bridge._values["fieldMin"],self.bridge._values["fieldMax"]),(-.2,.3));self.assertTrue(any(row["status"]!="ok" for row in self.bridge._scan["rows"]))
        def test_completed_range_scan_is_invalidated_by_each_scan_input(self):
            for key,value in (("cwGroup",""),("temperatureMode","measured"),("temperatureMin",10.),("temperatureMax",20.),("fieldHalfwidth",.25)):
                with self.subTest(parameter=key):
                    original=self.bridge._values[key]
                    self.bridge.invoke("scan-ranges");self.wait();self.assertTrue(self.bridge._scan["rows"])
                    self.bridge.invoke("scan");self.assertTrue(self.bridge.snapshot["scanRows"])
                    self.bridge.setParameter(key,value)
                    self.assertIsNone(self.bridge._scan);self.assertEqual(self.bridge.snapshot["scanRows"],[])
                    self.bridge.setParameter(key,original);self.wait()
        def test_range_scan_preserves_results_when_display_controls_change(self):
            self.bridge.invoke("scan-ranges");self.wait();payload=self.bridge._scan
            for key,value in (("view","mcd"),("palette","plasma"),("visibleWeighted",False),("showConditions",False),("fieldRecord",self.bridge._members()[0].record_id)):
                with self.subTest(parameter=key):
                    self.bridge.setParameter(key,value);self.assertIs(self.bridge._scan,payload)
            self.wait()
        def test_range_scan_cannot_export_stale_rows_after_temperature_bound_changes(self):
            self.bridge.invoke("scan-ranges");self.wait()
            self.assertEqual({row["temperature_k"] for row in self.bridge._scan["rows"]},{3.,4.,6.,9.,15.,25.})
            self.bridge.setParameter("temperatureMin",10.);self.wait()
            with patch.object(science,"write_csv") as write_csv:
                self.bridge.invoke("export-scan")
                self.assertFalse(self.bridge.snapshot["busy"]);self.assertIn("Run the range scan",self.bridge.snapshot["error"])
                write_csv.assert_not_called()
            self.bridge.setParameter("view","cw");self.bridge.invoke("scan-ranges");self.wait()
            self.assertEqual({row["temperature_k"] for row in self.bridge._scan["rows"]},{15.,25.})
            self.bridge.invoke("export-scan");self.wait()
            paths=self.bridge.snapshot["resultPaths"]
            metadata=json.loads(Path(next(path for path in paths if Path(path).name=="scan.json")).read_text())
            self.assertEqual(metadata["parameters"]["temperatureMin"],10.)
            self.assertEqual({row["temperature_k"] for row in metadata["rows"]},{15.,25.})
        def test_range_scan_input_changes_discard_late_worker_completion(self):
            self.bridge.invoke("scan-ranges");self.wait();old=self.bridge._scan
            self.bridge.setParameter("view","mcd");self.wait()
            for key,value in (("cwGroup",""),("temperatureMode","measured"),("temperatureMin",10.),("temperatureMax",20.),("fieldHalfwidth",.25)):
                with self.subTest(parameter=key):
                    original=self.bridge._values[key];entered=Event();release=Event();returned=Event()
                    def delayed(*args):
                        entered.set();release.wait(5);returned.set();return old
                    with patch.object(science,"scan_ranges",side_effect=delayed):
                        self.bridge.invoke("scan-ranges");deadline=time.monotonic()+5
                        while not entered.is_set() and time.monotonic()<deadline:settle()
                        self.assertTrue(entered.is_set());token=self.bridge._tokens["scan"]
                        self.bridge.setParameter(key,value)
                        self.assertIsNone(self.bridge._scan);self.assertTrue(token.cancelled)
                        release.set();self.assertTrue(returned.wait(5));self.wait()
                        self.assertIsNone(self.bridge._scan);self.assertNotEqual(self.bridge._values["view"],"scan")
                    self.bridge.setParameter(key,original);self.wait()
        def test_batch_theta_uses_explicit_groups_and_actual_common_ranges(self):
            self.bridge.invoke("check-all");self.bridge.setParameter("thetaSensitivity",False);self.bridge.invoke("compare-theta");self.wait()
            rows=self.bridge._theta["rows"];self.assertEqual(len(rows),4);self.assertTrue(all(row["status"]=="ok" for row in rows));self.assertTrue(all(abs(row["theta_k"]+4)<1e-6 for row in rows))
            self.assertTrue(all(row["halfwidth_t"]==.2 for row in rows));self.assertEqual(len(self.bridge._theta_entries()),2)
            first=self.bridge._series[0];self.bridge._store.assign(first,self.bridge._series,[first.records[0].record_id],"Other resonance");self.bridge._theta_mapping.clear()
            with self.assertRaises(ValueError):self.bridge._theta_entries()
        def test_scientific_exports_include_cw_curves_and_theta_diagnostics(self):
            self.cw();self.bridge.invoke("export-cw");self.wait();paths=self.bridge.snapshot["resultPaths"]
            self.assertTrue(any(Path(path).name=="fit.json" for path in paths));metadata=json.loads(Path(next(path for path in paths if Path(path).name=="fit.json")).read_text())
            self.assertIn("mcd_curves_index",metadata);self.assertTrue(any(Path(path).name=="method_comparison.png" for path in paths))
            self.bridge.invoke("check-all");self.bridge.setParameter("thetaSensitivity",False);self.bridge.invoke("compare-theta");self.wait();self.bridge.invoke("export-theta");self.wait()
            self.assertTrue(any(Path(path).name=="comparison.json" for path in self.bridge.snapshot["resultPaths"]))
        def test_full_series_package_export_uses_exact_checked_conditions(self):
            included=self.bridge._members();self.bridge.includeCondition(included[0].record_id,False);self.wait();self.bridge.invoke("export");self.wait()
            paths=self.bridge.snapshot["resultPaths"];self.assertTrue(any(path.endswith(".xlsx") for path in paths));self.assertTrue(any(path.endswith(".json") for path in paths))
            self.assertTrue(any(path.endswith(".png") for path in paths))
        def test_real_standalone_qml_wide_compact_and_every_scientific_view(self):
            engine=QQmlEngine();warnings=[];engine.warnings.connect(lambda values:warnings.extend(value.toString() for value in values))
            path=Path(__file__).resolve().parents[1]/"ui_qt"/"quick"/"qml"/"Organizer.qml";component=QQmlComponent(engine,QUrl.fromLocalFile(str(path)))
            window=component.createWithInitialProperties(dict(organizerBridge=self.bridge));self.assertIsNotNone(window,"\n".join(error.toString() for error in component.errors()))
            window.resize(1440,900);window.show();settle(150);self.assertFalse(window.grabWindow().isNull())
            from tests.quick_selection_checks import assert_text_selection_palette
            for mode in ("light", "dark", "high-contrast"):
                self.bridge._appearance.setMode(mode);settle(30)
                assert_text_selection_palette(self,window,self.bridge.colors)
            self.bridge._appearance.setMode("light");settle(30)
            capture=os.environ.get("DPTK_ORGANIZER_CAPTURE")
            if capture:window.grabWindow().save(capture+"-series.png")
            self.bridge.setParameter("view","slopes");settle(120)
            if capture:window.grabWindow().save(capture+"-slopes.png")
            self.cw();settle(100)
            if capture:window.grabWindow().save(capture+"-cw.png")
            self.bridge.setParameter("view","methods");settle(100);self.bridge.invoke("theta");settle(50);self.bridge.invoke("close-sheet")
            window.resize(760,600);self.bridge.setParameter("showConditions",False);self.bridge.setParameter("view","mcd");settle(100)
            if capture:window.grabWindow().save(capture+"-compact.png")
            self.assertFalse(warnings,"\n".join(warnings));self.assertNotIn("PySide6.QtWidgets",sys.modules);self.assertNotIn("matplotlib.backends.backend_qtagg",sys.modules)
            window.close();settle()
            self.assertTrue(self.bridge._closed,"The QML close action must reach its exposed bridge slot")
            self.assertFalse(warnings,"\n".join(warnings))
            window.deleteLater();settle()
    suite=unittest.TestSuite([Cases("test_real_standalone_qml_wide_compact_and_every_scientific_view")]) if "--qml-only" in sys.argv else unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__=="__main__":
    if "--isolated" in sys.argv:raise SystemExit(_isolated())
    unittest.main()
