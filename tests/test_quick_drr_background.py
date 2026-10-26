"""Native background transactions and real Quick previews, in an isolated process."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickBackgroundTests(unittest.TestCase):
    def test_native_background_workspace(self):
        with TemporaryDirectory() as temporary:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software", QT_SCALE_FACTOR="1",
                       MPLBACKEND="agg", MPLCONFIGDIR=temporary, LOCALAPPDATA=temporary,
                       XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary, QML_DISK_CACHE_PATH=temporary)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import csv
    from dataclasses import replace
    import json
    from threading import Event
    import time
    from unittest.mock import patch
    import numpy as np
    from PySide6.QtCore import QPointF, QSettings, QUrl
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication
    from PySide6.QtQml import QQmlComponent, QQmlEngine
    from PySide6.QtQuick import QQuickItem, QQuickWindow
    from core.drr_composite_background import compose_background, load_composite_source, CompositeBackground, assign_region, save_composite_background, read_composite_background
    from core.drr_pn_store import DrrPnStore
    from core.drr_pn_pairing import DrrPnPair
    from core.drr_raw_preview import load_preview
    from ui_qt.quick.plot_surface import register_plot_surface
    from ui_qt.quick.drr_background_bridge import DrrBackgroundBridge
    from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft, prepare_application, commit_application, generate_pn, import_pn, parse_regions, load_with_recipe, background_input_signatures, compare_bn, prepare_fixed_application

    app = QGuiApplication([])
    if os.name == "nt":
        QFontDatabase.addApplicationFont(str(Path(os.environ["WINDIR"]) / "Fonts" / "segoeui.ttf"))
        app.setFont(QFont("Segoe UI", 9))
    register_plot_surface()
    colors = dict(surface="#f3f5f8", control="#ffffff", text="#20242b", muted="#606978", border="#c6cbd3",
        accent="#0067c0", hover="#edf1f6", pressed="#e0e7ef", error="#b42318", disabled="#939aa5",
        selectedText="#ffffff", inset="#e9edf2", divider="#dce1e8", borderHover="#aeb7c4", borderDisabled="#dce1e8",
        controlDisabled="#edf0f4", focus="#20242b", accentText="#0067c0", selected="#e6f0fa", selectedBorder="#b4d6fa", errorSurface="#fdf1f0")

    def settle(milliseconds=15):
        deadline = time.monotonic() + milliseconds/1000
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.001)
        app.processEvents()

    class Cases(unittest.TestCase):
        prefix = "YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_"

        def setUp(self):
            self.temporary = TemporaryDirectory()
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name)
            self.energy = np.linspace(1.45, 2.05, 301)
            self.base = np.exp(9 + .4*(self.energy-1.7) - .5*(self.energy-1.7)**2)
            self.sample = self.write("TG_rep01", 0)
            self.sample2 = self.write("TG_rep02", 1)
            self.p = self.write("Pbackground", -1)
            self.n = self.write("Nbackground", 1, constant=True)
            self.bn = self.write("BNbackground", 0, constant=True)
            self.context = BackgroundContext(str(self.root), (self.sample, self.sample2))
            self.settings = QSettings(str(self.root / "native.ini"), QSettings.Format.IniFormat)
            self.bridge = DrrBackgroundBridge(settings=self.settings)
            self.addCleanup(self.bridge.close)
            self.bridge.open(self.context)
            self.wait()

        def write(self, token, offset, *, constant=False):
            path = self.root / (self.prefix + token + ".csv")
            constant = constant or "background" in token.lower()
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Vbg", "Vtg", *self.energy])
                for gate in range(16):
                    value = offset*10 if constant else gate-8
                    intensity = self.base*(1 + .003*offset + (.02*np.exp(-((self.energy-1.72)/.012)**2)*(gate+1)/16 if not constant else 0))
                    writer.writerow([value, value, *intensity])
            return path.name

        def wait(self):
            deadline = time.monotonic()+30
            while self.bridge.snapshot["busy"] and time.monotonic() < deadline:
                settle()
            self.assertFalse(self.bridge.snapshot["busy"], "native background task timed out")
            self.assertFalse(self.bridge.snapshot["error"], self.bridge.snapshot["error"])
            settle()

        def test_self_commit_is_explicit_and_keeps_measurement_group(self):
            signals = []
            self.bridge.applyRequested.connect(signals.append)
            self.bridge.invoke("apply")
            self.wait()
            self.assertEqual(len(signals), 1)
            self.assertFalse(self.settings.contains("drr/bn_auto_compare"))
            recipe = commit_application(signals[0], current_context=self.context, settings=self.settings)
            self.assertEqual(recipe.mode, "Self (last frame)")
            self.assertEqual(tuple(item.measurement_file for item in recipe.assignments), self.context.measurements)
            self.assertEqual(recipe.files, ())

        def test_apply_rejects_changed_context_and_source_after_preparation(self):
            prepared = prepare_application(self.context, BackgroundDraft())
            other = replace(self.context, measurements=(self.sample,))
            with self.assertRaisesRegex(ValueError, "context"):
                commit_application(prepared, current_context=other, settings=self.settings)
            with (self.root/self.sample).open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                commit_application(prepared, current_context=self.context, settings=self.settings)
            self.assertFalse(self.settings.contains("drr/bn_auto_compare"))

        def test_common_bn_external_grid_and_fixed_flag(self):
            draft = BackgroundDraft(kind="bn", files=(self.bn,), which="all", fixed=True)
            prepared = prepare_application(self.context, draft)
            recipe = commit_application(prepared, current_context=self.context, settings=self.settings)
            self.assertEqual((recipe.mode, recipe.files, recipe.which, recipe.fixed), ("External", (self.bn,), "all", True))
            wrong = self.root/"wrong.csv"
            wrong.write_text("Vbg,Vtg,1.5,1.6\n0,0,1,2\n")
            with self.assertRaisesRegex(ValueError, "spectral|grid"):
                prepare_application(self.context, replace(draft, files=(wrong.name,)))

        def test_fixed_common_recipe_rebinds_without_reusing_group_assignments(self):
            first = prepare_application(self.context, BackgroundDraft(kind="bn", files=(self.bn,), which="all", fixed=True))
            context = replace(self.context, measurements=(self.sample2,))
            prepared = prepare_fixed_application(first.recipe, context)
            self.assertEqual(tuple(item.measurement_file for item in prepared.recipe.assignments), (self.sample2,))
            self.assertTrue(prepared.recipe.fixed)
            self.assertEqual(prepared.recipe.files, (self.bn,))
            self.assertIsNone(prepared.draft.bn_plan)
            self.assertIsNone(prepared.draft.pn_plan)
            with self.assertRaisesRegex(ValueError, "explicitly fixed"):
                prepare_fixed_application(replace(first.recipe, fixed=False), context)
            with self.assertRaisesRegex(ValueError, "experiment folders"):
                prepare_fixed_application(first.recipe, replace(context, folder=str(self.root/"another-experiment")))

        def test_owner_loading_locks_edits_and_cancellation_reaches_owner(self):
            cancelled = []
            self.bridge.applicationCancelRequested.connect(lambda: cancelled.append(True))
            self.bridge.application_started()
            self.bridge.setChoice("kind", 0)
            self.bridge.setText("regions", "1.6 1.7")
            self.bridge.selectFiles("background", [self.bn])
            self.assertEqual(self.bridge.draft.kind, "self")
            self.assertEqual(self.bridge.snapshot["texts"]["regions"], "")
            self.assertTrue(self.bridge.snapshot["busy"])
            self.bridge.invoke("cancel-task")
            self.assertEqual(cancelled, [True])
            self.assertFalse(self.bridge.snapshot["busy"])

        def test_source_drafts_survive_kind_switch_and_cancel_never_applies(self):
            applied = []
            self.bridge.applyRequested.connect(applied.append)
            self.bridge.setChoice("kind", 0)
            self.bridge.selectFiles("background", [self.bn])
            self.bridge.setChoice("kind", 4)
            self.bridge.setChoice("kind", 0)
            self.assertEqual(self.bridge.draft.files, (self.bn,))
            self.bridge.invoke("cancel")
            self.assertFalse(applied)
            self.assertFalse(self.bridge.snapshot["opened"])

        def test_pn_pool_is_explicit_and_warnings_require_review(self):
            self.bridge.setChoice("kind", 1)
            self.bridge.setChoice("method", 1)
            self.bridge.invoke("match")
            deadline = time.monotonic()+5
            while self.bridge.snapshot["busy"] and time.monotonic() < deadline:
                settle()
            self.assertIn("pool explicitly", self.bridge.snapshot["error"])
            self.bridge.selectFiles("background", [self.p])
            self.bridge.invoke("match")
            self.wait()
            plan = self.bridge.draft.pn_plan
            self.assertFalse(plan.automatic_candidates)
            self.assertEqual(plan.candidate_files, (self.p,))
            self.assertTrue(plan.ready)
            if plan.warnings:
                with self.assertRaisesRegex(ValueError, "acknowledg|warning"):
                    prepare_application(self.context, self.bridge.draft)
            self.bridge.setFlag("acknowledged", True)
            prepared = prepare_application(self.context, self.bridge.draft)
            recipe = commit_application(prepared, current_context=self.context, settings=self.settings)
            self.assertEqual(len(recipe.assignments), 2)
            self.assertEqual(recipe.assignments[0].baseline_files, (self.p,))

        def test_pn_sources_are_independent_and_opposite_role_rejected(self):
            self.bridge.selectFiles("P", [QUrl.fromLocalFile(str(self.root/self.p))])
            self.bridge.selectFiles("N", [self.n])
            self.bridge.selectFiles("P", [self.bn])
            self.assertIn("cannot be relabelled", self.bridge.snapshot["error"])
            self.assertEqual(self.bridge.snapshot["pFiles"], [self.p])
            self.assertEqual(self.bridge.snapshot["nFiles"], [self.n])
            self.assertEqual(self.bridge.snapshot["choices"]["generation"]["current"], 1)
            self.assertEqual(self.bridge.snapshot["training"], [dict(file=self.sample, selected=False), dict(file=self.sample2, selected=False)])

        def test_real_pn_generation_save_preview_and_application_are_separate(self):
            pair = DrrPnPair((self.p,), (self.n,))
            version, document, report = generate_pn(self.context, pair, mode="pn-only")
            self.assertFalse(report["scientifically_validated"])
            store = DrrPnStore(self.root)
            self.assertIsNone(store.find_applied(self.context.measurements))
            prepared = prepare_application(self.context, BackgroundDraft(kind="pn", version=version, which="all"))
            self.assertIsNone(store.find_applied(self.context.measurements))
            trial = load_with_recipe(self.context, prepared.recipe, prepared=prepared)
            self.assertEqual(trial.input_signatures, background_input_signatures(self.context, prepared.recipe))
            self.assertIsNone(store.find_applied(self.context.measurements))
            recipe = commit_application(prepared, current_context=self.context, settings=self.settings)
            self.assertEqual(store.find_applied(self.context.measurements).version_id, version.version_id)
            loaded = read_composite_background(self.root/recipe.files[0])
            np.testing.assert_array_equal(compose_background(document)[1], compose_background(loaded)[1])
            reused, _, _ = generate_pn(self.context, pair, mode="pn-only")
            self.assertEqual(reused.version_id, version.version_id)
            forced, _, _ = generate_pn(self.context, pair, mode="pn-only", force=True)
            self.assertNotEqual(forced.version_id, version.version_id)

        def test_reconstruction_requires_two_explicit_training_roles(self):
            with self.assertRaisesRegex(ValueError, "two distinct"):
                generate_pn(self.context, DrrPnPair((self.p,), (self.n,)), mode="reconstruction")
            self.assertFalse((self.root/"Processed Data"/"DRR"/"PN Backgrounds"/"manifest.json").exists())

        def test_legacy_per_file_fit_protection_uses_same_group_merge(self):
            recipe = dict(protected_regions_by_file={self.sample: [[1.6, 1.65]], self.sample2: [[1.63, 1.7]]},
                          automatic_protection=False, shared_texture=False)
            self.bridge.open(self.context, BackgroundDraft(kind="fitted", fitted_result=recipe))
            self.wait()
            self.assertEqual(parse_regions(self.bridge.snapshot["texts"]["regions"]), ((1.6, 1.7),))
            from core.drr_fitted_background import settings_for_group
            self.assertEqual(self.bridge._fit_recipe()["protected_regions"], settings_for_group(recipe, self.context.measurements).protected_regions)

        def test_native_composite_regions_preserve_exact_points_and_undo(self):
            self.bridge.invoke("page:composite")
            self.bridge.selectFiles("composite-add", [self.p])
            self.wait()
            self.bridge.selectFiles("composite-add", [self.n])
            self.wait()
            self.bridge.setChoice("compositeSource", 1)
            self.bridge.setNumber("regionLow", 1.65)
            self.bridge.setNumber("regionHigh", 1.8)
            before = compose_background(self.bridge._document)[1].copy()
            self.bridge.invoke("composite-region")
            self.assertEqual(len(self.bridge.snapshot["regions"]), 1)
            energy, values, _ = compose_background(self.bridge._document)
            mask = (energy >= 1.65) & (energy < 1.8)
            np.testing.assert_array_equal(values[mask], self.bridge._document.sources[1].intensity[mask])
            np.testing.assert_array_equal(values[~mask], before[~mask])
            self.bridge.selectRegion(0)
            self.bridge.setNumber("regionLow", 1.7)
            self.bridge.invoke("composite-region")
            self.assertEqual(len(self.bridge.snapshot["regions"]), 1)
            self.assertEqual(self.bridge.snapshot["regions"][0]["low"], 1.7)
            self.bridge.removeRegion(0)
            self.assertEqual(self.bridge.snapshot["regions"], [])
            self.bridge.invoke("composite-undo")
            self.bridge.invoke("composite-undo")
            self.bridge.invoke("composite-undo")
            np.testing.assert_array_equal(compose_background(self.bridge._document)[1], before)
            output = str(self.root/"manual.csv")
            self.bridge.selectFiles("composite-save", [output])
            self.wait()
            self.assertEqual(self.bridge.draft.kind, "custom")
            self.assertIsNone(DrrPnStore(self.root).find_applied(self.context.measurements))

        def test_raw_gate_reuses_heatmap_and_updates_linecut_without_full_draw(self):
            self.bridge.invoke("page:raw");self.bridge.invoke("preview-raw");self.wait();settle(60)
            axes=tuple(self.bridge.canvas.figure.axes);heatmap=self.bridge._raw_heatmap
            mesh=heatmap.collections[0];line=self.bridge._raw_spectrum_line
            draws=self.bridge.canvas.draw_count;blits=self.bridge.canvas.blit_count
            self.bridge.setNumber("gate",5);settle(40)
            self.assertEqual(tuple(self.bridge.canvas.figure.axes),axes)
            self.assertIs(heatmap.collections[0],mesh)
            self.assertIs(self.bridge._raw_spectrum_line,line)
            np.testing.assert_array_equal(line.get_ydata(),self.bridge._raw_preview.cube.Z[5])
            self.assertEqual(self.bridge.canvas.draw_count,draws)
            self.assertGreater(self.bridge.canvas.blit_count,blits)
            self.assertIn(f"{self.bridge._raw_preview.cube.gate[5]:g}",self.bridge._raw_spectrum_axis.get_title())

        def test_fitted_actual_preview_matches_core_and_templates_save_recipe_only(self):
            self.bridge.invoke("page:fit")
            self.bridge.setFlag("automatic_protection", True)
            self.bridge.setFlag("shared_texture", False)
            self.bridge.invoke("fit")
            self.wait()
            from core.drr_fitted_background import prepare_fitted_group, fit_prepared_group
            raw = prepare_fitted_group(self.root, self.context.measurements)
            result, reference = fit_prepared_group(raw, self.bridge._fit_recipe())
            np.testing.assert_array_equal(reference.background(), self.bridge._fit_reference.background())
            self.assertGreater(self.bridge.canvas.backend.frameGeneration, 0)
            self.assertEqual(self.bridge.canvas.figure.axes[0].get_xlim(), (float(raw.cube.energy[0]), float(raw.cube.energy[-1])))
            self.bridge.setText("templateName", "Synthetic isolated recipe")
            self.bridge.invoke("save-template")
            saved = json.loads(self.settings.value("drr/background-templates-v1"))
            self.assertEqual(set(saved["Synthetic isolated recipe"]), set(self.bridge._fit_recipe()))
            self.bridge.invoke("use-fit")
            prepared = prepare_application(self.context, self.bridge.draft)
            self.assertEqual(prepared.recipe.mode, "Fitted (sample)")
            self.assertEqual(prepared.recipe.assignments, ())
            trial = load_with_recipe(self.context, prepared.recipe, prepared=prepared)
            self.assertEqual(len(trial.references), 1)
            np.testing.assert_array_equal(trial.cube.Z, result.Z)
            self.assertEqual(trial.raw_snapshot.numerical_path, "fitted_raw_average")

        def test_heterogeneous_and_regional_loads_match_existing_core_exactly(self):
            from core.drr_sources import DrrMeasurementAssignment
            from core.drr_background_regions import DrrBackgroundRegion
            from core.data_io import load_drr_resolved_cube
            assignments = (DrrMeasurementAssignment(self.sample, "External", (self.p,), "first"),
                           DrrMeasurementAssignment(self.sample2, "External", (self.n,), "all"))
            prepared = prepare_application(self.context, BackgroundDraft(kind="pn", saved_assignments=assignments))
            result = load_with_recipe(self.context, prepared.recipe, prepared=prepared)
            expected = load_drr_resolved_cube(str(self.root), self.context.measurements, assignments)
            np.testing.assert_array_equal(result.cube.Z, expected.Z)
            self.assertEqual(result.cube.drr_numerical_path, "heterogeneous")
            regions = (DrrBackgroundRegion(1.65, 1.8, (self.n,), "all"),)
            prepared = prepare_application(self.context, BackgroundDraft(kind="custom", files=(self.p,), regions=regions))
            result = load_with_recipe(self.context, prepared.recipe, prepared=prepared)
            expected = load_drr_resolved_cube(str(self.root), self.context.measurements, prepared.recipe.assignments)
            np.testing.assert_array_equal(result.cube.Z, expected.Z)
            self.assertEqual(tuple(item.regions for item in result.raw_snapshot.assignments), (regions, regions))

        def test_trial_load_failure_does_not_publish_pn_application(self):
            version, _, _ = generate_pn(self.context, DrrPnPair((self.p,), (self.n,)), mode="pn-only")
            prepared = prepare_application(self.context, BackgroundDraft(kind="pn", version=version))
            (self.root/self.sample).unlink()
            with self.assertRaises((ValueError, OSError)):
                load_with_recipe(self.context, prepared.recipe, prepared=prepared)
            self.assertIsNone(DrrPnStore(self.root).find_applied((self.sample2,)))
            manifest = json.loads(DrrPnStore(self.root).manifest_path.read_text())
            self.assertFalse(manifest.get("uses"))

        def test_stale_version_requires_explicit_snapshot_and_no_automatic_old_selection(self):
            version, _, _ = generate_pn(self.context, DrrPnPair((self.p,), (self.n,)), mode="pn-only")
            (self.root/self.p).unlink()
            stale = DrrPnStore(self.root).get_version(version.version_id)
            self.assertFalse(stale.fresh)
            with self.assertRaisesRegex(ValueError, "stale|Version"):
                prepare_application(self.context, BackgroundDraft(kind="pn", version=stale))
            prepared = prepare_application(self.context, BackgroundDraft(kind="pn", version=stale, allow_stale=True))
            trial = load_with_recipe(self.context, prepared.recipe, prepared=prepared)
            self.assertTrue(np.isfinite(trial.cube.Z).any())
            self.assertIsNone(DrrPnStore(self.root).find_applied(self.context.measurements))

        def test_import_rejects_unknown_reconstruction_namespace(self):
            p = load_composite_source(self.root, (self.p,), "all", key="P")
            n = load_composite_source(self.root, (self.n,), "all", key="N")
            generated = replace(p, key="unknown", generation_json=json.dumps(dict(algorithm_version="pn_raw_unrecognized", settings={})))
            path = save_composite_background(self.root/"unknown-method.csv", CompositeBackground((p, n, generated), "unknown"))
            with self.assertRaisesRegex(ValueError, "Unsupported reconstruction"):
                import_pn(self.context, DrrPnPair((self.p,), (self.n,)), path)
            self.assertIsNone(DrrPnStore(self.root).find_applied(self.context.measurements))

        def test_real_bn_preview_keeps_original_without_joint_acceptance(self):
            payload = compare_bn(self.context, (self.bn,), "all")
            self.assertFalse(payload["eligible"])
            self.assertIsNotNone(payload["version_id"])
            self.assertEqual(payload["preview"]["drr"].shape, (2, len(self.energy)))
            self.bridge._bn_payload = payload
            self.bridge.invoke("page:bn")
            self.assertEqual(len(self.bridge.canvas.figure.axes), 3)
            from core.drr_bn_store import DrrBnStore
            store = DrrBnStore(self.root)
            self.assertIsNone(store.find_applied(self.context.measurements))
            original = BackgroundDraft(kind="bn", files=(self.bn,), which="all", calibration=dict(payload=payload, original=True))
            prepared = prepare_application(self.context, original)
            trial = load_with_recipe(self.context, prepared.recipe, prepared=prepared)
            self.assertTrue(np.isfinite(trial.cube.Z).any())
            with self.assertRaisesRegex(ValueError, "joint DR/R"):
                prepare_application(self.context, replace(original, calibration=dict(payload=payload, original=False)))
            self.assertIsNone(store.find_applied(self.context.measurements))

        def test_raw_preview_is_read_only_and_matches_core_reference(self):
            before = (self.root/self.sample).read_bytes()
            self.bridge.invoke("page:raw")
            self.bridge.setChoice("rawScope", 1)
            self.bridge.invoke("preview-raw")
            self.wait()
            expected = load_preview(self.bridge._make_raw_snapshot(), "measurement", "average")
            np.testing.assert_array_equal(expected.cube.Z, self.bridge._raw_preview.cube.Z)
            self.assertEqual((self.root/self.sample).read_bytes(), before)
            self.assertIsNotNone(self.bridge._raw_heatmap)
            self.assertEqual(len(self.bridge.canvas.figure.axes), 3)
            self.assertFalse(self.settings.contains("drr/bn_auto_compare"))

        def test_stale_worker_cannot_apply_after_context_change(self):
            release, started = Event(), Event()
            results = []
            def work(token):
                started.set()
                release.wait(3)
                return "old"
            self.bridge._run(work, results.append, "controlled obsolete worker")
            self.assertTrue(started.wait(1))
            self.bridge.open(replace(self.context, measurements=(self.sample,)))
            release.set()
            self.wait()
            self.assertEqual(results, [])
            self.assertEqual(self.bridge.context.measurements, (self.sample,))

        def test_validated_numeric_drafts_preserve_previous_value(self):
            self.bridge.setFlag("auto_bn_compare", True)
            self.assertFalse(self.bridge.snapshot["flags"]["auto_bn_compare"])
            self.assertIn("unavailable", self.bridge.snapshot["error"])
            self.bridge.setNumber("smoothing", float("nan"))
            self.assertEqual(self.bridge.snapshot["numbers"]["smoothing"], .001)
            self.assertIn("finite", self.bridge.snapshot["error"])
            self.bridge.setNumber("periodic_components", 9)
            self.assertEqual(self.bridge.snapshot["numbers"]["periodic_components"], 3)
            self.assertFalse(self.bridge.snapshot["applyEnabled"])
            self.assertEqual(parse_regions("1.5 1.6; 1.7, 1.8"), ((1.5, 1.6), (1.7, 1.8)))
            with self.assertRaises(ValueError):
                parse_regions("1.8 1.7")

        def test_real_qml_pages_render_without_widgets_or_errors(self):
            engine = QQmlEngine()
            warnings = []
            engine.warnings.connect(lambda messages: warnings.extend(message.toString() for message in messages))
            path = Path(__file__).resolve().parents[1]/"ui_qt"/"quick"/"qml"/"DrrBackgroundSettings.qml"
            component = QQmlComponent(engine, QUrl.fromLocalFile(str(path)))
            item = component.createWithInitialProperties(dict(bridge=self.bridge, colors=colors))
            self.assertIsNotNone(item, "\n".join(error.toString() for error in component.errors()))
            window = QQuickWindow()
            window.resize(1020, 720)
            item.setParentItem(window.contentItem())
            item.setWidth(1020)
            item.setHeight(720)
            window.show()
            settle(20)
            self.assertFalse(self.bridge.snapshot["hasPreview"])
            empty = item.findChild(QQuickItem, "drr_background_empty_preview")
            self.assertIsNotNone(empty)
            navigation = item.findChild(QQuickItem, "drr_background_plot_pan")
            self.assertFalse(navigation.isEnabled())
            self.bridge.setFlag("automatic_protection", True)
            self.bridge.setFlag("shared_texture", False)
            self.bridge.invoke("fit")
            self.wait()
            for page in ("settings", "pn", "fit", "composite", "bn", "raw"):
                self.bridge.invoke("page:"+page)
                settle(35)
                capture = os.environ.get("DPTK_BACKGROUND_CAPTURE")
                if capture and page in {"pn", "fit"}:
                    window.grabWindow().save(capture + "-" + page + ".png")
            self.assertFalse(window.grabWindow().isNull())
            item.setWidth(700)
            item.setHeight(600)
            window.resize(700, 600)
            self.bridge.invoke("page:fit")
            settle(40)
            plot = item.findChild(QQuickItem, "drr_background_plot")
            self.assertGreaterEqual(plot.height(), 125)
            for name in ("drr_background_tools", "drr_background_page_settings", "drr_background_cancel", "drr_background_apply"):
                action = item.findChild(QQuickItem, name)
                self.assertIsNotNone(action, name)
                self.assertTrue(action.isVisible(), name)
                left = action.mapToScene(QPointF()).x()
                self.assertGreaterEqual(left, 0, name)
                self.assertLessEqual(left + action.width(), window.width(), name)
            if os.environ.get("DPTK_BACKGROUND_CAPTURE"):
                window.grabWindow().save(os.environ["DPTK_BACKGROUND_CAPTURE"] + "-compact-fit.png")
            self.assertFalse(warnings, "\n".join(warnings))
            self.assertNotIn("PySide6.QtWidgets", sys.modules)
            self.assertNotIn("matplotlib.backends.backend_qtagg", sys.modules)
            window.close()
            item.setParentItem(None)
            item.deleteLater()
            settle()

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
