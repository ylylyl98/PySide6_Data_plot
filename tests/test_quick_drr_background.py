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
            for mode in ("--cold-settings", "--isolated"):
                with self.subTest(mode=mode):
                    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), mode],
                        cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=180)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _cold_settings():
    """The first background choice must not pay for unused plotting/analysis."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QGuiApplication
    from ui_qt.quick.drr_background_bridge import DrrBackgroundBridge
    from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft

    app = QGuiApplication([])
    with TemporaryDirectory() as folder:
        bridge = DrrBackgroundBridge(settings=QSettings(str(Path(folder) / "cold.ini"), QSettings.Format.IniFormat))
        try:
            context = BackgroundContext(folder, ("measurement.csv",))
            bridge.open(context)
            assert bridge.draft.which == "last"
            for frame in (0, 1):
                bridge.setChoice("frame", frame)
                assert bridge.draft.which == ("first", "last")[frame]
            bridge.invoke("page:raw")
            bridge.invoke("plot:home")
            bridge.invoke("page:settings")
            bridge.invoke("cancel")
            # Reopening a saved draft restores its choice without constructing
            # a preview or importing the fitting code for its default values.
            bridge.open(context, BackgroundDraft(which="first"))
            assert bridge.draft.which == "first"
            assert bridge.canvas is None and bridge.plotBackend is None
            assert bridge.snapshot["plotTool"] == ""
            # Entering BN calibration or choosing a method must stay lazy.
            bridge.open(context, BackgroundDraft(kind="bn", files=("BNbackground.csv",), which="all"))
            bridge.invoke("page:bn")
            bridge.setChoice("bnCalibrationMethod", 0)
            bridge.setChoice("bnCalibrationMethod", 1)
            assert bridge.canvas is None and not bridge.snapshot["busy"]
            forbidden = {"core.loader", "core.processing_run", "core.drr_bn_calibration", "core.drr_fitted_background"}
            unexpected = [name for name in sys.modules if name in forbidden
                or name.split(".", 1)[0] in {"scipy", "matplotlib"}]
            assert not unexpected, unexpected
            app.processEvents()
        finally:
            bridge.close()
    return 0


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import csv
    from dataclasses import replace
    import json
    from threading import Event
    import time
    from unittest.mock import patch
    import numpy as np
    from PySide6.QtCore import QEvent, QPointF, QSettings, Qt, QUrl
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication, QKeyEvent
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

        def test_direct_bn_original_applies_without_generating_a_candidate(self):
            draft = BackgroundDraft(kind="bn", files=(self.bn,), which="all")
            self.bridge.open(self.context, draft)
            self.wait()
            signals = []
            self.bridge.applyRequested.connect(signals.append)
            self.bridge.invoke("apply-bn-original")
            self.wait()
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0].recipe.files, (self.bn,))
            self.assertIsNone(signals[0].draft.calibration)
            self.assertEqual(self.bridge.draft, draft)
            self.assertFalse(self.settings.contains("drr/bn_auto_compare"))
            from core.drr_bn_store import DrrBnStore
            store = DrrBnStore(self.root)
            self.assertFalse(store.has_original_choice(self.context.measurements, (self.bn,), "all"))
            commit_application(signals[0], current_context=self.context, settings=self.settings)
            self.assertTrue(store.has_original_choice(self.context.measurements, (self.bn,), "all"))

        def test_direct_bn_candidate_is_transactional_and_requires_current_preview(self):
            draft = BackgroundDraft(kind="bn", files=(self.bn,), which="all")
            self.bridge.open(self.context, draft)
            self.wait()
            signals = []
            self.bridge.applyRequested.connect(signals.append)
            self.bridge.invoke("apply-bn-candidate")
            self.assertIn("eligible calibration preview", self.bridge.snapshot["error"])
            payload = dict(eligible=True, candidates=[], report={})
            self.bridge._bn_payload = payload
            with patch.object(self.bridge, "_run", side_effect=lambda work, complete, message: complete(work(None))), \
                 patch("ui_qt.quick.drr_background_bridge.prepare_application", side_effect=ValueError("source changed")):
                self.bridge.invoke("apply-bn-candidate")
            self.assertIn("source changed", self.bridge.snapshot["error"])
            self.assertEqual(signals, [])
            self.assertEqual(self.bridge.draft, draft)
            prepared = object()
            with patch.object(self.bridge, "_run", side_effect=lambda work, complete, message: complete(work(None))), \
                 patch("ui_qt.quick.drr_background_bridge.prepare_application", return_value=prepared) as prepare:
                self.bridge.invoke("apply-bn-candidate")
                proposal = prepare.call_args.args[1]
                self.assertEqual(proposal.calibration, dict(payload=payload, original=False))
                self.assertEqual(signals, [prepared])
                self.assertEqual(self.bridge.draft, draft)
                self.bridge.invoke("apply-bn-original")
                self.assertTrue(prepare.call_args.args[1].calibration["original"])
            self.bridge.setChoice("bnCalibrationMethod", 0)
            self.bridge.invoke("apply-bn-candidate")
            self.assertIn("eligible calibration preview", self.bridge.snapshot["error"])
            self.assertEqual(len(signals), 2)
            self.assertFalse(self.settings.contains("drr/bn_auto_compare"))

        def test_direct_original_bn_commit_failure_retains_selection(self):
            from core.drr_bn_store import DrrBnStore
            store = DrrBnStore(self.root)
            store.deactivate(self.context.measurements, (self.n,), "first", restore_original=True)
            before = store.manifest_path.read_bytes()
            draft = BackgroundDraft(kind="bn", files=(self.bn,), which="all")
            prepared = prepare_application(self.context, draft, original_bn=True)
            self.assertEqual(store.manifest_path.read_bytes(), before)
            with patch.object(DrrBnStore, "_atomic_json", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    commit_application(prepared, current_context=self.context, settings=self.settings)
            self.assertEqual(store.manifest_path.read_bytes(), before)
            self.assertFalse(self.settings.contains("drr/bn_auto_compare"))
            with self.assertRaisesRegex(ValueError, "context"):
                commit_application(prepared, current_context=replace(self.context, measurements=(self.sample,)), settings=self.settings)
            self.assertEqual(store.manifest_path.read_bytes(), before)

        def test_direct_original_bn_requires_common_measured_sources_without_preview(self):
            from core.drr_sources import DrrMeasurementAssignment
            assignments = tuple(DrrMeasurementAssignment(name, "External", (source,), "all")
                                for name, source in zip(self.context.measurements, (self.bn, self.n)))
            draft = BackgroundDraft(kind="bn", saved_assignments=assignments)
            with self.assertRaisesRegex(ValueError, "generated comparison"):
                prepare_application(self.context, draft, original_bn=True)

        def test_direct_bn_apply_rejects_duplicate_and_obsolete_requests(self):
            draft = BackgroundDraft(kind="bn", files=(self.bn,), which="all")
            self.bridge.open(self.context, draft)
            self.wait()
            release, started = Event(), Event()
            self.addCleanup(release.set)
            signals = []
            self.bridge.applyRequested.connect(signals.append)
            def prepare(context, proposal, **kwargs):
                started.set()
                release.wait(3)
                return "obsolete"
            with patch("ui_qt.quick.drr_background_bridge.prepare_application", side_effect=prepare) as work:
                self.bridge.invoke("apply-bn-original")
                self.assertTrue(started.wait(1))
                self.bridge.invoke("apply-bn-original")
                self.assertEqual(work.call_count, 1)
                self.bridge.open(replace(self.context, measurements=(self.sample,)), draft)
                release.set()
                self.wait()
                settle(100)
            self.assertEqual(signals, [])

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

        def test_multiple_bn_per_file_default_preview_and_joint_option(self):
            from types import ModuleType
            from core.drr_sources import DrrMeasurementAssignment
            original = [DrrMeasurementAssignment(name, "External", (self.bn,), "all") for name in self.context.measurements]
            candidate = [DrrMeasurementAssignment(self.sample, "External", ("managed.csv",), "last"), original[1]]
            payload = dict(batch=True, eligible=True, status="per_measurement", summary="One candidate eligible",
                source_identity=dict(root=str(self.root), measurements=[dict(path=v) for v in self.context.measurements], y_axis=self.context.y_axis),
                settings=dict(derivative_window=self.context.derivative_window, derivative_polyorder=self.context.derivative_polyorder),
                original_assignments=[v.to_dict() for v in original], candidate_assignments=[v.to_dict() for v in candidate],
                items=[dict(measurement=self.sample, version_id="v1", status="single_pair", eligible=True, summary="eligible"),
                       dict(measurement=self.sample2, version_id="v2", status="original", eligible=False, summary="fallback")],
                energy=self.energy, gates=np.arange(3), original_drr=np.ones((3, len(self.energy))),
                candidate_drr=np.full((3, len(self.energy)), 2.), original_d2=np.ones((3, len(self.energy))),
                candidate_d2=np.full((3, len(self.energy)), 2.), candidates=[], report={}, exports={}, elapsed_seconds=1.)
            module = ModuleType("ui_qt.quick.drr_bn_batch_workflow")
            calls = []
            def compare(*args, **kwargs):
                calls.append((args, kwargs))
                return payload
            module.compare_bn_batch = compare
            self.bridge.open(self.context, BackgroundDraft(kind="bn", files=(self.bn,), which="all"))
            self.assertTrue(self.bridge.snapshot["bnIndividualCalibration"])
            with patch.dict(sys.modules, {"ui_qt.quick.drr_bn_batch_workflow": module}), \
                 patch.object(self.bridge, "_run", side_effect=lambda work, complete, message: complete(work(None))):
                self.bridge.invoke("page:bn")
                self.bridge._compare_bn()
                self.bridge._refresh()
                self.assertEqual(len(calls), 1)
                self.assertTrue(calls[0][1]["settings"]["single_full_range"])
                self.assertEqual(len(self.bridge.canvas.figure.axes), 6)
                axes = self.bridge.canvas.figure.axes
                self.assertEqual([axis.collections[0].get_clim() for axis in axes[:3]], [(1., 2.)]*3)
                self.assertEqual([axis.collections[0].get_clim() for axis in axes[3:]], [(-2., 2.)]*3)
                self.assertIn("not applied", axes[2].get_title())
                self.assertIn("Original BN retained", self.bridge.snapshot["bnBatchItems"][1]["status"])
                self.assertTrue(self.bridge.snapshot["canUseCalibrated"])
                self.bridge.invoke("use-bn-candidate")
                self.assertEqual(self.bridge._raw_assignments(), tuple(candidate))
                draft = self.bridge.draft
                self.bridge.open(self.context, draft)
                self.assertTrue(self.bridge.restore_comparison(payload))
                wrong = dict(payload, source_identity={**payload["source_identity"], "y_axis": "wrong"})
                self.assertFalse(self.bridge.restore_comparison(wrong))
                self.bridge.invoke("page:bn")
                engine = QQmlEngine()
                component = QQmlComponent(engine, QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/"ui_qt"/"quick"/"qml"/"DrrBackgroundSettings.qml")))
                item = component.createWithInitialProperties(dict(bridge=self.bridge, colors=colors))
                self.assertIsNotNone(item, "\n".join(error.toString() for error in component.errors()))
                window = QQuickWindow()
                window.resize(1020, 720)
                item.setParentItem(window.contentItem())
                item.setSize(window.size())
                window.show()
                settle(35)
                self.assertTrue(item.findChild(QQuickItem, "drr_background_flag_bn.per_measurement").isVisible())
                self.assertFalse(item.findChild(QQuickItem, "drr_background_flag_bn.single_full_range").isVisible())
                item.findChild(QQuickItem, "drr_bn_advanced_toggle").clicked.emit()
                settle()
                self.assertTrue(item.findChild(QQuickItem, "drr_background_flag_bn.single_full_range").isVisible())
                self.assertFalse(item.findChild(QQuickItem, "bn_weak_feature_warning").isVisible())
                item.findChild(QQuickItem, "drr_bn_details_toggle").clicked.emit()
                settle()
                self.assertTrue(item.findChild(QQuickItem, "bn_weak_feature_warning").isVisible())
                self.bridge.setFlag("bn.per_measurement", False)
                settle()
                self.assertFalse(item.findChild(QQuickItem, "drr_background_flag_bn.single_full_range").isVisible())
                window.close()
                item.deleteLater()
                settle()
                self.assertFalse(self.bridge.snapshot["bnIndividualCalibration"])
                self.assertFalse(self.bridge.snapshot["canUseCalibrated"])
                with patch("ui_qt.quick.drr_background_workflow.compare_bn", return_value=payload) as joint:
                    self.bridge._compare_bn()
                    joint.assert_called_once()

        def test_single_bn_trend_settings_route_and_invalidate_preview(self):
            context = replace(self.context, measurements=(self.sample,))
            self.bridge.open(context, BackgroundDraft(kind="bn", files=(self.bn,), which="all"))
            self.assertFalse(self.bridge.snapshot["flags"]["bn.single_trend_enabled"])
            self.assertEqual(self.bridge.snapshot["choices"]["bnCalibrationMethod"]["current"], 1)
            self.assertEqual(self.bridge.snapshot["choices"]["bnTrendOrder"]["current"], 0)
            self.bridge._bn_payload = dict(eligible=True, candidates=[], report={})
            self.bridge._refresh()
            self.assertTrue(self.bridge.snapshot["canUseCalibrated"])
            self.bridge.setChoice("bnCalibrationMethod", 0)
            self.assertFalse(self.bridge.snapshot["canUseCalibrated"])
            self.bridge.setText("bn.single_trend_regions", "1.46 1.53; 1.94 2.02")
            self.bridge.setChoice("bnTrendOrder", 2)
            self.bridge.setNumber("bn.single_trend_max_sensitivity", .03)
            with patch("ui_qt.quick.drr_background_workflow.compare_bn") as compare, \
                 patch.object(self.bridge, "_run", side_effect=lambda work, complete, message: work(None)):
                self.bridge._compare_bn()
                config = compare.call_args.kwargs["settings"]
                self.assertTrue(config["single_trend_enabled"])
                self.assertEqual(config["single_trend_regions"], ((1.46, 1.53), (1.94, 2.02)))
                self.assertEqual(config["single_trend_order"], "quadratic")
                self.assertEqual(config["single_trend_max_sensitivity"], .03)
                self.assertNotIn("single_regions", config)
            self.bridge.setFlag("bn.single_trend_enabled", False)
            with patch("ui_qt.quick.drr_background_workflow.compare_bn") as compare, \
                 patch.object(self.bridge, "_run", side_effect=lambda work, complete, message: work(None)):
                self.bridge._compare_bn()
                self.assertFalse(compare.call_args.kwargs["settings"]["single_trend_enabled"])
            from types import ModuleType
            module = ModuleType("ui_qt.quick.drr_bn_batch_workflow")
            from unittest.mock import Mock
            module.compare_bn_batch = Mock()
            self.bridge.open(self.context, BackgroundDraft(kind="bn", files=(self.bn,), which="all"))
            self.bridge.setFlag("bn.single_trend_enabled", True)
            with patch.dict(sys.modules, {"ui_qt.quick.drr_bn_batch_workflow": module}), \
                 patch.object(self.bridge, "_run", side_effect=lambda work, complete, message: work(None)):
                self.bridge._compare_bn()
                self.assertTrue(module.compare_bn_batch.call_args.kwargs["settings"]["single_trend_enabled"])
                self.assertEqual(module.compare_bn_batch.call_args.kwargs["settings"]["single_trend_order"], "quadratic")

        def test_single_bn_trend_report_restores_settings_and_diagnostics(self):
            context = replace(self.context, measurements=(self.sample,))
            self.bridge.open(context, BackgroundDraft(kind="bn", files=(self.bn,), which="all"))
            payload = dict(eligible=True, candidates=[], summary="Trend ready", status="single_pair", reasons=[],
                source_identity=dict(root=str(self.root), measurements=[dict(path=self.sample)],
                    baseline_files=[dict(path=self.bn)], baseline_which="all", y_axis=context.y_axis),
                settings=dict(derivative_window=context.derivative_window, derivative_polyorder=context.derivative_polyorder,
                    single_trend_enabled=True, single_trend_order="linear", single_trend_regions=((1.46,1.53),(1.94,2.02)),
                    single_trend_max_sensitivity=.03),
                report=dict(optimization_policy="shared_bn_trend_v1", selected_candidate="trend_linear",
                    models=dict(trend_linear=dict(cv_original_rms=.1,cv_candidate_rms=.04,max_sensitivity=.012,sensitivity_pass=True))))
            with patch.object(self.bridge, "_draw_preview"):
                self.assertTrue(self.bridge.restore_comparison(payload))
            state=self.bridge.snapshot
            self.assertTrue(state["flags"]["bn.single_trend_enabled"])
            self.assertEqual(state["choices"]["bnCalibrationMethod"]["current"], 0)
            self.assertEqual(state["choices"]["bnTrendOrder"]["current"], 1)
            self.assertEqual(state["numbers"]["bn.single_trend_max_sensitivity"], .03)
            self.assertEqual(parse_regions(state["texts"]["bn.single_trend_regions"]), ((1.46,1.53),(1.94,2.02)))
            self.assertIn("Shared linear trend", state["bnModelSummary"])
            self.assertIn("1.20%", state["bnModelSummary"])
            self.assertIn("D2 not validated", state["bnModelSummary"])
            payload["report"] = dict(optimization_policy="shared_bn_trend_v2", selected_candidate="trend_linear",
                models=dict(trend_linear=dict(strength=.375, eligible=True, d2_guard=dict(max_relative_change=.049))))
            with patch.object(self.bridge, "_draw_preview"):
                self.assertTrue(self.bridge.restore_comparison(payload))
            self.assertIn("Adaptive linear", self.bridge.snapshot["bnModelSummary"])
            self.assertIn("4.9%", self.bridge.snapshot["bnModelSummary"])
            self.bridge.setChoice("bnTrendOrder", 2)
            self.assertFalse(self.bridge.snapshot["canUseCalibrated"])

        def test_single_bn_full_range_setting_reaches_comparison(self):
            context = replace(self.context, measurements=(self.sample,))
            self.bridge.open(context, BackgroundDraft(kind="bn", files=(self.bn,), which="all"))
            with patch("ui_qt.quick.drr_background_workflow.compare_bn") as compare, \
                 patch.object(self.bridge, "_run", side_effect=lambda work, complete, message: work(None)):
                self.bridge._compare_bn()
                self.assertTrue(compare.call_args.kwargs["settings"]["single_full_range"])
                self.bridge.setFlag("bn.single_full_range", False)
                self.bridge._compare_bn()
                self.assertFalse(compare.call_args.kwargs["settings"]["single_full_range"])

        def test_single_bn_payload_renders_output_and_weak_feature_warning(self):
            context = replace(self.context, measurements=(self.sample,))
            with patch("core.drr_bn_single_exports.save_single_pair_products", return_value=dict(
                    status="preview_only", paths={"comparison_png": "comparison.png"}, directory=str(self.root), elapsed_seconds=0.1)):
                payload = compare_bn(context, (self.bn,), "all")
            self.bridge.open(context, BackgroundDraft(kind="bn", files=(self.bn,), which="all"))
            payload["report"]["optimization_policy"] = "multi_bn_single_v1"
            self.bridge._bn_payload = payload
            self.bridge.invoke("page:bn")
            self.bridge._refresh()
            self.assertEqual(len(self.bridge.canvas.figure.axes), 6)
            np.testing.assert_allclose(payload["preview"]["maps_drr"][0],
                                       payload["mean_raw"]/payload["original_background"]-1)
            axes = self.bridge.canvas.figure.axes
            self.assertEqual(axes[0].collections[0].get_clim(), axes[1].collections[0].get_clim())
            self.assertEqual(axes[2].collections[0].get_clim(), axes[3].collections[0].get_clim())
            self.assertEqual(axes[2].collections[0].get_clim()[0], -axes[2].collections[0].get_clim()[1])
            self.assertEqual(axes[4].get_ylabel(), "DR/R")
            self.assertEqual(axes[5].get_ylabel(), "D2 (eV⁻²)")
            self.assertEqual(axes[0].get_ylabel(), payload["preview"]["gate_label"])
            payload["eligible"] = False
            payload["preview"]["candidate_available"] = True
            self.bridge._draw_preview()
            self.assertIn("not eligible", self.bridge.canvas.figure.axes[1].get_title())
            payload["eligible"] = True
            self.bridge._draw_preview()
            self.assertIn("Eligible within-scan", self.bridge.canvas.figure.axes[1].get_title())
            payload["eligible"] = False
            self.bridge._draw_preview()
            self.assertIn("Multi BN measured texture templates", self.bridge.snapshot["bnModelSummary"])
            payload["report"] = {"single_pair_scope": {"model": "phase_ten_knots"}}
            self.bridge._refresh()
            self.assertIn("smooth energy-dependent phase", self.bridge.snapshot["bnModelSummary"])
            payload["report"]["single_pair_scope"]["model"] = "phase_with_shared_high_frequency"
            self.bridge._refresh()
            self.assertIn("phase and shared fine-response correction", self.bridge.snapshot["bnModelSummary"])
            self.assertTrue(self.bridge.snapshot["bnSingleMeasurement"])
            self.assertIn("comparison.png", self.bridge.snapshot["bnOutputSummary"])
            self.assertEqual(self.bridge.snapshot["canUseCalibrated"], payload["eligible"])
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
            settle(35)
            capture = os.environ.get("DPTK_BN_CAPTURE")
            if capture:
                settle(500)
                window.grabWindow().save(capture)
            warning = item.findChild(QQuickItem, "bn_weak_feature_warning")
            outputs = item.findChild(QQuickItem, "bn_outputs")
            self.assertFalse(warning.isVisible())
            self.assertFalse(item.findChild(QQuickItem, "bn_model_summary").isVisible())
            with patch.object(self.bridge, "_draw_preview") as draw:
                item.findChild(QQuickItem, "drr_bn_details_toggle").clicked.emit()
                settle()
                draw.assert_not_called()
            self.assertTrue(warning.isVisible())
            self.assertIn("Real sample peaks and weak features", warning.property("text"))
            self.assertTrue(item.findChild(QQuickItem, "bn_model_summary").isVisible())
            self.assertTrue(outputs.isVisible())
            self.assertIn("1 file generated", outputs.property("text"))
            self.assertNotIn("comparison.png", outputs.property("text"))
            self.assertIn("comparison.png", self.bridge.snapshot["bnOutputSummary"])
            self.assertEqual(self.bridge.snapshot["bnOutputCount"], 1)
            self.assertTrue(item.findChild(QQuickItem, "drr_background_open-bn-output-folder").isVisible())
            with patch("PySide6.QtGui.QDesktopServices.openUrl", return_value=True) as opener:
                self.bridge.invoke("open-bn-output-folder")
                opener.assert_called_once()
            multi_note = item.findChild(QQuickItem, "bn_multi_reference_explanation")
            self.assertTrue(multi_note.isVisible())
            self.bridge.setFlag("bn.single_trend_enabled", True)
            settle()
            trend_note = item.findChild(QQuickItem, "bn_trend_explanation")
            self.assertTrue(trend_note.isVisible())
            self.assertIn("Changes absolute DR/R", trend_note.property("text"))
            self.assertFalse(item.findChild(QQuickItem, "drr_background_apply").isVisible())
            self.assertTrue(item.findChild(QQuickItem, "drr_background_generate-bn").isVisible())
            self.assertTrue(item.findChild(QQuickItem, "drr_background_compare-bn").isVisible())
            self.assertTrue(item.findChild(QQuickItem, "drr_background_choice_bnCalibrationMethod").isVisible())
            self.assertFalse(item.findChild(QQuickItem, "drr_background_editor_bn.single_trend_regions").isVisible())
            item.findChild(QQuickItem, "drr_bn_advanced_toggle").clicked.emit()
            settle()
            self.assertTrue(item.findChild(QQuickItem, "drr_background_editor_bn.single_trend_regions").isVisible())
            self.assertFalse(item.findChild(QQuickItem, "drr_background_editor_bn.taper_width").isVisible())
            self.assertFalse(item.findChild(QQuickItem, "drr_background_flag_bn.single_full_range").isVisible())
            self.bridge.setFlag("bn.single_trend_enabled", False)
            settle()
            self.assertIn("One measured BN file is enough", multi_note.property("text"))
            self.assertIn("not independent sample repeats", multi_note.property("text"))
            self.assertTrue(self.bridge.snapshot["flags"]["bn.single_full_range"])
            self.assertFalse(item.findChild(QQuickItem, "drr_background_editor_bn.single_regions").isVisible())
            self.assertFalse(item.findChild(QQuickItem, "drr_background_editor_bn.protected_regions").isVisible())
            self.bridge.setFlag("bn.single_full_range", False)
            settle()
            self.assertTrue(item.findChild(QQuickItem, "drr_background_editor_bn.single_regions").isVisible())
            self.assertTrue(item.findChild(QQuickItem, "drr_background_editor_bn.protected_regions").isVisible())
            self.assertFalse(self.bridge.snapshot["canUseCalibrated"])
            self.assertFalse(item.findChild(QQuickItem, "drr_background_editor_bn.uncertain_regions").isVisible())
            self.assertFalse(item.findChild(QQuickItem, "drr_background_editor_bn.trusted_regions").isVisible())
            self.assertIsNone(item.findChild(QQuickItem, "drr_background_editor_bn.train_region.low"))
            self.assertFalse(warnings, "\n".join(warnings))
            window.close()
            item.deleteLater()
            settle()

        def test_direct_bn_button_generates_automatic_trend_without_method_page(self):
            context = replace(self.context, measurements=(self.sample,))
            self.bridge.open(context, BackgroundDraft(kind="bn", files=(self.bn,), which="all"))
            self.bridge.setChoice("bnCalibrationMethod", 1)
            self.bridge.setChoice("bnTrendOrder", 2)
            self.bridge.setText("bn.single_trend_regions", "1.45 1.55; 1.95 2.05")
            self.bridge._bn_payload = dict(summary="old texture", eligible=True)
            payload = dict(summary="automatic trend", eligible=True, candidates=[], report={}, settings=dict(single_trend_enabled=True))
            self.bridge.set_current_background_label("Original BN")
            generated = []
            self.bridge.calibrationGenerated.connect(generated.append)
            applications = []
            self.bridge.applyRequested.connect(applications.append)
            engine = QQmlEngine()
            path = Path(__file__).resolve().parents[1]/"ui_qt"/"quick"/"qml"/"DrrBackgroundSettings.qml"
            component = QQmlComponent(engine, QUrl.fromLocalFile(str(path)))
            item = component.createWithInitialProperties(dict(bridge=self.bridge, colors=colors))
            self.assertIsNotNone(item)
            window = QQuickWindow()
            window.resize(760, 640)
            item.setParentItem(window.contentItem())
            item.setWidth(760)
            item.setHeight(640)
            window.show()
            settle(100)
            button = item.findChild(QQuickItem, "drr_background_calibrate-bn")
            self.assertTrue(button.isVisible())
            self.assertTrue(button.isEnabled())
            self.assertEqual(button.property("text"), "Calibrate & use · Auto trend")
            self.assertFalse(item.findChild(QQuickItem, "drr_background_choice_bnCalibrationMethod").isVisible())
            capture = os.environ.get("DPTK_BN_DIRECT_CAPTURE")
            if capture:
                window.grabWindow().save(capture)
            with patch("ui_qt.quick.drr_background_workflow.compare_bn", return_value=payload) as compare, \
                 patch("ui_qt.quick.drr_background_bridge.prepare_application", return_value="prepared") as prepare, \
                 patch.object(self.bridge, "_run", side_effect=lambda work, complete, message: complete(work(None))), \
                 patch.object(self.bridge, "_draw_preview") as draw:
                button.clicked.emit()
                self.assertEqual(generated, [payload])
                self.assertEqual(applications, ["prepared"])
                self.assertFalse(prepare.call_args.args[1].calibration["original"])
                self.assertEqual(self.bridge.snapshot["page"], "settings")
                self.assertEqual(compare.call_args.args[1:3], ((self.bn,), "all"))
                config = compare.call_args.kwargs["settings"]
                self.assertTrue(config["single_trend_enabled"])
                self.assertEqual(config["single_trend_order"], "auto")
                self.assertEqual(config["single_trend_regions"], ())
                self.assertFalse(compare.call_args.kwargs["export_products"])
                settle()
                current = item.findChild(QQuickItem, "bn_current_background")
                ready = item.findChild(QQuickItem, "bn_generated_background")
                self.assertEqual(current.property("text"), "Current: Original BN")
                self.assertEqual(ready.property("text"), "Ready: Auto trend · not applied")
                self.assertTrue(ready.isVisible())
                self.assertFalse(item.findChild(QQuickItem, "drr_background_apply").isVisible())
                self.assertIsNone(item.findChild(QQuickItem, "drr_background_export-bn"))
                if capture:
                    window.grabWindow().save(capture.replace(".png", "-ready.png"))
                draw.assert_not_called()
                self.bridge.invoke("page:bn")
                self.bridge.setChoice("bnCalibrationMethod", 1)
                self.bridge.invoke("generate-bn")
                self.assertFalse(compare.call_args.kwargs["settings"]["single_trend_enabled"])
                self.assertEqual(applications, ["prepared"])
            window.close()
            item.setParentItem(None)
            item.deleteLater()
            settle()
            window.deleteLater()
            QGuiApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            settle()

        def test_direct_bn_batch_and_failure_do_not_reuse_previous_candidate(self):
            self.bridge.open(self.context, BackgroundDraft(kind="bn", files=(self.bn,), which="all"))
            self.bridge.setFlag("bn.per_measurement", False)
            self.bridge._bn_payload = dict(summary="old texture", eligible=True)
            generated = []
            self.bridge.calibrationGenerated.connect(generated.append)
            with patch("ui_qt.quick.drr_bn_batch_workflow.compare_bn_batch", side_effect=ValueError("failed current inputs")) as compare, \
                 patch.object(self.bridge, "_run", side_effect=lambda work, complete, message: complete(work(None))):
                self.bridge.invoke("calibrate-bn")
                compare.assert_called_once()
                self.assertTrue(compare.call_args.kwargs["settings"]["single_trend_enabled"])
                self.assertTrue(self.bridge.snapshot["flags"]["bn.per_measurement"])
                self.assertEqual(generated, [])
                self.assertIn("failed current inputs", self.bridge.snapshot["error"])
                self.assertFalse(self.bridge.snapshot["canUseCalibrated"])

        def test_bn_picker_uses_measured_inputs_of_applied_calibration(self):
            context = replace(self.context, measurements=(self.sample,))
            generated = self.root / "copy_BNbackground.csv"
            generated.write_text("Vbg,Vtg,BNCalibrationVersion,730\n0,0,bn_test,10\n", encoding="utf-8")
            payload = dict(source_identity=dict(baseline_files=[dict(path=self.bn)], baseline_which="all"))
            draft = BackgroundDraft(kind="bn", files=(generated.name,),
                                    calibration=dict(payload=payload, original=False))
            self.bridge.open(context, draft)
            self.assertEqual(self.bridge._background_files(), (self.bn,))
            self.assertEqual(self.bridge.snapshot["files"], [self.bn])
            self.assertIsNotNone(self.bridge.draft.calibration)
            self.bridge.selectFiles("background", [str(generated)])
            self.assertIn("measurement-specific", self.bridge.snapshot["error"])
            self.assertEqual(self.bridge.draft, draft)
            result = dict(summary="recalibrated", eligible=False, candidates=[], report={})
            with patch("ui_qt.quick.drr_background_workflow.compare_bn", return_value=result) as compare, \
                 patch.object(self.bridge, "_run", side_effect=lambda work, complete, message: complete(work(None))):
                self.bridge.invoke("calibrate-bn")
                self.assertEqual(compare.call_args.args[1:3], ((self.bn,), "all"))

        def test_bn_generation_returns_result_without_mandatory_preview_or_refit(self):
            context = replace(self.context, measurements=(self.sample,))
            self.bridge.open(context, BackgroundDraft(kind="bn", files=(self.bn,), which="all"))
            payload = dict(summary="generated", eligible=True, candidates=[], report={})
            generated = []
            self.bridge.calibrationGenerated.connect(generated.append)
            with patch("ui_qt.quick.drr_background_workflow.compare_bn", return_value=payload) as compare, \
                 patch.object(self.bridge, "_run", side_effect=lambda work, complete, message: complete(work(None))), \
                 patch.object(self.bridge, "_draw_preview") as draw:
                self.bridge.invoke("generate-bn")
                self.assertEqual(generated, [payload])
                draw.assert_not_called()
                self.bridge.invoke("generate-bn")
                compare.assert_called_once()
                self.assertEqual(len(generated), 2)
                self.bridge.invoke("compare-bn")
                compare.assert_called_once()
                draw.assert_called_once()

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
            smoothing = item.findChild(QQuickItem, "drr_background_editor_smoothing")
            intervals = item.findChild(QQuickItem, "drr_background_editor_spline_intervals")
            def key(code, text="", modifier=Qt.NoModifier):
                for event in (QEvent.KeyPress, QEvent.KeyRelease):
                    app.sendEvent(window, QKeyEvent(event, code, modifier, text))
            def type_number(field, text):
                field.forceActiveFocus()
                key(Qt.Key_A, modifier=Qt.ControlModifier)
                for char in text:
                    key(0, char)
                key(Qt.Key_Return, "\r")
                settle()
            type_number(smoothing, "1.23456789e-6")
            self.assertEqual(self.bridge.snapshot["numbers"]["smoothing"], 1.23456789e-6)
            self.assertEqual(smoothing.property("text"), "1.2346e-6")
            intervals.forceActiveFocus()
            settle()
            self.assertEqual(smoothing.property("text"), "1.2346e-6")
            smoothing.forceActiveFocus()
            self.assertEqual(smoothing.property("text"), "1.2346e-6")
            key(Qt.Key_Return, "\r")
            self.assertEqual(self.bridge.snapshot["numbers"]["smoothing"], 1.23456789e-6)
            type_number(smoothing, "1e")
            self.assertTrue(smoothing.property("invalid"))
            self.assertEqual(self.bridge.snapshot["numbers"]["smoothing"], 1.23456789e-6)
            key(Qt.Key_Escape)
            self.assertFalse(smoothing.property("invalid"))
            self.assertEqual(smoothing.property("text"), "1.2346e-6")
            self.assertEqual(self.bridge.snapshot["numbers"]["smoothing"], 1.23456789e-6)
            type_number(intervals, "1.5e1")
            self.assertEqual(self.bridge.snapshot["numbers"]["spline_intervals"], 15)
            type_number(intervals, "1.55e1")
            self.assertTrue(intervals.property("invalid"))
            self.assertEqual(self.bridge.snapshot["numbers"]["spline_intervals"], 15)
            key(Qt.Key_Escape)
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
    if "--cold-settings" in sys.argv:
        raise SystemExit(_cold_settings())
    if "--isolated" in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
