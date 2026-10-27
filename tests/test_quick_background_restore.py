"""Exact raw CSV history restoration uses the saved numerical recipe."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_background_regions import DrrBackgroundRegion
from core.drr_sources import DrrMeasurementAssignment
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope
from ui_qt.quick.drr_background_restore import restore_saved_background
from ui_qt.quick.drr_workflow import DrrRequest, load_drr


class QuickBackgroundRestoreTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="quick-background-restore-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "Processed Data" / "DRR"
        self.output.mkdir(parents=True)
        environment = patch.dict(os.environ, MPLBACKEND="agg",
            MPLCONFIGDIR=str(self.root / "matplotlib"), LOCALAPPDATA=str(self.root / "local"),
            XDG_CONFIG_HOME=str(self.root / "config"), XDG_CACHE_HOME=str(self.root / "cache"))
        environment.start()
        self.addCleanup(environment.stop)
        self.wavelengths = np.linspace(630., 690., 41)
        for name, offset in (("a.csv", 0), ("b.csv", 8), ("old.csv", 30),
                             ("latest.csv", 50), ("other.csv", 70)):
            self.csv(name, offset)

    def csv(self, name, offset=0, wavelengths=None):
        axis = self.wavelengths if wavelengths is None else wavelengths
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = ["Vbg,Vtg,Vbias," + ",".join(format(value, ".17g") for value in axis)]
        for gate in (0, 1, 2):
            intensity = 100 + offset + 3 * gate + np.linspace(0., 1., len(axis)) ** 2
            rows.append(",".join(map(str, (gate, gate, 0, *intensity))))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    def request(self, selected=("a.csv",), **kwargs):
        return DrrRequest(tuple(str(self.root / name) for name in selected),
            experiment_folder=str(self.root), y_axis="tg", window=9, **kwargs)

    def export(self, name, selected=("a.csv",), backgrounds=(), *, stamp=1000,
               assignments=None, baseline="External", which="last", processing=None):
        from core.export import write_export_metadata
        values = dict(baseline_selection=baseline, baseline_which=which)
        if assignments is not None:
            values["drr_background_assignments"] = [item.to_dict() for item in assignments]
        values.update(processing or {})
        path = write_export_metadata(str(self.root), [self.output / f"{name}.dat"],
            operation="DR/R", input_files=[*(("measurement", item) for item in selected),
                                           *(("background", item) for item in backgrounds)],
            processing=values)
        os.utime(path, (stamp, stamp))
        return path

    def managed_bn(self):
        """Real immutable store plus transport-only accepted certificate."""
        from core.drr_bn_store import DrrBnStore, source_records
        from core.drr_sources import portable_source_name
        from tests.bn_joint_fixtures import accepted_joint_report
        selected = ("a.csv", "b.csv", "c.csv", "d.csv")
        self.csv("c.csv", 13)
        self.csv("d.csv", 17)
        self.csv("reference_BNbackground.csv", 40)
        energy = np.sort(1240 / np.round(self.wavelengths, 9))
        original = np.full(len(energy), 140.)
        candidate = original + .25
        identity = dict(schema_version=1, root=str(self.root),
            measurements=source_records(self.root, selected),
            baseline_files=source_records(self.root, ("reference_BNbackground.csv",)),
            baseline_which="all", spectral_columns=self.wavelengths.tolist(),
            y_axis="tg", repeat_order=list(selected), ordering={})
        payload = dict(source_identity=identity, report=accepted_joint_report(9, 2),
            settings=dict(derivative_window=9, derivative_polyorder=2),
            energy=energy, gates=np.arange(3.), original_background=original,
            candidate_background=candidate,
            candidate_backgrounds={"original": original, "local_shift": candidate},
            mean_raw=np.full((3, len(energy)), 114.),
            input_snapshots={"measurement_0": np.full((3, len(energy)), 114.)},
            eligible=True, status="accepted", selected_candidate="local_shift",
            summary="Accepted transport fixture", reasons=[], candidates=[])
        store = DrrBnStore(self.root)
        version = store.save_result(payload)
        store.activate(version, measurements=selected)
        name = portable_source_name(self.root, version.csv_path)
        assignments = tuple(DrrMeasurementAssignment(item, "External", (name,), "last") for item in selected)
        self.export("managed", selected, (name,), assignments=assignments)
        return store, version, selected

    def test_managed_bn_restores_group_bound_calibration_and_exact_background(self):
        from core.loader import load_drr_avg
        store, version, selected = self.managed_bn()
        manifest = store.manifest_path.read_bytes()
        request = restore_saved_background(self.request(selected))
        recipe = request.background_recipe
        self.assertEqual(recipe.draft.kind, "bn")
        self.assertFalse(recipe.fixed)
        self.assertFalse(recipe.draft.calibration["original"])
        self.assertEqual(recipe.draft.calibration["payload"]["version_id"], version.version_id)
        self.assertEqual(recipe.draft.calibration["payload"]["csv_path"], str(version.csv_path))
        self.assertEqual(tuple(item.baseline_files for item in recipe.assignments), (recipe.files,) * len(selected))
        loaded = load_drr(request)
        payload = store.load_result(version)
        expected = load_drr_avg(str(self.root), selected, bg_mode="external", y_axis="tg",
            external_vector=payload["candidate_background"], external_energy=payload["energy"])
        np.testing.assert_array_equal(loaded.source.Z, expected.Z)
        self.assertEqual(store.manifest_path.read_bytes(), manifest)

    def test_restored_managed_bn_stale_certificate_and_copied_marker_still_block(self):
        store, version, selected = self.managed_bn()
        with (self.root / selected[0]).open("a", encoding="utf-8") as stream:
            stream.write("\n")
        request = restore_saved_background(self.request(selected))
        self.assertIsNotNone(request.background_recipe.draft.calibration)
        with self.assertRaisesRegex(ValueError, "current joint"):
            load_drr(request)
        copied = self.root / "copied.csv"
        copied.write_bytes(version.csv_path.read_bytes())
        assignments = tuple(DrrMeasurementAssignment(item, "External", (copied.name,), "last") for item in selected)
        self.export("copied", selected, (copied.name,), assignments=assignments, stamp=2000)
        with self.assertRaisesRegex(ValueError, "no verified version"):
            restore_saved_background(self.request(selected))

    def test_no_exact_group_retains_original_request_without_member_guessing(self):
        self.export("single", backgrounds=("old.csv",))
        request = self.request(("a.csv", "b.csv"))
        with patch("core.drr_saved_background.resolve_saved_group_background") as resolver:
            self.assertIs(restore_saved_background(request), request)
        resolver.assert_not_called()

    def test_exact_history_scan_reuses_identities_only_within_one_request(self):
        from core.drr_sources import DrrSavedRecipe, resolve_source_path
        from ui_qt.quick import drr_background_restore as restoration
        from ui_qt.quick.drr_workflow import background_context
        records = [DrrSavedRecipe(('b.csv', 'a.csv'), (), 'Self (last frame)',
                    'last', str(self.output / f'{index}.metadata.json'), float(index))
                   for index in range(30)]
        context = background_context(self.request(('a.csv', 'b.csv')))
        with patch.object(restoration, '_read_drr_metadata', return_value=({}, records)), \
                patch.object(restoration, 'resolve_source_path', wraps=resolve_source_path) as resolve:
            self.assertEqual(restoration._latest_exact_records(context), (records[-1],))
            self.assertEqual(resolve.call_count, 2)
            self.assertEqual(restoration._latest_exact_records(context), (records[-1],))
            self.assertEqual(resolve.call_count, 4)

    def test_imported_maps_and_empty_request_skip_history(self):
        for files in ((), (str(self.root / "saved.dat"),), (str(self.root / "map.xlsx"),)):
            request = DrrRequest(files)
            with patch("ui_qt.quick.drr_background_restore._read_drr_metadata") as history:
                self.assertIs(restore_saved_background(request), request)
            history.assert_not_called()

    def test_latest_exact_common_background_preserves_frame_and_numeric_result(self):
        from core.data_io import load_drr_external_cube
        self.export("old", ("a.csv", "b.csv"), ("old.csv",), stamp=1000)
        self.export("latest", ("b.csv", "a.csv"), ("latest.csv",), stamp=2000, which="first")
        self.export("newer-single", ("a.csv",), ("other.csv",), stamp=3000)
        request = restore_saved_background(self.request(("a.csv", "b.csv")), auto_bn_compare=True)
        recipe = request.background_recipe
        self.assertEqual((request.background_mode, request.background_frame), ("external", "first"))
        self.assertEqual(recipe.files, ("latest.csv",))
        self.assertEqual(recipe.context.measurements, ("a.csv", "b.csv"))
        self.assertEqual(recipe.draft.files, ("latest.csv",))
        self.assertFalse(recipe.fixed)
        self.assertTrue(recipe.auto_bn_compare)
        result = load_drr(request)
        expected = load_drr_external_cube(str(self.root), ("a.csv", "b.csv"), ("latest.csv",),
            baseline_which="first", y_axis="tg")
        np.testing.assert_array_equal(result.source.Z, expected.Z)

    def test_saved_self_first_is_restored_instead_of_current_self_last(self):
        from core.data_io import load_drr_self_cube
        self.export("self", baseline="Self (first frame)", which="first")
        request = restore_saved_background(self.request())
        self.assertEqual(request.background_mode, "self-first")
        self.assertEqual(request.background_recipe.draft.kind, "self")
        self.assertEqual(request.background_recipe.files, ())
        expected = load_drr_self_cube(str(self.root), ("a.csv",), use_first_frame=True, y_axis="tg")
        np.testing.assert_array_equal(load_drr(request).source.Z, expected.Z)

    def test_heterogeneous_backgrounds_keep_member_grid_and_frame_choices(self):
        from core.data_io import load_drr_resolved_cube
        self.csv("b.csv", 8, self.wavelengths + .5)
        self.csv("other.csv", 70, self.wavelengths + .5)
        assignments = (DrrMeasurementAssignment("a.csv", "External", ("old.csv",), "first"),
                       DrrMeasurementAssignment("b.csv", "External", ("other.csv",), "all"))
        self.export("members", ("a.csv", "b.csv"), ("old.csv", "other.csv"), assignments=assignments)
        request = restore_saved_background(self.request(("a.csv", "b.csv")))
        self.assertEqual([(item.baseline_files, item.baseline_which)
            for item in request.background_recipe.draft.saved_assignments],
            [(("old.csv",), "first"), (("other.csv",), "all")])
        expected = load_drr_resolved_cube(str(self.root), ("a.csv", "b.csv"), assignments, y_axis="tg")
        np.testing.assert_array_equal(load_drr(request).source.Z, expected.Z)
        self.assert_reapply_preserves_members(request, expected)

    def assert_reapply_preserves_members(self, request, expected):
        from dataclasses import replace
        from ui_qt.quick.drr_background_workflow import prepare_application, load_with_recipe
        recipe = request.background_recipe
        prepared = prepare_application(recipe.context, replace(recipe.draft, fixed=True))
        self.assertEqual(prepared.recipe.assignments, recipe.assignments)
        self.assertFalse(prepared.recipe.fixed)
        reapplied = load_with_recipe(recipe.context, prepared.recipe, prepared=prepared)
        np.testing.assert_array_equal(reapplied.cube.Z, expected.Z)

    def test_heterogeneous_measured_bn_reapply_preserves_each_saved_reference(self):
        from core.data_io import load_drr_resolved_cube
        self.csv("BNbackground_one.csv", 40)
        self.csv("BNbackground_two.csv", 60)
        assignments = (DrrMeasurementAssignment("a.csv", "External", ("BNbackground_one.csv",), "first"),
                       DrrMeasurementAssignment("b.csv", "External", ("BNbackground_two.csv",), "all"))
        self.export("members-bn", ("a.csv", "b.csv"),
            ("BNbackground_one.csv", "BNbackground_two.csv"), assignments=assignments)
        request = restore_saved_background(self.request(("a.csv", "b.csv")))
        self.assertEqual(request.background_recipe.draft.kind, "bn")
        expected = load_drr_resolved_cube(str(self.root), ("a.csv", "b.csv"), assignments, y_axis="tg")
        np.testing.assert_array_equal(load_drr(request).source.Z, expected.Z)
        self.assert_reapply_preserves_members(request, expected)

    def test_mixed_self_methods_remain_per_member_assignments(self):
        from core.data_io import load_drr_resolved_cube
        assignments = (DrrMeasurementAssignment("a.csv", "Self (first frame)", (), "first"),
                       DrrMeasurementAssignment("b.csv", "Self (last frame)", (), "last"))
        self.export("mixed-self", ("a.csv", "b.csv"), assignments=assignments)
        request = restore_saved_background(self.request(("a.csv", "b.csv")))
        self.assertEqual(request.background_recipe.draft.kind, "custom")
        self.assertEqual(len(request.background_recipe.draft.saved_assignments), 2)
        expected = load_drr_resolved_cube(str(self.root), ("a.csv", "b.csv"), assignments, y_axis="tg")
        np.testing.assert_array_equal(load_drr(request).source.Z, expected.Z)
        self.assert_reapply_preserves_members(request, expected)

    def test_common_regions_retain_default_and_override_without_flattening_draft(self):
        from core.data_io import load_drr_resolved_cube
        region = DrrBackgroundRegion(1.86, 1.90, ("other.csv",), "all")
        assignments = (DrrMeasurementAssignment("a.csv", "External", ("latest.csv",), "first", regions=(region,)),)
        self.export("regions", backgrounds=("latest.csv", "other.csv"), assignments=assignments)
        request = restore_saved_background(self.request())
        recipe = request.background_recipe
        self.assertEqual(recipe.files, ("latest.csv", "other.csv"))
        self.assertEqual(recipe.draft.files, ("latest.csv",))
        self.assertEqual(recipe.draft.regions, (region,))
        expected = load_drr_resolved_cube(str(self.root), ("a.csv",), assignments, y_axis="tg")
        np.testing.assert_array_equal(load_drr(request).source.Z, expected.Z)

    def test_missing_or_malformed_latest_never_uses_old_background_or_self(self):
        self.export("old", backgrounds=("old.csv",), stamp=1000)
        latest = self.export("latest", backgrounds=("latest.csv",), stamp=2000)
        (self.root / "latest.csv").unlink()
        with self.assertRaisesRegex(ValueError, "recorded baseline missing"):
            restore_saved_background(self.request())
        self.csv("latest.csv", 50)
        payload = json.loads(latest.read_text(encoding="utf-8"))
        payload["processing"]["drr_background_assignments"] = None
        latest.write_text(json.dumps(payload), encoding="utf-8")
        os.utime(latest, (2000, 2000))
        with self.assertRaisesRegex(ValueError, "malformed saved assignment"):
            restore_saved_background(self.request())

    def test_equal_time_conflicting_latest_records_remain_blocked(self):
        self.export("one", backgrounds=("old.csv",))
        self.export("two", backgrounds=("latest.csv",))
        with self.assertRaisesRegex(ValueError, "conflicting saved assignments"):
            restore_saved_background(self.request())

    def test_saved_grid_mismatch_blocks_without_fallback(self):
        self.export("latest", backgrounds=("latest.csv",))
        self.csv("latest.csv", 50, self.wavelengths[::-1])
        with self.assertRaisesRegex(ValueError, "exact native spectral grid"):
            restore_saved_background(self.request())

    def test_fitted_group_restores_settings_and_normal_fit_loading(self):
        from core.drr_fitted_background import BackgroundFitSettings, load_fitted_group
        settings = asdict(BackgroundFitSettings(protected_regions=((1.86, 1.89),),
            method="polynomial", polynomial_degree=2, shared_texture=False))
        self.export("fitted", ("a.csv", "b.csv"), baseline="Fitted (sample)",
            assignments=(), processing={"fitted_background": settings, "numerical_path": "fitted_raw_average"})
        request = restore_saved_background(self.request(("a.csv", "b.csv")))
        self.assertEqual(request.background_mode, "fitted")
        self.assertEqual(request.background_recipe.draft.kind, "fitted")
        self.assertEqual(request.background_recipe.fitted_settings, json.loads(json.dumps(settings)))
        result = load_drr(request)
        self.assertEqual(len(result.background_references), 1)
        expected, _ = load_fitted_group(str(self.root), ("a.csv", "b.csv"), settings, y_axis="tg")
        np.testing.assert_array_equal(result.source.Z, expected.Z)

    def test_fitted_missing_invalid_or_conflicting_settings_are_actionable(self):
        self.export("fitted-missing", baseline="Fitted (sample)", assignments=())
        with self.assertRaisesRegex(ValueError, "no fitting settings"):
            restore_saved_background(self.request())
        self.export("fitted-invalid", baseline="Fitted (sample)", assignments=(), stamp=2000,
            processing={"fitted_background": {"protected_regions": []}})
        with self.assertRaisesRegex(ValueError, "fitted background cannot be restored"):
            restore_saved_background(self.request())
        self.export("external", backgrounds=("latest.csv",), stamp=2000)
        with self.assertRaisesRegex(ValueError, "conflicting or incomplete fitted"):
            restore_saved_background(self.request())

    def test_restoration_honors_worker_cancellation(self):
        self.export("latest", backgrounds=("latest.csv",))
        cancellation = LoadCancellation()
        cancellation.cancel()
        with self.assertRaises(LoadCancelled), load_scope(cancellation):
            restore_saved_background(self.request())


class QuickBackgroundRestoreBridgeTests(unittest.TestCase):
    def test_restored_mapping_editor_in_isolated_gui_process(self):
        with TemporaryDirectory(prefix="quick-restore-editor-") as temporary:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                MPLBACKEND="agg", MPLCONFIGDIR=temporary, LOCALAPPDATA=temporary,
                XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, "-c",
                "from tests.test_quick_background_restore import _bridge_isolated; _bridge_isolated()"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _bridge_isolated():
    from dataclasses import replace
    from threading import Event
    import time
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QGuiApplication
    from ui_qt.quick.drr_background_bridge import DrrBackgroundBridge

    application = QGuiApplication([])

    class Cases(QuickBackgroundRestoreTests):
        def wait_for_bridge(self, bridge):
            deadline = time.monotonic() + 10
            while bridge.snapshot["busy"] and time.monotonic() < deadline:
                application.processEvents()
                time.sleep(.001)
            application.processEvents()
            self.assertFalse(bridge.snapshot["busy"], "raw preview worker timed out")
            self.assertFalse(bridge.snapshot["error"], bridge.snapshot["error"])

        def test_saved_frame_edit_changes_effective_mapping_and_numeric_result(self):
            from core.data_io import load_drr_resolved_cube
            from ui_qt.quick.drr_background_workflow import load_with_recipe, prepare_application
            settings = QSettings(str(self.root / "frames.ini"), QSettings.Format.IniFormat)
            bridge = DrrBackgroundBridge(settings=settings)
            self.addCleanup(bridge.close)
            region = DrrBackgroundRegion(1.86, 1.90, ("latest.csv",), "last")
            scenarios = (
                (DrrMeasurementAssignment("a.csv", "External", ("old.csv",), "first", regions=(region,)),
                 DrrMeasurementAssignment("b.csv", "External", ("other.csv",), "last")),
                (DrrMeasurementAssignment("a.csv", "Self (first frame)", (), "first"),
                 DrrMeasurementAssignment("b.csv", "Self (last frame)", (), "last")),
            )
            for index, assignments in enumerate(scenarios):
                with self.subTest(index=index):
                    files = tuple(dict.fromkeys(name for item in assignments for name in item.baseline_source_files))
                    self.export(f"frame-{index}", ("a.csv", "b.csv"), files,
                        assignments=assignments, stamp=1000 * (index + 1))
                    request = restore_saved_background(self.request(("a.csv", "b.csv")))
                    original = load_drr(request).source.Z.copy()
                    bridge.open(request.background_recipe.context, request.background_recipe.draft)
                    bridge.setChoice("frame", 0)
                    expected_assignments = tuple(replace(item, baseline_which="first",
                        baseline_mode="External" if item.baseline_mode == "External" else "Self (first frame)",
                        regions=tuple(replace(part, baseline_which="first") for part in item.regions))
                        for item in request.background_recipe.assignments)
                    prepared = prepare_application(bridge.context, bridge.draft)
                    self.assertEqual(prepared.recipe.assignments, expected_assignments)
                    self.assertEqual(bridge.snapshot["choices"]["frame"]["current"], 0)
                    actual = load_with_recipe(bridge.context, prepared.recipe).cube.Z
                    expected = load_drr_resolved_cube(str(self.root), ("a.csv", "b.csv"),
                        expected_assignments, y_axis="tg")
                    np.testing.assert_array_equal(actual, expected.Z)
                    self.assertFalse(np.allclose(actual, original))
                    if index == 1:
                        previous = bridge.draft
                        bridge.setChoice("frame", 2)
                        self.assertIs(bridge.draft, previous)
                        self.assertIn("First frame or Last frame", bridge.snapshot["error"])
                        self.assertEqual(bridge.snapshot["choices"]["frame"]["current"], 0)
                    else:
                        bridge.setChoice("frame", 2)
                        all_assignments = tuple(replace(item, baseline_which="all",
                            regions=tuple(replace(part, baseline_which="all") for part in item.regions))
                            for item in expected_assignments)
                        prepared = prepare_application(bridge.context, bridge.draft)
                        self.assertEqual(prepared.recipe.assignments, all_assignments)
                        actual = load_with_recipe(bridge.context, prepared.recipe).cube.Z
                        expected = load_drr_resolved_cube(str(self.root), ("a.csv", "b.csv"),
                            all_assignments, y_axis="tg")
                        np.testing.assert_array_equal(actual, expected.Z)

        def test_raw_preview_rebind_clears_plot_source_and_navigation(self):
            from ui_qt.quick.drr_background_workflow import BackgroundContext
            settings = QSettings(str(self.root / "raw-switch.ini"), QSettings.Format.IniFormat)
            bridge = DrrBackgroundBridge(settings=settings)
            self.addCleanup(bridge.close)
            bridge.open(BackgroundContext(str(self.root), ("a.csv", "b.csv"), "tg"))
            bridge.invoke("page:raw")
            bridge.setChoice("rawSource", 1)
            bridge.invoke("preview-raw")
            self.wait_for_bridge(bridge)
            old_preview, old_blitter = bridge._raw_preview, bridge._raw_gate_blitter
            self.assertIsNotNone(old_preview)
            bridge._navigation.push_current()
            bridge._navigation.pan()
            bridge.setNumber("gate", 2)
            bridge.open(BackgroundContext(str(self.root), ("latest.csv", "other.csv"), "tg"))
            bridge.invoke("page:raw")
            state = bridge.snapshot
            self.assertFalse(state["hasPreview"])
            self.assertEqual(state["texts"]["rawSource"], "")
            self.assertEqual(state["numbers"]["gate"], 0)
            self.assertEqual(state["plotTool"], "")
            self.assertIsNone(bridge._raw_preview_key)
            self.assertIsNone(bridge._raw_gate_blitter)
            self.assertIsNone(old_blitter._draw_cid)
            self.assertFalse(any(axis.lines or axis.images for axis in bridge.canvas.figure.axes))
            self.assertIsNone(bridge._navigation._nav_stack())
            bridge.setChoice("rawSource", 1)
            bridge.invoke("preview-raw")
            self.wait_for_bridge(bridge)
            self.assertIsNot(bridge._raw_preview, old_preview)
            self.assertIn("other.csv", bridge._raw_preview.description)
            self.assertEqual(bridge.snapshot["texts"]["rawSource"], "other.csv")
            self.assertTrue(bridge.snapshot["hasPreview"])
            self.assertTrue(np.all(bridge._raw_preview.cube.Z > old_preview.cube.Z))

        def test_raw_preview_from_previous_context_cannot_complete_after_rebind(self):
            from core.drr_raw_preview import load_preview
            from ui_qt.quick.drr_background_workflow import BackgroundContext
            settings = QSettings(str(self.root / "raw-inflight.ini"), QSettings.Format.IniFormat)
            bridge = DrrBackgroundBridge(settings=settings)
            self.addCleanup(bridge.close)
            bridge.open(BackgroundContext(str(self.root), ("a.csv",), "tg"))
            bridge.invoke("page:raw")
            started, release = Event(), Event()
            completions = []
            bridge._finished.connect(lambda epoch, value, error: completions.append(epoch))
            self.addCleanup(release.set)
            old_preview = load_preview(bridge._make_raw_snapshot(), "measurement", "single", source="a.csv")

            def delayed_preview(*args, **kwargs):
                started.set()
                release.wait(10)
                return old_preview

            with patch("core.drr_raw_preview.load_preview", side_effect=delayed_preview):
                bridge.invoke("preview-raw")
                self.assertTrue(started.wait(5))
                old_epoch = bridge._epoch
                bridge.open(BackgroundContext(str(self.root), ("b.csv",), "tg"))
                bridge.invoke("page:raw")
                self.assertGreater(bridge._epoch, old_epoch)
                release.set()
                # Deliver the worker's queued completion before checking state.
                deadline = time.monotonic() + 5
                while old_epoch not in completions and time.monotonic() < deadline:
                    application.processEvents()
                    time.sleep(.001)
                self.assertIn(old_epoch, completions)
            self.assertIsNone(bridge._raw_preview)
            self.assertFalse(bridge.snapshot["hasPreview"])
            bridge.invoke("preview-raw")
            self.wait_for_bridge(bridge)
            self.assertIn("b.csv", bridge._raw_preview.description)

        def test_managed_bn_editor_disables_keep_and_restores_comparison(self):
            _store, version, selected = self.managed_bn()
            request = restore_saved_background(self.request(selected))
            recipe = request.background_recipe
            settings = QSettings(str(self.root / "managed-editor.ini"), QSettings.Format.IniFormat)
            bridge = DrrBackgroundBridge(settings=settings, auto_compare_supported=True)
            self.addCleanup(bridge.close)
            bridge.open(recipe.context, recipe.draft)
            self.assertFalse(bridge.snapshot["canFix"])
            self.assertFalse(bridge.snapshot["flags"]["fixed"])
            self.assertTrue(bridge.restore_comparison(recipe.draft.calibration["payload"]))
            self.assertEqual(bridge._bn_payload["version_id"], version.version_id)
            bridge.invoke("page:bn")
            self.assertTrue(bridge.snapshot["hasPreview"])

        def test_saved_mapping_editor_keeps_selected_files_and_noop_edits(self):
            settings = QSettings(str(self.root / "editor.ini"), QSettings.Format.IniFormat)
            bridge = DrrBackgroundBridge(settings=settings)
            self.addCleanup(bridge.close)
            self.csv("BNbackground_one.csv", 40)
            self.csv("BNbackground_two.csv", 60)
            scenarios = (
                ("custom", (DrrMeasurementAssignment("a.csv", "External", ("old.csv",), "first"),
                            DrrMeasurementAssignment("b.csv", "External", ("other.csv",), "all"))),
                ("bn", (DrrMeasurementAssignment("a.csv", "External", ("BNbackground_one.csv",), "first"),
                        DrrMeasurementAssignment("b.csv", "External", ("BNbackground_two.csv",), "all"))),
                ("custom", (DrrMeasurementAssignment("a.csv", "Self (first frame)", (), "first"),
                            DrrMeasurementAssignment("b.csv", "Self (last frame)", (), "last"))),
            )
            for index, (kind, assignments) in enumerate(scenarios):
                with self.subTest(kind=kind, index=index):
                    files = tuple(dict.fromkeys(name for item in assignments for name in item.baseline_source_files))
                    self.export(f"editor-{index}", ("a.csv", "b.csv"), files,
                        assignments=assignments, stamp=1000 * (index + 1))
                    request = restore_saved_background(self.request(("a.csv", "b.csv")))
                    recipe = request.background_recipe
                    bridge.open(recipe.context, recipe.draft)
                    state = bridge.snapshot
                    self.assertEqual(state["kind"], kind)
                    self.assertTrue(state["hasSavedAssignments"])
                    self.assertFalse(state["canFix"])
                    self.assertEqual(len(state["mappings"]), 2)
                    self.assertEqual([row["measurement"] for row in state["mappings"]], ["a.csv", "b.csv"])
                    self.assertEqual([row["background"] for row in state["mappings"]],
                        [", ".join(item.baseline_files) if item.baseline_mode == "External" else item.baseline_mode
                         for item in assignments])
                    self.assertEqual(bridge._background_files(), files)
                    self.assertEqual(tuple(state["referencePaths"]), files)
                    self.assertEqual(state["selectedCandidates"] if kind == "bn" else state["files"], list(files))

                    original_draft = bridge.draft
                    bridge._file_selection_session = (recipe.context, kind, bridge._method, "background")
                    bridge._accept_file_selection("drr", [str(self.root / name) for name in files])
                    self.assertIs(bridge.draft, original_draft)
                    bridge._inline_selection_visible = True
                    bridge._inline_selection_session = (recipe.context, kind, bridge._method)
                    bridge._edit_inline_selection(list(files))
                    self.assertIs(bridge.draft, original_draft)
                    bridge.setInlineSelectionVisible(False)

                    bridge._file_selection_session = (recipe.context, kind, bridge._method, "background")
                    bridge._accept_file_selection("drr", [str(self.root / "latest.csv")])
                    self.assertFalse(bridge.draft.saved_assignments)
                    self.assertEqual(bridge._background_files(), ("latest.csv",))
                    self.assertFalse(bridge.snapshot["hasSavedAssignments"])
                    self.assertEqual(bridge.snapshot["mappings"], [])
            application.processEvents()

    suite = unittest.TestSuite([Cases("test_saved_mapping_editor_keeps_selected_files_and_noop_edits"),
        Cases("test_managed_bn_editor_disables_keep_and_restores_comparison"),
        Cases("test_saved_frame_edit_changes_effective_mapping_and_numeric_result"),
        Cases("test_raw_preview_rebind_clears_plot_source_and_navigation"),
        Cases("test_raw_preview_from_previous_context_cannot_complete_after_rebind")])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    unittest.main()
