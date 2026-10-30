"""Scientific/native DRR core checks using synthetic, isolated sources only.

This covers loading, differentiation, verified background recipes and exports.
The native advanced background UI has separate isolated-process coverage.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")
os.environ.setdefault("MPLBACKEND", "agg")

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QSettings, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest

from ui_qt.quick.drr_workflow import DrrRequest, export_drr, load_drr, process_drr
from tests.quick_drr_helpers import apply_self_background


class QuickDrrWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QGuiApplication.instance() or QGuiApplication([])

    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="quick-drr-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        environment = dict(MPLCONFIGDIR=str(self.root / "matplotlib"),
                           XDG_CACHE_HOME=str(self.root / "cache"),
                           XDG_CONFIG_HOME=str(self.root / "config"),
                           LOCALAPPDATA=str(self.root / "local"))
        self.environment = patch.dict(os.environ, environment)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        from core import processing_run
        from core.raw_spectrum_cache import RawSpectrumCache
        cache = patch.object(processing_run, "_RAW_SPECTRUM_CACHE", RawSpectrumCache(max_bytes=0))
        cache.start()
        self.addCleanup(cache.stop)

    def csv(self, name="sample_TG_rep01.csv", *, offset=0., wavelengths=None):
        wavelengths = np.linspace(630., 690., 41) if wavelengths is None else np.asarray(wavelengths)
        gate = np.linspace(-2., 2., 9)
        energy_term = np.linspace(-1., 1., len(wavelengths))
        intensity = 130 + offset + 15 * gate[:, None] + 4 * energy_term[None, :] ** 2 + 3 * gate[:, None] * energy_term[None, :]
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "Vbg,Vtg,Vbias," + ",".join(format(value, ".17g") for value in wavelengths)
        rows = [header]
        rows.extend(",".join(format(value, ".17g") for value in [gate_value * .5, gate_value, .1 + gate_value * .02, *values])
                    for gate_value, values in zip(gate, intensity))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    def dat(self, name="imported.dat", *, points=41):
        energy, gate = np.linspace(1.7, 2.1, points), np.linspace(-2., 2., 9)
        values = 3 * gate[:, None] * energy[None, :] + energy[None, :] ** 2 + .2 * gate[:, None] ** 2
        path = self.root / name
        lines = ["Photon energy\t" + "\t".join(format(value, ".17g") for value in gate)]
        lines.extend("\t".join(format(value, ".17g") for value in [x, *values[:, index]]) for index, x in enumerate(energy))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.with_suffix(".metadata.json").write_text(json.dumps({"operation": "DR/R", "plot": {
            "title": "Imported synthetic map", "ylabel": "Bias", "cbar_label": "DR/R", "y_unit": "V"}}), encoding="utf-8")
        return path

    def request(self, files, **kwargs):
        return DrrRequest(tuple(str(path) for path in files), y_axis="tg", window=9, polyorder=2, **kwargs)

    def params(self, cube):
        from core.plotting import HeatmapParams
        finite = cube.Z[np.isfinite(cube.Z)]
        lower, upper = float(finite.min()), float(finite.max())
        if lower == upper:
            upper = lower + 1
        return HeatmapParams(cube.title, "Energy", cube.gate_label, cube.cbar_label, lower, upper,
                             (float(min(cube.energy)), float(max(cube.energy))),
                             (float(min(cube.gate)), float(max(cube.gate))), cmap="RdBu_r")

    def export(self, products, view="pair", **kwargs):
        return export_drr(products, folder=str(self.root), raw_params=self.params(products.raw),
                          second_params=self.params(products.second) if products.second is not None else None,
                          view=view, **kwargs)

    def wait(self, predicate, *, timeout=30):
        # Real PNG writes and font rendering can exceed five seconds on CI.
        # Wait for completion, with a wall-clock bound for a stuck operation.
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            QTest.qWait(5)
        self.fail("Native DRR operation did not finish")

    def bridge(self):
        from ui_qt.quick.application_bridge import QuickApplicationBridge
        settings = QSettings(str(self.root / "settings.ini"), QSettings.IniFormat)
        bridge = QuickApplicationBridge(settings=settings)
        def close():
            bridge.close()
            bridge._executor.shutdown(wait=True)
            self.app.processEvents()
        self.addCleanup(close)
        bridge.selectWorkflow("drr")
        return bridge

    def load_native(self, source, *, bridge=None):
        bridge = bridge or self.bridge()
        bridge.selectFiles([QUrl.fromLocalFile(str(source))])
        self.wait(lambda: not bridge.snapshot["busy"] and not bridge._pending)
        if bridge.snapshot["backgroundRequired"]:
            apply_self_background(bridge, self.wait)
        self.wait_plot(bridge)
        self.assertTrue(bridge.snapshot["canExport"], bridge.snapshot["status"])
        return bridge

    def wait_plot(self, bridge):
        self.wait(lambda: bridge._canvas is not None and bridge._maps
                  and bridge._canvas.frame_generation > 0 and not bridge._plot_timer.isActive()
                  and not bridge._canvas.backend._timer.isActive())

    def tweak_saved_display(self, bridge):
        state = bridge.drrDisplayBridge
        for key, text in (("axis.xmin", "1.82"), ("axis.xmax", "1.94"),
                          ("raw.vmin", "-0.4"), ("raw.vmax", "0.3")):
            self.assertTrue(state.setNumber(key, text), state.error)
        for key in ("fix.axis.xmin", "fix.axis.xmax", "fix.raw.vmin", "fix.raw.vmax", "memory.global"):
            self.assertTrue(state.setFlag(key, True), state.error)
        plasma = next(index for index, (key, _label) in enumerate(state._cmaps) if key == "plasma")
        self.assertTrue(state.setChoice("raw.cmap", plasma))
        self.wait_plot(bridge)
        return state

    def test_self_first_and_last_match_existing_core_bit_for_bit(self):
        from core.data_io import load_drr_self_cube
        files = self.csv(), self.csv("sample_TG_rep02.csv", offset=7)
        before = [path.read_bytes() for path in files]
        for mode, first in (("self-first", True), ("self-last", False)):
            with self.subTest(mode=mode):
                products = load_drr(self.request(files, background_mode=mode))
                expected = load_drr_self_cube(str(self.root), tuple(map(str, files)), use_first_frame=first, y_axis="tg")
                np.testing.assert_array_equal(products.raw.energy, expected.energy)
                np.testing.assert_array_equal(products.raw.gate, expected.gate)
                np.testing.assert_array_equal(products.raw.Z, expected.Z)
                np.testing.assert_array_equal(products.raw.Z[0 if first else -1], np.zeros(41))
                self.assertEqual(products.raw.gate_coordinates.keys(), expected.gate_coordinates.keys())
                for key in expected.gate_coordinates:
                    np.testing.assert_array_equal(products.raw.gate_coordinates[key], expected.gate_coordinates[key])
        self.assertEqual([path.read_bytes() for path in files], before)

    def test_external_baseline_frames_match_existing_core_bit_for_bit(self):
        from core.data_io import load_drr_external_cube
        files = self.csv(), self.csv("sample_TG_rep02.csv", offset=7)
        backgrounds = self.csv("background01.csv", offset=40), self.csv("background02.csv", offset=55)
        for frame in ("first", "last", "all"):
            with self.subTest(frame=frame):
                request = self.request(files, backgrounds=tuple(map(str, backgrounds)), background_mode="external", background_frame=frame)
                result = load_drr(request)
                expected = load_drr_external_cube(str(self.root), tuple(map(str, files)), tuple(map(str, backgrounds)), baseline_which=frame, y_axis="tg")
                np.testing.assert_array_equal(result.raw.Z, expected.Z)
                np.testing.assert_array_equal(result.raw.energy, expected.energy)
                np.testing.assert_array_equal(result.raw.gate, expected.gate)

    def test_external_grid_identity_and_managed_bn_guards_cannot_be_bypassed(self):
        source = self.csv()
        for axis in (np.linspace(630., 690., 40), np.linspace(630., 690., 41)[::-1], np.linspace(630., 690., 41) + .25):
            with self.subTest(points=len(axis), first=float(axis[0])):
                background = self.csv("wrong_grid.csv", offset=40, wavelengths=axis)
                with self.assertRaisesRegex(ValueError, "exact spectral grid"):
                    load_drr(self.request([source], backgrounds=(str(background),), background_mode="external"))
        background = self.csv("background.csv", offset=40)
        with patch("core.drr_bn_store.validate_bn_background_usage", side_effect=ValueError("BN exact group protection")) as guard:
            with self.assertRaisesRegex(ValueError, "BN exact group protection"):
                load_drr(self.request([source], backgrounds=(str(background),), background_mode="external"))
        guard.assert_called_once_with(str(self.root), (str(source),), (str(background),), y_axis="tg")
        managed = self.csv(".drr_bn_calibration/unverified.csv", offset=40)
        with self.assertRaisesRegex(ValueError, "verified|version|calibration"):
            load_drr(self.request([source], backgrounds=(str(managed),), background_mode="external"))

    def test_raw_first_energy_second_and_mixed_derivatives_are_exact_core_outputs(self):
        from core.processing import apply_sg_derivative_energy
        source = self.csv()
        for derivative in (2, ("mixed", 7)):
            with self.subTest(derivative=derivative):
                request = self.request([source], raw_derivative=1, second_derivative=derivative)
                products = load_drr(request)
                expected_raw, raw_window = apply_sg_derivative_energy(products.source, derivative=1, window_length=9, polyorder=2)
                expected_second, second_window = apply_sg_derivative_energy(products.source, derivative=derivative, window_length=9, polyorder=2)
                np.testing.assert_array_equal(products.raw.Z, expected_raw.Z)
                np.testing.assert_array_equal(products.second.Z, expected_second.Z)
                self.assertEqual((products.raw_window, products.second_window), (raw_window, second_window))
                self.assertEqual(products.second.cbar_label, expected_second.cbar_label)

    def test_dat_and_xlsx_are_imported_without_ratio_processing(self):
        from core.data_io import load_drr_map_cube
        from core.loader import load_dat
        from openpyxl import Workbook
        dat = self.dat()
        background = self.csv("unrelated_background.csv", offset=99)
        result = load_drr(self.request([dat], backgrounds=(str(background),), background_mode="external"))
        expected = load_dat(dat)
        np.testing.assert_array_equal(result.raw.Z, expected.Z)
        self.assertEqual(result.raw.gate_label, "Bias")
        self.assertEqual(result.raw.gate_unit, "V")
        self.assertEqual(result.request.background_mode, "imported")
        self.assertEqual(result.request.backgrounds, ())
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "dR_R"
        sheet.append(["0", *expected.gate])
        for index, energy in enumerate(expected.energy):
            sheet.append([energy, *expected.Z[:, index]])
        xlsx = self.root / "precomputed.xlsx"
        workbook.save(xlsx)
        result = load_drr(self.request([xlsx], background_mode="self-first"))
        expected = load_drr_map_cube(str(self.root), xlsx.name, y_axis="tg")
        np.testing.assert_array_equal(result.raw.Z, expected.Z)
        self.assertEqual(result.request.background_mode, "imported")
        with self.assertRaisesRegex(ValueError, "one precomputed"):
            load_drr(self.request([dat, self.csv()]))

    def test_export_dat_values_legacy_recipe_and_mixed_metadata(self):
        source, background = self.csv(), self.csv("background.csv", offset=40)
        request = self.request([source], backgrounds=(str(background),), background_mode="external", background_frame="all", raw_derivative=1, second_derivative=("mixed", 7))
        products = load_drr(request)
        with patch("core.export._save_heatmap_png", side_effect=lambda path, *_args, **_kwargs: Path(path).write_bytes(b"test png")):
            outputs = self.export(products)
        self.assertEqual(set(outputs), {"raw_png", "raw_dat", "second_png", "second_dat"})
        for name, cube, derivative in (("raw", products.raw, 1), ("second", products.second, 2)):
            dat = outputs[name + "_dat"]
            np.testing.assert_allclose(np.loadtxt(dat, skiprows=1)[:, 1:].T, cube.Z, rtol=1e-10, atol=1e-10)
            metadata = json.loads(dat.with_suffix(".metadata.json").read_text(encoding="utf-8"))
            processing = metadata["processing"]
            self.assertEqual(processing["baseline_selection"], "External")
            self.assertEqual(processing["baseline_which"], "all")
            self.assertEqual(processing["derivative_order"], derivative)
            self.assertEqual(processing["savgol_window"], 9)
            self.assertEqual(processing["savgol_polyorder"], 2)
            self.assertEqual({row["role"] for row in metadata["sources"]}, {"measurement", "background"})
            by_role = {row["role"]: row for row in metadata["sources"]}
            self.assertEqual(by_role["measurement"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            if name == "second":
                self.assertEqual(processing["derivative_axes"], "XY")
                self.assertEqual(processing["savgol_window_y"], 7)
                self.assertEqual(processing["y_derivative_coordinate"], cube.gate_label)

    def test_second_only_export_includes_raw_with_derivative_ranges_and_own_colors(self):
        products = load_drr(self.request([self.csv()]))
        raw_params = replace(self.params(products.raw), cmap="plasma", vmin=-.4, vmax=.3)
        second_params = replace(self.params(products.second), xlim=(1.835, 1.925),
                                ylim=(-1., 1.2), cmap="viridis", vmin=-10., vmax=10.)
        original_raw_ranges = raw_params.xlim, raw_params.ylim
        with patch("core.export._save_heatmap_png", side_effect=lambda path, *_args, **_kwargs: Path(path).write_bytes(b"test png")):
            outputs = export_drr(products, folder=str(self.root), raw_params=raw_params,
                                 second_params=second_params, view="second")
        self.assertEqual(set(outputs), {"raw_png", "raw_dat", "second_png", "second_dat"})
        for name, params in (("raw", raw_params), ("second", second_params)):
            metadata = json.loads(outputs[name + "_dat"].with_suffix(".metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["plot"]["xlim"], list(second_params.xlim))
            self.assertEqual(metadata["plot"]["ylim"], list(second_params.ylim))
            self.assertEqual(metadata["plot"]["cmap"], params.cmap)
            self.assertEqual(metadata["plot"]["vmin"], params.vmin)
            self.assertEqual(metadata["plot"]["vmax"], params.vmax)
        self.assertEqual((raw_params.xlim, raw_params.ylim), original_raw_ranges)

    def test_transient_unknown_gate_profile_is_rechecked_for_next_fixed_group(self):
        from core import drr_sources
        from ui_qt.quick import drr_background_workflow as background
        source, next_source = self.csv(), self.csv("next_measurement.csv", offset=7.)
        references = self.csv("unknown_reference_a.csv", offset=40.), self.csv("unknown_reference_b.csv", offset=50.)
        for gate, reference in enumerate(references):
            lines = reference.read_text().splitlines()
            rows = [line.split(",") for line in lines[1:]]
            reference.write_text(lines[0] + "\n" + "\n".join(
                ",".join([str(gate), str(gate), "0", *row[3:]]) for row in rows) + "\n")
        context = background.BackgroundContext(str(self.root), (str(source),), "tg", 9, 2)
        next_context = replace(context, measurements=(str(next_source),))
        draft = background.BackgroundDraft(kind="custom", files=tuple(map(str, references)), which="all", fixed=True)
        actual_profile = drr_sources.inspect_csv_gate_profile
        def transient_unknown(path, **kwargs):
            if Path(path) == references[0]:
                # The gate reader returns this profile after a transient
                # OSError; surrounding content hashes can still succeed.
                return drr_sources.DrrGateProfile(None, None)
            return actual_profile(path, **kwargs)
        with patch.object(drr_sources, "assess_background_gate_files",
                          wraps=drr_sources.assess_background_gate_files) as assess:
            with patch.object(drr_sources, "inspect_csv_gate_profile", side_effect=transient_unknown):
                prepared = background.prepare_application(context, draft)
            self.assertTrue(prepared.recipe.fixed)
            with self.assertRaisesRegex(ValueError, "Different constant background gates"):
                background.prepare_fixed_application(prepared.recipe, next_context)
            self.assertEqual(assess.call_count, 2)

    def test_fixed_background_gate_checks_follow_content_across_measurement_groups(self):
        from core import drr_sources
        from ui_qt.quick import drr_background_workflow as background
        source, next_source = self.csv(), self.csv("next_measurement.csv", offset=7.)
        references = self.csv("reference_a.csv", offset=40.), self.csv("reference_b.csv", offset=50.)
        for reference in references:
            lines = reference.read_text().splitlines()
            rows = [line.split(",") for line in lines[1:]]
            reference.write_text(lines[0] + "\n" + "\n".join(
                ",".join(["0", "0", "0", *row[3:]]) for row in rows) + "\n")
        context = background.BackgroundContext(str(self.root), (str(source),), "tg", 9, 2)
        next_context = replace(context, measurements=(str(next_source),))
        draft = background.BackgroundDraft(kind="custom", files=tuple(map(str, references)), which="all", fixed=True)
        with patch.object(drr_sources, "assess_background_gate_files",
                          wraps=drr_sources.assess_background_gate_files) as assess:
            prepared = background.prepare_application(context, draft)
            following = background.prepare_fixed_application(prepared.recipe, next_context)
            self.assertTrue(following.recipe.fixed)
            self.assertEqual(assess.call_count, 1)
            # A changed constant gate must be rejected even if size/mtime are
            # unchanged; the reusable summary is keyed by actual file bytes.
            changed = references[1]
            stamp = changed.stat()
            changed.write_bytes(changed.read_bytes().replace(b"\n0,0,0,", b"\n1,1,0,"))
            self.assertEqual(changed.stat().st_size, stamp.st_size)
            os.utime(changed, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, "Different constant background gates"):
                background.prepare_fixed_application(prepared.recipe, next_context)
            self.assertEqual(assess.call_count, 2)
            changed.unlink()
            with self.assertRaises(OSError):
                background.prepare_fixed_application(prepared.recipe, next_context)

    def test_recipe_load_shares_content_checks_and_still_rejects_changed_exports(self):
        from ui_qt.quick import drr_background_workflow as background
        source, reference = self.csv(), self.csv("background.csv", offset=40.)
        context = background.BackgroundContext(str(self.root), (str(source),), "tg", 9, 2)
        prepared = background.prepare_application(context,
            background.BackgroundDraft(kind="custom", files=(str(reference),), which="all"))
        request = self.request([source], background_mode="external", backgrounds=prepared.recipe.files,
            experiment_folder=str(self.root), background_recipe=prepared.recipe)
        with patch.object(background, "background_input_signatures",
                          wraps=background.background_input_signatures) as fingerprints:
            products = load_drr(request, prepared_background=prepared)
        self.assertEqual(fingerprints.call_count, 2)
        self.assertEqual(products.input_signatures, background.background_input_signatures(context, prepared.recipe))
        stamp = reference.stat()
        contents = reference.read_bytes()
        reference.write_bytes(contents.replace(b"170", b"171", 1))
        self.assertNotEqual(reference.read_bytes(), contents)
        self.assertEqual(reference.stat().st_size, stamp.st_size)
        os.utime(reference, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "changed since loading"):
            self.export(products)

    def test_recipe_load_checks_changes_during_validation_and_data_read(self):
        from core import data_io
        from ui_qt.quick import drr_background_workflow as background
        source, reference = self.csv(), self.csv("background.csv", offset=40.)
        context = background.BackgroundContext(str(self.root), (str(source),), "tg", 9, 2)
        prepared = background.prepare_application(context,
            background.BackgroundDraft(kind="custom", files=(str(reference),), which="all"))
        request = self.request([source], background_mode="external", backgrounds=prepared.recipe.files,
            experiment_folder=str(self.root), background_recipe=prepared.recipe)
        original = reference.read_bytes()
        stamp = reference.stat()
        def change_bytes(*_args, **_kwargs):
            reference.write_bytes(original.replace(b"170", b"171", 1))
            os.utime(reference, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            self.assertNotEqual(reference.read_bytes(), original)
            self.assertEqual(reference.stat().st_size, stamp.st_size)
        actual_load = data_io.load_drr_resolved_cube
        def changed_during_read(*args, **kwargs):
            result = actual_load(*args, **kwargs)
            change_bytes()
            return result
        for target, effect in (("ui_qt.quick.drr_workflow._validate_managed_settings", change_bytes),
                               ("core.data_io.load_drr_resolved_cube", changed_during_read)):
            with self.subTest(stage=target):
                reference.write_bytes(original)
                os.utime(reference, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                with patch(target, side_effect=effect), self.assertRaisesRegex(ValueError, "changed during loading"):
                    load_drr(request, prepared_background=prepared)

    def test_advanced_heterogeneous_recipe_preserves_exact_products_and_portable_export(self):
        from core.drr_sources import DrrMeasurementAssignment
        from core.drr_background_regions import DrrBackgroundRegion
        from core.data_io import load_drr_resolved_cube
        from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft, prepare_application
        files = self.csv(), self.csv("sample_TG_rep02.csv", offset=7.)
        backgrounds = self.csv("background01.csv", offset=40.), self.csv("background02.csv", offset=55.)
        regional = self.csv("regional.csv", offset=60.)
        context = BackgroundContext(str(self.root), tuple(map(str, files)), "tg", 9, 2)
        assignments = tuple(DrrMeasurementAssignment(source.name, "External", (background.name,), which)
            for source, background, which in zip(files, backgrounds, ("first", "all")))
        assignments = (replace(assignments[0], regions=(DrrBackgroundRegion(1.83, 1.9, (regional.name,), "all"),)), assignments[1])
        prepared = prepare_application(context, BackgroundDraft(kind="pn", saved_assignments=assignments))
        request = self.request(files, background_mode="external", backgrounds=prepared.recipe.files,
            experiment_folder=str(self.root), background_recipe=prepared.recipe)
        products = load_drr(request, prepared_background=prepared)
        expected = load_drr_resolved_cube(str(self.root), context.measurements, assignments, y_axis="tg")
        np.testing.assert_array_equal(products.source.Z, expected.Z)
        self.assertEqual(products.raw_snapshot.assignments, assignments)
        reprocessed = process_drr(products.source, replace(request, raw_derivative=1),
            input_signatures=products.input_signatures, background_references=products.background_references,
            raw_snapshot=products.raw_snapshot)
        self.assertIs(reprocessed.raw_snapshot, products.raw_snapshot)
        with patch("core.export._save_heatmap_png", side_effect=lambda path, *_args, **_kwargs: Path(path).write_bytes(b"test png")):
            outputs = self.export(reprocessed)
        for key in ("raw_dat", "second_dat"):
            metadata = json.loads(outputs[key].with_suffix(".metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["processing"]["numerical_path"], "heterogeneous")
            self.assertEqual(metadata["processing"]["selection_method"], "explicit_per_measurement")
            self.assertEqual(metadata["processing"]["drr_background_assignments"], [item.to_dict() for item in assignments])
            self.assertTrue(all(not Path(row["name"]).is_absolute() for row in metadata["sources"]))
            self.assertEqual({row["name"] for row in metadata["sources"] if row["role"] == "background"},
                {path.name for path in (*backgrounds, regional)})
        backgrounds[1].write_text(backgrounds[1].read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "changed since loading"):
            self.export(products)

    def test_fitted_export_saves_actual_denominator_and_reprocess_retains_reference(self):
        from core.drr_fitted_background import load_fitted_group
        from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft, prepare_application
        files = self.csv(), self.csv("sample_TG_rep02.csv", offset=7.)
        context = BackgroundContext(str(self.root), tuple(map(str, files)), "tg", 9, 2)
        settings = dict(automatic_protection=True, shared_texture=False, method="polynomial", polynomial_degree=2)
        prepared = prepare_application(context, BackgroundDraft(kind="fitted", fitted_result=settings))
        request = self.request(files, background_mode="fitted", experiment_folder=str(self.root), background_recipe=prepared.recipe)
        products = load_drr(request, prepared_background=prepared)
        expected, refs = load_fitted_group(str(self.root), context.measurements, settings, y_axis="tg")
        np.testing.assert_array_equal(products.source.Z, expected.Z)
        np.testing.assert_array_equal(products.background_references[0].background(), refs[0].background())
        reprocessed = process_drr(products.source, replace(request, raw_derivative=1),
            input_signatures=products.input_signatures, background_references=products.background_references,
            raw_snapshot=products.raw_snapshot)
        self.assertIs(reprocessed.background_references[0], products.background_references[0])
        with patch("core.export._save_heatmap_png", side_effect=lambda path, *_args, **_kwargs: Path(path).write_bytes(b"test png")):
            outputs = self.export(reprocessed)
        denominator = outputs["background_0"]
        self.assertTrue(denominator.is_relative_to(self.root))
        np.testing.assert_allclose(np.loadtxt(denominator, skiprows=1)[:, 1:].T, refs[0].background(), rtol=1e-10)
        metadata = json.loads(outputs["raw_dat"].with_suffix(".metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["processing"]["numerical_path"], "fitted_raw_average")
        self.assertEqual(metadata["processing"]["selection_method"], "fitted_sample")
        self.assertEqual(metadata["processing"]["fitted_background_files"], [str(denominator.relative_to(self.root))])
        self.assertEqual(json.loads(denominator.with_suffix(".metadata.json").read_text())["operation"], "Fitted background")

    def test_self_first_and_imported_exports_keep_distinct_recipe_provenance(self):
        for file, mode, expected in ((self.csv(), "self-first", "Self (first frame)"), (self.dat(), "external", "Imported map")):
            with self.subTest(mode=mode):
                products = load_drr(self.request([file], background_mode=mode))
                with patch("core.export._save_heatmap_png", side_effect=lambda path, *_args, **_kwargs: Path(path).write_bytes(b"test png")):
                    outputs = self.export(products, "raw")
                metadata = json.loads(outputs["raw_dat"].with_suffix(".metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["processing"]["baseline_selection"], expected)
                self.assertEqual(metadata["processing"]["background_mode"], products.request.background_mode)

    def test_changed_or_missing_measurement_and_background_block_export_before_writing(self):
        for role, missing in (("measurement", False), ("background", False), ("measurement", True)):
            with self.subTest(role=role, missing=missing):
                source = self.csv()
                background = self.csv("background.csv", offset=40)
                products = load_drr(self.request([source], backgrounds=(str(background),), background_mode="external"))
                path = source if role == "measurement" else background
                if missing:
                    path.unlink()
                else:
                    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                with patch("core.export.export_drr_png_and_dat") as writer:
                    with self.assertRaisesRegex(ValueError, "changed|disappeared|Reload"):
                        self.export(products)
                writer.assert_not_called()

    def test_missing_derivative_pair_and_invalid_view_do_not_partially_export(self):
        source = self.dat(points=3)
        products = load_drr(self.request([source]))
        self.assertIsNone(products.second)
        self.assertIn("at least 5", products.second_error)
        for view in ("pair", "second", "not-a-view"):
            with self.subTest(view=view), patch("core.export.export_drr_png_and_dat") as writer:
                with self.assertRaises(ValueError):
                    self.export(products, view)
                writer.assert_not_called()
        self.assertFalse((self.root / "Processed Data").exists())

    def test_clamped_mixed_y_export_metadata_records_actual_and_requested_windows(self):
        products = load_drr(self.request([self.csv()], second_derivative=("mixed", 21)))
        with patch("core.export._save_heatmap_png", side_effect=lambda path, *_args, **_kwargs: Path(path).write_bytes(b"test png")):
            outputs = self.export(products, "second")
        dat = outputs["second_dat"]
        metadata = json.loads(dat.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["processing"]["savgol_window_y"], 9)
        self.assertEqual(metadata["processing"]["savgol_window_y_requested"], 21)
        self.assertTrue(dat.stem.endswith("_dXdY"))
        self.assertNotIn("_Y9_", dat.name)

    def test_dat_metadata_changes_removal_or_appearance_invalidate_loaded_products(self):
        for change in ("changed", "deleted", "appeared"):
            with self.subTest(change=change):
                source = self.dat()
                sidecar = source.with_suffix(".metadata.json")
                if change == "appeared":
                    sidecar.unlink()
                products = load_drr(self.request([source]))
                if change == "deleted":
                    sidecar.unlink()
                else:
                    sidecar.write_text(json.dumps({"plot": {"ylabel": "Different Y coordinate"}}), encoding="utf-8")
                with patch("core.export.export_drr_png_and_dat") as writer:
                    with self.assertRaisesRegex(ValueError, "changed|disappeared|Reload"):
                        self.export(products, "raw")
                writer.assert_not_called()

    def test_missing_raw_plot_parameters_fail_before_any_pair_output(self):
        products = load_drr(self.request([self.csv()]))
        for view in ("pair", "second"):
            with self.subTest(view=view), patch("core.export.export_drr_png_and_dat") as writer:
                with self.assertRaisesRegex(ValueError, "raw|parameters"):
                    export_drr(products, folder=str(self.root), raw_params=None,
                               second_params=self.params(products.second), view=view)
                writer.assert_not_called()

    def test_native_bridge_failed_new_source_clears_exportable_old_data(self):
        source = self.csv()
        bridge = self.load_native(source)
        self.assertTrue(bridge.snapshot["canExport"])
        invalid = self.root / "invalid.csv"
        invalid.write_text("malformed\n", encoding="utf-8")
        bridge.selectFiles([QUrl.fromLocalFile(str(invalid))])
        self.assertFalse(bridge.snapshot["canExport"])
        self.wait(lambda: not bridge.snapshot["busy"] and not bridge._pending)
        self.assertTrue(bridge.snapshot["backgroundRequired"])
        background = bridge.drrBackgroundBridge
        background.invoke("apply")
        self.wait(lambda: not background.snapshot["busy"] and not bridge._pending)
        self.assertTrue(background.snapshot["error"])
        self.assertFalse(bridge.snapshot["canExport"])
        self.assertIsNone(bridge._drr_products)
        with patch("ui_qt.quick.drr_workflow.export_drr") as writer:
            bridge.invoke("export")
        writer.assert_not_called()

    def test_native_bridge_busy_export_does_not_cancel_loading_or_publish_stale_result(self):
        bridge = self.bridge()
        source = self.csv()
        entered, release = Event(), Event()
        original = load_drr
        def delayed(request, **kwargs):
            entered.set()
            release.wait(5)
            return original(request, **kwargs)
        bridge.selectFiles([QUrl.fromLocalFile(str(source))])
        self.wait(lambda: not bridge.snapshot["busy"] and not bridge._pending)
        self.assertTrue(bridge.snapshot["backgroundRequired"])
        with patch("ui_qt.quick.drr_workflow.load_drr", side_effect=delayed):
            bridge.drrBackgroundBridge.invoke("apply")
            self.wait(entered.is_set)
            sequence = bridge._sequence
            bridge.invoke("export")
            self.assertEqual(bridge._sequence, sequence)
            self.assertTrue(bridge.snapshot["busy"])
            release.set()
            self.wait(lambda: not bridge.snapshot["busy"] and not bridge._pending)
        self.assertEqual(bridge._source, source)
        self.assertTrue(bridge.snapshot["canExport"])

    def test_native_import_reprocessing_preserves_import_mode_and_exact_derivative(self):
        from core.processing import apply_sg_derivative_energy
        bridge = self.bridge()
        source = self.dat()
        bridge.selectFiles([QUrl.fromLocalFile(str(source))])
        self.wait(lambda: not bridge.snapshot["busy"] and not bridge._pending)
        bridge.setParameter("rawDerivative", "first")
        self.wait(lambda: not bridge.snapshot["busy"] and not bridge._pending)
        products = bridge._drr_products
        self.assertEqual(products.request.background_mode, "imported")
        expected, _ = apply_sg_derivative_energy(products.source, derivative=1,
                                                window_length=products.request.window,
                                                polyorder=products.request.polyorder)
        np.testing.assert_array_equal(products.raw.Z, expected.Z)
        self.assertTrue(products.input_signatures)
        self.assertTrue(bridge.snapshot["canExport"])

    def test_plot_product_switch_selects_corresponding_display_tab(self):
        from PySide6.QtCore import QEvent, QObject
        from PySide6.QtQml import QQmlComponent, QQmlEngine
        bridge = self.load_native(self.csv())
        state = bridge._drr_display
        engine = QQmlEngine()
        component = QQmlComponent(engine, QUrl.fromLocalFile(str(
            Path(__file__).resolve().parents[1] / "ui_qt/quick/qml/Display.qml")))
        panel = component.createWithInitialProperties(dict(bridge=state))
        self.assertIsNotNone(panel, str(component.errors()))
        panel.setParent(engine)
        def dispose_display(component=component):
            # Keep both the component context and engine alive through cleanup.
            from shiboken6 import isValid
            if isValid(panel):
                panel.deleteLater()
            self.app.sendPostedEvents(None, QEvent.DeferredDelete)
            engine.deleteLater()
            self.app.sendPostedEvents(None, QEvent.DeferredDelete)
        self.addCleanup(dispose_display)
        tabs = panel.findChild(QObject, "drr_color_tabs")
        self.assertIsNotNone(tabs)
        for view, expected in (("second", 1), ("pair", 1), ("raw", 0), ("pair", 0), ("second", 1)):
            with self.subTest(view=view, tab=expected):
                bridge.setParameter("drrView", view)
                self.wait_plot(bridge)
                self.assertEqual(bridge.snapshot["drrView"], view)
                self.assertEqual(state.snapshot["colorProduct"], expected)
                self.assertEqual(tabs.property("currentIndex"), expected)
        # Editing the other color bank remains independent of the plotted view.
        state.setChoice("colorProduct", 0)
        self.assertEqual(bridge.snapshot["drrView"], "second")
        bridge.setParameter("drrView", "second")
        self.assertEqual(state.snapshot["colorProduct"], 1)
        # An unavailable product cannot change either the plot or its tab.
        bridge.setParameter("drrView", "raw")
        self.wait_plot(bridge)
        bridge._drr_products = replace(bridge._drr_products, second=None, second_error="Unavailable derivative")
        bridge.setParameter("drrView", "second")
        self.assertEqual(bridge.snapshot["drrView"], "raw")
        self.assertEqual(state.snapshot["colorProduct"], 0)

    def test_native_pair_plot_and_gate_linecuts_use_each_exact_scientific_product(self):
        from core.processing import nearest_gate_spectrum
        bridge = self.load_native(self.csv())
        bridge.setParameter("drrView", "pair")
        self.wait(lambda: bridge._canvas is not None and len(bridge._maps) == 2
                  and bridge._canvas.frame_generation > 0 and not bridge._plot_timer.isActive()
                  and not bridge._canvas.backend._timer.isActive())
        self.assertFalse(bridge.snapshot["railVisible"])
        self.assertTrue(bridge.snapshot["controlsVisible"])
        self.assertIs(bridge._plot_cubes[0], bridge._drr_products.raw)
        self.assertIs(bridge._plot_cubes[1], bridge._drr_products.second)
        axes = tuple(axis for axis, _line in bridge._maps)
        bridge.setParameter("gate", 1.7)
        self.app.processEvents()
        for cube, (_axis, line), (_map, gate_line) in zip(bridge._plot_cubes, bridge._spectra, bridge._maps):
            gate, values = nearest_gate_spectrum(cube, 1.7)
            np.testing.assert_array_equal(line.get_xdata(), cube.energy)
            np.testing.assert_array_equal(line.get_ydata(), values)
            np.testing.assert_array_equal(gate_line.get_ydata(), [gate, gate])
        self.assertEqual(tuple(axis for axis, _line in bridge._maps), axes)
        bridge.setParameter("drrView", "second")
        self.wait(lambda: len(bridge._maps) == 1 and not bridge._plot_timer.isActive())
        self.assertIs(bridge._plot_cubes[0], bridge._drr_products.second)

    def test_native_second_only_background_export_saves_both_products_at_visible_ranges(self):
        from core import export as export_core
        from core.loader import load_dat
        from PIL import Image

        bridge = self.load_native(self.csv())
        bridge.setParameter("drrView", "second")
        self.tweak_saved_display(bridge)
        self.wait_plot(bridge)
        products = bridge._drr_products
        self.assertEqual(len(bridge._maps), 1)
        self.assertIs(bridge._plot_cubes[0], products.second)
        axis = bridge._maps[0][0]
        viewport = dict(xlim=[1.835, 1.925], ylim=[-1., 1.2])
        axis.set_xlim(*viewport["xlim"])
        axis.set_ylim(*viewport["ylim"])
        bridge.setParameter("export.pngLayout", "wide")
        bridge.setParameter("export.pngWidth", 1.5)

        entered, release = Event(), Event()
        original_writer = export_core._save_heatmap_png
        def paused_writer(*args, **kwargs):
            entered.set()
            release.wait(10)
            return original_writer(*args, **kwargs)
        with patch.object(export_core, "_save_heatmap_png", side_effect=paused_writer):
            bridge.invoke("export")
            try:
                self.wait(entered.is_set)
                self.assertTrue(bridge.snapshot["busy"])
                self.assertEqual(bridge.snapshot["drrView"], "second")
                self.assertEqual(len(bridge._maps), 1)
            finally:
                release.set()
            # This writes four full-resolution PNGs (both products, original
            # and wide), twice the work of the single-product export below.
            # Hosted Windows runners can exceed the usual 30-second bound.
            self.wait(lambda: not bridge.snapshot["busy"] and not bridge._pending, timeout=90)
        self.assertIn("Export completed", bridge.snapshot["status"])
        directory = self.root / "Processed Data" / "DRR"
        pngs, dats = list(directory.glob("*.png")), list(directory.glob("*.dat"))
        self.assertEqual(len(pngs), 4)
        self.assertEqual(len(dats), 2)
        for path in pngs:
            with Image.open(path) as png:
                self.assertEqual(png.size, (1800, 930) if path.stem.endswith("_1.5x") else (1200, 930))
        for path in dats:
            metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
            expected = products.second if metadata["processing"].get("derivative_order") == 2 else products.raw
            restored = load_dat(path)
            np.testing.assert_allclose(restored.energy, expected.energy)
            np.testing.assert_allclose(restored.gate, expected.gate)
            np.testing.assert_allclose(restored.Z, expected.Z, rtol=1e-9, atol=1e-10)
            self.assertEqual(metadata["plot"]["xlim"], viewport["xlim"])
            self.assertEqual(metadata["plot"]["ylim"], viewport["ylim"])
            self.assertEqual(metadata["ui_state"]["viewport"], viewport)
            self.assertNotIn("display_save_pending", metadata)
            for figure in metadata["figures"].values():
                self.assertEqual(figure["plot"]["xlim"], viewport["xlim"])
                self.assertEqual(figure["plot"]["ylim"], viewport["ylim"])
            if expected is products.raw:
                self.assertEqual(metadata["plot"]["cmap"], "plasma")
                self.assertEqual(metadata["plot"]["vmin"], -.4)
                self.assertEqual(metadata["plot"]["vmax"], .3)
        self.assertEqual(bridge.snapshot["drrView"], "second")
        self.assertEqual(len(bridge._maps), 1)
        self.assertIs(bridge._plot_cubes[0], products.second)
        np.testing.assert_allclose(axis.get_xlim(), viewport["xlim"])
        np.testing.assert_allclose(axis.get_ylim(), viewport["ylim"])

    def test_native_saved_display_roundtrip_restores_file_fixes_colors_and_exported_viewport(self):
        from core.drr_display_settings import INDEX_NAME, find_candidates, source_key
        from core import export as export_core
        from PIL import Image

        source = self.csv("sample_660nmc_TG_rep01.csv")
        bridge = self.load_native(source)
        state = self.tweak_saved_display(bridge)
        self.assertEqual(state.snapshot["flags"]["fix.axis.xmin"]["text"], "Fix M")
        viewport = dict(xlim=[1.835, 1.925], ylim=[-1.0, 1.2])
        bridge._maps[0][0].set_xlim(*viewport["xlim"])
        bridge._maps[0][0].set_ylim(*viewport["ylim"])
        directory = self.root / "Processed Data" / "DRR"
        index_path = directory / INDEX_NAME
        self.assertFalse(index_path.exists())
        self.assertFalse(bridge.settings.contains("drr/display_center_templates"))

        entered, release = Event(), Event()
        original_writer = export_core._save_heatmap_png
        def paused_writer(*args, **kwargs):
            entered.set()
            release.wait(10)
            return original_writer(*args, **kwargs)
        with patch.object(export_core, "_save_heatmap_png", side_effect=paused_writer):
            bridge.invoke("export")
            self.wait(entered.is_set)
            self.assertTrue(bridge.snapshot["busy"])
            self.assertFalse(index_path.exists())
            self.assertFalse(bridge.settings.contains("drr/display_center_templates"))
            release.set()
            self.wait(lambda: not bridge.snapshot["busy"] and not bridge._pending)
        self.assertIn("Export completed", bridge.snapshot["status"])
        self.assertTrue(index_path.exists())
        index = json.loads(index_path.read_text(encoding="utf-8"))
        saved = index["files"][source_key(str(self.root), [source.name])]
        self.assertEqual(saved["ranges"]["xmin"], 1.82)
        self.assertEqual(saved["ranges"]["xmax"], 1.94)
        self.assertEqual(saved["raw"]["cmap"], "plasma")
        self.assertEqual(saved["viewport"], viewport)
        self.assertTrue(saved["fixes"]["xmin"])
        self.assertTrue(bridge.settings.contains("drr/display_center_templates"))
        metadata_path = next(directory.glob("*.metadata.json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["ui_state"], saved)
        self.assertNotIn("display_save_pending", metadata)
        self.assertEqual(metadata["plot"]["xlim"], viewport["xlim"])
        self.assertEqual(metadata["plot"]["ylim"], viewport["ylim"])
        with Image.open(next(directory.glob("*.png"))) as png:
            self.assertGreater(png.width, 300)
            self.assertGreater(png.height, 200)
        bridge.close()

        restored = self.load_native(source)
        restored_state = restored.drrDisplayBridge
        self.assertIsNotNone(restored._drr_products.history_candidates["exact"])
        self.assertEqual(restored_state.snapshot["numbers"]["axis.xmin"]["value"], 1.82)
        self.assertEqual(restored_state.snapshot["numbers"]["axis.xmax"]["value"], 1.94)
        self.assertEqual(restored_state.snapshot["numbers"]["raw.vmin"]["value"], -.4)
        self.assertEqual(restored_state.snapshot["numbers"]["raw.vmax"]["value"], .3)
        self.assertEqual(restored_state.snapshot["flags"]["fix.axis.xmin"]["text"], "Fix F")
        # Explicit color locks now survive restart as manual preferences; they
        # retain priority over the same file's portable display snapshot.
        self.assertEqual(restored_state.snapshot["flags"]["fix.raw.vmin"]["text"], "Fix M")
        params = restored_state.heatmap_params(restored._drr_products.raw)
        self.assertEqual(params.cmap, "plasma")
        np.testing.assert_array_equal(restored._maps[0][0].get_xlim(), viewport["xlim"])
        np.testing.assert_array_equal(restored._maps[0][0].get_ylim(), viewport["ylim"])
        self.assertIn("Restored this file", restored_state.snapshot["texts"]["memoryStatus"]["text"])
        self.assertIsNotNone(find_candidates(str(self.root), [source.name])["exact"])

        restored.close()
        restored.settings.remove("drr/display_memory/manual_ranges")
        restored.settings.sync()
        file_only = self.load_native(source)
        self.assertEqual(file_only.drrDisplayBridge.snapshot["flags"]["fix.raw.vmin"]["text"], "Fix F")
        self.assertEqual(file_only.drrDisplayBridge.snapshot["numbers"]["raw.vmin"]["value"], -.4)

    def test_failed_paired_native_export_does_not_publish_pending_display_history(self):
        from core.drr_display_settings import INDEX_NAME, find_candidates
        from core import export as export_core

        source = self.csv("failed_660nmc_TG_rep01.csv")
        bridge = self.load_native(source)
        state = self.tweak_saved_display(bridge)
        bridge.setParameter("drrView", "pair")
        self.wait_plot(bridge)
        original_writer = export_core.export_drr_png_and_dat
        calls = []
        def fail_second(*args, **kwargs):
            calls.append(kwargs["cube"])
            if len(calls) == 2:
                raise RuntimeError("Synthetic derivative export failure")
            return original_writer(*args, **kwargs)
        with (patch.object(export_core, "export_drr_png_and_dat", side_effect=fail_second),
              patch.object(state, "record_saved_state", wraps=state.record_saved_state) as record_saved):
            bridge.invoke("export")
            self.wait(lambda: not bridge.snapshot["busy"] and not bridge._pending)
        self.assertEqual(len(calls), 2)
        self.assertIn("Synthetic derivative export failure", bridge.snapshot["status"])
        record_saved.assert_not_called()
        directory = self.root / "Processed Data" / "DRR"
        self.assertFalse((directory / INDEX_NAME).exists())
        self.assertFalse(bridge.settings.contains("drr/display_center_templates"))
        partial_metadata = json.loads(next(directory.glob("*.metadata.json")).read_text(encoding="utf-8"))
        self.assertTrue(partial_metadata["display_save_pending"])
        self.assertEqual(partial_metadata["ui_state"]["raw"]["cmap"], "plasma")
        self.assertIsNone(find_candidates(str(self.root), [source.name])["exact"])
        bridge.close()
        restored = self.load_native(source)
        self.assertIsNone(restored._drr_products.history_candidates["exact"])
        self.assertFalse(restored.drrDisplayBridge.snapshot["flags"]["fix.axis.xmin"]["checked"])
        self.assertNotEqual(restored.drrDisplayBridge.heatmap_params(restored._drr_products.raw).cmap, "plasma")


if __name__ == "__main__":
    unittest.main()
