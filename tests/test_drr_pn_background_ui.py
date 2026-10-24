import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QThreadPool, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QTabWidget

from core.drr_composite_background import compose_background
from core.drr_pn_generator import ALGORITHM_VERSION, default_settings, generate_pn_background
from core.drr_pn_pairing import DrrPnPair
from core.drr_pn_store import DrrPnStore
from ui_qt.drr_pn_background_dialog import (DrrPnBackgroundDialog, load_pn_version,
                                           _assisted_import_dependencies, _reconstruction_import_dependencies)
from ui_qt.theme import install_theme, theme_manager


def _pump_worker_events(app):
    app.processEvents()
    # qWait's native loop can starve Python QRunnable work of the GIL.
    # Yield it explicitly while still delivering queued Qt signals.
    time.sleep(0.005)
    app.processEvents()


class PnBackgroundDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode="light")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)
        self.dialogs = []
        self.windows = []
        self.energy = np.linspace(1.4, 1.9, 181)
        self.pname = "test_1K_1T_RotIn17.5deg_730nmc_0.3sx10_Pbackground.csv"
        self.nname = self.pname.replace("Pbackground", "Nbackground")
        self.sample = self.pname.replace("Pbackground", "sample")
        for name, base in ((self.pname, 100.), (self.nname, 110.), (self.sample, 125.)):
            self.write(name, base)
        self.pair = DrrPnPair((self.pname,), (self.nname,))
        self.store = DrrPnStore(self.root)

    def write(self, name, base):
        wave = 1240 / self.energy[::-1]
        values = base * (1 + .02 * np.sin(self.energy * 130))
        header = "Vbg,Vtg," + ",".join(f"{v:.17g}" for v in wave)
        rows = [f"{gate},0," + ",".join(f"{v:.17g}" for v in values[::-1] * scale)
                for gate, scale in ((0, 1), (1, 1.1))]
        (self.root / name).write_text("\n".join([header, *rows]) + "\n")

    def save(self, *, used=True):
        key = self.store.prepare(self.pair, settings=default_settings(), algorithm_version=ALGORITHM_VERSION)
        version = self.store.save(generate_pn_background(self.root, self.pair).document, key)
        if used:
            # Most legacy-history fixtures represent an explicitly used version;
            # this is not scientific acceptance or a measurement association.
            self.store.mark_used(version.version_id)
        return version

    def dialog(self, **kwargs):
        options = dict(folder=self.root, pool=self.pool, select_files=lambda current, title: current,
                       measurements=(self.sample,))
        options.update(kwargs)
        dialog = DrrPnBackgroundDialog(**options)
        self.dialogs.append(dialog)
        self.wait(dialog)
        return dialog

    def wait(self, dialog, *, timeout=15):
        self.app.processEvents()
        deadline = time.monotonic() + timeout
        while dialog._worker is not None and time.monotonic() < deadline:
            _pump_worker_events(self.app)
        self.assertIsNone(dialog._worker, dialog.status.text())

    def choose(self, dialog):
        dialog._set_pair(self.pair)
        dialog._refresh_versions()
        self.wait(dialog)

    def tearDown(self):
        for widget in (*self.dialogs, *self.windows):
            widget.close()
        self.pool.waitForDone(10000)
        self.app.processEvents()
        for widget in (*self.dialogs, *self.windows):
            widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.temp.cleanup()

    def test_pair_selection_autoloads_saved_version_without_generating_or_applying(self):
        version = self.save()
        dialog = self.dialog()
        with patch("ui_qt.drr_pn_background_dialog.generate_pn_background",
                   side_effect=AssertionError("Selection must not regenerate")):
            self.choose(dialog)
        self.assertEqual(dialog._current.version_id, version.version_id)
        self.assertTrue(dialog.use_btn.isEnabled())
        self.assertEqual(dialog._payload["compatibility_error"], "")
        self.assertEqual(len(dialog._payload["reference_sources"]), 2)
        self.assertIsNone(self.store.find_applied((self.sample,)))
        self.assertIsNone(dialog.selected_version)

    def test_preview_distinguishes_saved_output_from_recorded_source_frames(self):
        version = self.save()
        payload = load_pn_version(self.root, version)
        saved_files = {path: path.read_bytes() for path in (version.csv_path, version.metadata_path)}
        before = compose_background(payload["document"])
        dialog = self.dialog(measurements=())
        for role, which in (("P", "first"), ("N", "last")):
            combo = dialog.frames[role]
            blocked = combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(which))
            combo.blockSignals(blocked)
        dialog._show_preview(payload)
        self.assertIn("Saved composite · single-frame background", dialog.method_label.text())
        self.assertIn("P: all frames averaged; N: all frames averaged", dialog.method_label.text())
        self.assertIn("one generated background spectrum", dialog.method_label.toolTip())
        self.assertEqual(dialog.frames["P"].currentData(), "first")
        self.assertEqual(dialog.frames["N"].currentData(), "last")
        self.assertIs(dialog._current, version)
        for path, content in saved_files.items():
            self.assertEqual(path.read_bytes(), content)
        after = compose_background(payload["document"])
        for expected, actual in zip(before[:2], after[:2]):
            np.testing.assert_array_equal(actual, expected)

    def test_history_preview_reports_recorded_first_and_last_sources(self):
        key = self.store.prepare(self.pair, p_which="first", n_which="last",
                                 settings=default_settings(), algorithm_version=ALGORITHM_VERSION)
        result = generate_pn_background(self.root, self.pair, "first", "last")
        version = self.store.save(result.document, key)
        dialog = self.dialog(measurements=())
        dialog.history.setCurrentIndex(dialog.history.findData(version.version_id))
        self.wait(dialog)
        self.assertEqual(dialog._current.version_id, version.version_id)
        self.assertIn("Saved composite · single-frame background", dialog.method_label.text())
        self.assertIn("P: first frame; N: last frame", dialog.method_label.text())
        self.assertNotIn("all frames averaged", dialog.method_label.text())

    def test_missing_cache_does_not_generate_until_clicked_and_regenerate_keeps_history(self):
        dialog = self.dialog()
        dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("pn_only"))
        self.choose(dialog)
        self.assertFalse(dialog.use_btn.isEnabled())
        self.assertFalse(self.store.directory.exists())
        dialog._generate(False)
        self.wait(dialog, timeout=60)
        self.assertIsNotNone(dialog._current, dialog.status.text())
        first = dialog._current
        dialog._generate(True)
        self.wait(dialog, timeout=60)
        self.assertNotEqual(first.version_id, dialog._current.version_id)
        self.assertTrue(first.csv_path.exists())
        self.assertEqual(len(self.store.list_versions(self.pair)), 2)

    def assisted_stub(self, key, *, result=None):
        request = SimpleNamespace(key=key, training_files=(self.sample,), evaluation_files=(),
                                  settings=default_settings())
        api = SimpleNamespace(
            prepare_assisted_request=Mock(return_value=request),
            generate_assisted_background=Mock(return_value=result),
            assess_assisted_result=Mock(return_value={"status": "diagnostic_only"}),
            save_assessment=Mock(return_value=None),
            matching_sample_context=Mock(return_value=True),
        )
        return api, request

    def test_generation_mode_defaults_and_switch_does_not_generate(self):
        dialog = self.dialog()
        self.choose(dialog)
        self.assertEqual(dialog.generation_mode.currentData(), "saved")
        self.assertIn("main view", dialog.sample_role_label.text())
        self.assertFalse(dialog.generate_btn.isEnabled())
        self.assertFalse(dialog.regenerate_btn.isEnabled())
        self.assertFalse(dialog.transition_spin.isEnabled())
        self.assertIn("Import saved", dialog.status.text())
        before = dialog._context()
        with patch("ui_qt.drr_pn_background_dialog.generate_pn_background") as generate, \
                patch("ui_qt.drr_pn_background_dialog._assisted_api") as assisted:
            dialog._generate(False)
            dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("pn_only"))
        self.assertNotEqual(before, dialog._context())
        self.assertTrue(dialog.generate_btn.isEnabled())
        generate.assert_not_called()
        assisted.assert_not_called()
        self.assertFalse(self.store.directory.exists())
        empty = self.dialog(measurements=())
        self.assertEqual(empty.generation_mode.currentData(), "saved")
        self.assertFalse(empty.generation_mode.model().item(empty.generation_mode.findData("assisted")).isEnabled())

    def reconstruction_dialog(self):
        names = tuple(self.sample.replace("sample", f"sample_rep{index+1}") for index in range(4))
        for index, name in enumerate(names):
            self.write(name, 125+index)
        dialog = self.dialog(measurements=names)
        dialog.advanced_btn.setChecked(True)
        self.choose(dialog)
        dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("reconstruction"))
        return dialog, names

    def time_catalog(self):
        from core.drr_sources import DrrSource
        entries, names = [], {}
        for suffix, stamp in (("001", 1000.), ("002", 2000.)):
            for role, offset in (("P", 0.), ("N", 30.)):
                name = self.pname.replace("Pbackground", f"{role}background_{suffix}")
                self.write(name, 100. if role == "P" else 110.)
                lines = (self.root / name).read_text().splitlines()
                gate = "-5,-5" if role == "P" else "5,5"
                rows = [gate + "," + line.split(",", 2)[2] for line in lines[1:]]
                (self.root / name).write_text("\n".join([lines[0], *(rows * 2)]) + "\n")
                names[role, suffix] = name
                sign = -5. if role == "P" else 5.
                entries.append(DrrSource(name, name, suffix, "2026-10-20", stamp+offset, True,
                    classification="background", gate_varies=False, frame_count=4,
                    gate_grid=((sign, sign),)*4, spectral_grid=tuple(1240/self.energy[::-1]),
                    gate_labels=("Vbg", "Vtg"), grid_complete=True))
        repeat = self.sample.replace("sample", "sample_repeat")
        self.write(repeat, 126.)
        for name in (self.sample, repeat):
            entries.append(DrrSource(name, name, "sample", "2026-10-20", 3000., False,
                metadata_role="measurement", spectral_grid=tuple(1240/self.energy[::-1]), grid_complete=True))
        return entries, names, (self.sample, repeat)

    def select_time_row(self, dialog, role, name):
        row = next(i for i, entry in enumerate(dialog._role_entries[role]) if entry.source.source == name)
        dialog.role_tabs.setCurrentIndex(0 if role == "P" else 1)
        index = dialog.role_models[role].index(row, 0)
        dialog.role_tables[role].setCurrentIndex(index)
        dialog.role_tables[role].clicked.emit(index)

    def test_time_tabs_select_single_files_and_keep_selected_pair_visible(self):
        catalog, names, samples = self.time_catalog()
        dialog = self.dialog(catalog=catalog, measurements=samples, sample_scoped=True)
        self.assertFalse(dialog.advanced_btn.isChecked())
        self.assertEqual([entry.source.source for entry in dialog._role_entries["P"]],
                         [names["P", "002"], names["P", "001"]])
        self.assertEqual(dialog.role_models["P"].horizontalHeaderItem(0).text(), "Modified time")
        self.assertEqual(dialog.role_models["P"].item(0, 2).text(), "4")
        with patch.object(dialog, "_select_files", side_effect=AssertionError("row clicks must not open a picker")):
            self.select_time_row(dialog, "P", names["P", "001"])
            self.wait(dialog)
        self.assertIsNone(dialog.pair)
        self.assertEqual(dialog._n_files, ())
        self.assertFalse(dialog.generate_btn.isEnabled())
        self.select_time_row(dialog, "N", names["N", "002"])
        self.wait(dialog)
        self.assertEqual(dialog.pair, DrrPnPair((names["P", "001"],), (names["N", "002"],)))
        self.assertEqual(dialog.role_models["N"].item(0, 3).data(Qt.UserRole), 1030.)
        self.assertEqual(dialog.role_models["N"].item(1, 3).data(Qt.UserRole), 30.)
        self.assertIn("time difference", dialog.condition_label.text())
        self.assertFalse(dialog.role_tabs.isAncestorOf(dialog.source_labels["P"]))
        self.assertTrue(dialog.generate_btn.isEnabled())
        self.assertIsNone(dialog.selected_version)
        self.assertFalse(self.store.directory.exists())

    def test_either_side_can_be_replaced_or_cleared_without_changing_the_other(self):
        catalog, names, samples = self.time_catalog()
        dialog = self.dialog(catalog=catalog, measurements=samples, sample_scoped=True)
        self.select_time_row(dialog, "P", names["P", "001"])
        self.wait(dialog)
        self.select_time_row(dialog, "N", names["N", "001"])
        self.wait(dialog)
        self.select_time_row(dialog, "P", names["P", "002"])
        self.wait(dialog)
        self.assertEqual(dialog._n_files, (names["N", "001"],))
        self.assertEqual(dialog._p_files, (names["P", "002"],))
        self.assertIn("Known conditions match", dialog.condition_label.text())
        dialog._clear_role("P")
        self.wait(dialog)
        self.assertIsNone(dialog.pair)
        self.assertEqual(dialog._n_files, (names["N", "001"],))
        self.assertFalse(dialog.generate_btn.isEnabled())

    def test_other_conditions_filter_preserves_time_order_and_exposes_conflict(self):
        from dataclasses import replace
        catalog, names, samples = self.time_catalog()
        conflict = replace(catalog[0], source="other_"+catalog[0].source,
            filename=catalog[0].filename.replace("1T_", "9T_"), modified_time=4000.)
        dialog = self.dialog(catalog=[*catalog, conflict], measurements=samples, sample_scoped=True)
        self.assertNotIn(conflict.source, [e.source.source for e in dialog._role_entries["P"]])
        dialog.show_other_conditions.setChecked(True)
        self.assertEqual(dialog._role_entries["P"][0].source.source, conflict.source)
        self.select_time_row(dialog, "P", conflict.source)
        self.wait(dialog)
        self.assertEqual(dialog._n_files, ())
        self.assertIn("measurement conflicts", dialog.role_models["P"].item(0, 1).toolTip())
        other = replace(catalog[1], source="other_"+catalog[1].source,
            filename=catalog[1].filename.replace("1T_", "9T_"), modified_time=4030.)
        other_dialog = self.dialog(catalog=[*catalog, conflict, other], measurements=samples, sample_scoped=True)
        other_dialog.show_other_conditions.setChecked(True)
        self.select_time_row(other_dialog, "P", conflict.source)
        self.wait(other_dialog)
        self.select_time_row(other_dialog, "N", other.source)
        self.wait(other_dialog)
        self.assertIn("Conditions need review", other_dialog.condition_label.text())
        self.assertNotIn("Known conditions match", other_dialog.condition_label.text())

    def test_real_mouse_click_preserves_file_rows_and_selected_index(self):
        catalog, names, samples = self.time_catalog()
        dialog = self.dialog(catalog=catalog, measurements=samples, sample_scoped=True)
        dialog.show()
        self.app.processEvents()
        for role, suffix in (("P", "001"), ("N", "002"), ("P", "002")):
            dialog.role_tabs.setCurrentIndex(0 if role == "P" else 1)
            table = dialog.role_tables[role]
            row = next(i for i, entry in enumerate(dialog._role_entries[role])
                       if entry.source.source == names[role, suffix])
            item = dialog.role_models[role].item(row, 1)
            index = item.index()
            table.scrollTo(index)
            self.app.processEvents()
            QTest.mouseClick(table.viewport(), Qt.LeftButton, pos=table.visualRect(index).center())
            self.wait(dialog)
            self.assertIs(dialog.role_models[role].item(row, 1), item)
            self.assertEqual(table.currentIndex().row(), row)
            self.assertEqual(table.textElideMode(), Qt.ElideNone)
            self.assertEqual(dialog._p_files if role == "P" else dialog._n_files, (names[role, suffix],))
            for selected_role, files in (("P", dialog._p_files), ("N", dialog._n_files)):
                selected_rows = dialog.role_tables[selected_role].selectionModel().selectedRows()
                self.assertEqual([dialog._role_entries[selected_role][index.row()].source.source
                                  for index in selected_rows], list(files))
        self.assertEqual(dialog._n_files, (names["N", "002"],))

    def test_new_role_click_cancels_late_worker_and_does_not_restore_old_pair(self):
        catalog, names, samples = self.time_catalog()
        dialog = self.dialog(catalog=catalog, measurements=samples, sample_scoped=True)
        entered, release = threading.Event(), threading.Event()
        consume = Mock()
        def blocked():
            entered.set()
            release.wait(5)
            return "old result"
        dialog._start(blocked, consume, "Loading old preview")
        self.assertTrue(entered.wait(2))
        cancellation = dialog._worker.cancellation
        self.select_time_row(dialog, "N", names["N", "002"])
        release.set()
        self.wait(dialog)
        self.assertTrue(cancellation.cancelled)
        consume.assert_not_called()
        self.assertEqual(dialog._p_files, ())
        self.assertEqual(dialog._n_files, (names["N", "002"],))
        self.assertIsNone(dialog._current)

    def test_raw_selection_after_association_restores_generate_and_clear_stays_clear(self):
        catalog, names, samples = self.time_catalog()
        version = self.save()
        self.store.mark_used(version.version_id, measurements=samples)
        dialog = self.dialog(catalog=catalog, measurements=samples, sample_scoped=True)
        self.assertEqual(dialog.generation_mode.currentData(), "saved")
        self.select_time_row(dialog, "P", names["P", "001"])
        self.wait(dialog)
        self.assertEqual(dialog.generation_mode.currentData(), "reconstruction")
        self.assertTrue(dialog.generate_btn.isEnabled())
        self.assertIsNone(dialog._current)
        dialog._clear_role("P")
        self.wait(dialog)
        self.assertEqual(dialog._p_files, ())
        self.assertIsNone(dialog._current)
        self.assertEqual(self.store.find_applied(samples).version_id, version.version_id)

    def test_training_repeat_labels_keep_distinct_tails_and_full_path_tooltips(self):
        dialog, names = self.reconstruction_dialog()
        self.assertEqual(dialog._training_files(), (names[0], names[2]))
        self.assertTrue(dialog.training_summary.text().startswith("Sample repeats: automatic"))
        self.assertIn("loaded saved version uses its recorded inputs", dialog.training_summary.toolTip())
        dialog.manual_training_check.setChecked(True)
        for combo in dialog.training_repeats:
            self.assertEqual(tuple(combo.itemData(i) for i in range(combo.count())), names)
            for index, name in enumerate(names):
                self.assertEqual(combo.itemText(index), f"{index+1}. …{Path(name).name[-30:]}")
                self.assertEqual(combo.itemData(index, Qt.ToolTipRole), str((self.root / name).resolve()))
            self.assertEqual(combo.toolTip(), str((self.root / combo.currentData()).resolve()))
        dialog.training_repeats[1].setCurrentIndex(1)
        self.assertEqual(dialog.training_repeats[1].toolTip(), str((self.root / names[1]).resolve()))
        self.assertEqual(dialog.training_repeats[1].currentData(), names[1])

    def test_automatic_repeats_generate_without_opening_advanced(self):
        catalog, sources, samples = self.time_catalog()
        with patch("ui_qt.drr_pn_background_dialog._reconstruction_api") as prepare_api:
            dialog = self.dialog(catalog=catalog, measurements=samples, sample_scoped=True)
            for role in ("P", "N"):
                self.select_time_row(dialog, role, sources[role, "001"])
                self.wait(dialog)
        prepare_api.assert_not_called()
        self.assertEqual(dialog._training_files(), samples)
        self.assertFalse(dialog.advanced_btn.isChecked())
        self.assertFalse(dialog.manual_training_check.isChecked())
        self.assertTrue(dialog.manual_training_controls.isHidden())
        self.assertTrue(all(not combo.isEnabled() for combo in dialog.training_repeats))
        self.assertTrue(dialog.generate_btn.isEnabled())
        api, _, _ = self.reconstruction_stub(samples, pair=dialog.pair)
        with patch("ui_qt.drr_pn_background_dialog._reconstruction_api", return_value=api):
            dialog.generate_btn.click()
            self.wait(dialog)
        self.assertEqual(api.prepare_reconstruction_request.call_args.kwargs["training_files"], samples)
        self.assertIsNotNone(dialog._current)
        self.assertIsNone(dialog.selected_version)
        self.assertIsNone(self.store.find_applied(samples))

    def test_automatic_numbered_repeats_and_optional_manual_round_trip(self):
        ordered = tuple(self.sample.replace("sample", f"sample_rep{index:02}_007") for index in range(1, 4))
        for name in ordered:
            self.write(name, 125.)
        with patch("ui_qt.drr_pn_background_dialog._reconstruction_api") as api:
            dialog = self.dialog(measurements=ordered[::-1], sample_scoped=True)
            self.choose(dialog)
            automatic = (ordered[0], ordered[2])
            self.assertEqual(dialog._training_files(), automatic)
            self.assertEqual(tuple(combo.currentData() for combo in dialog.training_repeats), automatic)
            context = dialog._context()
            # Dormant manual widgets cannot alter automatic inputs or worker identity.
            dialog.training_repeats[0].setCurrentIndex(1)
            self.assertEqual(dialog._training_files(), automatic)
            self.assertEqual(dialog._context(), context)
            dialog.advanced_btn.setChecked(True)
            dialog.manual_training_check.setChecked(True)
            self.assertEqual(dialog._training_files(), (ordered[1], ordered[2]))
            self.assertTrue(dialog.manual_training_controls.isVisibleTo(dialog))
            self.assertTrue(all(combo.isEnabled() for combo in dialog.training_repeats))
            dialog.manual_training_check.setChecked(False)
            self.assertEqual(dialog._training_files(), automatic)
            self.assertEqual(dialog._context(), context)
            self.assertTrue(dialog.manual_training_controls.isHidden())
        api.assert_not_called()
        self.assertFalse(self.store.directory.exists())

    def test_automatic_repeats_explain_missing_group_without_duplicating_sample(self):
        dialog = self.dialog(sample_scoped=True)
        self.choose(dialog)
        self.assertEqual(dialog._training_files(), ())
        self.assertFalse(dialog.generate_btn.isEnabled())
        self.assertIn("two distinct", dialog.training_summary.text())
        self.assertIn("two distinct", dialog.generate_btn.toolTip())
        with patch("ui_qt.drr_pn_background_dialog._reconstruction_api") as api:
            dialog._generate(False)
        api.assert_not_called()

    def test_saved_preview_preserves_automatic_and_manual_choices(self):
        version = self.save()
        dialog, names = self.reconstruction_dialog()
        payload = load_pn_version(self.root, version, names)
        dialog._show_preview(payload)
        self.assertFalse(dialog.manual_training_check.isChecked())
        self.assertEqual(dialog._training_files(), (names[0], names[2]))
        dialog.manual_training_check.setChecked(True)
        dialog.training_repeats[1].setCurrentIndex(1)
        dialog._show_preview(payload)
        self.assertTrue(dialog.manual_training_check.isChecked())
        self.assertEqual(dialog._training_files(), (names[0], names[1]))
        dialog.manual_training_check.setChecked(False)
        self.assertEqual(dialog._training_files(), (names[0], names[2]))
        self.assertIsNone(dialog.selected_version)

    def reconstruction_stub(self, names, *, pair=None):
        from dataclasses import replace
        import json
        pair = pair or self.pair
        training = names[::2] if len(names) >= 3 else names
        evaluation = tuple(name for name in names if name not in training)
        settings = {"sample_analysis": {"training_files": list(training)}}
        algorithm = "pn_raw_conditioned_bounded_texture_v1"
        key = self.store.prepare(pair, settings=settings, algorithm_version=algorithm,
                                 dependencies=training)
        document = generate_pn_background(self.root, pair).document
        source = next(s for s in document.sources if s.key == document.default_source)
        report = json.loads(source.generation_json)
        report.update(algorithm_version=algorithm, settings=settings, scientifically_validated=False,
                      source_models_refitted_from_raw=True,
                      render_diagnostics={"full": {"partial": True, "applied_count": 1,
                          "applied_repairs": [{"opposite_exclusive_indices": [2, 3]}], "repair_plan": [
                          {"apply": False, "reasons": ["shared_or_overlapping_repair_bridge"]}]}})
        generated = replace(source, generation_json=json.dumps(report))
        document = replace(document, sources=tuple(generated if s.key == generated.key else s
                                                   for s in document.sources))
        request = SimpleNamespace(key=key, training_files=training, evaluation_files=evaluation,
                                  settings=settings)
        api = SimpleNamespace(prepare_reconstruction_request=Mock(return_value=request),
                              generate_reconstruction_background=Mock(return_value=SimpleNamespace(document=document)),
                              GENERALIZED_ALGORITHM_VERSION=algorithm)
        return api, request, document

    def test_raw_reconstruction_build_gate_cannot_be_bypassed(self):
        with patch("ui_qt.drr_pn_background_dialog.RECONSTRUCTION_ENABLED", False):
            dialog = self.dialog()
        self.choose(dialog)
        item = dialog.generation_mode.model().item(dialog.generation_mode.findData("reconstruction"))
        self.assertFalse(item.isEnabled())
        self.assertEqual(dialog.generation_mode.currentData(), "saved")
        with patch("ui_qt.drr_pn_background_dialog._reconstruction_api") as api:
            dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("reconstruction"))
            dialog._generate(False)
        api.assert_not_called()
        self.assertFalse(dialog.generate_btn.isEnabled())
        self.assertIn("not enabled", dialog.status.text())

    def test_reconstruction_is_available_only_by_explicit_selection_and_generation(self):
        dialog, _ = self.reconstruction_dialog()
        dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("saved"))
        self.assertFalse(dialog.generate_btn.isEnabled())
        self.assertFalse(dialog.training_controls.isVisibleTo(dialog))
        with patch("ui_qt.drr_pn_background_dialog._reconstruction_api") as api:
            dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("reconstruction"))
            self.assertTrue(dialog.generate_btn.isEnabled())
            self.assertTrue(dialog.training_controls.isVisibleTo(dialog))
            dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("saved"))
        api.assert_not_called()
        self.assertFalse(self.store.directory.exists())
        self.assertIsNone(dialog.selected_version)

    def test_raw_reconstruction_roles_and_input_constraints(self):
        dialog, names = self.reconstruction_dialog()
        self.assertEqual(dialog.generation_mode.currentData(), "reconstruction")
        self.assertEqual(dialog._training_files(), (names[0], names[2]))
        self.assertTrue(dialog.generate_btn.isEnabled())
        self.assertFalse(dialog.transition_spin.isEnabled())
        self.assertIn("2 other selected", dialog.sample_role_label.text())
        dialog.manual_training_check.setChecked(True)
        dialog.training_repeats[1].setCurrentIndex(0)
        self.assertFalse(dialog.generate_btn.isEnabled())
        self.assertIn("two distinct", dialog.sample_role_label.text())
        dialog.training_repeats[1].setCurrentIndex(2)
        dialog.frames["P"].setCurrentIndex(dialog.frames["P"].findData("first"))
        self.wait(dialog)
        self.assertFalse(dialog.generate_btn.isEnabled())
        self.assertIn("All frames", dialog.sample_role_label.text())

    def test_raw_reconstruction_generate_saves_draft_and_passes_exact_roles(self):
        dialog, names = self.reconstruction_dialog()
        api, request, _ = self.reconstruction_stub(names)
        def generate(*_args, **kwargs):
            kwargs["progress"]("source_v12", "start")
            self.assertFalse(kwargs["cancelled"]())
            return api.generate_reconstruction_background.return_value
        api.generate_reconstruction_background.side_effect = generate
        with patch("ui_qt.drr_pn_background_dialog._reconstruction_api", return_value=api):
            dialog._generate(False)
            self.wait(dialog)
        api.prepare_reconstruction_request.assert_called_once_with(
            str(self.root), self.pair, names, training_files=names[::2], method="conditioned_bounded")
        self.assertIs(api.generate_reconstruction_background.call_args.kwargs["request"], request)
        self.assertFalse(api.generate_reconstruction_background.call_args.kwargs["force_source_refit"])
        self.assertTrue(dialog._current.is_draft)
        self.assertIn("P/N reconstruction draft", dialog.method_label.text())
        self.assertIn("1 regional repair(s) deferred", dialog.method_label.text())
        self.assertIn("shared or overlapping repair bridge", dialog.method_label.text())
        self.assertIn("opposite-source evidence at 2 energy point(s)", dialog.method_label.text())
        self.assertIn("did not enter generation", dialog.status.text())
        self.assertIsNone(dialog.selected_version)
        self.assertIsNone(self.store.find_applied(names))

    def test_cached_raw_sources_keep_training_context_and_regenerate_forces_refit(self):
        import json
        from dataclasses import replace
        dialog, names = self.reconstruction_dialog()
        api, request, document = self.reconstruction_stub(names)
        source = next(s for s in document.sources if s.key == document.default_source)
        report = json.loads(source.generation_json)
        report["source_models_refitted_from_raw"] = False
        report["execution"] = {"source_preparation": {"reused": True, "scope": "process"}}
        generated = replace(source, generation_json=json.dumps(report))
        document = replace(document, sources=tuple(generated if s.key == generated.key else s for s in document.sources))
        api.generate_reconstruction_background.return_value = SimpleNamespace(document=document)
        with patch("ui_qt.drr_pn_background_dialog._reconstruction_api", return_value=api):
            dialog._generate(False)
            self.wait(dialog)
            first = dialog._current.version_id
            self.assertTrue(dialog._payload["sample_context_match"])
            self.assertIn("source fits reused", dialog.method_label.text())
            dialog._generate(True)
            self.wait(dialog)
        self.assertTrue(api.generate_reconstruction_background.call_args.kwargs["force_source_refit"])
        self.assertNotEqual(first, dialog._current.version_id)

    def test_reconstruction_enabled_default_preserves_saved_autoload(self):
        version = self.save()
        with patch("ui_qt.drr_pn_background_dialog.RECONSTRUCTION_ENABLED", True):
            dialog = self.dialog()
        with patch("ui_qt.drr_pn_background_dialog._reconstruction_api") as api:
            self.choose(dialog)
        self.assertEqual(dialog.generation_mode.currentData(), "saved")
        self.assertEqual(dialog._current.version_id, version.version_id)
        self.assertTrue(dialog.use_btn.isEnabled())
        self.assertFalse(dialog.generate_btn.isEnabled())  # This fixture has one sample repeat.
        api.assert_not_called()

    def test_reconstruction_import_checks_training_and_implementation_dependencies(self):
        dialog, names = self.reconstruction_dialog()
        api, request, document = self.reconstruction_stub(names)
        version = self.store.save(document, request.key)
        with patch("ui_qt.drr_pn_background_dialog.QFileDialog.getOpenFileName",
                   return_value=(str(version.csv_path), "CSV")), \
                patch("ui_qt.drr_pn_background_dialog._reconstruction_api", return_value=api), \
                patch("ui_qt.drr_pn_background_dialog._reconstruction_import_dependencies",
                      side_effect=ValueError("original training-file fingerprints missing")) as validate:
            dialog._import()
            self.wait(dialog)
        validate.assert_called_once_with(str(self.root), str(version.csv_path), request.settings, api,
                                         pair=self.pair, p_which="all", n_which="all")
        self.assertIn("original training-file fingerprints missing", dialog.status.text())
        self.assertEqual(len(self.store.list_versions(self.pair)), 1)
        self.assertIsNone(self.store.preferred_version(self.pair))

    def public_reconstruction_export(self):
        """Exercise the real public loader/writer; stub only the expensive fitter."""
        from core.drr_composite_background import save_composite_background
        from core.drr_pn_reconstruction.io import generate_from_files
        from core.drr_pn_reconstruction.pipeline import GENERALIZED_ALGORITHM_VERSION, GENERALIZED_PROTOCOL
        for name in (self.pname, self.nname):
            lines = (self.root / name).read_text().splitlines()
            (self.root / name).write_text("\n".join([lines[0], *lines[1:], *lines[1:]]) + "\n")
        training = (self.sample, self.sample.replace("sample", "sample_repeat2"))
        self.write(training[1], 126)
        code = self.root / "test_reconstruction_kernel.py"
        code.write_text("test kernel, not a scientific model", encoding="utf8")
        api = SimpleNamespace(GENERALIZED_ALGORITHM_VERSION=GENERALIZED_ALGORITHM_VERSION,
                              dependency_paths=lambda: (code,))
        def fitted(e, pn, *_args, **_kwargs):
            return {"background": pn[0].copy()}, {
                "algorithm_version": GENERALIZED_ALGORITHM_VERSION, "settings": dict(GENERALIZED_PROTOCOL),
                "source_models_refitted_from_raw": True, "scientifically_validated": False}
        with patch("core.drr_pn_reconstruction.io.dependency_paths", return_value=(code,)), \
                patch("core.drr_pn_reconstruction.pipeline.reconstruct_background", side_effect=fitted):
            result = generate_from_files(self.root, self.pname, self.nname, training, method="generalized")
        path = self.root / "public_native_export.csv"
        save_composite_background(path, result.document)
        return path, result, training, code, api

    def test_public_reconstruction_native_export_import_preserves_dependencies_and_roles(self):
        path, result, training, code, api = self.public_reconstruction_export()
        settings = result.report["settings"]
        deps = _reconstruction_import_dependencies(self.root, path, settings, api,
            pair=self.pair, p_which="all", n_which="all")
        self.assertEqual(deps, (code.resolve(), *((self.root / name).resolve() for name in training)))
        version = self.store.import_saved(path, self.pair, algorithm_version=result.report["algorithm_version"],
                                          settings=settings, dependencies=deps)
        np.testing.assert_array_equal(compose_background(load_pn_version(self.root, version)["document"])[1],
                                      result.arrays["background"])
        self.assertTrue(version.fresh)
        self.assertTrue(version.is_draft)
        self.assertIsNone(self.store.preferred_version(self.pair))
        with self.assertRaisesRegex(ValueError, "original P/N roles"):
            _reconstruction_import_dependencies(self.root, path, settings, api,
                pair=DrrPnPair(self.pair.n_files, self.pair.p_files), p_which="all", n_which="all")
        self.write(training[1], 155)
        with self.assertRaisesRegex(ValueError, "input.*changed"):
            _reconstruction_import_dependencies(self.root, path, settings, api,
                pair=self.pair, p_which="all", n_which="all")
        self.assertFalse(self.store.list_versions(self.pair)[0].fresh)

    def test_public_reconstruction_native_export_rejects_changed_code_and_settings(self):
        path, result, _training, code, api = self.public_reconstruction_export()
        with self.assertRaisesRegex(ValueError, "settings"):
            _reconstruction_import_dependencies(self.root, path, {**result.report["settings"], "extra": 1}, api,
                pair=self.pair, p_which="all", n_which="all")
        code.write_text("changed kernel", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "implementation.*changed"):
            _reconstruction_import_dependencies(self.root, path, result.report["settings"], api,
                pair=self.pair, p_which="all", n_which="all")

    def test_known_reconstruction_import_cannot_bypass_proof_without_flag(self):
        from dataclasses import replace
        import json
        from core.drr_composite_background import save_composite_background
        from core.drr_pn_reconstruction import pipeline
        _path, result, training, code, api = self.public_reconstruction_export()
        dialog = self.dialog()
        self.choose(dialog)
        methods = ((pipeline.ALGORITHM_VERSION, pipeline.PROTOCOL),
                   (pipeline.GENERALIZED_ALGORITHM_VERSION, pipeline.GENERALIZED_PROTOCOL),
                   (pipeline.LOCAL_ALGORITHM_VERSION, pipeline.LOCAL_PROTOCOL),
                   (pipeline.CONDITIONED_ALGORITHM_VERSION, pipeline.CONDITIONED_PROTOCOL))
        for method, protocol in methods:
            for missing_flag in (True, False):
                with self.subTest(method=method, missing_flag=missing_flag):
                    report = {**result.report, "algorithm_version": method, "settings": dict(protocol)}
                    if missing_flag:
                        report.pop("source_models_refitted_from_raw", None)
                    else:
                        report["source_models_refitted_from_raw"] = False
                    generated = replace(result.document.sources[-1], generation_json=json.dumps(report))
                    document = replace(result.document, sources=(*result.document.sources[:-1], generated))
                    path = self.root / f"{method}_{missing_flag}.csv"
                    save_composite_background(path, document)
                    # A complete proof is accepted for all three declared protocols.
                    deps = _reconstruction_import_dependencies(self.root, path, report["settings"], api,
                        pair=self.pair, p_which="all", n_which="all")
                    self.assertEqual(deps, (code.resolve(), *((self.root / name).resolve() for name in training)))
                    # The actual dialog dispatcher must still invoke this validator
                    # when the method flag is missing/false. A changed dependency
                    # must never be re-associated as a fresh generic import.
                    original_code = code.read_text(encoding="utf8")
                    code.write_text("changed implementation", encoding="utf8")
                    try:
                        with patch("ui_qt.drr_pn_background_dialog.QFileDialog.getOpenFileName",
                                   return_value=(str(path), "CSV")), \
                                patch("ui_qt.drr_pn_background_dialog._reconstruction_api", return_value=api), \
                                patch.object(dialog._store, "import_saved") as save:
                            dialog._import()
                            self.wait(dialog)
                        save.assert_not_called()
                        self.assertIn("implementation", dialog.status.text())
                        self.assertIn("changed", dialog.status.text())
                    finally:
                        code.write_text(original_code, encoding="utf8")
        self.assertEqual(self.store.list_versions(self.pair), ())

    def test_unknown_reconstruction_version_is_not_a_generic_fresh_import(self):
        from dataclasses import replace
        import json
        from core.drr_composite_background import save_composite_background
        _path, result, _training, _code, api = self.public_reconstruction_export()
        path = self.root / "unsupported_reconstruction_export.csv"
        report = {**result.report, "algorithm_version": "pn_raw_adaptive_regions_v999",
                  "source_models_refitted_from_raw": False}
        generated = replace(result.document.sources[-1], generation_json=json.dumps(report))
        save_composite_background(path, replace(result.document,
            sources=(*result.document.sources[:-1], generated)))
        dialog = self.dialog()
        self.choose(dialog)
        with patch("ui_qt.drr_pn_background_dialog.QFileDialog.getOpenFileName", return_value=(str(path), "CSV")), \
                patch("ui_qt.drr_pn_background_dialog._reconstruction_api", return_value=api), \
                patch.object(dialog._store, "import_saved") as save:
            dialog._import()
            self.wait(dialog)
        save.assert_not_called()
        self.assertIn("Unsupported reconstruction algorithm", dialog.status.text())
        self.assertEqual(self.store.list_versions(self.pair), ())

    def test_reconstruction_store_export_supports_its_original_dependency_proof(self):
        from dataclasses import replace
        import json
        _path, result, training, code, api = self.public_reconstruction_export()
        settings = {**result.report["settings"], "sample_analysis": {
            "training_files": list(training), "subset": "all gates and full native energy", "y_axis": "auto",
            "split_policy": "explicit-two-training-repeats; all other selected files check only"}}
        report = {**result.report, "settings": settings}
        del report["input_records"]
        del report["implementation_sha256"]
        generated = replace(result.document.sources[-1], generation_json=json.dumps(report))
        document = replace(result.document, sources=(*result.document.sources[:-1], generated))
        key = self.store.prepare(self.pair, algorithm_version=report["algorithm_version"], settings=settings,
                                 dependencies=(code, *training))
        version = self.store.save(document, key)
        deps = _reconstruction_import_dependencies(self.root, version.csv_path, settings, api,
            pair=self.pair, p_which="all", n_which="all")
        self.assertEqual(deps, (code.resolve(), *((self.root / name).resolve() for name in training)))

    def test_raw_reconstruction_reuses_fingerprint_without_refit_or_context_override(self):
        dialog, names = self.reconstruction_dialog()
        api, request, document = self.reconstruction_stub(names)
        version = self.store.save(document, request.key)
        with patch("ui_qt.drr_pn_background_dialog._reconstruction_api", return_value=api):
            dialog._generate(False)
            self.wait(dialog)
        self.assertEqual(dialog._current.version_id, version.version_id)
        api.generate_reconstruction_background.assert_not_called()
        self.assertIn("no fitting", dialog.status.text())
        self.assertIsNone(self.store.last_used(self.pair))

    def test_raw_reconstruction_training_change_cancels_late_worker(self):
        dialog, names = self.reconstruction_dialog()
        dialog.manual_training_check.setChecked(True)
        api, request, _ = self.reconstruction_stub(names)
        entered, release = threading.Event(), threading.Event()
        def prepare(*_args, **_kwargs):
            entered.set()
            release.wait(5)
            return request
        api.prepare_reconstruction_request.side_effect = prepare
        with patch("ui_qt.drr_pn_background_dialog._reconstruction_api", return_value=api):
            dialog._generate(False)
            deadline = time.monotonic()+5
            while not entered.is_set() and time.monotonic()<deadline:
                _pump_worker_events(self.app)
            try:
                self.assertTrue(entered.is_set())
                self.assertFalse(dialog.training_repeats[0].isEnabled())
                dialog.training_repeats[0].setCurrentIndex(1)
            finally:
                release.set()
            self.wait(dialog)
        api.generate_reconstruction_background.assert_not_called()
        self.assertIsNone(dialog._current)
        self.assertEqual(self.store.list_versions(self.pair), ())

    def test_assisted_generate_reuses_request_key_without_regeneration_or_assessment(self):
        version = self.save()
        key = self.store.prepare(self.pair, settings=default_settings(), algorithm_version=ALGORITHM_VERSION)
        api, request = self.assisted_stub(key)
        with patch("ui_qt.drr_pn_background_dialog._assisted_api", return_value=api):
            dialog = self.dialog()
            self.choose(dialog)
            api.prepare_assisted_request.assert_not_called()
            dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("assisted"))
            dialog._generate(False)
            self.wait(dialog, timeout=60)
        self.assertEqual(dialog._current.version_id, version.version_id)
        api.prepare_assisted_request.assert_called_once()
        self.assertEqual(api.prepare_assisted_request.call_args.kwargs["measurements"], (self.sample,))
        api.generate_assisted_background.assert_not_called()
        api.assess_assisted_result.assert_not_called()
        api.save_assessment.assert_not_called()
        self.assertIn("not rerun", dialog.status.text())
        self.assertIn("Training: 1", dialog.sample_role_label.text())
        self.assertIsNone(self.store.find_applied((self.sample,)))

    def test_assisted_assessment_failure_keeps_draft_and_reports_failure(self):
        version = self.save()
        key = self.store.prepare(self.pair, settings=default_settings(), algorithm_version=ALGORITHM_VERSION)
        result = SimpleNamespace(document=load_pn_version(self.root, version)["document"])
        api, request = self.assisted_stub(key, result=result)
        api.assess_assisted_result.side_effect = ValueError("check repeat grid changed")
        with patch("ui_qt.drr_pn_background_dialog._assisted_api", return_value=api):
            dialog = self.dialog()
            self.choose(dialog)
            dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("assisted"))
            dialog._generate(True)
            self.wait(dialog, timeout=60)
        self.assertNotEqual(dialog._current.version_id, version.version_id)
        self.assertTrue(version.csv_path.exists())
        self.assertEqual(len(self.store.list_versions(self.pair)), 2)
        self.assertIn("diagnostics failed", dialog.status.text())
        self.assertIn("check repeat grid changed", dialog.status.text())
        self.assertEqual(api.save_assessment.call_args.args[2]["status"], "error")
        self.assertIsNone(self.store.find_applied((self.sample,)))

    def test_store_preference_is_not_overridden_by_training_context(self):
        older = self.save()
        key = self.store.prepare(self.pair, settings=default_settings(), algorithm_version=ALGORITHM_VERSION)
        document = load_pn_version(self.root, older)["document"]
        newer = self.store.save(document, key, force_new=True)
        api, _ = self.assisted_stub(key)
        api.matching_sample_context.side_effect = lambda version, measurements: version.version_id == newer.version_id
        with patch("ui_qt.drr_pn_background_dialog._assisted_api", return_value=api), \
                patch("core.loader.load_pl", side_effect=AssertionError("History must not load sample spectra")):
            dialog = self.dialog()
            self.choose(dialog)
            self.assertEqual(dialog._current.version_id, older.version_id)
            self.assertIn("not linked to the current training-file context", dialog.status.text())
            dialog.history.setCurrentIndex(dialog.history.findData(newer.version_id))
            self.wait(dialog)
        self.assertEqual(dialog._current.version_id, newer.version_id)
        self.assertTrue(dialog.use_btn.isEnabled())
        api.prepare_assisted_request.assert_not_called()
        api.generate_assisted_background.assert_not_called()

    def test_only_experimental_history_is_not_selected_but_can_be_explicitly_previewed(self):
        draft = self.save(used=False)
        dialog = self.dialog()
        self.choose(dialog)
        self.assertIsNone(dialog._current)
        self.assertFalse(dialog.use_btn.isEnabled())
        self.assertEqual(dialog.generation_mode.currentData(), "saved")
        dialog.history.setCurrentIndex(dialog.history.findData(draft.version_id))
        self.wait(dialog)
        self.assertEqual(dialog._current.version_id, draft.version_id)
        self.assertTrue(dialog.use_btn.isEnabled())
        self.assertIn("draft", dialog.method_label.text())
        self.assertIsNone(self.store.last_used(self.pair))
        self.assertIsNone(self.store.find_applied((self.sample,)))

    def test_mode_change_cancels_pending_assisted_result_without_selecting_it(self):
        version = self.save()
        key = self.store.prepare(self.pair, settings=default_settings(), algorithm_version=ALGORITHM_VERSION)
        api, request = self.assisted_stub(key)
        entered, release = threading.Event(), threading.Event()
        def prepare(*args, **kwargs):
            entered.set()
            release.wait(5)
            return request
        api.prepare_assisted_request.side_effect = prepare
        with patch("ui_qt.drr_pn_background_dialog._assisted_api", return_value=api):
            dialog = self.dialog()
            self.choose(dialog)
            old_payload = dialog._payload
            dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("assisted"))
            dialog._generate(False)
            deadline = time.monotonic() + 5
            while not entered.is_set() and time.monotonic() < deadline:
                _pump_worker_events(self.app)
            try:
                self.assertTrue(entered.is_set())
                self.assertFalse(dialog.generation_mode.isEnabled())
                dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("pn_only"))
            finally:
                release.set()
            self.wait(dialog)
        self.assertIs(dialog._payload, old_payload)
        self.assertEqual(dialog._current.version_id, version.version_id)
        api.generate_assisted_background.assert_not_called()
        self.assertIsNone(dialog.selected_version)

    def test_explicit_pn_only_generate_never_prepares_samples(self):
        dialog = self.dialog()
        dialog._set_pair(self.pair)
        dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("pn_only"))
        api = SimpleNamespace(matching_sample_context=Mock(return_value=False),
                              prepare_assisted_request=Mock(side_effect=AssertionError("PN only")))
        with patch("ui_qt.drr_pn_background_dialog._assisted_api", return_value=api):
            dialog._generate(False)
            self.wait(dialog, timeout=60)
        api.prepare_assisted_request.assert_not_called()
        self.assertEqual(dialog._current.algorithm_version, ALGORITHM_VERSION)
        self.assertIn("P/N-only draft", dialog.status.text())

    def test_real_assisted_generate_saves_separate_diagnostics_without_applying(self):
        from core.drr_pn_assisted import ASSISTED_ALGORITHM_VERSION
        dialog = self.dialog()
        dialog._set_pair(self.pair)
        dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("assisted"))
        dialog._generate(False)
        self.wait(dialog, timeout=60)
        self.assertIsNotNone(dialog._current, dialog.status.text())
        self.assertEqual(dialog._current.algorithm_version, ASSISTED_ALGORITHM_VERSION)
        self.assertIn("Training: 1", dialog.sample_role_label.text())
        self.assertIn("No withheld files", dialog.sample_role_label.text())
        self.assertIn("not scientific acceptance", dialog.status.text())
        self.assertEqual(len(list((self.store.directory / "Evaluations").glob("*.json"))), 1)
        self.assertIsNone(self.store.find_applied((self.sample,)))

    def test_assisted_import_preserves_training_dependencies_and_rejects_changed_input(self):
        import hashlib
        import json
        from core.drr_composite_background import composite_metadata_path
        version = self.save()
        metadata_path = composite_metadata_path(version.csv_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["pn_generation"]["dependencies"] = [
            {"path": self.sample, "sha256": hashlib.sha256((self.root / self.sample).read_bytes()).hexdigest()}]
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        settings = {"sample_analysis": {"training_files": [self.sample]}}
        api = SimpleNamespace(dependency_paths=lambda: ())
        deps = _assisted_import_dependencies(self.root, version.csv_path, settings, api)
        self.assertEqual(deps, ((self.root / self.sample).resolve(),))
        self.write(self.sample, 130.)
        with self.assertRaisesRegex(ValueError, "missing, changed"):
            _assisted_import_dependencies(self.root, version.csv_path, settings, api)
        with self.assertRaisesRegex(ValueError, "original training-file fingerprints"):
            _assisted_import_dependencies(self.root, version.csv_path, {}, api)

    def test_assisted_import_rejects_missing_or_changed_implementation_fingerprints(self):
        import hashlib
        import json
        from core.drr_composite_background import composite_metadata_path
        version = self.save()
        code = self.root / "assisted_wrapper.py"
        code.write_text("original implementation", encoding="utf-8")
        metadata_path = composite_metadata_path(version.csv_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["pn_generation"]["dependencies"] = [
            {"path": self.sample, "sha256": hashlib.sha256((self.root / self.sample).read_bytes()).hexdigest()},
            {"path": "old-machine/core/assisted_wrapper.py", "sha256": hashlib.sha256(code.read_bytes()).hexdigest()}]
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        settings = {"sample_analysis": {"training_files": [self.sample]}}
        api = SimpleNamespace(dependency_paths=lambda: (code,))
        self.assertEqual(_assisted_import_dependencies(self.root, version.csv_path, settings, api)[0], code.resolve())
        code.write_text("different implementation", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "original implementation"):
            _assisted_import_dependencies(self.root, version.csv_path, settings, api)
        metadata["pn_generation"]["dependencies"].pop()
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "original implementation"):
            _assisted_import_dependencies(self.root, version.csv_path, settings, api)

    def test_preview_discloses_sample_fallback_and_full_unresolved_range(self):
        import json
        from dataclasses import replace
        version = self.save()
        payload = load_pn_version(self.root, version)
        document = payload["document"]
        source = next(s for s in document.sources if s.key == document.default_source)
        report = json.loads(source.generation_json)
        report.update(samples_analyzed=True, assistance_applied=False,
                      fallback_reason="no_consistent_matching_witness_protection_only",
                      degradation_reasons=["insufficient_repeat_support"],
                      unresolved_regions=[[float(source.energy[0]), float(source.energy[-1])]],
                      render_diagnostics={"full": {"partial": False, "no_supported_repairs": True,
                                                    "applied_count": 0, "repair_plan": []}})
        changed = replace(source, generation_json=json.dumps(report))
        payload["document"] = replace(document, sources=tuple(
            changed if s.key == source.key else s for s in document.sources))
        dialog = self.dialog(measurements=())
        dialog._show_preview(payload)
        dialog._controls()
        self.assertIn("Samples analyzed; no supported sample-driven change", dialog.method_label.text())
        self.assertIn("No consistent sample-matching evidence", dialog.method_label.toolTip())
        self.assertIn("insufficient repeat support", dialog.method_label.toolTip())
        self.assertIn("full-range unresolved", dialog.method_label.text())
        self.assertIn("No regional repair supported", dialog.method_label.text())
        self.assertNotIn("no_consistent_matching", dialog.method_label.text())
        self.assertLessEqual(dialog.method_label.text().count("\n"), 1)
        self.assertTrue(dialog.use_btn.isEnabled())
        dialog.resize(1120, 850)
        dialog.show()
        self.app.processEvents()
        self.assertLessEqual(dialog.height(), 850)
        for widget in (dialog.method_label, dialog.canvas, dialog.status, dialog.use_btn):
            if dialog.controls_scroll.widget().isAncestorOf(widget):
                dialog.controls_scroll.ensureWidgetVisible(widget)
                self.app.processEvents()
            point = widget.mapTo(dialog, widget.rect().topLeft())
            self.assertLessEqual(point.y() + widget.height(), dialog.height())

    def test_background_preview_does_not_prepare_measurement_maps_or_derivatives(self):
        version = self.save()
        with patch("core.drr_fitted_background.prepare_fitted_group",
                   side_effect=AssertionError("Measurements belong to the main view")), \
                patch("core.processing.apply_sg_derivative_energy",
                      side_effect=AssertionError("Derivatives belong to the main view")):
            payload = load_pn_version(self.root, version, (self.sample,))
        e, b, _ = compose_background(payload["document"])
        self.assertEqual(len(b), len(self.energy))
        self.assertEqual(payload["compatibility_error"], "")
        self.assertNotIn("maps", payload)

    def test_history_can_open_without_original_sources_but_requires_explicit_snapshot_use(self):
        version = self.save()
        (self.root / self.pname).unlink()
        dialog = self.dialog()
        dialog.history.setCurrentIndex(dialog.history.findData(version.version_id))
        self.wait(dialog)
        self.assertEqual(dialog._current.status, "missing_sources")
        self.assertEqual(dialog.pair, self.pair)
        self.assertFalse(dialog.use_btn.isEnabled())
        dialog.historical.setChecked(True)
        self.assertTrue(dialog.use_btn.isEnabled())
        dialog._use()
        self.wait(dialog)
        self.assertEqual(dialog.result(), QDialog.Accepted)
        self.assertTrue(dialog.allow_stale)
        self.assertIsNone(self.store.find_applied((self.sample,)))

    def test_incompatible_complete_grid_disables_use_instead_of_truncating_preview(self):
        version = self.save()
        text = (self.root / self.sample).read_text().splitlines()
        (self.root / self.sample).write_text("\n".join(",".join(line.split(",")[:-1]) for line in text) + "\n")
        dialog = self.dialog()
        self.choose(dialog)
        self.assertIsNotNone(dialog._current)
        self.assertTrue(dialog._payload["compatibility_error"])
        self.assertIn("complete measurement grid", dialog.status.text())
        self.assertFalse(dialog.use_btn.isEnabled())

    def test_close_during_worker_does_not_select_or_apply_late_result(self):
        version = self.save()
        dialog = self.dialog()
        entered, release = threading.Event(), threading.Event()
        original = dialog._preview
        def slow(v):
            entered.set()
            release.wait(5)
            return original(v)
        with patch.object(dialog, "_preview", side_effect=slow):
            dialog._set_pair(self.pair)
            dialog._refresh_versions()
            deadline = time.monotonic() + 5
            while not entered.is_set() and time.monotonic() < deadline:
                _pump_worker_events(self.app)
            try:
                self.assertTrue(entered.is_set())
                dialog.reject()
            finally:
                release.set()
            self.wait(dialog)
        self.assertIsNone(dialog.selected_version)
        self.assertIsNone(dialog._current)
        self.assertIsNone(self.store.find_applied((self.sample,)))

    def test_background_dialog_has_no_duplicate_map_tabs_or_display_controls(self):
        dialog = self.dialog()
        self.assertEqual(dialog.findChildren(QTabWidget), [dialog.role_tabs])
        self.assertEqual(dialog.role_tabs.count(), 2)
        self.assertFalse(dialog.advanced_btn.isChecked())
        self.assertFalse(hasattr(dialog, "contrast"))
        self.assertFalse(hasattr(dialog, "range_spins"))
        self.assertIn("main view", dialog.method_label.text())

    def main_window(self):
        from ui_qt.main_window import MainWindow
        settings = QSettings(str(self.root / "main-window.ini"), QSettings.IniFormat)
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None), \
                patch.object(MainWindow, "_schedule_automatic_update_check", lambda _self: None), \
                patch("ui_qt.main_window.QSettings", return_value=settings), \
                patch("ui_qt.presentation_widget.QSettings", return_value=settings):
            window = MainWindow()
        self.windows.append(window)
        return window

    def test_cancel_build_pn_preserves_assignments(self):
        from core.drr_sources import DrrMeasurementAssignment
        window = self.main_window()
        old = (DrrMeasurementAssignment("sample.csv", baseline_mode="External",
                                      baseline_files=("old.csv",), baseline_which="all"),)
        window._drr_assignments = old
        window._drr_assignments_automatic = True
        window.drr_baseline_files_manual = ["old.csv"]
        window.current_folder = str(self.root)
        with patch("ui_qt.drr_background_workspace.DrrBackgroundWorkspace") as factory, \
                patch.object(window, "_start_load") as load:
            factory.return_value.exec.return_value = QDialog.Rejected
            window.drr_controller._edit_drr_pn_background()
        self.assertEqual(window._drr_assignments, old)
        self.assertTrue(window._drr_assignments_automatic)
        self.assertEqual(window.drr_baseline_files_manual, ["old.csv"])
        load.assert_not_called()

    def test_accepted_pn_picker_routes_one_version_through_guarded_apply(self):
        window = self.main_window()
        version = self.save()
        window.drr_baseline_files_manual = ["old.csv"]
        from ui_qt.drr_background_workspace import BackgroundSelection
        window.current_folder = str(self.root)
        with patch("ui_qt.drr_background_workspace.DrrBackgroundWorkspace") as factory, \
                patch("ui_qt.drr_background_workspace_workflow.apply_pn_version") as apply, \
                patch.object(window, "_start_load") as load:
            factory.return_value.exec.return_value = QDialog.Accepted
            factory.return_value.selection = BackgroundSelection(version=version, allow_stale=True)
            window.drr_controller._edit_drr_pn_background()
        apply.assert_called_once_with(window, version, allow_stale=True)
        self.assertEqual(window.drr_baseline_files_manual, ["old.csv"])
        load.assert_not_called()

    def test_minimum_layout_themes_and_keyboard(self):
        dialog = self.dialog(measurements=())
        dialog.advanced_btn.setChecked(True)
        dialog.resize(dialog.minimumSize())
        dialog.show()
        manager = theme_manager()
        self.addCleanup(manager.set_mode, "light")
        for mode in ("light", "dark"):
            manager.set_mode(mode)
            self.app.processEvents()
            dialog.canvas.draw()
            self.assertEqual(dialog.canvas._resolved_theme.name, mode)
            self.assertEqual(dialog.styleSheet(), "")
            for widget in (dialog.table, dialog.history, dialog.use_btn, dialog.close_btn,
                           dialog.generate_btn, dialog.frames["P"], dialog.transition_spin,
                           dialog.generation_mode, dialog.sample_role_label):
                if dialog.controls_scroll.widget().isAncestorOf(widget):
                    dialog.controls_scroll.ensureWidgetVisible(widget)
                    self.app.processEvents()
                point = widget.mapTo(dialog, widget.rect().topLeft())
                self.assertGreaterEqual(point.x(), 0)
                self.assertGreaterEqual(point.y(), 0)
                self.assertLessEqual(point.x() + widget.width(), dialog.width())
                self.assertLessEqual(point.y() + widget.height(), dialog.height())
        dialog.table.setFocus()
        QTest.keyClick(dialog.table, Qt.Key_Tab)
        self.assertIsNot(self.app.focusWidget(), dialog.table)
        self.assertIsNone(dialog._worker)
        self.assertFalse(self.store.directory.exists())
        QTest.keyClick(dialog, Qt.Key_Escape)
        self.assertTrue(dialog._closed)

    def test_reconstruction_minimum_layout_themes_and_keyboard(self):
        dialog, _ = self.reconstruction_dialog()
        dialog.manual_training_check.setChecked(True)
        dialog.resize(dialog.minimumSize())
        dialog.show()
        manager = theme_manager()
        self.addCleanup(manager.set_mode, "light")
        for mode in ("light", "dark"):
            manager.set_mode(mode)
            self.app.processEvents()
            for widget in (*dialog.training_repeats, dialog.training_controls, dialog.sample_role_label,
                           dialog.generate_btn, dialog.use_btn, dialog.close_btn, dialog.canvas):
                if dialog.controls_scroll.widget().isAncestorOf(widget):
                    dialog.controls_scroll.ensureWidgetVisible(widget)
                    self.app.processEvents()
                point = widget.mapTo(dialog, widget.rect().topLeft())
                self.assertGreaterEqual(point.x(), 0)
                self.assertGreaterEqual(point.y(), 0)
                self.assertLessEqual(point.x() + widget.width(), dialog.width())
                self.assertLessEqual(point.y() + widget.height(), dialog.height())
        first, second = dialog.training_repeats
        first.setFocus()
        QTest.keyClick(first, Qt.Key_Tab)
        self.assertIs(self.app.focusWidget(), second)
        self.assertTrue(second.accessibleName())
        self.assertIsNone(dialog._worker)


    def test_region_table_loads_real_metadata_and_clears_with_selection(self):
        version = self.save()
        dialog = self.dialog()
        self.choose(dialog)
        self.assertGreater(dialog.region_model.rowCount(), 0)
        self.assertFalse(dialog.region_table.isHidden())
        self.assertIn("native-grid check passed", dialog.region_scope_label.text())
        self.assertIn("Not scientifically validated", dialog.region_scope_label.text())
        self.assertIn("not scientific acceptance", dialog.region_scope_label.text())
        self.assertTrue(dialog.use_btn.isEnabled())
        self.assertIsNone(dialog.selected_version)
        self.assertIsNone(self.store.find_applied((self.sample,)))
        for row in range(dialog.region_model.rowCount()):
            self.assertFalse(dialog.region_model.item(row, 0).isEditable())
        dialog._invalidate()
        self.assertEqual(dialog.region_model.rowCount(), 0)
        self.assertTrue(dialog.region_table.isHidden())
        self.assertIn("Load a saved version", dialog.region_scope_label.text())
        self.assertIsNone(self.store.find_applied((self.sample,)))

    def test_region_table_missing_and_optional_records_do_not_infer_acceptance(self):
        import json
        from dataclasses import replace
        version = self.save()
        dialog = self.dialog()
        payload = load_pn_version(self.root, version)
        document = payload["document"]
        source = next(s for s in document.sources if s.key == document.default_source)
        for report in ({}, dict(scientifically_validated=True,
            unsupported_regions=[[1.5, 1.6]],
            region_applicability=dict(schema_version=1, regions=[dict(
                energy_bounds=[1.6, 1.7], source="P/N common fine", status="supported_candidate",
                correction_applied=True, reason="Conditional candidate only")]))):
            with self.subTest(report=report):
                changed = replace(source, generation_json=json.dumps(report))
                payload["document"] = replace(document, sources=tuple(
                    changed if s.key == source.key else s for s in document.sources))
                dialog._show_preview(payload)
                dialog._controls()
                cells = [dialog.region_model.item(r, c).text()
                         for r in range(dialog.region_model.rowCount()) for c in range(5)]
                self.assertIn("Not recorded", cells)
                self.assertIn("Scientific review of this measurement is not recorded", dialog.region_scope_label.text())
                self.assertIn("no measurement grid checked", dialog.region_scope_label.text())
                self.assertTrue(dialog.use_btn.isEnabled())
                if report:
                    self.assertIn("Unsupported", cells)
                    self.assertIn("Optional fine correction", cells)
                    self.assertIn("Conditional correction candidate", cells)
                else:
                    self.assertEqual(dialog.region_model.rowCount(), 1)

    def test_region_table_loaded_layout_themes_and_keyboard(self):
        self.save()
        dialog = self.dialog()
        self.choose(dialog)
        dialog.advanced_btn.setChecked(True)
        dialog.resize(dialog.minimumSize())
        dialog.show()
        manager = theme_manager()
        self.addCleanup(manager.set_mode, "light")
        for mode in ("light", "dark"):
            manager.set_mode(mode)
            self.app.processEvents()
            for widget in (dialog.region_scope_label, dialog.region_table):
                dialog.controls_scroll.ensureWidgetVisible(widget)
                self.app.processEvents()
                point = widget.mapTo(dialog, widget.rect().topLeft())
                self.assertGreaterEqual(point.x(), 0)
                self.assertGreaterEqual(point.y(), 0)
                self.assertLessEqual(point.x()+widget.width(), dialog.width())
                self.assertLessEqual(point.y()+widget.height(), dialog.height())
        dialog.region_table.setFocus()
        self.app.processEvents()
        self.assertTrue(dialog.region_table.hasFocus())
        QTest.keyClick(dialog.region_table, Qt.Key_Down)
        self.assertIsNone(dialog.selected_version)
        self.assertIsNone(self.store.find_applied((self.sample,)))

    def test_conditioned_region_summary_counts_actual_deferred_routes(self):
        import json
        from dataclasses import replace
        version = self.save()
        payload = load_pn_version(self.root, version)
        document = payload["document"]
        source = next(s for s in document.sources if s.key == document.default_source)
        report = dict(scientifically_validated=False, render_diagnostics={"full": dict(
            partial=True, applied_count=0, opposite_exclusive_count=2,
            regional_plan=[dict(source="P", core_indices=[20, 80], apply=False, route="compact_only")],
            local_plan=[dict(source="P", core_indices=[30, 50], apply=False, route="compact_deferred",
                             reasons=["overlapping_requested_supports"])])})
        changed = replace(source, generation_json=json.dumps(report))
        payload["document"] = replace(document, sources=tuple(
            changed if s.key == source.key else s for s in document.sources))
        dialog = self.dialog()
        dialog._show_preview(payload)
        self.assertIn("1 regional repair(s) deferred", dialog.method_label.text())
        self.assertIn("opposite-source evidence at 2", dialog.method_label.text())

    def test_bounded_preview_separates_base_repairs_and_optional_fine_correction(self):
        import json
        from dataclasses import replace
        version = self.save()
        payload = load_pn_version(self.root, version)
        document = payload["document"]
        source = next(s for s in document.sources if s.key == document.default_source)
        dialog = self.dialog()
        for selected in (True, False):
            report = dict(scientifically_validated=False, source_models_refitted_from_raw=True,
                render_diagnostics={"full": dict(correction_selected=selected,
                    final_polynomial_available=False)},
                conditioned_parent_diagnostics={"full": dict(applied_count=3,
                    regional_plan=[dict(source="P", core_indices=[20, 80], apply=True)])})
            changed = replace(source, generation_json=json.dumps(report))
            payload["document"] = replace(document, sources=tuple(
                changed if s.key == source.key else s for s in document.sources))
            dialog._show_preview(payload)
            text = dialog.method_label.text()
            self.assertIn("Base: 3 regional repair(s)", text)
            self.assertIn("Supported fine correction included" if selected else
                          "Fine correction withheld; base background retained", text)
            self.assertNotIn("0 regional repair(s)", text)
            self.assertIn("reported separately", dialog.method_label.toolTip())
        report.pop("conditioned_parent_diagnostics")
        report["render_diagnostics"]["full"]["parent_diagnostics_path"] = "conditioned_parent_diagnostics.full"
        changed = replace(source, generation_json=json.dumps(report))
        payload["document"] = replace(document, sources=tuple(
            changed if s.key == source.key else s for s in document.sources))
        dialog._show_preview(payload)
        self.assertIn("Base construction not recorded", dialog.method_label.text())
        self.assertNotIn("0 regional repair(s)", dialog.method_label.text())
        self.assertIsNone(dialog.selected_version)
        self.assertIsNone(self.store.find_applied((self.sample,)))


if __name__ == "__main__":
    unittest.main()
