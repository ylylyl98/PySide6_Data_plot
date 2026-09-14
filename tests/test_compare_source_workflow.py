import os
import json
import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QPlainTextEdit

from ui_qt.controllers_compare import CompareController
from ui_qt.main_window import MainWindow
from ui_qt.source_picker_dialog import SourcePickerDialog
from tests.ui_test_helpers import wait_for_file_catalog


class _CompareOwner:
    def __init__(self):
        self.pl_available_files = [
            "sample_PL.csv",
            "sample_REF.csv",
            "sample_unknown.csv",
            "saved.dat",
        ]
        self.available_files = list(self.pl_available_files)
        self.cmp_source_filter_combo = QComboBox()
        self.cmp_source_filter_combo.addItem("PL raw sources", "pl")
        self.cmp_source_filter_combo.addItem("All raw data", "all")
        self.cmp_channel_combos = {key: QComboBox() for key in ("KK", "KKp", "KpK", "KpKp")}
        self.cmp_display_preset_combo = QComboBox()
        self.cmp_display_preset_combo.addItem("KK + KKp")
        self.cmp_show_checks = {}
        self.cmp_assignment_summary = QPlainTextEdit()
        self.cmp_group_power_tolerance_percent = 5.0
        self.cmp_in_k_angle_spin = QDoubleSpinBox()
        self.cmp_in_kp_angle_spin = QDoubleSpinBox()
        self.cmp_out_k_angle_spin = QDoubleSpinBox()
        self.cmp_out_kp_angle_spin = QDoubleSpinBox()
        self.cmp_angle_tolerance_spin = QDoubleSpinBox()
        self.cmp_in_k_angle_spin.setValue(0.0)
        self.cmp_in_kp_angle_spin.setValue(45.0)
        self.cmp_out_k_angle_spin.setValue(0.0)
        self.cmp_out_kp_angle_spin.setValue(45.0)
        self.cmp_angle_tolerance_spin.setValue(15.0)
        self.loaded = None
        self._append_log = lambda *_args: None
        self._invalidate_export_move_sources = lambda: None


class CompareSourceWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.owner = _CompareOwner()
        self.controller = CompareController(self.owner)

    def test_default_candidates_are_pl_only_and_exclude_dat(self):
        self.assertEqual(self.controller._cmp_assign_candidate_files(), ["sample_PL.csv"])

    def test_all_raw_data_includes_ref_and_unknown_but_excludes_dat(self):
        self.owner.cmp_source_filter_combo.setCurrentIndex(1)
        self.assertEqual(
            self.controller._cmp_assign_candidate_files(),
            ["sample_PL.csv", "sample_REF.csv", "sample_unknown.csv"],
        )

    def test_manual_assignment_survives_filter_change_with_exact_path(self):
        self.owner.cmp_source_filter_combo.setCurrentIndex(1)
        self.controller._cmp_set_channel_combo_items()
        combo = self.owner.cmp_channel_combos["KK"]
        combo.setCurrentText("sample_REF.csv")
        self.owner.cmp_source_filter_combo.setCurrentIndex(0)
        self.controller._cmp_set_channel_combo_items()
        self.assertEqual(combo.currentText(), "sample_REF.csv")
        self.assertIn("outside", combo.itemData(combo.findText("sample_REF.csv"), Qt.ToolTipRole))

    def test_unlabeled_file_remains_manually_selectable(self):
        self.controller._cmp_set_channel_combo_items()
        combo = self.owner.cmp_channel_combos["KK"]
        self.assertGreaterEqual(combo.findText("sample_PL.csv"), 0)
        combo.setCurrentText("sample_PL.csv")
        self.assertEqual(combo.currentText(), "sample_PL.csv")

    def test_swap_kk_and_kkp_preserves_other_channels(self):
        for combo in self.owner.cmp_channel_combos.values():
            combo.addItems(["", "kk.csv", "kkp.csv", "kpk.csv", "kpkp.csv"])
        self.owner.cmp_channel_combos["KK"].setCurrentText("kk.csv")
        self.owner.cmp_channel_combos["KKp"].setCurrentText("kkp.csv")
        self.owner.cmp_channel_combos["KpK"].setCurrentText("kpk.csv")
        self.owner.cmp_channel_combos["KpKp"].setCurrentText("kpkp.csv")
        self.owner._cmp_update_assignment_summary = lambda: None
        self.controller._cmp_swap_kk_channels()
        self.assertEqual(self.owner.cmp_channel_combos["KK"].currentText(), "kkp.csv")
        self.assertEqual(self.owner.cmp_channel_combos["KKp"].currentText(), "kk.csv")
        self.assertEqual(self.owner.cmp_channel_combos["KpK"].currentText(), "kpk.csv")
        self.assertEqual(self.owner.cmp_channel_combos["KpKp"].currentText(), "kpkp.csv")

    def test_clear_group_clears_hidden_channel_state(self):
        for combo in self.owner.cmp_channel_combos.values():
            combo.addItems(["", "assigned.csv"])
            combo.setCurrentText("assigned.csv")
        self.owner.cmp_selected_group_key = "group-1"
        self.owner.cmp_selected_group_label = "Group 1"
        self.owner.cmp_selected_group_sources = ("assigned.csv",)
        self.owner._cmp_update_assignment_summary = lambda: None
        self.controller._cmp_clear_group()
        self.assertFalse(self.owner.cmp_selected_group_key)
        self.assertTrue(all(not combo.currentText() for combo in self.owner.cmp_channel_combos.values()))

    def _angle_owner(self, files):
        owner = _CompareOwner()
        owner.pl_available_files = list(files)
        owner.available_files = list(files)
        owner.cmp_source_filter_combo.setCurrentIndex(1)
        return owner, CompareController(owner)

    def test_mapping_switch_reclassifies_selected_group_in_both_directions(self):
        files = ['sample_PL_Rot10deg.csv', 'sample_PL_Rot145deg.csv']
        owner, controller = self._angle_owner(files)
        owner.cmp_rotation_mapping_combo = QComboBox()
        owner.cmp_rotation_mapping_combo.addItem('Rot1 input', 'rot1_input')
        owner.cmp_rotation_mapping_combo.addItem('Rot1 output', 'rot1_output')
        controller._cmp_set_channel_combo_items()
        controller._cmp_auto_assign_channels()
        self.assertEqual(owner.cmp_channel_combos['KpK'].currentText(), files[1])
        group_key = owner.cmp_selected_group_key
        owner.cmp_rotation_mapping_combo.setCurrentIndex(1)
        controller._on_cmp_angle_reference_changed()
        self.assertEqual(owner.cmp_selected_group_key, group_key)
        self.assertEqual(owner.cmp_channel_combos['KKp'].currentText(), files[1])
        self.assertFalse(owner.cmp_channel_combos['KpK'].currentText())
        owner.cmp_rotation_mapping_combo.setCurrentIndex(0)
        controller._on_cmp_angle_reference_changed()
        self.assertEqual(owner.cmp_channel_combos['KpK'].currentText(), files[1])
        self.assertFalse(owner.cmp_channel_combos['KKp'].currentText())

    def test_auto_detect_infers_selected_group_without_mixing_stage_groups(self):
        files = [
            "YZ212_1077uW_Rot224deg_Stage200.csv",
            "YZ212_1078uW_Rot269deg_Stage200.csv",
            "YZ212_1077uW_Rot224deg_Stage3600.csv",
            "YZ212_1078uW_Rot269deg_Stage3600.csv",
        ]
        owner, controller = self._angle_owner(files)
        controller._cmp_set_channel_combo_items()
        controller._cmp_auto_assign_channels()

        self.assertEqual(owner.cmp_out_k_angle_spin.value(), 24.0)
        self.assertEqual(owner.cmp_out_kp_angle_spin.value(), 69.0)
        self.assertIn("stage200", owner.cmp_selected_group_key)
        self.assertEqual(
            owner.cmp_channel_combos["KK"].currentText(),
            "YZ212_1077uW_Rot224deg_Stage200.csv",
        )
        self.assertEqual(
            owner.cmp_channel_combos["KKp"].currentText(),
            "YZ212_1078uW_Rot269deg_Stage200.csv",
        )

    def test_manual_swap_survives_refresh_after_inferred_references(self):
        files = [
            "YZ212_1077uW_Rot224deg_Stage200.csv",
            "YZ212_1078uW_Rot269deg_Stage200.csv",
        ]
        owner, controller = self._angle_owner(files)
        controller._cmp_set_channel_combo_items()
        controller._cmp_auto_assign_channels()
        controller._cmp_swap_kk_channels()
        swapped = {
            key: combo.currentText() for key, combo in owner.cmp_channel_combos.items()
        }

        controller._cmp_auto_assign_channels()

        self.assertEqual(
            {key: combo.currentText() for key, combo in owner.cmp_channel_combos.items()},
            swapped,
        )

    def test_deleted_manual_mapping_survives_refresh_when_group_remains(self):
        files = [
            "YZ212_1077uW_Rot224deg_Stage200.csv",
            "YZ212_1078uW_Rot269deg_Stage200.csv",
        ]
        owner, controller = self._angle_owner(files)
        controller._cmp_set_channel_combo_items()
        controller._cmp_auto_assign_channels()
        missing = "manual_REF.csv"
        owner.cmp_channel_combos["KK"].addItem(missing)
        owner.cmp_channel_combos["KK"].setCurrentText(missing)
        expected = {
            key: combo.currentText() for key, combo in owner.cmp_channel_combos.items()
        }
        controller._cmp_auto_assign_channels()
        self.assertEqual(
            {key: combo.currentText() for key, combo in owner.cmp_channel_combos.items()},
            expected,
        )

    def test_missing_manual_mapping_does_not_reinfer_selected_group_references(self):
        files = [
            "YZ212_1077uW_Rot224deg_Stage200.csv",
            "YZ212_1078uW_Rot269deg_Stage200.csv",
        ]
        owner, controller = self._angle_owner(files)
        controller._cmp_set_channel_combo_items()
        group = controller._cmp_source_groups()[0]
        owner.cmp_selected_group_key = controller._cmp_group_value(group, "key", "")
        owner.cmp_selected_group_sources = tuple(files)
        owner.cmp_channel_combos["KK"].addItem("manual_REF.csv")
        owner.cmp_channel_combos["KK"].setCurrentText("manual_REF.csv")
        owner.cmp_channel_combos["KKp"].setCurrentText(files[0])
        expected = {"KK": "manual_REF.csv", "KKp": files[0]}
        controller._cmp_auto_assign_channels(preserve_existing=True, allow_inference=True)
        self.assertEqual(
            {key: controller._cmp_current_mapping().get(key) for key in expected},
            expected,
        )
        self.assertEqual(owner.cmp_out_k_angle_spin.value(), 0.0)
        self.assertEqual(owner.cmp_out_kp_angle_spin.value(), 45.0)

    def test_hidden_missing_mapping_does_not_block_visible_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("kk.csv", "kkp.csv"):
                (root / name).write_text("x", encoding="utf-8")
            owner, controller = self._angle_owner(["kk.csv", "kkp.csv"])
            owner.current_folder = str(root)
            owner.cmp_channel_combos["KK"].addItems(["", "kk.csv"])
            owner.cmp_channel_combos["KKp"].addItems(["", "kkp.csv"])
            owner.cmp_channel_combos["KpK"].addItems(["", "missing-hidden.csv"])
            owner.cmp_channel_combos["KpKp"].addItems(["", "missing-hidden-2.csv"])
            owner.cmp_channel_combos["KK"].setCurrentText("kk.csv")
            owner.cmp_channel_combos["KKp"].setCurrentText("kkp.csv")
            owner.cmp_channel_combos["KpK"].setCurrentText("missing-hidden.csv")
            owner.cmp_channel_combos["KpKp"].setCurrentText("missing-hidden-2.csv")
            owner.cmp_display_preset_combo.setCurrentText("KK + KKp")
            selection = controller._cmp_selection_from_ui()
            self.assertEqual(selection.visible_order, ("KK", "KKp"))

    def test_multi_cluster_group_does_not_guess_references(self):
        files = [
            "YZ212_1077uW_Rot224deg_Stage200.csv",
            "YZ212_1078uW_Rot250deg_Stage200.csv",
            "YZ212_1079uW_Rot269deg_Stage200.csv",
        ]
        owner, controller = self._angle_owner(files)
        controller._cmp_set_channel_combo_items()
        controller._cmp_auto_assign_channels()

        self.assertEqual(owner.cmp_out_k_angle_spin.value(), 0.0)
        self.assertEqual(owner.cmp_out_kp_angle_spin.value(), 45.0)
        self.assertFalse(owner.cmp_channel_combos["KK"].currentText())
        self.assertFalse(owner.cmp_channel_combos["KpK"].currentText())

    def test_angle_edit_reclassifies_without_inference(self):
        files = [
            "YZ212_1077uW_Rot224deg_Stage200.csv",
            "YZ212_1078uW_Rot269deg_Stage200.csv",
        ]
        owner, controller = self._angle_owner(files)
        controller._cmp_set_channel_combo_items()
        controller._cmp_auto_assign_channels()

        owner.cmp_out_k_angle_spin.setValue(10.0)
        controller._on_cmp_angle_reference_changed()

        self.assertEqual(owner.cmp_out_k_angle_spin.value(), 10.0)
        self.assertEqual(owner.cmp_out_kp_angle_spin.value(), 69.0)
        self.assertEqual(owner.cmp_channel_combos["KKp"].currentText(), files[1])

    @staticmethod
    def _write_compare_file(root: Path, name: str) -> None:
        (root / name).write_text("x", encoding="utf-8")

    def _real_compare_window(self, root: Path) -> MainWindow:
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            window = MainWindow()
        window._set_current_folder(str(root), remember=False)
        wait_for_file_catalog(window)
        window.cmp_source_filter_combo.setCurrentIndex(1)
        window.compare_controller._cmp_set_channel_combo_items()
        return window

    def test_compare_dialog_async_refresh_adds_file_and_preserves_manual_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group_a = [
                "YZ212_1077uW_Rot224deg_Stage200.csv",
                "YZ212_1078uW_Rot269deg_Stage200.csv",
            ]
            group_b = [
                "YZ212_2077uW_Rot124deg_Stage300.csv",
                "YZ212_2078uW_Rot169deg_Stage300.csv",
            ]
            for name in group_a:
                self._write_compare_file(root, name)
            window = self._real_compare_window(root)
            try:
                controller = window.compare_controller
                controller._cmp_auto_assign_channels()
                controller._cmp_swap_kk_channels()
                expected_mapping = controller._cmp_current_mapping()
                expected_key = window.cmp_selected_group_key
                observed = {}

                def fake_exec(dialog):
                    tolerance = dialog.findChild(QDoubleSpinBox)
                    tolerance.setValue(7.5)
                    for name in group_b:
                        self._write_compare_file(root, name)
                    dialog.refresh_button.click()
                    for _ in range(400):
                        QTest.qWait(10)
                        if (
                            group_b[0] in window.pl_available_files
                            and not window._file_refresh_running
                            and any(
                                group_b[0] in dialog.source_list.item(i).text()
                                for i in range(dialog.source_list.count())
                            )
                        ):
                            break
                    observed["has_new"] = any(
                        group_b[0] in dialog.source_list.item(i).text()
                        for i in range(dialog.source_list.count())
                    )
                    observed["items"] = [
                        str(dialog.source_list.item(i).data(Qt.UserRole))
                        for i in range(dialog.source_list.count())
                    ]
                    observed["mapping"] = controller._cmp_current_mapping()
                    observed["key"] = window.cmp_selected_group_key
                    observed["tolerance"] = tolerance.value()
                    dialog.reject()
                    return SourcePickerDialog.Rejected

                with patch.object(SourcePickerDialog, "exec", fake_exec):
                    controller._cmp_open_group_dialog()
                self.assertTrue(observed["has_new"], observed)
                self.assertEqual(observed["mapping"], expected_mapping)
                self.assertEqual(observed["key"], expected_key)
                self.assertEqual(observed["tolerance"], 7.5)
            finally:
                window.close()

    def test_compare_group_picker_shows_status_filter_and_orders_groups_by_newest_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = {
                "old": ("old_KK.csv", "old_KKp.csv"),
                "new": ("new_KK.csv", "new_KKp.csv"),
            }
            for name in groups["old"] + groups["new"]:
                self._write_compare_file(root, name)
            os.utime(root / "old_KK.csv", (1000, 1000))
            os.utime(root / "old_KKp.csv", (1001, 1001))
            os.utime(root / "new_KK.csv", (3000, 3000))
            os.utime(root / "new_KKp.csv", (3001, 3001))
            history = root / "Processed Data" / "Compare"
            history.mkdir(parents=True)
            (history / "new.metadata.json").write_text(
                json.dumps({
                    "operation": "Compare/KK",
                    "created_utc": "2026-09-10T12:00:00+00:00",
                    "sources": [{"role": "source", "name": "new_KK.csv"}],
                    "processing": {
                        "compare_source_mapping": {
                            "KK": "new_KK.csv",
                            "KKp": "new_KKp.csv",
                        },
                    },
                }), encoding="utf-8"
            )
            (history / "old.metadata.json").write_text(
                json.dumps({
                    "operation": "Compare/KK",
                    "created_utc": "2026-09-09T12:00:00+00:00",
                    "sources": [{"role": "source", "name": "old_KK.csv"}],
                }), encoding="utf-8"
            )

            window = self._real_compare_window(root)
            try:
                # The picker consumes the owner-published Compare snapshot;
                # force one real worker pass after creating the fixture files.
                window._refresh_file_lists(auto=False, mode="Compare")
                wait_for_file_catalog(window)
                seen = {}
                history_calls = []
                controller = window.compare_controller
                real_refresh_history = CompareController._cmp_refresh_history_cache

                def track_history_refresh(instance, *, force=False):
                    history_calls.append(bool(force))
                    return real_refresh_history(instance, force=force)

                def inspect(dialog):
                    seen["controls"] = [
                        (combo.currentText(), combo.findData("all"), combo.findData("new"), combo.findData("processed"))
                        for combo in dialog.findChildren(QComboBox)
                    ]
                    seen["items"] = [
                        str(dialog.source_list.item(i).data(Qt.UserRole))
                        for i in range(dialog.source_list.count())
                    ]
                    status = next(combo for combo in dialog.findChildren(QComboBox) if combo.findData("processed") >= 0)
                    status.setCurrentIndex(status.findData("processed"))
                    seen["processed"] = [
                        str(dialog.source_list.item(i).data(Qt.UserRole))
                        for i in range(dialog.source_list.count())
                    ]
                    status.setCurrentIndex(status.findData("mixed"))
                    seen["mixed"] = [
                        str(dialog.source_list.item(i).data(Qt.UserRole))
                        for i in range(dialog.source_list.count())
                    ]
                    dialog.reject()
                    return SourcePickerDialog.Rejected

                with (
                    patch.object(CompareController, "_cmp_refresh_history_cache", track_history_refresh),
                    patch.object(SourcePickerDialog, "exec", inspect),
                ):
                    controller._cmp_open_group_dialog()
                self.assertTrue(any(all(index >= 0 for index in row[1:]) for row in seen["controls"]), seen)
                self.assertEqual(len(seen["items"]), 2)
                self.assertIn("new", seen["items"][0])
                self.assertIn("old", seen["items"][1])
                self.assertEqual(len(seen["processed"]), 1)
                self.assertIn("new", seen["processed"][0])
                self.assertEqual(len(seen["mixed"]), 1)
                self.assertIn("old", seen["mixed"][0])
                self.assertTrue(history_calls)
                self.assertFalse(any(history_calls), history_calls)
            finally:
                window.close()

    def test_compare_group_status_keeps_ambiguous_legacy_history_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [
                "run_a/shared_KK.csv", "run_a/shared_KKp.csv",
                "run_b/shared_KK.csv", "run_b/shared_KKp.csv",
            ]
            for source in sources:
                path = root / source
                path.parent.mkdir(parents=True, exist_ok=True)
                self._write_compare_file(path.parent, path.name)
            history = root / "Processed Data" / "Compare"
            history.mkdir(parents=True)
            (history / "legacy.metadata.json").write_text(
                json.dumps({
                    "operation": "Compare/KK",
                    "sources": [{"role": "source", "name": "shared_KK.csv"}],
                }), encoding="utf-8"
            )
            owner, controller = self._angle_owner(sources)
            owner.current_folder = str(root)
            from ui_qt.main_window import _scan_folder_sources_worker
            payload = _scan_folder_sources_worker(str(root), mode="Compare", progress=None, log=None)
            scope = next(item for item in payload if isinstance(item, dict) and item.get("tag") == "catalog_scope")
            owner._cmp_history_records = list(scope.get("compare_history", ()))
            owner._cmp_history_cache_folder = str(root)
            controller._cmp_refresh_history_cache()
            groups = controller._cmp_source_groups()
            self.assertEqual(
                {controller._cmp_group_status(group) for group in groups},
                {"unknown"},
            )

    def test_compare_dialog_deleted_current_group_has_no_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group_a = [
                "YZ212_1077uW_Rot224deg_Stage200.csv",
                "YZ212_1078uW_Rot269deg_Stage200.csv",
            ]
            group_b = [
                "YZ212_2077uW_Rot124deg_Stage300.csv",
                "YZ212_2078uW_Rot169deg_Stage300.csv",
            ]
            for name in group_a:
                self._write_compare_file(root, name)
            window = self._real_compare_window(root)
            try:
                controller = window.compare_controller
                controller._cmp_auto_assign_channels()
                expected_mapping = controller._cmp_current_mapping()
                selected_key = window.cmp_selected_group_key
                observed = {}

                def fake_exec(dialog):
                    tolerance = dialog.findChild(QDoubleSpinBox)
                    tolerance.setValue(8.0)
                    for name in group_a:
                        (root / name).unlink()
                    for name in group_b:
                        self._write_compare_file(root, name)
                    dialog.refresh_button.click()
                    for _ in range(400):
                        QTest.qWait(10)
                        if (
                            group_b[0] in window.pl_available_files
                            and not window._file_refresh_running
                            and dialog.source_list.currentItem() is None
                            and any(
                                group_b[0] in dialog.source_list.item(i).text()
                                for i in range(dialog.source_list.count())
                            )
                        ):
                            break
                    observed["current"] = dialog.source_list.currentItem()
                    observed["ok"] = dialog.ok_button.isEnabled()
                    observed["mapping"] = controller._cmp_current_mapping()
                    observed["missing"] = controller._cmp_missing_sources(expected_mapping)
                    dialog.reject()
                    return SourcePickerDialog.Rejected

                with patch.object(SourcePickerDialog, "exec", fake_exec):
                    controller._cmp_open_group_dialog()
                self.assertIsNone(observed["current"])
                self.assertFalse(observed["ok"])
                self.assertEqual(observed["mapping"], expected_mapping)
                self.assertTrue(observed["missing"])
                self.assertEqual(window.cmp_selected_group_key, selected_key)
            finally:
                window.close()

    def test_compare_refresh_queued_behind_scan_applies_latest_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = [
                "YZ212_1077uW_Rot224deg_Stage200.csv",
                "YZ212_1078uW_Rot269deg_Stage200.csv",
            ]
            added = [
                "YZ212_2077uW_Rot124deg_Stage300.csv",
                "YZ212_2078uW_Rot169deg_Stage300.csv",
            ]
            for name in initial:
                self._write_compare_file(root, name)
            window = self._real_compare_window(root)
            release = threading.Event()
            first_ready = threading.Event()
            calls = {"count": 0}
            window.folder_refresh_timer.stop()
            watched = list(window.folder_watcher.directories())
            if watched:
                window.folder_watcher.removePaths(watched)
            from ui_qt import main_window as main_window_module
            real_scan = main_window_module._scan_folder_sources_worker

            def delayed_scan(folder, *, progress, log, **kwargs):
                result = real_scan(folder, progress=progress, log=log, **kwargs)
                calls["count"] += 1
                if calls["count"] == 1:
                    first_ready.set()
                    release.wait(3.0)
                return result

            try:
                with patch.object(main_window_module, "_scan_folder_sources_worker", delayed_scan):
                    window._refresh_file_lists(auto=False, mode="Compare")
                    self.assertTrue(first_ready.wait(3.0))

                    observed = {}

                    def fake_exec(dialog):
                        for name in added:
                            self._write_compare_file(root, name)
                        dialog.refresh_button.click()
                        release.set()
                        for _ in range(600):
                            QTest.qWait(10)
                            if (
                                added[0] in window.pl_available_files
                                and not window._file_refresh_running
                                and not window._file_refresh_pending
                            ):
                                break
                        QTest.qWait(100)
                        observed["has_new"] = any(
                            added[0] in dialog.source_list.item(i).text()
                            for i in range(dialog.source_list.count())
                        )
                        observed["items"] = [
                            str(dialog.source_list.item(i).data(Qt.UserRole))
                            for i in range(dialog.source_list.count())
                        ]
                        observed["calls"] = calls["count"]
                        observed["catalog"] = list(window.pl_available_files)
                        observed["pending"] = bool(window._file_refresh_pending)
                        dialog.reject()
                        return SourcePickerDialog.Rejected

                    with patch.object(SourcePickerDialog, "exec", fake_exec):
                        window.compare_controller._cmp_open_group_dialog()
                self.assertGreaterEqual(calls["count"], 2)
                self.assertTrue(observed["has_new"], observed)
            finally:
                release.set()
                window.close()

if __name__ == "__main__":
    unittest.main()
