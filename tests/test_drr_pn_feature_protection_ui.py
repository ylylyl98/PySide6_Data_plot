"""Feature-protection UI contracts with temporary inputs and no reconstruction fit."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from tests import test_drr_pn_background_ui as _fixtures
from tests import test_drr_pn_generation_progress_ui as _progress_fixtures
from core.drr_pn_store import _file_digest
from core.load_cancellation import LoadCancelled, LoadCancellation
from ui_qt.drr_pn_background_dialog import _reconstruction_import_dependencies
from ui_qt.theme import install_theme, theme_manager


class PnFeatureProtectionUiTests(unittest.TestCase):
    write = _fixtures.PnBackgroundDialogTests.write
    dialog = _fixtures.PnBackgroundDialogTests.dialog
    choose = _fixtures.PnBackgroundDialogTests.choose
    reconstruction_dialog = _fixtures.PnBackgroundDialogTests.reconstruction_dialog
    reconstruction_stub = _fixtures.PnBackgroundDialogTests.reconstruction_stub
    wait_until = _progress_fixtures.PnGenerationProgressUiTests.wait_until
    wait = _progress_fixtures.PnGenerationProgressUiTests.wait

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode="light")

    def setUp(self):
        _fixtures.PnBackgroundDialogTests.setUp(self)

    def tearDown(self):
        _fixtures.PnBackgroundDialogTests.tearDown(self)

    def range(self, dialog, lower="1.62", upper="1.67", source="P", *, index=0):
        _, lo, hi, combo, _ = dialog.feature_protection_rows[index]
        lo.setText(lower)
        hi.setText(upper)
        combo.setCurrentIndex(combo.findData(source))

    def test_opt_in_blank_fields_validation_and_off_keeps_original_request(self):
        dialog, _ = self.reconstruction_dialog()
        self.assertFalse(dialog.feature_protection_check.isChecked())
        self.assertEqual(dialog._feature_protection(), [])
        self.assertTrue(dialog.feature_protection_panel.isHidden())
        _, lower, upper, source, _ = dialog.feature_protection_rows[0]
        self.assertEqual((lower.text(), upper.text(), source.currentData()), ("", "", None))
        self.assertTrue(lower.placeholderText())
        self.assertTrue(upper.placeholderText())
        dialog.feature_protection_check.setChecked(True)
        self.assertFalse(dialog.generate_btn.isEnabled())
        self.assertIn("limits", dialog.generate_btn.toolTip())
        self.range(dialog)
        self.assertTrue(dialog.generate_btn.isEnabled())
        for lo, hi in (("nan", "1.7"), ("inf", "1.7"), ("1.8", "1.7"), ("1.7", "1.7")):
            with self.subTest(lower=lo, upper=hi):
                self.range(dialog, lo, hi)
                self.assertFalse(dialog.generate_btn.isEnabled())
        dialog.feature_protection_check.setChecked(False)
        self.assertEqual(dialog._feature_protection(), [])
        self.assertTrue(dialog.generate_btn.isEnabled())
        self.assertFalse(self.store.directory.exists())

    def test_multiple_ranges_are_preserved_sorted_and_overlap_rejected(self):
        dialog, _ = self.reconstruction_dialog()
        dialog.feature_protection_check.setChecked(True)
        self.range(dialog, "1.77", "1.81", "N")
        dialog.feature_protection_add.click()
        self.range(dialog, "1.62", "1.67", "P", index=1)
        self.assertEqual(dialog._feature_protection(), [
            {"xmin_ev": 1.62, "xmax_ev": 1.67, "source": "P"},
            {"xmin_ev": 1.77, "xmax_ev": 1.81, "source": "N"}])
        self.range(dialog, "1.66", "1.78", "P", index=1)
        self.assertFalse(dialog.generate_btn.isEnabled())
        dialog.feature_protection_rows[1][-1].click()
        self.assertTrue(dialog.generate_btn.isEnabled())
        self.assertEqual(len(dialog._feature_protection()), 1)

    def test_generate_snapshots_all_ranges_and_empty_does_not_add_keyword(self):
        dialog, names = self.reconstruction_dialog()
        api = SimpleNamespace(prepare_reconstruction_request=Mock(side_effect=LoadCancelled()))
        for enabled in (False, True):
            dialog.feature_protection_check.setChecked(enabled)
            if enabled:
                self.range(dialog)
            with patch.object(dialog, "_start") as start, patch(
                    "ui_qt.drr_pn_background_dialog._reconstruction_api", return_value=api):
                dialog._generate(False)
                callback = start.call_args.args[0]
                with self.assertRaises(LoadCancelled):
                    callback(Mock(), Mock())
            expected = dict(training_files=names[::2], method="conditioned_bounded")
            if enabled:
                expected["feature_protection"] = [{"xmin_ev": 1.62, "xmax_ev": 1.67, "source": "P"}]
            self.assertEqual(api.prepare_reconstruction_request.call_args.kwargs, expected)
        self.assertFalse(self.store.directory.exists())

    def test_edit_invalidates_preview_and_pending_delivery_and_disables_busy_fields(self):
        dialog, _ = self.reconstruction_dialog()
        dialog._current = SimpleNamespace(status="fresh", fresh=True)
        dialog._payload = {"compatibility_error": ""}
        dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("saved"))
        cancellation = LoadCancellation()
        dialog._worker = SimpleNamespace(cancellation=cancellation)
        token = dialog._token, dialog._context()
        dialog._controls()
        self.assertFalse(dialog.feature_protection_check.isEnabled())
        self.assertFalse(dialog.feature_protection_rows[0][1].isEnabled())
        # Programmatic edits must be safe even though actual busy controls cannot be clicked.
        dialog.feature_protection_check.setChecked(True)
        self.assertTrue(cancellation.cancelled)
        self.assertNotEqual(token, (dialog._token, dialog._context()))
        self.assertEqual(dialog.generation_mode.currentData(), "reconstruction")
        self.assertIsNone(dialog._current)
        self.assertIsNone(dialog._payload)
        self.assertFalse(dialog.use_btn.isEnabled())
        dialog._consume = Mock()
        dialog._result((token, "stale preview"))
        dialog._consume.assert_not_called()
        dialog._worker = None
        dialog._controls()
        self.assertTrue(dialog.feature_protection_check.isEnabled())
        dialog.feature_protection_preview.setText("Partial preview")
        dialog.feature_protection_preview.show()
        dialog._worker = SimpleNamespace(cancellation=LoadCancellation())
        dialog._active_context = (dialog._token, dialog._context())
        dialog._consume = Mock(side_effect=ValueError("preview failed"))
        dialog._result((dialog._active_context, None))
        self.assertTrue(dialog.feature_protection_preview.isHidden())
        self.assertEqual(dialog.feature_protection_preview.text(), "")
        dialog._worker = None

    def test_history_restores_every_range_without_editing_or_new_group_reuse(self):
        dialog, names = self.reconstruction_dialog()
        api, request, document = self.reconstruction_stub(names)
        regions = [{"xmin_ev": 1.62, "xmax_ev": 1.67, "source": "P"},
                   {"xmin_ev": 1.77, "xmax_ev": 1.81, "source": "N"}]
        settings = {**request.settings, "feature_protection": regions}
        key = self.store.prepare(self.pair, settings=settings,
            algorithm_version=request.key.provenance["algorithm_version"], dependencies=names[::2])
        sources = []
        for source in document.sources:
            if source.key == document.default_source:
                report = json.loads(source.generation_json)
                report["settings"] = settings
                report["feature_protection"] = {"full": {"whole_axis_gain_minmax": [.79, 1.31]}}
                report["conditioned_parent_diagnostics"] = {"full": {}}
                report["render_diagnostics"]["full"]["correction_selected"] = True
                source = replace(source, generation_json=json.dumps(report))
            sources.append(source)
        version = self.store.save(replace(document, sources=tuple(sources)), key)
        payload = dialog._preview(version)
        token = dialog._token
        dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("saved"))
        dialog._show_preview(payload)
        self.assertEqual(dialog._token, token)
        self.assertEqual(dialog.generation_mode.currentData(), "saved")
        self.assertEqual(dialog._feature_protection(), regions)
        self.assertEqual(len(dialog.feature_protection_rows), 2)
        self.assertEqual(dialog._current.version_id, version.version_id)
        preview_text = dialog.feature_protection_preview.text()
        self.assertIn("P: 1.62–1.67 eV", preview_text)
        self.assertIn("N: 1.77–1.81 eV", preview_text)
        self.assertIn("Background scale can change across the full spectrum; compare ΔR/R and d²E.", preview_text)
        self.assertIn("pre-protection background: 0.79–1.31×", preview_text)
        self.assertFalse(dialog.feature_protection_preview.isHidden())
        self.assertIn("fine correction included before protection", dialog.method_label.text())
        dialog.generation_mode.setCurrentIndex(dialog.generation_mode.findData("reconstruction"))
        self.assertEqual(dialog._feature_protection(), regions)
        self.assertTrue(dialog.generate_btn.isEnabled())
        other = self.dialog(measurements=(self.sample,), sample_scoped=True)
        self.assertFalse(other.feature_protection_check.isChecked())
        self.assertEqual(other._feature_protection(), [])
        self.assertTrue(other.feature_protection_preview.isHidden())
        dialog.feature_protection_rows[0][1].setText("1.63")
        self.assertTrue(dialog.feature_protection_preview.isHidden())
        self.assertEqual(dialog.feature_protection_preview.text(), "")

    def test_import_validates_full_protection_settings_for_native_and_store_style(self):
        from core.drr_pn_reconstruction.pipeline import CONDITIONED_BOUNDED_ALGORITHM_VERSION, CONDITIONED_BOUNDED_PROTOCOL
        names = [self.pname, self.nname, self.sample, self.sample.replace("sample", "repeat2")]
        self.write(names[-1], 126.)
        code = self.root / "fake_kernel.py"
        code.write_text("test only", encoding="utf-8")
        regions = [{"xmin_ev": 1.62, "xmax_ev": 1.67, "source": "P"},
                   {"xmin_ev": 1.77, "xmax_ev": 1.81, "source": "N"}]
        for with_training_settings in (False, True):
            settings = {**CONDITIONED_BOUNDED_PROTOCOL, "feature_protection": regions}
            if with_training_settings:
                settings["sample_analysis"] = {"training_files": names[2:], "subset": "all gates and full native energy",
                    "y_axis": "auto", "split_policy": "explicit-two-training-repeats; all other selected files check only"}
            report = {"algorithm_version": CONDITIONED_BOUNDED_ALGORITHM_VERSION, "settings": settings,
                "input_records": [{"path": name, "role": role, "sha256": _file_digest(self.root / name)}
                    for name, role in zip(names, ("P", "N", "training", "training"))],
                "implementation_sha256": {str(code): _file_digest(code)}}
            doc = SimpleNamespace(default_source="result", sources=(
                SimpleNamespace(key="result", generation_json=json.dumps(report)),))
            with patch("ui_qt.drr_pn_background_dialog.read_composite_background", return_value=doc):
                dependencies = _reconstruction_import_dependencies(self.root, self.root / "unused.csv", settings,
                    SimpleNamespace(dependency_paths=lambda: (code,)), pair=self.pair, p_which="all", n_which="all")
                self.assertEqual(dependencies, (code, *(self.root / name for name in names[2:])))
                with self.assertRaisesRegex(ValueError, "settings"):
                    _reconstruction_import_dependencies(self.root, self.root / "unused.csv",
                        {**settings, "feature_protection": regions[:1]},
                        SimpleNamespace(dependency_paths=lambda: (code,)), pair=self.pair, p_which="all", n_which="all")

    def test_light_dark_layout_and_keyboard_with_multiple_ranges(self):
        dialog, _ = self.reconstruction_dialog()
        dialog.advanced_btn.setChecked(False)
        dialog.feature_protection_check.setChecked(True)
        self.range(dialog)
        dialog.feature_protection_add.click()
        self.range(dialog, "1.77", "1.81", "N", index=1)
        dialog.resize(dialog.minimumSize())
        dialog.show()
        manager = theme_manager()
        self.addCleanup(manager.set_mode, "light")
        for mode in ("light", "dark"):
            manager.set_mode(mode)
            self.app.processEvents()
            widgets = [dialog.feature_protection_check, dialog.feature_protection_add, dialog.generate_btn,
                       dialog.use_btn, dialog.close_btn]
            widgets.extend(field for row in dialog.feature_protection_rows for field in row[1:])
            for widget in widgets:
                if dialog.controls_scroll.widget().isAncestorOf(widget):
                    dialog.controls_scroll.ensureWidgetVisible(widget)
                    self.app.processEvents()
                point = widget.mapTo(dialog, widget.rect().topLeft())
                self.assertGreaterEqual(point.x(), 0)
                self.assertGreaterEqual(point.y(), 0)
                self.assertLessEqual(point.x() + widget.width(), dialog.width())
                self.assertLessEqual(point.y() + widget.height(), dialog.height())
            _, lower, upper, source, _ = dialog.feature_protection_rows[0]
            lower.setFocus()
            QTest.keyClick(lower, Qt.Key_Tab)
            self.assertIs(self.app.focusWidget(), upper)
            QTest.keyClick(upper, Qt.Key_Tab)
            self.assertIs(self.app.focusWidget(), source)


if __name__ == "__main__":
    unittest.main()
