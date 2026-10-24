from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QThreadPool, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from core.drr_composite_background import (
    CompositeBackground, CompositeRegion, CompositeSource, save_composite_background,
)
from ui_qt.drr_composite_background_dialog import (
    DrrCompositeBackgroundDialog, _reload_source, _suggest_auto,
)
from ui_qt.theme import install_theme


class AutoCompositeDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.original_font = cls.app.font()
        cls.test_font = -1
        if not QFontDatabase.families():
            cls.test_font = QFontDatabase.addApplicationFont(str(
                Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"))
            cls.app.setFont(QFont("DejaVu Sans", 9))

    @classmethod
    def tearDownClass(cls):
        if cls.test_font >= 0:
            cls.app.setFont(cls.original_font)
            QFontDatabase.removeApplicationFont(cls.test_font)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.settings = QSettings(str(self.root / "settings.ini"), QSettings.IniFormat)
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)
        self.dialogs = []
        self.energy = np.linspace(1., 2., 101)
        self.report = {
            "schema_version": 1, "report_kind": "auto_composite_suggestion",
            "algorithm": "adaptive_masked_regions_v1", "original_default_source": "p",
            "estimated_regions": [[1.2, 1.3]], "unresolved_regions": [[1.45, 1.55]],
            "unsupported_regions": [[1.7, 1.8]],
            "texture_missing_regions": [[1.45, 1.5]],
        }

    def tearDown(self):
        for dialog in self.dialogs:
            dialog.close()
        self.pool.waitForDone(5000)
        self.app.processEvents()
        for dialog in self.dialogs:
            dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.app.processEvents()
        install_theme(self.app, mode="light")
        self.temporary.cleanup()

    def source(self, key, offset):
        return CompositeSource(key, key + ".csv", (str(self.root / (key + ".csv")),), "last",
            self.energy, tuple(map(str, self.energy)),
            offset + 3 * self.energy + .15 * np.sin(self.energy * 90))

    def document(self):
        return CompositeBackground((self.source("p", 20.), self.source("n", 22.)), "p")

    def suggestion(self, base=None):
        base = base or self.document()
        generated = replace(base.sources[0], key="auto", label="Automatic suggestion",
                            generation_json=json.dumps(self.report))
        return SimpleNamespace(document=CompositeBackground((*base.sources, generated), "auto", base.regions),
                               report=self.report)

    def dialog(self, **kwargs):
        args = dict(folder=str(self.root), select_files=lambda _: [], pool=self.pool)
        args.update(kwargs)
        dialog = DrrCompositeBackgroundDialog(**args)
        self.dialogs.append(dialog)
        return dialog

    def wait_job(self, dialog):
        deadline = time.monotonic() + 7
        while dialog._worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(5)
        self.assertIsNone(dialog._worker, dialog.status.text())

    def test_defaults_require_two_original_sources_and_guard_direct_call(self):
        empty = self.dialog()
        self.assertFalse(empty.auto_button.isEnabled())
        self.assertFalse(empty.reconstruct_check.isChecked())
        self.assertTrue(empty.preserve_regions_check.isChecked())
        self.assertEqual(empty.max_gap_spin.value(), 60.)
        self.assertFalse(empty.max_gap_spin.isEnabled())
        derived = self.suggestion().document.sources[-1]
        one = self.dialog(initial_document=CompositeBackground((self.source("p", 20.), derived), "p"))
        self.assertFalse(one.auto_button.isEnabled())
        one.auto_suggest()
        self.assertIsNone(one._worker)
        self.assertIn("two original", one.status.text())
        two = self.dialog(initial_document=self.document())
        self.assertTrue(two.auto_button.isEnabled())
        two.reconstruct_check.setChecked(True)
        self.assertTrue(two.max_gap_spin.isEnabled())
        self.assertIn("texture", two.reconstruct_check.toolTip())

    def test_auto_runs_off_gui_thread_one_undo_preserves_selection_and_does_not_save(self):
        original = self.document()
        dialog = self.dialog(initial_document=original)
        dialog.source_table.setCurrentIndex(dialog.source_model.index(1, 0))
        suggested = self.suggestion(original)
        calls = []
        def suggest(document, reconstruct, gap, preserve):
            calls.append((threading.get_ident(), document, reconstruct, gap, preserve))
            return suggested
        with patch("ui_qt.drr_composite_background_dialog._suggest_auto", suggest):
            dialog.auto_button.click()
            self.assertFalse(dialog.auto_button.isEnabled())
            self.assertFalse(dialog.save_button.isEnabled())
            self.wait_job(dialog)
        self.assertNotEqual(calls[0][0], threading.get_ident())
        self.assertIs(calls[0][1], original)
        self.assertEqual(calls[0][2:], (False, .06, True))
        self.assertEqual(dialog.document.default_source, suggested.document.default_source)
        self.assertEqual(dialog.document.regions, suggested.document.regions)
        self.assertEqual(len(dialog.document.sources), len(suggested.document.sources))
        for actual, expected in zip(dialog.document.sources, suggested.document.sources):
            self.assertIs(actual, expected)
        self.assertEqual(dialog.selected_source().key, "n")
        self.assertEqual(len(dialog._undo), 1)
        self.assertIsNone(dialog.saved_path)
        self.assertEqual(dialog.result(), QDialog.Rejected)
        self.assertFalse(list(self.root.glob("*.csv")))
        dialog.undo()
        self.assertIs(dialog.document, original)
        self.assertEqual(len(dialog._undo), 0)

    def test_settings_are_frozen_at_dispatch_and_converted_to_ev(self):
        dialog = self.dialog(initial_document=self.document())
        dialog.reconstruct_check.setChecked(True)
        dialog.max_gap_spin.setValue(85)
        dialog.preserve_regions_check.setChecked(False)
        calls = []
        def suggest(document, *settings):
            calls.append(settings)
            return self.suggestion(document)
        with patch("ui_qt.drr_composite_background_dialog._suggest_auto", suggest):
            dialog.auto_suggest()
            dialog.max_gap_spin.setValue(100)
            self.wait_job(dialog)
        self.assertEqual(calls, [(True, .085, False)])

    def test_core_adapter_passes_settings_and_manual_region_policy(self):
        original = self.document()
        calls = []
        def suggest(document, **kwargs):
            calls.append((document, kwargs))
            return "result"
        with patch("core.drr_auto_composite.suggest_composite_background", suggest):
            self.assertEqual(_suggest_auto(original, True, .055, False), "result")
        self.assertIs(calls[0][0], original)
        settings = calls[0][1]["settings"]
        self.assertTrue(settings.reconstruct_gaps)
        self.assertEqual(settings.max_gap_ev, .055)
        self.assertFalse(calls[0][1]["preserve_regions"])

    def test_generated_snapshots_disable_reload_and_low_level_rejects_it(self):
        document = self.suggestion().document
        dialog = self.dialog(initial_document=document)
        dialog.source_table.setCurrentIndex(dialog.source_model.index(2, 0))
        self.assertEqual(dialog.source_model.item(2, 1).text(), "Estimated")
        self.assertFalse(dialog.frame_combo.isEnabled())
        self.assertFalse(dialog.reload_button.isEnabled())
        with patch("ui_qt.drr_composite_background_dialog.load_composite_source",
                   side_effect=AssertionError("Generated source must not read original CSV")):
            dialog.reload_source()
            self.assertIsNone(dialog._worker)
            with self.assertRaisesRegex(ValueError, "frozen snapshots"):
                _reload_source(str(self.root), document, "auto", "first")
        self.assertIs(dialog.document, document)
        dialog.source_table.setCurrentIndex(dialog.source_model.index(0, 0))
        self.assertTrue(dialog.frame_combo.isEnabled())
        self.assertTrue(dialog.reload_button.isEnabled())
        self.assertEqual(dialog.frame_combo.currentData(), "last")

    def test_hatches_and_review_text_follow_only_effectively_used_ranges_and_undo(self):
        suggested = self.suggestion().document
        dialog = self.dialog(initial_document=suggested)
        self.assertEqual(dialog._diagnostic_spans["estimated_regions"], ((1.2, 1.3),))
        self.assertIn("Review 3 marked range(s)", dialog.auto_review.text())
        self.assertIn("model estimates", dialog.auto_review.text())
        self.assertIn("retain a raw source", dialog.auto_review.text())
        self.assertIn("insufficient support", dialog.auto_review.text())
        self.assertIn("1 range(s) lack reliable interference texture", dialog.auto_review.text())
        self.assertEqual({p.get_hatch() for p in dialog.ax.patches if p.get_hatch()}, {"///", "xxx", "..."})
        self.assertIn("Estimated (model)", [t.get_text() for t in dialog.ax.get_legend().get_texts()])
        self.assertEqual(dialog.ax.get_xlim(), (1., 2.))
        dialog.lower_edit.setText("1.2")
        dialog.upper_edit.setText("1.25")
        dialog.use_source()
        self.assertEqual(dialog._diagnostic_spans["estimated_regions"], ((1.25, 1.3),))
        dialog.new_region()
        dialog.lower_edit.setText("1.0")
        dialog.upper_edit.setText("2.0")
        dialog.use_source()
        self.assertTrue(all(not spans for spans in dialog._diagnostic_spans.values()))
        self.assertEqual(dialog._texture_missing_spans, ())
        self.assertTrue(dialog.auto_review.isHidden())
        self.assertFalse(any(p.get_hatch() for p in dialog.ax.patches))
        dialog.undo()
        self.assertEqual(dialog._diagnostic_spans["estimated_regions"], ((1.25, 1.3),))

    def test_generated_without_estimates_still_requires_review(self):
        self.report.update(estimated_regions=[], unresolved_regions=[], unsupported_regions=[], texture_missing_regions=[])
        dialog = self.dialog(initial_document=self.suggestion().document)
        self.assertEqual(dialog.source_model.item(2, 1).text(), "Generated")
        self.assertIn("Automatic source choices need review", dialog.auto_review.text())
        self.assertFalse(dialog.auto_review.isHidden())

    def test_failure_keeps_document_and_reenables_actions(self):
        original = self.document()
        dialog = self.dialog(initial_document=original)
        with patch("ui_qt.drr_composite_background_dialog._suggest_auto", side_effect=ValueError("No valid overlap")):
            dialog.auto_suggest()
            self.wait_job(dialog)
        self.assertIs(dialog.document, original)
        self.assertEqual(dialog._undo, [])
        self.assertIn("No valid overlap", dialog.status.text())
        self.assertTrue(dialog.auto_button.isEnabled())
        self.assertTrue(dialog.save_button.isEnabled())

    def test_outdated_result_cannot_overwrite_new_document(self):
        original = self.document()
        dialog = self.dialog(initial_document=original)
        release = threading.Event()
        def suggest(*_):
            release.wait(5)
            return self.suggestion(original)
        with patch("ui_qt.drr_composite_background_dialog._suggest_auto", suggest):
            try:
                dialog.auto_suggest()
                replacement = replace(original, default_source="n")
                dialog._commit(replacement)
                committed = dialog.document
                self.assertEqual(committed.default_source, "n")
                self.assertEqual(committed.regions, replacement.regions)
            finally:
                release.set()
            self.wait_job(dialog)
        self.assertIs(dialog.document, committed)
        self.assertEqual(len(dialog._undo), 1)
        self.assertIn("outdated suggestion was discarded", dialog.status.text())

    def test_escape_cancels_running_suggestion_without_mutating_draft(self):
        original = self.document()
        dialog = self.dialog(initial_document=original)
        dialog.show()
        self.app.processEvents()
        release = threading.Event()
        def suggest(*_):
            release.wait(5)
            return self.suggestion(original)
        with patch("ui_qt.drr_composite_background_dialog._suggest_auto", suggest):
            try:
                dialog.auto_suggest()
                worker = dialog._worker
                QTest.keyClick(dialog, Qt.Key_Escape)
                self.assertTrue(worker.cancellation.cancelled)
            finally:
                release.set()
            self.wait_job(dialog)
        self.assertIs(dialog.document, original)
        self.assertEqual(dialog._undo, [])
        self.assertIsNone(dialog.saved_path)

    def test_reopen_keeps_frozen_diagnostics_without_original_files(self):
        saved = self.root / "suggestion.csv"
        save_composite_background(saved, self.suggestion().document)
        dialog = self.dialog(initial_files=(str(saved),))
        self.app.processEvents()
        self.wait_job(dialog)
        self.assertEqual(dialog._diagnostic_spans["estimated_regions"], ((1.2, 1.3),))
        self.assertIn("model estimates", dialog.auto_review.text())
        dialog.source_table.setCurrentIndex(dialog.source_model.index(2, 0))
        self.assertFalse(dialog.reload_button.isEnabled())
        self.assertTrue(dialog.auto_button.isEnabled())

    def test_theme_minimum_layout_and_keyboard_operate_auto_controls(self):
        dialog = self.dialog(initial_document=self.suggestion().document)
        dialog.resize(820, 660)
        dialog.show()
        for theme in ("light", "dark"):
            install_theme(self.app, mode=theme)
            self.app.processEvents()
            dialog.canvas.draw()
            for widget in (dialog.auto_button, dialog.reconstruct_check, dialog.max_gap_spin,
                           dialog.preserve_regions_check, dialog.auto_review, dialog.save_button):
                self.assertTrue(dialog.rect().contains(widget.mapTo(dialog, widget.rect().topLeft())))
                self.assertTrue(dialog.rect().contains(widget.mapTo(dialog, widget.rect().bottomRight())))
            self.assertGreaterEqual(dialog.canvas.height(), 180)
        dialog.reconstruct_check.setFocus()
        QTest.keyClick(dialog.reconstruct_check, Qt.Key_Space)
        self.assertTrue(dialog.reconstruct_check.isChecked())
        QTest.keyClick(dialog.reconstruct_check, Qt.Key_Tab)
        self.assertTrue(dialog.max_gap_spin.hasFocus())


if __name__ == "__main__":
    unittest.main()
