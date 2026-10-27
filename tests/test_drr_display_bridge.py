"""The Quick boundary keeps legacy scientific/state semantics behind commands."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QLocale, QSettings, QSignalBlocker, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.loader import DataCube
from tests.ui_test_helpers import dispose_owned_window
from ui_qt.common import LoadedState
from ui_qt.main_window import MainWindow
from ui_qt.quick.drr_display_bridge import DrrDisplayBridge
from ui_qt.theme import install_theme


class DrrDisplayBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode="light")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        settings = QSettings(str(Path(self.tmp.name) / "ui.ini"), QSettings.IniFormat)
        with (patch("ui_qt.main_window.QSettings", return_value=settings),
              patch.object(MainWindow, "_restore_last_folder", lambda self: None),
              patch.object(MainWindow, "_schedule_automatic_update_check", lambda self: None)):
            self.w = MainWindow()
        self.w.setAttribute(Qt.WA_DontShowOnScreen)
        self.w._suspend_drr_autoplot = True
        self.bridge = DrrDisplayBridge(self.w)

    def tearDown(self):
        self.bridge.setActive(False)
        dispose_owned_window(self.w)
        self.tmp.cleanup()

    def test_snapshot_covers_display_and_contains_only_plain_values(self):
        snapshot = self.bridge.snapshot
        self.assertEqual(len(snapshot["numbers"]), 25)
        self.assertEqual(len(snapshot["flags"]), 29)
        self.assertEqual(len(snapshot["choices"]), 6)
        self.assertEqual(len(snapshot["actions"]), 12)
        self.assertEqual(len(snapshot["texts"]), 4)
        json.dumps(snapshot, allow_nan=False)
        self.assertEqual(snapshot["numbers"]["axis.xmin"]["fixKey"], "fix.axis.xmin")
        self.assertEqual(snapshot["numbers"]["second.vmin"]["fixKey"], "")
        snapshot["numbers"]["axis.xmin"]["value"] = 999
        self.assertNotEqual(self.bridge.snapshot["numbers"]["axis.xmin"]["value"], 999)

    def test_scientific_commit_keeps_precision_and_emits_once(self):
        spin = self.w.drr_second_vmin_spin
        spin.setValue(-1)
        values, finished = [], []
        spin.valueChanged.connect(values.append)
        spin.editingFinished.connect(lambda: finished.append(True))
        self.bridge.setEditing("second.vmin", True)
        self.assertIs(self.bridge.editingWidget(), spin)
        self.assertTrue(self.bridge.setNumber("second.vmin", "1.234567e-3"))
        self.assertAlmostEqual(spin.value(), .00123457, places=8)
        self.assertEqual(values, [.00123457])
        self.assertEqual(finished, [True])
        self.assertFalse(self.bridge.editing)
        self.assertFalse(spin.lineEdit().isModified())
        # An Enter followed by blur must not create another numeric command.
        self.assertTrue(self.bridge.setNumber("second.vmin", ".00123457"))
        self.assertEqual(values, [.00123457])
        self.assertEqual(finished, [True])

    def test_invalid_nonfinite_and_out_of_range_text_preserve_the_value(self):
        spin = self.w.drr_second_vmin_spin
        spin.setValue(-.25)
        values = []
        spin.valueChanged.connect(values.append)
        for text in ("-", "1e", "nan", "inf", "1e99", ""):
            with self.subTest(text=text):
                self.assertFalse(self.bridge.setNumber("second.vmin", text))
                self.assertFalse(self.bridge.stepNumber("second.vmin", 1, text))
                self.assertEqual(spin.value(), -.25)
                self.assertTrue(self.bridge.error)
                self.assertTrue(self.bridge.snapshot["numbers"]["second.vmin"]["error"])
        self.assertEqual(values, [])
        self.assertTrue(self.bridge.setNumber("second.vmin", "-.125"))
        self.assertEqual(self.bridge.error, "")

    def test_locale_commit_and_full_precision_idle_text_do_not_change_storage(self):
        spin = self.w.drr_second_vmin_spin
        spin.setLocale(QLocale(QLocale.German))
        self.assertTrue(self.bridge.setNumber("second.vmin", "0,12345678"))
        self.assertEqual(spin.value(), .12345678)
        item = self.bridge.snapshot["numbers"]["second.vmin"]
        self.assertEqual(item["editText"], "0,12345678")
        self.assertEqual(item["text"], "0,1235")
        for _ in range(3):
            self.bridge.refresh()
        self.assertEqual(spin.value(), .12345678)

    def test_adaptive_step_merges_pending_exponent_text_and_retains_arrow_debounce(self):
        lower, upper = self.w.drr_spins["vmin"], self.w.drr_spins["vmax"]
        for spin in (lower, upper):
            spin.setDecimals(8)
        lower.setValue(0)
        upper.setValue(200)
        values, finished, during_step = [], [], []
        upper.valueChanged.connect(values.append)
        upper.valueChanged.connect(lambda _: during_step.append(self.bridge.stepping))
        upper.editingFinished.connect(lambda: finished.append(True))
        self.bridge.setEditing("raw.vmax", True)
        self.assertTrue(self.bridge.stepNumber("raw.vmax", 1, "2e0"))
        self.assertEqual(upper.value(), 2.01)
        self.assertEqual(values, [2.01])
        self.assertEqual(during_step, [True])
        self.assertFalse(self.bridge.stepping)
        self.assertEqual(finished, [])
        self.assertFalse(self.bridge.editing)
        self.assertTrue(self.bridge.stepNumber("raw.vmax", -1))
        self.assertEqual(upper.value(), 2)

    def test_same_value_explicit_edit_finishes_once_to_release_pending_history(self):
        spin = self.w.drr_second_vmin_spin
        spin.setValue(.125)
        finished = []
        spin.editingFinished.connect(lambda: finished.append(True))
        self.bridge.setEditing("second.vmin", True)
        self.assertTrue(self.bridge.setNumber("second.vmin", ".125"))
        self.assertEqual(finished, [True])
        self.assertTrue(self.bridge.setNumber("second.vmin", ".125"))
        self.assertEqual(finished, [True])

    def test_silent_auto_updates_are_refreshed_without_feedback_or_precision_loss(self):
        spin = self.w.drr_second_vmin_spin
        values, snapshots = [], []
        spin.valueChanged.connect(values.append)
        self.bridge.snapshotChanged.connect(lambda: snapshots.append(True))
        with QSignalBlocker(spin):
            spin.setValue(.00000123)
        self.w.drr_fix_checks["xmin"].setText("Fix λ")
        self.w.drr_fix_checks["xmin"].setToolTip("Inherited wavelength scope")
        self.w.drr_memory_status.setText("Restored saved file")
        self.bridge.refresh()
        item = self.bridge.snapshot["numbers"]["second.vmin"]
        self.assertEqual(item["value"], .00000123)
        self.assertIn("e", item["text"])
        self.assertEqual(self.bridge.snapshot["flags"]["fix.axis.xmin"]["text"], "Fix λ")
        self.assertEqual(self.bridge.snapshot["texts"]["memoryStatus"]["text"], "Restored saved file")
        self.assertEqual(values, [])
        self.assertEqual(len(snapshots), 1)
        self.bridge.refresh()
        self.assertEqual(len(snapshots), 1)

    def test_shared_boundaries_product_navigation_and_hidden_values_are_independent(self):
        w, bridge = self.w, self.bridge
        cube = DataCube(np.linspace(1, 4, 21), np.array([0., 1.]),
                        np.linspace(-.5, .5, 42).reshape(2, 21), "Gate", "Test", "DR/R")
        w.loaded = LoadedState(mode="DRR", folder="", primary_file="synthetic.xlsx",
                              selected_files=["synthetic.xlsx"], cube=cube)
        w._update_action_states()
        w.drr_spins["xmin"].setValue(1)
        w.drr_spins["xmax"].setValue(4)
        self.assertTrue(bridge.setChoice("regionCount", 2))
        self.assertTrue(bridge.setNumber("boundary.x0", "2"))
        self.assertTrue(bridge.setNumber("boundary.x1", "3"))
        # Unchecked regional bounds intentionally follow data in the current
        # controllers; lock manual limits before testing their retained state.
        self.assertTrue(bridge.setFlag("fix.raw.middle_vmax", True))
        self.assertTrue(bridge.setFlag("fix.second.middle_vmax", True))
        self.assertTrue(bridge.setNumber("raw.middle_vmax", ".7"))
        self.assertTrue(bridge.setNumber("second.middle_vmax", "8"))
        before = w._drr_plot_view
        self.assertTrue(bridge.setChoice("colorProduct", 1))
        self.assertEqual(w._drr_plot_view, before)
        self.assertEqual(bridge.snapshot["regionCount"], 3)
        self.assertTrue(bridge.snapshot["numbers"]["boundary.x1"]["visible"])
        self.assertTrue(bridge.snapshot["numbers"]["second.middle_vmax"]["visible"])
        self.assertFalse(bridge.snapshot["numbers"]["raw.middle_vmax"]["visible"])
        bridge.setChoice("regionCount", 0)
        self.assertFalse(bridge.snapshot["numbers"]["boundary.x0"]["visible"])
        self.assertEqual(w.drr_split_spins["middle_vmax"].value(), .7)
        self.assertEqual(w.drr_second_split_spins["middle_vmax"].value(), 8)

    def test_disabled_choices_and_commands_are_rejected(self):
        combo = self.w.drr_right_yaxis_combo
        combo.model().item(1).setEnabled(False)
        self.bridge.refresh()
        self.assertFalse(self.bridge.snapshot["choices"]["yaxis.right"]["options"][1]["enabled"])
        self.assertFalse(self.bridge.setChoice("yaxis.right", 1))
        self.assertEqual(combo.currentIndex(), 0)
        self.w.drr_auto_x_btn.setEnabled(False)
        self.w.drr_spins["xmin"].setEnabled(False)
        self.w.drr_memory_lock_x_chk.setEnabled(False)
        self.assertFalse(self.bridge.invoke("auto.x"))
        self.assertFalse(self.bridge.setNumber("axis.xmin", "2"))
        self.assertFalse(self.bridge.setFlag("memory.lock_x", True))
        self.assertFalse(self.bridge.invoke("unknown"))

    def test_advanced_axis_choice_exposes_its_controls_and_fix_provenance_remains_manual(self):
        self.bridge.setChoice("yaxis.left", self.w.drr_yaxis_combo.count() - 1)
        self.assertTrue(self.bridge.snapshot["advancedYAxis"])
        self.assertTrue(self.bridge.snapshot["numbers"]["yaxis.a"]["visible"])
        self.bridge.setNumber("yaxis.a", "2.5")
        self.assertEqual(self.w.drr_yaxis_a_spin.value(), 2.5)
        self.bridge.setFlag("fix.axis.xmin", True)
        self.assertEqual(self.bridge.snapshot["flags"]["fix.axis.xmin"]["text"], "Fix M")
        self.assertTrue(self.w.drr_fix_checks["xmin"].isChecked())

    def test_raw_auto_command_retains_second_product_limits(self):
        w = self.w
        cube = DataCube(np.linspace(1, 2, 21), np.array([0., 1.]),
                        np.linspace(-.5, .5, 42).reshape(2, 21), "Gate", "Test", "DR/R")
        w.loaded = LoadedState(mode="DRR", folder="", primary_file="synthetic.xlsx",
                              selected_files=["synthetic.xlsx"], cube=cube)
        w._drr_plot_view = "second"
        w.drr_second_vmin_spin.setValue(-4)
        w.drr_second_vmax_spin.setValue(4)
        w._update_action_states()
        self.assertTrue(self.bridge.invoke("auto.raw"))
        self.assertLess(w.drr_spins["vmin"].value(), w.drr_spins["vmax"].value())
        self.assertEqual(w.drr_second_vmin_spin.value(), -4)
        self.assertEqual(w.drr_second_vmax_spin.value(), 4)

    def test_visible_poll_catches_blocked_changes_and_stops_when_inactive(self):
        self.bridge.setActive(True)
        spin = self.w.drr_second_vmin_spin
        with QSignalBlocker(spin):
            spin.setValue(.125)
        QTest.qWait(250)
        self.assertEqual(self.bridge.snapshot["numbers"]["second.vmin"]["value"], .125)
        self.bridge.setEditing("axis.xmin", True)
        self.bridge.setActive(False)
        self.assertFalse(self.bridge.editing)
        with QSignalBlocker(spin):
            spin.setValue(.25)
        QTest.qWait(250)
        self.assertEqual(self.bridge.snapshot["numbers"]["second.vmin"]["value"], .125)
        self.bridge.refresh()
        self.assertEqual(self.bridge.snapshot["numbers"]["second.vmin"]["value"], .25)


if __name__ == "__main__":
    unittest.main()
