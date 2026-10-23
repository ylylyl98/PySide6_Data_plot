"""Exercise BN entrypoints in the real DRR sidebar and controller wiring."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
import numpy as np
from PySide6.QtCore import QPoint, QRect, QSettings, Qt
from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from tests.ui_test_helpers import dispose_owned_window
from core.loader import DataCube
from ui_qt.common import LoadedState
from ui_qt.main_window import MainWindow
from ui_qt.theme import install_theme


class BnEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.original_font = cls.app.font()
        cls.test_font = -1
        cls.fallback_font = None
        cls.font_patch = None
        system_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
        button_font = QFont(system_font)
        button_font.setWeight(QFont.Weight.DemiBold)
        # Earlier offscreen fixtures register and remove a normal-weight
        # fallback. Qt can retain its normal glyph cache while the themed
        # button's demibold face has no font engine. Probe both actual weights
        # before deciding this class needs its own application font.
        if not all(QFontMetrics(font).inFont("A") for font in (system_font, button_font)):
            font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
            cls.test_font = QFontDatabase.addApplicationFont(str(font_path))
            cls.fallback_font = QFont("DejaVu Sans", 9)
        cls.theme = install_theme(cls.app, mode="light")
        if cls.fallback_font is not None:
            # Offscreen has no system font. A previous test can leave its
            # missing-font lookup cached even after a fallback is registered.
            # Supply this test's owned font before theme/QSS polishing; setting
            # only app.font() afterward leaves some labels using missing glyphs.
            cls.font_patch = patch.object(cls.theme, "_build_font", return_value=cls.fallback_font)
            cls.font_patch.start()
            cls.theme.apply()

    @classmethod
    def set_theme(cls, mode):
        cls.theme.set_mode(mode)

    @classmethod
    def tearDownClass(cls):
        if cls.font_patch is not None:
            cls.font_patch.stop()
        if cls.test_font >= 0:
            cls.app.setFont(cls.original_font)
            QFontDatabase.removeApplicationFont(cls.test_font)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.settings = QSettings(str(self.root / "ui.ini"), QSettings.IniFormat)
        self.set_theme("light")
        with (patch("ui_qt.main_window.QSettings", return_value=self.settings),
              patch.object(MainWindow, "_restore_last_folder", lambda self: None),
              patch.object(MainWindow, "_schedule_automatic_update_check", lambda self: None)):
            self.w = MainWindow()
        self.w.setAttribute(Qt.WA_DontShowOnScreen)
        self.w._suspend_drr_autoplot = True
        self.w.resize(1180, 820)
        self.w.show()
        self.w.tabs.setCurrentIndex(1)
        self.w.drr_sidebar_tabs.setCurrentIndex(0)
        self.app.processEvents()

    def tearDown(self):
        dispose_owned_window(self.w)
        self.set_theme("light")
        self.tmp.cleanup()

    def select_external(self):
        """Select a real temporary recipe; no loader or calibration is invoked."""
        w = self.w
        w.drr_baseline_combo.setCurrentText("External")
        w.current_folder = str(self.root)
        w.drr_selected_files = ["sample.csv"]
        w.drr_baseline_files_manual = ["sample_BNbackground.csv"]
        for name in (*w.drr_selected_files, *w.drr_baseline_files_manual):
            (self.root / name).write_text("Vbg,Vtg,630,640,650\n0,0,100,110,120\n")
        w._drr_assignments = ()
        w._drr_assignments_automatic = False
        w._drr_background_regions = ()
        w.drr_controller._update_drr_selection_labels()
        self.app.processEvents()

    def test_external_entrypoints_require_a_usable_csv_group(self):
        w = self.w
        w.drr_baseline_combo.setCurrentText("External")
        self.app.processEvents()
        self.assertTrue(w.drr_edit_baselines_btn.isVisible())
        self.assertFalse(w.drr_bn_calibration_btn.isVisible())
        self.assertFalse(w.drr_bn_auto_chk.isVisible())
        self.assertFalse(w.drr_bn_calibration_btn.isEnabled())
        self.assertFalse(w.drr_bn_auto_chk.isEnabled())
        self.select_external()
        self.assertFalse(w.drr_bn_calibration_btn.isVisible())
        self.assertFalse(w.drr_bn_auto_chk.isVisible())
        self.assertTrue(w.drr_bn_calibration_btn.isEnabled())
        self.assertTrue(w.drr_bn_auto_chk.isEnabled())
        for mode in ("Self (first frame)", "Fitted (sample)", "Self (last frame)"):
            with self.subTest(mode=mode):
                w.drr_baseline_combo.setCurrentText(mode)
                self.app.processEvents()
                self.assertFalse(w.drr_bn_calibration_btn.isVisible())
                self.assertFalse(w.drr_bn_auto_chk.isVisible())
                self.assertFalse(w.drr_bn_calibration_btn.isEnabled())
                self.assertFalse(w.drr_bn_auto_chk.isEnabled())
                self.assertTrue(w.drr_edit_baselines_btn.isVisible())
        w.drr_baseline_combo.setCurrentText("External")
        w.drr_selected_files = ["map.xlsx"]
        w.drr_controller._update_drr_selection_labels()
        self.assertFalse(w.drr_bn_calibration_btn.isEnabled())
        self.assertFalse(w.drr_bn_auto_chk.isEnabled())

    def test_actual_settings_button_dispatches_unified_workflow(self):
        self.select_external()
        w = self.w
        with patch("ui_qt.drr_background_settings_workflow.edit_settings") as edit:
            QTest.mouseClick(w.drr_edit_baselines_btn, Qt.LeftButton)
            edit.assert_called_once_with(w, initial_kind=None)

    def test_hidden_auto_adapter_still_saves_applied_preference(self):
        self.select_external()
        w = self.w
        with patch("ui_qt.drr_bn_workflow.auto_toggled") as auto:
            w.drr_bn_auto_chk.setChecked(True)
            auto.assert_called_once_with(w, True)
            self.assertTrue(self.settings.value("drr/bn_auto_compare", type=bool))
            w.drr_bn_auto_chk.setChecked(False)
            self.assertEqual(auto.call_count, 2)
            auto.assert_called_with(w, False)
            self.assertFalse(self.settings.value("drr/bn_auto_compare", type=bool))

    def test_calibrated_recipe_cannot_be_pinned_to_another_group(self):
        self.select_external()
        w = self.w
        w.drr_pin_baseline_chk.setChecked(True)
        self.assertTrue(w.drr_pin_baseline_chk.isChecked())
        w.drr_baseline_files_manual = [".drr_bn_calibration/bn_test.csv"]
        w.drr_controller._update_drr_selection_labels()
        self.assertFalse(w.drr_pin_baseline_chk.isChecked())
        self.assertFalse(w.drr_pin_baseline_chk.isEnabled())
        self.assertFalse(w.drr_baseline_combine_combo.isEnabled())
        self.assertIn("this exact group", w.drr_bn_status.text())
        # Also guard a programmatic request, which can check a disabled widget.
        w.drr_pin_baseline_chk.setChecked(True)
        self.assertFalse(w.drr_pin_baseline_chk.isChecked())
        w.drr_baseline_files_manual = ["sample_BNbackground.csv"]
        w.drr_controller._update_drr_selection_labels()
        self.assertTrue(w.drr_pin_baseline_chk.isEnabled())
        self.assertTrue(w.drr_baseline_combine_combo.isEnabled())

    def test_unverified_sg_blocks_direct_async_and_export_derivatives(self):
        self.select_external()
        w = self.w
        w.loaded = LoadedState(mode='DRR', folder=str(self.root), selected_files=w.drr_selected_files,
                               cube=DataCube(np.linspace(1.8, 2., 101), np.array([0., 1.]),
                                             np.ones((2, 101)), 'Gate', 'BN test', 'DR/R'))
        w.drr_baseline_files_manual = ['.drr_bn_calibration/bn_test.csv']
        w.last_plotted_mode = 'DRR'
        w._drr_limits_from_controls = True
        with (patch('ui_qt.drr_bn_workflow.managed_joint_validation_current', return_value=False),
              patch('ui_qt.drr_bn_workflow.ensure_managed_bn_validation') as verify,
              patch.object(w, '_ensure_loaded_matches_ui_params'),
              patch.object(w, '_show_error') as error,
              patch.object(w.thread_pool, 'start') as submit):
            with self.assertRaisesRegex(ValueError, 'BN joint DR/R and D2E'):
                w.drr_controller._drr_cube_with_metadata(2)
            self.assertIs(w.drr_controller._drr_cube_with_metadata(None)[0], w.loaded.cube)
            self.assertFalse(w._prepare_drr_display_derivative(2))
            w._start_export('DRR')
            self.assertIn('BN joint DR/R and D2E', error.call_args.args[0])
            verify.assert_called_with(w)
            submit.assert_not_called()

    def assert_bn_controls_fit(self):
        w = self.w
        viewport = w.drr_tab_scroll.viewport()
        for widget in (w.drr_edit_baselines_btn, w.drr_baseline_summary):
            self.assertTrue(widget.isVisible(), widget.objectName())
            self.assertTrue(QFontMetrics(widget.font()).inFont("A"), widget.objectName())
            bounds = QRect(widget.mapTo(viewport, QPoint(0, 0)), widget.size())
            self.assertTrue(viewport.rect().contains(bounds), (widget.objectName(), bounds, viewport.rect()))
            self.assertTrue(widget.parentWidget().contentsRect().contains(widget.geometry()), widget.objectName())
            self.assertGreaterEqual(widget.height(), widget.minimumSizeHint().height(), widget.objectName())
        self.assertGreaterEqual(w.drr_edit_baselines_btn.width(), w.drr_edit_baselines_btn.sizeHint().width())
        for legacy in (w.drr_bn_calibration_btn, w.drr_bn_auto_chk, w.drr_bn_status):
            self.assertFalse(legacy.isVisible(), legacy.objectName())
        self.assertEqual(w.drr_tab_scroll.horizontalScrollBar().maximum(), 0)
        self.assertEqual(w.drr_tab_scroll.verticalScrollBar().maximum(), 0)

    def test_new_controls_fit_at_1180_by_820_in_both_themes(self):
        self.select_external()
        for theme in ("light", "dark"):
            with self.subTest(theme=theme):
                self.set_theme(theme)
                self.app.processEvents()
                self.assert_bn_controls_fit()


if __name__ == "__main__":
    unittest.main()
