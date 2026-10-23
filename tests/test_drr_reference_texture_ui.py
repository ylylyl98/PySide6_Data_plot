from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QEventLoop, QSettings, QThreadPool, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.drr_fitted_background import estimate_background
from tests.test_drr_fitted_background import SETTINGS, synthetic, write_csv
from tests.ui_test_helpers import dispose_owned_window
from ui_qt.drr_background_dialog import DrrBackgroundDialog, reference_texture_status
from ui_qt.theme import install_theme


class ReferenceTextureUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode='light')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw, _, _ = synthetic()
        write_csv(self.root/'sample.csv', self.raw)
        self.reference_path = self.root/'explicitly selected N.csv'
        write_csv(self.reference_path, self.raw)
        self.pool = QThreadPool()
        self.widgets = []
        environment = patch.dict(os.environ, DPTK_FITTED_CACHE_DIR=str(self.root/'cache'))
        environment.start()
        self.addCleanup(environment.stop)

    def tearDown(self):
        for widget in reversed(self.widgets):
            widget.close()
        self.assertTrue(self.pool.waitForDone(5000))
        self.app.processEvents()
        for widget in reversed(self.widgets):
            dispose_owned_window(widget)
        self.tmp.cleanup()

    def wait(self, predicate):
        if not predicate():
            loop, poll, timeout = QEventLoop(), QTimer(), QTimer()
            poll.timeout.connect(lambda: loop.quit() if predicate() else None)
            timeout.setSingleShot(True)
            timeout.timeout.connect(loop.quit)
            poll.start(10)
            timeout.start(15000)
            loop.exec()
            poll.stop()
            timeout.stop()
        self.assertTrue(predicate(), 'Background dialog worker did not finish')

    def dialog(self, **kwargs):
        options = dict(y_axis='bg', settings=asdict(SETTINGS), pool=self.pool)
        options.update(kwargs)
        dialog = DrrBackgroundDialog(self.root, ['sample.csv'], **options)
        self.widgets.append(dialog)
        dialog.show()
        self.wait(lambda: dialog.raw is not None and dialog.worker is None)
        return dialog

    def select_reference(self, dialog):
        with patch('ui_qt.drr_background_dialog.QFileDialog.getOpenFileName',
                   return_value=(str(self.reference_path), 'CSV measurements (*.csv)')) as choose:
            QTest.mouseClick(dialog.reference_browse_button, Qt.LeftButton)
        self.assertEqual(choose.call_count, 1)
        self.assertEqual(choose.call_args.args[3], 'CSV measurements (*.csv)')
        self.assertEqual(dialog.reference_source_edit.text(), str(self.reference_path.resolve()))

    def fitted_reference(self, dialog, report=None):
        _, reference = estimate_background(self.raw, SETTINGS, source='sample.csv')
        return replace(reference, settings=dialog.settings(),
                       diagnostics={**reference.diagnostics, 'reference_texture': report or {}})

    def test_default_is_empty_explicit_read_only_and_cancel_preserves_preview(self):
        dialog = self.dialog()
        self.assertEqual(dialog.settings().texture_reference_source, '')
        self.assertTrue(dialog.reference_source_edit.isReadOnly())
        self.assertTrue(dialog.reference_browse_button.isEnabled())
        self.assertFalse(dialog.reference_clear_button.isEnabled())
        self.assertIn('same sample position and optical path', dialog.reference_source_edit.toolTip())
        self.assertIn('sample envelope is fitted independently', dialog.reference_source_edit.accessibleDescription())
        dialog.reference_source_edit.setFocus()
        QTest.keyClicks(dialog.reference_source_edit, 'unselected.csv')
        self.assertEqual(dialog.reference_source_edit.text(), '')
        reference = self.fitted_reference(dialog)
        dialog.ready((dialog.generation, self.raw, reference))
        generation = dialog.generation
        with patch('ui_qt.drr_background_dialog.QFileDialog.getOpenFileName', return_value=('', '')):
            dialog.browse_reference()
        self.assertEqual(dialog.generation, generation)
        self.assertIs(dialog.reference, reference)

    def test_browse_and_keyboard_clear_invalidate_fit_and_preserve_periodic_choice(self):
        dialog = self.dialog()
        dialog.periodic_check.setChecked(True)
        dialog.ready((dialog.generation, self.raw, self.fitted_reference(dialog)))
        generation = dialog.generation
        self.select_reference(dialog)
        self.assertGreater(dialog.generation, generation)
        self.assertIsNone(dialog.reference)
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertTrue(dialog.periodic_check.isChecked())
        self.assertTrue(dialog.reference_clear_button.isEnabled())
        self.assertEqual(dialog.recipe()['texture_reference_source'], str(self.reference_path.resolve()))
        self.assertIn(str(self.reference_path.resolve()), dialog.reference_source_edit.toolTip())
        dialog.reference_clear_button.setFocus()
        QTest.keyClick(dialog.reference_clear_button, Qt.Key_Space)
        self.assertEqual(dialog.settings().texture_reference_source, '')
        self.assertFalse(dialog.reference_clear_button.isEnabled())
        self.assertTrue(dialog.periodic_check.isChecked())

    def test_template_restore_and_apply_keep_explicit_source_dependency(self):
        store = QSettings(str(self.root/'settings.ini'), QSettings.IniFormat)
        dialog = self.dialog(settings_store=store)
        self.select_reference(dialog)
        dialog.save_template('Same optical path')
        saved = json.loads(store.value('drr/background-templates-v1'))['Same optical path']
        self.assertEqual(saved['texture_reference_source'], str(self.reference_path.resolve()))
        dialog.clear_reference()
        dialog.load_template()
        self.assertEqual(dialog.settings().texture_reference_source, saved['texture_reference_source'])
        dialog.ready((dialog.generation, self.raw, self.fitted_reference(dialog)))
        applied = []
        dialog.applied.connect(applied.append)
        dialog.apply_fit()
        self.assertEqual(applied, [dialog.recipe()])
        self.assertEqual(applied[0]['texture_reference_source'], saved['texture_reference_source'])
        old_recipe = {key: value for key, value in asdict(SETTINGS).items()
                      if key != 'texture_reference_source'}
        dialog.restore_recipe(old_recipe)
        self.assertEqual(dialog.reference_source_edit.text(), '')
        self.assertFalse(dialog.reference_clear_button.isEnabled())

    def test_busy_and_shared_off_block_actions_without_discarding_reference(self):
        dialog = self.dialog()
        self.select_reference(dialog)
        selected = dialog.settings().texture_reference_source
        with patch.object(self.pool, 'start'):
            dialog.fit_preview()
        self.assertFalse(dialog.reference_browse_button.isEnabled())
        self.assertFalse(dialog.reference_clear_button.isEnabled())
        self.assertTrue(dialog.reference_source_edit.isEnabled())
        with patch('ui_qt.drr_background_dialog.QFileDialog.getOpenFileName') as choose:
            dialog.browse_reference()
            dialog.clear_reference()
        choose.assert_not_called()
        self.assertEqual(dialog.reference_source_edit.text(), selected)
        dialog.finished_work()
        self.assertTrue(dialog.reference_browse_button.isEnabled())
        self.assertTrue(dialog.reference_clear_button.isEnabled())
        dialog.texture_check.setChecked(False)
        self.assertFalse(dialog.reference_browse_button.isEnabled())
        self.assertFalse(dialog.reference_clear_button.isEnabled())
        self.assertEqual(dialog.recipe()['texture_reference_source'], selected)
        reference = self.fitted_reference(dialog)
        self.assertIn('inactive while shared texture is off', reference_texture_status(reference))
        dialog.texture_check.setChecked(True)
        self.assertTrue(dialog.reference_browse_button.isEnabled())
        self.assertTrue(dialog.reference_clear_button.isEnabled())

    def test_status_reports_actual_application_and_rejection_in_preview_and_checks(self):
        dialog = self.dialog()
        self.select_reference(dialog)
        rejected = dict(status='Rejected', applied_fraction=0, rejection_reason='Held-out phase shifts were inconsistent.',
                        windows=[dict(status='Rejected')])
        reference = self.fitted_reference(dialog, rejected)
        dialog.ready((dialog.generation, self.raw, reference))
        self.assertIn('no validated correction applied', dialog.status.text())
        self.assertIn('Held-out phase shifts were inconsistent', dialog.status.text())
        self.assertIn('Validated windows: 0', dialog.status.text())
        self.assertIn('no validated correction applied', dialog.check_table.item(0, 4).text())
        applied = dict(status='Applied', applied_fraction=.126, additional_corrected_fraction=.054,
                       windows=[dict(status='Applied'), dict(status='Rejected')],
                       source_metadata=dict(source=str(self.reference_path), sha256='a'*64))
        texture = .002*np.sin(self.raw.energy*800)
        coverage = np.full(len(self.raw.energy), .75)
        reference = replace(reference, log_texture=texture, texture_coverage=coverage,
                            diagnostics={**reference.diagnostics, 'reference_texture': applied})
        dialog.ready((dialog.generation, self.raw, reference))
        self.assertIn('applied at 12.6% of spectral points', dialog.status.text())
        self.assertIn('Additional corrected spectral points: 5.4%', dialog.status.text())
        self.assertIn('Validated windows: 1', dialog.status.text())
        dialog.view_combo.setCurrentIndex(2)
        np.testing.assert_allclose(dialog.texture_line.get_ydata(), 100*np.expm1(texture))
        dialog.view_combo.setCurrentIndex(3)
        np.testing.assert_allclose(dialog.coverage_line.get_ydata(), 100*coverage)
        self.assertIn('not confidence', dialog.coverage_line.get_label())
        no_application = replace(reference, diagnostics={'reference_texture': dict(status='Applied', applied_fraction=0)})
        self.assertIn('no validated correction applied', reference_texture_status(no_application))
        self.assertNotIn('applied at', reference_texture_status(no_application))
        combined = replace(reference, settings=replace(reference.settings, periodic_texture=True),
            diagnostics={**reference.diagnostics, 'periodic_texture': dict(status='Applied',
                applied_fraction=.015, additional_corrected_fraction=0)})
        dialog.ready((dialog.generation, self.raw, combined))
        self.assertIn('additional corrected spectral points: 0%', dialog.status.text())
        self.assertIn('Prediction applied at 1.5% of spectral points', dialog.status.text())

    def test_missing_reference_error_keeps_selected_path_and_disables_apply(self):
        dialog = self.dialog()
        self.select_reference(dialog)
        with patch.object(self.pool, 'start'):
            dialog.fit_preview()
        dialog.failed('Fringe reference file is missing. Select an existing CSV.')
        dialog.finished_work()
        self.assertIn('reference file is missing', dialog.status.text())
        self.assertEqual(dialog.reference_source_edit.text(), str(self.reference_path.resolve()))
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertIsNone(dialog.reference)
        self.assertTrue(dialog.reference_browse_button.isEnabled())


if __name__ == '__main__':
    unittest.main()
