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
from ui_qt.drr_background_dialog import DrrBackgroundDialog, periodic_texture_status
from ui_qt.theme import install_theme


class PeriodicTextureUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode='light')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw, _, _ = synthetic()
        write_csv(self.root / 'sample.csv', self.raw)
        self.pool = QThreadPool()
        self.widgets = []
        environment = patch.dict(os.environ, DPTK_FITTED_CACHE_DIR=str(self.root / 'cache'))
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

    def test_opt_in_dependency_and_keyboard_changes_invalidate_previous_fit(self):
        dialog = self.dialog()
        self.assertFalse(dialog.periodic_check.isChecked())
        self.assertTrue(dialog.periodic_check.isEnabled())
        self.assertFalse(dialog.periodic_components_spin.isEnabled())
        self.assertEqual(dialog.periodic_components_spin.value(), 3)
        self.assertIn('per local fit', dialog.periodic_components_spin.accessibleName())
        self.assertIn('per local spectral fit', dialog.periodic_components_spin.toolTip())
        dialog.fit_preview()
        self.wait(lambda: dialog.reference is not None and dialog.worker is None)
        self.assertTrue(dialog.apply_button.isEnabled())
        dialog.periodic_check.setFocus()
        QTest.keyClick(dialog.periodic_check, Qt.Key_Space)
        self.assertTrue(dialog.periodic_check.isChecked())
        self.assertTrue(dialog.periodic_components_spin.isEnabled())
        self.assertIsNone(dialog.reference)
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.texture_check.setChecked(False)
        self.assertFalse(dialog.periodic_check.isEnabled())
        self.assertFalse(dialog.periodic_components_spin.isEnabled())
        self.assertTrue(dialog.periodic_check.isChecked())
        self.assertTrue(dialog.recipe()['periodic_texture'])
        dialog.texture_check.setChecked(True)
        self.assertTrue(dialog.periodic_check.isEnabled())
        self.assertTrue(dialog.periodic_components_spin.isEnabled())
        generation = dialog.generation
        dialog.periodic_components_spin.setValue(5)
        self.assertGreater(dialog.generation, generation)
        self.assertEqual(dialog.settings().periodic_components, 5)

    def test_recipe_template_restore_and_apply_preserve_periodic_parameters(self):
        store = QSettings(str(self.root / 'settings.ini'), QSettings.IniFormat)
        dialog = self.dialog(settings_store=store)
        dialog.periodic_check.setChecked(True)
        dialog.periodic_components_spin.setValue(4)
        dialog.save_template('Periodic sample')
        saved = json.loads(store.value('drr/background-templates-v1'))['Periodic sample']
        self.assertTrue(saved['periodic_texture'])
        self.assertEqual(saved['periodic_components'], 4)
        dialog.periodic_check.setChecked(False)
        dialog.periodic_components_spin.setValue(1)
        dialog.load_template()
        self.assertTrue(dialog.periodic_check.isChecked())
        self.assertEqual(dialog.periodic_components_spin.value(), 4)
        _, reference = estimate_background(self.raw, SETTINGS, source='sample.csv')
        reference = replace(reference, settings=dialog.settings())
        dialog.ready((dialog.generation, self.raw, reference))
        applied = []
        dialog.applied.connect(applied.append)
        dialog.apply_fit()
        self.assertEqual(applied, [dialog.recipe()])
        self.assertTrue(applied[0]['periodic_texture'])
        self.assertEqual(applied[0]['periodic_components'], 4)
        old_recipe = {key: value for key, value in asdict(SETTINGS).items()
                      if not key.startswith('periodic_')}
        dialog.restore_recipe(old_recipe)
        self.assertFalse(dialog.periodic_check.isChecked())
        self.assertEqual(dialog.periodic_components_spin.value(), 3)

    def test_busy_state_blocks_periodic_controls_and_restores_dependency(self):
        dialog = self.dialog()
        dialog.periodic_check.setChecked(True)
        with patch.object(self.pool, 'start'):
            dialog.fit_preview()
        self.assertIsNotNone(dialog.worker)
        self.assertFalse(dialog.periodic_check.isEnabled())
        self.assertFalse(dialog.periodic_components_spin.isEnabled())
        QTest.keyClick(dialog.periodic_check, Qt.Key_Space)
        self.assertTrue(dialog.periodic_check.isChecked())
        dialog.finished_work()
        self.assertTrue(dialog.periodic_check.isEnabled())
        self.assertTrue(dialog.periodic_components_spin.isEnabled())
        dialog.texture_check.setChecked(False)
        with patch.object(self.pool, 'start'):
            dialog.fit_preview()
        dialog.finished_work()
        self.assertFalse(dialog.periodic_check.isEnabled())
        self.assertFalse(dialog.periodic_components_spin.isEnabled())

    def test_rejected_predictions_are_explicit_and_views_use_final_applied_arrays(self):
        dialog = self.dialog()
        settings = replace(SETTINGS, periodic_texture=True)
        dialog.restore_recipe(asdict(settings))
        _, reference = estimate_background(self.raw, SETTINGS, source='sample.csv')
        rejected = dict(status='Rejected', reason='Held-out bands did not improve.',
                        applied_fraction=0, additional_corrected_fraction=0)
        reference = replace(reference, settings=settings,
                            diagnostics={**reference.diagnostics, 'periodic_texture': rejected})
        dialog.ready((dialog.generation, self.raw, reference))
        self.assertIn('no validated prediction applied', dialog.status.text())
        self.assertIn('Held-out bands did not improve', dialog.status.text())
        self.assertIn('no validated prediction applied', dialog.check_table.item(0, 4).text())
        # Deliberately disagree with the pre-extension shared coverage report.
        coverage = np.ones(len(self.raw.energy))
        texture = .002 * np.sin(self.raw.energy * 800)
        applied = dict(status='Applied', applied_fraction=.18,
                       additional_corrected_fraction=.12, components=[{'period': .007}])
        reference = replace(reference, log_texture=texture, texture_coverage=coverage,
                            diagnostics={**reference.diagnostics,
                                         'shared_texture': {'corrected_fraction': .44},
                                         'periodic_texture': applied})
        dialog.ready((dialog.generation, self.raw, reference))
        self.assertIn('additional corrected spectral points: 12%', dialog.status.text())
        self.assertIn('Corrected spectral points: 100%', dialog.status.text())
        self.assertIn('Components: 1', dialog.status.text())
        dialog.view_combo.setCurrentIndex(2)
        np.testing.assert_allclose(dialog.texture_line.get_ydata(), 100*np.expm1(texture))
        dialog.view_combo.setCurrentIndex(3)
        np.testing.assert_array_equal(dialog.coverage_line.get_ydata(), 100*coverage)
        self.assertIn('not confidence', dialog.coverage_line.get_label())

    def test_no_prediction_cannot_be_reported_as_success(self):
        _, reference = estimate_background(self.raw, SETTINGS)
        reference = replace(reference, settings=replace(SETTINGS, periodic_texture=True),
                            diagnostics={'periodic_texture': {'status': 'Applied',
                                                             'applied_fraction': 0}})
        text = periodic_texture_status(reference)
        self.assertIn('no validated prediction applied', text)
        self.assertNotIn('Periodic gaps: Applied', text)
        disabled = replace(reference, settings=replace(reference.settings, shared_texture=False))
        self.assertIn('inactive while shared texture is off', periodic_texture_status(disabled))

    def test_local_window_component_counts_and_rejection_reason_are_unambiguous(self):
        _, reference = estimate_background(self.raw, SETTINGS)
        report = dict(status='Applied', applied_fraction=.2, additional_corrected_fraction=.1,
                      components=[{}] * 5,
                      windows=[dict(status='Applied', component_count=3, components=[{}] * 3),
                               dict(status='Applied', components=[{}] * 2),
                               dict(status='Rejected', component_count=5)])
        reference = replace(reference, settings=replace(SETTINGS, periodic_texture=True),
                            diagnostics={'periodic_texture': report})
        text = periodic_texture_status(reference)
        self.assertIn('Validated windows: 2', text)
        self.assertIn('Max. components per local fit: 3', text)
        self.assertNotIn('Components: 5', text)
        report = dict(status='Rejected', applied_fraction=0,
                      rejection_reason='Held-out bands did not improve.')
        reference = replace(reference, diagnostics={'periodic_texture': report})
        self.assertIn('Held-out bands did not improve', periodic_texture_status(reference))


if __name__ == '__main__':
    unittest.main()
