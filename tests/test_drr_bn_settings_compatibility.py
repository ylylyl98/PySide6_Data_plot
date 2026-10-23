"""Adding multiple fit bands preserves the existing settings constructor."""
import unittest

from core.drr_bn_calibration import BnCalibrationSettings


class BnSettingsCompatibilityTests(unittest.TestCase):
    def test_existing_positional_settings_keep_their_meaning(self):
        core, training, validation = (1.8, 2.0), (1.8, 1.85), (1.9, 1.95)
        settings = BnCalibrationSettings(core, training, validation, .02)
        self.assertEqual(settings.core_region, core)
        self.assertEqual(settings.train_region, training)
        self.assertEqual(settings.validation_region, validation)
        self.assertEqual(settings.taper_width, .02)
        self.assertEqual(settings.train_regions, ())
        bands = ((1.8, 1.82), (1.83, 1.85))
        updated = BnCalibrationSettings(core, training, validation, .02, train_regions=bands)
        self.assertEqual(updated.validation_region, validation)
        self.assertEqual(updated.train_regions, bands)


if __name__ == '__main__':
    unittest.main()
