from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_background_review import blocked_validation
from core.drr_fitted_background import BackgroundFitSettings, estimate_background, export_references, read_reference_raw
from core.loader import DataCube, load_dat


def sample():
    energy = np.linspace(1.4, 2., 601)
    gate = np.linspace(-1, 1, 31)
    x = (energy-1.7)/.3
    envelope = np.exp(8+.2*x+.08*x*x+gate[:, None]*(.1+.03*x))
    fringe = .003*np.sin(2*np.pi*energy/.009)
    raw = DataCube(energy, gate, envelope*np.exp(fringe), 'BG', 'sample', 'counts')
    return raw, BackgroundFitSettings(((1.67, 1.73),))


class ReferenceBackgroundIntegrationTests(unittest.TestCase):
    def test_source_defaults_empty_and_roundtrips_without_requiring_a_file(self):
        settings = BackgroundFitSettings.from_mapping(dict(protected_regions=[[1.6, 1.7]]))
        self.assertEqual(settings.texture_reference_source, '')
        for source in ('nearby/reference.csv', 'D:/measurements/reference.csv'):
            selected = replace(settings, texture_reference_source=source)
            self.assertEqual(BackgroundFitSettings.from_mapping(asdict(selected)), selected)
        for invalid in (None, True, ['reference.csv']):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                BackgroundFitSettings.from_mapping({**asdict(settings), 'texture_reference_source': invalid})

    def test_enabled_source_requires_loaded_raw_frames(self):
        raw, settings = sample()
        with self.assertRaisesRegex(ValueError, 'reference was not loaded'):
            estimate_background(raw, replace(settings, texture_reference_source='selected.csv'), assess_quality=False)

    def test_shared_texture_off_retains_source_but_does_not_require_loading(self):
        raw, settings = sample()
        disabled = replace(settings, shared_texture=False)
        original, _ = estimate_background(raw, disabled, assess_quality=False)
        selected = replace(disabled, texture_reference_source='selected.csv')
        result, reference = estimate_background(raw, selected, assess_quality=False)
        np.testing.assert_array_equal(result.Z, original.Z)
        self.assertEqual(reference.settings.texture_reference_source, 'selected.csv')
        self.assertEqual(reference.diagnostics['reference_texture']['status'], 'Disabled')
        self.assertIn('Enable shared texture', reference.diagnostics['reference_texture']['reason'])
        self.assertFalse(reference.metadata()['fringe_reference_used'])
        self.assertFalse(reference.metadata()['bn_reference_used'])

    def test_envelope_holdout_explicitly_clears_reference_and_periodic_options(self):
        raw, settings = sample()
        for automatic in (False, True):
            with self.subTest(automatic=automatic):
                _, reference = estimate_background(raw, replace(settings, automatic_protection=automatic), assess_quality=False)
                reference.settings = replace(reference.settings, texture_reference_source='selected.csv', periodic_texture=True)
                with patch('core.drr_fitted_background.estimate_background', wraps=estimate_background) as fit:
                    score = blocked_validation(raw, reference)
                self.assertGreater(score['blocks'], 0)
                self.assertGreater(fit.call_count, 0)
                for call in fit.call_args_list:
                    temporary = call.args[1]
                    self.assertFalse(temporary.shared_texture)
                    self.assertFalse(temporary.periodic_texture)
                    self.assertEqual(temporary.texture_reference_source, '')
                    self.assertNotIn('texture_reference', call.kwargs)
                self.assertEqual(score['scope'], 'Smooth envelope only')

    def test_rejected_reference_retains_exact_existing_background(self):
        raw, settings = sample()
        baseline, previous = estimate_background(raw, settings, assess_quality=False)
        zero = np.zeros_like(raw.energy)
        report = dict(status='Not validated', reason='Contiguous holdout failed', corrected_fraction=0.)
        with patch('core.drr_reference_texture.reference_texture', return_value=(zero, zero.copy(), report)):
            actual, reference = estimate_background(raw, replace(settings, texture_reference_source='selected.csv'),
                assess_quality=False, texture_reference=raw)
        np.testing.assert_array_equal(actual.Z, baseline.Z)
        np.testing.assert_array_equal(reference.background(), previous.background())
        self.assertEqual(reference.diagnostics['reference_texture']['applied_fraction'], 0.)
        self.assertEqual(reference.diagnostics['reference_texture']['reason'], report['reason'])
        self.assertFalse(reference.metadata()['fringe_reference_used'])

    def test_reference_precedes_periodic_and_each_coverage_is_applied_once(self):
        raw, settings = sample()
        shared_coverage = np.linspace(0., 1., len(raw.energy))
        shared = .003*shared_coverage
        reference_coverage = np.full_like(raw.energy, .5)
        reference_prediction = -.002*reference_coverage
        periodic_coverage = np.full_like(raw.energy, .4)
        periodic_prediction = .001*periodic_coverage
        source = replace(raw, energy=raw.energy[::-1], Z=raw.Z[:4, ::-1], gate=np.arange(4.))
        original = source.Z.copy()
        metadata = dict(source='D:/example/reference.csv', sha256='a'*64, signature=[1, 2], role='fringe_only')
        enabled = replace(settings, texture_reference_source=metadata['source'], periodic_texture=True)
        with patch('core.drr_background_structure.shared_texture', return_value=(shared, shared_coverage, {})), \
                patch('core.drr_reference_texture.reference_texture',
                      return_value=(reference_prediction, reference_coverage, {'status': 'Applied'})) as predictor, \
                patch('core.drr_periodic_texture.periodic_texture',
                      return_value=(periodic_prediction, periodic_coverage, {'status': 'Applied'})), \
                patch('core.drr_texture_reference_io.load_texture_reference', side_effect=AssertionError('pure fit cannot load files')):
            result, reference = estimate_background(raw, enabled, assess_quality=False,
                texture_reference=source, texture_reference_meta=metadata)
        after_reference = shared+(1-shared_coverage)*reference_prediction
        intermediate_coverage = shared_coverage+(1-shared_coverage)*reference_coverage
        expected = after_reference+(1-intermediate_coverage)*periodic_prediction
        expected_coverage = intermediate_coverage+(1-intermediate_coverage)*periodic_coverage
        np.testing.assert_array_equal(reference.log_texture, expected)
        np.testing.assert_array_equal(reference.texture_coverage, expected_coverage)
        np.testing.assert_allclose(result.Z, raw.Z/reference.background()-1, atol=1e-14)
        np.testing.assert_array_equal(predictor.call_args.args[1], np.log(raw.Z))
        np.testing.assert_array_equal(predictor.call_args.args[3], raw.energy)
        np.testing.assert_array_equal(predictor.call_args.args[4], np.log(raw.Z[:4]))
        np.testing.assert_array_equal(predictor.call_args.args[5], shared_coverage)
        np.testing.assert_array_equal(source.Z, original)
        self.assertEqual(reference.log_texture[-1], shared[-1])
        self.assertEqual(reference.diagnostics['reference_texture']['source_metadata'], metadata)
        self.assertTrue(reference.metadata()['fringe_reference_used'])
        self.assertFalse(reference.metadata()['bn_reference_used'])
        self.assertEqual(reference.metadata()['fringe_reference_role'], 'fringe_only')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/export_references(directory, [reference])[0]
            saved = load_dat(path)
            np.testing.assert_allclose(saved.Z, reference.background(), rtol=1e-11)
            exported = json.loads(path.with_suffix('.metadata.json').read_text())['processing']
            self.assertEqual(exported['diagnostics']['reference_texture']['source_metadata'], metadata)
            self.assertTrue(exported['fringe_reference_used'])

    def test_invalid_reference_frames_or_prediction_fail_explicitly(self):
        raw, settings = sample()
        enabled = replace(settings, texture_reference_source='selected.csv')
        invalid = raw.Z.copy()
        invalid[0, 0] = 0.
        with patch('core.drr_reference_texture.reference_texture') as predictor:
            with self.assertRaisesRegex(ValueError, 'positive raw intensities'):
                estimate_background(raw, enabled, assess_quality=False, texture_reference=replace(raw, Z=invalid))
        predictor.assert_not_called()
        bad = np.ones_like(raw.energy)*1.1
        with patch('core.drr_reference_texture.reference_texture', return_value=(np.zeros_like(bad), bad, {})):
            with self.assertRaisesRegex(ValueError, 'invalid correction arrays'):
                estimate_background(raw, enabled, assess_quality=False, texture_reference=raw)

    def test_inspector_rejects_changed_reference_before_loading_sample(self):
        raw, settings = sample()
        _, reference = estimate_background(raw, settings, assess_quality=False)
        reference.settings = replace(settings, texture_reference_source='selected.csv')
        metadata = dict(source='D:/example/reference.csv', sha256='a'*64)
        reference.diagnostics['reference_texture'] = dict(source_metadata=metadata)
        with patch('core.drr_texture_reference_io.verify_texture_reference', side_effect=ValueError('reference changed')) as verify, \
                patch('core.drr_fitted_background.load_intensity') as load:
            with self.assertRaisesRegex(ValueError, 'reference changed'):
                read_reference_raw('unused', reference)
        verify.assert_called_once_with(metadata)
        load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
