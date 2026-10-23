from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_background_cache import fit_cached
from core.drr_background_review import blocked_validation
from core.drr_fitted_background import BackgroundFitSettings, estimate_background, export_references
from core.loader import DataCube, load_dat


def sample():
    energy = np.linspace(1.4, 2., 601)
    gate = np.linspace(-1, 1, 31)
    x = (energy-1.7)/.3
    envelope = np.exp(8+.2*x+.08*x*x+gate[:,None]*(.1+.03*x))
    fringe = .003*np.sin(2*np.pi*energy/.009)
    peaks = .06*np.exp(-.5*((energy-1.7)/.003)**2)
    raw = DataCube(energy, gate, envelope*np.exp(fringe)*(1+peaks), 'BG', 'sample', 'counts')
    return raw, BackgroundFitSettings(((1.67,1.73),))


class PeriodicBackgroundIntegrationTests(unittest.TestCase):
    def test_legacy_settings_default_off_and_invalid_recipes_fail(self):
        settings = BackgroundFitSettings.from_mapping(dict(protected_regions=[[1.6,1.7]]))
        self.assertFalse(settings.periodic_texture)
        self.assertEqual(settings.periodic_components,3)
        for values in ({'periodic_texture':'yes'}, {'periodic_components':0},
                       {'periodic_components':6}, {'periodic_components':True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BackgroundFitSettings.from_mapping({**asdict(settings), **values})
        with patch('core.drr_periodic_texture.periodic_texture') as predictor:
            raw, settings = sample()
            estimate_background(raw,settings,assess_quality=False)
            estimate_background(raw,replace(settings,shared_texture=False,periodic_texture=True),assess_quality=False)
        predictor.assert_not_called()

    def test_failed_validation_retains_exact_existing_product(self):
        raw, settings = sample()
        old, old_ref = estimate_background(raw,settings,assess_quality=False)
        zeros = np.zeros(len(raw.energy))
        with patch('core.drr_periodic_texture.periodic_texture',return_value=(zeros,zeros.copy(),
                   {'status':'Rejected','reason':'Held-out prediction failed','corrected_fraction':0.})):
            new, ref = estimate_background(raw,replace(settings,periodic_texture=True))
        np.testing.assert_array_equal(new.Z,old.Z)
        np.testing.assert_array_equal(ref.background(),old_ref.background())
        self.assertEqual(ref.diagnostics['periodic_texture']['additional_corrected_fraction'],0.)
        self.assertTrue(any('added no correction' in warning for warning in ref.diagnostics['warnings']))

    def test_prediction_persists_through_cache_and_exact_background_export(self):
        raw, settings = sample()
        source_copy = raw.Z.copy()
        baseline, baseline_ref = estimate_background(raw,settings,assess_quality=False)
        gap = (raw.energy>1.67)&(raw.energy<1.73)
        prediction = .002*np.sin(2*np.pi*raw.energy/.009)*gap
        coverage = gap.astype(float)
        report = dict(status='Applied',corrected_fraction=float(gap.mean()),components=1,
                      validation='Controlled integration fixture')
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,DPTK_FITTED_CACHE_DIR=directory+'/cache'):
            kwargs = dict(source='sample.csv',signature=(1,2),source_path=Path(directory)/'sample.csv')
            enabled = replace(settings,periodic_texture=True,periodic_components=2)
            with patch('core.drr_periodic_texture.periodic_texture',return_value=(prediction,coverage,report)) as predictor:
                first, ref = fit_cached(raw,enabled,**kwargs)
            self.assertEqual(predictor.call_args.kwargs['maximum_components'],2)
            np.testing.assert_array_equal(first.Z[:,~gap],baseline.Z[:,~gap])
            self.assertGreater(float(np.max(abs(first.Z[:,gap]-baseline.Z[:,gap]))),.001)
            self.assertGreater(ref.diagnostics['periodic_texture']['additional_corrected_fraction'],0)
            with patch('core.drr_background_cache.estimate_background',side_effect=AssertionError('must reuse exact cache')):
                restored, cached = fit_cached(raw,enabled,**kwargs)
            self.assertTrue(cached.cache_hit)
            np.testing.assert_array_equal(first.Z,restored.Z)
            self.assertEqual(cached.diagnostics,json.loads(json.dumps(ref.diagnostics)))
            self.assertEqual(cached.settings,enabled)
            paths = export_references(directory,[cached])
            saved = load_dat(Path(directory)/paths[0])
            np.testing.assert_allclose(saved.Z,cached.background(),rtol=1e-11)
            metadata = json.loads((Path(directory)/paths[0]).with_suffix('.metadata.json').read_text())['processing']
            self.assertTrue(metadata['settings']['periodic_texture'])
            self.assertEqual(metadata['diagnostics']['periodic_texture'],cached.diagnostics['periodic_texture'])
            # Disabling the feature cannot reuse its denominator.
            disabled, disabled_ref = fit_cached(raw,settings,**kwargs)
            self.assertFalse(disabled_ref.cache_hit)
            np.testing.assert_array_equal(disabled.Z,baseline.Z)
        np.testing.assert_array_equal(raw.Z,source_copy)
        np.testing.assert_array_equal(ref.background(include_texture=False),baseline_ref.background(include_texture=False))

    def test_partial_coverage_blends_weighted_prediction_only_once(self):
        raw, settings = sample()
        shared_coverage = np.linspace(0.,1.,len(raw.energy))
        prediction_coverage = np.full_like(raw.energy,.5)
        shared = .003*shared_coverage
        prediction = -.002*prediction_coverage
        with patch('core.drr_background_structure.shared_texture',return_value=(shared,shared_coverage,{})), \
             patch('core.drr_periodic_texture.periodic_texture',return_value=(prediction,prediction_coverage,{})):
            result, ref = estimate_background(raw,replace(settings,periodic_texture=True),assess_quality=False)
        expected = shared + (1-shared_coverage)*prediction
        np.testing.assert_array_equal(ref.log_texture,expected)
        np.testing.assert_array_equal(ref.texture_coverage,shared_coverage+(1-shared_coverage)*prediction_coverage)
        np.testing.assert_allclose(result.Z,raw.Z/(ref.background(include_texture=False)*np.exp(expected))-1,atol=1e-14)
        self.assertEqual(ref.log_texture[-1],shared[-1])

    def test_envelope_heldout_score_does_not_validate_periodic_predictions(self):
        raw, settings = sample()
        _, ref = estimate_background(raw,settings,assess_quality=False)
        ref.settings = replace(settings,periodic_texture=True)
        with patch('core.drr_periodic_texture.periodic_texture') as predictor:
            score = blocked_validation(raw,ref)
        predictor.assert_not_called()
        self.assertEqual(score['scope'],'Smooth envelope only')
        self.assertGreater(score['blocks'],0)


if __name__ == '__main__':
    unittest.main()
