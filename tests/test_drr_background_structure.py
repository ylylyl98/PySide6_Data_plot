from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from core.loader import DataCube, load_dat
from core.drr_fitted_background import BackgroundFitSettings, estimate_background, export_references, _robust_gain_tilt, _robust_fit
from core.drr_background_cache import fit_cached
from core.drr_background_structure import shared_texture, _local_quadratic_trend
from core.processing import apply_sg_derivative_energy


def structured_sample():
    energy = np.linspace(1.4, 2.05, 1301)
    gate = np.linspace(-1, 1, 81)
    x = (energy-energy.mean())/.325
    envelope = np.exp(8+.3*x+.12*x*x+gate[:,None]*(.2+.04*x))
    fringe = .003*np.sin(2*np.pi*energy/.0047)+.002*np.cos(2*np.pi*energy/.0081)
    background = envelope*np.exp(fringe)
    centers = 1.55+.18*(gate+1)/2
    peaks = .12*np.exp(-.5*((energy-centers[:,None])/.0025)**2)
    peaks -= .09*np.exp(-.5*((energy-1.81)/.003)**2)
    rng = np.random.default_rng(921)
    raw = background*(1+peaks)*np.exp(rng.normal(0,.0004,background.shape))
    return DataCube(energy,gate,raw,'Gate (V)','structured','counts'), background, peaks, centers


class BackgroundStructureTests(unittest.TestCase):
    def test_texture_uses_two_available_gate_groups_when_one_scan_half_is_protected(self):
        energy = np.linspace(1.4, 2.05, 1301)
        gate = np.linspace(-1, 1, 100)
        fringe = .003*np.sin(2*np.pi*energy/.0061)
        rng = np.random.default_rng(41)
        values = fringe+gate[:,None]*(.2+.1*energy+.03*energy**2)
        values += rng.normal(0, .0002, values.shape)
        masks = np.ones(values.shape, bool)
        # The moving resonance occupies the first 60 gates in this band.
        masks[:60, (energy>1.54)&(energy<1.73)] = False
        values[:60] += .3*np.exp(-.5*((energy-1.63)/.005)**2)
        # No independent sample exists in this stationary protected band.
        masks[:, (energy>1.78)&(energy<1.83)] = False
        values += .2*np.exp(-.5*((energy-1.805)/.003)**2)
        correction, coverage, _ = shared_texture(energy, values, masks)
        band = (energy>1.58)&(energy<1.69)
        self.assertGreater(float(np.mean(coverage[band]>.9)), .95)
        self.assertLess(float(np.sqrt(np.mean((correction[band]-fringe[band])**2))), .0004)
        np.testing.assert_array_equal(correction[~masks.any(axis=0)], 0.)
        # If only eleven unprotected gates remain, do not fit a template there.
        masks[:89, (energy>1.54)&(energy<1.73)] = False
        correction, _, _ = shared_texture(energy, values, masks)
        np.testing.assert_array_equal(correction[band], 0.)

    def test_energy_trend_preserves_quadratics_on_nonuniform_detector_grid(self):
        energy = np.sort(1239.841984/np.linspace(580, 870, 1000))
        values = np.array([2+3*energy-4*energy**2, -1+.7*energy+.2*energy**2])
        np.testing.assert_allclose(_local_quadratic_trend(energy, values, .035), values, atol=3e-13, rtol=0)

    def test_local_masks_recover_support_without_sacrificing_moving_or_stationary_peaks(self):
        raw, background, peaks, centers = structured_sample()
        original = raw.Z.copy()
        settings = BackgroundFitSettings(automatic_protection=True)
        result, ref = estimate_background(raw, settings)
        smooth, smooth_ref = estimate_background(raw, replace(settings, shared_texture=False))
        masks = ref.anchor_masks()
        self.assertGreater(masks.mean(), masks.all(axis=0).mean()+.15)
        for i, center in enumerate(centers):
            self.assertFalse(masks[i,np.argmin(abs(raw.energy-center))])
            self.assertFalse(masks[i,np.argmin(abs(raw.energy-1.81))])
        peak_pixels = peaks > .06
        self.assertLess(float(np.max(abs(result.Z[peak_pixels]-peaks[peak_pixels]))), .006)
        fixed = abs(raw.energy-1.81)<.002
        self.assertLess(float(np.max(abs(result.Z[:,fixed]-peaks[:,fixed]))), .006)
        np.testing.assert_array_equal(ref.log_texture[fixed], 0.)
        quiet = (raw.energy>1.88)&(raw.energy<2.)
        derivative = lambda cube: apply_sg_derivative_energy(cube, derivative=2, window_length=21, polyorder=2)[0].Z
        before = np.sqrt(np.mean(derivative(smooth)[:,quiet]**2))
        after = np.sqrt(np.mean(derivative(result)[:,quiet]**2))
        self.assertLess(after, .4*before)
        np.testing.assert_array_equal(ref.background(include_texture=False), smooth_ref.background())
        np.testing.assert_array_equal(raw.Z, original)

    def test_weak_stationary_peak_is_preserved_by_explicit_manual_protection(self):
        raw, _, _, _ = structured_sample()
        weak = .005*np.exp(-.5*((raw.energy-1.93)/.0015)**2)
        raw = replace(raw,Z=raw.Z*(1+weak))
        settings = BackgroundFitSettings(((1.52,1.76),(1.79,1.83),(1.92,1.94)))
        corrected, ref = estimate_background(raw, settings)
        smooth, _ = estimate_background(raw, replace(settings,shared_texture=False))
        band = (raw.energy>=1.92)&(raw.energy<=1.94)
        np.testing.assert_array_equal(ref.log_texture[band],0.)
        np.testing.assert_array_equal(corrected.Z[:,band],smooth.Z[:,band])
        self.assertTrue(np.any(ref.texture_coverage>.5))

    def test_noise_and_phase_inconsistent_patterns_are_not_removed(self):
        raw, _, _, _ = structured_sample()
        rng = np.random.default_rng(34)
        noise = rng.normal(0,.001,raw.Z.shape)
        masks = np.ones(raw.Z.shape,bool)
        for values in (noise, noise+np.r_[np.ones(40),-np.ones(41)][:,None]*.003*np.sin(2*np.pi*raw.energy/.004)):
            correction, coverage, report = shared_texture(raw.energy,values,masks)
            np.testing.assert_array_equal(correction,0.)
            self.assertEqual(report['corrected_fraction'],0.)
        correction, _, report = shared_texture(raw.energy,noise[:5],masks[:5])
        np.testing.assert_array_equal(correction,0.)
        self.assertEqual(report['status'],'Insufficient gates')

    def test_masked_gain_fits_match_independent_robust_solutions(self):
        rng = np.random.default_rng(80)
        x = np.linspace(-1,1,451)
        values = rng.normal(size=(75,1))+rng.normal(size=(75,1))*x+rng.normal(0,.01,(75,len(x)))
        masks = rng.uniform(size=values.shape)>.35
        values[:,170:177] += 2
        design = np.column_stack([np.ones_like(x),x])
        expected = np.array([_robust_fit(design[m],y[m]) for y,m in zip(values,masks)])
        np.testing.assert_allclose(_robust_gain_tilt(x,values,masks),expected,rtol=1e-9,atol=1e-10)

    def test_cache_and_export_preserve_exact_texture_and_per_gate_masks(self):
        raw, _, _, _ = structured_sample()
        settings = BackgroundFitSettings(automatic_protection=True)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,DPTK_FITTED_CACHE_DIR=directory+'/cache'):
            kwargs=dict(source='sample.csv',signature=(1,2),source_path=Path(directory)/'sample.csv')
            first, ref=fit_cached(raw,settings,**kwargs)
            restored, cached=fit_cached(raw,settings,**kwargs)
            self.assertTrue(cached.cache_hit)
            for key in ('gate_fit_mask','log_texture','texture_coverage'):
                np.testing.assert_array_equal(getattr(cached,key),getattr(ref,key))
            np.testing.assert_array_equal(first.Z,restored.Z)
            files=export_references(directory,[cached])
            saved=load_dat(Path(directory)/files[0])
            np.testing.assert_allclose(saved.Z,ref.background(),rtol=1e-11)
            payload=json.loads((Path(directory)/files[0]).with_suffix('.metadata.json').read_text())['processing']
            np.testing.assert_array_equal(payload['log_texture'],ref.log_texture)
            self.assertEqual(len(payload['protected_regions_by_gate_eV']),len(raw.gate))
            # Corrupted masks cannot silently become valid cached results.
            path=next((Path(directory)/'cache').glob('*.npz'))
            with np.load(path,allow_pickle=False) as stored:
                data={key:stored[key] for key in stored.files}
            data['gate_fit_mask']=np.ones((2,3),bool)
            np.savez_compressed(path,**data)
            _, repaired=fit_cached(raw,settings,**kwargs)
            self.assertFalse(repaired.cache_hit)


if __name__=='__main__':
    unittest.main()
