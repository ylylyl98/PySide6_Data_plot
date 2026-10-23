from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_background_cache import fit_cached
from core.drr_fitted_background import load_intensity, source_signature
from tests.test_drr_fitted_background import SETTINGS, synthetic, write_csv


class ReferenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sample = self.root/'sample.csv'
        self.reference = self.root/'N.csv'
        raw, _, _ = synthetic()
        write_csv(self.sample, raw)
        self.reference_cube = replace(raw, gate=np.zeros(12), Z=np.full((12, len(raw.energy)), 1000.))
        write_csv(self.reference, self.reference_cube)
        self.raw = load_intensity(self.root, self.sample.name)
        self.settings = replace(SETTINGS, texture_reference_source=str(self.reference))
        env = patch.dict(os.environ, DPTK_FITTED_CACHE_DIR=str(self.root/'cache'))
        env.start()
        self.addCleanup(env.stop)

    def fit(self, settings=None):
        return fit_cached(self.raw, settings or self.settings, source=self.sample.name,
                          signature=source_signature(self.sample), source_path=self.sample)

    def rejected(self, energy, sample, anchors, reference_energy, reference_frames, coverage):
        return np.zeros_like(energy), np.zeros_like(energy), dict(status='Rejected', reason='Synthetic validation failure')

    def test_reference_hash_invalidates_fit_even_with_unchanged_file_size_and_timestamp(self):
        with patch('core.drr_reference_texture.reference_texture', side_effect=self.rejected) as model:
            _, first = self.fit()
            _, cached = self.fit()
            self.assertFalse(first.cache_hit)
            self.assertTrue(cached.cache_hit)
            self.assertEqual(model.call_count, 1)
            stamp = self.reference.stat()
            write_csv(self.reference, replace(self.reference_cube, Z=np.full(self.reference_cube.Z.shape, 2000.)))
            self.assertEqual(self.reference.stat().st_size, stamp.st_size)
            os.utime(self.reference, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            _, changed = self.fit()
            self.assertFalse(changed.cache_hit)
            self.assertEqual(model.call_count, 2)
            np.testing.assert_allclose(model.call_args.args[4], np.log(2000.), rtol=0, atol=0)
            self.assertNotEqual(first.diagnostics['reference_texture']['source_metadata']['sha256'],
                                changed.diagnostics['reference_texture']['source_metadata']['sha256'])

    def test_missing_reference_does_not_reuse_saved_fit_but_disabled_reference_is_inactive(self):
        with patch('core.drr_reference_texture.reference_texture', side_effect=self.rejected):
            self.fit()
        self.reference.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.fit()
        with patch('core.drr_texture_reference_io.load_texture_reference') as load:
            _, inactive = self.fit(replace(self.settings, shared_texture=False))
            load.assert_not_called()
        self.assertEqual(inactive.diagnostics['reference_texture']['status'], 'Disabled')

    def test_file_change_during_fit_cannot_publish_a_result_or_cache(self):
        def changed_reference(*args):
            write_csv(self.reference, replace(self.reference_cube, Z=self.reference_cube.Z*2))
            return self.rejected(*args)
        with patch('core.drr_reference_texture.reference_texture', side_effect=changed_reference):
            with self.assertRaisesRegex(ValueError, 'changed after fitting'):
                self.fit()
        self.assertFalse(list((self.root/'cache').glob('*.npz')))


if __name__ == '__main__':
    unittest.main()
