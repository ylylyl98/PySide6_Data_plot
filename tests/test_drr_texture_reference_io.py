from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_texture_reference_io import load_texture_reference, verify_texture_reference
from core.loader import DataCube
from tests.test_drr_fitted_background import write_csv


class TextureReferenceIoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'Nbackground.csv'
        self.sample = self.root / 'sample.csv'
        self.raw = DataCube(np.linspace(1.4, 2., 120), np.zeros(12),
                            np.full((12, 120), 1000.), 'BG', 'reference', 'counts')
        write_csv(self.path, self.raw)

    def test_constant_gate_reference_retains_every_independent_frame(self):
        cube, metadata = load_texture_reference(self.path.name, measurement_path=self.sample)
        np.testing.assert_array_equal(cube.gate, np.arange(12))
        np.testing.assert_array_equal(cube.Z, self.raw.Z)
        self.assertEqual(cube.gate_label, 'Reference frame')
        self.assertEqual(metadata['source'], str(self.path.resolve()))
        self.assertEqual(metadata['role'], 'fringe_only')
        self.assertEqual(len(metadata['sha256']), 64)
        verify_texture_reference(metadata)
        cube.Z[:] = 5
        again, _ = load_texture_reference(str(self.path), measurement_path=self.sample)
        np.testing.assert_array_equal(again.Z, self.raw.Z)

    def test_reference_content_change_with_same_size_and_timestamp_cannot_reuse_raw_cache(self):
        _, old = load_texture_reference(str(self.path), measurement_path=self.sample)
        stamp = self.path.stat()
        write_csv(self.path, replace(self.raw, Z=np.full(self.raw.Z.shape, 2000.)))
        self.assertEqual(self.path.stat().st_size, stamp.st_size)
        os.utime(self.path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        changed, metadata = load_texture_reference(str(self.path), measurement_path=self.sample)
        self.assertNotEqual(metadata['sha256'], old['sha256'])
        np.testing.assert_array_equal(changed.Z, 2000.)
        with self.assertRaisesRegex(ValueError, 'changed after fitting'):
            verify_texture_reference(old)

    def test_invalid_missing_and_deleted_reference_are_explicit_errors(self):
        for source in ('', 'missing.csv'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                load_texture_reference(source, measurement_path=self.sample)
        _, metadata = load_texture_reference(str(self.path), measurement_path=self.sample)
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, 'no longer available'):
            verify_texture_reference(metadata)
        with self.assertRaisesRegex(ValueError, 'identity is unavailable'):
            verify_texture_reference({})
        for value in (0., -1., float('nan')):
            write_csv(self.path, replace(self.raw, Z=np.full(self.raw.Z.shape, value)))
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive'):
                load_texture_reference(str(self.path), measurement_path=self.sample)

    def test_changed_file_during_loading_and_cancellation_are_rejected(self):
        from core.drr_texture_reference_io import _snapshot
        first = _snapshot(self.path)
        with patch('core.drr_texture_reference_io._snapshot',
                   side_effect=[first, {**first, 'sha256': 'changed'}]):
            with self.assertRaisesRegex(ValueError, 'changed while loading'):
                load_texture_reference(str(self.path), measurement_path=self.sample)
        with patch('core.drr_texture_reference_io.check_load_cancelled', side_effect=RuntimeError('cancelled')):
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                load_texture_reference(str(self.path), measurement_path=self.sample)


if __name__ == '__main__':
    unittest.main()
