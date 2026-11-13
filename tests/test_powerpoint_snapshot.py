import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ui_qt.quick.powerpoint_snapshot import PresentationImageReader


class SnapshotReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / 'original.pptx'
        self.file.write_bytes(b'original file')
        self.pres = MagicMock()
        self.pres.Saved = -1
        self.pres.FullName = str(self.file)
        self.reader = PresentationImageReader()
        self.calls = []
        def read(path, slide, shape):
            self.calls.append(str(path))
            return b'image'
        self.patch = patch('ui_qt.quick.powerpoint_snapshot.read_shape_image', side_effect=read)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_clean_baseline_uses_disk_only_after_fresh_snapshot(self):
        self.assertEqual(self.reader.read(self.pres, 1, 2), b'image')
        self.assertEqual(self.pres.SaveCopyAs.call_count, 1)
        self.assertEqual(self.reader.read(self.pres, 1, 2), b'image')
        self.assertEqual(self.pres.SaveCopyAs.call_count, 1)
        self.assertEqual(self.calls[-1], str(self.file))

    def test_dirty_deck_never_uses_cached_disk(self):
        self.reader.read(self.pres, 1, 2)
        self.pres.Saved = 0
        self.reader.read(self.pres, 1, 2)
        self.assertEqual(self.pres.SaveCopyAs.call_count, 2)

    def test_disk_change_invalidates_cached_baseline(self):
        self.reader.read(self.pres, 1, 2)
        self.file.write_bytes(b'changed file with different size')
        self.reader.read(self.pres, 1, 2)
        self.assertEqual(self.pres.SaveCopyAs.call_count, 2)

    def test_saved_status_changed_during_disk_read_falls_back(self):
        self.reader.read(self.pres, 1, 2)
        def read(path, slide, shape):
            if str(path) == str(self.file): self.pres.Saved = 0
            return b'image'
        with patch('ui_qt.quick.powerpoint_snapshot.read_shape_image', side_effect=read):
            self.reader.read(self.pres, 1, 2)
        self.assertEqual(self.pres.SaveCopyAs.call_count, 2)

    def test_disk_read_failure_falls_back_to_snapshot(self):
        self.reader.read(self.pres, 1, 2)
        def read(path, slide, shape):
            if str(path) == str(self.file): raise OSError('locked')
            return b'image'
        with patch('ui_qt.quick.powerpoint_snapshot.read_shape_image', side_effect=read):
            self.assertEqual(self.reader.read(self.pres, 1, 2), b'image')
        self.assertEqual(self.pres.SaveCopyAs.call_count, 2)

    def test_other_presentation_never_reuses_baseline(self):
        self.reader.read(self.pres, 1, 2)
        other = MagicMock()
        other.Saved = -1
        other.FullName = str(self.file)
        self.reader.read(other, 1, 2)
        self.assertEqual(other.SaveCopyAs.call_count, 1)

    def test_different_disk_bytes_never_create_baseline(self):
        def read(path, slide, shape):
            return b'disk-image' if str(path) == str(self.file) else b'live-image'
        with patch('ui_qt.quick.powerpoint_snapshot.read_shape_image', side_effect=read):
            self.assertEqual(self.reader.read(self.pres, 1, 2), b'live-image')
            self.assertEqual(self.reader.read(self.pres, 1, 2), b'live-image')
        self.assertEqual(self.pres.SaveCopyAs.call_count, 2)
        self.assertFalse(self.reader._baselines)

    def test_baselines_are_lru_bounded_and_clearable(self):
        for shape in range(16):
            self.reader.read(self.pres, 1, shape)
        self.reader.read(self.pres, 1, 0)  # Refresh the oldest entry.
        self.reader.read(self.pres, 1, 16)
        self.assertEqual(len(self.reader._baselines), 16)
        self.assertIn((1, 0), self.reader._baselines)
        self.assertNotIn((1, 1), self.reader._baselines)
        self.reader.clear()
        self.assertFalse(self.reader._baselines)
        self.assertFalse(self.reader._verified_images)

    def test_dirty_deck_uses_only_live_byte_verified_candidate(self):
        self.reader.read(self.pres, 1, 2)
        self.pres.Saved = 0
        with patch.object(self.reader, 'matches_live_image', return_value=True) as match:
            self.assertEqual(self.reader.read(self.pres, 1, 2), b'image')
        match.assert_called_once_with(self.pres, 1, 2, b'image')
        self.assertEqual(self.pres.SaveCopyAs.call_count, 1)

    def test_different_live_candidate_forces_fresh_snapshot(self):
        self.reader.read(self.pres, 1, 2)
        self.pres.Saved = 0
        with patch.object(self.reader, 'matches_live_image', return_value=False), \
             patch('ui_qt.quick.powerpoint_snapshot.read_shape_image', return_value=b'changed image'):
            self.assertEqual(self.reader.read(self.pres, 1, 2), b'changed image')
        self.assertEqual(self.pres.SaveCopyAs.call_count, 2)
        self.assertEqual(self.reader._verified_images[(1, 2)][1], b'changed image')

    def test_first_open_disk_candidate_requires_live_match_even_dirty(self):
        self.pres.Saved = 0
        with patch.object(self.reader, 'matches_live_image', return_value=True) as match:
            self.assertEqual(self.reader.read(self.pres, 1, 2), b'image')
        match.assert_called_once_with(self.pres, 1, 2, b'image')
        self.pres.SaveCopyAs.assert_not_called()
        self.assertFalse(self.reader._baselines)

    def test_cross_proxy_cached_image_requires_current_live_match(self):
        self.reader.remember_verified(self.pres, 1, 2, b'cached')
        other = MagicMock(Saved=0, FullName=str(self.file))
        with patch.object(self.reader, 'matches_live_image', return_value=True) as match:
            self.assertEqual(self.reader.read(other, 1, 2), b'cached')
        match.assert_called_once_with(other, 1, 2, b'cached')
        other.SaveCopyAs.assert_not_called()

    def test_other_presentation_different_live_picture_never_uses_candidate(self):
        self.reader.remember_verified(self.pres, 1, 2, b'old')
        other = MagicMock(Saved=0, FullName=str(self.file))
        def read(path, slide, shape):
            return b'disk' if str(path) == str(self.file) else b'current'
        with patch('ui_qt.quick.powerpoint_snapshot.read_shape_image', side_effect=read), \
             patch.object(self.reader, 'matches_live_image', return_value=False) as match:
            self.assertEqual(self.reader.read(other, 1, 2), b'current')
        self.assertEqual(match.call_count, 2)
        other.SaveCopyAs.assert_called_once()

    def test_live_export_error_or_stale_disk_falls_back(self):
        self.pres.Saved = 0
        shapes = self.pres.Slides.FindBySlideID.return_value.Shapes
        shapes.Count = 1
        shapes.return_value.Id = 2
        with patch('ui_qt.quick.powerpoint_snapshot.read_shape_image', return_value=b'\x89PNG\r\n\x1a\nimage'), \
             patch('ui_qt.quick.powerpoint_emf.export_shape_png', side_effect=OSError('unsupported')) as export:
            self.assertEqual(self.reader.read(self.pres, 1, 2), b'\x89PNG\r\n\x1a\nimage')
        export.assert_called_once()
        self.pres.SaveCopyAs.assert_called_once()

    def test_successfully_updated_image_replaces_cached_original(self):
        self.reader.read(self.pres, 1, 2)
        self.reader.remember_verified(self.pres, 1, 2, b'new image')
        self.pres.Saved = 0
        with patch.object(self.reader, 'matches_live_image', return_value=True) as match:
            self.assertEqual(self.reader.read(self.pres, 1, 2), b'new image')
        match.assert_called_once_with(self.pres, 1, 2, b'new image')
