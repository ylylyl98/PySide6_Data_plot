from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from tempfile import TemporaryDirectory

import unittest
import sys
from ui_qt.quick.powerpoint_service import PowerPointService, _check_safe, _replace_picture_native, _restore_picture_geometry, _verified_image_fingerprint, _slide_shape_state, _assert_slide_shape_state, _select_matches, _is_picture, _find_shape, _covering_pictures


class PowerPointServiceTests(unittest.TestCase):
    def test_unchanged_geometry_performs_no_writes(self):
        class Tracked(SimpleNamespace):
            def __setattr__(self, name, value):
                if name != 'writes' and hasattr(self, 'writes'):
                    self.writes.append((name, value))
                super().__setattr__(name, value)

        picture = Tracked(CropLeft=1, CropRight=2, CropTop=3, CropBottom=4)
        picture.writes = []
        shape = Tracked(Left=10, Top=20, Width=100, Height=80, Rotation=0,
                        LockAspectRatio=-1, PictureFormat=picture)
        shape.writes = []
        props = dict(Left=10, Top=20, Width=100, Height=80, Rotation=0, LockAspectRatio=-1)
        crops = dict(CropLeft=1, CropRight=2, CropTop=3, CropBottom=4)
        _restore_picture_geometry(shape, props, crops)
        self.assertEqual(shape.writes, [])
        self.assertEqual(picture.writes, [])

    def test_changed_crop_triggers_full_geometry_restoration(self):
        class Tracked(SimpleNamespace):
            def __setattr__(self, name, value):
                if name != 'writes' and hasattr(self, 'writes'):
                    self.writes.append((name, value))
                super().__setattr__(name, value)

        picture = Tracked(CropLeft=9, CropRight=2, CropTop=3, CropBottom=4)
        picture.writes = []
        shape = Tracked(Left=10, Top=20, Width=100, Height=80, Rotation=0,
                        LockAspectRatio=-1, PictureFormat=picture)
        shape.writes = []
        props = dict(Left=10, Top=20, Width=100, Height=80, Rotation=0, LockAspectRatio=-1)
        crops = dict(CropLeft=1, CropRight=2, CropTop=3, CropBottom=4)
        _restore_picture_geometry(shape, props, crops)
        self.assertEqual(picture.CropLeft, 1)
        self.assertEqual(len(picture.writes), 4)
        self.assertIn(('LockAspectRatio', 0), shape.writes)
        self.assertIn(('LockAspectRatio', -1), shape.writes)
        self.assertGreaterEqual(len(shape.writes), 11)

    def test_failed_crop_restoration_is_rejected(self):
        class UnwritableCrop:
            CropLeft = property(lambda self: 9, lambda self, value: None)
            CropRight = 2
            CropTop = 3
            CropBottom = 4

        shape = SimpleNamespace(Left=10, Top=20, Width=100, Height=80, Rotation=0,
                                LockAspectRatio=-1, PictureFormat=UnwritableCrop())
        props = dict(Left=10, Top=20, Width=100, Height=80, Rotation=0, LockAspectRatio=-1)
        crops = dict(CropLeft=1, CropRight=2, CropTop=3, CropBottom=4)
        with self.assertRaisesRegex(ValueError, 'Cannot preserve picture crop CropLeft'):
            _restore_picture_geometry(shape, props, crops)

    def test_rejects_non_picture(self):
        with self.assertRaises(ValueError):
            _check_safe(None, SimpleNamespace(Type=6))


    def test_missing_binding_does_not_mutate_deck(self):
        service = PowerPointService()
        service.update_figure('missing', 'missing.png')
        assert service._commands.get_nowait() == ('update', 'missing', 'missing.png', None)
        service.close()


    def test_rejects_picture_with_action(self):
        old = MagicMock()
        old.Type = 13
        type(old).ParentGroup = property(lambda _: (_ for _ in ()).throw(RuntimeError()))
        old.ActionSettings.return_value.Action = 7
        with self.assertRaisesRegex(ValueError, 'action'):
            _check_safe(MagicMock(), old)


    def test_bind_preserves_multiple_targets(self):
        bindings = {}
        record = SimpleNamespace(record_id='ABC')
        for shape_id in (1, 2):
            shape = MagicMock()
            shape.Id = shape_id
            slide = MagicMock()
            slide.Shapes.Count = 1
            slide.Shapes.return_value = shape
            pres = MagicMock()
            pres.Slides.FindBySlideID.return_value = slide
            PowerPointService._bind_target(bindings, (pres, 10, shape_id), record)
        assert len(bindings['abc']) == 2


    def test_root_and_bind_commands_are_queued(self):
        service = PowerPointService()
        service.set_root('experiment')
        service.bind_png('figure.png')
        assert service._commands.get_nowait() == ('root', 'experiment')
        assert service._commands.get_nowait() == ('bind', 'figure.png')
        service.close()


    def test_rebinding_removes_prior_record(self):
        shape = MagicMock()
        shape.Id = 1
        slide = MagicMock()
        slide.Shapes.Count = 1
        slide.Shapes.return_value = shape
        pres = MagicMock()
        pres.Slides.FindBySlideID.return_value = slide
        target = (pres, 10, 1)
        bindings = {}
        PowerPointService._bind_target(bindings, target, SimpleNamespace(record_id='old'))
        PowerPointService._bind_target(bindings, target, SimpleNamespace(record_id='new'))
        assert not bindings['old']
        assert bindings['new'] == [target]

    def test_tag_disambiguates_only_pixel_matches(self):
        a = SimpleNamespace(record_id='aaa')
        b = SimpleNamespace(record_id='bbb')
        self.assertEqual(_select_matches([a, b], 'BBB'), [b])
        self.assertEqual(_select_matches([a], 'BBB'), [])
        self.assertEqual(_select_matches([a, b], ''), [a, b])

    def test_picture_placeholder_guard(self):
        self.assertTrue(_is_picture(SimpleNamespace(Type=13)))
        self.assertTrue(_is_picture(SimpleNamespace(Type=14, PlaceholderFormat=SimpleNamespace(ContainedType=13))))
        self.assertFalse(_is_picture(SimpleNamespace(Type=14, PlaceholderFormat=SimpleNamespace(ContainedType=1))))
        self.assertFalse(_is_picture(SimpleNamespace(Type=14)))
        self.assertFalse(_is_picture(SimpleNamespace(Type=6)))

    def test_missing_shape_has_explanatory_error(self):
        slide = MagicMock()
        slide.Shapes.Count = 0
        with self.assertRaisesRegex(ValueError, 'associated picture no longer exists'):
            _find_shape(slide, 25)

    def test_name_hint_checks_identity_and_falls_back(self):
        wrong = SimpleNamespace(Id=10)
        right = SimpleNamespace(Id=25)
        slide = MagicMock()
        slide.Shapes.Count = 2
        slide.Shapes.side_effect = lambda key: wrong if key in ('duplicate', 1) else right
        self.assertIs(_find_shape(slide, 25, 'duplicate'), right)
        self.assertEqual(slide.Shapes.call_args_list, [unittest.mock.call('duplicate'),
                                                      unittest.mock.call(1),
                                                      unittest.mock.call(2)])

    def test_valid_name_hint_avoids_scan(self):
        right = SimpleNamespace(Id=25)
        slide = MagicMock()
        slide.Shapes.return_value = right
        self.assertIs(_find_shape(slide, 25, 'target'), right)
        slide.Shapes.assert_called_once_with('target')

    def test_covering_pictures_respects_zorder_and_area(self):
        def shape(shape_id, z, left, top, width, height, kind=13):
            return SimpleNamespace(Id=shape_id, ZOrderPosition=z, Left=left, Top=top,
                                   Width=width, Height=height, Type=kind, Name=f'picture-{shape_id}')
        target = shape(1, 3, 0, 0, 100, 100)
        shapes = [target, shape(2, 4, 50, 0, 50, 100),
                  shape(3, 2, 0, 0, 100, 100), shape(4, 5, 200, 0, 100, 100),
                  shape(5, 6, 95, 0, 5, 100), shape(6, 7, 0, 0, 100, 100, 1)]
        slide = MagicMock()
        slide.Shapes.Count = len(shapes)
        slide.Shapes.side_effect = lambda index: shapes[index - 1]
        self.assertEqual(_covering_pictures(slide, target), ['picture-2'])

    def test_slide_shape_state_detects_changed_identity_or_geometry(self):
        shape = SimpleNamespace(Id=9, Type=14, Left=10, Top=20, Width=30, Height=40, Rotation=0)
        slide = MagicMock()
        slide.Shapes.Count = 1
        slide.Shapes.return_value = shape
        state = _slide_shape_state(slide)
        _assert_slide_shape_state(slide, state)
        shape.Width = 31
        with self.assertRaisesRegex(ValueError, 'changed slide shape 9'):
            _assert_slide_shape_state(slide, state)
        shape.Width = 30
        shape.Id = 10
        with self.assertRaisesRegex(ValueError, 'shape identities'):
            _assert_slide_shape_state(slide, state)

    def test_picture_placeholder_passes_safe_check(self):
        shape = MagicMock()
        shape.Type = 14
        shape.PlaceholderFormat.ContainedType = 13
        type(shape).ParentGroup = property(lambda _: (_ for _ in ()).throw(RuntimeError()))
        shape.ActionSettings.return_value.Action = 0
        slide = MagicMock()
        slide.TimeLine.MainSequence.Count = 0
        slide.TimeLine.InteractiveSequences.Count = 0
        _check_safe(slide, shape)

    @unittest.skipUnless(sys.platform == 'win32', 'Native PowerPoint mock requires pywin32')
    def test_native_failure_attempts_original_image_recovery(self):
        slide = MagicMock()
        shape = MagicMock()
        shape.Id, shape.Type, shape.Name = 3, 13, 'Picture 3'
        shape.Left, shape.Top, shape.Width, shape.Height, shape.Rotation = 1, 2, 100, 80, 0
        shape.LockAspectRatio, shape.ZOrderPosition = 0, 1
        shape.HorizontalFlip = shape.VerticalFlip = 0
        shape.AlternativeText = ''
        shape.Tags.Count = 0
        for name in ('CropLeft', 'CropRight', 'CropTop', 'CropBottom'):
            setattr(shape.PictureFormat, name, 0)
        slide.SlideID, slide.SlideIndex = 42, 1
        slide.Parent.FullName = 'example.pptx'
        slide.Parent.Windows.return_value.Caption = 'example.pptx'
        slide.Shapes.Count = 1
        slide.Shapes.return_value = shape
        with TemporaryDirectory() as directory:
            new = Path(directory) / 'new.png'
            original = Path(directory) / 'old.png'
            new.write_bytes(b'new')
            original.write_bytes(b'old')
            with patch('ui_qt.quick.powerpoint_service._check_safe'), \
                 patch('ui_qt.quick.powerpoint_service._restore_picture_geometry'), \
                 patch('win32gui.EnumWindows', side_effect=lambda callback, _: callback(123, None)), \
                 patch('win32gui.GetClassName', return_value='PPTFrameClass'), \
                 patch('win32gui.GetWindowText', return_value='example.pptx - PowerPoint'), \
                 patch('win32gui.SetForegroundWindow'), \
                 patch('win32gui.GetForegroundWindow', return_value=123), \
                 patch('core.pptx_figure_index._fingerprint', side_effect=lambda value: value) as fingerprint, \
                 patch('core.pptx_figure_index.read_shape_image', side_effect=[b'new', b'old']) as embedded_read, \
                 patch('ui_qt.quick.powerpoint_native_replace.replace_picture_from_file', side_effect=[RuntimeError('dialog failed'), None]) as native:
                with self.assertRaisesRegex(ValueError, 'replacement PNG changed'):
                    _replace_picture_native(slide, shape, new, 'new-id', original,
                                            replacement_fingerprint=b'old-fingerprint',
                                            replacement_image_bytes=b'old-file')
                native.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, 'original picture restored'):
                    _replace_picture_native(slide, shape, new, 'new-id', original,
                                            replacement_fingerprint=b'new', original_fingerprint=b'old',
                                            replacement_image_bytes=b'new', original_image_bytes=b'old')
                self.assertEqual(embedded_read.call_count, 2)  # Both checks read actual PowerPoint content.
                self.assertEqual(fingerprint.call_count, 1)  # Exact original bytes reuse the trusted hash.
                self.assertEqual(native.call_count, 2)
                self.assertEqual(native.call_args_list[1].args[-1], original)

    def test_rebinding_same_tag_does_not_write_tag(self):
        shape = MagicMock()
        shape.Id = 1
        shape.Tags.return_value = 'ABC'
        slide = MagicMock()
        slide.Shapes.Count = 1
        slide.Shapes.return_value = shape
        pres = MagicMock()
        pres.Slides.FindBySlideID.return_value = slide
        PowerPointService._bind_target({}, (pres, 10, 1), SimpleNamespace(record_id='abc'))
        shape.Tags.Add.assert_not_called()

    def test_unpaired_stale_fingerprint_does_not_bypass_verification(self):
        with patch('core.pptx_figure_index._fingerprint', return_value='new-hash') as fingerprint:
            self.assertEqual(_verified_image_fingerprint(b'new-file', None, 'old-hash'), 'new-hash')
            fingerprint.assert_called_once_with(b'new-file')
            fingerprint.reset_mock()
            self.assertEqual(_verified_image_fingerprint(b'new-file', b'old-file', 'old-hash'), 'new-hash')
            fingerprint.assert_called_once_with(b'new-file')
