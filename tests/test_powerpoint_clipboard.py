from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from PIL import Image
from ui_qt.quick.powerpoint_clipboard import picture_clipboard, ClipboardUnsupported, ClipboardRestoreError

class Fake:
    png_format=49152;exclude_format=49153
    def __init__(self,formats): self.data=dict(formats); self.own=9; self.seq=1; self.lock=False; self.empties=0; self.fail=False; self.dirty=False; self.after_close=None
    def create_owner(self): return 10
    def destroy_owner(self,hwnd): pass
    def open(self,hwnd): assert not self.lock;self.lock=True
    def close(self):
        assert self.lock
        self.lock=False
        if self.dirty:
            self.seq+=1; self.dirty=False
        if self.after_close:
            callback,self.after_close=self.after_close,None
            callback()
    def owner(self): return self.own
    def sequence(self): return self.seq
    def formats(self): assert self.lock;return list(self.data)
    def read(self,fmt,max_bytes=64 * 1024 * 1024):
        assert self.lock
        if len(self.data[fmt]) > max_bytes: raise ClipboardUnsupported('oversized')
        return self.data[fmt]
    def prepare(self,items): return [[k,v] for k,v in items]
    def free(self,handles): pass
    def empty(self): assert self.lock;self.data={};self.own=10;self.seq+=1;self.empties+=1;self.dirty=True
    def put(self,handles):
        assert self.lock
        if self.fail: raise OSError('failed')
        for k,v in handles:self.data[k]=v
    def copy(self,content): assert not self.lock;self.data=dict(content);self.own=20;self.seq+=1

class ClipboardTests(TestCase):
    def image(self, folder, alpha=255):
        path=Path(folder)/'test.png';Image.new('RGBA',(2,2),(1,2,3,alpha)).save(path);return path
    def test_restore_original_bytes_under_lock(self):
        with TemporaryDirectory() as tmp:
            api=Fake([(13,b'a\x00\x00\x00')]); original=dict(api.data)
            with picture_clipboard(self.image(tmp),_api=api) as info:
                self.assertIn(api.exclude_format,api.data);self.assertIn(8,api.data)
            self.assertEqual(api.data,original);self.assertTrue(info['restored'])
            self.assertEqual(info['sequence_before_close'] + 1, info['sequence_after_close'])
            self.assertEqual(info['sequence_after_close'], info['sequence_before_restore'])
            self.assertTrue(info['owner_after_close_is_ours'])
            self.assertTrue(info['owner_before_restore_is_ours'])
    def test_concurrent_copy_preserved(self):
        with TemporaryDirectory() as tmp:
            api=Fake([(13,b'old')])
            with picture_clipboard(self.image(tmp,20),_api=api) as info:
                self.assertNotIn(8,api.data);self.assertEqual(api.data[17][-1],20)
                api.copy([(13,b'new')])
            self.assertEqual(api.data,{13:b'new'});self.assertTrue(info['clipboard_changed'])
    def test_unsupported_refuses_before_empty(self):
        with TemporaryDirectory() as tmp:
            api=Fake([(50000,b'private')])
            with self.assertRaises(ClipboardUnsupported):
                with picture_clipboard(self.image(tmp),_api=api):pass
            self.assertEqual(api.empties,0);self.assertEqual(api.data,{50000:b'private'})
    def test_failed_restoration_raises(self):
        with TemporaryDirectory() as tmp:
            api=Fake([(13,b'old')])
            with self.assertRaises(ClipboardRestoreError):
                with picture_clipboard(self.image(tmp),_api=api):api.fail=True

    def test_restore_open_failure_is_explicit(self):
        from unittest.mock import patch
        with TemporaryDirectory() as tmp:
            api=Fake([(13,b'old')]); original_open=api.open
            def blocked(hwnd):
                if api.empties: raise OSError('busy')
                original_open(hwnd)
            api.open=blocked
            with patch('ui_qt.quick.powerpoint_clipboard.time.sleep'):
                with self.assertRaises(ClipboardRestoreError):
                    with picture_clipboard(self.image(tmp),_api=api): pass

    def test_failed_initial_put_and_failed_rollback_are_explicit(self):
        with TemporaryDirectory() as tmp:
            api=Fake([(13,b'old')]); api.fail=True
            with self.assertRaises(ClipboardRestoreError):
                with picture_clipboard(self.image(tmp),_api=api): pass

    def test_restore_readback_mismatch_raises(self):
        with TemporaryDirectory() as tmp:
            api=Fake([(13,b'old')]); oldread=api.read
            def read(fmt,max_bytes=64*1024*1024):
                return b'bad' if api.empties == 2 else oldread(fmt,max_bytes)
            api.read=read
            with self.assertRaises(ClipboardRestoreError):
                with picture_clipboard(self.image(tmp),_api=api): pass

    def test_copy_before_post_close_capture_is_preserved(self):
        with TemporaryDirectory() as tmp:
            api=Fake([(13,b'old')])
            api.after_close=lambda:api.copy([(13,b'new')])
            with picture_clipboard(self.image(tmp),_api=api) as info: pass
            self.assertEqual(api.data,{13:b'new'})
            self.assertFalse(info['owner_after_close_is_ours'])
            self.assertTrue(info['clipboard_changed'])
