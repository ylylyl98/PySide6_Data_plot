import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ui_qt.quick.powerpoint_temp import temporary_directory


class PowerPointTemporaryDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.TemporaryDirectory()
        self.addCleanup(self.base.cleanup)
        self.root = Path(self.base.name)

    def test_environment_override_and_cleanup(self):
        with patch.dict(os.environ, {'DPTK_PPT_TEMP_DIR': str(self.root)}):
            with temporary_directory() as path:
                self.assertEqual(Path(path).parent, self.root)
                self.assertTrue(Path(path).is_dir())
            self.assertFalse(Path(path).exists())

    def test_local_config_and_override_precedence(self):
        settings = self.root / 'DPTK' / 'PowerPoint' / 'settings.json'
        settings.parent.mkdir(parents=True)
        configured = self.root / 'configured'
        configured.mkdir()
        settings.write_text(json.dumps({'temp_directory': str(configured)}), encoding='utf-8')
        with patch.dict(os.environ, {'LOCALAPPDATA': str(self.root)}, clear=True):
            with temporary_directory() as path:
                self.assertEqual(Path(path).parent, configured)
            with patch.dict(os.environ, {'DPTK_PPT_TEMP_DIR': str(self.root)}):
                with temporary_directory() as path:
                    self.assertEqual(Path(path).parent, self.root)

    def test_default_uses_standard_temporary_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            with temporary_directory() as path:
                self.assertEqual(Path(path).parent, Path(tempfile.gettempdir()))

    def test_invalid_configured_directory_does_not_fallback(self):
        with patch.dict(os.environ, {'DPTK_PPT_TEMP_DIR': str(self.root / 'missing')}):
            with self.assertRaisesRegex(RuntimeError, 'configured PowerPoint temporary directory is unavailable'):
                temporary_directory()

    def test_invalid_local_config_does_not_fallback(self):
        settings = self.root / 'DPTK' / 'PowerPoint' / 'settings.json'
        settings.parent.mkdir(parents=True)
        settings.write_text('{bad', encoding='utf-8')
        with patch.dict(os.environ, {'LOCALAPPDATA': str(self.root)}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'Could not read the PowerPoint temporary-directory setting'):
                temporary_directory()
