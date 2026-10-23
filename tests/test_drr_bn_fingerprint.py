"""Source and simulated PYZ fingerprints remain strict across installations."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.drr_bn_fingerprint import (module_fingerprint, python_runtime_identity,
                                     semantic_code_digest)
from core.drr_bn_service import _algorithm_identity


class _FrozenLoader:
    def __init__(self, source, filename):
        self.code = compile(source, filename, 'exec')
        self.requested = []

    def get_code(self, name):
        self.requested.append(name)
        return self.code


def frozen_module(source, directory):
    filename = str(Path(directory) / 'core' / 'frozen_numerics.py')
    loader = _FrozenLoader(source, filename)
    return SimpleNamespace(__file__=filename, __spec__=SimpleNamespace(loader=loader))


class PipelineFingerprintTests(unittest.TestCase):
    def test_development_identity_hashes_real_source_bytes_and_all_modules(self):
        first = _algorithm_identity()
        self.assertEqual(first, _algorithm_identity())
        self.assertEqual(first['protocol'], 'd2_shrinkage_v1')
        expected = {'core.drr_bn_service', 'core.drr_bn_calibration', 'core.drr_bn_joint', 'core.drr_bn_d2_objective',
                    'core.processing', 'core.processing_run', 'core.loader',
                    'core.drr_segmented_derivative', 'core.drr_sources',
                    'core.drr_bn_store', 'core.drr_bn_fingerprint'}
        self.assertEqual(set(first['module_fingerprints']), expected)
        self.assertEqual(first['python_runtime'], python_runtime_identity())
        root = Path(__file__).resolve().parents[1]
        for name, item in first['module_fingerprints'].items():
            with self.subTest(module=name):
                source = root.joinpath(*name.split('.')).with_suffix('.py')
                self.assertEqual(item['kind'], 'source_bytes')
                self.assertEqual(item['sha256'], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_source_bytes_include_comment_changes_even_with_equal_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'numerics.py'
            path.write_text('FACTOR = 2\n# first\n', encoding='utf-8')
            module = SimpleNamespace(__file__=str(path), __spec__=SimpleNamespace(loader=None))
            before = module_fingerprint('numerics', module=module)
            path.write_text('FACTOR = 2\n# other\n', encoding='utf-8')
            after = module_fingerprint('numerics', module=module)
        self.assertEqual(before['kind'], 'source_bytes')
        self.assertNotEqual(before['sha256'], after['sha256'])

    def test_frozen_loader_code_has_no_dependency_on_extraction_path(self):
        source = 'FACTOR = 3\ndef nested(x):\n    return x + FACTOR\n'
        with tempfile.TemporaryDirectory() as directory:
            one = frozen_module(source, Path(directory) / '_MEI_first')
            two = frozen_module(source, Path(directory) / '_MEI_second')
            first = module_fingerprint('core.frozen_numerics', module=one)
            self.assertEqual(first, module_fingerprint('core.frozen_numerics', module=one))
            self.assertEqual(first, module_fingerprint('core.frozen_numerics', module=two))
        self.assertEqual(first['kind'], 'loader_code')
        self.assertEqual(one.__spec__.loader.requested, ['core.frozen_numerics']*2)

    def test_every_module_uses_loader_fallback_in_simulated_pyz_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            def load_from(extraction):
                def load(name):
                    return frozen_module(f'MODULE_NAME = {name!r}\nFACTOR = 3\n', extraction)
                return load
            with patch('core.drr_bn_fingerprint.importlib.import_module',
                       side_effect=load_from(Path(directory) / '_MEI_one')):
                first = _algorithm_identity()
            with patch('core.drr_bn_fingerprint.importlib.import_module',
                       side_effect=load_from(Path(directory) / '_MEI_two')):
                second = _algorithm_identity()
        self.assertEqual(first, second)
        self.assertEqual(len(first['module_fingerprints']), 11)
        self.assertTrue(all(item['kind'] == 'loader_code'
                            for item in first['module_fingerprints'].values()))

    def test_changed_module_and_nested_function_constants_change_digest(self):
        for original, changed in (
            ('FACTOR = 2\n', 'FACTOR = 3\n'),
            ('def f(x):\n    return x + 2\n', 'def f(x):\n    return x * 2\n'),
            ('def outer():\n    def inner():\n        return 2\n    return inner\n',
             'def outer():\n    def inner():\n        return 3\n    return inner\n'),
        ):
            with self.subTest(original=original):
                self.assertNotEqual(semantic_code_digest(compile(original, 'one.py', 'exec')),
                                    semantic_code_digest(compile(changed, 'two.py', 'exec')))

    def test_nested_filenames_and_line_positions_do_not_change_semantic_digest(self):
        source = 'def outer(x):\n    def inner(y):\n        return y*2\n    return inner(x)\n'
        original = compile(source, 'C:/old/_MEI_one/core/recipe.py', 'exec')
        relocated = compile('\n\n'+source, 'D:/new/_MEI_two/core/recipe.py', 'exec')
        self.assertEqual(semantic_code_digest(original), semantic_code_digest(relocated))

    def test_frozenset_constants_are_stable_across_python_hash_seeds(self):
        script = ('from core.drr_bn_fingerprint import semantic_code_digest; '
                  'print(semantic_code_digest(compile("def f(x):\\n    return x in '
                  '{\\\"alpha\\\", \\\"beta\\\", \\\"gamma\\\"}\\n", "different.py", "exec")))')
        root = Path(__file__).resolve().parents[1]
        values = [subprocess.check_output([sys.executable, '-X', 'utf8', '-c', script],
                    cwd=root, env={**os.environ, 'PYTHONHASHSEED': seed}, text=True).strip()
                  for seed in ('1', '2')]
        self.assertEqual(values[0], values[1])

    def test_missing_source_and_loader_fail_instead_of_skipping_module(self):
        module = SimpleNamespace(__file__='missing/frozen_module.py', __spec__=SimpleNamespace(loader=None))
        with self.assertRaisesRegex(ValueError, 'source and loader code are unavailable'):
            module_fingerprint('core.missing', module=module)
        module.__spec__.loader = SimpleNamespace(get_code=lambda name: None)
        with self.assertRaisesRegex(ValueError, 'did not provide a module code object'):
            module_fingerprint('core.missing', module=module)

    def test_runtime_version_is_recorded_for_bytecode_compatibility(self):
        identity = python_runtime_identity()
        self.assertEqual(identity['implementation'], sys.implementation.name)
        self.assertEqual(identity['cache_tag'], sys.implementation.cache_tag)
        self.assertEqual(identity['version'], list(sys.version_info))


if __name__ == '__main__':
    unittest.main()
