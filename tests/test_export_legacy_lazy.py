import subprocess
import sys
import unittest


class ExportLegacyLazyTests(unittest.TestCase):
    def test_lazy_alias_compatibility_in_fresh_process(self):
        code = r"""
import sys
from core import export_legacy
assert 'core.shg' not in sys.modules
assert 'core.shg_fit' not in sys.modules
assert callable(export_legacy.export_pl_pngs_and_dat)
assert 'core.shg' not in sys.modules
assert 'core.shg_fit' not in sys.modules
from core.export_legacy import (ShgProcessResult, ShgSettings, ShgSweepData,
                                ShgAngularFitResult, ShgTwistFitResult)
from core import shg, shg_fit
assert export_legacy.ShgProcessResult is ShgProcessResult is shg.ShgProcessResult
assert export_legacy.ShgSettings is ShgSettings is shg.ShgSettings
assert export_legacy.ShgSweepData is ShgSweepData is shg.ShgSweepData
assert export_legacy.ShgAngularFitResult is ShgAngularFitResult is shg_fit.ShgAngularFitResult
assert export_legacy.ShgTwistFitResult is ShgTwistFitResult is shg_fit.ShgTwistFitResult
try:
    export_legacy.not_an_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown attribute did not raise')
"""
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_star_import_exposes_legacy_shg_aliases(self):
        code = r"""
import sys
import core.export_legacy
assert 'core.shg' not in sys.modules
assert 'core.shg_fit' not in sys.modules
from core.export_legacy import *
from core import shg, shg_fit
assert ShgProcessResult is shg.ShgProcessResult
assert ShgSettings is shg.ShgSettings
assert ShgSweepData is shg.ShgSweepData
assert ShgAngularFitResult is shg_fit.ShgAngularFitResult
assert ShgTwistFitResult is shg_fit.ShgTwistFitResult
"""
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
