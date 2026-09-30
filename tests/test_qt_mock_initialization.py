"""Keep native Qt initialization regressions observable as test failures."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class QtMockInitializationTests(unittest.TestCase):
    def test_mocked_window_fixtures_initialize_in_a_fresh_process(self):
        # Plain MagicMock class attributes can be mistaken for Qt slot metadata
        # by PySide6 6.11. Exercise the real fixtures without an existing cached
        # metaobject, and contain any native crash within the child process.
        # Run the deferred-load fixture first to also catch QCoreApplication
        # creation that prevents subsequent QWidget fixtures from initializing.
        script = """
import os
import unittest
if os.name == 'nt':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'tests.test_power_prevalidated_load.PowerPrevalidatedLoadTests',
    'tests.test_colormaps.ColormapUiTests.test_all_selectors_have_required_order_and_legacy_defaults',
    'tests.test_astra_workflow_review.AstraPeakShiftEventTests.test_close_cancels_worker_before_waiting_for_completion',
    'tests.test_drr_analysis_window.WorkspaceWindowTests.test_workspace_restores_p2p_results_and_fixed_view_without_recalculation',
    'tests.test_workflow_latency.PeakShiftDispatchTests.test_default_analysis_dispatches_without_waiting_for_worker',
    'tests.test_drr_comparison_ui.ComparisonUiTests.test_unchanged_sources_reuse_group_cache_and_same_group_is_noop',
])
assert suite.countTestCases() == 7
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
"""
        environment = dict(os.environ, QT_QPA_PLATFORM='offscreen')
        result = subprocess.run(
            [sys.executable, '-X', 'faulthandler', '-c', script],
            cwd=Path(__file__).resolve().parents[1], env=environment,
            capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
