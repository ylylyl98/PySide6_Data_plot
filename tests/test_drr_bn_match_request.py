"""The BN action is an explicit, measured-reference-only request."""
import tempfile
from pathlib import Path
from threading import Event
import unittest
from unittest.mock import patch

from core.drr_auto_external import resolve_auto_external
from core.drr_sources import discover_drr_sources, preview_drr_sources
from tests.test_drr_auto_external import write_timed_bn_group


class MeasuredBnRequestTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.selected, self.backgrounds = write_timed_bn_group(self.root)

    def resolve(self, catalog=None, **kwargs):
        return resolve_auto_external(
            self.root, discover_drr_sources(self.root) if catalog is None else catalog,
            self.selected, measured_bn_only=True, **kwargs)

    def test_explicit_bn_request_skips_stores_history_and_generic_candidates(self):
        catalog = discover_drr_sources(self.root)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name in ('DrrBnStore', 'DrrPnStore', '_read_drr_metadata',
                         'rank_baseline_candidates', 'discover_drr_sources'):
                stack.enter_context(patch('core.drr_auto_external.' + name,
                                          side_effect=AssertionError(name + ' must not be used')))
            result = self.resolve(catalog)
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual([a.measurement_file for a in result.assignments], self.selected)
        self.assertEqual([a.baseline_files for a in result.assignments],
                         [(self.backgrounds[0],)] * 9 + [(self.backgrounds[1],)] * 6)
        self.assertTrue(all(a.baseline_which == 'all' for a in result.assignments))

    def test_missing_bn_never_falls_back_to_p_or_n_or_arbitrary_spectrum(self):
        for name in self.backgrounds:
            (self.root / name).rename(self.root / name.replace('BNbackground', 'Pbackground'))
        result = self.resolve()
        self.assertFalse(result.resolved)
        self.assertEqual(result.unresolved_measurements, tuple(self.selected))
        self.assertIn('No matching measured BN', result.reason)

    def test_generated_references_cannot_be_renamed_into_measured_bn(self):
        from core.drr_bn_store import BN_MARKER
        from core.drr_composite_background import COMPOSITE_MARKER
        catalog = discover_drr_sources(self.root)
        for name, marker in zip(self.backgrounds, (BN_MARKER, COMPOSITE_MARKER)):
            path = self.root / name
            path.write_text(marker + ',' + path.read_text())
        result = self.resolve(catalog)
        self.assertFalse(result.resolved)
        self.assertIn('No matching measured BN', result.reason)

    def test_preview_catalog_completes_only_measurement_and_bn_profiles(self):
        irrelevant = self.root / 'unrelated.csv'
        irrelevant.write_text('Not a spectral CSV\n')
        catalog = preview_drr_sources(self.root, include_all=True)
        from core.drr_sources import inspect_csv_gate_profile
        reads = []
        def inspect(path, **kwargs):
            reads.append(Path(path).name)
            return inspect_csv_gate_profile(path, **kwargs)
        with patch('core.drr_auto_external.inspect_csv_gate_profile', side_effect=inspect):
            result = self.resolve(catalog)
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual(set(reads), set(self.selected) | set(self.backgrounds))

    def test_cancelled_request_has_no_assignments(self):
        cancel = Event()
        cancel.set()
        result = self.resolve([], cancelled=cancel)
        self.assertFalse(result.resolved)
        self.assertFalse(result.assignments)


if __name__ == '__main__':
    unittest.main()
