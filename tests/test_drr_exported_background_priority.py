"""An exact exported group restores the background that produced its result."""
import json
import unittest
from unittest.mock import patch

from core.drr_auto_external import resolve_auto_external
from core.drr_sources import portable_source_name
from tests import test_drr_auto_external as fixtures


class ExportedBackgroundPriorityTests(unittest.TestCase):
    setUp = fixtures.AppliedPnExternalTests.setUp

    def exported(self, background, which='last'):
        history = self.root / 'Processed Data' / 'DRR'
        history.mkdir(parents=True, exist_ok=True)
        path = history / 'saved_result.metadata.json'
        path.write_text(json.dumps({
            'operation': 'DR/R',
            'sources': [
                *({'role': 'measurement', 'source_path': str(self.root / name)}
                  for name in self.selected),
                {'role': 'background', 'source_path': str(self.root / background)},
            ],
            'processing': {'baseline_selection': 'External', 'baseline_which': which},
        }), encoding='utf-8')
        return path

    def resolve(self):
        with patch('core.drr_auto_external.DrrPnStore.find_applied',
                   side_effect=AssertionError('An export must precede the applied-version store.')):
            return resolve_auto_external(self.root, (), self.selected)

    def test_exported_files_and_frame_choice_precede_another_applied_pn_version(self):
        self.exported('Nback.csv', 'first')
        result = self.resolve()
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual(result.assignments[0].baseline_files, ('Nback.csv',))
        self.assertEqual(result.assignments[0].baseline_which, 'first')

    def test_exported_frozen_composite_remains_usable_when_original_inputs_change(self):
        name = portable_source_name(self.root, self.version.csv_path)
        self.exported(name)
        saved_bytes = self.version.csv_path.read_bytes()
        source = self.root / 'Pback.csv'
        source.write_text(source.read_text().replace('10', '11'))
        self.assertFalse(self.store.get_version(self.version.version_id).fresh)
        result = self.resolve()
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual(result.assignments[0].baseline_files, (name,))
        self.assertEqual(self.version.csv_path.read_bytes(), saved_bytes)

    def test_missing_exported_background_does_not_fall_back_to_applied_version(self):
        self.exported('missing_background.csv')
        result = self.resolve()
        self.assertFalse(result.resolved)
        self.assertEqual(result.unresolved_measurements, self.selected)


if __name__ == '__main__':
    unittest.main()
