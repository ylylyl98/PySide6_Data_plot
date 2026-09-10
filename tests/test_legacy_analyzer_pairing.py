import tempfile
import unittest
from pathlib import Path

from core.data_io import get_power_series_sources
from core.processing import parse_compare_rotation_angles, group_compare_sources
from core.power_workflow import group_power_measurement_sources


class LegacyAnalyzerPairingTests(unittest.TestCase):
    def test_shared_parser_and_compare_group(self):
        names = [f'yz212_p7_deg{angle}_doping=-1_690nm_002.csv' for angle in (24, 69)]
        self.assertEqual(parse_compare_rotation_angles(names[0]).rot2, 24)
        self.assertEqual(parse_compare_rotation_angles('sample_deg-24p5.csv').rot2, -24.5)
        self.assertIsNone(parse_compare_rotation_angles('sample_nodeg24.csv').rot2)
        self.assertEqual(parse_compare_rotation_angles('sample_deg24_Rot269deg.csv').rot2, 69)
        groups = group_compare_sources(names, in_k_angle=0, out_k_angle=24, out_kp_angle=69)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].mapping, dict(zip(('KK', 'KKp'), names)))

    def test_power_auto_pair_preserves_runs_and_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            names = [f'yz212_p7_deg{angle}_doping=-1_690nm_002.csv' for angle in (24, 69)]
            names += ['yz212_p7_deg24_doping=-1_690nm_003.csv',
                      'yz212_p7_deg69_doping=-2_690nm_002.csv']
            for i, name in enumerate(names):
                Path(folder, name).write_text(f'Power_uW,1.4,1.5\n{1+i/10},2,3\n{2+i/10},4,5\n')
            sources = get_power_series_sources(folder, names)
            for refs in ({'in_k': 0, 'out_k': 24, 'out_kp': 69},
                         {'in_k': 0, 'out_k': 0, 'out_kp': 45}):
                groups = group_power_measurement_sources(folder, sources, angle_refs=refs)
                self.assertEqual(len(groups), 3)
                pair = next(g for g in groups if len(g.sources) == 2)
                self.assertEqual(pair.mapping, {'KK': 'csv::'+names[0], 'KKp': 'csv::'+names[1]})
                self.assertFalse(pair.duplicates)

