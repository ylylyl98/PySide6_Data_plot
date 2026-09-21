import json
import tempfile
import unittest
from pathlib import Path

from core.drr_comparison_groups import parse_condition, build_catalog, ComparisonStore


class ComparisonTests(unittest.TestCase):
    def test_normalization_and_strict_spot_gate(self):
        a=parse_condition('YZ365_p5n2_1T_1.67KREF_720nmc_0p08sx40_Rot49deg_TG−1.087BG=18.csv')
        b=parse_condition('YZ365_2T_p5n2_1.670KREF_720nmc_0p08sx10_Rot49deg_TG-1.087BG=18.0.csv')
        self.assertEqual(a['group_key'],b['group_key'])
        for old,new in [('p5n2','p3n2'),('=18','=19'),('1.087','1.08'),('720','760')]:
            c=parse_condition('YZ365_p5n2_1T_1.67KREF_720nmc_0p08sx40_Rot49deg_TG-1.087BG=18.csv'.replace(old,new))
            self.assertNotEqual(a['group_key'],c['group_key'])
        self.assertIsNone(parse_condition('unknown.csv'))
        c=parse_condition('YZ365_p5n2_2T_1.67KREF_720nmc_0p08sx10_RotIn49deg_TG-1.087BG=18_rep01.csv')
        self.assertEqual(a['group_key'],c['group_key'])
        plus=parse_condition('YZ365_p5n2_1T_1.67KREF_720nmc_Rot49deg_TG+1.087BG=18.csv')
        self.assertEqual(plus['gate'],'TG+1.087BG=18')
        self.assertIsNone(parse_condition('YZ365_p5n2_1T_1.67KREF_720nmc_Rot49deg_TG=1_BG=2.csv'))

    def test_catalog_does_not_merge_uncertain_or_repeats(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=[]
            for i,b in enumerate((0,1,1)):
                p=Path(folder)/f'{i}.metadata.json'
                p.write_text(json.dumps({'sources':[{'role':'measurement','filename':f'YZ365_p5n2_{b}T_1.67KREF_720nmc_Rot49deg_TG-1.087BG=18.csv'}],'processing':{}}))
                paths.append(p)
            groups,entries=build_catalog(paths)
            self.assertEqual(len(groups),1)
            self.assertEqual(len(next(iter(groups.values()))['members']),3)
            self.assertEqual(len(entries),3)

    def test_exclusions_survive_rescan_and_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            store=ComparisonStore(folder)
            store.set_member('g','a',False)
            self.assertEqual(ComparisonStore(folder).members('g',['a','b']),['b'])
            store.set_member('g','c',True)
            self.assertEqual(store.members('g',['a','b']),['b','c'])
            store.reset('g')
            self.assertEqual(store.members('g',['a','b']),['a','b'])
