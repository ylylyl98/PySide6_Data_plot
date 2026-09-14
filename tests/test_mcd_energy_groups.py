from types import SimpleNamespace
import unittest
from core.mcd_energy_groups import initial_energy_groups, safe_energy_edges


def record(name, energy, field):
    return SimpleNamespace(record_id=name, center_ev=energy, width_mev=5,
                           condition_value=lambda key: field if key == 'E-field' else None)


class EnergyGroupTests(unittest.TestCase):
    def test_groups_do_not_chain_across_a_wide_energy_range(self):
        rows = [record('a', 1.600, 0), record('b', 1.604, 1), record('c', 1.608, 2)]
        groups = initial_energy_groups(rows, 5)
        self.assertEqual(groups['a'], groups['b'])
        self.assertNotEqual(groups['a'], groups['c'])

    def test_duplicate_field_and_large_energy_jump_are_not_connected(self):
        rows = [record('a',1.6,0), record('b',1.601,1), record('c',1.602,1),
                record('d',1.603,2),record('e',1.63,3)]
        self.assertEqual(safe_energy_edges(rows,5), [])
        self.assertEqual(safe_energy_edges([rows[0],rows[1]],5), [(0,1)])

    def test_same_group_sparse_fields_connect_but_other_doping_does_not(self):
        rows = [record('a',1.6,0),record('b',1.601,1),record('c',1.602,100)]
        self.assertEqual(safe_energy_edges(rows,5),[(0,1),(1,2)])
        rows[0].condition_value = lambda k: {'E-field':0,'Doping':6.3}.get(k)
        rows[1].condition_value = lambda k: {'E-field':1,'Doping':11.4}.get(k)
        self.assertEqual(safe_energy_edges(rows[:2],5),[])

    def test_group_shades_follow_field_and_keep_same_field_color(self):
        from core.mcd_energy_groups import energy_record_colors
        rows = [record('a',1.6,0),record('b',1.6,10),record('c',1.6,10)]
        colors = energy_record_colors(rows,dict.fromkeys(['a','b','c'],'G'),{'G':'#336699'})
        self.assertNotEqual(colors['a'],colors['b'])
        self.assertEqual(colors['b'],colors['c'])
        self.assertGreater(sum(colors['a'][:3]),sum(colors['b'][:3]))

    def test_manual_group_allows_sparse_steps_but_not_ambiguous_windows(self):
        rows = [record('a',1.6,0),record('b',1.601,1),record('c',1.602,100)]
        self.assertEqual(safe_energy_edges(rows,5,('a','b','c')),[(0,1),(1,2)])
        rows.append(record('d',1.602,100))
        self.assertEqual(safe_energy_edges(rows,5,('a','b','c','d')),[(0,1)])

    def test_compact_group_numbers_preserve_membership_and_numeric_order(self):
        from core.mcd_energy_groups import compact_group_numbers
        original = {'a':'Group 1','b':'Group 2','c':'Group 4','d':'Group 4','e':'Custom'}
        self.assertEqual(compact_group_numbers(original), {'a':'Group 1','b':'Group 2','c':'Group 3','d':'Group 3','e':'Custom'})
        self.assertEqual(compact_group_numbers({'a':'Group 10','b':'Group 2'}),{'a':'Group 2','b':'Group 1'})

    def test_orange_gradient_stays_in_rgba_range_after_merge(self):
        from core.mcd_energy_groups import energy_record_colors
        from matplotlib.colors import to_hex
        rows=[record(str(f),1.58,f) for f in range(10,31)]
        colors=energy_record_colors(rows,{r.record_id:'Group 2' for r in rows},{'Group 2':'#ff7f0e'})
        for color in colors.values():
            self.assertTrue(all(0 <= value <= 1 for value in color),color)
            to_hex(color)

    def test_field_rank_spreads_dense_values_evenly(self):
        from core.mcd_energy_groups import energy_record_colors
        rows=[record(str(f),1.6,f) for f in (0,18,19,20)]
        colors=energy_record_colors(rows,{r.record_id:'G' for r in rows},{'G':'#2ca02c'})
        import numpy as np
        distances=[np.linalg.norm(np.array(colors[str(a)])-np.array(colors[str(b)])) for a,b in ((0,18),(18,19),(19,20))]
        self.assertLess(max(distances)/min(distances),2.5)

    def test_legend_omits_excluded_groups_without_recoloring(self):
        from core.mcd_energy_groups import draw_energy_slope_panels,energy_group_colors
        from matplotlib.figure import Figure
        rows=[record('a',1.6,0),record('b',1.62,10)]
        groups={'a':'Group 1','b':'Group 2','excluded':'Group 5'}
        fig=Figure()
        artists=draw_energy_slope_panels(fig,rows[1:],groups,(),('B increasing',))
        self.assertEqual([t.get_text() for t in fig.axes[0].get_legend().get_texts()],['Group 2','Inc'])
        self.assertEqual(len(artists),1)

    def test_visible_numbering_keeps_excluded_groups_distinct(self):
        from core.mcd_energy_groups import visible_group_numbers
        groups={str(i):f'Group {i}' for i in range(1,6)}
        result=visible_group_numbers(groups,{'2','3','4'})
        self.assertEqual([result[k] for k in ('2','3','4')],['Group 1','Group 2','Group 3'])
        self.assertEqual(len(set(result.values())),5)
        self.assertEqual(visible_group_numbers(result,{'2','3','4'}),result)
