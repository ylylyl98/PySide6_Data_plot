import unittest

import numpy as np

from core.peak_auto_selection import _continuity_reference


class ContinuityTrajectoryTests(unittest.TestCase):
    @staticmethod
    def point(row, energy, polarity='peak', prefix='track'):
        return dict(row_index=row, energy=energy, polarity=polarity,
                    candidate_id=f'{prefix}:{row}')

    def test_curved_track_and_single_missing_rows(self):
        points = [self.point(i, 1 + .0003 * i + .000015 * i * i)
                  for i in range(24) if i not in (7, 16)]
        ids = _continuity_reference(points, np.arange(24), .2)
        self.assertGreaterEqual(len(ids), 19)

    def test_dense_random_candidates_do_not_form_a_track(self):
        rng = np.random.default_rng(123)
        points = [self.point(i, 1 + .1 * rng.random(), prefix=f'noise{j}')
                  for i in range(40) for j in range(20)]
        ids = _continuity_reference(points, np.arange(40), .2)
        self.assertLess(len(ids), len(points) * .05)

    def test_gap_and_polarity_break_paths(self):
        points = [self.point(i, 1 + .0002 * i, 'peak' if i % 2 else 'dip')
                  for i in range(10)]
        self.assertEqual(_continuity_reference(points, np.arange(10), .2), set())
        gates = np.array([0, 1, 2, 20, 21, 22])
        points = [self.point(i, 1 + .0002 * i) for i in range(6)]
        self.assertEqual(_continuity_reference(points, gates, .2), set())

    def test_reverse_input_order_and_irregular_scan_spacing(self):
        gates = np.cumsum([0.] + [1., 1.1, .9, 1.] * 5)
        points = [self.point(i, 1 + .0004 * gates[i] + .000008 * gates[i] ** 2)
                  for i in range(len(gates))]
        expected = _continuity_reference(points, gates, .2)
        reversed_points = [dict(p, row_index=len(gates) - 1 - p['row_index'])
                           for p in reversed(points)]
        actual = _continuity_reference(reversed_points, gates[::-1], .2)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(points))

    def test_crossing_tracks_and_five_row_scan(self):
        points = [self.point(i, 1 + sign * .0005 * (i - 10), prefix=str(sign))
                  for i in range(21) for sign in (-1, 1)]
        ids = _continuity_reference(points, np.arange(21), .2)
        for sign in (-1, 1):
            self.assertGreaterEqual(sum(key.startswith(str(sign) + ':') for key in ids), 18)
        five = [self.point(i, 1 + i * .0003) for i in range(5)]
        self.assertEqual(len(_continuity_reference(five, np.arange(5), .2)), 5)

    def test_early_gaps_do_not_poison_later_long_track(self):
        points = [self.point(i, 1 + .0001 * i) for i in range(150)
                  if not (i < 75 and i % 3 == 1)]
        ids = _continuity_reference(points, np.arange(150), .2)
        self.assertTrue(all(f'track:{i}' in ids for i in range(120, 150)))

    def test_local_guard_requires_distinct_rows_same_polarity_and_scan_block(self):
        from core.peak_trajectories import local_cluster_reference
        gates = np.array([0., 1., 2., 20., 21., 22.])
        points = [self.point(1, 1., prefix='a'), self.point(2, 1., prefix='b'),
                  self.point(3, 1., prefix='c')]
        self.assertFalse(local_cluster_reference(points, gates, .2))
        points = [self.point(i, 1., 'peak' if i != 1 else 'dip') for i in range(3)]
        self.assertFalse(local_cluster_reference(points, gates, .2))
        points = [self.point(0, 1., prefix=f'a{i}') for i in range(4)]
        self.assertFalse(local_cluster_reference(points, gates, .2))
        points = [self.point(i, 1.) for i in range(3)]
        self.assertEqual(len(local_cluster_reference(points, gates, .2)), 3)

    def test_cancellation(self):
        points = [self.point(i, 1) for i in range(10)]
        with self.assertRaises(InterruptedError):
            _continuity_reference(points, np.arange(10), .2, lambda: True)


if __name__ == '__main__':
    unittest.main()
