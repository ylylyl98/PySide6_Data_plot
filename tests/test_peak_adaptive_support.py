"""Adaptive support extends only clear, width-compatible row matches."""
import copy
import unittest

import numpy as np

from core.peak_quality import supported_candidates


def samples(width=8.):
    return [dict(candidate_id=str(i), row_index=i, polarity='peak',
                 energy=1 + (.0015 if i % 2 else 0), width_mev=width)
            for i in range(5)]


def ids(points, gates=np.arange(5.), tolerance=1., adaptive=True, cancelled=None):
    return {p['candidate_id'] for p in supported_candidates(
        points, gates, 5, tolerance, cancelled, adaptive=adaptive)}


class AdaptiveSupportTests(unittest.TestCase):
    def test_broad_centroid_jitter_recovers_without_changing_input(self):
        points = samples()
        original = copy.deepcopy(points)
        self.assertNotIn('2', ids(points, adaptive=False))
        self.assertIn('2', ids(points))
        self.assertEqual(points, original)

    def test_narrow_missing_invalid_width_or_zero_tolerance_never_expand(self):
        for width in (1., None, float('nan'), -1.):
            points = samples(width)
            self.assertEqual(ids(points), ids(points, adaptive=False))
        points = samples()
        self.assertEqual(ids(points, tolerance=0), ids(points, tolerance=0, adaptive=False))
        for point in points:
            point.pop('width_mev')
        self.assertEqual(ids(points), ids(points, adaptive=False))

    def test_added_matches_are_mutual_and_unambiguous(self):
        points = samples()
        points.append(dict(candidate_id='rival', row_index=1, polarity='peak',
                           energy=.9985, width_mev=8.))
        self.assertNotIn('2', ids(points))
        points = samples()
        # The target is nearest to a competing candidate in the other row.
        points.append(dict(candidate_id='closer', row_index=2, polarity='peak',
                           energy=1.0014, width_mev=8.))
        self.assertNotIn('2', ids(points))

    def test_broad_peak_never_extends_past_twice_base(self):
        points = [dict(p, energy=1 + (.0021 if p['row_index'] % 2 else 0),
                       width_mev=100.) for p in samples()]
        self.assertEqual(ids(points), ids(points, adaptive=False))
        self.assertNotIn('2', ids(points))

    def test_original_matches_survive_ambiguous_nearest_neighbors(self):
        points = [dict(p, energy=1.) for p in samples()]
        # A competing same-row point makes mutual-nearest evidence ambiguous,
        # but existing base matches must not be removed.
        points += [dict(candidate_id=f'rival:{i}', row_index=i, polarity='peak',
                        energy=1.00005, width_mev=8.) for i in range(5)]
        original = ids(points, adaptive=False)
        self.assertTrue(original)
        self.assertTrue(original <= ids(points))

    def test_polarity_scan_gap_reverse_and_cancel(self):
        points = samples()
        reverse = [dict(p, row_index=4-p['row_index']) for p in points]
        self.assertEqual(ids(points), ids(reverse, gates=np.arange(4, -1, -1.)))
        across_gap = supported_candidates(points, np.array([0., 1., 2., 20., 21.]),
                                          0, 1., adaptive=True)
        self.assertEqual([p['neighbor_available'] for p in across_gap], [3, 3, 3, 2, 2])
        alternating = [dict(p, polarity='dip' if p['row_index'] % 2 else 'peak') for p in points]
        self.assertEqual(ids(alternating), ids(alternating, adaptive=False))
        with self.assertRaisesRegex(RuntimeError, 'cancel'):
            ids(points, cancelled=lambda: True)


if __name__ == '__main__':
    unittest.main()
