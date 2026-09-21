import csv
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

from core.mcd_slope_refit import refit_record_slopes
from tests.test_curie_weiss_ui import records


def write_curved_trace(record):
    # Inner points: y=2B+1; outer points bend. Return branch: y=-3B+4.
    fields = [-.3, -.2, -.1, 0., .1, .2, .3]
    inc = [-.2, .6, .8, 1., 1.2, 1.4, 2.2]
    with record.trace_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        columns = []
        for suffix in ('increasing', 'decreasing'):
            columns.extend([f'B_{suffix}_T', f'corrected_signed_mean_{suffix}',
                            f'corrected_field_signed_absolute_mean_{suffix}', f'corrected_integral_{suffix}'])
        writer.writerow(columns)
        for x, y, xd in zip(fields, inc, fields[::-1]):
            yd = -3*xd+4
            writer.writerow([x, y, y, y, xd, yd, yd, yd])


class SlopeRefitTests(unittest.TestCase):
    def test_bounds_change_slope_without_merging_return_branch(self):
        with tempfile.TemporaryDirectory() as folder:
            record = records(Path(folder))[0]
            write_curved_trace(record)
            narrow = {f['branch']:f for f in refit_record_slopes(record, -.2, .2)}
            wide = {f['branch']:f for f in refit_record_slopes(record, -.3, .3)}
            self.assertAlmostEqual(narrow['B increasing']['slope'], 2.)
            self.assertAlmostEqual(wide['B increasing']['slope'], 23/7)
            self.assertAlmostEqual(narrow['B decreasing']['slope'], -3.)
            self.assertAlmostEqual(narrow['B increasing']['intercept'], 1.)
            self.assertEqual(narrow['B increasing']['n'], 5)
            self.assertEqual(sum(p['used'] for p in narrow['B increasing']['samples']), 5)
            self.assertGreater(wide['B increasing']['slope_se'], 0.)

    def test_sparse_invalid_and_missing_curve_never_reuse_saved_slope(self):
        with tempfile.TemporaryDirectory() as folder:
            record = records(Path(folder))[0]
            write_curved_trace(record)
            fitted = refit_record_slopes(record, -.05, .05)
            self.assertTrue(all(f['slope'] is None for f in fitted))
            for low, high in [(0, 0), (.2, -.2), (float('nan'), .2)]:
                with self.assertRaises(ValueError):
                    refit_record_slopes(record, low, high)
            with self.assertRaises(OSError):
                refit_record_slopes(replace(record, trace_path=Path(folder)/'missing.csv'), -.2, .2)


if __name__ == '__main__':
    unittest.main()
