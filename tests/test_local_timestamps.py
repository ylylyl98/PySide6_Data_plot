from datetime import datetime, timezone
import unittest

from ui_qt.time_format import format_local_timestamp


class LocalTimestampTests(unittest.TestCase):
    def test_saved_utc_uses_system_local_timezone(self):
        stamp = '2026-09-14T17:02:00+00:00'
        expected = datetime.fromtimestamp(
            datetime(2026, 9, 14, 17, 2, tzinfo=timezone.utc).timestamp()
        ).strftime('%Y-%m-%d %H:%M')
        self.assertEqual(format_local_timestamp(stamp), expected)
        self.assertEqual(format_local_timestamp(stamp.replace('+00:00', 'Z')), expected)

    def test_explicit_offset_represents_the_same_instant(self):
        self.assertEqual(format_local_timestamp('2026-09-14T13:02:00-04:00'),
                         format_local_timestamp('2026-09-14T17:02:00+00:00'))

    def test_legacy_naive_timestamp_is_not_shifted(self):
        self.assertEqual(format_local_timestamp('2026-09-14T13:02:30'),
                         '2026-09-14 13:02')

    def test_missing_and_unrecognized_timestamps_remain_readable(self):
        for value in ('', 'unknown', '2026-09-14'):
            self.assertEqual(format_local_timestamp(value), value)
