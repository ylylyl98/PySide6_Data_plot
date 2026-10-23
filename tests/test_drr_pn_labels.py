from types import SimpleNamespace
import unittest

from ui_qt.drr_pn_labels import source_frame_summary


class PnSourceFrameLabelTests(unittest.TestCase):
    def test_recorded_frame_choices_are_readable(self):
        for p_which, n_which, expected in (
            ("all", "all", "P: all frames averaged; N: all frames averaged"),
            ("first", "last", "P: first frame; N: last frame"),
            ("last", "first", "P: last frame; N: first frame"),
            ("first", "all", "P: first frame; N: all frames averaged"),
        ):
            with self.subTest(p_which=p_which, n_which=n_which):
                version = SimpleNamespace(p_which=p_which, n_which=n_which)
                self.assertEqual(source_frame_summary(version), expected)

    def test_unknown_or_missing_saved_choices_do_not_imply_all_frames(self):
        for version in (None, SimpleNamespace(), SimpleNamespace(p_which="all"),
                        SimpleNamespace(p_which="all", n_which="unknown"),
                        SimpleNamespace(p_which="unknown", n_which="all")):
            with self.subTest(version=version):
                self.assertEqual(source_frame_summary(version), "P/N source averaging: see saved version")


if __name__ == "__main__":
    unittest.main()
