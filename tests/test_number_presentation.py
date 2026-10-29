"""Scientific display text never becomes a replacement for the model value."""
import unittest

from PySide6.QtCore import QLocale

from ui_qt.quick.number_presentation import display_text


class NumberPresentationTests(unittest.TestCase):
    def test_semantic_precision_and_no_padding(self):
        locale = QLocale.c()
        for key, value, expected in (
            ("axis.xmin", 1.23456789, "1.2346"),
            ("boundary.x0", 1.750000000001, "1.75"),
            ("axis.ymax", 12.3456789, "12.346"),
            ("raw.vmin", -.123456789, "-0.12346"),
            ("yaxis.a", 1.0000000001, "1"),
            ("log.vmin", 1.234567e-10, "1.2346e-10"),
            ("axis.xmin", -0., "0"),
        ):
            with self.subTest(key=key, value=value):
                self.assertEqual(display_text(value, key, locale), expected)

    def test_localized_display_remains_parseable(self):
        locale = QLocale(QLocale.German)
        self.assertEqual(display_text(1.23456789, "axis.xmin", locale), "1,2346")
        for value in (1e-10, -2.5e7, 1234.56789):
            text = display_text(value, "raw.vmin", locale)
            parsed, valid = locale.toDouble(text)
            self.assertTrue(valid, text)
            self.assertAlmostEqual(parsed / value, 1, places=4)

    def test_locale_numeric_parser_supports_scientific_input(self):
        for locale in (QLocale.c(), QLocale(QLocale.German)):
            for text, expected in (("1e3", 1000), ("1e-6", .000001), ("-25e3", -25000)):
                self.assertEqual(locale.toDouble(text), (expected, True))


if __name__ == "__main__":
    unittest.main()
