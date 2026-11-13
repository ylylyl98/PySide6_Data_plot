import unittest
from ui_qt.quick.powerpoint_mouse import DoubleClickDetector

class DoubleClickDetectorTests(unittest.TestCase):
    def test_double_click_and_reset(self):
        detector = DoubleClickDetector(500, 8, 8)
        self.assertFalse(detector.click(100, 40, 50))
        self.assertTrue(detector.click(200, 42, 51))
        self.assertFalse(detector.click(300, 42, 51))

    def test_time_distance_and_clock_wrap(self):
        detector = DoubleClickDetector(500, 8, 8)
        self.assertFalse(detector.click(100, 0, 0))
        self.assertFalse(detector.click(700, 0, 0))
        self.assertFalse(detector.click(710, 20, 0))
        detector.last = (0xfffffff0, 20, 0)
        self.assertTrue(detector.click(20, 20, 0))
