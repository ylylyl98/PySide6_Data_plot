import unittest
from ui_qt.controllers_drr import _drr_condition_values, format_drr_source_summary


class GateExpressionDisplayTests(unittest.TestCase):
    def test_compound_condition_is_not_split_into_gate_values(self):
        for sign in ('-', '−'):
            filename=f'YZ365_p5n2_1T_1.67KREF_720nmc_TG{sign}1.087BG=18.csv'
            values=_drr_condition_values(filename)
            self.assertEqual(values.get('gate_expression'),'TG−1.087BG=18')
            self.assertNotIn('tg',values)
            self.assertNotIn('bg',values)
            self.assertIn('TG−1.087BG=18',format_drr_source_summary(filename))

    def test_independent_gate_values_remain_independent(self):
        values=_drr_condition_values('sample_TG=-1.087_BG=18.csv')
        self.assertEqual(values['tg'],'-1.087')
        self.assertEqual(values['bg'],'18')

    def test_decimal_coefficients_and_signed_constant(self):
        values=_drr_condition_values('sample_2TG+1p087BG=-18_001.csv')
        self.assertEqual(values.get('gate_expression'),'2TG+1.087BG=-18')
