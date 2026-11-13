from types import SimpleNamespace
from unittest import TestCase, mock
from ui_qt.quick.powerpoint_application import PowerPointApplicationMixin


class Fake(PowerPointApplicationMixin):
    def __init__(self):
        self._ppt_init(); self.messages=[]; self._ppt_service=SimpleNamespace(update_figure=lambda *args: None)
    def _message(self, message): self.messages.append(message)


class TimingTests(TestCase):
    def test_correlates_record_and_preserves_newer_pending_save(self):
        app=Fake(); old=SimpleNamespace(record_id='old'); app._ppt_record=old
        a=SimpleNamespace(record_id='a',png_path='a.png',dat_path='a.dat')
        b=SimpleNamespace(record_id='b',png_path='b.png',dat_path='b.dat')
        app._ppt_finish('ppt-export',(old,a,10.))
        app._ppt_finish('ppt-export',(old,b,20.))
        with mock.patch('ui_qt.quick.powerpoint_application.perf_counter',return_value=25.):
            app._ppt_updated('wrong',a)
            self.assertIn('a',app._ppt_update_timings)
            app._ppt_updated('old',a)
            self.assertIs(app._ppt_save_record,b)
            self.assertTrue(any('15.000 s' in message for message in app.messages))
            app._ppt_updated('old',b)
            self.assertIsNone(app._ppt_save_record)
            self.assertTrue(any('5.000 s' in message for message in app.messages))
            self.assertEqual(app._ppt_update_timings,{})

    def test_synchronous_service_failure_clears_timing(self):
        app=Fake(); old=SimpleNamespace(record_id='old')
        record=SimpleNamespace(record_id='new',png_path='new.png',dat_path='new.dat')
        app._ppt_service.update_figure=mock.Mock(side_effect=RuntimeError('failure'))
        with self.assertRaises(RuntimeError): app._ppt_finish('ppt-export',(old,record,1.))
        self.assertEqual(app._ppt_update_timings,{})
        self.assertIsNone(app._ppt_save_record)
