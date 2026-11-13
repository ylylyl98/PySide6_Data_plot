from types import SimpleNamespace
from unittest import TestCase
from ui_qt.quick.pl_application import PlApplicationMixin
from ui_qt.quick.powerpoint_application import PowerPointApplicationMixin


class Catalog:
    def __init__(self): self._dialog_open=False;self.busy=False;self.prepares=[];self._generation=0;self.requests=[];self._busy=False
    def setForegroundBusy(self, value): self.busy=value
    def prepare(self,*args): self.prepares.append(args)
    def open(self,*args): self._dialog_open=True
    def refreshCurrent(self,*args,**kwargs):
        self._generation += 1;self._busy=True;self.requests.append((args,kwargs,self.busy))

class Signal:
    def emit(self,*args): pass

class Owner(PlApplicationMixin,PowerPointApplicationMixin):
    def __init__(self):
        self._source_catalog=Catalog();self._closed=False;self._busy=False;self._pending={}
        self._catalog_child_busy=False;self._catalog_prepare_scope=None;self._catalog_dialog_seen=False
        self._catalog_explicit_scope=None;self._catalog_explicit_generation=None;self._pl_auto_next_pending=None
        self._workflow='drr';self._folder='experiment';self._sources=[];self._ppt_record=None
        self._ppt_applying=False;self.sourceCatalogRequested=Signal()

class CatalogPriorityTests(TestCase):
    def test_applying_and_active_figure_do_not_prepare(self):
        owner=Owner(); owner._ppt_applying=True;owner._sync_source_catalog()
        self.assertTrue(owner._source_catalog.busy);self.assertEqual(owner._source_catalog.prepares,[])
        owner._ppt_applying=False;owner._ppt_record=object();owner._sync_source_catalog()
        self.assertTrue(owner._source_catalog.busy);self.assertEqual(owner._source_catalog.prepares,[])
    def test_explicit_picker_and_cancel(self):
        owner=Owner();owner._ppt_record=object();owner._sync_source_catalog()
        owner._pl_open_catalog(True)
        self.assertFalse(owner._source_catalog.busy)
        owner._source_catalog._dialog_open=False;owner._pl_catalog_dialog_changed()
        self.assertTrue(owner._source_catalog.busy)
    def test_normal_and_failed_apply_resume(self):
        owner=Owner();owner._sync_source_catalog()
        self.assertEqual(owner._source_catalog.prepares,[('drr','experiment')]);self.assertFalse(owner._source_catalog.busy)
        def fail(kind,result):
            self.assertTrue(owner._source_catalog.busy)
            raise ValueError('bad figure')
        owner._ppt_finish_apply=fail
        with self.assertRaises(ValueError):owner._ppt_finish('ppt-load',None)
        self.assertFalse(owner._ppt_applying);self.assertFalse(owner._source_catalog.busy)

    def test_explicit_refresh_runs_then_pauses_after_success_or_error(self):
        owner=Owner(); owner._ppt_record=object(); owner._sync_source_catalog()
        owner._pl_refresh_catalog_explicit(force=True)
        self.assertFalse(owner._source_catalog.busy)
        self.assertFalse(owner._source_catalog.requests[-1][2])
        self.assertTrue(owner._source_catalog.requests[-1][1]['force'])
        owner._pl_catalog_finished('catalog', .2)
        self.assertTrue(owner._source_catalog.busy)
        self.assertIsNone(owner._catalog_explicit_scope)
        owner._pl_refresh_catalog_explicit()
        owner._pl_catalog_finished('catalog', .2)  # errors emit the same completion signal
        self.assertTrue(owner._source_catalog.busy)

    def test_explicit_refresh_exception_and_cancellation_clear_override(self):
        owner=Owner();owner._ppt_record=object();owner._sync_source_catalog()
        def fail(*args,**kwargs):raise ValueError('failure')
        owner._source_catalog.refreshCurrent=fail
        with self.assertRaises(ValueError):owner._pl_refresh_catalog_explicit()
        self.assertTrue(owner._source_catalog.busy)
        self.assertIsNone(owner._catalog_explicit_scope)
        owner._source_catalog.refreshCurrent=lambda *args,**kwargs: (setattr(owner._source_catalog, '_generation', 1), setattr(owner._source_catalog, '_busy', True))
        owner._pl_refresh_catalog_explicit()
        owner._source_catalog._generation+=1  # another operation cancels the scan
        owner._sync_source_catalog()
        self.assertIsNone(owner._catalog_explicit_scope)
        self.assertTrue(owner._source_catalog.busy)

    def test_refresh_not_started_clears_override(self):
        owner=Owner();owner._ppt_record=object();owner._sync_source_catalog()
        owner._source_catalog.refreshCurrent=lambda *args,**kwargs: None
        owner._pl_refresh_catalog_explicit()
        self.assertIsNone(owner._catalog_explicit_scope)
        self.assertTrue(owner._source_catalog.busy)
