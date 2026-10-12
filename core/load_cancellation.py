"""Cooperative, thread-scoped cancellation for read-only load workflows."""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event


class LoadCancelled(Exception):
    """The load was superseded or invalidated; it is not a loading error."""


class LoadCancellation:
    def __init__(self):
        self._event = Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()

    def check(self):
        if self.cancelled:
            raise LoadCancelled()


_current = ContextVar('load_cancellation', default=None)


def check_load_cancelled():
    cancellation = _current.get()
    if cancellation is not None:
        cancellation.check()


@contextmanager
def load_scope(cancellation):
    scope = _current.set(cancellation)
    try:
        check_load_cancelled()
        yield
    finally:
        _current.reset(scope)
