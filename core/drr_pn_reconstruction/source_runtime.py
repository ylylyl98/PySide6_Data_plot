"""Cancellation hooks for the private source-fitting kernels (no data IO)."""
from contextlib import contextmanager
from contextvars import ContextVar

from scipy.optimize import least_squares as _least_squares

_cancelled = ContextVar("pn_reconstruction_cancelled", default=None)


def check_cancelled():
    callback = _cancelled.get()
    if callback is not None and callback():
        raise InterruptedError("P/N source reconstruction cancelled")


@contextmanager
def cancellation_scope(callback):
    token = _cancelled.set(callback)
    try:
        check_cancelled()
        yield
    finally:
        _cancelled.reset(token)


def least_squares(fun, *args, **kwargs):
    """Keep solver mathematics unchanged; check cancellation per evaluation."""
    def checked(*values, **options):
        check_cancelled()
        return fun(*values, **options)
    jac = kwargs.get("jac")
    if callable(jac):
        def checked_jac(*values, **options):
            check_cancelled()
            return jac(*values, **options)
        kwargs["jac"] = checked_jac
    return _least_squares(checked, *args, **kwargs)
