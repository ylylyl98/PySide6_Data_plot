"""Thread-scoped elapsed times for load workers; nested stages are inclusive."""
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

_durations = ContextVar('load_stage_durations', default=None)


@contextmanager
def collect_load_timings():
    durations = {}
    token = _durations.set(durations)
    try:
        yield durations
    finally:
        _durations.reset(token)


@contextmanager
def time_load_stage(name):
    durations = _durations.get()
    if durations is None:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        durations[name] = durations.get(name, 0.) + perf_counter() - started
