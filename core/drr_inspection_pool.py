"""Bounded process workers for large, uncached DRR CSV inventories."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context

from core.load_cancellation import LoadCancelled, check_load_checkpoint, load_scope


_stop = None


def _initialize(stop):
    global _stop
    _stop = stop


class _Cancellation:
    def check(self):
        if _stop is not None and _stop.is_set():
            raise LoadCancelled()


def _inspect(path):
    from core.drr_sources import inspect_csv_gate_profile
    with load_scope(_Cancellation()):
        return inspect_csv_gate_profile(path, max_rows=None)


def inspect_csv_profiles(paths, *, on_profile=None):
    """Return path/profile pairs; startup failure falls back to caller's scan.

    At most two processes and four submitted files are retained. Cancellation
    stops the current rows before joining workers, including frozen Windows
    executables whose entry point calls multiprocessing.freeze_support().
    """
    context = get_context('spawn')
    stop = None
    pool = None
    pending = {}
    result = {}
    remaining = iter(paths)
    try:
        stop = context.Event()
        pool = ProcessPoolExecutor(max_workers=2, mp_context=context,
                                   initializer=_initialize, initargs=(stop,))
        while True:
            check_load_checkpoint()
            while len(pending) < 4:
                path = next(remaining, None)
                if path is None:
                    break
                pending[pool.submit(_inspect, str(path))] = path
            if not pending:
                return result
            completed, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
            for future in completed:
                path = pending.pop(future)
                profile = future.result()
                result[path] = profile
                if on_profile is not None:
                    on_profile(path, profile)
    except (BrokenProcessPool, OSError, RuntimeError):
        # A restricted host may not permit child processes. Never fail a
        # normal catalog refresh solely because acceleration is unavailable.
        return result
    finally:
        if stop is not None:
            stop.set()
        for future in pending:
            future.cancel()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
