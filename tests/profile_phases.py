"""Small, opt-in phase timing probe for focused profiler runs."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _emit(event: str, phase: str, **fields: object) -> None:
    path = os.environ.get("PROFILE_PHASE_LOG")
    if not path:
        return
    payload = {
        "event": event,
        "phase": phase,
        "test_id": os.environ.get("PROFILE_TEST_CURRENT_ID"),
        "full_id": os.environ.get("PROFILE_TEST_CURRENT_ID", ""),
        "segment_id": os.environ.get("PROFILE_TEST_CURRENT_SEGMENT_ID"),
        "attempt_id": os.environ.get("PROFILE_TEST_CURRENT_ATTEMPT_ID"),
        "timestamp_utc": time.time(),
        "monotonic_ns": time.perf_counter_ns(),
        **fields,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


@contextmanager
def profile_phase(name: str) -> Iterator[None]:
    """Measure one phase only when the bounded profiler requested it."""

    started = time.perf_counter()
    _emit("PHASE_START", name)
    try:
        yield
    finally:
        _emit("PHASE_END", name, duration_s=time.perf_counter() - started)


def emit_phase_duration(name: str, duration_s: float, **fields: object) -> None:
    """Record a child-process phase measured by its own monotonic clock."""

    _emit("PHASE_END", name, duration_s=duration_s, **fields)
