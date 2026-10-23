"""Explicit fringe-reference loading with content-bound cache identity.

A constant gate reference contains repeated frames, not a gate sweep. Keep
every frame in acquisition order and use frame indices for its DataCube axis.
"""
import hashlib
from pathlib import Path

import numpy as np

from core.loader import DataCube
from core.load_cancellation import check_load_cancelled
from core.raw_spectrum_cache import RawSpectrumCache


_REFERENCE_CACHE = RawSpectrumCache(max_bytes=32 * 1024 * 1024)


def _snapshot(path):
    check_load_cancelled()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            check_load_cancelled()
            digest.update(chunk)
    after = path.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise ValueError('Fringe reference changed while reading; reload after acquisition finishes.')
    return dict(source=str(path), sha256=digest.hexdigest(),
                signature=[after.st_mtime_ns, after.st_size], role='fringe_only')


def load_texture_reference(source, *, measurement_path):
    """Load a selected CSV; relative recipes resolve beside the measurement."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError('Select a fringe reference CSV first.')
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = Path(measurement_path).resolve().parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f'Fringe reference is missing: {path}. Select it again or clear the reference.')
    if path.suffix.lower() != '.csv':
        raise ValueError('Fringe reference must be a raw intensity CSV.')
    before = _snapshot(path)
    from core.processing_run import _RAW_PARSE_VERSION, _parse_raw_spectrum
    # Include content in the raw-cache key. Replacing a reference while retaining
    # its size and timestamp must never reuse stale frame arrays.
    arrays = _REFERENCE_CACHE.load(path, ('fringe', _RAW_PARSE_VERSION, before['sha256']),
                                   _parse_raw_spectrum)
    energy, _, _, _, intensity = arrays
    energy, intensity = np.asarray(energy, float), np.asarray(intensity, float)
    if (energy.ndim != 1 or intensity.ndim != 2 or intensity.shape[1] != len(energy)
            or not len(intensity) or len(energy) < 7
            or not np.isfinite(energy).all() or np.any(energy <= 0)
            or not np.isfinite(intensity).all() or np.any(intensity <= 0)):
        raise ValueError('Fringe reference requires finite positive energy and raw intensities in every frame.')
    order = np.argsort(energy)
    energy = energy[order].copy()
    if np.any(np.diff(energy) <= 0):
        raise ValueError('Fringe reference requires distinct spectral coordinates.')
    after = _snapshot(path)
    if before != after:
        raise ValueError('Fringe reference changed while loading; reload after acquisition finishes.')
    cube = DataCube(energy, np.arange(len(intensity), dtype=float), intensity[:, order].copy(),
                    'Reference frame', path.name, 'Intensity (counts)')
    return cube, after


def verify_texture_reference(metadata):
    """Reject an inspector/cache result whose selected reference has changed."""
    if not isinstance(metadata, dict) or not metadata.get('source') or not metadata.get('sha256'):
        raise ValueError('Fringe-reference identity is unavailable; refit before inspecting this background.')
    path = Path(metadata['source'])
    try:
        current = _snapshot(path)
    except OSError as exc:
        raise ValueError('Fringe reference is no longer available; reload or clear the selected reference.') from exc
    if current['sha256'] != metadata['sha256']:
        raise ValueError('Fringe reference changed after fitting; reload DRR before inspecting its background.')
    check_load_cancelled()
