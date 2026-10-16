"""Bounded, disposable disk cache of compact fitted denominators (never raw data)."""
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4
from zipfile import BadZipFile
import numpy as np

from core.drr_fitted_background import BackgroundFitSettings, FittedReference, estimate_background
from core.load_cancellation import check_load_cancelled

CACHE_VERSION = 9  # Bump whenever numerical fitting or quality checks change.
_lock = RLock()


def cache_directory():
    override = os.environ.get('DPTK_FITTED_CACHE_DIR')
    if override:
        return Path(override)
    base = os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_CACHE_HOME')
    return (Path(base) if base else Path.home()/'.cache')/'DPTK'/'fitted-background'


def _key(cube, settings, source_path, y_axis):
    digest = hashlib.sha256(json.dumps(dict(version=CACHE_VERSION, settings=asdict(settings),
        source=str(Path(source_path).resolve()), y_axis=y_axis), sort_keys=True).encode())
    for array in (cube.energy, cube.gate, cube.Z):
        check_load_cancelled()
        canonical = np.ascontiguousarray(array, dtype='<f8')
        digest.update(str(canonical.shape).encode())
        digest.update(memoryview(canonical).cast('B'))
    return digest.hexdigest()


def _prune(directory):
    entries = []
    for path in directory.glob('*.npz'):
        if len(path.stem) == 64 and all(c in '0123456789abcdef' for c in path.stem):
            try:
                stat = path.stat()
                entries.append((stat.st_mtime_ns, stat.st_size, path))
            except OSError:
                pass
    total = sum(size for _, size, _ in entries)
    entries.sort()
    while entries and (len(entries) > 256 or total > 128*1024*1024):
        _, size, path = entries.pop(0)
        try:
            path.unlink(missing_ok=True)
            total -= size
        except OSError:
            pass


def fit_cached(cube, settings, *, source, signature, source_path, y_axis='auto'):
    settings = BackgroundFitSettings.from_mapping(settings)
    key = _key(cube, settings, source_path, y_axis)
    directory = cache_directory()
    path = directory/f'{key}.npz'
    order = np.argsort(cube.energy)
    ordered = replace(cube, energy=np.asarray(cube.energy)[order], Z=np.asarray(cube.Z)[:, order])
    try:
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['key'].item()) != key:
                raise ValueError('Cache identity mismatch')
            shape, gain, mask = saved['log_shape'], saved['gain_tilt'], saved['fit_mask']
            diagnostics = json.loads(str(saved['diagnostics'].item()))
            effective_settings = BackgroundFitSettings.from_mapping(json.loads(str(saved['effective_settings'].item())))
            gate_mask, texture, coverage = saved['gate_fit_mask'], saved['log_texture'], saved['texture_coverage']
            comparable = replace(effective_settings, protected_regions=settings.protected_regions) if settings.automatic_protection else effective_settings
            if (comparable != settings or effective_settings.automatic_protection != bool(gate_mask.size)
                    or effective_settings.shared_texture != bool(texture.size)):
                raise ValueError('Cache settings or components mismatch')
            if (shape.shape != ordered.energy.shape or gain.shape != (len(cube.gate), 2)
                    or mask.shape != shape.shape or mask.dtype != bool or not mask.any()
                    or not np.isfinite(shape).all() or not np.isfinite(gain).all()
                    or not isinstance(diagnostics, dict)):
                raise ValueError('Invalid cache arrays')
            if (gate_mask.size and (gate_mask.shape != ordered.Z.shape or gate_mask.dtype != bool
                                    or np.any(gate_mask.sum(axis=1) < 24))):
                raise ValueError('Invalid per-gate protection')
            if any(a.size and (a.shape != shape.shape or not np.isfinite(a).all()) for a in (texture, coverage)):
                raise ValueError('Invalid texture arrays')
            if texture.size != coverage.size or np.any((coverage < 0) | (coverage > 1)):
                raise ValueError('Invalid texture coverage')
            reference = FittedReference(source, signature, ordered.energy.copy(), cube.gate.copy(),
                shape.copy(), gain.copy(), mask.copy(), cube.gate_label, effective_settings,
                float(saved['anchor_rms'].item()), diagnostics, cache_hit=True,
                gate_fit_mask=gate_mask.copy() if gate_mask.size else None,
                log_texture=texture.copy() if texture.size else None,
                texture_coverage=coverage.copy() if coverage.size else None)
            background = reference.background()
            if not np.isfinite(background).all() or np.any(background <= 0):
                raise ValueError('Invalid cached background')
            check_load_cancelled()
            return replace(ordered, Z=ordered.Z/background-1, title=f'{cube.title} (fitted reference)',
                           cbar_label='I/B_fit - 1'), reference
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        pass
    result, reference = estimate_background(cube, settings, source=source, signature=signature)
    temporary = directory/f'.{uuid4().hex}.tmp'
    try:
        check_load_cancelled()
        with _lock:
            directory.mkdir(parents=True, exist_ok=True)
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, key=key, log_shape=reference.log_shape,
                    gain_tilt=reference.gain_tilt, fit_mask=reference.fit_mask,
                    effective_settings=json.dumps(asdict(reference.settings)),
                    gate_fit_mask=reference.gate_fit_mask if reference.gate_fit_mask is not None else np.array([], bool),
                    log_texture=reference.log_texture if reference.log_texture is not None else np.array([]),
                    texture_coverage=reference.texture_coverage if reference.texture_coverage is not None else np.array([]),
                    anchor_rms=reference.anchor_rms_percent, diagnostics=json.dumps(reference.diagnostics))
            temporary.replace(path)
            _prune(directory)
    except OSError:
        pass  # A read-only/full cache must never prevent a valid fit.
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return result, reference
