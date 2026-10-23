"""Catalog-only ordering for explicit BN selection; never an automatic match."""
from __future__ import annotations

import math

import numpy as np

from core.drr_baseline_candidates import (
    DrrBaselineRecommendation, parse_drr_acquisition_conditions, spectral_grids_match,
)
from core.drr_measured_bn import is_measured_bn_name


def _known_grid(source):
    values = np.asarray(source.spectral_grid, dtype=float).ravel()
    return values.size >= 2 and bool(np.all(np.isfinite(values)))


def bn_grid_state(measurements, candidate):
    """0=all cached grids match, 1=unknown, 2=a known grid conflicts."""
    if candidate is None or not _known_grid(candidate):
        return 1
    unknown = not measurements
    for measurement in measurements:
        if not _known_grid(measurement):
            unknown = True
        elif not spectral_grids_match(measurement, candidate):
            return 2
    return 1 if unknown else 0


def _rotations(conditions):
    return dict((arm.replace('rotation', 'rot'), value)
                for part in (conditions.rotation or '').split(';')
                if '=' in part for arm, _, value in (part.partition('='),))


def _rotation_state(measurements, candidate):
    # RotIn and RotOut describe different arms. An absent arm is unknown,
    # while a contradiction in any known arm cannot count as a match.
    unknown = not measurements or not candidate
    for expected in measurements:
        if any(arm in candidate and candidate[arm] != value for arm, value in expected.items()):
            return 2
        if not expected or expected != candidate:
            unknown = True
    return 1 if unknown else 0


def _angle_text(rotations):
    labels = {'rotin': 'RotIn', 'rotout': 'RotOut', 'rot': 'Rotation'}
    return ', '.join(f'{labels.get(arm, arm)} {value}°' for arm, value in sorted(rotations.items())) or 'unknown'


def _condition_state(measurements, candidate, key):
    value = getattr(candidate, key)
    expected = [getattr(item, key) for item in measurements]
    if value is not None and any(item is not None and item != value for item in expected):
        return 2
    return 1 if value is None or not expected or any(item is None for item in expected) else 0


def rank_bn_candidates(measurements, candidates):
    """Show same-angle BN first using cached metadata, with explicit fallbacks.

    Unknown grids stay available for validation when applying a manual draft;
    known incompatible grids are shown last and rejected by the picker. This
    helper performs no file, spectral-body or acquisition-log reads.
    """
    measurements = tuple(measurements)
    conditions = tuple(parse_drr_acquisition_conditions(item.filename) for item in measurements)
    rotations = tuple(_rotations(item) for item in conditions)
    keys = ('sample', 'position', 'temperature', 'magnetic_field', 'polarization',
            'power', 'exposure', 'accumulation')
    ranked = []
    for source in candidates:
        if not is_measured_bn_name(source.filename):
            continue
        parsed = parse_drr_acquisition_conditions(source.filename)
        angle = _rotations(parsed)
        angle_state = _rotation_state(rotations, angle)
        grid_state = bn_grid_state(measurements, source)
        states = tuple(_condition_state(conditions, parsed, key) for key in keys)
        label = ('Same angle: ' if angle_state == 0 else
                 'Angle not fully known: ' if angle_state == 1 else 'Different angle: ') + _angle_text(angle)
        grid_label = ('spectral grid matches all measurements', 'spectral grid needs validation',
                      'incompatible spectral grid')[grid_state]
        differences = [key.replace('_', ' ') for key, state in zip(keys, states) if state == 2]
        reason = label + '; ' + grid_label
        if differences:
            reason += '; different ' + ', '.join(differences)
        modified = float(source.modified_time)
        sort_key = (grid_state == 2, angle_state, states, grid_state,
                    -modified if math.isfinite(modified) and modified > 0 else float('inf'),
                    source.filename.casefold(), source.source.casefold(), source.source)
        ranked.append((sort_key, DrrBaselineRecommendation(source, 3 if grid_state == 2 else angle_state, reason)))
    return tuple(item for _, item in sorted(ranked, key=lambda entry: entry[0]))
