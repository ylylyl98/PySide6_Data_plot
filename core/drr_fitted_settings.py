"""Fitted-background settings and validation without fitting or I/O imports."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BackgroundFitSettings:
    protected_regions: tuple[tuple[float, float], ...] = ()
    spline_intervals: int = 10
    smoothing: float = .001
    method: str = 'spline'
    polynomial_degree: int = 3
    automatic_protection: bool = False
    shared_texture: bool = True
    periodic_texture: bool = False
    periodic_components: int = 3
    texture_reference_source: str = ''

    def validate(self):
        import numpy as np
        if not isinstance(self.automatic_protection, bool):
            raise ValueError('Automatic peak protection must be enabled or disabled.')
        if not isinstance(self.shared_texture, bool):
            raise ValueError('Shared texture correction must be enabled or disabled.')
        if not isinstance(self.periodic_texture, bool):
            raise ValueError('Periodic gap prediction must be enabled or disabled.')
        if not isinstance(self.texture_reference_source, str):
            raise ValueError('Fringe reference source must be a file path or an empty string.')
        if (isinstance(self.periodic_components, bool) or not isinstance(self.periodic_components, int)
                or not 1 <= self.periodic_components <= 5):
            raise ValueError('Maximum periodic components must be an integer between 1 and 5.')
        if not self.protected_regions and not self.automatic_protection:
            raise ValueError('Select protected resonance regions in Fit / view background first.')
        if not isinstance(self.spline_intervals, int) or not 4 <= self.spline_intervals <= 40:
            raise ValueError('Spline intervals must be an integer between 4 and 40.')
        if not np.isfinite(self.smoothing) or not .000001 <= self.smoothing <= 10:
            raise ValueError('Smoothing must be between 0.000001 and 10.')
        if self.method not in {'spline', 'polynomial'}:
            raise ValueError('Choose spline or polynomial background fitting.')
        if not isinstance(self.polynomial_degree, int) or not 2 <= self.polynomial_degree <= 4:
            raise ValueError('Polynomial degree must be an integer between 2 and 4.')
        for lo, hi in self.protected_regions:
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError('Each protected region needs two finite, increasing energy limits.')
        return self

    @classmethod
    def from_mapping(cls, values):
        if isinstance(values, cls):
            return values.validate()
        if not isinstance(values, dict):
            raise ValueError('Background settings must be a mapping.')
        return cls(tuple(tuple(map(float, pair)) for pair in values.get('protected_regions', ())),
                   values.get('spline_intervals', 10), float(values.get('smoothing', .001)),
                   values.get('method', 'spline'), values.get('polynomial_degree', 3),
                   values.get('automatic_protection', False), values.get('shared_texture', True),
                   values.get('periodic_texture', False), values.get('periodic_components', 3),
                   values.get('texture_reference_source', '')).validate()


def settings_for_source(settings, source):
    if isinstance(settings, BackgroundFitSettings):
        return settings.validate()
    if not isinstance(settings, dict):
        raise ValueError('Background recipe must be a mapping.')
    overrides = settings.get('protected_regions_by_file', {})
    if not isinstance(overrides, dict):
        raise ValueError('Per-file protected regions must be a mapping.')
    return BackgroundFitSettings.from_mapping({**settings,
        'protected_regions': overrides.get(source, settings.get('protected_regions', ()))})


def validate_recipe(settings, files):
    settings_for_group(settings, files)


def settings_for_group(settings, files):
    """Migrate legacy manual per-file protection conservatively to one fit."""
    import numpy as np
    if isinstance(settings, BackgroundFitSettings):
        return settings.validate()
    if not isinstance(settings, dict):
        raise ValueError('Background recipe must be a mapping.')
    overrides = settings.get('protected_regions_by_file', {})
    if not isinstance(overrides, dict):
        raise ValueError('Per-file protected regions must be a mapping.')
    if settings.get('automatic_protection', False):
        return BackgroundFitSettings.from_mapping(settings)
    regions = []
    for source in files:
        regions.extend(overrides.get(source, settings.get('protected_regions', ())))
    merged = []
    for lo, hi in sorted(tuple(map(float, pair)) for pair in regions):
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError('Each protected region needs two finite, increasing energy limits.')
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return BackgroundFitSettings.from_mapping({**settings, 'protected_regions': merged})
