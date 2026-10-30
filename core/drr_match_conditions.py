"""Shared acquisition-condition policy for DRR background matching."""

CONDITION_LABELS = {
    "sample": "Sample",
    "position": "Position",
    "temperature": "Temperature",
    "magnetic_field": "Magnetic field",
    "polarization": "Polarization",
    "rotation": "Rotation",
    "power": "Power",
    "exposure": "Exposure",
    "accumulation": "Accumulation",
    "wavelength_center_nm": "Wavelength center (nm)",
}


def normalize_relaxed_conditions(values):
    """Validate selected field keys and return them in canonical display order."""
    if isinstance(values, (str, bytes)):
        raise ValueError("Relaxed conditions must be a collection of field keys.")
    try:
        selected = set(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("Relaxed conditions must be a collection of field keys.") from exc
    if any(not isinstance(key, str) or key not in CONDITION_LABELS for key in selected):
        raise ValueError("Unknown relaxed acquisition condition.")
    return tuple(key for key in CONDITION_LABELS if key in selected)
