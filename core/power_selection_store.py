"""Portable, explicitly confirmed Power channel assignments.

Only source keys relative to the experiment are stored. Missing keys are kept
when reading so the picker can show a missing assignment instead of replacing it.
"""
import json
import os
from pathlib import Path
import tempfile


SELECTION_FILE = '.power-selection.json'
ROLES = ('single', 'KK', 'KKp')


def _read(folder):
    path = Path(folder) / SELECTION_FILE
    if not path.exists():
        return {'version': 1, 'groups': {}}
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('version') != 1 or not isinstance(value.get('groups'), dict):
        raise ValueError('Unrecognized saved Power selection format.')
    return value


def load_power_selections(folder, sources=None):
    """Return saved role keys; the picker checks them against its current catalog."""
    if not folder:
        return {}
    value = _read(folder)
    result = {}
    for group, assignments in value['groups'].items():
        if not isinstance(group, str) or not isinstance(assignments, dict):
            continue
        result[group] = {role: key.replace('\\', '/') for role, key in assignments.items()
                         if role in ROLES and isinstance(key, str)}
    return result


def save_power_selection(folder, group_key, assignments, sources):
    """Atomically remember a user-confirmed assignment without modifying CSVs."""
    if not folder or not Path(folder).is_dir():
        raise ValueError('Select an experiment folder before saving a pairing.')
    choices = {role: str(assignments.get(role) or '') for role in ROLES}
    if not group_key or any(key and key not in sources for key in choices.values()):
        raise ValueError('Cannot remember a pairing whose source files are missing.')
    if choices['KK'] and choices['KK'] == choices['KKp']:
        raise ValueError('KK and KKp must use different sweeps.')
    value = _read(folder)
    value['groups'][str(group_key)] = choices
    destination = Path(folder) / SELECTION_FILE
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=folder,
                                         prefix='.power-selection-', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
