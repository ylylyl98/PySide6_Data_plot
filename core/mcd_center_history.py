"""Per-source MCD window history, independent of the current recommendations."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from core.source_identity import match_source_identity


def _window_key(value):
    try:
        center, width = float(value['center_ev']), float(value['width_mev'])
        if math.isfinite(center) and math.isfinite(width) and width > 0:
            return round(center, 9), round(width, 6)
    except (KeyError, TypeError, ValueError):
        pass
    return None


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def read_center_history(root, source, *, extra_roots=()):
    """Read every matching formally saved window off the UI thread."""
    from core.data_io import list_mcd_csv_files
    root = Path(root)
    source = str(source).replace('\\', '/')
    sources = list_mcd_csv_files(str(root))
    if source not in sources:
        sources.append(source)
    found = {}

    def add(value, stamp, evidence, count=1):
        key = _window_key(value)
        if key is None:
            return
        row = found.setdefault(key, {'center_ev': key[0], 'width_mev': key[1],
                                     'last_used': '', 'uses': 0, 'records': []})
        if evidence in row['records']:
            return
        row['records'].append(evidence)
        row['uses'] += count
        row['last_used'] = max(row['last_used'], str(stamp))

    roots = [root / 'Processed Data' / 'MCD', (root / source).parent / 'Processed Data' / 'MCD',
             *[Path(p) for p in extra_roots if p]]
    seen = set()
    for directory in roots:
        if not directory.is_dir():
            continue
        for path in directory.rglob('*_MCD_settings*.json'):
            if any(part.startswith('.staging-') for part in path.parts):
                continue
            identity = str(path.resolve()).casefold()
            if identity in seen:
                continue
            seen.add(identity)
            payload = _read_json(path)
            if str(payload.get('workflow', '')).casefold() != 'mcd':
                continue
            descriptors = payload.get('sources', [])
            descriptor = next((d for d in descriptors if isinstance(d, dict)
                               and str(d.get('role', '')).casefold() == 'measurement'), {}) if isinstance(descriptors, list) else {}
            matched, _ = match_source_identity(root, sources,
                relative_path=payload.get('source_relative_path', descriptor.get('source_relative_path', '')),
                path=payload.get('source_path', descriptor.get('source_path', '')),
                legacy_name=payload.get('source_file', descriptor.get('filename', '')))
            if matched != source:
                continue
            try:
                stamp = payload.get('created_utc') or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            except OSError:
                continue
            windows = payload.get('windows', [])
            windows = [windows] if isinstance(windows, dict) else windows if isinstance(windows, list) else []
            for window in [payload.get('mcd_b', {}), *windows]:
                add(window, stamp, str(path))
    return sorted(found.values(), key=lambda row: (row['center_ev'], row['width_mev']))


def merge_center_candidates(recommendations, history):
    """Preserve H/R identity and merge coincident energy/width markers."""
    output = [dict(item) for item in recommendations]
    for item in output:
        if item.get('kind') == 'window':
            item['display_id'] = 'R' + str(item.get('label', item.get('display_id', '')))
    for number, record in enumerate(sorted(history, key=lambda r: (r['center_ev'], r['width_mev'])), 1):
        key = _window_key(record)
        if key is None:
            continue
        item = next((item for item in output if item.get('kind') == 'window' and _window_key(item) == key), None)
        if item is None:
            item = {'id': f'history-{key[0]:.9f}-{key[1]:.6f}', 'domain': 'mcd', 'source': 'MCD history',
                    'kind': 'window', 'center_ev': key[0], 'energy_ev': key[0], 'width_mev': key[1],
                    'recommended': False}
            output.append(item)
            item['display_id'] = f'H{number}'
        else:
            item['display_id'] = f'H{number}/{item["display_id"]}'
        item['history'] = dict(record)
    return output
