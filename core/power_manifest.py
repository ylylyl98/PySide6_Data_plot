"""Resolve whole sweeps through their acquisition run, condition and repeat."""
import json
import math
import re
from pathlib import Path

from core.processing import classify_compare_channel, infer_compare_angle_references


def manifest_assignments(folder, manifests, refs, tolerance):
    root = Path(folder).resolve()
    assignments = {}
    for name in manifests:
        path = (root / name).resolve()
        if '_manifest' not in path.stem.casefold():
            continue
        try:
            document = json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            continue
        rows = document.get('conditions', []) if isinstance(document, dict) else []
        if not isinstance(rows, list):
            continue
        groups = {}
        for row in rows:
            if not isinstance(row, dict) or not row.get('file') or row.get('condition_index') is None:
                continue
            recorded = Path(str(row['file']).replace('\\', '/'))
            target = recorded if recorded.is_absolute() else path.parent / recorded
            if not target.exists():
                target = path.parent / recorded.name
            try:
                relative = target.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            rotations = row.get('rotation_requested', {})
            if not isinstance(rotations, dict):
                continue
            try:
                angles = {k: float(v) for k, v in rotations.items() if k in ('rot1', 'rot2')}
            except (ValueError, TypeError):
                continue
            if not angles or not all(math.isfinite(v) for v in angles.values()):
                continue
            angle_name = '_'.join(f'{k}{v:g}deg' for k, v in angles.items()) + '.csv'
            # Signed voltages remain part of identity, even for malformed manifests.
            gates = re.findall(r'(?:Vbg|Vtg|Vb)[+\-]?\d+(?:[pP.]\d+)?', target.stem, re.I)
            context = (path.relative_to(root).as_posix() +
                       f"/condition={row['condition_index']}/repeat={row.get('repeat', 1)}/" + '_'.join(gates))
            groups.setdefault(context, {})[relative.casefold()] = angle_name
        for context, members in groups.items():
            local_refs = dict(refs or {})
            if len(members) == 2:
                inferred = infer_compare_angle_references(list(members.values()),
                    in_k_anchor=local_refs.get('in_k', 0.), out_k_anchor=local_refs.get('out_k', 0.))
                for key in ('in_k', 'in_kp', 'out_k', 'out_kp'):
                    value = getattr(inferred, key)
                    if value is not None:
                        local_refs[key] = value
            for relative, angle_name in members.items():
                role = classify_compare_channel(angle_name,
                    in_k_angle=local_refs.get('in_k', 0.), out_k_angle=local_refs.get('out_k', 0.),
                    in_kp_angle=local_refs.get('in_kp'), out_kp_angle=local_refs.get('out_kp'),
                    tolerance=tolerance)
                assignments.setdefault(relative, set()).add((context, role if role in ('KK', 'KKp') else None))
    return {name: next(iter(values)) if len(values) == 1 else (name + '/conflicting metadata', None)
            for name, values in assignments.items()}
