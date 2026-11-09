"""Explicit reference roles; filename evidence never certifies physical identity."""
from pathlib import Path
import re

# Keep these token patterns aligned with drr_pn_pairing and drr_measured_bn.
_PN = re.compile(r'(?<![A-Za-z0-9])(?P<role>[PN])back(?:ground)?\d*(?=$|[_\-\s(\[])', re.I)
_BN = re.compile(r'(?<![a-z0-9])bn[ _-]*back(?:ground)?(?=$|[^a-z])', re.I)


def validate_background_role(role, paths):
    """Reject named role conflicts; preserve unnamed manually assigned files."""
    role=str(role).lower()
    if role not in {'custom','p','n','bn'}:
        raise ValueError('Choose a supported background role: custom, P, N or BN.')
    found=set(); sources=[]; manual=[]
    for value in paths:
        path=Path(value).resolve()
        hints={m.group('role').lower() for m in _PN.finditer(path.stem)}
        if _BN.search(path.stem): hints.add('bn')
        if len(hints)>1:
            raise ValueError(f'Background filename contains conflicting role hints: {path.name}. Choose separately identified sources.')
        hint=next(iter(hints),'unknown')
        if hint!='unknown': found.add(hint)
        else: manual.append(str(path))
        if role!='custom' and hint not in (role,'unknown'):
            raise ValueError(f'{path.name} is identified as {hint.upper()}, not {role.upper()}; choose the matching background group.')
        sources.append(dict(path=str(path),filename_role=hint,
            assignment='manual_unknown' if hint=='unknown' else 'explicit_custom' if role=='custom' else 'matching_hint'))
    if role=='custom' and len(found)>1:
        raise ValueError('Custom references contain different P, N or BN roles; select each role in its own background group.')
    return dict(role=role,named_roles=sorted(found),manual_paths=manual,sources=sources,weighting_scope='active_group_only')


def inspect_background_roles(paths, role='custom'):
    return validate_background_role(role, paths)
