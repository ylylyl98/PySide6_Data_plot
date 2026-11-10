"""Human-readable names and locations for peak analysis output."""
from __future__ import annotations
import json
import os
from pathlib import Path
import unicodedata


def _primary(dataset):
    if dataset.get('processed_path'):
        return str(dataset['processed_path'])
    source = dataset.get('source') or ''
    try:
        parsed = json.loads(source) if isinstance(source, str) else source
    except (ValueError, TypeError):
        parsed = source
    if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
        parsed = parsed[1]
    if not parsed:
        raise ValueError('No processed file or source path is available for analysis output.')
    return str(parsed)


def output_stem(dataset):
    """Preserve the complete filename with readable ASCII normalization."""
    stem = Path(_primary(dataset)).stem
    text = unicodedata.normalize('NFKC', stem).translate(str.maketrans({'−':'-', '–':'-', '—':'-', 'μ':'u', 'µ':'u', 'Δ':'Delta', 'δ':'delta', 'Ω':'Ohm', '±':'+-'}))
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    if not text or any(ord(c) < 32 or ord(c) > 126 or c in '<>:"/\\|?*' for c in text) or text.endswith((' ', '.')):
        raise ValueError(f'Cannot create a readable ASCII output name from {stem!r}; please rename the source file.')
    return text


def output_folder(dataset, fallback_root):
    """Return processed file's parent, or the experiment's processed directory."""
    path = Path(_primary(dataset)).expanduser()
    if not path.is_absolute():
        path = Path(fallback_root) / path
    path = Path(os.path.abspath(path))
    parts = path.parts
    index = next ( (i for i,p in enumerate(parts) if p.casefold() == 'initial data'), None)
    if index is not None:
        kind = str(dataset.get('kind') or 'Unknown')
        if not kind.isascii() or not kind.isalnum():
            raise ValueError('Invalid analysis kind for output directory.')
        return Path(*parts[:index]) / 'Processed Data' / kind
    return path.parent if path.suffix else path
