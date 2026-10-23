"""Reproducible BN pipeline identities for source and PYZ installations.

Development fingerprints cover the real source bytes. A frozen module's
``__file__`` need not exist: its loader supplies the compiled module instead.
Code identities exclude build/extraction paths and debugging line positions,
while retaining executable instructions, exception handlers and all constants.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import CodeType

from core.load_cancellation import check_load_cancelled


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'), allow_nan=False).encode('ascii')


def _constant(value):
    # Typed constants avoid ambiguities such as False == 0. Sorting the encoded
    # frozenset members also makes identities independent of PYTHONHASHSEED.
    if isinstance(value, CodeType):
        return ['code', _code(value)]
    if value is None:
        return ['none']
    if value is Ellipsis:
        return ['ellipsis']
    if isinstance(value, bool):
        return ['bool', value]
    if isinstance(value, int):
        return ['int', str(value)]
    if isinstance(value, float):
        return ['float64', struct.pack('>d', value).hex()]
    if isinstance(value, complex):
        return ['complex128', struct.pack('>dd', value.real, value.imag).hex()]
    if isinstance(value, str):
        return ['str', value]
    if isinstance(value, bytes):
        return ['bytes', value.hex()]
    if isinstance(value, tuple):
        return ['tuple', [_constant(item) for item in value]]
    if isinstance(value, frozenset):
        return ['frozenset', sorted((_constant(item) for item in value), key=_encoded)]
    if isinstance(value, slice):
        return ['slice', [_constant(item) for item in (value.start, value.stop, value.step)]]
    raise TypeError(f'Unsupported code constant type: {type(value).__name__}')


def _code(code):
    # Deliberately exclude co_filename, co_firstlineno and the line/position
    # tables: a PYZ loader rewrites paths under a different _MEIPASS each run.
    # The exception table changes execution and must remain in the identity.
    fields = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_nlocals',
              'co_stacksize', 'co_flags', 'co_name', 'co_qualname', 'co_names',
              'co_varnames', 'co_freevars', 'co_cellvars')
    result = {field: _constant(getattr(code, field)) for field in fields if hasattr(code, field)}
    result['bytecode'] = code.co_code.hex()
    result['exception_table'] = getattr(code, 'co_exceptiontable', b'').hex()
    result['constants'] = [_constant(value) for value in code.co_consts]
    return result


def semantic_code_digest(code):
    """Hash complete executable code, recursively, without machine-local paths."""
    if not isinstance(code, CodeType):
        raise TypeError('A module code object is required for a frozen fingerprint.')
    return hashlib.sha256(_encoded(_code(code))).hexdigest()


def python_runtime_identity():
    """Bytecode and exception-table semantics are specific to this runtime."""
    version = sys.version_info
    return dict(implementation=sys.implementation.name,
                cache_tag=sys.implementation.cache_tag,
                version=[version.major, version.minor, version.micro,
                         version.releaselevel, version.serial])


def _source_path(module):
    filename = getattr(module, '__file__', None)
    if not filename:
        return None
    path = Path(filename)
    if path.suffix.lower() == '.pyc':
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except ValueError:
            path = path.with_suffix('.py')
    return path if path.suffix.lower() in ('.py', '.pyw') and path.is_file() else None


def _source_digest(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            check_load_cancelled()
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Pipeline source changed while fingerprinting: {path.name}')
    return digest.hexdigest()


def module_fingerprint(name, *, module=None):
    """Read real source bytes or a module loader's code; never omit a module."""
    check_load_cancelled()
    module = importlib.import_module(name) if module is None else module
    source = _source_path(module)
    if source is not None:
        return dict(kind='source_bytes', sha256=_source_digest(source))
    spec = getattr(module, '__spec__', None)
    loader = getattr(spec, 'loader', None) or getattr(module, '__loader__', None)
    get_code = getattr(loader, 'get_code', None)
    if not callable(get_code):
        raise ValueError(f'Cannot fingerprint pipeline module {name}: source and loader code are unavailable.')
    code = get_code(name)
    if not isinstance(code, CodeType):
        raise ValueError(f'Cannot fingerprint pipeline module {name}: loader did not provide a module code object.')
    return dict(kind='loader_code', sha256=semantic_code_digest(code))
