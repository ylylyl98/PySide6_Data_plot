"""Exact, portable lookup of exported figures and embedded PowerPoint images."""
from dataclasses import dataclass
from collections import OrderedDict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import struct
from zipfile import ZipFile
from xml.etree import ElementTree as ET

from PIL import Image


@dataclass(frozen=True)
class FigureRecord:
    png_path: Path
    dat_path: Path
    workflow: str
    role: str
    plot: dict
    snapshot: dict
    record_id: str
    metadata_path: Path | None = None


def _fingerprint(data):
    with Image.open(BytesIO(data)) as image:
        rgba = image.convert('RGBA')
        return sha256(f'{rgba.width}x{rgba.height}:'.encode() + rgba.tobytes()).hexdigest()


def _png_content_fingerprint(data):
    """Hash PNG rendering chunks; ancillary text/time metadata is ignored.

    IDAT chunks are concatenated so chunk boundaries do not change the hash.
    This is a fast exact-encoding prefilter, followed by RGBA verification.
    """
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    chunks, idat, offset, ended = [], bytearray(), 8, False
    while offset + 12 <= len(data):
        size = struct.unpack_from('>I', data, offset)[0]
        end = offset + 12 + size
        if end > len(data):
            return None
        kind, payload = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + size]
        if kind == b'IDAT':
            idat.extend(payload)
        elif kind in (b'IHDR', b'PLTE', b'tRNS'):
            chunks.append((kind, payload))
        elif kind == b'IEND':
            ended = True
            break
        # Unknown critical chunks could affect decoding; fall back to pixels.
        elif kind and not (kind[0] & 32):
            return None
        offset = end
    if not ended or not idat or not chunks or chunks[0][0] != b'IHDR':
        return None
    digest = sha256()
    for kind, payload in chunks + [(b'IDAT', idat)]:
        digest.update(kind)
        digest.update(struct.pack('>I', len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _png_chunk_lengths(stream, size):
    """Cheap necessary condition for identical encoded PNG rendering chunks.

    Ancillary chunks are ignored and IDAT payload lengths are summed, so
    splitting or joining IDAT chunks cannot exclude an exact encoding match.
    Invalid or unknown critical structure yields None (never an exclusion).
    """
    if size < 20 or stream.read(8) != b"\x89PNG\r\n\x1a\n":
        return None
    pieces, total, offset, count = [], 0, 8, 0
    while offset + 12 <= size and count < 100000:
        header = stream.read(8)
        if len(header) != 8:
            return None
        length, kind = struct.unpack('>I4s', header)
        end = offset + 12 + length
        if end > size:
            return None
        if kind == b'IDAT':
            total += length
        elif kind in (b'IHDR', b'PLTE', b'tRNS'):
            pieces.append((kind, length))
        elif kind == b'IEND':
            if length or not total or not pieces or pieces[0] != (b'IHDR', 13):
                return None
            return tuple(pieces), total
        elif not kind or not (kind[0] & 32):
            return None
        stream.seek(length + 4, 1)  # payload and CRC, without reading either
        offset = end
        count += 1
    return None


def _file_key(path, stat):
    return (os.path.normcase(os.path.abspath(path)), stat.st_dev, stat.st_ino,
            stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _role(name, workflow):
    stem = Path(name).stem.lower()
    wide = bool(re.search(r'_\d+(?:\.\d+)?x$', stem))
    stem = re.sub(r'_\d+(?:\.\d+)?x$', '', stem)
    if workflow == 'pl':
        role = next((x for x in ('linear', 'log') if stem.endswith('_' + x)), '')
    else:
        role = next((x for x in ('second', 'raw') if re.search(r'(?:^|_)' + x + r'(?:_|$)', stem)), 'raw')
    return role + ('_wide' if wide and role else '')


class FigureIndex:
    def __init__(self, root):
        self.root = Path(root)
        self._records = {}
        self._cache = {}
        self._structural_cache = {}
        self._size_cache = {}
        self._length_cache = {}
        self._metadata_cache = OrderedDict()

    def _hash(self, path):
        stat = path.stat()
        key = _file_key(path, stat)
        if key not in self._cache:
            self._cache[key] = _fingerprint(path.read_bytes())
        return self._cache[key]

    def _add(self, png, dat, workflow, role, plot, snapshot, metadata_path=None):
        png, dat = Path(png), Path(dat)
        if workflow not in ('pl', 'drr') or not role or not png.is_file() or not dat.is_file():
            return None
        # Sidecars and output symlinks must remain inside this experiment's
        # Processed Data tree, even when a metadata filename looks local.
        allowed = (self.root / 'Processed Data').resolve()
        if not png.resolve().is_relative_to(allowed) or not dat.resolve().is_relative_to(allowed):
            return None
        try:
            digest = self._hash(png)
        except (OSError, ValueError):
            return None
        rid = sha256((str(png.resolve()) + '\0' + digest).encode('utf-8')).hexdigest()
        record = FigureRecord(png.resolve(), dat.resolve(), workflow, role, dict(plot or {}), dict(snapshot or {}), rid, Path(metadata_path).resolve() if metadata_path is not None else None)
        self._records[rid] = record
        return record

    def _metadata(self, path):
        stat = path.stat()
        key = os.path.normcase(os.path.abspath(path))
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._metadata_cache.get(key)
        if cached is not None and cached[0] == signature:
            self._metadata_cache.move_to_end(key)
            return cached[1]
        payload = json.loads(path.read_text(encoding='utf-8'))
        after = path.stat()
        if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('Metadata changed while reading')
        self._metadata_cache[key] = (signature, payload)
        self._metadata_cache.move_to_end(key)
        if len(self._metadata_cache) > 4096:
            self._metadata_cache.popitem(last=False)
        return payload

    def _dimensions(self, path):
        stat = path.stat()
        key = _file_key(path, stat)
        if key not in self._size_cache:
            with Image.open(path) as image:
                self._size_cache[key] = image.size
        return self._size_cache[key]

    def _candidates(self, png_path=None, *, metadata_path=None):
        directory = self.root / 'Processed Data'
        if not directory.is_dir():
            return
        allowed = directory.resolve()
        target = Path(png_path).resolve() if png_path is not None else None
        if target is not None and not target.is_relative_to(allowed):
            return
        if metadata_path is not None:
            metadata = Path(metadata_path).resolve()
            # A completed export knows its exact sidecar. Keep the same
            # manifest/content validation without inspecting unrelated history.
            if target is None or metadata.parent != target.parent or not metadata.name.endswith('.metadata.json'):
                return
            metadata_files = (metadata,)
        else:
            # Unknown/legacy PNGs still need the full sibling-sidecar search.
            metadata_files = (target.parent.glob('*.metadata.json') if target is not None
                              else directory.rglob('*.metadata.json'))
        for metadata in metadata_files:
            try:
                if not metadata.resolve().is_relative_to(allowed):
                    continue
                payload = self._metadata(metadata)
                if payload.get('display_save_pending') or payload.get('pl_display_save_pending'):
                    continue
                workflow = str(payload.get('operation', payload.get('workflow', ''))).lower()
                workflow = 'drr' if workflow in ('dr/r', 'drr') else workflow
                if workflow not in ('pl', 'drr'):
                    continue
                figures = payload.get('figures') or {}
                if not isinstance(figures, dict):
                    continue
                outputs = payload.get('outputs') or []
                manifest = payload.get('output_manifest') or []
                data_names = {str(x.get('filename', '')) for x in manifest if isinstance(x, dict) and x.get('role') == 'data'}
                data_names.update(str(x) for x in outputs if isinstance(x, str) and x.lower().endswith('.dat'))
                # Older PL exports carried a linear/log plot bank and an
                # explicit output list but no per-figure mapping. Accept only
                # the exact names listed by that metadata and one data file.
                if workflow == 'pl' and not figures:
                    bank = payload.get('plot')
                    if isinstance(bank, dict) and len(data_names) == 1:
                        for name in outputs:
                            if not isinstance(name, str) or Path(name).name != name:
                                continue
                            role = _role(name, workflow)
                            base_role = role.removesuffix('_wide')
                            if role and isinstance(bank.get(base_role), dict):
                                figures[name] = {'data_file': next(iter(data_names)), 'plot': bank[base_role]}
                for name, figure in figures.items():
                    if target is not None and name != target.name:
                        continue
                    if not isinstance(name, str) or Path(name).name != name or not name.lower().endswith('.png') or not isinstance(figure, dict):
                        continue
                    data = figure.get('data_file')
                    if not isinstance(data, str) or Path(data).name != data or data not in data_names:
                        continue
                    plot = figure.get('plot')
                    if not isinstance(plot, dict):
                        continue
                    snapshot = payload.get('pl_display' if workflow == 'pl' else 'ui_state') or {}
                    if not isinstance(snapshot, dict):
                        continue
                    explicit_role = figure.get('role')
                    valid_roles = ({'linear', 'log', 'linear_wide', 'log_wide'} if workflow == 'pl'
                                   else {'raw', 'second', 'raw_wide', 'second_wide'})
                    if explicit_role is not None and explicit_role not in valid_roles:
                        continue
                    role = explicit_role or _role(name, workflow)
                    if workflow == 'drr' and explicit_role is None:
                        processing = payload.get('processing')
                        if not isinstance(processing, dict) or 'derivative_order' not in processing:
                            continue
                        derivative = processing['derivative_order']
                        if derivative not in (None, 0, 1, 2):
                            continue
                        role = ('second' if derivative == 2 else 'raw') + ('_wide' if role.endswith('_wide') else '')
                    png, dat = metadata.parent / name, metadata.parent / data
                    if not role or not png.is_file() or not dat.is_file():
                        continue
                    if not png.resolve().is_relative_to(allowed) or not dat.resolve().is_relative_to(allowed):
                        continue
                    yield (png, dat, workflow, role, plot, snapshot, metadata)
            except (OSError, ValueError, TypeError, AttributeError):
                continue

    def scan(self):
        self._records.clear()
        for candidate in self._candidates():
            self._add(*candidate)
        return list(self._records.values())

    def _structural_hash(self, path):
        stat = path.stat()
        key = _file_key(path, stat)
        if key not in self._structural_cache:
            self._structural_cache[key] = _png_content_fingerprint(path.read_bytes())
        return self._structural_cache[key]

    def _chunk_lengths(self, path):
        stat = path.stat()
        key = _file_key(path, stat)
        if key not in self._length_cache:
            with path.open('rb') as stream:
                self._length_cache[key] = _png_chunk_lengths(stream, stat.st_size)
        return self._length_cache[key]

    def lookup_image(self, image_bytes, record_id=None, *, target_pixels=None):
        """Find matching exports, preferring identical PNG content encoding.

        Exact content matches are RGBA-verified and all duplicates returned.
        If none exist, compare pixels against every candidate. A recompressed
        pixel-identical export is omitted when an encoding match exists. An
        existing record_id is checked against current metadata and RGBA pixels.
        """
        if record_id:
            known = self._records.get(str(record_id))
            if known is not None:
                current = self.find_png(known.png_path)
                target = target_pixels if target_pixels is not None else _fingerprint(image_bytes)
                verified = [record for record in current if record.record_id == str(record_id)
                            and self._hash(record.png_path) == target]
                if verified:
                    return verified
        with Image.open(BytesIO(image_bytes)) as target_image:
            target_size = target_image.size
        candidates = []
        for candidate in self._candidates():
            try:
                if self._dimensions(candidate[0]) == target_size:
                    candidates.append(candidate)
            except (OSError, ValueError):
                continue
        structural = _png_content_fingerprint(image_bytes)
        lengths = _png_chunk_lengths(BytesIO(image_bytes), len(image_bytes))
        target_pixels = target_pixels if target_pixels is not None else _fingerprint(image_bytes)
        matched = []
        if structural:
            for candidate in candidates:
                try:
                    candidate_lengths = self._chunk_lengths(candidate[0]) if lengths is not None else None
                    if candidate_lengths is not None and lengths is not None and candidate_lengths != lengths:
                        continue
                    if self._structural_hash(candidate[0]) == structural and self._hash(candidate[0]) == target_pixels:
                        matched.append(candidate)
                except (OSError, ValueError):
                    continue
        if not matched:
            for candidate in candidates:
                try:
                    if self._hash(candidate[0]) == target_pixels:
                        matched.append(candidate)
                except (OSError, ValueError):
                    continue
        return self._add_unique(matched)

    def _add_unique(self, candidates):
        # Match scan(): the last valid sidecar wins for a repeated record ID.
        records = {}
        for candidate in candidates:
            record = self._add(*candidate)
            if record is not None:
                records[record.record_id] = record
        return list(records.values())

    def find_png(self, path, *, metadata_path=None):
        """Validate one PNG, optionally against its known export sidecar only."""
        target = Path(path).resolve()
        return self._add_unique(item for item in self._candidates(target, metadata_path=metadata_path)
                                if item[0].resolve() == target)

    def register_exports(self, paths, workflow, snapshot):
        """Register confirmed output paths; prefer their freshly written metadata."""
        wanted = {Path(value).resolve() for value in paths.values() if str(value).lower().endswith('.png')}
        candidates = (item for path in wanted for item in self._candidates(path)
                      if item[0].resolve() == path and item[2] == workflow.lower())
        return self._add_unique(candidates)

    def get(self, record_id):
        return self._records.get(record_id)

    def match_image(self, image_bytes):
        digest = _fingerprint(image_bytes)
        result = []
        for record in self._records.values():
            try:
                current = self._hash(record.png_path)
                current_id = sha256((str(record.png_path.resolve()) + '\0' + current).encode('utf-8')).hexdigest()
                if current == digest and current_id == record.record_id:
                    result.append(record)
            except (OSError, ValueError):
                pass
        return result


_P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
_R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
_REL = 'http://schemas.openxmlformats.org/package/2006/relationships'


def _target(base, target):
    if not target or '\\' in target or target.startswith('/') or ':' in target:
        raise ValueError('Unsafe or external relationship target')
    path = posixpath.normpath(posixpath.join(posixpath.dirname(base), target))
    if path == '..' or path.startswith('../') or path.startswith('/'):
        raise ValueError('Relationship escapes archive')
    return path


def _relationships(archive, part):
    path = PurePosixPath(part)
    relpath = str(path.parent / '_rels' / (path.name + '.rels'))
    tree = ET.fromstring(archive.read(relpath))
    return {item.attrib['Id']: item for item in tree.findall(f'{{{_REL}}}Relationship')}


def _resolve_rel(archive, part, rid):
    rel = _relationships(archive, part).get(rid)
    if rel is None or rel.get('TargetMode', '').lower() == 'external':
        raise ValueError('Missing or external relationship')
    return _target(part, rel.get('Target', ''))


def read_shape_image(pptx_path, slide_id, shape_id):
    """Read an embedded picture by PowerPoint slide ID and shape ID (including groups)."""
    with ZipFile(pptx_path) as archive:
        presentation = 'ppt/presentation.xml'
        tree = ET.fromstring(archive.read(presentation))
        slide = next((x for x in tree.findall(f'.//{{{_P}}}sldId') if x.get('id') == str(slide_id)), None)
        if slide is None:
            raise ValueError('Slide ID not found')
        part = _resolve_rel(archive, presentation, slide.get(f'{{{_R}}}id'))
        root = ET.fromstring(archive.read(part))
        for pic in root.iter(f'{{{_P}}}pic'):
            props = pic.find(f'./{{{_P}}}nvPicPr/{{{_P}}}cNvPr')
            if props is not None and props.get('id') == str(shape_id):
                blip = pic.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
                rid = blip.get(f'{{{_R}}}embed') if blip is not None else None
                if not rid:
                    raise ValueError('Picture is not embedded')
                return archive.read(_resolve_rel(archive, part, rid))
        raise ValueError('Picture shape ID not found')
