"""Validated local path hints for PowerPoint figure lookup across restarts."""
from collections import OrderedDict
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from core.pptx_figure_index import FigureIndex, _fingerprint


class PictureLookup:
    def __init__(self, root, cache_dir=None):
        self.root = Path(root).resolve()
        self.index = FigureIndex(self.root)
        base = Path(cache_dir) if cache_dir is not None else (
            Path(os.environ['LOCALAPPDATA']) / 'DPTK' / 'PowerPoint' if os.environ.get('LOCALAPPDATA') else None)
        self.cache_path = base / (sha256(str(self.root).encode('utf-8')).hexdigest() + '.json') if base else None
        self._hints = OrderedDict()
        if self.cache_path:
            try:
                data = json.loads(self.cache_path.read_text(encoding='utf-8'))
                if isinstance(data, dict) and data.get('version') == 1 and isinstance(data.get('records'), dict):
                    for key, value in list(data['records'].items())[-1000:]:
                        if self._valid_id(key) and isinstance(value, str) and self._hint_path(value) is not None:
                            self._hints[key] = value
            except (OSError, ValueError, TypeError):
                pass

    @staticmethod
    def _valid_id(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

    def _hint_path(self, relative):
        try:
            name = Path(relative)
            if name.is_absolute():
                return None
            path = (self.root / name).resolve()
            if not path.is_relative_to((self.root / 'Processed Data').resolve()):
                return None
            return path
        except (OSError, ValueError, TypeError):
            return None

    def _remember(self, records):
        changed = False
        for record in records:
            try:
                path = record.png_path.resolve()
                if not path.is_relative_to((self.root / 'Processed Data').resolve()):
                    continue
                relative = str(path.relative_to(self.root))
            except (OSError, ValueError):
                continue
            if self._hints.get(record.record_id) != relative:
                self._hints[record.record_id] = relative
                changed = True
            self._hints.move_to_end(record.record_id)
        while len(self._hints) > 1000:
            self._hints.popitem(last=False)
            changed = True
        if not changed or self.cache_path is None:
            return
        temporary = None
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile('w', encoding='utf-8', dir=self.cache_path.parent,
                                    prefix='.dptk-', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({'version': 1, 'records': dict(self._hints)}, stream, ensure_ascii=False)
            os.replace(temporary, self.cache_path)
        except OSError:
            pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def find_png(self, path):
        records = self.index.find_png(path)
        self._remember(records)
        return records

    def prepare(self, cancelled=lambda: False):
        """Warm cheap search data on the index worker, without decoding images."""
        for candidate in self.index._candidates():
            if cancelled():
                return
            try:
                self.index._dimensions(candidate[0])
                self.index._chunk_lengths(candidate[0])
            except (OSError, ValueError):
                continue

    def match(self, data, tag=None):
        pixels = _fingerprint(data)
        tag = str(tag or '').lower()
        if self._valid_id(tag):
            hint = self._hints.get(tag)
            path = self._hint_path(hint) if hint else None
            if path is not None:
                for record in self.index.find_png(path):
                    actual_id = sha256((str(record.png_path.resolve()) + '\0' + pixels).encode('utf-8')).hexdigest()
                    if record.record_id == tag == actual_id:
                        self._remember([record])
                        return [record], pixels
        records = self.index.lookup_image(data, record_id=tag or None, target_pixels=pixels)
        if tag:
            records = [record for record in records if record.record_id.lower() == tag]
        self._remember(records)
        return records, pixels
