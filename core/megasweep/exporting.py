"""Non-overwriting app exports; identical CSV snapshots are reused."""
from dataclasses import dataclass
import filecmp
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading

from . import megasweep_analysis as analysis

_publish_lock = threading.Lock()


@dataclass(frozen=True)
class ExportResult:
    path: Path
    created: bool

    @property
    def message(self):
        return (f'Saved {self.path.suffix[1:].upper()}: {self.path}' if self.created
                else f'Unchanged CSV; skipped: {self.path}')


def export_file(path, writer):
    """Render first, then choose a version without overwriting an existing file."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.export-', suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        writer(str(temporary))
        with _publish_lock:
            pattern = re.compile(re.escape(path.stem) + r'_(\d{3,})' + re.escape(path.suffix), re.IGNORECASE)
            versions = [(1, path)] if path.exists() else []
            for candidate in path.parent.iterdir():
                match = pattern.fullmatch(candidate.name)
                if match and int(match[1]) >= 2:
                    versions.append((int(match[1]), candidate))
            versions.sort(key=lambda item: item[0])
            if path.suffix.lower() == '.csv':
                for _, candidate in versions:
                    if candidate.is_file() and filecmp.cmp(temporary, candidate, shallow=False):
                        return ExportResult(candidate, False)
            number = max((number for number, _ in versions), default=0) + 1
            while True:
                target = path if number == 1 else path.with_name(f'{path.stem}_{number:03d}{path.suffix}')
                try:
                    destination = target.open('xb')
                except FileExistsError:
                    number += 1
                    continue
                try:
                    with destination, temporary.open('rb') as source:
                        shutil.copyfileobj(source, destination)
                except BaseException:
                    target.unlink(missing_ok=True)
                    raise
                return ExportResult(target, True)
    finally:
        temporary.unlink(missing_ok=True)


def save_dataframe(frame, path):
    return export_file(path, lambda target: frame.to_csv(target, index=False))


def save_map_csv(x, y, z, path, **kwargs):
    return export_file(path, lambda target: analysis.save_map_csv(x, y, z, target, **kwargs))


def save_line_csv(line_cut, path):
    return export_file(path, lambda target: analysis.save_line_csv(line_cut, target))


def save_figure(figure, path, **kwargs):
    return export_file(path, lambda target: analysis.save_figure(figure, target, **kwargs))


def save_figure_with_axes_size(figure, path, axes_size, **kwargs):
    return export_file(path, lambda target: analysis.save_figure_with_axes_size(figure, target, axes_size, **kwargs))
