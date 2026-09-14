"""Qt-free worker adapter for MCD Organizer exports."""

from __future__ import annotations

from core.mcd_extract import export_mcd_extract


def mcd_extract_export_worker(records, folder, *, progress=None, log=None, **options):
    result = export_mcd_extract(records, folder, **options)
    if progress is not None:
        progress.emit(100)
    return result
