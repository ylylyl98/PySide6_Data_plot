"""Streaming workbook support for application analysis exports (no GUI work)."""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile


@dataclass
class AnalysisTable:
    name: str
    headers: list
    rows: object


def cell_value(value):
    """Keep unavailable numerics blank and source strings literal, not formulas."""
    if value is None:
        return None
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def analysis_filename(source, suffix, directory, *, fallback='analysis'):
    """Preserve experimental conditions; only shorten when the path requires it."""
    stem = Path(str(source)).stem or fallback
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', stem).strip(' .') or fallback
    if stem.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *[f'{s}{i}' for s in ('COM', 'LPT') for i in range(1, 10)]}:
        stem = '_' + stem
    tail = suffix + '.xlsx'
    available = min(180, 240 - len(str(Path(directory).resolve())) - 1) - len(tail)
    if available < 18:
        raise ValueError('Export folder is too deep. Select a shorter destination path.')
    if len(stem) > available:
        stem = stem[:available-9] + '_' + sha256(stem.encode('utf8')).hexdigest()[:8]
    return stem + tail


def write_analysis_workbook(path, tables, *, cancelled=None):
    """Publish only a complete workbook; never silently truncate Excel limits."""
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from core.load_cancellation import check_load_cancelled

    def check():
        check_load_cancelled()
        if cancelled and cancelled():
            raise InterruptedError('Analysis export cancelled.')

    path = Path(path)
    check()
    book = Workbook(write_only=True)
    book.properties.description = 'Numeric analysis results. Blank cells mean unavailable; see Settings for definitions.'
    temporary = None
    try:
        for table in tables:
            check()
            count = len(table.headers)
            if not 0 < count <= 16384:
                raise ValueError(f'{table.name}: too many columns for Excel; export fewer branches or use CSV.')
            sheet = book.create_sheet(table.name)
            sheet.freeze_panes = 'B2' if table.name not in {'Settings', 'Sources'} else 'A2'
            sheet.sheet_view.showGridLines = False
            sheet.row_dimensions[1].height = 44
            for index, name in enumerate(table.headers, 1):
                sheet.column_dimensions[get_column_letter(index)].width = min(34, max(17, len(str(name)) + 2))
            if table.name in {'Settings', 'Sources'}:
                sheet.column_dimensions['A'].width = 32
                sheet.column_dimensions['B'].width = 100
            header = []
            for name in table.headers:
                c = WriteOnlyCell(sheet, str(name)); c.data_type = 's'
                c.font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
                c.fill = PatternFill('solid', fgColor='215E8A')
                c.alignment = Alignment(wrap_text=True, vertical='center')
                header.append(c)
            sheet.append(header)
            n = 1
            for row in table.rows:
                n += 1
                if n > 1048576:
                    raise ValueError(f'{table.name}: exceeds Excel row limit; export fewer spectra or use CSV.')
                if n % 256 == 0:
                    check()
                if len(row) != count:
                    raise ValueError(f'{table.name}: inconsistent data columns at row {n}.')
                output = []
                for raw in row:
                    value = cell_value(raw)
                    if isinstance(value, str) and len(value) > 32767:
                        raise ValueError(f'{table.name}: text exceeds Excel cell limit; see JSON or reduce export scope.')
                    c = WriteOnlyCell(sheet, value)
                    if isinstance(value, str):
                        c.data_type = 's'
                    elif isinstance(value, float):
                        c.number_format = ('0.000E+00' if value != 0 and (abs(value) < 1e-5 or abs(value) >= 1e9)
                                           else '0.#########')
                    if table.name in {'Settings', 'Sources'}:
                        c.alignment = Alignment(wrap_text=True, vertical='top')
                    output.append(c)
                sheet.append(output)
            sheet.auto_filter.ref = f'A1:{get_column_letter(count)}{n}'
        check()
        with NamedTemporaryFile(prefix='.analysis-', suffix='.xlsx', dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        book.save(temporary)
        check()
        os.replace(temporary, path)
        return path
    finally:
        # Workbook.close() only closes read-only archives. Streaming worksheets
        # must be closed explicitly when cancellation precedes save(), otherwise
        # their XML generators keep Windows temporary files locked.
        for sheet in book.worksheets:
            if not sheet.closed:
                sheet.close()
            writer = getattr(sheet, '_writer', None)
            if writer is not None and isinstance(writer.out, str) and os.path.exists(writer.out):
                writer.cleanup()
        book.close()
        if temporary is not None and temporary.exists():
            temporary.unlink()
