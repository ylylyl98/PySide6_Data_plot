"""Export each selected optical window's complete MCD(B) curve and an index."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping, Sequence

import pandas as pd

from core.mcd_extract import BRANCHES, McdBranch, ProcessedMcdRecord, load_branch_traces


def export_mcd_curve_csvs(
    records: Sequence[ProcessedMcdRecord],
    output_dir: str | Path,
    *,
    branches: Sequence[McdBranch] = BRANCHES,
    energy_groups: Mapping[str, str] | None = None,
    temperature_mode: str = 'setpoint',
) -> Path:
    """Write independent XY pairs per record; never merge, refit or clip curves.

    Callers supply the included records, and for CW also apply the chosen group
    and temperature interval. The index retains the full conditions and source
    paths, so repeated temperatures/windows remain identifiable. Its CSV paths
    are relative to the index. Read/write failures propagate to the export UI.
    """
    if not records or not branches:
        raise ValueError('Select records and at least one B-sweep branch to export.')
    if temperature_mode not in ('setpoint', 'measured'):
        raise ValueError('Unknown temperature source for MCD curve export.')
    if any(branch not in BRANCHES for branch in branches):
        raise ValueError('Unknown MCD B-sweep branch.')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    groups = energy_groups or {}
    group_names = list(dict.fromkeys(groups.get(r.record_id) or 'Unassigned' for r in records))
    # The numeric prefix keeps user group names distinct even after sanitizing.
    folders = {name: f'{i:02d}_' + (re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('._-')[:48] or 'Group')
               for i, name in enumerate(group_names, 1)}
    index_rows = []
    for number, record in enumerate(records, 1):
        group = groups.get(record.record_id) or 'Unassigned'
        catalog_t = record.condition_value('T')
        temperature = record.temperature_measured_k if temperature_mode == 'measured' else catalog_t
        temperature_source = ('measured' if temperature_mode == 'measured'
                              else record.condition_sources.get('T', 'catalog'))
        temperature_label = f'{temperature:g}' if temperature is not None else 'unknown'
        relative = Path(folders[group]) / f'T{temperature_label}K_E{record.center_ev:.6f}eV_{number:04d}.csv'
        try:
            traces = load_branch_traces(record, branches, primary_only=True)
        except KeyError as exc:
            raise ValueError(f'{record.trace_path.name}: cannot read the selected MCD window: {exc}') from exc
        columns = {}
        counts = {}
        for branch in branches:
            suffix = 'increasing' if branch == 'B increasing' else 'decreasing'
            block = traces.loc[traces['branch'] == branch]
            # Series reset their indices so unequal native grids pad with empty
            # cells, without pairing measurements by equal field or sorting B.
            columns[f'B_{suffix}_T'] = block['B_T'].reset_index(drop=True)
            columns[f'MCD_{suffix}'] = block['corrected_signed_mean'].reset_index(drop=True)
            counts[f'{suffix}_points'] = len(block)
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns).to_csv(path, index=False, encoding='utf-8-sig')
        index_rows.append(dict(
            csv_file=relative.as_posix(), record_id=record.record_id, energy_group=group,
            temperature_K=temperature, temperature_source=temperature_source,
            temperature_catalog_K=catalog_t, temperature_setpoint_K=record.temperature_setpoint_k,
            temperature_measured_K=record.temperature_measured_k,
            doping_V=record.condition_value('Doping'), efield_V=record.condition_value('E-field'),
            vbias_V=record.condition_value('Vbias'), energy_ev=record.center_ev,
            width_mev=record.width_mev, saved_fit_window_T=record.fit_window_t,
            source_file=record.source_file, trace_path=str(record.trace_path),
            settings_path=str(record.settings_path), **counts,
        ))
    index = out / 'index.csv'
    pd.DataFrame(index_rows).to_csv(index, index=False, encoding='utf-8-sig')
    return index
