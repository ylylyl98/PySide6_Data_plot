# Compare save audit — 2026-09-11

## Scope and reproduction

This is a read-only performance audit. The live experiment was only read from
`D:\instrument_control_v3_1\YZ303\Initial Data`; no live output was written,
moved, or deleted. The two inputs were copied to temporary folders under
`artifacts/compare-save-audit-*`, then loaded and exported through the existing
Compare path using the project `.venv` (`Python 3.13.9`, `PySide6 6.11.2`,
`NumPy 2.4.2`, `Matplotlib 3.10.8`, `pandas 3.0.5`).

The exact group used by the current save was:

| input | size | loaded cube |
|---|---:|---:|
| `...0.975uW...RotOut45deg...csv` | 1,151,714 B | 201 × 1,340 |
| `...0.976uW...RotOut90deg...csv` | 1,145,338 B | 201 × 1,340 |

The export used the saved live parameters: linear scale, background `596`,
`turbo`, clipping enabled, `vmin=-11`, `vmax=10500.5845`, and VP enabled.
Each run produced three PNGs, three DAT files, and three metadata files. The
reproduction script and raw timing JSON are [profile_compare_save_audit.py](../../../artifacts/profile_compare_save_audit.py)
and [compare-save-audit-profile.json](../../../artifacts/compare-save-audit-profile.json).

## Measured worker path

Production `_start_export` creates a `Worker(self._export_task, ...)` at
`ui_qt/main_window.py:8007`, so the export body runs off the GUI thread. The
Compare branch calls `export_compare_panels` at `ui_qt/main_window.py:8435`.
Two complete reproductions gave 11.42 s on the first run and 8.18 s on the
second run; the first run includes normal Matplotlib cold-start variance.

| stage | run 1 | run 2 | calls/run | observed behavior |
|---|---:|---:|---:|---|
| `background_correct_cube` | 0.004 s | 0.003 s | 2 | negligible |
| `valley_polarization_cube` | 0.017 s | 0.013 s | 1 | negligible |
| PNG rendering + save | 5.10 s | 3.53 s | 3 | substantial, but not the largest stage |
| DAT formatting + write | 6.17 s | 4.41 s | 3 | largest combined stage; formatting and file I/O were not split |
| metadata + source descriptors | 0.10 s | 0.21 s | 3 + 4 | small; two unique ~1.15 MB SHA-256 reads, then cache reuse |
| total export worker wall time | 11.42 s | 8.18 s | — | sequential sum of three panels |

Output sizes were approximately 0.17 MB (KK PNG), 0.16 MB (KKp PNG), and
0.57 MB (VP PNG), versus 0.94 MB, 0.92 MB, and 3.41 MB for their DAT files.
The per-panel timings show the same pattern: VP PNG 1.09–2.58 s and VP DAT
0.96–2.32 s; KK/KKp DAT writes varied from 0.73–2.76 s despite similar file
sizes, so filesystem/OS variance contributes to wall time but does not change
the stage ranking.

The code explains the DAT cost: `core/export.py:1588-1615` builds every output
cell as Python strings and then joins the complete 3.4 MB VP table before one
`Path.write_text`. The measurement combines that formatting work with the
write, so it cannot by itself assign the time to disk versus CPU/string
construction. PNGs each rebuild a complete 8.0 × 6.2 inch, 150 DPI Matplotlib
figure (`core/export.py:327-349`, `:459`) and call `savefig` with
`bbox_inches="tight"`; the figure-build portion is nested inside the PNG
timing (2.89 s of run 1 and 1.91 s of run 2 across all three figures). The
Compare loop does this three times (`core/export.py:1637-1695`, `:1719-1760`).
There is no duplicate numerical processing: correction is two calls, VP
calculation is one call, and each panel is rendered/written once. The
`tight_layout` warning in the existing app log is from interactive plotting at
`main_window.py:7278`, not from this export renderer; the export path uses
`bbox_inches="tight"` and does not call `tight_layout`.

## GUI completion tail

After the worker result is delivered, Compare runs
`_cmp_refresh_history_cache(force=True)` and `_cmp_update_group_badge` at
`ui_qt/main_window.py:8523-8524`. On the live folder, the cache scan parsed
165 `*.metadata.json` records in 0.049 s, and the badge identity/history match
took 0.066 s (0.115 s combined). This is about 1–1.5% of the 8.18 s warm
export and cannot explain the perceived save delay. It is GUI-thread work and
can briefly delay repainting after the worker completes, but the dominant wait
is the worker's sequential DAT and PNG writes.

## Diagnosis

The measured bottleneck is serialized output generation, led by DAT table
formatting/writing and followed by three independent Matplotlib render/save
operations. Metadata SHA-256 work, automatic moving, and Compare history
refresh are not root causes for this group. The current code sets
`files_to_move` but returns `moved=0`; no automatic source move occurred in
this path. Any optimization should therefore start with a measured DAT writer
and/or reducing repeated PNG figure construction, while preserving the current
default output format and numerical data.
