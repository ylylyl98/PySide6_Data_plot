# Gate/cursor response audit — 2026-09-11

## Scope and method

This was a diagnostic-only run. `scripts/profile_gate_response.py` constructs
the production `ui_qt.main_window.MainWindow`, injects deterministic synthetic
`DataCube`/`McdResult` arrays, drives the real controller slots, and waits for
the Qt Matplotlib canvas `draw_event` (or `canvas.blit` for normal MCD cursor
updates). It measures already-loaded synthetic data only: source matching,
file I/O, and worker loading are deliberately excluded by the harness. It
does not write application settings. The harness uses `QT_QPA_PLATFORM=offscreen`,
Fusion style, Qt 6.10.2, Matplotlib 3.10.7, NumPy 2.3.4, Python 3.13.9 on
Windows 11; these values are useful for comparison but are not user-window
latencies.

Each measured case uses eight distinct control values. `single` processes the
event loop after each input. `rapid` sends all eight values before pumping the
event loop and reports one coalesced final-draw observation; it is not a
statistical p50/p95 sample. Arrays are 64x128 and 256x512. p50/p95/max are
milliseconds from the first control input in that case to the next real canvas
completion. Compare/DRR dispatch delay is measured separately from input to
`_plot_mode`; PL/Power gate paths update existing artists directly, so
dispatch is N/A. DRR latency already includes its approximately 90 ms debounce;
the separate-wait column shows that component rather than adding it again.

## Results

| Page | Control and path | Scenario | Latency (p50 / p95 / max ms) | Separate wait | Main bottleneck | Priority |
|---|---|---|---:|---:|---|---|
| PL | `pl_spins[gate]` → `_on_pl_gate_changed` → existing spectrum/gate artists | 64x128 single / rapid observation | 129 / 209 / 231; 195 obs. | N/A (direct) | Full Matplotlib draw; no axes rebuild | P2 |
| PL | same | 256x512 single / rapid observation | 161 / 252 / 264; 233 obs. | N/A (direct) | draw cost/event backlog in burst | P2 |
| DRR | `drr_spins[gate]` → 90 ms `_schedule_plot_redraw` → DRR gate-only update | 64x128 single / rapid observation | 221 / 326 / 346; 305 obs. | 90 / 97 / 98 | debounce plus canvas draw | P1 |
| DRR | same | 256x512 single / rapid observation | 213 / 279 / 311; 357 obs. | 86 / 96 / 97 | debounce plus draw/event backlog | P1 |
| Compare | `cmp_spins[gate]` → 90 ms redraw → corrected cubes + linecut + axes | 64x128 single / rapid observation | 355 / 508 / 524; 348 obs. | 87 / 94 / 100 | Full compare replot, `tight_layout`, repeated correction | P0 |
| Compare | same | 256x512 single / rapid observation | 465 / 639 / 654; 504 obs. | 97 / 104 / 104 | Full axes reconstruction and draw | P0 |
| Power | `power_spins[gate]` → `_update_power_compare_spectrum_and_lines` | 64x128 single / rapid observation | 130 / 215 / 236; 190 obs. | N/A (direct) | Spectrum/line update then canvas draw | P1 |
| Power | same | 256x512 single / rapid observation | 158 / 256 / 267; 254 obs. | N/A (direct) | Larger line arrays and draw | P1 |
| MCD | `mcd_window_center_spin` → 40 ms center-refresh timer → trace refresh → `canvas.blit` | 64x128 single / rapid observation | 86 / 90 / 91; 85 obs. | 40 ms timer (static call-chain) | trace recomputation + normal blit | P1 |
| MCD Peak Shift | B-field combo / canvas B cursor and feature cursor | static only | N/A | N/A | Requires analyzed/local-fit result and selected track; no valid fixture was available without source-backed analysis | P1 (measure next) |
| SHG | wavelength integration gate and angle cursor | static only | N/A | N/A | `editingFinished`/angle changes request source-backed worker reprocess; no source read was permitted | P1 (measure next) |
| Slides | no gate/field/cursor control | N/A | N/A | N/A | — | — |
| Tools | no gate/field/cursor control | N/A | N/A | N/A | — | — |

The measured values are synthetic-data/offscreen evidence. They must not be
described as real experimental-data performance. Because the harness replaces
`_ensure_loaded_matches_ui_params` to prevent file-backed reloads, these
measurements say nothing about source consistency checks or production file
I/O. Compare is the clear first
optimization target because a gate-only edit still rebuilds corrected panels,
axes, and layout. DRR has a low-cost gate-only update after its debounce, but
the current 90 ms wait is visible. PL and Power preserve their existing axes
and have the lowest steady single-input latency among the measured pages.

## Static call-chain evidence

- Shared debounce/dispatch: `MainWindow._schedule_plot_redraw` and
  `_run_scheduled_plot_redraw` in `ui_qt/main_window.py:1948-1970`.
- PL gate signal and artist update: `ui_qt/main_window.py:2255`,
  `ui_qt/controllers_pl.py:567-570`, `:751-773`.
- DRR gate enters the redraw scheduler in
  `ui_qt/controllers_drr.py:388-430`; the DRR gate-only path and line update
  are in `ui_qt/main_window.py:6657-6668` and
  `ui_qt/controllers_drr.py:1546-1568`.
- Compare gate is wired with the other plot spins at
  `ui_qt/main_window.py:2290-2292`; its handler always schedules a redraw in
  `ui_qt/controllers_compare.py:1039-1069`, while the main plot branch
  reconstructs compare axes at `ui_qt/main_window.py:7150-7277`.
- Power gate is wired at `ui_qt/main_window.py:2309-2310`; the handler takes
  the direct spectrum/line path at `ui_qt/controllers_power.py:767-781`, with
  rendering in `:676-749`.
- MCD's energy cursor is `mcd_window_center_spin` at
  `ui_qt/feature_pages.py:1246-1290`, wired at
  `ui_qt/main_window.py:2439`; center-only changes are coalesced by the
  controller's 40 ms timer in `ui_qt/controllers_mcd.py:223-257`.
- MCD Peak Shift field/feature drag starts and updates through
  `ui_qt/main_window.py:6461-6524`; its field selector is created at
  `ui_qt/feature_pages.py:1364-1375`.
- SHG controls are created at `ui_qt/feature_pages.py:2646-2676` and
  `:2735-2746`; angle changes are wired at
  `ui_qt/main_window.py:2339`, while integration edits are committed through
  `:2341-2349` and request `_request_shg_reprocess` at
  `ui_qt/controllers_shg.py:380-415`.

## Reproduction

From the repository root:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe scripts/profile_gate_response.py `
  2>&1 | Tee-Object artifacts\gate-audit-run.txt
```

The script creates temporary QSettings and Matplotlib cache directories via
the existing preview harness. The warning about `tight_layout` comes from the
production Compare plot and was retained as evidence rather than suppressed.
