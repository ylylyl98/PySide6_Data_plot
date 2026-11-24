# Megasweep

[Project home](../../README.md) · [简体中文](../zh-CN/megasweep.md)

Megasweep is a native Qt Quick workspace for two-coordinate optical sweeps and
scalar transport measurements. Open **Megasweep** in the workflow rail, then
choose **PL**, **Reflection**, or **Transport**. Each mode retains its own source,
backgrounds, parameters and display settings.

## Select data

Use the application's **Folder…** and **File…** actions, or **Open file…** in the
workspace. The shared source picker supports search, optical/transport candidate
filters and external files. Classification uses headers only; actual compatibility
is checked when loading. Generated `Processed Data/` and `*_outputs/` directories
are excluded from the measurement inventory.

By default, the picker lists compatible files in the current mode's remembered
folder and reveals its selected file. PL, Reflection and Transport separately
remember their browsing folder, last file and search scope across restarts.
Remembered files are not automatically loaded or analyzed. Reopening clears old
search text so it cannot hide the selected file.

| Search scope | Files shown |
| --- | --- |
| Current folder (default) | CSVs in the browsing folder |
| Include subfolders | The browsing folder and its subfolders |
| Entire experiment | The experiment root and its subfolders |

Choosing a new Folder prioritizes that folder without clearing loaded data.
Selecting a file moves the next File picker to its parent directory. For an
external file outside the current experiment, that parent becomes the new
experiment root.

Changing the browsing folder does not move a loaded measurement's output folder.
Switching modes or workflows does not replace another mode's selected CSV.
Use Reload after acquisition changes a file. Parameter changes operate on the
loaded snapshot rather than continuously rereading a growing CSV.

Optical CSVs have two sweep-coordinate columns and numeric wavelength headers in
nm. Transport CSVs have scalar numeric columns; supported unit rows and acquisition
metadata are retained. Select the scan axes and signal channel explicitly when
the available columns are ambiguous.

## Analyze

- **PL:** integrated intensity with automatic or manual baseline, peak energy,
  original or transformed doping/electric-field coordinates, and available bias
  current/resistance maps.
- **Reflection:** background-normalized spectra, peak-to-peak, fixed-energy and
  peak-position maps. The reference preview compares the raw signal, selected
  background and normalized spectrum at a measured coordinate.
- **Transport:** a scalar map, one-dimensional curve or fixed-coordinate cut,
  with available scan-direction and pass filters.
- **Spectral slices:** choose original or transformed coordinates, a fixed axis
  and measured value; step through coordinates and optionally calculate the
  second derivative on the measured energy grid. Batch export supports one or
  both fixed-axis series.

Reflection peak-to-peak and peak-position maps start with **Auto Energy Window**
enabled and use the full measured energy range. Editing either bound saves a
manual window for that scan. Enable Auto Energy Window to restore the full range.
PL retains its original initial window. Automatic background scaling uses separate
reference windows that must contain usable measured channels.

### Reflection workflow

1. Use **Open scan…** below the mode tabs in the left sidebar. Check the resolved X and Y columns, then use **Background** to select a reference with matching wavelength channels.
2. In **Analysis map**, select **Map quantity** to switch between peak-to-peak RC, RC at one energy, peak position or an available electrical signal. The automatic target for fixed-energy RC is the measured channel nearest the midpoint of the energy range. Edit it to use a specific energy. **Plot limits** above the plot groups the X, Y and color bounds. Type a bound and press Enter or leave the field to update that range. Each axis direction button immediately reverses that axis. Automatic map colors use the 2nd and 98th percentiles for at least 50 finite values and otherwise use the extrema, including when the percentiles coincide. Choose **RdBu_r** (default) or **RdBu** in the Color group. Turning on **Color → Auto** updates the color scale immediately.
3. In **Reference spectra**, type an X or Y coordinate and press Enter or leave the field. The preview selects the nearest measured value on that axis, then the closest available value on the other axis. Both fields show the selected scan point. Use Repeat for duplicate measurements. The RC plot marks the selected measurement. Selecting a matching background opens this preview automatically. New scans use the measured background at scale 1. Under **Background → Normalization**, turn on **Auto Background Scale** to estimate one multiplier for the entire scan. **Auto-select scale windows** selects stable intervals of the calibration spectrum/background ratio from the current file and displays their energy bounds. Review them for broad features. Turn off automatic window selection to edit the intervals under **Advanced reference settings**. Reopening a scan restores its saved settings.
4. In **Spectral cut**, choose the direction and fixed coordinate. To follow X, hold Y constant. To follow doping, hold E-field constant and choose the gate formula. The cut includes measured points within the coordinate tolerance. Previous and Next step through available cuts.

**Gate coordinates** assigns TG and BG from the selected column names. If both names are ambiguous, enabling a gate transform reveals **Top gate (TG)**. Select one paired definition under **Doping (D) and E-field (E)**: either D = TG + r × BG and E = TG − r × BG, or D = r × TG + BG and E = r × TG − BG. Both equations use the same lever-arm ratio r. The same choice applies to maps and spectral cuts.

Inside the per-scan folder, filenames use the analysis quantity, and its energy setting, for example `rc_at_1.65eV.csv` or `rc_p2p_1.6-1.7eV.png`. Map titles show the same setting. Peak-position exports also identify the peak, dip or automatic feature selection. Identical CSV, PNG and JSON exports reuse the existing files. Changed results add a suffix such as `_002`. The matching JSON records the full source path.

## Reflection backgrounds

Reflection uses the same background file browser components as DRR, with a
separate selection. Open **Background references** to choose CSV files from the
current browsing folder and its subfolders, excluding analysis output folders.
Use **Browse files…** to add references from elsewhere. Selection changes apply to
the current measurement. Choose **Custom, P, N or BN** in the background window.
Each role retains its own files and browsing location. Only the current role's
references are used, as a shared reference for every sweep point; other roles
are not pooled into the calculation. Switching measurements restores that file's
role groups. Changing Folder does not move a remembered role browsing location.

Explicit Pback/Nback/BNbackground filename labels filter candidates and detect
role conflicts. Unlabelled files may be assigned manually, subject to spectral
channel validation. Custom rejects a pool containing multiple explicitly named
roles. Legacy single-pool selections are retained under Custom without guessing
roles; mixed labelled references must be regrouped before calculation.
Its numerical background processing follows Megasweep:

| Setting | Meaning |
| --- | --- |
| All frames equally (default) | Average all selected frames across the selected files |
| First / last frame | Average the selected first / last frame from each file |
| Files equally | Average within each file, then give each file equal weight |

The first and last settings retain their original Megasweep meanings. All-frames
and equal-file averaging differ when files contain different numbers of frames.
Spectral channels must match before a background is applied. The browser's filename
classification does not certify numerical compatibility.

P+N reconstruction is not performed. DRR-specific fitted backgrounds and BN calibration are not
implicitly applied to Megasweep measurements by sharing the browser.

## Exports and movies

Use **Save CSV + PNG** in the upper right or press **Ctrl+Shift+S** to export the current result.
Megasweep exports a 432-pixel-wide PNG at 96 DPI by default. Drag it onto a blank widescreen DeckCraft slide for a 4.5-inch picture with a 3 × 2.25 inch plotting area. The colorbar sits 0.05 inches from the right edge of the plot. When editing an existing slide picture, DeckCraft keeps its current size.

New analyses write below `Processed Data/Megasweep/<source-stem>/` in the source's
bound experiment folder. Imported custom output directories remain part of their
analysis recipe. Review the actual output paths reported after exporting.
PNG uses the selected display limits and color scale; CSV retains the numerical
data. A JSON sidecar records processing and source/background provenance.
Saving the same CSV, PNG and JSON reuses the existing files. A changed export gets a new version.

In a connected DeckCraft presentation, double-click a Megasweep picture to reopen the original scan with its saved background and analysis settings. After editing, use the upper-right save action to publish the revised figure and replace the selected picture. Press **Ctrl+S** in DeckCraft to save the presentation. Keep the PNG, CSV and JSON together in the experiment's Megasweep export folder, with the original measurement files available.

For optical slices, **Export video…** opens coordinate-range, stride, playback,
frame-rate, size, color and axis controls. Preview, play/pause and the timeline
inspect generated frames. FFmpeg is provided by `imageio-ffmpeg` and included in
the application packaging configuration. Encoding runs in the background.
Cancelling a batch keeps completed export groups and stops further work.

## Saved settings and development

Native Megasweep recipes are stored separately from original Megasweep sessions.
On loading a source, compatible original per-CSV recipes may be imported without
modifying the original session files. Files and modes have distinct identities;
missing references must be resolved before Reflection can recalculate.
Legacy multiple-view layouts and line-cut tables are not fully restored. Import
notices in the log identify omitted settings; review the current recipe and
regenerate the plots.

The numerical implementation originates from
[`ylylyl98/Megasweep_data_analysis` at `2612f6f`](https://github.com/ylylyl98/Megasweep_data_analysis/tree/2612f6f9c1a0946ab62c9d9b7c221bdcb8f101ff).
It is namespaced under `core.megasweep`; native bridges and QML live alongside the
other Quick workspaces. No standalone Widgets window is created.

```powershell
.venv\Scripts\python.exe run_quick.py --workflow megasweep
.venv\Scripts\python.exe run_quick.py --workflow megasweep --megasweep-mode Transport --open "transport.csv"
.venv\Scripts\python.exe scripts\check_megasweep_integration.py
```

The integration check generates synthetic files and isolated settings, exercises
three modes and exports, and writes screenshots/reporting beneath `build/megasweep/`.
It does not operate on experimental source files.
