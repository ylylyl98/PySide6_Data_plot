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

The initial optical energy window follows the original Megasweep application.
Set it to a range present in your measurement before calculating. An out-of-range
window is an error, not a request to extrapolate data. Automatic background scaling
likewise requires measured channels in its configured reference windows.

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

New analyses write below `Processed Data/Megasweep/<source-stem>/` in the source's
bound experiment folder. Imported custom output directories remain part of their
analysis recipe. Review the actual output paths reported after exporting.
PNG uses the selected display limits and color scale; CSV retains the numerical
data. A JSON sidecar records processing and source/background provenance.
Existing exports are preserved using versioned destinations.

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
