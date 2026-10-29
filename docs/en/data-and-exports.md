# Data, exports, and source-file safety

[Project home](../../README.md) · [简体中文](../zh-CN/data-and-exports.md)

This reference describes output formats, portable experiment folders, and source-file maintenance. For the controls used to create these outputs, see the [user guide](user-guide.md). DRR reference provenance is explained in the [background reference](background-reference.md).

## Contents

- [Experiment folder layout](#experiment-folder-layout)
- [Raw-data lifecycle and cleanup](#raw-data-lifecycle-and-cleanup)
- [Export locations and contents](#export-locations-and-contents)
- [Metadata and collision handling](#metadata-and-collision-handling)
- [DAT re-import and Origin use](#dat-re-import-and-origin-use)
- [PNG dimensions and paired exports](#png-dimensions-and-paired-exports)
- [Export completion and provenance](#export-completion-and-provenance)

## Experiment folder layout

Choose the experiment folder, such as `YZ327`, as the app's data folder. Canonical
acquisition files belong in `Initial Data`. The app writes new processed results
under workflow-specific directories without moving historical results.

```text
YZ327/
├── Initial Data/                         # canonical acquisition/raw files
│   ├── raw_001.csv
│   └── raw_002.csv
├── Processed Data/                       # generated results
│   ├── PL/
│   ├── DRR/
│   ├── Compare/
│   ├── MCD/<analysis package>/
│   ├── MCD Extracts/                       # Origin XLSX, comparison PNG, settings JSON
│   ├── SHG/<analysis package>/
│   └── Power Dependence/<group or comparison package>/
├── working-copy.csv                      # temporary copy directly under YZ327
└── Initial data after processing/         # manual archive only
```

For normal CSV processing, selected working files remain directly under the
experiment folder because the file browser is more convenient for selecting
many raw files. Existing historical files directly under `Processed Data/`
remain valid; the application does not migrate or reorganize them.

## Raw-data lifecycle and cleanup

For PL, DRR, and Compare:

1. The acquisition application writes canonical files into `Initial Data/`.
2. Use the file manager to copy only the files needed for the
   analysis into the experiment folder.
3. Load and process those root-level working copies in the app.
4. Results are written to `Processed Data/<workflow>/`.
5. If **Clean verified source copies after successful export** is enabled, the
   app may remove a root-level duplicate only after its matching
   `Initial Data/<filename>` file exists and the SHA-256 hashes match.

Cleanup never deletes canonical files. A missing canonical file, invalid path
relationship, hash mismatch, failed export, disabled option, or unverified/manual
source leaves the working file in place. A file selected directly from
`Initial Data/` is never treated as a disposable temporary copy.

The **Move Exported Sources** button is separate and manual. It moves explicitly
selected source files to the legacy `Initial data after processing/` archive.
That folder is not canonical raw storage, automatic cleanup storage, or normal
processed-result storage.

## Export locations and contents

| Workflow | New export location | Typical contents |
|---|---|---|
| PL | `Processed Data/PL/` | DAT, metadata sidecar, linear/log PNG figures |
| DRR | `Processed Data/DRR/` | averaged or derivative DAT, metadata, PNG |
| Compare | `Processed Data/Compare/` | channel DAT/PNG/metadata and VP results |
| MCD | `Processed Data/MCD/<analysis package>/` | map DAT/PNG/metadata, MCD(B) CSV/PNG, diagnostics, settings |
| SHG | `Processed Data/SHG/<analysis package>/` | `shg.xlsx` with angular results and all-angle spectra, `analysis_settings.json` |
| Power Dependence | `Processed Data/Power Dependence/<package>/` | KK/KKp/VP DAT, PNG, metadata |

MCD package names use a readable source, energy, and window identity such as
`<source>_MCD_E1.650000eV_W5meV`. SHG single analyses use bounded names such as
`single_<short-source>_515nm_<identity>`; comparisons use
`compare_<short-reference>_vs_<short-sample>_515nm_<identity>`. The identity
is an eight-character source hash; repeated exports receive a numeric suffix.
### SHG workbook and settings

SHG packages contain `shg.xlsx` and `analysis_settings.json` by default. The
workbook opens on `Overview`, which prominently identifies Reference A and
Sample B (or the single measurement) with separate filenames and full source
paths. It also lists background sources, processing windows, export time and a
worksheet guide, so a workbook copied on its own still identifies the compared
files. Long paths wrap within the overview. Optional CSV copies include
`overview.csv` with the same source information.
The `Plot` sheet contains independent angle/intensity/error column groups
for each source, `Fit` contains smooth model curves within each actual fit range,
`Summary` records fitted parameters and twist results or their unavailable status,
and `Diagnostics` retains the complete per-acquisition results and source rows.
Missing values are blank rather than zero. Optional CSV copies use the same tables.

`Spectra_Raw` and `Spectra_Corrected` store every acquisition as a separate column:
the first column is wavelength in nm and the remaining columns are intensity in
counts, labelled with the processed measured angle and source-row identity.
Repeated angles are preserved, including records excluded from the angular fit;
their status and original/processed angles remain in the diagnostics and metadata.
For comparisons, `Reference_Raw`/`Reference_Corrected` and
`Sample_Raw`/`Sample_Corrected` retain each source's own wavelength grid. Raw
spectra retain the full recorded band. Corrected spectra use cosmic-cleaned data
minus the fitted background, restricted to the peak and sideband interval for
local/external background models; no-background mode retains the full band.
Optional `Spectra_Cleaned` (or `Reference_Cleaned`/`Sample_Cleaned`) sheets retain
the full band after cosmic-ray removal without subtracting a background. Spectral
counts are distinct from integrated angular intensities in counts·nm.

In Origin, import the desired XLSX sheet with its first row as column names.
For spectra, designate the first column X and the selected angle columns Y. For
`Plot`, designate each independent group X/Y/Y Error; for `Fit`, use X/Y groups.
XLSX column labels describe these roles but do not automatically set Origin's
column designations. Select any angle columns to overlay their spectra, without
choosing export angles in advance. Numerical rows remain plain numeric cells.

The JSON keeps full source identities, processing and fitting settings, spectrum
column mappings, units, and the output manifest. Existing SHG CSV/JSON packages
remain readable and are not renamed. The legacy CSV export APIs remain available
to scripts. New paths include a Windows-compatible length budget; overly deep
experiment roots receive an actionable error.
### Power package names

Power packages use one of:
`<group>_PowerDep`, `<KK-group>_vs_<KKp-group>_IntensityCompare`, or
`<KK-group>_vs_<KKp-group>_VP`. Package names are sanitized for Windows and
macOS. Related numerical files, metadata, figures, and diagnostics stay
together.

## Metadata and collision handling

Metadata sidecars use the DAT file's own directory and are optional for reading
the numerical data. Where emitted, the shared metadata core records schema and
app versions, workflow, dataset type, creation time, sources/provenance,
processing, plot settings, outputs, and an output manifest. Workflow-specific
fields record details such as DRR backgrounds, MCD windows, SHG fits, or Power
pairing and alignment.

Repeated exports do not silently overwrite earlier results. Simple result stems
use collision-safe suffixes such as `_01`; complex MCD, SHG, and Power analyses
use collision-safe package directories. Associated DAT, PNG, and JSON files
retain the same logical stem or package.

## DAT re-import and Origin use

Exported DAT files remain tab-delimited, Origin-friendly numerical matrices and
can be opened without JSON. The app can load DAT files from old root-level
`Processed Data`, the new workflow folders, or other supported user-selected
locations. Sidecar metadata is searched beside the DAT file, so moving a DAT
does not require the experiment root to be known. When present, metadata can
restore labels, units, plot context, and processing information.

The PL tab reopens these DAT files and supports an adjacent `.metadata.json`
sidecar or `<file>.plotmeta.json`. Without a sidecar it uses safe generic labels.

For imported DAT colormaps, the PL Y-axis selector supports `Y`, `Doping`,
`Electric field`, `Gate voltage`, and `Custom` labels, with an optional unit.
Without metadata, neutral generic labels are used.

The released binaries and CI validation target Windows. Documentation uses
portable folder names and the application uses cross-platform path handling;
this path portability does not establish support for another operating system.

Compatibility note: the active PySide GUI routes new exports to the workflow
locations above. Some lower-level legacy Python helper functions retain their
old configurable/default `Processed Data` destination for existing scripts;
calling those helpers directly is separate from the normal GUI workflow.

## PNG dimensions and paired exports

Heatmap PNGs default to `8.0 × 6.2 in @ 150 DPI` (`1200 × 930 px`), independently
of the window size. In the native PL/DRR toolbar, **PNG 1.0× / Wide PNG** and the
width multiplier sit beside **Save PNG + DAT**. A fresh settings profile starts
at **PNG 1.0×**. The selected mode and multiplier are shared export preferences,
retained across measurements, group changes, PL/DRR tabs, and application restarts.
The multiplier defaults to **2.00×** and accepts **1.00×–3.00×**. At **1.50×** the
image is `1800 × 930 px`; at **2.00×** it is `2400 × 930 px`.

Height, font sizes, displayed data ranges, and DAT values stay unchanged. Wider
heatmaps fit top spacing to the title and colorbars and reduce side margins to
the space needed by labels, ticks, and colorbars. The on-screen canvas keeps its
current size. Select **PNG 1.0×** to return to the original export layout.

| Export | PNG 1.0× | Wide PNG, for example 2.00× | Numerical files |
|---|---|---|---|
| PL **Save PNG + DAT** | Original Linear and Log PNGs | Original and wide Linear and Log PNGs: four images | One full-grid DAT shared by both scales and widths |
| DRR paired save / **Save both · PNG + DAT** | One original PNG per product | Original and wide PNGs per product: four images for two products | One full-grid DAT per product: two DATs |

When only DRR's second energy derivative (D2E) is displayed, **Save PNG + DAT**
also saves the DRR map in the background. Both products use the displayed D2E
X/Y ranges, including toolbar zoom, while retaining their own color scales. The
display stays on D2E and each DAT retains its complete numerical grid.

The original-width PNG keeps its normal filename. Wide PNGs append a multiplier
such as `_2.0x.png`. Other multipliers, including 1.50×, also retain the 1.0×
image. DRR repeat saves reuse matching images and repair missing selected widths
without duplicating DAT files. PL saves create a new collision-safe result stem
and DAT for each export. The legacy Widgets DRR toolbar's PNG-only save follows
the same width rule using one frozen figure. The native shell provides
**Save PNG + DAT**, without that legacy PNG-only toolbar action.

The native shell's shared **PNG 1.0× / Wide PNG** preference is available for
PL/DRR; it is not a shared native Compare/VP or Power export setting. Native MCD
has its own **PNG width multiplier** control, defaulting to 1× with a 1–4× range.
The legacy Widgets shell has broader toolbar width handling for main-plot PNGs,
PNG + DAT heatmaps (including Compare/VP and Power), and MCD Save Results;
this is a separate compatibility path. Its toolbar PNG action widens the current
figure while leaving the on-screen plot unchanged. PL writes separate
Linear/Log images; the desktop no longer offers an extra combined side-by-side
PL PNG.

DRR PNG titles use whole-word fitting with a bounded per-font cache, falling
back to character wrapping for long identifiers while keeping math tokens
complete. Export meshes use original cell boundaries at full resolution inside
the visible X/Y range, plus adjacent guard cells for clipping; cells outside the
viewport are omitted from the PNG mesh. DAT files and interactive data retain
all measurements. Singular or nearly singular ranges and invalid log bounds
retain the full extent for Matplotlib's usual automatic range handling. Measured
right axes, normalization, dimensions and saved-image reuse are preserved.
Generated title markers show only the current product: `(DR/R)
(d2(DR/R)/dE2)` becomes `(d2(DR/R)/dE2)`. Metadata retains the original title.
Saving refreshes PNGs produced under the earlier title policy while reusing
unchanged DAT files.

## Export completion and provenance

Successful saves report completion in the status bar and record output paths in
**Log**, preserving the current bottom-panel visibility and scientific analysis
text. **Analysis results** remains for peak finding, fitting, and other scientific
results. SHG shows a compact relative results-folder path after exporting; hover
for the full path, or use **Open folder** / **Copy path**. Receipts list the folder
once and then the filenames.

SHG's **Save XLSX + settings** writes the workbook and settings JSON described
above. Optional CSV copies and full-band cosmic-cleaned spectra are off by
default and do not trigger reprocessing. Select angle columns after importing
the workbook into Origin; export includes all measured angles.

Data Maintenance binds **Move Exported Sources** to a verified successful
export. Optional cleanup removes only verified temporary copies while preserving
canonical originals; changing sources or outputs invalidates the export receipt.
Checks should use synthetic files in temporary folders. See
[development and verification](development.md) for worker timing and test scope,
or [troubleshooting](troubleshooting.md) for failed exports.
