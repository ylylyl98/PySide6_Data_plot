# Troubleshooting

[Home](../../README.md) · **English** | [简体中文](../zh-CN/troubleshooting.md)

Start with the visible status/error message and the workflow in which it appeared.
Keep original acquisition files while diagnosing a problem.

## Contents

- [Installation and startup](#installation-and-startup)
- [Finding and loading data](#finding-and-loading-data)
- [Display and number inputs](#display-and-number-inputs)
- [Backgrounds and scientific results](#backgrounds-and-scientific-results)
- [Export and restoration](#export-and-restoration)
- [Updates](#updates)
- [Reporting a reproducible problem](#reporting-a-reproducible-problem)

## Installation and startup

| Symptom | What to check |
| --- | --- |
| Cannot find an application in the download | Use the installer or portable ZIP from [Releases](https://github.com/ylylyl98/PySide6_Data_plot/releases/latest), not the source-code archive. |
| Portable application cannot start after moving it | Extract and keep the entire `PySide6_Data_Plot` directory. Run `PySide6_Data_Plot.exe` with its `_internal` directory and other bundled files present. |
| Windows shows a SmartScreen warning | The installer may be unsigned. Check its source and the release's checksum; do not disable system security to diagnose an unrelated startup error. |
| Source checkout reports an import/dependency error | Use the project virtual environment and install `requirements.txt`. Windows CI uses Python 3.13; see [development](development.md). |
| Interface or rendering differs between systems | Record Windows version, display scaling, theme and graphics details. Offscreen checks do not validate every native GPU/font configuration. |

For source diagnostics, follow the isolated commands in the development guide.
Use synthetic data and temporary settings for checks. Do not clear application
settings as a first troubleshooting step: they can contain deliberate manual locks
and saved choices.

## Finding and loading data

| Message or symptom | What to check |
| --- | --- |
| `No CSV files found` | Check the selected folder, file extension and workflow filter. The usual CSV workflow uses working copies directly in the experiment root; nested acquisitions are not necessarily listed by every selector. |
| `Folder does not exist` / `Selected path is not a folder` | Select the folder again, especially after moving an experiment or disconnecting a drive. |
| A source is missing from a catalog | Review the search/filter and selection scope, then refresh or reopen the selector. Source lists and processed-history lists serve different purposes. |
| CSV cannot be interpreted | Confirm that its headers and numerical columns match the selected workflow. PL, power sweeps, MCD and SHG do not share one universal CSV schema. |
| A processed DAT has generic axis labels | Numerical DAT can load without metadata. Keep its matching metadata sidecar beside it to restore available labels and processing context. |
| Pairing is ambiguous | Inspect full filenames and conditions and resolve the assignment explicitly. Modification-time ordering does not establish channel identity. |

See [data and exports](data-and-exports.md) for file conventions and the
[user guide](user-guide.md) for selection and grouping behavior.

## Display and number inputs

- **The value appears rounded:** idle text is deliberately compact. Focus the
  input to see its configured editing precision. Formatting does not reduce the
  stored scientific data. Scientific notation such as `1e-6` is accepted where
  the field's precision and range permit it.
- **A typed value has not applied:** press Enter or leave the field. Check its
  inline error. Non-finite values, reversed limits and invalid Log limits are
  rejected rather than silently applied.
- **A PL color tab will not switch:** correct the invalid draft or press Escape
  to cancel it. Legal drafts are committed before switching.
- **Auto leaves an endpoint unchanged:** check its Fix. Auto preserves fixed
  endpoints. In PL, if both endpoints are fixed, that group's Auto is disabled.
- **Log Auto has no result:** confirm that the visible XY range/region contains
  positive finite samples. An empty region or no positive samples retains the
  previous limits with an explanation.
- **An edit seems to follow a file change:** explicit manual locks can survive
  file changes and restart. File-restored locks and manual locks have different
  precedence. Review [display memory](user-guide.md) before removing settings.
- **A region disappears after zooming:** PL draws only the intersections of the
  physical regions with the current viewport. Zooming does not relocate the
  stored boundaries.
- **Blank or invalid plot:** check finite numeric coordinates and intensities,
  selected source, XY range, color limits and Log positivity before changing the
  processing method.

## Backgrounds and scientific results

A newly selected DRR CSV without a saved background waits for a background choice.
Cancelling the background dialog does not calculate a provisional Self result.
Reopen background selection, choose compatible references and apply the draft.

For `Compute failed` or a rejected background, inspect reference type, selected
files, frame policy, gate coverage and acquisition conditions. Missing or changed
saved references need explicit repair; they should not be replaced with an
unreviewed background just to obtain a plot.

Warnings about ambiguity, stale files, unresolved matches or failed validation
carry different meanings. Consult [background and analysis reference](background-reference.md)
for their actual guards and the distinction between a numerically valid output
and a scientifically suitable background. Historical experiment thresholds are
not universal evidence for another measurement.

## Export and restoration

| Symptom | What to check |
| --- | --- |
| Exported files seem missing | Check the workflow directory under `Processed Data/`, the chosen destination and the completion status. MCD, SHG and Power may create package subdirectories. |
| Export reports a path or write error | Check drive availability, write access, path length and whether another application has the target open. A shorter experiment path may be needed for deeply nested packages. |
| Multiple similarly named results appear | Repeated exports use collision-safe suffixes or new package folders. Use metadata and timestamps to identify the intended result. |
| Wide PL export contains four PNGs | This is intentional when the multiplier is greater than 1×: original and wide images are kept for both Linear and Log. DAT remains complete. |
| Derivative-only DRR export also contains ΔR/R | This is intentional: the underlying ΔR/R and derivative products are both saved. |
| Display settings do not restore after moving data | Keep matching DAT/metadata together and preserve the experiment structure for file-index restoration. Source identity and coordinate semantics must match; unrelated same-named files must not share settings. |
| Working copies remain after export | Cleanup is optional and requires a verified matching canonical file in `Initial Data/`. Failed, changed or unverified exports retain their sources. |
| Origin imports the wrong columns | Choose the intended DAT or XLSX sheet and assign X/Y/Y Error columns explicitly where needed. See the SHG and Origin instructions in [data and exports](data-and-exports.md). |

A failed or superseded export must not be treated as a successful saved state.
Check the completion message and expected output set before cleaning up files.

## Updates

Use **Help → Check for Updates…**. The application checks stable GitHub Releases;
source commits and Actions artifacts do not by themselves constitute an update.

If checking fails, confirm network access to GitHub and retry. If a newer release
has no usable installer/checksum assets, the updater cannot offer a verified
installation. For a checksum failure, do not run the rejected partial download;
retry from the published release and report a persistent mismatch.

You can also open [Releases](https://github.com/ylylyl98/PySide6_Data_plot/releases/latest)
manually. Installing an application update and pulling a source checkout are
separate operations. Source users should follow the repository's Git synchronization
rules rather than assuming an installer updates their checkout.

## Reporting a reproducible problem

Open an [issue](https://github.com/ylylyl98/PySide6_Data_plot/issues) with:

1. App version, Windows version and installer/portable/source launch method.
2. Workflow and source type, plus relevant processing/display choices.
3. Minimal steps, expected behavior and actual behavior.
4. Exact visible error and relevant log excerpts or screenshots.
5. A small synthetic or redacted example, including needed metadata sidecars.

For plotting issues, include XY range, Linear/Log or DRR product, region count and
Fix settings. For export issues, include the selected PNG layout, output types and
whether the operation finished. Remove private paths or measurement information
that is not needed to reproduce the problem.
