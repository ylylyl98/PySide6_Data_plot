# Source selection reliability design

Date: 2026-09-10. Execution requested by the user: Astra writes the plan, Luna implements it, Astra reviews the completed changes. This written design records the already authorized direction; no additional approval checkpoint is needed.

## Goal and boundaries

Make existing file selection and processing-history displays reliable while retaining each tab's selection model. This is a bounded repair across existing flows, not a shared picker redesign. The implementation is split into five independently testable tasks in the accompanying plan.

## Global constraints

- Keep the existing Python/PySide6 stack and dependencies; use `python -m unittest` because pytest is absent in the default Python 3.13 environment.
- Work on `codex/file-selection-reliability` in the existing workspace; preserve all pre-existing uncommitted Rot1/Rot2 mapping changes and tests. Do not create a separate worktree, reset files, stage, or commit.
- Preserve per-tab business differences: PL single raw/result selection, DRR measurements plus independent backgrounds, Compare groups plus manual channel mappings, MCD shared source, Power sweep/role groups, and SHG single/reference/sample/background choices.
- Processing status means saved history, not verified freshness for the current source bytes or current UI settings. Do not hash measurements during catalog refresh or introduce a generic processing-status framework.
- Keep source identities in item data and metadata; status decorations must not become loader paths.
- Preserve explicit Clear and Cancel behavior. Refresh must not silently substitute unrelated sources when a prior selection disappears.

## Verified code findings

1. `CompareController._cmp_open_group_dialog` connects Refresh to `_refresh_view`, which only regroups the in-memory catalog. Main-window catalog scans already run asynchronously. `SourcePickerDialog.restore_selection` selects the only remaining row even when a prior choice disappeared. Compare combo refresh drops a missing mapped source before auto-assignment, allowing incomplete old choices to become a different assignment.
2. `core.data_io.discover_pl_processing_status` recognizes an exact relative path but then appends *all* basename matches. `core.mcd.discover_mcd_processing_status` uses basenames exclusively. These are real duplicate-name association bugs.
3. DRR's dialog already performs asynchronous refresh with generation/closed-dialog guards. Its chosen list does not validate missing files. Main-window `_on_drr_catalog_refresh_result` also silently removes absent measurement/manual-background selections. Both boundaries need correction; the discovery worker itself is not missing.
4. Compare exports already write `Compare/<channel>` and `Compare/VP` operations below `Processed Data/Compare`. An old intensity panel metadata file contains only that panel's input, while VP records have role-specific KK and KKp sources. Existing records do not prove that all intensity panels were exported together for a given combination.
5. Power picker rows already display New/Processed/Combined using `processed_source_names`. The missing display is the main sweep and KK/KKp role combo rows. Do not reimplement the already working picker status.
6. Peak Shift's badge currently says Selected, not Processed. It shares MCD source selection and should explicitly show *MCD source history*, without implying a Peak Shift analysis exists.
7. SHG intentionally lists all root CSVs. `core.shg` already recognizes measured-angle aliases and numeric wavelength columns. Add explanatory hints, not a format-based exclusion policy.
8. MCD discovery recursively prefers a case-insensitive `mcd` directory if it contains CSVs, then falls back to root CSVs. Keep this priority, including the existing root fallback when the dedicated directory is empty.

No applicable AGENTS.md was found in the repository or its parent directories during planning.

## Design decisions

### Source history identity and uncertainty

Introduce one small source-identity helper, with no UI or status policy. An explicit experiment-relative path is authoritative, including a root-level filename stored in an explicitly relative field. An absolute path inside the current experiment is also exact. For legacy metadata, a path containing directories is exact; a basename can fall back only when it identifies one candidate. Once usable exact identity exists but points to no candidate, do not downgrade to a different same-name source. Normalize slash separators and Windows case consistently. A moved experiment may use portable relative identity; do not guess arbitrary suffixes from stale absolute paths.

Keep `discover_pl_processing_status` and `discover_mcd_processing_status` returning `dict[str, str]`. Add an optional keyword-only `ambiguous_sources: set[str] | None = None`; populate it with candidates affected by basename-only ambiguous records. Exact evidence wins for an individual candidate, so remove confirmed candidates from the final ambiguity set. A source without confirmed or ambiguous evidence is New; ambiguity displays `History unknown — legacy metadata matches multiple files`. Unknown sources appear under All, are not counted as New or Processed, and are not chosen by PL auto-advance. Existing saved DAT handling stays intact. Add an Unknown filter/count only on PL and MCD, using a neutral existing badge state.

PL and MCD badges and picker hints clarify that saved history does not validate current settings. Add explicit source identity to future MCD settings exports while retaining legacy `source_file` and `filename` fields. Existing PL descriptors already carry relative `name` and absolute `source_path`; retain their format.

### Compare refresh and selection

Use the existing main-window catalog worker and notify the open Compare picker only after its result has been applied. A small owner signal is preferable to a second scan or UI-thread polling. Refresh retains local search, local power tolerance, selected group key, existing manual mapping, angle references, and swap flag. Disable its button while the request is outstanding and restore it on completion/error. Bind connections to the dialog lifetime; closing it or changing folders must not apply stale dialog state.

A previously selected group that disappears yields an explicit missing message and no substituted row. Scope the one-row-auto-selection opt-out to Compare with a backward-compatible optional `SourcePickerDialog` argument; do not change other tabs' selection policies. A mapped path that disappears remains represented in the combo/model as missing; preserve surviving assignments and disable or reject loading with a readable missing-path error. Do not refill a deleted channel from other group members during automatic refresh. Explicit group selection and Auto Assign still intentionally replace mappings. Initial auto-assignment and explicit Clear retain existing behavior.

### DRR missing choices

Keep user-chosen missing measurements and manual external baselines in selection state. Show `Missing` in chosen rows and summary, retain their original identity, and require the user to remove or replace them. Validate using `resolve_source_path(...).is_file()`, not membership in the filtered catalog: external absolute baselines and paths omitted by a format/type filter can remain valid. Validate after refresh, before accepting the dialog, and at the actual main-window DRR load boundary. Reject the entire request if any required file is missing, including missing automatic-assignment baseline paths; never run a subset. Preserve existing automatic-background unresolved behavior and complete-group repeat expansion when all existing selected files are valid.

### Compare history

Read only existing Compare metadata, ignore malformed/non-dict payloads, and ignore PL history. Cache parsed history at catalog refresh/export completion, not on every plot-control change. Current Intensity shows matching channel/source history; current VP requires exact role-specific KK and KKp identities in the same `Compare/VP` record. Swapping source roles must alter the match. Keep intensity and VP histories separate.

For future Compare exports add `compare_source_mapping` to metadata processing fields for both panel and VP outputs. This gives future Intensity records enough evidence for exact-combination matching. Records with this mapping must match the selected active channel/source combination before counting as combination history; a map with extra hidden channels may match the same active subset, but every active role must be present with the same source. Old panel-only records may be displayed under `Individual panel history; combined selection not verified`, with their matching channel and timestamp. They must never produce an overall Processed/current-combination badge. A record from a different full mapping is excluded, even if one panel matches. All successful matches say `Saved history`, never `Current`, `Up to date`, or an unqualified processed-current claim. Settings values may be shown in details as historical metadata; no settings equality/freshness engine is required.

Use the existing Compare group badge/details and assignment summary. No new dialog, process, database, history sidebar, or output loading feature is needed. A missing metadata directory means `No saved Compare history found`, not proof that raw data have never been processed elsewhere.

### Small presentation changes

Power's main/role group rows use the same all-records-processed predicate already used in its picker. Partial groups remain New under the existing binary policy; do not invent partial-state semantics in this task. Keep all item keys and ordering unchanged.

MCD picker hint states the actual scan scope (`mcd/**` or root fallback) and that its status is saved MCD history. Peak Shift mirrors the source's MCD historical/unknown/new text with an explicit `MCD source history` prefix and never describes the Peak Shift pipeline as processed.

SHG adds a static hint describing measured-angle aliases and numeric wavelength columns, and explicitly notes that all root CSVs remain available. Static hints avoid rescanning every file merely for decoration.

## Exclusions

No unified picker or status framework, no new Peak Shift processing/export pipeline, no MCD discovery-priority changes, no aggressive SHG filtering, no rework of Power status discovery, no measurement hashing on refresh, no export migration/deletion, no changes to Rot1/Rot2 channel semantics, no speculative repairs to working picker features, and no automatic commits.

## Acceptance

Duplicate-name regressions must demonstrate exact association and unknown legacy ambiguity. A Compare Refresh performed while the dialog is open must see newly created files and retain manual swaps; deleting its prior group must not select the only unrelated row. DRR deleting one selected measurement or baseline must remain visible and block acceptance/loading. Compare tests must distinguish PL from Compare, intensity from VP, KK/KKp swaps, full-map mismatch, and old individual-panel history. Existing 94 baseline Compare/PL/DRR/Power/MCD source tests must remain passing, with focused new tests and the full unittest suite run at the final verification gate.
