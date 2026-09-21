# DRR comparison organizer implementation plan

Goal: select magnetic-field comparison groups by sample, spot and gate condition; retain manual membership and prior analysis without changing source files.

Architecture: a metadata catalog and persistent override store are independent of numeric AnalysisDataset objects. A comparison browser previews groups before opening. Group workspaces and immutable dataset identities reuse the existing pickle-free NPZ session format. P2P range/SG variants are archived independently.

Approved scope: magnetic-field grouping, manual include/exclude/restore, duplicate and processing warnings, open-group restoration, shared-window P2P. No automatic averaging, interpolation or fixed-Y extraction in this first iteration.

- [x] Test and implement conservative condition parsing and catalog grouping. Use measurement-source filenames in current metadata; verify all averaged sources agree. Structured condition metadata is not present in this catalog. Normalize minus signs and numeric spellings. Never group across sample/spot/gate relationship/temperature/optics. Keep uncertain entries separate.
- [x] Test persistent manual membership overrides and separate dataset/group/analysis storage, atomic writes, stable keys, new-source invalidation.
- [x] Add group preview browser, explicit open, exclusions/restoration and manual additions. Keep raw-file import as secondary workflow.
- [x] Integrate group switching with save-before-switch, restoration of per-file results, current range and P2P state. Preserve prior range variants.
- [x] Validate with synthetic tests, actual metadata catalog and offscreen UI. Document practical limits and verification.
