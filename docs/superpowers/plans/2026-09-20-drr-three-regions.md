# DRR three-region color scales

**Goal:** Add optional three-region DRR color limits and two shared boundaries, with separate raw/second-derivative limits and non-overlapping right-side colorbars in preview and PNG.

**Architecture:** Extend the existing split-scale dataclass with optional second boundary and middle limits. Keep two-region calls backward compatible. Share region masks between drawing and artist reuse. Add a focused DRR control helper for three-region UI and persistence; keep other workflows unchanged.

**Approved design:** Current conversation: 1/2/3 regions; shared boundary positions for raw and d2E; independent per-product limits, Auto and Fix; three vertical bars stacked at right; widen PNG canvas while preserving heatmap size. DAT values unchanged; JSON stores color settings.

- [x] Write failing core/UI/export regression tests.
- [x] Implement validated three-region masks, render metadata, and artist reuse.
- [x] Implement region selection, controls, Auto/Fix, and saved settings.
- [x] Implement matching preview/export colorbars, headers and JSON metadata.
- [x] Run relevant regressions and render/inspect preview and exported PNG; review code before completion.

Validation uses synthetic contrasting regions, invalid/coincident boundaries, raw and derivative products, old two-region regressions, and temporary settings/cache directories. Do not commit unrelated existing changes.
