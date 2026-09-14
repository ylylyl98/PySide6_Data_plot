# MCD Center Workflow Implementation Plan

> **For agentic workers:** Implement locally with the existing dirty worktree preserved; a single Astra read-only review follows this implementation.

**Goal:** Correct MCD recommendation ranking, spectrum center semantics, per-field feature association, candidate overlays, and compact unified-page controls using the existing real-data workflow.

**Architecture:** Keep detection pure and JSON-friendly in `core/mcd_analysis.py`, adding evidence metadata without deleting the full candidate catalog. Keep presentation changes in `ui_qt/mcd_unified_page.py`, where candidate filtering and overlays are lightweight drawing operations. Preserve existing worker ownership, source processing, and legacy APIs.

**Tech Stack:** Python, NumPy/SciPy, Matplotlib, PySide6, existing unittest/Qt tests.

**Spec:** `artifacts/center-audit/report.md`, `artifacts/center-audit/summary.json`, and `artifacts/center-audit/diagnostics.json`.

## Global Constraints

- Preserve all pre-existing dirty changes and never modify the original CSV or `artifacts/center-audit/`.
- Keep full measured source, branch, field-sign, and feature-kind identity in the catalog.
- Use measured current-B points and separate reference-center metadata; never hard-code expected energies.
- Keep Inc/Dec display switches independent and preserve Shift/Energy/Splitting exclusivity with Overlay independent.
- Save new verification artifacts under `artifacts/center-implementation/`.

---

### Task 1: Pure center and candidate evidence

**Files:**
- Modify: `core/mcd_analysis.py`
- Test: `tests/test_mcd_analysis.py`

- [ ] Add failing tests for sub-grid local centers, one candidate per field per track, and recommendation metadata that keeps all candidates.
- [ ] Implement explicit raw/residual detection mode metadata, return the quadratic local center, and use a local raw locator by default.
- [ ] Change row clustering to reject duplicate field assignments within one feature group while retaining separate measured identities.
- [ ] Add data-driven MCD recommendation scores based on normalized field response and branch agreement, marking one representative per energy group while retaining all candidates.
- [ ] Run the focused analysis tests and verify the real CSV catalog has no same-field duplicate support in one feature and exposes recommendation metadata.

### Task 2: Unified candidate presentation

**Files:**
- Modify: `ui_qt/mcd_unified_page.py`
- Test: `tests/test_mcd_unified_workflow.py`

- [ ] Add failing tests for full-catalog filtering, candidate overlays on both axes, immediate clearing/reselection, and bounded candidate control width.
- [ ] Remove display truncation; show recommended markers in labels while keeping the complete catalog available in MCD/Spectrum/All.
- [ ] Redraw all visible candidate markers immediately on filter and selection changes, with outlined selected lines, triangle tags, IDs, and Window-labeled translucent regions.
- [ ] Bound the candidate combo width and use compact, wrapped rows while keeping Inc/Dec and Overlay semantics intact.
- [ ] Run focused Qt tests in offscreen mode and render a real-data screenshot to `artifacts/center-implementation/`.

### Task 3: Source status verification and handoff

**Files:**
- Modify only if required: `ui_qt/main_window.py`
- Create: `artifacts/center-implementation/` verification outputs and `changed-manifest.json`

- [ ] Verify Source status separates filename, processing state, and saved time; make a local wording adjustment only if the current implementation still conflates them.
- [ ] Run changed-file compilation, focused tests, the existing reproduction/analyze scripts against a copied result, and `git diff --check`.
- [ ] Write measured before/after candidate counts, center labels, screenshot paths, and known limitations to the implementation report.

