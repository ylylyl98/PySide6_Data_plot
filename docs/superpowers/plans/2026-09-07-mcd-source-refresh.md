# MCD source synchronization and automatic peak analysis

> **For agentic workers:** Execute this approved, bounded change inline, task by task, using test-driven-development and verification-before-completion.

**Goal:** Selecting an MCD CSV loads matching angles and data automatically; entering Peak Shift analyzes only when needed.

**Architecture:** Keep the existing MCD processing cache and local-fit cache. Bind detected angles to resolved source path, size, and modification time. Deduplicate pending angle scans and local fits. Use a lightweight source/settings analysis key for tab activation rather than triggering Load on every tab switch.

**Tech Stack:** Python, PySide6, unittest, existing NumPy processing.

**Spec:** User-approved design in this task: synchronize file/angles first, then automatic Peak Shift analysis, without redundant calculation.

## Global constraints

- No new dependencies or changes to scientific fitting algorithms.
- Never apply angles or asynchronous results from another source/version.
- Display-only changes and repeated tab entry must not read CSVs or restart fits.
- Retain manual controls and existing processing caches.
- Match 73 and 72.999 degrees using the existing absolute 0.01-degree readback tolerance in core/mcd.py. Preserve CSV values; reject missing or ambiguous matches.

## Task 1: Bind angles to the selected file version

**Files:** `ui_qt/controllers_mcd.py`, `ui_qt/main_window.py`, `tests/test_mcd_auto_refresh.py`.

**Interfaces:** `_mcd_angles_ready()` validates the selected source fingerprint; `_mcd_detect_available_angles()` deduplicates identical pending work; `_request_mcd_load()` resumes after detection; `_start_load("MCD")` enforces the same readiness check for all entry points.

- [x] Add regression tests for missing/invalid replacement files, a changed file during angle detection, and repeated catalog refresh while an automatic load is pending. Assert selected angle values, valid readiness, and exactly one worker dispatch for unchanged detection.
- [x] Run `python -m unittest tests.test_mcd_auto_refresh -v` and confirm new tests fail before implementation.
- [x] Clear readiness before detection/errors; retain a `(resolved path, size, mtime_ns)` ready/pending fingerprint. Reject callbacks with changed fingerprints and repeat detection while preserving load intent. Validate selected angle membership and distinctness.
- [x] Route direct MCD loads through readiness; preserve pending reload when another load is running.
- [x] Run the source tests and existing MCD processing tests.

## Task 2: Analyze on tab entry without duplicate work

**Files:** `ui_qt/main_window.py`, `ui_qt/feature_pages.py`, `tests/test_mcd_auto_refresh.py`.

**Interfaces:** `_ensure_mcd_peak_analysis()` checks source/settings freshness; `_request_mcd_local_fit()` deduplicates its exact pending cache key before cancellation; completion/error clears pending state.

- [x] Add regression tests: first tab entry produces results; repeated entry preserves result identity and performs no analysis; changed processing parameters trigger analysis; entry during a local fit preserves the same worker and cancellation token.
- [x] Run `python -m unittest tests.test_mcd_auto_refresh -v` and confirm new tests fail before implementation.
- [x] Use source identity and computation settings as freshness key, excluding display controls. Record keys for successful results and pending fits. Use existing caches and skip requests matching an active fit.
- [x] Call the same ensure method on tab activation and MCD load completion; retain current results when the identical cached source is republished.
- [x] Run source/peak tests plus MCD processing/local-fit/display/valley suites; inspect the diff and record results below.

## Verification results

### Follow-up: bidirectional tab plotting

- [x] Entering MCD redraws the MCD view when a valid MCD result is already loaded. No load is submitted; entering an empty MCD tab remains safe.
- [x] Regression test loads a synthetic paired CSV through the real processing pipeline, switches MCD/Peak Shift twice, checks the active axes and mode, and verifies that loaded data and peak results retain identity without Load or peak analysis calls.

### Follow-up: default Raw and second-derivative results

- [x] Default Peak tracker to Raw spectrum, retaining the existing joint Raw/Second derivative result plotting and optional Local mixed fit.
- [x] Update the tooltip to explain that the selector controls candidates while both finding methods appear in the result plot.
- [x] Add a default-tab-entry test using two channels with a finite field-dependent separation: both method curves must be present and no local worker submitted. Explicitly select Local mixed fit in tests dedicated to that optional method.

Completed 2026-09-07. Added 11 regression tests in tests/test_mcd_auto_refresh.py; confirmed failures before fixes, then passing results. Full command: python -m unittest tests.test_mcd_auto_refresh tests.test_mcd_shared_source tests.test_mcd_peak_shift tests.test_mcd tests.test_mcd_local_fit tests.test_mcd_peak_display tests.test_mcd_valley_split -q. Result: 131 tests run, 130 passed, 1 skipped, no failures. Independent code review found a local-fit method-switch cancellation race; fixed and regression-tested. git diff --check passed. Verification used offscreen Qt and synthetic fixtures; the user CSV/live application was not available for reproduction.


Follow-up validation: python -m unittest tests.test_mcd_peak_shift tests.test_mcd_auto_refresh tests.test_mcd_shared_source -q; 59 tests passed. git diff --check passed.
Bidirectional-tab validation: python -m unittest tests.test_mcd tests.test_mcd_auto_refresh tests.test_mcd_peak_shift tests.test_mcd_shared_source -q; 121 tests run, 120 passed, 1 skipped. git diff --check passed.
