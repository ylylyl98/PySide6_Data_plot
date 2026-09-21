# Local application and main integration

**Goal:** Preserve the current local application, integrate origin/main, and verify the resulting application before any push.

**Approach:** Record the local source, tests and documentation in a checkpoint commit. Exclude generated artifacts. Merge origin/main using normal Git history, examining every conflict against the checkpoint and upstream implementation. Do not overwrite either side wholesale or force-push.

**Validation:** Run imports, the CI representative suite, DRR Save Both/export regressions, and the full suite with visible progress and bounded diagnostics. Report any reproducible remaining failures separately from merge failures.

- [x] Checkpoint local source and tests; preserve generated files locally (`7650ec1`).
- [x] Integrate latest origin/main and review resulting changes (`cf94348`).
- [x] Run focused verification, investigate failures, and obtain code review; full-suite limitations are recorded in the report.
- [x] Record the integration and test results in `docs/superpowers/reports/2026-09-21-main-integration.md`; leave remote unchanged.
