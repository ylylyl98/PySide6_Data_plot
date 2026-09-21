# DRR export latency investigation

Current-code synthetic timings, temporary output only:

- 295 x 1340 raw/second PNG+DAT pair: 1.091 seconds fresh, 0.009 seconds reused.
- Peak XLSX with 500 / 1000 / 2000 points: 0.612 / 1.553 / 6.701 seconds before.
- Same XLSX fixtures after the fix: 0.241 / 0.301 / 0.613 seconds.

The peak workbook row writer appended values, then accessed
`sheet[sheet.max_row]` to mark string cells as literal text. Those dimension
lookups repeatedly scan the growing worksheet, causing quadratic work.
It now constructs just the new row's Cell objects with literal string types
before appending. Sheet contents, formatting, formula-like text handling,
numeric precision and exported image resolution are unchanged.

The regression test reproduces the repeated dimension scan and verifies row
placement, string type and numeric values. All 21 tests in test_drr_peak_export,
test_drr_dual_export and test_drr_dual_save_workflow passed.

This confirms and fixes a bottleneck when peak analysis is included. It does
not establish the cause of the user's live export slowdown: whether that save
contains peak analysis, its data size and the slow stage were not yet supplied.
The pair timing excludes GUI preparation, derivative computation, real source
hashing and post-export catalog refresh. No live measurement files were changed.
