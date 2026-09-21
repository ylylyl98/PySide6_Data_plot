# Limit results: table refresh freeze

The previous candidate-copy optimization did not address the visible table refresh. QHeaderView.ResizeToContents rescanned cells during every QTableWidget.setItem, even with updates disabled. A shown 100-row, nine-column table reproduced 3.31 seconds versus 0.006 seconds with sizing suspended. A 500-row workspace page took 99.88 seconds under profiling; setItem accounted for 99.73 seconds.

WorkspacePeakController.populate_table now temporarily freezes header sizing across all nine columns, blocks table signals, and restores header modes, signal state and painting in finally. Candidate data and scientific calculations are unchanged.

Verification: a regression test fails before the change and confirms sizing stays suspended for every populated cell afterward. All 31 analysis-window, candidate-filter and batch-UI tests pass. The shown 500-row page refresh including paint measured 0.050 seconds after the fix; a full filter cycle with overlay disabled measured 0.378 seconds including debounce. Benchmarks are synthetic and machine-specific, not a latency guarantee.
