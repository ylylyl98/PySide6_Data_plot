# Reuse processed DRR groups

Analysis discovers saved groups under the main app's current folder in
`Processed Data/DRR`. The left-hand list supports filtering, multi-selection,
refresh and double-click reuse. Opening the existing standalone Analysis from
another main-app folder forwards the folder through its local socket and
refreshes the catalog after any active job finishes.

Each group uses the exported raw DRR DAT and its metadata directly. Original
measurement combinations, background recipes and source descriptions are kept
as provenance; CSV loading, alignment and background subtraction are not run.
SG defaults are restored within the second-derivative controls' supported range;
the original saved processing remains unchanged in provenance.
Duplicate imports preserve existing dataset settings/results. Derivative-only
exports, incomplete pairs and ambiguous metadata are excluded. Unsaved current
DRR data can still be sent through the main window's snapshot shortcut.

Discovery and loading run through existing worker/progress/cancellation UI.
Tests cover multi-file groups with missing raw sources, exact numeric reuse,
derivative rejection, duplicate selection, and unsupported saved SG defaults.
Read-only real-data verification found 305 eligible groups in YZ365 and loaded
one 111-by-1340 matrix with its three-reference provenance.
