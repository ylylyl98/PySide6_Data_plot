# DRR Analysis process isolation

The main DRR entry now launches a detached analysis process, matching the MCD
Organizer lifecycle. A local socket forwards repeated opens and immutable
processed-data snapshots to the existing workspace. Closing the main app does
not close Analysis. Closing and reopening Analysis loads updated source code.
The main app must be restarted once to pick up this changed launch entry.

The default workspace is `%LOCALAPPDATA%/DPTK/DRR Analysis/workspace.npz`.
It stores copied numeric data, provenance, per-dataset parameters, accepted
results, current dataset and plot views. Saves use atomic replacement and loads
do not allow pickle. Graceful close saves accepted state; in-flight work is
cancelled rather than resumed. This is not crash recovery or live code reload.
`Save workspace…` also allows choosing a destination; the standalone CLI accepts
`--session PATH` for opening a separately saved workspace.

Corrupt originals are preserved. Failed saves keep the window open, and a
workspace without a usable persistence path requests a save destination before
closing with data. Successfully persisted private inbox snapshots are removed.

Validation: unit coverage includes numeric/result/view round trips, invalid
archives, detached main-window launch, close/reopen restoration and preventing
silent discard. A real subprocess test exercises local-socket delivery, reuse
of the running instance, graceful shutdown and restored datasets on relaunch.
