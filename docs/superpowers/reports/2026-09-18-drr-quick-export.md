# Direct group loading and one-click P2P exports

- Double-clicking a comparison group or pressing Open group loads it directly. Manage members is separate; applying membership reloads the active group only. The member list is labelled Loaded datasets.
- P2P Save CSV and Save PNG no longer open a file picker. They write beneath the measurement folder's `Processed Data/DRR Analysis/<group>/` directory. Names include the group, Energy interval, raw/SG configuration and export scope. Exclusive reservation adds `_v002`, `_v003`, etc. rather than overwriting existing exports.
- Save as provides explicit CSV/PNG file pickers without changing the default directory. CSV retains all calculated datasets for Origin; PNG captures the current figure.
- Results show the saved path and an Open folder button. Pending calculations cannot be exported. Failed automatic exports remove their reserved file. Long paths do not force the sidebar wider.

Verification: 38 related unittest cases passed; the final focused run of 8 export/comparison/P2P tests passed after the sidebar adjustment. Tests cover no-dialog exports, versions, valid PNG/CSV output, folder opening, SG naming, explicit Save As and direct group loading. Offscreen 1500×950 inspection confirmed zero horizontal sidebar overflow after exporting a long filename.
