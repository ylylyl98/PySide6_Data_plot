# DRR seeded tracking adjustment

Added optional Target branch mode after full-range candidate detection produced
many false positives on BG=18 data. Pick a seed on a heatmap or enter its energy
and Y manually; select one source and explicitly select Peaks or Dips. Seed
picking consumes the click so legacy peak deletion cannot run. A star indicates
the requested seed. Existing All candidates mode remains explicitly exploratory.

Track the nearest qualified seed bidirectionally in sorted Y. Candidate gates:
prominence, robust noise scale, physical half-prominence width, displacement,
width/prominence ratios, normalized local shape correlation, and ambiguity.
Second derivatives use a robust amplitude floor because SG noise is correlated.
Missing rows remain blank, recovered segments are separate, and traversal stops
after the configured number of consecutive gaps. No Y interpolation, forced
connection, resonance-center fitting, or automatic physical interpretation.

User-visible settings: seed E/Y, source, Feature, target noise multiplier,
minimum width and max missing rows. Existing max Y-step shift and prominence
also apply. Min spacing / max peaks controls are for exploratory mode; seeded
mode chooses at most one qualified extremum per row and does not truncate the
candidate list by prominence rank.

Real-data validation: use second derivative SG31/2, Peaks, X=1.820–1.875 eV,
all Y, prominence=.30, shift=1.5 meV, noise multiplier=4, width=1 meV,
max missing rows=2. Seeds (energy eV,Y V): (1.845,12), (1.852,6.609),
(1.856,0). These independent runs retain 57, 7 and 15 rows, respectively.
The figure combines three runs for comparison; the app replaces the current
result on each analysis. No manual removal was used. This is conservative
feature extraction, not a validated physical mode assignment. Middle portions
remain incomplete.

Verification: 48 focused numeric, export, UI and integration tests passed.
Regression cases include stronger neighboring branches, ambiguous splitting,
missing rows, raw white noise, correlated derivative noise, and seed clicks
without legacy deletion. Native hidden-window UI capture inspected. Export
smoke check confirmed untracked Y rows remain blank. Independent bounded review
found implicit polarity selection; this was removed and explicitly tested.
No application restart, executable build, or source data modification.

Comparison: artifacts/drr-batch-peaks/parameter-study/seed_tracking_comparison.png.
