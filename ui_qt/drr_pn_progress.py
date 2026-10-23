"""Translate reconstruction events into UI stage milestones, never an ETA.

This helper owns no Qt objects and calls no reconstruction code. Its two
outputs only require an ``emit(value)`` method. The GUI must verify that the
result still belongs to the active request before displaying the final stage.
"""

STAGES = (
    "verify_inputs", "load_raw", "v12", "v20", "v21", "v24", "v29", "v31", "v32",
    "regions", "render_full", "render_first_half", "render_last_half",
    "raw_common_texture", "bounded_texture", "apply_bounded_texture",
    "save", "preview", "ready",
)
TOTAL_STAGES = len(STAGES)
STAGE_PROGRESS_NOTE = (
    "Stage progress; stages take different amounts of time. "
    "This is not an estimate of time remaining."
)

_SOURCE_STAGES = ("v12", "v20", "v21", "v24", "v29", "v31", "v32")
_LABELS = {
    "verify_inputs": "Checking selected files and saved versions…",
    "load_raw": "Reading original spectra…",
    **{stage: f"Fitting P/N source spectra ({index}/7)…"
       for index, stage in enumerate(_SOURCE_STAGES, 1)},
    "regions": "Checking sample-supported regions…",
    "render_full": "Reconstructing background…",
    "render_first_half": "Checking first-half repeatability…",
    "render_last_half": "Checking last-half repeatability…",
    "raw_common_texture": "Checking common P/N fine structure…",
    "bounded_texture": "Checking supported fine correction…",
    "apply_bounded_texture": "Constructing the final draft…",
    "save": "Saving draft…",
    "preview": "Loading background preview…",
    "ready": "Finishing background preview…",
}
_INDICES = {stage: index for index, stage in enumerate(STAGES)}
_ALIASES = {f"source_{stage}": stage for stage in _SOURCE_STAGES}


class ReconstructionProgress:
    """Emit monotonic completed-stage counts from zero through eighteen.

    Nested region/render events have variable counts and reused names. They
    neither advance this fixed stage sequence nor replace its friendly label.
    A cache hit skips verified source-fit stages. Reusing a saved result can
    jump directly to preview. Neither path permits this helper to emit 19:
    only the GUI's accepted success result may complete the progress bar.
    """

    def __init__(self, progress_signal, log_signal):
        self._progress = progress_signal
        self._log = log_signal
        self._completed = 0
        self._emitted = False

    def _advance(self, completed):
        completed = max(self._completed, min(completed, TOTAL_STAGES - 1))
        if not self._emitted or completed != self._completed:
            self._completed = completed
            self._emitted = True
            self._progress.emit(completed)

    def start(self, stage):
        """Announce a known stage without moving backwards."""
        self.event(stage, "start")

    def complete(self, stage):
        """Record a known stage's completion, reserving final success for GUI."""
        self.event(stage, "complete")

    def event(self, stage, event):
        """Accept source/pipeline callbacks and ignore unknown nested events."""
        if stage == "feature_protection":
            if event == "start":
                self._log.emit("Preserving the selected real-feature ranges…")
            return
        if stage in ("source_preparation", "source_preparation_cache") and event == "hit":
            completed = _INDICES["v32"] + 1
            if self._completed <= completed:
                self._log.emit("Reusing verified P/N source fits; checking current training spectra…")
            self._advance(completed)
            return
        if stage == "source_preparation":
            if event == "start":
                self.complete("load_raw")
            elif event == "complete":
                self.complete("v32")
            return
        stage = _ALIASES.get(stage, stage)
        if stage not in _INDICES or event not in ("start", "complete"):
            return
        index = _INDICES[stage]
        if event == "start":
            if index >= self._completed:
                self._log.emit(_LABELS[stage])
            self._advance(index)
        else:
            self._advance(index + 1)
