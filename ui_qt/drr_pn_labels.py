"""Display labels for saved P/N composites and their recorded source frames."""

SAVED_COMPOSITE_LABEL = "Saved composite · single-frame background"
SAVED_COMPOSITE_TOOLTIP = (
    "This saved composite contains one generated background spectrum. "
    "P/N source frame choices are recorded separately for this version."
)


def source_frame_summary(version) -> str:
    """Describe a saved version without inferring choices from current controls."""
    frame_labels = {"all": "all frames averaged", "first": "first frame", "last": "last frame"}
    p_which = getattr(version, "p_which", None)
    n_which = getattr(version, "n_which", None)
    if p_which not in frame_labels or n_which not in frame_labels:
        return "P/N source averaging: see saved version"
    return f"P: {frame_labels[p_which]}; N: {frame_labels[n_which]}"
