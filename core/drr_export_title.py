"""Remove recognized acquisition bookkeeping from PNG titles only.

Unknown conditions are retained. Never use this function for paths or source
metadata: the unsimplified title is the provenance record.
"""
import re

_DURATION = r'\d+(?:[p.]\d+)?\s*(?:ms|us|µs|s)'
_EXPOSURE = rf'{_DURATION}\s*[x×]\s*\d+'
TITLE_POLICY_VERSION = 3

# These are generated product labels, not arbitrary parenthesized conditions.
# Mixed-coordinate labels may contain the gate unit's own parentheses.
_PRODUCT = (r'\((?:DR/R|d\(DR/R\)/dE|d2\(DR/R\)/dE2|'
            r'd2\(DR/R\)/\(dE\s+d(?:[^()]|\([^()]*\))+\)|'
            r'dE|d2E|dXdY|Advanced first derivative,\s*dE)\)')
_PRODUCT_CHAIN = re.compile(rf'(?<!\S){_PRODUCT}(?:\s+{_PRODUCT})*\s*$', re.IGNORECASE)


def background_label(processing: dict | None) -> str:
    """Describe the applied reference type, never infer it from a filename."""
    if not isinstance(processing, dict):
        return ""
    mode = processing.get("background_mode") or processing.get("baseline_selection", "")
    if not isinstance(mode, str):
        return ""
    self_modes = {"self-first": "First", "Self (first frame)": "First",
                  "self-last": "Last", "Self (last frame)": "Last"}
    if mode in self_modes:
        return self_modes[mode]
    if mode in {"imported", "None", "Imported map"}:
        return ""
    if mode in {"fitted", "Fitted (sample)"} or processing.get("selection_method") == "fitted_sample":
        return "Fitted"
    assignments = processing.get("drr_background_assignments") or ()
    if (not isinstance(assignments, (tuple, list))
            or any(not isinstance(item, dict) or item.get("regions") for item in assignments)):
        return ""
    if any(not isinstance(item.get("baseline_mode", ""), str) for item in assignments):
        return ""
    modes = {item.get("baseline_mode", "") for item in assignments}
    if modes and modes <= self_modes.keys():
        labels = {self_modes[item] for item in modes}
        return next(iter(labels)) if len(labels) == 1 else ""
    if any(item != "External" for item in modes):
        return ""
    kind = processing.get("background_kind")
    return ({"p": "P", "n": "N", "pn": "P+N", "bn": "BN", "fitted": "Fitted"}.get(kind, "")
            if isinstance(kind, str) else "")


def imported_background_label(metadata: dict | None) -> str:
    """Retain a verified short label when a saved map is exported again."""
    if not isinstance(metadata, dict):
        return ""
    label = background_label(metadata.get("processing"))
    if label:
        return label
    saved = metadata.get("png_background")
    return saved if isinstance(saved, str) and saved in {"P", "N", "P+N", "BN", "Fitted", "First", "Last"} else ""


def display_title(original: str, *, background: str = "") -> str:
    protected = []
    def protect(match):
        protected.append(match.group())
        return f'PROTECTEDTITLETOKEN{len(protected)-1}TOKEN'
    text = re.sub(r'\$[^$]*\$', protect, str(original))
    # Explicit file-valued keys only; never match bare BG or a gate equation.
    text = re.sub(r'''(?i)\b(?:background|baseline)\s*(?:file)?\s*[:=]\s*(?:"[^"]*"|'[^']*'|[^\s|~]+\.(?:csv|xlsx|dat|txt))''', '', text)
    text = re.sub(r'(?i)\b(?:background|baseline)\s+(?:method|mode)\s*[:=]\s*(?:self_first|self_last|external|first|last)(?!\w)', '', text)
    # Protect unknown explicit assignments, including their values. A value
    # such as custom=END is an experimental condition, not an END marker.
    text = re.sub(r'\b[A-Za-z][A-Za-z0-9-]*\s*[:=]\s*[^\s|~_]+', protect, text)
    text = re.sub(r'(?<!\w)20\d{2}-\d{2}-\d{2}[_ T]\d{2}[-:]\d{2}[-:]\d{2}(?:\.\d+)?Z?(?!\w)', '', text)
    text = re.sub(r'(?i)\.(?:csv|xlsx?|dat|txt|png|json)(?=$|\s|[|~])', '', text)
    fields = re.split(r'[_~|]', text)
    kept = []
    for field in fields:
        field = field.strip()
        if re.fullmatch(r'(?i)(?:session\s*[:=#-]?\s*\d+|END)', field):
            continue
        if re.fullmatch(rf'(?i)(?:(?:exposure|integration(?: time)?)\s*[:=]?\s*)?{_DURATION}(?:\s*[x×]\s*\d+)?', field):
            continue
        if re.fullmatch(r'(?i)(?:external background|self (?:first|last)(?: frame)?|(?:background|baseline) (?:method|mode)\s*[:=]\s*(?:external|self_first|self_last|first|last))', field):
            continue
        # Only unambiguously date/time-shaped bookkeeping, not arbitrary IDs.
        field = re.sub(r'(?<!\w)20\d{2}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?(?!\w)', '', field)
        field = re.sub(r'(?<!\w)\d{2}:\d{2}:\d{2}(?:\.\d+)?(?!\w)', '', field)
        field = re.sub(rf'(?i)(?<![\w=]){_EXPOSURE}(?!\w)', '', field)
        field = re.sub(r'(?i)(?<!\w)session\s*[:=#-]?\s*\d+(?!\w)', '', field)
        field = re.sub(r'(?i)(?<!\w)END(?!\w)', '', field)
        # Known processing_run suffix: keep product identity and remove its
        # background method / averaging bookkeeping, including derivatives.
        field = re.sub(r'(?i)(\((?:DR/R|d\(DR/R\)/dE|d2\(DR/R\)/dE2)),?\s+(?:first|last|external),\s*avg\s+\d+\)', r'\1)', field)
        if field.strip():
            kept.append(field.strip())
    result = re.sub(r'\s+', ' ', ' '.join(kept)).strip()
    # The colorbar identifies the quantity. Remove generated trailing product
    # labels and the camera token from the PNG, preserving source provenance.
    result = re.sub(r'(?i)(?<!\S)PIXIS(?!\S)', '', result)
    result = _PRODUCT_CHAIN.sub('', result).strip()
    result = re.sub(r'\s+', ' ', result).strip()
    for index, value in enumerate(protected):
        result = result.replace(f'PROTECTEDTITLETOKEN{index}TOKEN', value)
    return f'{result} · {background}' if result and background else result or background
