"""Remove recognized acquisition bookkeeping from PNG titles only.

Unknown conditions are retained. Never use this function for paths or source
metadata: the unsimplified title is the provenance record.
"""
import re

_DURATION = r'\d+(?:[p.]\d+)?\s*(?:ms|us|µs|s)'
_EXPOSURE = rf'{_DURATION}\s*[x×]\s*\d+'


def display_title(original: str) -> str:
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
    for index, value in enumerate(protected):
        result = result.replace(f'PROTECTEDTITLETOKEN{index}TOKEN', value)
    return result
