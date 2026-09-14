"""Display persisted timestamps in the operating system's local timezone."""

from datetime import datetime


def format_local_timestamp(value: str) -> str:
    """Convert offset-aware ISO timestamps; keep legacy local values local."""
    text = str(value or '').strip()
    if len(text) <= 10:
        return text
    try:
        stamp = datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError:
        return text
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone()
    return stamp.strftime('%Y-%m-%d %H:%M')
