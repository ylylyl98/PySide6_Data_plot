"""Per-machine location for temporary PowerPoint copies and recovery files."""
import json
import os
from pathlib import Path
import tempfile


def _configured_directory():
    override = os.environ.get('DPTK_PPT_TEMP_DIR')
    if override is not None:
        if not override.strip():
            raise RuntimeError('DPTK_PPT_TEMP_DIR must name an existing writable directory.')
        return Path(override).expanduser()
    local = os.environ.get('LOCALAPPDATA')
    if not local:
        return None
    settings = Path(local) / 'DPTK' / 'PowerPoint' / 'settings.json'
    if not settings.exists():
        return None
    try:
        data = json.loads(settings.read_text(encoding='utf-8'))
        value = data['temp_directory']
        if not isinstance(value, str) or not value.strip():
            raise ValueError('temp_directory must be a non-empty string')
        return Path(value).expanduser()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(f'Could not read the PowerPoint temporary-directory setting: {exc}') from exc


def temporary_directory():
    """Create an automatically cleaned temporary directory, honoring local config."""
    directory = _configured_directory()
    try:
        return tempfile.TemporaryDirectory(prefix='dptk-ppt-', dir=directory)
    except (OSError, ValueError) as exc:
        if directory is None:
            raise RuntimeError(f'Could not create a temporary PowerPoint directory: {exc}') from exc
        raise RuntimeError(f'The configured PowerPoint temporary directory is unavailable: {directory} ({exc})') from exc
