"""Update checking and verified installer download for DPTK Desktop.

This module is intentionally dependency-free: it uses only the Python
standard library.  Network access is injectable so the parsing and decision
logic can be unit-tested without contacting GitHub.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

RELEASES_URL = "https://api.github.com/repos/ylylyl98/PySide6_Data_plot/releases"
SUMS_FILENAME = "SHA256SUMS.txt"
INSTALLER_PREFIX = "DPTK-Setup-v"
INSTALLER_SUFFIX = "-Windows-x64.exe"
USER_AGENT = "DPTK-Desktop-Updater"
DEFAULT_TIMEOUT = 10.0
_CHUNK_SIZE = 1024 * 256


def parse_version(text: str) -> tuple[int, int, int] | None:
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    if not candidate:
        return None
    if candidate.startswith("v"):
        candidate = candidate[1:]
    parts = candidate.split(".")
    if len(parts) != 3:
        return None
    values: list[int] = []
    for part in parts:
        if not part or not part.isascii() or not part.isdigit():
            return None
        if len(part) > 1 and part[0] == "0":
            return None
        values.append(int(part))
    return (values[0], values[1], values[2])


def format_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def expected_installer_name(version: tuple[int, int, int]) -> str:
    return INSTALLER_PREFIX + format_version(version) + INSTALLER_SUFFIX


def _validate_asset_url(
    url: str,
    version: tuple[int, int, int],
    expected_basename: str,
) -> bool:
    """Return True only for a trusted GitHub release asset URL.

    The URL must be HTTPS on exactly ``github.com``, carry no query or
    fragment, and its path must be exactly the GitHub download path for this
    repository and release tag ending in the expected basename.  Anything
    else (other hosts, path traversal, mismatched tags, extra segments) is
    rejected.
    """
    if not isinstance(url, str):
        return False
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    if parts.scheme != "https":
        return False
    if parts.netloc != "github.com":
        return False
    if parts.query or parts.fragment:
        return False
    expected_path = (
        "/ylylyl98/PySide6_Data_plot/releases/download/"
        + "v"
        + format_version(version)
        + "/"
        + expected_basename
    )
    return parts.path == expected_path


@dataclass(frozen=True)
class ReleaseInfo:
    tag: str
    version: tuple[int, int, int]
    draft: bool
    prerelease: bool
    html_url: str
    body: str
    asset_urls: dict[str, str]


@dataclass(frozen=True)
class CheckResult:
    status: str
    current_version: tuple[int, int, int]
    latest_version: tuple[int, int, int] | None = None
    release: ReleaseInfo | None = None
    installer_url: str | None = None
    sums_url: str | None = None
    message: str = ""
    error_kind: str | None = None


@dataclass(frozen=True)
class DownloadResult:
    status: str
    installer_path: str | None = None
    expected_sha256: str | None = None
    message: str = ""
    error_kind: str | None = None


class UpdateCheckError(Exception):
    def __init__(self, kind: str, user_message: str):
        super().__init__(user_message)
        self.kind = kind
        self.user_message = user_message


def parse_release(data: object) -> ReleaseInfo | None:
    if not isinstance(data, dict):
        return None
    raw_tag = data.get("tag_name")
    if not isinstance(raw_tag, str):
        return None
    version = parse_version(raw_tag)
    if version is None:
        return None
    html_url = data.get("html_url")
    body = data.get("body")
    asset_urls: dict[str, str] = {}
    assets = data.get("assets")
    if isinstance(assets, list):
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            name = asset.get("name")
            url = asset.get("browser_download_url")
            if isinstance(name, str) and isinstance(url, str) and name and url:
                asset_urls[name] = url
    return ReleaseInfo(
        tag=raw_tag,
        version=version,
        draft=bool(data.get("draft", False)),
        prerelease=bool(data.get("prerelease", False)),
        html_url=str(html_url) if isinstance(html_url, str) else "",
        body=str(body) if isinstance(body, str) else "",
        asset_urls=asset_urls,
    )

def newest_stable_release(releases: Iterable[object]) -> ReleaseInfo | None:
    best: ReleaseInfo | None = None
    for data in releases:
        release = parse_release(data)
        if release is None or release.draft or release.prerelease:
            continue
        if best is None or release.version > best.version:
            best = release
    return best


def expected_assets(release: ReleaseInfo) -> tuple[str, str] | None:
    installer = expected_installer_name(release.version)
    installer_url = release.asset_urls.get(installer)
    sums_url = release.asset_urls.get(SUMS_FILENAME)
    if installer_url is None or sums_url is None:
        return None
    if not _validate_asset_url(installer_url, release.version, installer):
        return None
    if not _validate_asset_url(sums_url, release.version, SUMS_FILENAME):
        return None
    return (installer_url, sums_url)


def _open(opener, url: str, timeout: float):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        return opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise UpdateCheckError("http", "The update server responded with HTTP " + str(exc.code) + ".") from exc
    except TimeoutError as exc:
        raise UpdateCheckError("timeout", "The update check timed out.") from exc
    except urllib.error.URLError as exc:
        raise UpdateCheckError("offline", "Could not reach the update server. Check your connection and try again.") from exc
    except OSError as exc:
        raise UpdateCheckError("offline", "Could not reach the update server. Check your connection and try again.") from exc


def _fetch_bytes(opener, url: str, timeout: float) -> bytes:
    with _open(opener, url, timeout) as response:
        return response.read()


def check_for_update(
    current_version: str,
    opener=None,
    timeout: float = DEFAULT_TIMEOUT,
) -> CheckResult:
    current = parse_version(current_version)
    if current is None:
        return CheckResult(
            status="error",
            current_version=(0, 0, 0),
            message="The installed application version is invalid.",
            error_kind="invalid_version",
        )
    if opener is None:
        opener = urllib.request.build_opener()
    try:
        raw = _fetch_bytes(opener, RELEASES_URL, timeout)
    except UpdateCheckError as exc:
        return CheckResult(
            status="error",
            current_version=current,
            message=exc.user_message,
            error_kind=exc.kind,
        )
    try:
        releases = json.loads(raw.decode("utf-8", "replace"))
    except (json.JSONDecodeError, ValueError):
        return CheckResult(
            status="error",
            current_version=current,
            message="The update server returned an invalid response.",
            error_kind="json",
        )
    if not isinstance(releases, list):
        return CheckResult(
            status="error",
            current_version=current,
            message="The update server returned an invalid response.",
            error_kind="json",
        )
    release = newest_stable_release(releases)
    if release is None or release.version <= current:
        return CheckResult(
            status="up_to_date",
            current_version=current,
            latest_version=release.version if release else None,
            release=release,
            message="You are running the latest version.",
        )
    assets = expected_assets(release)
    if assets is None:
        return CheckResult(
            status="up_to_date",
            current_version=current,
            latest_version=release.version,
            release=release,
            message="A newer release is missing its installer files.",
        )
    installer_url, sums_url = assets
    return CheckResult(
        status="update_available",
        current_version=current,
        latest_version=release.version,
        release=release,
        installer_url=installer_url,
        sums_url=sums_url,
        message="Version " + format_version(release.version) + " is available.",
    )


def parse_sums_checksum(text: str, expected_filename: str) -> str | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) < 2:
            continue
        digest = tokens[0]
        filename = " ".join(tokens[1:])
        if filename.startswith("*"):
            filename = filename[1:].strip()
        if filename != expected_filename:
            continue
        if len(digest) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in digest):
            return digest.lower()
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _stream_download(
    opener,
    url: str,
    dest_path: Path,
    timeout: float,
    progress: Callable[[int], None] | None,
) -> None:
    with _open(opener, url, timeout) as response:
        total: int | None = None
        headers = getattr(response, "headers", None)
        if headers is not None and hasattr(headers, "get"):
            length = headers.get("Content-Length")
            if length is not None:
                try:
                    total = int(str(length))
                except ValueError:
                    total = None
        downloaded = 0
        with open(dest_path, "wb") as handle:
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if progress is not None and total is not None and total > 0:
                    progress(min(100, downloaded * 100 // total))


def download_installer(
    installer_url: str,
    sums_url: str,
    expected_filename: str,
    dest_dir: str | Path,
    opener=None,
    timeout: float = DEFAULT_TIMEOUT,
    progress: Callable[[int], None] | None = None,
) -> DownloadResult:
    parsed_name_version = parse_version(
        expected_filename[len(INSTALLER_PREFIX):-len(INSTALLER_SUFFIX)]
        if expected_filename.startswith(INSTALLER_PREFIX) and expected_filename.endswith(INSTALLER_SUFFIX)
        else ""
    )
    if parsed_name_version is None or expected_filename != expected_installer_name(parsed_name_version):
        return DownloadResult(
            status="error",
            message="The requested installer filename is not trusted.",
            error_kind="unexpected_asset",
        )
    if not _validate_asset_url(installer_url, parsed_name_version, expected_filename):
        return DownloadResult(
            status="error",
            message="The installer download link is not trusted.",
            error_kind="unexpected_asset",
        )
    if not _validate_asset_url(sums_url, parsed_name_version, SUMS_FILENAME):
        return DownloadResult(
            status="error",
            message="The checksum download link is not trusted.",
            error_kind="unexpected_asset",
        )
    if opener is None:
        opener = urllib.request.build_opener()
    dest = Path(dest_dir)
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError:
        return DownloadResult(
            status="error",
            message="Could not create the download folder.",
            error_kind="filesystem",
        )
    final_path = dest / expected_filename
    part_path = dest / (expected_filename + ".part")
    try:
        _stream_download(opener, installer_url, part_path, timeout, progress)
    except UpdateCheckError as exc:
        _safe_unlink(part_path)
        return DownloadResult(status="error", message=exc.user_message, error_kind=exc.kind)
    try:
        sums_raw = _fetch_bytes(opener, sums_url, timeout)
    except UpdateCheckError as exc:
        _safe_unlink(part_path)
        return DownloadResult(status="error", message=exc.user_message, error_kind=exc.kind)
    expected_digest = parse_sums_checksum(sums_raw.decode("utf-8", "replace"), expected_filename)
    if expected_digest is None:
        _safe_unlink(part_path)
        return DownloadResult(
            status="error",
            message="Could not find a checksum for the installer.",
            error_kind="checksum_missing",
        )
    if sha256_file(part_path) != expected_digest:
        _safe_unlink(part_path)
        return DownloadResult(
            status="error",
            message="The downloaded installer failed checksum verification.",
            error_kind="checksum_mismatch",
        )
    try:
        part_path.replace(final_path)
    except OSError:
        _safe_unlink(part_path)
        return DownloadResult(
            status="error",
            message="Could not finalize the downloaded installer.",
            error_kind="filesystem",
        )
    return DownloadResult(
        status="ok",
        installer_path=str(final_path),
        expected_sha256=expected_digest,
        message="Downloaded " + expected_filename + ".",
    )
