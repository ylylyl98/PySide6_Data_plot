from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from core import update_checker
from core.update_checker import (
    RELEASES_URL,
    SUMS_FILENAME,
    _validate_asset_url,
    expected_installer_name,
    newest_stable_release,
    parse_sums_checksum,
    parse_version,
)


class FakeResponse:
    def __init__(self, data, headers=None):
        if isinstance(data, str):
            data = data.encode('utf-8')
        self._io = io.BytesIO(data)
        if headers is None:
            headers = {'Content-Length': str(len(data))}
        self.headers = headers

    def read(self, size=-1):
        return self._io.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    def __init__(self, routes, error=None):
        self.routes = routes
        self.error = error
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        url = request.full_url if hasattr(request, 'full_url') else request.get_full_url()
        data = self.routes.get(url)
        if data is None:
            raise urllib.error.HTTPError(url, 404, 'Not Found', None, None)
        if callable(data):
            data = data()
        if isinstance(data, str):
            data = data.encode('utf-8')
        return FakeResponse(data)


def make_release(tag, draft=False, prerelease=False, assets=(), html_url=''):
    asset_list = [{'name': name, 'browser_download_url': url} for name, url in assets]
    return {
        'tag_name': tag,
        'draft': draft,
        'prerelease': prerelease,
        'html_url': html_url,
        'assets': asset_list,
    }


def installer_url(version):
    return (
        'https://github.com/ylylyl98/PySide6_Data_plot/releases/download/'
        'v' + update_checker.format_version(version) + '/'
        + expected_installer_name(version)
    )


def sums_url(version):
    return (
        'https://github.com/ylylyl98/PySide6_Data_plot/releases/download/'
        'v' + update_checker.format_version(version) + '/SHA256SUMS.txt'
    )


def release_payload(releases):
    return json.dumps(releases).encode('utf-8')


class ParseVersionTests(unittest.TestCase):
    def test_valid(self):
        cases = {
            '1.0.0': (1, 0, 0),
            'v1.2.3': (1, 2, 3),
            '0.0.0': (0, 0, 0),
            '10.20.30': (10, 20, 30),
        }
        for text, expected in cases.items():
            self.assertEqual(parse_version(text), expected)

    def test_invalid(self):
        bad = ('', '1', '1.2', '1.2.3.4', 'v', '1.2.x', '01.2.3', '1.02.3', '1.2.03', '1.2.3-beta', 'V1.2.3')
        for text in bad:
            self.assertIsNone(parse_version(text))

    def test_non_string(self):
        self.assertIsNone(parse_version(None))
        self.assertIsNone(parse_version(1))


class NamingTests(unittest.TestCase):
    def test_expected_installer_name(self):
        self.assertEqual(expected_installer_name((1, 2, 3)), 'DPTK-Setup-v1.2.3-Windows-x64.exe')


class SumsTests(unittest.TestCase):
    def test_finds_exact_filename(self):
        digest = 'a' * 64
        text = digest + '  other.exe' + chr(10) + digest + ' *DPTK-Setup-v1.0.0-Windows-x64.exe'
        self.assertEqual(parse_sums_checksum(text, 'DPTK-Setup-v1.0.0-Windows-x64.exe'), digest)

    def test_missing(self):
        self.assertIsNone(parse_sums_checksum('a' * 64 + '  other.exe', 'DPTK-Setup-v1.0.0-Windows-x64.exe'))

    def test_malformed_digest_ignored(self):
        self.assertIsNone(parse_sums_checksum('nothex  DPTK-Setup-v1.0.0-Windows-x64.exe', 'DPTK-Setup-v1.0.0-Windows-x64.exe'))


class ReleaseSelectionTests(unittest.TestCase):
    def test_excludes_draft_and_prerelease(self):
        releases = [
            make_release('v1.0.0'),
            make_release('v2.0.0', prerelease=True),
            make_release('v1.5.0', draft=True),
            make_release('v3.0.0'),
            {'not': 'a release'},
            'garbage',
        ]
        result = newest_stable_release(releases)
        self.assertIsNotNone(result)
        self.assertEqual(result.version, (3, 0, 0))

    def test_empty(self):
        self.assertIsNone(newest_stable_release([]))

    def test_ignores_malformed_tags(self):
        self.assertIsNone(newest_stable_release([make_release('not-a-version')]))


class CheckTests(unittest.TestCase):
    def _opener(self, releases, error=None):
        return FakeOpener({RELEASES_URL: release_payload(releases)}, error=error)

    def test_update_available(self):
        version = (1, 1, 0)
        releases = [make_release('v1.1.0', assets=[(expected_installer_name(version), installer_url(version)), ('SHA256SUMS.txt', sums_url(version))])]
        result = update_checker.check_for_update('1.0.0', opener=self._opener(releases))
        self.assertEqual(result.status, 'update_available')
        self.assertEqual(result.latest_version, version)
        self.assertEqual(result.installer_url, installer_url(version))
        self.assertEqual(result.sums_url, sums_url(version))

    def test_up_to_date(self):
        version = (1, 0, 0)
        releases = [make_release('v1.0.0', assets=[(expected_installer_name(version), installer_url(version)), ('SHA256SUMS.txt', sums_url(version))])]
        result = update_checker.check_for_update('1.0.0', opener=self._opener(releases))
        self.assertEqual(result.status, 'up_to_date')

    def test_no_downgrade(self):
        releases = [make_release('v0.9.0')]
        result = update_checker.check_for_update('2.0.0', opener=self._opener(releases))
        self.assertEqual(result.status, 'up_to_date')

    def test_unexpected_installer_ignored(self):
        releases = [make_release('v1.1.0', assets=[('Other-Setup-v1.1.0-Windows-x64.exe', 'https://example.com/other.exe'), ('SHA256SUMS.txt', sums_url((1, 1, 0)))])]
        result = update_checker.check_for_update('1.0.0', opener=self._opener(releases))
        self.assertEqual(result.status, 'up_to_date')

    def test_missing_sums_ignored(self):
        version = (1, 1, 0)
        releases = [make_release('v1.1.0', assets=[(expected_installer_name(version), installer_url(version))])]
        result = update_checker.check_for_update('1.0.0', opener=self._opener(releases))
        self.assertEqual(result.status, 'up_to_date')

    def test_prerelease_newer_ignored(self):
        stable_version = (1, 0, 1)
        releases = [
            make_release('v1.1.0', prerelease=True, assets=[(expected_installer_name((1, 1, 0)), installer_url((1, 1, 0))), ('SHA256SUMS.txt', sums_url((1, 1, 0)))]),
            make_release('v1.0.1', assets=[(expected_installer_name(stable_version), installer_url(stable_version)), ('SHA256SUMS.txt', sums_url(stable_version))]),
        ]
        result = update_checker.check_for_update('1.0.0', opener=self._opener(releases))
        self.assertEqual(result.status, 'update_available')
        self.assertEqual(result.latest_version, stable_version)

    def test_offline(self):
        result = update_checker.check_for_update('1.0.0', opener=FakeOpener({}, error=urllib.error.URLError('no network')))
        self.assertEqual(result.status, 'error')
        self.assertEqual(result.error_kind, 'offline')

    def test_timeout(self):
        result = update_checker.check_for_update('1.0.0', opener=FakeOpener({}, error=TimeoutError('timed out')))
        self.assertEqual(result.status, 'error')
        self.assertEqual(result.error_kind, 'timeout')

    def test_http_error(self):
        error = urllib.error.HTTPError(RELEASES_URL, 500, 'Server Error', None, None)
        result = update_checker.check_for_update('1.0.0', opener=FakeOpener({}, error=error))
        self.assertEqual(result.status, 'error')
        self.assertEqual(result.error_kind, 'http')

    def test_invalid_json(self):
        result = update_checker.check_for_update('1.0.0', opener=FakeOpener({RELEASES_URL: b'not json'}))
        self.assertEqual(result.status, 'error')
        self.assertEqual(result.error_kind, 'json')

    def test_json_not_list(self):
        result = update_checker.check_for_update('1.0.0', opener=FakeOpener({RELEASES_URL: b'{"x": 1}'}))
        self.assertEqual(result.status, 'error')
        self.assertEqual(result.error_kind, 'json')

    def test_sets_user_agent(self):
        releases = [make_release('v1.0.0')]
        opener = self._opener(releases)
        update_checker.check_for_update('1.0.0', opener=opener)
        self.assertEqual(len(opener.requests), 1)
        self.assertEqual(opener.requests[0].get_header('User-agent'), update_checker.USER_AGENT)


class DownloadTests(unittest.TestCase):
    def test_success(self):
        version = (1, 1, 0)
        name = expected_installer_name(version)
        payload = b'installer-bytes'
        digest = hashlib.sha256(payload).hexdigest()
        routes = {installer_url(version): payload, sums_url(version): digest + '  ' + name}
        opener = FakeOpener(routes)
        with tempfile.TemporaryDirectory() as td:
            result = update_checker.download_installer(installer_url(version), sums_url(version), name, td, opener=opener)
            self.assertEqual(result.status, 'ok')
            path = Path(result.installer_path)
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(result.expected_sha256, digest)

    def test_rejects_unexpected_filename(self):
        with tempfile.TemporaryDirectory() as td:
            result = update_checker.download_installer(
                'https://example.com/evil.exe', sums_url((1, 0, 0)), 'evil.exe', td,
                opener=FakeOpener({}),
            )
            self.assertEqual(result.status, 'error')
            self.assertEqual(result.error_kind, 'unexpected_asset')

    def test_checksum_mismatch(self):
        version = (1, 1, 0)
        name = expected_installer_name(version)
        routes = {installer_url(version): b'installer-bytes', sums_url(version): ('0' * 64) + '  ' + name}
        opener = FakeOpener(routes)
        with tempfile.TemporaryDirectory() as td:
            result = update_checker.download_installer(installer_url(version), sums_url(version), name, td, opener=opener)
            self.assertEqual(result.status, 'error')
            self.assertEqual(result.error_kind, 'checksum_mismatch')
            self.assertFalse((Path(td) / name).exists())
            self.assertFalse((Path(td) / (name + '.part')).exists())

    def test_missing_checksum(self):
        version = (1, 1, 0)
        name = expected_installer_name(version)
        routes = {installer_url(version): b'installer-bytes', sums_url(version): ('a' * 64) + '  other.exe'}
        opener = FakeOpener(routes)
        with tempfile.TemporaryDirectory() as td:
            result = update_checker.download_installer(installer_url(version), sums_url(version), name, td, opener=opener)
            self.assertEqual(result.status, 'error')
            self.assertEqual(result.error_kind, 'checksum_missing')
            self.assertFalse((Path(td) / (name + '.part')).exists())

    def test_offline(self):
        version = (1, 1, 0)
        name = expected_installer_name(version)
        opener = FakeOpener({}, error=urllib.error.URLError('no network'))
        with tempfile.TemporaryDirectory() as td:
            result = update_checker.download_installer(installer_url(version), sums_url(version), name, td, opener=opener)
            self.assertEqual(result.status, 'error')
            self.assertEqual(result.error_kind, 'offline')
            self.assertFalse((Path(td) / (name + '.part')).exists())

    def test_sums_http_error_cleans_part(self):
        version = (1, 1, 0)
        name = expected_installer_name(version)
        opener = FakeOpener({installer_url(version): b'installer-bytes'})
        with tempfile.TemporaryDirectory() as td:
            result = update_checker.download_installer(installer_url(version), sums_url(version), name, td, opener=opener)
            self.assertEqual(result.status, 'error')
            self.assertEqual(result.error_kind, 'http')
            self.assertFalse((Path(td) / (name + '.part')).exists())

    def test_reports_progress(self):
        version = (1, 1, 0)
        name = expected_installer_name(version)
        payload = b'installer-bytes'
        digest = hashlib.sha256(payload).hexdigest()
        routes = {installer_url(version): payload, sums_url(version): digest + '  ' + name}
        progress_calls = []
        opener = FakeOpener(routes)
        with tempfile.TemporaryDirectory() as td:
            update_checker.download_installer(installer_url(version), sums_url(version), name, td, opener=opener, progress=progress_calls.append)
        self.assertTrue(progress_calls)
        self.assertLessEqual(max(progress_calls), 100)


class AssetUrlValidationTests(unittest.TestCase):
    def _url(self, tag, basename):
        return 'https://github.com/ylylyl98/PySide6_Data_plot/releases/download/' + tag + '/' + basename

    def test_valid_installer(self):
        url = self._url('v1.2.3', 'DPTK-Setup-v1.2.3-Windows-x64.exe')
        self.assertTrue(_validate_asset_url(url, (1, 2, 3), 'DPTK-Setup-v1.2.3-Windows-x64.exe'))

    def test_valid_sums(self):
        url = self._url('v1.2.3', SUMS_FILENAME)
        self.assertTrue(_validate_asset_url(url, (1, 2, 3), SUMS_FILENAME))

    def test_rejects_http(self):
        url = self._url('v1.2.3', SUMS_FILENAME).replace('https://', 'http://')
        self.assertFalse(_validate_asset_url(url, (1, 2, 3), SUMS_FILENAME))

    def test_rejects_wrong_host(self):
        url = 'https://evil.com/ylylyl98/PySide6_Data_plot/releases/download/v1.2.3/' + SUMS_FILENAME
        self.assertFalse(_validate_asset_url(url, (1, 2, 3), SUMS_FILENAME))

    def test_rejects_host_prefix_trick(self):
        url = 'https://github.com.evil.com/ylylyl98/PySide6_Data_plot/releases/download/v1.2.3/' + SUMS_FILENAME
        self.assertFalse(_validate_asset_url(url, (1, 2, 3), SUMS_FILENAME))

    def test_rejects_userinfo_trick(self):
        url = 'https://github.com@evil.com/ylylyl98/PySide6_Data_plot/releases/download/v1.2.3/' + SUMS_FILENAME
        self.assertFalse(_validate_asset_url(url, (1, 2, 3), SUMS_FILENAME))

    def test_rejects_mismatched_tag(self):
        url = self._url('v9.9.9', 'DPTK-Setup-v1.2.3-Windows-x64.exe')
        self.assertFalse(_validate_asset_url(url, (1, 2, 3), 'DPTK-Setup-v1.2.3-Windows-x64.exe'))

    def test_rejects_non_v_tag(self):
        url = self._url('1.2.3', SUMS_FILENAME)
        self.assertFalse(_validate_asset_url(url, (1, 2, 3), SUMS_FILENAME))

    def test_rejects_wrong_basename(self):
        url = self._url('v1.2.3', 'other.exe')
        self.assertFalse(_validate_asset_url(url, (1, 2, 3), SUMS_FILENAME))

    def test_rejects_query(self):
        url = self._url('v1.2.3', SUMS_FILENAME) + '?download=1'
        self.assertFalse(_validate_asset_url(url, (1, 2, 3), SUMS_FILENAME))

    def test_rejects_fragment(self):
        url = self._url('v1.2.3', SUMS_FILENAME) + '#frag'
        self.assertFalse(_validate_asset_url(url, (1, 2, 3), SUMS_FILENAME))

    def test_rejects_traversal(self):
        url = self._url('v1.2.3', '..') + '/..' + '/' + SUMS_FILENAME
        self.assertFalse(_validate_asset_url(url, (1, 2, 3), SUMS_FILENAME))

    def test_rejects_extra_segment(self):
        url = self._url('v1.2.3', 'sub') + '/' + SUMS_FILENAME
        self.assertFalse(_validate_asset_url(url, (1, 2, 3), SUMS_FILENAME))


class UrlIntegrationTests(unittest.TestCase):
    def _opener(self, releases, error=None):
        return FakeOpener({RELEASES_URL: release_payload(releases)}, error=error)

    def test_check_rejects_installer_with_wrong_tag(self):
        version = (1, 1, 0)
        bad_installer = installer_url((9, 9, 9))
        releases = [make_release('v1.1.0', assets=[(expected_installer_name(version), bad_installer), (SUMS_FILENAME, sums_url(version))])]
        result = update_checker.check_for_update('1.0.0', opener=self._opener(releases))
        self.assertEqual(result.status, 'up_to_date')

    def test_check_rejects_sums_with_query(self):
        version = (1, 1, 0)
        bad_sums = sums_url(version) + '?download=1'
        releases = [make_release('v1.1.0', assets=[(expected_installer_name(version), installer_url(version)), (SUMS_FILENAME, bad_sums)])]
        result = update_checker.check_for_update('1.0.0', opener=self._opener(releases))
        self.assertEqual(result.status, 'up_to_date')

    def test_download_rejects_installer_outside_github(self):
        version = (1, 1, 0)
        name = expected_installer_name(version)
        with tempfile.TemporaryDirectory() as td:
            result = update_checker.download_installer(
                'https://example.com/' + name, sums_url(version), name, td,
                opener=FakeOpener({}),
            )
            self.assertEqual(result.status, 'error')
            self.assertEqual(result.error_kind, 'unexpected_asset')

    def test_download_rejects_sums_with_wrong_tag(self):
        version = (1, 1, 0)
        name = expected_installer_name(version)
        with tempfile.TemporaryDirectory() as td:
            result = update_checker.download_installer(
                installer_url(version), sums_url((9, 9, 9)), name, td,
                opener=FakeOpener({}),
            )
            self.assertEqual(result.status, 'error')
            self.assertEqual(result.error_kind, 'unexpected_asset')


if __name__ == '__main__':
    unittest.main()
