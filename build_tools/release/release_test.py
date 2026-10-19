"""Release tooling checks: local fixtures only, no publishing or network access."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile

from build_tools.make_release import git, prepare_release
from build_tools.publish_release import (
    github_release, publish_github, pypi_has_wheel, release_wheel, sha256,
)
from build_tools.versioning import (
    is_prerelease, next_dev_version, read_package_version, validate_release_version, validate_tag,
)


class VersionTest(unittest.TestCase):
    def test_releases_and_bumps(self):
        for version, following, prerelease in (
            ('0.3.0', '0.4.0.dev0', False),
            ('1.2.3a0', '1.2.3a1.dev0', True),
            ('1.2.3b9', '1.2.3b10.dev0', True),
            ('1.2.3rc1', '1.2.3rc2.dev0', True),
        ):
            with self.subTest(version=version):
                self.assertEqual(validate_release_version(version), version)
                self.assertEqual(next_dev_version(version), following)
                self.assertEqual(is_prerelease(version), prerelease)

    def test_reject_noncanonical_or_nonrelease_versions(self):
        for version in ('v1.2.3', '1.2', '01.2.3', '1.2.3-rc1', '1.2.3rc01',
                        '1.2.3.dev0', '1.2.3.post1', '1.2.3+local', '1.2.3\n'):
            with self.subTest(version=version), self.assertRaises(ValueError):
                validate_release_version(version)

    def test_checked_in_version_and_tag_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            version_file = root / 'version.json'
            version_file.write_text('{"package-version": "1.2.3rc1"}')
            with patch.dict(os.environ, HRX_SOLUTIONS_PACKAGE_VERSION='9.9.9'):
                self.assertEqual(read_package_version(root), '1.2.3rc1')
            self.assertEqual(validate_tag('v1.2.3rc1', root), '1.2.3rc1')
            for tag in ('1.2.3rc1', 'v1.2.3', 'v1.2.3.dev0'):
                with self.assertRaises(ValueError):
                    validate_tag(tag, root)
            version_file.write_text('{"package-version": "1.2.3rc2.dev0"}')
            self.assertEqual(read_package_version(root), '1.2.3rc2.dev0')
            version_file.write_text('{"package-version": "garbage"}')
            with self.assertRaises(ValueError):
                read_package_version(root)


class PreparationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        git(self.root, 'init', '-b', 'main')
        for key, value in (('user.name', 'Release Test'), ('user.email', 'test@example.invalid'),
                           ('commit.gpgsign', 'false'), ('tag.gpgsign', 'false'),
                           ('core.hooksPath', '/dev/null')):
            git(self.root, 'config', key, value)
        (self.root / 'version.json').write_text('{"package-version": "0.3.0.dev0"}\n')
        git(self.root, 'add', '.')
        git(self.root, 'commit', '-m', 'Initial version')
        self.initial = git(self.root, 'rev-parse', 'HEAD')

    def prepare(self, version='0.3.0', **options):
        with redirect_stdout(io.StringIO()):
            prepare_release(version, root=self.root, **options)

    def test_dry_run_has_no_effect(self):
        before = (self.root / 'version.json').read_bytes()
        self.prepare(bump_dev=True, dry_run=True)
        self.assertEqual((self.root / 'version.json').read_bytes(), before)
        self.assertEqual(git(self.root, 'rev-parse', 'HEAD'), self.initial)
        self.assertEqual(git(self.root, 'tag', '--list'), '')
        self.assertEqual(git(self.root, 'status', '--porcelain'), '')

    def test_release_and_development_commits(self):
        self.prepare(bump_dev=True)
        self.assertEqual(read_package_version(self.root), '0.4.0.dev0')
        self.assertEqual(json.loads(git(self.root, 'show', 'v0.3.0:version.json'))['package-version'], '0.3.0')
        self.assertEqual(git(self.root, 'cat-file', '-t', 'v0.3.0'), 'tag')
        self.assertEqual(git(self.root, 'rev-list', '--count', 'HEAD'), '3')
        self.assertIn('Signed-off-by: Release Test', git(self.root, 'log', '-1', '--format=%B'))
        self.assertEqual(git(self.root, 'status', '--porcelain'), '')

    def test_prerelease_bump(self):
        self.prepare('0.3.0rc1', bump_dev=True)
        self.assertEqual(read_package_version(self.root), '0.3.0rc2.dev0')
        self.assertEqual(json.loads(git(self.root, 'show', 'v0.3.0rc1:version.json'))['package-version'], '0.3.0rc1')

    def test_without_bump(self):
        self.prepare()
        self.assertEqual(read_package_version(self.root), '0.3.0')
        self.assertEqual(git(self.root, 'rev-parse', 'v0.3.0^{}'), git(self.root, 'rev-parse', 'HEAD'))

    def test_already_recorded_release_needs_no_empty_commit(self):
        (self.root / 'version.json').write_text('{"package-version": "0.3.0"}')
        git(self.root, 'commit', '-am', 'Set release version')
        before = git(self.root, 'rev-parse', 'HEAD')
        self.prepare()
        self.assertEqual(git(self.root, 'rev-parse', 'HEAD'), before)

    def test_dirty_tree_rejected_even_for_dry_run(self):
        (self.root / 'untracked').write_text('user work')
        for dry_run in (False, True):
            with self.assertRaisesRegex(ValueError, 'not clean'):
                self.prepare(dry_run=dry_run)
        self.assertEqual(git(self.root, 'rev-parse', 'HEAD'), self.initial)

    def test_existing_tag_rejected_before_mutation(self):
        git(self.root, 'tag', 'v0.3.0')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.prepare()
        self.assertEqual(read_package_version(self.root), '0.3.0.dev0')

    def test_wrong_branch_and_detached_head_rejected(self):
        git(self.root, 'switch', '-c', 'feature')
        with self.assertRaisesRegex(ValueError, 'main branch'):
            self.prepare()
        git(self.root, 'checkout', '--detach')
        with self.assertRaisesRegex(ValueError, 'main branch'):
            self.prepare()


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.wheel = self.directory / 'hrx_solutions-0.3.0-py3-none-manylinux_2_28_x86_64.whl'
        with zipfile.ZipFile(self.wheel, 'w') as archive:
            archive.writestr('hrx_solutions-0.3.0.dist-info/METADATA',
                             'Metadata-Version: 2.4\nName: hrx-solutions\nVersion: 0.3.0\n')

    def test_filename_and_metadata(self):
        self.assertEqual(release_wheel(self.directory, '0.3.0'), self.wheel)
        with self.assertRaises(ValueError):
            release_wheel(self.directory, '0.3.1')
        with zipfile.ZipFile(self.wheel, 'w') as archive:
            archive.writestr('hrx_solutions-0.3.0.dist-info/METADATA', 'Name: other\nVersion: 0.3.0\n')
        with self.assertRaisesRegex(ValueError, 'metadata'):
            release_wheel(self.directory, '0.3.0')

    def test_native_linux_tag_rejected(self):
        self.wheel.rename(self.directory / self.wheel.name.replace('manylinux_2_28', 'linux'))
        with self.assertRaisesRegex(ValueError, 'manylinux'):
            release_wheel(self.directory, '0.3.0')

    def test_multiple_wheels_rejected(self):
        (self.directory / 'extra.whl').touch()
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            release_wheel(self.directory, '0.3.0')

    def existing_release(self):
        return {'draft': False, 'prerelease': False, 'assets': [{'name': self.wheel.name}]}

    def test_github_creation_has_assets_and_no_clobber(self):
        with patch('build_tools.publish_release.github_release', return_value=None), \
             patch('build_tools.publish_release.gh') as command:
            publish_github(self.wheel, '0.3.0', 'ROCm/hrx-solutions', 'v0.3.0')
        args = command.call_args.args
        self.assertEqual(args[:3], ('release', 'create', 'v0.3.0'))
        self.assertIn('--verify-tag', args)
        self.assertNotIn('--clobber', args)
        self.assertEqual((self.directory / 'SHA256SUMS').read_text(), f'{sha256(self.wheel)}  {self.wheel.name}\n')

    def test_github_prerelease_not_latest(self):
        with patch('build_tools.publish_release.github_release', return_value=None), \
             patch('build_tools.publish_release.gh') as command:
            publish_github(self.wheel, '0.3.0rc1', 'ROCm/hrx-solutions', 'v0.3.0rc1')
        self.assertIn('--prerelease', command.call_args.args)
        self.assertIn('--latest=false', command.call_args.args)

    def test_identical_github_wheel_kept_and_missing_checksum_uploaded(self):
        def download(*args):
            if args[1] == 'download':
                (Path(args[args.index('--dir') + 1]) / self.wheel.name).write_bytes(self.wheel.read_bytes())
        with patch('build_tools.publish_release.github_release', return_value=self.existing_release()), \
             patch('build_tools.publish_release.gh', side_effect=download) as command:
            publish_github(self.wheel, '0.3.0', 'ROCm/hrx-solutions', 'v0.3.0')
        uploads = [call.args for call in command.call_args_list if call.args[1] == 'upload']
        self.assertEqual(len(uploads), 1)
        self.assertIn(str(self.directory / 'SHA256SUMS'), uploads[0])
        self.assertNotIn(str(self.wheel), uploads[0])

    def test_conflicting_github_wheel_not_overwritten(self):
        def download(*args):
            (Path(args[args.index('--dir') + 1]) / self.wheel.name).write_bytes(b'different')
        with patch('build_tools.publish_release.github_release', return_value=self.existing_release()), \
             patch('build_tools.publish_release.gh', side_effect=download) as command:
            with self.assertRaisesRegex(ValueError, 'differs'):
                publish_github(self.wheel, '0.3.0', 'ROCm/hrx-solutions', 'v0.3.0')
        self.assertEqual(command.call_count, 1)

    def test_github_lookup_only_treats_404_as_absent(self):
        with patch('build_tools.publish_release.subprocess.run') as run:
            run.return_value = subprocess.CompletedProcess([], 1, '', 'gh: Not Found (HTTP 404)')
            self.assertIsNone(github_release('owner/repo', 'v0.3.0'))
            run.return_value = subprocess.CompletedProcess([], 1, '', 'HTTP 403')
            with self.assertRaises(RuntimeError):
                github_release('owner/repo', 'v0.3.0')

    def test_pypi_identical_conflicting_missing_and_yanked(self):
        entry = {'filename': self.wheel.name, 'digests': {'sha256': sha256(self.wheel)}}
        with patch('build_tools.publish_release.urllib.request.urlopen') as open_url:
            def response():
                open_url.return_value = io.BytesIO(json.dumps({'urls': [entry]}).encode())
            response()
            self.assertTrue(pypi_has_wheel(self.wheel, '0.3.0'))
            entry['yanked'] = True
            response()
            with self.assertRaisesRegex(ValueError, 'yanked'):
                pypi_has_wheel(self.wheel, '0.3.0')
            entry['digests']['sha256'] = 'bad'
            response()
            with self.assertRaisesRegex(ValueError, 'different artifacts'):
                pypi_has_wheel(self.wheel, '0.3.0')
            open_url.side_effect = urllib.error.HTTPError('url', 404, 'Not Found', {}, None)
            self.assertFalse(pypi_has_wheel(self.wheel, '0.3.0'))
            open_url.side_effect = urllib.error.HTTPError('url', 503, 'Unavailable', {}, None)
            with self.assertRaises(urllib.error.HTTPError):
                pypi_has_wheel(self.wheel, '0.3.0')


if __name__ == '__main__':
    unittest.main()
