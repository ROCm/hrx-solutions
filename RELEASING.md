# Releasing HRX Solutions

A release tag builds one Linux x86-64 manylinux wheel, checks it, attaches it
and SHA-256 checksums to a GitHub release, then publishes the same wheel to
PyPI. The wheel supports Python 3.9 and newer. Releases contain no model
weights. There is no nightly PyPI publication or source-distribution upload.

## Version conventions

`version.json` is the source of truth for local builds and CI. Its
`package-version` is read by setuptools; there is no environment-variable
version override. `MODULE.bazel` pins the HRX source dependency.

| Kind | Package version | Release tag |
| --- | --- | --- |
| Stable | `0.3.0` | `v0.3.0` |
| Alpha | `0.3.0a1` | `v0.3.0a1` |
| Beta | `0.3.0b1` | `v0.3.0b1` |
| Release candidate | `0.3.0rc1` | `v0.3.0rc1` |
| Development | `0.3.0.dev0` or `0.3.0rc2.dev0` | None |

Use canonical PEP 440 spelling. Bare numeric tags, hyphenated prerelease
suffixes, local versions (`+...`), post releases, and development release tags
are rejected. The version in the tagged commit must exactly match the tag
without its leading `v`.

## One-time trusted publisher setup

Complete this setup before the first release push. No PyPI token needs to be
stored in GitHub Actions.

1. In `ROCm/hrx-solutions` on GitHub, create an environment named **`pypi`**
   under Settings → Environments. Restrict deployment tags to `v*`. If you
   configure required reviewers, the publishing job will wait for their approval.
2. On PyPI, configure a trusted publisher with these exact values:

   | Setting | Value |
   | --- | --- |
   | Project name | `hrx-solutions` |
   | Owner | `ROCm` |
   | Repository | `hrx-solutions` |
   | Workflow filename | `release_python_wheel.yml` |
   | Environment | `pypi` |

   If the project does not exist yet, create a **pending publisher** in your
   PyPI account's Publishing settings. If it already exists, its owner must
   add the publisher under the project's Publishing settings. Changing a
   previous package's name or publisher does not configure this project.
3. Confirm the repository can use the existing
   `aws-linux-scale-rocm-prod` build runner and fetch the TheRock toolchain.
   The PyPI job itself runs on GitHub-hosted Ubuntu with `id-token: write`.

See PyPI's [trusted publisher setup](https://docs.pypi.org/trusted-publishers/)
and [publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
A workflow file alone cannot register the publisher on PyPI.

## Prepare a release locally

Use a clean, up-to-date `main` checkout with Git author/committer identity
configured. Commit the implementation and documentation changes first. Fetch
remote tags before preparing a release so the local duplicate-tag check sees
existing releases:

```bash
git switch main
git pull --ff-only origin main
git fetch origin --tags
python -m build_tools.release.make_release --version 0.3.0 --bump-dev --dry-run
```

The helper requires only the Python standard library. Both dry runs and real
runs check the branch, working tree, version syntax, Git identity, and local
tag availability. A dry run changes no files, commits, or tags.

After reviewing the dry run:

```bash
python -m build_tools.release.make_release --version 0.3.0 --bump-dev
git log -3 --oneline
git show v0.3.0:version.json
```

The helper:

1. Sets the release version and creates a signed-off commit. If it is already
   set, it uses the existing commit without creating an empty one.
2. Creates an annotated `v0.3.0` tag at the release commit.
3. With `--bump-dev`, commits the next development version. A stable release
   advances the minor version (`0.3.0` → `0.4.0.dev0`); a prerelease advances
   its sequence (`0.3.0rc1` → `0.3.0rc2.dev0`). Omit the flag to leave the
   release version checked out.

The helper never pushes. If a Git hook or commit fails, inspect `git status`,
`git log`, and the local tag before retrying; completed steps are not rolled back.

## Push and watch the release

For the initial release, finish the trusted publisher setup and review the
local commits together before this step. Push the branch and **only the
selected tag** atomically:

```bash
git push --atomic origin main refs/tags/v0.3.0
```

The [Release Python Wheel workflow](https://github.com/ROCm/hrx-solutions/actions/workflows/release_python_wheel.yml)
runs these stages:

1. Check the tag against the version recorded in its commit.
2. Check release tooling, build with the existing pinned HRX source and
   TheRock SDK selection, repair the wheel with auditwheel, and validate its
   contents. A clean Ubuntu environment installs it and checks CLI startup.
3. Check the release wheel's filename/version, metadata, and PyPI description
   with `twine check --strict`.
4. Create a GitHub release with generated notes, the wheel, and `SHA256SUMS`.
   Alpha, beta, and release-candidate versions are marked as prereleases and
   are not marked latest.
5. Publish the same Actions wheel artifact to PyPI through OIDC. PyPI records
   prerelease versions using their PEP 440 version strings.

Build jobs and pull requests have no PyPI publishing permission. Runs for the
same release tag are serialized without cancelling an active run. The
`release_type` CI input selects the **TheRock toolchain channel**, independently
of the HRX Solutions package version.

Check the GitHub assets and PyPI project after success. Set `RELEASE_VERSION`
to the exact published version, then check the installed commands:

```bash
python -m pip install --upgrade "hrx-solutions==${RELEASE_VERSION:?set the published version}"
loom-serve --help
loom-control --help
```

The management command is now `loom-control`. Update scripts using `loom-hem`;
the wheel no longer installs that command. State and configuration are unchanged.

Pin the exact version for an installation check, including prerelease suffixes.
Ordinary pip upgrades generally prefer stable releases; use `--pre` when
intentionally selecting prereleases without an exact version.

## Failures and retries

- **Tag/version mismatch:** nothing is built or published. Inspect the tagged
  `version.json`. Do not move an existing release tag; prepare a new version.
- **Build or smoke failure:** fix the problem and prepare a new version if a
  source change is needed. A transient infrastructure failure can be rerun.
- **GitHub upload failure:** rerun failed jobs from the original Actions run.
  Existing assets are downloaded and compared; identical assets are kept,
  missing assets are uploaded, and conflicting files are rejected. There is
  no `--clobber` behavior.
- **PyPI identity or environment failure:** the GitHub release may already
  exist. Check the exact owner, repository, workflow filename, environment,
  and any required environment approval. Correct setup and rerun failed jobs.
- **Upload succeeded but the job failed afterward:** the retry compares the
  PyPI wheel's SHA-256 digest with the original artifact and skips publication
  only when they match. A conflicting or yanked wheel fails explicitly.

Use **Re-run failed jobs** to retain the original build artifact after a
publishing failure. Do not rebuild and try to overwrite an uploaded wheel:
PyPI versions/files are immutable, and build outputs can differ. If the
original Actions artifact has expired, stop and recover the exact wheel from
the GitHub release and verify `SHA256SUMS` before any manual recovery. Changed
artifacts need a new version. Do not delete an uploaded file expecting to
reuse its filename.

## Local release-tool checks

```bash
python -m pip install 'packaging>=24.2'
python -m unittest discover -s build_tools -p '*_test.py' -v
```

These checks exercise version rules, release preparation in temporary Git
repositories, wheel metadata, and publishing retries with mocked services.
They do not create tags in this repository or contact publishing services.
