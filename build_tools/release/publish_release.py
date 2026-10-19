#!/usr/bin/env python3
"""Validate tagged artifacts, attach GitHub assets, and check PyPI retry state."""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile

from packaging.utils import parse_wheel_filename

try:
    from ..versioning import is_prerelease, validate_tag
except ImportError:
    from build_tools.versioning import is_prerelease, validate_tag


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_wheel(directory: Path, version: str) -> Path:
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected exactly one wheel in {directory}, found {len(wheels)}")
    wheel = wheels[0]
    name, actual_version, build, tags = parse_wheel_filename(wheel.name)
    if name != "hrx-solutions" or str(actual_version) != version or build:
        raise ValueError(f"Wheel filename does not match hrx-solutions {version}: {wheel.name}")
    if not tags or any(
        tag.interpreter != "py3" or tag.abi != "none"
        or not tag.platform.startswith("manylinux")
        or not tag.platform.endswith("_x86_64") for tag in tags
    ):
        raise ValueError(f"Expected a py3-none manylinux x86-64 wheel: {wheel.name}")
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(members) != 1:
            raise ValueError("Wheel must contain exactly one METADATA file")
        metadata = email.parser.BytesParser().parsebytes(archive.read(members[0]))
        if metadata["Name"] != "hrx-solutions" or metadata["Version"] != version:
            raise ValueError("Wheel metadata does not match the release package and version")
    return wheel


def gh(*args: str) -> str:
    return subprocess.run(
        ["gh", *args], check=True, text=True, stdout=subprocess.PIPE
    ).stdout


def github_release(repository: str, tag: str) -> dict | None:
    result = subprocess.run(
        ["gh", "api", f"repos/{repository}/releases/tags/{tag}"],
        text=True, capture_output=True,
    )
    if result.returncode == 0:
        return json.loads(result.stdout)
    if "HTTP 404" in result.stderr:
        return None
    raise RuntimeError(f"Could not inspect GitHub release: {result.stderr.strip()}")


def publish_github(wheel: Path, version: str, repository: str, tag: str) -> None:
    checksum = wheel.parent / "SHA256SUMS"
    checksum.write_text(f"{sha256(wheel)}  {wheel.name}\n", encoding="utf-8")
    expected = {path.name: path for path in (wheel, checksum)}
    existing = github_release(repository, tag)
    if existing is None:
        options = ["--prerelease", "--latest=false"] if is_prerelease(version) else []
        gh("release", "create", tag, str(wheel), str(checksum),
           "--repo", repository, "--verify-tag", "--title", f"HRX Solutions {tag}",
           "--generate-notes", *options)
        return
    if existing["draft"] or existing["prerelease"] != is_prerelease(version):
        raise ValueError("Existing GitHub release has conflicting draft/prerelease status")
    names = {asset["name"] for asset in existing["assets"]}
    unexpected = {name for name in names if name.endswith(".whl")} - expected.keys()
    if unexpected:
        raise ValueError(f"Existing release has unexpected wheels: {sorted(unexpected)}")
    # Check every overlapping asset before uploading any missing assets. Never clobber.
    with tempfile.TemporaryDirectory(prefix="hrx-release-") as temporary:
        for name in sorted(names & expected.keys()):
            gh("release", "download", tag, "--repo", repository,
               "--pattern", name, "--dir", temporary)
            if sha256(Path(temporary) / name) != sha256(expected[name]):
                raise ValueError(f"Existing GitHub asset differs: {name}; use a new version")
    missing = [str(path) for name, path in expected.items() if name not in names]
    if missing:
        gh("release", "upload", tag, *missing, "--repo", repository)
    else:
        print("GitHub release already contains identical assets.")


def pypi_has_wheel(wheel: Path, version: str) -> bool:
    request = urllib.request.Request(
        f"https://pypi.org/pypi/hrx-solutions/{version}/json",
        headers={"User-Agent": "hrx-solutions-release"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            release = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    matches = [entry for entry in release["urls"] if entry["filename"] == wheel.name]
    if len(matches) != 1 or matches[0]["digests"]["sha256"] != sha256(wheel):
        raise ValueError("PyPI version exists with different artifacts; use a new version")
    if matches[0].get("yanked"):
        raise ValueError("Existing PyPI wheel is yanked; review it before retrying")
    return True


def output(name: str, value: str) -> None:
    print(f"{name}={value}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            stream.write(f"{name}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("validate-tag", "validate-wheel", "github", "pypi-status"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--wheel-dir", type=Path, default=Path("dist"))
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    args = parser.parse_args()
    try:
        version = validate_tag(args.tag)
        if args.operation == "validate-tag":
            output("package_version", version)
            output("prerelease", str(is_prerelease(version)).lower())
            return 0
        wheel = release_wheel(args.wheel_dir, version)
        if args.operation == "github":
            if not args.repository:
                raise ValueError("--repository or GITHUB_REPOSITORY is required")
            publish_github(wheel, version, args.repository, args.tag)
        elif args.operation == "pypi-status":
            output("exists", str(pypi_has_wheel(wheel, version)).lower())
        else:
            print(f"Validated release wheel: {wheel}")
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Release failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
