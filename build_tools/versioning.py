"""Shared, dependency-free package and release version rules."""

from __future__ import annotations

import json
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
_NUMBER = r"(?:0|[1-9][0-9]*)"
_RELEASE = rf"({_NUMBER})\.({_NUMBER})\.({_NUMBER})(?:(a|b|rc)({_NUMBER}))?"
RELEASE_RE = re.compile(_RELEASE)
PACKAGE_RE = re.compile(rf"{_RELEASE}(?:\.dev{_NUMBER})?")


def read_package_version(root: Path = REPO_ROOT) -> str:
    """Read the canonical version; builds never consult environment overrides."""
    data = json.loads((root / "version.json").read_text(encoding="utf-8"))
    version = data.get("package-version")
    if not isinstance(version, str) or not PACKAGE_RE.fullmatch(version):
        raise ValueError(f"Invalid package-version in version.json: {version!r}")
    return version


def validate_release_version(version: str) -> str:
    if not RELEASE_RE.fullmatch(version):
        raise ValueError(
            f"Invalid release version {version!r}; expected X.Y.Z, "
            "X.Y.ZaN, X.Y.ZbN, or X.Y.ZrcN (canonical PEP 440 spelling)"
        )
    return version


def validate_tag(tag: str, root: Path = REPO_ROOT) -> str:
    if not tag.startswith("v"):
        raise ValueError(f"Release tags must start with 'v': {tag!r}")
    version = validate_release_version(tag[1:])
    recorded = read_package_version(root)
    if version != recorded:
        raise ValueError(f"Tag {tag!r} does not match version.json ({recorded})")
    return version


def is_prerelease(version: str) -> bool:
    validate_release_version(version)
    return RELEASE_RE.fullmatch(version).group(4) is not None


def next_dev_version(version: str) -> str:
    validate_release_version(version)
    major, minor, patch, phase, number = RELEASE_RE.fullmatch(version).groups()
    if phase:
        return f"{major}.{minor}.{patch}{phase}{int(number) + 1}.dev0"
    return f"{major}.{int(minor) + 1}.0.dev0"
