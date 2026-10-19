#!/usr/bin/env python3
"""Prepare release commits and a local tag; never push or publish."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess

try:
    from ..versioning import REPO_ROOT, next_dev_version, read_package_version, validate_release_version
except ImportError:
    from build_tools.versioning import REPO_ROOT, next_dev_version, read_package_version, validate_release_version


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def prepare_release(version: str, *, bump_dev: bool = False,
                    dry_run: bool = False, root: Path = REPO_ROOT) -> None:
    validate_release_version(version)
    current = read_package_version(root)
    if git(root, "branch", "--show-current") != "main":
        raise ValueError("Prepare releases from the main branch")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Working tree is not clean; commit or stash changes first")
    tag = f"v{version}"
    if git(root, "tag", "--list", tag):
        raise ValueError(f"Tag {tag} already exists; do not move published tags")
    # Check identity before changing the version file or creating either commit.
    git(root, "var", "GIT_AUTHOR_IDENT")
    git(root, "var", "GIT_COMMITTER_IDENT")

    def run(*args: str) -> None:
        print("+ " + shlex.join(["git", *args]), flush=True)
        if not dry_run:
            git(root, *args)

    def commit_version(value: str, message: str) -> None:
        print(f"version.json: package-version = {value}", flush=True)
        if not dry_run:
            path = root / "version.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["package-version"] = value
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        run("add", "--", "version.json")
        run("commit", "--signoff", "-m", message)

    if current != version:
        commit_version(version, f"Release {tag}")
    run("tag", "-a", tag, "-m", f"HRX Solutions {tag}")
    if bump_dev:
        dev = next_dev_version(version)
        commit_version(dev, f"Bump to {dev}")
    print("\nNext, after reviewing the commits and trusted publisher setup:")
    print(f"  git push --atomic origin main refs/tags/{tag}")
    print("The tag triggers GitHub release assets and PyPI publishing.")
    if dry_run:
        print("Dry run: no files, commits, or tags were changed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--bump-dev", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        prepare_release(args.version, bump_dev=args.bump_dev, dry_run=args.dry_run)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Release preparation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
