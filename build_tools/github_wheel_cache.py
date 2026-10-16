# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Best-effort publication verification and bounded rolling-cache maintenance."""
import argparse
from datetime import datetime
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

FAMILY = "wheel-bazel-v1-linux-x86_64-"
KEY = re.compile(re.escape(FAMILY) + r"[0-9]+-[0-9]+-[a-z0-9-]+\Z")
REF = "refs/heads/main"


class CacheAPI:
    def __init__(self, repository, token):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Invalid repository")
        self.base = f"https://api.github.com/repos/{repository}/actions/caches"
        self.token = token

    def request(self, suffix="", method="GET"):
        request = urllib.request.Request(self.base + suffix, method=method, headers={
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.load(response) if method == "GET" else None
        except urllib.error.HTTPError as error:
            if method == "DELETE" and error.code == 404:
                return None
            raise

    def entries(self, key):
        entries = {}
        expected = None
        # Bound both API work and diagnostics. Incomplete listings never authorize deletes.
        for page in range(1, 101):
            query = urllib.parse.urlencode(dict(key=key, ref=REF, per_page=100,
                                                page=page, sort="created_at", direction="asc"))
            result = self.request("?" + query)
            total = result["total_count"]
            if expected is not None and total != expected:
                raise ValueError("Cache listing changed during pagination")
            expected = total
            batch = result["actions_caches"]
            for entry in batch:
                entries[entry["id"]] = entry
            if len(batch) < 100:
                if len(entries) != expected:
                    raise ValueError("Incomplete cache listing")
                return list(entries.values())
        raise ValueError("Cache listing exceeds page budget")

    def delete(self, identifier):
        self.request(f"/{identifier}", method="DELETE")


def eligible(entry, version=None):
    return (isinstance(entry.get("id"), int) and entry["id"] > 0
            and KEY.fullmatch(entry.get("key", "")) is not None
            and entry.get("ref") == REF and bool(entry.get("version"))
            and (version is None or entry["version"] == version)
            and isinstance(entry.get("size_in_bytes"), int) and entry["size_in_bytes"] > 0)


def rank(entry):
    return datetime.fromisoformat(entry["created_at"].replace("Z", "+00:00")), entry["id"]


def maintain(api, attempted_key, sleep=time.sleep):
    if KEY.fullmatch(attempted_key) is None:
        raise ValueError("Unexpected attempted cache key")
    publication = None
    for attempt in range(3):
        matches = [entry for entry in api.entries(attempted_key)
                   if eligible(entry) and entry["key"] == attempted_key]
        if len(matches) == 1:
            publication = matches[0]
            break
        if len(matches) > 1:
            raise ValueError("Ambiguous published cache version")
        if attempt < 2:
            sleep(2)
    if publication is None:
        return dict(verified=False, deleted=0, reason="Publication not visible; nothing deleted")
    entries = [entry for entry in api.entries(FAMILY)
               if eligible(entry, publication["version"])]
    if not any(entry["id"] == publication["id"] for entry in entries):
        return dict(verified=True, deleted=0, reason="Publication missing from complete listing")
    # Parse every timestamp before deleting anything, so malformed data fails closed.
    ordered = sorted(entries, key=rank)
    newest = ordered[-1]
    deleted = 0
    for entry in ordered[:-1]:
        api.delete(entry["id"])
        deleted += 1
    return dict(verified=True, deleted=deleted, preserved_id=newest["id"],
                stored_bytes=publication["size_in_bytes"], reason="Maintenance completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempted-key", required=True)
    args = parser.parse_args()
    try:
        result = maintain(CacheAPI(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"]),
                          args.attempted_key)
    except Exception as error:
        # Cache maintenance must never turn a valid wheel into a failed build.
        result = dict(verified=False, reason=f"Maintenance skipped ({type(error).__name__})")
    print(json.dumps(result, sort_keys=True))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as stream:
                stream.write("\nWheel cache maintenance: `" + json.dumps(result, sort_keys=True) + "`\n")
        except OSError:
            pass


if __name__ == "__main__":
    main()
