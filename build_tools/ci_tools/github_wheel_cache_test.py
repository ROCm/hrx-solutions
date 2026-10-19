# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
import unittest
from unittest.mock import patch

from build_tools.ci_tools import github_wheel_cache as cache


def entry(identifier, key=None, **changes):
    result = dict(id=identifier, key=key or f"{cache.FAMILY}{identifier}-1-manylinux-core",
                  ref=cache.REF, version="archive-v1", size_in_bytes=44,
                  created_at=f"2026-10-15T00:00:{identifier:02d}Z")
    result.update(changes)
    return result


class FakeAPI:
    def __init__(self, publication, listing):
        self.publication, self.listing, self.deleted = publication, listing, []

    def entries(self, key):
        return self.listing if key in (cache.FAMILY, cache.PILOT_FAMILY) else self.publication

    def delete(self, identifier):
        self.deleted.append(identifier)


class MaintenanceTest(unittest.TestCase):
    def test_warning_success_without_publication_deletes_nothing(self):
        api = FakeAPI([], [entry(1)])
        result = cache.maintain(api, entry(2)["key"], sleep=lambda _: None)
        self.assertFalse(result["verified"])
        self.assertEqual(api.deleted, [])

    def test_preserves_later_overlapping_publication_and_unrelated_entries(self):
        current = entry(2)
        api = FakeAPI([current], [entry(1), current, entry(3),
                                entry(4, ref="refs/pull/1/merge"),
                                entry(5, version="other"),
                                entry(6, key="other-family-6"),
                                entry(7, key=cache.FAMILY + "malformed")])
        result = cache.maintain(api, current["key"])
        self.assertEqual(api.deleted, [1, 2])
        self.assertEqual(result["preserved_id"], 3)

    def test_absent_publication_in_listing_deletes_nothing(self):
        api = FakeAPI([entry(2)], [entry(1)])
        cache.maintain(api, entry(2)["key"])
        self.assertEqual(api.deleted, [])

    def test_malformed_timestamp_fails_before_any_delete(self):
        api = FakeAPI([entry(2)], [entry(1), entry(2), entry(3, created_at="bad")])
        with self.assertRaises(ValueError):
            cache.maintain(api, entry(2)["key"])
        self.assertEqual(api.deleted, [])

    def test_tie_breaker_is_id(self):
        first, second = entry(1), entry(2, created_at=entry(1)["created_at"])
        api = FakeAPI([first], [second, first])
        cache.maintain(api, first["key"])
        self.assertEqual(api.deleted, [1])

    def test_ambiguous_archive_versions_fail_closed(self):
        api = FakeAPI([entry(1), entry(2, key=entry(1)["key"], version="other")], [])
        with self.assertRaises(ValueError):
            cache.maintain(api, entry(1)["key"])
        self.assertEqual(api.deleted, [])

    def test_partial_pagination_fails_closed(self):
        api = cache.CacheAPI("owner/repo", "fake")
        with patch.object(api, "request", return_value={"total_count": 2, "actions_caches": [entry(1)]}):
            with self.assertRaises(ValueError):
                api.entries(cache.FAMILY)

    def test_every_page_is_loaded(self):
        api = cache.CacheAPI("owner/repo", "fake")
        first = [entry(1, id=i) for i in range(1, 101)]
        with patch.object(api, "request", side_effect=[
                {"total_count": 101, "actions_caches": first},
                {"total_count": 101, "actions_caches": [entry(2, id=101)]}]) as request:
            self.assertEqual(len(api.entries(cache.FAMILY)), 101)
        self.assertIn("page=2", request.call_args.args[0])

    def test_listing_failure_deletes_nothing(self):
        api = FakeAPI([entry(2)], [])
        with patch.object(api, "entries", side_effect=[[entry(2)], PermissionError()]):
            with self.assertRaises(PermissionError):
                cache.maintain(api, entry(2)["key"])
        self.assertEqual(api.deleted, [])

    def test_pilot_cleanup_never_deletes_main_or_other_ref_entries(self):
        older = entry(1, key=cache.PILOT_FAMILY + "1-1-manylinux-core", ref=cache.PILOT_REF)
        newest = entry(2, key=cache.PILOT_FAMILY + "2-1-manylinux-core", ref=cache.PILOT_REF)
        api = FakeAPI([newest], [older, newest, entry(3),
                               entry(4, key=cache.PILOT_FAMILY + "4-1-manylinux-core")])
        result = cache.maintain(api, newest["key"], pilot=True)
        self.assertEqual(api.deleted, [1])
        self.assertEqual(result["preserved_id"], 2)
        with self.assertRaises(ValueError):
            cache.maintain(api, newest["key"])

    def test_pilot_api_uses_only_fixed_pilot_ref(self):
        api = cache.CacheAPI("owner/repo", "fake", pilot=True)
        with patch.object(api, "request", return_value={"total_count": 0, "actions_caches": []}) as request:
            api.entries(cache.PILOT_FAMILY)
        self.assertIn("ref=refs%2Fheads%2Fpilot%2Fwheel-bazel-cache", request.call_args.args[0])

    def test_cli_redacts_token_from_bounded_error_detail(self):
        with patch.dict("os.environ", {"GH_TOKEN": "secret-token", "GITHUB_REPOSITORY": "owner/repo"}, clear=True), patch("sys.argv", ["cache", "--attempted-key", entry(1)["key"]]), patch.object(cache, "maintain", side_effect=ValueError("secret-token bad response")), patch("builtins.print") as output:
            cache.main()
        self.assertNotIn("secret-token", output.call_args.args[0])
        self.assertIn("bad response", output.call_args.args[0])

    def test_cli_api_failure_remains_successful(self):
        with patch.dict("os.environ", {}, clear=True), patch("sys.argv", ["cache", "--attempted-key", entry(1)["key"]]), patch("builtins.print") as output:
            cache.main()
        self.assertIn('"verified": false', output.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
