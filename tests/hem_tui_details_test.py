"""Focused checks for TUI detail identity and field normalization."""

from __future__ import annotations

import unittest

from binding.python import hem_tui, hem_tui_details


class TuiDetailsTest(unittest.TestCase):
    def test_exact_model_identity_does_not_match_replacement(self):
        original = hem_tui.ServiceObservation(
            "chat", "http://127.0.0.1:9999", "ready", "original",
            kind="model", server="local", launch_token="launch-one",
            incarnation=17)
        replacement = hem_tui.ServiceObservation(
            "chat", "http://127.0.0.1:8888", "ready", "replacement",
            kind="model", server="local", launch_token="launch-two",
            incarnation=18)
        identity = hem_tui_details.service_identity(original)
        self.assertIs(original, hem_tui_details.find_service([original], identity))
        self.assertIsNone(hem_tui_details.find_service([replacement], identity))

    def test_server_identity_ignores_model_only_incarnation(self):
        server = hem_tui.ServiceObservation(
            "local", "http://127.0.0.1:9999", "running", "server",
            server="local", launch_token="launch-one", incarnation=91)
        identity = hem_tui_details.service_identity(server)
        self.assertEqual(identity.incarnation, None)

    def test_detail_fields_normalize_only_absent_values(self):
        model = hem_tui.ServiceObservation(
            "chat", "", "", "", kind="model", server="local",
            launch_token=None, incarnation=0, recorded_state="",
            process_state="unknown", http_state="")
        fields = dict(hem_tui_details.detail_fields(model))
        self.assertEqual(fields["Endpoint"], "Unavailable")
        self.assertEqual(fields["Launch identity"], "Unavailable")
        self.assertEqual(fields["Recorded state"], "Unavailable")
        self.assertEqual(fields["HTTP"], "Unavailable")
        self.assertEqual(fields["Incarnation"], "0")
        self.assertEqual(fields["Process"], "unknown")


if __name__ == "__main__":
    unittest.main()
