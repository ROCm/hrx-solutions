"""Headless checks for the first usable Loom Control TUI shell."""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import threading
import time
import unittest
from unittest.mock import patch

from binding.python import hem_cli, hem_tui
from binding.python.hem_tui_logs import LogBatch, LogEntry


def rendered_text(widget):
    """Render through Textual's public Widget API across supported versions."""
    return str(widget.render())


def log_text(widget):
    return "\n".join(line.text for line in widget.lines)


class DelayedProvider:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def list_services(self):
        self.started.set()
        await self.release.wait()
        return [
            hem_tui.ServiceObservation(
                "test", "http://127.0.0.1:9999", "ready", "Sample service"),
            hem_tui.ServiceObservation(
                "other", "http://127.0.0.1:9998", "stopped", "Not running"),
        ]


class RefreshProvider:
    def __init__(self):
        self.calls = 0
        self.active = 0
        self.maximum_active = 0
        self.refresh_started = asyncio.Event()
        self.release_refresh = asyncio.Event()

    async def list_services(self):
        self.calls += 1
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.calls > 1:
                self.refresh_started.set()
                await self.release_refresh.wait()
            return [hem_tui.ServiceObservation(
                f"service-{self.calls}", "http://127.0.0.1:9999",
                "running", "fixture")]
        finally:
            self.active -= 1


class ThreadedBlockingProvider:
    def __init__(self):
        self.started = threading.Event()
        self.finished = threading.Event()

    async def list_services(self):
        def blocking():
            self.started.set()
            try:
                time.sleep(0.08)
                return [hem_tui.ServiceObservation(
                    "late", "http://127.0.0.1:9999", "running", "late result")]
            finally:
                self.finished.set()
        return await asyncio.to_thread(blocking)


class PollTracker:
    def __init__(self):
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()

    def enter(self):
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)

    def leave(self):
        with self.lock:
            self.active -= 1


class ControlledLogSource:
    """A real blocking-thread poll with the production close/wait contract."""

    def __init__(self, label, tracker):
        self.label = label
        self.tracker = tracker
        self.calls = 0
        self.closed = False
        self.blocking = threading.Event()
        self.release_late = threading.Event()
        self.idle = threading.Event()
        self.idle.set()

    async def poll(self):
        self.calls += 1
        if self.calls == 1:
            return LogBatch((LogEntry("data", self.label + "-first"),))
        self.idle.clear()
        def blocking():
            self.tracker.enter()
            self.blocking.set()
            try:
                self.release_late.wait(1)
                return LogBatch((LogEntry("data", self.label + "-late"),))
            finally:
                self.tracker.leave()
                self.idle.set()
        return await asyncio.to_thread(blocking)

    def close(self):
        self.closed = True

    async def wait_closed(self):
        while not self.idle.is_set():
            await asyncio.sleep(0.01)


class FakeLogProvider:
    def __init__(self):
        self.sources = []
        self.tracker = PollTracker()

    def open_source(self, service):
        source = ControlledLogSource(service.name, self.tracker)
        self.sources.append(source)
        return source


class TuiTest(unittest.IsolatedAsyncioTestCase):
    async def test_complete_keyboard_flow_has_contextual_help_and_focus(self):
        services = [hem_tui.ServiceObservation(
            "local", "http://127.0.0.1:9999", "running", "fixture",
            server="local", launch_token="launch-one")]
        logs = FakeLogProvider()
        app = hem_tui.create_app(hem_tui.FakeServiceProvider(services), logs)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            listing = app.query_one("#services")
            self.assertTrue(listing.has_focus)
            self.assertIn("Enter Details", rendered_text(app.query_one("#help")))

            await pilot.press("enter")
            await pilot.pause()
            self.assertFalse(listing.display)
            self.assertIn("Esc Back", rendered_text(app.query_one("#help")))
            self.assertIn("Server identity: local",
                          rendered_text(app.query_one("#details")))

            await pilot.press("l")
            await pilot.pause()
            source = logs.sources[0]
            self.assertTrue(app.query_one("#logs").has_focus)
            self.assertIn("Space Pause/Resume",
                          rendered_text(app.query_one("#help")))
            self.assertIn("local-first", log_text(app.query_one("#logs")))
            self.assertTrue(await asyncio.to_thread(source.blocking.wait, 1))
            source.release_late.set()

            await pilot.press("escape")
            await pilot.pause()
            self.assertIn("Server identity: local",
                          rendered_text(app.query_one("#details")))
            await pilot.press("escape")
            await pilot.pause()
            self.assertTrue(listing.display)
            self.assertTrue(listing.has_focus)
            await pilot.press("escape")
            await pilot.pause()
            self.assertTrue(app.is_running)
            await pilot.press("r")
            await pilot.pause()
            self.assertIn("local", rendered_text(app.query_one("#details")))
            await pilot.press("q")
        self.assertFalse(app.is_running)

    async def test_overview_refresh_preserves_only_exact_selected_identity(self):
        alpha = hem_tui.ServiceObservation(
            "alpha", "http://127.0.0.1:9999", "running", "alpha",
            server="alpha", launch_token="launch-alpha")
        beta = hem_tui.ServiceObservation(
            "beta", "http://127.0.0.1:9998", "running", "beta-old",
            server="beta", launch_token="launch-beta")
        provider = hem_tui.FakeServiceProvider([alpha, beta])
        logs = FakeLogProvider()
        app = hem_tui.create_app(provider, logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("down")
            provider.services = [beta, alpha]
            await pilot.press("r")
            await pilot.pause()
            self.assertEqual(app.query_one("#services").index, 0)
            self.assertIn("beta-old", rendered_text(app.query_one("#details")))

            provider.services = [hem_tui.ServiceObservation(
                "beta", "http://127.0.0.1:8888", "stopped", "beta-replaced",
                server="beta", launch_token="launch-replacement"), alpha]
            await pilot.press("r")
            await pilot.pause()
            self.assertIsNone(app.query_one("#services").index)
            overview = rendered_text(app.query_one("#details"))
            self.assertIn("Previous selection is unavailable", overview)
            self.assertNotIn("beta-replaced", overview)
            await pilot.press("enter", "l")
            await pilot.pause()
            self.assertIsNone(app._detail_identity)
            self.assertEqual(logs.sources, [])
            await pilot.press("q")

    async def test_disappeared_selection_does_not_retarget_after_reorder(self):
        alpha = hem_tui.ServiceObservation(
            "alpha", "http://127.0.0.1:9999", "running", "alpha",
            server="alpha", launch_token="launch-alpha")
        beta = hem_tui.ServiceObservation(
            "beta", "http://127.0.0.1:9998", "running", "beta",
            server="beta", launch_token="launch-beta")
        gamma = hem_tui.ServiceObservation(
            "gamma", "http://127.0.0.1:9997", "running", "gamma",
            server="gamma", launch_token="launch-gamma")
        provider = hem_tui.FakeServiceProvider([alpha, beta, gamma])
        logs = FakeLogProvider()
        app = hem_tui.create_app(provider, logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("down")
            self.assertIn("beta", rendered_text(app.query_one("#details")))

            provider.services = [gamma, alpha]
            await pilot.press("r")
            await pilot.pause()
            self.assertIsNone(app.query_one("#services").index)
            overview = rendered_text(app.query_one("#details"))
            self.assertIn("Previous selection is unavailable", overview)
            self.assertNotIn("alpha", overview)
            self.assertNotIn("gamma", overview)

            await pilot.press("enter", "l")
            await pilot.pause()
            self.assertIsNone(app._detail_identity)
            self.assertEqual(logs.sources, [])
            await pilot.press("q")

    async def test_normal_and_narrow_layout_keep_identity_and_state(self):
        service = hem_tui.ServiceObservation(
            "local", "http://127.0.0.1:9999", "unavailable", "HTTP unavailable",
            server="local", launch_token="launch-one")
        for size, stacked in (((100, 30), False), ((42, 20), True)):
            with self.subTest(size=size), patch.dict(os.environ, {"NO_COLOR": "1"}):
                app = hem_tui.create_app(hem_tui.FakeServiceProvider([service]))
                async with app.run_test(size=size) as pilot:
                    await pilot.pause()
                    listing = app.query_one("#services")
                    details = app.query_one("#details")
                    self.assertEqual(listing.region.y != details.region.y, stacked)
                    self.assertIn("local", rendered_text(details))
                    self.assertIn("State: unavailable", rendered_text(details))
                    self.assertIn("Kind: server", rendered_text(details))
                    self.assertIn("q Quit", rendered_text(app.query_one("#help")))
                    await pilot.press("q")

    async def test_loading_populated_state_navigation_and_quit(self):
        provider = DelayedProvider()
        app = hem_tui.create_app(provider)
        async with app.run_test() as pilot:
            await provider.started.wait()
            self.assertIn("Loading services", str(app.query_one("#state").render()))
            provider.release.set()
            await pilot.pause()
            self.assertIn("test", str(app.query_one("#details").render()))
            self.assertIn("ready", str(app.query_one("#details").render()))
            await pilot.press("down")
            await pilot.pause()
            self.assertIn("other", str(app.query_one("#details").render()))
            await pilot.press("q")
            await pilot.pause()
        self.assertFalse(app.is_running)

    async def test_empty_and_provider_error_states(self):
        for provider, expected in (
            (hem_tui.FakeServiceProvider(services=[]), "No services configured"),
            (hem_tui.FakeServiceProvider(error=RuntimeError("[bold red]offline[/bold red]")),
             "Provider unavailable: [bold red]offline[/bold red]"),
        ):
            app = hem_tui.create_app(provider)
            async with app.run_test() as pilot:
                await pilot.pause()
                state = app.query_one("#state")
                self.assertIn(expected, rendered_text(state))
                await pilot.press("q")

    async def test_default_provider_keeps_absent_namespace_absent(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "missing"
            with patch.dict("os.environ", {"LOOM_SERVE_DIR": str(root)}):
                app = hem_tui.create_app()
                async with app.run_test() as pilot:
                    await pilot.pause()
                    state = rendered_text(app.query_one("#state"))
                    self.assertIn("No services configured", state)
                    self.assertFalse(root.exists())
                    await pilot.press("q")

    async def test_refresh_is_single_flight_without_retargeting_overview(self):
        provider = RefreshProvider()
        app = hem_tui.create_app(provider)
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertIn("service-1", str(app.query_one("#details").render()))
            await pilot.press("r")
            await provider.refresh_started.wait()
            await pilot.press("r", "r", "r")
            self.assertEqual(provider.calls, 2)
            self.assertEqual(provider.maximum_active, 1)
            provider.release_refresh.set()
            await pilot.pause()
            self.assertIsNone(app.query_one("#services").index)
            self.assertIn("Previous selection is unavailable",
                          rendered_text(app.query_one("#details")))
            self.assertNotIn("service-2",
                             rendered_text(app.query_one("#details")))
            await pilot.press("q")
        self.assertEqual(provider.active, 0)

    async def test_late_refresh_generation_does_not_repaint(self):
        provider = RefreshProvider()
        app = hem_tui.create_app(provider)
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertIn("service-1", str(app.query_one("#details").render()))
            await pilot.press("r")
            await provider.refresh_started.wait()
            app._load_generation += 1
            provider.release_refresh.set()
            await pilot.pause()
            self.assertIn("service-1", str(app.query_one("#details").render()))
            self.assertNotIn("service-2", str(app.query_one("#details").render()))
            await pilot.press("q")

    async def test_quit_cancels_publish_while_blocking_work_finishes(self):
        provider = ThreadedBlockingProvider()
        app = hem_tui.create_app(provider)
        async with app.run_test() as pilot:
            await asyncio.to_thread(provider.started.wait, 1)
            await pilot.press("q")
        self.assertFalse(app.is_running)
        self.assertTrue(await asyncio.to_thread(provider.finished.wait, 1))
        self.assertEqual(app.observations, ())

    async def test_provider_fields_are_rendered_as_text(self):
        provider = hem_tui.FakeServiceProvider(services=[hem_tui.ServiceObservation(
            "[bold red]HOST[/bold red]", "[link=https://example.test]endpoint[/link]",
            "[reverse]ready[/reverse]", "[italic]summary[/italic]")])
        app = hem_tui.create_app(provider)
        async with app.run_test() as pilot:
            await pilot.pause()
            details = rendered_text(app.query_one("#details"))
            self.assertIn("[bold red]HOST[/bold red]", details)
            self.assertIn("[link=https://example.test]endpoint[/link]", details)
            self.assertIn("[reverse]ready[/reverse]", details)
            self.assertIn("[italic]summary[/italic]", details)
            await pilot.press("q")

    async def test_server_details_and_back_to_overview(self):
        provider = hem_tui.FakeServiceProvider(services=[
            hem_tui.ServiceObservation(
                "local", "http://127.0.0.1:9999", "running", "fixture",
                server="local", launch_token="launch-one",
                recorded_state="running", process_state="running",
                http_state="available"),
        ])
        app = hem_tui.create_app(provider)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            details = rendered_text(app.query_one("#details"))
            self.assertIn("Endpoint: http://127.0.0.1:9999", details)
            self.assertIn("Server identity: local", details)
            self.assertIn("Launch identity: launch-one", details)
            self.assertIn("Recorded state: running", details)
            self.assertIn("Process: running", details)
            self.assertIn("HTTP: available", details)
            self.assertFalse(app.query_one("#services").display)
            await pilot.press("escape")
            await pilot.pause()
            self.assertTrue(app.is_running)
            self.assertTrue(app.query_one("#services").display)
            self.assertIn("fixture", rendered_text(app.query_one("#details")))
            await pilot.press("q")

    async def test_model_details_include_readiness_and_incarnation(self):
        provider = hem_tui.FakeServiceProvider(services=[
            hem_tui.ServiceObservation(
                "local", "http://127.0.0.1:9999", "running", "server",
                server="local", launch_token="launch-one"),
            hem_tui.ServiceObservation(
                "chat", "http://127.0.0.1:9999", "ready", "model",
                kind="model", server="local", launch_token="launch-one",
                incarnation=17, recorded_state="registered",
                process_state="running", http_state="available"),
        ])
        app = hem_tui.create_app(provider)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("down", "enter")
            await pilot.pause()
            details = rendered_text(app.query_one("#details"))
            self.assertIn("Model identity: chat", details)
            self.assertIn("Server identity: local", details)
            self.assertIn("Launch identity: launch-one", details)
            self.assertIn("Incarnation: 17", details)
            self.assertIn("Readiness: ready", details)
            await pilot.press("q")

    async def test_missing_detail_fields_are_plainly_unavailable(self):
        provider = hem_tui.FakeServiceProvider(services=[
            hem_tui.ServiceObservation(
                "chat", "", "", "", kind="model", server="local",
                launch_token=None, incarnation=None, recorded_state="",
                process_state="", http_state=""),
        ])
        app = hem_tui.create_app(provider)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            details = rendered_text(app.query_one("#details"))
            for label in ("Endpoint", "Launch identity", "Recorded state",
                          "Process", "HTTP", "Incarnation", "Readiness"):
                self.assertIn(f"{label}: Unavailable", details)
            await pilot.press("q")

    async def test_detail_values_cannot_inject_rich_markup(self):
        hostile = "[bold red][link=https://example.test]owned[/link][/bold red]"
        provider = hem_tui.FakeServiceProvider(services=[
            hem_tui.ServiceObservation(
                hostile, hostile, hostile, hostile, kind="model",
                server=hostile, launch_token=hostile, incarnation=9,
                recorded_state=hostile, process_state=hostile,
                http_state=hostile),
        ])
        app = hem_tui.create_app(provider)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            rendered = app.query_one("#details").render()
            self.assertIn(hostile, str(rendered))
            styles = " ".join(str(span.style).lower()
                              for span in getattr(rendered, "spans", ()))
            self.assertNotIn("red", styles)
            self.assertNotIn("link", styles)
            await pilot.press("q")

    async def test_refresh_does_not_substitute_replacement_identity(self):
        provider = hem_tui.FakeServiceProvider(services=[
            hem_tui.ServiceObservation(
                "chat", "http://127.0.0.1:9999", "ready", "old",
                kind="model", server="local", launch_token="launch-one",
                incarnation=17, recorded_state="registered",
                process_state="running", http_state="available"),
        ])
        app = hem_tui.create_app(provider)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            provider.services = [hem_tui.ServiceObservation(
                "chat", "http://127.0.0.1:8888", "ready", "replacement",
                kind="model", server="local", launch_token="launch-two",
                incarnation=18, recorded_state="registered",
                process_state="running", http_state="available")]
            await pilot.press("r")
            await pilot.pause()
            details = rendered_text(app.query_one("#details"))
            self.assertIn("Selected identity changed or is unavailable", details)
            self.assertNotIn("launch-two", details)
            self.assertNotIn("8888", details)
            self.assertNotIn("Incarnation: 18", details)
            self.assertIsNone(app.query_one("#services").index)
            await pilot.press("escape")
            await pilot.pause()
            details = rendered_text(app.query_one("#details"))
            self.assertIn("Select a service to inspect its summary", details)
            self.assertNotIn("replacement", details)
            await pilot.press("q")

    async def test_refresh_restores_exact_detail_row_before_back(self):
        selected = hem_tui.ServiceObservation(
            "beta", "http://127.0.0.1:9998", "running", "beta-old",
            server="beta", launch_token="launch-beta")
        provider = hem_tui.FakeServiceProvider(services=[
            hem_tui.ServiceObservation(
                "alpha", "http://127.0.0.1:9999", "running", "alpha",
                server="alpha", launch_token="launch-alpha"),
            selected,
        ])
        app = hem_tui.create_app(provider)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("down", "enter")
            await pilot.pause()
            provider.services = [
                hem_tui.ServiceObservation(
                    "beta", "http://127.0.0.1:9998", "running", "beta-new",
                    server="beta", launch_token="launch-beta"),
                hem_tui.ServiceObservation(
                    "alpha", "http://127.0.0.1:9999", "running", "alpha",
                    server="alpha", launch_token="launch-alpha"),
            ]
            await pilot.press("r")
            await pilot.pause()
            self.assertEqual(app.query_one("#services").index, 0)
            await pilot.press("escape")
            await pilot.pause()
            details = rendered_text(app.query_one("#details"))
            self.assertIn("beta-new", details)
            self.assertNotIn("alpha", details)
            await pilot.press("q")

    async def test_unverified_identity_is_not_rendered_as_current(self):
        provider = hem_tui.FakeServiceProvider(services=[
            hem_tui.ServiceObservation(
                "chat", "http://127.0.0.1:9999", "unknown", "changed",
                kind="model", server="local", launch_token="launch-one",
                incarnation=17, identity_verified=False),
        ])
        app = hem_tui.create_app(provider)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            details = rendered_text(app.query_one("#details"))
            self.assertIn("Selected identity changed or is unavailable", details)
            self.assertNotIn("Endpoint:", details)
            await pilot.press("q")

    async def test_logs_follow_pause_back_switch_and_suppress_late_output(self):
        services = [
            hem_tui.ServiceObservation(
                "one", "http://127.0.0.1:9999", "running", "fixture",
                server="one", launch_token="launch-one"),
            hem_tui.ServiceObservation(
                "two", "http://127.0.0.1:9998", "running", "fixture",
                server="two", launch_token="launch-two"),
        ]
        logs = FakeLogProvider()
        app = hem_tui.create_app(hem_tui.FakeServiceProvider(services), logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("l")
            await pilot.pause()
            first = logs.sources[0]
            self.assertIn("one-first", log_text(app.query_one("#logs")))
            await pilot.press("space")
            paused_calls = first.calls
            await asyncio.sleep(0.2)
            self.assertEqual(first.calls, paused_calls)
            await pilot.press("space")
            await pilot.pause()
            self.assertTrue(await asyncio.to_thread(first.blocking.wait, 1))
            back = asyncio.create_task(pilot.press("escape"))
            await asyncio.sleep(0.05)
            self.assertFalse(back.done())
            self.assertEqual(logs.tracker.active, 1)
            app.action_logs()
            self.assertEqual(len(logs.sources), 1)
            first.release_late.set()
            await back
            self.assertTrue(first.closed)
            self.assertEqual(logs.tracker.active, 0)
            self.assertNotIn("one-late", log_text(app.query_one("#logs")))
            await pilot.press("down", "l")
            await pilot.pause()
            second = logs.sources[1]
            self.assertIn("two-first", log_text(app.query_one("#logs")))
            self.assertNotIn("one-late", log_text(app.query_one("#logs")))
            self.assertTrue(await asyncio.to_thread(second.blocking.wait, 1))
            quit_task = asyncio.create_task(pilot.press("q"))
            await asyncio.sleep(0.05)
            self.assertFalse(quit_task.done())
            self.assertTrue(app.is_running)
            self.assertEqual(logs.tracker.maximum_active, 1)
            second.release_late.set()
            await quit_task
        self.assertTrue(second.closed)
        self.assertEqual(logs.tracker.active, 0)
        self.assertFalse(app.is_running)

    def test_cli_dispatch_and_help(self):
        with patch("binding.python.hem_tui.run", return_value=0) as run:
            self.assertEqual(hem_cli.main(["tui"]), 0)
            run.assert_called_once()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                hem_cli.main(["tui", "--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("loom-control tui", output.getvalue())
        args = hem_cli.parse_args(["tui"])
        self.assertEqual((args.resource, args.command), ("tui", "run"))


if __name__ == "__main__":
    unittest.main()
