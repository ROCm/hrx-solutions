"""Headless checks for the first usable Loom Control TUI shell."""

from __future__ import annotations

import asyncio
import contextlib
import io
import threading
import time
import unittest
from unittest.mock import patch

from binding.python import hem_cli, hem_tui


def rendered_text(widget):
    """Render through Textual's public Widget API across supported versions."""
    return str(widget.render())


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


class TuiTest(unittest.IsolatedAsyncioTestCase):
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

    async def test_refresh_is_single_flight_and_updates_the_overview(self):
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
            self.assertIn("service-2", str(app.query_one("#details").render()))
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
