"""Textual shell for interactive, read-only Loom Control observation."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from .hem_tui_details import (
    ServiceIdentity,
    detail_fields,
    find_service,
    service_identity,
)


@dataclass(frozen=True)
class ServiceObservation:
    """Small presentation model shared by providers and the TUI shell."""

    name: str
    endpoint: str
    state: str
    summary: str
    kind: str = "server"
    server: str = ""
    launch_token: Optional[str] = None
    incarnation: Optional[int] = None
    recorded_state: str = "unknown"
    process_state: str = "unknown"
    http_state: str = "unknown"
    identity_verified: bool = True


class ServiceProvider(Protocol):
    """Supplies observations without exposing backend or CLI internals."""

    async def list_services(self) -> Sequence[ServiceObservation]:
        ...


class FakeServiceProvider:
    """Deterministic sample provider for tests and UI previews."""

    def __init__(self, services: Optional[list[ServiceObservation]] = None,
                 error: Optional[Exception] = None) -> None:
        self.services = list(services if services is not None else [
            ServiceObservation("local", "http://127.0.0.1:8080", "ready",
                               "Listening · no models loaded"),
            ServiceObservation("staging", "http://127.0.0.1:8081", "stopped",
                               "Not running"),
        ])
        self.error = error

    async def list_services(self) -> list[ServiceObservation]:
        if self.error is not None:
            raise self.error
        return list(self.services)


def create_app(provider: Optional[ServiceProvider] = None, log_provider=None):
    """Build the interactive app; importing the CLI does not import Textual."""
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal
    from textual.widgets import Footer, Header, Label, ListItem, ListView, RichLog, Static
    from rich.markup import escape
    from rich.text import Text

    class LoomControlTui(App[None]):
        TITLE = "Loom Control"
        SUB_TITLE = "Local service observer"
        CSS = """
        #body { height: 1fr; }
        #services { width: 36%; min-width: 24; border: round $primary; }
        #details { width: 1fr; border: round $primary; padding: 1 2; }
        #logs { width: 1fr; border: round $primary; padding: 1 2; display: none; }
        #state { height: 1; padding: 0 1; }
        #help { height: 2; dock: bottom; padding: 0 1; background: $boost; }
        .selected { text-style: bold; }
        #body.compact { layout: vertical; }
        #body.compact #services { width: 1fr; min-width: 0; height: 2fr; }
        #body.compact #details { width: 1fr; height: 3fr; padding: 0 1; }
        #body.compact #logs { width: 1fr; height: 1fr; padding: 0 1; }
        """
        OVERVIEW_HELP = (
            "↑/↓ Select   Enter Details   l Logs\n"
            "r Refresh   q Quit"
        )
        DETAILS_HELP = "Esc Back   l Logs\nr Refresh   q Quit"
        LOGS_HELP = "↑/↓ Scroll   Space Pause/Resume\nEsc Back   q Quit"
        BINDINGS = [
            ("enter", "open_details", "Details"),
            ("r", "refresh", "Refresh"),
            ("l", "logs", "Logs"),
            ("space", "pause_logs", "Pause/Resume"),
            ("q", "quit", "Quit"),
            ("escape", "back", "Back"),
        ]

        def __init__(self, data_provider: ServiceProvider, logs) -> None:
            super().__init__()
            self.provider = data_provider
            self.observations: Sequence[ServiceObservation] = ()
            self._load_generation = 0
            self._loading = False
            self._detail_identity: Optional[ServiceIdentity] = None
            self._overview_identity: Optional[ServiceIdentity] = None
            self._overview_selection_established = False
            self.log_provider = logs
            self._log_generation = 0
            self._log_source = None
            self._log_closing = False
            self._log_paused = False
            self._log_lines = deque()
            self._log_characters = 0

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("Loading services…", id="state")
            with Horizontal(id="body"):
                yield ListView(id="services", disabled=True)
                yield Static("Waiting for service observations.", id="details")
                yield RichLog(id="logs", markup=False, wrap=True, max_lines=500)
            yield Static(self.OVERVIEW_HELP, id="help")
            yield Footer()

        def on_mount(self) -> None:
            self._set_compact_layout(self.size.width <= 60)
            self.request_refresh()

        def on_resize(self, event) -> None:
            self._set_compact_layout(event.size.width <= 60)

        def _set_compact_layout(self, compact: bool) -> None:
            self.query_one("#body", Horizontal).set_class(compact, "compact")

        def request_refresh(self) -> None:
            if self._loading:
                return
            self._loading = True
            self._load_generation += 1
            generation = self._load_generation
            self.run_worker(self.load_services(generation), name="load-services")

        def action_refresh(self) -> None:
            if self._log_source is not None:
                return
            self.request_refresh()

        def action_open_details(self) -> None:
            if self._detail_identity is not None:
                return
            listing = self.query_one("#services", ListView)
            index = listing.index
            if index is None or not 0 <= index < len(self.observations):
                return
            self._detail_identity = service_identity(self.observations[index])
            listing.display = False
            self.query_one("#help", Static).update(self.DETAILS_HELP)
            self.show_details()

        async def action_back(self) -> None:
            if self._log_source is not None:
                if not await self._stop_logs():
                    return
                self.query_one("#logs", RichLog).display = False
                self.query_one("#details", Static).display = True
                if self._detail_identity is None:
                    self.query_one("#services", ListView).display = True
                    self.query_one("#help", Static).update(self.OVERVIEW_HELP)
                    self.query_one("#services", ListView).focus()
                else:
                    self.query_one("#help", Static).update(self.DETAILS_HELP)
                    self.show_details()
                self.query_one("#state", Static).update(
                    f"{len(self.observations)} service(s) · observations")
                return
            if self._detail_identity is None:
                return
            self._detail_identity = None
            listing = self.query_one("#services", ListView)
            listing.display = True
            self.query_one("#help", Static).update(self.OVERVIEW_HELP)
            listing.focus()
            if listing.index is None:
                self.query_one("#details", Static).update(
                    "Select a service to inspect its summary.")
            else:
                self.show_selection(listing.index)

        def on_unmount(self) -> None:
            self._load_generation += 1
            self._request_log_close()

        def action_logs(self) -> None:
            if self._log_source is not None or self._log_closing:
                return
            index = self.query_one("#services", ListView).index
            if index is None or not 0 <= index < len(self.observations):
                return
            selection = self.observations[index]
            try:
                source = self.log_provider.open_source(selection)
            except Exception as error:
                self.query_one("#state", Static).update(
                    f"Logs unavailable: {escape(str(error))}")
                return
            self._log_source = source
            self._log_paused = False
            self._log_lines.clear()
            self._log_characters = 0
            self._log_generation += 1
            generation = self._log_generation
            self.query_one("#services", ListView).display = False
            self.query_one("#details", Static).display = False
            log_view = self.query_one("#logs", RichLog)
            log_view.display = True
            log_view.clear()
            log_view.write("Waiting for log output…")
            log_view.focus()
            self.query_one("#state", Static).update(
                f"{escape(selection.name)} · {escape(selection.kind)} logs · following")
            self.query_one("#help", Static).update(self.LOGS_HELP)
            self.run_worker(self._follow_logs(source, generation),
                            name="follow-logs", group="logs", exclusive=True)

        def action_pause_logs(self) -> None:
            if self._log_source is None:
                return
            self._log_paused = not self._log_paused
            state = "paused" if self._log_paused else "following"
            self.query_one("#state", Static).update(f"Logs · {state}")

        async def action_quit(self) -> None:
            if self._log_source is not None and not await self._stop_logs():
                return
            self.exit()

        def _request_log_close(self) -> None:
            self._log_generation += 1
            source = self._log_source
            if source is not None:
                source.close()

        async def _stop_logs(self) -> bool:
            source = self._log_source
            if source is None:
                return True
            if self._log_closing:
                return False
            self._log_closing = True
            self._request_log_close()
            self.query_one("#state", Static).update("Closing log source…")
            try:
                await source.wait_closed()
            except TimeoutError:
                self.query_one("#state", Static).update(
                    "Log source is still closing")
                return False
            finally:
                self._log_closing = False
            if self._log_source is source:
                self._log_source = None
            return True

        async def _follow_logs(self, source, generation: int) -> None:
            while (generation == self._log_generation and
                   source is self._log_source and self.is_running):
                if self._log_paused:
                    await asyncio.sleep(0.05)
                    continue
                try:
                    batch = await source.poll()
                except Exception:
                    if generation == self._log_generation and source is self._log_source:
                        self._append_log("unavailable", "log source unavailable")
                    return
                if (generation != self._log_generation or
                        source is not self._log_source or not self.is_running):
                    return
                for entry in batch.entries:
                    self._append_log(entry.kind, entry.text)
                if batch.ended:
                    return
                await asyncio.sleep(0.15)

        def _append_log(self, kind: str, text: str) -> None:
            prefix = "" if kind == "data" else f"[{kind}] "
            rows = text.splitlines() or [""]
            for row in rows:
                rendered = prefix + row
                self._log_lines.append(rendered)
                self._log_characters += len(rendered)
            while (len(self._log_lines) > 500 or self._log_characters > 256 * 1024):
                self._log_characters -= len(self._log_lines.popleft())
            log_view = self.query_one("#logs", RichLog)
            log_view.clear()
            for line in self._log_lines:
                log_view.write(line)

        async def load_services(self, generation: int) -> None:
            state = self.query_one("#state", Static)
            try:
                observations = tuple(await self.provider.list_services())
            except Exception as error:  # Provider failures must remain in the UI.
                if generation != self._load_generation or not self.is_running:
                    return
                self.observations = ()
                state.update(f"Provider unavailable: {escape(str(error))}")
                listing = self.query_one("#services", ListView)
                listing.clear()
                listing.disabled = True
                self.query_one("#details", Static).update(
                    "Service observations could not be loaded.\n"
                    "Check the provider and try again.")
                return
            finally:
                if generation == self._load_generation:
                    self._loading = False

            if generation != self._load_generation or not self.is_running:
                return
            listing = self.query_one("#services", ListView)
            self.observations = observations
            listing.clear()
            if not self.observations:
                listing.disabled = True
                listing.index = None
                state.update("No services configured")
                self.query_one("#details", Static).update(
                    "No locally managed Loom Serve services were found.")
                if self._detail_identity is not None:
                    self.show_details()
                return

            for service in self.observations:
                prefix = "  ↳ " if service.kind == "model" else ""
                label = f"{prefix}{escape(service.name)}  ·  {escape(service.state)}"
                listing.append(ListItem(Label(label)))
            listing.disabled = False
            state.update(f"{len(self.observations)} service(s) · observations")
            if self._detail_identity is not None:
                listing.index = next((
                    index for index, service in enumerate(self.observations)
                    if service_identity(service) == self._detail_identity
                ), None)
                self.show_details()
            else:
                matching_index = next((
                    index for index, service in enumerate(self.observations)
                    if service_identity(service) == self._overview_identity
                ), None)
                if not self._overview_selection_established:
                    matching_index = 0
                listing.index = matching_index
                listing.focus()
                if matching_index is None:
                    self.query_one("#details", Static).update(
                        "Previous selection is unavailable.\n"
                        "Select a current service to continue.")
                else:
                    self.show_selection(matching_index)

        def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
            self.show_selection(event.list_view.index)

        def on_list_view_selected(self, _event: ListView.Selected) -> None:
            self.action_open_details()

        def show_selection(self, index: Optional[int]) -> None:
            if self._detail_identity is not None:
                return
            if index is None or not 0 <= index < len(self.observations):
                return
            service = self.observations[index]
            self._overview_identity = service_identity(service)
            self._overview_selection_established = True
            self.query_one("#details", Static).update(
                f"[b]{escape(service.name)}[/b]\n"
                f"State: {escape(service.state)}\n"
                f"Kind: {escape(service.kind)}\n"
                f"Endpoint: {escape(service.endpoint)}\n\n"
                f"{escape(service.summary)}")

        def show_details(self) -> None:
            identity = self._detail_identity
            if identity is None:
                return
            service = find_service(self.observations, identity)
            details = self.query_one("#details", Static)
            if service is None or not service.identity_verified:
                message = Text()
                message.append(identity.name or "Selected service", style="bold")
                message.append("\n\nSelected identity changed or is unavailable.")
                details.update(message)
                return
            content = Text()
            content.append(service.name or "Unavailable", style="bold")
            content.append("\n")
            for label, value in detail_fields(service):
                content.append(f"{label}: ", style="bold")
                content.append(value)
                content.append("\n")
            details.update(content)

    if provider is None:
        from .hem_tui_provider import LocalServiceProvider
        provider = LocalServiceProvider()
    if log_provider is None:
        from .hem_tui_logs import LocalLogProvider
        log_provider = LocalLogProvider()
    return LoomControlTui(provider, log_provider)


def run(args=None, deadline=None) -> int:
    """Launch the app, leaving terminal setup and restoration to Textual."""
    create_app().run()
    return 0
