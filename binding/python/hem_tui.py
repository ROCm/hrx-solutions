"""Textual shell for interactive, read-only Loom Control observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence


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


def create_app(provider: Optional[ServiceProvider] = None):
    """Build the interactive app; importing the CLI does not import Textual."""
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal
    from textual.widgets import Footer, Header, Label, ListItem, ListView, Static
    from rich.markup import escape

    class LoomControlTui(App[None]):
        TITLE = "Loom Control"
        SUB_TITLE = "Local service observer"
        CSS = """
        #body { height: 1fr; }
        #services { width: 36%; min-width: 24; border: round $primary; }
        #details { width: 1fr; border: round $primary; padding: 1 2; }
        #state { height: 1; padding: 0 1; }
        #help { height: 1; dock: bottom; padding: 0 1; background: $boost; }
        .selected { text-style: bold; }
        """
        BINDINGS = [
            ("r", "refresh", "Refresh"),
            ("q", "quit", "Quit"),
            ("escape", "quit", "Quit"),
        ]

        def __init__(self, data_provider: ServiceProvider) -> None:
            super().__init__()
            self.provider = data_provider
            self.observations: Sequence[ServiceObservation] = ()
            self._load_generation = 0
            self._loading = False

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("Loading services…", id="state")
            with Horizontal(id="body"):
                yield ListView(id="services", disabled=True)
                yield Static("Waiting for service observations.", id="details")
            yield Static("↑/↓ Select   r Refresh   q Quit", id="help")
            yield Footer()

        def on_mount(self) -> None:
            self.request_refresh()

        def request_refresh(self) -> None:
            if self._loading:
                return
            self._loading = True
            self._load_generation += 1
            generation = self._load_generation
            self.run_worker(self.load_services(generation), name="load-services")

        def action_refresh(self) -> None:
            self.request_refresh()

        def on_unmount(self) -> None:
            self._load_generation += 1

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
            self.observations = observations

            listing = self.query_one("#services", ListView)
            listing.clear()
            if not self.observations:
                listing.disabled = True
                state.update("No services configured")
                self.query_one("#details", Static).update(
                    "No locally managed Loom Serve services were found.")
                return

            for service in self.observations:
                prefix = "  ↳ " if service.kind == "model" else ""
                label = f"{prefix}{escape(service.name)}  ·  {escape(service.state)}"
                listing.append(ListItem(Label(label)))
            listing.disabled = False
            state.update(f"{len(self.observations)} service(s) · observations")
            listing.index = 0
            listing.focus()
            self.show_selection(0)

        def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
            self.show_selection(event.list_view.index)

        def show_selection(self, index: Optional[int]) -> None:
            if index is None or not 0 <= index < len(self.observations):
                return
            service = self.observations[index]
            self.query_one("#details", Static).update(
                f"[b]{escape(service.name)}[/b]\n"
                f"State: {escape(service.state)}\n"
                f"Kind: {escape(service.kind)}\n"
                f"Endpoint: {escape(service.endpoint)}\n\n"
                f"{escape(service.summary)}")

    if provider is None:
        from .hem_tui_provider import LocalServiceProvider
        provider = LocalServiceProvider()
    return LoomControlTui(provider)


def run(args=None, deadline=None) -> int:
    """Launch the app, leaving terminal setup and restoration to Textual."""
    create_app().run()
    return 0
