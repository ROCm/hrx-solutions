"""Identity and presentation helpers for basic Loom Control TUI details."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


UNAVAILABLE = "Unavailable"


@dataclass(frozen=True)
class ServiceIdentity:
    """Stable selection key for one observed server launch or model instance."""

    kind: str
    server: str
    name: str
    launch_token: Optional[str]
    incarnation: Optional[int]


def service_identity(service) -> ServiceIdentity:
    """Return the complete managed identity represented by an observation."""
    kind = service.kind
    return ServiceIdentity(
        kind=kind,
        server=service.server or service.name,
        name=service.name,
        launch_token=service.launch_token,
        incarnation=service.incarnation if kind == "model" else None,
    )


def find_service(services, identity: ServiceIdentity):
    """Find only an exact identity; never substitute a replacement launch."""
    for service in services:
        if service_identity(service) == identity:
            return service
    return None


def display_value(value) -> str:
    """Normalize an absent value without inventing service state."""
    if value is None or value == "":
        return UNAVAILABLE
    return str(value)


def detail_fields(service) -> Tuple[Tuple[str, str], ...]:
    """Build the intentionally small first-release details record."""
    fields = [
        ("Kind", display_value(service.kind)),
        ("Endpoint", display_value(service.endpoint)),
        ("Server identity", display_value(service.server or service.name)),
        ("Launch identity", display_value(service.launch_token)),
        ("Recorded state", display_value(service.recorded_state)),
        ("Process", display_value(service.process_state)),
        ("HTTP", display_value(service.http_state)),
    ]
    if service.kind == "model":
        fields.extend((
            ("Model identity", display_value(service.name)),
            ("Incarnation", display_value(service.incarnation)),
            ("Readiness", display_value(service.state)),
        ))
    return tuple(fields)
