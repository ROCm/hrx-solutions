"""Pure Pi wire contracts. No file, SQLite, process or network operations.

These schemas describe current configuration, never live inventory or history.
Operational owners must validate untrusted input; TypedDict is documentation,
not a runtime validator. Native profile acceptance still needs live GET checks.
"""

from __future__ import annotations

import base64
import re
from typing import Dict, List, TypedDict

from ...hem_common import InvalidInput


INTEGRATION_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
ENDPOINT_SCHEMA_VERSION = 1
TARGET_SCHEMA_VERSION = 2
EXTENSION_VERSION = "1"
PI_PACKAGE = "@earendil-works/pi-coding-agent"
PI_VERSIONS = ("1.3.0",)
NODE_MIN_VERSION = (22, 19, 0)
NATIVE_PROFILE = "loom-text-serve-61417137-v1"
NATIVE_COMMIT = "61417137f334cfa828b38d6e9b0f2d7190503f16"
# Exactly one prior owned generation can be inspected and explicitly refreshed.
# New projections and installations always emit the current identity above.
PREVIOUS_NATIVE_PROFILE = "loom-text-serve-ed6b79c-v1"
PREVIOUS_NATIVE_COMMIT = "ed6b79c757c738dea7eaecd6ca55a7df73cf6185"
READABLE_NATIVE_PROFILES = (NATIVE_PROFILE, PREVIOUS_NATIVE_PROFILE)
OUTPUT_CAP = 8192
REFRESH_TIMEOUT = 3.0
DISCOVERY_PATHS = ("/v1/models", "/v1/status")
MANIFEST_PATH = "loom-hem/config.json"
EXTENSION_PATH = "extensions/loom-hem"
ENDPOINT_PATH = "integrations/pi/endpoints.json"


class PiTarget(TypedDict):
    target: str  # canonical absolute Pi agent directory
    integration_id: str  # canonical UUID4; stable until disable completes


class Endpoint(TypedDict):
    server: str
    provider: str
    base_url: str  # numeric loopback http origin plus /v1; no userinfo/query
    native_profile: str


class EndpointDocument(TypedDict):
    schema_version: int
    integration_id: str
    hem_root: str
    servers: List[Endpoint]  # sorted by exact server name; never model inventory


class Manifest(TypedDict):
    schema_version: int
    integration_version: int
    integration_id: str
    enabled: bool
    hem_root: str
    target: str
    extension_version: str
    native_profile: str
    native_commit: str
    pi_package: str
    pi_versions: List[str]
    endpoint_file: str
    artifacts: Dict[str, str]  # target-relative regular files -> SHA256 hex


def provider_id(server: str) -> str:
    """Injective, case-preserving encoding; never lossy punctuation folding."""
    if not isinstance(server, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", server):
        raise InvalidInput("invalid Pi server name")
    encoded = base64.urlsafe_b64encode(server.encode("ascii")).decode("ascii").rstrip("=")
    return "loom-hem-" + encoded


class PiResult(dict):
    """Useful preview/failure payload plus the existing CLI dispatch exit hook."""

    def __init__(self, state: str, *, configured: bool, changed: bool,
                 completed: bool, exit_code: int = 0, accepted=None, **fields):
        if type(exit_code) is not int or exit_code not in range(5):
            raise InvalidInput("Pi result exit code must be 0, 1, 2, 3 or 4")
        super().__init__(state=state, configured=configured, changed=changed,
                         completed=completed, **fields)
        if accepted is not None:
            self["accepted"] = accepted
        self.exit_code = exit_code
