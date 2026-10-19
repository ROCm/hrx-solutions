"""Bounded, credential-safe management HTTP for qualified Open WebUI.

No native inference client policy is changed here. Full configuration documents
(including foreign secrets) exist only in memory; callers own display/recovery.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import re
import socket
import threading
from urllib.parse import urlsplit

from .hem_common import Deadline, DeadlineExpired, HemError, InvalidInput, UncertainOutcome

QUALIFIED_VERSIONS = ("0.12.3",)
MAX_BYTES = 2 * 1024 * 1024
_FIELDS = {"ENABLE_OPENAI_API", "OPENAI_API_BASE_URLS", "OPENAI_API_KEYS", "OPENAI_API_CONFIGS"}


def canonical_target(value: str) -> str:
    """Return a direct numeric-loopback HTTP origin; never resolve hostnames."""
    try:
        if not isinstance(value, str) or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError
        parsed = urlsplit(value)
        address = ipaddress.ip_address(parsed.hostname or "")
        port = parsed.port
        if (parsed.scheme != "http" or not address.is_loopback or "%" in parsed.netloc
                or parsed.username is not None or parsed.password is not None
                or port is None or port == 0 or parsed.path not in ("", "/")
                or parsed.query or parsed.fragment or "?" in value or "#" in value):
            raise ValueError
        host = f"[{address.compressed}]" if address.version == 6 else address.compressed
        return f"http://{host}:{port}"
    except (ValueError, TypeError):
        raise InvalidInput("WebUI URL must be a numeric-loopback http://address:port origin") from None


def _json_bytes(value: object) -> bytes:
    # Reject coercions (non-string keys, tuples, NaN), not just malformed bytes.
    def check(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise ValueError
    try:
        check(value)
        body = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(body) > MAX_BYTES:
            raise ValueError
        return body
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise InvalidInput("WebUI JSON must be strict and at most 2 MiB") from None


def validate_config(value: object) -> dict:
    """Validate the losslessly round-trippable four-field management document."""
    _json_bytes(value)
    if type(value) is not dict or set(value) != _FIELDS:
        raise InvalidInput("WebUI configuration has unsupported top-level fields")
    urls, keys, configs = (value[name] for name in ("OPENAI_API_BASE_URLS", "OPENAI_API_KEYS", "OPENAI_API_CONFIGS"))
    if (type(value["ENABLE_OPENAI_API"]) is not bool or type(urls) is not list
            or type(keys) is not list or len(urls) != len(keys)
            or any(type(item) is not str for item in urls + keys) or type(configs) is not dict):
        raise InvalidInput("WebUI configuration has invalid enable, URL, key or connection fields")
    for index, config in configs.items():
        if (not re.fullmatch(r"0|[1-9][0-9]*", index) or len(index) > len(str(len(urls)))
                or int(index) >= len(urls) or type(config) is not dict):
            raise InvalidInput("WebUI configuration has an invalid connection index or shape")
    return value


def _decode(body: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError
            result[key] = value
        return result
    def nonfinite(_):
        raise ValueError
    try:
        value = json.loads(body.decode("utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)
        _json_bytes(value)
        return value
    except (ValueError, UnicodeError, RecursionError, InvalidInput):
        raise InvalidInput("WebUI response is not bounded strict JSON") from None


class WebUIClient:
    """One shared deadline, no redirects/proxies/retries or credential display."""

    def __init__(self, target: str, token: str, deadline: Deadline):
        self.target = canonical_target(target)
        if (not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9._~+/=-]+", token)):
            raise InvalidInput("OPENWEBUI_TOKEN must contain a nonempty bearer credential")
        self._token = token
        self.deadline = deadline

    def version(self) -> dict:
        value = self._request("GET", "/api/version")
        if (type(value) is not dict or type(value.get("version")) is not str
                or not value["version"] or type(value.get("deployment_id")) is not str):
            raise InvalidInput("WebUI version response lacks version/deployment identity strings")
        return value

    def config(self) -> dict:
        return validate_config(self._request("GET", "/openai/config"))

    def update(self, configuration: dict) -> dict:
        payload = _json_bytes(validate_config(configuration))
        value = self._request("POST", "/openai/config/update", payload)
        try:
            return validate_config(value)
        except InvalidInput:
            raise UncertainOutcome("WebUI accepted update but returned an invalid configuration; inspect before retrying",
                                   accepted=True) from None

    def _request(self, method, path, payload=None):
        parsed = urlsplit(self.target)
        address = ipaddress.ip_address(parsed.hostname)
        budget = self.deadline
        submitted = False
        status = None
        timer = None
        try:
            remaining = budget.remaining()
        except DeadlineExpired:
            raise DeadlineExpired("WebUI management deadline expired before submission",
                                  accepted=False if method == "POST" else None) from None
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=remaining)
        try:
            # Numeric literals only: bypass DNS and environment proxy handlers.
            sock = socket.socket(socket.AF_INET6 if address.version == 6 else socket.AF_INET, socket.SOCK_STREAM)
            connection.sock = sock
            sock.settimeout(budget.remaining())
            sock.connect((address.compressed, parsed.port))
            def interrupt():
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            timer = threading.Timer(budget.remaining(), interrupt)
            timer.daemon = True
            timer.start()
            sock.settimeout(budget.remaining())
            headers = {"Authorization": "Bearer " + self._token, "Accept": "application/json", "Connection": "close"}
            if payload is not None:
                headers["Content-Type"] = "application/json"
            budget.remaining()
            submitted = True  # Partial request writes cannot safely be replayed.
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            status = response.status
            # Do not read or render rejection bodies, which can echo credentials.
            if not 200 <= status < 300:
                if method == "POST" and status >= 500:
                    raise UncertainOutcome("WebUI update returned a server error; inspect before retrying")
                label = "authorization rejected (admin token/API-key route access required)" if status in (401, 403) else "management request rejected"
                raise HemError(f"WebUI HTTP {status}: {label}", accepted=False if method == "POST" else None)
            length = response.getheader("Content-Length")
            if length is not None and (not length.isdecimal() or len(length) > 10 or int(length) > MAX_BYTES):
                raise InvalidInput("WebUI response exceeds the 2 MiB bound or has invalid length")
            body = response.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES:
                raise InvalidInput("WebUI response exceeds the 2 MiB bound")
            if length is not None and len(body) != int(length):
                raise http.client.IncompleteRead(body)
            budget.remaining()
            return _decode(body)
        except InvalidInput:
            if method == "POST" and submitted:
                raise UncertainOutcome("WebUI accepted update but returned an invalid response; inspect before retrying",
                                       accepted=True if status is not None and 200 <= status < 300 else None) from None
            raise
        except (OSError, http.client.HTTPException, DeadlineExpired):
            if method == "POST" and submitted:
                raise UncertainOutcome("WebUI update response incomplete; inspect before retrying",
                                       accepted=True if status is not None and 200 <= status < 300 else None) from None
            try:
                budget.remaining()
            except DeadlineExpired:
                raise DeadlineExpired("WebUI management deadline expired", accepted=False if method == "POST" else None) from None
            raise HemError("WebUI management service is unreachable or the connection failed",
                           accepted=False if method == "POST" else None) from None
        finally:
            if timer is not None:
                timer.cancel()
            connection.close()
