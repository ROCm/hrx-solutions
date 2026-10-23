"""Bounded, credential-safe management HTTP for qualified Open WebUI.

No native inference client policy is changed here. Full configuration documents
(including foreign secrets) exist only in memory; callers own display/recovery.
"""

from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import threading
from urllib.parse import urlsplit
from urllib.parse import quote

from ...hem_common import Deadline, DeadlineExpired, HemError, InvalidInput, UncertainOutcome

from .hem_openwebui_wire import (MAX_BYTES, canonical_target,
    json_bytes as _json_bytes, decode_json as _decode, validate_config, validate_audio)


QUALIFIED_VERSIONS = ("0.12.3",)


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

    def audio_config(self):
        return validate_audio(self._request("GET", "/api/v1/audio/config"))

    def update_audio(self, configuration):
        validate_audio(configuration)
        return self._request("POST", "/api/v1/audio/config/update", _json_bytes(configuration))

    def model(self, model_id):
        return self._request("GET", "/api/v1/models/model?id=" + quote(model_id, safe=""), missing_ok=True)

    def create_model(self, body):
        return self._request("POST", "/api/v1/models/create", _json_bytes(body))

    def delete_model(self, model_id):
        return self._request("POST", "/api/v1/models/model/delete", _json_bytes(dict(id=model_id)))

    def _request(self, method, path, payload=None, *, missing_ok=False):
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
            if missing_ok and method == "GET" and status == 404:
                return None
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
