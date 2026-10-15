"""Native HTTP transport and observations, without a model lifecycle engine.

Routes/shapes follow HRX ed6b79c router.c and http/logs.c. No retries,
redirects, proxy routing, process adoption, or automatic lifecycle actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
from pathlib import Path
import queue
import socket
import threading
import time
from urllib.parse import urlsplit

from . import hem_display
from .hem_common import Deadline, DeadlineExpired, HemError, InvalidInput, UncertainOutcome


@dataclass(frozen=True)
class NativeResponse:
    status: int
    body: bytes
    accepted: bool | None = None

    def json(self):
        try:
            return json.loads(self.body)
        except (ValueError, UnicodeError) as error:
            raise HemError("native response is not valid JSON", accepted=self.accepted) from error


class NativeClient:
    """Direct HTTP client with a shared optional lifecycle Deadline.

    ownership(socket, deadline) must prove the actual connected socket belongs
    to the pinned process, or raise UncertainOutcome. It runs BEFORE transmitting
    request bytes. ownership_after(deadline) rechecks pinned process identity
    after the response. Native Connection: close means the accepted socket can
    already be gone then; this second check deliberately checks process identity.
    Managed callers must provide BOTH hooks. Process ownership belongs to caller.
    Explicit external URL control passes no hook and gains no process rights.
    """

    def __init__(self, endpoint, deadline=None, ownership=None, *, ownership_after=None, emit=hem_display.emit,
                 secrets=()):
        parsed = urlsplit(endpoint)
        try:
            port = parsed.port
        except ValueError as error:
            raise InvalidInput("invalid native endpoint port") from error
        if (parsed.scheme != "http" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in ("", "/")
                or parsed.query or parsed.fragment):
            raise InvalidInput("native endpoint must be a direct http://host:port origin")
        self.endpoint = endpoint.rstrip("/")
        self.host, self.port = parsed.hostname, port or 80
        self.deadline, self.ownership = deadline, ownership
        if (ownership is None) != (ownership_after is None):
            raise InvalidInput("managed HTTP requires both socket ownership and post-response identity checks")
        self.ownership_after = ownership_after
        self.emit, self.secrets = emit, secrets

    def get(self, path, *, display=True, headers=None):
        return self._request("GET", path, display=display, headers=headers)

    def post(self, path, body_path):
        path_obj = Path(body_path).absolute()
        try:
            body = path_obj.read_bytes()
        except OSError as error:
            raise InvalidInput(f"cannot read native request document: {error}") from error
        return self._request("POST", path, body=body, body_path=path_obj)

    def _request(self, method, path, *, body=None, body_path=None, display=True, headers=None):
        if not path.startswith("/") or path.startswith("//") or any(c in path for c in "\r\n?#"):
            raise InvalidInput("native request path must be an absolute path without query or fragment")
        # A bounded standalone observation still has one total I/O budget.
        budget = self.deadline or Deadline.after()
        budget.remaining()
        equivalent = hem_display.http(method, self.endpoint + path,
                                      body_path=body_path, body=body, secrets=self.secrets)
        request_headers = dict(headers or {})
        # Only native log selectors are exposed; no hidden authorization headers
        # or transport headers may make the printed curl differ from the request.
        if any(key not in ("X-Loom-Model", "X-Loom-Cursor") for key in request_headers):
            raise InvalidInput("unsupported native request header")
        if any(not isinstance(value, str) or any(c in value for c in "\r\n")
               for value in request_headers.values()):
            raise InvalidInput("invalid native request header value")
        if request_headers:
            equivalent += " " + hem_display.command(
                [part for key, value in request_headers.items() for part in ("--header", f"{key}: {value}")],
                secrets=self.secrets)
        if display:
            self.emit(equivalent)
        connection = http.client.HTTPConnection(self.host, self.port, timeout=budget.remaining())
        submitted = False
        status = None
        timer = None
        try:
            connection.sock = _connect(self.host, self.port, budget)
            connected_socket = connection.sock
            # Interrupt trickled headers/body as well as individual blocked reads.
            # A socket timeout alone resets per recv and is not a total deadline.
            def interrupt():
                try:
                    connected_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            timer = threading.Timer(budget.remaining(), interrupt)
            timer.daemon = True
            timer.start()
            if self.ownership is not None:
                self.ownership(connected_socket, budget)
            connected_socket.settimeout(budget.remaining())
            if body is not None:
                request_headers["Content-Type"] = "application/json"
            submitted = True  # Any partial write from request() is uncertain.
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            status = response.status
            payload = response.read()
            if self.ownership_after is not None:
                self.ownership_after(budget)
            budget.remaining()
            if 200 <= status < 300:
                return NativeResponse(status, payload, True if method == "POST" else None)
            diagnostic = payload.decode('utf-8', errors='replace')
            for secret in self.secrets:
                if secret:
                    diagnostic = diagnostic.replace(secret, "<redacted>")
            raise HemError(f"native HTTP {status}: {diagnostic}",
                           accepted=False if method == "POST" else None)
        except UncertainOutcome as error:
            # The post-response identity check can be uncertain even though
            # this socket already delivered an authoritative acknowledgment.
            if method == "POST" and status is not None:
                error.accepted = 200 <= status < 300
            raise
        except (OSError, http.client.HTTPException, DeadlineExpired) as error:
            if method == "POST" and submitted:
                if status is not None and 200 <= status < 300:
                    # Headers acknowledge acceptance even if the body is lost.
                    try:
                        budget.remaining()
                    except DeadlineExpired:
                        raise DeadlineExpired("native accepted request; response incomplete at deadline",
                                              accepted=True) from error
                    raise HemError("native accepted request but response body was lost; inspect status", accepted=True) from error
                if status is not None and status >= 400:
                    raise HemError(f"native HTTP {status}; error response body was lost", accepted=False) from error
                raise UncertainOutcome("native mutation response lost; inspect before retrying",
                                       accepted=None) from error
            budget.remaining()
            raise HemError(f"native HTTP observation/connection failed: {error}",
                           accepted=False if method == "POST" else None) from error
        finally:
            if timer is not None:
                timer.cancel()
            connection.close()

    def status(self, *, display=True):
        document = self.get("/v1/status", display=display).json()
        _inventory(document, require_incarnation=True)
        return document

    def models(self):
        document = self.get("/v1/models").json()
        _inventory(document, require_incarnation=False)
        return document

    def logs(self, *, model=None, cursor=None):
        """Bounded native NDJSON snapshot; follow streaming belongs to log CLI."""
        headers = {}
        if model is not None:
            headers["X-Loom-Model"] = model
        if cursor is not None:
            headers["X-Loom-Cursor"] = cursor
        return self.get("/logz", headers=headers)

    def wait(self, predicate, *, accepted=True, interval=0.05):
        """Poll native status only; never replay a mutation or infer acceptance.

        Callers provide the lifecycle completion predicate. The same deadline is
        used for all requests/sleeps. Explicit accepted=None preserves uncertainty
        if observation cannot resolve an earlier lost response.
        """
        if self.deadline is None:
            raise InvalidInput("native completion wait requires the invocation deadline")
        if interval <= 0:
            raise InvalidInput("poll interval must be positive")
        first = True
        try:
            while True:
                self.deadline.remaining()
                try:
                    document = self.status(display=first)
                except DeadlineExpired:
                    # Exhausting this same budget does not add uncertainty to
                    # previously acknowledged work. Classify expiry below using
                    # its known acceptance, regardless of the poll boundary.
                    raise
                except HemError as error:
                    # Acceptance does not establish completion. A failed or
                    # invalid observation cannot tell us the mutation's outcome.
                    raise UncertainOutcome(
                        f"native completion observation failed: {error}", accepted=accepted) from error
                first = False
                if predicate(document):
                    return document
                time.sleep(min(interval, self.deadline.remaining()))
        except DeadlineExpired as error:
            if accepted is None:
                raise UncertainOutcome("native outcome remains uncertain at deadline; inspect before retrying") from error
            raise DeadlineExpired("native accepted work remains pending at deadline", accepted=accepted) from error
        except HemError as error:
            # Predicate errors come from successfully observed native state,
            # so retain their classification (for example preparation failure).
            if error.accepted is None:
                error.accepted = accepted
            raise


def _connect(host, port, budget):
    # getaddrinfo has no socket timeout. Bound the caller's DNS wait without
    # letting an unresolved hostname consume the lifecycle budget indefinitely.
    answers = queue.Queue(maxsize=1)
    def resolve():
        try:
            answers.put((socket.getaddrinfo(host, port, type=socket.SOCK_STREAM), None))
        except OSError as error:
            answers.put((None, error))
    resolver = threading.Thread(target=resolve, daemon=True)
    resolver.start()
    try:
        addresses, error = answers.get(timeout=budget.remaining())
    except queue.Empty as error:
        raise DeadlineExpired("deadline expired resolving native endpoint") from error
    if error is not None:
        raise error
    last_error = None
    for family, kind, proto, _, address in addresses:
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(budget.remaining())
            sock.connect(address)
            return sock
        except (OSError, DeadlineExpired) as error:
            last_error = error
            sock.close()
    if last_error is not None:
        raise last_error
    raise OSError("native endpoint resolved to no addresses")


def _inventory(document, *, require_incarnation):
    if not isinstance(document, dict) or document.get("object") != "list" or not isinstance(document.get("data"), list):
        raise HemError("native inventory response has an invalid shape")
    if require_incarnation and not isinstance(document.get("server"), dict):
        raise HemError("retirement/registration observation requires /v1/status, not discovery")
    seen = set()
    for item in document["data"]:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or item["id"] in seen or not isinstance(item.get("loom"), dict)):
            raise HemError("native inventory contains invalid or duplicate registrations")
        seen.add(item["id"])
        incarnation = item["loom"].get("incarnation")
        if require_incarnation and (type(incarnation) is not int or incarnation <= 0):
            raise HemError("native status lacks a valid registration incarnation")
        if any(key in item["loom"] and not isinstance(item["loom"][key], dict)
               for key in ("status", "parameters")):
            raise HemError("native status contains invalid state/parameter details")
    return document["data"]


def registration(document, name, incarnation=None):
    """Find an exact status incarnation, never from discovery /v1/models."""
    for item in _inventory(document, require_incarnation=True):
        if item["id"] == name and (incarnation is None or item["loom"]["incarnation"] == incarnation):
            return item
    return None


def retired(document, name, incarnation):
    if type(incarnation) is not int or incarnation <= 0:
        raise InvalidInput("retirement requires the original positive native incarnation")
    return registration(document, name, incarnation) is None


def active_ready(document, name, incarnation):
    """Native activation completion while admission remains closed."""
    item = registration(document, name, incarnation)
    if item is None:
        raise HemError("expected native registration incarnation disappeared")
    loom = item["loom"]
    if loom.get("error") is not None:
        raise HemError(f"native preparation failed: {loom['error']}", accepted=True)
    return (loom.get("registration") == "ready" and loom.get("parameter_operation") == "idle"
            and loom.get("parameters", {}).get("active") is True
            and loom.get("status", {}).get("admission") == "drained")
