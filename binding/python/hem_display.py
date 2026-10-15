"""Inspectable equivalents on stderr, never an execution or trace facility."""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path
from typing import Mapping, Sequence, TextIO
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .hem_common import InvalidInput


_SENSITIVE = re.compile(
    r"(?:^|[_-])(token|password|passwd|secret|credentials?|authorization|api[_-]?key)(?:$|[_-])",
    re.I,
)
_REDACTED = "<redacted>"


def _url(value: str) -> str:
    if not value.startswith(("http://", "https://")):
        return value
    parsed = urlsplit(value)
    host = parsed.netloc.rsplit("@", 1)[-1]
    if "@" in parsed.netloc:
        host = _REDACTED + "@" + host
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query = parsed.query
    if any(_SENSITIVE.search(key) for key, _ in pairs):
        query = urlencode([(key, _REDACTED if _SENSITIVE.search(key) else val)
                           for key, val in pairs])
    return urlunsplit((parsed.scheme, host, parsed.path, query, parsed.fragment))


def command(argv: Sequence[str], *, environment: Mapping[str, str] | None = None,
            secrets: Sequence[str] = ()) -> str:
    """Quote argv faithfully except for explicit or conventional credentials.

    Callers supply secrets for domain-specific values that cannot be inferred.
    Never execute this display string: redacted commands need credentials again.
    """
    values = []
    redact_next = False
    for raw in argv:
        value = str(raw)
        if redact_next:
            value = _REDACTED
            redact_next = False
        elif value.startswith("--") and _SENSITIVE.search(value.split("=", 1)[0]):
            if "=" in value:
                value = value.split("=", 1)[0] + "=" + _REDACTED
            else:
                redact_next = True
        elif value.startswith(("--user=", "--oauth2-bearer=")):
            value = value.split("=", 1)[0] + "=" + _REDACTED
        elif value.startswith("-u") and not value.startswith("--") and len(value) > 2:
            value = "-u" + _REDACTED
        elif value in ("-u", "--user", "--oauth2-bearer"):
            redact_next = True
        elif ":" in value and _SENSITIVE.search(
                value.split(":", 1)[0].removeprefix("--header=").removeprefix("-H")):
            value = value.split(":", 1)[0] + ": " + _REDACTED
        else:
            value = _url(value)
        for secret in secrets:
            if secret:
                value = value.replace(secret, _REDACTED)
        values.append(value)
    assignments = []
    for key, raw in (environment or {}).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise InvalidInput(f"invalid environment variable name: {key}")
        value = _REDACTED if _SENSITIVE.search(key) else _url(raw)
        for secret in secrets:
            if secret:
                value = value.replace(secret, _REDACTED)
        assignments.append(f"{key}={value}")
    rendered = shlex.join((["env", *assignments] if assignments else []) + values)
    if _REDACTED in rendered or "%3Credacted%3E" in rendered:
        rendered += "  # credentials redacted; replay needs existing credentials"
    return rendered


def http(method: str, url: str, *, body_path: Path | None = None,
         body: bytes | None = None, secrets: Sequence[str] = ()) -> str:
    """Show the actual published body path, verifying its transport bytes.

    The adapter must send these same bytes and keep the file inspectable. This
    helper does not create files or decide their lifetime or publication policy.
    """
    argv = ["curl", "--fail-with-body", "--silent", "--show-error", "--request", method, url]
    if body_path is not None or body is not None:
        if body_path is None or body is None:
            raise InvalidInput("HTTP equivalent requires both the body file and submitted bytes")
        try:
            actual = body_path.read_bytes()
        except OSError as error:
            raise InvalidInput(f"cannot inspect HTTP body file: {error}") from error
        if actual != body:
            raise InvalidInput("HTTP body file differs from submitted bytes")
        argv += ["--header", "Content-Type: application/json", "--data-binary", "@" + str(body_path)]
    return command(argv, secrets=secrets)


def emit(equivalent: str, *, stream: TextIO | None = None) -> None:
    print(equivalent, file=sys.stderr if stream is None else stream, flush=True)
