"""Bounded, identity-pinned log sources for the read-only Loom Control TUI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import threading
from typing import Optional, Sequence

from .hem_common import Deadline, HemError, InvalidInput
from .hem_http import NativeClient, registration
from .hem_process import PinnedProcess, _confirmed_exit
from .hem_state import _name
from .hem_tui_provider import _read_snapshot
from .runtime_paths import state_root

MAX_LOG_BATCH_BYTES = 256 * 1024
MAX_NATIVE_LOG_BYTES = 1024 * 1024


class _StreamChanged(HemError):
    pass


class _TrackedLogSource:
    """Keep cancellation from outliving the blocking poll it started."""

    def _init_tracking(self, timeout: float) -> None:
        self.timeout = timeout
        self._closed = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._active_lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._active_polls = 0

    async def poll(self) -> LogBatch:
        with self._active_lock:
            if self._closed.is_set():
                return LogBatch()
            self._active_polls += 1
            self._idle.clear()
        task = asyncio.create_task(asyncio.to_thread(self._run_poll))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # asyncio cannot stop the thread. Keep this coroutine alive until
            # the source's own deadline releases the descriptor/socket.
            await asyncio.shield(task)
            raise
        finally:
            with self._active_lock:
                self._active_polls -= 1
                if self._active_polls == 0:
                    self._idle.set()

    def _run_poll(self) -> LogBatch:
        with self._poll_lock:
            if self._closed.is_set():
                return LogBatch()
            return self._poll()

    def close(self) -> None:
        self._closed.set()

    async def wait_closed(self) -> None:
        """Wait at most one poll budget plus scheduling margin."""
        loop = asyncio.get_running_loop()
        expires = loop.time() + self.timeout + 0.5
        while not self._idle.is_set():
            if loop.time() >= expires:
                raise TimeoutError("log source did not release its active poll")
            await asyncio.sleep(0.01)


@dataclass(frozen=True)
class LogEntry:
    """One display-safe chunk or explicit continuity marker."""

    kind: str
    text: str


@dataclass(frozen=True)
class LogBatch:
    entries: Sequence[LogEntry] = ()
    ended: bool = False


@dataclass(frozen=True)
class _FileRead:
    entries: Sequence[LogEntry] = ()
    bytes_read: int = 0
    at_eof: bool = False
    continuity_lost: bool = False


@dataclass(frozen=True)
class LogSelection:
    kind: str
    server: str
    launch_token: str
    model: Optional[str] = None
    incarnation: Optional[int] = None

    @classmethod
    def from_observation(cls, observation):
        if observation.kind not in ("server", "model"):
            raise InvalidInput("logs require a managed server or model selection")
        if not observation.server or not observation.launch_token:
            raise InvalidInput("selected service has no current managed launch")
        server, launch_token = _name(observation.server), _name(observation.launch_token)
        if observation.kind == "model":
            if (not observation.name or type(observation.incarnation) is not int
                    or observation.incarnation <= 0):
                raise InvalidInput("selected model has no current managed incarnation")
            return cls("model", server, launch_token,
                       _name(observation.name), observation.incarnation)
        return cls("server", server, launch_token)


def _records(root: Path, selection: LogSelection, deadline: Deadline):
    servers, instances = _read_snapshot(root, deadline)
    server = next((item.record for item in servers
                   if item.record["id"] == selection.server and
                   item.record["launch_token"] == selection.launch_token), None)
    instance = None
    if selection.kind == "model":
        instance = next((item.record for item in instances
                         if item.record["server_id"] == selection.server and
                         item.record["launch_token"] == selection.launch_token and
                         item.record["name"] == selection.model and
                         item.record["incarnation"] == selection.incarnation), None)
    return server, instance


class ServerLogSource(_TrackedLogSource):
    """Follow fixed retained process files resolved from a managed launch."""

    _NAMES = ("process.log", "process.log.1", "bootstrap-error.txt")

    def __init__(self, root: Path, selection: LogSelection, *, timeout: float,
                 batch_bytes: int = MAX_LOG_BATCH_BYTES):
        self.root, self.selection = root, selection
        self._init_tracking(timeout)
        self.batch_bytes = batch_bytes
        self._identity = None
        self._offset = 0
        self._unavailable = False
        self._ended = False

    def _managed_directory(self, record) -> Path:
        expected = self.root / "servers" / self.selection.server / self.selection.launch_token
        build = record.get("build")
        if not isinstance(build, dict) or Path(build.get("directory", "")) != expected:
            raise HemError("managed server log directory is unavailable")
        return expected

    def _open_log(self, _directory: Path):
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        directory_fd = os.open(str(self.root), directory_flags)
        try:
            for component in ("servers", self.selection.server,
                              self.selection.launch_token):
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            directory_info = os.fstat(directory_fd)
            if (not stat.S_ISDIR(directory_info.st_mode)
                    or directory_info.st_uid != os.getuid()):
                raise HemError("managed server log directory is unavailable")
            for name in self._NAMES:
                try:
                    flags = os.O_RDONLY | os.O_CLOEXEC
                    if hasattr(os, "O_NOFOLLOW"):
                        flags |= os.O_NOFOLLOW
                    descriptor = os.open(name, flags, dir_fd=directory_fd)
                except FileNotFoundError:
                    continue
                try:
                    info = os.fstat(descriptor)
                    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                        raise HemError("managed server log file is unavailable")
                    return descriptor, name, info
                except Exception:
                    os.close(descriptor)
                    raise
            return None
        finally:
            os.close(directory_fd)

    def _read_file(self, record, limit=None):
        limit = self.batch_bytes if limit is None else limit
        try:
            opened = self._open_log(self._managed_directory(record))
        except (HemError, OSError):
            opened = None
        if opened is None:
            continuity_lost = self._identity is not None
            if self._unavailable:
                return _FileRead(at_eof=not continuity_lost,
                                 continuity_lost=continuity_lost)
            self._unavailable = True
            entries = []
            if continuity_lost:
                entries.append(LogEntry("gap", "server log file disappeared"))
            entries.append(LogEntry("unavailable", "server log unavailable"))
            return _FileRead(tuple(entries), at_eof=not continuity_lost,
                             continuity_lost=continuity_lost)
        descriptor, _name_on_disk, info = opened
        entries = []
        try:
            identity = (info.st_dev, info.st_ino)
            if self._identity is None:
                self._offset = max(0, info.st_size - self.batch_bytes)
                if self._offset:
                    entries.append(LogEntry("gap", "earlier server log bytes omitted"))
            elif identity != self._identity:
                entries.append(LogEntry("reset", "server log file changed"))
                self._offset = max(0, info.st_size - self.batch_bytes)
                if self._offset:
                    entries.append(LogEntry("gap", "earlier replacement log bytes omitted"))
            elif info.st_size < self._offset:
                entries.append(LogEntry("gap", "server log was truncated"))
                self._offset = 0
            self._identity = identity
            self._unavailable = False
            os.lseek(descriptor, self._offset, os.SEEK_SET)
            data = os.read(descriptor, limit)
            self._offset += len(data)
            if data:
                entries.append(LogEntry("data", data.decode("utf-8", errors="replace")))
            final_info = os.fstat(descriptor)
            at_eof = ((final_info.st_dev, final_info.st_ino) == identity
                      and self._offset >= final_info.st_size)
        finally:
            os.close(descriptor)
        return _FileRead(tuple(entries), len(data), at_eof)

    def _poll(self) -> LogBatch:
        if self._closed.is_set() or self._ended:
            return LogBatch(ended=self._ended)
        deadline = Deadline.after(self.timeout)
        record, _ = _records(self.root, self.selection, deadline)
        if record is None:
            self._ended = True
            return LogBatch((LogEntry("end", "server launch ended"),), True)
        initial = self._read_file(record)
        entries = list(initial.entries)
        current, _ = _records(self.root, self.selection, deadline)
        if self._closed.is_set():
            return LogBatch()
        if current is None:
            if not initial.at_eof and not initial.continuity_lost:
                entries.append(LogEntry(
                    "gap", "server launch ended before retained log was drained"))
            self._ended = True
            return LogBatch(tuple(entries) +
                            (LogEntry("end", "server launch ended"),), True)
        try:
            process = current.get("process")
            process_ended = bool(process and _confirmed_exit(process))
        except (HemError, KeyError, TypeError, ValueError, OSError):
            process_ended = False
        if process_ended:
            logger = process.get("logger") if isinstance(process, dict) else None
            try:
                logger_ended = not logger or _confirmed_exit(logger)
            except (HemError, KeyError, TypeError, ValueError, OSError):
                logger_ended = False
            if not logger_ended:
                return LogBatch(tuple(entries))
            # Reopen/re-stat after logger exit is proven. Consume only the
            # remainder of this poll's byte budget; later polls drain the rest.
            final = self._read_file(
                current, max(0, self.batch_bytes - initial.bytes_read))
            entries.extend(final.entries)
            if not final.at_eof and not final.continuity_lost:
                return LogBatch(tuple(entries))
            self._ended = True
            return LogBatch(tuple(entries) +
                            (LogEntry("end", "server process ended"),), True)
        return LogBatch(tuple(entries))


def _parse_native_batch(payload: bytes, selection: LogSelection,
                        previous_cursor: Optional[str]):
    if not payload.endswith(b"\n"):
        raise HemError("native log response is incomplete")
    try:
        records = [json.loads(line) for line in payload.splitlines()]
    except (ValueError, UnicodeError) as error:
        raise HemError("native log response is malformed") from error
    if (len(records) < 2 or any(not isinstance(item, dict) for item in records)
            or records[0].get("event") != "stream"
            or records[-1].get("event") != "cursor"):
        raise HemError("native log response is incomplete")
    header, footer = records[0], records[-1]
    identity = header.get("identity")
    if (header.get("scope") != "model" or header.get("model") != selection.model
            or not isinstance(identity, str) or len(identity.split(":")) != 2
            or not all(part.isascii() and part.isdecimal()
                       for part in identity.split(":"))):
        raise HemError("native log response has an invalid stream identity")
    if previous_cursor is not None:
        if not previous_cursor.startswith(identity + ":"):
            raise _StreamChanged(
                "native log stream changed without rejecting its cursor")
        try:
            position = int(previous_cursor.rsplit(":", 1)[1])
        except (ValueError, IndexError) as error:
            raise HemError("native log cursor is malformed") from error
    else:
        position = 0
    for item in records[1:-1]:
        event = item.get("event")
        if not isinstance(event, str):
            raise HemError("native log event is malformed")
        if event == "gap":
            end = item.get("to")
            if item.get("from") != position or type(end) is not int or end <= position:
                raise HemError("native log gap is malformed")
            position = end
        elif event != "counters":
            if type(item.get("sequence")) is not int or item["sequence"] != position:
                raise HemError("native log response has an unmarked discontinuity")
            position += 1
    cursor = footer.get("value")
    if footer.get("next") != position or cursor != f"{identity}:{position}":
        raise HemError("native log cursor does not cover the batch")
    entries = []
    for item in records[1:-1]:
        if item["event"] == "gap":
            entries.append(LogEntry("gap", f"native log gap {item['from']}..{item['to']}"))
        else:
            entries.append(LogEntry("data", json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))))
    return identity, cursor, tuple(entries)


class NativeModelLogSource(_TrackedLogSource):
    """Poll managed model telemetry without joining different incarnations."""

    def __init__(self, root: Path, selection: LogSelection, *, timeout: float):
        self.root, self.selection = root, selection
        self._init_tracking(timeout)
        self._cursor = None
        self._stream_identity = None
        self._reset_pending = False
        self._unavailable = False

    def _unavailable_batch(self, text="model log unavailable"):
        if self._unavailable:
            return LogBatch()
        self._unavailable = True
        return LogBatch((LogEntry("unavailable", text),))

    def _poll(self) -> LogBatch:
        if self._closed.is_set():
            return LogBatch()
        deadline = Deadline.after(self.timeout)
        record, instance = _records(self.root, self.selection, deadline)
        if record is None or instance is None or not record.get("process") or not record.get("build"):
            return self._unavailable_batch()
        try:
            with PinnedProcess(record["process"], record["build"]["executable"],
                               emit=lambda _message: None) as process:
                client = NativeClient(
                    record["endpoint"], deadline, process.connection,
                    ownership_after=process.verify, emit=lambda _message: None,
                )
                before = client.status(display=False, max_response_bytes=MAX_NATIVE_LOG_BYTES)
                if registration(before, self.selection.model,
                                self.selection.incarnation) is None:
                    return self._unavailable_batch()
                try:
                    sent_cursor = self._cursor is not None
                    response = client.logs(
                        model=self.selection.model, cursor=self._cursor,
                        max_response_bytes=MAX_NATIVE_LOG_BYTES)
                except HemError as error:
                    # Only a complete native rejection invalidates a cursor;
                    # observed headers on a malformed/oversized body do not.
                    status = error.status if error.reason == "http_status" else None
                    if status in (400, 409) and sent_cursor:
                        self._cursor = None
                        self._reset_pending = True
                        self._unavailable = False
                        return LogBatch((LogEntry("reset", "native log cursor reset"),))
                    if status == 404:
                        return self._unavailable_batch()
                    return self._unavailable_batch("model log unavailable")
                after = client.status(display=False, max_response_bytes=MAX_NATIVE_LOG_BYTES)
                if registration(after, self.selection.model,
                                self.selection.incarnation) is None:
                    return self._unavailable_batch()
        except (HemError, KeyError, TypeError, ValueError, OSError):
            return self._unavailable_batch()
        try:
            identity, cursor, entries = _parse_native_batch(
                response.body, self.selection, self._cursor)
        except _StreamChanged:
            self._cursor = None
            self._reset_pending = True
            return LogBatch((LogEntry("reset", "native log stream changed"),))
        except HemError:
            return self._unavailable_batch("model log response unavailable")
        current, current_instance = _records(self.root, self.selection, deadline)
        if self._closed.is_set():
            return LogBatch()
        if current is None or current_instance is None:
            return self._unavailable_batch()
        if (self._stream_identity is not None and identity != self._stream_identity
                and not self._reset_pending):
            self._cursor = None
            self._reset_pending = True
            return LogBatch((LogEntry("reset", "native log stream changed"),))
        prefix = ()
        if self._reset_pending:
            prefix = (LogEntry("reset", "native log stream restarted"),)
        self._stream_identity, self._cursor = identity, cursor
        self._reset_pending = False
        self._unavailable = False
        return LogBatch(prefix + entries)


class LocalLogProvider:
    """Create only sources authorized by an existing TUI observation identity."""

    def __init__(self, root=None, *, timeout=1.0,
                 batch_bytes=MAX_LOG_BATCH_BYTES):
        self.root = Path(root) if root is not None else state_root()
        if not self.root.is_absolute():
            raise ValueError("TUI state root must be absolute")
        if (type(batch_bytes) is not int or not 0 < batch_bytes <= 4 * 1024 * 1024):
            raise ValueError("TUI log batch limit must be between 1 and 4194304 bytes")
        self.timeout, self.batch_bytes = timeout, batch_bytes

    def open_source(self, observation):
        selection = LogSelection.from_observation(observation)
        if selection.kind == "server":
            return ServerLogSource(self.root, selection, timeout=self.timeout,
                                   batch_bytes=self.batch_bytes)
        return NativeModelLogSource(self.root, selection, timeout=self.timeout)
