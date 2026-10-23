"""Identity, continuity, and bounds checks for TUI log sources."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import http.server
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from binding.python.tui import hem_tui, hem_tui_logs
from binding.python.hem_common import HemError, UncertainOutcome
from binding.python.control.hem_state import Registry


def observation(kind="server"):
    return hem_tui.ServiceObservation(
        name="chat" if kind == "model" else "local",
        endpoint="http://127.0.0.1:1", state="running", summary="fixture",
        kind=kind, server="local", launch_token="launch-one",
        incarnation=17 if kind == "model" else None,
    )


def native_status():
    return json.dumps({
        "object": "list", "server": {"worker_count": 1},
        "data": [{
            "id": "chat", "loom": {
                "incarnation": 17, "registration": "ready",
                "parameter_operation": "idle", "parameters": {"active": True},
                "status": {"admission": "open", "phase": "idle"},
            },
        }],
    }).encode()


def native_batch(records=(), position=0, identity="100:1"):
    header = {"event": "stream", "scope": "model", "model": "chat",
              "identity": identity, "oldest": 0, "head": position,
              "capacity": 128, "partial": False, "counters_available": False}
    footer = {"event": "cursor", "next": position,
              "value": f"{identity}:{position}"}
    return b"".join(json.dumps(item).encode() + b"\n"
                    for item in (header, *records, footer))


@contextlib.contextmanager
def endpoint(log_replies):
    replies = collections.deque(log_replies)
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, dict(self.headers)))
            if self.path == "/v1/status":
                status, body = 200, native_status()
            else:
                status, body = replies.popleft()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class FakePinned:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def connection(self, _socket, _deadline):
        pass

    def verify(self, _deadline=None):
        pass


class SourceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "state"
        self.registry = Registry(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def running_server(self, endpoint="http://127.0.0.1:1"):
        self.registry.create_server("local", saved_settings={})
        record = self.registry.reserve_launch(
            "local", launch_token="launch-one", current_settings={}, endpoint=endpoint)
        directory = self.root / "servers/local/launch-one"
        directory.mkdir(parents=True, exist_ok=True)
        return self.registry.update_server(
            "local", {"build": {"directory": str(directory), "executable": {}},
                      "process": {"fixture": True}, "phase": "running", "pending": None},
            expected_token="launch-one", expected_revision=record["revision"])

    async def test_server_append_repeated_bytes_rotation_truncation_and_end(self):
        record = self.running_server()
        path = self.root / "servers/local/launch-one/process.log"
        path.write_bytes(b"same\n")
        source = hem_tui_logs.LocalLogProvider(
            self.root, timeout=1, batch_bytes=32).open_source(observation())
        first = await source.poll()
        self.assertEqual(first.entries[-1].text, "same\n")
        with path.open("ab") as stream:
            stream.write(b"same\n")
        self.assertEqual((await source.poll()).entries[-1].text, "same\n")

        path.replace(path.with_name("process.log.1"))
        self.assertEqual((await source.poll()).entries, ())
        path.write_bytes(b"replacement\n")
        rotated = await source.poll()
        self.assertEqual(rotated.entries[0].kind, "reset")
        self.assertIn("replacement", rotated.entries[-1].text)

        path.write_bytes(b"tiny\n")
        truncated = await source.poll()
        self.assertEqual(truncated.entries[0].kind, "gap")
        self.assertIn("tiny", truncated.entries[-1].text)

        path.unlink()
        path.with_name("process.log.1").unlink()
        self.assertEqual(
            [entry.kind for entry in (await source.poll()).entries],
            ["gap", "unavailable"])
        path.write_text("back\n")
        self.assertEqual((await source.poll()).entries[0].kind, "reset")

        self.registry.clear_server("local", expected_token="launch-one",
                                   expected_revision=record["revision"])
        ended = await source.poll()
        self.assertTrue(ended.ended)
        self.assertEqual(ended.entries[0].kind, "end")

    async def test_server_initial_tail_is_bounded_and_paths_are_internal(self):
        record = self.running_server()
        path = self.root / "servers/local/launch-one/process.log"
        path.write_bytes(b"0123456789")
        source = hem_tui_logs.LocalLogProvider(
            self.root, timeout=1, batch_bytes=4).open_source(observation())
        batch = await source.poll()
        self.assertEqual([item.kind for item in batch.entries], ["gap", "data"])
        self.assertEqual(batch.entries[-1].text, "6789")

        directory = path.parent
        retained = directory.with_name("launch-one-retained")
        outside = self.root.parent / "outside-directory"
        outside.mkdir()
        directory.rename(retained)
        directory.symlink_to(outside, target_is_directory=True)
        symlinked = hem_tui_logs.LocalLogProvider(
            self.root, timeout=1).open_source(observation())
        self.assertEqual((await symlinked.poll()).entries[0].kind, "unavailable")
        directory.unlink()
        retained.rename(directory)

        self.registry.update_server(
            "local", {"build": {"directory": str(self.root.parent / "outside"),
                                  "executable": {}}},
            expected_token="launch-one", expected_revision=record["revision"])
        rejected = hem_tui_logs.LocalLogProvider(
            self.root, timeout=1).open_source(observation())
        self.assertEqual((await rejected.poll()).entries[0].kind, "unavailable")
        with self.assertRaisesRegex(Exception, "managed"):
            hem_tui_logs.LocalLogProvider(self.root).open_source(
                hem_tui.ServiceObservation("x", "file:///tmp/private", "running", "x",
                                           server="missing", launch_token=None))
        for server in ("../outside", "line\nbreak"):
            with self.subTest(server=server), self.assertRaisesRegex(Exception, "IDs"):
                hem_tui_logs.LocalLogProvider(self.root).open_source(
                    hem_tui.ServiceObservation("x", "http://127.0.0.1:1", "running", "x",
                                               server=server, launch_token="launch-one"))

    async def test_server_reports_confirmed_process_end_after_final_bytes(self):
        self.running_server()
        path = self.root / "servers/local/launch-one/process.log"
        path.write_text("final line\n")
        source = hem_tui_logs.LocalLogProvider(
            self.root, timeout=1).open_source(observation())
        with patch.object(hem_tui_logs, "_confirmed_exit", return_value=True):
            batch = await source.poll()
        self.assertTrue(batch.ended)
        self.assertEqual([entry.kind for entry in batch.entries], ["data", "end"])
        self.assertIn("final line", batch.entries[0].text)

    async def test_server_waits_for_logger_and_reads_bytes_written_after_native_exit(self):
        record = self.running_server()
        process = {"pid": "native", "logger": {"pid": "logger"}}
        self.registry.update_server(
            "local", {"process": process}, expected_token="launch-one",
            expected_revision=record["revision"])
        path = self.root / "servers/local/launch-one/process.log"
        path.write_text("before exit\n")
        logger_done = [False]
        appended = [False]
        def exited(item):
            if item["pid"] == "logger" and logger_done[0] and not appended[0]:
                with path.open("a") as stream:
                    stream.write("after native exit\n")
                appended[0] = True
            return item["pid"] == "native" or logger_done[0]
        source = hem_tui_logs.LocalLogProvider(
            self.root, timeout=1).open_source(observation())
        with patch.object(hem_tui_logs, "_confirmed_exit", side_effect=exited):
            first = await source.poll()
            self.assertFalse(first.ended)
            logger_done[0] = True
            final = await source.poll()
        self.assertTrue(final.ended)
        self.assertIn("after native exit", final.entries[0].text)
        self.assertEqual(final.entries[-1].kind, "end")

    async def test_server_drains_multiple_bounded_batches_before_one_end(self):
        record = self.running_server()
        process = {"pid": "native", "logger": {"pid": "logger"}}
        self.registry.update_server(
            "local", {"process": process}, expected_token="launch-one",
            expected_revision=record["revision"])
        path = self.root / "servers/local/launch-one/process.log"
        path.write_bytes(b"seed")
        source = hem_tui_logs.LocalLogProvider(
            self.root, timeout=1, batch_bytes=8).open_source(observation())
        with patch.object(hem_tui_logs, "_confirmed_exit", return_value=False):
            await source.poll()

        appended = b"abcdefgh" + b"ijklmnop" + b"qrstuvwx" + b"yz"
        with path.open("ab") as stream:
            stream.write(appended)
        batches = []
        with patch.object(hem_tui_logs, "_confirmed_exit", return_value=True):
            for _ in range(10):
                batches.append(await source.poll())
                if batches[-1].ended:
                    break

        data = b"".join(
            entry.text.encode() for batch in batches for entry in batch.entries
            if entry.kind == "data")
        self.assertEqual(data, appended)
        self.assertTrue(all(
            sum(len(entry.text.encode()) for entry in batch.entries
                if entry.kind == "data") <= 8
            for batch in batches))
        self.assertTrue(batches[-1].ended)
        self.assertTrue(all(not batch.ended for batch in batches[:-1]))
        self.assertEqual(
            sum(entry.kind == "end" for batch in batches for entry in batch.entries), 1)
        self.assertEqual((await source.poll()).entries, ())

    async def test_server_marks_gap_if_file_disappears_before_exit_drain(self):
        self.running_server()
        path = self.root / "servers/local/launch-one/process.log"
        path.write_bytes(b"observed")
        source = hem_tui_logs.LocalLogProvider(
            self.root, timeout=1, batch_bytes=4).open_source(observation())
        with patch.object(hem_tui_logs, "_confirmed_exit", return_value=False):
            await source.poll()
        path.unlink()

        with patch.object(hem_tui_logs, "_confirmed_exit", return_value=True):
            final = await source.poll()
        self.assertTrue(final.ended)
        self.assertEqual(
            [entry.kind for entry in final.entries], ["gap", "unavailable", "end"])

    async def test_server_keeps_bounded_bytes_when_launch_vanishes_mid_poll(self):
        record = self.running_server()
        path = self.root / "servers/local/launch-one/process.log"
        path.write_bytes(b"seed")
        source = hem_tui_logs.LocalLogProvider(
            self.root, timeout=1, batch_bytes=4).open_source(observation())
        with patch.object(hem_tui_logs, "_confirmed_exit", return_value=False):
            await source.poll()
        with path.open("ab") as stream:
            stream.write(b"abcdefgh")

        read_records = hem_tui_logs._records
        snapshots = [0]
        def clear_after_first_snapshot(*args):
            result = read_records(*args)
            snapshots[0] += 1
            if snapshots[0] == 1:
                self.registry.clear_server(
                    "local", expected_token="launch-one",
                    expected_revision=record["revision"])
            return result
        with patch.object(hem_tui_logs, "_records",
                          side_effect=clear_after_first_snapshot):
            final = await source.poll()

        self.assertEqual(snapshots[0], 2)
        self.assertTrue(final.ended)
        self.assertEqual(
            [(entry.kind, entry.text) for entry in final.entries],
            [("data", "abcd"),
             ("gap", "server launch ended before retained log was drained"),
             ("end", "server launch ended")])
        self.assertEqual((await source.poll()).entries, ())

    async def test_tracked_source_serializes_polls_and_waits_for_blocking_thread(self):
        selection = hem_tui_logs.LogSelection.from_observation(observation())
        source = hem_tui_logs.ServerLogSource(
            self.root, selection, timeout=.5, batch_bytes=32)
        started = threading.Event()
        release = threading.Event()
        counts = {"active": 0, "maximum": 0}
        lock = threading.Lock()
        def blocking_poll():
            with lock:
                counts["active"] += 1
                counts["maximum"] = max(counts["maximum"], counts["active"])
            started.set()
            try:
                release.wait(1)
                return hem_tui_logs.LogBatch()
            finally:
                with lock:
                    counts["active"] -= 1
        with patch.object(source, "_poll", side_effect=blocking_poll):
            first = asyncio.create_task(source.poll())
            second = asyncio.create_task(source.poll())
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            source.close()
            closing = asyncio.create_task(source.wait_closed())
            await asyncio.sleep(.05)
            self.assertFalse(closing.done())
            self.assertEqual(counts["active"], 1)
            release.set()
            await first
            await second
            await closing
        self.assertEqual(counts, {"active": 0, "maximum": 1})

    async def test_native_cursor_gap_reset_loss_and_incarnation_boundary(self):
        records = ({"event": "stream_begin", "sequence": 0},)
        gap_records = ({"event": "gap", "from": 1, "to": 4},
                       {"event": "request_end", "sequence": 4})
        replies = [(200, native_batch(records, 1)),
                   (200, native_batch(gap_records, 5)),
                   (409, b"changed"),
                   (200, native_batch((), 0, identity="200:2")),
                   (404, b"gone")]
        with endpoint(replies) as (url, requests):
            self.running_server(url)
            self.registry.create_instance(
                "local", "launch-one", "chat", incarnation=17)
            source = hem_tui_logs.LocalLogProvider(
                self.root, timeout=2).open_source(observation("model"))
            with patch.object(hem_tui_logs, "PinnedProcess", return_value=FakePinned()):
                first = await source.poll()
                second = await source.poll()
                reset = await source.poll()
                restarted = await source.poll()
                lost = await source.poll()
        self.assertIn("stream_begin", first.entries[0].text)
        self.assertEqual(second.entries[0].kind, "gap")
        self.assertEqual(reset.entries[0].kind, "reset")
        self.assertEqual(restarted.entries[0].kind, "reset")
        self.assertEqual(lost.entries[0].kind, "unavailable")
        log_requests = [headers for path, headers in requests if path == "/logz"]
        self.assertNotIn("X-Loom-Cursor", log_requests[0])
        self.assertEqual(log_requests[1]["X-Loom-Cursor"], "100:1:1")
        self.assertNotIn("X-Loom-Cursor", log_requests[3])

    async def test_native_malformed_oversized_and_changed_without_409_do_not_advance(self):
        replies = [(200, native_batch((), 0)),
                   (200, b"not-json\n"),
                   (409, b"x" * (hem_tui_logs.MAX_NATIVE_LOG_BYTES + 1)),
                   (200, native_batch((), 0, identity="200:1"))]
        with endpoint(replies) as (url, _requests):
            self.running_server(url)
            self.registry.create_instance(
                "local", "launch-one", "chat", incarnation=17)
            source = hem_tui_logs.LocalLogProvider(
                self.root, timeout=2).open_source(observation("model"))
            with patch.object(hem_tui_logs, "PinnedProcess", return_value=FakePinned()):
                await source.poll()
                self.assertEqual((await source.poll()).entries[0].kind, "unavailable")
                self.assertEqual((await source.poll()).entries, ())
                changed = await source.poll()
        self.assertEqual(changed.entries[0].kind, "reset")

    async def test_native_http_400_marks_cursor_reset_without_same_poll_retry(self):
        replies = [(200, native_batch((), 0)), (400, b"bad cursor"),
                   (200, native_batch((), 0, identity="200:1"))]
        with endpoint(replies) as (url, requests):
            self.running_server(url)
            self.registry.create_instance(
                "local", "launch-one", "chat", incarnation=17)
            source = hem_tui_logs.LocalLogProvider(
                self.root, timeout=2).open_source(observation("model"))
            with patch.object(hem_tui_logs, "PinnedProcess", return_value=FakePinned()):
                await source.poll()
                reset = await source.poll()
                self.assertEqual(len([path for path, _ in requests if path == "/logz"]), 2)
                restarted = await source.poll()
        self.assertEqual(reset.entries[0].kind, "reset")
        self.assertEqual(restarted.entries[0].kind, "reset")

    async def test_native_rejection_with_uncertain_ownership_retains_cursor(self):
        class UncertainAfterLogs(FakePinned):
            checks = 0

            def verify(self, _deadline=None):
                self.checks += 1
                if self.checks == 2:
                    raise UncertainOutcome("response ownership is unknown")

        with endpoint([(200, native_batch((), 0)), (409, b"cursor changed")]) as (url, requests):
            self.running_server(url)
            self.registry.create_instance(
                "local", "launch-one", "chat", incarnation=17)
            source = hem_tui_logs.LocalLogProvider(
                self.root, timeout=2).open_source(observation("model"))
            with patch.object(hem_tui_logs, "PinnedProcess", return_value=FakePinned()):
                await source.poll()
            cursor = source._cursor
            pinned = UncertainAfterLogs()
            with patch.object(hem_tui_logs, "PinnedProcess", return_value=pinned):
                unavailable = await source.poll()
            self.assertEqual(unavailable.entries[0].kind, "unavailable")
            self.assertEqual(source._cursor, cursor)
            self.assertFalse(source._reset_pending)
            self.assertEqual(pinned.checks, 2)
        self.assertEqual(len([path for path, _ in requests if path == "/logz"]), 2)

    async def test_native_cursor_policy_uses_metadata_instead_of_diagnostic_prose(self):
        replies = [(200, native_batch((), 0))]
        with endpoint(replies) as (url, requests):
            self.running_server(url)
            self.registry.create_instance(
                "local", "launch-one", "chat", incarnation=17)
            source = hem_tui_logs.LocalLogProvider(
                self.root, timeout=2).open_source(observation("model"))
            with patch.object(hem_tui_logs, "PinnedProcess", return_value=FakePinned()):
                await source.poll()
                with patch.object(hem_tui_logs.NativeClient, "logs", side_effect=HemError(
                        "cursor no longer usable", status=409, reason="http_status")):
                    reset = await source.poll()
                self.assertEqual(reset.entries[0].kind, "reset")
                source._cursor = "100:1:0"
                with patch.object(hem_tui_logs.NativeClient, "logs", side_effect=HemError(
                        "native HTTP 409: misleading diagnostic", reason="connection_failed")):
                    unavailable = await source.poll()
                self.assertEqual(unavailable.entries[0].kind, "unavailable")
                self.assertEqual(source._cursor, "100:1:0")
        self.assertEqual(len([path for path, _ in requests if path == "/logz"]), 1)

    async def test_native_fresh_400_and_409_are_unavailable_not_cursor_resets(self):
        for status in (400, 409):
            with (self.subTest(status=status), tempfile.TemporaryDirectory() as temporary,
                  endpoint([(status, b"fresh rejected")]) as (url, requests)):
                root = Path(temporary) / "state"
                registry = Registry(root)
                registry.create_server("local", saved_settings={})
                record = registry.reserve_launch(
                    "local", launch_token="launch-one", current_settings={}, endpoint=url)
                directory = root / "servers/local/launch-one"
                directory.mkdir(parents=True)
                registry.update_server(
                    "local", {"build": {"directory": str(directory), "executable": {}},
                              "process": {"fixture": True}, "phase": "running",
                              "pending": None}, expected_token="launch-one",
                    expected_revision=record["revision"])
                registry.create_instance(
                    "local", "launch-one", "chat", incarnation=17)
                source = hem_tui_logs.LocalLogProvider(
                    root, timeout=2).open_source(observation("model"))
                with patch.object(hem_tui_logs, "PinnedProcess", return_value=FakePinned()):
                    result = await source.poll()
                self.assertEqual(result.entries[0].kind, "unavailable")
                headers = next(headers for path, headers in requests if path == "/logz")
                self.assertNotIn("X-Loom-Cursor", headers)

    def test_parser_rejects_unmarked_discontinuity_and_bad_footer(self):
        selection = hem_tui_logs.LogSelection.from_observation(observation("model"))
        with self.assertRaisesRegex(HemError, "unmarked"):
            hem_tui_logs._parse_native_batch(
                native_batch(({"event": "request", "sequence": 3},), 4),
                selection, None)
        malformed = native_batch((), 0).replace(
            b'"value": "100:1:0"', b'"value": "100:1:9"')
        with self.assertRaises(HemError):
            hem_tui_logs._parse_native_batch(malformed, selection, None)


if __name__ == "__main__":
    unittest.main()
