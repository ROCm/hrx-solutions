"""Real loopback HTTP protocol tests; no native payload or shared services."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from binding.python.hem_common import Deadline, DeadlineExpired, HemError, InvalidInput, UncertainOutcome
from binding.python.hem_http import NativeClient, active_ready, registration, retired


def inventory(*items):
    return {"object": "list", "data": list(items), "server": {"worker_count": 1}}


def model(incarnation=17, **overrides):
    loom = dict(incarnation=incarnation, registration="ready", parameter_operation="idle",
                parameters={"active": True}, status={"admission": "drained"})
    loom.update(overrides)
    return dict(id="chat", loom=loom)


@contextmanager
def fixture(respond=None):
    records = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch()

        def dispatch(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            records.append((self.command, self.path, body, dict(self.headers)))
            if respond is not None:
                try:
                    respond(self, records)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                reply(self, inventory(model()))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.005), daemon=True)
    worker.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port), records
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def reply(handler, payload, status=200):
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)


class TransportTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.body = Path(directory.name) / "body with spaces.json"
        self.body.write_bytes(b'{ "model": "chat", "mode": "drain" }\n')

    def test_exact_routes_bytes_headers_and_inspectable_curl(self):
        output = []
        with fixture() as (endpoint, requests):
            client = NativeClient(endpoint, Deadline.after(), emit=output.append)
            for path in ("/readyz", "/v1/models", "/v1/status"):
                client.get(path)
            for action in ("register", "activate", "undrain", "unregister"):
                client.post("/v1/models/" + action, self.body)
            client.logs(model="chat", cursor="1:2:0")
        self.assertEqual([item[1] for item in requests], ["/readyz", "/v1/models", "/v1/status",
                         "/v1/models/register", "/v1/models/activate", "/v1/models/undrain",
                         "/v1/models/unregister", "/logz"])
        for _, _, body, headers in requests[3:7]:
            self.assertEqual(body, self.body.read_bytes())
            self.assertEqual(headers["Content-Type"], "application/json")
        self.assertIn("@" + str(self.body), shlex.split(output[3]))
        self.assertEqual(requests[-1][3]["X-Loom-Model"], "chat")
        self.assertIn("X-Loom-Cursor: 1:2:0", shlex.split(output[-1]))

    def test_ownership_precedes_any_bytes_and_post_checks_closed_response(self):
        checks = []
        deadline = Deadline.after()
        with fixture() as (endpoint, requests):
            def before(sock, budget):
                self.assertIs(budget, deadline)
                self.assertEqual(sock.getpeername()[1], int(endpoint.rsplit(":", 1)[1]))
                self.assertEqual(requests, [])
                checks.append("connected before bytes")
            def after(budget):
                self.assertIs(budget, deadline)
                self.assertEqual(len(requests), 1)
                checks.append("process after response")
            client = NativeClient(endpoint, deadline, before, ownership_after=after, emit=lambda _: None)
            client.post("/v1/models/register", self.body)
        self.assertEqual(checks, ["connected before bytes", "process after response"])

    def test_ownership_refusal_transmits_nothing(self):
        with fixture() as (endpoint, requests):
            def refuse(sock, deadline):
                raise UncertainOutcome("unrelated listener")
            client = NativeClient(endpoint, ownership=refuse, ownership_after=lambda _: None, emit=lambda _: None)
            with self.assertRaises(UncertainOutcome):
                client.post("/v1/models/register", self.body)
            self.assertEqual(requests, [])

    def test_post_response_process_exit_is_uncertain(self):
        with fixture() as (endpoint, _):
            def exited(deadline):
                raise UncertainOutcome("pinned process exited during observation")
            client = NativeClient(endpoint, ownership=lambda *args: None, ownership_after=exited, emit=lambda _: None)
            with self.assertRaises(UncertainOutcome):
                client.status()

    def test_acknowledgment_survives_post_response_ownership_uncertainty(self):
        for status, accepted in ((202, True), (409, False)):
            with self.subTest(status=status), fixture(lambda h, _: reply(h, {}, status)) as (endpoint, requests):
                def unknown_identity(_):
                    raise UncertainOutcome("process identity cannot be rechecked")
                client = NativeClient(endpoint, Deadline.after(), ownership=lambda *_: None,
                                      ownership_after=unknown_identity, emit=lambda _: None)
                with self.assertRaises(UncertainOutcome) as caught:
                    client.post("/v1/models/register", self.body)
                self.assertEqual(caught.exception.exit_code, 4)
                self.assertIs(caught.exception.accepted, accepted)
                self.assertEqual(len(requests), 1)

    def test_ownership_budget_expiry_never_transmits(self):
        deadline = Deadline.after(0.04)
        with fixture() as (endpoint, requests):
            client = NativeClient(endpoint, deadline, lambda *args: time.sleep(0.06),
                                  ownership_after=lambda _: None, emit=lambda _: None)
            with self.assertRaises(DeadlineExpired):
                client.post("/v1/models/register", self.body)
            self.assertEqual(requests, [])

    def test_expired_budget_never_connects_or_displays(self):
        with fixture() as (endpoint, requests):
            output = []
            client = NativeClient(endpoint, Deadline(0), emit=output.append)
            with self.assertRaises(DeadlineExpired):
                client.post("/v1/models/register", self.body)
            self.assertEqual(requests, [])
            self.assertEqual(output, [])

    def test_lost_mutation_response_is_uncertain_without_retry(self):
        def drop(handler, records):
            handler.connection.shutdown(socket.SHUT_RDWR)
        with fixture(drop) as (endpoint, requests):
            client = NativeClient(endpoint, Deadline.after(), emit=lambda _: None)
            with self.assertRaises(UncertainOutcome) as caught:
                client.post("/v1/models/register", self.body)
            self.assertIsNone(caught.exception.accepted)
            self.assertEqual(len(requests), 1)

    def test_mutation_timeout_before_headers_uncertainty_beats_expiry(self):
        def stall(handler, records):
            time.sleep(0.12)
        with fixture(stall) as (endpoint, requests):
            client = NativeClient(endpoint, Deadline.after(0.04), emit=lambda _: None)
            with self.assertRaises(UncertainOutcome) as caught:
                client.post("/v1/models/undrain", self.body)
            self.assertEqual(caught.exception.exit_code, 4)
            self.assertEqual(len(requests), 1)

    def test_acknowledged_request_incomplete_at_deadline_is_known_accepted(self):
        def stall(handler, records):
            handler.send_response(202)
            handler.send_header("Content-Length", "999")
            handler.end_headers()
            handler.wfile.flush()
            time.sleep(0.12)
        with fixture(stall) as (endpoint, _):
            client = NativeClient(endpoint, Deadline.after(0.04), emit=lambda _: None)
            with self.assertRaises(DeadlineExpired) as caught:
                client.post("/v1/models/register", self.body)
            self.assertTrue(caught.exception.accepted)

    def test_known_rejection_includes_native_diagnostic(self):
        with fixture(lambda h, _: reply(h, {"error": "already exists"}, 409)) as (endpoint, requests):
            with self.assertRaises(HemError) as caught:
                NativeClient(endpoint, emit=lambda _: None).post("/v1/models/register", self.body)
            self.assertEqual(caught.exception.exit_code, 1)
            self.assertIs(caught.exception.accepted, False)
            self.assertIn("already exists", str(caught.exception))
            self.assertEqual(len(requests), 1)

    def test_invalid_json_does_not_erase_acknowledged_acceptance(self):
        def malformed(handler, records):
            handler.send_response(202)
            handler.end_headers()
            handler.wfile.write(b"invalid")
        with fixture(malformed) as (endpoint, _):
            with self.assertRaises(HemError) as caught:
                NativeClient(endpoint, emit=lambda _: None).post("/v1/models/register", self.body).json()
            self.assertIs(caught.exception.accepted, True)

    def test_error_diagnostics_redact_explicit_credentials(self):
        with fixture(lambda h, _: reply(h, {"error": "secret-token invalid"}, 400)) as (endpoint, _):
            with self.assertRaises(HemError) as caught:
                NativeClient(endpoint, secrets=("secret-token",), emit=lambda _: None).get("/v1/status")
            self.assertNotIn("secret-token", str(caught.exception))

    def test_redirect_never_replays_mutation(self):
        with fixture(lambda h, _: reply(h, {}, 307)) as (endpoint, requests):
            with self.assertRaises(HemError):
                NativeClient(endpoint, emit=lambda _: None).post("/v1/models/register", self.body)
            self.assertEqual(len(requests), 1)

    def test_trickled_body_cannot_extend_total_observation_budget(self):
        def trickle(handler, records):
            handler.send_response(200)
            handler.send_header("Content-Length", "1000")
            handler.end_headers()
            for _ in range(20):
                handler.wfile.write(b"x")
                handler.wfile.flush()
                time.sleep(0.01)
        with fixture(trickle) as (endpoint, _):
            start = time.monotonic()
            with self.assertRaises(DeadlineExpired):
                NativeClient(endpoint, Deadline.after(0.05), emit=lambda _: None).get("/v1/status")
            self.assertLess(time.monotonic() - start, 0.18)

    def test_status_response_limit_stops_oversized_and_streaming_bodies(self):
        def oversized(handler, _records):
            body = b"x" * 65
            handler.send_response(200)
            handler.send_header("Content-Length", str(len(body)))
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.wfile.write(body)

        with fixture(oversized) as (endpoint, _):
            with self.assertRaisesRegex(HemError, "exceeds 64 bytes"):
                NativeClient(endpoint, Deadline.after(), emit=lambda _: None).status(
                    max_response_bytes=64)

        closed = threading.Event()
        def streaming(handler, _records):
            handler.send_response(200)
            handler.send_header("Connection", "close")
            handler.end_headers()
            try:
                for _ in range(20):
                    handler.wfile.write(b"abcdefgh")
                    handler.wfile.flush()
                    time.sleep(0.005)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                closed.set()

        with fixture(streaming) as (endpoint, _):
            start = time.monotonic()
            with self.assertRaisesRegex(HemError, "exceeds 32 bytes"):
                NativeClient(endpoint, Deadline.after(1), emit=lambda _: None).status(
                    max_response_bytes=32)
            self.assertLess(time.monotonic() - start, 0.3)
        self.assertTrue(closed.wait(1))

    def test_status_without_explicit_limit_retains_existing_behavior(self):
        document = inventory(model())
        document["ignored-padding"] = "x" * (1024 * 1024 + 1)
        with fixture(lambda handler, _records: reply(handler, document)) as (endpoint, _):
            observed = NativeClient(
                endpoint, Deadline.after(2), emit=lambda _: None).status()
        self.assertEqual(observed["data"][0]["id"], "chat")
        self.assertGreater(len(observed["ignored-padding"]), 1024 * 1024)

    def test_response_limit_must_be_a_positive_integer(self):
        with fixture() as (endpoint, requests):
            for value in (0, -1, 1.5, True):
                with self.subTest(value=value), self.assertRaises(InvalidInput):
                    NativeClient(endpoint, emit=lambda _: None).status(
                        max_response_bytes=value)
            self.assertEqual(requests, [])

    def test_name_resolution_is_bounded_before_any_mutation(self):
        original = socket.getaddrinfo
        def delayed(*args, **kwargs):
            time.sleep(0.1)
            return original(*args, **kwargs)
        with fixture() as (endpoint, requests), patch("socket.getaddrinfo", side_effect=delayed):
            with self.assertRaises(DeadlineExpired):
                NativeClient(endpoint, Deadline.after(0.03), emit=lambda _: None).post("/v1/models/register", self.body)
            self.assertEqual(requests, [])

    def test_direct_requests_ignore_proxy_environment(self):
        with fixture() as (endpoint, requests), patch.dict("os.environ", {"http_proxy": "http://127.0.0.1:1", "HTTP_PROXY": "http://127.0.0.1:1"}):
            NativeClient(endpoint, emit=lambda _: None).status()
            self.assertEqual(len(requests), 1)

    def test_invalid_endpoints_headers_and_missing_body_reject_before_http(self):
        for endpoint in ("https://127.0.0.1", "http://user:secret@localhost", "http://localhost/prefix", "http://localhost:bad", "http://localhost/?token=secret"):
            with self.subTest(endpoint=endpoint), self.assertRaises(InvalidInput):
                NativeClient(endpoint)
        with self.assertRaises(InvalidInput):
            NativeClient("http://127.0.0.1", ownership=lambda *args: None)
        with fixture() as (endpoint, requests):
            client = NativeClient(endpoint, emit=lambda _: None)
            for path in ("//otherhost", "/v1/status?token=secret", "/v1/status\r\nX: y"):
                with self.assertRaises(InvalidInput):
                    client.get(path)
            with self.assertRaises(InvalidInput):
                client.get("/logz", headers={"Authorization": "Bearer secret"})
            with self.assertRaises(InvalidInput):
                client.logs(model="chat\r\nX: y")
            with self.assertRaises(InvalidInput):
                client.post("/v1/models/register", self.body.parent / "missing")
            self.assertEqual(requests, [])

    def test_explicit_secrets_redacted_in_display_but_not_changed_on_wire(self):
        output = []
        with fixture() as (endpoint, requests):
            NativeClient(endpoint, secrets=("private-model",), emit=output.append).logs(model="private-model")
            self.assertEqual(requests[0][3]["X-Loom-Model"], "private-model")
        self.assertNotIn("private-model", output[0])
        self.assertIn("redacted", output[0])


class ObservationTest(unittest.TestCase):
    def test_original_incarnation_retirement_and_discovery_not_sufficient(self):
        self.assertFalse(retired(inventory(model()), "chat", 17))
        self.assertTrue(retired(inventory(model(18)), "chat", 17))
        self.assertIsNone(registration(inventory(model(18)), "chat", 17))
        self.assertTrue(retired(inventory(), "chat", 17))
        with self.assertRaises(HemError):
            retired({"object": "list", "data": []}, "chat", 17)
        with self.assertRaises(InvalidInput):
            retired(inventory(), "chat", None)

    def test_activation_requires_all_native_conditions(self):
        self.assertTrue(active_ready(inventory(model()), "chat", 17))
        for override in (dict(registration="preparing"), dict(parameter_operation="activating"),
                         dict(parameters={"active": False}), dict(status={"admission": "open"})):
            self.assertFalse(active_ready(inventory(model(**override)), "chat", 17))
        with self.assertRaises(HemError):
            active_ready(inventory(model(error={"message": "compilation failed"})), "chat", 17)
        with self.assertRaises(HemError):
            active_ready(inventory(model(18)), "chat", 17)

    def test_invalid_or_duplicate_status_cannot_prove_retirement(self):
        for document in ({}, inventory(model(), model()), inventory(dict(id="chat", loom={})),
                         inventory(model(True)), inventory(model(parameters=None))):
            with self.subTest(document=document), self.assertRaises(HemError):
                retired(document, "chat", 17)

    def test_wait_uses_status_only_prints_once_and_preserves_shared_budget(self):
        def sequence(handler, requests):
            reply(handler, inventory(model()) if len(requests) < 3 else inventory(model(18)))
        output = []
        with fixture(sequence) as (endpoint, requests):
            deadline = Deadline.after()
            client = NativeClient(endpoint, deadline, emit=output.append)
            document = client.wait(lambda d: retired(d, "chat", 17), interval=0.001)
            self.assertIs(client.deadline, deadline)
            self.assertEqual(len(requests), 3)
            self.assertEqual(len(output), 1)
            self.assertEqual({r[1] for r in requests}, {"/v1/status"})
            self.assertTrue(retired(document, "chat", 17))

    def test_known_pending_timeout_vs_unresolved_mutation(self):
        for accepted, expected in ((True, DeadlineExpired), (None, UncertainOutcome)):
            with self.subTest(accepted=accepted):
                now = [0.0]
                deadline = Deadline.after(1, clock=lambda: now[0])
                client = NativeClient("http://127.0.0.1:1", deadline, emit=lambda _: None)
                def expire_during_sleep(seconds):
                    now[0] = deadline.expires_at
                with patch.object(client, "status", return_value=inventory(model())) as observe, \
                     patch("binding.python.hem_http.time.sleep", side_effect=expire_during_sleep):
                    with self.assertRaises(expected) as caught:
                        client.wait(lambda _: False, accepted=accepted)
                self.assertIs(caught.exception.accepted, accepted)
                observe.assert_called_once_with(display=True)

    def test_deadline_boundaries_preserve_known_or_unknown_acceptance(self):
        for boundary in ("before-poll", "during-poll", "connection-preparation", "after-pending-observation"):
            for accepted, expected in ((True, DeadlineExpired), (None, UncertainOutcome)):
                with self.subTest(boundary=boundary, accepted=accepted):
                    now = [0.0]
                    deadline = Deadline.after(1, clock=lambda: now[0])
                    client = NativeClient("http://127.0.0.1:1", deadline, emit=lambda _: None)
                    def expire(*args, **kwargs):
                        now[0] = deadline.expires_at
                        return deadline.remaining()
                    def pending(*args, **kwargs):
                        now[0] = deadline.expires_at
                        return inventory(model())
                    if boundary == "before-poll":
                        now[0] = deadline.expires_at
                        observation = patch.object(client, "status")
                    elif boundary == "during-poll":
                        observation = patch.object(client, "status", side_effect=expire)
                    elif boundary == "connection-preparation":
                        observation = patch("binding.python.hem_http._connect", side_effect=expire)
                    else:
                        observation = patch.object(client, "status", side_effect=pending)
                    with observation as observed, patch("binding.python.hem_http.time.sleep") as sleep:
                        with self.assertRaises(expected) as caught:
                            client.wait(lambda _: False, accepted=accepted)
                    self.assertIs(caught.exception.accepted, accepted)
                    self.assertEqual(caught.exception.exit_code, 3 if accepted else 4)
                    self.assertEqual(observed.call_count, 0 if boundary == "before-poll" else 1)
                    sleep.assert_not_called()

    def test_observation_failure_does_not_erase_uncertain_or_known_acceptance(self):
        with fixture(lambda h, _: reply(h, {}, 503)) as (endpoint, _):
            for accepted in (None, True):
                with self.assertRaises(UncertainOutcome) as caught:
                    NativeClient(endpoint, Deadline.after(), emit=lambda _: None).wait(lambda _: False, accepted=accepted)
                self.assertIs(caught.exception.accepted, accepted)

    def test_acknowledged_unregister_with_failed_observation_is_uncertain(self):
        def sequence(failure):
            def respond(handler, requests):
                if handler.command == "POST":
                    reply(handler, {}, 202)
                else:
                    failure(handler)
            return respond
        def drop(handler):
            handler.connection.shutdown(socket.SHUT_RDWR)
        def malformed_json(handler):
            handler.send_response(200)
            handler.end_headers()
            handler.wfile.write(b"not JSON")
        for failure in (lambda h: reply(h, {}, 503), lambda h: reply(h, {}), malformed_json, drop):
            with self.subTest(failure=failure), fixture(sequence(failure)) as (endpoint, requests):
                directory = tempfile.TemporaryDirectory()
                self.addCleanup(directory.cleanup)
                body = Path(directory.name) / "unregister.json"
                body.write_text('{"model":"chat","mode":"drain"}')
                client = NativeClient(endpoint, Deadline.after(), emit=lambda _: None)
                acknowledged = client.post("/v1/models/unregister", body)
                with self.assertRaises(UncertainOutcome) as caught:
                    client.wait(lambda d: retired(d, "chat", 17), accepted=acknowledged.accepted)
                self.assertEqual(caught.exception.exit_code, 4)
                self.assertIs(caught.exception.accepted, True)
                self.assertEqual([r[0] for r in requests], ["POST", "GET"])

    def test_wait_preserves_acceptance_on_post_observation_ownership_uncertainty(self):
        with fixture() as (endpoint, _):
            def unknown_identity(_):
                raise UncertainOutcome("process identity became unknown")
            client = NativeClient(endpoint, Deadline.after(), ownership=lambda *_: None,
                                  ownership_after=unknown_identity, emit=lambda _: None)
            with self.assertRaises(UncertainOutcome) as caught:
                client.wait(lambda _: False, accepted=True)
            self.assertIs(caught.exception.accepted, True)

    def test_positively_observed_native_preparation_failure_remains_operational_failure(self):
        with fixture(lambda h, _: reply(h, inventory(model(error={"message": "compilation failed"})))) as (endpoint, _):
            with self.assertRaises(HemError) as caught:
                NativeClient(endpoint, Deadline.after(), emit=lambda _: None).wait(
                    lambda d: active_ready(d, "chat", 17), accepted=True)
            self.assertEqual(caught.exception.exit_code, 1)
            self.assertIs(caught.exception.accepted, True)


if __name__ == "__main__":
    unittest.main()
