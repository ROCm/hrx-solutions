"""Bounded protocol fault fixtures for WebUI management, not model acceptance."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import copy
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch

from binding.python.hem_common import Deadline, DeadlineExpired, HemError, InvalidInput, UncertainOutcome
from binding.python.hem_openwebui_http import MAX_BYTES, QUALIFIED_VERSIONS, WebUIClient, canonical_target, validate_config

TOKEN = "admin.private.token"
FOREIGN = "foreign.private.key"


def config():
    return {"ENABLE_OPENAI_API": False, "OPENAI_API_BASE_URLS": ["https://foreign.invalid/v1", "http://127.0.0.1:18080/v1"],
            "OPENAI_API_KEYS": [FOREIGN, ""], "OPENAI_API_CONFIGS": {"0": {"custom": {"nested": [None, True, 1.25]}}, "1": {"auth_type": "none"}}}


def reply(handler, body, status=200, **headers):
    if not isinstance(body, bytes):
        body = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Length", str(len(body)))
    for key, value in headers.items():
        handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(body)


@contextmanager
def fixture(respond):
    records = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.dispatch()
        def do_POST(self):
            self.dispatch()
        def dispatch(self):
            records.append((self.command, self.path, dict(self.headers), self.rfile.read(int(self.headers.get("Content-Length", "0")))))
            try:
                respond(self, records)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.005), daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", records
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


class ValidationTest(unittest.TestCase):
    def test_target_canonicalization_without_dns_or_credentials(self):
        for value, expected in (("http://127.0.0.1:3000/", "http://127.0.0.1:3000"),
                                ("http://127.2.3.4:03000", "http://127.2.3.4:3000"),
                                ("http://[0:0:0:0:0:0:0:1]:3000", "http://[::1]:3000")):
            self.assertEqual(canonical_target(value), expected)
        for value in (None, "", "http://localhost:3000", "http://127.0.0.1", "http://127.0.0.1:0",
                      "http://127.0.0.1:65536", "http://127.0.0.1:bad", "https://127.0.0.1:3000",
                      "http://user:secret@127.0.0.1:3000", "http://192.168.0.1:3000", "http://[::]:3000",
                      "http://127.0.0.1:3000/base", "http://127.0.0.1:3000?", "http://127.0.0.1:3000#",
                      "http://127.0.0.1:3000\n", "http://[::1%lo]:3000"):
            with self.subTest(value=value), self.assertRaises(InvalidInput) as caught:
                canonical_target(value)
            self.assertNotIn("secret", str(caught.exception))

    def test_config_is_lossless_and_sparse(self):
        value = config()
        del value["OPENAI_API_CONFIGS"]["0"]
        before = copy.deepcopy(value)
        self.assertIs(validate_config(value), value)
        self.assertEqual(value, before)
        self.assertEqual(QUALIFIED_VERSIONS, ("0.12.3",))

    def test_invalid_config_shapes_and_nonjson_values(self):
        variants = [{}, [], dict(config(), extra="value")]
        for field, values in (("ENABLE_OPENAI_API", (1, None)), ("OPENAI_API_KEYS", ([], ["one", 2])),
                              ("OPENAI_API_BASE_URLS", ("string", [None, "url"])),
                              ("OPENAI_API_CONFIGS", ({"00": {}}, {"-1": {}}, {"2": {}}, {"1": []}, {"+1": {}}, {"1.0": {}}, {"0": {1: "key"}}, {"0": {"bad": float("nan")}}, {"0": {"bad": (1, 2)}}))):
            variants.extend(dict(config(), **{field: value}) for value in values)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                validate_config(value)
        oversized = config()
        oversized["OPENAI_API_KEYS"][0] = "x" * MAX_BYTES
        with self.assertRaises(InvalidInput):
            validate_config(oversized)

    def test_refused_credentials_never_echo_value(self):
        for token in ("", None, "bad secret", "bad\r\nX: secret", "sécret"):
            with self.subTest(token=token), self.assertRaises(InvalidInput) as caught:
                WebUIClient("http://127.0.0.1:3000", token, Deadline.after())
            self.assertNotIn("secret", str(caught.exception))
        self.assertNotIn(TOKEN, repr(WebUIClient("http://127.0.0.1:3000", TOKEN, Deadline.after())))


class TransportTest(unittest.TestCase):
    def assert_safe(self, error):
        for secret in (TOKEN, FOREIGN):
            self.assertNotIn(secret, str(error))
            self.assertNotIn(secret, repr(error))
        self.assertTrue(error.__suppress_context__ or error.__context__ is None)

    def test_exact_routes_auth_body_no_proxies_no_dns_or_display(self):
        version = {"version": "0.12.3", "deployment_id": ""}
        def respond(h, _):
            reply(h, version if h.path == "/api/version" else config())
        with fixture(respond) as (target, records), patch.dict("os.environ", {"HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1", "ALL_PROXY": "http://127.0.0.1:1"}), patch("socket.getaddrinfo", side_effect=AssertionError("unexpected DNS")), patch("builtins.print") as output:
            client = WebUIClient(target, TOKEN, Deadline.after())
            self.assertEqual(client.version(), version)
            self.assertEqual(client.config(), config())
            self.assertEqual(client.update(config()), config())
        output.assert_not_called()
        self.assertEqual([(r[0], r[1]) for r in records], [("GET", "/api/version"), ("GET", "/openai/config"), ("POST", "/openai/config/update")])
        for _, _, headers, _ in records:
            self.assertEqual(headers["Authorization"], "Bearer " + TOKEN)
            self.assertNotIn("Cookie", headers)
        self.assertEqual(json.loads(records[2][3]), config())
        self.assertEqual(records[2][2]["Content-Type"], "application/json")

    def test_version_shape_not_mutation_qualification(self):
        for value in ({"version": "99.0", "deployment_id": ""}, {"version": "0.12.3", "deployment_id": "identity"}):
            with fixture(lambda h, _: reply(h, value)) as (target, _):
                self.assertEqual(WebUIClient(target, TOKEN, Deadline.after()).version(), value)
        for value in ({}, {"version": "0.12.3"}, {"version": "", "deployment_id": ""}, {"version": "0.12.3", "deployment_id": None}):
            with fixture(lambda h, _: reply(h, value)) as (target, _), self.assertRaises(InvalidInput):
                WebUIClient(target, TOKEN, Deadline.after()).version()

    def test_http_errors_and_redirect_never_echo_body_or_follow(self):
        for status in (301, 307, 401, 403, 422, 500, 503):
            with self.subTest(status=status), fixture(lambda h, _: reply(h, (TOKEN + FOREIGN).encode(), status, Location="http://127.0.0.1:1")) as (target, records):
                for method in ("config", "update"):
                    with self.assertRaises(HemError) as caught:
                        client = WebUIClient(target, TOKEN, Deadline.after())
                        getattr(client, method)(*([config()] if method == "update" else []))
                    self.assert_safe(caught.exception)
                    uncertain = method == "update" and status >= 500
                    self.assertEqual(caught.exception.exit_code, 4 if uncertain else 1)
                    self.assertIs(caught.exception.accepted, None if method == "config" or uncertain else False)
                self.assertEqual(len(records), 2)

    def test_invalid_bounded_json_and_config_after_accepted_update(self):
        invalid = [b"not JSON", b'{"duplicate":1,"duplicate":2}', b'{"a":{"x":1,"x":2}}', b'{"bad":NaN}', b'{"bad":Infinity}', b'{"bad":1e999}', b'\xff', b"[]", b"{}", b"[" * 1200 + b"]" * 1200, b" " * (MAX_BYTES + 1)]
        for body in invalid:
            with self.subTest(body=body[:40]), fixture(lambda h, _: reply(h, body)) as (target, records):
                client = WebUIClient(target, TOKEN, Deadline.after())
                with self.assertRaises(InvalidInput) as observed:
                    client.config()
                self.assert_safe(observed.exception)
                with self.assertRaises(UncertainOutcome) as submitted:
                    client.update(config())
                self.assertTrue(submitted.exception.accepted)
                self.assert_safe(submitted.exception)
                self.assertEqual(len(records), 2)

    def test_request_validation_precedes_network(self):
        with fixture(lambda h, _: reply(h, config())) as (target, records):
            with self.assertRaises(InvalidInput):
                WebUIClient(target, TOKEN, Deadline.after()).update({})
            self.assertEqual(records, [])

    def test_preconnect_failure_and_expiry_are_known_not_submitted(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.close()
        with self.assertRaises(HemError) as caught:
            WebUIClient(f"http://127.0.0.1:{port}", TOKEN, Deadline.after()).update(config())
        self.assertEqual(caught.exception.exit_code, 1)
        self.assertIs(caught.exception.accepted, False)
        self.assert_safe(caught.exception)
        with fixture(lambda h, _: reply(h, config())) as (target, records):
            with self.assertRaises(DeadlineExpired) as expired:
                WebUIClient(target, TOKEN, Deadline(0)).update(config())
            self.assertIs(expired.exception.accepted, False)
            self.assertEqual(records, [])

    def test_lost_response_before_and_after_acknowledgment(self):
        def drop(h, _):
            h.connection.shutdown(socket.SHUT_RDWR)
        def truncated(h, _):
            h.send_response(200)
            h.send_header("Content-Length", "1000")
            h.end_headers()
            h.wfile.write(b"{}")
        for respond, accepted in ((drop, None), (truncated, True)):
            with fixture(respond) as (target, records), self.assertRaises(UncertainOutcome) as caught:
                WebUIClient(target, TOKEN, Deadline.after()).update(config())
            self.assertIs(caught.exception.accepted, accepted)
            self.assertEqual(len(records), 1)
            self.assert_safe(caught.exception)

    def test_total_deadline_interrupts_slow_headers_and_body(self):
        def slow_headers(h, _):
            for byte in b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n":
                h.connection.send(bytes([byte]))
                time.sleep(0.01)
        def slow_body(h, _):
            h.send_response(200)
            h.send_header("Content-Length", "1000")
            h.end_headers()
            for _ in range(30):
                h.wfile.write(b"x")
                h.wfile.flush()
                time.sleep(0.01)
        for respond in (slow_headers, slow_body):
            for update in (False, True):
                with self.subTest(respond=respond, update=update), fixture(respond) as (target, _):
                    start = time.monotonic()
                    with self.assertRaises(UncertainOutcome if update else DeadlineExpired) as caught:
                        client = WebUIClient(target, TOKEN, Deadline.after(0.06))
                        client.update(config()) if update else client.config()
                    self.assertLess(time.monotonic() - start, 0.25)
                    self.assert_safe(caught.exception)

    def test_one_shared_deadline_is_not_renewed_per_request(self):
        def delayed(h, _):
            time.sleep(0.04)
            reply(h, config())
        with fixture(delayed) as (target, records):
            client = WebUIClient(target, TOKEN, Deadline.after(0.065))
            client.config()
            with self.assertRaises(DeadlineExpired):
                client.config()
            self.assertEqual(len(records), 2)

    def test_audio_model_wire_routes_and_only_model_404_is_absence(self):
        from binding.python import hem_openwebui_stt as stt
        audio = dict(tts=dict.fromkeys(stt.TTS_STRINGS, ''),
                     stt=dict(dict.fromkeys(stt.STT_STRINGS, ''), MISTRAL_USE_CHAT_COMPLETIONS=False),
                     realtime={'OPENAI_API_KEY': FOREIGN})
        def respond(h, _):
            if h.command == 'GET' and h.path.startswith('/api/v1/models/model?id='):
                reply(h, {'secret': FOREIGN}, status=404)
            else:
                reply(h, audio if '/audio/' in h.path else True)
        with fixture(respond) as (target, records):
            client = WebUIClient(target, TOKEN, Deadline.after())
            self.assertEqual(client.audio_config(), audio)
            body = {'tts': audio['tts'], 'stt': audio['stt']}
            client.update_audio(body)
            self.assertIsNone(client.model('prefix.name'))
            client.create_model({'id': 'prefix.name'})
            client.delete_model('prefix.name')
            self.assertEqual([(r[0], r[1]) for r in records], [
                ('GET', '/api/v1/audio/config'), ('POST', '/api/v1/audio/config/update'),
                ('GET', '/api/v1/models/model?id=prefix.name'), ('POST', '/api/v1/models/create'),
                ('POST', '/api/v1/models/model/delete')])
            self.assertEqual(json.loads(records[1][3]), body)
            self.assertEqual(json.loads(records[-1][3]), {'id': 'prefix.name'})
            self.assertTrue(all(r[2]['Authorization'] == 'Bearer ' + TOKEN for r in records))
        for code in (401, 403, 500):
            with fixture(lambda h, _: reply(h, {'secret': FOREIGN}, status=code)) as (target, _):
                with self.assertRaises(HemError) as error: WebUIClient(target, TOKEN, Deadline.after()).model('target')
                self.assertNotIn(FOREIGN, str(error.exception))

    def test_oversize_without_content_length_is_bounded(self):
        def oversized(h, _):
            h.send_response(200)
            h.end_headers()
            h.wfile.write(b" " * (MAX_BYTES + 2))
        with fixture(oversized) as (target, _), self.assertRaises(InvalidInput):
            WebUIClient(target, TOKEN, Deadline.after()).config()


if __name__ == "__main__":
    unittest.main()
