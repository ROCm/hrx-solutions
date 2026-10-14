"""Captured HTTP response acceptance checks."""

import io
import json
import unittest

from smoke.assert_response import parse_sse


def sse(text="Hello!", finish="stop", done=True):
    events = [
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return io.StringIO(body + ("data: [DONE]\n\n" if done else ""))


class ResponseTest(unittest.TestCase):
    def test_completed_exact_responses(self):
        self.assertEqual(parse_sse(sse("\nHello! ")), "Hello!")

    def test_sse_rejects_wrong_empty_gibberish_and_truncated(self):
        for stream in [
            sse(""),
            sse("Hello! extra"),
            sse("asdf"),
            sse(finish="length"),
            sse(done=False),
            sse(finish=None),
            io.StringIO('data: {"error":"broken"}\n\n'),
            io.StringIO("data: [DONE]"),
        ]:
            with self.subTest(stream=stream.getvalue()), self.assertRaises(ValueError):
                parse_sse(stream)


if __name__ == "__main__":
    unittest.main()
