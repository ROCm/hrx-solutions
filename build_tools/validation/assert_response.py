#!/usr/bin/env python3
"""Check that a captured chat stream completed with exactly Hello!."""
import argparse
import json
from pathlib import Path

EXPECTED = "Hello!"
def exact_response(text):
    if text.strip() != EXPECTED:
        raise ValueError(f"Expected {EXPECTED!r}; received {text!r}")
    return text.strip()


def parse_sse(lines):
    """Consume SSE frames, rejecting errors, truncation and missing completion."""
    content, finish, done, frame = [], None, False, []
    for line in lines:
        line = line.rstrip("\r\n")
        if line:
            if line.startswith("data:"):
                frame.append(line[5:].lstrip(" "))
            continue
        if not frame:
            continue
        data, frame = "\n".join(frame), []
        if done:
            raise ValueError("Data after terminal [DONE]")
        if data == "[DONE]":
            done = True
            continue
        event = json.loads(data)
        if "error" in event:
            raise ValueError(f"Server error: {event['error']}")
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("Unexpected extra completion choice")
            delta = choice.get("delta", {})
            if delta.get("tool_calls") or delta.get("reasoning_content"):
                raise ValueError("Unexpected tool or thinking output")
            if delta.get("content"):
                if finish is not None:
                    raise ValueError("Content after finished completion")
                content.append(delta["content"])
            if choice.get("finish_reason") is not None:
                if finish is not None:
                    raise ValueError("Duplicate finish reason")
                finish = choice["finish_reason"]
    if frame or not done or finish != "stop":
        raise ValueError(
            f"Incomplete/truncated stream: finish={finish!r}, done={done}, pending_frame={bool(frame)}"
        )
    return exact_response("".join(content))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    with args.input.open() as stream:
        response = parse_sse(stream)
    print(json.dumps({"response": response, "status": "passed"}))


if __name__ == "__main__":
    main()
