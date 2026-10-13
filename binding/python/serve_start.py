"""Convenient command for starting an installed Loom model."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from . import setup_cli


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Start a model from the store using its recorded pipeline by default.

    Remaining arguments are forwarded to ``loom-serve-setup serve`` so the
    setup command remains the single source of truth for model lookup and
    server options.
    """
    parser = argparse.ArgumentParser(
        prog="loom-serve-start",
        description="Start an installed model without specifying GGUF or tokenizer paths.",
    )
    parser.add_argument("model_id", help="ID used when the model was imported")
    parser.add_argument("--pipeline", help="override the model's recorded pipeline")
    parser.add_argument("--foreground", action="store_true", help="run Loom Serve in the foreground")
    parser.add_argument("--json", action="store_true", help="print the managed server record as JSON")
    args, server_options = parser.parse_known_args(argv)

    # setup_cli performs the manifest, model-file, and catalog checks before
    # replacing this process with the packaged server.
    command = ["serve", args.model_id]
    if args.pipeline:
        command.extend(("--pipeline", args.pipeline))
    if args.foreground:
        command.append("--foreground")
    if args.json:
        command.append("--json")
    return setup_cli.main([*command, *server_options])


if __name__ == "__main__":
    raise SystemExit(main())
