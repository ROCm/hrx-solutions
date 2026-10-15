"""Thin four-resource entry point for Loom Hem."""

from __future__ import annotations

import argparse
import json
import sys

from . import hem_arguments, hem_instance_cli, hem_model_cli, hem_pipeline_cli, hem_server_cli
from .hem_common import Deadline, HemError, InvalidInput, result


DOMAINS = {"model": hem_model_cli, "pipeline": hem_pipeline_cli,
           "server": hem_server_cli, "instance": hem_instance_cli}


class Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **dict(kwargs, allow_abbrev=False))

    def error(self, message):
        raise InvalidInput(message)


def parser():
    root = Parser(prog="loom-hem", description="Thin setup and control for Loom Serve")
    hem_arguments.output(root)
    types = root.add_subparsers(dest="resource", required=True)
    for name, module in DOMAINS.items():
        domain = types.add_parser(name)
        hem_arguments.output(domain)
        module.configure(domain)
    return root


def parse_args(argv=None):
    args = parser().parse_args(argv)
    if args.resource == "server":
        hem_server_cli.validate(args)
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    json_output = "--json" in argv
    args = None
    try:
        args = parse_args(argv)
        json_output = getattr(args, "json", False)
        # Downloads and follow streams intentionally have no lifecycle budget.
        deadline = Deadline.after(args.timeout) if hasattr(args, "timeout") else None
        response = DOMAINS[args.resource].run(args, deadline)
    except HemError as error:
        print(str(error), file=sys.stderr)
        if error.accepted is not None or error.exit_code == 4:
            acceptance = 'unknown' if error.accepted is None else str(error.accepted).lower()
            print(f"accepted: {acceptance}", file=sys.stderr)
        if json_output:
            identifiers = {}
            if args is not None:
                for key in ("model", "server", "instance", "pipeline"):
                    value = getattr(args, key, None)
                    if value is not None:
                        identifiers[key] = value
                if args.resource == "server" and getattr(args, "name", None):
                    identifiers["server"] = args.name
                if args.resource == "model" and getattr(args, "id", None):
                    identifiers["model"] = args.id
                if args.resource == "instance" and args.command in ("start", "export"):
                    identifiers["instance"] = args.name or args.model
            print(json.dumps(result("uncertain" if error.exit_code == 4 else "failed",
                                    completed=False, error=str(error),
                                    accepted=error.accepted, **identifiers)))
        return error.exit_code
    if json_output:
        print(json.dumps(response))
    else:
        print(json.dumps(response, indent=2))
    return getattr(response, 'exit_code', 0)


if __name__ == "__main__":
    raise SystemExit(main())
