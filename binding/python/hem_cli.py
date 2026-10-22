"""Thin resource entry point for Loom Control."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import textwrap

from . import hem_arguments, hem_help, hem_instance_cli, hem_model_cli, hem_pi_cli, hem_opencode_cli, hem_omp_cli, hem_openwebui_cli, hem_pipeline_cli, hem_server_cli, hem_render
from .hem_common import Deadline, HemError, InvalidInput, result


DOMAINS = {"openwebui": hem_openwebui_cli, "omp": hem_omp_cli, "model": hem_model_cli, "pipeline": hem_pipeline_cli,
           "server": hem_server_cli, "instance": hem_instance_cli, "pi": hem_pi_cli, "opencode": hem_opencode_cli}


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _split_lines(self, text, width):
        return textwrap.wrap(text, width, break_long_words=False, break_on_hyphens=False)

    def _format_action(self, action):
        if isinstance(action, argparse._SubParsersAction):
            # Usage names the selector; the command list needs no duplicate
            # choice-enumeration heading above its descriptive entries.
            return ''.join(self._format_action(choice) for choice in action._get_subactions())
        return super()._format_action(action)


class Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **dict(kwargs, allow_abbrev=False,
                                     formatter_class=HelpFormatter))

    def error(self, message):
        error = InvalidInput(message)
        error.parser = self
        raise error

    def parse_args(self, args=None, namespace=None):
        parsed, remaining = self.parse_known_args(args, namespace)
        if remaining:
            getattr(parsed, '_help_parser', self).error(
                'unrecognized Open WebUI arguments; use --help (credentials use OPENWEBUI_TOKEN)'
                if parsed.resource == 'openwebui' else 'unrecognized arguments: ' + ' '.join(remaining))
        return parsed


def parser():
    root = Parser(prog="loom-control", description="Thin setup and control for Loom Serve")
    hem_arguments.output(root)
    types = root.add_subparsers(dest="resource", title="Resources", metavar="RESOURCE")
    tui = types.add_parser("tui", help=hem_help.RESOURCES["tui"],
                           description=hem_help.RESOURCES["tui"])
    tui.set_defaults(command="run", _help_parser=tui)
    domains = {}
    for name, module in DOMAINS.items():
        domain = types.add_parser(name, help=hem_help.RESOURCES[name])
        domains[name] = domain
        hem_arguments.output(domain)
        module.configure(domain)
    hem_help.describe(root, domains)
    return root


def parse_args(argv=None):
    args = parser().parse_args(argv)
    if args.resource in ("server", "model") and args.command is not None:
        try:
            DOMAINS[args.resource].validate(args)
        except InvalidInput as error:
            error.parser = args._help_parser
            raise
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    json_output = "--json" in argv
    args = None
    try:
        args = parse_args(argv)
        json_output = getattr(args, "json", False)
        if args.resource is None or getattr(args, "command", None) is None:
            args._help_parser.print_help()
            return 0
        # Downloads and follow streams intentionally have no lifecycle budget.
        deadline = Deadline.after(args.timeout) if hasattr(args, "timeout") else None
        if args.resource == "tui":
            from . import hem_tui
            return hem_tui.run(args, deadline)
        response = DOMAINS[args.resource].run(args, deadline)
    except HemError as error:
        message = hem_openwebui_cli.safe_error(error) if 'openwebui' in argv else str(error)
        print(message, file=sys.stderr)
        if isinstance(error, InvalidInput) and not json_output:
            context = getattr(error, "parser", None) or (args._help_parser if args else None)
            if context is not None:
                context.print_usage(sys.stderr)
                print(f"For details: {context.prog} --help", file=sys.stderr)
        if error.accepted is not None or error.exit_code == 4:
            acceptance = 'unknown' if error.accepted is None else str(error.accepted).lower()
            print(f"accepted: {acceptance}", file=sys.stderr)
        if not json_output and args is not None and (error.accepted is True or error.exit_code in (3, 4)):
            if args.resource == 'instance':
                identifier = getattr(args, 'instance', None) or getattr(args, 'name', None) or getattr(args, 'model', None)
                if identifier:
                    print('Inspect before retrying: ' + hem_render.target(args, 'status', identifier), file=sys.stderr)
            elif args.resource == 'server':
                identifier = getattr(args, 'server', None) or getattr(args, 'name', None)
                if identifier:
                    print('Inspect before retrying: ' + shlex.join(['loom-control', 'server', 'status', identifier]), file=sys.stderr)
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
                                    completed=False, error=message,
                                    accepted=error.accepted, **identifiers)))
        return error.exit_code
    if json_output:
        print(json.dumps(response))
    else:
        rendered = hem_render.render(args, response)
        if rendered:
            sys.stdout.write(rendered)
            if args.command != 'logs' and not rendered.endswith('\n'):
                sys.stdout.write('\n')
    return getattr(response, 'exit_code', 0)


if __name__ == "__main__":
    raise SystemExit(main())
