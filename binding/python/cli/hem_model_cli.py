"""Thin public adapters for local model acquisition and inventory."""

import os
from pathlib import Path

from . import hem_arguments as arguments
from .. import hem_display
from ..models import hem_models, model_download
from ..hem_common import Deadline, InvalidInput, result
from ..control.hem_state import Registry


def configure(parser):
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("available", "download", "import", "validate", "list", "show"):
        child = commands.add_parser(command)
        arguments.output(child)
        if command == "download":
            child.add_argument("recipe")
            child.add_argument("--id", required=True, type=arguments.name)
            child.add_argument("--offline", action="store_true")
            child.add_argument("--dry-run", action="store_true")
        elif command in ("import", "validate", "show"):
            child.add_argument("model", type=arguments.name)
        if command in ("download", "import", "validate"):
            arguments.selection_assertions(child)
        if command == "import":
            child.add_argument("--recipe", required=True)
            inputs = child.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--gguf", nargs="+", type=Path)
            inputs.add_argument("--assets-dir", type=Path)
            child.add_argument("--tokenizer", type=Path)
            child.add_argument("--alias", action="store_true")


def validate(args):
    """Preserve input modes; explicit recipe selects the concrete executable."""
    if args.command == 'import':
        if args.assets_dir is not None:
            if args.tokenizer is not None:
                raise InvalidInput('--assets-dir cannot combine with --tokenizer')
        elif args.tokenizer is None:
            raise InvalidInput('--gguf requires --tokenizer')


def run(args, deadline):
    validate(args)
    if args.command == "available":
        return result("available", completed=True, items=hem_models.available(deadline=deadline))
    # Preview must not initialize even an empty model registry.
    if args.command == "download" and args.dry_run:
        preview = hem_models.download(None, args.recipe, model_id=args.id,
                                      executable=args.executable, modality=args.modality, offline=args.offline, dry_run=True, deadline=deadline)
        secrets = [os.environ.get(key, '') for key in
                   ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_HUB_TOKEN')]
        for artifact in preview['artifacts']:
            if artifact['source']['kind'] == 'huggingface':
                hem_display.emit(hem_display.command(model_download.command(artifact['source']),
                                                     secrets=secrets))
        hem_display.emit('Preview only: no acquisition or local validation performed')
        return dict(preview, completed=True, offline=args.offline)
    # Explicit recipe documents and programs are admitted before Registry can
    # initialize state. Reuse this operation's actual admitted documents.
    selection = None
    if args.command in ('import', 'download'):
        selection = hem_models._select(args.recipe, executable=args.executable,
                                        modality=args.modality, deadline=deadline or Deadline.after())
    registry = Registry(deadline=deadline)
    if args.command == "list":
        return result("inventory", completed=True, items=hem_models.inventory(registry))
    if args.command == "show":
        record = hem_models.show(registry, args.model)
        record['instances'] = registry.references(args.model)
    else:
        if args.command == "import":
            if args.assets_dir is not None:
                hem_display.emit('Local acquisition/presence checks only; native loading remains unprobed')
                record = hem_models.import_assets(registry, args.model, directory=args.assets_dir,
                                         alias=args.alias, recipe_id=args.recipe, executable=args.executable, modality=args.modality, _selection=selection)
            else:
                hem_display.emit('Local model acquisition and selected artifact checks; native loading remains unprobed')
                record = hem_models.import_model(registry, args.model, ggufs=args.gguf,
                                        tokenizer=args.tokenizer, alias=args.alias, recipe_id=args.recipe,
                                        executable=args.executable, modality=args.modality, _selection=selection)
        elif args.command == "download":
            hem_display.emit('Selected recipe and prepared-asset checks; native loading remains unprobed')
            record = hem_models.download(registry, args.recipe, model_id=args.id,
                                executable=args.executable, modality=args.modality, offline=args.offline, deadline=deadline, _selection=selection)
        elif args.command == "validate":
            hem_display.emit('Selected recipe and prepared-asset checks; native loading remains unprobed')
            record = hem_models.validate(registry, args.model, executable=args.executable, modality=args.modality)
    return dict(record, completed=True)
