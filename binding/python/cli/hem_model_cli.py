"""Thin public adapters for local model acquisition and inventory."""

import os
from pathlib import Path
from dataclasses import asdict

from . import hem_arguments as arguments
from .. import hem_display
from ..models import pipelines, recipes, hem_models, model_download
from ..models.recipe_types import HFSource
from ..hem_common import InvalidInput, result
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
            child.add_argument("--pipeline", type=arguments.name)
        if command == "import":
            inputs = child.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--gguf", nargs="+", type=Path)
            inputs.add_argument("--assets-dir", type=Path)
            child.add_argument("--tokenizer", type=Path)
            child.add_argument("--alias", action="store_true")


def validate(args):
    """Keep existing import grammar; concrete bindings validate actual artifacts."""
    if args.command == 'import':
        if args.assets_dir is not None:
            if args.tokenizer is not None or args.pipeline is None:
                raise InvalidInput('--assets-dir requires --pipeline and cannot combine with --tokenizer')
        elif args.tokenizer is None:
            raise InvalidInput('--gguf requires --tokenizer')


def run(args, deadline):
    validate(args)
    if args.command == 'import' and args.pipeline is not None:
        descriptor = pipelines.get(args.pipeline).describe()
        if args.assets_dir is None and not any('gguf' in item.formats for item in descriptor.artifacts):
            raise InvalidInput('Selected pipeline requires --assets-dir')
    if args.command == 'download' and args.pipeline is not None:
        if recipes.load(args.recipe).pipeline != args.pipeline:
            raise InvalidInput('--pipeline differs from the selected recipe')
    if args.command == "available":
        return result("available", completed=True, items=hem_models.available())
    # Preview must not initialize even an empty model registry.
    if args.command == "download" and args.dry_run:
        preview = hem_models.download(None, args.recipe, model_id=args.id,
                                      offline=args.offline, dry_run=True)
        secrets = [os.environ.get(key, '') for key in
                   ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_HUB_TOKEN')]
        for artifact in preview.artifacts:
            if isinstance(artifact.source, HFSource):
                hem_display.emit(hem_display.command(model_download.command(artifact.source),
                                                     secrets=secrets))
        hem_display.emit('Preview only: no acquisition or local validation performed')
        document = asdict(preview)
        for artifact in document['artifacts']:
            if artifact['cache_path'] is not None:
                artifact['cache_path'] = str(artifact['cache_path'])
        return dict(document, completed=True, offline=args.offline)
    if args.command == 'import' and args.assets_dir is not None:
        pipelines.get(args.pipeline).describe()  # Reject bad data before reserving state.
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
                hem_models.import_assets(registry, args.model, directory=args.assets_dir,
                                         alias=args.alias, pipeline=args.pipeline)
            else:
                hem_display.emit('Local model acquisition and selected artifact checks; native loading remains unprobed')
                hem_models.import_model(registry, args.model, ggufs=args.gguf,
                                        tokenizer=args.tokenizer, alias=args.alias,
                                        pipeline=args.pipeline)
        elif args.command == "download":
            hem_display.emit('Selected recipe and prepared-asset checks; native loading remains unprobed')
            hem_models.download(registry, args.recipe, model_id=args.id,
                                offline=args.offline)
        elif args.command == "validate":
            hem_display.emit('Selected recipe and prepared-asset checks; native loading remains unprobed')
            hem_models.validate(registry, args.model, pipeline=args.pipeline)
        record = hem_models.show(registry, args.id if args.command == 'download' else args.model)
    return dict(record, completed=True)
