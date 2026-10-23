"""Thin public adapters for local model acquisition and inventory."""

from pathlib import Path

from . import hem_arguments as arguments
from . import hem_display
from .models import curated_catalog, hem_models
from .hem_common import InvalidInput, result
from .hem_state import Registry


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
    """Pure grammar checks shared by parser and direct CLI dispatch."""
    if args.command == 'import':
        if args.assets_dir is not None:
            if args.tokenizer is not None or args.pipeline not in curated_catalog.CATALOGS:
                raise InvalidInput('--assets-dir requires a curated --pipeline and cannot combine with --tokenizer')
        elif args.pipeline in curated_catalog.CATALOGS:
            raise InvalidInput('Curated pipelines require --assets-dir')
        elif args.tokenizer is None:
            raise InvalidInput('--gguf requires --tokenizer')


def run(args, deadline):
    validate(args)
    if args.command == "available":
        return result("available", completed=True, items=hem_models.available())
    # Preview must not initialize even an empty model registry.
    if args.command == "download" and args.dry_run:
        preview = hem_models.download(None, args.recipe, model_id=args.id,
                                      pipeline=args.pipeline, offline=args.offline, dry_run=True)
        for command in preview['commands']:
            hem_display.emit(command)
        hem_display.emit('Preview only: no acquisition or local validation performed')
        return preview
    if args.command == 'import' and args.assets_dir is not None:
        curated_catalog.load_catalog(args.pipeline)  # Reject bad data before reserving state.
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
                hem_display.emit('Local model acquisition and packaged pipeline validation (no native equivalent)')
                hem_models.import_model(registry, args.model, ggufs=args.gguf,
                                        tokenizer=args.tokenizer, alias=args.alias,
                                        pipeline=args.pipeline)
        elif args.command == "download":
            hem_display.emit('Local packaged pipeline validation for Qwen; curated acquisition/presence checks only (no native equivalent)')
            hem_models.download(registry, args.recipe, model_id=args.id,
                                pipeline=args.pipeline, offline=args.offline)
        elif args.command == "validate":
            hem_display.emit('Local packaged pipeline validation for Qwen; curated acquisition/presence checks only (no native equivalent)')
            hem_models.validate(registry, args.model, pipeline=args.pipeline)
        record = hem_models.show(registry, args.id if args.command == 'download' else args.model)
    return dict(record, completed=True)
