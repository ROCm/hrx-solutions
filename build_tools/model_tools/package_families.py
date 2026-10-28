"""Uniform selected executable packaging from authored catalog/build data."""
from pathlib import PurePosixPath

from binding.python.models import recipes, pipelines
from . import loom_catalog as compiler


def build_targets():
    return (compiler.DECODER_TARGET, compiler.CATALOG_TARGET)


def prepare(query, repository, *, selected=None):
    """Admit selected declarations and resolve required products before clearing."""
    root = repository / 'loom-pipeline'
    builds = compiler.load_builds(root / 'catalog_sources.bzl')
    references = tuple(sorted(builds if selected is None else selected))
    if not references or len(references) != len(set(references)) or not set(references) <= builds.keys():
        raise ValueError('Missing, duplicate or unregistered selected executable')
    if selected is None:
        authored = {p.relative_to(root).as_posix() for p in root.glob('**/executable.json')}
        if authored != builds.keys():
            raise ValueError('Build membership differs from authored executable inventory')
    catalogs, identifiers, recipe_ids = {}, set(), set()
    for reference in references:
        source, identity = compiler.admit_source(repository, reference, builds[reference])
        admitted, parsed_identity = pipelines.parse(compiler.source_path(root, reference).read_bytes())
        if admitted != source or identity != parsed_identity or source['id'] in identifiers:
            raise ValueError('Changed or duplicate selected executable declaration')
        identifiers.add(source['id'])
        recipe_paths = []
        for path in sorted((root / PurePosixPath(reference).parent / 'recipes').glob('*.json')):
            path.resolve(strict=True).relative_to(root.resolve())
            document, recipe_identity = recipes.parse(path.read_bytes())
            if (document['executable'] != reference or path.name != document['id'] + '.json'
                    or document['id'] in recipe_ids):
                raise ValueError('Duplicate, misplaced or cross-catalog recipe')
            pipelines.validate_files(source, document['files'], document.get('preparation_inputs'))
            recipe_ids.add(document['id'])
            recipe_paths.append((path.relative_to(root).as_posix(), recipe_identity))
        catalogs[reference] = (source, identity, tuple(recipe_paths))
    outputs = compiler.query_outputs(query, repository, builds, references)
    return compiler.query_decoder(query), {ref:builds[ref] for ref in references}, catalogs, outputs


def stage(package_root, prepared, repository):
    decoder, builds, catalogs, products = prepared
    outputs = {}
    for reference, build in builds.items():
        source, identity, recipe_paths = catalogs[reference]
        compiler.normalize_catalog(package_root, repository, reference, build, source, identity,
                                   products, decoder, outputs)
        for name, expected in recipe_paths:
            content = compiler.source_path(repository / 'loom-pipeline', name).read_bytes()
            if compiler.digest(content) != expected:
                raise ValueError('Authored recipe changed during build')
            compiler.write_once(package_root, 'loom-pipeline/' + name, content, outputs)
    return list(outputs.values())


def stage_helpers(native_root, prepared, hrx_source, *, repository):
    _, builds, catalogs, _ = prepared
    outputs = {}
    for reference, build in builds.items():
        if build['helper'] is not None:
            source, _, _ = catalogs[reference]
            compiler.stage_trusted_payload(native_root.parent, repository, hrx_source,
                build['helper'], source['preparation']['operation'], outputs)
    return list(outputs.values())
