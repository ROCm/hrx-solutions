"""Fixed current wheel producers; independent acceptance must not iterate these."""
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from binding.python.models import recipes, pipelines
from . import loom_catalog, moonshine_catalog, moonshine_preparer


# One ordered choice binds each build target, output interpretation and package
# stage. No discovery, environment selection or runtime registration is involved.
PRODUCERS = (
    (loom_catalog.CATALOG_TARGET,
     lambda query, repository: loom_catalog.query_bytecode_outputs(query, repository),
     loom_catalog.stage_package, None),
    (moonshine_catalog.TARGET,
     lambda query, repository: moonshine_catalog.query_outputs(query),
     moonshine_catalog.stage_package, moonshine_preparer.stage),
)


def build_targets():
    targets = (loom_catalog.DECODER_TARGET, *(target for target, _, _, _ in PRODUCERS))
    if len(targets) != len(set(targets)):
        raise ValueError('Duplicate packaging producer target')
    return targets


def recipe_bundle(repository):
    """Bind authored selections to code-owned installed assets before staging.

    The finite producers still admit compiler inputs and trusted helper bytes.
    Recipe text can select a known pipeline, never a target, tool or callback.
    """
    selected, identifiers, declared = set(), set(), set()
    root = repository / 'loom-pipeline'
    for path in sorted(root.glob('**/recipes/*.json')):
        path.resolve(strict=True).relative_to(root.resolve())
        recipe = recipes.parse(path.read_bytes())
        if recipe.id in identifiers or path.name != recipe.id + '.json':
            raise ValueError('Duplicate or misplaced authored recipe: ' + str(path))
        identifiers.add(recipe.id)
        binding = pipelines.get(recipe.pipeline)
        descriptor = binding.describe()
        if descriptor.modality != recipe.modality:
            raise ValueError('Recipe/pipeline modality mismatch: ' + recipe.id)
        binding.validate_artifacts(recipe.artifacts)
        catalog = PurePosixPath(descriptor.bundle.catalog_relative_path)
        if (str(catalog) != descriptor.bundle.catalog_relative_path or catalog.is_absolute()
                or not catalog.parts or '..' in catalog.parts or '\\' in str(catalog)):
            raise ValueError('Unsafe bundle catalog path')
        local = set()
        for entry in descriptor.bundle.files:
            logical = PurePosixPath(entry.path)
            if (str(logical) != entry.path or logical.is_absolute() or '..' in logical.parts
                    or not logical.parts or '\\' in entry.path or '\0' in entry.path
                    or logical.parts[0] not in ('loom-pipeline', '_native')
                    or entry.path in local):
                raise ValueError('Unsafe or duplicate pipeline bundle file: ' + entry.path)
            local.add(entry.path)
        authored = 'loom-pipeline/' + path.relative_to(root).as_posix()
        if authored not in local:
            raise ValueError('Pipeline bundle omits selected authored recipe: ' + authored)
        if not path.relative_to(root).is_relative_to(Path(catalog)):
            raise ValueError('Authored recipe is outside selected pipeline catalog')
        selected.update(local)
        declared.add(authored)
    if not identifiers:
        raise ValueError('No authored recipes selected for packaging')
    if {p for p in selected if '/recipes/' in p} != declared:
        raise ValueError('Pipeline bundle declares an unregistered recipe')
    return tuple(sorted(selected))


def prepare(query, repository):
    """Resolve complete family products before clearing the existing build tree."""
    build_targets()  # Refuse duplicate declarations before queries or clearing.
    bundle = recipe_bundle(repository)
    products = tuple((stage, read(query, repository), helper)
                     for _, read, stage, helper in PRODUCERS)
    return loom_catalog.query_decoder(query), products, bundle


def stage(package_root, prepared, repository):
    decoder, products, bundle = prepared
    # Producers materialize only admitted compiler products. The descriptor union
    # chooses which of those assets enter the wheel; no source-tree copy/prune.
    selected = {p for p in bundle if p.startswith('loom-pipeline/')}
    selected.add('loom-pipeline/llm/catalog-build.json')  # compiler provenance
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        produced = {}
        for produce, outputs, _ in products:
            for path in produce(staging, outputs, repository, decoder=decoder):
                name = path.relative_to(staging).as_posix()
                if name in produced:
                    raise ValueError('Duplicate compiled/catalog product: ' + name)
                produced[name] = path
        missing = selected - produced.keys()
        if missing:
            raise ValueError('Pipeline bundle missing compiled/catalog products: ' + repr(sorted(missing)))
        result = []
        for name in sorted(selected):
            destination = package_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(produced[name], destination)
            result.append(destination)
        return result


def stage_helpers(native_root, prepared, hrx_source):
    """Stage selected family-owned trusted payloads after generic native clearing."""
    _, products, _ = prepared
    return [path for _, _, helper in products if helper is not None
            for path in helper(native_root, hrx_source)]


def validate_bundle(package_root, prepared):
    """Require every declared asset after both catalog and helper staging."""
    _, _, bundle = prepared
    for name in bundle:
        path = package_root / name
        path.resolve(strict=True).relative_to(package_root.resolve())
        if not path.is_file() or path.is_symlink():
            raise ValueError('Pipeline bundle missing regular installed asset: ' + name)
