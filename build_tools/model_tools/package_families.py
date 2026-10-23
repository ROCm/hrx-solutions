"""Fixed current wheel producers; independent acceptance must not iterate these."""
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


def prepare(query, repository):
    """Resolve complete family products before clearing the existing build tree."""
    build_targets()  # Refuse duplicate declarations before queries or clearing.
    products = tuple((stage, read(query, repository), helper)
                     for _, read, stage, helper in PRODUCERS)
    return loom_catalog.query_decoder(query), products


def stage(package_root, prepared, repository):
    decoder, products = prepared
    return [path for produce, outputs, _ in products
            for path in produce(package_root, outputs, repository, decoder=decoder)]


def stage_helpers(native_root, prepared, hrx_source):
    """Stage selected family-owned trusted payloads after generic native clearing."""
    _, products = prepared
    return [path for _, _, helper in products if helper is not None
            for path in helper(native_root, hrx_source)]
