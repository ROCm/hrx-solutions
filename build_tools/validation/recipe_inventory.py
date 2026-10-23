"""Independent release expectations for the three supported authored recipes.

These are acceptance oracles, never inputs to the packaging producers or runtime
selector. Keep coordinates literal so changing a producer cannot bless its own
missing, swapped or newly introduced declarations.
"""
import stat


EXPECTED_RECIPES = {
    'llm/qwen/qwen_08b/recipes/qwen35-08b-q5-k-xl.json': (
        'qwen35-08b-q5-k-xl', 'chat', 'qwen-08b', (
            ('weights', 'weights', 'gguf', 'model.gguf', 'unsloth/Qwen3.5-0.8B-GGUF',
             'b97f009053190ee1de01f886721d5e17d2bf4622', 'Qwen3.5-0.8B-UD-Q5_K_XL.gguf'),
            ('tokenizer', 'tokenizer', 'json', 'tokenizer.json', 'Qwen/Qwen3.5-0.8B',
             '2fc06364715b967f1860aea9cf38778875588b17', 'tokenizer.json'),
        )),
    'llm/qwen/qwen_27b/recipes/qwen38-27b-q5-k-xl.json': (
        'qwen38-27b-q5-k-xl', 'chat', 'qwen', (
            ('weights', 'weights', 'gguf', 'model.gguf', 'unsloth/Qwen3.8-27B-GGUF',
             '408fcc1807ab264d8cd644a7c4c0f58fbd32eebc', 'Qwen3.8-27B-UD-Q5_K_XL.gguf'),
            ('tokenizer', 'tokenizer', 'json', 'tokenizer.json', 'Qwen/Qwen3.8-27B',
             '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0', 'tokenizer.json'),
        )),
    'recognition/moonshine/streaming_tiny/recipes/moonshine-streaming-tiny.json': (
        'moonshine-streaming-tiny', 'asr', 'moonshine-streaming-tiny', (
            ('weights', 'weights', 'safetensors', 'model.safetensors',
             'moonshine-ai/moonshine-streaming-tiny',
             'f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a', 'model.safetensors'),
            ('tokenizer', 'tokenizer', 'json', 'tokenizer.json',
             'moonshine-ai/moonshine-streaming-tiny',
             'f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a', 'tokenizer.json'),
            ('configuration', 'configuration', 'json', 'config.json',
             'moonshine-ai/moonshine-streaming-tiny',
             'f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a', 'config.json'),
            ('rotary', 'rotary', 'safetensors', 'rotary.safetensors', None, None, None),
        )),
}


def validate_values(read, paths, parse):
    """Consume actual bytes with the common parser and independent expectations."""
    paths = tuple(paths)
    if len(paths) != len(set(paths)) or set(paths) != set(EXPECTED_RECIPES):
        raise RuntimeError('Authored recipe inventory missing, duplicate or unregistered: ' + repr(paths))
    for path, expected in EXPECTED_RECIPES.items():
        recipe = parse(read(path))
        artifacts = tuple((a.id, a.role, a.format, a.path,
                           a.source.repo_id if a.source.kind == 'huggingface' else None,
                           a.source.revision if a.source.kind == 'huggingface' else None,
                           a.source.filename if a.source.kind == 'huggingface' else None)
                          for a in recipe.artifacts)
        if (recipe.id, recipe.modality, recipe.pipeline, artifacts) != expected:
            raise RuntimeError('Authored recipe identity/coordinates differ from release contract: ' + path)


def validate_wheel(archive, repository, parse):
    prefix = 'hrx_solutions/loom-pipeline/'
    members = [info for info in archive.infolist()
               if info.filename.startswith(prefix) and '/recipes/' in info.filename and not info.is_dir()]
    paths = [info.filename[len(prefix):] for info in members]
    validate_values(lambda p: archive.read(prefix + p), paths, parse)
    for info in members:
        path = info.filename[len(prefix):]
        if (stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG)
                or archive.read(info) != (repository / 'loom-pipeline' / path).read_bytes()):
            raise RuntimeError('Wheel recipe is not the exact regular authored file: ' + path)


def validate_sdist(archive, repository, parse):
    members = [m for m in archive.getmembers() if '/loom-pipeline/' in m.name
               and '/recipes/' in m.name and not m.isdir()]
    paths = [m.name.split('/loom-pipeline/', 1)[1] for m in members]
    if any(not m.isfile() for m in members):
        raise RuntimeError('Source recipe must be a regular authored file')
    by_path = dict(zip(paths, members))
    read = lambda p: archive.extractfile(by_path[p]).read()
    validate_values(read, paths, parse)
    for path in paths:
        if read(path) != (repository / 'loom-pipeline' / path).read_bytes():
            raise RuntimeError('Source recipe differs from authored file: ' + path)


def validate_installed(package, parse, source_root=None):
    root = package / 'loom-pipeline'
    paths = [p.relative_to(root).as_posix() for p in root.glob('**/recipes/*') if not p.is_dir()]
    validate_values(lambda p: (root / p).read_bytes(), paths, parse)
    for path in paths:
        target = root / path
        if target.is_symlink() or not target.is_file():
            raise RuntimeError('Installed recipe is not a regular file: ' + path)
        if source_root is not None and target.read_bytes() != (source_root / 'loom-pipeline' / path).read_bytes():
            raise RuntimeError('Installed recipe differs from authored file: ' + path)
