"""Selected trusted preparer identity shared by wheel build and runtime."""

REVISION = '61417137f334cfa828b38d6e9b0f2d7190503f16'
FILES = {
    'prepare_assets.py': 'e42af9c0bb50c2e75bd0178163bcb7e309f395ef2e94a43b50c8d33a1d6d7de9',
    'model.json': '56cc217f34c5e22a65050e47b87db2ab20064a1e0f125be84463ff7a4ad3306f',
}
RELATIVE = 'share/loom-serve/preparers/moonshine'
UPSTREAM = 'experimental/loom_serve/models/moonshine'

# Code-owned input/output contract for this exact fixed helper, not recipe knobs.
INPUT_REPOSITORY = 'moonshine-ai/moonshine-streaming-tiny'
INPUT_REVISION = 'f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a'
INPUT_FILES = ('model.safetensors', 'tokenizer.json', 'config.json')
GENERATED_FILES = (('rotary', 'safetensors', 'rotary.safetensors'),)
