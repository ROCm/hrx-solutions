"""Typed contract doubles for configuration orchestration, without native claims."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json

from binding.python.models import hem_config, recipes, modalities
from binding.python.models.recipe_types import (
    ArtifactRequirement, BundleFile, BundleManifest, Configurable, Constraints,
    Fixed, PipelineDescriptor, PreparedArtifact, PreparedModel, ProtocolDescriptor,
    RecognitionPrograms, TextPrograms, Unsupported,
)
from binding.python.hem_common import InvalidInput

RECIPE_PATHS = {
    'qwen-08b': 'llm/qwen/qwen_08b/recipes/qwen35-08b-q5-k-xl.json',
    'qwen': 'llm/qwen/qwen_27b/recipes/qwen38-27b-q5-k-xl.json',
    'moonshine-streaming-tiny': 'recognition/moonshine/streaming_tiny/recipes/moonshine-streaming-tiny.json',
}
SOURCE = Path(__file__).resolve().parents[1] / 'loom-pipeline'


def recipe_for(pipeline):
    return recipes.parse((SOURCE / RECIPE_PATHS[pipeline]).read_bytes())


def descriptor_for(recipe):
    if recipe.modality == 'chat':
        parameters = {key: Configurable('integer', default, Constraints(1, maximum))
                      for key, default, maximum in [('rows', 4, 16), ('context_capacity', 8192, 262144),
                          ('pool_capacity', 32768, 4194304), ('prefill_capacity', 128, 262144)]}
        parameters['mtp'] = Fixed('boolean', False) if recipe.pipeline == 'qwen-08b' else Configurable('boolean', True)
        parameters['mtp_depth'] = (Unsupported('integer', 'not supported') if recipe.pipeline == 'qwen-08b' else
            Configurable('integer', 3, Constraints(1, 3), {'parameter': 'mtp', 'equals': True}))
        programs = TextPrograms('program.loom', 'program.loom', ('program.loom',))
        protocol = ProtocolDescriptor('chat', 'text', '/v1/chat/completions', 'json', ('text',), ('text',))
    else:
        parameters = {'pending_requests': Fixed('integer', 4), 'options': Fixed('object', {})}
        programs = RecognitionPrograms('program.loom', 'program.loom', ('program.loom',))
        protocol = ProtocolDescriptor('transcription', 'recognition', '/v1/audio/transcriptions', 'multipart', ('audio',), ('text',))
    relative = str(Path(RECIPE_PATHS[recipe.pipeline]).parent.parent)
    return PipelineDescriptor(recipe.pipeline, recipe.modality, parameters, protocol,
                              tuple(ArtifactRequirement(a.role, (a.format,), 1, 1) for a in recipe.artifacts),
                              None, programs, BundleManifest(relative, (BundleFile('program.loom', 'program'),)))


def prepared_for(root, recipe):
    directory = root / 'model'
    directory.mkdir(exist_ok=True)
    files = []
    for item in recipe.artifacts:
        path = directory / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic asset, not native readiness')
        files.append(PreparedArtifact(item.id, item.role, item.format, item.path, path,
                     'hf-reference' if item.source.kind == 'huggingface' else 'generated-owned'))
    return PreparedModel('local-fixture', recipe.id, recipe.identity, recipe.pipeline, directory, tuple(files))


class ConfigurationFixture:
    def configure(self, pipeline='qwen-08b'):
        self.recipe = recipe_for(pipeline)
        self.descriptor = descriptor_for(self.recipe)
        self.catalog_root = self.root / 'loom-pipeline'
        self.catalog = self.catalog_root / self.descriptor.bundle.catalog_relative_path
        self.catalog.mkdir(parents=True)
        (self.catalog / 'program.loom').write_text('synthetic program identity')
        self.model = prepared_for(self.root, self.recipe)
        self.binding = SimpleNamespace(describe=lambda: self.descriptor, validate_artifacts=lambda artifacts: None)
        def validate(settings, descriptor):
            if descriptor.modality == 'chat':
                if settings['prefill_capacity'] > settings['context_capacity']:
                    raise InvalidInput('prefill_capacity must not exceed context_capacity')
                if settings['prefill_capacity'] < settings.get('mtp_depth', 0) + 1:
                    raise InvalidInput('prefill_capacity must fit MTP')
        def definition(settings, prepared, programs):
            return dict(kind=self.descriptor.protocol.native_kind, prepare=programs.prepare,
                        control=programs.control, sources=['program.loom'], settings=dict(settings))
        self.modality = SimpleNamespace(validate=validate, service_definition=definition)
        for patcher in [patch.object(hem_config.pipelines, 'catalog_roots', return_value=(self.catalog_root,)),
                        patch.object(hem_config.pipelines, 'get', return_value=self.binding),
                        patch.object(modalities, 'get', return_value=self.modality)]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def config_file(self, contents):
        path = self.root / 'configuration.json'
        path.write_text(contents)
        return path
