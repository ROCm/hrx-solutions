"""Actual declarations/policy and plain inspected records; no native execution."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from unittest.mock import patch

from binding.python.models import pipelines, recipes, modalities
from tests.gguf_fixtures import write_gguf

SOURCE = next(root for root in pipelines.catalog_roots() if root.is_dir())


def recipe_for(identifier):
    return recipes.load(identifier)[0]


def record_for(root, recipe, identity):
    executable, _catalog, executable_identity = pipelines.load(recipe['executable'])
    directory = Path(root) / 'model'
    directory.mkdir()
    paths, assets, evidence = {}, [], []
    for purpose, item in recipe['files'].items():
        path = directory / item['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        format = executable['assets'][purpose]['format']
        if format == 'gguf':
            write_gguf(path, tensors=[('fixture', [1], 0, 0)])
        else:
            path.write_bytes(b'{}')  # Presence fixture; no helper or native load.
        paths[purpose] = path
        assets.append(dict(id=purpose, role=purpose, format=format,
                           logical_path=item['path'], ownership='owned',
                           prepared_ownership='caller-owned', source_path=str(path)))
        st = path.stat()
        evidence.append(dict(path=str(path), device=st.st_dev, inode=st.st_ino,
                             size=st.st_size, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns))
    reference = recipe['executable']
    operation = pipelines.validate_files(executable, recipe['files'])
    preparation = {'id':operation['id'], 'declaration':operation} if operation is not None else None
    record = dict(id='fixture', directory=str(directory), pipeline=reference, assets=assets,
                  provenance=dict(recipe_id=recipe['id'], recipe_identity=identity,
                    pipeline=reference, declaration=deepcopy(recipe), preparation=preparation, executable=dict(reference=reference,
                      id=executable['id'], identity=executable_identity, declaration=deepcopy(executable))))
    return record, paths, tuple(paths.values()), evidence


class ConfigurationFixture:
    def configure(self, identifier='qwen35-08b-q5-k-xl'):
        self.catalog_root = self.root / 'loom-pipeline'
        shutil.copytree(SOURCE, self.catalog_root)
        patcher = patch.object(pipelines, 'catalog_roots', return_value=(self.catalog_root,))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.recipe, self.recipe_identity = recipes.load(identifier)
        self.reference = self.recipe['executable']
        self.recipe_path = self.catalog_root / Path(self.reference).parent / 'recipes' / (identifier + '.json')
        self.executable, self.catalog, self.executable_identity = pipelines.load(self.reference)
        self.serving = self.executable['serving']
        self.modality = modalities.get(self.executable['modality'])
        self.record, self.paths, self.dependencies, self.evidence = record_for(
            self.root, self.recipe, self.recipe_identity)

    def replace_recipe(self, **values):
        document = dict(self.recipe, **values)
        self.recipe_path.write_text(json.dumps(document))
        self.recipe, self.recipe_identity = recipes.load(document['id'])
        self.record['provenance'].update(recipe_identity=self.recipe_identity,
                                         declaration=deepcopy(self.recipe))

    def config_file(self, contents):
        path = self.root / 'configuration.json'
        path.write_text(contents)
        return path
