"""Real authored catalogs with explicit synthetic compiler/tool fixtures for unit tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch

from . import loom_catalog, package_families

REPOSITORY = Path(__file__).resolve().parents[2]


class CatalogFixture:
    def __init__(self, root):
        self.root = root
        self.repository = root / 'source'
        shutil.copytree(REPOSITORY / 'loom-pipeline', self.repository / 'loom-pipeline', symlinks=True)
        contract = 'binding/python/models/preparation/moonshine-operation.json'
        destination = self.repository / contract
        destination.parent.mkdir(parents=True)
        self.operation = json.loads((REPOSITORY / contract).read_bytes())
        self.tools = {'prepare_assets.py': b'# trusted synthetic fixture\n', 'model.json': b'{}\n'}
        self.operation['tool_files'] = {name: hashlib.sha256(data).hexdigest() for name, data in self.tools.items()}
        destination.write_text(json.dumps(self.operation, indent=2) + '\n')
        shutil.copyfile(REPOSITORY / '.bazelversion', self.repository / '.bazelversion')
        self.builds = loom_catalog.load_builds(self.repository / 'loom-pipeline/catalog_sources.bzl')
        self.operations = {self.operation['id']: self.operation}
        for build in self.builds.values():
            helper = build['helper']
            if helper and not (self.repository / helper['contract']).exists():
                declaration = json.loads((REPOSITORY / helper['contract']).read_bytes())
                declaration['tool_files'] = dict(self.operation['tool_files'])
                (self.repository / helper['contract']).write_text(json.dumps(declaration, indent=2) + '\n')
                self.operations[declaration['id']] = declaration
        self.products, self.decoded, self.queried = {}, {}, []
        self.hrx = root / 'hrx'
        self.hrx.mkdir()
        (self.hrx / 'LICENSE').write_bytes(b'license')
        self.decoder = root / 'decoder'
        self.decoder.touch()
        for build in self.builds.values():
            if build['helper']:
                directory = self.hrx / build['helper']['upstream']
                directory.mkdir(parents=True, exist_ok=True)
                for name, content in self.tools.items():
                    (directory / name).write_bytes(content)
            for output, group in build['groups'].items():
                name = build['root'] + '/' + output
                path = root / 'bazel-bin/loom-pipeline/bytecode' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'LOOM-unit-fixture:' + name.encode())
                self.products[name] = path
                texts = []
                for source in group['sources']:
                    text = (self.repository / 'loom-pipeline' / source).read_text()
                    if group['policy'] and source == group['policy']['source']:
                        text = loom_catalog.preparation_bytecode_source(text, group['policy']['from'], group['policy']['to'], group['policy'].get('symbol', 'policy'))
                    texts.append(text)
                self.decoded[path] = '\n'.join(texts)

    def query(self, target):
        self.queried.append(target)
        return tuple(self.products.values()) if target == loom_catalog.CATALOG_TARGET else (self.decoder,)

    def decode(self, arguments, **kwargs):
        Path(arguments[-1].removeprefix('--output=')).write_text(self.decoded[Path(arguments[1])])

    def stage(self, package, selected=None):
        with patch.object(loom_catalog.subprocess, 'run', side_effect=self.decode):
            prepared = package_families.prepare(self.query, self.repository, selected=selected)
            outputs = package_families.stage(package, prepared, self.repository)
        outputs += package_families.stage_helpers(package / '_native', prepared, self.hrx, repository=self.repository)
        return outputs

    def release(self, reviewed):
        # Independent oracle is explicitly altered for synthetic trusted tool bytes;
        # it is never learned from staged output or read by production compilation.
        result = deepcopy(reviewed)
        for operation in result['operations'].values():
            selected = self.operations[operation['declaration']['id']]
            operation['declaration'] = deepcopy(selected)
            operation['provenance']['files'] = deepcopy(selected['tool_files'])
        return result
