"""Model validation assets and removal of the obsolete native checker."""

from pathlib import Path
import os
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from build_tools.wheel_tools import python_packaging as packaging
from build_tools.model_tools import loom_catalog
from build_tools.wheel_tools.validate_wheel import HEM_MODULES, OBSOLETE_MODULES, OBSOLETE_ENTRY_POINTS, REQUIRED_ENTRY_POINTS, PI_ASSETS, validate_wheel
from binding.python import hem_pi_assets, hem_pi_contracts
import hashlib
import json
import io
import tarfile
from build_tools.wheel_tools.validate_wheel import validate_pi_sdist
from build_tools.versioning import read_package_version


class PackagingTest(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.decode = patch.object(loom_catalog, 'validate_bytecode_artifact').start()
        self.addCleanup(patch.stopall)

    def test_build_includes_catalog_but_native_mapping_only_contains_server(self):
        hrx=self.root/'hrx'
        hrx.mkdir()
        (hrx/'.bazelversion').write_text('8.3.1')
        with patch.object(packaging,'discover_hrx_source',return_value=hrx), patch.object(packaging,'_query_bazel_output',return_value=self.root/'server') as query, patch.object(packaging.subprocess,'run') as run:
            outputs=packaging.build_native_executables(self.root/'bazel',self.root/'rocm')
        self.assertEqual(set(outputs),{'loom-serve'})
        self.assertEqual(packaging.BAZEL_TARGETS,{'loom-serve':'//:loom-serve'})
        self.assertEqual(query.call_count,1)
        self.assertIn(packaging.BAZEL_TARGETS['loom-serve'],run.call_args.args[0])
        self.assertEqual(run.call_args.args[0][-1],loom_catalog.CATALOG_TARGET)
        self.assertNotIn('epoch_check',' '.join(run.call_args.args[0]))

    def test_explicit_cache_disable_overrides_stale_rc_for_build_and_query(self):
        with patch.dict(os.environ, {"HRX_BAZEL_CACHE_ENABLED": "0"}), patch.object(packaging, "_query_bazel_output", return_value=self.root / "server") as query, patch.object(packaging.subprocess, "run") as run:
            packaging.build_native_executables(self.root / "bazel", self.root / "rocm")
        self.assertIn("--disk_cache=", run.call_args.args[0])
        self.assertIn("--disk_cache=", query.call_args.args[2])
        self.assertIn("--action_env=HRX_WHEEL_COMPILER_INPUTS=", run.call_args.args[0])
        self.assertIn("--host_action_env=HRX_WHEEL_COMPILER_INPUTS=", run.call_args.args[0])

    def test_metrics_only_apply_to_native_build(self):
        with patch.dict(os.environ, {"HRX_BAZEL_METRICS_DIR": str(self.root / "metrics")}), patch.object(packaging, "_query_bazel_output", return_value=self.root / "server") as query, patch.object(packaging.subprocess, "run") as run:
            packaging.build_native_executables(self.root / "bazel", self.root / "rocm")
        self.assertTrue(any(arg.startswith("--profile=") for arg in run.call_args.args[0]))
        self.assertTrue(any(arg.startswith("--build_event_json_file=") for arg in run.call_args.args[0]))
        self.assertFalse(any(arg.startswith("--profile=") or arg.startswith("--build_event_json_file=") for arg in query.call_args.args[2]))

    def test_staging_removes_stale_binary_and_preserves_runtime(self):
        rocm=self.root/'rocm'
        for name in packaging.ROCM_RUNTIME_ROOTS:
            path=rocm/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(name)
        hrx=self.root/'hrx'
        hrx.mkdir()
        (hrx/'LICENSE').write_text('license')
        server=self.root/'server'
        server.write_text('server')
        native=self.root/'_native'
        (native/'bin').mkdir(parents=True)
        (native/'bin/loom-qwen-epoch-check').write_text('stale checker')
        (native/'lib').mkdir()
        (native/'lib/stale.so').write_text('stale library')
        outputs=packaging.stage_native_payload(native,{'loom-serve':server},rocm,hrx,needed_libraries=lambda _:())
        self.assertEqual({path.name for path in (native/'bin').iterdir()},{'loom-serve'})
        self.assertFalse((native/'lib/stale.so').exists())
        self.assertTrue((native/'bin/loom-serve').stat().st_mode & 0o111)
        for name in packaging.ROCM_RUNTIME_ROOTS:
            self.assertIn(native/name,outputs)

    def wheel(self, *, obsolete=False, missing=None):
        package=self.root/'staged'
        inventory = packaging.REPO_ROOT / 'loom-pipeline/catalog_sources.bzl'
        products = {}
        for architecture, variant in loom_catalog.variants(inventory).items():
            product = self.root / 'outputs' / variant['output']
            product.parent.mkdir(parents=True, exist_ok=True)
            product.write_bytes(b'LOOMfixture')
            products[architecture] = product
        catalog=loom_catalog.stage_bytecode_catalog(package / 'loom-pipeline/llm', products, inventory, decoder=Path("fixture-decoder"))
        members={
            'hrx_solutions/_native/bin/loom-serve':b'server',
            'hrx_solutions/_native/share/loom-serve/LICENSE':b'license',
            'hrx_solutions/_native/lib/libhsa-runtime64.so.1':b'hsa',
            'hrx_solutions/_native/lib/libhsa-amd-aqlprofile64.so.1':b'profile',
        }
        members.update({'hrx_solutions/'+path.relative_to(package).as_posix():path.read_bytes() for path in catalog})
        members.update({'hrx_solutions/pi_extension/'+name:
                        (packaging.REPO_ROOT/'binding/python/pi_extension'/name).read_bytes()
                        for name in PI_ASSETS})
        for name in ('pipeline_validation.py','runtime_paths.py','model_store.py','model_download.py','gguf.py','qwen_validation.py','loom_serve.py', *(name+'.py' for name in HEM_MODULES)):
            members['hrx_solutions/'+name]=(packaging.REPO_ROOT/'binding/python'/name).read_bytes()
        version=read_package_version()
        prefix=f'hrx_solutions-{version}.dist-info/'
        members[prefix+'METADATA']=(f'Metadata-Version: 2.1\nName: hrx-solutions\nVersion: {version}\nRequires-Python: >=3.9\nRequires-Dist: textual<2,>=0.47\nDescription-Content-Type: text/markdown\n\n'+(packaging.REPO_ROOT/'PYPI_README.md').read_text()).encode()
        members[prefix+'entry_points.txt']=('[console_scripts]\n'+''.join(f'{command} = {target}\n' for command,target in REQUIRED_ENTRY_POINTS.items())).encode()
        if obsolete:
            members['hrx_solutions/_native/bin/loom-qwen-epoch-check']=b'obsolete'
        if missing:
            del members[missing]
        wheel=self.root/'fixture.whl'
        with zipfile.ZipFile(wheel,'w') as archive:
            for name,data in members.items():
                archive.writestr(name,data)
        return wheel

    def test_pi_resources_match_contract_and_loader_metadata(self):
        assets = hem_pi_assets.bundled_assets()
        prefix = hem_pi_contracts.EXTENSION_PATH + '/'
        self.assertEqual(set(assets), {prefix + name for name in PI_ASSETS})
        self.assertEqual(hem_pi_assets.asset_digests(assets),
                         {name: hashlib.sha256(data).hexdigest() for name, data in assets.items()})
        profile = json.loads(assets[prefix+'profile.json'])
        for key, value in profile.items():
            expected = getattr(hem_pi_contracts, key.upper())
            self.assertEqual(value, list(expected) if isinstance(expected, tuple) else expected)
        metadata = json.loads(assets[prefix+'package.json'])
        self.assertEqual(metadata['pi']['extensions'], ['./index.ts'])
        self.assertFalse(metadata.get('dependencies'))
        self.assertFalse(metadata.get('scripts'))

    def test_wheel_requires_every_pi_resource_and_reviewed_bytes(self):
        for name in PI_ASSETS:
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'missing setup assets'):
                validate_wheel(self.wheel(missing='hrx_solutions/pi_extension/'+name))
        wheel = self.wheel()
        with zipfile.ZipFile(wheel) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        members['hrx_solutions/pi_extension/index.ts'] += b'\n// changed\n'
        with zipfile.ZipFile(wheel, 'w') as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        with self.assertRaisesRegex(RuntimeError, 'Pi asset differs'):
            validate_wheel(wheel)

    def test_sdist_requires_identical_regular_pi_resources(self):
        for failure in (None, 'missing', 'changed', 'symlink'):
            path = self.root/'source.tar.gz'
            with tarfile.open(path, 'w:gz') as archive:
                for name in PI_ASSETS:
                    data = (packaging.REPO_ROOT/'binding/python/pi_extension'/name).read_bytes()
                    if name == 'index.ts' and failure == 'missing':
                        continue
                    if name == 'index.ts' and failure == 'changed':
                        data += b'// changed'
                    member = tarfile.TarInfo('hrx-solutions/binding/python/pi_extension/'+name)
                    member.size = len(data)
                    if name == 'index.ts' and failure == 'symlink':
                        member.type = tarfile.SYMTYPE
                        member.linkname = '/unreviewed/index.ts'
                    archive.addfile(member, io.BytesIO(data))
            with self.subTest(failure=failure):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, 'Pi asset'):
                        validate_pi_sdist(path)
                else:
                    validate_pi_sdist(path)

    def test_wheel_contains_new_validation_assets(self):
        libraries,size=validate_wheel(self.wheel())
        self.assertEqual(libraries,2)
        self.assertGreater(size,0)
        with zipfile.ZipFile(self.root/'fixture.whl') as archive:
            catalog = [name for name in archive.namelist() if name.startswith('hrx_solutions/loom-pipeline/llm/')]
            self.assertEqual(len(catalog), 8)
            self.assertEqual(sum(name.endswith('.loombc') for name in catalog), 2)
            self.assertFalse(any(name.endswith('sources.txt') for name in catalog))
        with self.assertRaisesRegex(RuntimeError,'missing setup assets.*compatibility.json'):
            validate_wheel(self.wheel(missing='hrx_solutions/loom-pipeline/llm/qwen/qwen_27b/compatibility.json'))
        with self.assertRaisesRegex(RuntimeError,'missing setup assets.*gguf.py'):
            validate_wheel(self.wheel(missing='hrx_solutions/gguf.py'))

    def test_wheel_requires_both_architecture_catalogs(self):
        for filename in ('pipeline.loombc', 'pipeline.json', 'compatibility.json'):
            with self.subTest(filename=filename), self.assertRaisesRegex(RuntimeError, 'missing setup assets'):
                validate_wheel(self.wheel(missing='hrx_solutions/loom-pipeline/llm/qwen/qwen_08b/'+filename))

    def test_wheel_rejects_source_fallback_corruption_and_metadata_drift(self):
        mutations = ('source', 'bytecode', 'extra', 'descriptor', 'profile',
                     'variant', 'source_hash', 'source_inventory', 'bytecode_hash', 'format')
        for mutation in mutations:
            wheel = self.wheel()
            with zipfile.ZipFile(wheel) as archive:
                members = {name: archive.read(name) for name in archive.namelist()}
            prefix = 'hrx_solutions/loom-pipeline/llm/'
            local = prefix + 'qwen/qwen_08b/'
            if mutation == 'source':
                members[local+'control.loom'] = b'hidden source fallback'
            elif mutation == 'bytecode':
                members[local+'pipeline.loombc'] = b'LOOMstale'
            elif mutation == 'extra':
                members[local+'weights.loombc'] = b'LOOMauxiliary'
            elif mutation in ('descriptor', 'profile'):
                name = local + ('pipeline.json' if mutation == 'descriptor' else 'compatibility.json')
                data = json.loads(members[name])
                data['variant' if mutation == 'descriptor' else 'identifier'] = 'wrong'
                members[name] = json.dumps(data).encode()
            else:
                name = prefix + 'catalog-build.json'
                data = json.loads(members[name])
                variant = data['variants']['qwen_08b']
                if mutation == 'variant':
                    variant['identifier'] = 'wrong'
                elif mutation == 'source_hash':
                    variant['sources'][next(iter(variant['sources']))] = '0'*64
                elif mutation == 'source_inventory':
                    del variant['sources'][next(iter(variant['sources']))]
                elif mutation == 'bytecode_hash':
                    variant['bytecode_sha256'] = '0'*64
                else:
                    data['format'] = 1
                members[name] = json.dumps(data).encode()
            with zipfile.ZipFile(wheel, 'w') as archive:
                for name, data in members.items():
                    archive.writestr(name, data)
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                validate_wheel(wheel)

    def test_wheel_requires_every_hem_component_and_entrypoint(self):
        for module in HEM_MODULES:
            with self.subTest(module=module), self.assertRaisesRegex(RuntimeError, module):
                validate_wheel(self.wheel(missing='hrx_solutions/'+module+'.py'))
        for target in REQUIRED_ENTRY_POINTS.values():
            module = target.split(":")[0].replace(".", "/") + ".py"
            with self.subTest(entrypoint_module=module), self.assertRaisesRegex(RuntimeError, 'missing setup assets'):
                validate_wheel(self.wheel(missing=module))
        wheel = self.wheel()
        with zipfile.ZipFile(wheel) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        entrypoint = next(name for name in members if name.endswith('/entry_points.txt'))
        original = members[entrypoint]
        for command, target in REQUIRED_ENTRY_POINTS.items():
            with self.subTest(command=command):
                members[entrypoint] = original.replace(f'{command} = {target}'.encode(),
                                                       f'{command} = wrong.module:main'.encode())
                with zipfile.ZipFile(wheel, 'w') as archive:
                    for name, data in members.items():
                        archive.writestr(name, data)
                with self.assertRaisesRegex(RuntimeError, command):
                    validate_wheel(wheel)

    def test_wheel_rejects_retired_manager_surface(self):
        for module in OBSOLETE_MODULES:
            wheel = self.wheel()
            with zipfile.ZipFile(wheel, 'a') as archive:
                archive.writestr('hrx_solutions/' + module + '.py', b'retired')
            with self.subTest(module=module), self.assertRaisesRegex(RuntimeError, 'retired manager modules'):
                validate_wheel(wheel)
        for command in OBSOLETE_ENTRY_POINTS:
            wheel = self.wheel()
            with zipfile.ZipFile(wheel) as archive:
                members = {name: archive.read(name) for name in archive.namelist()}
            entrypoint = next(name for name in members if name.endswith('/entry_points.txt'))
            members[entrypoint] += f'{command} = hrx_solutions.setup_cli:main\n'.encode()
            with zipfile.ZipFile(wheel, 'w') as archive:
                for name, data in members.items():
                    archive.writestr(name, data)
            with self.subTest(command=command), self.assertRaisesRegex(RuntimeError, 'retired commands'):
                validate_wheel(wheel)

    def test_wheel_requires_control_and_rejects_hem_even_alongside_control(self):
        # Literal metadata is independent of the validator's required/retired sets.
        for scripts, error in (
            ('loom-serve = hrx_solutions.loom_serve:main\n', 'does not expose the loom-control command'),
            ('loom-serve = hrx_solutions.loom_serve:main\n'
             'loom-hem = hrx_solutions.hem_cli:main\n', 'retired commands: loom-hem'),
            ('loom-serve = hrx_solutions.loom_serve:main\n'
             'loom-control = hrx_solutions.hem_cli:main\n'
             'loom-hem = hrx_solutions.hem_cli:main\n', 'retired commands: loom-hem'),
        ):
            with self.subTest(scripts=scripts):
                wheel = self.wheel()
                with zipfile.ZipFile(wheel) as archive:
                    members = {name: archive.read(name) for name in archive.namelist()}
                entrypoint = next(name for name in members if name.endswith('/entry_points.txt'))
                members[entrypoint] = ('[console_scripts]\n' + scripts).encode()
                with zipfile.ZipFile(wheel, 'w') as archive:
                    for name, data in members.items():
                        archive.writestr(name, data)
                with self.assertRaisesRegex(RuntimeError, error):
                    validate_wheel(wheel)

    def test_wheel_rejects_old_checker(self):
        with self.assertRaisesRegex(RuntimeError,'obsolete model execution checkers'):
            validate_wheel(self.wheel(obsolete=True))


if __name__ == '__main__':
    unittest.main()
