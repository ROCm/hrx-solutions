"""Model validation assets and removal of the obsolete native checker."""

from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from build_tools import python_packaging as packaging
from build_tools.validate_wheel import HEM_MODULES, OBSOLETE_MODULES, OBSOLETE_ENTRY_POINTS, REQUIRED_ENTRY_POINTS, validate_wheel
from build_tools.versioning import read_package_version


class PackagingTest(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)

    def test_build_targets_only_server(self):
        hrx=self.root/'hrx'
        hrx.mkdir()
        (hrx/'.bazelversion').write_text('8.3.1')
        with patch.object(packaging,'discover_hrx_source',return_value=hrx), patch.object(packaging,'_query_bazel_output',return_value=self.root/'server') as query, patch.object(packaging.subprocess,'run') as run:
            outputs=packaging.build_native_executables(self.root/'bazel',self.root/'rocm')
        self.assertEqual(set(outputs),{'loom-serve'})
        self.assertEqual(packaging.BAZEL_TARGETS,{'loom-serve':'//:loom-serve'})
        self.assertEqual(query.call_count,1)
        self.assertEqual(run.call_args.args[0][-1],packaging.BAZEL_TARGETS['loom-serve'])
        self.assertNotIn('epoch_check',' '.join(run.call_args.args[0]))

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
        catalog=packaging.stage_loom_source_catalog(package)
        members={
            'hrx_solutions/_native/bin/loom-serve':b'server',
            'hrx_solutions/_native/share/loom-serve/LICENSE':b'license',
            'hrx_solutions/_native/lib/libhsa-runtime64.so.1':b'hsa',
            'hrx_solutions/_native/lib/libhsa-amd-aqlprofile64.so.1':b'profile',
        }
        members.update({'hrx_solutions/'+path.relative_to(package).as_posix():path.read_bytes() for path in catalog})
        for name in ('pipeline_validation.py','runtime_paths.py','model_store.py','model_download.py','gguf.py','qwen_validation.py','loom_serve.py', *(name+'.py' for name in HEM_MODULES)):
            members['hrx_solutions/'+name]=(packaging.REPO_ROOT/'binding/python'/name).read_bytes()
        version=read_package_version()
        prefix=f'hrx_solutions-{version}.dist-info/'
        members[prefix+'METADATA']=(f'Metadata-Version: 2.1\nName: hrx-solutions\nVersion: {version}\nDescription-Content-Type: text/markdown\n\n'+(packaging.REPO_ROOT/'PYPI_README.md').read_text()).encode()
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

    def test_wheel_contains_new_validation_assets(self):
        libraries,size=validate_wheel(self.wheel())
        self.assertEqual(libraries,2)
        self.assertGreater(size,0)
        with self.assertRaisesRegex(RuntimeError,'missing setup assets.*compatibility.json'):
            validate_wheel(self.wheel(missing='hrx_solutions/loom-pipeline/llm/qwen/qwen_27b/compatibility.json'))
        with self.assertRaisesRegex(RuntimeError,'missing setup assets.*gguf.py'):
            validate_wheel(self.wheel(missing='hrx_solutions/gguf.py'))

    def test_wheel_requires_both_architecture_catalogs(self):
        for filename in ('sources.txt', 'config.loom', 'prepare.loom', 'control.loom', 'weights.loom', 'compatibility.json'):
            with self.subTest(filename=filename), self.assertRaisesRegex(RuntimeError, 'missing setup assets'):
                validate_wheel(self.wheel(missing='hrx_solutions/loom-pipeline/llm/qwen/qwen_08b/'+filename))

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

    def test_wheel_rejects_old_checker(self):
        with self.assertRaisesRegex(RuntimeError,'obsolete model execution checkers'):
            validate_wheel(self.wheel(obsolete=True))


if __name__ == '__main__':
    unittest.main()
