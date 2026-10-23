"""Model validation assets and removal of the obsolete native checker."""

from pathlib import Path
from dataclasses import FrozenInstanceError
import subprocess
import sys
import shlex
from contextlib import ExitStack, contextmanager
import os
import json
import tempfile
import shutil
import unittest
from unittest.mock import patch
import zipfile

from build_tools.wheel_tools import python_packaging as packaging
from build_tools.wheel_tools import validate_wheel as wheel_validation
from build_tools.model_tools import loom_catalog, moonshine_catalog, moonshine_preparer
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
        self.preparer_data = {'prepare_assets.py': b'# synthetic trusted tool\n', 'model.json': b'{}\n'}
        patch.object(moonshine_preparer, 'FILES', {name:hashlib.sha256(data).hexdigest()
                                                for name,data in self.preparer_data.items()}).start()
        self.addCleanup(patch.stopall)

    def context(self, *, environment=None, options=None):
        return packaging.BazelBuildContext(self.root/'bazel', self.root,
            options if options is not None else packaging.wheel_build_options(self.root/'rocm'),
            environment if environment is not None else packaging.bazel_environment())

    def test_build_includes_catalog_but_native_mapping_only_contains_server(self):
        context=self.context()
        with patch.object(packaging,'_native_output',return_value=self.root/'server') as query, patch.object(packaging.subprocess,'run') as run:
            outputs=packaging.build_native_executables(context)
        self.assertEqual(set(outputs),{'loom-serve'})
        self.assertEqual(packaging.BAZEL_TARGETS,{'loom-serve':'//:loom-serve'})
        query.assert_called_once_with(context, '//:loom-serve')
        self.assertIn(packaging.BAZEL_TARGETS['loom-serve'],run.call_args.args[0])
        self.assertIn(loom_catalog.CATALOG_TARGET,run.call_args.args[0])
        self.assertEqual(run.call_args.args[0][-1],moonshine_catalog.TARGET)
        self.assertNotIn('epoch_check',' '.join(run.call_args.args[0]))

    def test_explicit_cache_disable_overrides_stale_rc_for_build_and_query(self):
        with patch.dict(os.environ, {"HRX_BAZEL_CACHE_ENABLED": "0"}):
            context=self.context()
        with patch.object(packaging, "_native_output", return_value=self.root / "server"), patch.object(packaging.subprocess, "run") as run:
            packaging.build_native_executables(context)
            context.query('//:loom-serve')
        for call in run.call_args_list:
            for option in ('--disk_cache=', '--action_env=HRX_WHEEL_COMPILER_INPUTS=',
                           '--host_action_env=HRX_WHEEL_COMPILER_INPUTS='):
                self.assertIn(option, call.args[0])

    def test_catalog_stages_share_one_context_and_required_decoder_lookup(self):
        bazel, rocm, decoder = (self.root / name for name in ('bazel', 'rocm', 'decoder'))
        options = ('--config=opt', '--disk_cache=')
        environment = {'HRX_BAZEL_CACHE_ENABLED': '0'}
        command = packaging.BazelBuildPy(packaging.BinaryDistribution())
        command.build_lib = str(self.root / 'build')
        with ExitStack() as stack:
            mocks={}
            for name, result in (
                ('validate_wheel_platform', None), ('discover_bazel', bazel),
                ('discover_rocm_root', rocm), ('build_native_executables', {}),
                ('wheel_build_options', options), ('bazel_environment', environment),
                ('discover_hrx_source', self.root / 'hrx'), ('stage_native_payload', []),
            ):
                mocks[name]=stack.enter_context(patch.object(packaging, name, return_value=result))
            stack.enter_context(patch.object(packaging.build_py, 'run'))
            stack.enter_context(patch.object(packaging.package_families, 'stage_helpers', return_value=[]))
            bytecodes=stack.enter_context(patch.object(loom_catalog, 'query_bytecode_outputs', return_value={}))
            artifacts=stack.enter_context(patch.object(moonshine_catalog, 'query_outputs', return_value={}))
            lookup = stack.enter_context(patch.object(loom_catalog, 'query_decoder', return_value=decoder))
            qwen = stack.enter_context(patch.object(loom_catalog, 'stage_bytecode_catalog', return_value=[]))
            moonshine = stack.enter_context(patch.object(moonshine_catalog, 'stage', return_value=[]))
            command.run()
        context=mocks['build_native_executables'].call_args.args[0]
        mocks['wheel_build_options'].assert_called_once_with(rocm)
        mocks['bazel_environment'].assert_called_once_with()
        bytecodes.assert_called_once_with(context.query, context.repository)
        lookup.assert_called_once_with(context.query)
        artifacts.assert_called_once_with(context.query)
        mocks['discover_hrx_source'].assert_called_once_with(context)
        self.assertEqual((context.executable, context.options, dict(context.environment)), (bazel, options, environment))
        self.assertIs(qwen.call_args.kwargs['decoder'], decoder)
        self.assertIs(moonshine.call_args.kwargs['decoder'], decoder)

    @contextmanager
    def buildpy_fixture(self):
        repository = packaging.REPO_ROOT
        products = {}
        for architecture, variant in loom_catalog.variants(repository / 'loom-pipeline/catalog_sources.bzl').items():
            path = self.root / 'bazel-out/loom-pipeline/bytecode' / variant['output']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'LOOM-qwen-fixture-' + architecture.encode())
            products[architecture] = path
        moonshine = {}
        for name in moonshine_catalog.ARTIFACTS:
            path = self.root / 'bazel-out/loom-pipeline' / moonshine_catalog.RELATIVE / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'LOOM-moonshine-fixture-' + name.encode())
            moonshine[name] = path
        decoder = self.root / 'decoder'; decoder.write_bytes(b'fixture')
        queried, staged = [], []
        def query(context, target):
            queried.append((context, target))
            return {loom_catalog.CATALOG_TARGET: tuple(products.values()),
                    moonshine_catalog.TARGET: tuple(moonshine.values()),
                    loom_catalog.DECODER_TARGET: (decoder,)}[target]
        qwen_stage, moonshine_stage = loom_catalog.stage_bytecode_catalog, moonshine_catalog.stage
        def stage_qwen(*args, **kwargs):
            staged.append('qwen')
            return qwen_stage(*args, **kwargs)
        def stage_moonshine(*args, **kwargs):
            staged.append('moonshine')
            return moonshine_stage(*args, **kwargs)
        def fixture_roots(path, sources, root, decoder, policy=False):
            return moonshine_catalog.roots('\n'.join((root/source).read_text() for source in sources))
        hrx, rocm = self.root / 'selected-hrx', self.root / 'rocm'
        tool = hrx / moonshine_preparer.UPSTREAM
        tool.mkdir(parents=True, exist_ok=True)
        (hrx / 'LICENSE').write_bytes(b'selected HRX license')
        for name, content in self.preparer_data.items():
            (tool / name).write_bytes(content)
        for name in packaging.ROCM_RUNTIME_ROOTS:
            path = rocm / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        server = self.root / 'server'; server.write_bytes(b'native fixture')
        runtime_payload = packaging.resolve_rocm_runtime_payload
        command = packaging.BazelBuildPy(packaging.BinaryDistribution())
        command.build_lib = str(self.root / 'build')
        with ExitStack() as stack:
            for name, result in (('validate_wheel_platform', None), ('discover_bazel', self.root/'bazel'),
                                 ('discover_rocm_root', rocm), ('build_native_executables', {'loom-serve': server}),
                                 ('discover_hrx_source', hrx)):
                stack.enter_context(patch.object(packaging, name, return_value=result))
            package = Path(command.build_lib) / 'hrx_solutions'
            python_file = package / '__init__.py'
            def python_stage():
                package.mkdir(parents=True)
                python_file.write_bytes(b'# python fixture')
            stack.enter_context(patch.object(packaging.build_py, 'run', side_effect=python_stage))
            stack.enter_context(patch.object(packaging.build_py, 'get_outputs', side_effect=lambda *args: [str(python_file)]))
            stack.enter_context(patch.object(packaging, 'resolve_rocm_runtime_payload',
                side_effect=lambda root, **kwargs: runtime_payload(root, needed_libraries=lambda _: ())))
            stack.enter_context(patch.object(packaging.BazelBuildContext, 'query', query))
            stack.enter_context(patch.object(loom_catalog, 'stage_bytecode_catalog', side_effect=stage_qwen))
            stack.enter_context(patch.object(moonshine_catalog, 'stage', side_effect=stage_moonshine))
            stack.enter_context(patch.object(moonshine_catalog, 'validate_artifact', side_effect=fixture_roots))
            yield command, package, products, moonshine, hrx, queried, staged

    def test_buildpy_actual_family_queries_stage_catalogs_and_owned_helper_outputs(self):
        with self.buildpy_fixture() as fixture:
            command, package, products, moonshine, hrx, queried, staged = fixture
            command.run()
            first_outputs = command.get_outputs()
            helper = package / '_native' / moonshine_preparer.RELATIVE
            self.assertEqual(set(helper.iterdir()), {helper / name for name in (*self.preparer_data, 'LICENSE', 'provenance.json')})
            for name, content in self.preparer_data.items():
                self.assertEqual((helper / name).read_bytes(), content)
            self.assertEqual((helper / 'LICENSE').read_bytes(), (hrx / 'LICENSE').read_bytes())
            self.assertEqual(json.loads((helper / 'provenance.json').read_text())['files'], moonshine_preparer.FILES)
            # Repeat the real build/stage boundary with stale previous native/helper files.
            (helper / 'stale.py').write_bytes(b'old helper garbage')
            stale_native = package / '_native/bin/retired-tool'
            stale_native.write_bytes(b'old native garbage')
            command.run()
            final_outputs = command.get_outputs()
            self.assertFalse((helper / 'stale.py').exists())
            self.assertFalse(stale_native.exists())
        self.assertEqual(staged, ['qwen', 'moonshine'] * 2)
        self.assertEqual([target for _, target in queried],
                         [loom_catalog.CATALOG_TARGET, moonshine_catalog.TARGET, loom_catalog.DECODER_TARGET] * 2)
        self.assertTrue(all(context is queried[0][0] for context, _ in queried[:3]))
        self.assertTrue(all(context is queried[3][0] for context, _ in queried[3:]))
        self.assertEqual(set(first_outputs), set(final_outputs))
        self.assertEqual(len(final_outputs), len(set(final_outputs)))
        self.assertEqual({str(path) for path in package.rglob('*') if path.is_file()}, set(final_outputs))
        self.assertEqual(len(command._catalog_outputs), len(set(command._catalog_outputs)))
        package = Path(command.build_lib) / 'hrx_solutions'
        self.assertEqual({path for path in (package / 'loom-pipeline').rglob('*') if path.is_file()}, set(command._catalog_outputs))
        self.assertEqual(sum(path.suffix == '.loombc' for path in command._catalog_outputs), 6)
        for architecture, path in products.items():
            self.assertEqual((package/'loom-pipeline/llm/qwen'/architecture/'pipeline.loombc').read_bytes(), path.read_bytes())
        for name, path in moonshine.items():
            self.assertEqual((package/'loom-pipeline'/moonshine_catalog.RELATIVE/name).read_bytes(), path.read_bytes())

    def test_duplicate_family_producer_refused_before_actual_buildpy_clears_package(self):
        producers = packaging.package_families.PRODUCERS
        with self.buildpy_fixture() as fixture:
            command, package, _, _, _, queried, staged = fixture
            retained = package / 'retained.txt'
            package.mkdir(parents=True); retained.write_bytes(b'existing package')
            with patch.object(packaging.package_families, 'PRODUCERS', (producers[0], *producers)):
                with self.assertRaisesRegex(ValueError, 'Duplicate.*producer'):
                    command.run()
            self.assertEqual(retained.read_bytes(), b'existing package')
            self.assertEqual(queried, [])
            self.assertEqual(staged, [])

    def test_actual_packaging_omitted_required_family_fails_independent_wheel_acceptance(self):
        # Keep only independent Python/metadata/resources from the positive fixture;
        # every generated catalog/native/helper member below comes from BuildPy.
        with zipfile.ZipFile(self.wheel()) as archive:
            base = {name: archive.read(name) for name in archive.namelist()
                    if not name.startswith(('hrx_solutions/loom-pipeline/', 'hrx_solutions/_native/'))}
        producers = packaging.package_families.PRODUCERS
        for omitted in (None, loom_catalog.CATALOG_TARGET, moonshine_catalog.TARGET):
            selected = tuple(producer for producer in producers if producer[0] != omitted)
            # Keep the omission active through acceptance, so a checker derived
            # from the producer tuple would incorrectly skip required payloads.
            with self.subTest(omitted=omitted), patch.object(
                    packaging.package_families, 'PRODUCERS', selected), self.buildpy_fixture() as fixture:
                command, package, _, _, _, _, _ = fixture
                command.run()
                outputs = command.get_outputs()
                self.assertEqual(len(outputs), len(set(outputs)))
                self.assertEqual(set(outputs), {str(path) for path in package.rglob('*') if path.is_file()})
                members = dict(base)
                members.update({'hrx_solutions/' + path.relative_to(package).as_posix(): path.read_bytes()
                                for path in package.rglob('*') if path.is_file()})
                candidate = self.root / 'actual-package-fixture.whl'
                with zipfile.ZipFile(candidate, 'w') as archive:
                    for name, content in members.items():
                        archive.writestr(name, content)
                with patch.object(loom_catalog, 'validate_archive', wraps=loom_catalog.validate_archive) as qwen, patch.object(
                        moonshine_catalog, 'validate_archive', wraps=moonshine_catalog.validate_archive) as moonshine, patch.object(
                        moonshine_preparer, 'validate_archive', wraps=moonshine_preparer.validate_archive) as helper:
                    if omitted is None:
                        self.assertEqual(wheel_validation.validate_wheel(candidate)[0], 2)
                        self.assertEqual((qwen.call_count, moonshine.call_count, helper.call_count), (1, 1, 1))
                    elif omitted == loom_catalog.CATALOG_TARGET:
                        # Literal required Qwen inventory is independent of production.
                        with self.assertRaisesRegex(RuntimeError, 'missing setup assets.*qwen'):
                            wheel_validation.validate_wheel(candidate)
                    else:
                        # Existing required Moonshine validator runs despite omission.
                        with self.assertRaisesRegex(ValueError, 'Moonshine installed catalog allowlist'):
                            wheel_validation.validate_wheel(candidate)
                        self.assertEqual((qwen.call_count, moonshine.call_count), (1, 1))
                        self.assertFalse((package / '_native' / moonshine_preparer.RELATIVE).exists())

    def test_incomplete_family_refused_before_existing_package_is_cleared(self):
        command = packaging.BazelBuildPy(packaging.BinaryDistribution())
        command.build_lib = str(self.root / 'build')
        retained = Path(command.build_lib) / 'hrx_solutions/retained.txt'
        retained.parent.mkdir(parents=True); retained.write_bytes(b'existing payload')
        for family, reader in ((loom_catalog, 'query_bytecode_outputs'), (moonshine_catalog, 'query_outputs')):
            with self.subTest(family=family.__name__), ExitStack() as stack:
                for name, result in (('validate_wheel_platform', None), ('discover_bazel', self.root/'bazel'),
                                     ('discover_rocm_root', self.root/'rocm'), ('build_native_executables', {})):
                    stack.enter_context(patch.object(packaging, name, return_value=result))
                build = stack.enter_context(patch.object(packaging.build_py, 'run'))
                stage = stack.enter_context(patch.object(packaging.package_families, 'stage'))
                stack.enter_context(patch.object(loom_catalog, 'query_bytecode_outputs', return_value={}))
                stack.enter_context(patch.object(moonshine_catalog, 'query_outputs', return_value={}))
                stack.enter_context(patch.object(family, reader, side_effect=ValueError('incomplete required family')))
                with self.assertRaisesRegex(ValueError, 'incomplete required family'):
                    command.run()
                build.assert_not_called(); stage.assert_not_called()
                self.assertEqual(retained.read_bytes(), b'existing payload')

    def test_build_and_all_queries_keep_snapshot_and_metrics_stay_build_only(self):
        environment={'BUILD':'original', 'HRX_BAZEL_METRICS_DIR':str(self.root/'metrics')}
        options=['--config=opt', '--disk_cache=']
        context=self.context(environment=environment,options=options)
        server=self.root/'server';server.write_bytes(b'executable')
        def run(arguments, **kwargs):
            # Simulate an inherited environment/options change during the build.
            self.assertEqual(kwargs['env']['BUILD'],'original')
            environment['BUILD']='changed';options.append('--unreviewed-option')
            kwargs['env']['BUILD']='child-only'
            return subprocess.CompletedProcess(arguments,0,stdout='server\n')
        with patch.object(packaging.subprocess,'run',side_effect=run) as process:
            self.assertEqual(packaging.build_native_executables(context), {'loom-serve':server})
            for target in (loom_catalog.DECODER_TARGET,loom_catalog.CATALOG_TARGET,moonshine_catalog.TARGET):
                self.assertEqual(context.query(target),(server,))
        for call in process.call_args_list:
            args=call.args[0]
            self.assertEqual(args[0],str(context.executable))
            self.assertEqual(call.kwargs['cwd'],context.repository)
            # The mock changed its private child env; the immutable source stayed original.
            self.assertEqual(context.environment['BUILD'],'original')
            self.assertNotIn('--unreviewed-option',args)
            self.assertIn('--disk_cache=',args)
            metrics=any(arg.startswith(('--profile=','--build_event_json_file=')) for arg in args)
            self.assertEqual(metrics,args[1]=='build')
        with self.assertRaises(TypeError):context.environment['BUILD']='mutate'
        with self.assertRaises(FrozenInstanceError):context.options=()

    def test_hrx_source_lookup_uses_the_build_snapshot(self):
        context=self.context(environment={'BUILD':'original'})
        source=self.root/'external/selected-hrx'
        source.mkdir(parents=True);(source/'MODULE.bazel').write_text('module()')
        def run(arguments, **kwargs):
            self.assertEqual(arguments[0],str(context.executable))
            self.assertEqual(kwargs['cwd'],context.repository)
            self.assertEqual(kwargs['env'],{'BUILD':'original'})
            if arguments[1]=='mod': output='{"hrx_system":"selected-hrx"}'
            elif arguments[1]=='info': output=str(self.root)
            else: output=''
            return subprocess.CompletedProcess(arguments,0,stdout=output)
        with patch.object(packaging,'bazel_environment',side_effect=AssertionError('recomputed environment')), patch.object(packaging.subprocess,'run',side_effect=run) as process:
            self.assertEqual(packaging.discover_hrx_source(context),source)
        self.assertEqual([call.args[0][1] for call in process.call_args_list],['fetch','mod','info'])

    def test_wheel_bootstrap_payload_discovers_source_before_rocm_setup(self):
        script=(packaging.REPO_ROOT/'.github/scripts/build_python_wheel.sh').read_text()
        line=next(line for line in script.splitlines() if line.startswith('HRX_SOURCE_DIR=$('))
        # Execute the actual shell-owned -c payload, rather than a copied helper
        # call that could keep passing while the build entry becomes stale.
        payload,=shlex.split(line.split(' -c ',1)[1][:-1])
        source=self.root/'external/bootstrap-hrx'
        source.mkdir(parents=True);(source/'MODULE.bazel').write_text('module()')
        trace=self.root/'bazel-calls.jsonl'
        bazel=self.root/'fake-bazel'
        bazel.write_text(
            '#!'+sys.executable+'\nimport json,sys\nfrom pathlib import Path\n'
            +f'with Path({str(trace)!r}).open("a") as stream: stream.write(json.dumps(sys.argv[1:])+"\\n")\n'
            +'if sys.argv[1]=="mod": print(json.dumps({"hrx_system":"bootstrap-hrx"}))\n'
            +f'elif sys.argv[1]=="info": print({str(self.root)!r})\n')
        bazel.chmod(0o700)
        environment=os.environ.copy()
        environment.update(BAZEL=str(bazel),IREE_ROCM_PATH=str(self.root/'absent-rocm'),
                           HRX_ROCM_ROOT=str(self.root/'absent-rocm'))
        process=subprocess.run([sys.executable,'-c',payload],cwd=packaging.REPO_ROOT,
                               env=environment,capture_output=True,text=True)
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertEqual(process.stdout.strip(),str(source))
        self.assertEqual([json.loads(line) for line in trace.read_text().splitlines()],
                         [['fetch','--repo=@hrx_system'],['mod','dump_repo_mapping',''],['info','output_base']])
        self.assertFalse((self.root/'absent-rocm').exists())

    def test_query_preserves_order_symlink_locator_and_native_singleton_checks(self):
        target=self.root/'actual';target.write_bytes(b'LOOM')
        alias=self.root/'alias';alias.symlink_to(target)
        context=self.context(environment={})
        with patch.object(packaging.subprocess,'run',return_value=subprocess.CompletedProcess([],0,stdout='alias\nactual\n')):
            self.assertEqual(context.query('target'),(alias,target))
            with self.assertRaisesRegex(RuntimeError,'Expected one'):
                packaging._native_output(context,'target')
        for content,message in (('', 'Expected one'), ('missing\n','does not exist')):
            with patch.object(packaging.subprocess,'run',return_value=subprocess.CompletedProcess([],0,stdout=content)):
                with self.assertRaisesRegex(RuntimeError,message):packaging._native_output(context,'target')
        with patch.object(packaging.subprocess,'run',return_value=subprocess.CompletedProcess([],0,stdout='alias\n')):
            self.assertEqual(packaging._native_output(context,'target'),target)

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
        old_helper = native / moonshine_preparer.RELATIVE
        old_helper.mkdir(parents=True)
        (old_helper / 'stale.py').write_text('stale helper')
        outputs=packaging.stage_native_payload(native,{'loom-serve':server},rocm,hrx,needed_libraries=lambda _:())
        self.assertEqual({path.name for path in (native/'bin').iterdir()},{'loom-serve'})
        self.assertFalse((native/'lib/stale.so').exists())
        self.assertFalse((native / moonshine_preparer.RELATIVE).exists())
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
        moonshine_products = {}
        for name in moonshine_catalog.ARTIFACTS:
            product = self.root / 'outputs' / name
            product.parent.mkdir(parents=True, exist_ok=True)
            product.write_bytes(b'LOOMfixture')
            moonshine_products[name] = product
        def fixture_roots(path, sources, root, decoder, policy=False):
            return moonshine_catalog.roots('\n'.join((root/source).read_text() for source in sources))
        with patch.object(moonshine_catalog, 'validate_artifact', side_effect=fixture_roots):
            catalog.extend(moonshine_catalog.stage(package/'loom-pipeline'/moonshine_catalog.RELATIVE,
                moonshine_products,packaging.REPO_ROOT,decoder=Path('fixture-decoder')))
        members={
            'hrx_solutions/_native/bin/loom-serve':b'server',
            'hrx_solutions/_native/share/loom-serve/LICENSE':b'license',
            'hrx_solutions/_native/lib/libhsa-runtime64.so.1':b'hsa',
            'hrx_solutions/_native/lib/libhsa-amd-aqlprofile64.so.1':b'profile',
        }
        members.update({'hrx_solutions/'+path.relative_to(package).as_posix():path.read_bytes() for path in catalog})
        hrx = self.root / 'fixture-hrx'
        tool = hrx / moonshine_preparer.UPSTREAM
        tool.mkdir(parents=True, exist_ok=True)
        (hrx/'LICENSE').write_text('license')
        for name,data in self.preparer_data.items():
            (tool/name).write_bytes(data)
        destination = package / '_native' / moonshine_preparer.RELATIVE
        if destination.exists():
            shutil.rmtree(destination)
        moonshine_preparer.stage(package/'_native', hrx)
        members.update({'hrx_solutions/'+path.relative_to(package).as_posix():path.read_bytes()
                        for path in destination.iterdir()})
        members.update({'hrx_solutions/pi_extension/'+name:
                        (packaging.REPO_ROOT/'binding/python/pi_extension'/name).read_bytes()
                        for name in PI_ASSETS})
        members.update({'hrx_solutions/'+name:(packaging.REPO_ROOT/'binding/python'/name).read_bytes()
                        for name in wheel_validation.CLIENT_ASSETS})
        for name in ('moonshine_preparation.py','moonshine_preparer_identity.py','curated_catalog.py','pipeline_validation.py','runtime_paths.py','model_store.py','model_download.py','gguf.py','qwen_validation.py','loom_serve.py', *(name+'.py' for name in HEM_MODULES)):
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

    def test_client_resources_materialize_one_authored_companion(self):
        from binding.python import hem_omp, hem_opencode
        for directory,installer in (('omp_extension',hem_omp),('opencode_plugin',hem_opencode)):
            expected={name.removeprefix(directory+'/') for name in wheel_validation.CLIENT_ASSETS
                      if name.startswith(directory+'/')} | {'native_protocol.ts'}
            assets=installer.bundled_assets()
            self.assertEqual(set(assets),expected)
            for name,data in assets.items():
                source=packaging.REPO_ROOT/'binding/python'
                source=source/name if name=='native_protocol.ts' else source/directory/name
                self.assertEqual(data,source.read_bytes())
            self.assertFalse((packaging.REPO_ROOT/'binding/python'/directory/'native_protocol.ts').exists())

    def test_wheel_requires_each_client_resource_and_exact_reviewed_bytes_once(self):
        for name in wheel_validation.CLIENT_ASSETS:
            with self.subTest(missing=name), self.assertRaisesRegex(RuntimeError,'missing setup assets'):
                validate_wheel(self.wheel(missing='hrx_solutions/'+name))
        for failure in ('changed','duplicate'):
            with self.subTest(failure=failure):
                wheel=self.wheel()
                with zipfile.ZipFile(wheel) as archive:
                    members={name:archive.read(name) for name in archive.namelist()}
                member='hrx_solutions/native_protocol.ts'
                if failure=='changed':members[member]+=b'// changed'
                with zipfile.ZipFile(wheel,'w') as archive:
                    for name,data in members.items():archive.writestr(name,data)
                    if failure=='duplicate':
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore',UserWarning)
                            archive.writestr(member,members[member])
                with self.assertRaisesRegex(RuntimeError,'client asset duplicates or differs'):
                    validate_wheel(wheel)

    def test_sdist_requires_each_exact_regular_client_resource_once(self):
        for failure in (None,'missing','changed','symlink','duplicate'):
            path=self.root/'client-source.tar.gz'
            with tarfile.open(path,'w:gz') as archive:
                for name in wheel_validation.CLIENT_ASSETS:
                    data=(packaging.REPO_ROOT/'binding/python'/name).read_bytes()
                    if name=='native_protocol.ts' and failure=='missing':continue
                    if name=='native_protocol.ts' and failure=='changed':data+=b'// changed'
                    member=tarfile.TarInfo('hrx-solutions/binding/python/'+name);member.size=len(data)
                    if name=='native_protocol.ts' and failure=='symlink':
                        member.type=tarfile.SYMTYPE;member.linkname='/unreviewed/native_protocol.ts'
                    archive.addfile(member,io.BytesIO(data))
                    if name=='native_protocol.ts' and failure=='duplicate':archive.addfile(member,io.BytesIO(data))
            with self.subTest(failure=failure):
                if failure:
                    with self.assertRaisesRegex(RuntimeError,'client asset'):
                        wheel_validation.validate_client_sdist(path)
                else:wheel_validation.validate_client_sdist(path)

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
