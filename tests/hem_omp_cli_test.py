"""Public resource grammar, passive checks and client-independent removal."""
from contextlib import redirect_stdout,redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from binding.python import hem_cli
from binding.python.cli import hem_omp_cli as cli
from binding.python.integrations.omp import hem_omp as owner
from binding.python.hem_common import Deadline, InvalidInput
from binding.python.hem_state import Registry


class OMPCliTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.target=self.base/'agent';self.root=self.base/'state'
        self.environment=dict(LOOM_SERVE_DIR=str(self.root),PI_CODING_AGENT_DIR=str(self.target),OMP_PROFILE='',PI_PROFILE='')
        self.addCleanup(patch.stopall);patch.dict(os.environ,self.environment).start()

    def invoke(self,*flags,expected=0):
        output,errors=io.StringIO(),io.StringIO()
        with redirect_stdout(output),redirect_stderr(errors):
            code=hem_cli.main(['omp','configure','--json',*flags])
        self.assertEqual(code,expected,output.getvalue()+errors.getvalue())
        return json.loads(output.getvalue()),errors.getvalue()

    def contents(self):
        return {str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in self.base.rglob('*') if p.is_file()}

    def test_configure_repeat_check_and_absent_client_disable(self):
        with patch.object(owner,'inspect_runtime',return_value={'qualification':'test metadata'}):
            result,errors=self.invoke();self.assertEqual(result['state'],'configured');self.assertIn('No native Loom equivalent',errors)
            self.assertTrue((self.root/'clients/endpoints.json').exists())
            self.assertFalse(self.invoke()[0]['changed']);before=self.contents()
            self.invoke('--check');self.assertEqual(before,self.contents())
        with patch.object(owner,'inspect_runtime',side_effect=AssertionError('disable must not inspect client')):
            self.invoke('--disable');self.assertTrue((self.root/'clients/endpoints.json').exists())
        self.assertFalse(list(self.root.rglob('*.sock')))

    def test_explicit_configure_refreshes_prior_owned_projection_after_passive_check(self):
        from binding.python.integrations.pi.hem_pi_contracts import NATIVE_PROFILE, PREVIOUS_NATIVE_PROFILE
        Registry(self.root).create_server('local', saved_settings=dict(device='task', listen='127.0.0.1:18080'))
        with patch.object(owner, 'inspect_runtime', return_value={'qualification': 'test metadata'}):
            self.invoke()
            path = self.root / 'clients/endpoints.json'
            prior = json.loads(path.read_bytes())
            prior['servers'][0]['native_profile'] = PREVIOUS_NATIVE_PROFILE
            path.write_text(json.dumps(prior))
            before = self.contents()
            self.invoke('--check', expected=1)
            self.assertEqual(self.contents(), before)
            self.invoke()
            self.assertEqual(json.loads(path.read_bytes())['servers'][0]['native_profile'], NATIVE_PROFILE)

    def test_missing_client_check_separate_findings_no_writes(self):
        with patch.object(owner,'inspect_runtime',side_effect=InvalidInput('client absent')),patch('subprocess.run',side_effect=AssertionError('passive check')):
            result,_=self.invoke('--check',expected=1)
        self.assertEqual(result['state'],'absent');self.assertEqual(result['runtime_compatibility']['state'],'unqualified')
        self.assertFalse(self.root.exists());self.assertFalse(self.target.exists())

    def test_unsupported_client_configure_no_writes(self):
        with patch.object(owner,'inspect_runtime',side_effect=InvalidInput('unsupported')):self.invoke(expected=2)
        self.assertFalse(self.root.exists());self.assertFalse(self.target.exists())

    def test_profile_precedence_and_explicit_conflict(self):
        with patch.dict(os.environ,{'OMP_PROFILE':'selected','PI_PROFILE':'ignored'}), patch.object(Path,'home',return_value=self.base):
            self.assertEqual(owner.resolve_target(),self.base/'.omp/profiles/selected/agent')
            with self.assertRaises(InvalidInput):owner.resolve_target(self.target)
        with patch.dict(os.environ,{'OMP_PROFILE':'','PI_PROFILE':'ignored'}):self.assertEqual(owner.resolve_target(),self.target)
        with patch.dict(os.environ,{'OMP_PROFILE':'../unsafe'}):
            with self.assertRaises(InvalidInput):owner.resolve_target()
        with self.assertRaises(InvalidInput):owner.resolve_target(Path('relative'))

    def test_profile_normalization_cases(self):
        cases={'':'',' ':'','default':'',' default ':'','selected':'selected',' abc ':'abc',
               'a'*64:'a'*64,'a'*65:None,'Upper':None,'con':None,'nul.txt':None,'com0':None,
               '.':None,'..':None,'abc.':None,'a/b':None,'default2':'default2'}
        for value,expected in cases.items():
            with self.subTest(value=value):
                if expected is None:
                    with self.assertRaises(InvalidInput):owner.normalize_profile(value)
                else:self.assertEqual(owner.normalize_profile(value),expected)
        with patch.dict(os.environ,{'OMP_PROFILE':' default ','PI_PROFILE':'named'}):
            self.assertEqual(owner.resolve_target(),self.target)
            self.assertEqual(owner.resolve_target(self.base/'explicit'),self.base/'explicit')

    def test_direct_entry_qualifies_and_check_disable_remain_safe(self):
        with patch.object(owner,'inspect_runtime',side_effect=InvalidInput('unsupported')):
            with self.assertRaisesRegex(InvalidInput,'unsupported'):
                owner.configure_omp(self.root,self.target,Deadline.after())
            checked=owner.configure_omp(self.root,self.target,Deadline.after(),check=True)
        self.assertEqual(checked['runtime_compatibility']['state'],'unqualified')
        self.assertFalse(self.root.exists() or self.target.exists())
        with patch.object(owner,'inspect_runtime',side_effect=AssertionError('disable probe')):
            disabled=owner.configure_omp(self.root,self.target,Deadline.after(),disable=True)
        self.assertEqual(disabled['state'],'disabled')
        self.assertFalse(self.root.exists() or self.target.exists())
        with patch.object(owner,'inspect_runtime',return_value={'qualification':'direct test'}):
            configured=owner.configure_omp(self.root,self.target,Deadline.after())
        self.assertTrue(configured['configured'])
        self.assertEqual(configured['runtime_compatibility']['state'],'compatible')
        self.assertTrue((self.root/'clients/endpoints.json').exists())

    def test_known_yaml_conflict_precedes_root_and_endpoint_publication(self):
        self.target.mkdir(mode=0o700)
        yaml=self.target/'models.yml'
        yaml.write_text('providers:\n  local: {auth: none}\n')
        before=self.contents()
        with patch.object(owner,'inspect_runtime',return_value={'qualification':'test'}), patch.object(owner,'Registry',side_effect=AssertionError('must preflight before Registry')):
            self.invoke(expected=2)
            with self.assertRaises(InvalidInput):
                owner.configure_omp(self.root,self.target,Deadline.after())
        self.assertEqual(before,self.contents())
        self.assertFalse(self.root.exists())

    def test_one_public_operation_and_asset_read_with_fresh_locked_preparation(self):
        with patch.object(owner,'inspect_runtime',return_value={'qualification':'test'}), patch.object(owner,'configure_omp',wraps=owner.configure_omp) as operation, patch.object(owner,'bundled_assets',wraps=owner.bundled_assets) as assets, patch.object(owner,'_prepare',wraps=owner._prepare) as prepare:
            self.invoke()
        operation.assert_called_once()
        self.assertFalse(operation.call_args.kwargs['check'])
        assets.assert_called_once_with()
        # Initial private conflict preflight and the fresh locked reread remain;
        # there is no intervening public check operation or third preparation.
        self.assertEqual(prepare.call_count,2)

    def test_conflict_after_preflight_is_refused_by_fresh_locked_reread(self):
        publish=owner.ensure_endpoints
        def publish_then_edit(registry,deadline):
            answer=publish(registry,deadline)
            self.target.mkdir(mode=0o700)
            (self.target/'models.yml').write_text('providers:\n  local: {auth: none}\n')
            return answer
        with patch.object(owner,'inspect_runtime',return_value={'qualification':'test'}), patch.object(owner,'ensure_endpoints',side_effect=publish_then_edit):
            with self.assertRaises(InvalidInput):
                owner.configure_omp(self.root,self.target,Deadline.after())
        self.assertFalse((self.target/owner.PLUGIN).exists())
        self.assertEqual((self.target/'models.yml').read_text(),'providers:\n  local: {auth: none}\n')

    def test_fixed_dispatch_keeps_explicit_omp_namespace_and_deadline(self):
        from binding.python.integrations.hem_integrations import configure_integration
        deadline=Deadline.after()
        for mode in ('configure','check','disable'):
            with patch.object(owner,'configure_omp',return_value={}) as operation:
                configure_integration('omp',self.root,deadline,mode=mode,omp_dir=self.target)
                operation.assert_called_once_with(self.root,self.target,deadline,
                    check=mode=='check',disable=mode=='disable')

    def test_grammar_rejects_mode_conflict_and_abbreviations(self):
        self.invoke('--check','--disable',expected=2);self.invoke('--chec',expected=2)
        for invocation in (['omp'],['omp','--help'],['omp','configure','--help']):
            with redirect_stdout(io.StringIO()):
                try:code=hem_cli.main(invocation)
                except SystemExit as error:code=error.code
            self.assertEqual(code,0)

    def test_passive_runtime_no_execution_real_qualified_package_when_supplied(self):
        package=os.environ.get('LOOM_OMP_TEST_PACKAGE');bun=os.environ.get('LOOM_OMP_TEST_BUN')
        if not package or not bun:self.skipTest('qualified external npm/Bun fixture required')
        bindir=self.base/'bin';bindir.mkdir(mode=0o700)
        metadata=json.loads((Path(package)/'package.json').read_text())
        (bindir/'omp').symlink_to(Path(package)/metadata['bin']['omp']);(bindir/'bun').symlink_to(bun)
        with patch.dict(os.environ,{'PATH':str(bindir)}),patch('subprocess.run',side_effect=AssertionError('passive')):
            runtime=owner.inspect_runtime(__import__('binding.python.hem_common',fromlist=['Deadline']).Deadline.after(),passive=True)
        self.assertEqual(runtime['version'],'18.11.2')


if __name__=='__main__':unittest.main()
