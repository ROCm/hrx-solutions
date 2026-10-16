"""Public resource grammar, passive checks and client-independent removal."""
from contextlib import redirect_stdout,redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from binding.python import hem_cli,hem_omp_cli as cli
from binding.python.hem_common import InvalidInput
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
        with patch.object(cli,'inspect_runtime',return_value={'qualification':'test metadata'}):
            result,errors=self.invoke();self.assertEqual(result['state'],'configured');self.assertIn('No native Loom equivalent',errors)
            self.assertTrue((self.root/'clients/endpoints.json').exists())
            self.assertFalse(self.invoke()[0]['changed']);before=self.contents()
            self.invoke('--check');self.assertEqual(before,self.contents())
        with patch.object(cli,'inspect_runtime',side_effect=AssertionError('disable must not inspect client')):
            self.invoke('--disable');self.assertTrue((self.root/'clients/endpoints.json').exists())
        self.assertFalse(list(self.root.rglob('*.sock')))

    def test_missing_client_check_separate_findings_no_writes(self):
        with patch.object(cli,'inspect_runtime',side_effect=InvalidInput('client absent')),patch('subprocess.run',side_effect=AssertionError('passive check')):
            result,_=self.invoke('--check',expected=1)
        self.assertEqual(result['state'],'absent');self.assertEqual(result['runtime_compatibility']['state'],'unqualified')
        self.assertFalse(self.root.exists());self.assertFalse(self.target.exists())

    def test_unsupported_client_configure_no_writes(self):
        with patch.object(cli,'inspect_runtime',side_effect=InvalidInput('unsupported')):self.invoke(expected=2)
        self.assertFalse(self.root.exists());self.assertFalse(self.target.exists())

    def test_profile_precedence_and_explicit_conflict(self):
        with patch.dict(os.environ,{'OMP_PROFILE':'selected','PI_PROFILE':'ignored'}), patch.object(Path,'home',return_value=self.base):
            self.assertEqual(cli.resolve_target(),self.base/'.omp/profiles/selected/agent')
            with self.assertRaises(InvalidInput):cli.resolve_target(self.target)
        with patch.dict(os.environ,{'OMP_PROFILE':'','PI_PROFILE':'ignored'}):self.assertEqual(cli.resolve_target(),self.target)
        with patch.dict(os.environ,{'OMP_PROFILE':'../unsafe'}):
            with self.assertRaises(InvalidInput):cli.resolve_target()
        with self.assertRaises(InvalidInput):cli.resolve_target(Path('relative'))

    def test_profile_normalization_cases(self):
        cases={'':'',' ':'','default':'',' default ':'','selected':'selected',' abc ':'abc',
               'a'*64:'a'*64,'a'*65:None,'Upper':None,'con':None,'nul.txt':None,'com0':None,
               '.':None,'..':None,'abc.':None,'a/b':None,'default2':'default2'}
        for value,expected in cases.items():
            with self.subTest(value=value):
                if expected is None:
                    with self.assertRaises(InvalidInput):cli.normalize_profile(value)
                else:self.assertEqual(cli.normalize_profile(value),expected)
        with patch.dict(os.environ,{'OMP_PROFILE':' default ','PI_PROFILE':'named'}):
            self.assertEqual(cli.resolve_target(),self.target)
            self.assertEqual(cli.resolve_target(self.base/'explicit'),self.base/'explicit')

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
            runtime=cli.inspect_runtime(__import__('binding.python.hem_common',fromlist=['Deadline']).Deadline.after(),passive=True)
        self.assertEqual(runtime['version'],'18.11.2')


if __name__=='__main__':unittest.main()
