"""Generic compiler closure proof; decoder fixtures do not qualify native code."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from build_tools.model_tools import loom_catalog as compiler


class CompiledProgramTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'novel/runtime/program.loom'
        self.source.parent.mkdir(parents=True)
        self.original = ('func.def public @entry()\nfunc.def private @internal()\n'
                         'command.program.def public @step()\nconfig.def @capacity = 8\n')
        self.source.write_text(self.original)
        self.artifact = self.root / 'module.loombc'
        self.artifact.write_bytes(b'LOOMfixture')
        self.decoded = self.original

    def validate(self, **kwargs):
        def decode(arguments, **options):
            self.assertTrue(options['check'])
            Path(arguments[-1].removeprefix('--output=')).write_text(self.decoded)
        with patch.object(compiler.subprocess, 'run', side_effect=decode):
            return compiler.validate_compiled_program(self.artifact,
                ['novel/runtime/program.loom'], self.root, Path('fixture-decoder'), **kwargs)

    def test_public_command_private_and_config_closure_observations(self):
        proof = self.validate()
        expected = {'functions': ['entry'], 'commands': ['step'], 'configs': ['capacity']}
        self.assertEqual(proof, {'source_roots': expected, 'compiled_roots': expected,
                                 'source_function_count': 2, 'compiled_function_count': 2})

    def test_missing_added_and_changed_roots_and_private_count_rejected(self):
        for changed in (
            self.original.replace('public @entry', 'public @other'),
            self.original.replace('command.program.def public @step()\n', ''),
            self.original.replace('private @internal', 'public @internal'),
            self.original.replace('func.def private @internal()\n', ''),
            self.original.replace('config.def @capacity = 8\n', ''),
            self.original + 'func.def private @extra()\n',
            self.original + 'config.def @extra = 1\n',
        ):
            self.decoded = changed
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'closure'):
                self.validate()

    def test_existing_compiler_generated_config_tolerance_is_explicit(self):
        self.decoded += 'config.def @compiler_generated = 1\n'
        proof = self.validate(exact_configs=False)
        self.assertIn('compiler_generated', proof['compiled_roots']['configs'])
        self.decoded = self.decoded.replace('config.def @capacity = 8\n', '')
        with self.assertRaisesRegex(ValueError, 'closure'):
            self.validate(exact_configs=False)

    def test_policy_projection_requires_exact_old_and_new_declaration(self):
        old = 'global.rodata.def @policy = bytes("' + b'support.loom'.hex() + '")\n'
        new = 'global.rodata.def @policy = bytes("' + b'support.loombc'.hex() + '")\n'
        self.source.write_text(self.original + old)
        self.decoded = self.original + new
        self.validate(policy=('support.loom', 'support.loombc'))
        for changed in (self.original + old, self.original + new * 2, self.original):
            self.decoded = changed
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'policy'):
                self.validate(policy=('support.loom', 'support.loombc'))

    def test_missing_duplicate_escape_and_wrong_encoding_rejected(self):
        for sources in ([], ['novel/runtime/program.loom'] * 2, ['../outside.loom'], ['/outside.loom']):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                compiler.validate_compiled_program(self.artifact, sources, self.root, Path('decoder'))
        self.artifact.write_bytes(b'not bytecode')
        with self.assertRaisesRegex(ValueError, 'compiled program'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
