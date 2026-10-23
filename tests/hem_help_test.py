"""Discoverable guidance and parser contracts without assets or services."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

from binding.python import hem_cli
from binding.python.cli import hem_help
from binding.python.models import hem_config


class HelpTest(unittest.TestCase):
    def invoke(self, argv):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            try:
                code = hem_cli.main(argv)
            except SystemExit as exit:
                code = exit.code
        return code, output.getvalue(), errors.getvalue()

    def test_guidance_is_identical_and_side_effect_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / 'must-not-exist'
            with patch.dict('os.environ', {'LOOM_SERVE_DIR': str(state)}):
                # The dispatcher is the boundary before all registry, model,
                # download, network and native operations.
                with patch.multiple(hem_cli.hem_model_cli, run=unittest.mock.DEFAULT), \
                     patch.multiple(hem_cli.hem_pipeline_cli, run=unittest.mock.DEFAULT), \
                     patch.multiple(hem_cli.hem_server_cli, run=unittest.mock.DEFAULT), \
                     patch.multiple(hem_cli.hem_instance_cli, run=unittest.mock.DEFAULT), \
                     patch.multiple(hem_cli.hem_pi_cli, run=unittest.mock.DEFAULT), \
                     patch.multiple(hem_cli.hem_opencode_cli, run=unittest.mock.DEFAULT), \
                     patch.multiple(hem_cli.hem_omp_cli, run=unittest.mock.DEFAULT), \
                     patch.multiple(hem_cli.hem_openwebui_cli, run=unittest.mock.DEFAULT):
                    for resource in (None, *hem_cli.DOMAINS):
                        argv = [resource] if resource else []
                        expected = self.invoke(argv + ['--help'])
                        self.assertEqual(expected[0], 0)
                        self.assertEqual(expected[2], '')
                        self.assertIn('usage: loom-control', expected[1])
                        self.assertNotIn('loom-hem', expected[1])
                        for invocation in (argv, ['--json'] + argv, argv + ['--json']):
                            self.assertEqual(self.invoke(invocation), expected)
                    for module in hem_cli.DOMAINS.values():
                        module.run.assert_not_called()
            self.assertFalse(state.exists())

    def test_every_action_and_argument_has_help_and_valid_example(self):
        root = hem_cli.parser()
        domains = next(a for a in root._actions if isinstance(a, argparse._SubParsersAction)).choices
        count = 0
        for resource, domain in domains.items():
            if resource == 'tui':
                continue
            commands = next(a for a in domain._actions if isinstance(a, argparse._SubParsersAction)).choices
            for command, child in commands.items():
                count += 1
                with self.subTest(resource=resource, command=command):
                    self.assertTrue(child.description)
                    for action in child._actions:
                        self.assertTrue(action.help, action.dest)
                        self.assertNotEqual(action.help, argparse.SUPPRESS)
                    for example in hem_help.ACTIONS[resource][command][1].splitlines():
                        words = shlex.split(example)
                        if words[0] == 'loom-control':
                            words = words[1:]
                        args = hem_cli.parse_args(words)
                        self.assertEqual((args.resource, args.command), (resource, command))
                    code, output, errors = self.invoke([resource, command, '--help'])
                    self.assertEqual(code, 0)
                    self.assertIn('Examples:', output)
                    self.assertIn(f'usage: loom-control {resource} {command}', output)
                    self.assertNotIn('loom-hem', output)
                    self.assertEqual(errors, '')
        self.assertEqual(count, 26)

    def test_resource_and_action_lists_have_no_redundant_choice_heading(self):
        _, output, _ = self.invoke([])
        self.assertIn('Resources:', output)
        self.assertIn('RESOURCE ...', output)
        self.assertNotIn('{model,pipeline,server,instance}', output)
        self.assertNotIn('\n  RESOURCE', output)
        for resource, description in hem_help.RESOURCES.items():
            self.assertIn(resource, output)
            self.assertIn(description, ' '.join(output.split()))
            if resource == 'tui':
                _, guidance, _ = self.invoke([resource, '--help'])
                self.assertIn('usage: loom-control tui', guidance)
                self.assertIn('Interactive', guidance)
                continue
            _, guidance, _ = self.invoke([resource])
            self.assertIn('Actions:', guidance)
            self.assertIn('ACTION ...', guidance)
            self.assertNotIn('{', guidance)
            self.assertNotIn('\n  ACTION', guidance)
            normalized = ' '.join(guidance.split())
            for action, (description, _) in hem_help.ACTIONS[resource].items():
                self.assertIn(action, guidance)
                self.assertIn(description, normalized)

    def test_contextual_errors_and_unchanged_json_error_contract(self):
        for argv, context in ((['model', 'import'], 'loom-control model import'),
                              (['server', 'start'], 'loom-control server start'),
                              (['instance', 'list', '--server', 'a', '--url', 'http://localhost:8080'], 'loom-control instance list'),
                              (['pipeline', 'unknown'], 'loom-control pipeline')):
            with self.subTest(argv=argv):
                code, output, errors = self.invoke(argv)
                self.assertEqual(code, 2)
                self.assertEqual(output, '')
                self.assertIn(f'For details: {context} --help', errors)
                self.assertIn('usage:', errors)
                self.assertNotIn('Get started:', errors)
                for position in (0, 1, len(argv)):
                    code, output, errors = self.invoke(argv[:position] + ['--json'] + argv[position:])
                    self.assertEqual(code, 2)
                    response = json.loads(output)
                    self.assertEqual(response['state'], 'failed')
                    self.assertFalse(response['completed'])
                    self.assertNotIn('For details:', errors)

    def test_reference_matches_validation_and_pipeline_metadata(self):
        for command in ('start', 'export'):
            _, output, _ = self.invoke(['instance', command, '--help'])
            for key, (default, maximum, _) in hem_config.SETTINGS.items():
                self.assertIn(f'{key}: integer 1..{maximum}; default {default}', output)
                self.assertEqual(hem_config.LIMITS[key], maximum)
                if key != 'mtp_depth':
                    self.assertEqual(hem_config.DEFAULTS[key], default)
            for term in ('flat JSON', 'Flags override file', 'pipeline defaults',
                         'Duplicate/unknown keys', 'null', 'nonfinite', 'memory fit',
                         'prefill_capacity', 'disabled MTP'):
                self.assertIn(term, output)
        for pipeline in ('qwen', 'qwen-08b'):
            info = hem_config.pipeline_info(pipeline)
            self.assertEqual(info['defaults']['rows'], hem_config.SETTINGS['rows'][0])
            supported = info['capabilities']['mtp']
            self.assertEqual(info['defaults']['mtp_depth'],
                             hem_config.SETTINGS['mtp_depth'][0] if supported else None)
            self.assertEqual(info['limits']['mtp_depth'],
                             hem_config.LIMITS['mtp_depth'] if supported else None)

    def test_download_pipeline_defaults_to_recipe_instead_of_auto_selection(self):
        _, output, _ = self.invoke(['model', 'download', '--help'])
        output = ' '.join(output.split())
        self.assertIn("recipe's declared pipeline", output)
        self.assertNotIn('unique compatible pipeline', output)
        for resource, command in (('model', 'import'), ('model', 'validate'),
                                  ('instance', 'start'), ('instance', 'export')):
            with self.subTest(resource=resource, command=command):
                _, output, _ = self.invoke([resource, command, '--help'])
                output = ' '.join(output.split())
                self.assertIn('unique compatible pipeline', output)
                self.assertNotIn("recipe's declared pipeline", output)


if __name__ == '__main__':
    unittest.main()
