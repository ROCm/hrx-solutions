"""Complete-root integrity and checked emission across selected recipe setup."""
from contextlib import redirect_stderr
from dataclasses import replace
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from binding.python.models import hem_config
from binding.python.hem_common import Deadline, DeadlineExpired, InvalidInput
from binding.python.control import hem_process
from tests.recipe_config_fixture import ConfigurationFixture


class PipelineCatalogTest(ConfigurationFixture, unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.configure()

    def resolved(self, **kwargs):
        return hem_config.resolve(self.recipe, self.model, **kwargs)

    def test_checked_identity_covers_full_authoritative_source_and_installed_roots(self):
        other = self.catalog_root / 'other/policy.loom'
        other.parent.mkdir(); other.write_text('outside selected leaf')
        checked = self.resolved()
        self.assertEqual(checked.catalog_identity.relative_path, Path(self.descriptor.bundle.catalog_relative_path))
        self.assertEqual(checked.catalog_identity.tree_sha256, hem_process.tree_identity(self.catalog_root, Deadline.after()))
        installed = self.root / 'installed/loom-pipeline'
        shutil.copytree(self.catalog_root, installed)
        with patch.object(hem_config.pipelines, 'catalog_roots', return_value=(installed,)):
            repeated = self.resolved()
        self.assertEqual(repeated.catalog_identity, checked.catalog_identity)

    def test_catalog_identity_brackets_descriptor_artifact_and_definition_reads(self):
        for stage in ('descriptor', 'artifacts', 'definition'):
            with self.subTest(stage=stage):
                policy = self.catalog_root / 'outside-policy'
                policy.write_text('before')
                def drift(*args):
                    policy.write_text('after')
                    return self.descriptor if stage == 'descriptor' else {'kind': 'text'} if stage == 'definition' else None
                target, name = ((self.binding, 'describe') if stage == 'descriptor' else
                                (self.binding, 'validate_artifacts') if stage == 'artifacts' else
                                (self.modality, 'service_definition'))
                with patch.object(target, name, side_effect=drift), self.assertRaisesRegex(InvalidInput, 'changed'):
                    self.resolved()

    def test_checked_value_refuses_catalog_without_authoritative_root(self):
        foreign = self.root / 'foreign'; foreign.mkdir()
        with patch.object(hem_config, '_catalog', return_value=foreign), self.assertRaisesRegex(InvalidInput, 'outside'):
            self.resolved()

    def test_root_identity_and_definition_share_original_deadline(self):
        budget = Deadline.after()
        original = hem_process.tree_identity
        with patch.object(hem_process, 'tree_identity', wraps=original) as identity:
            self.resolved(deadline=budget)
        self.assertGreaterEqual(identity.call_count, 2)
        self.assertTrue(all(call.args[1] is budget for call in identity.call_args_list))
        clock = [0.0]; budget = Deadline(1.0, clock=lambda: clock[0])
        def expire(*args): clock[0] = 2.0; return self.descriptor
        with patch.object(self.binding, 'describe', side_effect=expire), self.assertRaises(DeadlineExpired):
            self.resolved(deadline=budget)

    def test_emission_only_rebases_checked_definition_without_reselecting(self):
        checked = self.resolved()
        retained = self.root / 'retained'; shutil.copytree(self.catalog_root, retained)
        selected = retained / checked.catalog_identity.relative_path
        with patch.object(hem_config.pipelines, 'get', side_effect=AssertionError('reselection')), \
             patch.object(self.modality, 'validate', side_effect=AssertionError('revalidation')):
            emitted = hem_config.deployment_document(checked, 'other', catalog=selected, device=2)
        self.assertEqual(emitted['definition'], checked.definition_document())
        self.assertEqual(emitted['name'], 'other')
        self.assertEqual(emitted['base_directory'], str(selected))
        self.assertEqual(emitted['device'], 2)

    def test_retained_complete_root_drift_and_different_program_form_are_refused(self):
        checked = self.resolved()
        for where in ('outside-selected', str(checked.catalog_identity.relative_path / 'program.loom')):
            retained = self.root / 'retained'; shutil.copytree(self.catalog_root, retained)
            target = retained / where; target.write_text('changed')
            with self.assertRaisesRegex(InvalidInput, 'differs'):
                hem_config.deployment_document(checked, 'alias', catalog=retained / checked.catalog_identity.relative_path)
            shutil.rmtree(retained)

    def test_emission_requires_exact_relative_location_before_reading_root(self):
        checked = self.resolved()
        forged = replace(checked, catalog_identity=replace(checked.catalog_identity, relative_path=Path('../bad')))
        with patch.object(hem_process, 'tree_identity', side_effect=AssertionError('read foreign root')), self.assertRaises(InvalidInput):
            hem_config.deployment_document(forged, 'alias', catalog=self.catalog)

    def test_emission_root_reads_do_not_refresh_expired_invocation_budget(self):
        checked = self.resolved(); budget = Deadline(0.0, clock=lambda: 1.0)
        with patch.object(hem_process, 'tree_identity', side_effect=AssertionError('expired budget read')), self.assertRaises(DeadlineExpired):
            hem_config.deployment_document(checked, 'alias', catalog=self.catalog, deadline=budget)

    def test_offline_exports_two_names_without_native_calls_or_named_catalogs(self):
        checked = self.resolved()
        with patch.object(hem_config, 'resolve', return_value=checked), \
             patch.object(hem_config.pipelines, 'get', side_effect=AssertionError('selected binding again')), \
             patch('binding.python.control.hem_state.Registry', side_effect=AssertionError('state')), \
             patch('subprocess.Popen', side_effect=AssertionError('native')), redirect_stderr(io.StringIO()):
            for name in ('first', 'second'):
                result = hem_config.export_deployment(self.recipe, self.model, self.root / (name + '.json'), name)
                self.assertTrue(result['completed'])
