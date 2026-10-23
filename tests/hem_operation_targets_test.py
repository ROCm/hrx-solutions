"""Fresh target admission and same-original read-only consumption contracts."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from binding.python import hem_operation_targets as targets, hem_process, hem_state
from binding.python.models import hem_model_operations as operations
from binding.python.hem_common import Deadline, DeadlineExpired, HemError, InvalidInput, UncertainOutcome
from binding.python.hem_state import Registry


DIGESTS = dict(native_sha256='a' * 64, payload_sha256='b' * 64, catalog_sha256='c' * 64)


def status(operation='transcription', *, incarnation=7):
    profile = operations.operation_profile(operation)
    return dict(object='list', server={}, data=[dict(id='narrator-west',
        input_modalities=list(profile.input_modalities), output_modalities=list(profile.output_modalities),
        loom=dict(kind=profile.native_kind, endpoint=profile.endpoint, incarnation=incarnation,
                  registration='ready', parameter_operation='idle', error=None,
                  parameters={'primary': {'active': True}}, status={'admission': 'open'}))])


class OperationTargetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.deadline = Deadline.after(20)
        self.registry = Registry(Path(self.temp.name) / 'state')
        self.registry.create_server('studio', saved_settings={'listen': '127.0.0.1:18080', 'device': 'cpu'})
        server = self.registry.reserve_launch('studio', launch_token='launch-a', current_settings={},
                                              endpoint='http://127.0.0.1:18080/')
        self.registry.update_server('studio', dict(phase='running', pending=None,
            process=dict(pid=123, boot_id='boot', start_ticks=42),
            build=dict(DIGESTS, directory='/retained', executable=dict(device=1, inode=2))),
            expected_token='launch-a', expected_revision=server['revision'])
        self.registry.create_instance('studio', 'launch-a', 'narrator-west', incarnation=7)
        self.document = status()
        self.manager = Mock(package_root=Path('/package'))
        self.manager.pin.side_effect = lambda record: self.pin(record)
        self.manager.client.return_value.status.side_effect = lambda **kwargs: deepcopy(self.document)
        self.pins = []
        self.factory = self.enter(patch.object(targets, 'ProcessManager', return_value=self.manager))
        self.build = self.enter(patch.object(targets, 'require_build', return_value=Path('/retained/loom-pipeline')))

    def enter(self, context):
        value = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return value

    @contextmanager
    def pin(self, record):
        self.pins.append(deepcopy(record))
        yield Mock()

    def resolve(self, operation='transcription'):
        return targets.resolve_operation_target(self.registry, 'studio', 'narrator-west', operation, self.deadline)

    def assert_failure_reason(self, reason, callback=None):
        with self.assertRaises(HemError) as caught:
            (callback or self.resolve)()
        self.assertEqual(caught.exception.reason, reason)
        if reason == 'unknown':
            self.assertIsInstance(caught.exception, UncertainOutcome)
        return caught.exception

    def update_server(self, **changes):
        row = self.registry.get_server('studio')
        self.registry.update_server('studio', changes, expected_token=row['launch_token'], expected_revision=row['revision'])

    def test_profiles_are_inert_and_finite(self):
        code = "from binding.python.models import hem_model_operations; import sys; assert not any(name.endswith(('hem_state', 'hem_config', 'hem_models', 'hem_openwebui')) for name in sys.modules)"
        subprocess.run([sys.executable, '-c', code], check=True)
        with self.assertRaises(ValueError):operations.operation_profile('synthesis')
        with self.assertRaises(FrozenInstanceError):operations.CHAT.endpoint = '/elsewhere'

    def test_current_operations_keep_exact_identity_and_original_deadline(self):
        for operation in ('chat', 'transcription'):
            with self.subTest(operation=operation):
                self.document = status(operation)
                with patch.object(self.registry, 'lock', wraps=self.registry.lock) as lock:
                    target = self.resolve(operation)
                lock.assert_called_once_with('server', 'studio', deadline=self.deadline)
                self.assertEqual((target.server_id, target.instance_name, target.origin, target.launch_token,
                                  target.incarnation), ('studio', 'narrator-west', 'http://127.0.0.1:18080', 'launch-a', 7))
                self.assertEqual(target.profile, operations.operation_profile(operation))
                self.assertEqual(target.build_identity, tuple(DIGESTS.items()))
                self.assertEqual((target.registration, target.parameter_operation, target.primary_active,
                                  target.error, target.admission), ('ready', 'idle', True, None, 'open'))
                self.assertIsNone(target.tracked_profile_identity)
                self.assertIsNone(self.registry.deadline)
                self.assertIs(self.manager.client.call_args.args[2], self.deadline)
                self.assertIs(self.build.call_args.args[2], self.deadline)
                with self.assertRaises(FrozenInstanceError):target.incarnation = 8

    def test_revalidation_is_same_original_without_lock_reconcile_or_registry_creation(self):
        target = self.resolve()
        with patch.object(self.registry, 'lock', side_effect=AssertionError('no ranked lock')), \
                patch.object(self.manager, 'observe_and_reconcile', side_effect=AssertionError('no reconcile')), \
                patch.object(self.registry, 'update_server', side_effect=AssertionError('no writes')), \
                patch.object(self.registry, 'get_model', side_effect=AssertionError('no model reads')), \
                patch.object(hem_state, 'Registry', side_effect=AssertionError('no Registry construction')):
            self.assertIsNone(targets.revalidate_operation_target(self.registry, target, self.deadline))
        self.assertEqual(len(self.pins), 2)
        self.assertEqual(self.pins[0], self.pins[1])
        self.assertIsNone(self.registry.deadline)

    def test_invalid_selectors_and_operation_fail_before_native_observation(self):
        for args in (('bad/name', 'narrator-west', 'chat'), ('studio', '../name', 'chat'),
                     ('studio', 'narrator-west', 'synthesis')):
            with self.assertRaises(InvalidInput):
                targets.resolve_operation_target(self.registry, *args, self.deadline)
        self.factory.assert_not_called()

    def test_confirmed_missing_stopped_and_unresolved_are_distinct(self):
        self.assert_failure_reason('missing', lambda: targets.resolve_operation_target(self.registry, 'absent', 'name', 'chat', self.deadline))
        self.assert_failure_reason('missing', lambda: targets.resolve_operation_target(self.registry, 'studio', 'absent', 'chat', self.deadline))
        self.update_server(phase='unknown')
        self.assert_failure_reason('unknown')
        self.update_server(phase='stopped')  # A leftover launch is unresolved, not stopped proof.
        self.assert_failure_reason('unknown')
        row = self.registry.get_server('studio')
        self.registry.clear_server('studio', expected_token=row['launch_token'], expected_revision=row['revision'])
        self.assert_failure_reason('unavailable')
        self.manager.client.assert_not_called()

    def test_pending_and_missing_incarnation_are_unknown(self):
        for changes in ({'pending': {'action': 'start'}}, {'pending': None, 'incarnation': None}):
            row = self.registry.get_instance('studio', 'launch-a', 'narrator-west')
            self.registry.update_instance('studio', 'launch-a', 'narrator-west', changes, expected_revision=row['revision'])
            self.assert_failure_reason('unknown')
        self.manager.client.assert_not_called()

    def test_exact_profile_mismatch_is_incompatible_and_malformed_is_unknown(self):
        for owner, key, value in (('loom', 'kind', 'speech'), ('loom', 'endpoint', '/v1/audio/speech'),
                                  ('row', 'input_modalities', ['text']), ('row', 'output_modalities', ['audio'])):
            with self.subTest(key=key):
                self.document = status()
                row = self.document['data'][0]
                (row['loom'] if owner == 'loom' else row)[key] = value
                self.assert_failure_reason('incompatible')
        self.document = status()
        del self.document['data'][0]['input_modalities']
        self.assert_failure_reason('unknown')

    def test_native_readiness_requires_primary_ready_idle_no_error_and_open(self):
        for changes in ({'registration': 'loading'}, {'parameter_operation': 'loading'}, {'error': 'failed'},
                        {'parameters': {'primary': {'active': False}, 'image': {'active': True}}},
                        {'status': {'admission': 'drained'}}):
            with self.subTest(changes=changes):
                self.document = status()
                self.document['data'][0]['loom'].update(changes)
                self.assert_failure_reason('unavailable')
        self.document = status()
        del self.document['data'][0]['loom']['parameters']
        self.assert_failure_reason('unknown')

    def test_missing_and_replaced_native_incarnation_never_inherit_proof(self):
        target = self.resolve()
        self.document = dict(object='list', server={}, data=[])
        self.assert_failure_reason('unavailable')
        self.assert_failure_reason('unknown', lambda: targets.revalidate_operation_target(self.registry, target, self.deadline))
        self.document = status(incarnation=8)
        self.assert_failure_reason('unknown')
        self.assert_failure_reason('unknown', lambda: targets.revalidate_operation_target(self.registry, target, self.deadline))

    def test_unknown_socket_http_and_malformed_status_stay_unknown(self):
        for error in (UncertainOutcome('socket ownership unavailable'), HemError('refused', reason='connection_refused')):
            with self.subTest(error=error):
                self.manager.client.return_value.status.side_effect = error
                self.assert_failure_reason('unknown')
        self.manager.client.return_value.status.side_effect = lambda **kwargs: {'object': 'list', 'data': []}
        self.assert_failure_reason('unknown')

    def test_build_mismatch_requires_complete_typed_recorded_digests(self):
        self.build.side_effect = HemError('build differs', reason='build_mismatch')
        self.assert_failure_reason('incompatible')
        original = deepcopy(self.registry.get_server('studio')['build'])
        for key in DIGESTS:
            for value in (None, 3, 'invalid', 'a' * 63):
                with self.subTest(key=key, value=value):
                    build = dict(original)
                    if value is None:del build[key]
                    else:build[key] = value
                    self.update_server(build=build)
                    self.build.reset_mock()
                    self.assert_failure_reason('unknown')
                    self.build.assert_not_called()  # No duplicate hashing/comparison.
        self.manager.client.assert_not_called()

    def test_unreadable_build_identity_is_unknown(self):
        for error in (HemError('cannot read payload'), OSError('unreadable')):
            self.build.side_effect = error
            self.assert_failure_reason('unknown')
        self.manager.client.assert_not_called()

    def test_record_change_during_status_refuses_initial_and_revalidation(self):
        for initial in (True, False):
            with self.subTest(initial=initial):
                self.manager.client.return_value.status.side_effect = lambda **kwargs: deepcopy(self.document)
                self.document = status(incarnation=self.registry.get_instance('studio', 'launch-a', 'narrator-west')['incarnation'])
                target = None if initial else self.resolve()
                def observe(**kwargs):
                    row = self.registry.get_instance('studio', 'launch-a', 'narrator-west')
                    self.registry.update_instance('studio', 'launch-a', 'narrator-west',
                        {'incarnation': row['incarnation'] + 1}, expected_revision=row['revision'])
                    return deepcopy(self.document)
                self.document = status(incarnation=self.registry.get_instance('studio', 'launch-a', 'narrator-west')['incarnation'])
                self.manager.client.return_value.status.side_effect = observe
                self.assert_failure_reason('unknown', self.resolve if initial else lambda: targets.revalidate_operation_target(self.registry, target, self.deadline))

    def assert_admission_failure_rereads_identity(self, *, initial, mismatch):
        # Unchanged identity retains the native category; replacing the tracked
        # incarnation during the same response must instead remain unknown.
        for changed in (False, True):
            self.manager.client.return_value.status.side_effect = lambda **kwargs: deepcopy(self.document)
            target = None if initial else self.resolve()
            def observe(**kwargs):
                self.assertIs(self.registry.deadline, self.deadline)
                if changed:
                    row = self.registry.get_instance('studio', 'launch-a', 'narrator-west')
                    self.registry.update_instance('studio', 'launch-a', 'narrator-west',
                        {'incarnation': 8}, expected_revision=row['revision'])
                document = deepcopy(self.document)
                document['data'][0]['loom'].update(mismatch)
                return document
            self.manager.client.return_value.status.side_effect = observe
            reason = 'unknown' if changed else ('unavailable' if 'registration' in mismatch else 'incompatible')
            callback = self.resolve if initial else lambda: targets.revalidate_operation_target(self.registry, target, self.deadline)
            self.assert_failure_reason(reason, callback)
            self.assertIsNone(self.registry.deadline)

    def test_initial_notready_failure_rereads_tracked_identity(self):
        self.assert_admission_failure_rereads_identity(initial=True, mismatch={'registration': 'loading'})

    def test_initial_profile_failure_rereads_tracked_identity(self):
        self.assert_admission_failure_rereads_identity(initial=True, mismatch={'kind': 'speech'})

    def test_revalidation_notready_failure_rereads_tracked_identity(self):
        self.assert_admission_failure_rereads_identity(initial=False, mismatch={'registration': 'loading'})

    def test_revalidation_profile_failure_rereads_tracked_identity(self):
        self.assert_admission_failure_rereads_identity(initial=False, mismatch={'kind': 'speech'})

    def test_failure_reread_keeps_original_deadline_expiry(self):
        for initial in (True, False):
            clock = [0]
            self.deadline = Deadline(10, clock=lambda: clock[0])
            self.manager.client.return_value.status.side_effect = lambda **kwargs: deepcopy(self.document)
            target = None if initial else self.resolve()
            def expired_status(**kwargs):
                self.assertIs(self.registry.deadline, self.deadline)
                clock[0] = 11
                document = deepcopy(self.document)
                document['data'][0]['loom']['registration'] = 'loading'
                return document
            self.manager.client.return_value.status.side_effect = expired_status
            callback = self.resolve if initial else lambda: targets.revalidate_operation_target(self.registry, target, self.deadline)
            with self.assertRaises(DeadlineExpired):callback()
            self.assertIsNone(self.registry.deadline)

    def test_changed_server_identity_or_revision_refuses_before_new_io(self):
        changes = ({'endpoint': 'http://127.0.0.1:18081'},
                   {'process': dict(pid=456, boot_id='other', start_ticks=77)},
                   {'phase': 'running'})  # Even a revision-only write refuses.
        for change in changes:
            with self.subTest(change=change):
                target = self.resolve()
                self.update_server(**change)
                calls = self.manager.client.call_count
                self.assert_failure_reason('unknown', lambda: targets.revalidate_operation_target(self.registry, target, self.deadline))
                self.assertEqual(self.manager.client.call_count, calls)

    def test_new_launch_same_name_and_incarnation_cannot_revalidate(self):
        target = self.resolve()
        server = self.registry.get_server('studio')
        self.registry.clear_server('studio', expected_token=server['launch_token'], expected_revision=server['revision'])
        server = self.registry.reserve_launch('studio', launch_token='launch-b', current_settings={},
                                              endpoint=target.origin)
        self.registry.update_server('studio', dict(phase='running', pending=None,
            process=self.pins[0]['process'], build=self.pins[0]['build']),
            expected_token='launch-b', expected_revision=server['revision'])
        self.registry.create_instance('studio', 'launch-b', target.instance_name, incarnation=target.incarnation)
        calls = self.manager.client.call_count
        self.assert_failure_reason('unknown', lambda: targets.revalidate_operation_target(self.registry, target, self.deadline))
        self.assertEqual(self.manager.client.call_count, calls)

    def test_server_revision_changed_during_status_is_unknown(self):
        def observe(**kwargs):
            self.assertIs(self.registry.deadline, self.deadline)
            self.update_server(phase='running')
            return deepcopy(self.document)
        self.manager.client.return_value.status.side_effect = observe
        self.assert_failure_reason('unknown')

    def test_ambiguous_rows_or_invalid_origin_are_unknown(self):
        rows = self.registry.list_instances(server_id='studio')
        with patch.object(self.registry, 'list_instances', return_value=rows + rows):
            self.assert_failure_reason('unknown')
        self.update_server(endpoint='http://example.com:18080')
        self.assert_failure_reason('unknown')
        self.manager.client.assert_not_called()

    def test_corrupt_registry_is_unknown_not_missing(self):
        with patch.object(self.registry, 'list_servers', side_effect=HemError('corrupt row')):
            self.assert_failure_reason('unknown')
        self.manager.client.assert_not_called()

    def test_server_lock_failure_is_unknown_without_observation(self):
        for error in (HemError('lock ownership unavailable'), OSError('lock unreadable')):
            with patch.object(self.registry, 'lock', side_effect=error):
                self.assert_failure_reason('unknown')
            self.assertIsNone(self.registry.deadline)
        self.manager.client.assert_not_called()

    def test_deadline_not_restarted_and_registry_budget_restored(self):
        self.deadline = Deadline(0, clock=lambda: 1)
        with self.assertRaises(DeadlineExpired):self.resolve()
        self.assertIsNone(self.registry.deadline)
        self.manager.client.assert_not_called()


class BuildReasonTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.current = Path(temporary.name) / 'current'
        self.retained = Path(temporary.name) / 'retained'
        (self.current / '_native/bin').mkdir(parents=True)
        (self.current / 'loom-pipeline').mkdir()
        # Identity hashing needs files, not an executable process or compiler.
        (self.current / '_native/bin/loom-serve').write_bytes(b'identity-only fixture')
        (self.current / 'loom-pipeline/profile.json').write_text('{}')
        shutil.copytree(self.current, self.retained)
        self.deadline = Deadline.after()
        self.record = {'build': dict(hem_process.package_identity(self.current, self.deadline), directory=str(self.retained))}

    def test_only_existing_digest_comparison_tags_mismatch(self):
        self.assertEqual(hem_process.require_build(self.record, self.current, self.deadline), self.retained / 'loom-pipeline')
        (self.retained / 'loom-pipeline/profile.json').write_text('{"changed": true}')
        with self.assertRaises(HemError) as caught:
            hem_process.require_build(self.record, self.current, self.deadline)
        error = caught.exception
        self.assertIs(type(error), HemError)
        self.assertEqual(error.args, ('serving build/catalog differs; shut down and launch a new empty server',))
        self.assertEqual((error.reason, error.exit_code, error.accepted), ('build_mismatch', 1, None))
        with self.assertRaises(UncertainOutcome) as caught:hem_process.require_build({}, self.current, self.deadline)
        self.assertIsNone(caught.exception.reason)

    def test_readable_payloads_with_bad_record_can_tag_but_core_refuses_unknown(self):
        for value in (None, 'malformed'):
            build = dict(self.record['build'])
            if value is None:del build['catalog_sha256']
            else:build['catalog_sha256'] = value
            with self.assertRaises(HemError) as caught:
                hem_process.require_build({'build': build}, self.current, self.deadline)
            self.assertEqual(caught.exception.reason, 'build_mismatch')
            with patch.object(targets, 'require_build', wraps=hem_process.require_build) as require:
                with self.assertRaises(UncertainOutcome) as unknown:
                    targets._build({'build': build}, Mock(package_root=self.current), self.deadline)
            self.assertEqual(unknown.exception.reason, 'unknown')
            require.assert_not_called()

    def test_unreadable_payload_and_deadline_leave_reason_unset(self):
        (self.retained / '_native/bin/loom-serve').unlink()
        with self.assertRaises(HemError) as caught:
            hem_process.require_build(self.record, self.current, self.deadline)
        self.assertIsNone(caught.exception.reason)
        with self.assertRaises(UncertainOutcome) as unknown:
            targets._build(self.record, Mock(package_root=self.current), self.deadline)
        self.assertEqual(unknown.exception.reason, 'unknown')
        with self.assertRaises(DeadlineExpired) as caught:
            hem_process.require_build(self.record, self.current, Deadline(0, clock=lambda: 1))
        self.assertIsNone(caught.exception.reason)


if __name__ == '__main__':unittest.main()
