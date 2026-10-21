"""Pure ownership tests: secret preservation, minimal crash intent and no adoption."""
from copy import deepcopy
import importlib
import json
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

PACKAGE = os.environ.get('HEM_TEST_PACKAGE', 'binding.python')
state = importlib.import_module(PACKAGE + '.hem_openwebui_state')
common = importlib.import_module(PACKAGE + '.hem_common')
contracts = importlib.import_module(PACKAGE + '.hem_pi_contracts')
files = importlib.import_module(PACKAGE + '.hem_pi_files')

ROOT = Path('/nonexistent/owned-root')
TARGET = 'http://127.0.0.1:3000'
SECRET = 'foreign-secret-never-persist'


def row(server='main', port=18080):
    provider = contracts.provider_id(server)
    return dict(server=server, provider=provider, base_url=f'http://127.0.0.1:{port}/v1',
                config=dict(enable=True, auth_type='none', prefix_id=provider,
                            model_ids=[], connection_type='local'))


def remote(rows=(), *, foreign=False, enabled=True):
    rows = list(rows)
    value = dict(ENABLE_OPENAI_API=enabled, OPENAI_API_BASE_URLS=[r['base_url'] for r in rows],
                 OPENAI_API_KEYS=['' for _ in rows],
                 OPENAI_API_CONFIGS={str(i): deepcopy(r['config']) for i, r in enumerate(rows)})
    if foreign:
        value['OPENAI_API_BASE_URLS'].append('https://unrelated.example/custom')
        value['OPENAI_API_KEYS'].append(SECRET)
        value['OPENAI_API_CONFIGS'][str(len(rows))] = dict(prefix_id='foreign', headers={'unknown': SECRET}, custom=[1, None])
    return value


def document(rows=(), identity=''):
    result = state.decode_document(None, ROOT, TARGET)
    result.update(connections=sorted(deepcopy(list(rows)), key=lambda r: r['provider']), deployment_id=identity)
    return result


def plan(current, owned, wanted, mode='configure', identity=''):
    return state.plan_change(current, owned, sorted(deepcopy(wanted), key=lambda r: r['provider']),
                             mode=mode, deployment_id=identity)


class OwnershipTest(unittest.TestCase):
    def test_known_prior_endpoint_profile_preserves_webui_connections(self):
        PREVIOUS_NATIVE_PROFILE = contracts.PREVIOUS_NATIVE_PROFILE
        endpoints = dict(schema_version=1, owner='loom-control-clients', hem_root=str(ROOT), servers=[
            dict(server='main', provider=contracts.provider_id('main'), base_url='http://127.0.0.1:18080/v1',
                 native_profile=PREVIOUS_NATIVE_PROFILE)])
        self.assertEqual(state.desired_connections(endpoints), [row()])
        endpoints['servers'][0]['native_profile'] = 'unknown'
        with self.assertRaises(common.InvalidInput): state.desired_connections(endpoints)

    def test_pure_paths_empty_decode_and_projection(self):
        with patch.object(Path, 'resolve', side_effect=AssertionError('filesystem access')), patch('builtins.open', side_effect=AssertionError('I/O')):
            value = document()
            path = state.ownership_path(ROOT, TARGET + '/')
            self.assertEqual(len(path.stem), 64)
            self.assertEqual(path.parent, ROOT / 'clients/openwebui')
            endpoint = dict(schema_version=1, owner='loom-control-clients', hem_root=str(ROOT),
                            servers=[{k: v for k, v in row().items() if k != 'config'} | {'native_profile': contracts.NATIVE_PROFILE}])
            self.assertEqual(state.desired_connections(endpoint), [row()])
            self.assertEqual(value['pending'], None)
            plan(remote(), value, [row()])

    def test_decode_rejects_corruption_duplicates_unknown_identity_and_size(self):
        valid = document([row()])
        variants = [b'not-json', b'[]', b'{"a":1,"a":2}', b'[' * 2000,
                    b'{"nan":NaN}', b'x' * (state.MAX_BYTES + 1)]
        for changes in ({'owner': 'foreign'}, {'schema_version': True}, {'hem_root': '/different'},
                        {'target_url': TARGET + '/'}, {'models': []}, {'deployment_id': None},
                        {'connections': [row(), row()]}, {'connections': [dict(row(), api_key='')]},
                        {'connections': [dict(row(), provider='wrong')]},
                        {'connections': [dict(row(), config=dict(row()['config'], enable=1))]}):
            variants.append(json.dumps(dict(valid, **changes)).encode())
        for value in variants:
            with self.subTest(value=value[:80]), self.assertRaises(common.InvalidInput):
                state.decode_document(value, ROOT, TARGET)
        for root in ('relative', '/a/../b', '/a//b', '/a/.', '//root'):
            with self.subTest(root=root), self.assertRaises(common.InvalidInput):
                state.decode_document(None, root, TARGET)
        for target in ('http://localhost:3000', 'http://user@127.0.0.1:3000', 'https://127.0.0.1:3000',
                       'http://127.0.0.1:0', TARGET + '/path', TARGET + '?x', TARGET + '#f'):
            with self.assertRaises(common.InvalidInput):
                state.ownership_path(ROOT, target)

    def test_append_preserves_foreign_secrets_only_in_memory_no_input_mutation(self):
        current = remote(foreign=True)
        original = deepcopy(current)
        result = plan(current, document(), [row()])
        self.assertTrue(result.changed)
        self.assertEqual(result.remote_config['OPENAI_API_KEYS'], [SECRET, ''])
        self.assertEqual(result.remote_config['OPENAI_API_CONFIGS']['0'], current['OPENAI_API_CONFIGS']['0'])
        self.assertEqual(current, original)
        self.assertNotIn(SECRET, repr(result))
        for value in (result.pending_document, result.next_document, result.findings):
            self.assertNotIn(SECRET, json.dumps(value))
        self.assertEqual(result.pending_document['pending']['before'], [])
        self.assertEqual(result.pending_document['pending']['after'], [row()])
        result.remote_config['OPENAI_API_CONFIGS']['0']['headers']['unknown'] = 'changed'
        self.assertEqual(current, original)

    def test_reconcile_replace_owned_in_place_remove_stale_and_reindex_sparse_foreign(self):
        first, second = row('a'), row('b', 18081)
        current = remote([first], foreign=True)
        current['OPENAI_API_BASE_URLS'].append('foreign-no-config')
        current['OPENAI_API_KEYS'].append('another-key')
        update = row('a', 19080)
        result = plan(current, document([first]), [update, second])
        self.assertEqual(result.remote_config['OPENAI_API_BASE_URLS'], [update['base_url'], 'https://unrelated.example/custom', 'foreign-no-config', second['base_url']])
        self.assertNotIn('2', result.remote_config['OPENAI_API_CONFIGS'])
        removal = plan(current, document([first]), [], mode='disable')
        self.assertEqual(removal.remote_config['OPENAI_API_KEYS'], [SECRET, 'another-key'])
        self.assertEqual(set(removal.remote_config['OPENAI_API_CONFIGS']), {'0'})
        self.assertEqual(removal.next_document['connections'], [])
        self.assertEqual(removal.pending_document['pending']['before'], [first])
        self.assertEqual(removal.pending_document['pending']['after'], [])

    def test_unknown_foreign_prefix_shape_and_fields_preserved(self):
        current = remote(foreign=True)
        current['OPENAI_API_CONFIGS']['0']['prefix_id'] = {'opaque': [None]}
        result = plan(current, document(), [row()])
        self.assertEqual(result.remote_config['OPENAI_API_CONFIGS']['0'], current['OPENAI_API_CONFIGS']['0'])

    def test_idempotent_no_post_initial_identity_only_mutation_not_check(self):
        owned = document([row()])
        for mode in ('configure', 'check'):
            result = plan(remote([row()]), owned, [row()], mode=mode)
            self.assertFalse(result.changed)
            self.assertIsNone(result.remote_config)
        result = plan(remote([row()]), owned, [row()], identity='id')
        self.assertTrue(result.changed)
        self.assertIsNone(result.remote_config)
        self.assertEqual(result.next_document['deployment_id'], 'id')
        check = plan(remote([row()]), owned, [row()], identity='id', mode='check')
        self.assertIsNone(check.next_document)
        self.assertEqual(owned['deployment_id'], '')

    def test_no_takeover_duplicate_prefix_and_exact_owned_shape_drift(self):
        result = plan(remote([row()]), document(), [row()])
        self.assertEqual(result.configuration, 'drift')
        self.assertIsNone(result.remote_config)
        variants = [remote([row(), row()])]
        for field, value in (('headers', {}), ('auth_type', 'bearer'), ('enable', 1), ('unknown', None)):
            changed = remote([row()])
            changed['OPENAI_API_CONFIGS']['0'][field] = value
            variants.append(changed)
        changed = remote([row()])
        changed['OPENAI_API_KEYS'][0] = SECRET
        variants.append(changed)
        for current in variants:
            result = plan(current, document([row()]), [row(), row('b')])
            self.assertEqual(result.configuration, 'drift')
            self.assertIsNone(result.remote_config)
            self.assertIsNone(result.pending_document)
            self.assertNotIn(SECRET, repr(result))

    def test_disable_partial_retains_evidence_and_does_not_persist_live_edits(self):
        a, b = row('a'), row('b', 18081)
        current = remote([a, b], foreign=True)
        current['OPENAI_API_KEYS'][0] = SECRET
        result = plan(current, document([a, b]), [], mode='disable')
        self.assertTrue(result.partial)
        self.assertEqual(result.configuration, 'partial')
        self.assertEqual(result.next_document['connections'], [a])
        self.assertEqual(result.remote_config['OPENAI_API_KEYS'], [SECRET, SECRET])
        self.assertEqual(result.pending_document['pending']['before'], [b])
        self.assertNotIn(SECRET, json.dumps(result.pending_document))
        retained = plan(current, document([a]), [], mode='disable')
        self.assertTrue(retained.partial)
        self.assertIsNone(retained.remote_config)

    def test_global_disabled_blocks_configure_but_cleanup_preserves_false(self):
        current = remote([row()], enabled=False)
        for mode in ('configure', 'check'):
            result = plan(current, document([row()]), [row()], mode=mode)
            self.assertEqual(result.configuration, 'drift')
            self.assertIsNone(result.remote_config)
        result = plan(current, document([row()]), [], mode='disable')
        self.assertFalse(result.remote_config['ENABLE_OPENAI_API'])
        self.assertEqual(plan(remote(enabled=False), document(), [], mode='disable').configuration, 'absent')

    def test_target_identity_change_disappearance_and_pending_identity(self):
        for identity in ('', 'other'):
            result = plan(remote([row()]), document([row()], 'known'), [row()], identity=identity)
            self.assertEqual(result.recovery, 'conflict')
            self.assertIsNone(result.remote_config)
        pending = plan(remote(), document(), [row()], identity='new').pending_document
        self.assertEqual(plan(remote([row()]), pending, [row()]).recovery, 'conflict')

    def test_pending_completed_unchanged_and_third_state_no_blind_replay(self):
        current = remote(foreign=True)
        staged = plan(current, document(), [row()])
        intent = staged.pending_document
        completed = plan(staged.remote_config, intent, [row()])
        self.assertEqual(completed.recovery, 'completed')
        self.assertIsNone(completed.remote_config)
        self.assertIsNone(completed.next_document['pending'])
        # Current foreign changes are preserved in a freshly calculated retry.
        foreign_changed = deepcopy(current)
        foreign_changed['OPENAI_API_KEYS'][0] = 'new-foreign-key'
        retry = plan(foreign_changed, intent, [row()])
        self.assertEqual(retry.recovery, 'unchanged')
        self.assertEqual(retry.remote_config['OPENAI_API_KEYS'][0], 'new-foreign-key')
        third = deepcopy(staged.remote_config)
        third['OPENAI_API_KEYS'][1] = SECRET
        conflict = plan(third, intent, [row()])
        self.assertEqual(conflict.recovery, 'conflict')
        self.assertIsNone(conflict.remote_config)
        self.assertIsNone(conflict.next_document)
        check = plan(staged.remote_config, intent, [row()], mode='check')
        self.assertEqual(check.recovery, 'completed')
        self.assertEqual(check.configuration, 'pending')
        self.assertIsNone(check.next_document)
        # Disable resolves completed configure before deciding what it owns.
        disable = plan(staged.remote_config, intent, [], mode='disable')
        self.assertEqual(disable.pending_document['pending']['operation'], 'disable')
        self.assertEqual(disable.next_document['connections'], [])

    def test_missing_owned_recreation_and_disable_recovery(self):
        owned = document([row()])
        recreate = plan(remote(), owned, [row()])
        self.assertEqual(recreate.pending_document['pending']['before'], [])
        self.assertEqual(recreate.pending_document['pending']['after'], [row()])
        self.assertEqual(plan(remote(), recreate.pending_document, [row()]).recovery, 'unchanged')
        removal = plan(remote([row()]), owned, [], mode='disable')
        done = plan(remote(), removal.pending_document, [], mode='disable')
        self.assertEqual(done.recovery, 'completed')
        self.assertEqual(done.next_document['connections'], [])
        absent = plan(remote(), owned, [], mode='disable')
        self.assertIsNone(absent.remote_config)
        self.assertTrue(absent.changed)

    def test_pending_excludes_unchanged_and_rejects_invalid_shapes(self):
        a, b = row('a'), row('b', 18081)
        result = plan(remote([a]), document([a]), [a, b])
        self.assertEqual(result.pending_document['pending']['before'], [])
        self.assertEqual(result.pending_document['pending']['after'], [b])
        valid = result.pending_document
        for change in ({'operation': 'check'}, {'token': SECRET}, {'before': [b]}, {'after': []},
                       {'before': [a], 'after': [a]}, {'deployment_id': None}):
            value = deepcopy(valid)
            value['pending'].update(change)
            with self.assertRaises(common.InvalidInput):
                state.decode_document(json.dumps(value).encode(), ROOT, TARGET)

    def test_owned_file_boundary_rejects_alias_and_expected_replace_race(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'state'
            path = state.ownership_path(root, TARGET)
            value = state.decode_document(None, root, TARGET)
            data = files.encoded(value)
            files.atomic_replace(path, data, expected=None, deadline=common.Deadline.after())
            self.assertEqual(state.decode_document(files.read_owned(path), root, TARGET), value)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            path.write_bytes(b'user change')
            with self.assertRaises(common.InvalidInput):
                files.atomic_replace(path, data, expected=data, deadline=common.Deadline.after())
            self.assertEqual(path.read_bytes(), b'user change')
            path.unlink()
            outside = Path(directory) / 'external'
            outside.write_bytes(data)
            path.symlink_to(outside)
            with self.assertRaises(common.InvalidInput):
                files.read_owned(path)
            self.assertEqual(outside.read_bytes(), data)

    def test_pending_owned_file_roundtrip_contains_only_current_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'state'
            owned = state.decode_document(None, root, TARGET)
            staged = plan(remote(foreign=True), owned, [row()])
            path = state.ownership_path(root, TARGET)
            data = files.encoded(staged.pending_document)
            files.atomic_replace(path, data, expected=None, deadline=common.Deadline.after())
            loaded = state.decode_document(files.read_owned(path), root, TARGET)
            self.assertNotIn(SECRET.encode(), data)
            result = plan(staged.remote_config, loaded, [row()])
            files.atomic_replace(path, files.encoded(result.next_document), expected=data,
                                 deadline=common.Deadline.after())
            final = state.decode_document(files.read_owned(path), root, TARGET)
            self.assertIsNone(final['pending'])
            self.assertEqual(final['connections'], [row()])

    def test_target_canonicalization_agrees_with_transport(self):
        transport = importlib.import_module(PACKAGE + '.hem_openwebui_http')
        for target in ('http://[::1]:3000', 'http://[0:0:0:0:0:0:0:1]:3000/', TARGET + '/'):
            canonical = transport.canonical_target(target)
            self.assertEqual(state.decode_document(None, ROOT, target)['target_url'], canonical)
            self.assertEqual(state.ownership_path(ROOT, target), state.ownership_path(ROOT, canonical))
        for target in ('http://[::1%lo]:3000', TARGET + '?', TARGET + '#', TARGET + '/?',
                       TARGET + '/#', TARGET + '\x7f', 'http://127.0.0.1:3000\u2003'):
            with self.subTest(target=target):
                for canonicalizer in (transport.canonical_target, lambda value: state.ownership_path(ROOT, value),
                                      lambda value: state.decode_document(None, ROOT, value)):
                    with self.assertRaises(common.InvalidInput):
                        canonicalizer(target)

    def test_remote_rejects_nested_python_json_coercions(self):
        transport = importlib.import_module(PACKAGE + '.hem_openwebui_http')
        for nested in ((1, 2), {1: 'nonstring'}, {'nested': (1,)}, [{'nested': {False: None}}],
                       float('nan'), float('inf'), {1, 2}, b'bytes'):
            current = remote(foreign=True)
            current['OPENAI_API_CONFIGS']['0']['opaque'] = nested
            with self.subTest(nested=repr(nested)):
                with self.assertRaises(common.InvalidInput):
                    transport.validate_config(current)
                with self.assertRaises(common.InvalidInput):
                    plan(current, document(), [row()])
        current = remote(foreign=True)
        current['OPENAI_API_CONFIGS']['0']['opaque'] = {'array': [None, True, 1, 1.5, 'str']}
        self.assertEqual(plan(current, document(), [row()]).remote_config['OPENAI_API_CONFIGS']['0'],
                         current['OPENAI_API_CONFIGS']['0'])

    def test_remote_malformed_envelope_rejected_without_secret_echo(self):
        for current in ({}, dict(remote(), OPENAI_API_KEYS=['secret']),
                        dict(remote([row()]), OPENAI_API_CONFIGS={'00': {}}),
                        dict(remote([row()]), OPENAI_API_CONFIGS={'1': {}}),
                        dict(remote(), extra=SECRET), dict(remote(), ENABLE_OPENAI_API=1)):
            with self.assertRaises(common.InvalidInput) as caught:
                plan(current, document(), [row()])
            self.assertNotIn(SECRET, str(caught.exception))


if __name__ == '__main__':
    unittest.main()
