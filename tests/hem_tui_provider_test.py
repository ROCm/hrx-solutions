"""Read-only inventory and observation checks for the Loom Control TUI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

from binding.python.hem_common import HemError, UncertainOutcome
from tests.hem_lifecycle_fixture import fixture
from binding.python.control.hem_state import Registry
from binding.python.control import hem_state
from binding.python.tui import hem_tui_provider


def tree_state(root):
    if not root.exists():
        return None
    result = {}
    for path in sorted((root, *root.rglob("*"))):
        relative = str(path.relative_to(root)) or "."
        info = path.lstat()
        result[relative] = (info.st_mode, info.st_mtime_ns,
                            path.read_bytes() if path.is_file() else None)
    return result


class LocalServiceProviderTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "state"

    def running_server(self, registry, name, token):
        registry.create_server(name, saved_settings={"device": "test"})
        row = registry.reserve_launch(
            name, launch_token=token, current_settings={"device": "test"},
            endpoint="http://127.0.0.1:9999")
        return registry.update_server(
            name,
            {"phase": "running", "pending": None,
             "process": {"pid": 10, "boot_id": "boot", "start_ticks": 20,
                         "executable": {"device": 1, "inode": 2}},
             "build": {"executable": {"device": 1, "inode": 2}}},
            expected_token=token, expected_revision=row["revision"])

    async def test_saved_model_pipeline_is_joined_in_read_only_log_snapshot(self):
        from binding.python.models import hem_models
        from binding.python.hem_common import Deadline
        from binding.python.tui.hem_tui_logs import LogSelection, _records
        from tests.gguf_fixtures import write_gguf
        weights = self.root.parent / 'weights.gguf'; write_gguf(weights, tensors=[('fixture', [1], 0, 0)])
        tokenizer = weights.with_suffix('.json'); tokenizer.write_text('{}')
        registry = Registry(self.root)
        hem_models.import_model(registry, 'local', ggufs=[weights], tokenizer=tokenizer, pipeline='qwen-08b')
        self.running_server(registry, 'server', 'launch')
        registry.create_instance('server', 'launch', 'chat', model_id='local', incarnation=17)
        before = tree_state(self.root)
        with patch.object(Registry, '__init__', side_effect=AssertionError('passive Registry init')):
            _, row = _records(self.root, LogSelection('model', 'server', 'launch', 'chat', 17), Deadline.after())
        self.assertEqual(row['pipeline'], 'qwen-08b')
        self.assertEqual(tree_state(self.root), before)

    async def test_absent_namespace_is_empty_and_remains_absent(self):
        provider = hem_tui_provider.LocalServiceProvider(self.root)
        with patch.object(Registry, '__init__', side_effect=AssertionError('passive Registry init')), patch.object(
                hem_state, '_migrate', side_effect=AssertionError('passive migration')):
            self.assertEqual(await provider.list_services(), ())
        self.assertFalse(self.root.exists())

    async def test_existing_inventory_read_performs_no_authored_writes(self):
        registry = Registry(self.root)
        registry.create_server("stopped", saved_settings={"device": "test"})
        before = tree_state(self.root)
        with patch.object(Registry, '__init__', side_effect=AssertionError('passive Registry init')), patch.object(
                Registry, 'lock', side_effect=AssertionError('passive resource lock')), patch.object(
                hem_state, '_migrate', side_effect=AssertionError('passive migration')), patch.object(
                hem_state, '_directory', side_effect=AssertionError('passive directory creation')):
            observations = await hem_tui_provider.LocalServiceProvider(self.root).list_services()
        self.assertEqual([(row.name, row.state) for row in observations],
                         [("stopped", "stopped")])
        self.assertEqual(tree_state(self.root), before)

    async def test_populated_server_and_instance_are_normalized(self):
        registry = Registry(self.root)
        self.running_server(registry, "local", "launch-one")
        registry.create_instance("local", "launch-one", "chat", incarnation=17)
        native = {"object": "list", "data": [{"id": "chat", "loom": {
            "incarnation": 17, "registration": "ready"}}], "server": {}}
        health = hem_tui_provider._ServerHealth("running", "available", native)
        with patch.object(hem_tui_provider, "_observe_server", return_value=health):
            rows = await hem_tui_provider.LocalServiceProvider(self.root).list_services()
        self.assertEqual([(row.kind, row.name, row.state) for row in rows],
                         [("server", "local", "running"),
                          ("model", "chat", "ready")])
        self.assertEqual(rows[1].server, "local")
        self.assertEqual(rows[1].incarnation, 17)

    async def test_socket_association_uncertainty_is_unknown_and_passive(self):
        class UnassociatedProcess:
            def __init__(self, *_args, **_kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                pass
            def connection(self, *_args):
                raise UncertainOutcome("socket association is unknown")
            def verify(self, *_args):
                pass

        registry = Registry(self.root)
        row = self.running_server(registry, "local", "launch-one")
        registry.create_instance("local", "launch-one", "chat", incarnation=17)
        with fixture() as (url, requests):
            registry.update_server('local', {'endpoint': url}, expected_token=row['launch_token'],
                                   expected_revision=row['revision'])
            before = tree_state(self.root)
            read_snapshot = hem_tui_provider._read_snapshot
            with patch.object(hem_tui_provider, 'PinnedProcess', UnassociatedProcess), \
                 patch.object(hem_tui_provider, '_read_snapshot', wraps=read_snapshot) as snapshots:
                rows = await hem_tui_provider.LocalServiceProvider(self.root).list_services()
            self.assertEqual(snapshots.call_count, 2)
            self.assertEqual([(row.kind, row.state) for row in rows],
                             [('server', 'unknown'), ('model', 'unknown')])
            self.assertEqual(requests, [])  # Refused before transmitting HTTP.
            self.assertEqual(tree_state(self.root), before)

    async def test_failed_observation_preserves_other_rows(self):
        registry = Registry(self.root)
        self.running_server(registry, "bad", "launch-bad")
        self.running_server(registry, "good", "launch-good")

        def observe(record, _deadline):
            if record["id"] == "bad":
                raise HemError("fixture failure")
            return hem_tui_provider._ServerHealth(
                "running", "available", {"object": "list", "data": [], "server": {}})

        with patch.object(hem_tui_provider, "_observe_server", side_effect=observe):
            rows = await hem_tui_provider.LocalServiceProvider(self.root).list_services()
        self.assertEqual({row.name: row.state for row in rows},
                         {"bad": "unknown", "good": "running"})

    async def test_one_total_deadline_stops_new_observations_and_releases_resources(self):
        registry = Registry(self.root)
        for name in ("one", "two", "three"):
            self.running_server(registry, name, "launch-" + name)
        calls = []
        active = 0

        def blocking(record, _deadline):
            nonlocal active
            calls.append(record["id"])
            active += 1
            try:
                time.sleep(0.06)
                return hem_tui_provider._ServerHealth("running", "available", {
                    "object": "list", "data": [], "server": {}})
            finally:
                active -= 1

        started = time.monotonic()
        with patch.object(hem_tui_provider, "_observe_server", side_effect=blocking):
            rows = await hem_tui_provider.LocalServiceProvider(
                self.root, timeout=0.03).list_services()
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertEqual(calls, ["one"])
        self.assertEqual(active, 0)
        self.assertEqual({row.name: row.state for row in rows},
                         {"one": "unknown", "two": "unknown", "three": "unknown"})

    async def test_authoritative_schema_and_launch_relationship_are_required(self):
        registry = Registry(self.root)
        self.running_server(registry, "local", "launch-one")
        registry.create_instance("local", "launch-one", "chat", incarnation=17)
        database = self.root / "control.sqlite3"
        connection = sqlite3.connect(database)
        try:
            row = connection.execute(
                "SELECT data FROM instances WHERE name='chat'").fetchone()
            data = json.loads(row[0])
            data["launch_token"] = "missing-launch"
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute(
                "UPDATE instances SET launch_token=?,data=? WHERE name='chat'",
                ("missing-launch", json.dumps(data, sort_keys=True, separators=(",", ":"))))
            connection.commit()
        finally:
            connection.close()
        before = tree_state(self.root)
        with self.assertRaisesRegex(HemError, "broken references"):
            await hem_tui_provider.LocalServiceProvider(self.root).list_services()
        self.assertEqual(tree_state(self.root), before)

        other = Path(self.temporary.name) / "missing-table"
        Registry(other)
        connection = sqlite3.connect(other / "control.sqlite3")
        try:
            connection.execute("DROP TABLE pi_target")
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(HemError, "schema constraints"):
            await hem_tui_provider.LocalServiceProvider(other).list_services()

    async def test_sql_server_token_must_match_json_even_when_foreign_keys_are_valid(self):
        registry = Registry(self.root)
        self.running_server(registry, 'local', 'launch-one')
        registry.create_instance('local', 'launch-one', 'chat', incarnation=17)
        with sqlite3.connect(registry.database) as database:
            data = json.loads(database.execute("SELECT data FROM servers WHERE id='local'").fetchone()[0])
            data['launch_token'] = 'different-json-token'
            database.execute("UPDATE servers SET data=? WHERE id='local'", (json.dumps(data),))
            self.assertIsNone(database.execute('PRAGMA foreign_key_check').fetchone())
        # Registry supplies only the SQL id; TUI has additional SQL token proof.
        self.assertEqual(registry.get_server('local')['launch_token'], 'different-json-token')
        before = tree_state(self.root)
        with patch.object(Registry, '__init__', side_effect=AssertionError('passive Registry init')), patch.object(
                hem_tui_provider, '_observe_server', side_effect=AssertionError('corrupt inventory observed')):
            with self.assertRaisesRegex(HemError, 'corrupt state record; inventory retained'):
                await hem_tui_provider.LocalServiceProvider(self.root).list_services()
        self.assertEqual(tree_state(self.root), before)

    async def test_inventory_row_cap_fails_read_only_and_within_budget(self):
        registry = Registry(self.root)
        registry.create_server("one", saved_settings={"device": "test"})
        registry.create_server("two", saved_settings={"device": "test"})
        before = tree_state(self.root)
        started = time.monotonic()
        with patch.object(hem_tui_provider, "MAX_INVENTORY_ROWS", 1):
            with self.assertRaisesRegex(HemError, "observation limit"):
                await hem_tui_provider.LocalServiceProvider(
                    self.root, timeout=.2).list_services()
        self.assertLess(time.monotonic() - started, .5)
        self.assertEqual(tree_state(self.root), before)

    def test_large_postprocessing_preserves_completed_rows_and_bounds_unknown_tail(self):
        servers = []
        instances = []
        health = {}
        for index in range(100):
            server_id = f"server-{index:03d}"
            token = f"launch-{index:03d}"
            servers.append(hem_tui_provider._ServerSnapshot({
                "id": server_id, "launch_token": token, "revision": 1,
                "phase": "running", "endpoint": "http://127.0.0.1:9999"}))
            instances.append(hem_tui_provider._InstanceSnapshot({
                "server_id": server_id, "launch_token": token,
                "name": f"model-{index:03d}", "incarnation": 1, "revision": 1,
                "pending": None}))
            health[server_id] = hem_tui_provider._ServerHealth(
                "running", "unavailable")
        ticks = [0]

        def clock():
            ticks[0] += 1
            return ticks[0] / 1000
        deadline = hem_tui_provider.Deadline(.34, clock=clock)
        started = time.monotonic()
        rows = hem_tui_provider._build_observations(
            servers, instances, health, (servers, instances), deadline=deadline)
        self.assertLess(time.monotonic() - started, .1)
        self.assertEqual(len(rows), 200)
        self.assertIn("unavailable", {row.state for row in rows})
        self.assertEqual(rows[-1].state, "unknown")
        self.assertGreaterEqual(ticks[0], 340)

    def test_replaced_launch_never_publishes_old_observation(self):
        registry = Registry(self.root)
        self.running_server(registry, "local", "launch-one")
        first = hem_tui_provider._read_snapshot(
            self.root, hem_tui_provider.Deadline.after(1))
        old = first[0][0].record
        replacement = dict(old, launch_token="launch-two", revision=old["revision"] + 1,
                           endpoint="http://127.0.0.1:8888")
        current = ((hem_tui_provider._ServerSnapshot(replacement),), ())
        health = {"local": hem_tui_provider._ServerHealth(
            "running", "available", {"object": "list", "data": [], "server": {}})}
        rows = hem_tui_provider._build_observations(*first, health, current)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].name, rows[0].state, rows[0].endpoint),
                         ("local", "unknown", ""))
        self.assertFalse(rows[0].identity_verified)

    def test_replaced_instance_incarnation_is_not_joined(self):
        registry = Registry(self.root)
        self.running_server(registry, "local", "launch-one")
        registry.create_instance("local", "launch-one", "chat", incarnation=17)
        servers, instances = hem_tui_provider._read_snapshot(
            self.root, hem_tui_provider.Deadline.after(1))
        replacement = dict(instances[0].record, incarnation=18,
                           revision=instances[0].record["revision"] + 1)
        current = (servers, (hem_tui_provider._InstanceSnapshot(replacement),))
        native = {"object": "list", "data": [{"id": "chat", "loom": {
            "incarnation": 17, "registration": "ready"}}], "server": {}}
        health = {"local": hem_tui_provider._ServerHealth("running", "available", native)}
        rows = hem_tui_provider._build_observations(servers, instances, health, current)
        self.assertEqual([(row.kind, row.name, row.state) for row in rows],
                         [("server", "local", "running"),
                          ("model", "chat", "unknown")])
        self.assertFalse(rows[1].identity_verified)

    def test_repeated_failed_http_observation_releases_process_pins(self):
        active = 0
        maximum = 0
        status_limits = []

        class FakePinned:
            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                nonlocal active, maximum
                active += 1
                maximum = max(maximum, active)
                return self

            def __exit__(self, *_args):
                nonlocal active
                active -= 1

            def connection(self, *_args):
                return None

            def verify(self, *_args):
                return None

        class FailedClient:
            def __init__(self, *_args, **_kwargs):
                pass

            def status(self, **_kwargs):
                status_limits.append(_kwargs.get("max_response_bytes"))
                raise HemError("fixture HTTP failure")

        record = {
            "launch_token": "launch-one", "endpoint": "http://127.0.0.1:9999",
            "process": {"pid": 10}, "build": {"executable": {}},
        }
        with patch.object(hem_tui_provider, "PinnedProcess", FakePinned), \
             patch.object(hem_tui_provider, "NativeClient", FailedClient):
            for _ in range(20):
                health = hem_tui_provider._observe_server(
                    record, hem_tui_provider.Deadline.after(1))
                self.assertEqual((health.process, health.http),
                                 ("running", "unavailable"))
        self.assertEqual(active, 0)
        self.assertEqual(maximum, 1)
        self.assertEqual(status_limits, [hem_tui_provider.MAX_TUI_STATUS_BYTES] * 20)


if __name__ == "__main__":
    unittest.main()
