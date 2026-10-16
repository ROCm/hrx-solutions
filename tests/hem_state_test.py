"""Real SQLite, subprocess locks and filesystem crash checkpoints for Hem."""

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from binding.python import hem_state
from binding.python.hem_common import Deadline, DeadlineExpired, HemError, InvalidInput
from binding.python.hem_state import Registry


class StateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "state"
        self.registry = Registry(self.root)

    def child(self, code, *arguments):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env.update(PYTHONNOUSERSITE="1", LOOM_SERVE_DIR=str(self.root))
        return subprocess.Popen([sys.executable, "-c", code, str(self.root), *arguments],
                                cwd=Path(__file__).resolve().parents[1], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def finish(self, process, expected=0):
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, expected, stderr)
        return stdout

    def model(self, model_id="small", valid=True):
        assets = [dict(logical_path="model.gguf", source_path="/external/model.gguf", ownership="owned"),
                  dict(logical_path="tokenizer.json", source_path="/external/tokenizer.json", ownership="owned")]
        with self.registry.lock("model", model_id):
            record = self.registry.prepare_model(model_id, assets=assets)
            directory = Path(record["pending"]["path"])
            (directory / "model.gguf").write_bytes(b"weights")
            (directory / "tokenizer.json").write_text(model_id)
            self.registry.publish_model(model_id)
            return self.registry.record_validation(model_id, valid=valid, pipeline="qwen-08b" if valid else None,
                                                   evidence={"metadata": "fixture"})

    def server(self, server_id="local", token="launch1"):
        self.registry.create_server(server_id, saved_settings={"device": "cpu"})
        row = self.registry.reserve_launch(server_id, launch_token=token, current_settings={"device": "cpu"},
                                           endpoint="http://127.0.0.1:9999")
        return self.registry.update_server(server_id, {"phase": "running", "pending": None},
                                           expected_token=token, expected_revision=row["revision"])

    def test_root_environment_and_schema(self):
        with patch.dict(os.environ, {"LOOM_SERVE_DIR": str(self.root)}):
            self.assertEqual(Registry().root, self.root)
        with self.assertRaises(InvalidInput):
            Registry("relative")
        with sqlite3.connect(self.registry.database) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 2)
            self.assertEqual(db.execute("PRAGMA journal_mode").fetchone()[0], "delete")
            self.assertEqual({row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")},
                             {"models", "servers", "instances", "pi_target"})
        with self.registry._transaction() as db:
            self.assertEqual(db.execute("PRAGMA synchronous").fetchone()[0], 2)
            self.assertEqual(db.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        self.assertEqual(self.registry.database.stat().st_mode & 0o777, 0o600)

    def test_corrupt_newer_and_malformed_schema_never_reset(self):
        self.registry.create_server("saved", saved_settings={"keep": True})
        with sqlite3.connect(self.registry.database) as db:
            db.execute("PRAGMA user_version=99")
        before = self.registry.database.read_bytes()
        with self.assertRaisesRegex(HemError, "unsupported"):
            Registry(self.root)
        self.assertEqual(self.registry.database.read_bytes(), before)
        broken = self.root.parent / "broken"
        broken.mkdir()
        (broken / "control.sqlite3").write_bytes(b"not sqlite")
        with self.assertRaises(HemError):
            Registry(broken)
        self.assertEqual((broken / "control.sqlite3").read_bytes(), b"not sqlite")
        malformed = self.root.parent / "malformed"
        Registry(malformed)
        with sqlite3.connect(malformed / "control.sqlite3") as db:
            db.execute("ALTER TABLE models RENAME COLUMN data TO wrong")
        with self.assertRaisesRegex(HemError, "schema"):
            Registry(malformed)

    def test_exception_and_process_death_roll_back_transaction(self):
        self.registry.create_server("saved", saved_settings={"original": True})
        with self.assertRaises(RuntimeError):
            with self.registry._transaction() as db:
                db.execute("DELETE FROM servers")
                raise RuntimeError("interrupted")
        self.assertEqual(len(self.registry.list_servers()), 1)
        code = '''from binding.python.hem_state import Registry
import os,sys
r=Registry(sys.argv[1])
with r._transaction() as db:
    db.execute("DELETE FROM servers")
    os._exit(19)
'''
        self.finish(self.child(code), 19)
        self.assertTrue(Registry(self.root).get_server("saved")["saved_settings"]["original"])

    def test_newer_wal_database_is_refused_without_checkpoint_or_mode_change(self):
        with sqlite3.connect(self.registry.database) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA user_version=99")
            db.commit()
            before = self.registry.database.read_bytes()
            wal = Path(str(self.registry.database) + "-wal")
            wal_before = wal.read_bytes()
            with self.assertRaisesRegex(HemError, "unsupported state schema 99"):
                Registry(self.root)
            self.assertEqual(self.registry.database.read_bytes(), before)
            self.assertEqual(wal.read_bytes(), wal_before)
            self.assertEqual(db.execute("PRAGMA journal_mode").fetchone()[0], "wal")

    def test_concurrent_name_and_launch_claims_are_unique(self):
        code = '''from binding.python.hem_state import Registry
from binding.python.hem_common import HemError
import sys
r=Registry(sys.argv[1])
try:
    r.create_server("same",saved_settings={})
    print("created")
except HemError:
    print("conflict")
'''
        processes = [self.child(code) for _ in range(5)]
        outputs = [self.finish(process).strip() for process in processes]
        self.assertEqual(outputs.count("created"), 1)
        self.assertEqual(outputs.count("conflict"), 4)
        code = '''from binding.python.hem_state import Registry
from binding.python.hem_common import HemError
import sys
r=Registry(sys.argv[1])
try:
    r.reserve_launch("same",launch_token=sys.argv[2],current_settings={},endpoint="http://local")
    print("reserved")
except HemError:
    print("conflict")
'''
        processes = [self.child(code, f"token{i}") for i in range(5)]
        outputs = [self.finish(process).strip() for process in processes]
        self.assertEqual(outputs.count("reserved"), 1)
        self.assertEqual(len(self.registry.list_servers()), 1)

    def test_caller_crash_releases_lock_but_keeps_pending(self):
        self.registry.create_server("local", saved_settings={})
        code = '''from binding.python.hem_state import Registry
import sys,os
r=Registry(sys.argv[1])
with r.lock("server","local"):
    r.reserve_launch("local",launch_token="owned",current_settings={},endpoint="http://local")
    os._exit(17)
'''
        self.finish(self.child(code), 17)
        with self.registry.lock("server", "local", deadline=Deadline.after(.2)):
            self.assertEqual(self.registry.get_server("local")["pending"]["stage"], "reserved")
            with self.assertRaisesRegex(HemError, "unresolved"):
                self.registry.reserve_launch("local", launch_token="new", current_settings={}, endpoint="http://local")

    def test_lock_contention_deadline_and_independent_servers(self):
        code = '''from binding.python.hem_state import Registry
from binding.python.hem_common import Deadline,DeadlineExpired
import sys
r=Registry(sys.argv[1])
try:
    with r.lock("server",sys.argv[2],deadline=Deadline.after(.1)):
        print("locked")
except DeadlineExpired:
    print("timeout")
'''
        with self.registry.lock("server", "one"):
            self.assertEqual(self.finish(self.child(code, "one")).strip(), "timeout")
            self.assertEqual(self.finish(self.child(code, "two")).strip(), "locked")
            with self.assertRaisesRegex(HemError, "before"):
                with self.registry.lock("model", "small"):
                    self.fail("reversed ordering")
        with self.registry.lock("model", "small"), self.registry.lock("server", "one") as held:
            self.assertFalse(os.get_inheritable(held.fileno()))

    def test_sqlite_contention_uses_callers_deadline(self):
        with self.registry._transaction():
            code = '''from binding.python.hem_state import Registry
from binding.python.hem_common import Deadline,DeadlineExpired
import sys
try:
    Registry(sys.argv[1],deadline=Deadline.after(.05))
except DeadlineExpired:
    print("timeout")
'''
            self.assertEqual(self.finish(self.child(code)).strip(), "timeout")

    def test_inherited_lock_survives_parent_close_until_helper_exec_or_exit(self):
        with self.registry.lock("server", "local") as held:
            helper = subprocess.Popen([sys.executable, "-c", "import sys;print('ready',flush=True);sys.stdin.readline()"],
                                      pass_fds=(held.fileno(),), stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.addCleanup(lambda: helper.poll() is None and helper.kill())
            self.assertEqual(helper.stdout.readline().strip(), "ready")
        with self.assertRaises(DeadlineExpired):
            with self.registry.lock("server", "local", deadline=Deadline.after(.05)):
                self.fail("helper still owns inherited lock")
        helper.communicate("exit\n", timeout=3)
        self.assertEqual(helper.returncode, 0)
        with self.registry.lock("server", "local", deadline=Deadline.after(.1)):
            pass

    def test_publication_invalid_qualified_and_tokenizer_scoping(self):
        invalid = self.model("small", valid=False)
        large = self.model("large")
        self.assertEqual(invalid["pending"]["stage"], "published-invalid")
        self.assertIsNone(invalid["pipeline"])
        self.assertIsNone(large["pending"])
        self.assertTrue(large["validation"]["valid"])
        self.assertEqual((Path(invalid["directory"]) / "tokenizer.json").read_text(), "small")
        self.assertEqual((Path(large["directory"]) / "tokenizer.json").read_text(), "large")
        server = self.server()
        with self.assertRaisesRegex(HemError, "not qualified"):
            self.registry.create_instance("local", "launch1", "no", model_id="small")
        fixed = self.registry.record_validation("small", valid=True, pipeline="qwen-08b", evidence={"retry": True})
        self.assertIsNone(fixed["pending"])
        self.assertEqual(len(self.registry.list_models()), 2)
        self.registry.create_instance("local", "launch1", "yes", model_id="small")
        self.assertEqual(len(self.registry.references("small")), 1)
        self.assertEqual(server["launch_token"], "launch1")

    def test_published_before_qualification_is_not_valid(self):
        assets = [dict(logical_path="model.gguf", source_path="/external/model", ownership="reference")]
        record = self.registry.prepare_model("small", assets=assets)
        Path(record["pending"]["path"], "model.gguf").symlink_to("/missing/external/model")
        current = self.registry.publish_model("small")
        self.assertEqual(current["pending"], {"stage": "qualification"})
        self.assertIsNone(current["validation"])
        self.assertTrue((Path(current["directory"]) / "model.gguf").is_symlink())

    def test_crash_after_rename_can_finish_only_identified_publication(self):
        assets = [dict(logical_path="tokenizer.json", source_path="/external/tok", ownership="owned")]
        initial = self.registry.prepare_model("small", assets=assets)
        Path(initial["pending"]["path"], "tokenizer.json").write_text("kept")
        original = hem_state._sync_dir
        def fail_after_rename(path):
            if path == self.root / "models" and (path / "small").exists():
                raise RuntimeError("power loss before record update")
            original(path)
        with patch.object(hem_state, "_sync_dir", side_effect=fail_after_rename):
            with self.assertRaises(RuntimeError):
                self.registry.publish_model("small")
        self.assertEqual(self.registry.get_model("small")["pending"]["stage"], "staging")
        record = Registry(self.root).publish_model("small")
        self.assertEqual(record["pending"], {"stage": "qualification"})
        self.assertEqual((Path(record["directory"]) / "tokenizer.json").read_text(), "kept")

    def test_process_death_at_publication_checkpoint_retains_recovery_identity(self):
        assets = [dict(logical_path="tokenizer.json", source_path="/external/tok", ownership="owned")]
        initial = self.registry.prepare_model("small", assets=assets)
        Path(initial["pending"]["path"], "tokenizer.json").write_text("durable")
        code = '''from binding.python import hem_state
import os,sys
r=hem_state.Registry(sys.argv[1])
sync=hem_state._sync_dir
def interrupted(path):
    sync(path)
    if path == r.root / "models" and (path / "small").exists():
        os._exit(23)
hem_state._sync_dir=interrupted
with r.lock("model","small"):
    r.publish_model("small")
'''
        self.finish(self.child(code), 23)
        with self.registry.lock("model", "small"):
            current = self.registry.publish_model("small")
        self.assertEqual(current["pending"]["stage"], "qualification")
        self.assertIsNone(current["validation"])
        self.assertEqual((Path(current["directory"]) / "tokenizer.json").read_text(), "durable")

    def test_interruption_before_staging_mkdir_is_recoverable(self):
        assets = [dict(logical_path="tokenizer.json", source_path="/external/tok", ownership="owned")]
        original = Path.mkdir
        def fail_staging(path, *args, **kwargs):
            if ".staging-" in path.name:
                raise RuntimeError("interrupted after reservation")
            return original(path, *args, **kwargs)
        with patch.object(Path, "mkdir", fail_staging):
            with self.assertRaises(RuntimeError):
                self.registry.prepare_model("small", assets=assets)
        self.assertEqual(self.registry.get_model("small")["pending"]["stage"], "staging")
        record = self.registry.prepare_model("small", assets=assets)
        self.assertTrue(Path(record["pending"]["path"]).is_dir())

    def test_reference_alias_never_changes_external_inputs(self):
        external = self.root.parent / "shared.gguf"
        external.write_bytes(b"immutable fixture")
        evidence = (external.stat().st_mtime_ns, external.stat().st_size)
        assets = [dict(logical_path="model.gguf", source_path=str(external), ownership="reference")]
        current = self.registry.prepare_model("alias", assets=assets)
        Path(current["pending"]["path"], "model.gguf").symlink_to(external)
        published = self.registry.publish_model("alias")
        self.assertEqual((external.stat().st_mtime_ns, external.stat().st_size), evidence)
        self.assertEqual(external.read_bytes(), b"immutable fixture")
        self.assertEqual((Path(published["directory"]) / "model.gguf").resolve(), external)

    def test_publication_never_replaces_or_adopts_directory(self):
        assets = [dict(logical_path="tokenizer.json", source_path="/external/tok", ownership="owned")]
        first = self.registry.prepare_model("small", assets=assets)
        Path(first["pending"]["path"], "tokenizer.json").write_text("new")
        destination = self.root / "models" / "small"
        destination.mkdir()
        (destination / "tokenizer.json").write_text("original")
        with self.assertRaisesRegex(HemError, "already exists"):
            self.registry.publish_model("small")
        self.assertEqual((destination / "tokenizer.json").read_text(), "original")
        Path(first["pending"]["path"], "tokenizer.json").unlink()
        Path(first["pending"]["path"]).rmdir()
        with self.assertRaisesRegex(HemError, "unproven"):
            self.registry.publish_model("small")
        orphan = self.root / "models" / "orphan"
        orphan.mkdir()
        with self.assertRaisesRegex(HemError, "untracked"):
            self.registry.prepare_model("orphan", assets=assets)
        self.assertEqual(len(self.registry.list_models()), 1)

    def test_model_retries_conflicting_mappings_and_unsafe_paths(self):
        assets = [dict(logical_path="a.gguf", source_path="/external/a", ownership="reference")]
        original = self.registry.prepare_model("same", assets=assets, provenance={"recipe": "test"})
        again = self.registry.prepare_model("same", assets=assets, provenance={"recipe": "test"})
        self.assertEqual(original, again)
        with self.assertRaisesRegex(HemError, "different mappings"):
            self.registry.prepare_model("same", assets=assets, provenance={"recipe": "other"})
        for logical in ("../outside", "/absolute", ""):
            with self.subTest(logical=logical), self.assertRaises(InvalidInput):
                self.registry.prepare_model("invalid", assets=[dict(assets[0], logical_path=logical)])
        with self.assertRaises(InvalidInput):
            self.registry.prepare_model("invalid", assets=assets + assets)
        with self.assertRaisesRegex(InvalidInput, "immutable"):
            self.registry.update_model("same", {"assets": []}, expected_revision=1)

    def test_generated_documents_replace_atomically_and_confine_paths(self):
        document = self.registry.write_document("servers/local/request.json", b'{"a":1}')
        self.registry.write_document("servers/local/request.json", b'{"a":2}')
        self.assertEqual(document.read_bytes(), b'{"a":2}')
        self.assertEqual(list(document.parent.glob("*.tmp")), [])
        for path in ("../escape", "/absolute", "models/small/model.gguf", "control.sqlite3"):
            with self.subTest(path=path), self.assertRaises(InvalidInput):
                self.registry.write_document(path, b"bad")
        (self.root / "alias").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(HemError):
            self.registry.write_document("alias/escape", b"bad")
        with patch.object(hem_state.os, "replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                self.registry.write_document("servers/local/request.json", b"wrong")
        self.assertEqual(document.read_bytes(), b'{"a":2}')

    def test_launch_token_and_revision_reject_stale_updates(self):
        first = self.server()
        row = self.registry.update_server("local", {"pending": {"action": "shutdown"}},
                                          expected_token="launch1", expected_revision=first["revision"])
        with self.assertRaisesRegex(HemError, "changed"):
            self.registry.update_server("local", {"phase": "unknown"}, expected_token="launch1",
                                        expected_revision=first["revision"])
        with self.assertRaisesRegex(HemError, "accepting"):
            self.registry.create_instance("local", "launch1", "blocked")
        with self.assertRaisesRegex(HemError, "changed"):
            self.registry.clear_server("local", expected_token="old", expected_revision=row["revision"])
        self.registry.clear_server("local", expected_token="launch1", expected_revision=row["revision"])
        self.registry.reserve_launch("local", launch_token="launch2", current_settings={}, endpoint="http://local")
        with self.assertRaisesRegex(HemError, "changed"):
            self.registry.update_server("local", {"phase": "running"}, expected_token="launch1", expected_revision=first["revision"])

    def test_completed_lifecycles_keep_only_current_rows(self):
        self.model()
        self.registry.create_server("local", saved_settings={})
        for index in range(30):
            token = f"launch{index}"
            server = self.registry.reserve_launch("local", launch_token=token, current_settings={}, endpoint="http://local")
            server = self.registry.update_server("local", {"phase": "running", "pending": None}, expected_token=token,
                                                 expected_revision=server["revision"])
            instance = self.registry.create_instance("local", token, "chat", model_id="small", settings={"context": 8192},
                                                     document="launch/request.json", dependencies=["weights"],
                                                     pending={"action": "start", "stage": "register"})
            instance = self.registry.update_instance("local", token, "chat", {"pending": None, "incarnation": str(index)},
                                                     expected_revision=instance["revision"])
            self.registry.retire_instance("local", token, "chat", expected_revision=instance["revision"])
            self.registry.clear_server("local", expected_token=token, expected_revision=server["revision"])
        self.assertEqual(len(self.registry.list_models()), 1)
        self.assertEqual(len(self.registry.list_servers()), 1)
        self.assertEqual(self.registry.list_instances(), [])
        self.assertEqual(self.registry.references("small"), [])
        self.assertIsNone(self.registry.get_server("local")["pending"])

    def test_references_span_servers_and_exit_clears_only_one(self):
        self.model()
        one = self.server("one", "a")
        self.server("two", "b")
        self.registry.create_instance("one", "a", "chat", model_id="small", pending={"action": "start"})
        self.registry.create_instance("two", "b", "chat", model_id="small", pending={"action": "stop"})
        self.assertEqual(len(self.registry.references("small")), 2)
        with self.assertRaises(HemError):
            self.registry.clear_server("one", expected_token="a", expected_revision=one["revision"] - 1)
        self.assertEqual(len(self.registry.references("small")), 2)  # DELETE rolled back on stale CAS
        self.registry.clear_server("one", expected_token="a", expected_revision=one["revision"])
        self.assertEqual([row["server_id"] for row in self.registry.references("small")], ["two"])


if __name__ == "__main__":
    unittest.main()
