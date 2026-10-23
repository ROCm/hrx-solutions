"""Public Pi CLI subprocesses against real installer/SQLite and fake metadata."""
from contextlib import redirect_stderr
import http.server
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from binding.python import hem_cli, hem_pi
from binding.python.cli import hem_pi_cli
from binding.python.hem_common import Deadline, DeadlineExpired, HemError, InvalidInput, UncertainOutcome
from binding.python.hem_pi_contracts import MANIFEST_PATH
from binding.python.hem_state import Registry


class PiCliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root, self.target = self.base / "state", self.base / "pi"
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.package = self.base / "npm/node_modules/@earendil-works/pi-coding-agent"
        self.package.mkdir(parents=True)
        self.metadata = dict(name="@earendil-works/pi-coding-agent", version="1.3.0", bin={"pi":"dist/cli.js"},
            exports={".":{"import":"./dist/index.js"}}, dependencies={"@earendil-works/pi-ai":"1.3.0"})
        self.write_metadata()
        (self.package / "dist").mkdir()
        cli = self.package / "dist/cli.js"
        cli.write_text("#!/usr/bin/env node\nthrow new Error('Pi must not execute');\n")
        cli.chmod(0o755)
        (self.package / "dist/index.js").write_text("throw new Error('public entry must not execute');")
        (self.bin / "pi").symlink_to(cli)
        self.ai = self.package.parent / "pi-ai"
        (self.ai / "dist").mkdir(parents=True)
        (self.ai / "package.json").write_text(json.dumps(dict(name="@earendil-works/pi-ai", version="1.3.0", exports={".":{"import":"./dist/index.js"}})))
        (self.ai / "dist/index.js").write_text("throw new Error('pi-ai must not execute');")
        self.node("22.22.1")
        self.env = {key:value for key,value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
        self.env.update(PATH=str(self.bin), LOOM_SERVE_DIR=str(self.root), PI_CODING_AGENT_DIR=str(self.target),
                        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", NODE_OPTIONS="--require /must-not-load", NODE_PATH="/must-not-load")

    def node(self, version):
        node = self.bin / "node"
        node.write_text("#!/bin/sh\n[ -z \"$NODE_OPTIONS$NODE_PATH\" ] || exit 23\n[ \"$1\" = --version ] || exit 24\nprintf 'v" + version + "\\n'\n")
        node.chmod(0o755)

    def write_metadata(self):
        (self.package / "package.json").write_text(json.dumps(self.metadata))

    def invoke(self, *args, expected=0, env=None):
        process = subprocess.run([sys.executable, "-m", "binding.python.hem_cli", "pi", "configure", "--json", *args],
            cwd=Path(__file__).resolve().parents[1], env=env or self.env, text=True, capture_output=True, timeout=10)
        self.assertEqual(process.returncode, expected, process.stderr + process.stdout)
        self.assertNotIn("Traceback", process.stderr)
        return json.loads(process.stdout), process.stderr

    def snapshot(self):
        return {str(p.relative_to(self.base)):(p.lstat().st_mode, p.lstat().st_mtime_ns,
            p.read_bytes() if p.is_file() and not p.is_symlink() else os.readlink(p) if p.is_symlink() else None)
            for p in self.base.rglob("*")}

    def test_absent_check_and_missing_parents_are_passive(self):
        target = self.base / "missing/parents/agent"
        before = self.snapshot()
        body, stderr = self.invoke("--check", "--pi-dir", str(target), expected=1)
        self.assertEqual(body["state"], "changes_required")
        self.assertTrue(body["completed"])
        self.assertFalse(body["changed"] or body["configured"])
        self.assertNotIn("accepted", body)
        self.assertEqual(before, self.snapshot())
        self.assertIn("No native Loom equivalent", stderr)
        self.assertTrue(all(path.startswith((str(target), str(self.root))) for path in body["affected_paths"]))

    def test_install_idempotent_passive_check_disable_preserve_user_data(self):
        self.target.mkdir(mode=0o700)
        models = dict(providers={"hrx-loom":dict(baseUrl="http://127.0.0.1:18080/v1", apiKey="secret-sentinel", models=[dict(id="old-qwen")]), "other":dict(apiKey="secret-sentinel")})
        for name, data in (("models.json", json.dumps(models)), ("settings.json", '{"defaultModel":"old-qwen"}'), ("auth.json", 'credentials-secret-sentinel')):
            (self.target / name).write_text(data)
        user = {p.name:p.read_bytes() for p in self.target.iterdir()}
        body, stderr = self.invoke()
        self.assertTrue(body["configured"] and body["changed"] and body["accepted"])
        self.assertEqual(body["runtime"]["version"], "1.3.0")
        self.assertEqual(body["runtime_compatibility"]["state"], "compatible")
        self.assertEqual(body["legacy"][0]["ids"], ["old-qwen"])
        self.assertIn("Export/back up", stderr)
        self.assertNotIn("secret-sentinel", json.dumps(body) + stderr)
        body, _ = self.invoke()
        self.assertFalse(body["changed"])
        before = self.snapshot()
        body, _ = self.invoke("--check")
        self.assertTrue(body["configured"])
        self.assertEqual(before, self.snapshot())
        body, _ = self.invoke("--disable")
        self.assertEqual(body["state"], "disabled")
        self.assertFalse((self.target / MANIFEST_PATH).exists())
        self.assertEqual(user, {name:(self.target / name).read_bytes() for name in user})
        body, _ = self.invoke("--disable")
        self.assertFalse(body["changed"])

    def test_unsupported_runtime_and_missing_public_entries_refuse_without_writes(self):
        for version in ("1.1.0", "1.4.0", None):
            self.metadata["version"] = version
            self.write_metadata()
            before = self.snapshot()
            body, stderr = self.invoke(expected=2)
            self.assertIn("unsupported", stderr)
            self.assertFalse(body["completed"])
            self.assertEqual(before, self.snapshot())
        self.metadata["version"] = "1.3.0"
        self.write_metadata()
        self.node("22.18.0")
        before = self.snapshot()
        self.invoke(expected=2)
        self.assertEqual(before, self.snapshot())
        self.node("22.22.1")
        (self.package / "dist/index.js").unlink()
        before = self.snapshot()
        self.invoke(expected=2)
        self.assertEqual(before, self.snapshot())

    def test_check_retains_findings_without_qualified_runtime(self):
        self.invoke()
        for failure in ("missing_pi", "unsupported_pi", "missing_node", "unsupported_node", "broken_node"):
            with self.subTest(failure=failure):
                self.metadata["version"] = "1.3.0"
                self.write_metadata()
                if not (self.bin / "pi").exists():
                    (self.bin / "pi").symlink_to(self.package / "dist/cli.js")
                self.node("22.22.1")
                if failure == "missing_pi":
                    (self.bin / "pi").unlink()
                elif failure == "unsupported_pi":
                    self.metadata["version"] = "1.1.0"
                    self.write_metadata()
                elif failure == "missing_node":
                    (self.bin / "node").unlink()
                elif failure == "unsupported_node":
                    self.node("22.18.0")
                else:
                    # Executable selected on PATH, but its interpreter is absent.
                    (self.bin / "node").write_text("#!/missing/interpreter\n")
                before = self.snapshot()
                body, stderr = self.invoke("--check", expected=1)
                self.assertTrue(body["completed"] and body["configured"])
                self.assertEqual(body["state"], "configured")
                self.assertIsNone(body["runtime"])
                self.assertEqual(body["runtime_compatibility"]["state"], "unqualified")
                self.assertIn("Next:", stderr)
                self.assertIn("--check", body["runtime_compatibility"]["next_action"])
                self.assertEqual(before, self.snapshot())
                self.invoke(expected=2)
                self.assertEqual(before, self.snapshot())

    def test_runtime_free_disable_preserves_user_files_and_rejects_unsafe_cleanup(self):
        self.invoke()
        (self.bin / "pi").unlink()
        (self.bin / "node").unlink()
        extension = self.target / "extensions/loom-hem"
        index = extension / "index.ts"
        original = index.read_bytes()
        index.write_bytes(original + b"\n// user edit\n")
        foreign = extension / "user.txt"
        foreign.write_text("user file")
        before = self.snapshot()
        _, stderr = self.invoke("--disable", expected=2)
        self.assertIn("unknown Pi extension content retained", stderr)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(foreign.read_text(), "user file")
        foreign.unlink()  # Explicit user cleanup for this test-owned file only.
        body, stderr = self.invoke("--disable", expected=1)
        self.assertEqual(body["state"], "cleanup_required")
        self.assertIn(str(index), body["retained_paths"])
        self.assertIn("restore retained owned files", stderr)
        self.assertEqual(index.read_bytes(), original + b"\n// user edit\n")
        self.assertEqual(body["runtime_compatibility"]["state"], "not_checked")
        index.write_bytes(original)
        body, _ = self.invoke("--disable")
        self.assertEqual(body["state"], "disabled")
        self.assertIsNone(body["runtime"])
        before = self.snapshot()
        self.invoke("--disable")
        self.assertEqual(before, self.snapshot())

    def test_absent_runtime_check_with_absent_schema1_and_corrupt_state_is_passive(self):
        (self.bin / "pi").unlink()
        before = self.snapshot()
        body, _ = self.invoke("--check", expected=1)
        self.assertEqual(body["state"], "changes_required")
        self.assertEqual(body["runtime_compatibility"]["state"], "unqualified")
        self.assertEqual(before, self.snapshot())
        registry = Registry(self.root)
        with sqlite3.connect(registry.database) as db:
            db.execute("DROP TABLE pi_target")
            db.execute("PRAGMA user_version=1")
        before = self.snapshot()
        self.invoke("--check", expected=1)
        self.assertEqual(before, self.snapshot())
        registry.database.write_bytes(b"corrupt SQLite")
        before = self.snapshot()
        body, stderr = self.invoke("--check", expected=1)
        self.assertFalse(body["completed"])
        self.assertIn("state snapshot unavailable; inventory retained", stderr)
        self.assertEqual(before, self.snapshot())

    def test_disable_never_qualifies_runtime_and_stronger_refusals_are_preserved(self):
        self.invoke()
        args = hem_cli.parse_args(["pi", "configure", "--disable"])
        with patch.dict(os.environ, self.env), patch.object(hem_pi, "inspect_runtime", side_effect=AssertionError("runtime probe forbidden")):
            body = hem_pi_cli.run(args, Deadline.after())
        self.assertEqual(body["state"], "disabled")
        self.invoke()
        (self.bin / "pi").unlink()
        before = self.snapshot()
        self.invoke("--check", "--pi-dir", str(self.base / "other"), expected=2)
        self.invoke("--disable", "--pi-dir", str(self.base / "other"), expected=2)
        self.assertEqual(before, self.snapshot())
        args = hem_cli.parse_args(["pi", "configure", "--check"])
        with patch.dict(os.environ, self.env), patch.object(hem_pi, "_configure_owned", side_effect=UncertainOutcome("uncertain ownership")):
            with self.assertRaises(UncertainOutcome):
                hem_pi_cli.run(args, Deadline.after())

    def test_check_inaccessible_runtime_retains_passive_configuration_findings(self):
        self.invoke()
        args = hem_cli.parse_args(["pi", "configure", "--check"])
        before = self.snapshot()
        with patch.dict(os.environ, self.env), patch.object(hem_pi, "_metadata", side_effect=PermissionError("selected Pi metadata inaccessible")), redirect_stderr(io.StringIO()):
            body = hem_pi_cli.run(args, Deadline.after())
        self.assertEqual(body.exit_code, 1)
        self.assertTrue(body["configured"])
        self.assertIn("metadata inaccessible", body["runtime_compatibility"]["reason"])
        self.assertEqual(before, self.snapshot())

    def test_static_provider_collision_is_rejected_before_schema_or_ownership_writes(self):
        registry = Registry(self.root)
        for name in ("local", "stopped"):
            registry.create_server(name, saved_settings=dict(device="task", listen="127.0.0.1:1"))
        # Include the legacy schema: rejection must not migrate it or reserve a target.
        with sqlite3.connect(registry.database) as db:
            db.execute("DROP TABLE pi_target")
            db.execute("PRAGMA user_version=1")
        self.target.mkdir(mode=0o700)
        for provider in ("loom-hem-bG9jYWw", "loom-hem-c3RvcHBlZA"):
            with self.subTest(provider=provider):
                (self.target / "models.json").write_text(json.dumps(dict(providers={
                    provider: dict(apiKey="private-sentinel", models=[dict(id="private-model")])})))
                before = self.snapshot()
                for flags in ((), ("--check",)):
                    body, stderr = self.invoke(*flags, expected=2)
                    self.assertFalse(body["completed"])
                    self.assertIn("static provider IDs", stderr)
                    self.assertIn(provider, stderr)
                    self.assertIn("export/back up models.json", stderr)
                    self.assertNotIn("private-sentinel", stderr + json.dumps(body))
                    self.assertNotIn("private-model", stderr + json.dumps(body))
                    self.assertEqual(before, self.snapshot())

    def test_only_actual_provider_ids_conflict_and_disable_remains_repairable(self):
        Registry(self.root).create_server("local", saved_settings=dict(device="task", listen="127.0.0.1:1"))
        self.target.mkdir(mode=0o700)
        models = dict(providers={"loom-hem-bG9jYWw-neighbor": {}, "loom-hem-b3RoZXI": {},
            "other": dict(apiKey="private-sentinel"), "hrx-loom": dict(models=[dict(id="old-qwen")])})
        for name, value in (("models.json", json.dumps(models)), ("settings.json", '{"defaultModel":"old-qwen"}'),
                            ("auth.json", "private-sentinel")):
            (self.target / name).write_text(value)
        user = {p.name:p.read_bytes() for p in self.target.iterdir()}
        body, _ = self.invoke()
        self.assertTrue(body["configured"])
        self.assertEqual(body["legacy"][0]["ids"], ["old-qwen"])
        self.assertEqual(user, {name:(self.target / name).read_bytes() for name in user})
        models["providers"]["loom-hem-bG9jYWw"] = {}
        (self.target / "models.json").write_text(json.dumps(models))
        user["models.json"] = (self.target / "models.json").read_bytes()
        before = self.snapshot()
        self.invoke("--check", expected=2)
        self.assertEqual(before, self.snapshot())
        self.invoke(expected=2)
        self.assertEqual(before, self.snapshot())
        body, _ = self.invoke("--disable")
        self.assertEqual(body["state"], "disabled")
        self.assertFalse((self.target / MANIFEST_PATH).exists())
        self.assertEqual(user, {name:(self.target / name).read_bytes() for name in user})
        with sqlite3.connect(self.root / "control.sqlite3") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM pi_target").fetchone()[0], 0)

    def test_missing_and_ambiguous_launcher_and_dependency(self):
        (self.bin / "pi").unlink()
        before = self.snapshot()
        self.invoke(expected=2)
        self.assertEqual(before, self.snapshot())
        (self.bin / "pi").write_text("#!/bin/sh\nexit 99\n")
        (self.bin / "pi").chmod(0o755)
        before = self.snapshot()
        self.invoke(expected=2)
        self.assertEqual(before, self.snapshot())
        (self.bin / "pi").unlink()
        (self.bin / "pi").symlink_to(self.package / "dist/cli.js")
        self.metadata["bin"]["pi"] = "dist/index.js"
        self.write_metadata()
        self.invoke(expected=2)
        self.metadata["bin"]["pi"] = "dist/cli.js"
        self.write_metadata()
        (self.ai / "package.json").unlink()
        before = self.snapshot()
        self.invoke(expected=2)
        self.assertEqual(before, self.snapshot())

    def test_target_priority_canonical_alias_and_ownership_conflict(self):
        alternate = self.base / "alternate"
        alternate.mkdir(mode=0o700)
        alias = self.base / "alias"
        alias.symlink_to(alternate, target_is_directory=True)
        body, _ = self.invoke("--pi-dir", str(alias))
        self.assertEqual(body["target"], str(alternate))
        self.assertFalse(self.target.exists())
        before = self.snapshot()
        self.invoke(expected=2)
        self.assertEqual(before, self.snapshot())
        self.invoke("--disable", "--pi-dir", str(alias))

    def test_check_legacy_schema_and_drift_without_migration_or_locks(self):
        registry = Registry(self.root)
        with sqlite3.connect(registry.database) as db:
            db.execute("DROP TABLE pi_target")
            db.execute("PRAGMA user_version=1")
        before = self.snapshot()
        self.invoke("--check", expected=1)
        self.assertEqual(before, self.snapshot())
        with sqlite3.connect(registry.database) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 1)
        self.invoke()
        index = self.target / "extensions/loom-hem/index.ts"
        index.write_bytes(index.read_bytes() + b"\n// user edit\n")
        before = self.snapshot()
        self.invoke("--check", expected=1)
        self.assertEqual(before, self.snapshot())
        self.invoke(expected=2)
        self.assertEqual(before, self.snapshot())

    def test_setup_and_check_leave_native_discovery_to_runtime(self):
        requests = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(inner):
                requests.append(inner.path)
                inner.send_response(500)
                inner.end_headers()
            def log_message(*args):
                pass
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        Registry(self.root).create_server("native", saved_settings=dict(device="task", listen=f"127.0.0.1:{server.server_port}"))
        for args in ((), ("--check",)):
            body, stderr = self.invoke(*args)
            self.assertTrue(body["configured"])
            self.assertEqual(body["availability"], "not_observed")
            for removed in ("servers", "available", "availability_reason"):
                self.assertNotIn(removed, body)
            self.assertNotIn("curl ", stderr)
        self.assertEqual(requests, [])

    def test_direct_entry_qualifies_before_writes_and_check_stays_passive(self):
        self.node("22.18.0")
        before = self.snapshot()
        with patch.dict(os.environ, self.env):
            with self.assertRaisesRegex(InvalidInput, "Node"):
                hem_pi.configure_pi(self.root, self.target, Deadline.after())
            checked = hem_pi.configure_pi(self.root, self.target, Deadline.after(), check=True)
            self.assertEqual(checked["runtime_compatibility"]["state"], "unqualified")
            self.assertEqual(checked.exit_code, 1)
            disabled = hem_pi.configure_pi(self.root, self.target, Deadline.after(), disable=True)
            self.assertEqual(disabled["runtime_compatibility"]["state"], "not_checked")
        self.assertEqual(before, self.snapshot())
        self.node("22.22.1")
        with patch.dict(os.environ, self.env):
            result = hem_pi.configure_pi(self.root, self.target, Deadline.after())
        self.assertTrue(result["configured"])
        self.assertEqual(result["runtime"]["version"], "1.3.0")

    def test_fixed_dispatch_preserves_explicit_namespace_mode_and_deadline(self):
        from binding.python.hem_integrations import configure_integration
        deadline = Deadline.after()
        for mode in ("configure", "check", "disable"):
            with patch.object(hem_pi, "configure_pi", return_value={}) as owner:
                configure_integration("pi", self.root, deadline, mode=mode, pi_dir=self.target)
                owner.assert_called_once_with(self.root, self.target, deadline,
                    check=mode == "check", disable=mode == "disable")
        for integration, mode in (("other", "configure"), ("pi", "other")):
            with patch.object(hem_pi, "configure_pi") as owner:
                with self.assertRaises(InvalidInput):
                    configure_integration(integration, self.root, deadline, mode=mode)
                owner.assert_not_called()

    def test_assets_and_new_reservation_identity_are_selected_once(self):
        with patch.dict(os.environ, self.env), patch.object(hem_pi, "bundled_assets", wraps=hem_pi.bundled_assets) as assets, patch.object(hem_pi.uuid, "uuid4", wraps=hem_pi.uuid.uuid4) as identity:
            result = hem_pi.configure_pi(self.root, self.target, Deadline.after())
        self.assertTrue(result["configured"])
        assets.assert_called_once_with()
        identity.assert_called_once_with()

    def test_disable_after_snapshot_cannot_reuse_retired_generation(self):
        with patch.dict(os.environ, self.env):
            hem_pi.configure_pi(self.root, self.target, Deadline.after())
            old_identity = json.loads((self.target / MANIFEST_PATH).read_text())["integration_id"]
            snapshot = hem_pi.passive_snapshot

            def snapshot_then_disable(root, deadline):
                observed = snapshot(root, deadline)
                # Deterministically retire the observed generation before the
                # outer configure can acquire its mutation lock.
                with patch.object(hem_pi, "passive_snapshot", side_effect=snapshot):
                    disabled = hem_pi.configure_pi(root, self.target, deadline, disable=True)
                self.assertEqual(disabled["state"], "disabled")
                return observed

            with patch.object(hem_pi, "passive_snapshot", side_effect=snapshot_then_disable), patch.object(hem_pi, "set_target", wraps=hem_pi.set_target) as reserve:
                with self.assertRaisesRegex(InvalidInput, "reservation changed"):
                    hem_pi.configure_pi(self.root, self.target, Deadline.after())
                reserve.assert_not_called()
            self.assertFalse((self.target / MANIFEST_PATH).exists())
            self.assertIsNone(snapshot(self.root, Deadline.after())["pi_target"])
            fresh = hem_pi.configure_pi(self.root, self.target, Deadline.after())
            self.assertTrue(fresh["configured"])
            new_identity = json.loads((self.target / MANIFEST_PATH).read_text())["integration_id"]
            self.assertNotEqual(new_identity, old_identity)

    def test_malformed_metadata_fails_cleanly(self):
        for field in ("exports", "dependencies"):
            original = self.metadata[field]
            self.metadata[field] = []
            self.write_metadata()
            before = self.snapshot()
            self.invoke(expected=2)
            self.assertEqual(before, self.snapshot())
            self.metadata[field] = original
        self.write_metadata()
    def test_shared_projection_guard_runs_before_installer_writes(self):
        args = hem_cli.parse_args(["pi", "configure"])
        before = self.snapshot()
        with patch.dict(os.environ, self.env), patch.object(hem_pi, "project_endpoints", side_effect=InvalidInput("endpoint projection exceeds1MiB")), patch.object(hem_pi, "_configure_owned") as installer:
            with self.assertRaisesRegex(InvalidInput, "exceeds1MiB"):
                hem_pi_cli.run(args, Deadline.after())
            installer.assert_not_called()
        self.assertEqual(before, self.snapshot())

    def test_missing_packaged_source_is_known_failure_without_writes(self):
        args = hem_cli.parse_args(["pi", "configure"])
        before = self.snapshot()
        with patch.dict(os.environ,self.env), patch.object(hem_pi,"bundled_assets",side_effect=FileNotFoundError("missing")):
            with self.assertRaises(HemError) as raised:
                hem_pi_cli.run(args, Deadline.after())
        self.assertEqual(raised.exception.exit_code, 1)
        self.assertIs(raised.exception.accepted, False)
        self.assertEqual(before, self.snapshot())

    def test_public_dispatch_subprocess_preserves_accepted_pending_and_uncertain(self):
        # Faults originate inside real installer mutation; exercise production
        # CLI dispatch/compatibility/installer up to the failing reservation.
        program = """import sys
from unittest.mock import patch
from binding.python import hem_cli, hem_pi
from binding.python.hem_common import DeadlineExpired, UncertainOutcome
error = DeadlineExpired('accepted pending', accepted=True) if sys.argv[1]=='3' else UncertainOutcome('reservation uncertain')
with patch.object(hem_pi, 'set_target', side_effect=error):
    raise SystemExit(hem_cli.main(['pi','configure','--json']))
"""
        for expected in (3,4):
            with self.subTest(expected=expected):
                process = subprocess.run([sys.executable,'-c',program,str(expected)],
                    cwd=Path(__file__).resolve().parents[1],env=self.env,text=True,capture_output=True,timeout=10)
                self.assertEqual(process.returncode,expected,process.stderr)
                body = json.loads(process.stdout)
                self.assertFalse(body['completed'])
                self.assertIs(body['accepted'],True if expected==3 else None)
                self.assertIn('accepted: true' if expected==3 else 'accepted: unknown',process.stderr)

    def test_actual_run_preserves_error_acceptance_and_uncertainty(self):
        args = hem_cli.parse_args(["pi", "configure"])
        for error, code, accepted in ((HemError("rejected",accepted=False),1,False),
            (DeadlineExpired("pending",accepted=True),3,True), (UncertainOutcome("unknown"),4,None),
            (OSError("replace failed"),4,None)):
            with self.subTest(code=code), patch.dict(os.environ,self.env), patch.object(hem_pi,"inspect_runtime", return_value={}), patch.object(hem_pi,"_configure_owned",side_effect=error):
                with redirect_stderr(io.StringIO()), self.assertRaises(HemError) as raised:
                    hem_pi_cli.run(args, Deadline.after())
                self.assertEqual(raised.exception.exit_code,code)
                self.assertIs(raised.exception.accepted,accepted)


if __name__ == "__main__":
    unittest.main()
