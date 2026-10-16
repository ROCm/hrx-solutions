"""Real filesystem/SQLite crash boundaries and process writer contention."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from binding.python import hem_pi, hem_pi_files as files
from binding.python.hem_common import Deadline, InvalidInput, UncertainOutcome
from binding.python.hem_pi_assets import bundled_assets
from binding.python.hem_pi_contracts import MANIFEST_PATH, ENDPOINT_PATH
from binding.python.hem_pi_state import passive_snapshot, get_target, set_target
from binding.python.hem_state import Registry


def projection(root, target, servers):
    # File transaction fixtures use zero servers; pure live projection belongs to
    # hem_pi_endpoints tests. Combined gates exercise the production projector.
    return dict(schema_version=1, integration_id=target["integration_id"], hem_root=str(root), servers=[])


def projector_fixture():
    module = types.ModuleType("binding.python.hem_pi_endpoints")
    module.project_endpoints = projection
    return patch.dict(sys.modules, {module.__name__: module})


class InstallerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root, self.target = self.base / "root", self.base / "pi"
        self.assets = bundled_assets()
        fixture = projector_fixture()
        fixture.start()
        self.addCleanup(fixture.stop)

    def run_pi(self, **kwargs):
        return hem_pi.configure_pi(self.root, self.target, Deadline.after(5), **kwargs)

    def snapshot(self):
        return {str(p.relative_to(self.base)): (p.lstat().st_mode, p.lstat().st_mtime_ns,
                    p.read_bytes() if p.is_file() and not p.is_symlink() else None)
                for p in self.base.rglob("*")}

    def assert_configured(self):
        value = files.inspect_target(self.target, self.root)
        self.assertTrue(value["configured"], value)
        self.assertFalse(value["pending"])
        self.assertEqual(value["integration_id"], passive_snapshot(self.root, Deadline.after(5))["pi_target"]["integration_id"])

    def test_check_absent_has_no_files_or_registry_initialization(self):
        with patch.object(hem_pi, "Registry", side_effect=AssertionError("must be passive")):
            result = self.run_pi(check=True)
        self.assertEqual((result.exit_code, result["state"], result["changed"]), (1, "changes_required", False))
        self.assertEqual(self.snapshot(), {})

    def test_install_idempotence_check_disable_preserves_user_files(self):
        self.target.mkdir(mode=0o700)
        settings = self.target / "settings.json"
        settings.write_text('{"untouched":true}')
        first = self.run_pi()
        self.assertTrue(first["configured"])
        self.assert_configured()
        before = self.snapshot()
        self.assertFalse(self.run_pi()["changed"])
        self.assertEqual(self.run_pi(check=True).exit_code, 0)
        # SQLite opens may update the DB mtime, but managed Pi bytes stay stable.
        after = self.snapshot()
        self.assertEqual({k:v for k,v in before.items() if k.startswith("pi/")}, {k:v for k,v in after.items() if k.startswith("pi/")})
        result = self.run_pi(disable=True)
        self.assertEqual(result["state"], "disabled")
        self.assertIsNone(passive_snapshot(self.root, Deadline.after(5))["pi_target"])
        self.assertFalse((self.target / MANIFEST_PATH).exists())
        self.assertEqual(settings.read_text(), '{"untouched":true}')
        self.assertEqual(self.run_pi(disable=True)["changed"], False)

    def test_missing_disable_does_not_initialize_state(self):
        self.assertEqual(self.run_pi(disable=True)["state"], "disabled")
        self.assertEqual(self.snapshot(), {})

    def test_reservation_durable_before_any_pending_and_reused(self):
        original = hem_pi._prepare
        identity = []
        def crash(root, target, reserved, *args):
            identity.append(reserved)
            self.assertEqual(get_target(Registry(root)), reserved)
            self.assertFalse((target / files.PENDING_PATH).exists())
            raise RuntimeError("crash after reservation")
        with patch.object(hem_pi, "_prepare", side_effect=crash):
            with self.assertRaises(RuntimeError): self.run_pi()
        other = self.base / "other"
        with self.assertRaises(InvalidInput):
            hem_pi.configure_pi(self.root, other, Deadline.after(5))
        self.assertFalse(other.exists())
        self.run_pi()
        self.assert_configured()
        self.assertEqual(get_target(Registry(self.root)), identity[0])

    def test_reservation_only_disable_missing_target(self):
        registry = Registry(self.root)
        import uuid
        set_target(registry, dict(target=str(self.target), integration_id=str(uuid.uuid4())))
        self.assertEqual(self.run_pi(disable=True)["state"], "disabled")
        self.assertIsNone(get_target(registry))

    def test_each_replace_crash_recovers_without_new_identity(self):
        # plan, endpoint, six artifacts, authoritative manifest.
        for index in range(1, 10):
            with self.subTest(index=index), tempfile.TemporaryDirectory(dir=self.base) as tmp:
                root, target = Path(tmp)/"root", Path(tmp)/"pi"
                real = files.atomic_replace
                calls = 0
                def fault(*args, **kwargs):
                    nonlocal calls
                    real(*args, **kwargs)
                    calls += 1
                    if calls == index:
                        raise RuntimeError("power loss after replacement")
                with patch.object(files, "atomic_replace", side_effect=fault):
                    with self.assertRaises(RuntimeError):
                        hem_pi.configure_pi(root, target, Deadline.after(5))
                reserved = get_target(Registry(root))
                self.assertFalse(hem_pi.configure_pi(root, target, Deadline.after(5), check=True)["configured"])
                hem_pi.configure_pi(root, target, Deadline.after(5))
                self.assertTrue(files.inspect_target(target, root)["configured"])
                self.assertEqual(get_target(Registry(root)), reserved)

    def test_torn_actual_temporary_writes_recover_same_identity(self):
        for mode, suffix in (("configure", "plan.json"), ("configure", "index.ts"),
                             ("configure", "config.json"), ("disable", "plan.json"),
                             ("disable", "config.json")):
            for length in (0, 5):
                with self.subTest(mode=mode,suffix=suffix,length=length), tempfile.TemporaryDirectory(dir=self.base) as tmp:
                    root,target = Path(tmp)/"root",Path(tmp)/"pi"
                    if mode == "disable": hem_pi.configure_pi(root,target,Deadline.after(5))
                    real = os.fdopen
                    fired = []
                    class InterruptedWriter:
                        def __init__(self, stream): self.stream = stream
                        def __enter__(self): return self
                        def __exit__(self,*args): self.stream.close()
                        def write(inner, data):
                            # Real partial bytes become durable; interruption is
                            # before atomic_replace returns or replaces a file.
                            inner.stream.write(data[:length])
                            inner.stream.flush()
                            os.fsync(inner.stream.fileno())
                            fired.append(True)
                            raise RuntimeError("interrupted actual temp write")
                    def fdopen(fd, openmode, **kwargs):
                        destination = os.readlink(f"/proc/self/fd/{fd}")
                        stream = real(fd,openmode,**kwargs)
                        if openmode == "wb" and destination.endswith("."+suffix+".loom-hem-tmp") and not fired:
                            if suffix == "plan.json":
                                marker = target / files.PREPARING_PATH / mode
                                self.assertTrue(marker.is_dir())
                                self.assertIsNotNone(get_target(Registry(root)))
                            return InterruptedWriter(stream)
                        return stream
                    with patch.object(files.os,"fdopen",side_effect=fdopen):
                        with self.assertRaises(RuntimeError):
                            hem_pi.configure_pi(root,target,Deadline.after(5),disable=mode=="disable")
                    identity = get_target(Registry(root))
                    self.assertTrue(fired)
                    # Recovery follows durable intent even when the caller asks
                    # configure after an interrupted disable preparation.
                    result = hem_pi.configure_pi(root,target,Deadline.after(5))
                    if mode == "disable":
                        self.assertEqual(result["state"],"disabled")
                        self.assertIsNone(get_target(Registry(root)))
                    else:
                        self.assertTrue(result["configured"])
                        self.assertEqual(get_target(Registry(root)),identity)

    def test_preparation_marker_creation_crash_recovers(self):
        real = files.atomic_replace
        def fault(path,*args,**kwargs):
            if Path(path).name == "plan.json":
                marker = Path(path).parent / "configure"
                self.assertTrue(marker.is_dir())
                raise RuntimeError("crash after durable mode marker")
            return real(path,*args,**kwargs)
        with patch.object(files,"atomic_replace",side_effect=fault):
            with self.assertRaises(RuntimeError): self.run_pi()
        identity = get_target(Registry(self.root))
        self.run_pi()
        self.assertEqual(get_target(Registry(self.root)),identity)
        self.assert_configured()

    def test_crash_between_mode_marker_and_fingerprint_recovers(self):
        for mode in ("configure","disable"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir=self.base) as tmp:
                root,target=Path(tmp)/"root",Path(tmp)/"pi"
                if mode=="disable": hem_pi.configure_pi(root,target,Deadline.after(5))
                real=files.sync_dir
                def fault(path):
                    real(path)
                    preparing=target/files.PREPARING_PATH
                    if Path(path)==preparing and (preparing/mode).is_dir() and not list((preparing/mode).iterdir()):
                        raise RuntimeError("crash before fingerprint creation")
                with patch.object(files,"sync_dir",side_effect=fault):
                    with self.assertRaises(RuntimeError): hem_pi.configure_pi(root,target,Deadline.after(5),disable=mode=="disable")
                identity=get_target(Registry(root))
                result=hem_pi.configure_pi(root,target,Deadline.after(5))
                if mode=="disable":
                    self.assertEqual(result["state"],"disabled")
                else:
                    self.assertTrue(result["configured"])
                    self.assertEqual(get_target(Registry(root)),identity)

    def test_full_temporary_fsync_and_prerename_crashes_recover(self):
        for suffix in ("plan.json","index.ts"):
            for boundary in ("fsync", "replace"):
                with self.subTest(suffix=suffix,boundary=boundary), tempfile.TemporaryDirectory(dir=self.base) as tmp:
                    root,target = Path(tmp)/"root",Path(tmp)/"pi"
                    real_sync,real_replace = os.fsync,os.replace
                    fired = []
                    def sync(fd):
                        path = os.readlink(f"/proc/self/fd/{fd}")
                        if boundary=="fsync" and path.endswith("."+suffix+".loom-hem-tmp") and not fired:
                            fired.append(True)
                            raise RuntimeError("power loss at fsync")
                        return real_sync(fd)
                    def replace(source,destination):
                        if boundary=="replace" and str(source).endswith("."+suffix+".loom-hem-tmp") and not fired:
                            fired.append(True)
                            raise RuntimeError("power loss before rename")
                        return real_replace(source,destination)
                    with patch.object(files.os,"fsync",side_effect=sync), patch.object(files.os,"replace",side_effect=replace):
                        with self.assertRaises(RuntimeError): hem_pi.configure_pi(root,target,Deadline.after(5))
                    identity=get_target(Registry(root))
                    hem_pi.configure_pi(root,target,Deadline.after(5))
                    self.assertEqual(get_target(Registry(root)),identity)
                    self.assertTrue(files.inspect_target(target,root)["configured"])

    def test_foreign_temporary_bytes_and_markers_preserved(self):
        for stage in ("preparing","pending"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory(dir=self.base) as tmp:
                root,target=Path(tmp)/"root",Path(tmp)/"pi"
                real=files.atomic_replace
                def fault(path,data,**kwargs):
                    if (stage=="preparing" and Path(path).name=="plan.json") or (stage=="pending" and Path(path).name=="index.ts"):
                        temporary=files.staged_path(path)
                        temporary.write_bytes(b"foreign third bytes")
                        temporary.chmod(0o600)
                        raise RuntimeError("crash with conflicting temporary")
                    return real(path,data,**kwargs)
                with patch.object(files,"atomic_replace",side_effect=fault):
                    with self.assertRaises(RuntimeError): hem_pi.configure_pi(root,target,Deadline.after(5))
                before={p:p.read_bytes() for p in target.rglob("*.loom-hem-tmp")}
                self.assertTrue(before)
                with self.assertRaises(InvalidInput): hem_pi.configure_pi(root,target,Deadline.after(5))
                self.assertEqual(before,{p:p.read_bytes() for p in before})
        with patch.object(files,"atomic_replace",side_effect=RuntimeError("crash")):
            with self.assertRaises(RuntimeError): self.run_pi()
        marker=self.target/files.PREPARING_PATH/"configure"
        foreign=marker/"foreign"
        foreign.write_bytes(b"keep")
        with self.assertRaises(InvalidInput): self.run_pi()
        self.assertEqual(foreign.read_bytes(),b"keep")

    def test_changed_inputs_do_not_reconstruct_partial_preparation(self):
        real=files.atomic_replace
        def fault(path,data,**kwargs):
            if Path(path).name=="plan.json":
                temporary=files.staged_path(path)
                temporary.write_bytes(data[:5])
                temporary.chmod(0o600)
                raise RuntimeError("torn preparation")
            return real(path,data,**kwargs)
        with patch.object(files,"atomic_replace",side_effect=fault):
            with self.assertRaises(RuntimeError): self.run_pi()
        before=self.snapshot()
        changed=dict(self.assets)
        changed["extensions/loom-hem/discovery.ts"]+=b"\n// different wheel\n"
        with patch.object(hem_pi,"bundled_assets",return_value=changed):
            with self.assertRaisesRegex(InvalidInput,"inputs changed"): self.run_pi()
        self.assertEqual({k:v for k,v in before.items() if k.startswith("pi/")}, {k:v for k,v in self.snapshot().items() if k.startswith("pi/")})
        self.run_pi()
        self.assert_configured()

    def test_unknown_dedicated_package_content_refuses_before_writes(self):
        for installed in (False,True):
            with self.subTest(installed=installed),tempfile.TemporaryDirectory(dir=self.base) as tmp:
                root,target=Path(tmp)/"root",Path(tmp)/"pi"
                if installed: hem_pi.configure_pi(root,target,Deadline.after(5))
                else: (target/"extensions/loom-hem").mkdir(parents=True,mode=0o700)
                # mkdir parents follow the process umask, so make target private.
                for component in (target,target/"extensions",target/"extensions/loom-hem"): component.chmod(0o700)
                extra=target/"extensions/loom-hem/custom.ts"
                extra.write_bytes(b"user code")
                before={p:p.read_bytes() for p in target.rglob("*") if p.is_file()}
                with self.assertRaisesRegex(InvalidInput,"unknown Pi extension"):
                    hem_pi.configure_pi(root,target,Deadline.after(5))
                self.assertEqual(before,{p:p.read_bytes() for p in before})
                if not installed: self.assertFalse(root.exists())

    def test_pending_third_bytes_refused_without_overwrite(self):
        real = files.atomic_replace
        def fault(path, *args, **kwargs):
            real(path, *args, **kwargs)
            if str(path).endswith("index.ts"):
                raise RuntimeError("crash")
        with patch.object(files, "atomic_replace", side_effect=fault):
            with self.assertRaises(RuntimeError): self.run_pi()
        edited = self.target / "extensions/loom-hem/index.ts"
        edited.write_bytes(b"user edit")
        with self.assertRaises(InvalidInput): self.run_pi()
        self.assertEqual(edited.read_bytes(), b"user edit")
        self.assertTrue((self.target / files.PENDING_PATH).exists())

    def test_disable_retains_edits_and_pointer_until_restored(self):
        self.run_pi()
        edited = self.target / "extensions/loom-hem/index.ts"
        original = edited.read_bytes()
        edited.write_bytes(b"user edit")
        result = self.run_pi(disable=True)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result["retained_paths"], [str(edited)])
        self.assertFalse(json.loads((self.target / MANIFEST_PATH).read_text())["enabled"])
        self.assertIsNotNone(get_target(Registry(self.root)))
        self.assertEqual(edited.read_bytes(), b"user edit")
        edited.write_bytes(original)
        self.assertEqual(self.run_pi(disable=True)["state"], "disabled")
        self.assertIsNone(get_target(Registry(self.root)))

    def test_each_disable_removal_crash_retains_pointer_and_recovers(self):
        for index in range(1, 10):
            with self.subTest(index=index), tempfile.TemporaryDirectory(dir=self.base) as tmp:
                root, target = Path(tmp)/"root", Path(tmp)/"pi"
                hem_pi.configure_pi(root, target, Deadline.after(5))
                real = files.remove_owned
                calls = 0
                def fault(*args, **kwargs):
                    nonlocal calls
                    real(*args, **kwargs)
                    calls += 1
                    if calls == index: raise RuntimeError("power loss after removal")
                with patch.object(files, "remove_owned", side_effect=fault):
                    with self.assertRaises(RuntimeError): hem_pi.configure_pi(root,target,Deadline.after(5),disable=True)
                self.assertIsNotNone(get_target(Registry(root)))
                hem_pi.configure_pi(root, target, Deadline.after(5), disable=True)
                self.assertIsNone(get_target(Registry(root)))

    def test_foreign_manifest_future_schema_and_symlink_never_overwritten(self):
        self.run_pi()
        manifest = self.target / MANIFEST_PATH
        old = manifest.read_bytes()
        changed = json.loads(old)
        changed["schema_version"] = 999
        manifest.write_text(json.dumps(changed))
        before = manifest.read_bytes()
        with self.assertRaises(InvalidInput): self.run_pi()
        self.assertEqual(manifest.read_bytes(), before)
        manifest.write_bytes(old)
        asset = self.target / "extensions/loom-hem/index.ts"
        asset.unlink()
        external = self.base / "external"
        external.write_bytes(b"preserve")
        asset.symlink_to(external)
        with self.assertRaises(InvalidInput): self.run_pi()
        self.assertEqual(external.read_bytes(), b"preserve")

    def test_upgrade_uses_old_manifest_digests_and_preserves_identity(self):
        self.run_pi()
        identity = get_target(Registry(self.root))
        upgraded = dict(self.assets)
        upgraded["extensions/loom-hem/discovery.ts"] += b"\n// next release\n"
        with patch.object(hem_pi, "bundled_assets", return_value=upgraded):
            self.assertTrue(self.run_pi()["changed"])
            self.assertFalse(self.run_pi()["changed"])
        self.assertEqual(get_target(Registry(self.root)), identity)
        self.assertEqual((self.target / "extensions/loom-hem/discovery.ts").read_bytes(), upgraded["extensions/loom-hem/discovery.ts"])

    def test_fixed_pending_plan_rejects_extra_files(self):
        with patch.object(hem_pi, "_resume", side_effect=RuntimeError("crash")):
            with self.assertRaises(RuntimeError): self.run_pi()
        extra = self.target / files.PENDING_PATH / "foreign"
        extra.write_bytes(b"keep")
        with self.assertRaises(InvalidInput): self.run_pi()
        self.assertEqual(extra.read_bytes(), b"keep")

    def test_private_component_rejection_precedes_registration(self):
        self.target.mkdir(mode=0o700)
        (self.target / "extensions").mkdir(mode=0o777)
        (self.target / "extensions").chmod(0o777)
        with self.assertRaises(InvalidInput): self.run_pi()
        self.assertFalse(self.root.exists())

    def test_atomic_directory_sync_failure_reports_uncertain(self):
        self.target.mkdir(mode=0o700)
        path = self.target / "owned"
        with patch.object(files, "sync_dir", side_effect=OSError("fsync failed")):
            with self.assertRaises(UncertainOutcome):
                files.atomic_replace(path,b"new",expected=None,deadline=Deadline.after(5))
        self.assertEqual(path.read_bytes(), b"new")

    def test_concurrent_processes_same_target_and_conflicting_roots(self):
        code = """from pathlib import Path
import sys
from tests.hem_pi_installer_test import projector_fixture
from binding.python.hem_pi import configure_pi
from binding.python.hem_common import Deadline, InvalidInput
try:
 with projector_fixture():
  r=configure_pi(Path(sys.argv[1]),Path(sys.argv[2]),Deadline.after(5))
 print(r['state'])
except InvalidInput as e:
 print(str(e)); sys.exit(2)
"""
        env = dict(os.environ, PYTHONNOUSERSITE="1", LOOM_SERVE_DIR=str(self.root))
        env.pop("PYTHONPATH",None); env.pop("PYTHONHOME",None)
        args = [sys.executable,"-c",code,str(self.root),str(self.target)]
        workers = [subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
        for worker in workers:
            out, err = worker.communicate(timeout=10)
            self.assertEqual(worker.returncode,0,(out,err))
        self.assert_configured()
        other = self.base / "other-root"
        result = subprocess.run([sys.executable,"-c",code,str(other),str(self.target)],env=env,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,2,result.stderr)
        self.assert_configured()


if __name__ == "__main__": unittest.main()
