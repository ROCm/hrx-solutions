"""Real SQLite migration, passive reads and cross-process Pi publication locks."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

from binding.python import hem_state
from binding.python.hem_common import Deadline, DeadlineExpired, HemError, InvalidInput
from binding.python.hem_state import Registry
from binding.python.hem_pi_state import (passive_snapshot, integration_lock,
                                        target_lock, get_target, set_target, clear_target)


class PiStateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'state'
        self.target = Path(self.temp.name) / 'pi'
        self.identity = dict(target=str(self.target), integration_id=str(uuid.uuid4()))

    def legacy(self):
        self.root.mkdir(mode=0o700)
        db = sqlite3.connect(self.root / 'control.sqlite3')
        for statement in hem_state._SCHEMA:
            db.execute(statement)
        db.execute('PRAGMA user_version=1')
        model = dict(id='m',assets=[],provenance={'retained':True},directory='/outside',
                     pipeline='qwen-08b', validation={'valid':True}, pending=None)
        server = dict(id='s',launch_token='launch', saved_settings={'listen':'127.0.0.1'},
                      phase='running',pending=None,process={'pid':123},build={'identity':'retained'})
        instance = dict(server_id='s',launch_token='launch',name='i',model_id='m',settings={},
                        document='/retained',dependencies=[],incarnation='native-id',pending=None)
        db.execute('INSERT INTO models VALUES(?,7,?)', ('m',json.dumps(model)))
        db.execute('INSERT INTO servers VALUES(?,?,9,?)', ('s','launch',json.dumps(server)))
        db.execute('INSERT INTO instances VALUES(?,?,?,?,11,?)', ('s','launch','i','m',json.dumps(instance)))
        db.commit()
        db.close()

    def tree(self):
        return {str(p.relative_to(self.root)): (p.stat().st_mode,p.stat().st_mtime_ns,
                    p.read_bytes() if p.is_file() else None)
                for p in [self.root,*self.root.rglob('*')]}

    def child(self, code, *args):
        env = dict(os.environ, PYTHONNOUSERSITE='1', LOOM_SERVE_DIR=str(self.root))
        env.pop('PYTHONPATH',None); env.pop('PYTHONHOME',None)
        return subprocess.run([sys.executable,'-c',code,str(self.root),*map(str,args)],
                              env=env,capture_output=True,text=True,timeout=5)

    def test_absent_and_legacy_read_are_passive_even_read_only(self):
        with patch.object(hem_state.Registry,'__init__',side_effect=AssertionError('initialized')):
            self.assertEqual(passive_snapshot(self.root,Deadline.after()),
                             dict(schema_version=None,pi_target=None,servers=[]))
        self.assertFalse(self.root.exists())
        self.legacy()
        self.root.chmod(0o500)
        (self.root/'control.sqlite3').chmod(0o400)
        before=self.tree()
        with patch.object(hem_state.Registry,'__init__',side_effect=AssertionError('initialized')):
            snapshot=passive_snapshot(self.root,Deadline.after())
        self.assertEqual(snapshot['schema_version'],1)
        self.assertEqual(snapshot['servers'][0]['revision'],9)
        self.assertEqual(self.tree(),before)

    def test_migration_preserves_exact_inventory_and_constraints(self):
        self.legacy()
        with sqlite3.connect(self.root/'control.sqlite3') as db:
            before={table:db.execute(f'SELECT * FROM {table}').fetchall()
                    for table in ('models','servers','instances')}
        registry=Registry(self.root)
        with sqlite3.connect(registry.database) as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0],2)
            for table,rows in before.items():
                self.assertEqual(db.execute(f'SELECT * FROM {table}').fetchall(),rows)
        self.assertEqual(registry.get_instance('s','launch','i')['revision'],11)
        self.assertIsNone(get_target(registry))

    def test_migration_crash_rolls_back_table_and_version(self):
        self.legacy()
        code='''from binding.python import hem_state
import os,sys
def crash(db):
    db.execute(hem_state._PI_SCHEMA)
    db.execute('PRAGMA user_version=2')
    os._exit(23)
hem_state._migrate=crash
hem_state.Registry(sys.argv[1])
'''
        result=self.child(code)
        self.assertEqual(result.returncode,23,result.stderr)
        with sqlite3.connect(self.root/'control.sqlite3') as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0],1)
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='pi_target'").fetchone())
        self.assertEqual(Registry(self.root).get_server('s')['revision'],9)

    def test_unknown_schema_and_corrupt_rows_are_not_reset(self):
        registry=Registry(self.root)
        registry.create_server('s',saved_settings={})
        with sqlite3.connect(registry.database) as db:
            db.execute('PRAGMA user_version=99')
        before=self.tree()
        with self.assertRaisesRegex(HemError,'unsupported'):
            passive_snapshot(self.root,Deadline.after())
        self.assertEqual(self.tree(),before)
        with sqlite3.connect(registry.database) as db:
            db.execute('PRAGMA user_version=2')
            db.execute("UPDATE servers SET data='{}'")
        with self.assertRaisesRegex(HemError,'corrupt'):
            passive_snapshot(self.root,Deadline.after())

    def test_migration_refuses_unknown_tables_or_corrupt_legacy_records(self):
        self.legacy()
        with sqlite3.connect(self.root/'control.sqlite3') as db:
            db.execute('CREATE TABLE unrecognized (data TEXT)')
        with self.assertRaisesRegex(HemError,'schema'):
            Registry(self.root)
        with sqlite3.connect(self.root/'control.sqlite3') as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0],1)
            db.execute('DROP TABLE unrecognized')
            db.execute("UPDATE models SET data='{}'")
        with self.assertRaisesRegex(HemError,'corrupt'):
            Registry(self.root)
        with sqlite3.connect(self.root/'control.sqlite3') as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0],1)
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='pi_target'").fetchone())

    def test_passive_snapshot_never_acquires_publication_lock(self):
        Registry(self.root)
        with integration_lock(self.root,Deadline.after()):
            code='''from binding.python.hem_pi_state import passive_snapshot
from binding.python.hem_common import Deadline
from pathlib import Path
import sys
print(passive_snapshot(Path(sys.argv[1]),Deadline.after(.1))['schema_version'])
'''
            result=self.child(code)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.strip(),'2')

    def test_singleton_is_idempotent_conflict_safe_and_compare_deleted(self):
        registry=Registry(self.root)
        set_target(registry,self.identity)
        set_target(registry,self.identity)
        for conflicting in (dict(self.identity,target=str(self.target.parent/'other')),
                            dict(self.identity,integration_id=str(uuid.uuid4()))):
            with self.assertRaisesRegex(InvalidInput,'conflict'):
                set_target(registry,conflicting)
        with self.assertRaisesRegex(InvalidInput,'changed'):
            clear_target(registry,integration_id=str(uuid.uuid4()))
        self.assertEqual(get_target(registry),self.identity)
        self.assertFalse(self.target.exists()) # reservation precedes target artifact mutation
        self.assertEqual(passive_snapshot(self.root,Deadline.after())['pi_target'],self.identity)
        clear_target(registry,integration_id=self.identity['integration_id'])
        clear_target(registry,integration_id=self.identity['integration_id'])
        self.assertIsNone(get_target(registry))

    def test_reservation_survives_crash_without_claiming_installation(self):
        Registry(self.root)
        code='''from binding.python.hem_state import Registry
from binding.python.hem_pi_state import set_target
import json,os,sys
set_target(Registry(sys.argv[1]),json.loads(sys.argv[2]))
os._exit(24)
'''
        result=self.child(code,json.dumps(self.identity))
        self.assertEqual(result.returncode,24,result.stderr)
        self.assertEqual(get_target(Registry(self.root)),self.identity)
        self.assertFalse(self.target.exists())

    def test_atomic_concurrent_reservations_have_one_winner(self):
        registry=Registry(self.root)
        code='''from binding.python.hem_state import Registry
from binding.python.hem_pi_state import set_target
from binding.python.hem_common import InvalidInput
import json,sys
try:
    set_target(Registry(sys.argv[1]),json.loads(sys.argv[2]))
    print('reserved')
except InvalidInput:
    print('conflict')
'''
        processes=[]
        for _ in range(4):
            identity=dict(self.identity,integration_id=str(uuid.uuid4()))
            processes.append(subprocess.Popen([sys.executable,'-c',code,str(self.root),json.dumps(identity)],
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
        results=[]
        for process in processes:
            out,err=process.communicate(timeout=5)
            self.assertEqual(process.returncode,0,err)
            results.append(out.strip())
        self.assertEqual(results.count('reserved'),1)
        self.assertEqual(results.count('conflict'),3)
        with sqlite3.connect(registry.database) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM pi_target').fetchone()[0],1)

    def test_order_reentrancy_deadline_and_crash_release(self):
        registry=Registry(self.root)
        deadline=Deadline.after()
        with registry.lock('model','m'),registry.lock('server','s'),integration_lock(self.root,deadline):
            with target_lock(self.target,deadline):
                for lock in (registry.lock('server','other'),integration_lock(self.root,deadline),
                             target_lock(self.target,deadline)):
                    with self.assertRaises(HemError):
                        with lock: pass
            code='''from binding.python.hem_pi_state import integration_lock
from binding.python.hem_common import Deadline,DeadlineExpired
import sys
try:
    with integration_lock(__import__('pathlib').Path(sys.argv[1]),Deadline.after(.05)): pass
except DeadlineExpired: print('timeout')
'''
            result=self.child(code)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.strip(),'timeout')
        code='''from binding.python.hem_pi_state import integration_lock
from binding.python.hem_common import Deadline
from pathlib import Path
import sys,os
with integration_lock(Path(sys.argv[1]),Deadline.after()): os._exit(25)
'''
        self.assertEqual(self.child(code).returncode,25)
        with integration_lock(self.root,Deadline.after(.1)): pass

    def test_passive_reader_during_active_transaction_and_wal(self):
        registry=Registry(self.root)
        registry.create_server('old',saved_settings={})
        with self.assertRaises(RuntimeError):
            with registry._transaction() as writer:
                writer.execute('DELETE FROM servers')
                self.assertEqual([x['id'] for x in passive_snapshot(self.root,Deadline.after())['servers']],['old'])
                raise RuntimeError('rollback fixture')
        with sqlite3.connect(registry.database) as writer:
            writer.execute('PRAGMA journal_mode=WAL')
            writer.execute('UPDATE servers SET revision=revision+1')
            writer.commit()
            before={p.name:p.read_bytes() for p in self.root.glob('control.sqlite3*') if not p.name.endswith('-shm')}
            snapshot=passive_snapshot(self.root,Deadline.after())
            self.assertEqual(snapshot['servers'][0]['revision'],2)
            after={p.name:p.read_bytes() for p in self.root.glob('control.sqlite3*') if not p.name.endswith('-shm')}
            self.assertEqual(before,after)

    def test_unsafe_target_and_invalid_identity_refused(self):
        registry=Registry(self.root)
        for identity in (dict(self.identity,integration_id='bad'),dict(self.identity,target='relative'),
                         dict(self.identity,target=str(self.target/'..'/'pi'))):
            with self.assertRaises(InvalidInput): set_target(registry,identity)
        self.target.mkdir(mode=0o755)
        self.target.chmod(0o755)  # Exercise unsafe permissions even under umask 077.
        with self.assertRaises(HemError):
            with target_lock(self.target,Deadline.after()): pass
        self.assertFalse((self.target/'loom-hem.lock').exists())


if __name__=='__main__': unittest.main()
