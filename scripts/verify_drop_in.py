"""Verify an upstream-created database opens without vector dependencies.

Run with a Python that has sqlite-vec installed ONLY to create the upstream
fixture. The PerfectRecall subprocess explicitly blocks vector imports.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

CREATE = r'''
import json, sqlite3, hashlib
from pathlib import Path
from mnemosyne import Mnemosyne
from mnemosyne.core.beam import _vec_insert, EMBEDDING_DIM
items=[]
for bank in (None, 'existing-bank'):
    m=Mnemosyne(session_id='existing-session', bank=bank)
    mid=m.remember('Project Cedar uses PostgreSQL.', memory_type='fact')
    m.beam.conn.execute('INSERT INTO memory_embeddings(memory_id,embedding_json,model) VALUES (?,?,?)', (mid,'[0.1,0.2]','legacy-fixture'))
    m.beam.conn.commit()
    # A real vec0 table and shadow tables exercise the extension-free reopen.
    eid=m.beam.consolidate_to_episodic('Cedar previously used SQLite.', [])
    rowid=m.beam.conn.execute('SELECT rowid FROM episodic_memory WHERE id=?',(eid,)).fetchone()[0]
    _vec_insert(m.beam.conn,rowid,[1.0]+[0.0]*(EMBEDDING_DIM-1))
    m.beam.conn.commit()
    tables=[r[0] for r in m.beam.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vec_%' AND sql NOT LIKE '%VIRTUAL%'")]
    snapshot={t:[list(r) for r in m.beam.conn.execute('SELECT * FROM "'+t+'"')] for t in tables}
    # Byte values are encoded only for comparing synthetic fixture data.
    encode=lambda x:x.hex() if isinstance(x,bytes) else x
    snapshot={t:hashlib.sha256(json.dumps(rows,default=encode).encode()).hexdigest() for t,rows in snapshot.items()}
    items.append(dict(bank=bank,id=mid,path=str(m.db_path),vectors=snapshot))
    m.beam.conn.close();m.conn.close()
print(json.dumps(items))
'''
VERIFY = r'''
import importlib.abc,json,os,sys,hashlib
class BlockVectors(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','sqlite_vec','fastembed','onnxruntime'} or fullname=='mnemosyne.core.embeddings':
            raise AssertionError('Removed vector dependency imported: '+fullname)
sys.meta_path.insert(0,BlockVectors())
from perfectrecall import PerfectRecall
from mnemosyne.core import jev
class FakeJev:
    def snapshot(self): return dict(requests=0,input_tokens=0,output_tokens=0,cache_hits=0,resolved_model='fixture')
    def evaluate(self,state,questions,**kwargs):return {k:dict(type='noul',noul=.95) for k in questions}
    fanout=evaluate
jev.client=lambda:FakeJev()
for old in json.loads(os.environ['FIXTURE_DATA']):
    m=PerfectRecall(session_id='existing-session',bank=old['bank'])
    assert str(m.db_path)==old['path']
    assert m.get(old['id'])['content']=='Project Cedar uses PostgreSQL.'
    assert any(x['id']==old['id'] for x in m.recall('Cedar database', evidence_questions=['Does this memory name a Cedar database?']))
    assert m.beam.conn.execute('SELECT embedding_json FROM memory_embeddings WHERE memory_id=?',(old['id'],)).fetchone()[0]=='[0.1,0.2]'
    for table,rows in old['vectors'].items():
        now=[list(r) for r in m.beam.conn.execute('SELECT * FROM "'+table+'"')]
        now=hashlib.sha256(json.dumps(now,default=lambda x:x.hex() if isinstance(x,bytes) else x).encode()).hexdigest()
        assert now==rows,table
    from mnemosyne.dr.recovery import create_backup, restore_backup
    from pathlib import Path
    backup=create_backup(Path(m.db_path),Path(os.environ['HERMES_HOME'])/'backups')
    restored=Path(os.environ['HERMES_HOME'])/('restored-'+str(old['bank'])+'.db')
    result=restore_backup(Path(backup['backup_path']),restored)
    assert result['integrity_check'],result
    import sqlite3
    with sqlite3.connect(restored) as c:
        assert c.execute('SELECT content FROM working_memory WHERE id=?',(old['id'],)).fetchone()[0]=='Project Cedar uses PostgreSQL.'
    m.beam.conn.close();m.conn.close()
print(json.dumps(dict(status='passed',banks=2,legacy_vector_tables_preserved=True,blocked_vector_imports=True,backup_restore=True)))
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='perfectrecall-compat-') as tmp:
        env={k:v for k,v in os.environ.items() if not k.startswith(('MNEMOSYNE_','PERFECTRECALL_','JEVOSYNE_'))}
        env.update(HERMES_HOME=tmp,MNEMOSYNE_NO_EMBEDDINGS='1',MNEMOSYNE_WRITE_CLASSIFIER='off',MNEMOSYNE_LLM_ENABLED='0',PYTHONPATH=str(args.upstream_root.resolve()))
        source=subprocess.run([sys.executable,'-c',CREATE],env=env,cwd=tmp,text=True,capture_output=True)
        if source.returncode: raise RuntimeError(source.stderr)
        env['FIXTURE_DATA']=source.stdout.strip().splitlines()[-1]
        env['PYTHONPATH']=str(root)
        result=subprocess.run([sys.executable,'-c',VERIFY],env=env,cwd=tmp,text=True,capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        report=json.loads(result.stdout.strip().splitlines()[-1])
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
