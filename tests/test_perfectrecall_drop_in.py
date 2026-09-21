"""Existing database compatibility across a real process/package boundary."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(code, env):
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize('explicit_path', [False, True])
def test_existing_database_banks_and_configuration_are_used_in_place(tmp_path, explicit_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PERFECTRECALL_', 'MNEMOSYNE_'))}
    env.update(HERMES_HOME=str(tmp_path / 'hermes'), MNEMOSYNE_NO_EMBEDDINGS='1',
               MNEMOSYNE_WRITE_CLASSIFIER='off', MNEMOSYNE_PERSONA_FILE=str(tmp_path / 'persona.md'))
    if explicit_path:
        env['MNEMOSYNE_DATA_DIR'] = str(tmp_path / 'existing-custom-data')
    before = run('''
import json
from mnemosyne import Mnemosyne
items=[]
for bank in (None, 'existing-bank'):
    mem=Mnemosyne(session_id='existing-session', bank=bank)
    mid=mem.remember('Existing project uses PostgreSQL.', source='user')
    items.append(dict(bank=bank, id=mid, path=str(mem.db_path), content=mem.get(mid)['content']))
    mem.beam.conn.close()
    mem.conn.close()
print(json.dumps(items))
''', env)
    env['TEST_EXISTING_MEMORIES'] = json.dumps(before)
    after = run('''
import json, os
from perfectrecall import PerfectRecall, Mnemosyne, get, forget, update
from mnemosyne.core import jev
from mnemosyne.core.persona import DEFAULT_PERSONA_FILE
assert PerfectRecall is Mnemosyne
assert jev.enabled()
assert os.environ['MNEMOSYNE_WRITE_CLASSIFIER']=='off'
assert str(DEFAULT_PERSONA_FILE)==os.environ['MNEMOSYNE_PERSONA_FILE']
class FakeJev:
    def __init__(self): self.sent=[]
    def snapshot(self):
        return dict(requests=len(self.sent),input_tokens=0,output_tokens=0,cache_hits=0,resolved_model='test')
    def evaluate(self,state,questions,**kwargs):
        self.sent.append(state)
        return {key:dict(type='noul',noul=.99) for key in questions}
client=FakeJev()
jev.client=lambda:client
items=[]
for old in json.loads(os.environ['TEST_EXISTING_MEMORIES']):
    mem=PerfectRecall(session_id='existing-session', bank=old['bank'])
    hits=mem.recall('project database', evidence_questions=['Does this memory state the project database?'])
    assert [x['id'] for x in hits]==[old['id']]
    items.append(dict(bank=old['bank'],id=old['id'],path=str(mem.db_path),content=mem.get(old['id'])['content']))
    mem.beam.conn.close()
    mem.conn.close()
assert client.sent
print(json.dumps(items))
''', env)
    assert after == before
    expected_root = Path(env.get('MNEMOSYNE_DATA_DIR', str(tmp_path / 'hermes/mnemosyne/data')))
    assert expected_root in Path(after[0]['path']).parents
    assert not (tmp_path / '.perfectrecall').exists()


def test_public_exports_match_upstream_and_module_cli_works(tmp_path):
    import perfectrecall
    import mnemosyne
    assert set(mnemosyne.__all__).issubset(perfectrecall.__all__)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PERFECTRECALL_', 'MNEMOSYNE_'))}
    env['HERMES_HOME'] = str(tmp_path / 'hermes')
    for module in ('perfectrecall', 'perfectrecall.cli'):
        result = subprocess.run([sys.executable, '-m', module, 'jev-status'], env=env,
                                cwd=ROOT, text=True, capture_output=True, check=True)
        status = json.loads(result.stdout)
        assert status['backend'] == 'jev'
        assert Path(status['data_dir']) == tmp_path / 'hermes/mnemosyne/data'
    assert not (tmp_path / 'hermes').exists()  # Status must not create a fresh database.
