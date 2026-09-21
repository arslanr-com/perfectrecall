"""A write at capacity must survive timestamp ties instead of evicting itself."""
from datetime import datetime, timezone

from mnemosyne.core.beam import BeamMemory


def test_trim_keeps_newest_subsecond_then_insertion_order(tmp_path, monkeypatch):
    from mnemosyne.core import beam as module
    monkeypatch.setattr(module, 'WORKING_MEMORY_MAX_ITEMS', 2)
    memory = BeamMemory(session_id='retention', db_path=tmp_path / 'memory.db')
    stamp = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None).isoformat()
    try:
        memory.conn.executemany(
            'INSERT INTO working_memory(id,content,session_id,timestamp) VALUES (?,?,?,?)',
            [('latest-time', 'A', 'retention', stamp + '.900'),
             ('oldest-time', 'B', 'retention', stamp + '.100'),
             ('newer-insertion', 'C', 'retention', stamp + '.500')])
        memory.conn.commit()
        memory._trim_working_memory()
        assert {row[0] for row in memory.conn.execute('SELECT id FROM working_memory')} == {'latest-time', 'newer-insertion'}
        memory.conn.execute('INSERT INTO working_memory(id,content,session_id,timestamp) VALUES (?,?,?,?)',
                            ('same-time-newest', 'D', 'retention', stamp + '.500'))
        memory.conn.commit()
        memory._trim_working_memory()
        assert {row[0] for row in memory.conn.execute('SELECT id FROM working_memory')} == {'latest-time', 'same-time-newest'}
    finally:
        memory.conn.close()
