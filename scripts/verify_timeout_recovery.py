"""Verify timeout recovery through real Hermes with a deterministic transport.

No API calls. Run with Hermes' Python and an explicitly built wheel.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hermes-root', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='perfectrecall-timeout-') as directory:
        root = Path(directory)
        site, home = root / 'site', root / 'profile'
        with zipfile.ZipFile(args.wheel) as archive:
            archive.extractall(site)
        for key in tuple(os.environ):
            if key.startswith(('MNEMOSYNE_', 'PERFECTRECALL_', 'JEVOSYNE_')):
                del os.environ[key]
        os.environ.update(HERMES_HOME=str(home), PERFECTRECALL_LLM_ENABLED='0',
                          PERFECTRECALL_HOST_LLM_ENABLED='0', PERFECTRECALL_JEV_BATCH_MODE='question',
                          PERFECTRECALL_PREFETCH_BUDGET_SECONDS='.15')
        sys.path[:0] = [str(site), str(args.hermes_root.resolve())]
        from perfectrecall.install import configure_hermes
        from mnemosyne.core import jev
        from plugins.memory import load_memory_provider
        from agent.memory_manager import MemoryManager
        configure_hermes(home)
        def timeout_transport(payload, timeout):
            time.sleep(timeout + .005)
            raise TimeoutError()
        client = jev.JevClient('test-double', transport=timeout_transport)
        jev.client = lambda: client
        provider = load_memory_provider('perfectrecall', register_skills=False)
        provider.initialize('timeout-test', hermes_home=str(home), auto_sleep=False)
        provider._beam.conn.execute(
            'INSERT INTO working_memory(id,content,session_id,source,importance,scope,timestamp) VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)',
            ('target', 'Project Cedar uses PostgreSQL as its production database.', provider._beam.session_id, 'fact', .5, 'session'))
        provider._beam.conn.commit()
        manager = MemoryManager()
        manager.add_provider(provider)
        query = 'Which database does Project Cedar use?'
        started = time.monotonic()
        first = manager.prefetch_all(query, session_id='timeout-test')
        elapsed = time.monotonic() - started
        first_trace = dict(provider._last_prefetch)
        assert first == '' and first_trace['status'] == 'timed_out', first_trace
        assert elapsed < 1, elapsed
        assert not manager._external_prefetch_threads.get('perfectrecall')
        assert client.snapshot()['failures'] == 0
        assert client.snapshot()['deadline_exceeded'] == 1
        def success_transport(payload, timeout):
            return dict(model='test-double', answers={key: dict(type='noul', noul=.95)
                                                      for key in payload['questions']}, usage={})
        client._transport = success_transport
        next_context = manager.prefetch_all(query, session_id='timeout-test')
        assert 'PostgreSQL' in next_context
        report = dict(status='passed', live_api=False,
                      wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
                      host_timeout_seconds=manager._external_prefetch_timeout,
                      failed_prefetch_seconds=elapsed, failed_prefetch=first_trace,
                      immediate_followup_injected=True, lingering_worker=False, usage=client.snapshot())
        manager.shutdown_all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
