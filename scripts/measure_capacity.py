"""Measure synthetic full-corpus prefetch against Hermes' own deadline.

Requires Hermes' Python, a built wheel, and OPENROUTER_API_KEY. Makes paid
requests only with --live. Large scans can outlive the host's prefetch window.
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hermes-root',type=Path,required=True)
    parser.add_argument('--wheel',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--records',type=int,default=128)
    parser.add_argument('--workers',type=int,default=128)
    parser.add_argument('--diagnostic-workers',action='store_true',help='Allow a scheduling-only 512-worker experiment')
    parser.add_argument('--observe-seconds',type=float,default=300)
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    if not args.live:parser.error('--live is required; this script makes paid API calls')
    if not 1<=args.records<=10000:parser.error('--records must be 1..10000')
    if not 1<=args.workers<=256 and not(args.workers==512 and args.diagnostic_workers):parser.error('Use 1..256 workers, or explicitly allow the diagnostic 512 override')
    if not 1<=args.observe_seconds<=300:parser.error('--observe-seconds must be 1..300')
    with tempfile.TemporaryDirectory(prefix='perfectrecall-capacity-') as directory:
        scratch=Path(directory);site=scratch/'site';home=scratch/'home'
        with zipfile.ZipFile(args.wheel) as archive:archive.extractall(site)
        for key in tuple(os.environ):
            if key.startswith(('MNEMOSYNE_','PERFECTRECALL_','JEVOSYNE_')):del os.environ[key]
        os.environ.update(HERMES_HOME=str(home),PERFECTRECALL_HOST_LLM_ENABLED='0',PERFECTRECALL_LLM_ENABLED='0',PERFECTRECALL_JEV_WORKERS=str(min(args.workers,256)))
        sys.path[:0]=[str(site),str(args.hermes_root.resolve())]
        from perfectrecall.install import configure_hermes
        configure_hermes(home)
        from plugins.memory import load_memory_provider
        from agent.memory_manager import MemoryManager
        from mnemosyne.core import jev,jev_evidence
        if args.diagnostic_workers:jev_evidence.worker_count=lambda:args.workers
        provider=load_memory_provider('perfectrecall',register_skills=False)
        provider.initialize('capacity',hermes_home=str(home),auto_sleep=False)
        assert provider._beam is not None
        for n in range(args.records):
            content=('Project Cedar uses PostgreSQL as its production database.' if n==args.records-1 else f'Test project number {n:05d} uses SQLite as its production database.')
            provider._beam.conn.execute('INSERT INTO working_memory(id,content,session_id,source,importance,scope,timestamp) VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)',(f'scale-{n:05d}',content,provider._beam.session_id,'fact',.5,'session'))
        provider._beam.conn.commit()
        client=jev.client();transport=client._transport
        def bounded(payload,timeout):
            usage=client.snapshot()
            if usage['requests']>args.records*2+50 or usage['cost_usd']>.5:raise jev.JevError('Capacity request or cost budget reached')
            return transport(payload,timeout)
        client._transport=bounded
        observed={};original=provider.prefetch
        def traced(*a,**kw):
            start=time.monotonic()
            try:
                result=original(*a,**kw)
                observed.update(seconds=time.monotonic()-start,contains_target='PostgreSQL' in result)
                return result
            except Exception as exc:
                observed.update(seconds=time.monotonic()-start,error_type=type(exc).__name__)
                raise
        provider.prefetch=traced
        manager=MemoryManager();manager.add_provider(provider)
        start=time.monotonic()
        result=manager.prefetch_all('Which production database does Project Cedar use?',session_id='capacity')
        elapsed=time.monotonic()-start
        worker=manager._external_prefetch_threads.get('perfectrecall')
        deadline=time.monotonic()+args.observe_seconds
        while worker and worker.is_alive() and time.monotonic()<deadline:
            worker.join(min(30,max(0,deadline-time.monotonic())))
            if worker.is_alive():print('Waiting for the underlying scan; the host deadline has already passed.',flush=True)
        alive=bool(worker and worker.is_alive())
        report=dict(workers=args.workers,records=args.records,synthetic=True,diagnostic_override=args.diagnostic_workers,
            wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),host_elapsed_seconds=elapsed,
            host_timeout_seconds=manager._external_prefetch_timeout,injected_target='PostgreSQL' in result,
            worker_still_alive=alive,underlying=observed,usage=client.snapshot())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
        if alive:
            # The process owns this temporary profile; do not delete files while
            # its worker uses them. An incomplete observation remains incomplete.
            print('Observation incomplete; waiting for worker cleanup before exit.',flush=True)
            while worker.is_alive():worker.join(30)
        manager.shutdown_all()

if __name__=='__main__':main()
