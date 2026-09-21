"""Exercise an installed wheel with Hermes' actual loader and MemoryManager.

Use Hermes' Python. --live makes bounded paid calls with OPENROUTER_API_KEY.
The profile is temporary; no installed user profile is edited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hermes-root',type=Path,required=True)
    installation = parser.add_mutually_exclusive_group(required=True)
    installation.add_argument('--wheel',type=Path)
    installation.add_argument('--plugin-dir', type=Path, help='Verify a directory plugin without pip installing PerfectRecall')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--records', type=int, default=0, help='Synthetic existing mixed-length records, up to 10000')
    args=parser.parse_args()
    if not 0 <= args.records <= 10000:
        parser.error('--records must be between 0 and 10000')
    timings = {}
    # Bound the test's lifetime and keep the user's profile out of discovery.
    with tempfile.TemporaryDirectory(prefix='perfectrecall-hermes-') as directory:
        scratch=Path(directory);site=scratch/'site';home=scratch/'home'
        if args.wheel:
            with zipfile.ZipFile(args.wheel) as archive: archive.extractall(site)
        else:
            site = home/'plugins'/'perfectrecall'
            shutil.copytree(args.plugin_dir, site, ignore=shutil.ignore_patterns(
                '.git', 'build', 'dist', '*.egg-info', '__pycache__', '.pytest_cache', '.ruff_cache'))
        for key in tuple(os.environ):
            if key.startswith(('MNEMOSYNE_','PERFECTRECALL_','JEVOSYNE_')):del os.environ[key]
        os.environ.update(HERMES_HOME=str(home),PERFECTRECALL_HOST_LLM_ENABLED='0',
                          PERFECTRECALL_LLM_ENABLED='0')
        sys.path.insert(0, str(args.hermes_root.resolve()))
        if args.wheel:
            sys.path.insert(0, str(site))
            from perfectrecall.install import configure_hermes
            configure_hermes(home)
        else:
            # Real setup must discover and import the copied directory, with no
            # installed package or repository path helping resolve its imports.
            from hermes_cli.memory_setup import cmd_setup_provider
            cmd_setup_provider('perfectrecall')
            import yaml
            assert yaml.safe_load((home/'config.yaml').read_text())['memory']['provider']=='perfectrecall'
        from plugins.memory import load_memory_provider, find_provider_entry_point
        from agent.memory_manager import MemoryManager
        from agent.memory_provider import MemoryProvider
        from mnemosyne.core import jev
        if args.wheel:
            assert find_provider_entry_point('perfectrecall') is not None
        else:
            assert find_provider_entry_point('perfectrecall') is None, 'Use an environment without PerfectRecall installed'
        if args.live:
            client=jev.client();transport=client._transport
            def bounded(payload,timeout):
                usage = client.snapshot()
                if usage['requests'] > (2000 if args.records else 40) or usage['cost_usd'] >= 2:
                    raise jev.JevError('Smoke request or cost limit reached')
                return transport(payload,timeout)
            client._transport=bounded
        else:
            def transport(payload,timeout):
                answers={}
                for key,q in payload['questions'].items():
                    if q['type']=='choice':
                        chosen=next(iter(q['criteria']))
                        answers[key]=dict(type='choice',choice=chosen,confidence=1.,probabilities={x:float(x==chosen) for x in q['criteria']})
                    else:answers[key]=dict(type='noul',noul=.95)
                return dict(model='test-double',answers=answers,usage=dict(input_tokens=0,output_tokens=0))
            client=jev.JevClient('test-double',transport=transport)
            jev.client=lambda:client
        provider=load_memory_provider('perfectrecall',register_skills=False)
        assert isinstance(provider,MemoryProvider)
        assert provider.name=='perfectrecall'
        assert str(site) in sys.modules['perfectrecall'].__file__
        provider.initialize('first-provider',hermes_home=str(home),auto_sleep=False)
        assert provider._beam is not None,provider._init_error
        if args.records:
            from measure_recall_performance import corpus
            # Keep Cedar absent before automatic capture. Preserve the normal
            # retention policy when the write crosses the working-memory cap.
            fixtures = [(key, text.replace('Project Cedar', 'Project Birch'))
                        for key, text in corpus(args.records, 'mixed')]
            provider._beam.conn.executemany(
                'INSERT INTO working_memory(id,content,session_id,source,importance,scope,timestamp) VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)',
                [(key, text, provider._beam.session_id, 'fact', .5, 'session') for key, text in fixtures])
            provider._beam.conn.commit()
        manager=MemoryManager();manager.add_provider(provider)
        fact='Project Cedar uses PostgreSQL as its production database.'
        started = time.monotonic()
        manager.sync_all(fact,'Understood.',session_id='first-provider')
        timings['sync_dispatch_seconds'] = time.monotonic() - started
        assert manager.flush_pending(timeout=45),'Automatic sync timed out'
        timings['sync_completed_seconds'] = time.monotonic() - started
        diagnostics=provider._sync_turn_diagnostics()
        assert diagnostics['completed']==1 and diagnostics['failed']==0,diagnostics
        rows=[r[0] for r in provider._beam.conn.execute('SELECT content FROM working_memory')]
        assert any(fact in row for row in rows)
        assert all('[ASSISTANT]' not in row for row in rows)
        manager.shutdown_all()
        provider=load_memory_provider('perfectrecall',register_skills=False)
        provider.initialize('first-provider',hermes_home=str(home),auto_sleep=False)
        manager=MemoryManager();manager.add_provider(provider)
        started = time.monotonic()
        context=manager.prefetch_all('Which production database does Project Cedar use?',session_id='first-provider')
        timings['reopened_prefetch_seconds'] = time.monotonic() - started
        assert 'PostgreSQL' in context,repr(context)
        assert timings['reopened_prefetch_seconds'] < 8, 'Hermes prefetch exceeded its host window'
        if args.records:
            criteria = ['Does this memory identify the production database used by Project Cedar?',
                        'Does this memory describe where Project Cedar runs its production database?',
                        'Does this memory describe a change to the database used by Project Cedar?']
            for temperature in ('cold', 'warm'):
                before = client.snapshot()
                started = time.monotonic()
                hits = provider._beam.recall('Which production database does Project Cedar use?',
                                             top_k=5, evidence_questions=criteria, explain=True)
                timings[f'three_criteria_{temperature}_seconds'] = time.monotonic() - started
                timings[f'three_criteria_{temperature}_requests'] = client.snapshot()['requests'] - before['requests']
                assert any('PostgreSQL' in hit['content'] and 'Project Cedar' in hit['content'] for hit in hits['results'])
            assert timings['three_criteria_warm_requests'] == 0, 'A full warm scan unexpectedly called the API'
        final_records = provider._beam.conn.execute('SELECT COUNT(*) FROM working_memory').fetchone()[0]
        manager.shutdown_all()
        report=dict(status='passed',live_api=args.live,manual_memory_tool_calls=0,
            initial_records=args.records, final_records=final_records, timings=timings,
            prefetch_mode=getattr(provider, '_prefetch_mode', 'strict'),
            hermes_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.hermes_root,text=True).strip(),
            wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest() if args.wheel else None,
            installation='wheel' if args.wheel else 'directory-plugin',
            checks=dict(empty_profile=True,provider_discovered=True,actual_host_abc=True,
                automatic_capture=True,assistant_excluded=True,reopened_prefetch=True),usage=client.snapshot())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
