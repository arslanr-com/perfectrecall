"""Exercise an installed wheel with Hermes' actual loader and MemoryManager.

Use Hermes' Python. --live makes bounded paid calls with OPENROUTER_API_KEY.
The profile is temporary; no installed user profile is edited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hermes-root',type=Path,required=True)
    parser.add_argument('--wheel',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    # Bound the test's lifetime and keep the user's profile out of discovery.
    with tempfile.TemporaryDirectory(prefix='perfectrecall-hermes-') as directory:
        scratch=Path(directory);site=scratch/'site';home=scratch/'home'
        with zipfile.ZipFile(args.wheel) as archive: archive.extractall(site)
        for key in tuple(os.environ):
            if key.startswith(('MNEMOSYNE_','PERFECTRECALL_','JEVOSYNE_')):del os.environ[key]
        os.environ.update(HERMES_HOME=str(home),PERFECTRECALL_HOST_LLM_ENABLED='0',
                          PERFECTRECALL_LLM_ENABLED='0')
        sys.path[:0]=[str(site),str(args.hermes_root.resolve())]
        from perfectrecall.install import configure_hermes
        configure_hermes(home)
        from plugins.memory import load_memory_provider, find_provider_entry_point
        from agent.memory_manager import MemoryManager
        from agent.memory_provider import MemoryProvider
        from mnemosyne.core import jev
        assert find_provider_entry_point('perfectrecall') is not None
        if args.live:
            client=jev.client();transport=client._transport
            def bounded(payload,timeout):
                if client.snapshot()['requests']>40:raise jev.JevError('Smoke request limit reached')
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
        manager=MemoryManager();manager.add_provider(provider)
        fact='Project Cedar uses PostgreSQL as its production database.'
        manager.sync_all(fact,'Understood.',session_id='first-provider')
        assert manager.flush_pending(timeout=45),'Automatic sync timed out'
        diagnostics=provider._sync_turn_diagnostics()
        assert diagnostics['completed']==1 and diagnostics['failed']==0,diagnostics
        rows=[r[0] for r in provider._beam.conn.execute('SELECT content FROM working_memory')]
        assert any(fact in row for row in rows)
        assert all('[ASSISTANT]' not in row for row in rows)
        manager.shutdown_all()
        provider=load_memory_provider('perfectrecall',register_skills=False)
        provider.initialize('first-provider',hermes_home=str(home),auto_sleep=False)
        manager=MemoryManager();manager.add_provider(provider)
        context=manager.prefetch_all('Which production database does Project Cedar use?',session_id='first-provider')
        assert 'PostgreSQL' in context,repr(context)
        manager.shutdown_all()
        report=dict(status='passed',live_api=args.live,manual_memory_tool_calls=0,
            hermes_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.hermes_root,text=True).strip(),
            wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
            checks=dict(empty_profile=True,entrypoint_discovered=True,actual_host_abc=True,
                automatic_capture=True,assistant_excluded=True,reopened_prefetch=True),usage=client.snapshot())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
