"""Verify the wheel's real MCP stdio protocol with synthetic stored memory.

Requires the MCP extra in this Python environment. No network calls are made.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='perfectrecall-mcp-') as directory:
        scratch=Path(directory);site=scratch/'site';home=scratch/'home'
        with zipfile.ZipFile(args.wheel) as archive:archive.extractall(site)
        for key in tuple(os.environ):
            if key.startswith(('MNEMOSYNE_','PERFECTRECALL_','JEVOSYNE_')):del os.environ[key]
        os.environ.update(HERMES_HOME=str(home),PERFECTRECALL_WRITE_CLASSIFIER='off',
            PERFECTRECALL_LLM_ENABLED='0',PERFECTRECALL_HOST_LLM_ENABLED='0',PERFECTRECALL_CROSS_SESSION='1')
        sys.path.insert(0,str(site))
        from perfectrecall import PerfectRecall
        memory=PerfectRecall(session_id='fixture')
        mid=memory.remember('Project Cedar uses PostgreSQL.',memory_type='fact',scope='global')
        memory.beam.conn.close();memory.conn.close()
        from mcp import ClientSession,StdioServerParameters
        from mcp.client.stdio import stdio_client
        empty=scratch/'empty.env';empty.write_text('')
        env=dict(os.environ,PYTHONPATH=str(site))
        async def run():
            params=StdioServerParameters(command=sys.executable,args=['-m','perfectrecall','mcp','--env-file',str(empty)],env=env,cwd=str(scratch))
            with (scratch/'stderr.log').open('w') as errors:
                async with stdio_client(params,errlog=errors) as streams:
                    async with ClientSession(*streams,read_timeout_seconds=30) as client:
                        await client.initialize()
                        listing=await client.list_tools()
                        names={t.name for t in listing.tools}
                        assert {'mnemosyne_remember','mnemosyne_recall','mnemosyne_get','mnemosyne_forget'}<=names
                        recall=next(t for t in listing.tools if t.name=='mnemosyne_recall')
                        assert 'SHORT' in recall.description
                        result=await client.call_tool('mnemosyne_get',{'memory_id':mid})
                        assert not result.is_error,result
                        content=''.join(c.text for c in result.content if c.type=='text')
                        assert 'Project Cedar uses PostgreSQL.' in content,content
                        return len(names)
        count=asyncio.run(run())
    report=dict(status='passed',live_api=False,transport='stdio',tool_count=count,
                stored_record_read=True,legacy_tool_names=True,atomic_criteria_description=True,
                wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
