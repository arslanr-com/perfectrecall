"""Run with an isolated, freshly installed Python: python -I scripts/verify_install.py."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='perfectrecall-install-') as directory:
        env={k:v for k,v in os.environ.items() if not k.startswith(('MNEMOSYNE_','PERFECTRECALL_','JEVOSYNE_'))}
        env['HERMES_HOME']=str(Path(directory)/'home')
        for module in ('numpy','sqlite_vec','fastembed','onnxruntime'):
            assert importlib.util.find_spec(module) is None,'Use a fresh environment for this check'
        for command in ([sys.executable,'-I','-m','perfectrecall','--version'],
                        [sys.executable,'-I','-m','perfectrecall.install','--hermes-home',env['HERMES_HOME']]):
            subprocess.run(command,env=env,cwd=directory,check=True,capture_output=True,text=True)
        status=json.loads(subprocess.check_output([sys.executable,'-I','-m','perfectrecall','jev-status'],env=env,cwd=directory,text=True))
        assert status['product']=='PerfectRecall' and status['backend']=='jev'
        assert Path(status['data_dir'])==Path(env['HERMES_HOME'])/'mnemosyne/data'
        report=dict(status='passed',python=sys.version.split()[0],isolated_mode=bool(sys.flags.isolated),
                    installed_packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
                    checks=dict(no_previous_memory_provider=True,no_vector_dependencies=True,
                                module_cli=True,first_provider_installer=True,legacy_default_data_path=True))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
