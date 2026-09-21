"""Run a frozen evaluator with an empty profile and no inherited memory settings.

Use the separate historical virtualenv's Python. All remaining arguments are
passed unchanged to the original script, whose bytes and hash stay intact.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('script',type=Path)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    script=args.script.resolve()
    historical=Path(__file__).resolve().parent/'historical'
    if not script.is_relative_to(historical):parser.error('Choose a script under benchmarks/historical')
    env={k:v for k,v in os.environ.items() if not k.startswith(('MNEMOSYNE_','PERFECTRECALL_','JEVOSYNE_'))}
    with tempfile.TemporaryDirectory(prefix='perfectrecall-benchmark-profile-') as directory:
        env['HERMES_HOME']=directory
        result=subprocess.run([sys.executable,str(script),*args.arguments],env=env)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
