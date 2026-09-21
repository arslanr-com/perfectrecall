"""Reconstruct pinned evaluation runtimes outside the production package.

Requires Git and network access for the public upstream clone. No API calls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

BASELINE='199d4bc6662bd51c18275331ddeb51ac07387acf'
EXPECTED={
    'baseline':'4c94832d0b37c635e70ca6fa82c47cda4d2b3aa997039eb7216c1ec8004393b1',
    'confirmation':'edc43a41bccfc53e41cb23dd464ee244e8c041709a3100446b63ed2b60371034',
    'atomic':'183af1ec39b75157c200bfb6169ca1a147d3c3f52fff4567ba5ffb61e400689e',
}

def runtime_hash(root):
    return hashlib.sha256(b''.join(str(p.relative_to(root)).encode()+p.read_bytes()
        for p in sorted((root/'mnemosyne').rglob('*.py')))).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('benchmarks/work'))
    parser.add_argument('--upstream-local',type=Path,help='Optional existing clone containing the pinned commit')
    args=parser.parse_args();destination=args.output.resolve()
    destination.mkdir(parents=True,exist_ok=True)
    baseline=destination/'baseline'
    if not baseline.exists():
        source=str(args.upstream_local.resolve()) if args.upstream_local else 'https://github.com/mnemosyne-oss/mnemosyne.git'
        subprocess.run(['git','clone','--no-hardlinks',source,str(baseline)],check=True)
        subprocess.run(['git','-C',str(baseline),'checkout','--detach',BASELINE],check=True)
    if runtime_hash(baseline)!=EXPECTED['baseline']:raise ValueError('Baseline source differs from the frozen runtime')
    manifest={}
    for label in EXPECTED:
        root=destination/label
        if label!='baseline' and not root.exists():
            shutil.copytree(baseline,root,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
            patch=Path(__file__).resolve().parent/'historical'/f'{label}.patch'
            subprocess.run(['git','apply','--unidiff-zero',str(patch)],cwd=root,check=True)
        actual=runtime_hash(root)
        if actual!=EXPECTED[label]:raise ValueError(f'{label} source differs from the frozen runtime')
        manifest[label]=actual
    (destination/'verified-runtimes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
