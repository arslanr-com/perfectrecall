"""Audit publishable files without printing matched secret values."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','__pycache__','.pytest_cache','.ruff_cache','build','dist','.venv','venv','work','downloads','runs'}
RUNTIME={'perfectrecall','mnemosyne','hermes_memory_provider'}
REMOVED={'numpy','fastembed','onnxruntime','sqlite_vec','sentence_transformers'}


def source_files(root):
    result=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root,capture_output=True)
    if result.returncode==0:
        return [root/p for p in result.stdout.decode().split('\0') if p]
    return [p for p in root.rglob('*') if p.is_file() and not any(x in EXCLUDED or x.endswith('.egg-info') for x in p.relative_to(root).parts)]


def inspect(name,raw,secret=None):
    findings=[]
    suffix=Path(name).suffix
    if suffix in {'.db','.sqlite','.sqlite3','.pem','.key','.pyc'} or Path(name).name.startswith('.env'):
        findings.append('private or generated file type')
    if secret and secret in raw:findings.append('known credential bytes')
    if raw.startswith(b'SQLite format 3'):findings.append('SQLite database')
    try:text=raw.decode('utf-8')
    except UnicodeError:return findings
    patterns={
        'non-English Cyrillic text':r'[\u0400-\u04ff]',
        'private home path':r'/(?:Users|home)/[A-Za-z0-9_.-]+/',
        'OpenRouter credential':r'sk-or-v1-[A-Za-z0-9]{32,}',
        'GitHub credential':r'(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}',
        'private key material':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    }
    for label,pattern in patterns.items():
        if re.search(pattern,text):findings.append(label)
    parts=Path(name).parts
    if parts and parts[0] in RUNTIME and suffix=='.py':
        if Path(name).name in {'embeddings.py','binary_vectors.py','polyphonic_recall.py','query_cache.py'}:
            findings.append('removed production backend')
        try:tree=ast.parse(text)
        except SyntaxError:
            findings.append('invalid Python');return findings
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):mods=[a.name.split('.')[0] for a in node.names]
            elif isinstance(node,ast.ImportFrom):mods=[(node.module or '').split('.')[0]]
            else:continue
            if REMOVED.intersection(mods):findings.append('removed dependency import')
        if '/embeddings' in text:findings.append('embedding endpoint or implementation reference')
    return sorted(set(findings))


def artifact_entries(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith('/'):yield name,archive.read(name)
    else:
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    name=member.name.split('/',1)[-1]
                    yield name,archive.extractfile(member).read()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--secret-file',type=Path,help='Optional local key used only for exact-byte scanning')
    parser.add_argument('--artifact',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    secret=args.secret_file.read_bytes().strip() if args.secret_file else None
    findings=[];count=0
    entries=[(str(p.relative_to(ROOT)),p.read_bytes()) for p in source_files(ROOT)]
    for artifact in args.artifact:
        entries.extend(artifact_entries(artifact))
    for name,raw in entries:
        count+=1
        for reason in inspect(name,raw,secret):findings.append({'file':name,'finding':reason})
    report=dict(status='passed' if not findings else 'failed',files_checked=count,
                known_credential_checked=bool(secret),artifacts_checked=len(args.artifact),findings=findings,
                scope='Publishable working-tree files and explicitly supplied archives; no guarantee against every possible form of sensitive data')
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if findings:raise SystemExit(1)

if __name__=='__main__':main()
