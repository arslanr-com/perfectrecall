"""Check the independent dataset bytes and case selection before spending money."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    args=parser.parse_args();root=Path(__file__).parent
    spec=importlib.util.spec_from_file_location('selection',root/'historical/confirmation/evaluate_longmemeval.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    frozen=json.loads((root/'results/confirmation-freeze.json').read_text())
    raw=args.dataset.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==frozen['dataset_sha256'],'Dataset bytes differ from the published experiment'
    ids=[c['question_id'] for c in module.select_cases(json.loads(raw),**frozen['selection'])]
    assert ids==frozen['selected_ids'],'Case selection differs'
    print(json.dumps({'verified_cases':len(ids),'dataset_sha256':frozen['dataset_sha256']}))

if __name__=='__main__':main()
