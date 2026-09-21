"""Recompute paired quality results from every published case outcome."""
import json
from pathlib import Path


def summarize(cases):
    ids=[c['id'] for c in cases]
    if not cases or len(set(ids))!=len(ids):raise ValueError('Empty or duplicate case set')
    for c in cases:
        if type(c['baseline_correct']) is not bool or type(c['jev_correct']) is not bool:
            raise ValueError('Labels must be booleans')
    b=sum(not c['baseline_correct'] for c in cases);j=sum(not c['jev_correct'] for c in cases)
    n=len(cases)
    return dict(cases=n,baseline_errors=b,jev_errors=j,relative_error_reduction=1-j/b if b else None,
        baseline_accuracy=1-b/n,jev_accuracy=1-j/n,
        repairs=sum(not c['baseline_correct'] and c['jev_correct'] for c in cases),
        regressions=sum(c['baseline_correct'] and not c['jev_correct'] for c in cases))


def main():
    for name in ('longmemeval-independent','longmemeval-development','personamem-development'):
        data=json.loads((Path(__file__).parent/'results'/f'{name}.json').read_text())
        computed=summarize(data['cases'])
        assert computed==data['summary'],name
        print(name,json.dumps(computed))

if __name__=='__main__':main()
