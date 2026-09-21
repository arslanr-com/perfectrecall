#!/usr/bin/env python3
"""Compare complete, matched caller/tool runs without hiding failed cases."""
import argparse
import json
from pathlib import Path

from evaluate_answers import digest, paired_summary


MATCHED_SETTINGS = (
    'protocol', 'scope', 'dataset_sha256', 'selected_ids', 'harness_sha256',
    'client_sha256', 'caller_model', 'reasoning_effort', 'judge_model',
    'answer_max_tokens', 'searches', 'records', 'context_chars', 'chunk_chars',
)


def compare(baseline, jev):
    arms = {'baseline': baseline, 'jev': jev}
    indexed = {}
    for arm, report in arms.items():
        if report['status'] != 'completed' or report['plan']['backend'] != arm:
            raise ValueError('Both arms must be completed and correctly identified')
        selected = report['plan']['selected_ids']
        rows = report['results']
        indexed[arm] = {row['id']: row for row in rows}
        if (not selected or len(set(selected)) != len(selected)
                or len(indexed[arm]) != len(rows) or set(indexed[arm]) != set(selected)):
            raise ValueError('Results must cover every selected case exactly once')
        if any(type(row['correct']) is not bool for row in rows):
            raise ValueError('Every case needs a boolean correctness judgment')
    for field in MATCHED_SETTINGS:
        if baseline['plan'][field] != jev['plan'][field]:
            raise ValueError(f'Unmatched evaluation setting: {field}')
    rows = []
    for case_id in baseline['plan']['selected_ids']:
        a, b = indexed['baseline'][case_id], indexed['jev'][case_id]
        if (a['kind'], a['unanswerable']) != (b['kind'], b['unanswerable']):
            raise ValueError('Unmatched case labels')
        rows.append(dict(id=case_id, kind=a['kind'], unanswerable=a['unanswerable'],
                         baseline=dict(correct=a['correct']), jev=dict(correct=b['correct'])))
    groups = {kind: [r for r in rows if r['kind'] == kind] for kind in sorted({r['kind'] for r in rows})}
    groups.update(answerable=[r for r in rows if not r['unanswerable']],
                  unanswerable=[r for r in rows if r['unanswerable']])
    return dict(scope=baseline['plan']['scope'], summary=paired_summary(rows),
                by_group={key: paired_summary(group) for key, group in groups.items() if group},
                results=rows,
                limitation='Model-graded paired sample; development results do not establish independent superiority.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--jev', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = {name: path.read_bytes() for name, path in [('baseline', args.baseline), ('jev', args.jev)]}
    result = compare(*(json.loads(sources[name]) for name in ('baseline', 'jev')))
    result['input_sha256'] = {name: digest(raw) for name, raw in sources.items()}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result['summary']))
