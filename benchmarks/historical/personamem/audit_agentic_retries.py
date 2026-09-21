#!/usr/bin/env python3
"""Supplement the frozen primary comparison with a conservative retry bound."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from evaluate_answers import digest, paired_summary


def family_sign_flip(rows):
    """Exchange paired arms per question family, preserving within-family dependence."""
    differences = {}
    for row in rows:
        family = row['id'].removesuffix('_abs')
        difference = int(not row['baseline']['correct']) - int(not row['jev']['correct'])
        differences[family] = differences.get(family, 0) + difference
    observed = sum(differences.values())
    distribution = {0: 1}
    for value in differences.values():
        if not value:
            continue
        step = abs(value)
        updated = {}
        for total, count in distribution.items():
            for next_total in (total-step, total+step):
                updated[next_total] = updated.get(next_total, 0) + count
        distribution = updated
    return dict(families=len(differences), cases=len(rows), observed_error_difference=observed,
        family_sign_flip_one_sided_p=sum(n for d, n in distribution.items() if d >= observed)/sum(distribution.values()),
        method='Paired arm exchange within entire question families; includes answerable/absent variants in one block.')


def sensitivity(comparison, audit):
    rows = deepcopy(comparison['results'])
    index = {row['id']: row for row in rows}
    if not rows or len(index) != len(rows):
        raise ValueError('Require a nonempty comparison with unique case IDs')
    adjusted = set()
    for retry in audit['retries']:
        arm, case_id = retry['backend'], retry['case_id']
        if arm not in ('baseline', 'jev') or case_id not in index:
            raise ValueError('Retry must identify a compared case and a valid arm')
        index[case_id][arm]['correct'] = arm == 'baseline'
        adjusted.add((arm, case_id))
    return dict(scope='Conservative sensitivity only; does not replace primary grades',
        primary=paired_summary(comparison['results']), conservative=paired_summary(rows),
        primary_family_test=family_sign_flip(comparison['results']),
        conservative_family_test=family_sign_flip(rows),
        adjusted_cases=[dict(backend=arm, case_id=case_id) for arm, case_id in sorted(adjusted)],
        interpretation='Every retried baseline case is assumed correct; every retried Jev case is assumed incorrect. All original cases remain in the denominator.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw, audit = args.comparison.read_bytes(), args.audit.read_bytes()
    result = sensitivity(json.loads(raw), json.loads(audit))
    result['input_sha256'] = dict(comparison=digest(raw), audit=digest(audit))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result['conservative']))
