#!/usr/bin/env python3
"""Reserve disjoint personas, then adapt their complete histories for evaluation.

Source: https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2 (CC BY 4.0).
No provider calls. Gold preferences, snippets, profiles and answer options never
enter the memory corpus or the caller question. The source data is not bundled.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


DATASET_SHA256 = '95f2a8a324aab7baf2af937feae12731369e2abf7cad5ab3e170594cb25a3e52'
REVISION = 'ed956dea41521fc4499acbc63f966e0fd3c053ba'
SALT = 'jevosyne-personamem-v2-v1'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_query(raw):
    try:
        query = json.loads(raw)
    except json.JSONDecodeError:
        query = ast.literal_eval(raw)
    if (not isinstance(query, dict) or query.get('role') != 'user'
            or not isinstance(query.get('content'), str) or not query['content'].strip()):
        raise ValueError('Expected a nonempty user query')
    return query['content']


def row_id(row):
    return f"personamem-v2-{row['persona_id']}-{digest(parse_query(row['user_query']).encode())[:16]}"


def source_path(root, raw):
    path = PurePosixPath(raw)
    if (path.is_absolute() or '..' in path.parts or not path.parts
            or path.parts[0] != 'data' or path.suffix != '.json'):
        raise ValueError('History must be a relative JSON path inside data/')
    result = (root / str(path)).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('History escapes the dataset directory')
    return result


def reserve(rows):
    """One query per persona, selected without inspecting answers or histories."""
    groups = {}
    for row in rows:
        groups.setdefault(row['persona_id'], []).append(row)
    personas = sorted(groups, key=lambda p: digest(f'{SALT}:persona:{p}'.encode()))
    if len(personas) != 200:
        raise ValueError('Expected the pinned 200-persona benchmark')
    selected = []
    for index, persona in enumerate(personas):
        row = min(groups[persona], key=lambda r: digest(f'{SALT}:query:{row_id(r)}'.encode()))
        selected.append(dict(id=row_id(row), persona_id=persona,
                             split='development' if index < 20 else 'confirmation' if index < 120 else 'reserve',
                             history=row['chat_history_128k_link']))
    return dict(protocol='jevosyne-personamem-reservation-v1', source_revision=REVISION,
                dataset_sha256=DATASET_SHA256, salt=SALT, context='128k',
                selection='One hash-selected query per hash-ordered persona; 20 development, 100 sealed confirmation, 80 reserve.',
                cases=selected)


def convert(row, history):
    if isinstance(history, dict):
        if set(history) == {'metadata', 'chat_history'}:
            history = history['chat_history']
        elif set(history) == {'conversations'}:
            history = history['conversations']
        else:
            raise ValueError('Unexpected history wrapper; inspect development format before adaptation')
    if not isinstance(history, list) or not history:
        raise ValueError('Expected a complete nonempty conversation list')
    turns = []
    for turn in history:
        if (not isinstance(turn, dict) or turn.get('role') not in {'user', 'assistant', 'system', 'tool'}
                or not isinstance(turn.get('content'), str)):
            raise ValueError('Unsupported history turn; never silently omit data')
        turns.append(dict(role=turn['role'], content=turn['content']))
    if not isinstance(row.get('preference'), str) or not row['preference'].strip():
        raise ValueError('Missing preference reference')
    return dict(question_id=row_id(row), question_type='personamem-preference',
                question=parse_query(row['user_query']), question_date='not provided',
                answer=dict(preference=row['preference'], reference_response=row['correct_answer']),
                answer_session_ids=[], haystack_session_ids=['conversation'],
                haystack_dates=['not provided'], haystack_sessions=[turns],
                evaluation_metadata={k: row[k] for k in ('persona_id', 'pref_type', 'who', 'updated', 'conversation_scenario')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--reservation', type=Path, required=True)
    parser.add_argument('--split', choices=['development', 'confirmation', 'reserve'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    csv_path = args.dataset_root / 'benchmark/text/benchmark.csv'
    if digest(csv_path.read_bytes()) != DATASET_SHA256:
        raise ValueError('Source CSV does not match pinned revision')
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    reservation = reserve(rows)
    if args.reservation.exists():
        if json.loads(args.reservation.read_text()) != reservation:
            raise ValueError('Existing reservation differs; refusing replacement')
    else:
        args.reservation.parent.mkdir(parents=True, exist_ok=True)
        args.reservation.write_text(json.dumps(reservation, indent=2) + '\n')
    if args.split:
        if args.output is None:
            raise ValueError('--output is required with --split')
        by_id = {row_id(row): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError('Duplicate query IDs')
        cases, history_hashes = [], {}
        for selected in reservation['cases']:
            if selected['split'] != args.split:
                continue
            source = source_path(args.dataset_root, selected['history'])
            raw = source.read_bytes()
            history_hashes[selected['history']] = digest(raw)
            cases.append(convert(by_id[selected['id']], json.loads(raw)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(cases, ensure_ascii=False) + '\n')
        audit = dict(source_revision=REVISION, csv_sha256=DATASET_SHA256,
                     reservation_sha256=digest(args.reservation.read_bytes()), split=args.split,
                     output_sha256=digest(args.output.read_bytes()), cases=len(cases),
                     history_sha256=history_hashes,
                     transformation='Original roles and complete text only; no reference snippets, profiles or options; dates remain unspecified.')
        args.output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        print(json.dumps(dict(split=args.split, cases=len(cases), output_sha256=audit['output_sha256'])))
    else:
        print('Reserved 20 development, 100 confirmation and 80 future personas; no histories or answers displayed.')


if __name__ == '__main__':
    main()
