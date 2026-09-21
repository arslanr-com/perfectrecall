#!/usr/bin/env python3
"""Run every frozen evaluation case in bounded, independently resumable batches."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import subprocess
import sys

from evaluate_answers import digest
from evaluate_agentic_memory import search_schema
from evaluate_longmemeval import select_cases


def partition(ids, size):
    if size <= 0 or not ids or len(set(ids)) != len(ids):
        raise ValueError('Require unique cases and a positive batch size')
    return [ids[start:start+size] for start in range(0, len(ids), size)]


def merge(reports, selected):
    if not reports or any(r['status'] != 'completed' for r in reports):
        raise ValueError('Every batch must complete before aggregation')
    template = {k: v for k, v in reports[0]['plan'].items() if k != 'selected_ids'}
    rows, seen = [], []
    for report in reports:
        if {k: v for k, v in report['plan'].items() if k != 'selected_ids'} != template:
            raise ValueError('Batch protocols or source versions differ')
        ids = [r['id'] for r in report['results']]
        if ids != report['plan']['selected_ids']:
            raise ValueError('A batch must report each planned case in order')
        rows.extend(report['results'])
        seen.extend(ids)
    if seen != selected or len(set(seen)) != len(seen):
        raise ValueError('Batches must cover the entire frozen selection exactly once')
    result = dict(status='completed', plan=dict(template, selected_ids=selected),
                  results=rows, calls=[c for r in reports for c in r['calls']],
                  summary=dict(cases=len(rows), errors=sum(not r['correct'] for r in rows)))
    for category in ('embedding_usage', 'jev_usage'):
        stats = [r[category] for r in reports if category in r]
        if not stats:
            continue
        combined = {}
        for field in set().union(*(s.keys() for s in stats)):
            values = [s[field] for s in stats if field in s]
            if all(type(v) in (int, float) for v in values):
                combined[field] = sum(values)
            elif all(v == values[0] for v in values):
                combined[field] = values[0]
            else:
                raise ValueError(f'Provider metadata changed across batches: {field}')
        result[category] = combined
    return result


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def run(args):
    amounts = (args.caller_budget, args.jev_budget, args.embedding_budget)
    if (any(not math.isfinite(x) or x <= 0 for x in amounts)
            or args.jobs < 1 or args.max_jev_requests < 1
            or args.per_type < 1 or args.test_offset < 1):
        raise ValueError('Budgets and concurrency must be positive and finite')
    root = args.source_root.resolve()
    script = Path(__file__).with_name('evaluate_agentic_memory.py').resolve()
    raw = args.dataset.read_bytes()
    data = json.loads(raw)
    ids = [c['question_id'] for c in select_cases(data, 'test', args.per_type, args.test_offset)]
    if len(ids) != len({c['question_type'] for c in data})*args.per_type:
        raise ValueError('Dataset cannot supply every requested case')
    groups = partition(ids, args.batch_size)
    if args.max_jev_requests < len(groups):
        raise ValueError('Request budget cannot be divided among batches')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    directory = args.output.with_suffix('.parts')
    directory.mkdir(exist_ok=True)
    manifest_path = directory/'manifest.json'
    schema = search_schema(root)
    plan = dict(protocol='jevosyne-agentic-batches-v1', backend=args.backend, scope=args.scope,
        dataset_sha256=digest(raw), selected_ids=ids, groups=groups,
        runtime_sha256=digest(b''.join(str(p.relative_to(root)).encode()+p.read_bytes()
                              for p in sorted((root/'mnemosyne').rglob('*.py')))),
        schema_sha256=digest(json.dumps(schema, sort_keys=True).encode()),
        scripts={p.name: digest(p.read_bytes()) for p in [Path(__file__), script,
            script.with_name('evaluate_answers.py'), script.with_name('evaluate_longmemeval.py')]},
        caller_model=args.caller_model, reasoning_effort='high', judge_model=args.judge_model,
        answer_max_tokens=16384, searches=3, records=20, context_chars=120000,
        jobs=args.jobs, caller_budget=args.caller_budget, jev_budget=args.jev_budget,
        embedding_budget=args.embedding_budget, max_jev_requests=args.max_jev_requests)
    manifest = dict(status='running', plan=plan, batches={})
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['plan'] != plan:
            raise ValueError('Cannot resume after changing the frozen batch plan')
        manifest = previous
        manifest['status'] = 'running'
    # Freeze the complete selection and implementation before starting a child.
    save(manifest_path, manifest)
    total = len(groups)

    def batch(index):
        output = directory/f'batch-{index:03d}.json'
        command = [sys.executable, str(script), '--dataset', str(args.dataset.resolve()),
            '--source-root', str(root), '--api-key-file', str(args.api_key_file.resolve()),
            '--backend', args.backend, '--scope', args.scope, '--split', 'test',
            '--test-offset', str(args.test_offset), '--per-type', str(args.per_type),
            '--start-case', str(index*args.batch_size), '--limit', str(len(groups[index])),
            '--caller-model', args.caller_model, '--reasoning-effort', 'high',
            '--judge-model', args.judge_model, '--answer-max-tokens', '16384',
            '--searches', '3', '--records', '20', '--context-chars', '120000',
            '--caller-budget', str(args.caller_budget/total),
            '--jev-budget', str(args.jev_budget/total),
            '--embedding-budget', str(args.embedding_budget/total),
            '--max-jev-requests', str(args.max_jev_requests//total), '--output', str(output.resolve())]
        with (directory/f'batch-{index:03d}.log').open('a') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        return index, output, code

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(batch, index) for index in range(total)]
        for future in as_completed(futures):
            index, output, code = future.result()
            manifest['batches'][str(index)] = dict(output=output.name, exit_code=code)
            save(manifest_path, manifest)
            print(json.dumps(dict(batch=index, total_batches=total, exit_code=code)), flush=True)
    if any(b['exit_code'] != 0 for b in manifest['batches'].values()):
        manifest['status'] = 'failed'
        save(manifest_path, manifest)
        raise RuntimeError('One or more batches failed; preserve results and resume the same plan')
    paths = [directory/f'batch-{index:03d}.json' for index in range(total)]
    reports = [json.loads(p.read_text()) for p in paths]
    expected = dict(source_sha256=plan['runtime_sha256'], schema_sha256=plan['schema_sha256'],
        harness_sha256=plan['scripts']['evaluate_agentic_memory.py'],
        client_sha256=plan['scripts']['evaluate_answers.py'])
    expected.update({key: plan[key] for key in ('dataset_sha256', 'backend', 'scope',
        'caller_model', 'reasoning_effort', 'judge_model', 'answer_max_tokens',
        'searches', 'records', 'context_chars')})
    if any(any(r['plan'][key] != value for key, value in expected.items()) for r in reports):
        raise ValueError('A child used different code than the frozen parent plan')
    result = merge(reports, ids)
    result['batch_provenance'] = dict(manifest_sha256=digest(manifest_path.read_bytes()),
        files={p.name: digest(p.read_bytes()) for p in paths})
    manifest['status'] = 'completed'
    save(manifest_path, manifest)
    result['batch_provenance']['manifest_sha256'] = digest(manifest_path.read_bytes())
    save(args.output, result)
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('dataset', 'source-root', 'api-key-file', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--backend', choices=['baseline', 'jev'], required=True)
    parser.add_argument('--scope', choices=['diagnostic', 'frozen-independent'], default='diagnostic')
    parser.add_argument('--test-offset', type=int, required=True)
    parser.add_argument('--per-type', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--caller-model', default='openai/gpt-5.6-luna')
    parser.add_argument('--judge-model', default='openai/gpt-4.1')
    parser.add_argument('--caller-budget', type=float, default=5.)
    parser.add_argument('--jev-budget', type=float, default=5.)
    parser.add_argument('--embedding-budget', type=float, default=.5)
    parser.add_argument('--max-jev-requests', type=int, default=360000)
    run(parser.parse_args())
