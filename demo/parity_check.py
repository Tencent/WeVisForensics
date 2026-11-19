"""Compare the in-process transformers backend with recorded vLLM trajectories.

    python parity_check.py                                    # demo examples vs examples/cases.json
    python parity_check.py --videos a.mp4 b.mp4 --reference ref/   # ref/<stem>.json from scripts/infer.py --json

Reports, per video, the verdict, the inspector calls and windows, exact raw-text agreement when a
token-level reference is given, wall time, and peak GPU memory.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time
import uuid

import torch

import wvf_local

HERE = Path(__file__).resolve().parent


def calls_from_case(case):
    out = []
    for step in case['steps']:
        if not step.get('skill'):
            continue
        w = step.get('window') or {}
        span = (round(w['at'], 1),) if w.get('type') == 'point' else (
            (round(w['start'], 1), round(w['end'], 1)) if 'start' in w else ())
        out.append((step['skill'], span))
    return out


def calls_from_steps(steps):
    out = []
    for step in steps:
        action = step.get('action') or {}
        if action.get('type') == 'answer' or not action.get('skill'):
            continue
        if 'start_sec' in action:
            span = (round(float(action['start_sec']), 1), round(float(action['end_sec']), 1))
        elif 'at_sec' in action:
            span = (round(float(action['at_sec']), 1),)
        else:
            span = ()
        out.append((action['skill'], span))
    return out


def run(video, window):
    media = wvf_local.prepare(video, 0., 16., uuid.uuid4().hex) if window else None
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if media:
        traj = asyncio.run(wvf_local.AGENT.run(media['path'], duration=media['duration'],
                                               has_audio=media['has_audio']))
    else:
        traj = asyncio.run(wvf_local.AGENT.run(str(video)))
    return traj, time.monotonic() - started, torch.cuda.max_memory_allocated() / 2 ** 30


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--videos', nargs='*', type=Path)
    parser.add_argument('--reference', type=Path, help='directory of scripts/infer.py --json outputs')
    parser.add_argument('--output', type=Path, default=Path('parity_report.json'))
    args = parser.parse_args()

    if args.videos:
        jobs = [(v, None) for v in args.videos]
    else:
        cases = json.loads((HERE / 'examples' / 'cases.json').read_text(encoding='utf-8'))
        jobs = [(HERE / 'examples' / c['video'], c) for c in cases]

    rows = []
    for video, case in jobs:
        traj, seconds, peak = run(video, window=case is not None)
        mine = calls_from_steps(traj.steps)
        row = dict(video=video.name, answer=traj.answer, status=traj.status, calls=mine,
                   seconds=round(seconds, 1), peak_gib=round(peak, 1))
        if case is not None:
            row.update(ref_answer=case['answer'], ref_calls=calls_from_case(case))
        elif args.reference:
            ref = json.loads((args.reference / f'{video.stem}.json').read_text(encoding='utf-8'))
            raws, ref_raws = [s['raw'] for s in traj.steps], [s['raw'] for s in ref['steps']]
            row.update(ref_answer=ref.get('answer'), ref_calls=calls_from_steps(ref['steps']),
                       raw_identical=raws == ref_raws,
                       first_divergent_turn=next((i + 1 for i, (a, b) in enumerate(zip(raws, ref_raws)) if a != b),
                                                 None if len(raws) == len(ref_raws) else min(len(raws), len(ref_raws)) + 1))
        if 'ref_answer' in row:
            row.update(same_answer=row['answer'] == row['ref_answer'], same_calls=row['calls'] == row['ref_calls'])
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    compared = [r for r in rows if 'same_answer' in r]
    summary = dict(videos=len(rows), mean_seconds=round(sum(r['seconds'] for r in rows) / max(len(rows), 1), 1),
                   max_peak_gib=max((r['peak_gib'] for r in rows), default=0))
    if compared:
        summary.update(same_answer=sum(r['same_answer'] for r in compared),
                       same_calls=sum(r['same_calls'] for r in compared), compared=len(compared))
    if any('raw_identical' in r for r in rows):
        summary['raw_identical'] = sum(bool(r.get('raw_identical')) for r in rows)
    print(json.dumps(summary, ensure_ascii=False))
    args.output.write_text(json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
